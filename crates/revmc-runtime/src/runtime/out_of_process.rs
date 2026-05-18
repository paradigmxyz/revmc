//! Out-of-process JIT helper process and IPC.

use crate::{
    CompileTimings, EvmCompiler, EvmLlvmBackend, OptimizationLevel, eyre,
    runtime::{
        config::{CompilationKind, RuntimeConfig},
        storage::RuntimeCacheKey,
        worker::{
            CompileJob, JitObjectSuccess, SyncNotifier, WorkerResult, WorkerSuccess,
            compile_jit_object_artifact, create_compiler,
        },
    },
};
use alloy_primitives::{B256, Bytes};
use crossbeam_channel as chan;
use revm_context_interface::cfg::{GasParams, gas_params::GasId};
use revm_primitives::hardfork::SpecId;
use std::{
    io::{BufReader, BufWriter, Read, Write},
    ops::ControlFlow,
    os::unix::process::CommandExt,
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::{Duration, Instant},
};
use wait_timeout::ChildExt;
use wincode::{SchemaRead, SchemaWrite};

const HELPER_ENV: &str = "REVMC_JIT_HELPER";
const GAS_PARAM_COUNT: usize = 256;

type GasParamPairs = Vec<(u8, u64)>;

/// Runs the out-of-process JIT helper if this process was launched as one.
pub(super) fn maybe_run_jit_helper() -> eyre::Result<ControlFlow<()>> {
    if std::env::var_os(HELPER_ENV).is_none() {
        return Ok(ControlFlow::Continue(()));
    }
    run_jit_helper_stdio()?;
    Ok(ControlFlow::Break(()))
}

/// Compiles a job in the out-of-process helper.
pub(super) fn compile_job(
    job: CompileJob,
    config: &RuntimeConfig,
    helper: &HelperProcess,
) -> WorkerResult {
    let t0 = Instant::now();
    let outcome = run_helper_job(&job, config, helper);
    WorkerResult {
        key: job.key,
        outcome,
        kind: job.kind,
        sync_notifier: job.sync_notifier,
        generation: job.generation,
        compile_duration: t0.elapsed(),
        timings: CompileTimings::default(),
    }
}

fn run_helper_job(
    job: &CompileJob,
    config: &RuntimeConfig,
    helper: &HelperProcess,
) -> Result<WorkerSuccess, String> {
    helper.compile(job, config)
}

struct HelperIo {
    stdin: BufWriter<ChildStdin>,
    result_rx: chan::Receiver<Result<WorkerSuccess, String>>,
}

pub(super) struct HelperProcess {
    inner: Mutex<Option<Arc<HelperProcessInner>>>,
}

impl HelperProcess {
    pub(super) const fn new() -> Self {
        Self { inner: Mutex::new(None) }
    }

    fn compile(&self, job: &CompileJob, config: &RuntimeConfig) -> Result<WorkerSuccess, String> {
        let helper = {
            let mut slot = self.inner.lock().unwrap();
            if slot.as_ref().is_none_or(|helper| !helper.matches_config(config)) {
                debug!("spawning JIT helper");
                *slot = Some(Arc::new(HelperProcessInner::spawn(config)?));
            }
            slot.as_ref().unwrap().clone()
        };

        match helper.compile(job, config) {
            Ok(result) => Ok(result),
            Err(err) => {
                let mut slot = self.inner.lock().unwrap();
                if slot.as_ref().is_some_and(|current| Arc::ptr_eq(current, &helper)) {
                    warn!(error = %err, "discarding JIT helper after failed job");
                    *slot = None;
                }
                Err(err)
            }
        }
    }

    pub(super) fn cancel_in_flight(&self) {
        if let Some(helper) = self.inner.lock().unwrap().take() {
            helper.kill();
        }
    }
}

struct HelperProcessInner {
    path: PathBuf,
    init: HelperInit,
    child: Mutex<Child>,
    io: Mutex<HelperIo>,
    reader: Mutex<Option<JoinHandle<()>>>,
    shutdown_timeout: Duration,
}

impl HelperProcessInner {
    fn spawn(config: &RuntimeConfig) -> Result<Self, String> {
        let path = match &config.jit_helper_path {
            Some(path) => path.clone(),
            None => {
                std::env::current_exe().map_err(|e| format!("failed to locate current exe: {e}"))?
            }
        };
        let init = helper_init(config);
        let mut command = Command::new(&path);
        command
            .env(HELPER_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        apply_helper_limits(&mut command, config);

        let mut child = command.spawn().map_err(|e| format!("failed to spawn JIT helper: {e}"))?;
        let mut stdin = BufWriter::new(child.stdin.take().ok_or("helper stdin unavailable")?);
        write_init(&mut stdin, &init).map_err(|e| format!("failed to write helper init: {e}"))?;
        stdin.flush().map_err(|e| format!("failed to flush helper init: {e}"))?;
        let stdout = child.stdout.take().ok_or("helper stdout unavailable")?;
        let (result_tx, result_rx) = chan::bounded(1);
        let reader = std::thread::spawn(move || {
            let mut stdout = BufReader::new(stdout);
            loop {
                let result = read_helper_result(&mut stdout);
                let done = result.is_err();
                if result_tx.try_send(result).is_err() || done {
                    break;
                }
            }
        });
        Ok(Self {
            path,
            init,
            child: Mutex::new(child),
            io: Mutex::new(HelperIo { stdin, result_rx }),
            reader: Mutex::new(Some(reader)),
            shutdown_timeout: config.tuning.shutdown_timeout,
        })
    }

    fn matches_config(&self, config: &RuntimeConfig) -> bool {
        if self.init != helper_init(config) {
            return false;
        }
        match &config.jit_helper_path {
            Some(path) => self.path == *path,
            None => std::env::current_exe().map(|path| self.path == path).unwrap_or(false),
        }
    }

    fn compile(&self, job: &CompileJob, config: &RuntimeConfig) -> Result<WorkerSuccess, String> {
        let mut io = self.io.lock().unwrap();
        write_job(&mut io.stdin, job).map_err(|e| format!("failed to write helper job: {e}"))?;
        io.stdin.flush().map_err(|e| format!("failed to flush helper job: {e}"))?;

        match io.result_rx.recv_timeout(config.tuning.jit_timeout) {
            Ok(result) => result,
            Err(chan::RecvTimeoutError::Timeout) => {
                warn!(timeout = ?config.tuning.jit_timeout, "JIT helper timed out");
                self.kill();
                Err(format!(
                    "JIT helper timed out after {:?}; helper will be restarted",
                    config.tuning.jit_timeout
                ))
            }
            Err(chan::RecvTimeoutError::Disconnected) => {
                let status = self.child.lock().unwrap().try_wait().ok().flatten();
                let message = match status {
                    Some(status) => format!("JIT helper exited with {status}"),
                    None => "JIT helper disconnected".into(),
                };
                warn!(message, "JIT helper disconnected");
                Err(format!("{message}; helper will be restarted"))
            }
        }
    }

    fn kill(&self) -> bool {
        let mut child = self.child.lock().unwrap();
        if matches!(child.try_wait(), Ok(Some(_))) {
            return true;
        }
        kill_helper(&mut child);
        match child.wait_timeout(self.shutdown_timeout) {
            Ok(Some(_)) => true,
            Ok(None) => {
                warn!(timeout = ?self.shutdown_timeout, "timed out waiting for JIT helper exit");
                false
            }
            Err(err) => {
                warn!(%err, "failed to wait for JIT helper exit");
                false
            }
        }
    }
}

impl Drop for HelperProcessInner {
    fn drop(&mut self) {
        let exited = self.kill();
        if exited && let Some(reader) = self.reader.lock().unwrap().take() {
            let _ = reader.join();
        }
    }
}

fn apply_helper_limits(command: &mut Command, config: &RuntimeConfig) {
    let memory_limit = config.tuning.jit_helper_memory_limit_bytes;
    let cpu_count = config.tuning.jit_helper_cpu_count;

    // SAFETY: `pre_exec` runs in the child after fork and before exec. The closure only calls
    // libc process/resource/affinity syscalls and constructs an `io::Error` if they fail.
    unsafe {
        command.pre_exec(move || {
            set_process_group()?;
            if memory_limit > 0 {
                set_rlimit(libc::RLIMIT_AS as _, memory_limit)?;
            }
            if cpu_count > 0 {
                limit_cpu_affinity(cpu_count)?;
            }
            Ok(())
        });
    }
}

fn set_process_group() -> std::io::Result<()> {
    if unsafe { libc::setpgid(0, 0) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn kill_helper(child: &mut Child) {
    let pid = child.id() as libc::pid_t;
    if unsafe { libc::kill(-pid, libc::SIGKILL) } == 0 {
        return;
    }

    let err = std::io::Error::last_os_error();
    if err.raw_os_error() != Some(libc::ESRCH) {
        warn!(%err, "failed to kill JIT helper process group");
    }
    if let Err(err) = child.kill() {
        warn!(%err, "failed to kill JIT helper");
    }
}

fn set_rlimit(resource: libc::c_int, value: u64) -> std::io::Result<()> {
    let value = libc::rlim_t::try_from(value).unwrap_or(libc::rlim_t::MAX);
    let limit = libc::rlimit { rlim_cur: value, rlim_max: value };
    if unsafe { libc::setrlimit(resource as _, &limit) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn limit_cpu_affinity(cpu_count: usize) -> std::io::Result<()> {
    let mut current = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let size = std::mem::size_of::<libc::cpu_set_t>();
    if unsafe { libc::sched_getaffinity(0, size, &mut current) } != 0 {
        return Err(std::io::Error::last_os_error());
    }

    let mut limited = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let mut remaining = cpu_count;
    for cpu in 0..(8 * size) {
        if unsafe { libc::CPU_ISSET(cpu, &current) } {
            unsafe { libc::CPU_SET(cpu, &mut limited) };
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
    }
    if remaining == cpu_count {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "current CPU affinity mask is empty",
        ));
    }

    if unsafe { libc::sched_setaffinity(0, size, &limited) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn limit_cpu_affinity(_cpu_count: usize) -> std::io::Result<()> {
    Ok(())
}

#[derive(Clone, PartialEq, Eq, SchemaWrite, SchemaRead)]
struct HelperInit {
    debug_assertions: bool,
    no_dedup: bool,
    no_dse: bool,
    dump_dir: Option<String>,
    gas_params: Option<GasParamPairs>,
}

#[derive(SchemaWrite, SchemaRead)]
enum HelperRequest {
    Init(HelperInit),
    Compile(HelperCompile),
}

#[derive(SchemaWrite, SchemaRead)]
struct HelperCompile {
    code_hash: [u8; 32],
    spec_id: u8,
    opt_level: u8,
    symbol_name: String,
    bytecode: Vec<u8>,
}

#[derive(SchemaWrite, SchemaRead)]
enum HelperResponse {
    Ok { symbol_name: String, object_bytes: Vec<u8>, builtin_symbols: Vec<String> },
    Err { error: String },
}

fn write_init<W: Write + ?Sized>(w: &mut BufWriter<W>, init: &HelperInit) -> std::io::Result<()> {
    write_message(w, &HelperRequest::Init(init.clone()))
}

fn write_job<W: Write + ?Sized>(w: &mut BufWriter<W>, job: &CompileJob) -> std::io::Result<()> {
    let req = HelperRequest::Compile(HelperCompile {
        code_hash: job.key.code_hash.0,
        spec_id: job.key.spec_id as u8,
        opt_level: opt_level_to_u8(job.opt_level),
        symbol_name: job.symbol_name.clone(),
        bytecode: job.bytecode.to_vec(),
    });
    write_message(w, &req)
}

fn read_helper_result<R: Read + ?Sized>(r: &mut BufReader<R>) -> Result<WorkerSuccess, String> {
    match read_message(r).map_err(|e| format!("failed to decode helper result: {e}"))? {
        Some(HelperResponse::Ok { symbol_name, object_bytes, builtin_symbols }) => {
            Ok(WorkerSuccess::JitObject(JitObjectSuccess {
                symbol_name,
                object_bytes: Bytes::from(object_bytes),
                builtin_symbols,
            }))
        }
        Some(HelperResponse::Err { error }) => Err(error),
        None => Err("JIT helper closed stdout".into()),
    }
}

fn run_jit_helper_stdio() -> eyre::Result<()> {
    let mut stdin = BufReader::new(std::io::stdin().lock());
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    let mut compiler: Option<EvmCompiler<EvmLlvmBackend>> = None;
    let config = read_helper_init(&mut stdin)?;

    while let Some(job) = read_helper_job(&mut stdin)? {
        if compiler.is_none() {
            compiler = Some(create_compiler(&config, true).map_err(|e| eyre::eyre!(e))?);
        }
        let compiler = compiler.as_mut().unwrap();
        compiler.set_opt_level(job.opt_level);
        compiler.set_dump_to(job_dump_dir(&config, &job));

        let result = compile_jit_object_artifact(&job, compiler);
        if let Err(err) = compiler.clear_ir() {
            warn!(%err, "clear_ir failed");
        }
        write_helper_result(&mut stdout, result)?;
        stdout.flush()?;
    }

    Ok(())
}

fn read_helper_init<R: Read + ?Sized>(stdin: &mut BufReader<R>) -> eyre::Result<RuntimeConfig> {
    match read_message(stdin)? {
        Some(HelperRequest::Init(init)) => runtime_config_from_init(init),
        Some(HelperRequest::Compile(_)) => eyre::bail!("JIT helper received job before init"),
        None => eyre::bail!("JIT helper closed stdin before init"),
    }
}

fn read_helper_job<R: Read + ?Sized>(stdin: &mut BufReader<R>) -> eyre::Result<Option<CompileJob>> {
    let req = match read_message(stdin)? {
        Some(HelperRequest::Compile(req)) => req,
        Some(HelperRequest::Init(_)) => eyre::bail!("JIT helper received duplicate init"),
        None => return Ok(None),
    };
    let spec_id = SpecId::try_from_u8(req.spec_id).ok_or_else(|| eyre::eyre!("invalid spec id"))?;
    let opt_level = opt_level_from_u8(req.opt_level)?;

    let job = CompileJob {
        kind: CompilationKind::Jit,
        key: RuntimeCacheKey { code_hash: B256::from(req.code_hash), spec_id },
        bytecode: Bytes::from(req.bytecode),
        symbol_name: req.symbol_name,
        opt_level,
        sync_notifier: SyncNotifier::none(),
        generation: 0,
    };
    Ok(Some(job))
}

fn write_helper_result<W: Write + ?Sized>(
    stdout: &mut BufWriter<W>,
    result: Result<WorkerSuccess, String>,
) -> eyre::Result<()> {
    let response = match result {
        Ok(WorkerSuccess::JitObject(success)) => HelperResponse::Ok {
            symbol_name: success.symbol_name,
            object_bytes: success.object_bytes.to_vec(),
            builtin_symbols: success.builtin_symbols,
        },
        Ok(_) => unreachable!(),
        Err(error) => HelperResponse::Err { error },
    };
    write_message(stdout, &response)?;
    Ok(())
}

fn write_message<T, W: Write + ?Sized>(w: &mut BufWriter<W>, message: &T) -> std::io::Result<()>
where
    T: wincode::SchemaWrite<wincode::config::DefaultConfig, Src = T>,
{
    let len = wincode::serialized_size(message)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let len = u32::try_from(len)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "message too large"))?;
    w.write_all(&len.to_le_bytes())?;
    wincode::serialize_into(w, message)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

fn read_message<T, R: Read + ?Sized>(r: &mut BufReader<R>) -> std::io::Result<Option<T>>
where
    T: wincode::SchemaReadOwned<wincode::config::DefaultConfig, Dst = T>,
{
    let mut len = [0; 4];
    let n = r.read(&mut len[..1])?;
    if n == 0 {
        return Ok(None);
    }
    r.read_exact(&mut len[1..])?;
    let mut reader = BufReader::new(r.take(u64::from(u32::from_le_bytes(len))));
    let value = wincode::deserialize_from(&mut reader)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let remaining = reader.buffer().len() as u64 + reader.get_ref().limit();
    if remaining != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "trailing bytes in helper message",
        ));
    }
    Ok(Some(value))
}

fn opt_level_to_u8(level: OptimizationLevel) -> u8 {
    match level {
        OptimizationLevel::None => 0,
        OptimizationLevel::Less => 1,
        OptimizationLevel::Default => 2,
        OptimizationLevel::Aggressive => 3,
    }
}

fn opt_level_from_u8(level: u8) -> eyre::Result<OptimizationLevel> {
    Ok(match level {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        3 => OptimizationLevel::Aggressive,
        _ => eyre::bail!("invalid optimization level"),
    })
}

fn gas_params_to_pairs(gas_params: &GasParams) -> GasParamPairs {
    (0..GAS_PARAM_COUNT)
        .map(|i| {
            let id = i as u8;
            (id, gas_params.get(GasId::new(id)))
        })
        .collect()
}

fn gas_params_from_pairs(pairs: GasParamPairs) -> eyre::Result<GasParams> {
    if pairs.len() != GAS_PARAM_COUNT {
        eyre::bail!("invalid gas params length: {}", pairs.len());
    }

    let mut table = [0; GAS_PARAM_COUNT];
    let mut seen = [false; GAS_PARAM_COUNT];
    for (id, value) in pairs {
        let index = usize::from(id);
        if seen[index] {
            eyre::bail!("duplicate gas param id: {id}");
        }
        seen[index] = true;
        table[index] = value;
    }
    if let Some((id, _)) = seen.iter().enumerate().find(|(_, seen)| !**seen) {
        eyre::bail!("missing gas param id: {id}");
    }

    Ok(GasParams::new(Arc::new(table)))
}

fn helper_init(config: &RuntimeConfig) -> HelperInit {
    HelperInit {
        debug_assertions: config.debug_assertions,
        no_dedup: config.no_dedup,
        no_dse: config.no_dse,
        dump_dir: config.dump_dir.as_ref().map(|path| path.to_string_lossy().into_owned()),
        gas_params: config.gas_params.as_ref().map(gas_params_to_pairs),
    }
}

fn runtime_config_from_init(init: HelperInit) -> eyre::Result<RuntimeConfig> {
    Ok(RuntimeConfig {
        dump_dir: init.dump_dir.map(PathBuf::from),
        debug_assertions: init.debug_assertions,
        no_dedup: init.no_dedup,
        no_dse: init.no_dse,
        gas_params: init.gas_params.map(gas_params_from_pairs).transpose()?,
        ..Default::default()
    })
}

fn job_dump_dir(config: &RuntimeConfig, job: &CompileJob) -> Option<PathBuf> {
    config.dump_dir.as_ref().map(|base| {
        base.join(format!("{:?}", job.key.spec_id)).join(format!("{}", job.key.code_hash))
    })
}

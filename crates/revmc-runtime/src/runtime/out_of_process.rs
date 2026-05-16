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
use revm_primitives::hardfork::SpecId;
use std::{
    io::{BufReader, BufWriter, Read, Write},
    ops::ControlFlow,
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, Mutex},
    thread::JoinHandle,
    time::{Duration, Instant},
};
use wait_timeout::ChildExt;
use wincode::{SchemaRead, SchemaWrite};

const HELPER_ENV: &str = "REVMC_JIT_HELPER";

/// Runs the out-of-process JIT helper if this process was launched as one.
pub(super) fn maybe_run_jit_helper() -> eyre::Result<ControlFlow<()>> {
    if std::env::var_os(HELPER_ENV).is_none() {
        return Ok(ControlFlow::Continue(()));
    }
    run_jit_helper_stdio()?;
    Ok(ControlFlow::Break(()))
}

/// Compiles a job in the out-of-process helper.
pub(super) fn compile_job(job: CompileJob, config: &RuntimeConfig) -> WorkerResult {
    let t0 = Instant::now();
    let outcome = run_helper_job(&job, config);
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

/// Cancels in-flight out-of-process helper work.
pub(super) fn cancel_in_flight() {
    HELPER_PROCESS.reset();
}

fn run_helper_job(job: &CompileJob, config: &RuntimeConfig) -> Result<WorkerSuccess, String> {
    if config.gas_params.is_some() {
        return Err("out-of-process JIT does not support custom gas params yet".into());
    }
    if config.dump_dir.is_some() {
        return Err("out-of-process JIT does not support debug dumps yet".into());
    }

    helper_process().compile(job, config)
}

static HELPER_PROCESS: HelperProcess = HelperProcess::new();

fn helper_process() -> &'static HelperProcess {
    &HELPER_PROCESS
}

struct HelperIo {
    stdin: BufWriter<ChildStdin>,
    result_rx: chan::Receiver<Result<WorkerSuccess, String>>,
}

struct HelperProcess {
    inner: Mutex<Option<Arc<HelperProcessInner>>>,
}

impl HelperProcess {
    const fn new() -> Self {
        Self { inner: Mutex::new(None) }
    }

    fn compile(&self, job: &CompileJob, config: &RuntimeConfig) -> Result<WorkerSuccess, String> {
        let helper = {
            let mut slot = self.inner.lock().unwrap();
            if slot.as_ref().is_none_or(|helper| !helper.matches_config(config)) {
                *slot = Some(Arc::new(HelperProcessInner::spawn(config)?));
            }
            slot.as_ref().unwrap().clone()
        };

        match helper.compile(job, config) {
            Ok(result) => Ok(result),
            Err(err) => {
                let mut slot = self.inner.lock().unwrap();
                if slot.as_ref().is_some_and(|current| Arc::ptr_eq(current, &helper)) {
                    *slot = None;
                }
                Err(err)
            }
        }
    }

    fn reset(&self) {
        if let Some(helper) = self.inner.lock().unwrap().take() {
            helper.kill();
        }
    }
}

struct HelperProcessInner {
    path: PathBuf,
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
        let mut command = Command::new(&path);
        command
            .env(HELPER_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        apply_helper_limits(&mut command, config);

        let mut child = command.spawn().map_err(|e| format!("failed to spawn JIT helper: {e}"))?;
        let stdin = BufWriter::new(child.stdin.take().ok_or("helper stdin unavailable")?);
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
            child: Mutex::new(child),
            io: Mutex::new(HelperIo { stdin, result_rx }),
            reader: Mutex::new(Some(reader)),
            shutdown_timeout: config.tuning.shutdown_timeout,
        })
    }

    fn matches_config(&self, config: &RuntimeConfig) -> bool {
        match &config.jit_helper_path {
            Some(path) => self.path == *path,
            None => std::env::current_exe().map(|path| self.path == path).unwrap_or(false),
        }
    }

    fn compile(&self, job: &CompileJob, config: &RuntimeConfig) -> Result<WorkerSuccess, String> {
        let mut io = self.io.lock().unwrap();
        write_job(&mut io.stdin, job, config)
            .map_err(|e| format!("failed to write helper job: {e}"))?;
        io.stdin.flush().map_err(|e| format!("failed to flush helper job: {e}"))?;

        match io.result_rx.recv_timeout(config.tuning.jit_timeout) {
            Ok(result) => result,
            Err(chan::RecvTimeoutError::Timeout) => {
                self.kill();
                Err(format!("JIT helper timed out after {:?}", config.tuning.jit_timeout))
            }
            Err(chan::RecvTimeoutError::Disconnected) => {
                let status = self.child.lock().unwrap().try_wait().ok().flatten();
                Err(match status {
                    Some(status) => format!("JIT helper exited with {status}"),
                    None => "JIT helper disconnected".into(),
                })
            }
        }
    }

    fn kill(&self) -> bool {
        let mut child = self.child.lock().unwrap();
        if matches!(child.try_wait(), Ok(Some(_))) {
            return true;
        }
        if let Err(err) = child.kill() {
            warn!(%err, "failed to kill JIT helper");
        }
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

#[cfg(unix)]
fn apply_helper_limits(command: &mut Command, config: &RuntimeConfig) {
    use std::os::unix::process::CommandExt;

    let memory_limit = config.tuning.jit_helper_memory_limit_bytes;
    let cpu_time = config.tuning.jit_helper_cpu_time;
    if memory_limit == 0 && cpu_time.is_none() {
        return;
    }

    // SAFETY: `pre_exec` runs in the child after fork and before exec. The closure only calls
    // async-signal-safe libc `setrlimit` and constructs an `io::Error` if it fails.
    unsafe {
        command.pre_exec(move || {
            if memory_limit > 0 {
                set_rlimit(libc::RLIMIT_AS as _, memory_limit)?;
            }
            if let Some(cpu_time) = cpu_time {
                let seconds = cpu_time.as_secs().max(1);
                set_rlimit(libc::RLIMIT_CPU as _, seconds)?;
            }
            Ok(())
        });
    }
}

#[cfg(unix)]
fn set_rlimit(resource: libc::c_int, value: u64) -> std::io::Result<()> {
    let value = libc::rlim_t::try_from(value).unwrap_or(libc::rlim_t::MAX);
    let limit = libc::rlimit { rlim_cur: value, rlim_max: value };
    if unsafe { libc::setrlimit(resource as _, &limit) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(unix))]
fn apply_helper_limits(_command: &mut Command, _config: &RuntimeConfig) {}

#[derive(SchemaWrite, SchemaRead)]
struct HelperRequest {
    code_hash: [u8; 32],
    spec_id: u8,
    opt_level: u8,
    debug_assertions: bool,
    no_dedup: bool,
    no_dse: bool,
    symbol_name: String,
    bytecode: Vec<u8>,
}

#[derive(SchemaWrite, SchemaRead)]
enum HelperResponse {
    Ok { symbol_name: String, object_bytes: Vec<u8>, builtin_symbols: Vec<String> },
    Err { error: String },
}

fn write_job<W: Write + ?Sized>(
    w: &mut BufWriter<W>,
    job: &CompileJob,
    config: &RuntimeConfig,
) -> std::io::Result<()> {
    let req = HelperRequest {
        code_hash: job.key.code_hash.0,
        spec_id: job.key.spec_id as u8,
        opt_level: opt_level_to_u8(job.opt_level),
        debug_assertions: config.debug_assertions,
        no_dedup: config.no_dedup,
        no_dse: config.no_dse,
        symbol_name: job.symbol_name.clone(),
        bytecode: job.bytecode.to_vec(),
    };
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

    while let Some((job, config)) = read_helper_job(&mut stdin)? {
        if compiler.is_none() {
            compiler = Some(create_compiler(&config, true).map_err(|e| eyre::eyre!(e))?);
        }
        let compiler = compiler.as_mut().unwrap();
        compiler.set_opt_level(job.opt_level);
        compiler.debug_assertions(config.debug_assertions);
        compiler.set_dedup(!config.no_dedup);
        compiler.set_dse(!config.no_dse);

        let result = compile_jit_object_artifact(&job, compiler);
        if let Err(err) = compiler.clear_ir() {
            warn!(%err, "clear_ir failed");
        }
        write_helper_result(&mut stdout, result)?;
        stdout.flush()?;
    }

    Ok(())
}

fn read_helper_job<R: Read + ?Sized>(
    stdin: &mut BufReader<R>,
) -> eyre::Result<Option<(CompileJob, RuntimeConfig)>> {
    let req: Option<HelperRequest> = read_message(stdin)?;
    let Some(req) = req else { return Ok(None) };
    let spec_id = SpecId::try_from_u8(req.spec_id).ok_or_else(|| eyre::eyre!("invalid spec id"))?;
    let opt_level = opt_level_from_u8(req.opt_level)?;

    let config = RuntimeConfig {
        debug_assertions: req.debug_assertions,
        no_dedup: req.no_dedup,
        no_dse: req.no_dse,
        ..Default::default()
    };
    let job = CompileJob {
        kind: CompilationKind::Jit,
        key: RuntimeCacheKey { code_hash: B256::from(req.code_hash), spec_id },
        bytecode: Bytes::from(req.bytecode),
        symbol_name: req.symbol_name,
        opt_level,
        sync_notifier: SyncNotifier::none(),
        generation: 0,
    };
    Ok(Some((job, config)))
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

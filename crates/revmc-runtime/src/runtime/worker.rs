//! Background JIT and AOT compilation workers.
//!
//! Each worker thread owns a long-lived `EvmCompiler` instance tied to its
//! thread-local LLVM context.
//!
//! With ORCv2, JIT code lifetime is managed per-module via `ResourceTracker`s.
//! Each successful JIT compilation extracts the committed module's tracker and
//! returns it as a [`JitCodeBacking`], which frees the machine code on drop.
//! Workers no longer need to stay alive for code lifetime — they exit as soon
//! as the job channel closes.

use crate::{
    CompileTimings, EvmCompilerFn, OptimizationLevel, eyre,
    runtime::{
        config::{CompilationKind, JitMode, RuntimeConfig},
        storage::RuntimeCacheKey,
    },
};
use alloy_primitives::{B256, Bytes};
use crossbeam_channel as chan;
use rayon::{ThreadPool, ThreadPoolBuilder};
#[cfg(feature = "llvm")]
use std::{
    cell::RefCell,
    fs::File,
    io::{Read, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, Stdio},
    thread::JoinHandle,
    time::Instant,
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

#[cfg(feature = "llvm")]
use crate::{
    EvmCompiler, EvmLlvmBackend, Linker,
    llvm::{JitDylibGuard, orc::ResourceTracker},
};
#[cfg(feature = "llvm")]
use revm_primitives::hardfork::SpecId;

/// Notifier for synchronous compilation requests.
///
/// Wraps an optional sender that is notified when the compilation
/// completes (success or failure). Passed from the backend through
/// the worker and back, then fired after the result is processed.
pub(crate) struct SyncNotifier(Option<chan::Sender<()>>);

impl SyncNotifier {
    pub(crate) fn none() -> Self {
        Self(None)
    }

    pub(crate) fn new(tx: chan::Sender<()>) -> Self {
        Self(Some(tx))
    }

    pub(crate) fn notify(self) {
        if let Some(tx) = self.0 {
            let _ = tx.send(());
        }
    }
}

/// A compilation job sent from the backend to a worker.
#[derive(derive_more::Debug)]
pub(crate) struct CompileJob {
    /// Whether this job compiles JIT or AOT output.
    pub(crate) kind: CompilationKind,
    /// The key to compile for.
    pub(crate) key: RuntimeCacheKey,
    /// The raw bytecode to compile.
    pub(crate) bytecode: Bytes,
    /// The symbol name to use for the compiled function.
    pub(crate) symbol_name: String,
    /// Optimization level for compilation.
    pub(crate) opt_level: OptimizationLevel,
    /// Optional notifier for synchronous callers.
    #[debug(skip)]
    pub(crate) sync_notifier: SyncNotifier,
    /// Generation at the time the job was dispatched.
    pub(crate) generation: u64,
}

/// Result of a compilation attempt, sent back from a worker to the backend.
pub(crate) struct WorkerResult {
    /// The key that was compiled.
    pub(crate) key: RuntimeCacheKey,
    /// The compilation outcome.
    pub(crate) outcome: Result<WorkerSuccess, String>,
    /// Whether this was a JIT or AOT compilation job.
    pub(crate) kind: CompilationKind,
    /// Optional notifier for synchronous callers, passed through from the job.
    pub(crate) sync_notifier: SyncNotifier,
    /// Generation at the time the job was dispatched.
    pub(crate) generation: u64,
    /// Wall-clock time spent compiling.
    pub(crate) compile_duration: Duration,
    /// Per-phase timing breakdown from the compiler.
    pub(crate) timings: CompileTimings,
}

/// Successful compilation output.
pub(crate) enum WorkerSuccess {
    /// JIT compilation produced an in-memory function pointer.
    Jit(JitSuccess),
    /// AOT compilation produced shared-library bytes.
    Aot(AotSuccess),
    /// JIT compilation produced relocatable object bytes to link in the parent.
    JitObject(JitObjectSuccess),
}

/// Successful JIT compilation output.
pub(crate) struct JitSuccess {
    /// The compiled function pointer.
    pub(crate) func: EvmCompilerFn,
    /// Owns the JIT machine code via an ORCv2 `ResourceTracker`.
    /// Dropping this frees the compiled code.
    pub(crate) backing: Arc<JitCodeBacking>,
}

/// Successful AOT compilation output.
pub(crate) struct JitObjectSuccess {
    /// The symbol name in the object file.
    pub(crate) symbol_name: String,
    /// The raw relocatable object bytes.
    pub(crate) object_bytes: Vec<u8>,
    /// Builtin absolute symbols referenced by the object.
    pub(crate) builtin_symbols: Vec<String>,
}

pub(crate) struct AotSuccess {
    /// The symbol name in the shared library.
    pub(crate) symbol_name: String,
    /// The raw shared-library bytes (.so / .dylib).
    pub(crate) dylib_bytes: Vec<u8>,
    /// Length of the original bytecode.
    pub(crate) bytecode_len: usize,
}

/// Owns JIT-compiled machine code via an ORCv2 `ResourceTracker` and a
/// [`JitDylibGuard`].
///
/// The tracker provides per-entry code removal: dropping it calls
/// `tracker.remove()` which frees this entry's machine code.
/// The guard keeps the owning `JITDylib` alive — it won't be cleared
/// or recycled until all guards are dropped.
///
/// This enables true per-entry eviction: removing a `CompiledProgram`
/// from the resident map reclaims its machine code once all callers
/// release their `Arc<CompiledProgram>` handles.
pub(crate) struct JitCodeBacking {
    /// The tracker that owns this entry's machine code.
    /// Must be dropped (after `remove()`) BEFORE `_jd_guard` to ensure
    /// the JITDylib is still valid when we remove resources from it.
    #[cfg(feature = "llvm")]
    tracker: Option<ResourceTracker>,
    /// Keeps the owning JITDylib alive. Dropped after `tracker`.
    #[cfg(feature = "llvm")]
    _jd_guard: Arc<JitDylibGuard>,
}

impl JitCodeBacking {
    #[cfg(feature = "llvm")]
    pub(crate) fn new(tracker: ResourceTracker, jd_guard: Arc<JitDylibGuard>) -> Self {
        Self { tracker: Some(tracker), _jd_guard: jd_guard }
    }
}

#[cfg(feature = "llvm")]
impl Drop for JitCodeBacking {
    fn drop(&mut self) {
        if let Some(tracker) = self.tracker.take()
            && let Err(e) = tracker.remove()
        {
            warn!("failed to remove JIT code: {e}");
        }
    }
}

/// Handle to the worker pool.
pub(crate) struct WorkerPool {
    /// Rayon pool used to execute compilation jobs.
    pool: Option<ThreadPool>,
    /// Sender for worker results.
    result_tx: chan::Sender<WorkerResult>,
    /// Runtime configuration shared by spawned jobs.
    config: Arc<RuntimeConfig>,
    /// Number of queued jobs waiting to start.
    queued: Arc<AtomicUsize>,
    /// Maximum queued jobs accepted by the pool.
    queue_capacity: usize,
    /// Signals workers to stop accepting new work.
    shutdown: Arc<AtomicBool>,
}

impl WorkerPool {
    /// Creates and starts the worker pool.
    pub(crate) fn new(result_tx: chan::Sender<WorkerResult>, config: RuntimeConfig) -> Self {
        let worker_count = config.tuning.jit_worker_count;
        let queue_capacity = worker_count.saturating_mul(config.tuning.jit_worker_queue_capacity);
        let pool = (worker_count > 0).then(|| {
            ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .thread_name(|i| format!("revmc-{i:02}"))
                .exit_handler(|_| clear_thread_local_compilers())
                .build()
                .expect("failed to spawn compile workers")
        });

        Self {
            pool,
            result_tx,
            config: Arc::new(config),
            queued: Arc::new(AtomicUsize::new(0)),
            queue_capacity,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Tries to send a job to the worker pool.
    /// Returns the job back on failure (queue full or shut down).
    pub(crate) fn try_send(&mut self, job: CompileJob) -> Result<(), CompileJob> {
        if self.pool.is_none() || self.shutdown.load(Ordering::Acquire) {
            return Err(job);
        }

        let mut current = self.queued.load(Ordering::Acquire);
        loop {
            if current >= self.queue_capacity {
                return Err(job);
            }
            match self.queued.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }

        let pool = self.pool.as_ref().unwrap();
        let queued = self.queued.clone();
        let result_tx = self.result_tx.clone();
        let config = self.config.clone();
        pool.spawn_fifo(move || {
            queued.fetch_sub(1, Ordering::AcqRel);
            let result = compile_job(job, &config);
            let _ = result_tx.send(result);
        });
        Ok(())
    }

    /// Shuts down all workers after draining queued jobs.
    pub(crate) fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(pool) = &self.pool {
            pool.broadcast(|_| clear_thread_local_compilers());
        }
        self.pool.take();
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(feature = "llvm")]
fn clear_thread_local_compilers() {
    JIT_COMPILER.with_borrow_mut(Option::take);
    AOT_COMPILER.with_borrow_mut(Option::take);
    JIT_HELPER.with_borrow_mut(Option::take);
}

#[cfg(not(feature = "llvm"))]
fn clear_thread_local_compilers() {}

#[cfg(feature = "llvm")]
fn compile_job(job: CompileJob, config: &RuntimeConfig) -> WorkerResult {
    trace!(?job, "received job");
    match job.kind {
        CompilationKind::Jit if config.jit_mode == JitMode::OutOfProcess => {
            compile_job_out_of_process(job, config)
        }
        CompilationKind::Jit => {
            JIT_COMPILER.with_borrow_mut(|state| compile_with_state(job, config, state))
        }
        CompilationKind::Aot => {
            AOT_COMPILER.with_borrow_mut(|state| compile_with_state(job, config, state))
        }
    }
}

#[cfg(feature = "llvm")]
fn compile_job_out_of_process(job: CompileJob, config: &RuntimeConfig) -> WorkerResult {
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

const HELPER_ENV: &str = "REVMC_JIT_HELPER";

#[cfg(feature = "llvm")]
fn run_helper_job(job: &CompileJob, config: &RuntimeConfig) -> Result<WorkerSuccess, String> {
    if config.gas_params.is_some() {
        return Err("out-of-process JIT does not support custom gas params yet".into());
    }
    if config.dump_dir.is_some() {
        return Err("out-of-process JIT does not support debug dumps yet".into());
    }

    JIT_HELPER.with_borrow_mut(|slot| {
        if slot.as_ref().is_none_or(|helper| !helper.matches_config(config)) {
            *slot = Some(HelperProcess::spawn(config)?);
        }

        let helper = slot.as_mut().unwrap();
        match helper.compile(job, config) {
            Ok(result) => Ok(result),
            Err(err) => {
                *slot = None;
                Err(err)
            }
        }
    })
}

#[cfg(feature = "llvm")]
struct HelperProcess {
    path: PathBuf,
    child: Child,
    stdin: ChildStdin,
    result_rx: chan::Receiver<Result<WorkerSuccess, String>>,
    reader: Option<JoinHandle<()>>,
}

#[cfg(feature = "llvm")]
impl HelperProcess {
    fn spawn(config: &RuntimeConfig) -> Result<Self, String> {
        let path = match &config.jit_helper_path {
            Some(path) => path.clone(),
            None => {
                std::env::current_exe().map_err(|e| format!("failed to locate current exe: {e}"))?
            }
        };
        let mut child = Command::new(&path)
            .env(HELPER_ENV, "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("failed to spawn JIT helper: {e}"))?;
        let stdin = child.stdin.take().ok_or("helper stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("helper stdout unavailable")?;
        let (result_tx, result_rx) = chan::bounded(1);
        let reader = std::thread::spawn(move || {
            let mut stdout = stdout;
            loop {
                let result = read_helper_result(&mut stdout);
                if result_tx.send(result).is_err() {
                    break;
                }
            }
        });
        Ok(Self { path, child, stdin, result_rx, reader: Some(reader) })
    }

    fn matches_config(&self, config: &RuntimeConfig) -> bool {
        match &config.jit_helper_path {
            Some(path) => self.path == *path,
            None => std::env::current_exe().map(|path| self.path == path).unwrap_or(false),
        }
    }

    fn compile(
        &mut self,
        job: &CompileJob,
        config: &RuntimeConfig,
    ) -> Result<WorkerSuccess, String> {
        write_job(&mut self.stdin, job, config)
            .map_err(|e| format!("failed to write helper job: {e}"))?;
        self.stdin.flush().map_err(|e| format!("failed to flush helper job: {e}"))?;

        match self.result_rx.recv_timeout(config.tuning.jit_helper_timeout) {
            Ok(result) => result,
            Err(chan::RecvTimeoutError::Timeout) => {
                let _ = self.child.kill();
                Err(format!("JIT helper timed out after {:?}", config.tuning.jit_helper_timeout))
            }
            Err(chan::RecvTimeoutError::Disconnected) => {
                let status = self.child.try_wait().ok().flatten();
                Err(match status {
                    Some(status) => format!("JIT helper exited with {status}"),
                    None => "JIT helper disconnected".into(),
                })
            }
        }
    }
}

#[cfg(feature = "llvm")]
impl Drop for HelperProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

#[cfg(feature = "llvm")]
fn write_job(mut w: impl Write, job: &CompileJob, config: &RuntimeConfig) -> std::io::Result<()> {
    w.write_all(b"RJIT\0")?;
    w.write_all(&job.key.code_hash.0)?;
    w.write_all(&[job.key.spec_id as u8, opt_level_to_u8(job.opt_level)])?;
    w.write_all(&[
        u8::from(config.debug_assertions),
        u8::from(config.no_dedup),
        u8::from(config.no_dse),
    ])?;
    write_bytes(&mut w, job.symbol_name.as_bytes())?;
    write_bytes(&mut w, &job.bytecode)?;
    Ok(())
}

#[cfg(feature = "llvm")]
fn read_helper_result(mut r: impl Read) -> Result<WorkerSuccess, String> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag).map_err(|e| format!("failed to read helper result: {e}"))?;
    match tag[0] {
        0 => {
            let msg =
                read_string(&mut r).map_err(|e| format!("failed to read helper error: {e}"))?;
            Err(msg)
        }
        1 => {
            let symbol_name =
                read_string(&mut r).map_err(|e| format!("failed to read symbol name: {e}"))?;
            let object_bytes =
                read_vec(&mut r).map_err(|e| format!("failed to read object bytes: {e}"))?;
            let count =
                read_u32(&mut r).map_err(|e| format!("failed to read symbol count: {e}"))?;
            let mut builtin_symbols = Vec::with_capacity(count as usize);
            for _ in 0..count {
                builtin_symbols.push(
                    read_string(&mut r)
                        .map_err(|e| format!("failed to read builtin symbol: {e}"))?,
                );
            }
            Ok(WorkerSuccess::JitObject(JitObjectSuccess {
                symbol_name,
                object_bytes,
                builtin_symbols,
            }))
        }
        tag => Err(format!("unknown helper result tag {tag}")),
    }
}

#[cfg(feature = "llvm")]
pub(super) fn run_jit_helper_stdio() -> eyre::Result<()> {
    let mut stdin = std::io::stdin().lock();
    let mut stdout = std::io::stdout().lock();
    let mut compiler: Option<EvmCompiler<EvmLlvmBackend>> = None;

    while let Some((job, config)) = read_helper_job(&mut stdin)? {
        if compiler.is_none() {
            compiler = Some(create_compiler(&config, false).map_err(|e| eyre::eyre!(e))?);
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

#[cfg(feature = "llvm")]
fn read_helper_job(mut stdin: impl Read) -> eyre::Result<Option<(CompileJob, RuntimeConfig)>> {
    let mut magic = [0u8; 5];
    match stdin.read_exact(&mut magic) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    eyre::ensure!(&magic == b"RJIT\0", "invalid JIT helper request");

    let mut hash = [0u8; 32];
    stdin.read_exact(&mut hash)?;
    let mut fixed = [0u8; 5];
    stdin.read_exact(&mut fixed)?;
    let spec_id = SpecId::try_from_u8(fixed[0]).ok_or_else(|| eyre::eyre!("invalid spec id"))?;
    let opt_level = opt_level_from_u8(fixed[1])?;
    let symbol_name = read_string(&mut stdin)?;
    let bytecode = Bytes::from(read_vec(&mut stdin)?);

    let config = RuntimeConfig {
        debug_assertions: fixed[2] != 0,
        no_dedup: fixed[3] != 0,
        no_dse: fixed[4] != 0,
        ..Default::default()
    };
    let job = CompileJob {
        kind: CompilationKind::Jit,
        key: RuntimeCacheKey { code_hash: B256::from(hash), spec_id },
        bytecode,
        symbol_name,
        opt_level,
        sync_notifier: SyncNotifier::none(),
        generation: 0,
    };
    Ok(Some((job, config)))
}

#[cfg(feature = "llvm")]
fn write_helper_result(
    mut stdout: impl Write,
    result: Result<WorkerSuccess, String>,
) -> eyre::Result<()> {
    match result {
        Ok(WorkerSuccess::JitObject(success)) => {
            stdout.write_all(&[1])?;
            write_bytes(&mut stdout, success.symbol_name.as_bytes())?;
            write_bytes(&mut stdout, &success.object_bytes)?;
            stdout.write_all(&(success.builtin_symbols.len() as u32).to_le_bytes())?;
            for symbol in success.builtin_symbols {
                write_bytes(&mut stdout, symbol.as_bytes())?;
            }
        }
        Ok(_) => unreachable!(),
        Err(err) => {
            stdout.write_all(&[0])?;
            write_bytes(&mut stdout, err.as_bytes())?;
        }
    }
    Ok(())
}

#[cfg(feature = "llvm")]
fn write_bytes(mut w: impl Write, bytes: &[u8]) -> std::io::Result<()> {
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)
}

#[cfg(feature = "llvm")]
fn read_string(r: impl Read) -> std::io::Result<String> {
    String::from_utf8(read_vec(r)?)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

#[cfg(feature = "llvm")]
fn read_vec(mut r: impl Read) -> std::io::Result<Vec<u8>> {
    let len = read_u32(&mut r)? as usize;
    let mut bytes = vec![0; len];
    r.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[cfg(feature = "llvm")]
fn read_u32(mut r: impl Read) -> std::io::Result<u32> {
    let mut bytes = [0u8; 4];
    r.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

#[cfg(feature = "llvm")]
fn opt_level_to_u8(level: OptimizationLevel) -> u8 {
    match level {
        OptimizationLevel::None => 0,
        OptimizationLevel::Less => 1,
        OptimizationLevel::Default => 2,
        OptimizationLevel::Aggressive => 3,
    }
}

#[cfg(feature = "llvm")]
fn opt_level_from_u8(level: u8) -> eyre::Result<OptimizationLevel> {
    Ok(match level {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        3 => OptimizationLevel::Aggressive,
        _ => eyre::bail!("invalid optimization level"),
    })
}

#[cfg(feature = "llvm")]
fn compile_with_state(
    job: CompileJob,
    config: &RuntimeConfig,
    state_slot: &mut Option<CompilerState>,
) -> WorkerResult {
    let _span = match job.kind {
        CompilationKind::Jit => {
            debug_span!("jit_compile", hash=%job.key.code_hash, spec_id=?job.key.spec_id).entered()
        }
        CompilationKind::Aot => {
            debug_span!("aot_compile", hash=%job.key.code_hash, spec_id=?job.key.spec_id).entered()
        }
    };
    let t0 = Instant::now();

    if state_slot.is_none() {
        match CompilerState::new(config, job.kind) {
            Ok(s) => *state_slot = Some(s),
            Err(e) => {
                error!(error = %e, "failed to create LLVM backend");
                return WorkerResult {
                    key: job.key,
                    outcome: Err(e),
                    kind: job.kind,
                    sync_notifier: job.sync_notifier,
                    generation: job.generation,
                    compile_duration: t0.elapsed(),
                    timings: CompileTimings::default(),
                };
            }
        }
    }

    let state = state_slot.as_mut().unwrap();
    let compiler = &mut state.compiler;

    if job.kind == CompilationKind::Jit
        && let Some(base) = &config.dump_dir
    {
        let dir =
            base.join(format!("{:?}", job.key.spec_id)).join(format!("{}", job.key.code_hash));
        compiler.set_dump_to(Some(dir));
    }
    compiler.set_opt_level(job.opt_level);

    let outcome = match job.kind {
        CompilationKind::Jit if config.jit_mode == JitMode::OutOfProcess => {
            compile_jit_object_artifact(&job, compiler)
        }
        CompilationKind::Jit => compile_jit_artifact(&job, compiler),
        CompilationKind::Aot => compile_aot_artifact(&job, compiler),
    };
    let timings = compiler.take_timings();

    if let Err(err) = compiler.clear_ir() {
        warn!(%err, "clear_ir failed");
    }

    state.compilations_since_recycle += 1;
    if config.tuning.compiler_recycle_threshold > 0
        && state.compilations_since_recycle >= config.tuning.compiler_recycle_threshold
    {
        debug!(compilations_since_recycle = state.compilations_since_recycle, "recycling compiler");
        match CompilerState::new(config, job.kind) {
            Ok(new_state) => {
                *state_slot = Some(new_state);
                revmc_llvm::global_gc();
            }
            Err(e) => {
                error!(error = %e, "failed to recreate compiler");
                state.compilations_since_recycle = 0;
            }
        }
    }

    WorkerResult {
        key: job.key,
        outcome,
        kind: job.kind,
        sync_notifier: job.sync_notifier,
        generation: job.generation,
        compile_duration: t0.elapsed(),
        timings,
    }
}

#[cfg(feature = "llvm")]
struct CompilerState {
    compiler: EvmCompiler<EvmLlvmBackend>,
    compilations_since_recycle: usize,
}

#[cfg(feature = "llvm")]
impl CompilerState {
    fn new(config: &RuntimeConfig, kind: CompilationKind) -> Result<Self, String> {
        Ok(Self {
            compiler: create_compiler(config, kind == CompilationKind::Aot)?,
            compilations_since_recycle: 0,
        })
    }
}

#[cfg(feature = "llvm")]
thread_local! {
    static JIT_COMPILER: RefCell<Option<CompilerState>> = const { RefCell::new(None) };
    static AOT_COMPILER: RefCell<Option<CompilerState>> = const { RefCell::new(None) };
    static JIT_HELPER: RefCell<Option<HelperProcess>> = const { RefCell::new(None) };
}

#[cfg(feature = "llvm")]
fn create_compiler(
    config: &RuntimeConfig,
    aot: bool,
) -> Result<EvmCompiler<EvmLlvmBackend>, String> {
    let backend = EvmLlvmBackend::new(aot).map_err(|e| e.to_string())?;
    let mut compiler = EvmCompiler::new(backend);
    compiler.set_opt_level(if aot {
        config.tuning.aot_opt_level
    } else {
        config.tuning.jit_opt_level
    });
    compiler.debug_assertions(config.debug_assertions);
    compiler.set_dedup(!config.no_dedup);
    compiler.set_dse(!config.no_dse);
    if let Some(gas_params) = &config.gas_params {
        compiler.set_gas_params(gas_params.clone());
    }
    Ok(compiler)
}

#[cfg(feature = "llvm")]
fn compile_jit_artifact(
    job: &CompileJob,
    compiler: &mut EvmCompiler<EvmLlvmBackend>,
) -> Result<WorkerSuccess, String> {
    let result = unsafe { compiler.jit(&job.symbol_name, &job.bytecode[..], job.key.spec_id) };
    match result {
        Ok(func) => {
            let jd_guard = compiler.backend_mut().jit_dylib_guard();
            debug!("JIT compilation succeeded");
            let tracker = compiler
                .backend_mut()
                .take_last_resource_tracker()
                .expect("no ResourceTracker after JIT");
            let backing = Arc::new(JitCodeBacking::new(tracker, jd_guard));
            Ok(WorkerSuccess::Jit(JitSuccess { func, backing }))
        }
        Err(err) => {
            warn!(%err, "JIT compilation failed");
            Err(format!("{err}"))
        }
    }
}

/// Compiles a single bytecode to a shared library and returns the raw bytes.
#[cfg(feature = "llvm")]
fn compile_jit_object_artifact(
    job: &CompileJob,
    compiler: &mut EvmCompiler<EvmLlvmBackend>,
) -> Result<WorkerSuccess, String> {
    compiler
        .translate(&job.symbol_name, &job.bytecode[..], job.key.spec_id)
        .map_err(|e| format!("JIT object translate failed: {e}"))?;

    let mut object_bytes = Vec::new();
    compiler
        .write_object(&mut object_bytes)
        .map_err(|e| format!("JIT object write failed: {e}"))?;
    let builtin_symbols = compiler
        .backend()
        .pending_symbol_names()
        .into_iter()
        .map(|name| name.to_string_lossy().into_owned())
        .collect();

    debug!(
        bytecode_len = job.bytecode.len(),
        object_len = object_bytes.len(),
        "JIT object compilation succeeded",
    );

    Ok(WorkerSuccess::JitObject(JitObjectSuccess {
        symbol_name: job.symbol_name.clone(),
        object_bytes,
        builtin_symbols,
    }))
}

fn compile_aot_artifact(
    job: &CompileJob,
    compiler: &mut EvmCompiler<EvmLlvmBackend>,
) -> Result<WorkerSuccess, String> {
    compiler
        .translate(&job.symbol_name, &job.bytecode[..], job.key.spec_id)
        .map_err(|e| format!("AOT translate failed: {e}"))?;

    let tmp_dir = tempfile::tempdir().map_err(|e| format!("failed to create temp dir: {e}"))?;

    let obj_path = tmp_dir.path().join("a.o");
    let so_path = tmp_dir.path().join("a.so");

    compiler
        .write_object_to_file(&obj_path)
        .map_err(|e| format!("AOT write object failed: {e}"))?;

    let linker = Linker::new();
    linker
        .link(&so_path, [obj_path.to_str().unwrap()])
        .map_err(|e| format!("AOT link failed: {e}"))?;

    let mut dylib_bytes = Vec::new();
    File::open(&so_path)
        .and_then(|mut f| f.read_to_end(&mut dylib_bytes))
        .map_err(|e| format!("failed to read linked .so: {e}"))?;

    debug!(
        bytecode_len = job.bytecode.len(),
        dylib_len = dylib_bytes.len(),
        "AOT compilation succeeded",
    );

    Ok(WorkerSuccess::Aot(AotSuccess {
        symbol_name: job.symbol_name.clone(),
        dylib_bytes,
        bytecode_len: job.bytecode.len(),
    }))
}

#[cfg(not(feature = "llvm"))]
fn compile_job(job: CompileJob, _config: &RuntimeConfig) -> WorkerResult {
    WorkerResult {
        key: job.key,
        outcome: Err("LLVM backend not available".into()),
        kind: job.kind,
        sync_notifier: job.sync_notifier,
        generation: job.generation,
        compile_duration: Duration::ZERO,
        timings: CompileTimings::default(),
    }
}

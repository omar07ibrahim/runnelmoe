//! Fixed-width asynchronous dispatch for the authoritative synchronous reader.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use tokio::sync::oneshot;

use crate::{Control, PageSpec, ReadStats, StoreError, SyncReader, VerifiedPage};

/// Fixed resource limits for asynchronous page reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AsyncReaderConfig {
    worker_threads: usize,
    queue_capacity: usize,
}

impl AsyncReaderConfig {
    /// Hard ceiling preventing untrusted configuration from attempting an
    /// unbounded worker allocation.
    pub const MAX_WORKER_THREADS: usize = 256;
    /// Hard ceiling for queued, worker-owned requests.
    pub const MAX_QUEUE_CAPACITY: usize = 65_536;

    /// Creates and validates an asynchronous reader configuration.
    pub fn new(worker_threads: usize, queue_capacity: usize) -> Result<Self, StoreError> {
        if worker_threads == 0 {
            return Err(StoreError::invalid_config(
                "worker_threads",
                "must be greater than zero",
            ));
        }
        if worker_threads > Self::MAX_WORKER_THREADS {
            return Err(StoreError::invalid_config(
                "worker_threads",
                "exceeds the hard worker-thread ceiling",
            ));
        }
        if queue_capacity == 0 {
            return Err(StoreError::invalid_config(
                "queue_capacity",
                "must be greater than zero",
            ));
        }
        if queue_capacity > Self::MAX_QUEUE_CAPACITY {
            return Err(StoreError::invalid_config(
                "queue_capacity",
                "exceeds the hard queue-capacity ceiling",
            ));
        }
        Ok(Self {
            worker_threads,
            queue_capacity,
        })
    }

    #[must_use]
    pub const fn worker_threads(self) -> usize {
        self.worker_threads
    }

    #[must_use]
    pub const fn queue_capacity(self) -> usize {
        self.queue_capacity
    }
}

impl Default for AsyncReaderConfig {
    fn default() -> Self {
        Self {
            worker_threads: 2,
            queue_capacity: 32,
        }
    }
}

type ObservedRead = (Result<VerifiedPage, StoreError>, ReadStats);

enum Completion {
    Await(oneshot::Sender<ObservedRead>),
    Detached(Box<dyn FnOnce(ObservedRead) + Send + 'static>),
}

struct Request {
    spec: PageSpec,
    control: Control,
    completion: Completion,
    #[cfg(test)]
    gate: Option<WorkerGate>,
}

struct Inner {
    sender: Mutex<Option<SyncSender<Request>>>,
    workers: Mutex<Vec<JoinHandle<()>>>,
    accepting: AtomicBool,
    #[cfg(test)]
    worker_gate: Mutex<Option<WorkerGate>>,
}

impl Inner {
    fn close(&self) {
        self.accepting.store(false, Ordering::Release);
        let mut sender = lock_unpoisoned(&self.sender);
        let _ = sender.take();
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.accepting.store(false, Ordering::Release);
        let _ = lock_unpoisoned(&self.sender).take();
        let current_thread = thread::current().id();
        for worker in lock_unpoisoned(&self.workers).drain(..) {
            // A worker owns all request inputs and output until it terminates.
            // Joining here ensures the final reader drop cannot outlive them.
            // A detached completion may release the final reader owner on the
            // reader worker itself. That worker observes the disconnected
            // channel and exits after the callback, so it must be detached
            // rather than joined from itself.
            if worker.thread().id() != current_thread {
                let _ = worker.join();
            }
        }
    }
}

/// A cloneable asynchronous reader backed by a fixed worker set and bounded
/// admission queue.
#[derive(Clone)]
pub struct AsyncReader {
    inner: Arc<Inner>,
}

impl AsyncReader {
    /// Starts exactly `config.worker_threads()` reader threads.
    pub fn new(reader: SyncReader, config: AsyncReaderConfig) -> Result<Self, StoreError> {
        // Revalidate values even if a future constructor or deserializer is
        // added; resource-bearing code should not rely only on type privacy.
        let config = AsyncReaderConfig::new(config.worker_threads, config.queue_capacity)?;
        let (sender, receiver) = mpsc::sync_channel(config.queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut workers = Vec::with_capacity(config.worker_threads);

        for index in 0..config.worker_threads {
            let worker_receiver = Arc::clone(&receiver);
            let worker_reader = reader.clone();
            let name = format!("runnel-page-reader-{index}");
            match thread::Builder::new()
                .name(name)
                .spawn(move || worker_loop(&worker_reader, &worker_receiver))
            {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    drop(sender);
                    drop(receiver);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(StoreError::io("spawning async reader worker", &error));
                }
            }
        }

        Ok(Self {
            inner: Arc::new(Inner {
                sender: Mutex::new(Some(sender)),
                workers: Mutex::new(workers),
                accepting: AtomicBool::new(true),
                #[cfg(test)]
                worker_gate: Mutex::new(None),
            }),
        })
    }

    /// Reads and authenticates one page without blocking an async executor
    /// thread. A full queue fails immediately and does not allocate an
    /// unbounded waiter.
    pub async fn read(
        &self,
        spec: PageSpec,
        control: Control,
    ) -> Result<(VerifiedPage, ReadStats), StoreError> {
        let (result, stats) = self.read_observed(spec, control).await;
        result.map(|page| (page, stats))
    }

    pub(crate) async fn read_observed(&self, spec: PageSpec, control: Control) -> ObservedRead {
        if let Err(error) = control.check() {
            return (Err(error), ReadStats::default());
        }

        let (completion, result) = oneshot::channel();
        let request = Request {
            spec,
            control,
            completion: Completion::Await(completion),
            #[cfg(test)]
            gate: lock_unpoisoned(&self.inner.worker_gate).clone(),
        };
        if let Err(error) = self.submit(request) {
            return (Err(error), ReadStats::default());
        }

        // The worker retains the PageSpec, Control, and any in-progress buffer
        // even if this future is dropped. A reported terminal result is
        // therefore always after buffer ownership has returned from read().
        result.await.unwrap_or_else(|_| {
            (
                Err(StoreError::invariant(
                    "async reader worker lost a completion",
                )),
                ReadStats::default(),
            )
        })
    }

    /// Submits a read whose completion is owned by the fixed reader worker,
    /// independently of any Tokio task or runtime lifetime.
    pub(crate) fn submit_observed(
        &self,
        spec: PageSpec,
        control: Control,
        completion: impl FnOnce(ObservedRead) + Send + 'static,
    ) -> Result<(), StoreError> {
        control.check()?;
        self.submit(Request {
            spec,
            control,
            completion: Completion::Detached(Box::new(completion)),
            #[cfg(test)]
            gate: lock_unpoisoned(&self.inner.worker_gate).clone(),
        })
    }

    fn submit(&self, request: Request) -> Result<(), StoreError> {
        if !self.inner.accepting.load(Ordering::Acquire) {
            return Err(StoreError::shutdown());
        }
        let sender = lock_unpoisoned(&self.inner.sender);
        let Some(sender) = sender.as_ref() else {
            return Err(StoreError::shutdown());
        };
        match sender.try_send(request) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_request)) => Err(StoreError::queue_full()),
            Err(TrySendError::Disconnected(_request)) => Err(StoreError::shutdown()),
        }
    }

    #[cfg(test)]
    pub(crate) fn install_worker_gate(&self, gate: WorkerGate) {
        *lock_unpoisoned(&self.inner.worker_gate) = Some(gate);
    }

    /// Stops accepting work. Already admitted reads run to completion, keeping
    /// their buffers worker-owned until their completion channel is resolved.
    pub fn shutdown(&self) {
        self.inner.close();
    }

    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        !self.inner.accepting.load(Ordering::Acquire)
    }
}

impl fmt::Debug for AsyncReader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AsyncReader")
            .field("is_shutdown", &self.is_shutdown())
            .finish_non_exhaustive()
    }
}

fn worker_loop(reader: &SyncReader, receiver: &Mutex<Receiver<Request>>) {
    loop {
        let request = {
            // Receiver is not cloneable. The lock protects only dequeue, never
            // the positional read, so all fixed workers can execute in parallel.
            let receiver = lock_unpoisoned(receiver);
            receiver.recv()
        };
        let Ok(request) = request else {
            break;
        };

        #[cfg(test)]
        if let Some(gate) = request.gate {
            gate.arrive_and_wait();
        }
        let result = reader.read_observed(&request.spec, &request.control);
        // A dropped async caller intentionally discards the result here; the
        // worker still owned it through verification and destruction.
        match request.completion {
            Completion::Await(completion) => {
                let _ = completion.send(result);
            }
            Completion::Detached(completion) => completion(result),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) struct WorkerGate {
    state: Arc<(std::sync::Condvar, Mutex<WorkerGateState>)>,
}

#[cfg(test)]
#[derive(Debug, Default)]
struct WorkerGateState {
    entered: bool,
    released: bool,
}

#[cfg(test)]
impl WorkerGate {
    pub(crate) fn new() -> Self {
        Self {
            state: Arc::new((
                std::sync::Condvar::new(),
                Mutex::new(WorkerGateState::default()),
            )),
        }
    }

    fn arrive_and_wait(&self) {
        let (condition, state) = &*self.state;
        let mut state = lock_unpoisoned(state);
        state.entered = true;
        condition.notify_all();
        while !state.released {
            state = match condition.wait(state) {
                Ok(state) => state,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
    }

    pub(crate) fn wait_until_entered(&self) {
        let (condition, state) = &*self.state;
        let mut state = lock_unpoisoned(state);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !state.entered {
            let now = std::time::Instant::now();
            if now >= deadline {
                state.released = true;
                condition.notify_all();
                panic!("reader worker did not reach its deterministic test gate");
            }
            let (next, timeout) = match condition.wait_timeout(state, deadline - now) {
                Ok(result) => result,
                Err(poisoned) => poisoned.into_inner(),
            };
            state = next;
            if timeout.timed_out() && !state.entered {
                state.released = true;
                condition.notify_all();
                panic!("reader worker did not reach its deterministic test gate");
            }
        }
    }

    pub(crate) fn release(&self) {
        let (condition, state) = &*self.state;
        let mut state = lock_unpoisoned(state);
        state.released = true;
        condition.notify_all();
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::AsyncReaderConfig;
    use crate::ErrorCategory;

    #[test]
    fn rejects_unbounded_or_workerless_configuration() {
        let workerless = AsyncReaderConfig::new(0, 1).expect_err("zero workers must fail");
        let unbounded = AsyncReaderConfig::new(1, 0).expect_err("zero queue must fail");
        assert_eq!(workerless.category(), ErrorCategory::InvalidInput);
        assert_eq!(unbounded.category(), ErrorCategory::InvalidInput);
    }

    #[test]
    fn exposes_validated_limits() {
        let config = AsyncReaderConfig::new(3, 7).expect("valid limits");
        assert_eq!(config.worker_threads(), 3);
        assert_eq!(config.queue_capacity(), 7);
    }

    #[test]
    fn rejects_capacity_overflow_configuration() {
        assert!(AsyncReaderConfig::new(usize::MAX, 1).is_err());
        assert!(AsyncReaderConfig::new(1, usize::MAX).is_err());
    }
}

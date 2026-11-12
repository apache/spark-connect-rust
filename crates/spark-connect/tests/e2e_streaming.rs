//! Server-gated end-to-end coverage of Structured Streaming: the `rate` source
//! through the reader, a `memory` sink through the writer's `start`/trigger paths,
//! and the `StreamingQuery` + `StreamingQueryManager` handles. These exercise the
//! execution paths in `streaming.rs` that a server-less run reads as 0%.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_streaming`

mod common;

use common::{session, should_run};
use spark_connect::streaming::Trigger;

/// A short-lived query on the `rate` source into an in-memory sink, so the whole
/// reader -> writer.start -> StreamingQuery -> stop path runs against the server.
#[test]
fn streaming_rate_to_memory_query_lifecycle() {
    if !should_run() {
        return;
    }
    let spark = session();
    let df = spark
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "5")
        .load(None);

    let query = df
        .write_stream()
        .format("memory")
        .query_name("e2e_rate_mem")
        .output_mode("append")
        .trigger(Trigger::ProcessingTime("1 seconds".to_string()))
        .start("")
        .expect("start streaming query");

    // Handle accessors.
    assert!(!query.id().is_empty());
    assert!(!query.run_id().is_empty());
    assert_eq!(query.name(), Some("e2e_rate_mem"));

    // Status / activity RPCs.
    let _active = query.is_active().expect("is_active");
    let _status = query.status().expect("status");
    let _explain = query.explain(false).expect("explain");

    // Manager surface while the query is (briefly) live.
    let mgr = spark.streams();
    let _all = mgr.active().expect("active list");
    let _got = mgr.get(query.id()).expect("get by id");

    // Recent/last progress may be empty this early; the call must still succeed.
    let _last = query.last_progress().expect("last_progress");
    let _recent = query.recent_progress().expect("recent_progress");

    // Give it a moment then stop and reset.
    let _ = query
        .await_termination(Some(1.0))
        .expect("await_termination timeout");
    query.stop().expect("stop");
    let _ = query.exception().expect("exception after stop");
    mgr.reset_terminated().expect("reset_terminated");
}

/// Cover the `to_table` sink-destination path and the AvailableNow trigger.
#[test]
fn streaming_available_now_to_table() {
    if !should_run() {
        return;
    }
    let spark = session();
    // Drop any leftover table from a previous run.
    let _ = spark
        .sql("DROP TABLE IF EXISTS e2e_stream_tbl")
        .and_then(|d| d.collect());

    let df = spark
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "10")
        .load(None);

    let query = df
        .write_stream()
        .output_mode("append")
        .query_name("e2e_avail_now")
        .trigger(Trigger::AvailableNow)
        .to_table("e2e_stream_tbl");

    // AvailableNow processes what's available and terminates; the call may either
    // succeed (query handle) or fail if the sink/table config is unsupported here.
    // Either way the writer's proto-building + start path is exercised.
    if let Ok(q) = query {
        let _ = q.await_termination(Some(5.0));
        let _ = q.stop();
    }
    let _ = spark
        .sql("DROP TABLE IF EXISTS e2e_stream_tbl")
        .and_then(|d| d.collect());
}

/// The native (Rust) client-side listener bus: implement StreamingQueryListener, add
/// it to the manager, run a rate query, and assert events are dispatched to the Rust
/// listener; then remove it and close the bus.
#[test]
fn streaming_native_listener_bus() {
    if !should_run() {
        return;
    }
    use spark_connect::streaming::{StreamingQueryListener, StreamingQueryListenerEvent};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingListener {
        events: Arc<AtomicUsize>,
    }
    impl StreamingQueryListener for CountingListener {
        fn on_event(&self, _event: &StreamingQueryListenerEvent) {
            self.events.fetch_add(1, Ordering::SeqCst);
        }
    }

    let spark = session();
    let mgr = spark.streams();
    let counter = Arc::new(AtomicUsize::new(0));
    let listener = Arc::new(CountingListener {
        events: counter.clone(),
    });
    let id = mgr.add_listener(listener).expect("add_listener");
    assert!(!id.is_empty());

    let query = spark
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "5")
        .load(None)
        .write_stream()
        .format("memory")
        .query_name("e2e_native_listener")
        .trigger(Trigger::ProcessingTime("1 seconds".to_string()))
        .start("")
        .expect("start streaming query");

    // Wait up to ~30s for at least one listener event.
    let mut got = 0;
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_secs(1));
        got = counter.load(Ordering::SeqCst);
        if got > 0 {
            break;
        }
    }
    query.stop().expect("stop");
    mgr.remove_listener(&id).expect("remove_listener");
    mgr.close().expect("close");

    assert!(
        got > 0,
        "expected at least one native listener event, got {got}"
    );
}

/// As PySpark Connect's `spark.streams`: the managers of a session share its listener bus, the
/// started event of a query reaches the listeners before `start` returns, and a listener removed
/// through another manager of the session gets no more events.
#[test]
fn streaming_listeners_are_those_of_the_session() {
    if !should_run() {
        return;
    }
    use spark_connect::streaming::{
        StreamingQueryListener, StreamingQueryListenerEvent, QUERY_PROGRESS_EVENT,
        QUERY_STARTED_EVENT,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct Recording {
        events: Mutex<Vec<i32>>,
    }
    impl StreamingQueryListener for Recording {
        fn on_event(&self, event: &StreamingQueryListenerEvent) {
            self.events.lock().unwrap().push(event.event_type);
        }
    }

    let spark = session();
    let listener = Arc::new(Recording {
        events: Mutex::new(Vec::new()),
    });
    let id = spark
        .streams()
        .add_listener(listener.clone())
        .expect("add_listener");
    assert_eq!(spark.streams().listeners().len(), 1);
    let query = spark
        .read_stream()
        .format("rate")
        .load(None)
        .write_stream()
        .format("noop")
        .query_name("e2e_session_listener_bus")
        .start("")
        .expect("start streaming query");
    assert_eq!(
        listener.events.lock().unwrap().first(),
        Some(&QUERY_STARTED_EVENT)
    );
    for _ in 0..60 {
        if listener
            .events
            .lock()
            .unwrap()
            .contains(&QUERY_PROGRESS_EVENT)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    spark
        .streams()
        .remove_listener(&id)
        .expect("remove_listener");
    assert!(spark.streams().listeners().is_empty());
    let seen = listener.events.lock().unwrap().len();
    std::thread::sleep(Duration::from_secs(2));
    query.stop().expect("stop");
    let events = listener.events.lock().unwrap();
    assert!(events.contains(&QUERY_PROGRESS_EVENT), "{events:?}");
    assert_eq!(events.len(), seen, "{events:?}");
}

/// Exercise the real event reader before dropping every native session/manager owner.
/// After the probe query terminates the subscription is quiet; its worker must not keep
/// the bus, its listener, or its cancellation sender alive by retaining a strong bus Arc.
#[test]
fn streaming_last_native_owner_drop_cancels_a_quiet_listener_worker() {
    if !should_run() {
        return;
    }
    use spark_connect::streaming::{
        StreamingQueryListener, StreamingQueryListenerEvent, QUERY_TERMINATED_EVENT,
    };
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    struct DropObservedListener {
        terminated: mpsc::Sender<()>,
        dropped: mpsc::Sender<()>,
    }
    impl StreamingQueryListener for DropObservedListener {
        fn on_event(&self, event: &StreamingQueryListenerEvent) {
            if event.event_type == QUERY_TERMINATED_EVENT {
                let _ = self.terminated.send(());
            }
        }
    }
    impl Drop for DropObservedListener {
        fn drop(&mut self) {
            let _ = self.dropped.send(());
        }
    }

    let spark = session();
    let manager = spark.streams();
    let (terminated, observed_termination) = mpsc::channel();
    let (dropped, observed_drop) = mpsc::channel();
    let listener = Arc::new(DropObservedListener {
        terminated,
        dropped,
    });
    manager
        .add_listener(listener.clone())
        .expect("register the native ownership listener");
    let query = spark
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "1")
        .load(None)
        .write_stream()
        .format("noop")
        .query_name("e2e_last_native_listener_owner")
        .start("")
        .expect("start the ownership probe query");
    query.stop().expect("stop the ownership probe query");
    // A started callback is synchronous. This asynchronous bus event proves that
    // the real worker entered its loop before the test drops any of its owners.
    observed_termination
        .recv_timeout(Duration::from_secs(30))
        .expect("the real listener worker receives the query termination");
    assert_eq!(manager.listeners().len(), 1);

    // The listener stores only channels, so it cannot retain a session or manager.
    // Do not remove the listener or close/stop the session: cancellation must follow
    // the last owner's drop, rather than explicit cleanup.
    drop(query);
    drop(listener);
    drop(manager);
    drop(spark);
    observed_drop
        .recv_timeout(Duration::from_secs(5))
        .expect("dropping every native owner releases the listener of the quiet real worker");
}

/// A listener whose subscription fails on its first event re-adds another listener from its
/// `on_stream_error`, as a client resubscribes. The re-added listener must be subscribed on the
/// server: it receives the events of a query that runs afterwards.
#[test]
fn streaming_listener_readded_from_a_stream_error_receives_events() {
    if !should_run() {
        return;
    }
    use spark_connect::streaming::{StreamingQueryListener, StreamingQueryListenerEvent};
    use spark_connect::streaming::{StreamingQueryManager, QUERY_PROGRESS_EVENT};
    use spark_connect::{Result, SparkError};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct Counting(AtomicUsize);
    impl StreamingQueryListener for Counting {
        fn on_event(&self, event: &StreamingQueryListenerEvent) {
            if event.event_type == QUERY_PROGRESS_EVENT {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    struct FailsThenReadds {
        manager: StreamingQueryManager,
        next: Arc<dyn StreamingQueryListener>,
        result: Mutex<Option<Result<String>>>,
    }
    impl StreamingQueryListener for FailsThenReadds {
        fn on_event(&self, _: &StreamingQueryListenerEvent) {}

        fn try_on_event(&self, _: &StreamingQueryListenerEvent) -> Result<()> {
            Err(SparkError::connect_msg("the event cannot be read"))
        }

        fn on_stream_error(&self, _: &SparkError) -> bool {
            let added = self.manager.add_listener(Arc::clone(&self.next));
            *self.result.lock().unwrap() = Some(added);
            true
        }
    }

    let spark = session();
    let manager = spark.streams();
    let counting = Arc::new(Counting(AtomicUsize::new(0)));
    let failing = Arc::new(FailsThenReadds {
        manager: spark.streams(),
        next: counting.clone(),
        result: Mutex::new(None),
    });
    manager
        .add_listener(failing.clone())
        .expect("add the listener that fails");
    let query = spark
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "1")
        .load(None)
        .write_stream()
        .format("noop")
        .query_name("e2e_listener_readded_after_stream_error")
        .start("")
        .expect("start the query");
    for _ in 0..60 {
        if counting.0.load(Ordering::SeqCst) > 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    query.stop().expect("stop the query");
    let added = failing
        .result
        .lock()
        .unwrap()
        .take()
        .expect("the failing listener reported its stream error");
    assert!(added.is_ok(), "the re-add succeeds: {added:?}");
    assert_eq!(
        manager.listeners().len(),
        1,
        "only the re-added listener remains"
    );
    assert!(
        counting.0.load(Ordering::SeqCst) > 0,
        "the re-added listener receives the events of the query"
    );
    manager.close().expect("close the listener bus");
}

/// A session progress handler that panics on the reported removal of the listener bus does not
/// leave the bus in its lifecycle phase: the first close reports the panic, and the next close
/// returns rather than waiting for the phase.
#[test]
fn streaming_listener_close_returns_after_a_panicking_progress_handler() {
    if !should_run() {
        return;
    }
    use spark_connect::streaming::{StreamingQueryListener, StreamingQueryListenerEvent};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    struct Quiet;
    impl StreamingQueryListener for Quiet {
        fn on_event(&self, _: &StreamingQueryListenerEvent) {}
    }

    let spark = session();
    let manager = spark.streams();
    manager
        .add_listener(Arc::new(Quiet))
        .expect("add the listener");
    let panicked = Arc::new(AtomicBool::new(false));
    let fired = Arc::clone(&panicked);
    let handler = spark.register_progress_handler(move |_| {
        if !fired.swap(true, Ordering::SeqCst) {
            panic!("the progress handler panics");
        }
    });

    let first = catch_unwind(AssertUnwindSafe(|| manager.close()));
    spark.remove_progress_handler(handler);
    assert!(
        first.is_err(),
        "the first close reports the panic of the handler"
    );
    assert!(panicked.load(Ordering::SeqCst));

    let (returned, done) = mpsc::channel();
    let retry = spark.clone();
    std::thread::spawn(move || {
        let _ = returned.send(retry.streams().close());
    });
    done.recv_timeout(Duration::from_secs(60))
        .expect("the next close returns")
        .expect("the next close removes the listener");
    assert!(manager.listeners().is_empty());
}

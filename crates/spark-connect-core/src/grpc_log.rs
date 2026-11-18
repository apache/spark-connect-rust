//! The errors of a channel that the client goes on from, which grpcio logs.
//!
//! PySpark's client shows nothing of a proxy variable that is not an HTTP proxy, or of a file of
//! TLS roots that cannot be read: gRPC's C core logs them, to the stderr of the process, as errors
//! of its own (`E1118 16:53:41.241267 4026384 http_proxy_mapper.cc:159] 'https' scheme not
//! supported in proxy URI`), unless `GRPC_VERBOSITY` is `NONE`, and goes on without the proxy or
//! with its default roots. The core gives these errors to `tracing`, as warnings, and to the
//! listener of the process, which the Python binding sets to log them as grpcio does. Not a stable
//! API.

use std::panic::Location;
use std::sync::{Arc, RwLock};

/// An error of a channel that grpcio logs.
#[derive(Debug, Clone)]
pub struct Logged {
    /// What is wrong and what the client does instead. It never holds the value of a variable,
    /// as a proxy URI can hold a password.
    pub message: String,
    /// The call of the core that logs it, as grpcio's line names the source of its call.
    pub location: &'static Location<'static>,
}

/// What the client does with an error of a channel that grpcio logs.
pub type Listener = Arc<dyn Fn(&Logged) + Send + Sync>;

/// The listener of the process, which every channel tells.
static LISTENER: RwLock<Option<Listener>> = RwLock::new(None);

/// Set the listener of the errors of the channels of the process, or none.
pub fn set_listener(listener: Option<Listener>) {
    *LISTENER.write().unwrap_or_else(|e| e.into_inner()) = listener;
}

/// Log `message`, an error of a channel that grpcio logs: as a warning of `tracing`, and to the
/// listener of the process, if one is set, on the thread of the caller.
#[track_caller]
pub(crate) fn error(message: String) {
    let location = Location::caller();
    tracing::warn!("{message}");
    let listener = LISTENER.read().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(listener) = listener {
        listener(&Logged { message, location });
    }
}

/// What `f` logs on the thread that runs it, with a listener of the test's own. The tests that
/// listen take turns, and a listener of another test does not see what this one logs.
#[cfg(test)]
pub(crate) fn logged_by<T>(f: impl FnOnce() -> T) -> (T, Vec<Logged>) {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let thread = std::thread::current().id();
    let logged = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = logged.clone();
    set_listener(Some(Arc::new(move |entry: &Logged| {
        if std::thread::current().id() == thread {
            sink.lock().unwrap().push(entry.clone());
        }
    })));
    let output = f();
    set_listener(None);
    let logged = logged.lock().unwrap().clone();
    (output, logged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_go_to_the_listener_with_the_location_of_their_call() {
        let ((), logged) = logged_by(|| error("not using a proxy".to_string()));
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert_eq!(logged[0].message, "not using a proxy");
        assert!(logged[0].location.file().ends_with("grpc_log.rs"));
        // Without a listener, an error goes to `tracing` only.
        error("not using a proxy".to_string());
    }
}

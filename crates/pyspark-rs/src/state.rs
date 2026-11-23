//! The state of the objects of a session that PySpark keeps in their `__dict__`.
//!
//! PySpark's objects of a session, such as its catalog, its table-valued functions, its data
//! source registration and its runtime configuration, are plain Python objects: their `__init__`
//! sets their state as instance attributes, of any object that it is given, and their methods
//! read those attributes when they need them. The native classes of these objects keep their
//! state the same way, under PySpark's names in the `__dict__` of the object, which `__init__`
//! sets as PySpark's does, and their methods read it through the attribute that PySpark's method
//! reads, where it reads it ([`attr`]). So `vars()`, an assignment, a deletion, a property or a
//! `__getattr__` of a subclass, and the copies and the pickles of the state are PySpark's, and an
//! object whose `__init__` never ran raises the `AttributeError` that PySpark's raises.

use pyo3::prelude::*;
use pyo3::types::PyString;
use spark_connect::session::SparkSession;

use crate::session::PySparkSession;

/// `getattr(obj, name)`, as a method of PySpark reads `self.<name>`: of the instance dict, or of a
/// descriptor or a `__getattr__` of a subclass, with their errors. Within CPython's recursion
/// check, as a call of Python code is, so that a `__getattr__` that reads the attribute again
/// ends in a `RecursionError`, as PySpark's does, and not in an overflow of the native stack.
pub(crate) fn attr<'py>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: the GIL is held, and the string is a C string literal; the check that succeeds is
    // left by `Py_LeaveRecursiveCall` below, on every path.
    if unsafe { pyo3::ffi::Py_EnterRecursiveCall(c"".as_ptr()) } != 0 {
        return Err(PyErr::fetch(obj.py()));
    }
    let value = obj.getattr(name);
    // SAFETY: as above, of the check that `Py_EnterRecursiveCall` entered.
    unsafe { pyo3::ffi::Py_LeaveRecursiveCall() };
    value
}

/// Set the attributes of `entries` on `obj`, in their order, as PySpark's `__init__` assigns
/// them: through `setattr`, so that a descriptor or a `__setattr__` of a subclass takes them as
/// it takes PySpark's assignments, and `vars()` holds them in that order.
pub(crate) fn assign<'py>(
    obj: &Bound<'py, PyAny>,
    entries: &[(&Bound<'py, PyString>, &Bound<'py, PyAny>)],
) -> PyResult<()> {
    for (name, value) in entries {
        obj.setattr(*name, *value)?;
    }
    Ok(())
}

/// The `__getstate__` of an object whose state is its `__dict__`, as `object.__getstate__` gives
/// it of PySpark's plain object: the `__dict__` itself (with the values of the `__slots__` of a
/// subclass that are set), or None when there is neither. CPython's own refuses an object of a
/// native class, whose layout it does not know.
pub(crate) fn getstate<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    Ok(crate::reduce::plain_object_state(obj)?.unwrap_or_else(|| py.None().into_bound(py)))
}

/// The `__reduce_ex__` of such an object, as `object.__reduce_ex__` gives it of PySpark's plain
/// object: from the protocol 2, `object.__reduce_ex__` itself, of `copyreg.__newobj__` of the
/// class and the state of its `__getstate__` (or of the `__reduce__` of a subclass, which it
/// calls); below it, where copyreg refuses a class of a native base, the same form, which those
/// protocols store too ([`crate::reduce::plain_reduce`]).
pub(crate) fn reduce_ex<'py>(
    obj: &Bound<'py, PyAny>,
    protocol: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    if protocol >= 2 {
        return py
            .get_type::<PyAny>()
            .call_method1(pyo3::intern!(py, "__reduce_ex__"), (obj, protocol));
    }
    Ok(crate::reduce::plain_reduce(obj)?.into_any())
}

/// The refresher of the calls of a session, the `_RequestContext` of the state of its clients (see
/// `pyspark.sql.connect.client.core._client_context`), which the core keeps as long as the session
/// lives, of its client and of the clones that its executions hold. Once the session is gone, its
/// drop sets the `_closed` of the clients of the state, as the `__del__` of PySpark's session
/// object closes its client: a client that the program kept apart from its session is closed then
/// (see `_session` of `SparkConnectClient`).
pub(crate) struct ClientRefresher(Option<Py<PyAny>>);

impl ClientRefresher {
    pub(crate) fn new(refresh: Py<PyAny>) -> Self {
        ClientRefresher(Some(refresh))
    }

    /// The `_RequestContext`.
    pub(crate) fn refresh(&self) -> Option<&Py<PyAny>> {
        self.0.as_ref()
    }
}

impl Drop for ClientRefresher {
    fn drop(&mut self) {
        let Some(refresh) = self.0.take() else {
            return;
        };
        // Of an interpreter that is gone, nothing; of the exception that the thread was raising
        // kept as it was, as the drop may run while one propagates.
        let _ = Python::try_attach(|py| {
            crate::deliver::preserving_pending_exception(py, |py| {
                if let Err(error) = refresh.bind(py).call_method1("_set_closed", (true,)) {
                    error.write_unraisable(py, None);
                }
            })
        });
    }
}

/// The `_RequestContext` of the refresher `refresher` of a session (see [`ClientRefresher`]).
pub(crate) fn refresh_of(refresher: &(dyn std::any::Any + Send + Sync)) -> Option<&Py<PyAny>> {
    refresher.downcast_ref::<ClientRefresher>()?.refresh()
}

/// The attributes of the clients of `session` (`pyspark.sql.connect.client.core._client_state`),
/// of the state that the session keeps with the refresher of its calls (see
/// `_RequestContext`), or None of a session of no state yet.
pub(crate) fn client_state_of<'py>(
    py: Python<'py>,
    session: &SparkSession,
) -> Option<Bound<'py, pyo3::types::PyDict>> {
    let refresher = session.user_context_refresher()?;
    let refresh = refresh_of(refresher.as_ref())?;
    refresh
        .bind(py)
        .getattr(pyo3::intern!(py, "state"))
        .ok()?
        .cast_into::<pyo3::types::PyDict>()
        .ok()
}

/// Whether the client of `session` is closed, of the `_closed` of the attributes of its clients
/// (see [`client_state_of`]), as PySpark's `is_stopped` of a session reads the `_closed` of its
/// client, else of the session.
pub(crate) fn client_closed(py: Python<'_>, session: &SparkSession) -> bool {
    match client_state_of(py, session).and_then(|state| state.get_item("_closed").ok().flatten()) {
        Some(closed) => closed.is_truthy().unwrap_or(true),
        None => session.is_stopped(),
    }
}

/// The session of `client`, a client of Spark Connect of this client
/// (`pyspark.sql.connect.client.core.SparkConnectClient`, of its class or of a subclass), and not
/// of another object, a session included.
pub(crate) fn client_session(client: &Bound<'_, PyAny>) -> Option<SparkSession> {
    if client.is_instance_of::<PySparkSession>() {
        return None;
    }
    crate::session::session_of_client(client)
}

/// Whether the class of `client`, a client of this client, overrides its method `name`, which
/// PySpark's objects of a session call to make their requests: a request of an object of the drop-in
/// calls that method then, as PySpark's does, rather than the native transport of the client.
pub(crate) fn overrides(client: &Bound<'_, PyAny>, name: &Bound<'_, PyString>) -> PyResult<bool> {
    let py = client.py();
    static CLIENT: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> = pyo3::sync::PyOnceLock::new();
    let base = CLIENT.import(py, "pyspark.sql.connect.client.core", "SparkConnectClient")?;
    let own = client.get_type().getattr(name)?;
    Ok(!own.is(&base.getattr(name)?))
}

/// The error of `client`, a client of another class than this client's, of which a request of an
/// object of a session is made: the requests of the objects of a session are those of the native
/// transport of its client, and not of the methods of another object, as PySpark's are (see
/// `dev/parity/known-differences.md`).
pub(crate) fn foreign_client_error(client: &Bound<'_, PyAny>) -> PyErr {
    let name = client
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    pyo3::exceptions::PyTypeError::new_err(format!(
        "'{name}' object is not a SparkConnectClient of pyspark.sql.connect.client.core, whose \
         native transport makes the requests of the objects of a session"
    ))
}

/// The session of `value`, the object that a PySpark method reads where its class takes a session
/// (the `_sparkSession` of a catalog, the `_sparkSession` of the table-valued functions), through
/// which it makes its requests: the client of its attribute `client_attr` (`client`, or `_client`
/// where PySpark reads that), whose `AttributeError` is PySpark's, is of its session, of a session
/// of this client as of a proxy of one, and the own client of a session of this client is of its
/// session at once (see `PySparkSession::of_client`). A client of another class raises
/// [`foreign_client_error`].
pub(crate) fn session_of_value(
    value: &Bound<'_, PyAny>,
    client_attr: &Bound<'_, PyString>,
) -> PyResult<SparkSession> {
    let client = attr(value, client_attr)?;
    if let Ok(session) = value.cast::<PySparkSession>() {
        if let Some(own) = session
            .try_borrow()
            .ok()
            .and_then(|session| session.session_of_own_client(&client))
        {
            return Ok(own);
        }
    }
    client_session(&client).ok_or_else(|| foreign_client_error(&client))
}

//! PyO3 wrapper for data source registration.

use pyo3::prelude::*;
use spark_connect::datasource::{
    CommonInlineUserDefinedDataSourceExpression, PythonDataSourcePayload,
};

use crate::dataframe::{py_cloudpickle, py_version};
use crate::interrupt::interruptible;
use crate::session::PySparkSession;

/// Python accessor for data source registration.
///
/// Exposed as `SparkSession.dataSource` to register custom data sources. It is PySpark Connect's
/// class, `pyspark.sql.connect.datasource.DataSourceRegistration`, which that module re-exports:
/// `pyspark.sql.datasource.DataSourceRegistration` is PySpark's classic class, of which it is not.
/// As PySpark's plain class, it is subclassed: a subclass constructs of the arguments of its own
/// `__init__`, and its copies and pickles are of its class.
#[pyclass(
    name = "DataSourceRegistration",
    module = "pyspark.sql.connect.datasource",
    subclass,
    weakref,
    dict
)]
pub struct PyDataSourceRegistration {
    /// The session of the registration, as PySpark's `sparkSession`, which its `__init__` sets:
    /// None before it, as the registration that `__new__` makes has none.
    pub(crate) session: Option<Py<PySparkSession>>,
}

impl PyDataSourceRegistration {
    pub fn new(session: Py<PySparkSession>) -> Self {
        PyDataSourceRegistration {
            session: Some(session),
        }
    }

    /// The session of the registration, or the AttributeError of PySpark's registration whose
    /// `__init__` did not run, which has no `sparkSession`.
    fn session_of(slf: &Bound<'_, Self>) -> PyResult<Py<PySparkSession>> {
        let py = slf.py();
        match &slf.borrow().session {
            Some(session) => Ok(session.clone_ref(py)),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'{}' object has no attribute 'sparkSession'",
                slf.get_type().name()?
            ))),
        }
    }
}

#[pymethods]
impl PyDataSourceRegistration {
    /// A registration of no session, as the `object.__new__` of PySpark's plain class: of any
    /// arguments, which `__init__` takes, so that a subclass of another `__init__` is built as it
    /// is in PySpark, and an unpickle makes one of none. The class shows the signature of
    /// `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(sparkSession)")]
    fn py_new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        PyDataSourceRegistration { session: None }
    }

    /// `DataSourceRegistration(sparkSession)`, as PySpark Connect's `__init__`: the registration
    /// of the data sources of the session.
    #[allow(non_snake_case)]
    fn __init__(slf: &Bound<'_, Self>, sparkSession: Py<PySparkSession>) {
        let replaced = slf.borrow_mut().session.replace(sparkSession);
        drop(replaced);
    }

    /// The shallow copy of the data source registration, as PySpark's `copy.copy`: of the same
    /// session, and of the class of this one, with the instance attributes of this one.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let session = slf.borrow().session.as_ref().map(|s| s.clone_ref(py));
        let copy =
            crate::reduce::new_of_class(&slf.get_type(), PyDataSourceRegistration { session })?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.as_any())?;
        Ok(copy)
    }

    /// The reduce of the data source registration, as PySpark's pickle of its plain object:
    /// `copyreg.__newobj__` of its class, of the state of its `__dict__`, its `sparkSession`, which
    /// unpickles with it, and its instance attributes (see `__setstate__`). So PySpark loads the
    /// pickle as a registration of its own, and PySpark's pickle loads here.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::types::{PyDict, PyTuple};
        let py = slf.py();
        let state = PyDict::new(py);
        if let Some(session) = &slf.borrow().session {
            state.set_item("sparkSession", session.clone_ref(py))?;
        }
        state.call_method1("update", (slf.getattr(pyo3::intern!(py, "__dict__"))?,))?;
        let newobj = py.import("copyreg")?.getattr("__newobj__")?;
        let args = PyTuple::new(py, [slf.get_type()])?;
        let reduced = match state.is_empty() {
            true => PyTuple::new(py, [newobj, args.into_any()])?,
            false => PyTuple::new(py, [newobj, args.into_any(), state.into_any()])?,
        };
        Ok(reduced.into_any())
    }

    /// The state of the pickle of a registration, as pickle sets the `__dict__` of PySpark's: its
    /// session, of `sparkSession`, and its instance attributes, of the other entries.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
        let py = slf.py();
        let state = state.copy()?;
        if let Some(session) = state.get_item("sparkSession")? {
            Self::__init__(slf, session.extract()?);
            state.del_item("sparkSession")?;
        }
        slf.getattr(pyo3::intern!(py, "__dict__"))?
            .call_method1(pyo3::intern!(py, "update"), (state,))?;
        Ok(())
    }

    /// The session of the registration, as PySpark's `DataSourceRegistration.sparkSession`.
    #[getter(sparkSession)]
    fn spark_session(slf: &Bound<'_, Self>) -> PyResult<Py<PySparkSession>> {
        Self::session_of(slf)
    }

    /// As PySpark's registration, a Python object that holds its session, which Python's garbage
    /// collector sees: the session is reported, so a reference cycle through it, such as a session
    /// that keeps its `dataSource`, is freed. Only an `Option<Py<_>>` field is read, which is
    /// `None` while the object is created, so no [`crate::gc::GcReady`] is needed.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if let Some(session) = &self.session {
            visit.call(session)?;
        }
        Ok(())
    }

    /// Break a reference cycle that the garbage collector found unreachable.
    fn __clear__(&mut self) {
        self.session = None;
    }

    // Register a custom data source class.
    //
    // Mirrors `pyspark.sql.SparkSession.dataSource.register(dataSource)`, of PySpark's parameter.
    // The data source class is cloudpickled and sent to the server.
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataSourceRegistration.register)]
    fn register(slf: &Bound<'_, Self>, dataSource: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        // As PySpark's `self.sparkSession._client.register_data_source(dataSource)`, which reads
        // the session first.
        let session = Self::session_of(slf)?.borrow(py).session.clone();
        // The name of the classmethod `name()`, as PySpark Connect's: the AttributeError of a
        // class without one. Its `CommonInlineUserDefinedDataSource` sets the name in the string
        // field of its command, as protobuf converts it, before it pickles the class: bytes of
        // UTF-8 too, and protobuf's error of another value, such as an int or None.
        let name = crate::str_argument(&dataSource.call_method0("name")?)?;

        // Cloudpickle the class
        let command = py_cloudpickle(py, dataSource)?;
        let py_ver = py_version(py);

        // Create the payload and expression
        let payload = PythonDataSourcePayload::new(command, py_ver);
        let expr = CommonInlineUserDefinedDataSourceExpression::new(name, payload);

        // Register with the session (release GIL during the RPC)
        interruptible(py, &session, || {
            session.register_data_source(expr).map_err(Box::new)
        })?;

        Ok(())
    }
}

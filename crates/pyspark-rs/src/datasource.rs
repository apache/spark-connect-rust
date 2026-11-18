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
#[pyclass(
    name = "DataSourceRegistration",
    module = "pyspark.sql.connect.datasource",
    weakref,
    dict
)]
pub struct PyDataSourceRegistration {
    pub(crate) session: Py<PySparkSession>,
    /// Set once the value is written, as `__traverse__` reads `session`, which is not valid while
    /// the object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyDataSourceRegistration {
    pub fn new(session: Py<PySparkSession>) -> Self {
        PyDataSourceRegistration {
            session,
            gc_ready: crate::gc::GcReady::new(),
        }
    }
}

#[pymethods]
impl PyDataSourceRegistration {
    /// `DataSourceRegistration(sparkSession)`, as PySpark Connect's: the registration of the data
    /// sources of the session.
    #[new]
    #[allow(non_snake_case)]
    fn py_new(sparkSession: Py<PySparkSession>) -> Self {
        PyDataSourceRegistration::new(sparkSession)
    }

    /// The shallow copy of the data source registration, as PySpark's `copy.copy`: of the same
    /// session, with the instance attributes of this one.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let session = slf.borrow().session.clone_ref(py);
        let copy = Py::new(py, PyDataSourceRegistration::new(session))?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.bind(py).as_any())?;
        Ok(copy)
    }

    /// The reduce of the data source registration, as PySpark's: `dataSource` of its session,
    /// which unpickles with it, and its instance attributes.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let session = slf.borrow().session.clone_ref(py).into_bound(py).into_any();
        let reduced = crate::reduce::getattr_reduce(py, session, "dataSource")?;
        crate::reduce::reduce_with_instance_dict(slf.as_any(), reduced.into_any())
    }

    /// The session of the registration, as PySpark's `DataSourceRegistration.sparkSession`.
    #[getter(sparkSession)]
    fn spark_session(&self, py: Python<'_>) -> Py<PySparkSession> {
        self.session.clone_ref(py)
    }

    /// As PySpark's registration, a Python object that holds its session, which Python's garbage
    /// collector sees: the session is reported, so a reference cycle through it, such as a session
    /// that keeps its `dataSource`, is freed. Nothing of a registration that is being created.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        visit.call(&self.session)
    }

    /// Register a custom data source class.
    ///
    /// Mirrors `pyspark.sql.SparkSession.dataSource.register(dataSource)`, of PySpark's parameter.
    /// The data source class is cloudpickled and sent to the server.
    #[allow(non_snake_case)]
    fn register(&self, py: Python<'_>, dataSource: &Bound<'_, PyAny>) -> PyResult<()> {
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
        let session = self.session.borrow(py).session.clone();
        interruptible(py, &session, || {
            session.register_data_source(expr).map_err(Box::new)
        })?;

        Ok(())
    }
}

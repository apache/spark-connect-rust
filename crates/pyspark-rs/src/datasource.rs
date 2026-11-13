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
/// Exposed as `SparkSession.dataSource` to register custom data sources.
#[pyclass(
    name = "DataSourceRegistration",
    module = "pyspark.sql.datasource",
    weakref,
    dict
)]
pub struct PyDataSourceRegistration {
    pub(crate) session: Py<PySparkSession>,
}

impl PyDataSourceRegistration {
    pub fn new(session: Py<PySparkSession>) -> Self {
        PyDataSourceRegistration { session }
    }
}

#[pymethods]
impl PyDataSourceRegistration {
    /// The shallow copy of the data source registration, as PySpark's `copy.copy`: of the same
    /// session.
    fn __copy__(&self, py: Python<'_>) -> PyDataSourceRegistration {
        PyDataSourceRegistration::new(self.session.clone_ref(py))
    }

    /// The reduce of the data source registration, as PySpark's: `dataSource` of its session,
    /// which unpickles with it.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let session = self.session.clone_ref(py).into_bound(py).into_any();
        crate::reduce::getattr_reduce(py, session, "dataSource")
    }

    /// The session of the registration, as PySpark's `DataSourceRegistration.sparkSession`.
    #[getter(sparkSession)]
    fn spark_session(&self, py: Python<'_>) -> Py<PySparkSession> {
        self.session.clone_ref(py)
    }

    /// Register a custom data source class.
    ///
    /// Mirrors `pyspark.sql.SparkSession.dataSource.register`.
    /// The data source class is cloudpickled and sent to the server.
    fn register(&self, py: Python<'_>, data_source_class: &Bound<'_, PyAny>) -> PyResult<()> {
        // The name of the classmethod `name()`, as PySpark Connect's: the AttributeError of a
        // class without one.
        let name: String = data_source_class.call_method0("name")?.extract()?;

        // Cloudpickle the class
        let command = py_cloudpickle(py, data_source_class)?;
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

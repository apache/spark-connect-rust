//! PyO3 wrapper for data source registration.

use pyo3::prelude::*;
use spark_connect::datasource::{
    CommonInlineUserDefinedDataSourceExpression, PythonDataSourcePayload,
};

use crate::dataframe::{py_cloudpickle, py_version};
use crate::errors::ResultExt;
use crate::session::PySparkSession;

/// Python accessor for data source registration.
///
/// Exposed as `SparkSession.dataSource` to register custom data sources.
#[pyclass(name = "DataSourceRegistration")]
pub struct PyDataSourceRegistration {
    pub(crate) session: PySparkSession,
}

impl PyDataSourceRegistration {
    pub fn new(session: PySparkSession) -> Self {
        PyDataSourceRegistration { session }
    }
}

#[pymethods]
impl PyDataSourceRegistration {
    /// Register a custom data source class.
    ///
    /// Mirrors `pyspark.sql.SparkSession.dataSource.register`.
    /// The data source class is cloudpickled and sent to the server.
    fn register(&self, py: Python<'_>, data_source_class: &Bound<'_, PyAny>) -> PyResult<()> {
        // Get the name via the classmethod `name()`
        let name_result = data_source_class.call_method0("name");
        let name: String = match name_result {
            Ok(name_obj) => name_obj.extract()?,
            Err(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "DataSource class must have a classmethod name()",
                ))
            }
        };

        // Cloudpickle the class
        let command = py_cloudpickle(py, data_source_class)?;
        let py_ver = py_version(py);

        // Create the payload and expression
        let payload = PythonDataSourcePayload::new(command, py_ver);
        let expr = CommonInlineUserDefinedDataSourceExpression::new(name, payload);

        // Register with the session (release GIL during the RPC)
        py.detach(|| self.session.session.register_data_source(expr))
            .to_pyerr()?;

        Ok(())
    }
}

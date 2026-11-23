//! PyO3 wrapper for data source registration.

use pyo3::prelude::*;
use spark_connect::datasource::{
    CommonInlineUserDefinedDataSourceExpression, PythonDataSourcePayload,
};

use crate::dataframe::{py_cloudpickle, py_version};
use crate::session::PySparkSession;

// Python accessor for data source registration.
//
// Exposed as `SparkSession.dataSource` to register custom data sources. It is PySpark Connect's
// class, `pyspark.sql.connect.datasource.DataSourceRegistration`, which that module re-exports:
// `pyspark.sql.datasource.DataSourceRegistration` is PySpark's classic class, of which it is not.
// As PySpark's plain class, it keeps its state in its `__dict__`, its `sparkSession`, which its
// `register` reads, and it is subclassed and constructed of PySpark's arguments (see
// `crate::state`).
#[doc = connect_doc!(DataSourceRegistration)]
#[pyclass(
    name = "DataSourceRegistration",
    module = "pyspark.sql.connect.datasource",
    subclass,
    weakref,
    dict
)]
pub struct PyDataSourceRegistration {}

impl PyDataSourceRegistration {
    /// The registration of `session`, as `spark.dataSource` makes it:
    /// `DataSourceRegistration(spark)`, of the session object, which PySpark's `__init__` keeps
    /// as its `sparkSession`.
    pub(crate) fn of_session<'py>(
        session: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDataSourceRegistration>> {
        let registration = Bound::new(session.py(), PyDataSourceRegistration {})?;
        Self::__init__(&registration, session)?;
        Ok(registration)
    }
}

#[pymethods]
impl PyDataSourceRegistration {
    /// A registration of no state, as the `object.__new__` of PySpark's plain class: of any
    /// arguments, which `__init__` takes, so that a subclass of another `__init__` is built as it
    /// is in PySpark, and an unpickle makes one of none. The class shows the signature of
    /// `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(sparkSession)")]
    fn py_new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        PyDataSourceRegistration {}
    }

    /// `DataSourceRegistration(sparkSession)`, as PySpark Connect's `__init__`, which keeps its
    /// argument as it is, as the `sparkSession` of the registration.
    #[allow(non_snake_case)]
    fn __init__(slf: &Bound<'_, Self>, sparkSession: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        crate::state::assign(
            slf.as_any(),
            &[(pyo3::intern!(py, "sparkSession"), sparkSession)],
        )
    }

    /// The state of the registration, its `__dict__`, as `object.__getstate__` gives that of
    /// PySpark's plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the registration, as PySpark's plain object's: `copyreg.__newobj__` of its
    /// class, of its `__dict__`, so PySpark loads the pickle as a registration of its own, and
    /// PySpark's pickle loads here (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
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
        // the session and its `_client` first: the own client of a session of this client is of
        // its session at once; another client is of its session, or else of the
        // `register_data_source` of its own class, which PySpark calls, also of a subclass of this
        // client's that overrides it.
        let register = pyo3::intern!(py, "register_data_source");
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "sparkSession"))?;
        let client = crate::state::attr(&session, pyo3::intern!(py, "_client"))?;
        let own = session
            .cast::<PySparkSession>()
            .ok()
            .and_then(|session| session.try_borrow().ok()?.session_of_own_client(&client));
        let session = match own {
            Some(session) => session,
            None => match crate::state::client_session(&client) {
                Some(session) if !crate::state::overrides(&client, register)? => session,
                _ => {
                    client.call_method1(register, (dataSource,))?;
                    return Ok(());
                }
            },
        };
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
        // Of the bytes of the pickled class, of no part.
        crate::interrupt::interruptible_without_parts(py, &session, || {
            session.register_data_source(expr).map_err(Box::new)
        })?;

        Ok(())
    }
}

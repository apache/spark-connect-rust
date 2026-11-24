//! PyO3 wrapper for `spark_connect::tvf::TableValuedFunction` (`spark.tvf`).

use pyo3::prelude::*;
use spark_connect::tvf::TableValuedFunction;

use crate::dataframe::column_or_name;
use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

// Python wrapper for the table-valued-function namespace (`spark.tvf`). As PySpark's plain class,
// it keeps its state in its `__dict__`, its `_sparkSession`, which its functions read when they
// make their DataFrames, and it is subclassed and constructed of PySpark's arguments (see
// `crate::state`).
#[doc = connect_doc!(TableValuedFunction)]
#[pyclass(
    name = "TableValuedFunction",
    module = "pyspark.sql.connect.tvf",
    subclass,
    weakref,
    dict
)]
pub struct PyTableValuedFunction {}

impl PyTableValuedFunction {
    /// The table-valued functions of `session`, as `spark.tvf` makes them:
    /// `TableValuedFunction(spark)`, of the session object, which PySpark's `__init__` keeps as
    /// its `_sparkSession`.
    pub(crate) fn of_session<'py>(
        session: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTableValuedFunction>> {
        let tvf = Bound::new(session.py(), PyTableValuedFunction {})?;
        Self::__init__(&tvf, session)?;
        Ok(tvf)
    }

    /// The table-valued functions of the core of the session of `_sparkSession`, of whose plans
    /// they make their DataFrames, of the client of their requests (see `state::session_of_value`):
    /// of an object of no session of the drop-in, of the session of no server, as PySpark's `_fn`
    /// makes `DataFrame(plan, self._sparkSession)` of it as it is, whose requests then raise the
    /// error of its client (see `frame`).
    fn tvf_of(slf: &Bound<'_, Self>) -> PyResult<TableValuedFunction> {
        let py = slf.py();
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sparkSession"))?;
        match crate::state::session_of_value(&session, pyo3::intern!(py, "client")) {
            Ok(session) => Ok(session.tvf()),
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                Ok(crate::session::unconnected_session()?.tvf())
            }
            Err(err) => Err(err),
        }
    }

    /// `new`, the DataFrame of a function of these, as PySpark's `_fn` makes it,
    /// `DataFrame(plan, self._sparkSession)`: of `_sparkSession` as it is, as its `_session`,
    /// which its requests resolve (see `dataframe::Frame::of`).
    fn frame(
        slf: &Bound<'_, Self>,
        new: crate::dataframe::NewDataFrame,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let py = slf.py();
        let session = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sparkSession"))?;
        Ok(new.with_session(session.unbind()))
    }
}

/// The arguments of a table-valued function, as PySpark's `_fn` takes them: Columns or column
/// names, as `F._to_col` does, else `NOT_EXPECTED_TYPE` of `col`.
fn cols(items: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<spark_connect::column::Column>> {
    items.iter().map(column_or_name).collect()
}

#[pymethods]
impl PyTableValuedFunction {
    /// Table-valued functions of no state, as the `object.__new__` of PySpark's plain class: of
    /// any arguments, which `__init__` takes, so that a subclass of another `__init__` is built as
    /// it is in PySpark. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(sparkSession)")]
    fn py_new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        PyTableValuedFunction {}
    }

    /// `TableValuedFunction(sparkSession)`, as PySpark Connect's `__init__`, which keeps its
    /// argument as it is, as the `_sparkSession` of the functions.
    #[allow(non_snake_case)]
    fn __init__(slf: &Bound<'_, Self>, sparkSession: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        crate::state::assign(
            slf.as_any(),
            &[(pyo3::intern!(py, "_sparkSession"), sparkSession)],
        )
    }

    /// The state of the functions, their `__dict__`, as `object.__getstate__` gives that of
    /// PySpark's plain object, which a copy and a pickle keep (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the functions, as PySpark's plain object's: `copyreg.__newobj__` of their
    /// class, of their `__dict__` (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    // As PySpark's `spark.tvf.range`, which is `self._sparkSession.range(start, end, step,
    // numPartitions)`, a call of the `range` of the object, of PySpark's default step of 1.
    #[pyo3(
        signature = (start, end=None, step=crate::session::RangeStep::Omitted, numPartitions=None),
        text_signature = "($self, start, end=None, step=1, numPartitions=None)"
    )]
    #[allow(non_snake_case)]
    #[doc = connect_doc!(TableValuedFunction.range)]
    fn range<'py>(
        slf: &Bound<'py, Self>,
        start: &Bound<'py, PyAny>,
        end: Option<&Bound<'py, PyAny>>,
        step: crate::session::RangeStep,
        numPartitions: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let step = match step {
            crate::session::RangeStep::Omitted => 1i64.into_pyobject(py)?.into_any(),
            crate::session::RangeStep::Given(step) => step.into_bound(py),
        };
        crate::state::attr(slf.as_any(), pyo3::intern!(py, "_sparkSession"))?.call_method1(
            pyo3::intern!(py, "range"),
            (start, end, step, numPartitions),
        )
    }

    #[doc = connect_doc!(TableValuedFunction.explode)]
    fn explode(
        slf: &Bound<'_, Self>,
        collection: &Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let c = column_or_name(collection)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.explode(&c).to_pyerr()?),
        )
    }

    #[pyo3(name = "explode_outer")]
    #[doc = connect_doc!(TableValuedFunction.explode_outer)]
    fn explode_outer(
        slf: &Bound<'_, Self>,
        collection: &Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let c = column_or_name(collection)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.explode_outer(&c).to_pyerr()?),
        )
    }

    #[doc = connect_doc!(TableValuedFunction.inline)]
    fn inline(
        slf: &Bound<'_, Self>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let c = column_or_name(input)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.inline(&c).to_pyerr()?),
        )
    }

    #[pyo3(name = "inline_outer")]
    #[doc = connect_doc!(TableValuedFunction.inline_outer)]
    fn inline_outer(
        slf: &Bound<'_, Self>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let c = column_or_name(input)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.inline_outer(&c).to_pyerr()?),
        )
    }

    #[pyo3(name = "json_tuple", signature = (input, *fields))]
    #[doc = connect_doc!(TableValuedFunction.json_tuple)]
    fn json_tuple(
        slf: &Bound<'_, Self>,
        input: &Bound<'_, PyAny>,
        fields: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let py = slf.py();
        // As PySpark's, which asks for fields before it takes the arguments.
        if fields.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "field")],
            ));
        }
        let inp = column_or_name(input)?;
        let fs = cols(fields)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.json_tuple(&inp, fs).to_pyerr()?),
        )
    }

    #[doc = connect_doc!(TableValuedFunction.posexplode)]
    fn posexplode(
        slf: &Bound<'_, Self>,
        collection: &Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let c = column_or_name(collection)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.posexplode(&c).to_pyerr()?),
        )
    }

    #[pyo3(name = "posexplode_outer")]
    #[doc = connect_doc!(TableValuedFunction.posexplode_outer)]
    fn posexplode_outer(
        slf: &Bound<'_, Self>,
        collection: &Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let c = column_or_name(collection)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.posexplode_outer(&c).to_pyerr()?),
        )
    }

    #[pyo3(signature = (n, *fields))]
    #[doc = connect_doc!(TableValuedFunction.stack)]
    fn stack(
        slf: &Bound<'_, Self>,
        n: &Bound<'_, PyAny>,
        fields: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let nc = column_or_name(n)?;
        let fs = cols(fields)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.stack(&nc, fs).to_pyerr()?),
        )
    }

    #[doc = connect_doc!(TableValuedFunction.collations)]
    fn collations(slf: &Bound<'_, Self>) -> PyResult<crate::dataframe::NewDataFrame> {
        let tvf = Self::tvf_of(slf)?;
        Self::frame(
            slf,
            PyDataFrame::new(
                crate::gc::detach_without_parts(slf.py(), || tvf.collations()).to_pyerr()?,
            ),
        )
    }

    #[pyo3(name = "sql_keywords")]
    #[doc = connect_doc!(TableValuedFunction.sql_keywords)]
    fn sql_keywords(slf: &Bound<'_, Self>) -> PyResult<crate::dataframe::NewDataFrame> {
        let tvf = Self::tvf_of(slf)?;
        Self::frame(
            slf,
            PyDataFrame::new(
                crate::gc::detach_without_parts(slf.py(), || tvf.sql_keywords()).to_pyerr()?,
            ),
        )
    }

    #[pyo3(name = "variant_explode")]
    #[doc = connect_doc!(TableValuedFunction.variant_explode)]
    fn variant_explode(
        slf: &Bound<'_, Self>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let c = column_or_name(input)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.variant_explode(&c).to_pyerr()?),
        )
    }

    #[pyo3(name = "variant_explode_outer")]
    #[doc = connect_doc!(TableValuedFunction.variant_explode_outer)]
    fn variant_explode_outer(
        slf: &Bound<'_, Self>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<crate::dataframe::NewDataFrame> {
        let c = column_or_name(input)?;
        Self::frame(
            slf,
            PyDataFrame::new(Self::tvf_of(slf)?.variant_explode_outer(&c).to_pyerr()?),
        )
    }

    #[pyo3(name = "python_worker_logs")]
    #[doc = connect_doc!(TableValuedFunction.python_worker_logs)]
    fn python_worker_logs(slf: &Bound<'_, Self>) -> PyResult<crate::dataframe::NewDataFrame> {
        let tvf = Self::tvf_of(slf)?;
        Self::frame(
            slf,
            PyDataFrame::new(
                crate::gc::detach_without_parts(slf.py(), || tvf.python_worker_logs())
                    .to_pyerr()?,
            ),
        )
    }
}

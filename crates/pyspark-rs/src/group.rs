//! PyO3 wrapper for spark_connect::group::GroupedData.

use pyo3::prelude::*;
use spark_connect::group::{CoGroupedData as RustCoGroupedData, GroupedData as RustGroupedData};
use spark_connect::udf::eval_type;

use crate::dataframe::{build_map_udf, PyDataFrame};
use crate::errors::ResultExt;
use crate::functions::to_column;

/// Python wrapper for Spark GroupedData.
#[pyclass(name = "GroupedData", module = "pyspark.sql.group")]
pub struct PyGroupedData {
    pub(crate) grouped_data: RustGroupedData,
}

impl PyGroupedData {
    pub fn new(grouped_data: RustGroupedData) -> Self {
        PyGroupedData { grouped_data }
    }
}

#[pymethods]
impl PyGroupedData {
    /// Aggregate with expressions (accepts Column objects), or the dict form
    /// `agg({"col": "aggfunc"})` (PySpark parity), e.g. `agg({"age": "max"})`.
    #[pyo3(signature = (*cols))]
    fn agg(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        // Dict form: {column_name: aggregate_function_name} -> func(col(name)) per entry.
        if cols.len() == 1 {
            if let Ok(dict) = cols[0].cast::<pyo3::types::PyDict>() {
                let mut exprs = vec![];
                for (k, v) in dict.iter() {
                    let col_name: String = k.extract()?;
                    let func_name: String = v.extract()?;
                    let c = spark_connect::functions::col(&col_name);
                    // Build func(col) as an unresolved function resolved server-side, so
                    // aggregate names (max/min/sum/avg/count/...) all work regardless of
                    // whether they are in the generated dispatch.
                    let agg_expr = spark_connect::expression::Expression::UnresolvedFunction(
                        spark_connect::expression::UnresolvedFunction::new(
                            func_name,
                            vec![c.expression().clone()],
                        ),
                    );
                    exprs.push(agg_expr);
                }
                let df = self.grouped_data.agg(exprs);
                return Ok(PyDataFrame::new(df));
            }
        }
        let mut exprs = vec![];
        for col in cols {
            let rust_col = to_column(&col)?;
            exprs.push(rust_col.expression().clone());
        }
        let df = self.grouped_data.agg(exprs);
        Ok(PyDataFrame::new(df))
    }

    /// Count rows in each group.
    fn count(&self) -> PyResult<PyDataFrame> {
        let df = self.grouped_data.count();
        Ok(PyDataFrame::new(df))
    }

    /// Pivot on a column; `values=None` auto-discovers the distinct pivot values.
    #[pyo3(signature = (pivot_col, values=None))]
    fn pivot(
        &self,
        pivot_col: &str,
        values: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<PyGroupedData> {
        let pcol = spark_connect::functions::col(pivot_col);
        let vals = match values {
            None => None,
            Some(vs) => {
                let mut out = Vec::with_capacity(vs.len());
                for v in &vs {
                    out.push(crate::session::py_to_value(v)?);
                }
                Some(out)
            }
        };
        Ok(PyGroupedData::new(self.grouped_data.pivot(pcol, vals)))
    }

    /// Sum values in each group.
    #[pyo3(signature = (*cols))]
    fn sum(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.sum(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// Average values in each group.
    #[pyo3(signature = (*cols))]
    fn avg(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.avg(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// Minimum values in each group.
    #[pyo3(signature = (*cols))]
    fn min(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.min(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// Maximum values in each group.
    #[pyo3(signature = (*cols))]
    fn max(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.max(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// Mean (alias for avg).
    #[pyo3(signature = (*cols))]
    fn mean(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let col_names: Vec<String> = cols
            .iter()
            .map(|c| c.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let col_refs: Vec<&str> = col_names.iter().map(|s| s.as_str()).collect();
        let df = self.grouped_data.mean(col_refs);
        Ok(PyDataFrame::new(df))
    }

    /// `GroupedData.applyInPandas`.
    #[pyo3(name = "applyInPandas")]
    fn apply_in_pandas(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let cols = py.detach(|| self.grouped_data.input_columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "applyInPandas",
            &func,
            &schema,
            eval_type::SQL_GROUPED_MAP_PANDAS_UDF,
            &cols,
        )?;
        Ok(PyDataFrame::new(self.grouped_data.apply_in_pandas(udf)))
    }

    /// Deprecated grouped-map apply: `apply(udf)` where `udf` is a
    /// `SQL_GROUPED_MAP_PANDAS_UDF` pandas UDF. Delegates to `applyInPandas` with the
    /// UDF's wrapped function and return type (mirrors the official Connect API).
    fn apply(&self, py: Python<'_>, udf: Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let func = udf.getattr("func").map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "apply() expects a pandas_udf of type SQL_GROUPED_MAP_PANDAS_UDF",
            )
        })?;
        let schema = udf.getattr("returnType")?;
        self.apply_in_pandas(py, func, schema)
    }

    /// `GroupedData.applyInArrow`.
    #[pyo3(name = "applyInArrow")]
    fn apply_in_arrow(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let cols = py.detach(|| self.grouped_data.input_columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "applyInArrow",
            &func,
            &schema,
            eval_type::SQL_GROUPED_MAP_ARROW_UDF,
            &cols,
        )?;
        Ok(PyDataFrame::new(self.grouped_data.apply_in_arrow(udf)))
    }

    /// `GroupedData.cogroup`.
    #[pyo3(name = "cogroup")]
    fn cogroup(&self, other: &PyGroupedData) -> PyCoGroupedData {
        PyCoGroupedData {
            inner: self.grouped_data.cogroup(&other.grouped_data),
        }
    }

    /// `GroupedData.applyInPandasWithState` (stateful streaming).
    #[pyo3(name = "applyInPandasWithState")]
    #[pyo3(signature = (func, output_struct_type, state_struct_type, output_mode, timeout_conf))]
    fn apply_in_pandas_with_state(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        output_struct_type: Bound<'_, PyAny>,
        state_struct_type: Bound<'_, PyAny>,
        output_mode: &str,
        timeout_conf: &str,
    ) -> PyResult<PyDataFrame> {
        let cols = py.detach(|| self.grouped_data.input_columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "applyInPandasWithState",
            &func,
            &output_struct_type,
            eval_type::SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE,
            &cols,
        )?;
        let state_schema = crate::dataframe::resolve_datatype(&state_struct_type)?;
        Ok(PyDataFrame::new(
            self.grouped_data.apply_in_pandas_with_state(
                udf,
                state_schema,
                output_mode,
                timeout_conf,
            ),
        ))
    }

    /// `GroupedData.transformWithStateInPandas` — arbitrary stateful processing with a
    /// user `StatefulProcessor` (pandas output). Mirrors the Connect API: the processor is
    /// wrapped by `TransformWithStateInPandasUdfUtils` into a UDF (cloudpickled + run on the
    /// server), and the `GroupMap` proto carries the transform-with-state info.
    #[pyo3(name = "transformWithStateInPandas")]
    #[pyo3(signature = (statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName=""))]
    #[allow(non_snake_case)]
    fn transform_with_state_in_pandas(
        &self,
        py: Python<'_>,
        statefulProcessor: Bound<'_, PyAny>,
        outputStructType: Bound<'_, PyAny>,
        outputMode: &str,
        timeMode: &str,
        initialState: Option<PyRef<'_, PyGroupedData>>,
        eventTimeColumnName: &str,
    ) -> PyResult<PyDataFrame> {
        let (func, et) = twus_func(
            py,
            &statefulProcessor,
            timeMode,
            initialState.is_some(),
            true,
        )?;
        let cols = py.detach(|| self.grouped_data.input_columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "transformWithStateInPandas",
            &func,
            &outputStructType,
            et,
            &cols,
        )?;
        let out = crate::types::py_to_data_type(&outputStructType)?;
        let etc = (!eventTimeColumnName.is_empty()).then_some(eventTimeColumnName);
        Ok(PyDataFrame::new(
            self.grouped_data.transform_with_state_in_pandas(
                udf,
                out,
                outputMode,
                timeMode,
                etc,
                initialState.as_ref().map(|g| &g.grouped_data),
            ),
        ))
    }

    /// `GroupedData.transformWithState` (row output). Like `transformWithStateInPandas` but
    /// with the row-oriented eval type and no output schema on the plan node.
    #[pyo3(name = "transformWithState")]
    #[pyo3(signature = (statefulProcessor, outputStructType, outputMode, timeMode, initialState=None, eventTimeColumnName=""))]
    #[allow(non_snake_case)]
    fn transform_with_state(
        &self,
        py: Python<'_>,
        statefulProcessor: Bound<'_, PyAny>,
        outputStructType: Bound<'_, PyAny>,
        outputMode: &str,
        timeMode: &str,
        initialState: Option<PyRef<'_, PyGroupedData>>,
        eventTimeColumnName: &str,
    ) -> PyResult<PyDataFrame> {
        let (func, et) = twus_func(
            py,
            &statefulProcessor,
            timeMode,
            initialState.is_some(),
            false,
        )?;
        let cols = py.detach(|| self.grouped_data.input_columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "transformWithState",
            &func,
            &outputStructType,
            et,
            &cols,
        )?;
        let etc = (!eventTimeColumnName.is_empty()).then_some(eventTimeColumnName);
        Ok(PyDataFrame::new(self.grouped_data.transform_with_state(
            udf,
            outputMode,
            timeMode,
            etc,
            initialState.as_ref().map(|g| &g.grouped_data),
        )))
    }
}

/// Wrap a user `StatefulProcessor` into the transform-with-state UDF callable, returning it
/// with the matching `PythonEvalType`. Mirrors the Connect client's
/// `TransformWithStateInPandasUdfUtils` usage (pandas vs row output; with/without init state).
fn twus_func<'py>(
    py: Python<'py>,
    processor: &Bound<'py, PyAny>,
    time_mode: &str,
    has_init_state: bool,
    pandas: bool,
) -> PyResult<(Bound<'py, PyAny>, i32)> {
    let util = py
        .import("pyspark.sql.streaming.stateful_processor_util")?
        .getattr("TransformWithStateInPandasUdfUtils")?
        .call1((processor.clone(), time_mode))?;
    let (attr, et) = match (has_init_state, pandas) {
        (false, true) => (
            "transformWithStateUDF",
            eval_type::SQL_TRANSFORM_WITH_STATE_PANDAS_UDF,
        ),
        (true, true) => (
            "transformWithStateWithInitStateUDF",
            eval_type::SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF,
        ),
        (false, false) => (
            "transformWithStateUDF",
            eval_type::SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF,
        ),
        (true, false) => (
            "transformWithStateWithInitStateUDF",
            eval_type::SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF,
        ),
    };
    Ok((util.getattr(attr)?, et))
}

/// Python wrapper for cogrouped pandas/arrow ops (`pyspark.sql.group.PandasCogroupedOps`).
#[pyclass(name = "PandasCogroupedOps", module = "pyspark.sql.pandas.group_ops")]
pub struct PyCoGroupedData {
    pub(crate) inner: RustCoGroupedData,
}

#[pymethods]
impl PyCoGroupedData {
    /// `PandasCogroupedOps.applyInPandas`.
    #[pyo3(name = "applyInPandas")]
    fn apply_in_pandas(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let cols = py.detach(|| self.inner.input_columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "applyInPandas",
            &func,
            &schema,
            eval_type::SQL_COGROUPED_MAP_PANDAS_UDF,
            &cols,
        )?;
        Ok(PyDataFrame::new(self.inner.apply_in_pandas(udf)))
    }

    /// `PandasCogroupedOps.applyInArrow`.
    #[pyo3(name = "applyInArrow")]
    fn apply_in_arrow(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let cols = py.detach(|| self.inner.input_columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "applyInArrow",
            &func,
            &schema,
            eval_type::SQL_COGROUPED_MAP_ARROW_UDF,
            &cols,
        )?;
        Ok(PyDataFrame::new(self.inner.apply_in_arrow(udf)))
    }
}

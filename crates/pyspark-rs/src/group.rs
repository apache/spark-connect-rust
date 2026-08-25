//! PyO3 wrapper for spark_connect::group::GroupedData.

use pyo3::prelude::*;
use spark_connect::group::{CoGroupedData as RustCoGroupedData, GroupedData as RustGroupedData};
use spark_connect::udf::eval_type;

use crate::dataframe::{build_map_udf, PyDataFrame};
use crate::functions::to_column;

/// Python wrapper for Spark GroupedData.
#[pyclass(name = "GroupedData")]
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
    /// Aggregate with expressions (accepts Column objects).
    #[pyo3(signature = (*cols))]
    fn agg(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
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
        let udf = build_map_udf(
            py,
            "applyInPandas",
            &func,
            &schema,
            eval_type::SQL_GROUPED_MAP_PANDAS_UDF,
        )?;
        Ok(PyDataFrame::new(self.grouped_data.apply_in_pandas(udf)))
    }

    /// `GroupedData.applyInArrow`.
    #[pyo3(name = "applyInArrow")]
    fn apply_in_arrow(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let udf = build_map_udf(
            py,
            "applyInArrow",
            &func,
            &schema,
            eval_type::SQL_GROUPED_MAP_ARROW_UDF,
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
        let udf = build_map_udf(
            py,
            "applyInPandasWithState",
            &func,
            &output_struct_type,
            eval_type::SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE,
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

    /// `GroupedData.transformWithStateInPandas` (stateful streaming).
    #[pyo3(name = "transformWithStateInPandas")]
    #[pyo3(signature = (func, output_schema, output_mode, time_mode, event_time_column_name=None))]
    fn transform_with_state_in_pandas(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        output_schema: Bound<'_, PyAny>,
        output_mode: &str,
        time_mode: &str,
        event_time_column_name: Option<String>,
    ) -> PyResult<PyDataFrame> {
        let udf = build_map_udf(
            py,
            "transformWithStateInPandas",
            &func,
            &output_schema,
            eval_type::SQL_TRANSFORM_WITH_STATE_PANDAS_UDF,
        )?;
        let out = crate::dataframe::resolve_datatype(&output_schema)?;
        Ok(PyDataFrame::new(
            self.grouped_data.transform_with_state_in_pandas(
                udf,
                out,
                output_mode,
                time_mode,
                event_time_column_name.as_deref(),
                None,
            ),
        ))
    }

    /// `GroupedData.transformWithState` (row output, stateful streaming).
    #[pyo3(name = "transformWithState")]
    #[pyo3(signature = (func, output_schema, output_mode, time_mode, event_time_column_name=None))]
    fn transform_with_state(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        output_schema: Bound<'_, PyAny>,
        output_mode: &str,
        time_mode: &str,
        event_time_column_name: Option<String>,
    ) -> PyResult<PyDataFrame> {
        let udf = build_map_udf(
            py,
            "transformWithState",
            &func,
            &output_schema,
            eval_type::SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF,
        )?;
        Ok(PyDataFrame::new(self.grouped_data.transform_with_state(
            udf,
            output_mode,
            time_mode,
            event_time_column_name.as_deref(),
            None,
        )))
    }
}

/// Python wrapper for cogrouped pandas/arrow ops (`pyspark.sql.group.PandasCogroupedOps`).
#[pyclass(name = "PandasCogroupedOps")]
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
        let udf = build_map_udf(
            py,
            "applyInPandas",
            &func,
            &schema,
            eval_type::SQL_COGROUPED_MAP_PANDAS_UDF,
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
        let udf = build_map_udf(
            py,
            "applyInArrow",
            &func,
            &schema,
            eval_type::SQL_COGROUPED_MAP_ARROW_UDF,
        )?;
        Ok(PyDataFrame::new(self.inner.apply_in_arrow(udf)))
    }
}

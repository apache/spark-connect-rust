//! PyO3 wrapper for spark_connect::dataframe::DataFrame.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spark_connect::dataframe::{DataFrame, LocalRowIterator};
use spark_connect::plan::JoinType;
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::column::PyColumn;
use crate::errors::ResultExt;
use crate::group::PyGroupedData;
use crate::row::{value_to_py, PyRow};
use crate::streaming::PyDataStreamWriter;
use crate::types::PyDataType;

/// Python wrapper for a Spark DataFrame.
#[pyclass(name = "DataFrame")]
pub struct PyDataFrame {
    pub(crate) dataframe: DataFrame,
}

impl PyDataFrame {
    pub fn new(dataframe: DataFrame) -> Self {
        PyDataFrame { dataframe }
    }
}

/// Helper to convert arguments to a vector of Columns.
fn to_column_list(_args: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<spark_connect::column::Column>> {
    let mut cols = vec![];
    for arg in _args {
        // Try as PyColumn first
        if let Ok(pycol) = arg.extract::<Bound<'_, PyColumn>>() {
            cols.push(pycol.borrow().column.clone());
        } else {
            // Try as string (column name)
            if let Ok(name) = arg.extract::<String>() {
                cols.push(spark_connect::functions::col(&name));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Expected Column or column name (str)",
                ));
            }
        }
    }
    Ok(cols)
}

#[pymethods]
impl PyDataFrame {
    /// Select columns (accepts Column objects or string names).
    #[pyo3(signature = (*cols))]
    fn select(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let columns = to_column_list(cols)?;
        Ok(PyDataFrame::new(self.dataframe.select(columns)))
    }

    /// Select using SQL expressions.
    #[pyo3(signature = (*exprs))]
    fn selectExpr(&self, _py: Python<'_>, exprs: Vec<String>) -> PyDataFrame {
        let columns: Vec<_> = exprs
            .iter()
            .map(|e| spark_connect::functions::expr(e))
            .collect();
        PyDataFrame::new(self.dataframe.select(columns))
    }

    /// Filter rows by a condition.
    fn filter(&self, condition: &PyColumn) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.filter(condition.column.clone()))
    }

    /// Alias for filter (reference pyspark name is `where`).
    #[pyo3(name = "where")]
    fn where_(&self, condition: &PyColumn) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.filter(condition.column.clone()))
    }

    /// Add or replace a column.
    fn withColumn(&self, name: &str, col: &PyColumn) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.with_column(name, col.column.clone()))
    }

    /// Rename a column.
    fn withColumnRenamed(&self, existing: &str, new: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.with_column_renamed(existing, new))
    }

    /// Add or replace multiple columns from a {name: Column} mapping.
    #[pyo3(name = "withColumns")]
    fn with_columns(&self, cols: &Bound<'_, PyDict>) -> PyResult<PyDataFrame> {
        let mut pairs = Vec::with_capacity(cols.len());
        for (k, v) in cols.iter() {
            let name: String = k.extract()?;
            let pycol = v.extract::<PyRef<PyColumn>>()?;
            pairs.push((name, pycol.column.clone()));
        }
        Ok(PyDataFrame::new(self.dataframe.with_columns(pairs)))
    }

    /// Specify a relation hint (e.g. "broadcast"), with optional string parameters.
    #[pyo3(signature = (name, *parameters))]
    fn hint(&self, name: &str, parameters: Vec<String>) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.hint(name, parameters))
    }

    /// Drop columns.
    #[pyo3(signature = (*names))]
    fn drop(&self, _py: Python<'_>, names: Vec<String>) -> PyDataFrame {
        let col_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        PyDataFrame::new(self.dataframe.drop(col_refs))
    }

    /// Limit the number of rows.
    fn limit(&self, n: i32) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.limit(n))
    }

    /// Skip the first n rows.
    fn offset(&self, n: i32) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.offset(n))
    }

    /// Remove duplicate rows.
    fn distinct(&self) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.distinct())
    }

    /// Remove duplicate rows, optionally on specific columns.
    #[pyo3(signature = (subset=None))]
    fn dropDuplicates(&self, _py: Python<'_>, subset: Option<Vec<String>>) -> PyDataFrame {
        let col_refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|names| names.iter().map(|s| s.as_str()).collect());
        PyDataFrame::new(self.dataframe.drop_duplicates(col_refs))
    }

    /// Sort rows.
    #[pyo3(signature = (*cols))]
    fn sort(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let columns = to_column_list(cols)?;
        let exprs: Vec<_> = columns.iter().map(|c| c.expression().clone()).collect();
        Ok(PyDataFrame::new(self.dataframe.sort(exprs)))
    }

    /// Alias for sort.
    #[pyo3(signature = (*cols))]
    fn orderBy(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let columns = to_column_list(cols)?;
        let exprs: Vec<_> = columns.iter().map(|c| c.expression().clone()).collect();
        Ok(PyDataFrame::new(self.dataframe.order_by(exprs)))
    }

    /// Join with another DataFrame.
    ///
    /// `on` accepts a Column condition, a column-name string, or a list of
    /// column-name strings (name-based "using" join), mirroring
    /// `pyspark.sql.DataFrame.join(other, on=None, how=None)` (default inner).
    #[pyo3(signature = (other, on=None, how=None))]
    fn join(
        &self,
        other: &PyDataFrame,
        on: Option<Bound<'_, PyAny>>,
        how: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let join_type = match how.unwrap_or("inner").to_lowercase().as_str() {
            "inner" => JoinType::Inner,
            "left" | "leftouter" | "left_outer" => JoinType::LeftOuter,
            "right" | "rightouter" | "right_outer" => JoinType::RightOuter,
            "outer" | "full" | "fullouter" | "full_outer" => JoinType::FullOuter,
            "cross" => JoinType::Cross,
            "left_semi" | "leftsemi" | "semi" => JoinType::LeftSemi,
            "left_anti" | "leftanti" | "anti" => JoinType::LeftAnti,
            other => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid join type: {}",
                    other
                )))
            }
        };

        // Resolve `on`: None | Column | str | list[str].
        if let Some(bound) = on {
            if let Ok(col) = bound.extract::<PyRef<PyColumn>>() {
                let on_col = Some(col.column.clone());
                return Ok(PyDataFrame::new(self.dataframe.join(
                    &other.dataframe,
                    on_col,
                    join_type,
                )));
            }
            if let Ok(name) = bound.extract::<String>() {
                return Ok(PyDataFrame::new(self.dataframe.join_using(
                    &other.dataframe,
                    vec![name],
                    join_type,
                )));
            }
            if let Ok(names) = bound.extract::<Vec<String>>() {
                return Ok(PyDataFrame::new(self.dataframe.join_using(
                    &other.dataframe,
                    names,
                    join_type,
                )));
            }
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "join 'on' must be a Column, a str, or a list of str",
            ));
        }
        Ok(PyDataFrame::new(self.dataframe.join(
            &other.dataframe,
            None,
            join_type,
        )))
    }

    /// Cross join.
    fn crossJoin(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.cross_join(&other.dataframe))
    }

    /// Union with another DataFrame.
    fn union(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.union(&other.dataframe))
    }

    /// Union by name.
    fn unionByName(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.union_by_name(&other.dataframe))
    }

    /// Intersect with another DataFrame.
    fn intersect(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.intersect(&other.dataframe))
    }

    /// Intersect keeping duplicates (reference pyspark `intersectAll`).
    #[pyo3(name = "intersectAll")]
    fn intersect_all(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.intersect_all(&other.dataframe))
    }

    /// Subtract another DataFrame.
    fn subtract(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.subtract(&other.dataframe))
    }

    /// Repartition.
    fn repartition(&self, num_partitions: i32) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.repartition(num_partitions))
    }

    /// Coalesce.
    fn coalesce(&self, num_partitions: i32) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.coalesce(num_partitions))
    }

    /// Convert to DataFrame with new column names.
    #[pyo3(signature = (*names))]
    fn toDF(&self, _py: Python<'_>, names: Vec<String>) -> PyDataFrame {
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        PyDataFrame::new(self.dataframe.to_df(name_refs))
    }

    /// Alias this DataFrame.
    fn alias(&self, name: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.alias(name))
    }

    /// Group by columns for aggregation.
    #[pyo3(signature = (*cols))]
    fn groupBy(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyGroupedData> {
        let columns = to_column_list(cols)?;
        Ok(PyGroupedData::new(self.dataframe.group_by(columns)))
    }

    /// Rollup multi-dimensional aggregation.
    #[pyo3(signature = (*cols))]
    fn rollup(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyGroupedData> {
        Ok(PyGroupedData::new(
            self.dataframe.rollup(to_column_list(cols)?),
        ))
    }

    /// Cube multi-dimensional aggregation.
    #[pyo3(signature = (*cols))]
    fn cube(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyGroupedData> {
        Ok(PyGroupedData::new(
            self.dataframe.cube(to_column_list(cols)?),
        ))
    }

    /// Sort within each partition.
    #[pyo3(name = "sortWithinPartitions", signature = (*cols))]
    fn sort_within_partitions(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let exprs: Vec<_> = to_column_list(cols)?
            .iter()
            .map(|c| c.expression().clone())
            .collect();
        Ok(PyDataFrame::new(
            self.dataframe.sort_within_partitions(exprs),
        ))
    }

    /// Except all (keeps duplicates).
    #[pyo3(name = "exceptAll")]
    fn except_all(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.except_all(&other.dataframe))
    }

    /// Rename multiple columns from an {old: new} mapping.
    #[pyo3(name = "withColumnsRenamed")]
    fn with_columns_renamed(
        &self,
        renames: std::collections::HashMap<String, String>,
    ) -> PyDataFrame {
        PyDataFrame::new(
            self.dataframe
                .with_columns_renamed(renames.into_iter().collect()),
        )
    }

    /// Select columns whose names match a regex.
    #[pyo3(name = "colRegex")]
    fn col_regex(&self, col_name: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.col_regex(col_name))
    }

    /// Basic statistics for the given columns (count/mean/stddev/min/max).
    #[pyo3(signature = (*cols))]
    fn describe(&self, _py: Python<'_>, cols: Vec<String>) -> PyDataFrame {
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        PyDataFrame::new(self.dataframe.describe(refs))
    }

    /// Summary statistics (percentiles etc.).
    #[pyo3(signature = (*statistics))]
    fn summary(&self, _py: Python<'_>, statistics: Vec<String>) -> PyDataFrame {
        let refs: Vec<&str> = statistics.iter().map(|s| s.as_str()).collect();
        PyDataFrame::new(self.dataframe.summary(refs))
    }

    /// Persist with the default MEMORY_AND_DISK storage level.
    fn cache(&self, py: Python<'_>) -> PyResult<PyDataFrame> {
        let df = py.detach(|| self.dataframe.cache()).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Mark the DataFrame as non-persistent.
    #[pyo3(signature = (blocking=false))]
    fn unpersist(&self, py: Python<'_>, blocking: bool) -> PyResult<PyDataFrame> {
        let df = py
            .detach(|| self.dataframe.unpersist(blocking))
            .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Print the plan (the extended/mode flags are accepted for API parity).
    #[pyo3(signature = (extended=false, mode=None))]
    fn explain(&self, py: Python<'_>, extended: bool, mode: Option<String>) -> PyResult<()> {
        let _ = (extended, mode);
        py.detach(|| self.dataframe.explain()).to_pyerr()
    }

    /// Register this DataFrame as a temporary view.
    #[pyo3(name = "createOrReplaceTempView")]
    fn create_or_replace_temp_view(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| self.dataframe.create_or_replace_temp_view(name))
            .to_pyerr()
    }

    /// Drop rows containing nulls.
    #[pyo3(signature = (how="any", thresh=None, subset=None))]
    fn dropna(&self, how: &str, thresh: Option<i32>, subset: Option<Vec<String>>) -> PyDataFrame {
        let owned = subset;
        let refs = owned
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        PyDataFrame::new(self.dataframe.dropna(Some(how), thresh, refs))
    }

    /// Fill null values with a numeric value.
    #[pyo3(signature = (value, subset=None))]
    fn fillna(&self, value: i64, subset: Option<Vec<String>>) -> PyDataFrame {
        let owned = subset;
        let refs = owned
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        PyDataFrame::new(self.dataframe.fillna(value, refs))
    }

    /// Aggregate over the whole DataFrame (shorthand for `groupBy().agg(...)`).
    ///
    /// Mirrors `pyspark.sql.DataFrame.agg(*exprs)`.
    #[pyo3(signature = (*exprs))]
    fn agg(&self, _py: Python<'_>, exprs: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let columns = to_column_list(exprs)?;
        let expressions = columns.iter().map(|c| c.expression().clone()).collect();
        Ok(PyDataFrame::new(self.dataframe.agg(expressions)))
    }

    /// Collect all rows into memory.
    fn collect(&self, py: Python<'_>) -> PyResult<Vec<PyRow>> {
        // Release the GIL across the blocking RPC so other Python threads run.
        let rows = py.detach(|| self.dataframe.collect()).to_pyerr()?;
        Ok(rows.into_iter().map(PyRow::new).collect())
    }

    /// Return an iterator over rows without materializing the entire result in memory.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toLocalIterator(prefetchPartitions=False)`.
    /// Yields Row objects lazily as batches are fetched from the server.
    #[pyo3(name = "toLocalIterator")]
    #[pyo3(signature = (prefetchPartitions=false))]
    fn to_local_iterator(
        &self,
        py: Python<'_>,
        prefetchPartitions: bool,
    ) -> PyResult<PyLocalRowIterator> {
        let iterator = py
            .detach(|| self.dataframe.to_local_iterator(prefetchPartitions))
            .to_pyerr()?;
        Ok(PyLocalRowIterator::new(iterator))
    }

    /// Collect the DataFrame into a `pyarrow.Table`.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toArrow()`: the result is decoded from the
    /// server's Arrow IPC stream and handed to `pyarrow` (no row-by-row copy).
    #[pyo3(name = "toArrow")]
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::PyBytes;
        // `DataFrame::to_arrow` produces Arrow IPC *file* bytes (FileWriter).
        let bytes = py.detach(|| self.dataframe.to_arrow()).to_pyerr()?;
        let pa = py.import("pyarrow")?;
        let py_bytes = PyBytes::new(py, &bytes);
        let table = pa
            .getattr("ipc")?
            .call_method1("open_file", (py_bytes,))?
            .call_method0("read_all")?;
        Ok(table.unbind())
    }

    /// Collect the DataFrame into a pandas DataFrame.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toPandas()`: decodes the result via Arrow
    /// (`toArrow`) and converts the `pyarrow.Table` with `to_pandas()`.
    fn toPandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let table = self.to_arrow(py)?;
        let df = table.bind(py).call_method0("to_pandas")?;
        Ok(df.unbind())
    }

    /// Get the count of rows.
    fn count(&self, py: Python<'_>) -> PyResult<i64> {
        py.detach(|| self.dataframe.count()).to_pyerr()
    }

    /// Show the first n rows.
    fn show(&self, py: Python<'_>, n: usize) -> PyResult<()> {
        py.detach(|| self.dataframe.show(n)).to_pyerr()
    }

    /// Execution metrics from the most recent action. Mirrors `DataFrame.executionInfo`.
    #[pyo3(name = "executionInfo")]
    fn execution_info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let info = self.dataframe.execution_info().to_pyerr()?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("has_metrics", info.metrics.is_some())?;
        let names: Vec<String> = info
            .observed_metrics
            .iter()
            .map(|m| m.name.clone())
            .collect();
        dict.set_item("observed_metrics", names)?;
        Ok(dict.into_any().unbind())
    }

    /// Convert to a pandas-on-Spark DataFrame. Mirrors `DataFrame.pandas_api`;
    /// bridges into the vendored `pyspark.pandas`.
    #[pyo3(name = "pandas_api")]
    #[pyo3(signature = (index_col=None))]
    fn pandas_api(
        slf: Bound<'_, Self>,
        py: Python<'_>,
        index_col: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let ps = py.import("pyspark.pandas")?;
        let df_cls = ps.getattr("DataFrame")?;
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ic) = index_col {
            kwargs.set_item("index_col", ic)?;
        }
        Ok(df_cls.call((slf,), Some(&kwargs))?.unbind())
    }

    /// pandas-on-Spark plotting accessor. Mirrors `DataFrame.plot`.
    #[getter]
    fn plot(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let psdf = Self::pandas_api(slf, py, None)?;
        Ok(psdf.bind(py).getattr("plot")?.unbind())
    }

    /// Get the schema of this DataFrame.
    fn schema(&self) -> PyResult<PyDataType> {
        let schema = self.dataframe.schema().to_pyerr()?;
        Ok(PyDataType::new(schema))
    }

    /// Get the first row.
    fn first(&self, py: Python<'_>) -> PyResult<Option<PyRow>> {
        let row = py.detach(|| self.dataframe.first()).to_pyerr()?;
        Ok(row.map(PyRow::new))
    }

    /// Alias for first.
    fn head(&self, py: Python<'_>) -> PyResult<Option<PyRow>> {
        let row = py.detach(|| self.dataframe.first()).to_pyerr()?;
        Ok(row.map(PyRow::new))
    }

    /// Get the first n rows.
    fn take(&self, py: Python<'_>, n: usize) -> PyResult<Vec<PyRow>> {
        let rows = py.detach(|| self.dataframe.take(n)).to_pyerr()?;
        Ok(rows.into_iter().map(PyRow::new).collect())
    }

    /// Get column names.
    fn columns(&self) -> PyResult<Vec<String>> {
        self.dataframe.columns().to_pyerr()
    }

    /// Get the last n rows as a DataFrame (lazy evaluation).
    fn tail(&self, n: i32) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.tail(n))
    }

    /// Sample a fraction of rows.
    fn sample(&self, fraction: f64, seed: Option<i64>) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.sample(fraction, seed))
    }

    /// Check if the DataFrame is empty.
    fn isEmpty(&self) -> PyResult<bool> {
        self.dataframe.is_empty().to_pyerr()
    }

    /// Print the schema of the DataFrame.
    fn printSchema(&self) -> PyResult<()> {
        self.dataframe.show(0).to_pyerr()
    }

    /// Get the schema as a list of (name, type) tuples.
    fn dtypes(&self) -> PyResult<Vec<(String, String)>> {
        let schema = self.dataframe.schema().to_pyerr()?;
        match schema {
            spark_connect::types::DataType::Struct { fields } => {
                let dtypes = fields
                    .iter()
                    .map(|f| (f.name.clone(), f.data_type.simple_string()))
                    .collect();
                Ok(dtypes)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Schema is not a struct type",
            )),
        }
    }

    /// Additional set operation: unionAll (alias for union).
    fn unionAll(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.union(&other.dataframe))
    }

    /// The V1 writer interface (`df.write`).
    #[getter]
    fn write(&self) -> PyDataFrameWriter {
        PyDataFrameWriter {
            inner: Some(self.dataframe.write()),
        }
    }

    /// The V2 writer interface (`df.writeTo("table")`).
    #[pyo3(name = "writeTo")]
    fn write_to(&self, table_name: &str) -> PyDataFrameWriterV2 {
        PyDataFrameWriterV2 {
            inner: Some(self.dataframe.write_to(table_name)),
        }
    }

    /// The streaming writer interface (`df.writeStream`).
    #[getter]
    #[pyo3(name = "writeStream")]
    fn write_stream(&self) -> PyDataStreamWriter {
        let writer = self.dataframe.write_stream();
        PyDataStreamWriter::new(writer)
    }

    /// Null-handling functions (`df.na.drop()/fill()/replace()`).
    #[getter]
    fn na(&self) -> PyDataFrameNaFunctions {
        PyDataFrameNaFunctions {
            inner: self.dataframe.na(),
        }
    }

    /// Merge-into (`df.mergeInto("table", cond)`), returning a V2 merge builder.
    #[pyo3(name = "mergeInto")]
    fn merge_into(&self, table: &str, condition: &PyColumn) -> PyMergeIntoWriter {
        PyMergeIntoWriter {
            inner: Some(self.dataframe.merge_into(table, condition.column.clone())),
        }
    }

    /// Column access by name (`df["col"]`).
    fn __getitem__(&self, item: &str) -> PyColumn {
        PyColumn::new(spark_connect::functions::col(item))
    }

    fn __repr__(&self) -> String {
        "DataFrame".to_string()
    }

    /// `DataFrame.mapInPandas` — map an iterator of pandas DataFrames.
    #[pyo3(name = "mapInPandas", signature = (func, schema, is_barrier = false))]
    fn map_in_pandas(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
        is_barrier: bool,
    ) -> PyResult<PyDataFrame> {
        let udf = build_map_udf(
            py,
            "mapInPandas",
            &func,
            &schema,
            eval_type::SQL_MAP_PANDAS_ITER_UDF,
        )?;
        Ok(PyDataFrame::new(
            self.dataframe.map_in_pandas(udf, is_barrier),
        ))
    }

    /// `DataFrame.mapInArrow` — map an iterator of Arrow batches.
    #[pyo3(name = "mapInArrow", signature = (func, schema, is_barrier = false))]
    fn map_in_arrow(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
        is_barrier: bool,
    ) -> PyResult<PyDataFrame> {
        let udf = build_map_udf(
            py,
            "mapInArrow",
            &func,
            &schema,
            eval_type::SQL_MAP_ARROW_ITER_UDF,
        )?;
        Ok(PyDataFrame::new(
            self.dataframe.map_in_arrow(udf, is_barrier),
        ))
    }

    /// `DataFrame.foreach` — run a function per row for side effects.
    #[pyo3(name = "foreach")]
    fn foreach(&self, py: Python<'_>, func: Bound<'_, PyAny>) -> PyResult<()> {
        let udf = build_side_effect_udf(py, "foreach", &func)?;
        self.dataframe.foreach(udf).to_pyerr()
    }

    /// `DataFrame.foreachPartition` — run a function per partition for side effects.
    #[pyo3(name = "foreachPartition")]
    fn foreach_partition(&self, py: Python<'_>, func: Bound<'_, PyAny>) -> PyResult<()> {
        let udf = build_side_effect_udf(py, "foreachPartition", &func)?;
        self.dataframe.foreach_partition(udf).to_pyerr()
    }

    /// `DataFrame.nearestByJoin`.
    #[pyo3(name = "nearestByJoin", signature = (other, ranking_expression, num_results = 1, mode = "brute", direction = "asc", join_type = "inner"))]
    fn nearest_by_join(
        &self,
        other: &PyDataFrame,
        ranking_expression: &PyColumn,
        num_results: i32,
        mode: &str,
        direction: &str,
        join_type: &str,
    ) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.nearest_by_join(
            &other.dataframe,
            ranking_expression.column.clone(),
            num_results,
            mode,
            direction,
            join_type,
        ))
    }

    /// `DataFrame.metadataColumn`.
    #[pyo3(name = "metadataColumn")]
    fn metadata_column(&self, name: &str) -> PyColumn {
        PyColumn::new(self.dataframe.metadata_column(name))
    }
}

/// Python wrapper for `DataFrameNaFunctions` (`df.na`).
#[pyclass(name = "DataFrameNaFunctions")]
pub struct PyDataFrameNaFunctions {
    inner: spark_connect::group::NaFunctions,
}

#[pymethods]
impl PyDataFrameNaFunctions {
    #[pyo3(signature = (how=None, thresh=None, subset=None))]
    fn drop(
        &self,
        how: Option<String>,
        thresh: Option<i32>,
        subset: Option<Vec<String>>,
    ) -> PyDataFrame {
        let subset_refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        PyDataFrame::new(self.inner.drop(how.as_deref(), thresh, subset_refs))
    }

    #[pyo3(signature = (value, subset=None))]
    fn fill(&self, value: i64, subset: Option<Vec<String>>) -> PyDataFrame {
        let subset_refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        PyDataFrame::new(self.inner.fill(value, subset_refs))
    }

    #[pyo3(signature = (to_replace, subset=None))]
    fn replace(
        &self,
        to_replace: Vec<(String, String)>,
        subset: Option<Vec<String>>,
    ) -> PyDataFrame {
        let subset_refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        PyDataFrame::new(self.inner.replace(to_replace, subset_refs))
    }
}

/// Python wrapper for `MergeIntoWriter` (`df.mergeInto(...)`).
#[pyclass(name = "MergeIntoWriter")]
pub struct PyMergeIntoWriter {
    inner: Option<spark_connect::MergeIntoWriter>,
}

impl PyMergeIntoWriter {
    fn take(&mut self) -> PyResult<spark_connect::MergeIntoWriter> {
        self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("MergeIntoWriter already consumed")
        })
    }
}

#[pymethods]
impl PyMergeIntoWriter {
    #[pyo3(name = "whenMatched", signature = (condition=None))]
    fn when_matched(&mut self, condition: Option<&PyColumn>) -> PyResult<PyWhenMatched> {
        let cond = condition.map(|c| c.column.clone());
        Ok(PyWhenMatched {
            inner: Some(self.take()?.when_matched(cond)),
        })
    }

    #[pyo3(name = "whenNotMatched", signature = (condition=None))]
    fn when_not_matched(&mut self, condition: Option<&PyColumn>) -> PyResult<PyWhenNotMatched> {
        let cond = condition.map(|c| c.column.clone());
        Ok(PyWhenNotMatched {
            inner: Some(self.take()?.when_not_matched(cond)),
        })
    }

    #[pyo3(name = "whenNotMatchedBySource", signature = (condition=None))]
    fn when_not_matched_by_source(
        &mut self,
        condition: Option<&PyColumn>,
    ) -> PyResult<PyWhenNotMatchedBySource> {
        let cond = condition.map(|c| c.column.clone());
        Ok(PyWhenNotMatchedBySource {
            inner: Some(self.take()?.when_not_matched_by_source(cond)),
        })
    }

    #[pyo3(name = "withSchemaEvolution")]
    fn with_schema_evolution(&mut self) -> PyResult<PyMergeIntoWriter> {
        Ok(PyMergeIntoWriter {
            inner: Some(self.take()?.with_schema_evolution()),
        })
    }

    fn merge(&mut self) -> PyResult<()> {
        self.take()?.merge().to_pyerr()
    }
}

fn assignments_to_map(
    py: Python<'_>,
    assignments: std::collections::HashMap<String, Py<PyColumn>>,
) -> std::collections::HashMap<String, spark_connect::column::Column> {
    assignments
        .into_iter()
        .map(|(k, v)| (k, v.borrow(py).column.clone()))
        .collect()
}

/// `whenMatched(...)` clause builder.
#[pyclass(name = "WhenMatched")]
pub struct PyWhenMatched {
    inner: Option<spark_connect::merge::WhenMatched>,
}

#[pymethods]
impl PyWhenMatched {
    #[pyo3(name = "updateAll")]
    fn update_all(&mut self) -> PyResult<PyMergeIntoWriter> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("clause already consumed"))?;
        Ok(PyMergeIntoWriter {
            inner: Some(w.update_all()),
        })
    }

    fn update(
        &mut self,
        py: Python<'_>,
        assignments: std::collections::HashMap<String, Py<PyColumn>>,
    ) -> PyResult<PyMergeIntoWriter> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("clause already consumed"))?;
        Ok(PyMergeIntoWriter {
            inner: Some(w.update(assignments_to_map(py, assignments))),
        })
    }

    fn delete(&mut self) -> PyResult<PyMergeIntoWriter> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("clause already consumed"))?;
        Ok(PyMergeIntoWriter {
            inner: Some(w.delete()),
        })
    }
}

/// `whenNotMatched(...)` clause builder.
#[pyclass(name = "WhenNotMatched")]
pub struct PyWhenNotMatched {
    inner: Option<spark_connect::merge::WhenNotMatched>,
}

#[pymethods]
impl PyWhenNotMatched {
    #[pyo3(name = "insertAll")]
    fn insert_all(&mut self) -> PyResult<PyMergeIntoWriter> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("clause already consumed"))?;
        Ok(PyMergeIntoWriter {
            inner: Some(w.insert_all()),
        })
    }

    fn insert(
        &mut self,
        py: Python<'_>,
        assignments: std::collections::HashMap<String, Py<PyColumn>>,
    ) -> PyResult<PyMergeIntoWriter> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("clause already consumed"))?;
        Ok(PyMergeIntoWriter {
            inner: Some(w.insert(assignments_to_map(py, assignments))),
        })
    }
}

/// `whenNotMatchedBySource(...)` clause builder.
#[pyclass(name = "WhenNotMatchedBySource")]
pub struct PyWhenNotMatchedBySource {
    inner: Option<spark_connect::merge::WhenNotMatchedBySource>,
}

#[pymethods]
impl PyWhenNotMatchedBySource {
    #[pyo3(name = "updateAll")]
    fn update_all(&mut self) -> PyResult<PyMergeIntoWriter> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("clause already consumed"))?;
        Ok(PyMergeIntoWriter {
            inner: Some(w.update_all()),
        })
    }

    fn update(
        &mut self,
        py: Python<'_>,
        assignments: std::collections::HashMap<String, Py<PyColumn>>,
    ) -> PyResult<PyMergeIntoWriter> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("clause already consumed"))?;
        Ok(PyMergeIntoWriter {
            inner: Some(w.update(assignments_to_map(py, assignments))),
        })
    }

    fn delete(&mut self) -> PyResult<PyMergeIntoWriter> {
        let w = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("clause already consumed"))?;
        Ok(PyMergeIntoWriter {
            inner: Some(w.delete()),
        })
    }
}

/// Python wrapper for the V1 `DataFrameWriter` (`df.write`).
#[pyclass(name = "DataFrameWriter")]
pub struct PyDataFrameWriter {
    inner: Option<spark_connect::readwriter::DataFrameWriter>,
}

impl PyDataFrameWriter {
    fn take(&mut self) -> PyResult<spark_connect::readwriter::DataFrameWriter> {
        self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("DataFrameWriter already consumed")
        })
    }
}

#[pymethods]
impl PyDataFrameWriter {
    fn mode(&mut self, mode: &str) -> PyResult<PyDataFrameWriter> {
        Ok(PyDataFrameWriter {
            inner: Some(self.take()?.mode(mode)),
        })
    }

    fn format(&mut self, source: &str) -> PyResult<PyDataFrameWriter> {
        Ok(PyDataFrameWriter {
            inner: Some(self.take()?.format(source)),
        })
    }

    fn option(&mut self, key: &str, value: &str) -> PyResult<PyDataFrameWriter> {
        Ok(PyDataFrameWriter {
            inner: Some(self.take()?.option(key, value)),
        })
    }

    fn options(
        &mut self,
        options: std::collections::HashMap<String, String>,
    ) -> PyResult<PyDataFrameWriter> {
        Ok(PyDataFrameWriter {
            inner: Some(self.take()?.options(options)),
        })
    }

    #[pyo3(name = "partitionBy", signature = (*cols))]
    fn partition_by(&mut self, cols: Vec<String>) -> PyResult<PyDataFrameWriter> {
        Ok(PyDataFrameWriter {
            inner: Some(self.take()?.partition_by(cols)),
        })
    }

    #[pyo3(name = "bucketBy", signature = (num_buckets, *cols))]
    fn bucket_by(&mut self, num_buckets: i32, cols: Vec<String>) -> PyResult<PyDataFrameWriter> {
        Ok(PyDataFrameWriter {
            inner: Some(self.take()?.bucket_by(num_buckets, cols)),
        })
    }

    #[pyo3(name = "sortBy", signature = (*cols))]
    fn sort_by(&mut self, cols: Vec<String>) -> PyResult<PyDataFrameWriter> {
        Ok(PyDataFrameWriter {
            inner: Some(self.take()?.sort_by(cols)),
        })
    }

    #[pyo3(signature = (path=None))]
    fn save(&mut self, path: Option<String>) -> PyResult<()> {
        self.take()?.save(path.as_deref()).to_pyerr()
    }

    #[pyo3(name = "saveAsTable")]
    fn save_as_table(&mut self, table_name: &str) -> PyResult<()> {
        self.take()?.save_as_table(table_name).to_pyerr()
    }

    #[pyo3(name = "insertInto")]
    fn insert_into(&mut self, table_name: &str) -> PyResult<()> {
        self.take()?.insert_into(table_name).to_pyerr()
    }

    fn parquet(&mut self, path: &str) -> PyResult<()> {
        self.take()?.parquet(path).to_pyerr()
    }

    fn json(&mut self, path: &str) -> PyResult<()> {
        self.take()?.json(path).to_pyerr()
    }

    fn csv(&mut self, path: &str) -> PyResult<()> {
        self.take()?.csv(path).to_pyerr()
    }

    fn orc(&mut self, path: &str) -> PyResult<()> {
        self.take()?.orc(path).to_pyerr()
    }

    fn text(&mut self, path: &str) -> PyResult<()> {
        self.take()?.text(path).to_pyerr()
    }
}

/// Python wrapper for the V2 `DataFrameWriter` (`df.writeTo`).
#[pyclass(name = "DataFrameWriterV2")]
pub struct PyDataFrameWriterV2 {
    inner: Option<spark_connect::readwriter::DataFrameWriterV2>,
}

impl PyDataFrameWriterV2 {
    fn take(&mut self) -> PyResult<spark_connect::readwriter::DataFrameWriterV2> {
        self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("DataFrameWriterV2 already consumed")
        })
    }
}

#[pymethods]
impl PyDataFrameWriterV2 {
    fn using(&mut self, provider: &str) -> PyResult<PyDataFrameWriterV2> {
        Ok(PyDataFrameWriterV2 {
            inner: Some(self.take()?.using(provider)),
        })
    }

    fn option(&mut self, key: &str, value: &str) -> PyResult<PyDataFrameWriterV2> {
        Ok(PyDataFrameWriterV2 {
            inner: Some(self.take()?.option(key, value)),
        })
    }

    fn options(
        &mut self,
        options: std::collections::HashMap<String, String>,
    ) -> PyResult<PyDataFrameWriterV2> {
        Ok(PyDataFrameWriterV2 {
            inner: Some(self.take()?.options(options)),
        })
    }

    #[pyo3(name = "tableProperty")]
    fn table_property(&mut self, property: &str, value: &str) -> PyResult<PyDataFrameWriterV2> {
        Ok(PyDataFrameWriterV2 {
            inner: Some(self.take()?.table_property(property, value)),
        })
    }

    #[pyo3(name = "partitionedBy", signature = (*cols))]
    fn partitioned_by(&mut self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrameWriterV2> {
        let columns = to_column_list(cols)?;
        Ok(PyDataFrameWriterV2 {
            inner: Some(self.take()?.partition_by(columns)),
        })
    }

    fn create(&mut self) -> PyResult<()> {
        self.take()?.create().to_pyerr()
    }

    fn replace(&mut self) -> PyResult<()> {
        self.take()?.replace().to_pyerr()
    }

    #[pyo3(name = "createOrReplace")]
    fn create_or_replace(&mut self) -> PyResult<()> {
        self.take()?.create_or_replace().to_pyerr()
    }

    fn append(&mut self) -> PyResult<()> {
        self.take()?.append().to_pyerr()
    }

    fn overwrite(&mut self, condition: &PyColumn) -> PyResult<()> {
        self.take()?.overwrite(condition.column.clone()).to_pyerr()
    }

    #[pyo3(name = "overwritePartitions")]
    fn overwrite_partitions(&mut self) -> PyResult<()> {
        self.take()?.overwrite_partitions().to_pyerr()
    }
}

// ---- Shared helpers for closure-based (UDF) DataFrame/GroupedData methods ----

/// cloudpickle an arbitrary Python object to bytes (mirrors the scalar UDF path,
/// where the client serializes the callable for server-side execution).
///
/// Uses the bundled `pyspark.cloudpickle` (the skin vendors it, matching the worker's
/// `pyspark.cloudpickle`), so there is no external `cloudpickle` runtime dependency.
pub(crate) fn py_cloudpickle(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let cp = py.import("pyspark.cloudpickle")?;
    cp.call_method1("dumps", (obj,))?.extract()
}

/// The running Python's "major.minor" (recorded in the UDF payload).
pub(crate) fn py_version(py: Python<'_>) -> String {
    (|| -> Option<String> {
        let vi = py.import("sys").ok()?.getattr("version_info").ok()?;
        let major: i64 = vi.getattr("major").ok()?.extract().ok()?;
        let minor: i64 = vi.getattr("minor").ok()?.extract().ok()?;
        Some(format!("{major}.{minor}"))
    })()
    .unwrap_or_else(|| "3.11".to_string())
}

/// Resolve a Python schema argument (a `DataType` wrapper or a DDL string) into a Rust DataType.
pub(crate) fn resolve_datatype(
    schema: &Bound<'_, PyAny>,
) -> PyResult<spark_connect::types::DataType> {
    if let Ok(dt) = schema.extract::<PyRef<PyDataType>>() {
        return Ok(dt.inner.clone());
    }
    if let Ok(s) = schema.extract::<String>() {
        return spark_connect::types::DataType::from_ddl(&s).to_pyerr();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "schema must be a DataType or a DDL string",
    ))
}

/// Build a map/group UDF expression: cloudpickle `(func, schema)` as the command and
/// carry the resolved schema as the output type (mirrors the scalar UDF command layout).
pub(crate) fn build_map_udf(
    py: Python<'_>,
    name: &str,
    func: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
    eval_type_id: i32,
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let return_type = resolve_datatype(schema)?;
    let tup = PyTuple::new(py, [func.clone(), schema.clone()])?;
    let command = py_cloudpickle(py, tup.as_any())?;
    let payload = PythonUDFPayload::new(return_type, eval_type_id, command, py_version(py));
    Ok(CommonInlineUserDefinedFunctionExpression::new(
        name.to_string(),
        true,
        vec![],
        payload,
    ))
}

/// Build a side-effect UDF (foreach / foreachPartition): no output schema.
fn build_side_effect_udf(
    py: Python<'_>,
    name: &str,
    func: &Bound<'_, PyAny>,
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let command = py_cloudpickle(py, func)?;
    let payload = PythonUDFPayload::new(
        spark_connect::types::DataType::Struct { fields: vec![] },
        eval_type::SQL_MAP_ARROW_ITER_UDF,
        command,
        py_version(py),
    );
    Ok(CommonInlineUserDefinedFunctionExpression::new(
        name.to_string(),
        true,
        vec![],
        payload,
    ))
}

/// Python wrapper for LocalRowIterator yielding Row objects lazily.
///
/// Implements `__iter__` and `__next__` to make it a proper Python iterator.
#[pyclass(name = "LocalRowIterator")]
pub struct PyLocalRowIterator {
    iterator: LocalRowIterator,
}

impl PyLocalRowIterator {
    pub fn new(iterator: LocalRowIterator) -> Self {
        PyLocalRowIterator { iterator }
    }
}

#[pymethods]
impl PyLocalRowIterator {
    /// Make this object an iterator by returning self.
    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    /// Fetch the next row from the iterator.
    ///
    /// Releases the GIL during the blocking fetch to allow other threads to run.
    fn __next__(&mut self, py: Python<'_>) -> Option<PyRow> {
        loop {
            // Release the GIL while fetching from the Rust iterator (which may block on network I/O).
            let next_result = py.detach(|| self.iterator.next());
            match next_result {
                Some(Ok(row)) => return Some(PyRow::new(row)),
                Some(Err(_e)) => return None, // On error, stop iterating
                None => return None,
            }
        }
    }
}

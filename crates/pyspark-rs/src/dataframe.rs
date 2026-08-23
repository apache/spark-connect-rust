//! PyO3 wrapper for spark_connect::dataframe::DataFrame.

use pyo3::prelude::*;
use spark_connect::dataframe::DataFrame;
use spark_connect::plan::JoinType;

use crate::column::PyColumn;
use crate::errors::ResultExt;
use crate::group::PyGroupedData;
use crate::row::PyRow;
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

    /// Alias for filter.
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
    fn collect(&self) -> PyResult<Vec<PyRow>> {
        let rows = self.dataframe.collect().to_pyerr()?;
        Ok(rows.into_iter().map(PyRow::new).collect())
    }

    /// Collect the DataFrame into a pandas DataFrame.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toPandas()`: collects all rows and builds a
    /// `pandas.DataFrame` with the DataFrame's column names (dtypes inferred by pandas).
    fn toPandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyDict, PyList};
        let columns = self.dataframe.columns().to_pyerr()?;
        let rows = self.dataframe.collect().to_pyerr()?;

        let data = PyList::empty(py);
        for row in &rows {
            let py_row = PyList::empty(py);
            for i in 0..columns.len() {
                match row.get(i) {
                    Some(val) => py_row.append(crate::row::value_to_py(py, val))?,
                    None => py_row.append(py.None())?,
                }
            }
            data.append(py_row)?;
        }

        let pandas = py.import("pandas")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("columns", columns)?;
        let df = pandas.getattr("DataFrame")?.call((data,), Some(&kwargs))?;
        Ok(df.unbind())
    }

    /// Get the count of rows.
    fn count(&self) -> PyResult<i64> {
        self.dataframe.count().to_pyerr()
    }

    /// Show the first n rows.
    fn show(&self, n: usize) -> PyResult<()> {
        self.dataframe.show(n).to_pyerr()
    }

    /// Get the schema of this DataFrame.
    fn schema(&self) -> PyResult<PyDataType> {
        let schema = self.dataframe.schema().to_pyerr()?;
        Ok(PyDataType::new(schema))
    }

    /// Get the first row.
    fn first(&self) -> PyResult<Option<PyRow>> {
        let row = self.dataframe.first().to_pyerr()?;
        Ok(row.map(PyRow::new))
    }

    /// Alias for first.
    fn head(&self) -> PyResult<Option<PyRow>> {
        let row = self.dataframe.first().to_pyerr()?;
        Ok(row.map(PyRow::new))
    }

    /// Get the first n rows.
    fn take(&self, n: usize) -> PyResult<Vec<PyRow>> {
        let rows = self.dataframe.take(n).to_pyerr()?;
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
}

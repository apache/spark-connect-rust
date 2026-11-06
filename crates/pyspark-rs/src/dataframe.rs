//! PyO3 wrapper for spark_connect::dataframe::DataFrame.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};
use spark_connect::dataframe::{DataFrame, LocalRowIterator};
use spark_connect::plan::JoinType;
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::column::PyColumn;
use crate::errors::ResultExt;
use crate::functions::to_column;
use crate::group::PyGroupedData;
use crate::row::{FieldConverters, PyRow};
use crate::streaming::PyDataStreamWriter;
use crate::types::PyDataType;

/// Apply a single writer option (skipping None; bool->"true"/"false").
fn wset_opt(
    w: spark_connect::readwriter::DataFrameWriter,
    name: &str,
    v: Option<&Bound<'_, PyAny>>,
) -> PyResult<spark_connect::readwriter::DataFrameWriter> {
    match v {
        Some(x) => match crate::coerce_option_value(x)? {
            Some(sv) => Ok(w.option(name, &sv)),
            None => Ok(w),
        },
        None => Ok(w),
    }
}

/// Flatten `*cols` args that are each a str or a list of str (pyspark
/// partitionBy/bucketBy/sortBy accept both forms).
fn flatten_str_cols(cols: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let mut out = Vec::new();
    for c in cols {
        if let Ok(one) = c.extract::<String>() {
            out.push(one);
        } else {
            out.extend(c.extract::<Vec<String>>()?);
        }
    }
    Ok(out)
}

/// Python wrapper for a Spark DataFrame.
#[pyclass(name = "DataFrame", module = "pyspark.sql.connect.dataframe", subclass)]
pub struct PyDataFrame {
    pub(crate) dataframe: DataFrame,
}

impl PyDataFrame {
    /// The pandas DataFrame of the result, as `SparkConnectClient.to_pandas` makes it: the
    /// settings of the conversion first, in one request; then the Arrow result with the schema
    /// that the server sends, which `pyspark.sql.connect.client._to_pandas` converts.
    fn to_pandas(
        &self,
        py: Python<'_>,
        struct_handling_mode: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let settings = py
            .detach(|| {
                self.dataframe.spark_session().conf().get_many(&[
                    "spark.sql.session.timeZone",
                    "spark.sql.execution.pandas.structHandlingMode",
                    "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
                ])
            })
            .to_pyerr()?;
        let (batches, schema) = py
            .detach(|| self.dataframe.collect_record_batches_with_schema())
            .to_pyerr()?;
        let table = crate::pyarrow_export::record_batches_to_pyarrow(py, batches)?;
        let schema = match &schema {
            Some(schema) => crate::types::data_type_to_py(py, schema)?,
            None => py.None(),
        };
        let (metrics, observed) = match self.dataframe.execution_info() {
            Ok(info) => execution_metrics(py, &info)?,
            Err(_) => (
                pyo3::types::PyList::empty(py),
                pyo3::types::PyList::empty(py),
            ),
        };
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("timezone", &settings[0])?;
        match struct_handling_mode {
            Some(mode) => kwargs.set_item("struct_handling_mode", mode.str()?)?,
            None => kwargs.set_item("struct_handling_mode", &settings[1])?,
        }
        kwargs.set_item("self_destruct", &settings[2])?;
        let pdf = py.import("pyspark.sql.connect.client")?.call_method(
            "_to_pandas",
            (table, schema, metrics, observed),
            Some(&kwargs),
        )?;
        Ok(pdf.unbind())
    }

    /// The table of `show` and `_show_string`, for the arguments of `function`.
    fn show_table(
        &self,
        py: Python<'_>,
        function: &str,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        let [n, truncate, vertical] = bind_arguments(
            function,
            args,
            kwargs,
            ["n", "truncate", "vertical"],
            [
                20i32.into_pyobject(py)?.into_any(),
                pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                pyo3::types::PyBool::new(py, false).to_owned().into_any(),
            ],
        )?;
        let not_expected = |expected: &str, name: &str, value: &Bound<'_, PyAny>| -> PyErr {
            let type_name = value
                .get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_default();
            crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("expected_type", expected),
                    ("arg_name", name),
                    ("arg_type", &type_name),
                ],
            )
        };
        if n.is_instance_of::<pyo3::types::PyBool>() || !n.is_instance_of::<pyo3::types::PyInt>() {
            return Err(not_expected("int", "n", &n));
        }
        if !vertical.is_instance_of::<pyo3::types::PyBool>() {
            return Err(not_expected("bool", "vertical", &vertical));
        }
        let truncate =
            if truncate.is_instance_of::<pyo3::types::PyBool>() && truncate.is_truthy()? {
                20
            } else {
                // As `int(truncate)`, whose ValueError is PySpark's NOT_EXPECTED_TYPE.
                match py.import("builtins")?.getattr("int")?.call1((&truncate,)) {
                    Ok(value) => value.extract::<i32>()?,
                    Err(err) if err.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {
                        return Err(not_expected("bool", "truncate", &truncate));
                    }
                    Err(err) => return Err(err),
                }
            };
        let (n, vertical) = (n.extract::<i32>()?, vertical.extract::<bool>()?);
        py.detach(|| self.dataframe.show_string(n, truncate, vertical))
            .to_pyerr()
    }

    pub fn new(dataframe: DataFrame) -> Self {
        PyDataFrame { dataframe }
    }
}

/// Helper to convert arguments to a vector of Columns.
pub(crate) fn to_column_list(
    mut _args: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<spark_connect::column::Column>> {
    // pyspark unpacks a single list/tuple argument: df.select(["a", "b"]) == df.select("a","b").
    // (A single Column/str stays as-is.)
    if _args.len() == 1 {
        let only = &_args[0];
        if only.is_instance_of::<pyo3::types::PyList>()
            || only.is_instance_of::<pyo3::types::PyTuple>()
        {
            _args = only.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        }
    }
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

/// Build sort-order expressions for sort/orderBy/sortWithinPartitions, honoring an
/// optional `ascending` kwarg (a bool applied to all columns, or a per-column list of
/// bools/ints). Mirrors reference pyspark, which wraps each column with asc()/desc()
/// when `ascending` is given; without it, columns are used as-is.
fn sort_order_exprs(
    columns: Vec<spark_connect::column::Column>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<spark_connect::expression::Expression>> {
    let ascending = match kwargs {
        Some(kw) => kw.get_item("ascending")?,
        None => None,
    };
    let Some(ascending) = ascending else {
        return Ok(columns.iter().map(|c| c.expression().clone()).collect());
    };
    let flags: Vec<bool> = if let Ok(b) = ascending.extract::<bool>() {
        vec![b; columns.len()]
    } else if let Ok(i) = ascending.extract::<i64>() {
        vec![i != 0; columns.len()]
    } else if let Ok(list) = ascending.extract::<Vec<bool>>() {
        list
    } else if let Ok(list) = ascending.extract::<Vec<i64>>() {
        list.into_iter().map(|i| i != 0).collect()
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "ascending must be a bool, an int, or a list of bools/ints",
        ));
    };
    if flags.len() != columns.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "the list length of ascending ({}) must equal the number of sort columns ({})",
            flags.len(),
            columns.len()
        )));
    }
    Ok(columns
        .into_iter()
        .zip(flags)
        .map(|(c, asc)| (if asc { c.asc() } else { c.desc() }).expression().clone())
        .collect())
}

/// Map a pyspark `how` string to a JoinType (shared by join / lateralJoin).
fn parse_join_type(how: &str) -> PyResult<JoinType> {
    Ok(match how.to_lowercase().as_str() {
        "inner" => JoinType::Inner,
        "left" | "leftouter" | "left_outer" => JoinType::LeftOuter,
        "right" | "rightouter" | "right_outer" => JoinType::RightOuter,
        "outer" | "full" | "fullouter" | "full_outer" => JoinType::FullOuter,
        "cross" => JoinType::Cross,
        "left_semi" | "leftsemi" | "semi" => JoinType::LeftSemi,
        "left_anti" | "leftanti" | "anti" => JoinType::LeftAnti,
        other => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid join type: {other}"
            )))
        }
    })
}

/// Resolve a filter/where condition: a Column, or a SQL string parsed via `expr`.
fn cond_column(condition: &Bound<'_, PyAny>) -> PyResult<spark_connect::column::Column> {
    if let Ok(c) = condition.extract::<PyRef<PyColumn>>() {
        Ok(c.column.clone())
    } else {
        let s: String = condition.extract()?;
        Ok(spark_connect::functions::expr(&s))
    }
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
    #[pyo3(signature = (*expr))]
    fn selectExpr(&self, _py: Python<'_>, expr: Vec<String>) -> PyDataFrame {
        let columns: Vec<_> = expr
            .iter()
            .map(|e| spark_connect::functions::expr(e))
            .collect();
        PyDataFrame::new(self.dataframe.select(columns))
    }

    /// Filter rows by a Column condition or a SQL-string condition.
    fn filter(&self, condition: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            self.dataframe.filter(cond_column(condition)?),
        ))
    }

    /// Alias for filter (reference pyspark name is `where`).
    #[pyo3(name = "where")]
    fn where_(&self, condition: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            self.dataframe.filter(cond_column(condition)?),
        ))
    }

    /// Add or replace a column.
    fn withColumn(&self, colName: &str, col: &PyColumn) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.with_column(colName, col.column.clone()))
    }

    /// Rename a column.
    fn withColumnRenamed(&self, existing: &str, new: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.with_column_renamed(existing, new))
    }

    /// Add or replace multiple columns from a {name: Column} mapping.
    #[pyo3(name = "withColumns", signature = (*colsMap))]
    fn with_columns(&self, colsMap: Vec<Bound<'_, PyDict>>) -> PyResult<PyDataFrame> {
        let mut pairs = Vec::new();
        for d in &colsMap {
            for (k, v) in d.iter() {
                let name: String = k.extract()?;
                let pycol = v.extract::<PyRef<PyColumn>>()?;
                pairs.push((name, pycol.column.clone()));
            }
        }
        Ok(PyDataFrame::new(self.dataframe.with_columns(pairs)))
    }

    /// Specify a relation hint (e.g. "broadcast"), with optional parameters. Reference
    /// hint params may be ints (e.g. `df.hint("rebalance", 3)`), not just strings, so
    /// accept any value and stringify it; the core parses a numeric string back into a
    /// long literal, matching reference semantics.
    #[pyo3(signature = (name, *parameters))]
    fn hint(&self, name: &str, parameters: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let params: Vec<String> = parameters
            .iter()
            .map(|p| Ok(p.str()?.to_string()))
            .collect::<PyResult<_>>()?;
        Ok(PyDataFrame::new(self.dataframe.hint(name, params)))
    }

    /// Drop columns.
    #[pyo3(signature = (*cols))]
    fn drop(&self, _py: Python<'_>, cols: Vec<String>) -> PyDataFrame {
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        PyDataFrame::new(self.dataframe.drop(col_refs))
    }

    /// Limit the number of rows.
    fn limit(&self, num: i32) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.limit(num))
    }

    /// Skip the first n rows.
    fn offset(&self, num: i32) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.offset(num))
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

    /// Sort rows. `ascending` (a bool or list of bools, via kwargs) mirrors reference
    /// pyspark: it wraps each sort column with asc()/desc().
    #[pyo3(signature = (*cols, **kwargs))]
    fn sort(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = sort_order_exprs(to_column_list(cols)?, kwargs.as_ref())?;
        Ok(PyDataFrame::new(self.dataframe.sort(exprs)))
    }

    /// Alias for sort.
    #[pyo3(signature = (*cols, **kwargs))]
    fn orderBy(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = sort_order_exprs(to_column_list(cols)?, kwargs.as_ref())?;
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
        let join_type = parse_join_type(how.unwrap_or("inner"))?;

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

    /// Lateral (correlated) join. `on` is an optional Column condition; `how`
    /// mirrors `join` (default inner). Mirrors `DataFrame.lateralJoin`.
    #[pyo3(signature = (other, on=None, how=None))]
    fn lateralJoin(
        &self,
        other: &PyDataFrame,
        on: Option<PyRef<PyColumn>>,
        how: Option<&str>,
    ) -> PyResult<PyDataFrame> {
        let join_type = parse_join_type(how.unwrap_or("inner"))?;
        let on_col = on.map(|c| c.column.clone());
        Ok(PyDataFrame::new(self.dataframe.lateral_join(
            &other.dataframe,
            on_col,
            join_type,
        )))
    }

    /// Transpose (swap rows/columns). `indexColumn` optionally names the header
    /// column. Mirrors `DataFrame.transpose(indexColumn=None)`.
    #[pyo3(signature = (indexColumn=None))]
    #[allow(non_snake_case)]
    fn transpose(&self, indexColumn: Option<PyRef<PyColumn>>) -> PyResult<PyDataFrame> {
        let df = match indexColumn {
            Some(c) => self.dataframe.transpose_with_index(c.column.clone()),
            None => self.dataframe.transpose(),
        }
        .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Repartition into `numPartitions` by a range partitioning on the given
    /// columns. Mirrors `DataFrame.repartitionByRange(numPartitions, *cols)`.
    #[pyo3(signature = (numPartitions, *cols))]
    #[allow(non_snake_case)]
    fn repartitionByRange(
        &self,
        numPartitions: i32,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let columns = to_column_list(cols)?;
        let exprs: Vec<_> = columns.iter().map(|c| c.expression().clone()).collect();
        Ok(PyDataFrame::new(
            self.dataframe.repartition_by_range(numPartitions, exprs),
        ))
    }

    /// Repartition into `numPartitions` using a column's value directly as the shuffle
    /// partition id. Mirrors `DataFrame.repartitionById(numPartitions, partitionIdCol)`.
    #[pyo3(signature = (numPartitions, partitionIdCol))]
    #[allow(non_snake_case)]
    fn repartitionById(
        &self,
        numPartitions: i32,
        partitionIdCol: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        if numPartitions <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "numPartitions must be positive, got {numPartitions}"
            )));
        }
        // A `ColumnOrName`, as in pyspark: a string is a column name, not a literal.
        let col = if let Ok(c) = partitionIdCol.extract::<Bound<'_, PyColumn>>() {
            c.borrow().column.clone()
        } else if let Ok(name) = partitionIdCol.extract::<String>() {
            spark_connect::functions::col(&name)
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "partitionIdCol should be a Column or a column name (str)",
            ));
        };
        Ok(PyDataFrame::new(
            self.dataframe.repartition_by_id(numPartitions, col),
        ))
    }

    /// Append a distributed monotonically-increasing index column. Mirrors
    /// `DataFrame.zipWithIndex(indexColName="index")`.
    #[pyo3(signature = (indexColName="index"))]
    #[allow(non_snake_case)]
    fn zipWithIndex(&self, indexColName: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.zip_with_index(indexColName))
    }

    /// Zip this DataFrame with another column-wise (side-by-side). Mirrors
    /// `DataFrame.zip(other)`.
    fn zip(&self, other: &PyDataFrame) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            self.dataframe.zip(&other.dataframe).to_pyerr()?,
        ))
    }

    /// Union with another DataFrame.
    fn union(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.union(&other.dataframe))
    }

    /// Union by name, optionally filling columns missing on one side with null.
    #[pyo3(signature = (other, allowMissingColumns=false))]
    fn unionByName(&self, other: &PyDataFrame, allowMissingColumns: bool) -> PyDataFrame {
        PyDataFrame::new(
            self.dataframe
                .union_by_name_opt(&other.dataframe, allowMissingColumns),
        )
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

    /// Repartition. Mirrors reference `repartition(numPartitions, *cols)` where the
    /// first argument may instead be a Column/str - `df.repartition("country")` /
    /// `df.repartition(col("country"))` repartition by that column at the default
    /// partition count (no explicit number).
    #[pyo3(signature = (numPartitions, *cols))]
    fn repartition(
        &self,
        py: Python<'_>,
        numPartitions: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // First arg is an int -> a partition count; otherwise it's a partition column
        // (str/Column) and the count defaults to the server's (passed as 0 = unset).
        if let Ok(n) = numPartitions.extract::<i32>() {
            if cols.is_empty() {
                return Ok(PyDataFrame::new(self.dataframe.repartition(n)));
            }
            let exprs: Vec<_> = to_column_list(cols)?
                .iter()
                .map(|c| c.expression().clone())
                .collect();
            Ok(PyDataFrame::new(
                self.dataframe.repartition_by_expressions(n, exprs),
            ))
        } else {
            // Column-first form: the first arg is itself a partition column.
            let mut all = vec![numPartitions];
            all.extend(cols);
            let _ = py;
            let exprs: Vec<_> = to_column_list(all)?
                .iter()
                .map(|c| c.expression().clone())
                .collect();
            Ok(PyDataFrame::new(
                self.dataframe.repartition_by_expressions(0, exprs),
            ))
        }
    }

    /// Coalesce.
    fn coalesce(&self, numPartitions: i32) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.coalesce(numPartitions))
    }

    /// Convert to DataFrame with new column names.
    #[pyo3(signature = (*cols))]
    fn toDF(&self, _py: Python<'_>, cols: Vec<String>) -> PyDataFrame {
        let name_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        PyDataFrame::new(self.dataframe.to_df(name_refs))
    }

    /// Alias this DataFrame.
    fn alias(&self, alias: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.alias(alias))
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

    /// Grouping sets aggregation: `groupingSets([[a, b], [a], []], "a", "b")`.
    ///
    /// `*cols` (the overall grouping columns) is accepted for signature parity with
    /// reference pyspark; the connect core derives the grouping from the sets, so the
    /// trailing columns are currently informational.
    #[pyo3(name = "groupingSets", signature = (groupingSets, *cols))]
    #[allow(unused_variables)]
    fn grouping_sets(
        &self,
        groupingSets: Vec<Vec<Bound<'_, PyAny>>>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyGroupedData> {
        let mut out: Vec<Vec<spark_connect::column::Column>> =
            Vec::with_capacity(groupingSets.len());
        for s in groupingSets {
            out.push(to_column_list(s)?);
        }
        Ok(PyGroupedData::new(self.dataframe.grouping_sets(out)))
    }

    /// Drop duplicates within the event-time watermark.
    #[pyo3(name = "dropDuplicatesWithinWatermark", signature = (subset=None))]
    fn drop_duplicates_within_watermark(&self, subset: Option<Vec<String>>) -> PyDataFrame {
        let refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|names| names.iter().map(|s| s.as_str()).collect());
        PyDataFrame::new(self.dataframe.drop_duplicates_within_watermark(refs))
    }

    /// `toJSON()` returns an RDD in classic PySpark; Spark Connect has no RDD, so
    /// the official Connect client raises NotImplementedError - we match that exactly.
    /// (Use `df.select(to_json(struct("*")))` for a JSON-string DataFrame.)
    #[pyo3(name = "toJSON")]
    fn to_json(&self) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "toJSON() is not supported for Spark Connect (it returns an RDD)",
        ))
    }

    /// Whether this DataFrame is semantically equal to `other`.
    #[pyo3(name = "sameSemantics")]
    fn same_semantics(&self, py: Python<'_>, other: &PyDataFrame) -> PyResult<bool> {
        py.detach(|| self.dataframe.same_semantics(&other.dataframe))
            .to_pyerr()
    }

    /// The server-side hash of this DataFrame's logical plan.
    #[pyo3(name = "semanticHash")]
    fn semantic_hash(&self, py: Python<'_>) -> PyResult<i32> {
        py.detach(|| self.dataframe.semantic_hash()).to_pyerr()
    }

    /// Sort within each partition.
    #[pyo3(name = "sortWithinPartitions", signature = (*cols, **kwargs))]
    fn sort_within_partitions(
        &self,
        _py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let exprs = sort_order_exprs(to_column_list(cols)?, kwargs.as_ref())?;
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
        colsMap: std::collections::HashMap<String, String>,
    ) -> PyDataFrame {
        PyDataFrame::new(
            self.dataframe
                .with_columns_renamed(colsMap.into_iter().collect()),
        )
    }

    /// The columns of this DataFrame whose names match a regex, as a Column. Mirrors
    /// `DataFrame.colRegex`.
    #[pyo3(name = "colRegex")]
    fn col_regex(&self, colName: &str) -> PyColumn {
        PyColumn::new(self.dataframe.col_regex(colName))
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

    /// Print the plan with Python's `print`, as PySpark's `DataFrame.explain`, which prints
    /// `_explain_string(extended, mode)`.
    #[pyo3(signature = (extended=None, mode=None))]
    fn explain(
        &self,
        py: Python<'_>,
        extended: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let plan = self.explain_string(py, extended, mode)?;
        py.import("builtins")?.call_method1("print", (plan,))?;
        Ok(())
    }

    /// The plan that `explain` prints, as PySpark's `DataFrame._explain_string`: `extended` is
    /// a bool, for the "extended" or the "simple" mode, or a mode, and `mode` a mode, but not
    /// both; the mode is "simple" without either.
    #[pyo3(name = "_explain_string", signature = (extended=None, mode=None))]
    fn explain_string(
        &self,
        py: Python<'_>,
        extended: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        use pyo3::types::{PyBool, PyString};
        let explain_mode: String = match (extended, mode) {
            (Some(_), Some(_)) => {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_SET_TOGETHER",
                    &[("arg_list", "extended and mode")],
                ))
            }
            (None, None) => "simple".to_string(),
            (Some(extended), None) if extended.is_instance_of::<PyBool>() => {
                if extended.extract::<bool>()? {
                    "extended"
                } else {
                    "simple"
                }
                .to_string()
            }
            (Some(arg), None) | (None, Some(arg)) if arg.is_instance_of::<PyString>() => {
                arg.extract()?
            }
            (Some(arg), None) | (None, Some(arg)) => {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "bool or str"),
                        ("arg_name", "extended (optional) and mode (optional)"),
                        ("arg_type", &arg.get_type().str()?.to_string()),
                    ],
                ))
            }
        };
        py.detach(|| self.dataframe.explain_string(&explain_mode))
            .to_pyerr()
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

    /// Fill null values. `value` may be a scalar (optionally over `subset`) or a
    /// {column: value} mapping, mirroring `pyspark.sql.DataFrame.fillna`.
    #[pyo3(signature = (value, subset=None))]
    fn fillna(
        &self,
        value: &Bound<'_, PyAny>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        if let Ok(d) = value.cast::<PyDict>() {
            let mut pairs = Vec::with_capacity(d.len());
            for (k, v) in d.iter() {
                pairs.push((k.extract::<String>()?, crate::session::py_to_value(&v)?));
            }
            return Ok(PyDataFrame::new(self.dataframe.fillna_map(pairs)));
        }
        let val = crate::session::py_to_value(value)?;
        let owned = to_subset(subset)?;
        let refs: Option<Vec<&str>> = owned
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        Ok(PyDataFrame::new(self.dataframe.fillna_value(val, refs)))
    }

    /// Statistical functions (`df.stat`).
    #[getter]
    #[pyo3(name = "stat")]
    fn stat(&self) -> crate::stat::PyStatFunctions {
        crate::stat::PyStatFunctions::new(self.dataframe.stat())
    }

    /// `df.crosstab(col1, col2)` - pyspark exposes it on DataFrame as well as
    /// `df.stat.crosstab`.
    fn crosstab(&self, col1: &str, col2: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.stat().crosstab(col1, col2))
    }

    /// `df.freqItems(cols, support=0.01)` (also on df.stat).
    #[pyo3(name = "freqItems", signature = (cols, support=0.01))]
    fn freq_items(&self, cols: Vec<String>, support: f64) -> PyDataFrame {
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        PyDataFrame::new(self.dataframe.stat().freq_items(refs, support))
    }

    /// `df.approxQuantile(col, probabilities, relativeError)` -> list[float]
    /// (also on df.stat). The server returns one row whose column is the
    /// array(-of-arrays) of quantiles; for a single column we return the inner list.
    #[pyo3(name = "approxQuantile")]
    fn approx_quantile(
        &self,
        py: Python<'_>,
        col: &str,
        probabilities: Vec<f64>,
        relativeError: f64,
    ) -> PyResult<Vec<f64>> {
        use spark_connect::row::Value;
        let df = self
            .dataframe
            .stat()
            .approx_quantile(vec![col], probabilities, relativeError);
        let rows = py.detach(|| df.collect()).to_pyerr()?;
        let quantiles = match rows.first().and_then(|r| r.get(0)) {
            Some(Value::List(outer)) => match outer.first() {
                Some(Value::List(inner)) => inner.iter().filter_map(|v| v.as_f64()).collect(),
                _ => outer.iter().filter_map(|v| v.as_f64()).collect(),
            },
            _ => Vec::new(),
        };
        Ok(quantiles)
    }

    /// Observe named metrics on the stream/df. `observation` is an Observation
    /// object or a name string; the remaining args are aggregate metric columns.
    /// Mirrors `DataFrame.observe(observation, *exprs)`.
    #[pyo3(signature = (observation, *exprs))]
    fn observe(
        &self,
        observation: &Bound<'_, PyAny>,
        exprs: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let name =
            if let Ok(obs) = observation.extract::<PyRef<crate::observation::PyObservation>>() {
                obs.inner.name().to_string()
            } else {
                observation.extract::<String>()?
            };
        let columns = to_column_list(exprs)?;
        let e: Vec<_> = columns.iter().map(|c| c.expression().clone()).collect();
        Ok(PyDataFrame::new(self.dataframe.observe(&name, e)))
    }

    /// Register as a temporary view (session-scoped).
    #[pyo3(name = "createTempView")]
    fn create_temp_view(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| self.dataframe.create_temp_view(name))
            .to_pyerr()
    }

    /// Register as a global temporary view.
    #[pyo3(name = "createGlobalTempView")]
    fn create_global_temp_view(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| self.dataframe.create_global_temp_view(name))
            .to_pyerr()
    }

    /// Register as (or replace) a global temporary view.
    #[pyo3(name = "createOrReplaceGlobalTempView")]
    fn create_or_replace_global_temp_view(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| self.dataframe.create_or_replace_global_temp_view(name))
            .to_pyerr()
    }

    /// Deprecated alias for createOrReplaceTempView.
    #[pyo3(name = "registerTempTable")]
    fn register_temp_table(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| self.dataframe.register_temp_table(name))
            .to_pyerr()
    }

    /// Reliable checkpoint (materialize + truncate lineage).
    #[pyo3(signature = (eager=true))]
    #[allow(unused_variables)]
    fn checkpoint(&self, py: Python<'_>, eager: bool) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            py.detach(|| self.dataframe.checkpoint()).to_pyerr()?,
        ))
    }

    /// Local checkpoint. `storageLevel` is accepted for signature parity with reference
    /// pyspark; the connect core's local checkpoint does not yet take a storage level,
    /// so a supplied value is currently a no-op.
    #[pyo3(name = "localCheckpoint", signature = (eager=true, storageLevel=None))]
    #[allow(unused_variables)]
    fn local_checkpoint(
        &self,
        py: Python<'_>,
        eager: bool,
        storageLevel: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            py.detach(|| self.dataframe.local_checkpoint()).to_pyerr()?,
        ))
    }

    /// Pearson correlation (or `method`) between two columns.
    #[pyo3(signature = (col1, col2, method=None))]
    #[allow(unused_variables)]
    fn corr(&self, py: Python<'_>, col1: &str, col2: &str, method: Option<&str>) -> PyResult<f64> {
        py.detach(|| self.dataframe.stat().corr(col1, col2))
            .to_pyerr()
    }

    /// Sample covariance between two columns.
    fn cov(&self, py: Python<'_>, col1: &str, col2: &str) -> PyResult<f64> {
        py.detach(|| self.dataframe.stat().cov(col1, col2))
            .to_pyerr()
    }

    /// Stratified sample without replacement per key.
    #[pyo3(name = "sampleBy", signature = (col, fractions, seed=None))]
    fn sample_by(
        &self,
        col: &str,
        fractions: &Bound<'_, PyDict>,
        seed: Option<i64>,
    ) -> PyResult<PyDataFrame> {
        let mut fr = Vec::with_capacity(fractions.len());
        for (k, v) in fractions.iter() {
            let key = to_column(&k)?.expression().clone();
            fr.push((key, v.extract::<f64>()?));
        }
        Ok(PyDataFrame::new(
            self.dataframe.stat().sample_by(col, fr, seed),
        ))
    }

    /// Replace values. `to_replace` may be a {old: new} dict, or a value/list with `value`.
    #[pyo3(signature = (to_replace, value=None, subset=None))]
    fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: Option<&Bound<'_, PyAny>>,
        subset: Option<Vec<String>>,
    ) -> PyResult<PyDataFrame> {
        let mut pairs: Vec<(String, String)> = Vec::new();
        if let Ok(d) = to_replace.cast::<PyDict>() {
            for (k, v) in d.iter() {
                pairs.push((k.str()?.to_string(), v.str()?.to_string()));
            }
        } else if let (Ok(olds), Some(val)) = (to_replace.extract::<Vec<Bound<'_, PyAny>>>(), value)
        {
            if let Ok(news) = val.extract::<Vec<Bound<'_, PyAny>>>() {
                for (o, n) in olds.iter().zip(news.iter()) {
                    pairs.push((o.str()?.to_string(), n.str()?.to_string()));
                }
            } else {
                for o in olds {
                    pairs.push((o.str()?.to_string(), val.str()?.to_string()));
                }
            }
        } else if let Some(val) = value {
            pairs.push((to_replace.str()?.to_string(), val.str()?.to_string()));
        }
        let subset_refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|s| s.iter().map(|x| x.as_str()).collect());
        Ok(PyDataFrame::new(self.dataframe.replace(pairs, subset_refs)))
    }

    /// Reconcile to a target schema (StructType or DDL string).
    fn to(&self, schema: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let dt = crate::types::py_to_data_type(schema)?;
        Ok(PyDataFrame::new(self.dataframe.to(dt)))
    }

    /// Set metadata (a dict) on a column.
    #[pyo3(name = "withMetadata")]
    fn with_metadata(
        &self,
        columnName: &str,
        metadata: std::collections::HashMap<String, String>,
    ) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.with_metadata(columnName, metadata))
    }

    /// Define an event-time watermark.
    #[pyo3(name = "withWatermark")]
    fn with_watermark(&self, eventTime: &str, delayThreshold: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.with_watermark(eventTime, delayThreshold))
    }

    /// Randomly split into multiple DataFrames by the given weights.
    #[pyo3(name = "randomSplit", signature = (weights, seed=None))]
    fn random_split(&self, weights: Vec<f64>, seed: Option<i64>) -> Vec<PyDataFrame> {
        self.dataframe
            .random_split(weights, seed)
            .into_iter()
            .map(PyDataFrame::new)
            .collect()
    }

    /// The list of files that make up this DataFrame.
    #[pyo3(name = "inputFiles")]
    fn input_files(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| self.dataframe.input_files()).to_pyerr()
    }

    /// Whether collect/take can run without a Spark job (a local relation).
    #[pyo3(name = "isLocal")]
    fn is_local(&self) -> bool {
        self.dataframe.is_local()
    }

    /// Whether this (cached) DataFrame is cached.
    #[pyo3(name = "is_cached")]
    #[getter]
    fn is_cached(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.dataframe.is_cached()).to_pyerr()
    }

    /// Whether this DataFrame contains one or more sources that continuously return data.
    #[pyo3(name = "isStreaming")]
    #[getter]
    fn is_streaming(&self) -> bool {
        self.dataframe.is_streaming()
    }

    /// The storage level currently in use for this DataFrame.
    #[pyo3(name = "storageLevel")]
    #[getter]
    fn storage_level(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let sl = py.detach(|| self.dataframe.storage_level()).to_pyerr()?;
        let cls = py.import("pyspark.storagelevel")?.getattr("StorageLevel")?;
        Ok(cls
            .call1((
                sl.use_disk,
                sl.use_memory,
                sl.use_off_heap,
                sl.deserialized,
                sl.replication,
            ))?
            .unbind())
    }

    /// The SparkSession that created this DataFrame.
    #[pyo3(name = "sparkSession")]
    #[getter]
    fn spark_session(&self) -> crate::session::PySparkSession {
        crate::session::PySparkSession::new(self.dataframe.spark_session())
    }

    /// Not supported on Spark Connect — `DataFrame.rdd` requires the legacy RDD API.
    #[getter]
    fn rdd(&self) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "rdd() is not supported for Spark Connect",
        ))
    }

    /// The EXISTS subquery of this DataFrame, as a Column. Mirrors `DataFrame.exists()`.
    fn exists(&self) -> PyColumn {
        PyColumn::new(self.dataframe.exists())
    }

    /// The scalar subquery of this DataFrame, as a Column. Mirrors `DataFrame.scalar()`.
    fn scalar(&self) -> PyColumn {
        PyColumn::new(self.dataframe.scalar())
    }

    /// Return a `TableArg` wrapping this DataFrame, for use as a table-valued function
    /// argument. Mirrors `DataFrame.asTable()`.
    #[pyo3(name = "asTable")]
    fn as_table(&self) -> crate::tablearg::PyTableArg {
        crate::tablearg::PyTableArg::new(spark_connect::table_arg::TableArg::new(
            self.dataframe.clone(),
        ))
    }

    /// Deprecated alias for dropDuplicates.
    #[pyo3(name = "drop_duplicates", signature = (subset=None))]
    fn drop_duplicates(&self, subset: Option<Vec<String>>) -> PyDataFrame {
        let refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|s| s.iter().map(|x| x.as_str()).collect());
        PyDataFrame::new(self.dataframe.drop_duplicates(refs))
    }

    /// Alias for groupBy.
    #[pyo3(signature = (*cols))]
    fn groupby(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyGroupedData> {
        self.groupBy(py, cols)
    }

    /// Apply a function to this DataFrame and return its result:
    /// `df.transform(func, *args, **kwargs)` == `func(df, *args, **kwargs)`.
    #[pyo3(signature = (func, *args, **kwargs))]
    fn transform(
        &self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let this = Py::new(py, PyDataFrame::new(self.dataframe.clone()))?;
        let mut call_args: Vec<Py<PyAny>> = vec![this.into_any()];
        for a in args.iter() {
            call_args.push(a.unbind());
        }
        let tuple = PyTuple::new(py, call_args)?;
        Ok(func.call(tuple, kwargs)?.unbind())
    }

    /// Unpivot (wide-to-long). `values=None` unpivots all non-id columns.
    #[pyo3(signature = (ids, values=None, variableColumnName="variable", valueColumnName="value"))]
    fn unpivot(
        &self,
        ids: Vec<String>,
        values: Option<Vec<String>>,
        variableColumnName: &str,
        valueColumnName: &str,
    ) -> PyDataFrame {
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let owned = values;
        let val_refs = owned
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        PyDataFrame::new(self.dataframe.melt(
            id_refs,
            val_refs,
            variableColumnName,
            valueColumnName,
        ))
    }

    /// Alias for unpivot.
    #[pyo3(signature = (ids, values=None, variableColumnName="variable", valueColumnName="value"))]
    fn melt(
        &self,
        ids: Vec<String>,
        values: Option<Vec<String>>,
        variableColumnName: &str,
        valueColumnName: &str,
    ) -> PyDataFrame {
        let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();
        let owned = values;
        let val_refs = owned
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        PyDataFrame::new(self.dataframe.melt(
            id_refs,
            val_refs,
            variableColumnName,
            valueColumnName,
        ))
    }

    /// Persist the DataFrame. A storage-level argument is accepted for API parity;
    /// the default (MEMORY_AND_DISK) is used.
    #[pyo3(signature = (storageLevel=None))]
    fn persist(
        &self,
        py: Python<'_>,
        storageLevel: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        use spark_connect::storage::StorageLevelExt;
        // Honor the requested StorageLevel instead of always using MEMORY_AND_DISK.
        // A pyspark StorageLevel exposes useDisk/useMemory/useOffHeap/deserialized/
        // replication; map them onto the proto. Default (None) is MEMORY_AND_DISK_DESER,
        // matching reference DataFrame.persist().
        let level = match storageLevel {
            Some(obj) => spark_connect_proto::StorageLevel {
                use_disk: obj.getattr("useDisk")?.extract()?,
                use_memory: obj.getattr("useMemory")?.extract()?,
                use_off_heap: obj.getattr("useOffHeap")?.extract()?,
                deserialized: obj.getattr("deserialized")?.extract()?,
                replication: obj.getattr("replication")?.extract()?,
            },
            None => spark_connect_proto::StorageLevel::memory_and_disk_deser(),
        };
        let df = py.detach(|| self.dataframe.persist(level)).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Column access by attribute (`df.colname` -> Column), mirroring reference pyspark: the
    /// column of this DataFrame, by its plan ID. Names beginning with `_` raise
    /// AttributeError so normal attribute lookup works.
    fn __getattr__(&self, name: &str) -> PyResult<PyColumn> {
        if name.starts_with('_') {
            return Err(PyErr::new::<pyo3::exceptions::PyAttributeError, _>(
                name.to_string(),
            ));
        }
        Ok(PyColumn::new(self.dataframe.col(name)))
    }

    /// Class attributes and all column names, matching Python Connect.
    fn __dir__(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        // Inspect the class rather than the instance to avoid calling __dir__ again.
        let mut attrs = py.get_type::<Self>().dir()?.extract::<Vec<String>>()?;
        attrs.extend(self.columns(py)?);
        attrs.sort();
        attrs.dedup();
        Ok(attrs)
    }

    /// Bracket completion keeps column order, duplicates and non-identifiers.
    fn _ipython_key_completions_(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.columns(py)
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
        let (rows, schema) = py
            .detach(|| self.dataframe.collect_with_schema())
            .to_pyerr()?;
        rows_to_py(py, rows, schema.as_ref())
    }

    /// Return an iterator over rows without materializing the entire result in memory.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toLocalIterator(prefetchPartitions=False)`, a generator:
    /// the execution starts with the first row, and its errors are raised by the iteration.
    /// Yields Row objects lazily as batches are fetched from the server.
    #[pyo3(name = "toLocalIterator")]
    #[pyo3(signature = (prefetchPartitions=false))]
    fn to_local_iterator(&self, prefetchPartitions: bool) -> PyLocalRowIterator {
        PyLocalRowIterator::new(self.dataframe.clone(), prefetchPartitions)
    }

    /// Collect the DataFrame into a `pyarrow.Table`.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toArrow()`: the result is decoded from the
    /// server's Arrow IPC stream and handed to `pyarrow` through arrow-rs's
    /// reader conversion, sharing compatible decoded Arrow value buffers. Bitmap alignment
    /// and schemas requiring the IPC compatibility fallback can still copy.
    #[pyo3(name = "toArrow")]
    fn to_arrow(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let batches = py
            .detach(|| self.dataframe.collect_record_batches())
            .to_pyerr()?;
        crate::pyarrow_export::record_batches_to_pyarrow(py, batches)
    }

    /// Collect the DataFrame into a pandas DataFrame.
    ///
    /// Mirrors `pyspark.sql.DataFrame.toPandas()`, as `SparkConnectClient.to_pandas` converts
    /// the Arrow result: each column by its Spark type, in the session time zone.
    fn toPandas(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.to_pandas(py, None)
    }

    /// Internal Arrow-based pandas conversion used by pandas-on-Spark
    /// (`pyspark.sql.connect.dataframe.DataFrame._to_pandas`): `toPandas`, with
    /// `pandasStructHandlingMode` overriding `spark.sql.execution.pandas.structHandlingMode`,
    /// as in `SparkConnectClient.to_pandas`.
    #[pyo3(signature = (**kwargs))]
    fn _to_pandas(
        &self,
        py: Python<'_>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let mode = match kwargs {
            Some(kwargs) => kwargs.get_item("pandasStructHandlingMode")?,
            None => None,
        };
        self.to_pandas(py, mode)
    }

    /// Get the count of rows.
    fn count(&self, py: Python<'_>) -> PyResult<i64> {
        py.detach(|| self.dataframe.count()).to_pyerr()
    }

    /// Print the first `n` rows as the server formats them. Mirrors `DataFrame.show`, which
    /// prints the table of `_show_string` with Python's `print`.
    #[pyo3(signature = (*args, **kwargs), text_signature = "($self, n=20, truncate=True, vertical=False)")]
    fn show(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let table = self.show_table(py, "DataFrame.show", args, kwargs)?;
        py.import("builtins")?.call_method1("print", (table,))?;
        Ok(())
    }

    /// The table that `show` prints: the first `n` rows (20 by default), values truncated
    /// to 20 characters for `truncate=True`, to `int(truncate)` otherwise (0 for none), and
    /// vertical with a line per value. Mirrors `DataFrame._show_string`, with its errors,
    /// for any value given, None included.
    #[pyo3(name = "_show_string", signature = (*args, **kwargs), text_signature = "($self, n=20, truncate=True, vertical=False)")]
    fn show_string(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<String> {
        self.show_table(py, "DataFrame._show_string", args, kwargs)
    }

    /// Execution info from the most recent action, as a `pyspark.sql.metrics.ExecutionInfo`
    /// (plan metrics, observed-metric flows, and the ExecutePlan `operation_id`). Mirrors
    /// `DataFrame.executionInfo`; `None` before any action ran.
    #[getter]
    #[pyo3(name = "executionInfo")]
    fn execution_info(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Ok(info) = self.dataframe.execution_info() else {
            return Ok(py.None());
        };
        let metrics_mod = py.import("pyspark.sql.metrics")?;
        let (plan_metrics, observed) = execution_metrics(py, &info)?;
        let ei = metrics_mod.getattr("ExecutionInfo")?.call1((
            plan_metrics,
            observed,
            info.operation_id.clone(),
        ))?;
        Ok(ei.unbind())
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

    /// The schema of this DataFrame as a `StructType` (a property, mirroring
    /// `DataFrame.schema`) — materialized into the proper Python type classes so
    /// `df.schema.fields`, `df.schema["c"]`, `df.schema.fieldNames()` all work.
    #[getter]
    fn schema(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let schema = py.detach(|| self.dataframe.schema()).to_pyerr()?;
        crate::types::data_type_to_py(py, &schema)
    }

    /// Get the first row.
    fn first(&self, py: Python<'_>) -> PyResult<Option<PyRow>> {
        let (row, schema) = py
            .detach(|| self.dataframe.first_with_schema())
            .to_pyerr()?;
        Ok(rows_to_py(py, row.into_iter().collect(), schema.as_ref())?.pop())
    }

    /// Alias for first.
    #[pyo3(signature = (n=None))]
    fn head(&self, py: Python<'_>, n: Option<i32>) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;
        match n {
            // head() -> the first Row (or None), matching reference pyspark.
            None => self.first(py)?.into_py_any(py),
            // head(n) -> a list of the first n Rows.
            Some(k) => {
                let (rows, schema) = py
                    .detach(|| self.dataframe.limit(k).collect_with_schema())
                    .to_pyerr()?;
                rows_to_py(py, rows, schema.as_ref())?.into_py_any(py)
            }
        }
    }

    /// Get the first n rows.
    fn take(&self, py: Python<'_>, num: usize) -> PyResult<Vec<PyRow>> {
        let (rows, schema) = py
            .detach(|| self.dataframe.take_with_schema(num))
            .to_pyerr()?;
        rows_to_py(py, rows, schema.as_ref())
    }

    /// Column names (a property, mirroring `DataFrame.columns`).
    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| self.dataframe.columns().to_pyerr())
    }

    /// Get the last `num` rows as a list of Rows (eager, like pyspark `tail`).
    fn tail(&self, py: Python<'_>, num: i32) -> PyResult<Vec<PyRow>> {
        let (rows, schema) = py
            .detach(|| self.dataframe.tail(num).collect_with_schema())
            .to_pyerr()?;
        rows_to_py(py, rows, schema.as_ref())
    }

    /// Sample a fraction of rows. Mirrors `DataFrame.sample(withReplacement=None,
    /// fraction=None, seed=None)` including the legacy positional form
    /// `sample(fraction)` (where the first positional arg is the fraction).
    #[pyo3(signature = (withReplacement=None, fraction=None, seed=None))]
    fn sample(
        &self,
        withReplacement: Option<&Bound<'_, PyAny>>,
        fraction: Option<f64>,
        seed: Option<i64>,
    ) -> PyResult<PyDataFrame> {
        // Resolve the (withReplacement, fraction) overloads exactly like pyspark:
        // if the first positional is a float and no fraction was given, it IS the
        // fraction (legacy `sample(0.5)`); otherwise it's the bool withReplacement.
        let (replace, frac) = match (withReplacement, fraction) {
            (Some(w), None) => {
                if let Ok(f) = w.extract::<f64>() {
                    (false, f) // legacy sample(fraction)
                } else {
                    let b = w.extract::<bool>().map_err(|_| {
                        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "withReplacement must be a bool, or pass fraction as a float",
                        )
                    })?;
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "sample requires a fraction (got withReplacement={b} with no fraction)",
                    )));
                }
            }
            (Some(w), Some(f)) => {
                let b = w.extract::<bool>().unwrap_or(false);
                (b, f)
            }
            (None, Some(f)) => (false, f),
            (None, None) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "sample requires a fraction",
                ))
            }
        };
        Ok(PyDataFrame::new(
            self.dataframe.sample_opt(frac, replace, seed),
        ))
    }

    /// Check if the DataFrame is empty.
    fn isEmpty(&self) -> PyResult<bool> {
        self.dataframe.is_empty().to_pyerr()
    }

    /// Print the schema through Python stdout, honoring the maximum nesting depth.
    #[pyo3(signature = (level=None))]
    fn printSchema(&self, py: Python<'_>, level: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        // Reference's `if level:` selects the default for false values before
        // resolving the schema; positive fractional depths behave as ceil(level).
        let level = match level {
            Some(level) if level.is_truthy()? => Some(level),
            _ => None,
        };
        let schema = py.detach(|| self.dataframe.schema().to_pyerr())?;
        let depth = level
            .map(crate::types::normalize_tree_depth)
            .transpose()?
            .unwrap_or(i32::MAX);
        let tree = schema.tree_string_with_depth(depth).to_pyerr()?;
        py.import("builtins")?.getattr("print")?.call1((tree,))?;
        Ok(())
    }

    /// Get the schema as a list of (name, type) tuples.
    #[getter]
    fn dtypes(&self, py: Python<'_>) -> PyResult<Vec<(String, String)>> {
        let spark_connect::types::DataType::Struct { fields } =
            py.detach(|| self.dataframe.schema().to_pyerr())?
        else {
            unreachable!("schema() validates struct responses")
        };
        // As PySpark, a user-defined type is named by its simpleString(), such as `vector`,
        // which only its Python class knows.
        fields
            .into_iter()
            .map(|field| {
                let name = crate::types::simple_string_with_udts(py, &field.data_type)?;
                Ok((field.name, name))
            })
            .collect()
    }

    /// Additional set operation: unionAll (alias for union).
    fn unionAll(&self, other: &PyDataFrame) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.union(&other.dataframe))
    }

    /// The V1 writer interface (`df.write`).
    #[getter]
    fn write(&self) -> PyDataFrameWriter {
        PyDataFrameWriter {
            inner: self.dataframe.write(),
        }
    }

    /// The V2 writer interface (`df.writeTo("table")`).
    #[pyo3(name = "writeTo")]
    fn write_to(&self, table: &str) -> PyDataFrameWriterV2 {
        PyDataFrameWriterV2 {
            inner: self.dataframe.write_to(table),
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
            df: self.dataframe.clone(),
        }
    }

    /// Merge-into (`df.mergeInto("table", cond)`), returning a V2 merge builder.
    #[pyo3(name = "mergeInto")]
    fn merge_into(&self, table: &str, condition: &PyColumn) -> PyMergeIntoWriter {
        PyMergeIntoWriter {
            inner: self.dataframe.merge_into(table, condition.column.clone()),
        }
    }

    /// As PySpark's `DataFrame.__getitem__`: a column by name (`df["col"]`), of this DataFrame
    /// by its plan ID, as its star of `df["*"]`, or by position (`df[0]`), the rows that match
    /// a Column (`df[df.a > 1]`, as `filter`), or the projection of a list or tuple of columns
    /// (`df[["a", "b"]]`, as `select`).
    fn __getitem__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;

        let name = if let Ok(name) = item.cast::<pyo3::types::PyString>() {
            let name = name.to_cow()?;
            let column = if name == "*" {
                self.dataframe.star()
            } else {
                self.dataframe.col(&name)
            };
            return PyColumn::new(column).into_py_any(py);
        } else if item.is_instance_of::<PyColumn>() {
            return self.filter(item)?.into_py_any(py);
        } else if item.is_instance_of::<PyList>() || item.is_instance_of::<PyTuple>() {
            let cols = item.try_iter()?.collect::<PyResult<Vec<_>>>()?;
            return self.select(py, cols)?.into_py_any(py);
        } else if item.is_instance_of::<PyInt>() {
            let columns = self.columns(py)?;
            // Use Python list indexing for negative indices, bools (an int
            // subclass), and IndexError even for integers larger than isize.
            PyList::new(py, columns)?
                .as_any()
                .get_item(item)?
                .extract::<String>()?
        } else {
            let arg_type = item.get_type().name()?.to_string();
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("expected_type", "Column, int, list, str or tuple"),
                    ("arg_name", "item"),
                    ("arg_type", &arg_type),
                ],
            ));
        };
        PyColumn::new(spark_connect::functions::col(&name)).into_py_any(py)
    }

    /// Private column accessor used by pandas-on-Spark (`sdf._col(name)`); mirrors the
    /// connect DataFrame's `_col`, the column of this DataFrame by its plan ID. Backtick-quoted
    /// names arrive from `scol_for`.
    fn _col(&self, item: &str) -> PyColumn {
        PyColumn::new(self.dataframe.col(item))
    }

    /// As PySpark's `DataFrame.__repr__`: the table of `_show_string` when the session's
    /// `spark.sql.repl.eagerEval.enabled` is `true`, with its `maxNumRows` rows and its
    /// `truncate`, and `DataFrame[name: type, ...]` otherwise. Like PySpark, it reads the
    /// three settings, in one request, on every call.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        const EAGER_EVAL: [&str; 3] = [
            "spark.sql.repl.eagerEval.enabled",
            "spark.sql.repl.eagerEval.maxNumRows",
            "spark.sql.repl.eagerEval.truncate",
        ];
        let conf = self.dataframe.spark_session().conf();
        let values = py.detach(|| conf.get_many(&EAGER_EVAL)).to_pyerr()?;
        if values[0].as_deref() == Some("true") {
            // As PySpark's `int(...)` of each setting.
            let int = py.import("builtins")?.getattr("int")?;
            let n: i32 = int.call1((values[1].as_deref(),))?.extract()?;
            let truncate: i32 = int.call1((values[2].as_deref(),))?.extract()?;
            return py
                .detach(|| self.dataframe.show_string(n, truncate, false))
                .to_pyerr();
        }
        let fields = self
            .dtypes(py)?
            .into_iter()
            .map(|(name, dtype)| format!("{name}: {dtype}"))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!("DataFrame[{fields}]"))
    }

    /// `DataFrame.mapInPandas` — map an iterator of pandas DataFrames.
    ///
    /// `profile` (a ResourceProfile) is accepted for signature parity with reference
    /// pyspark; the connect core does not yet attach a resource profile id to the
    /// map-partitions plan, so a supplied profile is currently a no-op.
    #[pyo3(name = "mapInPandas", signature = (func, schema, barrier = false, profile = None))]
    #[allow(unused_variables)]
    fn map_in_pandas(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
        barrier: bool,
        profile: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        validate_map_function(py, &func, eval_type::SQL_MAP_PANDAS_ITER_UDF)?;
        let cols = py.detach(|| self.dataframe.columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "mapInPandas",
            &func,
            &schema,
            eval_type::SQL_MAP_PANDAS_ITER_UDF,
            &cols,
        )?;
        Ok(PyDataFrame::new(self.dataframe.map_in_pandas(udf, barrier)))
    }

    /// `DataFrame.mapInArrow` — map an iterator of Arrow batches. `profile` is accepted
    /// for signature parity (see `mapInPandas`); it is currently a no-op.
    #[pyo3(name = "mapInArrow", signature = (func, schema, barrier = false, profile = None))]
    #[allow(unused_variables)]
    fn map_in_arrow(
        &self,
        py: Python<'_>,
        func: Bound<'_, PyAny>,
        schema: Bound<'_, PyAny>,
        barrier: bool,
        profile: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        validate_map_function(py, &func, eval_type::SQL_MAP_ARROW_ITER_UDF)?;
        let cols = py.detach(|| self.dataframe.columns()).to_pyerr()?;
        let udf = build_map_udf(
            py,
            "mapInArrow",
            &func,
            &schema,
            eval_type::SQL_MAP_ARROW_ITER_UDF,
            &cols,
        )?;
        Ok(PyDataFrame::new(self.dataframe.map_in_arrow(udf, barrier)))
    }

    /// `DataFrame.foreach` — run a function per row for side effects.
    #[pyo3(name = "foreach")]
    fn foreach(&self, py: Python<'_>, f: Bound<'_, PyAny>) -> PyResult<()> {
        let udf = build_side_effect_udf(py, "foreach", &f)?;
        self.dataframe.foreach(udf).to_pyerr()
    }

    /// `DataFrame.foreachPartition` — run a function per partition for side effects.
    #[pyo3(name = "foreachPartition")]
    fn foreach_partition(&self, py: Python<'_>, f: Bound<'_, PyAny>) -> PyResult<()> {
        let udf = build_side_effect_udf(py, "foreachPartition", &f)?;
        self.dataframe.foreach_partition(udf).to_pyerr()
    }

    /// `DataFrame.nearestByJoin`.
    #[pyo3(name = "nearestByJoin", signature = (other, rankingExpression, numResults = 1, mode = "brute", direction = "asc", joinType = "inner"))]
    fn nearest_by_join(
        &self,
        other: &PyDataFrame,
        rankingExpression: &PyColumn,
        numResults: i32,
        mode: &str,
        direction: &str,
        joinType: &str,
    ) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.nearest_by_join(
            &other.dataframe,
            rankingExpression.column.clone(),
            numResults,
            mode,
            direction,
            joinType,
        ))
    }

    /// `DataFrame.metadataColumn`.
    #[pyo3(name = "metadataColumn")]
    fn metadata_column(&self, colName: &str) -> PyColumn {
        PyColumn::new(self.dataframe.metadata_column(colName))
    }
}

/// Python wrapper for `DataFrameNaFunctions` (`df.na`).
#[pyclass(name = "DataFrameNaFunctions", module = "pyspark.sql.dataframe")]
pub struct PyDataFrameNaFunctions {
    inner: spark_connect::group::NaFunctions,
    df: spark_connect::dataframe::DataFrame,
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

    /// Fill nulls; `value` may be a scalar (float / int / str / bool, optionally over
    /// `subset`) or a {column: value} mapping. Alias of `DataFrame.fillna`. PySpark fills
    /// only columns whose type matches the value's type (server-side).
    #[pyo3(signature = (value, subset=None))]
    fn fill(
        &self,
        value: &Bound<'_, PyAny>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        if let Ok(d) = value.cast::<PyDict>() {
            let mut pairs = Vec::with_capacity(d.len());
            for (k, v) in d.iter() {
                pairs.push((k.extract::<String>()?, crate::session::py_to_value(&v)?));
            }
            return Ok(PyDataFrame::new(self.df.fillna_map(pairs)));
        }
        let val = crate::session::py_to_value(value)?;
        let owned = to_subset(subset)?;
        let refs: Option<Vec<&str>> = owned
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        Ok(PyDataFrame::new(self.df.fillna_value(val, refs)))
    }

    /// Replace values, mirroring `DataFrameNaFunctions.replace`:
    /// `replace([old...], [new...], subset)`, `replace({old: new}, subset)`, or
    /// `replace(old, new, subset)`.
    #[pyo3(signature = (to_replace, value=None, subset=None))]
    fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: Option<&Bound<'_, PyAny>>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let pairs = build_replacement_pairs(to_replace, value)?;
        let owned = to_subset(subset)?;
        let subset_refs: Option<Vec<&str>> = owned
            .as_ref()
            .map(|v| v.iter().map(String::as_str).collect());
        Ok(PyDataFrame::new(self.inner.replace(pairs, subset_refs)))
    }
}

/// Coerce a `subset` argument (a str column name, a list of names, or None) to
/// an owned Vec<String>, mirroring reference pyspark which accepts either.
fn to_subset(subset: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    match subset {
        None => Ok(None),
        Some(s) => {
            if let Ok(name) = s.extract::<String>() {
                Ok(Some(vec![name]))
            } else {
                Ok(Some(s.extract::<Vec<String>>()?))
            }
        }
    }
}

/// Build (old, new) string replacement pairs from the reference `replace` forms.
fn build_replacement_pairs(
    to_replace: &Bound<'_, PyAny>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<(String, String)>> {
    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyDict, PyList};
    if let Ok(d) = to_replace.cast::<PyDict>() {
        let mut pairs = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            pairs.push((k.str()?.to_string(), v.str()?.to_string()));
        }
        return Ok(pairs);
    }
    if let Ok(olds) = to_replace.cast::<PyList>() {
        let val = value.ok_or_else(|| {
            PyValueError::new_err("replace with a list to_replace requires a value list")
        })?;
        let news = val
            .cast::<PyList>()
            .map_err(|_| PyValueError::new_err("value must be a list when to_replace is a list"))?;
        if olds.len() != news.len() {
            return Err(PyValueError::new_err(
                "to_replace and value lists must be the same length",
            ));
        }
        let mut pairs = Vec::with_capacity(olds.len());
        for (a, b) in olds.iter().zip(news.iter()) {
            pairs.push((a.str()?.to_string(), b.str()?.to_string()));
        }
        return Ok(pairs);
    }
    let val =
        value.ok_or_else(|| PyValueError::new_err("replace with a scalar requires a value"))?;
    Ok(vec![(
        to_replace.str()?.to_string(),
        val.str()?.to_string(),
    )])
}

/// Python wrapper for `MergeIntoWriter` (`df.mergeInto(...)`).
///
/// Mirrors PySpark's mutable writer: `withSchemaEvolution` and every clause action
/// (`whenMatched().updateAll()`, ...) change this object and return it, and `merge`
/// leaves it usable. The core writer is a consuming builder, so the wrapped value is
/// replaced in place and `merge` runs on a clone.
#[pyclass(name = "MergeIntoWriter", module = "pyspark.sql.merge")]
pub struct PyMergeIntoWriter {
    inner: spark_connect::MergeIntoWriter,
}

/// Apply a clause action to the parent writer in place and return that same writer
/// (PySpark's clause methods append to `self.writer` and return it).
fn merge_clause_action(
    py: Python<'_>,
    writer: &Py<PyMergeIntoWriter>,
    action: impl FnOnce(spark_connect::MergeIntoWriter) -> spark_connect::MergeIntoWriter,
) -> PyResult<Py<PyMergeIntoWriter>> {
    {
        let mut w = writer.bind(py).try_borrow_mut()?;
        w.inner = action(w.inner.clone());
    }
    Ok(writer.clone_ref(py))
}

#[pymethods]
impl PyMergeIntoWriter {
    /// Nested builder-class references, mirroring pyspark's `MergeIntoWriter.WhenMatched`
    /// etc. (the objects returned by `whenMatched()` / `whenNotMatched()` /
    /// `whenNotMatchedBySource()`).
    #[classattr]
    #[allow(non_snake_case)]
    fn WhenMatched(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PyWhenMatched>().unbind()
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn WhenNotMatched(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PyWhenNotMatched>().unbind()
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn WhenNotMatchedBySource(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PyWhenNotMatchedBySource>().unbind()
    }

    #[pyo3(name = "whenMatched", signature = (condition=None))]
    fn when_matched(slf: &Bound<'_, Self>, condition: Option<&PyColumn>) -> PyWhenMatched {
        PyWhenMatched {
            writer: slf.clone().unbind(),
            condition: condition.map(|c| c.column.clone()),
        }
    }

    #[pyo3(name = "whenNotMatched", signature = (condition=None))]
    fn when_not_matched(slf: &Bound<'_, Self>, condition: Option<&PyColumn>) -> PyWhenNotMatched {
        PyWhenNotMatched {
            writer: slf.clone().unbind(),
            condition: condition.map(|c| c.column.clone()),
        }
    }

    #[pyo3(name = "whenNotMatchedBySource", signature = (condition=None))]
    fn when_not_matched_by_source(
        slf: &Bound<'_, Self>,
        condition: Option<&PyColumn>,
    ) -> PyWhenNotMatchedBySource {
        PyWhenNotMatchedBySource {
            writer: slf.clone().unbind(),
            condition: condition.map(|c| c.column.clone()),
        }
    }

    #[pyo3(name = "withSchemaEvolution")]
    fn with_schema_evolution(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = slf.inner.clone().with_schema_evolution();
        slf
    }

    fn merge(&self) -> PyResult<()> {
        self.inner.clone().merge().to_pyerr()
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

/// `whenMatched(...)` clause builder. Holds its parent writer, like PySpark's
/// `MergeIntoWriter.WhenMatched`; each action may be called more than once.
#[pyclass(name = "WhenMatched", module = "pyspark.sql.merge")]
pub struct PyWhenMatched {
    writer: Py<PyMergeIntoWriter>,
    condition: Option<spark_connect::column::Column>,
}

#[pymethods]
impl PyWhenMatched {
    #[pyo3(name = "updateAll")]
    fn update_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let cond = self.condition.clone();
        merge_clause_action(py, &self.writer, |w| w.when_matched(cond).update_all())
    }

    fn update(
        &self,
        py: Python<'_>,
        assignments: std::collections::HashMap<String, Py<PyColumn>>,
    ) -> PyResult<Py<PyMergeIntoWriter>> {
        let cond = self.condition.clone();
        let map = assignments_to_map(py, assignments);
        merge_clause_action(py, &self.writer, |w| w.when_matched(cond).update(map))
    }

    fn delete(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let cond = self.condition.clone();
        merge_clause_action(py, &self.writer, |w| w.when_matched(cond).delete())
    }
}

/// `whenNotMatched(...)` clause builder. Holds its parent writer, like PySpark's
/// `MergeIntoWriter.WhenNotMatched`.
#[pyclass(name = "WhenNotMatched", module = "pyspark.sql.merge")]
pub struct PyWhenNotMatched {
    writer: Py<PyMergeIntoWriter>,
    condition: Option<spark_connect::column::Column>,
}

#[pymethods]
impl PyWhenNotMatched {
    #[pyo3(name = "insertAll")]
    fn insert_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let cond = self.condition.clone();
        merge_clause_action(py, &self.writer, |w| w.when_not_matched(cond).insert_all())
    }

    fn insert(
        &self,
        py: Python<'_>,
        assignments: std::collections::HashMap<String, Py<PyColumn>>,
    ) -> PyResult<Py<PyMergeIntoWriter>> {
        let cond = self.condition.clone();
        let map = assignments_to_map(py, assignments);
        merge_clause_action(py, &self.writer, |w| w.when_not_matched(cond).insert(map))
    }
}

/// `whenNotMatchedBySource(...)` clause builder. Holds its parent writer, like
/// PySpark's `MergeIntoWriter.WhenNotMatchedBySource`.
#[pyclass(name = "WhenNotMatchedBySource", module = "pyspark.sql.merge")]
pub struct PyWhenNotMatchedBySource {
    writer: Py<PyMergeIntoWriter>,
    condition: Option<spark_connect::column::Column>,
}

#[pymethods]
impl PyWhenNotMatchedBySource {
    #[pyo3(name = "updateAll")]
    fn update_all(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let cond = self.condition.clone();
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched_by_source(cond).update_all()
        })
    }

    fn update(
        &self,
        py: Python<'_>,
        assignments: std::collections::HashMap<String, Py<PyColumn>>,
    ) -> PyResult<Py<PyMergeIntoWriter>> {
        let cond = self.condition.clone();
        let map = assignments_to_map(py, assignments);
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched_by_source(cond).update(map)
        })
    }

    fn delete(&self, py: Python<'_>) -> PyResult<Py<PyMergeIntoWriter>> {
        let cond = self.condition.clone();
        merge_clause_action(py, &self.writer, |w| {
            w.when_not_matched_by_source(cond).delete()
        })
    }
}

/// Python wrapper for the V1 `DataFrameWriter` (`df.write`).
#[pyclass(name = "DataFrameWriter", module = "pyspark.sql.readwriter")]
pub struct PyDataFrameWriter {
    inner: spark_connect::readwriter::DataFrameWriter,
}

impl PyDataFrameWriter {}

/// Apply the shared `format` / `mode` / `partitionBy` / `**options` arguments that
/// reference `DataFrameWriter.save` and `.saveAsTable` accept inline, mirroring
/// `.format().mode().partitionBy().options()` chaining.
fn apply_write_common(
    mut w: spark_connect::readwriter::DataFrameWriter,
    format: Option<&str>,
    mode: Option<&str>,
    partitionBy: Option<&Bound<'_, PyAny>>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<spark_connect::readwriter::DataFrameWriter> {
    if let Some(f) = format {
        w = w.format(f);
    }
    if let Some(m) = mode {
        w = w.mode(m);
    }
    if let Some(pb) = partitionBy {
        let cols: Vec<String> = if let Ok(one) = pb.extract::<String>() {
            vec![one]
        } else {
            pb.extract::<Vec<String>>()?
        };
        w = w.partition_by(cols);
    }
    if let Some(opts) = options {
        let mut map = std::collections::HashMap::new();
        for (k, v) in opts.iter() {
            if let Some(val) = crate::coerce_option_value(&v)? {
                map.insert(k.str()?.to_string(), val);
            }
        }
        w = w.options(map);
    }
    Ok(w)
}

#[pymethods]
impl PyDataFrameWriter {
    fn mode<'py>(mut slf: PyRefMut<'py, Self>, saveMode: &str) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().mode(saveMode);
        Ok(slf)
    }

    fn format<'py>(mut slf: PyRefMut<'py, Self>, source: &str) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().format(source);
        Ok(slf)
    }

    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => {
                slf.inner = slf.inner.clone().option(key, &v);
                Ok(slf)
            }
            None => Ok(slf),
        }
    }

    // Mirrors reference `DataFrameWriter.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut map = std::collections::HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    map.insert(k.str()?.to_string(), val);
                }
            }
        }
        slf.inner = slf.inner.clone().options(map);
        Ok(slf)
    }

    #[pyo3(name = "partitionBy", signature = (*cols))]
    fn partition_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().partition_by(flatten_str_cols(cols)?);
        Ok(slf)
    }

    #[pyo3(name = "bucketBy", signature = (numBuckets, col, *cols))]
    fn bucket_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        numBuckets: i32,
        col: String,
        cols: Vec<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut all = vec![col];
        all.extend(cols);
        slf.inner = slf.inner.clone().bucket_by(numBuckets, all);
        Ok(slf)
    }

    #[pyo3(name = "sortBy", signature = (col, *cols))]
    fn sort_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        col: String,
        cols: Vec<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut all = vec![col];
        all.extend(cols);
        slf.inner = slf.inner.clone().sort_by(all);
        Ok(slf)
    }

    #[pyo3(name = "clusterBy", signature = (*cols))]
    fn cluster_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        cols: Vec<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().cluster_by(cols);
        Ok(slf)
    }

    /// Write to an external JDBC table. Connection details go through `properties`.
    #[pyo3(signature = (url, table, mode=None, properties=None))]
    fn jdbc(
        &mut self,
        url: &str,
        table: &str,
        mode: Option<&str>,
        properties: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<()> {
        // Same order as PySpark: format, mode, properties, then url/dbtable (which
        // win over same-named properties), all kept on the writer, then save.
        let mut w = self.inner.clone().format("jdbc");
        if let Some(m) = mode {
            w = w.mode(m);
        }
        if let Some(props) = properties {
            w = w.options(props);
        }
        w = w.option("url", url).option("dbtable", table);
        self.inner = w.clone();
        w.save(None).to_pyerr()
    }

    #[pyo3(signature = (path=None, format=None, mode=None, partitionBy=None, **options))]
    #[allow(non_snake_case)]
    fn save(
        &mut self,
        path: Option<String>,
        format: Option<&str>,
        mode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let w = apply_write_common(self.inner.clone(), format, mode, partitionBy, options)?;
        // PySpark sets the inline format/mode/partitionBy/options on the writer itself.
        self.inner = w.clone();
        w.save(path.as_deref()).to_pyerr()
    }

    #[pyo3(name = "saveAsTable", signature = (name, format=None, mode=None, partitionBy=None, **options))]
    #[allow(non_snake_case)]
    fn save_as_table(
        &mut self,
        name: &str,
        format: Option<&str>,
        mode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let w = apply_write_common(self.inner.clone(), format, mode, partitionBy, options)?;
        self.inner = w.clone();
        w.save_as_table(name).to_pyerr()
    }

    /// Insert into an existing table. `overwrite` maps to overwrite/append mode,
    /// mirroring reference `DataFrameWriter.insertInto(tableName, overwrite=None)`.
    #[pyo3(name = "insertInto", signature = (tableName, overwrite=None))]
    fn insert_into(&mut self, tableName: &str, overwrite: Option<bool>) -> PyResult<()> {
        let mut w = self.inner.clone();
        if let Some(ow) = overwrite {
            w = w.mode(if ow { "overwrite" } else { "append" });
        }
        // PySpark sets the overwrite/append mode on the writer itself.
        self.inner = w.clone();
        w.insert_into(tableName).to_pyerr()
    }

    /// Write as PARQUET - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, partitionBy=None, compression=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn parquet(
        &mut self,
        path: &str,
        mode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut w = self.inner.clone();
        if let Some(m) = mode {
            w = w.mode(&m.str()?.to_string());
        }
        if let Some(pb) = partitionBy {
            let cols: Vec<String> = if let Ok(one) = pb.extract::<String>() {
                vec![one]
            } else {
                pb.extract::<Vec<String>>()?
            };
            w = w.partition_by(cols);
        }
        w = wset_opt(w, "compression", compression)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner = w.clone().format("parquet");
        w.parquet(path).to_pyerr()
    }

    /// Write as JSON - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, compression=None, dateFormat=None, timestampFormat=None, lineSep=None, encoding=None, ignoreNullFields=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn json(
        &mut self,
        path: &str,
        mode: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        ignoreNullFields: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut w = self.inner.clone();
        if let Some(m) = mode {
            w = w.mode(&m.str()?.to_string());
        }
        w = wset_opt(w, "compression", compression)?;
        w = wset_opt(w, "dateFormat", dateFormat)?;
        w = wset_opt(w, "timestampFormat", timestampFormat)?;
        w = wset_opt(w, "lineSep", lineSep)?;
        w = wset_opt(w, "encoding", encoding)?;
        w = wset_opt(w, "ignoreNullFields", ignoreNullFields)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner = w.clone().format("json");
        w.json(path).to_pyerr()
    }

    /// Write as CSV - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, compression=None, sep=None, quote=None, escape=None, header=None, nullValue=None, escapeQuotes=None, quoteAll=None, dateFormat=None, timestampFormat=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, charToEscapeQuoteEscaping=None, encoding=None, emptyValue=None, lineSep=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn csv(
        &mut self,
        path: &str,
        mode: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
        sep: Option<&Bound<'_, PyAny>>,
        quote: Option<&Bound<'_, PyAny>>,
        escape: Option<&Bound<'_, PyAny>>,
        header: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        escapeQuotes: Option<&Bound<'_, PyAny>>,
        quoteAll: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        ignoreLeadingWhiteSpace: Option<&Bound<'_, PyAny>>,
        ignoreTrailingWhiteSpace: Option<&Bound<'_, PyAny>>,
        charToEscapeQuoteEscaping: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        emptyValue: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut w = self.inner.clone();
        if let Some(m) = mode {
            w = w.mode(&m.str()?.to_string());
        }
        w = wset_opt(w, "compression", compression)?;
        w = wset_opt(w, "sep", sep)?;
        w = wset_opt(w, "quote", quote)?;
        w = wset_opt(w, "escape", escape)?;
        w = wset_opt(w, "header", header)?;
        w = wset_opt(w, "nullValue", nullValue)?;
        w = wset_opt(w, "escapeQuotes", escapeQuotes)?;
        w = wset_opt(w, "quoteAll", quoteAll)?;
        w = wset_opt(w, "dateFormat", dateFormat)?;
        w = wset_opt(w, "timestampFormat", timestampFormat)?;
        w = wset_opt(w, "ignoreLeadingWhiteSpace", ignoreLeadingWhiteSpace)?;
        w = wset_opt(w, "ignoreTrailingWhiteSpace", ignoreTrailingWhiteSpace)?;
        w = wset_opt(w, "charToEscapeQuoteEscaping", charToEscapeQuoteEscaping)?;
        w = wset_opt(w, "encoding", encoding)?;
        w = wset_opt(w, "emptyValue", emptyValue)?;
        w = wset_opt(w, "lineSep", lineSep)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner = w.clone().format("csv");
        w.csv(path).to_pyerr()
    }

    /// Write as ORC - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, partitionBy=None, compression=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn orc(
        &mut self,
        path: &str,
        mode: Option<&Bound<'_, PyAny>>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut w = self.inner.clone();
        if let Some(m) = mode {
            w = w.mode(&m.str()?.to_string());
        }
        if let Some(pb) = partitionBy {
            let cols: Vec<String> = if let Ok(one) = pb.extract::<String>() {
                vec![one]
            } else {
                pb.extract::<Vec<String>>()?
            };
            w = w.partition_by(cols);
        }
        w = wset_opt(w, "compression", compression)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner = w.clone().format("orc");
        w.orc(path).to_pyerr()
    }

    /// Write as TEXT - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, compression=None, lineSep=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn text(
        &mut self,
        path: &str,
        compression: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut w = self.inner.clone();
        w = wset_opt(w, "compression", compression)?;
        w = wset_opt(w, "lineSep", lineSep)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner = w.clone().format("text");
        w.text(path).to_pyerr()
    }

    /// Write as XML - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, rowTag=None, mode=None, attributePrefix=None, valueTag=None, rootTag=None, declaration=None, arrayElementName=None, nullValue=None, dateFormat=None, timestampFormat=None, compression=None, encoding=None, validateName=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn xml(
        &mut self,
        path: &str,
        rowTag: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        attributePrefix: Option<&Bound<'_, PyAny>>,
        valueTag: Option<&Bound<'_, PyAny>>,
        rootTag: Option<&Bound<'_, PyAny>>,
        declaration: Option<&Bound<'_, PyAny>>,
        arrayElementName: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        compression: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        validateName: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut w = self.inner.clone();
        w = wset_opt(w, "rowTag", rowTag)?;
        if let Some(m) = mode {
            w = w.mode(&m.str()?.to_string());
        }
        w = wset_opt(w, "attributePrefix", attributePrefix)?;
        w = wset_opt(w, "valueTag", valueTag)?;
        w = wset_opt(w, "rootTag", rootTag)?;
        w = wset_opt(w, "declaration", declaration)?;
        w = wset_opt(w, "arrayElementName", arrayElementName)?;
        w = wset_opt(w, "nullValue", nullValue)?;
        w = wset_opt(w, "dateFormat", dateFormat)?;
        w = wset_opt(w, "timestampFormat", timestampFormat)?;
        w = wset_opt(w, "compression", compression)?;
        w = wset_opt(w, "encoding", encoding)?;
        w = wset_opt(w, "validateName", validateName)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner = w.clone().format("xml");
        w.xml(path).to_pyerr()
    }
}

/// Python wrapper for the V2 `DataFrameWriter` (`df.writeTo`).
#[pyclass(name = "DataFrameWriterV2", module = "pyspark.sql.readwriter")]
pub struct PyDataFrameWriterV2 {
    inner: spark_connect::readwriter::DataFrameWriterV2,
}

impl PyDataFrameWriterV2 {}

#[pymethods]
impl PyDataFrameWriterV2 {
    fn using<'py>(mut slf: PyRefMut<'py, Self>, provider: &str) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().using(provider);
        Ok(slf)
    }

    // None -> option left unset; bools -> "true"/"false" (reference `to_str`), same as
    // the other reader/writer option bindings.
    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        match crate::coerce_option_value(value)? {
            Some(v) => {
                slf.inner = slf.inner.clone().option(key, &v);
                Ok(slf)
            }
            None => Ok(slf),
        }
    }

    // Mirrors reference `DataFrameWriterV2.options(**options)`; None skipped, bools lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut map = std::collections::HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    map.insert(k.str()?.to_string(), val);
                }
            }
        }
        slf.inner = slf.inner.clone().options(map);
        Ok(slf)
    }

    #[pyo3(name = "tableProperty")]
    fn table_property<'py>(
        mut slf: PyRefMut<'py, Self>,
        property: &str,
        value: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().table_property(property, value);
        Ok(slf)
    }

    #[pyo3(name = "partitionedBy", signature = (col, *cols))]
    fn partitioned_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        col: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut all = vec![col];
        all.extend(cols);
        let columns = to_column_list(all)?;
        slf.inner = slf.inner.clone().partition_by(columns);
        Ok(slf)
    }

    #[pyo3(name = "clusterBy", signature = (col, *cols))]
    fn cluster_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        col: String,
        cols: Vec<String>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut all = vec![col];
        all.extend(cols);
        slf.inner = slf.inner.clone().cluster_by(all);
        Ok(slf)
    }

    fn create(&mut self) -> PyResult<()> {
        self.inner.clone().create().to_pyerr()
    }

    fn replace(&mut self) -> PyResult<()> {
        self.inner.clone().replace().to_pyerr()
    }

    #[pyo3(name = "createOrReplace")]
    fn create_or_replace(&mut self) -> PyResult<()> {
        self.inner.clone().create_or_replace().to_pyerr()
    }

    fn append(&mut self) -> PyResult<()> {
        self.inner.clone().append().to_pyerr()
    }

    fn overwrite(&mut self, condition: &PyColumn) -> PyResult<()> {
        self.inner
            .clone()
            .overwrite(condition.column.clone())
            .to_pyerr()
    }

    #[pyo3(name = "overwritePartitions")]
    fn overwrite_partitions(&mut self) -> PyResult<()> {
        self.inner.clone().overwrite_partitions().to_pyerr()
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

/// The arguments of a call, bound to `names` positionally and by keyword as Python binds a
/// signature of those names with `defaults`, raising Python's `TypeError`s of `function`
/// for too many, repeated or unexpected arguments. For a method that tells a value of None
/// from an argument not given.
fn bind_arguments<'py, const N: usize>(
    function: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    names: [&str; N],
    defaults: [Bound<'py, PyAny>; N],
) -> PyResult<[Bound<'py, PyAny>; N]> {
    if args.len() > N {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{function}() takes from 1 to {} positional arguments but {} were given",
            N + 1,
            args.len() + 1
        )));
    }
    let mut values: [Option<Bound<'py, PyAny>>; N] = std::array::from_fn(|_| None);
    for (i, value) in args.iter().enumerate() {
        values[i] = Some(value);
    }
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key: String = key.extract()?;
            let Some(i) = names.iter().position(|name| *name == key) else {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "{function}() got an unexpected keyword argument '{key}'"
                )));
            };
            if values[i].is_some() {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "{function}() got multiple values for argument '{key}'"
                )));
            }
            values[i] = Some(value);
        }
    }
    let mut defaults = defaults.into_iter();
    Ok(values.map(|value| {
        let default = defaults.next().expect("a default per name");
        value.unwrap_or(default)
    }))
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

/// Validate the function of a map UDF for its eval type with
/// `pyspark.sql.pandas.functions._validate_vectorized_udf`, as the reference client does
/// before it builds the UDF.
pub(crate) fn validate_map_function(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    eval_type_id: i32,
) -> PyResult<()> {
    py.import("pyspark.sql.pandas.functions")?
        .call_method1("_validate_vectorized_udf", (func, eval_type_id))?;
    Ok(())
}

/// The eval type of a grouped map function from its type hints, by `infer` of
/// `pyspark.sql.pandas.typehints` (the iterator forms, for example), as the reference
/// client infers it. `default` is used when the hints give none, and when they cannot be
/// read, after the reference client's `warning`.
pub(crate) fn infer_group_map_eval_type(
    py: Python<'_>,
    func: &Bound<'_, PyAny>,
    infer: &str,
    warning: &str,
    default: i32,
) -> PyResult<i32> {
    match py
        .import("pyspark.sql.pandas.typehints")?
        .call_method1(infer, (func,))
    {
        Ok(inferred) if inferred.is_none() => Ok(default),
        Ok(inferred) => inferred.extract(),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
            py.import("warnings")?
                .call_method1("warn", (warning, category))?;
            Ok(default)
        }
        Err(err) => Err(err),
    }
}

/// Build a map/group UDF expression: cloudpickle `(func, schema)` as the command and
/// carry the resolved schema as the output type (mirrors the scalar UDF command layout).
pub(crate) fn build_map_udf(
    py: Python<'_>,
    name: &str,
    func: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
    eval_type_id: i32,
    input_cols: &[String],
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let return_type = resolve_datatype(schema)?;
    // The worker reads the pickled type as a `DataType`, so a DDL string is pickled as the
    // type it parses to, as the reference client pickles the type of `_parse_ddl`.
    let pickled_type = if schema.is_instance_of::<pyo3::types::PyString>() {
        crate::types::data_type_to_py(py, &return_type)?.into_bound(py)
    } else {
        schema.clone()
    };
    let tup = PyTuple::new(py, [func.clone(), pickled_type])?;
    let command = py_cloudpickle(py, tup.as_any())?;
    let payload = PythonUDFPayload::new(return_type, eval_type_id, command, py_version(py));
    // The worker reconstructs the input pandas/Arrow DataFrame from the UDF's argument
    // columns; without them the input columns are unnamed (KeyError in the user fn).
    // Mirror the reference Connect client, which passes the child's columns as the
    // function arguments.
    let arguments = input_cols
        .iter()
        .map(|c| spark_connect::functions::col(c).expression().clone())
        .collect();
    Ok(CommonInlineUserDefinedFunctionExpression::new(
        name.to_string(),
        true,
        arguments,
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

/// The converters of the fields of a result of `schema` that deserialize its values of
/// user-defined types, as PySpark does when it collects them
/// (`pyspark.sql.types._udt_field_converters`); None without a user-defined type.
fn udt_field_converters(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<Option<FieldConverters>> {
    let Some(schema) = schema.filter(|schema| crate::types::has_udt(schema)) else {
        return Ok(None);
    };
    let schema = crate::types::data_type_to_py(py, schema)?;
    let converters = py
        .import("pyspark.sql.types")?
        .call_method1("_udt_field_converters", (schema,))?;
    if converters.is_none() {
        return Ok(None);
    }
    Ok(Some(std::sync::Arc::new(converters.extract()?)))
}

/// The collected `rows` of a result of `schema`, whose values of user-defined types are
/// deserialized when first read.
fn rows_to_py(
    py: Python<'_>,
    rows: Vec<spark_connect::row::Row>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<Vec<PyRow>> {
    let converters = udt_field_converters(py, schema)?;
    Ok(rows
        .into_iter()
        .map(|row| PyRow::with_converters(row, converters.clone()))
        .collect())
}

/// Python wrapper for LocalRowIterator yielding Row objects lazily.
///
/// Implements `__iter__` and `__next__` to make it a proper Python iterator.
#[pyclass(name = "LocalRowIterator", module = "pyspark.sql.dataframe")]
pub struct PyLocalRowIterator {
    dataframe: DataFrame,
    prefetch_partitions: bool,
    /// The rows of the execution, which the first row starts, as PySpark's generator does.
    iterator: Option<LocalRowIterator>,
    /// Whether the iteration ended, with its last row or an error, as a generator ends.
    finished: bool,
    /// The converters of the fields of user-defined types of the rows, once the schema of the
    /// result is known: with the first row.
    converters: Option<Option<FieldConverters>>,
}

impl PyLocalRowIterator {
    pub fn new(dataframe: DataFrame, prefetch_partitions: bool) -> Self {
        PyLocalRowIterator {
            dataframe,
            prefetch_partitions,
            iterator: None,
            finished: false,
            converters: None,
        }
    }

    /// The next row, starting the execution for the first.
    fn next_row(&mut self) -> Option<spark_connect_core::error::Result<spark_connect::row::Row>> {
        if self.iterator.is_none() {
            match self.dataframe.to_local_iterator(self.prefetch_partitions) {
                Ok(iterator) => self.iterator = Some(iterator),
                Err(error) => return Some(Err(error)),
            }
        }
        self.iterator.as_mut()?.next()
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
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyRow>> {
        if self.finished {
            return Ok(None);
        }
        // Release the GIL while fetching from the Rust iterator (which may block on network I/O).
        match py.detach(|| self.next_row()) {
            Some(Ok(row)) => {
                if self.converters.is_none() {
                    let schema = self.iterator.as_ref().and_then(LocalRowIterator::schema);
                    self.converters = Some(udt_field_converters(py, schema.as_ref())?);
                }
                let converters = self.converters.clone().flatten();
                Ok(Some(PyRow::with_converters(row, converters)))
            }
            // As PySpark raises the error of the execution, which ends the iteration.
            Some(Err(error)) => {
                self.finished = true;
                Err(crate::errors::spark_error_to_py_exception(error))
            }
            None => {
                self.finished = true;
                Ok(None)
            }
        }
    }
}

/// The metrics and observed metrics of an execution, as the lists of `PlanMetrics` and
/// `PlanObservedMetrics` that `SparkConnectClient._build_metrics` makes.
fn execution_metrics<'py>(
    py: Python<'py>,
    info: &spark_connect::session::ExecutionInfo,
) -> PyResult<(
    Bound<'py, pyo3::types::PyList>,
    Bound<'py, pyo3::types::PyList>,
)> {
    let metrics_mod = py.import("pyspark.sql.metrics")?;
    let plan_metrics_cls = metrics_mod.getattr("PlanMetrics")?;
    let metric_value_cls = metrics_mod.getattr("MetricValue")?;
    let plan_metrics = pyo3::types::PyList::empty(py);
    if let Some(m) = &info.metrics {
        for x in &m.metrics {
            let values = pyo3::types::PyList::empty(py);
            for (k, v) in &x.execution_metrics {
                values.append(metric_value_cls.call1((k, v.value, &v.metric_type))?)?;
            }
            plan_metrics.append(plan_metrics_cls.call1((&x.name, x.plan_id, x.parent, values))?)?;
        }
    }
    let observed_cls = py
        .import("pyspark.sql.connect.client")?
        .getattr("PlanObservedMetrics")?;
    let observed = pyo3::types::PyList::empty(py);
    for o in &info.observed_metrics {
        let values = o
            .values
            .iter()
            .map(|l| literal_to_py(py, l, None))
            .collect::<PyResult<Vec<_>>>()?;
        observed.append(observed_cls.call1((&o.name, values, o.keys.clone()))?)?;
    }
    Ok((plan_metrics, observed))
}

/// Convert an observed-metric literal to a Python value. Mirrors
/// `LiteralExpression._to_value`, which `PlanObservedMetrics.pairs` applies: the kinds it
/// decodes map to the same Python values, and for any other kind (which the reference
/// rejects with `UNSUPPORTED_LITERAL`) this returns an `_UnsupportedLiteral` marker, so
/// the error surfaces when `flows` is read, as in the reference, not when the
/// `ExecutionInfo` is built.
///
/// `data_type` is the literal's type when the enclosing literal carries it; a struct's
/// field names come from it, else from the literal's own `data_type`, else from the
/// deprecated `struct_type`.
fn literal_to_py(
    py: Python<'_>,
    literal: &spark_connect_proto::expression::Literal,
    data_type: Option<&spark_connect_proto::DataType>,
) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    use spark_connect_proto::data_type::Kind;
    use spark_connect_proto::expression::literal::LiteralType;
    let data_type = data_type.or(literal.data_type.as_ref());
    let datetime = || py.import("datetime");
    Ok(match &literal.literal_type {
        None | Some(LiteralType::Null(_)) => py.None(),
        Some(LiteralType::Boolean(b)) => b.into_py_any(py)?,
        Some(LiteralType::Byte(v))
        | Some(LiteralType::Short(v))
        | Some(LiteralType::Integer(v)) => v.into_py_any(py)?,
        Some(LiteralType::Long(v)) => v.into_py_any(py)?,
        Some(LiteralType::Float(v)) => (*v as f64).into_py_any(py)?,
        Some(LiteralType::Double(v)) => v.into_py_any(py)?,
        Some(LiteralType::String(s)) => s.into_py_any(py)?,
        Some(LiteralType::Binary(b)) => pyo3::types::PyBytes::new(py, b).into_any().unbind(),
        Some(LiteralType::Decimal(d)) => py
            .import("decimal")?
            .getattr("Decimal")?
            .call1((&d.value,))?
            .unbind(),
        // DateType.fromInternal: days since the epoch (ordinal 719163).
        Some(LiteralType::Date(days)) => datetime()?
            .getattr("date")?
            .call_method1("fromordinal", (*days as i64 + 719_163,))?
            .unbind(),
        // TimeType.fromInternal: nanoseconds of the day, truncated to microseconds.
        Some(LiteralType::Time(t)) => {
            let nano = t.nano;
            datetime()?
                .getattr("time")?
                .call1((
                    nano.div_euclid(3_600_000_000_000),
                    nano.rem_euclid(3_600_000_000_000) / 60_000_000_000,
                    nano.rem_euclid(60_000_000_000) / 1_000_000_000,
                    nano.rem_euclid(1_000_000_000) / 1_000,
                ))?
                .unbind()
        }
        // TimestampType.fromInternal: a naive local datetime, built from whole seconds so
        // the microseconds are exact.
        Some(LiteralType::Timestamp(ts)) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microsecond", ts.rem_euclid(1_000_000))?;
            datetime()?
                .getattr("datetime")?
                .call_method1("fromtimestamp", (ts.div_euclid(1_000_000),))?
                .call_method("replace", (), Some(&kwargs))?
                .unbind()
        }
        // TimestampNTZType.fromInternal: the UTC wall clock, without a zone.
        Some(LiteralType::TimestampNtz(ts)) => {
            let dt = datetime()?;
            let utc = dt.getattr("timezone")?.getattr("utc")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microsecond", ts.rem_euclid(1_000_000))?;
            kwargs.set_item("tzinfo", py.None())?;
            dt.getattr("datetime")?
                .call_method1("fromtimestamp", (ts.div_euclid(1_000_000), utc))?
                .call_method("replace", (), Some(&kwargs))?
                .unbind()
        }
        // DayTimeIntervalType.fromInternal: timedelta(microseconds=v).
        Some(LiteralType::DayTimeInterval(micros)) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("microseconds", micros)?;
            datetime()?
                .getattr("timedelta")?
                .call((), Some(&kwargs))?
                .unbind()
        }
        Some(LiteralType::Array(a)) => {
            let element_type = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Array(t)) => t.element_type.as_deref(),
                _ => None,
            };
            let items = a
                .elements
                .iter()
                .map(|e| literal_to_py(py, e, element_type))
                .collect::<PyResult<Vec<_>>>()?;
            items.into_py_any(py)?
        }
        Some(LiteralType::Map(m)) => {
            let (key_type, value_type) = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Map(t)) => (t.key_type.as_deref(), t.value_type.as_deref()),
                _ => (None, None),
            };
            let dict = pyo3::types::PyDict::new(py);
            for (k, v) in m.keys.iter().zip(&m.values) {
                dict.set_item(
                    literal_to_py(py, k, key_type)?,
                    literal_to_py(py, v, value_type)?,
                )?;
            }
            dict.into_any().unbind()
        }
        Some(LiteralType::Struct(s)) => {
            #[allow(deprecated)]
            let struct_type = match data_type.and_then(|t| t.kind.as_ref()) {
                Some(Kind::Struct(t)) => Some(t),
                _ => match s.struct_type.as_ref().and_then(|t| t.kind.as_ref()) {
                    Some(Kind::Struct(t)) => Some(t),
                    _ => None,
                },
            };
            let fields = struct_type.map(|t| t.fields.as_slice()).unwrap_or_default();
            let names = (0..s.elements.len())
                .map(|i| match fields.get(i) {
                    Some(f) => f.name.clone(),
                    None => format!("_{}", i + 1),
                })
                .collect::<Vec<_>>();
            let values = s
                .elements
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    literal_to_py(py, e, fields.get(i).and_then(|f| f.data_type.as_ref()))
                })
                .collect::<PyResult<Vec<_>>>()?;
            py.import("pyspark.sql.types")?
                .getattr("_create_row")?
                .call1((names, values))?
                .unbind()
        }
        Some(_) => py
            .import("pyspark.sql.connect.client")?
            .getattr("_UnsupportedLiteral")?
            .call1((format!("{literal:?}"),))?
            .unbind(),
    })
}

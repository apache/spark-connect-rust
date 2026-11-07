//! PyO3 wrapper for spark_connect::dataframe::DataFrame.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyTuple};
use spark_connect::dataframe::{DataFrame, LocalRowIterator};
use spark_connect::plan::JoinType;
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::column::PyColumn;
use crate::errors::ResultExt;
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
    /// `other`, a DataFrame of the session of this one, as PySpark's `_check_same_session`
    /// takes the DataFrame of a method: else `SESSION_NOT_SAME`, as for what is not a
    /// DataFrame.
    pub(crate) fn same_session<'py>(
        &self,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<PyRef<'py, PyDataFrame>> {
        match other.extract::<PyRef<'py, PyDataFrame>>() {
            Ok(df)
                if df.dataframe.spark_session().session_id()
                    == self.dataframe.spark_session().session_id() =>
            {
                Ok(df)
            }
            _ => Err(crate::column::raise_pyspark(
                other.py(),
                "SessionNotSameException",
                "SESSION_NOT_SAME",
                &[],
            )),
        }
    }

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

    /// The column at the 0-based index `index`, as PySpark's `df[index]`: `col` of the name of
    /// the column at that index of `columns`, of Python's list indexing, and of the origin of
    /// `col`.
    fn column_at(&self, py: Python<'_>, index: i64) -> PyResult<spark_connect::column::Column> {
        let columns = self.columns(py)?;
        let name: String = PyList::new(py, columns)?
            .as_any()
            .get_item(index)?
            .extract()?;
        let _origin = crate::origin::with_origin("col");
        Ok(spark_connect::functions::col(&name))
    }

    /// The grouping columns of `groupBy`, `rollup`, `cube` and `groupingSets`, as PySpark's
    /// `_to_cols` of ordinals: Columns, column names and the 1-based ordinals of columns, or
    /// a list or tuple of them.
    fn grouping_columns(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        arg_name: &str,
    ) -> PyResult<Vec<spark_connect::column::Column>> {
        self.grouping_column_list(py, unwrap_sequence(cols)?, arg_name)
    }

    /// The grouping columns of `cols`, each a Column, a column name or the 1-based ordinal of
    /// a column, as PySpark's `_to_cols` of ordinals takes the items of a sequence.
    fn grouping_column_list(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        arg_name: &str,
    ) -> PyResult<Vec<spark_connect::column::Column>> {
        use pyo3::types::{PyBool, PyString};
        cols.iter()
            .map(|c| {
                if let Ok(column) = c.extract::<PyRef<'_, PyColumn>>() {
                    return Ok(column.column.clone());
                }
                if c.is_instance_of::<PyString>() {
                    // As PySpark's `F._to_col`, by `col`, of its origin.
                    let _origin = crate::origin::with_origin("col");
                    return Ok(spark_connect::functions::col(&c.extract::<String>()?));
                }
                if c.is_instance_of::<PyInt>() && !c.is_instance_of::<PyBool>() {
                    let ordinal: i64 = c.extract()?;
                    if ordinal < 1 {
                        return Err(crate::column::raise_pyspark(
                            py,
                            "PySparkIndexError",
                            "INDEX_NOT_POSITIVE",
                            &[("index", &ordinal.to_string())],
                        ));
                    }
                    return self.column_at(py, ordinal - 1);
                }
                Err(not_expected_type(c, arg_name, "Column, int or str"))
            })
            .collect()
    }

    /// The sort orders of `sort`, `orderBy` and `sortWithinPartitions`, as PySpark's
    /// `_preapare_cols_for_sort`: Columns, column names and the 1-based ordinals of columns,
    /// descending for a negative one, or a list or tuple of them, each descending for a
    /// false `ascending`, or for a false one of a list of them.
    fn sort_columns(
        &self,
        py: Python<'_>,
        cols: Vec<Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<spark_connect::expression::Expression>> {
        use pyo3::types::{PyBool, PyString};
        if cols.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "cols")],
            ));
        }
        let mut columns = Vec::new();
        for c in unwrap_sequence(cols)? {
            let column = if c.is_instance_of::<PyInt>() && !c.is_instance_of::<PyBool>() {
                let ordinal: i64 = c.extract()?;
                match ordinal {
                    0 => {
                        return Err(crate::column::raise_pyspark(
                            py,
                            "PySparkIndexError",
                            "ZERO_INDEX",
                            &[],
                        ))
                    }
                    ordinal if ordinal > 0 => self.column_at(py, ordinal - 1)?,
                    ordinal => self.column_at(py, -ordinal - 1)?.desc(),
                }
            } else if let Ok(column) = c.extract::<PyRef<'_, PyColumn>>() {
                column.column.clone()
            } else if c.is_instance_of::<PyString>() {
                // As PySpark's `F._to_col`, by `col`, of its origin.
                let _origin = crate::origin::with_origin("col");
                spark_connect::functions::col(&c.extract::<String>()?)
            } else {
                return Err(not_expected_type(&c, "col", "Column, int or str"));
            };
            columns.push(column);
        }
        let ascending = match kwargs {
            Some(kwargs) => kwargs.get_item("ascending")?,
            None => None,
        };
        if let Some(ascending) = ascending {
            if ascending.is_instance_of::<PyInt>() {
                if !ascending.is_truthy()? {
                    columns = columns.into_iter().map(|c| c.desc()).collect();
                }
            } else if ascending.is_instance_of::<PyList>() {
                // As PySpark's zip of them, which keeps a column for each flag.
                let flags: Vec<bool> = ascending
                    .try_iter()?
                    .map(|flag| flag?.is_truthy())
                    .collect::<PyResult<_>>()?;
                columns = columns
                    .into_iter()
                    .zip(flags)
                    .map(|(c, asc)| if asc { c } else { c.desc() })
                    .collect();
            } else {
                return Err(not_expected_type(
                    &ascending,
                    "ascending",
                    "Column, int or str",
                ));
            }
        }
        Ok(columns.iter().map(|c| c.expression().clone()).collect())
    }
}

/// The arguments of a method of `*cols`, the elements of a single list or tuple of them, as
/// PySpark takes them.
fn unwrap_sequence(cols: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Bound<'_, PyAny>>> {
    match cols.as_slice() {
        [only] if only.is_instance_of::<PyList>() || only.is_instance_of::<PyTuple>() => {
            only.try_iter()?.collect()
        }
        _ => Ok(cols),
    }
}

/// PySpark's `NOT_EXPECTED_TYPE` error of the argument `name`, `value`, which should be of
/// the type `expected`.
fn not_expected_type(value: &Bound<'_, PyAny>, name: &str, expected: &str) -> PyErr {
    let type_name = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    crate::column::raise_pyspark(
        value.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("expected_type", expected),
            ("arg_name", name),
            ("arg_type", &type_name),
        ],
    )
}

/// PySpark's `VALUE_NOT_POSITIVE` error of `numPartitions`, unless it is positive.
fn positive_partitions(py: Python<'_>, num_partitions: i64) -> PyResult<i32> {
    if num_partitions > 0 {
        return Ok(i32::try_from(num_partitions)?);
    }
    Err(crate::column::raise_pyspark(
        py,
        "PySparkValueError",
        "VALUE_NOT_POSITIVE",
        &[
            ("arg_name", "numPartitions"),
            ("arg_value", &num_partitions.to_string()),
        ],
    ))
}

/// The `subset` of `dropDuplicates` and `dropDuplicatesWithinWatermark`, as PySpark's: None,
/// or a list or tuple of column names.
fn deduplicate_subset(subset: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<String>>> {
    let Some(subset) = subset else {
        return Ok(None);
    };
    if !(subset.is_instance_of::<PyList>() || subset.is_instance_of::<PyTuple>()) {
        return Err(not_expected_type(subset, "subset", "list or tuple"));
    }
    subset
        .try_iter()?
        .map(|column| {
            let column = column?;
            if column.is_instance_of::<pyo3::types::PyString>() {
                column.extract()
            } else {
                Err(not_expected_type(&column, "subset", "str"))
            }
        })
        .collect::<PyResult<Vec<String>>>()
        .map(Some)
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
                // As PySpark's `F._to_col`, by `col`, of its origin.
                let _origin = crate::origin::with_origin("col");
                cols.push(spark_connect::functions::col(&name));
            } else {
                // As PySpark's `_to_cols`.
                return Err(not_expected_type(&arg, "cols", "Column or str"));
            }
        }
    }
    Ok(cols)
}

/// The join types that `DataFrame.join` accepts, as the message of its `UNSUPPORTED_JOIN_TYPE`.
const JOIN_TYPES_SUPPORTED: &str = "'inner', 'outer', 'full', 'fullouter', 'full_outer', 'leftouter', 'left', 'left_outer', 'rightouter', 'right', 'right_outer', 'leftsemi', 'left_semi', 'semi', 'leftanti', 'left_anti', 'anti', 'cross'";

/// The join types that `DataFrame.lateralJoin` accepts, as the message of its
/// `UNSUPPORTED_JOIN_TYPE`.
const LATERAL_JOIN_TYPES_SUPPORTED: &str = "'inner', 'leftouter', 'left', 'left_outer', 'cross'";

/// Map a pyspark `how` to a JoinType, as the connect client's `plan.Join` (or `plan.LateralJoin`
/// when `lateral`) maps it. The `how` is lowercased with its underscores removed, and no `how`
/// is an inner join. One that is not accepted raises the `AnalysisException` of
/// `UNSUPPORTED_JOIN_TYPE` that PySpark raises locally, naming the normalized `how` (or the
/// `str()` of a `how` that is not a str).
fn parse_join_type(
    py: Python<'_>,
    how: Option<&Bound<'_, PyAny>>,
    lateral: bool,
) -> PyResult<JoinType> {
    let Some(how) = how.filter(|how| !how.is_none()) else {
        return Ok(JoinType::Inner);
    };
    let normalized = how
        .extract::<String>()
        .ok()
        .map(|how| how.to_lowercase().replace('_', ""));
    let join_type = match normalized.as_deref() {
        Some("inner") => Some(JoinType::Inner),
        Some("cross") => Some(JoinType::Cross),
        Some("left" | "leftouter") => Some(JoinType::LeftOuter),
        Some("right" | "rightouter") if !lateral => Some(JoinType::RightOuter),
        Some("outer" | "full" | "fullouter") if !lateral => Some(JoinType::FullOuter),
        Some("leftsemi" | "semi") if !lateral => Some(JoinType::LeftSemi),
        Some("leftanti" | "anti") if !lateral => Some(JoinType::LeftAnti),
        _ => None,
    };
    if let Some(join_type) = join_type {
        return Ok(join_type);
    }
    let typ = match normalized {
        Some(normalized) => normalized,
        None => how.str()?.to_string(),
    };
    let supported = if lateral {
        LATERAL_JOIN_TYPES_SUPPORTED
    } else {
        JOIN_TYPES_SUPPORTED
    };
    Err(crate::column::raise_pyspark(
        py,
        "AnalysisException",
        "UNSUPPORTED_JOIN_TYPE",
        &[("typ", typ.as_str()), ("supported", supported)],
    ))
}

/// The `AttributeError` that PySpark's join raises for an `on` that is neither a name, a
/// condition nor a list of them: the plan reads `to_plan` of it.
fn no_to_plan(obj: &Bound<'_, PyAny>) -> PyErr {
    let typ = obj
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    pyo3::exceptions::PyAttributeError::new_err(format!(
        "'{typ}' object has no attribute 'to_plan'"
    ))
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

/// An argument of `withWatermark`, as PySpark's: `NOT_EXPECTED_TYPE` for an empty one, or one
/// whose type is not str.
fn watermark_argument(value: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    if !value.is_truthy()?
        || !value
            .get_type()
            .is(value.py().get_type::<pyo3::types::PyString>())
    {
        return Err(crate::stat_arguments::not_expected_type(value, name, "str"));
    }
    value.extract()
}

#[pymethods]
impl PyDataFrame {
    /// Select columns (accepts Column objects or string names).
    #[pyo3(signature = (*cols))]
    fn select(&self, _py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let columns = to_column_list(cols)?;
        Ok(PyDataFrame::new(self.dataframe.select(columns)))
    }

    /// Select using SQL expressions, as PySpark's `DataFrame.selectExpr`: strings, or lists of
    /// them.
    #[pyo3(signature = (*expr))]
    fn selectExpr(&self, expr: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let expr = match expr.as_slice() {
            [only] if only.is_instance_of::<PyList>() => {
                only.try_iter()?.collect::<PyResult<Vec<_>>>()?
            }
            _ => expr,
        };
        let mut columns = Vec::new();
        for element in &expr {
            if element.is_instance_of::<pyo3::types::PyString>() {
                columns.push(spark_connect::functions::expr(
                    &element.extract::<String>()?,
                ));
            } else {
                for e in element.try_iter()? {
                    columns.push(spark_connect::functions::expr(&e?.extract::<String>()?));
                }
            }
        }
        Ok(PyDataFrame::new(self.dataframe.select(columns)))
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
        // As PySpark's: a Column or a str, else NOT_EXPECTED_TYPE.
        if !(condition.is_instance_of::<PyColumn>()
            || condition.is_instance_of::<pyo3::types::PyString>())
        {
            return Err(not_expected_type(condition, "condition", "Column or str"));
        }
        Ok(PyDataFrame::new(
            self.dataframe.filter(cond_column(condition)?),
        ))
    }

    /// Add or replace a column, of a Column, as PySpark's `DataFrame.withColumn`.
    fn withColumn(&self, colName: &str, col: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let Ok(column) = col.extract::<PyRef<'_, PyColumn>>() else {
            return Err(not_expected_type(col, "col", "Column"));
        };
        Ok(PyDataFrame::new(
            self.dataframe.with_column(colName, column.column.clone()),
        ))
    }

    /// Rename a column.
    fn withColumnRenamed(&self, existing: &str, new: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.with_column_renamed(existing, new))
    }

    /// Add or replace multiple columns from a {name: Column} mapping, as PySpark's
    /// `DataFrame.withColumns`, which asserts that it is given one, of Columns.
    #[pyo3(name = "withColumns", signature = (*colsMap))]
    fn with_columns(&self, colsMap: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let [cols_map] = colsMap.as_slice() else {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        };
        let Ok(cols_map) = cols_map.cast::<PyDict>() else {
            return Err(not_expected_type(cols_map, "colsMap", "dict"));
        };
        let mut pairs = Vec::with_capacity(cols_map.len());
        for (name, column) in cols_map.iter() {
            let Ok(column) = column.extract::<PyRef<'_, PyColumn>>() else {
                return Err(pyo3::exceptions::PyAssertionError::new_err(()));
            };
            pairs.push((name.extract::<String>()?, column.column.clone()));
        }
        Ok(PyDataFrame::new(self.dataframe.with_columns(pairs)))
    }

    /// Specify a relation hint (e.g. "broadcast"), with optional parameters. Reference
    /// hint params may be ints (e.g. `df.hint("rebalance", 3)`), not just strings, so
    /// accept any value and stringify it; the core parses a numeric string back into a
    /// long literal, matching reference semantics.
    #[pyo3(signature = (name, *parameters))]
    fn hint(
        &self,
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
        parameters: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        use pyo3::types::{PyFloat, PyString};
        // As PySpark's: a single list is the parameters, each a str, a float, an int, a
        // Column or a list of str, float or int, sent as `lit` of it.
        let parameters = match parameters.as_slice() {
            [only] if only.is_instance_of::<PyList>() => {
                only.try_iter()?.collect::<PyResult<Vec<_>>>()?
            }
            _ => parameters,
        };
        if !name.is_instance_of::<PyString>() {
            return Err(not_expected_type(name, "name", "str"));
        }
        let primitive = |p: &Bound<'_, PyAny>| {
            p.is_instance_of::<PyString>()
                || p.is_instance_of::<PyFloat>()
                || p.is_instance_of::<PyInt>()
        };
        let invalid = |item_type: &str| {
            crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "INVALID_ITEM_FOR_CONTAINER",
                &[
                    ("arg_name", "parameters"),
                    (
                        "allowed_types",
                        "str, float, int, Column, list[str], list[float], list[int]",
                    ),
                    ("item_type", item_type),
                ],
            )
        };
        let type_name =
            |p: &Bound<'_, PyAny>| -> PyResult<String> { Ok(p.get_type().name()?.to_string()) };
        let mut expressions = Vec::with_capacity(parameters.len());
        for p in &parameters {
            if p.is_instance_of::<PyList>() {
                let items: Vec<Bound<'_, PyAny>> = p.try_iter()?.collect::<PyResult<_>>()?;
                if !items.iter().all(primitive) {
                    // As PySpark's, which names the type of the first item.
                    let first = items
                        .first()
                        .map(type_name)
                        .transpose()?
                        .unwrap_or_default();
                    return Err(invalid(&format!("list[{first}]")));
                }
            } else if !(primitive(p) || p.extract::<PyRef<'_, PyColumn>>().is_ok()) {
                return Err(invalid(&type_name(p)?));
            }
            expressions.push(crate::functions::to_column(p)?.expression().clone());
        }
        let name: String = name.extract()?;
        Ok(PyDataFrame::new(
            self.dataframe.hint_expressions(&name, expressions),
        ))
    }

    /// Drop columns, of Columns or column names, as PySpark's `DataFrame.drop`.
    #[pyo3(signature = (*cols))]
    fn drop(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let (mut columns, mut names) = (Vec::new(), Vec::new());
        for column in &cols {
            if let Ok(column) = column.extract::<PyRef<'_, PyColumn>>() {
                columns.push(column.column.clone());
            } else if column.is_instance_of::<pyo3::types::PyString>() {
                names.push(column.extract::<String>()?);
            } else {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "Column or str"),
                        ("arg_name", "cols"),
                        ("arg_type", "tuple"),
                    ],
                ));
            }
        }
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        Ok(PyDataFrame::new(
            self.dataframe.drop_columns(columns, names),
        ))
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
    fn dropDuplicates(&self, subset: Option<&Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let subset = deduplicate_subset(subset)?;
        let col_refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|names| names.iter().map(|s| s.as_str()).collect());
        Ok(PyDataFrame::new(self.dataframe.drop_duplicates(col_refs)))
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
        let exprs = self.sort_columns(_py, cols, kwargs.as_ref())?;
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
        let exprs = self.sort_columns(_py, cols, kwargs.as_ref())?;
        Ok(PyDataFrame::new(self.dataframe.order_by(exprs)))
    }

    /// Join with another DataFrame.
    ///
    /// `on` is a Column condition, a column name, or a list of column names (a "using"
    /// join) or of Column conditions (joined on their conjunction), as the connect client's
    /// `plan.Join` takes it. Mirrors `pyspark.sql.DataFrame.join(other, on=None, how=None)`
    /// (default inner), including its errors.
    #[pyo3(signature = (other, on=None, how=None))]
    fn join(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        on: Option<Bound<'_, PyAny>>,
        how: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's `_check_same_session`, first.
        let other = self.same_session(other)?;
        // As PySpark's `plan.Join`, `on` is iterated before `how` is checked, so an `on`
        // that is not iterable is the error that is raised first.
        if let Some(on) = &on {
            if !on.is_instance_of::<PyColumn>() && !on.is_instance_of::<pyo3::types::PyString>() {
                on.try_iter()?;
            }
        }
        let join_type = parse_join_type(py, how.as_ref(), false)?;
        let Some(on) = on else {
            return Ok(PyDataFrame::new(self.dataframe.join(
                &other.dataframe,
                None,
                join_type,
            )));
        };
        if let Ok(col) = on.extract::<PyRef<PyColumn>>() {
            return Ok(PyDataFrame::new(self.dataframe.join(
                &other.dataframe,
                Some(col.column.clone()),
                join_type,
            )));
        }
        if let Ok(name) = on.extract::<String>() {
            return Ok(PyDataFrame::new(self.dataframe.join_using(
                &other.dataframe,
                vec![name],
                join_type,
            )));
        }
        if let Ok(list) = on.cast::<PyList>() {
            // An empty list is no condition at all.
            let Some(first) = list.iter().next() else {
                return Ok(PyDataFrame::new(self.dataframe.join(
                    &other.dataframe,
                    None,
                    join_type,
                )));
            };
            if first.is_instance_of::<pyo3::types::PyString>() {
                let names = list.extract::<Vec<String>>()?;
                return Ok(PyDataFrame::new(self.dataframe.join_using(
                    &other.dataframe,
                    names,
                    join_type,
                )));
            }
            let mut condition: Option<spark_connect::column::Column> = None;
            for item in list.iter() {
                let Ok(col) = item.extract::<PyRef<PyColumn>>() else {
                    return Err(no_to_plan(&item));
                };
                condition = Some(match condition {
                    None => col.column.clone(),
                    Some(left) => left.and(col.column.clone()),
                });
            }
            return Ok(PyDataFrame::new(self.dataframe.join(
                &other.dataframe,
                condition,
                join_type,
            )));
        }
        Err(no_to_plan(&on))
    }

    /// Cross join.
    fn crossJoin(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe.cross_join(&other.dataframe),
        ))
    }

    /// Lateral (correlated) join. `on` is an optional Column condition; `how` is inner by
    /// default and takes the join types of a lateral join. Mirrors `DataFrame.lateralJoin`.
    #[pyo3(signature = (other, on=None, how=None))]
    fn lateralJoin(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        on: Option<PyRef<PyColumn>>,
        how: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's `_check_same_session`, first.
        let other = self.same_session(other)?;
        let join_type = parse_join_type(py, how.as_ref(), true)?;
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
    fn transpose(&self, indexColumn: Option<&Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        // As PySpark's `F._to_col`: a Column or a column name, else NOT_EXPECTED_TYPE of `col`.
        let index = match indexColumn {
            None => None,
            Some(c) => Some(if let Ok(column) = c.extract::<PyRef<'_, PyColumn>>() {
                column.column.clone()
            } else if c.is_instance_of::<pyo3::types::PyString>() {
                let _origin = crate::origin::with_origin("col");
                spark_connect::functions::col(&c.extract::<String>()?)
            } else {
                return Err(not_expected_type(c, "col", "Column or str"));
            }),
        };
        let df = match index {
            Some(column) => self.dataframe.transpose_with_index(column),
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
        py: Python<'_>,
        numPartitions: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's: a positive partition count and columns, or columns alone, each sorted
        // ascending unless it is a sort order.
        let (num_partitions, cols) = if numPartitions.is_instance_of::<PyInt>() {
            let n = positive_partitions(py, numPartitions.extract()?)?;
            if cols.is_empty() {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_BE_EMPTY",
                    &[("item", "cols")],
                ));
            }
            (n, cols)
        } else if numPartitions.is_instance_of::<pyo3::types::PyString>()
            || numPartitions.extract::<PyRef<'_, PyColumn>>().is_ok()
        {
            let mut all = vec![numPartitions];
            all.extend(cols);
            (0, all)
        } else {
            return Err(not_expected_type(
                &numPartitions,
                "numPartitions",
                "Column, int or str",
            ));
        };
        let exprs: Vec<_> = to_column_list(cols)?
            .into_iter()
            .map(|c| match c.expression().node() {
                spark_connect::expression::ExpressionNode::SortOrder(_) => c.expression().clone(),
                _ => c.asc().expression().clone(),
            })
            .collect();
        Ok(PyDataFrame::new(
            self.dataframe.repartition_by_range(num_partitions, exprs),
        ))
    }

    /// Repartition into `numPartitions` using a column's value directly as the shuffle
    /// partition id. Mirrors `DataFrame.repartitionById(numPartitions, partitionIdCol)`.
    #[pyo3(signature = (numPartitions, partitionIdCol))]
    #[allow(non_snake_case)]
    fn repartitionById(
        &self,
        py: Python<'_>,
        numPartitions: &Bound<'_, PyAny>,
        partitionIdCol: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's: an int that is not a bool, and positive.
        if !numPartitions.is_instance_of::<PyInt>()
            || numPartitions.is_instance_of::<pyo3::types::PyBool>()
        {
            return Err(not_expected_type(numPartitions, "numPartitions", "int"));
        }
        let numPartitions = positive_partitions(py, numPartitions.extract()?)?;
        // A `ColumnOrName`, as in pyspark: a string is a column name, not a literal.
        let col = if let Ok(c) = partitionIdCol.extract::<Bound<'_, PyColumn>>() {
            c.borrow().column.clone()
        } else if let Ok(name) = partitionIdCol.extract::<String>() {
            // As PySpark's `F._to_col`, by `col`, of its origin.
            let _origin = crate::origin::with_origin("col");
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
    fn zip(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe.zip(&other.dataframe).to_pyerr()?,
        ))
    }

    /// Union with another DataFrame.
    fn union(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(self.dataframe.union(&other.dataframe)))
    }

    /// Union by name, optionally filling columns missing on one side with null.
    #[pyo3(signature = (other, allowMissingColumns=false))]
    fn unionByName(
        &self,
        other: &Bound<'_, PyAny>,
        allowMissingColumns: bool,
    ) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe
                .union_by_name_opt(&other.dataframe, allowMissingColumns),
        ))
    }

    /// Intersect with another DataFrame.
    fn intersect(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(self.dataframe.intersect(&other.dataframe)))
    }

    /// Intersect keeping duplicates (reference pyspark `intersectAll`).
    #[pyo3(name = "intersectAll")]
    fn intersect_all(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe.intersect_all(&other.dataframe),
        ))
    }

    /// Subtract another DataFrame.
    fn subtract(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(self.dataframe.subtract(&other.dataframe)))
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
        // As PySpark's: a positive partition count, or a partition column (str/Column), at
        // the server's count (passed as 0 = unset).
        if numPartitions.is_instance_of::<PyInt>() {
            let n = positive_partitions(py, numPartitions.extract()?)?;
            if cols.is_empty() {
                return Ok(PyDataFrame::new(self.dataframe.repartition(n)));
            }
            let exprs: Vec<_> = to_column_list(cols)?
                .iter()
                .map(|c| c.expression().clone())
                .collect();
            return Ok(PyDataFrame::new(
                self.dataframe.repartition_by_expressions(n, exprs),
            ));
        }
        if !(numPartitions.is_instance_of::<pyo3::types::PyString>()
            || numPartitions.extract::<PyRef<'_, PyColumn>>().is_ok())
        {
            return Err(not_expected_type(
                &numPartitions,
                "numPartitions",
                "Column or str",
            ));
        }
        let mut all = vec![numPartitions];
        all.extend(cols);
        let exprs: Vec<_> = to_column_list(all)?
            .iter()
            .map(|c| c.expression().clone())
            .collect();
        Ok(PyDataFrame::new(
            self.dataframe.repartition_by_expressions(0, exprs),
        ))
    }

    /// Coalesce.
    fn coalesce(&self, py: Python<'_>, numPartitions: i64) -> PyResult<PyDataFrame> {
        let num_partitions = positive_partitions(py, numPartitions)?;
        Ok(PyDataFrame::new(self.dataframe.coalesce(num_partitions)))
    }

    /// Convert to DataFrame with new column names.
    #[pyo3(signature = (*cols))]
    fn toDF(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        // As PySpark's, of strings.
        let mut names = Vec::with_capacity(cols.len());
        for col in &cols {
            if !col.is_instance_of::<pyo3::types::PyString>() {
                return Err(not_expected_type(col, "cols", "list[str]"));
            }
            names.push(col.extract::<String>()?);
        }
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        Ok(PyDataFrame::new(self.dataframe.to_df(name_refs)))
    }

    /// Alias this DataFrame.
    fn alias(&self, alias: &str) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.alias(alias))
    }

    /// Group by columns for aggregation.
    #[pyo3(signature = (*cols))]
    fn groupBy(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyGroupedData> {
        let columns = self.grouping_columns(py, cols, "cols")?;
        Ok(PyGroupedData::new(self.dataframe.group_by(columns)))
    }

    /// Rollup multi-dimensional aggregation.
    #[pyo3(signature = (*cols))]
    fn rollup(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyGroupedData> {
        Ok(PyGroupedData::new(
            self.dataframe
                .rollup(self.grouping_columns(py, cols, "cols")?),
        ))
    }

    /// Cube multi-dimensional aggregation.
    #[pyo3(signature = (*cols))]
    fn cube(&self, py: Python<'_>, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyGroupedData> {
        Ok(PyGroupedData::new(
            self.dataframe
                .cube(self.grouping_columns(py, cols, "cols")?),
        ))
    }

    /// Grouping sets aggregation: `groupingSets([[a, b], [a], []], "a", "b")`.
    ///
    /// As PySpark's: `*cols` are the grouping columns, the columns of the result besides the
    /// aggregations, of Columns, names and ordinals, as are those of the sets.
    #[pyo3(name = "groupingSets", signature = (groupingSets, *cols))]
    fn grouping_sets(
        &self,
        py: Python<'_>,
        groupingSets: &Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyGroupedData> {
        use pyo3::types::{PyInt, PyString};
        // As PySpark's `_to_cols(grouping_set)`: a set is a sequence of columns, or one column.
        let mut sets = Vec::new();
        for set in groupingSets.try_iter()? {
            let set = set?;
            let one = set.is_instance_of::<PyString>()
                || set.is_instance_of::<PyInt>()
                || set.is_instance_of::<PyColumn>();
            let sequence = !one
                && py
                    .import("collections.abc")?
                    .getattr("Sequence")
                    .and_then(|sequence| set.is_instance(&sequence))?;
            let items = if sequence {
                set.try_iter()?.collect::<PyResult<Vec<_>>>()?
            } else {
                vec![set]
            };
            sets.push(self.grouping_column_list(py, items, "groupingSets")?);
        }
        // As PySpark's `_to_cols(cols)`, of the columns themselves, not a list of them.
        let columns = self.grouping_column_list(py, cols, "cols")?;
        Ok(PyGroupedData::new(
            self.dataframe.grouping_sets_with_columns(sets, columns),
        ))
    }

    /// Drop duplicates within the event-time watermark.
    #[pyo3(name = "dropDuplicatesWithinWatermark", signature = (subset=None))]
    fn drop_duplicates_within_watermark(
        &self,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let subset = deduplicate_subset(subset)?;
        let refs: Option<Vec<&str>> = subset
            .as_ref()
            .map(|names| names.iter().map(|s| s.as_str()).collect());
        Ok(PyDataFrame::new(
            self.dataframe.drop_duplicates_within_watermark(refs),
        ))
    }

    /// `DataFrame.toJSON`, as PySpark Connect's: a DataFrame of each row as a JSON object
    /// string, in a column `value`.
    #[pyo3(name = "toJSON")]
    fn to_json(&self) -> PyDataFrame {
        PyDataFrame::new(self.dataframe.to_json_dataframe())
    }

    /// Whether this DataFrame is semantically equal to `other`.
    #[pyo3(name = "sameSemantics")]
    fn same_semantics(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        // As PySpark's: a DataFrame, else NOT_EXPECTED_TYPE, of the same session.
        if !other.is_instance_of::<PyDataFrame>() {
            return Err(not_expected_type(other, "other", "DataFrame"));
        }
        let other = self.same_session(other)?.dataframe.clone();
        py.detach(|| self.dataframe.same_semantics(&other))
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
        let exprs = self.sort_columns(_py, cols, kwargs.as_ref())?;
        Ok(PyDataFrame::new(
            self.dataframe.sort_within_partitions(exprs),
        ))
    }

    /// Except all (keeps duplicates).
    #[pyo3(name = "exceptAll")]
    fn except_all(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(
            self.dataframe.except_all(&other.dataframe),
        ))
    }

    /// Rename multiple columns from an {old: new} mapping.
    #[pyo3(name = "withColumnsRenamed")]
    fn with_columns_renamed(&self, colsMap: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        // As PySpark's, of a dict, whose renames it sends in order.
        let Ok(cols_map) = colsMap.cast::<PyDict>() else {
            return Err(not_expected_type(colsMap, "colsMap", "dict"));
        };
        let mut renames = Vec::with_capacity(cols_map.len());
        for (existing, new) in cols_map.iter() {
            renames.push((existing.extract::<String>()?, new.extract::<String>()?));
        }
        Ok(PyDataFrame::new(
            self.dataframe.with_columns_renamed(renames),
        ))
    }

    /// The columns of this DataFrame whose names match a regex, as a Column. Mirrors
    /// `DataFrame.colRegex`.
    #[pyo3(name = "colRegex")]
    fn col_regex(&self, colName: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        // As PySpark's: a str, else NOT_EXPECTED_TYPE.
        if !colName.is_instance_of::<pyo3::types::PyString>() {
            return Err(not_expected_type(colName, "colName", "str"));
        }
        Ok(PyColumn::new(
            self.dataframe.col_regex(&colName.extract::<String>()?),
        ))
    }

    /// Basic statistics for the given columns (count/mean/stddev/min/max), of column names or
    /// a list of them, as PySpark's `DataFrame.describe`.
    #[pyo3(signature = (*cols))]
    fn describe(&self, cols: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let cols = crate::stat_arguments::describe_columns(&cols)?;
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        Ok(PyDataFrame::new(self.dataframe.describe(refs)))
    }

    /// Summary statistics (percentiles etc.), of strings, as PySpark's `DataFrame.summary`.
    #[pyo3(signature = (*statistics))]
    fn summary(&self, statistics: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        let statistics = crate::stat_arguments::summary_statistics(&statistics)?;
        let refs: Vec<&str> = statistics.iter().map(|s| s.as_str()).collect();
        Ok(PyDataFrame::new(self.dataframe.summary(refs)))
    }

    /// Persist with the default MEMORY_AND_DISK storage level.
    fn cache<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<PyRef<'py, Self>> {
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        py.detach(|| df.cache().map_err(Box::new)).to_pyerr()?;
        Ok(slf)
    }

    /// Mark the DataFrame as non-persistent.
    #[pyo3(signature = (blocking=false))]
    fn unpersist<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        blocking: bool,
    ) -> PyResult<PyRef<'py, Self>> {
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        py.detach(|| df.unpersist(blocking).map_err(Box::new))
            .to_pyerr()?;
        Ok(slf)
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

    /// Drop rows containing nulls, as PySpark's `DataFrame.dropna`.
    #[pyo3(
        signature = (how=None, thresh=None, subset=None),
        text_signature = "($self, how='any', thresh=None, subset=None)"
    )]
    fn dropna(
        &self,
        how: Option<&Bound<'_, PyAny>>,
        thresh: Option<&Bound<'_, PyAny>>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::dropna(&self.dataframe, how, thresh, subset)
    }

    /// Fill null values, as PySpark's `DataFrame.fillna`: `value` is a scalar, over
    /// `subset`, or a {column: value} mapping.
    #[pyo3(signature = (value, subset=None))]
    fn fillna(
        &self,
        value: &Bound<'_, PyAny>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::fillna(&self.dataframe, value, subset)
    }

    /// Statistical functions (`df.stat`), of this DataFrame, their `df`, as PySpark's.
    #[getter]
    #[pyo3(name = "stat")]
    fn stat(slf: &Bound<'_, Self>) -> crate::stat::PyStatFunctions {
        crate::stat::PyStatFunctions::new(slf.borrow().dataframe.stat(), slf.clone().unbind())
    }

    /// `df.crosstab(col1, col2)` - pyspark exposes it on DataFrame as well as
    /// `df.stat.crosstab`.
    fn crosstab(&self, col1: &Bound<'_, PyAny>, col2: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        crate::stat_arguments::crosstab(&self.dataframe.stat(), col1, col2)
    }

    /// `df.freqItems(cols, support=None)` (also on df.stat).
    #[pyo3(name = "freqItems", signature = (cols, support=None))]
    fn freq_items(
        &self,
        cols: &Bound<'_, PyAny>,
        support: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::freq_items(&self.dataframe.stat(), cols, support)
    }

    /// `df.approxQuantile(col, probabilities, relativeError)` (also on df.stat): the
    /// quantiles of a column, or a list of them of each of a list or tuple of columns.
    #[pyo3(name = "approxQuantile")]
    fn approx_quantile(
        &self,
        py: Python<'_>,
        col: &Bound<'_, PyAny>,
        probabilities: &Bound<'_, PyAny>,
        relativeError: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        crate::stat_arguments::approx_quantile(
            py,
            &self.dataframe.stat(),
            col,
            probabilities,
            relativeError,
        )
    }

    /// Observe metrics, as PySpark's `DataFrame.observe(observation, *exprs)`: the Columns
    /// `exprs`, which an Observation receives, or which are observed by a name.
    #[pyo3(signature = (observation, *exprs))]
    fn observe(
        &self,
        py: Python<'_>,
        observation: &Bound<'_, PyAny>,
        exprs: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        if exprs.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "exprs")],
            ));
        }
        let mut columns = Vec::with_capacity(exprs.len());
        for expr in &exprs {
            match expr.extract::<PyRef<'_, PyColumn>>() {
                Ok(column) => columns.push(column.column.expression().clone()),
                Err(_) => {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "PySparkTypeError",
                        "NOT_EXPECTED_TYPE",
                        &[
                            ("arg_name", "exprs"),
                            ("expected_type", "list[Column]"),
                            ("arg_type", "tuple"),
                        ],
                    ))
                }
            }
        }
        if let Ok(observation) =
            observation.extract::<PyRef<'_, crate::observation::PyObservation>>()
        {
            let observation = &observation.inner;
            return py
                .detach(|| self.dataframe.observe_with(observation, columns))
                .to_pyerr()
                .map(PyDataFrame::new);
        }
        if observation.is_instance_of::<pyo3::types::PyString>() {
            let name: String = observation.extract()?;
            return Ok(PyDataFrame::new(self.dataframe.observe(&name, columns)));
        }
        let type_name = observation.get_type().name()?.to_string();
        Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("arg_name", "observation"),
                ("expected_type", "Observation or str"),
                ("arg_type", &type_name),
            ],
        ))
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

    /// Reliable checkpoint (materialize + truncate lineage), at once if `eager`, as PySpark's.
    #[pyo3(signature = (eager=true))]
    fn checkpoint(&self, py: Python<'_>, eager: bool) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            py.detach(|| self.dataframe.checkpoint_with(eager).map_err(Box::new))
                .to_pyerr()?,
        ))
    }

    /// Local checkpoint, at once if `eager`, kept at `storageLevel`, else at the server's
    /// default, as PySpark's.
    #[pyo3(name = "localCheckpoint", signature = (eager=true, storageLevel=None))]
    fn local_checkpoint(
        &self,
        py: Python<'_>,
        eager: bool,
        storageLevel: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let level = storageLevel
            .filter(|level| !level.is_none())
            .map(|level| crate::storagelevel::storage_level_proto(&level))
            .transpose()?;
        Ok(PyDataFrame::new(
            py.detach(|| {
                self.dataframe
                    .local_checkpoint_with(eager, level)
                    .map_err(Box::new)
            })
            .to_pyerr()?,
        ))
    }

    /// Pearson correlation between two columns; PySpark's `method` is only "pearson".
    #[pyo3(signature = (col1, col2, method=None))]
    fn corr(
        &self,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
        method: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<f64> {
        crate::stat_arguments::corr(py, &self.dataframe.stat(), col1, col2, method)
    }

    /// Sample covariance between two columns.
    fn cov(
        &self,
        py: Python<'_>,
        col1: &Bound<'_, PyAny>,
        col2: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        crate::stat_arguments::cov(py, &self.dataframe.stat(), col1, col2)
    }

    /// Stratified sample without replacement per key.
    #[pyo3(name = "sampleBy", signature = (col, fractions, seed=None))]
    fn sample_by(
        &self,
        col: &Bound<'_, PyAny>,
        fractions: &Bound<'_, PyAny>,
        seed: Option<i64>,
    ) -> PyResult<PyDataFrame> {
        let (col, fractions) = crate::stat::sample_by_arguments(col, fractions)?;
        Ok(PyDataFrame::new(
            self.dataframe.stat().sample_by(col, fractions, seed),
        ))
    }

    /// Replace values, as PySpark's `DataFrame.replace`: `to_replace` is a {old: new} dict, or
    /// a value or a list or tuple of them with `value`, a value or a list or tuple of them.
    #[pyo3(signature = (to_replace, value=ReplaceValue::Omitted, subset=None))]
    fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: ReplaceValue,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        replace_values(&self.dataframe, to_replace, value, subset)
    }

    /// Reconcile to a target schema (StructType or DDL string).
    fn to(&self, schema: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let dt = crate::types::py_to_data_type(schema)?;
        Ok(PyDataFrame::new(self.dataframe.to(dt)))
    }

    /// Set the metadata of a column, as PySpark's: a dict of any JSON values, which it sends as
    /// `json.dumps(metadata)`, and `NOT_EXPECTED_TYPE` for one that is not a dict.
    #[pyo3(name = "withMetadata")]
    fn with_metadata(
        &self,
        py: Python<'_>,
        columnName: &str,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        if !metadata.is_instance_of::<PyDict>() {
            return Err(crate::stat_arguments::not_expected_type(
                metadata, "metadata", "dict",
            ));
        }
        let json: String = py
            .import("json")?
            .call_method1("dumps", (metadata,))?
            .extract()?;
        Ok(PyDataFrame::new(
            self.dataframe.with_metadata_json(columnName, &json),
        ))
    }

    /// Define an event-time watermark, as PySpark's: `NOT_EXPECTED_TYPE` for an empty event time
    /// or delay, or one that is not a str.
    #[pyo3(name = "withWatermark")]
    fn with_watermark(
        &self,
        eventTime: &Bound<'_, PyAny>,
        delayThreshold: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let event_time = watermark_argument(eventTime, "eventTime")?;
        let delay_threshold = watermark_argument(delayThreshold, "delayThreshold")?;
        Ok(PyDataFrame::new(
            self.dataframe.with_watermark(&event_time, &delay_threshold),
        ))
    }

    /// Randomly split into multiple DataFrames by the given weights.
    /// As PySpark's: `VALUE_NOT_POSITIVE` for a negative weight and for weights of a sum that is
    /// not positive, of their `str`; without a seed, the client chooses one of every split.
    #[pyo3(name = "randomSplit", signature = (weights, seed=None))]
    fn random_split(
        &self,
        weights: &Bound<'_, PyAny>,
        seed: Option<i64>,
    ) -> PyResult<Vec<PyDataFrame>> {
        let py = weights.py();
        let not_positive = |arg_name: &str, value: &Bound<'_, PyAny>| -> PyResult<PyErr> {
            Ok(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "VALUE_NOT_POSITIVE",
                &[
                    ("arg_name", arg_name),
                    ("arg_value", &value.str()?.to_cow()?),
                ],
            ))
        };
        for weight in weights.try_iter()? {
            let weight = weight?;
            if weight.lt(0.0)? {
                return Err(not_positive("weights", &weight)?);
            }
        }
        let total = py.import("builtins")?.call_method1("sum", (weights,))?;
        if total.le(0)? {
            return Err(not_positive("sum(weights)", &total)?);
        }
        let weights: Vec<f64> = weights.extract()?;
        Ok(self
            .dataframe
            .random_split(weights, seed)
            .into_iter()
            .map(PyDataFrame::new)
            .collect())
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
    fn is_streaming(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.dataframe.is_streaming().map_err(Box::new))
            .to_pyerr()
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

    /// Not supported on Spark Connect, as PySpark Connect's: `NOT_IMPLEMENTED` of `rdd`, which
    /// needs the RDD API of a classic session.
    #[getter]
    fn rdd(&self, py: Python<'_>) -> PyResult<()> {
        Err(crate::column::raise_pyspark(
            py,
            "PySparkNotImplementedError",
            "NOT_IMPLEMENTED",
            &[("feature", "rdd")],
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
    fn drop_duplicates(&self, subset: Option<&Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        self.dropDuplicates(subset)
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

    /// Unpivot (wide-to-long), as PySpark's `DataFrame.unpivot`: `ids` and `values` are each a
    /// column or a column name, or a list or tuple of them, and `values=None` unpivots all
    /// non-id columns.
    #[pyo3(signature = (ids, values, variableColumnName, valueColumnName))]
    fn unpivot(
        &self,
        ids: &Bound<'_, PyAny>,
        values: Option<&Bound<'_, PyAny>>,
        variableColumnName: &str,
        valueColumnName: &str,
    ) -> PyResult<PyDataFrame> {
        if ids.is_none() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "ids must not be None",
            ));
        }
        let ids = unpivot_columns(ids)?;
        let values = values.map(unpivot_columns).transpose()?;
        Ok(PyDataFrame::new(self.dataframe.unpivot(
            ids,
            values,
            variableColumnName,
            valueColumnName,
        )))
    }

    /// The same function as `unpivot`, as PySpark's `DataFrame.melt`.
    #[pyo3(signature = (ids, values, variableColumnName, valueColumnName))]
    fn melt(
        &self,
        ids: &Bound<'_, PyAny>,
        values: Option<&Bound<'_, PyAny>>,
        variableColumnName: &str,
        valueColumnName: &str,
    ) -> PyResult<PyDataFrame> {
        self.unpivot(ids, values, variableColumnName, valueColumnName)
    }

    /// Persist the DataFrame at `storageLevel`, MEMORY_AND_DISK_DESER by default, as
    /// PySpark's `DataFrame.persist()`.
    #[pyo3(signature = (storageLevel=None))]
    fn persist<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        storageLevel: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyRef<'py, Self>> {
        use spark_connect::storage::StorageLevelExt;
        let level = match storageLevel {
            Some(level) => crate::storagelevel::storage_level_proto(&level)?,
            None => spark_connect_proto::StorageLevel::memory_and_disk_deser(),
        };
        // As PySpark's, which returns the DataFrame itself.
        let df = slf.dataframe.clone();
        py.detach(|| df.persist(level).map_err(Box::new))
            .to_pyerr()?;
        Ok(slf)
    }

    /// Column access by attribute (`df.colname` -> Column), mirroring reference pyspark: the
    /// column of this DataFrame, by its plan ID. Names beginning with `_` raise
    /// AttributeError so normal attribute lookup works.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<PyColumn> {
        // As PySpark Connect's: the attributes of the JVM of a classic DataFrame are not
        // supported, and a name is validated as a column only for a dunder name, which Python
        // may look up for an attribute, or with PYSPARK_VALIDATE_COLUMN_NAME_LEGACY.
        if ["_jseq", "_jdf", "_jmap", "_jcols", "rdd"].contains(&name) {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkAttributeError",
                "JVM_ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", name)],
            ));
        }
        let legacy = std::env::var("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY").as_deref() == Ok("1");
        if (legacy || name.starts_with("__")) && !self.columns(py)?.iter().any(|c| c == name) {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkAttributeError",
                "ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", name)],
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
    fn agg(&self, py: Python<'_>, exprs: Vec<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        if exprs.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "exprs")],
            ));
        }
        // As PySpark's: a dict of column names and functions, the aggregation of each, as of
        // `groupBy().agg`, else Columns, which it asserts.
        if let [only] = exprs.as_slice() {
            if let Ok(dict) = only.cast::<PyDict>() {
                // As PySpark's, whose `groupBy().agg()` of none of the entries asserts that
                // it has expressions.
                if dict.is_empty() {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "exprs should not be empty",
                    ));
                }
                let expressions = crate::group::dict_aggregates(&dict)?;
                return Ok(PyDataFrame::new(self.dataframe.agg(expressions)));
            }
        }
        let mut expressions = Vec::with_capacity(exprs.len());
        for expr in &exprs {
            let Ok(column) = expr.extract::<PyRef<'_, PyColumn>>() else {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "all exprs should be Expression",
                ));
            };
            expressions.push(column.column.expression().clone());
        }
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

    /// Sample a fraction of rows, as PySpark's `DataFrame.sample(withReplacement=None,
    /// fraction=None, seed=None)`, which takes `sample(fraction[, seed])` positionally as
    /// well: a bool and a float, a float, or a float and the seed.
    #[pyo3(signature = (withReplacement=None, fraction=None, seed=None))]
    fn sample(
        &self,
        py: Python<'_>,
        withReplacement: Option<&Bound<'_, PyAny>>,
        fraction: Option<&Bound<'_, PyAny>>,
        seed: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        use pyo3::types::{PyBool, PyFloat};
        let int = |value: &Bound<'_, PyAny>| -> PyResult<i64> {
            py.import("builtins")?
                .getattr("int")?
                .call1((value,))?
                .extract()
        };
        let (replace, frac, seed) = match (withReplacement, fraction) {
            (Some(w), Some(f)) if w.is_instance_of::<PyBool>() && f.is_instance_of::<PyFloat>() => {
                (w.extract()?, f.extract()?, seed.map(int).transpose()?)
            }
            (None, Some(f)) if f.is_instance_of::<PyFloat>() => {
                (false, f.extract()?, seed.map(int).transpose()?)
            }
            // `sample(fraction[, seed])`: the fraction is the first argument, and the seed the
            // second.
            (Some(w), f) if w.is_instance_of::<PyFloat>() => {
                (false, w.extract()?, f.map(int).transpose()?)
            }
            _ => {
                let type_name = |value: Option<&Bound<'_, PyAny>>| -> PyResult<String> {
                    Ok(match value {
                        Some(value) => value.get_type().name()?.to_string(),
                        None => "NoneType".to_string(),
                    })
                };
                let arg_types = [withReplacement, fraction, seed]
                    .into_iter()
                    .map(type_name)
                    .collect::<PyResult<Vec<_>>>()?
                    .join(", ");
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "NOT_EXPECTED_TYPE",
                    &[
                        ("expected_type", "bool, float or int"),
                        (
                            "arg_name",
                            "withReplacement (optional), fraction (required) and seed (optional)",
                        ),
                        ("arg_type", &arg_types),
                    ],
                ));
            }
        };
        Ok(PyDataFrame::new(
            self.dataframe.sample_opt(frac, replace, seed),
        ))
    }

    /// Check if the DataFrame is empty.
    fn isEmpty(&self, py: Python<'_>) -> PyResult<bool> {
        detached(py, || self.dataframe.is_empty().map_err(Box::new))
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
    fn unionAll(&self, other: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(self.dataframe.union(&other.dataframe)))
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

    /// Null-handling functions (`df.na.drop()/fill()/replace()`), of this DataFrame, their
    /// `df`, as PySpark's.
    #[getter]
    fn na(slf: &Bound<'_, Self>) -> PyDataFrameNaFunctions {
        PyDataFrameNaFunctions {
            df: slf.borrow().dataframe.clone(),
            frame: slf.clone().unbind(),
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
        // As PySpark's `F.col(self.columns[item])`, of the origin of `col`.
        let _origin = crate::origin::with_origin("col");
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

    /// `DataFrame.mapInPandas` — map an iterator of pandas DataFrames, in a stage of the
    /// resource profile `profile`, if any, as PySpark's.
    #[pyo3(name = "mapInPandas", signature = (func, schema, barrier = false, profile = None))]
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
            &self.dataframe.spark_session(),
            &func,
            &schema,
            eval_type::SQL_MAP_PANDAS_ITER_UDF,
            &cols,
        )?;
        let profile_id = profile_id_of(profile.as_ref())?;
        Ok(PyDataFrame::new(
            self.dataframe
                .map_in_pandas_with_profile(udf, barrier, profile_id),
        ))
    }

    /// `DataFrame.mapInArrow` — map an iterator of Arrow batches, in a stage of the
    /// resource profile `profile`, if any, as PySpark's.
    #[pyo3(name = "mapInArrow", signature = (func, schema, barrier = false, profile = None))]
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
            &self.dataframe.spark_session(),
            &func,
            &schema,
            eval_type::SQL_MAP_ARROW_ITER_UDF,
            &cols,
        )?;
        let profile_id = profile_id_of(profile.as_ref())?;
        Ok(PyDataFrame::new(
            self.dataframe
                .map_in_arrow_with_profile(udf, barrier, profile_id),
        ))
    }

    /// `DataFrame.foreach`, as PySpark Connect's: a UDF over a struct of the columns of each
    /// row calls `f` with the row. The UDF is Python, as it runs on the server
    /// (`pyspark.sql._foreach`).
    fn foreach(slf: &Bound<'_, Self>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py()
            .import("pyspark.sql._foreach")?
            .call_method1("foreach", (slf, f))?;
        Ok(())
    }

    /// `DataFrame.foreachPartition`, as PySpark Connect's: a `mapInArrow` function calls `f`
    /// with an iterator of the rows of each partition.
    #[pyo3(name = "foreachPartition")]
    fn foreach_partition(slf: &Bound<'_, Self>, f: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py()
            .import("pyspark.sql._foreach")?
            .call_method1("foreachPartition", (slf, f))?;
        Ok(())
    }

    /// `DataFrame.nearestByJoin`.
    #[pyo3(name = "nearestByJoin", signature = (other, rankingExpression, numResults = 1, mode = "brute", direction = "asc", joinType = "inner"))]
    fn nearest_by_join(
        &self,
        other: &Bound<'_, PyAny>,
        rankingExpression: &PyColumn,
        numResults: i32,
        mode: &str,
        direction: &str,
        joinType: &str,
    ) -> PyResult<PyDataFrame> {
        let other = self.same_session(other)?;
        Ok(PyDataFrame::new(self.dataframe.nearest_by_join(
            &other.dataframe,
            rankingExpression.column.clone(),
            numResults,
            mode,
            direction,
            joinType,
        )))
    }

    /// `DataFrame.metadataColumn`.
    #[pyo3(name = "metadataColumn")]
    fn metadata_column(&self, colName: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
        // As PySpark's: a str, else NOT_EXPECTED_TYPE.
        if !colName.is_instance_of::<pyo3::types::PyString>() {
            return Err(not_expected_type(colName, "colName", "str"));
        }
        Ok(PyColumn::new(
            self.dataframe
                .metadata_column(&colName.extract::<String>()?),
        ))
    }
}

/// Python wrapper for `DataFrameNaFunctions` (`df.na`).
#[pyclass(name = "DataFrameNaFunctions", module = "pyspark.sql.dataframe")]
pub struct PyDataFrameNaFunctions {
    df: spark_connect::dataframe::DataFrame,
    /// The DataFrame of `df.na`, as PySpark's `DataFrameNaFunctions.df`.
    frame: Py<PyDataFrame>,
}

#[pymethods]
impl PyDataFrameNaFunctions {
    /// The DataFrame of these functions, as PySpark's `DataFrameNaFunctions.df`: `df.na.df` is
    /// `df`.
    #[getter(df)]
    fn frame(&self, py: Python<'_>) -> Py<PyDataFrame> {
        self.frame.clone_ref(py)
    }

    /// Drop rows containing nulls, as PySpark's `DataFrameNaFunctions.drop`, which is
    /// `DataFrame.dropna`.
    #[pyo3(
        signature = (how=None, thresh=None, subset=None),
        text_signature = "($self, how='any', thresh=None, subset=None)"
    )]
    fn drop(
        &self,
        how: Option<&Bound<'_, PyAny>>,
        thresh: Option<&Bound<'_, PyAny>>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::dropna(&self.df, how, thresh, subset)
    }

    /// Fill nulls, as PySpark's `DataFrameNaFunctions.fill`, which is `DataFrame.fillna`.
    #[pyo3(signature = (value, subset=None))]
    fn fill(
        &self,
        value: &Bound<'_, PyAny>,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        crate::stat_arguments::fillna(&self.df, value, subset)
    }

    /// Replace values, as PySpark's `DataFrameNaFunctions.replace`, which is
    /// `DataFrame.replace`.
    #[pyo3(signature = (to_replace, value=ReplaceValue::Omitted, subset=None))]
    fn replace(
        &self,
        to_replace: &Bound<'_, PyAny>,
        value: ReplaceValue,
        subset: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        replace_values(&self.df, to_replace, value, subset)
    }
}

/// The `value` argument of `DataFrame.replace`, where PySpark tells an omitted value apart
/// from None: `replace(to_replace)` requires a dict `to_replace`, while
/// `replace(to_replace, None)` replaces with nulls.
pub enum ReplaceValue {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ReplaceValue {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(ReplaceValue::Given(obj.to_owned().unbind()))
    }
}

/// `df.replace(to_replace, value, subset)`, checked and built as PySpark's
/// `DataFrame.replace`: each value of `to_replace` (a {old: new} dict, or a value or a list or
/// tuple of them) is replaced by `value` (a value or a list or tuple of them, or None), all
/// bools, numbers or strings, sent as literals with an int as a float, in the columns of
/// `subset` (a column name, or a list or tuple of them).
fn replace_values(
    dataframe: &DataFrame,
    to_replace: &Bound<'_, PyAny>,
    value: ReplaceValue,
    subset: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyDataFrame> {
    use pyo3::types::{PyBool, PyFloat, PyString};
    use spark_connect::expression::LiteralExpression;

    let py = to_replace.py();
    let raise = |class_name: &str, error_class: &str, params: &[(&str, &str)]| {
        crate::column::raise_pyspark(py, class_name, error_class, params)
    };
    let type_name = |v: &Bound<'_, PyAny>| -> String {
        v.get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_default()
    };
    let is_sequence =
        |v: &Bound<'_, PyAny>| v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>();
    // A bool, a float, an int or a str: a bool is an int.
    let is_scalar = |v: &Bound<'_, PyAny>| {
        v.is_instance_of::<PyFloat>()
            || v.is_instance_of::<PyInt>()
            || v.is_instance_of::<PyString>()
    };
    let is_dict = to_replace.is_instance_of::<PyDict>();
    let value = match value {
        ReplaceValue::Given(value) => value.into_bound(py),
        ReplaceValue::Omitted if is_dict => py.None().into_bound(py),
        ReplaceValue::Omitted => {
            return Err(raise(
                "PySparkTypeError",
                "ARGUMENT_REQUIRED",
                &[("arg_name", "value"), ("condition", "`to_replace` is dict")],
            ))
        }
    };

    if !(is_scalar(to_replace) || is_sequence(to_replace) || is_dict) {
        return Err(raise(
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("expected_type", "bool, dict, float, int, str or tuple"),
                ("arg_name", "to_replace"),
                ("arg_type", &type_name(to_replace)),
            ],
        ));
    }
    if !(is_scalar(&value) || is_sequence(&value) || value.is_none() || is_dict) {
        return Err(raise(
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                (
                    "expected_type",
                    "bool, float, int, list, None, str or tuple",
                ),
                ("arg_name", "value"),
                ("arg_type", &type_name(&value)),
            ],
        ));
    }
    if is_sequence(to_replace) && is_sequence(&value) && to_replace.len()? != value.len()? {
        return Err(raise(
            "PySparkValueError",
            "LENGTH_SHOULD_BE_THE_SAME",
            &[
                ("arg1", "to_replace"),
                ("arg2", "value"),
                ("arg1_length", &to_replace.len()?.to_string()),
                ("arg2_length", &value.len()?.to_string()),
            ],
        ));
    }
    let subset: Option<Vec<String>> = match subset {
        None => None,
        Some(subset) if subset.is_instance_of::<PyString>() => Some(vec![subset.extract()?]),
        Some(subset) if is_sequence(subset) => Some(subset.extract()?),
        Some(subset) => {
            return Err(raise(
                "PySparkTypeError",
                "NOT_EXPECTED_TYPE",
                &[
                    ("arg_name", "subset"),
                    ("expected_type", "list, str or tuple"),
                    ("arg_type", &type_name(subset)),
                ],
            ))
        }
    };

    // The replacements as a dict, as PySpark builds it, so that equal values are one key.
    let replacements = if is_dict {
        if !value.is_none() {
            py.import("warnings")?.call_method1(
                "warn",
                ("to_replace is a dict and value is not None. value will be ignored.",),
            )?;
        }
        to_replace.cast::<PyDict>()?.clone()
    } else {
        let olds: Vec<Bound<'_, PyAny>> = if is_scalar(to_replace) {
            vec![to_replace.clone()]
        } else {
            to_replace.try_iter()?.collect::<PyResult<_>>()?
        };
        let news: Vec<Bound<'_, PyAny>> = if is_scalar(&value) || value.is_none() {
            vec![value.clone(); olds.len()]
        } else {
            value.try_iter()?.collect::<PyResult<_>>()?
        };
        let replacements = PyDict::new(py);
        for (old, new) in olds.iter().zip(news.iter()) {
            replacements.set_item(old, new)?;
        }
        replacements
    };

    // All of the values to replace, and all of their replacements but None, are bools,
    // numbers or strings.
    let olds: Vec<Bound<'_, PyAny>> = replacements.keys().iter().collect();
    let news: Vec<Bound<'_, PyAny>> = replacements
        .values()
        .iter()
        .filter(|new| !new.is_none())
        .collect();
    let all_of = |is_type: &dyn Fn(&Bound<'_, PyAny>) -> bool| {
        olds.iter().all(is_type) && news.iter().all(is_type)
    };
    if !(all_of(&|v| v.is_instance_of::<PyBool>())
        || all_of(&|v| v.is_instance_of::<PyString>())
        || all_of(&|v| v.is_instance_of::<PyFloat>() || v.is_instance_of::<PyInt>()))
    {
        return Err(raise("PySparkValueError", "MIXED_TYPE_REPLACEMENT", &[]));
    }

    let literal = |v: &Bound<'_, PyAny>| -> PyResult<LiteralExpression> {
        Ok(if v.is_none() {
            LiteralExpression::null(spark_connect::types::DataType::Null)
        } else if v.is_instance_of::<PyBool>() {
            LiteralExpression::boolean(v.extract()?)
        } else if v.is_instance_of::<PyString>() {
            LiteralExpression::string(v.extract::<String>()?)
        } else {
            // As `float(v)`, for an int as well.
            LiteralExpression::double(v.extract()?)
        })
    };
    let replacements = replacements
        .iter()
        .map(|(old, new)| Ok((literal(&old)?, literal(&new)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let subset: Option<Vec<&str>> = subset
        .as_ref()
        .map(|subset| subset.iter().map(String::as_str).collect());
    Ok(PyDataFrame::new(
        dataframe.replace_literals(replacements, subset),
    ))
}

/// The columns of the `ids` or `values` of `df.unpivot`, as PySpark's: a list or tuple of
/// columns or column names, or one of them.
fn unpivot_columns(cols: &Bound<'_, PyAny>) -> PyResult<Vec<spark_connect::column::Column>> {
    if cols.is_instance_of::<PyList>() || cols.is_instance_of::<PyTuple>() {
        cols.try_iter()?.map(|col| column_or_name(&col?)).collect()
    } else {
        Ok(vec![column_or_name(cols)?])
    }
}

/// A column, or a column by its name, as PySpark's `_to_col`.
pub(crate) fn column_or_name(col: &Bound<'_, PyAny>) -> PyResult<spark_connect::column::Column> {
    if let Ok(column) = col.extract::<PyRef<'_, PyColumn>>() {
        return Ok(column.column.clone());
    }
    if col.is_instance_of::<pyo3::types::PyString>() {
        // As PySpark's `F._to_col`, by `col`, of its origin.
        let _origin = crate::origin::with_origin("col");
        return Ok(spark_connect::functions::col(&col.extract::<String>()?));
    }
    let type_name = col
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    Err(crate::column::raise_pyspark(
        col.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("expected_type", "Column or str"),
            ("arg_name", "col"),
            ("arg_type", &type_name),
        ],
    ))
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

    fn merge(&self, py: Python<'_>) -> PyResult<()> {
        let writer = self.inner.clone();
        detached(py, || writer.merge().map_err(Box::new))
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
    /// The save mode, as PySpark's: `None` keeps the default, "error".
    fn mode<'py>(
        mut slf: PyRefMut<'py, Self>,
        saveMode: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if let Some(save_mode) = saveMode {
            slf.inner = slf.inner.clone().mode(save_mode);
        }
        Ok(slf)
    }

    /// The format of the write. As PySpark's, None unsets it, so that the server writes its
    /// default source.
    fn format<'py>(
        mut slf: PyRefMut<'py, Self>,
        source: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = match source {
            Some(source) => slf.inner.clone().format(source),
            None => slf.inner.clone().clear_format(),
        };
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
        numBuckets: &Bound<'_, PyAny>,
        col: &Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let num_buckets = crate::readwriter::bucket_count(numBuckets)?;
        let columns = crate::readwriter::writer_columns(col, &cols)?;
        slf.inner = slf.inner.clone().bucket_by(num_buckets, columns);
        Ok(slf)
    }

    #[pyo3(name = "sortBy", signature = (col, *cols))]
    fn sort_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        col: &Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let columns = crate::readwriter::writer_columns(col, &cols)?;
        slf.inner = slf.inner.clone().sort_by(columns);
        Ok(slf)
    }

    /// The clustering columns, as PySpark's: column names, or a list or tuple of them, at
    /// least one.
    #[pyo3(name = "clusterBy", signature = (*cols))]
    fn cluster_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let cols: Vec<String> = match cols.as_slice() {
            [only] if only.is_instance_of::<PyList>() || only.is_instance_of::<PyTuple>() => {
                only.extract()?
            }
            _ => cols.iter().map(|c| c.extract()).collect::<PyResult<_>>()?,
        };
        if cols.is_empty() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "clusterBy needs one or more clustering columns.",
            ));
        }
        slf.inner = slf.inner.clone().cluster_by(cols);
        Ok(slf)
    }

    /// Write to an external JDBC table. Connection details go through `properties`, options of
    /// any type (None skipped, bools lowercased), as PySpark's.
    #[pyo3(signature = (url, table, mode=None, properties=None))]
    fn jdbc(
        &mut self,
        py: Python<'_>,
        url: &str,
        table: &str,
        mode: Option<&str>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        // Same order as PySpark: format, mode, properties, then url/dbtable (which
        // win over same-named properties), all kept on the writer, then save.
        let mut w = self.inner.clone().format("jdbc");
        if let Some(m) = mode {
            w = w.mode(m);
        }
        if let Some(props) = properties {
            for (key, value) in props.iter() {
                if let Some(value) = crate::coerce_option_value(&value)? {
                    w = w.option(&key.str()?.to_cow()?, &value);
                }
            }
        }
        w = w.option("url", url).option("dbtable", table);
        self.inner = w.clone();
        detached(py, || w.save(None).map_err(Box::new))
    }

    #[pyo3(signature = (path=None, format=None, mode=None, partitionBy=None, **options))]
    #[allow(non_snake_case)]
    fn save(
        &mut self,
        py: Python<'_>,
        path: Option<String>,
        format: Option<&str>,
        mode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let w = apply_write_common(self.inner.clone(), format, mode, partitionBy, options)?;
        // PySpark sets the inline format/mode/partitionBy/options on the writer itself.
        self.inner = w.clone();
        detached(py, || w.save(path.as_deref()).map_err(Box::new))
    }

    #[pyo3(name = "saveAsTable", signature = (name, format=None, mode=None, partitionBy=None, **options))]
    #[allow(non_snake_case)]
    fn save_as_table(
        &mut self,
        py: Python<'_>,
        name: &str,
        format: Option<&str>,
        mode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let w = apply_write_common(self.inner.clone(), format, mode, partitionBy, options)?;
        self.inner = w.clone();
        detached(py, || w.save_as_table(name).map_err(Box::new))
    }

    /// Insert into an existing table. `overwrite` maps to overwrite/append mode,
    /// mirroring reference `DataFrameWriter.insertInto(tableName, overwrite=None)`.
    #[pyo3(name = "insertInto", signature = (tableName, overwrite=None))]
    fn insert_into(
        &mut self,
        py: Python<'_>,
        tableName: &str,
        overwrite: Option<bool>,
    ) -> PyResult<()> {
        let mut w = self.inner.clone();
        if let Some(ow) = overwrite {
            w = w.mode(if ow { "overwrite" } else { "append" });
        }
        // PySpark sets the overwrite/append mode on the writer itself.
        self.inner = w.clone();
        detached(py, || w.insert_into(tableName).map_err(Box::new))
    }

    /// Write as PARQUET - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, partitionBy=None, compression=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn parquet(
        &mut self,
        py: Python<'_>,
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
        detached(py, || w.parquet(path).map_err(Box::new))
    }

    /// Write as JSON - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, compression=None, dateFormat=None, timestampFormat=None, lineSep=None, encoding=None, ignoreNullFields=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn json(
        &mut self,
        py: Python<'_>,
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
        detached(py, || w.json(path).map_err(Box::new))
    }

    /// Write as CSV - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, compression=None, sep=None, quote=None, escape=None, header=None, nullValue=None, escapeQuotes=None, quoteAll=None, dateFormat=None, timestampFormat=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, charToEscapeQuoteEscaping=None, encoding=None, emptyValue=None, lineSep=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn csv(
        &mut self,
        py: Python<'_>,
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
        detached(py, || w.csv(path).map_err(Box::new))
    }

    /// Write as ORC - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, mode=None, partitionBy=None, compression=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn orc(
        &mut self,
        py: Python<'_>,
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
        detached(py, || w.orc(path).map_err(Box::new))
    }

    /// Write as TEXT - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, compression=None, lineSep=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn text(
        &mut self,
        py: Python<'_>,
        path: &str,
        compression: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let mut w = self.inner.clone();
        w = wset_opt(w, "compression", compression)?;
        w = wset_opt(w, "lineSep", lineSep)?;
        // PySpark sets the mode/options/format on the writer itself, then saves.
        self.inner = w.clone().format("text");
        detached(py, || w.text(path).map_err(Box::new))
    }

    /// Write as XML - full pyspark signature; each named option is
    /// applied when provided.
    #[pyo3(signature = (path, rowTag=None, mode=None, attributePrefix=None, valueTag=None, rootTag=None, declaration=None, arrayElementName=None, nullValue=None, dateFormat=None, timestampFormat=None, compression=None, encoding=None, validateName=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn xml(
        &mut self,
        py: Python<'_>,
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
        detached(py, || w.xml(path).map_err(Box::new))
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

    /// Set the table property `property` to `value`, or leave it unset for None, as PySpark's.
    #[pyo3(name = "tableProperty")]
    fn table_property<'py>(
        mut slf: PyRefMut<'py, Self>,
        property: &str,
        value: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = match value {
            Some(value) => slf.inner.clone().table_property(property, value),
            None => slf.inner.clone().unset_table_property(property),
        };
        Ok(slf)
    }

    /// Partition by the columns, each a Column or a column name, as PySpark's `F._to_col` of
    /// each, else `NOT_EXPECTED_TYPE` of `col`, also for a list.
    #[pyo3(name = "partitionedBy", signature = (col, *cols))]
    fn partitioned_by<'py>(
        mut slf: PyRefMut<'py, Self>,
        col: Bound<'_, PyAny>,
        cols: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let columns = std::iter::once(&col)
            .chain(&cols)
            .map(column_or_name)
            .collect::<PyResult<Vec<_>>>()?;
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

    fn create(&mut self, py: Python<'_>) -> PyResult<()> {
        let writer = self.inner.clone();
        detached(py, || writer.create().map_err(Box::new))
    }

    fn replace(&mut self, py: Python<'_>) -> PyResult<()> {
        let writer = self.inner.clone();
        detached(py, || writer.replace().map_err(Box::new))
    }

    #[pyo3(name = "createOrReplace")]
    fn create_or_replace(&mut self, py: Python<'_>) -> PyResult<()> {
        let writer = self.inner.clone();
        detached(py, || writer.create_or_replace().map_err(Box::new))
    }

    fn append(&mut self, py: Python<'_>) -> PyResult<()> {
        let writer = self.inner.clone();
        detached(py, || writer.append().map_err(Box::new))
    }

    /// Overwrite the rows of `condition`, a Column or a column name, as PySpark's `F._to_col`,
    /// else `NOT_EXPECTED_TYPE` of `col`.
    fn overwrite(&mut self, py: Python<'_>, condition: &Bound<'_, PyAny>) -> PyResult<()> {
        let (writer, condition) = (self.inner.clone(), column_or_name(condition)?);
        detached(py, || writer.overwrite(condition).map_err(Box::new))
    }

    #[pyo3(name = "overwritePartitions")]
    fn overwrite_partitions(&mut self, py: Python<'_>) -> PyResult<()> {
        let writer = self.inner.clone();
        detached(py, || writer.overwrite_partitions().map_err(Box::new))
    }
}

/// `f` without the GIL, as PySpark's requests to the server release it while they wait, so
/// that other Python threads run. `f` boxes its error, which keeps its result small.
pub(crate) fn detached<T: Send>(
    py: Python<'_>,
    f: impl FnOnce() -> Result<T, Box<spark_connect_core::error::SparkError>> + Send,
) -> PyResult<T> {
    py.detach(f).to_pyerr()
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

/// Resolve a Python schema argument (a `DataType` wrapper or a DDL string) into a Rust
/// DataType: a DDL string as the server of `session` parses it, as PySpark's `_parse_ddl`.
pub(crate) fn resolve_datatype(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    schema: &Bound<'_, PyAny>,
) -> PyResult<spark_connect::types::DataType> {
    if let Ok(dt) = schema.extract::<PyRef<PyDataType>>() {
        return Ok(dt.inner.clone());
    }
    if schema.is_instance_of::<pyo3::types::PyString>() {
        return crate::types::schema_data_type(py, session, schema);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "schema must be a DataType or a DDL string",
    ))
}

/// The id of the resource profile `profile` of `mapInPandas` and `mapInArrow`, as PySpark's
/// `MapPartitions` reads `profile.id`, which creates the profile on the server.
fn profile_id_of(profile: Option<&Bound<'_, PyAny>>) -> PyResult<Option<i32>> {
    match profile {
        Some(profile) if !profile.is_none() => Ok(Some(profile.getattr("id")?.extract()?)),
        _ => Ok(None),
    }
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

/// The name of the UDF of `func`, as PySpark's `UserDefinedFunction` names it: the
/// function's `__name__`, or the name of its class for a callable object without one.
fn udf_name(func: &Bound<'_, PyAny>) -> PyResult<String> {
    if func.hasattr("__name__")? {
        func.getattr("__name__")?.extract()
    } else {
        Ok(func.get_type().name()?.to_string())
    }
}

/// Build a map/group UDF expression: cloudpickle `(func, schema)` as the command and
/// carry the resolved schema as the output type (mirrors the scalar UDF command layout).
pub(crate) fn build_map_udf(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    func: &Bound<'_, PyAny>,
    schema: &Bound<'_, PyAny>,
    eval_type_id: i32,
    input_cols: &[String],
) -> PyResult<CommonInlineUserDefinedFunctionExpression> {
    let name = udf_name(func)?;
    let return_type = resolve_datatype(py, session, schema)?;
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
        name, true, arguments, payload,
    ))
}

/// The converters of the fields of a result of `schema` that deserialize its values of
/// user-defined types, as PySpark does when it collects them
/// (`pyspark.sql.types._udt_field_converters`); None without a user-defined type.
fn udt_field_converters(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<Option<FieldConverters>> {
    let Some(schema) = schema.filter(|schema| crate::types::has_converted_values(schema)) else {
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

/// Whether `data_type` holds a year-month interval, also nested in an array, a map, a struct or
/// a user-defined type (`pyspark.sql.types._has_type`).
fn has_year_month_interval(data_type: &spark_connect::types::DataType) -> bool {
    use spark_connect::types::DataType;
    match data_type {
        DataType::YearMonthInterval { .. } => true,
        DataType::Struct { fields } => fields
            .iter()
            .any(|field| has_year_month_interval(&field.data_type)),
        DataType::Array { element_type, .. } => has_year_month_interval(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_year_month_interval(key_type) || has_year_month_interval(value_type),
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } => has_year_month_interval(sql_type),
        _ => false,
    }
}

/// Raises as PySpark's Arrow conversion of the rows does for a result of `schema` with a
/// year-month interval, which Spark Connect cannot collect: the error is raised even when the
/// result has no rows.
fn check_collectable(
    py: Python<'_>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<()> {
    if schema.is_some_and(has_year_month_interval) {
        return Err(crate::column::raise_pyspark(
            py,
            "PySparkNotImplementedError",
            "NOT_IMPLEMENTED",
            &[(
                "feature",
                "Collecting a year-month interval value in Spark Connect",
            )],
        ));
    }
    Ok(())
}

/// The collected `rows` of a result of `schema`, whose values of user-defined types are
/// deserialized when first read.
fn rows_to_py(
    py: Python<'_>,
    rows: Vec<spark_connect::row::Row>,
    schema: Option<&spark_connect::types::DataType>,
) -> PyResult<Vec<PyRow>> {
    check_collectable(py, schema)?;
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
        let next = py.detach(|| self.next_row());
        if self.converters.is_none() {
            if let Some(schema) = self.iterator.as_ref().and_then(LocalRowIterator::schema) {
                // The result is known once its first message arrives, possibly with no rows.
                if let Err(error) = check_collectable(py, Some(&schema)) {
                    self.finished = true;
                    return Err(error);
                }
                self.converters = Some(udt_field_converters(py, Some(&schema))?);
            }
        }
        match next {
            Some(Ok(row)) => {
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
pub(crate) fn literal_to_py(
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

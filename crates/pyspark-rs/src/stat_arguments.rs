//! The NA and statistic functions of a DataFrame (`df.dropna`, `df.na.fill`, `df.stat.corr`,
//! ...), which check their arguments on the client as PySpark Connect's DataFrame does.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use spark_connect::dataframe::DataFrame;
use spark_connect::group::StatFunctions;

use crate::column::raise_pyspark;
use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;
use crate::interrupt::interruptible;

/// PySpark's `NOT_EXPECTED_TYPE` error of the argument `name`, `value`, which should be of
/// the type `expected`.
pub(crate) fn not_expected_type(value: &Bound<'_, PyAny>, name: &str, expected: &str) -> PyErr {
    let type_name = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    raise_pyspark(
        value.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("arg_name", name),
            ("expected_type", expected),
            ("arg_type", &type_name),
        ],
    )
}

fn is_sequence(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>()
}

/// A bool, an int, a float or a str: a bool is an int.
fn is_primitive(value: &Bound<'_, PyAny>) -> bool {
    value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyString>()
}

/// The argument `name`, a str, as PySpark's statistic functions take a column.
fn string_argument(value: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    if value.is_instance_of::<PyString>() {
        value.extract()
    } else {
        Err(not_expected_type(value, name, "str"))
    }
}

/// The columns of the `subset` of `dropna` and `fillna`, as PySpark's: a column name, or a
/// list or tuple of them; `expected` is the type that the error of another type names.
fn subset_columns(subset: Option<&Bound<'_, PyAny>>, expected: &str) -> PyResult<Vec<String>> {
    let Some(subset) = subset else {
        return Ok(Vec::new());
    };
    if subset.is_instance_of::<PyString>() {
        return Ok(vec![subset.extract()?]);
    }
    if !is_sequence(subset) {
        return Err(not_expected_type(subset, "subset", expected));
    }
    subset
        .try_iter()?
        .map(|column| {
            let column = column?;
            if column.is_instance_of::<PyString>() {
                column.extract()
            } else {
                Err(not_expected_type(&column, "cols", "list, str or tuple"))
            }
        })
        .collect()
}

fn column_refs(columns: &[String]) -> Option<Vec<&str>> {
    (!columns.is_empty()).then(|| columns.iter().map(String::as_str).collect())
}

/// `df.dropna(how, thresh, subset)` and `df.na.drop`, as PySpark's: `how` is a str, which the core
/// checks to be "any" or "all", else `VALUE_NOT_ALLOWED`, `thresh`, an int, overrides it, and
/// `subset` is a column name, or a list or tuple of them.
pub(crate) fn dropna(
    df: &DataFrame,
    how: Option<&Bound<'_, PyAny>>,
    thresh: Option<&Bound<'_, PyAny>>,
    subset: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyDataFrame> {
    let how = how.map(|how| string_argument(how, "how")).transpose()?;
    // PySpark checks `how` before `thresh` and `subset`, so the core's error of `how` comes first.
    let parsed = dropna_thresh_and_subset(thresh, subset);
    let (thresh, columns) = match &parsed {
        Ok((thresh, columns)) => (*thresh, column_refs(columns)),
        Err(_) => (None, None),
    };
    let dataframe = df.dropna(how.as_deref(), thresh, columns).to_pyerr()?;
    parsed?;
    Ok(PyDataFrame::new(dataframe))
}

/// The `thresh` and the `subset` of `dropna`, checked as PySpark's. The `subset` is checked
/// before the `thresh` is sent as a proto int, which [`proto_int32`] checks.
fn dropna_thresh_and_subset(
    thresh: Option<&Bound<'_, PyAny>>,
    subset: Option<&Bound<'_, PyAny>>,
) -> PyResult<(Option<i32>, Vec<String>)> {
    let thresh = match thresh {
        None => None,
        Some(thresh) if thresh.is_instance_of::<PyInt>() => Some(thresh),
        Some(thresh) => return Err(not_expected_type(thresh, "thresh", "int")),
    };
    let columns = subset_columns(subset, "list, str or tuple")?;
    let thresh = thresh.map(proto_int32).transpose()?;
    Ok((thresh, columns))
}

/// An int, the `thresh` of `dropna`, as the proto's int32. A bool, which is an int, is refused as
/// the proto refuses it, `TypeError`, and an int out of the range of an int32 is a `ValueError`.
fn proto_int32(thresh: &Bound<'_, PyAny>) -> PyResult<i32> {
    if thresh.is_instance_of::<PyBool>() {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Cannot set spark.connect.NADrop.min_non_nulls to {thresh}: {thresh} has type {}, \
             but expected one of: (int).",
            thresh.get_type()
        )));
    }
    match thresh.extract::<i32>() {
        Ok(thresh) => Ok(thresh),
        Err(_) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Value out of range: {}",
            thresh.str()?
        ))),
    }
}

/// `df.fillna(value, subset)` and `df.na.fill`, as PySpark's: `value` is a bool, an int, a
/// float or a str, over the columns of `subset`, or a non-empty dict of them by column names.
pub(crate) fn fillna(
    df: &DataFrame,
    value: &Bound<'_, PyAny>,
    subset: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyDataFrame> {
    let dict = value.cast::<PyDict>().ok();
    if dict.is_none() && !is_primitive(value) {
        return Err(not_expected_type(
            value,
            "value",
            "bool, dict, float, int or str",
        ));
    }
    if let Some(dict) = &dict {
        if dict.is_empty() {
            return Err(raise_pyspark(
                value.py(),
                "PySparkValueError",
                "CANNOT_BE_EMPTY",
                &[("item", "value")],
            ));
        }
        for (column, fill) in dict.iter() {
            if !column.is_instance_of::<PyString>() {
                return Err(not_expected_type(&column, "key type of dict", "str"));
            }
            if !is_primitive(&fill) {
                return Err(not_expected_type(
                    &fill,
                    "value type of dict",
                    "bool, float, int or str",
                ));
            }
        }
    }
    let columns = subset_columns(subset, "list or tuple")?;
    if let Some(dict) = dict {
        let mut pairs = Vec::with_capacity(dict.len());
        for (column, fill) in dict.iter() {
            pairs.push((column.extract()?, fill_value(&fill)?));
        }
        return Ok(PyDataFrame::new(df.fillna_map(pairs).to_pyerr()?));
    }
    Ok(PyDataFrame::new(
        df.fillna_value(fill_value(value)?, column_refs(&columns))
            .to_pyerr()?,
    ))
}

/// The value of a fill. As PySpark sends it as a proto literal, an int beyond a long is out of
/// range for the proto, which raises `ValueError`, not the error of a literal.
fn fill_value(value: &Bound<'_, PyAny>) -> PyResult<spark_connect::row::Value> {
    if value.is_instance_of::<PyInt>()
        && !value.is_instance_of::<pyo3::types::PyBool>()
        && value.extract::<i64>().is_err()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Value out of range: {}",
            value.str()?
        )));
    }
    crate::session::py_to_value(value)
}

/// The columns of `df.describe(*cols)`, as PySpark's: column names, or a list of them.
pub(crate) fn describe_columns(cols: &[Bound<'_, PyAny>]) -> PyResult<Vec<String>> {
    let cols: Vec<Bound<'_, PyAny>> = match cols {
        [only] if only.is_instance_of::<PyList>() => only.try_iter()?.collect::<PyResult<_>>()?,
        _ => cols.to_vec(),
    };
    let mut names = Vec::new();
    for column in cols {
        if column.is_instance_of::<PyString>() {
            names.push(column.extract()?);
        } else {
            for name in column.try_iter()? {
                names.push(name?.extract()?);
            }
        }
    }
    Ok(names)
}

/// The statistics of `df.summary(*statistics)`, as PySpark's: strings.
pub(crate) fn summary_statistics(statistics: &[Bound<'_, PyAny>]) -> PyResult<Vec<String>> {
    statistics
        .iter()
        .map(|statistic| {
            if statistic.is_instance_of::<PyString>() {
                statistic.extract()
            } else {
                Err(not_expected_type(statistic, "statistics", "list[str]"))
            }
        })
        .collect()
}

/// `df.cov(col1, col2)`, of two column names, as PySpark's.
pub(crate) fn cov(
    py: Python<'_>,
    stat: &StatFunctions,
    col1: &Bound<'_, PyAny>,
    col2: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let (col1, col2) = (
        string_argument(col1, "col1")?,
        string_argument(col2, "col2")?,
    );
    interruptible(py, &stat.spark_session(), || {
        stat.cov(&col1, &col2).map_err(Box::new)
    })
}

/// `df.corr(col1, col2, method)`, of two column names and the method "pearson", the default,
/// else `VALUE_NOT_ALLOWED`, as PySpark's.
pub(crate) fn corr(
    py: Python<'_>,
    stat: &StatFunctions,
    col1: &Bound<'_, PyAny>,
    col2: &Bound<'_, PyAny>,
    method: Option<&Bound<'_, PyAny>>,
) -> PyResult<f64> {
    let (col1, col2) = (
        string_argument(col1, "col1")?,
        string_argument(col2, "col2")?,
    );
    if let Some(method) = method {
        if method.is_truthy()? && !method.eq("pearson")? {
            return Err(raise_pyspark(
                py,
                "PySparkValueError",
                "VALUE_NOT_ALLOWED",
                &[("arg_name", "method"), ("allowed_values", "['pearson']")],
            ));
        }
    }
    interruptible(py, &stat.spark_session(), || {
        stat.corr(&col1, &col2).map_err(Box::new)
    })
}

/// `df.crosstab(col1, col2)`, of two column names, as PySpark's.
pub(crate) fn crosstab(
    stat: &StatFunctions,
    col1: &Bound<'_, PyAny>,
    col2: &Bound<'_, PyAny>,
) -> PyResult<PyDataFrame> {
    let (col1, col2) = (
        string_argument(col1, "col1")?,
        string_argument(col2, "col2")?,
    );
    Ok(PyDataFrame::new(stat.crosstab(&col1, &col2)))
}

/// `df.freqItems(cols, support)`, as PySpark's: `cols` is a list or tuple, and `support` is
/// 0.01 unless one is given.
pub(crate) fn freq_items(
    stat: &StatFunctions,
    cols: &Bound<'_, PyAny>,
    support: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyDataFrame> {
    if !is_sequence(cols) {
        return Err(not_expected_type(cols, "cols", "list or tuple"));
    }
    let cols: Vec<String> = cols.extract()?;
    let support = match support {
        Some(support) if support.is_truthy()? => support.extract()?,
        _ => 0.01,
    };
    let refs: Vec<&str> = cols.iter().map(String::as_str).collect();
    Ok(PyDataFrame::new(stat.freq_items(refs, support)))
}

/// `df.approxQuantile(col, probabilities, relativeError)`, as PySpark's: the quantiles of a
/// column, or a list of those of each of a list or tuple of columns, at `probabilities`, a
/// list or tuple of numbers from 0 to 1, with a non-negative `relativeError`.
pub(crate) fn approx_quantile(
    py: Python<'_>,
    stat: &StatFunctions,
    col: &Bound<'_, PyAny>,
    probabilities: &Bound<'_, PyAny>,
    relative_error: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    use spark_connect::row::Value;

    let one_column = col.is_instance_of::<PyString>();
    if !one_column && !is_sequence(col) {
        return Err(not_expected_type(col, "col", "list, str or tuple"));
    }
    let columns: Vec<Bound<'_, PyAny>> = if one_column {
        vec![col.clone()]
    } else {
        col.try_iter()?.collect::<PyResult<_>>()?
    };
    let columns: Vec<String> = columns
        .iter()
        .map(|column| {
            if column.is_instance_of::<PyString>() {
                column.extract()
            } else {
                Err(not_expected_type(column, "columns", "list[str]"))
            }
        })
        .collect::<PyResult<_>>()?;
    if !is_sequence(probabilities) {
        return Err(not_expected_type(
            probabilities,
            "probabilities",
            "list or tuple",
        ));
    }
    let mut quantiles = Vec::new();
    for probability in probabilities.try_iter()? {
        let probability = probability?;
        let number =
            probability.is_instance_of::<PyInt>() || probability.is_instance_of::<PyFloat>();
        let value: f64 = if number { probability.extract()? } else { 0.0 };
        // As PySpark's `p < 0 or p > 1`, which a NaN passes, unlike a range's `contains`.
        #[allow(clippy::manual_range_contains)]
        let out_of_range = value < 0.0 || value > 1.0;
        if !number || out_of_range {
            return Err(not_expected_type(
                &probability,
                "probabilities",
                "list[float, int]",
            ));
        }
        quantiles.push(value);
    }
    if !(relative_error.is_instance_of::<PyInt>() || relative_error.is_instance_of::<PyFloat>()) {
        return Err(not_expected_type(
            relative_error,
            "relativeError",
            "float or int",
        ));
    }
    let error: f64 = relative_error.extract()?;
    if error < 0.0 {
        return Err(raise_pyspark(
            py,
            "PySparkValueError",
            "NEGATIVE_VALUE",
            &[
                ("arg_name", "relativeError"),
                ("arg_value", &relative_error.str()?.to_string()),
            ],
        ));
    }
    let refs: Vec<&str> = columns.iter().map(String::as_str).collect();
    let df = stat.approx_quantile(refs, quantiles, error);
    let rows = interruptible(py, &df.spark_session(), || df.collect().map_err(Box::new))?;
    // One row, of the array of the quantiles of each column.
    let per_column: Vec<Vec<f64>> = match rows.first().and_then(|row| row.get(0)) {
        Some(Value::List(columns)) => columns
            .iter()
            .map(|quantiles| match quantiles {
                Value::List(values) => values.iter().filter_map(|v| v.as_f64()).collect(),
                _ => Vec::new(),
            })
            .collect(),
        _ => Vec::new(),
    };
    if one_column {
        per_column
            .into_iter()
            .next()
            .unwrap_or_default()
            .into_py_any(py)
    } else {
        per_column.into_py_any(py)
    }
}

//! The conversions of the arguments of the readers and the writers of Spark Connect, as PySpark's
//! plans and commands of them convert them when they are sent.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString};
use spark_connect::dataframe::DataFrame;
use spark_connect_core::error::SparkError;

use crate::errors::not_expected_type;
use crate::options::utf32le_of;

/// The DataFrame `df`, which fails each request with `error` when it is one: the error of the plan
/// that PySpark raises when it sends the plan of a read, whose arguments it converts there.
pub(crate) fn with_plan_error(df: DataFrame, error: Option<SparkError>) -> DataFrame {
    match error {
        Some(error) => df.unbuildable(error),
        None => df,
    }
}

/// The name of a table, as PySpark's `Read` plan sets it in its string field: a str that UTF-8
/// cannot encode is kept with the error of that plan, and the name is empty. Bytes are decoded as
/// [`crate::str_argument`] decodes them, and bytes that are not UTF-8 are kept with their decode
/// error. Any other object is kept too, and fails the plan with the TypeError of protobuf's string
/// field, as PySpark's `table` does at the action and not at the call.
pub(crate) fn table_name_of(name: &Bound<'_, PyAny>) -> PyResult<(String, Option<SparkError>)> {
    if name.is_instance_of::<PyString>() {
        if let Ok(text) = name.extract::<String>() {
            return Ok((text, None));
        }
        let error = SparkError::unencodable_str(&utf32le_of(name)?);
        return Ok((String::new(), Some(error)));
    }
    if let Ok(bytes) = name.cast::<PyBytes>() {
        return Ok(match crate::str_argument(name) {
            Ok(text) => (text, None),
            Err(_) => (
                String::new(),
                Some(SparkError::invalid_utf8(0, bytes.as_bytes())),
            ),
        });
    }
    Ok((String::new(), Some(crate::options::builtin_type_error())))
}

/// The path of a writer, as PySpark sets it in its protobuf string field, which the default
/// (upb) backend converts as [`crate::str_argument`] does: a str, or bytes decoded as UTF-8, and
/// the error of that conversion for another object.
pub(crate) fn writer_path(path: &Bound<'_, PyAny>) -> PyResult<String> {
    crate::str_argument(path)
}

/// The table of `saveAsTable` and `insertInto`, which PySpark's `WriteOperation` sets in its
/// string field after the bucket count and before it checks the save mode, as [`writer_path`]
/// converts a path. None sets no table, so that the write is that of `save()` without a path.
pub(crate) fn writer_table_name(name: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if name.is_none() {
        return Ok(None);
    }
    crate::str_argument(name).map(Some)
}

/// The column items of `partitionBy(*cols)` and `clusterBy(*cols)` of a writer, as PySpark's take
/// them: a list or tuple alone stands for its items, else the arguments are the items.
pub(crate) fn writer_column_items<'py>(
    cols: &[Bound<'py, PyAny>],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    match cols {
        [only]
            if only.is_instance_of::<pyo3::types::PyList>()
                || only.is_instance_of::<pyo3::types::PyTuple>() =>
        {
            only.try_iter()?.collect::<PyResult<_>>()
        }
        _ => Ok(cols.to_vec()),
    }
}

/// The column names of the items of a writer column argument, as [`crate::str_argument`] converts
/// each one, which PySpark sets in a string field of its command at the write. The names that
/// convert are returned with the first item that does not, which the write fails with.
pub(crate) fn writer_columns_kept(items: &[Bound<'_, PyAny>]) -> (Vec<String>, Option<Py<PyAny>>) {
    let mut names = Vec::new();
    let mut unsent = None;
    for item in items {
        match crate::str_argument(item) {
            Ok(name) => names.push(name),
            Err(_) => {
                unsent.get_or_insert_with(|| item.clone().unbind());
            }
        }
    }
    (names, unsent)
}

/// The partitioning columns of `partitionBy(*cols)` of a writer, as [`writer_columns_kept`].
pub(crate) fn writer_partition_columns(
    cols: &[Bound<'_, PyAny>],
) -> PyResult<(Vec<String>, Option<Py<PyAny>>)> {
    Ok(writer_columns_kept(&writer_column_items(cols)?))
}

/// The save mode of a write whose `mode` is not a str, as PySpark's `WriteOperation.command`
/// resolves it: `mode.lower()` is one of the four modes, or else UNSUPPORTED_OPERATION of the
/// mode, as for the bytes of `b"append"`. An object without `lower` raises its AttributeError.
pub(crate) fn unsent_save_mode(mode: &Bound<'_, PyAny>) -> PyResult<String> {
    let lowered = mode.call_method0("lower")?;
    if let Ok(text) = lowered.extract::<String>() {
        if ["append", "overwrite", "error", "ignore"].contains(&text.as_str()) {
            return Ok(text);
        }
    }
    // The operation is the mode object itself, which PySpark's message formats as it is.
    let py = mode.py();
    let params = PyDict::new(py);
    params.set_item("operation", mode)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("errorClass", "UNSUPPORTED_OPERATION")?;
    kwargs.set_item("messageParameters", params)?;
    let error = py
        .import("pyspark.errors")?
        .getattr("PySparkValueError")?
        .call((), Some(&kwargs))?;
    Err(PyErr::from_value(error))
}

/// The column names of a bucketing or sorting argument, which keeps the first column that does not
/// convert, as [`writer_columns_kept`] does.
pub(crate) fn writer_columns_kept_of(
    col: &Bound<'_, PyAny>,
    cols: &[Bound<'_, PyAny>],
) -> PyResult<(Vec<String>, Option<Py<PyAny>>)> {
    Ok(writer_columns_kept(&writer_columns_items(col, cols)?))
}

/// The items of a bucketing or sorting argument: `col` and each of `cols` are checked to be strs,
/// and a list or a tuple of `col` takes the others.
fn writer_columns_items<'py>(
    col: &Bound<'py, PyAny>,
    cols: &[Bound<'py, PyAny>],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let sequence =
        col.is_instance_of::<pyo3::types::PyList>() || col.is_instance_of::<pyo3::types::PyTuple>();
    let (col, cols) = if sequence {
        if !cols.is_empty() {
            let type_name = col.get_type().name()?.to_string();
            return Err(crate::column::raise_pyspark(
                col.py(),
                "PySparkValueError",
                "CANNOT_SET_TOGETHER",
                &[("arg_list", &format!("`col` of type {type_name} and `cols`"))],
            ));
        }
        // An empty list or tuple raises an IndexError, as PySpark's `col[0]`.
        let first = col.get_item(0)?;
        let rest: Vec<Bound<'_, PyAny>> = col.try_iter()?.skip(1).collect::<PyResult<_>>()?;
        (first, rest)
    } else {
        (col.clone(), cols.to_vec())
    };
    if let Some(other) = cols
        .iter()
        .find(|c| !c.is_instance_of::<pyo3::types::PyString>())
    {
        return Err(not_expected_type(other, "cols", "list[str]"));
    }
    if !col.is_instance_of::<pyo3::types::PyString>() {
        return Err(not_expected_type(&col, "col", "list[str]"));
    }
    Ok(std::iter::once(col).chain(cols).collect())
}

/// The int32 field of the number of buckets, which PySpark sets at the write.
const BUCKET_COUNT_FIELD: &str = "spark.connect.WriteOperation.BucketBy.num_buckets";

/// The number of buckets of `df.write.bucketBy`, as PySpark's: `bucketBy` checks that it is an
/// int, a bool included, and keeps it, and the `WriteOperation` of each write sends the bucketing
/// only of a count above 0, which it then sets in the int32 field.
pub(crate) enum BucketCount<'py> {
    /// A count that is not positive: no bucketing is sent.
    Unsent,
    /// The count of the buckets.
    Count(i32),
    /// A positive count that the int32 field refuses, a bool or an int beyond its range, with
    /// which each write fails, as [`check_refused_bucket_count`] raises.
    Refused(Bound<'py, PyAny>),
}

/// The [`BucketCount`] of `num_buckets`, else `NOT_EXPECTED_TYPE` of an argument that is not an
/// int, as PySpark's `bucketBy` raises at the call.
pub(crate) fn bucket_count<'py>(num_buckets: &Bound<'py, PyAny>) -> PyResult<BucketCount<'py>> {
    if !num_buckets.is_instance_of::<pyo3::types::PyInt>() {
        return Err(not_expected_type(num_buckets, "numBuckets", "int"));
    }
    if !num_buckets.gt(0)? {
        return Ok(BucketCount::Unsent);
    }
    Ok(
        match crate::stat_arguments::proto_int::<i32>(num_buckets, BUCKET_COUNT_FIELD) {
            Ok(count) => BucketCount::Count(count),
            Err(_) => BucketCount::Refused(num_buckets.clone()),
        },
    )
}

/// Set a count of [`BucketCount::Refused`] in the int32 field again, as PySpark's
/// `WriteOperation.command` does at each write, which fails with the error of protobuf's default
/// (upb) backend: the `TypeError` of a bool and the `ValueError` of an int out of the range.
pub(crate) fn check_refused_bucket_count(num_buckets: &Bound<'_, PyAny>) -> PyResult<()> {
    crate::stat_arguments::proto_int::<i32>(num_buckets, BUCKET_COUNT_FIELD).map(|_| ())
}

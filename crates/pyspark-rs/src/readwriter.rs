//! PyO3 wrapper for spark_connect::readwriter::DataFrameReader (`spark.read`).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::readwriter::DataFrameReader;

use crate::dataframe::PyDataFrame;
use crate::errors::{not_expected_type, ResultExt};
use crate::options::{
    check_keyword_keys, coerce_option_value, set_option_skip_none, set_options_skip_none,
    set_reader_kwargs_unset_none, set_reader_option_unset_none, DefaultedOption,
};

/// Python wrapper for the batch DataFrameReader.
///
/// PySpark's reader is mutable: `format`/`schema`/`option`/`options` change `self`
/// and return it, and a reader can be used again after a read. The core reader is a
/// consuming builder, so the setters replace the wrapped value in place and return
/// the same Python object, and the terminal reads build their DataFrame from a clone.
#[pyclass(
    name = "DataFrameReader",
    module = "pyspark.sql.readwriter",
    weakref,
    dict
)]
pub struct PyDataFrameReader {
    inner: DataFrameReader,
    /// The options of the reader, which a shallow copy of it shares, as PySpark's `_options` dict
    /// is shared by the copies of a reader. The core reader holds none of them (see `state`).
    options: std::sync::Arc<crate::dataframe::WriterState<HashMap<String, String>>>,
    /// The format of `format` when it is not a str, which PySpark keeps as it is and whose
    /// `DataSource` asserts at `load` ([`reader_format`]). Any other format replaces it.
    unsent_format: Option<Py<PyAny>>,
}

impl PyDataFrameReader {
    pub fn new(reader: DataFrameReader) -> Self {
        let options = reader.get_options().clone();
        PyDataFrameReader {
            inner: without_options(reader, &options),
            options: std::sync::Arc::new(crate::dataframe::WriterState::new(options)),
            unsent_format: None,
        }
    }

    /// The core reader of this reader, with its options, which the copies of this reader share.
    fn state(&self) -> DataFrameReader {
        self.inner.clone().options(self.options.get())
    }

    /// Keep `reader` as the state of this reader: its options are shared by the copies of this
    /// reader, and its format and schema are this reader's own.
    fn set_state(&mut self, reader: DataFrameReader) {
        let options = reader.get_options().clone();
        self.inner = without_options(reader, &options);
        self.options.set(options);
    }

    /// Keep the options of a format-specific read on this reader, as PySpark does
    /// (its `json`/`csv`/... set the options and the format on `self`, then `load`).
    /// Called before the path is checked, as PySpark's `load` sets the format first.
    fn commit(&mut self, r: &DataFrameReader, format: &str) {
        self.set_state(r.clone().format(format));
        self.unsent_format = None;
    }

    /// The DataFrame of `r`, this reader with the options of a call of `json`, `csv` or `xml`
    /// of `path`, as PySpark's: a DataFrame of strings, which it parses as `parse_format`,
    /// keeping the schema and the options on this reader but not the format, or else a path
    /// or a list of them, which it loads as `format`. `csv`, unlike the others, passes any
    /// other `path` to `load`, which asserts it. The schema, if any, is set only on those two
    /// paths, as PySpark's, so a path of another type keeps the options alone.
    fn read_or_parse(
        &mut self,
        r: DataFrameReader,
        schema: Option<&Bound<'_, PyAny>>,
        format: &str,
        parse_format: spark_connect::plan::ParseFormat,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's, this sets the format of the reader, which replaces an unsent one.
        self.unsent_format = None;
        if let Ok(df) = path.extract::<PyRef<'_, PyDataFrame>>() {
            let r = apply_optional_schema(r, schema)?;
            let parsed = r
                .clone()
                .parse(parse_format, &df.dataframe)
                .map(PyDataFrame::new)
                .to_pyerr()?;
            self.set_state(r);
            return Ok(parsed);
        }
        if path.is_instance_of::<pyo3::types::PyString>()
            || path.is_instance_of::<pyo3::types::PyList>()
            || parse_format == spark_connect::plan::ParseFormat::Csv
        {
            let r = apply_optional_schema(r, schema)?;
            self.commit(&r, format);
            let paths = paths_of(path)?;
            return Ok(PyDataFrame::new(r.format(format).load_paths(paths)));
        }
        // PySpark's options are set before the path is checked, so they stay on this reader.
        self.set_state(r);
        Err(not_expected_type(path, "path", "str, list, or DataFrame"))
    }
}

/// The reader without the options of `options`, which the reader keeps apart from its core state.
fn without_options(reader: DataFrameReader, options: &HashMap<String, String>) -> DataFrameReader {
    options
        .keys()
        .fold(reader, |reader, key| reader.unset_option(key))
}

/// The reader with the format `source` of PySpark's `format`, and the unsent format that it keeps.
/// None unsets the format. A str sets it. Any other object is kept as it is, and `load` fails its
/// `DataSource` assertion with it, as PySpark's `DataSource` takes only a str.
fn reader_format(
    reader: DataFrameReader,
    source: Option<&Bound<'_, PyAny>>,
) -> (DataFrameReader, Option<Py<PyAny>>) {
    match source {
        None => (reader.clear_format(), None),
        Some(source) => match source.extract::<String>() {
            Ok(text) => (reader.format(&text), None),
            Err(_) => (reader.clear_format(), Some(source.clone().unbind())),
        },
    }
}

/// The paths of `path`, a path or a list of them, or None for none, as PySpark's `load` takes
/// them: anything else fails its assertion.
fn paths_of(path: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if path.is_none() {
        return Ok(vec![]);
    }
    if path.is_instance_of::<pyo3::types::PyString>() {
        return Ok(vec![path.extract()?]);
    }
    if !path.is_instance_of::<pyo3::types::PyList>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    path_list(&path.try_iter()?.collect::<PyResult<Vec<_>>>()?)
}

/// The paths of `paths`, strings, as PySpark's `load` asserts them.
fn path_list(paths: &[Bound<'_, PyAny>]) -> PyResult<Vec<String>> {
    paths
        .iter()
        .map(|path| {
            if path.is_instance_of::<pyo3::types::PyString>() {
                path.extract()
            } else {
                Err(pyo3::exceptions::PyAssertionError::new_err(()))
            }
        })
        .collect()
}

/// The `predicates` of `jdbc`, which PySpark's `DataSource` plan asserts to be a list, and each of
/// its items a str: a bare `AssertionError` for a tuple, a str, or an item that is not a str.
fn jdbc_predicates(predicates: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let Ok(predicates) = predicates.cast::<pyo3::types::PyList>() else {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    };
    predicates
        .iter()
        .map(|predicate| {
            if predicate.is_instance_of::<pyo3::types::PyString>() {
                crate::str_argument(&predicate)
            } else {
                Err(pyo3::exceptions::PyAssertionError::new_err(()))
            }
        })
        .collect()
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

/// The columns of `df.write.bucketBy(numBuckets, col, *cols)` and `sortBy(col, *cols)`, as
/// PySpark's: column names, or a list or tuple of them alone.
pub(crate) fn writer_columns(
    col: &Bound<'_, PyAny>,
    cols: &[Bound<'_, PyAny>],
) -> PyResult<Vec<String>> {
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
    std::iter::once(&col)
        .chain(cols.iter())
        .map(|c| c.extract())
        .collect()
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

/// A reader `schema` as pyspark's `DataFrameReader.schema` and `DataStreamReader.schema`
/// send it: a StructType as its JSON, which keeps field metadata, and a string as is, for
/// the server to parse. Anything else raises `NOT_EXPECTED_TYPE`.
pub(crate) fn reader_schema(schema: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(schema) = schema.extract::<String>() {
        return Ok(schema);
    }
    if let Ok(st) = schema.extract::<PyRef<crate::types::PyStructType>>() {
        return Ok(spark_connect::types::DataType::Struct {
            fields: st.current_fields(schema.py())?,
        }
        .json());
    }
    let arg_type = schema.get_type().name()?.to_string();
    Err(crate::column::raise_pyspark(
        schema.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("arg_name", "schema"),
            ("expected_type", "str or struct type"),
            ("arg_type", &arg_type),
        ],
    ))
}

/// The `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` that PySpark's `DataFrameReader.changes`
/// and `DataStreamReader.changes` raise for a reader of a schema, of the message of the JVM
/// error class, which PySpark's error classes lack.
pub(crate) fn changes_of_schema_error(py: Python<'_>) -> PyErr {
    let exception = (|| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("errorClass", "_LEGACY_ERROR_TEMP_1189")?;
        let parameters = PyDict::new(py);
        parameters.set_item("operation", "changes")?;
        kwargs.set_item("messageParameters", parameters)?;
        py.import("pyspark.errors")?
            .getattr("AnalysisException")?
            .call(
                ("User specified schema not supported with `changes`.",),
                Some(&kwargs),
            )
    })();
    match exception {
        Ok(exception) => PyErr::from_value(exception),
        Err(err) => err,
    }
}

/// Apply the `schema` argument of a format method, as pyspark does through `schema()`.
fn apply_reader_schema(r: DataFrameReader, v: &Bound<'_, PyAny>) -> PyResult<DataFrameReader> {
    Ok(r.schema(reader_schema(v)?))
}

/// Apply the `schema` argument of a format method, if there is one.
fn apply_optional_schema(
    r: DataFrameReader,
    schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<DataFrameReader> {
    match schema {
        Some(v) => apply_reader_schema(r, v),
        None => Ok(r),
    }
}

#[pymethods]
impl PyDataFrameReader {
    /// The shallow copy of a reader, as PySpark's `copy.copy`: the copy shares the options of
    /// this reader, as its `_options` dict is shared, so an option that either one sets or
    /// unsets is seen by both, and has its own format, which is kept as it is.
    fn __copy__(&self, py: Python<'_>) -> PyDataFrameReader {
        PyDataFrameReader {
            inner: self.inner.clone(),
            options: std::sync::Arc::clone(&self.options),
            unsent_format: self
                .unsent_format
                .as_ref()
                .map(|format| format.clone_ref(py)),
        }
    }

    /// The reduce of a reader, as PySpark's pickle of its session and its state, which
    /// `_of_state` gives back.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use crate::reduce::{optional_object, state_reduce, str_dict};

        let session = crate::session::session_object(py, &self.inner.spark_session())?;
        let args = pyo3::types::PyTuple::new(
            py,
            [
                session.into_bound(py).into_any(),
                self.inner.get_format().into_pyobject(py)?.into_any(),
                self.inner.get_schema().into_pyobject(py)?.into_any(),
                str_dict(py, &self.options.get())?.into_any(),
                optional_object(py, self.unsent_format.as_ref()),
            ],
        )?;
        state_reduce(py, "reader_reduce", args)
    }

    /// A reader of its state, as `__reduce__` gives it: of the session `session`, the format,
    /// the schema, the options and the format that is kept (`unsent`).
    #[staticmethod]
    fn _of_state(
        py: Python<'_>,
        session: Py<crate::session::PySparkSession>,
        format: Option<String>,
        schema: String,
        options: HashMap<String, String>,
        unsent: Option<Py<PyAny>>,
    ) -> PyDataFrameReader {
        let mut reader = session.borrow(py).session.read();
        if let Some(format) = format {
            reader = reader.format(&format);
        }
        if !schema.is_empty() {
            reader = reader.schema(schema);
        }
        PyDataFrameReader {
            unsent_format: unsent,
            ..PyDataFrameReader::new(reader.options(options))
        }
    }

    /// Set the schema, if any, and the options of `options` that are not None, through its own
    /// `schema` and `option`, as PySpark's `OptionUtils._set_opts`, which pandas-on-Spark calls.
    #[pyo3(signature = (schema=None, **options))]
    fn _set_opts(
        slf: &Bound<'_, Self>,
        schema: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        crate::options::set_opts(slf.as_any(), schema.as_ref(), options.as_ref())
    }

    /// Set the source format (e.g. "parquet", "json", "csv"). As PySpark's, None unsets it, so
    /// that the server reads its default source, and an object that is not a str is kept until
    /// `load` fails its assertion.
    fn format<'py>(
        mut slf: PyRefMut<'py, Self>,
        source: Option<&Bound<'_, PyAny>>,
    ) -> PyRefMut<'py, Self> {
        let (reader, unsent) = reader_format(slf.state(), source);
        slf.set_state(reader);
        slf.unsent_format = unsent;
        slf
    }

    /// Set the schema: a StructType or a DDL string, as in pyspark.
    fn schema<'py>(
        mut slf: PyRefMut<'py, Self>,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let schema = reader_schema(schema)?;
        let state = slf.state().schema(schema);
        slf.set_state(state);
        Ok(slf)
    }

    /// Set a single read option. `None` unsets the option, as PySpark's does (not the
    /// string "None"); booleans lowercase to "true"/"false" (reference `to_str` semantics).
    /// PySpark keeps the options in a dict, which hashes every key, and its `DataSource` plan
    /// checks the keys of the values that are not None only.
    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if value.is_none() {
            key.hash()?;
            // Only a str that UTF-8 encodes is the key of an option that is set.
            if let Ok(key) = key.extract::<String>() {
                let state = slf.state().unset_option(&key);
                slf.set_state(state);
            }
            return Ok(slf);
        }
        let key = crate::options::reader_option_key(key)?;
        let state = set_reader_option_unset_none(slf.state(), &key, value)?;
        slf.set_state(state);
        Ok(slf)
    }

    /// Set multiple read options. Mirrors reference `DataFrameReader.options(**options)`
    /// - keyword args; a `None` value unsets the option and booleans lowercase.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let state = set_reader_kwargs_unset_none(slf.state(), options)?;
        slf.set_state(state);
        Ok(slf)
    }

    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if let Some(format) = &self.unsent_format {
            visit.call(format)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.unsent_format = None;
    }

    /// The configured source format (PySpark Connect's `DataFrameReader._format`).
    #[getter]
    fn _format(&self) -> Option<String> {
        self.inner.get_format().map(str::to_string)
    }

    /// The configured schema string, empty when unset (`DataFrameReader._schema`).
    #[getter]
    fn _schema(&self) -> String {
        self.inner.get_schema().to_string()
    }

    /// The configured read options (`DataFrameReader._options`); a copy.
    #[getter]
    fn _options(&self) -> HashMap<String, String> {
        self.options.get()
    }

    /// The JVM reader of classic PySpark, which PySpark Connect's raises
    /// `JVM_ATTRIBUTE_NOT_SUPPORTED` for.
    #[getter]
    fn _jreader(&self, py: Python<'_>) -> PyResult<()> {
        Err(crate::column::raise_pyspark(
            py,
            "PySparkAttributeError",
            "JVM_ATTRIBUTE_NOT_SUPPORTED",
            &[("attr_name", "_jreader")],
        ))
    }

    /// Load data from the (optional) path. Mirrors the reference
    /// `DataFrameReader.load(path=None, format=None, schema=None, **options)`: the
    /// one-shot form applies format/schema/options here before reading, equivalent to
    /// chaining `.format(...).schema(...).options(...).load(path)`.
    #[pyo3(signature = (path=None, format=None, schema=None, **options))]
    fn load(
        &mut self,
        path: Option<&Bound<'_, PyAny>>,
        format: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's load, which sets the format, the schema and the options on the reader in
        // turn, each one kept when a later one fails. CPython checks the keyword keys first.
        check_keyword_keys(options)?;
        let mut r = self.state();
        if let Some(f) = format {
            let (reader, unsent) = reader_format(r, Some(f));
            r = reader;
            self.unsent_format = unsent;
        }
        self.set_state(r.clone());
        if let Some(s) = schema {
            r = apply_reader_schema(r, s)?;
        }
        self.set_state(r.clone());
        r = set_reader_kwargs_unset_none(r, options)?;
        // Then its `DataSource` asserts the format, which precedes the check of the paths.
        self.set_state(r.clone());
        if self.unsent_format.is_some() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let paths = path.map(paths_of).transpose()?.unwrap_or_default();
        Ok(PyDataFrame::new(r.load_paths(paths)))
    }

    /// Read a table by name, which PySpark's `Read` plan sets in its string field as it is, as
    /// [`crate::str_argument`] converts it, as `changes` and those of the stream reader do.
    fn table(&self, tableName: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let tableName = crate::str_argument(tableName)?;
        Ok(PyDataFrame::new(self.state().table(&tableName)))
    }

    /// Read the CDC changes of a named table. Mirrors `DataFrameReader.changes`, which raises an
    /// `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` for a reader of a schema.
    #[allow(non_snake_case)]
    fn changes(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        if !self.inner.get_schema().is_empty() {
            return Err(changes_of_schema_error(py));
        }
        let tableName = crate::str_argument(tableName)?;
        Ok(PyDataFrame::new(self.state().changes(&tableName)))
    }

    /// Read JSON file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, samplingRatio=None, dropFieldIfAllNull=None, encoding=None, locale=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, allowNonNumericNumbers=None, useUnsafeRow=None, **unexpected), text_signature = "($self, path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, samplingRatio=None, dropFieldIfAllNull=None, encoding=None, locale=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, allowNonNumericNumbers=None, useUnsafeRow=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn json(
        &mut self,
        path: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        primitivesAsString: Option<&Bound<'_, PyAny>>,
        prefersDecimal: Option<&Bound<'_, PyAny>>,
        allowComments: Option<&Bound<'_, PyAny>>,
        allowUnquotedFieldNames: Option<&Bound<'_, PyAny>>,
        allowSingleQuotes: Option<&Bound<'_, PyAny>>,
        allowNumericLeadingZero: Option<&Bound<'_, PyAny>>,
        allowBackslashEscapingAnyCharacter: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        allowUnquotedControlChars: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        samplingRatio: Option<&Bound<'_, PyAny>>,
        dropFieldIfAllNull: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
        allowNonNumericNumbers: Option<&Bound<'_, PyAny>>,
        useUnsafeRow: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataFrameReader.json", unexpected)?;
        let mut r = self.state();
        r = set_options_skip_none(
            r,
            &[
                ("primitivesAsString", primitivesAsString),
                ("prefersDecimal", prefersDecimal),
                ("allowComments", allowComments),
                ("allowUnquotedFieldNames", allowUnquotedFieldNames),
                ("allowSingleQuotes", allowSingleQuotes),
                ("allowNumericLeadingZero", allowNumericLeadingZero),
                (
                    "allowBackslashEscapingAnyCharacter",
                    allowBackslashEscapingAnyCharacter,
                ),
                ("mode", mode),
                ("columnNameOfCorruptRecord", columnNameOfCorruptRecord),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("multiLine", multiLine),
                ("allowUnquotedControlChars", allowUnquotedControlChars),
                ("lineSep", lineSep),
                ("samplingRatio", samplingRatio),
                ("dropFieldIfAllNull", dropFieldIfAllNull),
                ("encoding", encoding),
                ("locale", locale),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
                ("modifiedBefore", modifiedBefore),
                ("modifiedAfter", modifiedAfter),
                ("allowNonNumericNumbers", allowNonNumericNumbers),
                ("useUnsafeRow", useUnsafeRow),
            ],
        )?;
        self.read_or_parse(
            r,
            schema,
            "json",
            spark_connect::plan::ParseFormat::Json,
            path,
        )
    }

    /// Read Parquet file(s). As PySpark's `parquet(*paths, **options)`, only the options
    /// `mergeSchema`, `pathGlobFilter`, `recursiveFileLookup`, `modifiedBefore`,
    /// `modifiedAfter`, `datetimeRebaseMode` and `int96RebaseMode` of `options` are set, and
    /// the others are ignored.
    #[pyo3(signature = (*paths, **options))]
    fn parquet(
        &mut self,
        paths: Vec<Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        check_keyword_keys(options)?;
        let mut r = self.state();
        for name in [
            "mergeSchema",
            "pathGlobFilter",
            "recursiveFileLookup",
            "modifiedBefore",
            "modifiedAfter",
            "datetimeRebaseMode",
            "int96RebaseMode",
        ] {
            let value = options.map(|o| o.get_item(name)).transpose()?.flatten();
            r = set_option_skip_none(r, name, value.as_ref())?;
        }
        self.commit(&r, "parquet");
        let paths = path_list(&paths)?;
        Ok(PyDataFrame::new(r.format("parquet").load_paths(paths)))
    }

    /// Read CSV file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, samplingRatio=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, unescapedQuoteHandling=None, **unexpected), text_signature = "($self, path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, samplingRatio=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, unescapedQuoteHandling=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn csv(
        &mut self,
        path: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        sep: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        quote: Option<&Bound<'_, PyAny>>,
        escape: Option<&Bound<'_, PyAny>>,
        comment: Option<&Bound<'_, PyAny>>,
        header: Option<&Bound<'_, PyAny>>,
        inferSchema: Option<&Bound<'_, PyAny>>,
        ignoreLeadingWhiteSpace: Option<&Bound<'_, PyAny>>,
        ignoreTrailingWhiteSpace: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        nanValue: Option<&Bound<'_, PyAny>>,
        positiveInf: Option<&Bound<'_, PyAny>>,
        negativeInf: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        maxColumns: Option<&Bound<'_, PyAny>>,
        maxCharsPerColumn: Option<&Bound<'_, PyAny>>,
        maxMalformedLogPerPartition: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        charToEscapeQuoteEscaping: Option<&Bound<'_, PyAny>>,
        samplingRatio: Option<&Bound<'_, PyAny>>,
        enforceSchema: Option<&Bound<'_, PyAny>>,
        emptyValue: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
        unescapedQuoteHandling: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataFrameReader.csv", unexpected)?;
        let mut r = self.state();
        r = set_options_skip_none(
            r,
            &[
                ("sep", sep),
                ("encoding", encoding),
                ("quote", quote),
                ("escape", escape),
                ("comment", comment),
                ("header", header),
                ("inferSchema", inferSchema),
                ("ignoreLeadingWhiteSpace", ignoreLeadingWhiteSpace),
                ("ignoreTrailingWhiteSpace", ignoreTrailingWhiteSpace),
                ("nullValue", nullValue),
                ("nanValue", nanValue),
                ("positiveInf", positiveInf),
                ("negativeInf", negativeInf),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("maxColumns", maxColumns),
                ("maxCharsPerColumn", maxCharsPerColumn),
                ("maxMalformedLogPerPartition", maxMalformedLogPerPartition),
                ("mode", mode),
                ("columnNameOfCorruptRecord", columnNameOfCorruptRecord),
                ("multiLine", multiLine),
                ("charToEscapeQuoteEscaping", charToEscapeQuoteEscaping),
                ("samplingRatio", samplingRatio),
                ("enforceSchema", enforceSchema),
                ("emptyValue", emptyValue),
                ("locale", locale),
                ("lineSep", lineSep),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
                ("modifiedBefore", modifiedBefore),
                ("modifiedAfter", modifiedAfter),
                ("unescapedQuoteHandling", unescapedQuoteHandling),
            ],
        )?;
        self.read_or_parse(
            r,
            schema,
            "csv",
            spark_connect::plan::ParseFormat::Csv,
            path,
        )
    }

    /// Read ORC file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, **unexpected), text_signature = "($self, path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn orc(
        &mut self,
        path: &Bound<'_, PyAny>,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataFrameReader.orc", unexpected)?;
        let mut r = self.state();
        r = set_options_skip_none(
            r,
            &[
                ("mergeSchema", mergeSchema),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
                ("modifiedBefore", modifiedBefore),
                ("modifiedAfter", modifiedAfter),
            ],
        )?;
        self.commit(&r, "orc");
        let paths = paths_of(path)?;
        Ok(PyDataFrame::new(r.format("orc").load_paths(paths)))
    }

    /// Read TEXT file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided. As PySpark's `wholetext=False`, an omitted
    /// `wholetext` sets `false`, also over an earlier `wholetext` option.
    #[pyo3(
        signature = (paths, wholetext=DefaultedOption::Omitted, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, **unexpected),
        text_signature = "($self, paths, wholetext=False, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn text(
        &mut self,
        py: Python<'_>,
        paths: &Bound<'_, PyAny>,
        wholetext: DefaultedOption,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataFrameReader.text", unexpected)?;
        let wholetext = wholetext.or_bool(py, false);
        let mut r = self.state();
        r = set_options_skip_none(
            r,
            &[
                ("wholetext", Some(&wholetext)),
                ("lineSep", lineSep),
                ("pathGlobFilter", pathGlobFilter),
                ("recursiveFileLookup", recursiveFileLookup),
                ("modifiedBefore", modifiedBefore),
                ("modifiedAfter", modifiedAfter),
            ],
        )?;
        self.commit(&r, "text");
        let paths = paths_of(paths)?;
        Ok(PyDataFrame::new(r.format("text").load_paths(paths)))
    }

    /// Read XML file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None, **unexpected), text_signature = "($self, path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn xml(
        &mut self,
        path: &Bound<'_, PyAny>,
        rowTag: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
        excludeAttribute: Option<&Bound<'_, PyAny>>,
        attributePrefix: Option<&Bound<'_, PyAny>>,
        valueTag: Option<&Bound<'_, PyAny>>,
        ignoreSurroundingSpaces: Option<&Bound<'_, PyAny>>,
        rowValidationXSDPath: Option<&Bound<'_, PyAny>>,
        ignoreNamespace: Option<&Bound<'_, PyAny>>,
        wildcardColName: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        inferSchema: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        samplingRatio: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
        unexpected: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        crate::options::check_unexpected_keywords("DataFrameReader.xml", unexpected)?;
        let mut r = self.state();
        r = set_option_skip_none(r, "rowTag", rowTag)?;
        if let Some(v) = schema {
            r = apply_reader_schema(r, v)?;
        }
        r = set_options_skip_none(
            r,
            &[
                ("excludeAttribute", excludeAttribute),
                ("attributePrefix", attributePrefix),
                ("valueTag", valueTag),
                ("ignoreSurroundingSpaces", ignoreSurroundingSpaces),
                ("rowValidationXSDPath", rowValidationXSDPath),
                ("ignoreNamespace", ignoreNamespace),
                ("wildcardColName", wildcardColName),
                ("encoding", encoding),
                ("inferSchema", inferSchema),
                ("nullValue", nullValue),
                ("dateFormat", dateFormat),
                ("timestampFormat", timestampFormat),
                ("mode", mode),
                ("columnNameOfCorruptRecord", columnNameOfCorruptRecord),
                ("multiLine", multiLine),
                ("samplingRatio", samplingRatio),
                ("locale", locale),
            ],
        )?;
        self.read_or_parse(r, None, "xml", spark_connect::plan::ParseFormat::Xml, path)
    }

    /// Read from a JDBC source. Mirrors `DataFrameReader.jdbc(url, table, column=None,
    /// lowerBound=None, upperBound=None, numPartitions=None, predicates=None, properties=None)`,
    /// which keeps the format, then the options, on the reader:
    /// - of a `column`, it asserts the bounds and `numPartitions`, and reads the partitions of
    ///   the column, ignoring `predicates`;
    /// - else, it reads the partitions of `predicates`, ignoring the bounds and `numPartitions`.
    ///
    /// The column, the bounds, `numPartitions` and the values of `properties` are options, of any
    /// type (None skipped, bools lowercased).
    #[pyo3(signature = (url, table, column=None, lowerBound=None, upperBound=None, numPartitions=None, predicates=None, properties=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn jdbc(
        &mut self,
        url: &Bound<'_, PyAny>,
        table: &Bound<'_, PyAny>,
        column: Option<&Bound<'_, PyAny>>,
        lowerBound: Option<&Bound<'_, PyAny>>,
        upperBound: Option<&Bound<'_, PyAny>>,
        numPartitions: Option<&Bound<'_, PyAny>>,
        predicates: Option<&Bound<'_, PyAny>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        self.set_state(self.state().format("jdbc"));
        self.unsent_format = None;
        let mut r = self.state();
        let predicates = match column {
            Some(column) => {
                for (bound, name) in [
                    (lowerBound, "lowerBound"),
                    (upperBound, "upperBound"),
                    (numPartitions, "numPartitions"),
                ] {
                    if bound.is_none() {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                            "{name} can not be None when ``column`` is specified"
                        )));
                    }
                }
                r = set_options_skip_none(
                    r,
                    &[
                        ("partitionColumn", Some(column)),
                        ("lowerBound", lowerBound),
                        ("upperBound", upperBound),
                        ("numPartitions", numPartitions),
                    ],
                )?;
                None
            }
            None => predicates,
        };
        if let Some(properties) = properties {
            // As PySpark's `options(**properties)`, whose unpacking raises for a key that is not
            // a str before any property is set, with the format and the options of a column kept
            // on the reader.
            if let Err(err) = check_keyword_keys(Some(properties)) {
                self.set_state(r);
                return Err(err);
            }
            for (key, value) in properties.iter() {
                r = set_option_skip_none(r, &crate::str_argument(&key)?, Some(&value))?;
            }
        }
        // PySpark's jdbc also sets url and dbtable as options, `options(url=url, dbtable=table)`,
        // which leaves an option of None unset, before it asserts the predicates.
        r = set_reader_option_unset_none(r, "url", url)?;
        r = set_reader_option_unset_none(r, "dbtable", table)?;
        self.set_state(r.clone());
        let predicates = predicates.map(jdbc_predicates).transpose()?;
        let url = coerce_option_value(url)?;
        let table = coerce_option_value(table)?;
        Ok(PyDataFrame::new(match (url, table, predicates) {
            (Some(url), Some(table), predicates) => r.jdbc(&url, &table, predicates),
            // Without predicates, the read is that of the options alone, which leave a None url or
            // dbtable unset, as the core's `jdbc` needs both as strs.
            (_, _, None) => r.load_paths(Vec::<String>::new()),
            // Known residual: with predicates, the core's `jdbc` sets a None url or dbtable as "".
            (url, table, predicates) => r.jdbc(
                &url.unwrap_or_default(),
                &table.unwrap_or_default(),
                predicates,
            ),
        }))
    }
}

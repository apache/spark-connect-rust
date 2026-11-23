//! PyO3 wrapper for spark_connect::readwriter::DataFrameReader (`spark.read`).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use spark_connect::dataframe::DataFrame;
use spark_connect::readwriter::DataFrameReader;
use spark_connect_core::error::SparkError;

use crate::dataframe::PyDataFrame;
use crate::errors::{not_expected_type, ResultExt};
use crate::options::{
    check_keyword_keys, option_key, put_option_skip_none, put_reader_option_of_copies,
    put_shared_option, reader_option_text, skip_none_shared, utf32le_of, utf32le_text,
    DefaultedOption, OptionKey, OptionText, UnencodableOptions,
};

// Python wrapper for the batch DataFrameReader.
//
// PySpark's reader is mutable: `format`/`schema`/`option`/`options` change `self`
// and return it, and a reader can be used again after a read. The core reader is a
// consuming builder, so the setters replace the wrapped value in place and return
// the same Python object, and the terminal reads build their DataFrame from a clone.
// As PySpark's plain class, it constructs of a session, `DataFrameReader(spark)`, and is
// subclassed: the setters return the object, of its subclass, and its copies and pickles are
// of its class.
#[doc = connect_doc!(DataFrameReader)]
#[pyclass(
    name = "DataFrameReader",
    module = "pyspark.sql.connect.readwriter",
    subclass,
    weakref,
    dict
)]
pub struct PyDataFrameReader {
    inner: DataFrameReader,
    /// The options of the reader, which a shallow copy of it shares, as PySpark's `_options` dict
    /// is shared by the copies of a reader. The core reader holds none of them (see `state`).
    options: std::sync::Arc<crate::dataframe::WriterState<HashMap<String, String>>>,
    /// The format of `format` when it is not a format of the core reader, which PySpark keeps as it
    /// is ([`UnsentFormat`]). Any other format replaces it.
    unsent_format: Option<UnsentFormat>,
    /// The options that UTF-8 cannot encode, which the copies of this reader share, as they share
    /// `options` ([`UnencodableOptions`]), of a lock that is taken only with the GIL, as Python's
    /// garbage collector reads them (see [`crate::options::SharedObjects`]).
    unencodable_options: std::sync::Arc<crate::gc::GilMutex<UnencodableOptions>>,
    /// The Python object of `unencodable_options`, which the copies of this reader share too, and
    /// which reports the keys and the values of its options that are objects to Python's garbage
    /// collector, as the `_options` dict of PySpark's copies holds them (see
    /// [`crate::options::SharedObjects`]).
    shared: Option<Py<crate::options::PySharedObjects>>,
    /// The UTF-32LE bytes of the schema, when it is a str that UTF-8 cannot encode, which the core
    /// reader leaves unset.
    unencodable_schema: Option<Vec<u8>>,
    /// Set once the value is written, as `__traverse__` reads `unsent_format`, which is not valid
    /// while the reader object is created (see [`crate::gc::GcReady`]).
    gc_ready: crate::gc::GcReady,
}

impl PyDataFrameReader {
    pub fn new(reader: DataFrameReader) -> Self {
        let options = reader.get_options().clone();
        let unencodable_options =
            std::sync::Arc::new(crate::gc::GilMutex::new(UnencodableOptions::default()));
        PyDataFrameReader {
            inner: without_options(reader, &options),
            options: std::sync::Arc::new(crate::dataframe::WriterState::new(options)),
            unsent_format: None,
            shared: crate::options::PySharedObjects::of(unencodable_options.clone()),
            unencodable_options,
            unencodable_schema: None,
            gc_ready: crate::gc::GcReady::new(),
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

    /// Set the option `key` to `value`, or unset it when None, on `r`, as PySpark's `option`:
    /// the entries that UTF-8 cannot encode are kept on this reader, and `r` takes the others.
    fn put_option(
        &self,
        r: DataFrameReader,
        key: &OptionKey,
        value: Option<&OptionText>,
    ) -> DataFrameReader {
        Python::attach(|py| put_shared_option(py, r, &self.unencodable_options, key, value))
    }

    /// Set the option `key` to `value`, or unset it when None, on the options that this reader
    /// shares with its copies, as PySpark's `option` sets its dict: no borrow of the reader is
    /// held while the keys are compared, as Python code ([`put_reader_option_of_copies`]).
    fn put_option_of_copies(slf: &Bound<'_, Self>, key: &OptionKey, value: Option<&OptionText>) {
        let (options, unencodable) = {
            let this = slf.borrow();
            (
                std::sync::Arc::clone(&this.options),
                std::sync::Arc::clone(&this.unencodable_options),
            )
        };
        put_reader_option_of_copies(slf.py(), &options, &unencodable, key, value);
    }

    /// `r` with the options of `options` set, **skipping** those that are None, as
    /// [`crate::options::set_options_skip_none`] does, and keeping the entries that UTF-8 cannot
    /// encode on this reader.
    fn skip_none_options(
        &self,
        r: DataFrameReader,
        options: &[(&str, Option<&Bound<'_, PyAny>>)],
    ) -> PyResult<DataFrameReader> {
        Python::attach(|py| {
            skip_none_shared(
                py,
                r,
                &self.unencodable_options,
                options,
                reader_option_text,
            )
        })
    }

    /// `r` with the option `key` set to `value`, **skipping** a value that is None, as
    /// [`Self::skip_none_options`] does for a key that is not a literal.
    fn skip_none_entry(
        &self,
        r: DataFrameReader,
        key: &OptionKey,
        value: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<DataFrameReader> {
        Python::attach(|py| {
            put_option_skip_none(
                py,
                r,
                &self.unencodable_options,
                key,
                value,
                reader_option_text,
            )
        })
    }

    /// `r` with the option `name` unset when `value` is None, and set otherwise, as PySpark's
    /// `options(name=value)` does.
    fn unset_option_value(
        &self,
        r: DataFrameReader,
        name: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<DataFrameReader> {
        let value = reader_option_text(value)?;
        let key = OptionKey::Text(OptionText::literal(name));
        Ok(self.put_option(r, &key, value.as_ref()))
    }

    /// `r` with the `**options` of a call set, **unsetting** those that are None, as PySpark's
    /// `options` and `load` do.
    fn unset_kwargs(
        &self,
        mut r: DataFrameReader,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<DataFrameReader> {
        check_keyword_keys(options)?;
        for (key, value) in options.into_iter().flat_map(|options| options.iter()) {
            let key = option_key(&key)?;
            let value = reader_option_text(&value)?;
            r = self.put_option(r, &key, value.as_ref());
        }
        Ok(r)
    }

    /// `r` with the schema `schema`, as [`Self::set_schema`] sets it on this reader.
    fn apply_optional_schema(
        &mut self,
        r: DataFrameReader,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<DataFrameReader> {
        match schema {
            Some(schema) => self.set_schema(r, schema),
            None => Ok(r),
        }
    }

    /// `r` with the schema `schema`, which this reader keeps: a str that UTF-8 cannot encode is
    /// kept as its bytes, and fails the plan, as PySpark keeps it until the plan is built.
    fn set_schema(
        &mut self,
        r: DataFrameReader,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<DataFrameReader> {
        match reader_schema(schema)? {
            Ok(text) => {
                self.unencodable_schema = None;
                Ok(r.schema(text))
            }
            Err(utf32le) => {
                self.unencodable_schema = Some(utf32le);
                Ok(r.schema(String::new()))
            }
        }
    }

    /// Fail a `DataSource` of this reader, which PySpark builds at its call and asserts that each
    /// key of its options is a str, with the bare `AssertionError` of that assertion.
    fn check_data_source(&self) -> PyResult<()> {
        if Python::attach(|py| self.unencodable_options.lock(py).has_object_key()) {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        Ok(())
    }

    /// The reader `r` with the options of the bytes keys that are UTF-8, which PySpark's `Read`
    /// plan sends as their text.
    fn with_decodable_keys(&self, r: DataFrameReader) -> DataFrameReader {
        Python::attach(|py| self.unencodable_options.lock(py).decodable_pairs())
            .into_iter()
            .fold(r, |r, (key, value)| r.option(&key, &value))
    }

    /// The error of the plan of this reader, as [`reader_plan_error`] gives it.
    fn plan_error(&self) -> Option<SparkError> {
        Python::attach(|py| {
            reader_plan_error(
                self.unsent_format.as_ref(),
                self.unencodable_schema.as_deref(),
                &self.unencodable_options.lock(py),
            )
        })
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
            let r = self.apply_optional_schema(r, schema)?;
            let parsed = r.clone().parse(parse_format, &df.dataframe).to_pyerr()?;
            let parsed = with_plan_error(parsed, self.plan_error());
            self.set_state(r);
            return Ok(PyDataFrame::new(parsed));
        }
        if path.is_instance_of::<PyString>()
            || path.is_instance_of::<PyList>()
            || parse_format == spark_connect::plan::ParseFormat::Csv
        {
            let r = self.apply_optional_schema(r, schema)?;
            self.commit(&r, format);
            self.check_data_source()?;
            let (paths, error) = paths_of(path)?;
            let df = r.format(format).load_paths(paths);
            let error = self.plan_error().or(error);
            return Ok(PyDataFrame::new(with_plan_error(df, error)));
        }
        // PySpark's options are set before the path is checked, so they stay on this reader.
        self.set_state(r);
        Err(not_expected_type(path, "path", "str, list, or DataFrame"))
    }
}

/// The format that a reader keeps when it is not a str that the core reader takes, as PySpark
/// keeps the object as it is. Another object fails the assertion of its `DataSource` at `load`.
/// A str that UTF-8 cannot encode, such as one with a lone surrogate, fails the plan that `load`
/// builds, with the error of its encoding, when PySpark sends the plan.
pub(crate) struct UnsentFormat {
    pub(crate) object: Py<PyAny>,
    /// The error of the plan, for a str that UTF-8 cannot encode, and None for any other object.
    pub(crate) unencodable: Option<SparkError>,
}

impl UnsentFormat {
    pub(crate) fn of(source: &Bound<'_, PyAny>) -> Self {
        UnsentFormat {
            object: source.clone().unbind(),
            unencodable: unencodable_str(source),
        }
    }

    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        UnsentFormat {
            object: self.object.clone_ref(py),
            unencodable: self.unencodable.clone(),
        }
    }
}

/// The error that PySpark's plan raises when it sends `text`, a str that UTF-8 cannot encode, such
/// as a lone surrogate, which protobuf's string field encodes. None for any other object, or a str
/// that encodes.
fn unencodable_str(text: &Bound<'_, PyAny>) -> Option<SparkError> {
    if !text.is_instance_of::<PyString>() || text.extract::<String>().is_ok() {
        return None;
    }
    let utf32le = text
        .call_method1("encode", ("utf-32-le", "surrogatepass"))
        .ok()?;
    let utf32le = utf32le.cast::<PyBytes>().ok()?;
    Some(SparkError::unencodable_str(utf32le.as_bytes()))
}

/// The DataFrame `df`, which fails each request with `error` when it is one: the error of the plan
/// that PySpark raises when it sends the plan of a read, whose arguments it converts there.
pub(crate) fn with_plan_error(df: DataFrame, error: Option<SparkError>) -> DataFrame {
    match error {
        Some(error) => df.unbuildable(error),
        None => df,
    }
}

/// The schema and the options that a reader keeps for UTF-8 cannot encode, as its reduce gives
/// them back: the schema as a str, or None, and the entries of the options as pairs of strs. A
/// reader of them is set with its public `schema` and `option`, which keep them the same way.
pub(crate) fn unencodable_state<'py>(
    py: Python<'py>,
    schema: Option<&[u8]>,
    options: &UnencodableOptions,
) -> PyResult<Bound<'py, PyAny>> {
    let schema = match schema {
        Some(utf32le) => utf32le_text(py, utf32le)?,
        None => py.None().into_bound(py),
    };
    let entries = PyList::empty(py);
    for entry in options.to_objects(py)? {
        entries.append(entry)?;
    }
    Ok(pyo3::types::PyTuple::new(py, [schema, entries.into_any()])?.into_any())
}

/// The schema and the entries of the options that a reader keeps for UTF-8 cannot encode, as
/// Python objects: the schema, or None, and the pairs of the key and the value of each entry.
pub(crate) type UnencodableState<'py> = (
    Option<Bound<'py, PyAny>>,
    Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
);

/// The schema and the entries of the options of an `unencodable_state`, as their Python objects.
pub(crate) fn unencodable_entries<'py>(
    state: &Bound<'py, PyAny>,
) -> PyResult<UnencodableState<'py>> {
    state.extract()
}

/// The error of the plan of a reader, which PySpark's `DataSource` builds from its format, its
/// schema and its options, in that order: the first of them that UTF-8 cannot encode. `unsent` is
/// the format that the reader keeps, `schema` the UTF-32LE bytes of the schema, if it is one.
pub(crate) fn reader_plan_error(
    unsent: Option<&UnsentFormat>,
    schema: Option<&[u8]>,
    options: &UnencodableOptions,
) -> Option<SparkError> {
    unsent
        .and_then(|format| format.unencodable.clone())
        .or_else(|| schema.map(SparkError::unencodable_str))
        .or_else(|| options.error())
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

/// The reader without the options of `options`, which the reader keeps apart from its core state.
fn without_options(reader: DataFrameReader, options: &HashMap<String, String>) -> DataFrameReader {
    options
        .keys()
        .fold(reader, |reader, key| reader.unset_option(key))
}

/// The reader with the format `source` of PySpark's `format`, and the unsent format that it keeps.
/// None unsets the format. A str sets it. Any other object is kept as it is ([`UnsentFormat`]).
fn reader_format(
    reader: DataFrameReader,
    source: Option<&Bound<'_, PyAny>>,
) -> (DataFrameReader, Option<UnsentFormat>) {
    match source {
        None => (reader.clear_format(), None),
        Some(source) => match source.extract::<String>() {
            Ok(text) => (reader.format(&text), None),
            Err(_) => (reader.clear_format(), Some(UnsentFormat::of(source))),
        },
    }
}

/// The paths of `path`, a path or a list of them, or None for none, as PySpark's `load` takes
/// them, and the error of the plan for the first str that UTF-8 cannot encode. Anything else fails
/// its assertion.
fn paths_of(path: &Bound<'_, PyAny>) -> PyResult<(Vec<String>, Option<SparkError>)> {
    if path.is_none() {
        return Ok((vec![], None));
    }
    if path.is_instance_of::<PyString>() {
        return path_list(std::slice::from_ref(path));
    }
    if !path.is_instance_of::<PyList>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    path_list(&path.try_iter()?.collect::<PyResult<Vec<_>>>()?)
}

/// The paths of `paths`, which PySpark's `load` asserts to be strings, and the error of the plan
/// for the first one that UTF-8 cannot encode. The other paths are sent as they are.
fn path_list(paths: &[Bound<'_, PyAny>]) -> PyResult<(Vec<String>, Option<SparkError>)> {
    if paths.iter().any(|path| !path.is_instance_of::<PyString>()) {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    let mut texts = Vec::with_capacity(paths.len());
    let mut error = None;
    for path in paths {
        match path.extract::<String>() {
            Ok(text) => texts.push(text),
            Err(_) => {
                texts.push(String::new());
                error = error.or_else(|| unencodable_str(path));
            }
        }
    }
    Ok((texts, error))
}

/// The `predicates` of `jdbc`, which PySpark's `DataSource` plan asserts to be a list, and each of
/// its items a str: a bare `AssertionError` for a tuple, a str, or an item that is not a str. The
/// error of the plan is that of the first str that UTF-8 cannot encode, which is kept as an empty
/// predicate.
fn jdbc_predicates(predicates: &Bound<'_, PyAny>) -> PyResult<(Vec<String>, Option<SparkError>)> {
    let Ok(predicates) = predicates.cast::<PyList>() else {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    };
    let mut texts = Vec::new();
    let mut error = None;
    for predicate in predicates.iter() {
        if !predicate.is_instance_of::<PyString>() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        match predicate.extract::<String>() {
            Ok(text) => texts.push(text),
            Err(_) => {
                texts.push(String::new());
                error = error.or_else(|| unencodable_str(&predicate));
            }
        }
    }
    Ok((texts, error))
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

/// A reader `schema` as pyspark's `DataFrameReader.schema` and `DataStreamReader.schema`
/// send it: a StructType as its `json()`, which keeps field metadata, of the methods of a Python
/// subclass, and a string as is, for the server to parse. A str that UTF-8 cannot encode is the
/// `Err` of its UTF-32LE bytes, whose plan fails. Anything else raises `NOT_EXPECTED_TYPE`.
pub(crate) fn reader_schema(schema: &Bound<'_, PyAny>) -> PyResult<Result<String, Vec<u8>>> {
    if schema.is_instance_of::<PyString>() {
        return Ok(match schema.extract::<String>() {
            Ok(text) => Ok(text),
            Err(_) => Err(utf32le_of(schema)?),
        });
    }
    if schema.is_instance_of::<crate::types::PyStructType>() {
        return Ok(Ok(schema
            .call_method0(pyo3::intern!(schema.py(), "json"))?
            .extract()?));
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

#[pymethods]
impl PyDataFrameReader {
    /// `DataFrameReader(client)`, as PySpark Connect's, of the session `client`: the reader that
    /// `spark.read` gives. As the `object.__new__` of PySpark's plain class, it takes the
    /// arguments of the `__init__` of a subclass too, of which the session is the first that is
    /// a session, and none, as an unpickle gives none (see `session::constructor_session`);
    /// `__init__` checks them.
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(client)")]
    fn py_new(
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let session = crate::session::constructor_session(args, kwargs, "client")?;
        Ok(PyDataFrameReader::new(session.read()))
    }

    /// As PySpark's `__init__(client)`, which makes the reader of the session anew: of no
    /// format, schema or options.
    fn __init__(slf: &Bound<'_, Self>, client: PyRef<'_, crate::session::PySparkSession>) {
        let reader = PyDataFrameReader::new(client.session.read());
        *slf.borrow_mut() = reader;
    }

    /// The shallow copy of a reader, as PySpark's `copy.copy`: the copy shares the options of
    /// this reader, as its `_options` dict is shared, so an option that either one sets or
    /// unsets is seen by both, and has its own format, which is kept as it is. It is of the
    /// class of this reader, as the copy of PySpark's of a subclass is.
    fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let this = slf.borrow();
        let copy = crate::reduce::new_of_class(
            &slf.get_type(),
            PyDataFrameReader {
                inner: this.inner.clone(),
                options: std::sync::Arc::clone(&this.options),
                unsent_format: this
                    .unsent_format
                    .as_ref()
                    .map(|format| format.clone_ref(py)),
                unencodable_options: std::sync::Arc::clone(&this.unencodable_options),
                shared: this.shared.as_ref().map(|shared| shared.clone_ref(py)),
                unencodable_schema: this.unencodable_schema.clone(),
                gc_ready: crate::gc::GcReady::new(),
            },
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.as_any())?;
        Ok(copy)
    }

    /// The reduce of a reader, as PySpark's pickle of its session and its state, which
    /// `_of_state` gives back, of the class of the reader (named in the reduce when it is a
    /// subclass), with the instance attributes of the reader.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        use crate::reduce::{optional_object, state_reduce, str_dict};

        let py = slf.py();
        let this = slf.borrow();
        let session = crate::session::session_object(py, &this.inner.spark_session())?;
        // Of a copy of the options, as the Python objects of the state are made apart from its lock.
        let options = this.unencodable_options.lock(py).clone_ref(py);
        let unencodable = unencodable_state(py, this.unencodable_schema.as_deref(), &options)?;
        let args = pyo3::types::PyTuple::new(
            py,
            [
                session.into_bound(py).into_any(),
                this.inner.get_format().into_pyobject(py)?.into_any(),
                this.inner.get_schema().into_pyobject(py)?.into_any(),
                str_dict(py, &this.options.get())?.into_any(),
                optional_object(py, this.unsent_format.as_ref().map(|f| &f.object)),
                unencodable,
            ],
        )?;
        let args = crate::reduce::with_subclass(slf.as_any(), py.get_type::<Self>(), args)?;
        let reduced = state_reduce(py, "reader_reduce", args)?;
        crate::reduce::reduce_with_instance_dict(slf.as_any(), reduced)
    }

    /// The state of a pickle of a reader, which pickle sets on the reader that `__new__` made of
    /// no arguments. Of PySpark's pickle, the state is the `__dict__` of its plain object, which
    /// holds `_options`: so the reader is of the session `_client`, and of the format, the schema
    /// and the options of `_format`, `_schema` and `_options`, which it takes as `format`,
    /// `schema` and `option` take them, and the other entries are its instance attributes. The
    /// state of the drop-in's pickle holds the instance attributes alone (see `__reduce__`), as a
    /// reader's `__dict__` cannot hold `_options`, a property of the reader.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyDict>) -> PyResult<()> {
        let py = slf.py();
        let state = state.copy()?;
        if state.contains("_options")? {
            let take = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
                let value = state.get_item(key)?;
                if value.is_some() {
                    state.del_item(key)?;
                }
                Ok(value)
            };
            if let Some(client) = take("_client")? {
                let session = client.cast_into::<crate::session::PySparkSession>()?;
                let reader = PyDataFrameReader::new(session.borrow().session.read());
                let replaced = std::mem::replace(&mut *slf.borrow_mut(), reader);
                drop(replaced);
            }
            if let Some(format) = take("_format")? {
                let source = (!format.is_none()).then_some(&format);
                Self::format(slf.borrow_mut(), source);
            }
            if let Some(schema) = take("_schema")? {
                Self::schema(slf.borrow_mut(), &schema)?;
            }
            if let Some(options) = take("_options")? {
                for (key, value) in options.cast_into::<PyDict>()?.iter() {
                    Self::option(slf, &key, &value)?;
                }
            }
        }
        slf.getattr(pyo3::intern!(py, "__dict__"))?
            .call_method1(pyo3::intern!(py, "update"), (state,))?;
        Ok(())
    }

    /// A reader of its state, as `__reduce__` gives it, of the class `cls`: of the session
    /// `session`, the format, the schema, the options, the format that is kept (`unsent`), and
    /// the schema and the options that UTF-8 cannot encode (`unencodable`, as
    /// [`unencodable_state`] gives them).
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn _of_state<'py>(
        cls: &Bound<'py, pyo3::types::PyType>,
        session: Py<crate::session::PySparkSession>,
        format: Option<String>,
        schema: String,
        options: HashMap<String, String>,
        unsent: Option<Py<PyAny>>,
        unencodable: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyDataFrameReader>> {
        let py = cls.py();
        let mut reader = session.borrow(py).session.read();
        if let Some(format) = format {
            reader = reader.format(&format);
        }
        if !schema.is_empty() {
            reader = reader.schema(schema);
        }
        let mut result = PyDataFrameReader {
            unsent_format: unsent.map(|object| UnsentFormat::of(object.bind(py))),
            ..PyDataFrameReader::new(reader.options(options))
        };
        let (schema, entries) = unencodable_entries(unencodable.bind(py))?;
        if let Some(schema) = schema {
            let state = result.set_schema(result.state(), &schema)?;
            result.set_state(state);
        }
        for (key, value) in entries {
            let key = option_key(&key)?;
            let value = reader_option_text(&value)?;
            let state = result.put_option(result.state(), &key, value.as_ref());
            result.set_state(state);
        }
        crate::reduce::new_of_class(cls, result)
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

    // Set the source format (e.g. "parquet", "json", "csv"). As PySpark's, None unsets it, so
    // that the server reads its default source, and an object that is not a str is kept until
    // `load` fails its assertion.
    #[doc = connect_doc!(DataFrameReader.format)]
    fn format<'py>(
        mut slf: PyRefMut<'py, Self>,
        source: Option<&Bound<'_, PyAny>>,
    ) -> PyRefMut<'py, Self> {
        let (reader, unsent) = reader_format(slf.state(), source);
        slf.set_state(reader);
        slf.unsent_format = unsent;
        slf
    }

    // Set the schema: a StructType or a DDL string, as in pyspark.
    #[doc = connect_doc!(DataFrameReader.schema)]
    fn schema<'py>(
        mut slf: PyRefMut<'py, Self>,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let state = slf.state();
        let state = slf.set_schema(state, schema)?;
        slf.set_state(state);
        Ok(slf)
    }

    // Set a single read option. `None` unsets the option, as PySpark's does (not the
    // string "None"); booleans lowercase to "true"/"false" (reference `to_str` semantics).
    // PySpark keeps the options in a dict, which hashes every key, and its `DataSource` plan
    // checks the keys of the values that are not None only.
    // The key and the value are converted, and the keys compared, as Python code, with no borrow
    // of the reader held, as PySpark's dict holds none: a key whose `__eq__` releases the GIL or
    // sets an option of this reader again stops no call of another thread, nor its own.
    #[doc = connect_doc!(DataFrameReader.option)]
    fn option<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        // As PySpark's dict: an option of None is kept as None, which the plan leaves out, so an
        // entry of this key is unset.
        let key = option_key(key)?;
        let value = reader_option_text(value)?;
        Self::put_option_of_copies(slf, &key, value.as_ref());
        Ok(slf.clone())
    }

    // Set multiple read options. Mirrors reference `DataFrameReader.options(**options)`
    // - keyword args; a `None` value unsets the option and booleans lowercase.
    // As PySpark's, an `option` of each in turn, so that one that fails to convert keeps those
    // before it, of no borrow of the reader held as `option`.
    #[pyo3(signature = (**options))]
    #[doc = connect_doc!(DataFrameReader.options)]
    fn options<'py>(
        slf: &Bound<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        check_keyword_keys(options)?;
        for (key, value) in options.into_iter().flat_map(|options| options.iter()) {
            let key = option_key(&key)?;
            let value = reader_option_text(&value)?;
            Self::put_option_of_copies(slf, &key, value.as_ref());
        }
        Ok(slf.clone())
    }

    /// The format that the reader keeps, a Python object, which Python's garbage collector sees,
    /// and the Python object of the options that it shares with its copies, which reports their
    /// keys and values that are objects; nothing of a reader object that is being created.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        if let Some(format) = &self.unsent_format {
            visit.call(&format.object)?;
        }
        if let Some(shared) = &self.shared {
            visit.call(shared)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.unsent_format = None;
        self.shared = None;
    }

    /// The session of the reader, as PySpark Connect's `DataFrameReader._client`, which its
    /// pickle keeps.
    #[getter]
    fn _client(&self, py: Python<'_>) -> PyResult<Py<crate::session::PySparkSession>> {
        crate::session::session_object(py, &self.inner.spark_session())
    }

    /// The configured source format (PySpark Connect's `DataFrameReader._format`), which is the
    /// object itself when it is not a str that the core reader takes, as PySpark keeps it.
    #[getter]
    fn _format(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match &self.unsent_format {
            Some(format) => Some(format.object.clone_ref(py)),
            None => self
                .inner
                .get_format()
                .map(|format| PyString::new(py, format).into_any().unbind()),
        }
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

    // Load data from the (optional) path. Mirrors the reference
    // `DataFrameReader.load(path=None, format=None, schema=None, **options)`: the
    // one-shot form applies format/schema/options here before reading, equivalent to
    // chaining `.format(...).schema(...).options(...).load(path)`.
    #[pyo3(signature = (path=None, format=None, schema=None, **options))]
    #[doc = connect_doc!(DataFrameReader.load)]
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
            r = self.set_schema(r, s)?;
        }
        self.set_state(r.clone());
        r = self.unset_kwargs(r, options)?;
        // Then its `DataSource` asserts the format, which precedes the check of the paths.
        self.set_state(r.clone());
        // A format that is not a str fails the assertion. A str that UTF-8 cannot encode fails the
        // plan, which precedes the paths in the plan of PySpark's `DataSource`.
        if self
            .unsent_format
            .as_ref()
            .is_some_and(|format| format.unencodable.is_none())
        {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        self.check_data_source()?;
        let (paths, path_error) = path.map(paths_of).transpose()?.unwrap_or_default();
        let df = r.load_paths(paths);
        Ok(PyDataFrame::new(with_plan_error(
            df,
            self.plan_error().or(path_error),
        )))
    }

    // Read a table by name, which PySpark's `Read` plan sets in its string field as it is, as
    // [`crate::str_argument`] converts it, as `changes` and those of the stream reader do. A
    // name or an option that UTF-8 cannot encode fails the plan.
    #[doc = connect_doc!(DataFrameReader.table)]
    fn table(&self, tableName: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let (name, error) = table_name_of(tableName)?;
        let py = tableName.py();
        let error = error.or_else(|| self.unencodable_options.lock(py).error());
        let r = self.with_decodable_keys(self.state());
        Ok(PyDataFrame::new(with_plan_error(r.table(&name), error)))
    }

    // Read the CDC changes of a named table. Mirrors `DataFrameReader.changes`, which raises an
    // `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` for a reader of a schema.
    #[allow(non_snake_case)]
    #[doc = connect_doc!(DataFrameReader.changes)]
    fn changes(&self, py: Python<'_>, tableName: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        if !self.inner.get_schema().is_empty() || self.unencodable_schema.is_some() {
            return Err(changes_of_schema_error(py));
        }
        let (name, error) = table_name_of(tableName)?;
        let error = error.or_else(|| self.unencodable_options.lock(py).error());
        let r = self.with_decodable_keys(self.state());
        Ok(PyDataFrame::new(with_plan_error(r.changes(&name), error)))
    }

    // Read JSON file(s) - full pyspark signature; each named option is
    // applied as a read option when provided.
    #[pyo3(signature = (path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, samplingRatio=None, dropFieldIfAllNull=None, encoding=None, locale=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, allowNonNumericNumbers=None, useUnsafeRow=None, **unexpected), text_signature = "($self, path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, samplingRatio=None, dropFieldIfAllNull=None, encoding=None, locale=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, allowNonNumericNumbers=None, useUnsafeRow=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataFrameReader.json)]
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
        r = self.skip_none_options(
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

    // Read Parquet file(s). As PySpark's `parquet(*paths, **options)`, only the options
    // `mergeSchema`, `pathGlobFilter`, `recursiveFileLookup`, `modifiedBefore`,
    // `modifiedAfter`, `datetimeRebaseMode` and `int96RebaseMode` of `options` are set, and
    // the others are ignored.
    #[pyo3(signature = (*paths, **options))]
    #[doc = connect_doc!(DataFrameReader.parquet)]
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
            r = self.skip_none_options(r, &[(name, value.as_ref())])?;
        }
        self.commit(&r, "parquet");
        self.check_data_source()?;
        let (paths, error) = path_list(&paths)?;
        let df = r.format("parquet").load_paths(paths);
        let error = self.plan_error().or(error);
        Ok(PyDataFrame::new(with_plan_error(df, error)))
    }

    // Read CSV file(s) - full pyspark signature; each named option is
    // applied as a read option when provided.
    #[pyo3(signature = (path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, samplingRatio=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, unescapedQuoteHandling=None, **unexpected), text_signature = "($self, path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, samplingRatio=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, unescapedQuoteHandling=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataFrameReader.csv)]
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
        r = self.skip_none_options(
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

    // Read ORC file(s) - full pyspark signature; each named option is
    // applied as a read option when provided.
    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, **unexpected), text_signature = "($self, path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataFrameReader.orc)]
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
        r = self.skip_none_options(
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
        self.check_data_source()?;
        let (paths, error) = paths_of(path)?;
        let df = r.format("orc").load_paths(paths);
        let error = self.plan_error().or(error);
        Ok(PyDataFrame::new(with_plan_error(df, error)))
    }

    // Read TEXT file(s) - full pyspark signature; each named option is
    // applied as a read option when provided. As PySpark's `wholetext=False`, an omitted
    // `wholetext` sets `false`, also over an earlier `wholetext` option.
    #[pyo3(
        signature = (paths, wholetext=DefaultedOption::Omitted, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, **unexpected),
        text_signature = "($self, paths, wholetext=False, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None)"
    )]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataFrameReader.text)]
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
        r = self.skip_none_options(
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
        self.check_data_source()?;
        let (paths, error) = paths_of(paths)?;
        let df = r.format("text").load_paths(paths);
        let error = self.plan_error().or(error);
        Ok(PyDataFrame::new(with_plan_error(df, error)))
    }

    // Read XML file(s) - full pyspark signature; each named option is
    // applied as a read option when provided.
    #[pyo3(signature = (path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None, **unexpected), text_signature = "($self, path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None)")]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataFrameReader.xml)]
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
        r = self.skip_none_options(r, &[("rowTag", rowTag)])?;
        r = self.apply_optional_schema(r, schema)?;
        r = self.skip_none_options(
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

    // Read from a JDBC source. Mirrors `DataFrameReader.jdbc(url, table, column=None,
    // lowerBound=None, upperBound=None, numPartitions=None, predicates=None, properties=None)`,
    // which keeps the format, then the options, on the reader:
    // - of a `column`, it asserts the bounds and `numPartitions`, and reads the partitions of
    //   the column, ignoring `predicates`;
    // - else, it reads the partitions of `predicates`, ignoring the bounds and `numPartitions`.
    //
    // The column, the bounds, `numPartitions` and the values of `properties` are options, of any
    // type (None skipped, bools lowercased).
    #[pyo3(signature = (url, table, column=None, lowerBound=None, upperBound=None, numPartitions=None, predicates=None, properties=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    #[doc = connect_doc!(DataFrameReader.jdbc)]
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
                r = self.skip_none_options(
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
                let key = option_key(&key)?;
                r = self.skip_none_entry(r, &key, Some(&value))?;
            }
        }
        // PySpark's jdbc also sets url and dbtable as options, `options(url=url, dbtable=table)`,
        // which leaves an option of None unset, before it asserts the predicates.
        r = self.unset_option_value(r, "url", url)?;
        r = self.unset_option_value(r, "dbtable", table)?;
        self.set_state(r.clone());
        let (predicates, predicate_error) = match predicates {
            Some(predicates) => {
                let (texts, error) = jdbc_predicates(predicates)?;
                (Some(texts), error)
            }
            None => (None, None),
        };
        // A url or a dbtable that UTF-8 cannot encode is an option of the plan, which fails it.
        let url = reader_option_text(url)?.and_then(OptionText::into_text);
        let table = reader_option_text(table)?.and_then(OptionText::into_text);
        let df = match (url, table, predicates) {
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
        };
        let error = self.plan_error().or(predicate_error);
        Ok(PyDataFrame::new(with_plan_error(df, error)))
    }
}

//! The options of the readers, the writers and the catalog, as PySpark converts them.
//!
//! PySpark treats an option of `None` in two ways, and each helper here is named after the one
//! it implements:
//!
//! - **skip**: the `None` is left out, and whatever the builder already holds under that key
//!   stays. This is `OptionUtils._set_opts`, which the format methods (`csv`, `json`, ...) pass
//!   their keyword arguments to, and the `option` and `options` of the writers and of the
//!   stream reader.
//! - **unset**: the `None` removes the key from the builder. This is `DataFrameReader.option`,
//!   `options` and `load`.
//!
//! Every other value goes through [`coerce_option_value`], or through [`stream_option_value`] for
//! the option and the format methods of the stream reader, which PySpark converts differently.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyString};
use spark_connect::readwriter::{DataFrameReader, DataFrameWriter};
use spark_connect::streaming::DataStreamReader;

/// Coerce a Python option value the way reference pyspark's `to_str` does:
/// `None` -> `None` (the option is left unset, NOT the literal string "None"), a
/// bool -> lowercase `"true"`/`"false"` (not Python's `"True"`/`"False"`), and
/// everything else -> its `str()`. Used by every reader/writer/conf `option(s)`
/// binding so option handling matches the reference client.
pub(crate) fn coerce_option_value(v: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if v.is_none() {
        return Ok(None);
    }
    if let Ok(b) = v.cast::<PyBool>() {
        return Ok(Some(if b.is_true() { "true" } else { "false" }.to_string()));
    }
    Ok(Some(str_text(v)?))
}

/// The `str()` of `v`, as the reference client sends it in a protobuf string field, which raises
/// the `UnicodeEncodeError` of a str that UTF-8 cannot encode (a lone surrogate), as
/// [`str_argument`], where a lossy conversion would send another text, such as another path.
pub(crate) fn str_text(v: &Bound<'_, PyAny>) -> PyResult<String> {
    str_argument(v.str()?.as_any())
}

/// The text of `v` that PySpark sets in a protobuf string field as it is, as protobuf's default
/// (upb) backend converts it: a str as its UTF-8, and bytes decoded as UTF-8, with the error of
/// that conversion: the `UnicodeEncodeError` of a str that UTF-8 cannot encode (a lone
/// surrogate), the `UnicodeDecodeError` of bytes that are not UTF-8, and the `TypeError` of
/// `PyUnicode_AsUTF8AndSize` for any other object, such as an int, a bytearray or None.
pub(crate) fn str_argument(v: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = v.cast::<PyString>() {
        return text.extract();
    }
    if v.is_instance_of::<PyBytes>() {
        // As upb's `PyUnicode_FromEncodedObject(obj, "utf-8", NULL)`.
        return PyString::from_encoded_object(v, Some(c"utf-8"), None)?.extract();
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "bad argument type for built-in operation",
    ))
}

/// The keys of the `**options` of a call, which PySpark's Python function takes: a key that is not
/// a str raises the `TypeError` of CPython's unpacking, `keywords must be strings`, before the call
/// does anything. Every method that takes `**options` checks its keys first, with this.
pub(crate) fn check_keyword_keys(options: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    if let Some(options) = options {
        if options
            .keys()
            .iter()
            .any(|key| !key.is_instance_of::<PyString>())
        {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "keywords must be strings",
            ));
        }
    }
    Ok(())
}

/// The keywords of a method that has a catch-all `**unexpected` for the keywords it does not take,
/// which keeps them out of its signature: the keys are checked as [`check_keyword_keys`] does, then
/// the first key raises CPython's `TypeError` of an unexpected keyword argument, with `method` as
/// the qualified name of the method. The message joins the key in Python, as CPython formats it:
/// a str key need not have UTF-8, such as one with a lone surrogate.
pub(crate) fn check_unexpected_keywords(
    method: &str,
    unexpected: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    check_keyword_keys(unexpected)?;
    if let Some(key) = unexpected.and_then(|dict| dict.keys().iter().next()) {
        let message = PyString::new(
            key.py(),
            &format!("{method}() got an unexpected keyword argument '"),
        )
        .add(&key)?
        .add("'")?;
        return Err(pyo3::exceptions::PyTypeError::new_err(message.unbind()));
    }
    Ok(())
}

/// The key of an option that PySpark keeps in a dict, which raises the `TypeError` of an
/// unhashable key, and sends in a protobuf string field as it is, as [`str_argument`] converts it:
/// the options of the writers, and of `SparkSession.builder.config`.
pub(crate) fn option_dict_key(key: &Bound<'_, PyAny>) -> PyResult<String> {
    key.hash()?;
    str_argument(key)
}

/// The key of an option of `DataFrameReader.option` and `DataStreamReader.option`, which PySpark
/// keeps in a dict, which raises the `TypeError` of an unhashable key, and whose `DataSource` plan
/// asserts that each key is a str, a bare `AssertionError` for any other key, bytes too.
pub(crate) fn reader_option_key(key: &Bound<'_, PyAny>) -> PyResult<String> {
    key.hash()?;
    if !key.is_instance_of::<PyString>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    str_argument(key)
}

/// A reader or writer that takes string options one at a time.
pub(crate) trait OptionBuilder: Sized {
    fn with_option(self, key: &str, value: &str) -> Self;
}

impl OptionBuilder for DataFrameReader {
    fn with_option(self, key: &str, value: &str) -> Self {
        self.option(key, value)
    }
}

impl OptionBuilder for DataFrameWriter {
    fn with_option(self, key: &str, value: &str) -> Self {
        self.option(key, value)
    }
}

impl OptionBuilder for DataStreamReader {
    fn with_option(self, key: &str, value: &str) -> Self {
        self.option(key, value)
    }
}

/// `builder` with the option `name` of `value`, **skipping** a `value` that is absent or None.
pub(crate) fn set_option_skip_none<B: OptionBuilder>(
    builder: B,
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<B> {
    match value {
        Some(value) => match coerce_option_value(value)? {
            Some(value) => Ok(builder.with_option(name, &value)),
            None => Ok(builder),
        },
        None => Ok(builder),
    }
}

/// `builder` with each of `options` in order, **skipping** those that are absent or None, as
/// [`set_option_skip_none`] one after the other: the first value that fails to convert raises.
pub(crate) fn set_options_skip_none<B: OptionBuilder>(
    mut builder: B,
    options: &[(&str, Option<&Bound<'_, PyAny>>)],
) -> PyResult<B> {
    for (name, value) in options {
        builder = set_option_skip_none(builder, name, *value)?;
    }
    Ok(builder)
}

/// The `**options` of a call as a map of strings, **skipping** those that are None.
pub(crate) fn kwargs_skip_none(
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, String>> {
    check_keyword_keys(options)?;
    let mut map = HashMap::new();
    if let Some(options) = options {
        for (key, value) in options.iter() {
            if let Some(value) = coerce_option_value(&value)? {
                map.insert(str_text(&key)?, value);
            }
        }
    }
    Ok(map)
}

/// `reader` with the option `key` of `value`, **unsetting** the key of a `value` of None, as
/// PySpark's `DataFrameReader.option`.
pub(crate) fn set_reader_option_unset_none(
    reader: DataFrameReader,
    key: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<DataFrameReader> {
    Ok(match coerce_option_value(value)? {
        Some(value) => reader.option(key, &value),
        None => reader.unset_option(key),
    })
}

/// `reader` with the `**options` of a call, **unsetting** the keys of those that are None, as
/// PySpark's `DataFrameReader.options` and `load`.
pub(crate) fn set_reader_kwargs_unset_none(
    mut reader: DataFrameReader,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<DataFrameReader> {
    check_keyword_keys(options)?;
    if let Some(options) = options {
        for (key, value) in options.iter() {
            let key = str_text(&key)?;
            reader = set_reader_option_unset_none(reader, &key, &value)?;
        }
    }
    Ok(reader)
}

/// The `**options` of `Catalog.createTable`, **skipping** those that are None. Unlike the
/// options of the readers and writers these are not converted: PySpark sets them in a protobuf
/// map of strings as they are, which takes a str, and bytes that are UTF-8, which it decodes, and
/// raises the error of [`str_argument`] for any other value.
pub(crate) fn str_kwargs_skip_none(
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, String>> {
    check_keyword_keys(options)?;
    let mut map = HashMap::new();
    for (key, value) in options.into_iter().flat_map(|options| options.iter()) {
        if value.is_none() {
            continue;
        }
        // The key first, as protobuf sets an entry of a map.
        let key = str_argument(&key)?;
        map.insert(key, str_argument(&value)?);
    }
    Ok(map)
}

/// The `_set_opts` of the readers and writers, as PySpark's `OptionUtils._set_opts`: the schema of
/// `slf`, if any, and its options of `options`, **skipping** those that are None, through its own
/// Python `schema` and `option`, which convert them and raise their errors.
pub(crate) fn set_opts(
    slf: &Bound<'_, PyAny>,
    schema: Option<&Bound<'_, PyAny>>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    check_keyword_keys(options)?;
    if let Some(schema) = schema.filter(|schema| !schema.is_none()) {
        slf.call_method1("schema", (schema,))?;
    }
    if let Some(options) = options {
        for (key, value) in options.iter() {
            if !value.is_none() {
                slf.call_method1("option", (key, value))?;
            }
        }
    }
    Ok(())
}

/// The text of an option of the stream reader, as `DataStreamReader.option` and its format methods
/// send it: the `str()` of the value, so a bool is `"True"`, where `options` and `load` lowercase
/// it through `to_str`. None is left out. A str that UTF-8 cannot encode raises, as
/// [`str_text`].
pub(crate) fn stream_option_value(v: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if v.is_none() {
        return Ok(None);
    }
    Ok(Some(str_text(v)?))
}

/// An option argument of a format method whose Python default is not None, which PySpark's
/// `_set_opts` then sets on every call, over an earlier option of the same name: `text`'s
/// `wholetext=False`. An omitted argument is that default, and None is skipped, as PySpark
/// tells them apart.
pub(crate) enum DefaultedOption {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for DefaultedOption {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(DefaultedOption::Given(obj.to_owned().unbind()))
    }
}

impl DefaultedOption {
    /// The value of the argument, `default` when it is omitted.
    pub(crate) fn or_bool<'py>(&self, py: Python<'py>, default: bool) -> Bound<'py, PyAny> {
        match self {
            DefaultedOption::Omitted => PyBool::new(py, default).to_owned().into_any(),
            DefaultedOption::Given(value) => value.bind(py).clone(),
        }
    }
}

/// `builder` with the option `name` of `value` for the stream reader, **skipping** a `value` that
/// is absent or None, as [`set_option_skip_none`] does with [`stream_option_value`].
pub(crate) fn set_stream_option_skip_none(
    builder: DataStreamReader,
    name: &str,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<DataStreamReader> {
    match value {
        Some(value) => match stream_option_value(value)? {
            Some(value) => Ok(builder.option(name, &value)),
            None => Ok(builder),
        },
        None => Ok(builder),
    }
}

/// `builder` with each of `options` in order for the stream reader, **skipping** those that are
/// absent or None, as [`set_options_skip_none`] does with [`set_stream_option_skip_none`].
pub(crate) fn set_stream_options_skip_none(
    mut builder: DataStreamReader,
    options: &[(&str, Option<&Bound<'_, PyAny>>)],
) -> PyResult<DataStreamReader> {
    for (name, value) in options {
        builder = set_stream_option_skip_none(builder, name, *value)?;
    }
    Ok(builder)
}

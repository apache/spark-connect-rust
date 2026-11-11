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
    Ok(Some(v.str()?.to_string()))
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
    let mut map = HashMap::new();
    if let Some(options) = options {
        for (key, value) in options.iter() {
            if let Some(value) = coerce_option_value(&value)? {
                map.insert(key.str()?.to_string(), value);
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
    if let Some(options) = options {
        for (key, value) in options.iter() {
            let key = key.str()?.to_string();
            reader = set_reader_option_unset_none(reader, &key, &value)?;
        }
    }
    Ok(reader)
}

/// The `**options` of `Catalog.createTable`, **skipping** those that are None. Unlike the
/// options of the readers and writers these are not converted: PySpark sets them in a protobuf
/// map of strings as they are. That map takes a str, and bytes that are UTF-8, which it decodes;
/// it raises a `TypeError` for any other type, and a `ValueError` for bytes that are not UTF-8.
pub(crate) fn str_kwargs_skip_none(
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, String>> {
    let mut map = HashMap::new();
    for (key, value) in options.into_iter().flat_map(|options| options.iter()) {
        if value.is_none() {
            continue;
        }
        let text = if let Ok(text) = value.cast::<PyString>() {
            text.to_cow()?.into_owned()
        } else if let Ok(bytes) = value.cast::<PyBytes>() {
            match std::str::from_utf8(bytes.as_bytes()) {
                Ok(text) => text.to_string(),
                Err(_) => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "{} has type bytes, but isn't valid UTF-8 encoding. Non-UTF-8 strings \
                         must be converted to unicode objects before being added.",
                        value.repr()?
                    )));
                }
            }
        } else {
            // As the protobuf map of strings that PySpark sets the options in.
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{} has type {}, but expected one of: (<class 'bytes'>, <class 'str'>)",
                value.repr()?,
                value.get_type().str()?
            )));
        };
        map.insert(key.extract()?, text);
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
/// it through `to_str`. None is left out.
pub(crate) fn stream_option_value(v: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if v.is_none() {
        return Ok(None);
    }
    Ok(Some(v.str()?.to_string()))
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

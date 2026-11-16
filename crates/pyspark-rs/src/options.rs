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
//! Every other value goes through [`coerce_option_value`], or through [`stream_option_text`] for
//! the option and the format methods of the stream reader, which PySpark converts differently.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use spark_connect::readwriter::{DataFrameReader, DataFrameWriter, DataFrameWriterV2};
use spark_connect::streaming::DataStreamReader;
use spark_connect_core::error::SparkError;

use crate::dataframe::WriterState;

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

// ---- The options that UTF-8 cannot encode, and the keys that are not strs ----
//
// PySpark keeps an option in a dict, and its plan encodes the strs of the dict when the plan is
// built: a key or a value with a lone surrogate fails the request that sends the plan, not the call
// that sets it. A key that is not a str fails the plan too, or, for a method whose `DataSource`
// asserts its keys, the call of that method. The core holds only the strs that encode, and the
// other entries are kept here, in the order of the dict, with the error that PySpark raises.

/// The UTF-32LE bytes of a str, which keep its lone surrogates: the form of
/// [`SparkError::unencodable_str`].
pub(crate) fn utf32le_of(text: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let encoded = text.call_method1("encode", ("utf-32-le", "surrogatepass"))?;
    Ok(encoded.cast::<PyBytes>()?.as_bytes().to_vec())
}

/// The str of UTF-32LE bytes of a str, as Python makes it, which keeps its lone surrogates.
pub(crate) fn utf32le_text<'py>(py: Python<'py>, utf32le: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    PyBytes::new(py, utf32le).call_method1("decode", ("utf-32-le", "surrogatepass"))
}

/// The text of an option or of its key: its UTF-32LE bytes, and the text that the core takes,
/// which is None for a str that UTF-8 cannot encode.
#[derive(Clone)]
pub(crate) struct OptionText {
    utf32le: Vec<u8>,
    pub(crate) text: Option<String>,
}

impl OptionText {
    /// The text of `text`, a str.
    pub(crate) fn of(text: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(OptionText {
            utf32le: utf32le_of(text)?,
            text: text.extract::<String>().ok(),
        })
    }

    /// The text of a name that this crate gives, which is ASCII.
    pub(crate) fn literal(name: &str) -> Self {
        OptionText {
            utf32le: name
                .chars()
                .flat_map(|c| (c as u32).to_le_bytes())
                .collect(),
            text: Some(name.to_string()),
        }
    }

    /// The text that the core reader takes, None for a str that UTF-8 cannot encode.
    pub(crate) fn into_text(self) -> Option<String> {
        self.text
    }

    /// The text of the value, as a Python str.
    fn to_object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        utf32le_text(py, &self.utf32le)
    }
}

/// What a key that is not a str is, which decides the error of its plan.
#[derive(Clone)]
pub(crate) enum ObjectKind {
    /// Bytes that are UTF-8, which protobuf sends as their text.
    Bytes(OptionText),
    /// Bytes that are not UTF-8, whose decoding error the plan raises.
    InvalidBytes(Vec<u8>),
    /// Any other object, which the plan rejects with a TypeError.
    Other,
}

/// The key of an option: a str, or another object, which PySpark keeps in its dict as it is.
#[derive(Clone)]
pub(crate) enum OptionKey {
    Text(OptionText),
    Object {
        object: std::sync::Arc<Py<PyAny>>,
        kind: ObjectKind,
    },
}

impl OptionKey {
    /// The text of the key that the core takes, for a str that encodes, and for bytes that are
    /// UTF-8; None for any other key.
    pub(crate) fn core_text(&self) -> Option<&OptionText> {
        match self {
            OptionKey::Text(text) => text.text.is_some().then_some(text),
            OptionKey::Object {
                kind: ObjectKind::Bytes(text),
                ..
            } => Some(text),
            OptionKey::Object { .. } => None,
        }
    }

    /// Whether `other` is the same key, as PySpark's dict takes it: the same str, or an object
    /// that is equal to it.
    fn same_as(&self, other: &OptionKey) -> bool {
        match (self, other) {
            (OptionKey::Text(a), OptionKey::Text(b)) => a.utf32le == b.utf32le,
            (OptionKey::Object { object: a, .. }, OptionKey::Object { object: b, .. }) => {
                Python::attach(|py| {
                    pyo3::types::PyAnyMethods::eq(a.bind(py), b.bind(py)).unwrap_or(false)
                })
            }
            _ => false,
        }
    }

    /// The key as the object that PySpark keeps.
    fn to_object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            OptionKey::Text(text) => text.to_object(py),
            OptionKey::Object { object, .. } => Ok(object.bind(py).clone()),
        }
    }
}

/// The key of an option, as PySpark's `option` keeps it in a dict, which hashes it first, so that
/// an unhashable key raises its TypeError at the call.
pub(crate) fn option_key(key: &Bound<'_, PyAny>) -> PyResult<OptionKey> {
    key.hash()?;
    option_object(key)
}

/// The object of an option, a key or a value, as PySpark's dict keeps it, without hashing it: a
/// str, bytes, or another object, which the plan rejects with a TypeError.
pub(crate) fn option_object(object: &Bound<'_, PyAny>) -> PyResult<OptionKey> {
    if object.is_instance_of::<PyString>() {
        return Ok(OptionKey::Text(OptionText::of(object)?));
    }
    let kind = if object.is_instance_of::<PyBytes>() {
        match object.call_method1("decode", ("utf-8",)) {
            Ok(text) => ObjectKind::Bytes(OptionText::of(&text)?),
            Err(_) => ObjectKind::InvalidBytes(object.cast::<PyBytes>()?.as_bytes().to_vec()),
        }
    } else {
        ObjectKind::Other
    };
    Ok(OptionKey::Object {
        object: std::sync::Arc::new(object.clone().unbind()),
        kind,
    })
}

/// The error that PySpark's plan raises for a key that is not a str, which protobuf rejects.
pub(crate) fn builtin_type_error() -> SparkError {
    SparkError {
        kind: spark_connect_core::error::SparkErrorKind::BuiltinTypeError,
        ..SparkError::value_msg("bad argument type for built-in operation")
    }
}

/// The error that the plan raises for a value that the core does not take, as PySpark's command
/// encodes it: a str that does not encode, bytes that are not UTF-8, or another object.
fn value_error(value: &OptionKey) -> Option<SparkError> {
    match value {
        OptionKey::Text(text) if text.text.is_none() => {
            Some(SparkError::unencodable_str(&text.utf32le))
        }
        OptionKey::Text(_)
        | OptionKey::Object {
            kind: ObjectKind::Bytes(_),
            ..
        } => None,
        OptionKey::Object {
            kind: ObjectKind::InvalidBytes(bytes),
            ..
        } => Some(SparkError::invalid_utf8(0, bytes)),
        OptionKey::Object {
            kind: ObjectKind::Other,
            ..
        } => Some(builtin_type_error()),
    }
}

/// The error that the plan raises for an entry: PySpark encodes the key first, then the value.
fn entry_error(key: &OptionKey, value: Option<&OptionKey>) -> Option<SparkError> {
    let bad_value = || value.and_then(value_error);
    match key {
        OptionKey::Text(text) if text.text.is_none() => {
            Some(SparkError::unencodable_str(&text.utf32le))
        }
        OptionKey::Text(_) => bad_value(),
        OptionKey::Object {
            kind: ObjectKind::Bytes(_),
            ..
        } => bad_value(),
        OptionKey::Object {
            kind: ObjectKind::InvalidBytes(bytes),
            ..
        } => Some(SparkError::invalid_utf8(0, bytes)),
        OptionKey::Object {
            kind: ObjectKind::Other,
            ..
        } => Some(builtin_type_error()),
    }
}

/// The entries of the options of a reader, writer, builder or table properties that the core
/// cannot take, in the order of PySpark's dict: a key that is not a str that encodes, or a value
/// that the core does not take, which is a str that does not encode, or an object that is not
/// a str or bytes that are UTF-8. The plan fails with the error of the first entry.
#[derive(Clone, Default)]
pub(crate) struct UnencodableOptions(Vec<(OptionKey, Option<OptionKey>)>);

impl UnencodableOptions {
    /// Whether an entry of `key` and `value` is kept: a key that is not a str, or a str that does
    /// not encode, or a value that the core does not take.
    fn kept(key: &OptionKey, value: Option<&OptionKey>) -> bool {
        match key {
            OptionKey::Text(text) => {
                text.text.is_none() || value.is_some_and(|v| v.core_text().is_none())
            }
            OptionKey::Object { .. } => true,
        }
    }

    fn position(&self, key: &OptionKey) -> Option<usize> {
        self.0.iter().position(|(entry, _)| entry.same_as(key))
    }

    /// Remove the entry of `key`, which is set to a value that the core takes, or unset.
    pub(crate) fn remove(&mut self, key: &OptionKey) {
        if let Some(position) = self.position(key) {
            self.0.remove(position);
        }
    }

    /// Set the entry of `key` to `value`, as PySpark's dict is set: an entry keeps its place, and
    /// is removed when neither its key nor its value needs it.
    pub(crate) fn set(&mut self, key: &OptionKey, value: Option<OptionKey>) {
        if !Self::kept(key, value.as_ref()) {
            self.remove(key);
            return;
        }
        match self.position(key) {
            Some(position) => self.0[position] = (key.clone(), value),
            None => self.0.push((key.clone(), value)),
        }
    }

    /// Whether an entry has a key that is not a str, which a reader's `DataSource` asserts.
    pub(crate) fn has_object_key(&self) -> bool {
        self.0
            .iter()
            .any(|(key, _)| matches!(key, OptionKey::Object { .. }))
    }

    /// The options of the bytes keys that are UTF-8, with their values, which PySpark's plan
    /// sends as the text of the key, for the values that encode.
    pub(crate) fn decodable_pairs(&self) -> Vec<(String, String)> {
        self.0
            .iter()
            .filter_map(|(key, value)| match (key, value) {
                (
                    OptionKey::Object {
                        kind: ObjectKind::Bytes(name),
                        ..
                    },
                    Some(value),
                ) => Some((name.text.clone()?, value.core_text()?.text.clone()?)),
                _ => None,
            })
            .collect()
    }

    /// The error of the plan for the first entry, or None when every entry encodes.
    pub(crate) fn error(&self) -> Option<SparkError> {
        self.0
            .iter()
            .find_map(|(key, value)| entry_error(key, value.as_ref()))
    }

    /// The entries as a list of pairs of Python objects, which a reduce gives back.
    pub(crate) fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let entries = PyList::empty(py);
        for (key, value) in self.to_objects(py)? {
            entries.append(PyTuple::new(py, [key, value])?)?;
        }
        Ok(entries)
    }

    /// The entries as Python objects, which a reduce gives back: the key, and the value, if any.
    pub(crate) fn to_objects<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        self.0
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    Some(value) => value.to_object(py)?,
                    None => py.None().into_bound(py),
                };
                Ok((key.to_object(py)?, value))
            })
            .collect()
    }
}

/// A reader that takes its options one at a time, and unsets one of them.
pub(crate) trait ReaderOptionBuilder: OptionBuilder {
    fn without_option(self, key: &str) -> Self;
}

impl ReaderOptionBuilder for DataFrameReader {
    fn without_option(self, key: &str) -> Self {
        self.unset_option(key)
    }
}

impl ReaderOptionBuilder for DataStreamReader {
    fn without_option(self, key: &str) -> Self {
        self.unset_option(key)
    }
}

/// `builder` with the option `key` set to `value`, or unset when `value` is None, as a reader's
/// `option` sets it. The core reader takes only a str key whose value encodes, and the other
/// entries are kept in `unencodable`.
pub(crate) fn put_reader_option<B: ReaderOptionBuilder>(
    builder: B,
    unencodable: &mut UnencodableOptions,
    key: &OptionKey,
    value: Option<&OptionText>,
) -> B {
    match value {
        Some(value) => unencodable.set(key, Some(OptionKey::Text(value.clone()))),
        None => unencodable.remove(key),
    }
    match key {
        OptionKey::Text(OptionText {
            text: Some(name), ..
        }) => match value {
            Some(OptionText {
                text: Some(value), ..
            }) => builder.with_option(name, value),
            _ => builder.without_option(name),
        },
        _ => builder,
    }
}

/// `builder` with the option `key` set to `value`, as a writer's `option` sets it: the core writer
/// takes a key that is a str, or bytes that are UTF-8, with a value that encodes, and the other
/// entries are kept in `unencodable`.
pub(crate) fn put_writer_option<B: OptionBuilder>(
    builder: B,
    unencodable: &mut UnencodableOptions,
    key: &OptionKey,
    value: &OptionText,
) -> B {
    match (
        key.core_text().and_then(|name| name.text.as_deref()),
        value.text.as_deref(),
    ) {
        (Some(name), Some(value)) => {
            unencodable.remove(key);
            builder.with_option(name, value)
        }
        _ => {
            unencodable.set(key, Some(OptionKey::Text(value.clone())));
            builder
        }
    }
}

/// `builder` with the option `key` set to `value`, or unset when `value` is None, on a reader whose
/// state is shared by its copies, as [`put_reader_option`] does.
pub(crate) fn put_shared_option<B: ReaderOptionBuilder>(
    builder: B,
    shared: &WriterState<UnencodableOptions>,
    key: &OptionKey,
    value: Option<&OptionText>,
) -> B {
    let mut unencodable = shared.get();
    let builder = put_reader_option(builder, &mut unencodable, key, value);
    shared.set(unencodable);
    builder
}

/// `builder` with the option `key` set to `value`, converted by `to_text`, **skipping** a `value`
/// that is absent or None, as [`set_option_skip_none`] does.
pub(crate) fn put_option_skip_none<B: ReaderOptionBuilder>(
    builder: B,
    unencodable: &mut UnencodableOptions,
    key: &OptionKey,
    value: Option<&Bound<'_, PyAny>>,
    to_text: fn(&Bound<'_, PyAny>) -> PyResult<Option<OptionText>>,
) -> PyResult<B> {
    let Some(value) = value else {
        return Ok(builder);
    };
    match to_text(value)? {
        Some(text) => Ok(put_reader_option(builder, unencodable, key, Some(&text))),
        None => Ok(builder),
    }
}

/// `builder` with each of `options` set, **skipping** those that are None, as
/// [`put_option_skip_none`] does for each one, on a reader whose state is shared by its copies.
pub(crate) fn skip_none_shared<B: ReaderOptionBuilder>(
    mut builder: B,
    shared: &WriterState<UnencodableOptions>,
    options: &[(&str, Option<&Bound<'_, PyAny>>)],
    to_text: fn(&Bound<'_, PyAny>) -> PyResult<Option<OptionText>>,
) -> PyResult<B> {
    let mut unencodable = shared.get();
    for (name, value) in options {
        let key = OptionKey::Text(OptionText::literal(name));
        builder = put_option_skip_none(builder, &mut unencodable, &key, *value, to_text)?;
    }
    shared.set(unencodable);
    Ok(builder)
}

/// The text of the value of a `DataFrameReader` option, as its `option` takes it: None for None,
/// a bool lowercased, and else the `str()` of the value.
pub(crate) fn reader_option_text(value: &Bound<'_, PyAny>) -> PyResult<Option<OptionText>> {
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(b) = value.cast::<PyBool>() {
        let text = if b.is_true() { "true" } else { "false" };
        return Ok(Some(OptionText::literal(text)));
    }
    stream_option_text(value)
}

/// The text of the value of a `DataStreamReader` option, as its `option` takes it: None for None,
/// else the `str()` of the value, so that a bool is `True`.
pub(crate) fn stream_option_text(value: &Bound<'_, PyAny>) -> PyResult<Option<OptionText>> {
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(OptionText::of(value.str()?.as_any())?))
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

impl OptionBuilder for DataFrameWriterV2 {
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

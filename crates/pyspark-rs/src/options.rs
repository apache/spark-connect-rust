//! The options of the session builder and the catalog, as PySpark converts them, and those of the
//! plans of the reads of PySpark's readers (see `reader_plans`).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyString};
use spark_connect_core::error::SparkError;

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
// asserts its keys, the call of that method. The core takes only the strs that encode: each entry
// of the dict is converted when the plan is built ([`plan_entry`]), and the plan fails with the
// error that PySpark raises of the first entry that the core does not take.

/// The UTF-32LE bytes of a str, which keep its lone surrogates: the form of
/// [`SparkError::unencodable_str`].
pub(crate) fn utf32le_of(text: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let encoded = text.call_method1("encode", ("utf-32-le", "surrogatepass"))?;
    Ok(encoded.cast::<PyBytes>()?.as_bytes().to_vec())
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

/// The key of an option: a str, or another object, which PySpark keeps in its dict as it is, of
/// the kind that decides the error of its plan.
pub(crate) enum OptionKey {
    Text(OptionText),
    Object { kind: ObjectKind },
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
    Ok(OptionKey::Object { kind })
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
fn entry_error(key: &OptionKey, value: &OptionKey) -> Option<SparkError> {
    let bad_value = || value_error(value);
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

/// The pair of an entry of `key` and `value` of the options of a plan, which PySpark sets in the map
/// of its message when it sends the plan, as the core takes it, or the error that the map raises, of
/// the key and then of the value: of a str that UTF-8 cannot encode, bytes that are not UTF-8, or
/// another object, None included.
pub(crate) fn plan_entry(
    key: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Result<(String, String), SparkError>> {
    let key = option_object(key)?;
    let value = option_object(value)?;
    if let Some(error) = entry_error(&key, &value) {
        return Ok(Err(error));
    }
    let text = |option: &OptionKey| option.core_text().and_then(|text| text.text.clone());
    Ok(match (text(&key), text(&value)) {
        (Some(key), Some(value)) => Ok((key, value)),
        _ => Err(builtin_type_error()),
    })
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

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

/// The Python object of an option, a key that is not a str or a value that is an object, which the
/// copies of the options of a state share: of one reference to the object, which the options of a
/// state report to Python's garbage collector while they alone hold it (see
/// [`UnencodableOptions::visit_objects`]). It is cloned only with the GIL ([`OptionObject::clone_ref`];
/// it has no `Clone`), so that the count of the options that hold it changes only while no
/// collection runs, which must get the same objects of a state in each of its passes (see
/// `crate::gc`): a drop without the GIL only makes it reported, which is safe.
pub(crate) struct OptionObject(std::sync::Arc<Py<PyAny>>);

impl OptionObject {
    fn new(object: Py<PyAny>) -> Self {
        OptionObject(std::sync::Arc::new(object))
    }

    /// Another holder of the object, with the GIL (`_py`).
    pub(crate) fn clone_ref(&self, _py: Python<'_>) -> Self {
        OptionObject(std::sync::Arc::clone(&self.0))
    }

    /// The object.
    pub(crate) fn bind<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.0.bind(py)
    }

    /// Whether `other` is of the same object, as a copy of it is. Of the address of the object,
    /// which the holders keep alive, so another object cannot take it meanwhile.
    fn is(&self, other: &OptionObject) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }

    /// Whether no other holder holds the object, which these options then report.
    fn held_alone(&self) -> bool {
        std::sync::Arc::strong_count(&self.0) == 1
    }
}

/// The key of an option: a str, or another object, which PySpark keeps in its dict as it is. It is
/// cloned with the GIL ([`OptionKey::clone_ref`]), as its object is.
pub(crate) enum OptionKey {
    Text(OptionText),
    Object {
        object: OptionObject,
        kind: ObjectKind,
    },
}

impl OptionKey {
    /// Another holder of the key, with the GIL.
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            OptionKey::Text(text) => OptionKey::Text(text.clone()),
            OptionKey::Object { object, kind } => OptionKey::Object {
                object: object.clone_ref(py),
                kind: kind.clone(),
            },
        }
    }

    /// Whether `other` is this key, as a copy of the options that hold it has it: the same str,
    /// or the same object.
    fn is(&self, other: &OptionKey) -> bool {
        match (self, other) {
            (OptionKey::Text(a), OptionKey::Text(b)) => a.utf32le == b.utf32le,
            (OptionKey::Object { object: a, .. }, OptionKey::Object { object: b, .. }) => a.is(b),
            _ => false,
        }
    }

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
    /// that is equal to it, of its `==`, which is Python code, so no lock of a state of options is
    /// held meanwhile (see [`change_options`]).
    fn same_as(&self, py: Python<'_>, other: &OptionKey) -> bool {
        match (self, other) {
            (OptionKey::Text(a), OptionKey::Text(b)) => a.utf32le == b.utf32le,
            (OptionKey::Object { object: a, .. }, OptionKey::Object { object: b, .. }) => {
                crate::gc::check_no_lock_held("the `==` of the keys of options");
                pyo3::types::PyAnyMethods::eq(a.bind(py), b.bind(py)).unwrap_or(false)
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

/// Whether `a` and `b` are the same value of an entry, as a copy of the options that hold it has
/// it.
fn same_value(a: Option<&OptionKey>, b: Option<&OptionKey>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.is(b),
        (None, None) => true,
        _ => false,
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
        object: OptionObject::new(object.clone().unbind()),
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
/// a str or bytes that are UTF-8. The plan fails with the error of the first entry. Copied with
/// the GIL ([`UnencodableOptions::clone_ref`]), as their objects are.
#[derive(Default)]
pub(crate) struct UnencodableOptions(Vec<(OptionKey, Option<OptionKey>)>);

impl UnencodableOptions {
    /// A copy of the entries, which holds their objects too, with the GIL.
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        UnencodableOptions(
            self.0
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone_ref(py),
                        value.as_ref().map(|value| value.clone_ref(py)),
                    )
                })
                .collect(),
        )
    }

    /// Whether these are the entries of `other`, a copy of them, as they are until either one
    /// changes: the same keys and values, of the same objects.
    fn same_entries(&self, other: &UnencodableOptions) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(&other.0)
                .all(|((key, value), (other_key, other_value))| {
                    key.is(other_key) && same_value(value.as_ref(), other_value.as_ref())
                })
    }

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

    /// The entry of `key`, of the `==` of the keys, which is Python code (see
    /// [`OptionKey::same_as`]).
    fn position(&self, py: Python<'_>, key: &OptionKey) -> Option<usize> {
        self.0.iter().position(|(entry, _)| entry.same_as(py, key))
    }

    /// Remove the entry of `key`, which is set to a value that the core takes, or unset.
    pub(crate) fn remove(&mut self, py: Python<'_>, key: &OptionKey) {
        if let Some(position) = self.position(py, key) {
            self.0.remove(position);
        }
    }

    /// Set the entry of `key` to `value`, as PySpark's dict is set: an entry keeps its place, and
    /// is removed when neither its key nor its value needs it.
    pub(crate) fn set(&mut self, py: Python<'_>, key: &OptionKey, value: Option<OptionKey>) {
        if !Self::kept(key, value.as_ref()) {
            self.remove(py, key);
            return;
        }
        match self.position(py, key) {
            Some(position) => self.0[position] = (key.clone_ref(py), value),
            None => self.0.push((key.clone_ref(py), value)),
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

    /// Report the Python objects of the entries, the keys that are not strs and the values that
    /// are objects, to Python's garbage collector, as PySpark's dict of its options holds them:
    /// those that only these options hold, so that a copy of them, which a request makes of the
    /// state, does not count one reference twice (see [`SharedObjects`]). A copy is made and
    /// dropped only with the GIL, or dropped without it, which only makes an object reported, so
    /// the objects reported are the same in each pass of a collection.
    pub(crate) fn visit_objects(
        &self,
        visit: &pyo3::PyVisit<'_>,
    ) -> Result<(), pyo3::PyTraverseError> {
        for (key, value) in &self.0 {
            for entry in std::iter::once(key).chain(value.as_ref()) {
                if let OptionKey::Object { object, .. } = entry {
                    if object.held_alone() {
                        visit.call(&*object.0)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// The entries as Python objects, which a reduce gives back: the key, and the value, if any.
    /// Of a copy of the options, as Python objects are made, which may run Python code.
    pub(crate) fn to_objects<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        crate::gc::check_no_lock_held("the Python objects of options made");
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

/// The Python objects of a state that the shallow copies of a reader, a writer or a session
/// builder share, as PySpark's copies share the attributes of their `__dict__`, such as the dict
/// of their options, whose keys and values may be any objects. A state that copies share has no
/// one holder among them that could report its objects to Python's garbage collector once, so the
/// state is held by a Python object too, a [`PySharedObjects`], which each copy holds and reports,
/// and which reports the objects: so each is reported once, and freed with the last copy that
/// holds the state, as in PySpark. The state of options is of a lock that is taken only with the
/// GIL ([`crate::gc::GilMutex`]), so a collection finds it as it is in each of its passes.
pub(crate) trait SharedObjects: Send + Sync {
    /// Report the Python objects that the state holds, unless its lock is held.
    fn visit(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError>;

    /// Drop the Python objects of the state, which the garbage collector found unreachable with
    /// every holder of the state. They are dropped once the lock is left, as their drop may run
    /// Python code.
    fn clear(&self, py: Python<'_>);
}

impl SharedObjects for crate::gc::GilMutex<UnencodableOptions> {
    fn visit(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        match self.try_lock_in_traverse(visit) {
            Some(options) => options.visit_objects(visit),
            None => Ok(()),
        }
    }

    fn clear(&self, py: Python<'_>) {
        let cleared = self
            .try_lock(py)
            .map(|mut options| std::mem::take(&mut *options));
        drop(cleared);
    }
}

/// Change the options of `state`, those that `options_of` gives of it, by `change`, which sets the
/// entry of a key of them, as PySpark's dict sets a key, and `commit` the change to `state` with
/// them. The keys are compared with their `==`, which is Python code: it may release the GIL, as
/// any Python function may, or set an option of the same state again, as a key that PySpark's dict
/// compares may. So `change` runs on a copy of the options, with no lock held, and the copy
/// replaces them unless they changed meanwhile; then it runs again, of the options as they are, as
/// a dict looks a key up again when its comparison changed the dict. The lock of the options is
/// held only to copy and to replace them, and `commit` runs under it, of no Python code; the
/// options that are replaced are dropped once it is released, as the drop of their objects may
/// run Python code. A call of `change` of a key that is a str runs once, as such keys are
/// compared with no Python code.
pub(crate) fn change_options<T, R>(
    py: Python<'_>,
    state: &crate::gc::GilMutex<T>,
    options_of: impl Fn(&mut T) -> &mut UnencodableOptions,
    mut change: impl FnMut(&mut UnencodableOptions) -> R,
    commit: impl FnOnce(&mut T, &R),
) -> R {
    let mut commit = Some(commit);
    loop {
        let base = options_of(&mut state.lock(py)).clone_ref(py);
        let mut changed = Some(base.clone_ref(py));
        let result = change(changed.as_mut().expect("the copy of the options"));
        let replaced = {
            let mut locked = state.lock(py);
            if options_of(&mut locked).same_entries(&base) {
                let replaced = std::mem::replace(
                    options_of(&mut locked),
                    changed.take().expect("the copy of the options"),
                );
                if let Some(commit) = commit.take() {
                    commit(&mut locked, &result);
                }
                Some(replaced)
            } else {
                None
            }
        };
        let committed = replaced.is_some();
        drop((replaced, changed, base));
        if committed {
            return result;
        }
    }
}

/// The entry of the option `key` of a reader, set to `value`, or unset when it is None, as a
/// reader's `option` sets it: `unencodable` keeps the entries that the core reader does not take,
/// and the change of the options of the core reader is returned, which takes only a str key that
/// encodes: its value, None for a value that is None or does not encode, which unsets it; and None
/// of another key.
fn reader_entry(
    py: Python<'_>,
    unencodable: &mut UnencodableOptions,
    key: &OptionKey,
    value: Option<&OptionText>,
) -> Option<(String, Option<String>)> {
    match value {
        Some(value) => unencodable.set(py, key, Some(OptionKey::Text(value.clone()))),
        None => unencodable.remove(py, key),
    }
    match key {
        OptionKey::Text(OptionText {
            text: Some(name), ..
        }) => Some((name.clone(), value.and_then(|value| value.text.clone()))),
        _ => None,
    }
}

/// `builder` with the change of its options that [`reader_entry`] gives.
fn with_reader_change<B: ReaderOptionBuilder>(
    builder: B,
    change: Option<(String, Option<String>)>,
) -> B {
    match change {
        Some((name, Some(value))) => builder.with_option(&name, &value),
        Some((name, None)) => builder.without_option(&name),
        None => builder,
    }
}

/// The entry of the option `key` of a writer, set to `value`, as a writer's `option` sets it:
/// `unencodable` keeps the entries that the core writer does not take, and the option that the
/// core writer takes is returned, of a key that is a str, or bytes that are UTF-8, with a value
/// that encodes.
pub(crate) fn writer_entry(
    py: Python<'_>,
    unencodable: &mut UnencodableOptions,
    key: &OptionKey,
    value: &OptionText,
) -> Option<(String, String)> {
    match (
        key.core_text().and_then(|name| name.text.clone()),
        value.text.clone(),
    ) {
        (Some(name), Some(value)) => {
            unencodable.remove(py, key);
            Some((name, value))
        }
        _ => {
            unencodable.set(py, key, Some(OptionKey::Text(value.clone())));
            None
        }
    }
}

/// The entry of the table property `key` of a V2 writer, set to `value`, or unset when it is
/// None, as its `tableProperty` sets it: `unencodable` keeps the entries that the core writer does
/// not take, and the change of its table properties is returned, of a key that is a str, or bytes
/// that are UTF-8: its value, None for a value that is None or does not encode, which unsets it;
/// and None of another key.
pub(crate) fn property_entry(
    py: Python<'_>,
    unencodable: &mut UnencodableOptions,
    key: &OptionKey,
    value: Option<&OptionKey>,
) -> Option<(String, Option<String>)> {
    let Some(name) = key.core_text().and_then(|name| name.text.clone()) else {
        unencodable.set(py, key, value.map(|value| value.clone_ref(py)));
        return None;
    };
    match value {
        Some(value) => match value.core_text().and_then(|text| text.text.clone()) {
            Some(text) => {
                unencodable.remove(py, key);
                Some((name, Some(text)))
            }
            None => {
                unencodable.set(py, key, Some(value.clone_ref(py)));
                Some((name, None))
            }
        },
        None => {
            unencodable.remove(py, key);
            Some((name, None))
        }
    }
}

/// The Python object that holds a state that the shallow copies of an object share, for Python's
/// garbage collector (see [`SharedObjects`]): each copy holds it and reports it in its
/// `__traverse__`, and it reports the objects of the state.
#[pyclass(name = "_SharedObjects", module = "pyspark._pyspark")]
pub(crate) struct PySharedObjects(std::sync::Arc<dyn SharedObjects>);

impl PySharedObjects {
    /// The Python object of `state`, for a new holder of it, or None when it cannot be made, which
    /// reports nothing of the state: its objects are kept alive then, rather than freed while
    /// reachable.
    pub(crate) fn of(state: std::sync::Arc<dyn SharedObjects>) -> Option<Py<PySharedObjects>> {
        Python::attach(|py| Py::new(py, PySharedObjects(state)).ok())
    }
}

#[pymethods]
impl PySharedObjects {
    /// The objects of the state. The object has no `__dict__`, so a collection does not meet it
    /// before its value is written.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        self.0.visit(&visit)
    }

    /// Break a reference cycle that the garbage collector found unreachable: as every holder of
    /// the state is unreachable then, the objects of the state are dropped.
    fn __clear__(&mut self) {
        Python::attach(|py| self.0.clear(py));
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

/// `builder` with the option `key` set to `value`, or unset when `value` is None, on a reader whose
/// entries that the core reader does not take its copies share, `shared`, as a reader's `option`
/// sets it ([`reader_entry`]): the keys are compared with no lock held ([`change_options`]).
pub(crate) fn put_shared_option<B: ReaderOptionBuilder>(
    py: Python<'_>,
    builder: B,
    shared: &crate::gc::GilMutex<UnencodableOptions>,
    key: &OptionKey,
    value: Option<&OptionText>,
) -> B {
    let change = change_options(
        py,
        shared,
        |options| options,
        |options| reader_entry(py, options, key, value),
        |_, _| {},
    );
    with_reader_change(builder, change)
}

/// Set the option `key` to `value`, or unset it when it is None, on the options that a reader and
/// its shallow copies share, as PySpark's `option` sets the dict of its options: `options`, the
/// options of the core reader, takes the option of a str key whose value encodes, and `shared`
/// keeps the other entries ([`reader_entry`]). The keys are compared with no lock held
/// ([`change_options`]), and the caller holds no borrow of the reader meanwhile, so that a key
/// whose `__eq__` releases the GIL, or sets an option of the reader again, stops no call of the
/// reader, as PySpark's dict stops none.
pub(crate) fn put_reader_option_of_copies(
    py: Python<'_>,
    options: &WriterState<HashMap<String, String>>,
    shared: &crate::gc::GilMutex<UnencodableOptions>,
    key: &OptionKey,
    value: Option<&OptionText>,
) {
    let change = change_options(
        py,
        shared,
        |options| options,
        |options| reader_entry(py, options, key, value),
        |_, _| {},
    );
    if let Some((name, value)) = change {
        // As the core reader's `option` and `unset_option`, of the map that its copies share.
        options.update(|mut options| {
            match value {
                Some(value) => {
                    options.insert(name, value);
                }
                None => {
                    options.remove(&name);
                }
            }
            options
        });
    }
}

/// `builder` with the option `key` set to `value`, converted by `to_text`, **skipping** a `value`
/// that is absent or None, as [`set_option_skip_none`] does, on a reader whose entries that the
/// core reader does not take its copies share, as [`put_shared_option`] sets them.
pub(crate) fn put_option_skip_none<B: ReaderOptionBuilder>(
    py: Python<'_>,
    builder: B,
    shared: &crate::gc::GilMutex<UnencodableOptions>,
    key: &OptionKey,
    value: Option<&Bound<'_, PyAny>>,
    to_text: fn(&Bound<'_, PyAny>) -> PyResult<Option<OptionText>>,
) -> PyResult<B> {
    let Some(value) = value else {
        return Ok(builder);
    };
    match to_text(value)? {
        Some(text) => Ok(put_shared_option(py, builder, shared, key, Some(&text))),
        None => Ok(builder),
    }
}

/// `builder` with each of `options` set, **skipping** those that are None, as
/// [`put_option_skip_none`] does for each one, on a reader whose entries that the core reader does
/// not take its copies share: the values are converted first, in order, as Python code, so the
/// first one that fails raises before any of them is set; their keys are strs.
pub(crate) fn skip_none_shared<B: ReaderOptionBuilder>(
    py: Python<'_>,
    builder: B,
    shared: &crate::gc::GilMutex<UnencodableOptions>,
    options: &[(&str, Option<&Bound<'_, PyAny>>)],
    to_text: fn(&Bound<'_, PyAny>) -> PyResult<Option<OptionText>>,
) -> PyResult<B> {
    let mut entries = Vec::new();
    for (name, value) in options {
        if let Some(value) = value {
            if let Some(text) = to_text(value)? {
                entries.push((OptionKey::Text(OptionText::literal(name)), text));
            }
        }
    }
    if entries.is_empty() {
        return Ok(builder);
    }
    let changes = change_options(
        py,
        shared,
        |options| options,
        |options| {
            entries
                .iter()
                .map(|(key, value)| reader_entry(py, options, key, Some(value)))
                .collect::<Vec<_>>()
        },
        |_, _| {},
    );
    Ok(changes.into_iter().fold(builder, with_reader_change))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::gc::GilMutex;

    /// Whether `gc.get_referents` of `shared` gives `object`, as a collection gets it.
    fn reports(py: Python<'_>, shared: &Py<PySharedObjects>, object: &Bound<'_, PyAny>) -> bool {
        let referents = py
            .import("gc")
            .unwrap()
            .call_method1("get_referents", (shared,))
            .unwrap();
        referents
            .cast::<PyList>()
            .unwrap()
            .iter()
            .any(|referent| referent.is(object))
    }

    /// Set the entry of `key` to the text `value` in the options of `state`, as an option of a
    /// reader is set.
    fn put(py: Python<'_>, state: &GilMutex<UnencodableOptions>, key: &OptionKey, value: &str) {
        let value = OptionText::literal(value);
        change_options(
            py,
            state,
            |options| options,
            |options| reader_entry(py, options, key, Some(&value)),
            |_, _| {},
        );
    }

    #[test]
    fn the_objects_of_options_are_reported_while_the_options_of_a_state_alone_hold_them() {
        let _alone = crate::gc::tests_of_the_state_alone();
        Python::initialize();
        Python::attach(|py| {
            let object = py.eval(c"object()", None, None).unwrap();
            let state = Arc::new(GilMutex::new(UnencodableOptions::default()));
            let shared = PySharedObjects::of(state.clone()).unwrap();
            let key = option_key(&object).unwrap();
            put(py, &state, &key, "v");
            // The key of the caller holds the object too, as the copy of a request does: the
            // options do not report it while it lives, which keeps it alive.
            assert!(!reports(py, &shared, &object));
            drop(key);
            assert!(reports(py, &shared, &object));
            let copy = state.lock(py).clone_ref(py);
            assert!(!reports(py, &shared, &object));
            drop(copy);
            assert!(reports(py, &shared, &object));
            // A state whose lock is held is not read, in any pass of a collection alike.
            {
                let _held = state.lock(py);
                assert!(!reports(py, &shared, &object));
            }
            assert!(reports(py, &shared, &object));
            // The clear of an unreachable state drops its objects.
            state.clear(py);
            assert!(!reports(py, &shared, &object));
            assert_eq!(state.lock(py).0.len(), 0);
        });
    }

    /// The flags of the comparisons of [`keys_are_compared_with_no_lock_held_and_again_once_the_options_changed`].
    static COMPARED_WITH_A_LOCK: AtomicBool = AtomicBool::new(false);
    static REENTERED: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn keys_are_compared_with_no_lock_held_and_again_once_the_options_changed() {
        let _alone = crate::gc::tests_of_the_state_alone();
        Python::initialize();
        Python::attach(|py| {
            let namespace = pyo3::types::PyDict::new(py);
            py.run(
                c"class Key:\n    def __init__(self, tag, on_eq=None):\n        self.tag = tag\n        self.on_eq = on_eq\n    def __hash__(self):\n        return 1\n    def __eq__(self, other):\n        on_eq, self.on_eq = self.on_eq, None\n        if on_eq is not None:\n            on_eq()\n        return self is other\n",
                Some(&namespace),
                None,
            )
            .unwrap();
            let class = namespace.get_item("Key").unwrap().unwrap();
            let state = Arc::new(GilMutex::new(UnencodableOptions::default()));
            let first = class.call1(("first",)).unwrap();
            put(py, &state, &option_key(&first).unwrap(), "1");
            // The comparison of a key with `first` sets another key of the same options, as the
            // `__eq__` of a key of PySpark's dict may: the options are not locked meanwhile, and
            // the change is made again of the options as they are then, which keep both keys,
            // in the order of a dict that looks its key up again.
            let inner = class.call1(("inner",)).unwrap();
            let inner_key = option_key(&inner).unwrap();
            let of_inner = state.clone();
            let on_eq = pyo3::types::PyCFunction::new_closure(py, None, None, move |args, _| {
                let py = args.py();
                if of_inner.try_lock(py).is_none() {
                    COMPARED_WITH_A_LOCK.store(true, Ordering::SeqCst);
                }
                REENTERED.fetch_add(1, Ordering::SeqCst);
                put(py, &of_inner, &inner_key, "2");
            })
            .unwrap();
            first.setattr("on_eq", on_eq).unwrap();
            let outer = class.call1(("outer",)).unwrap();
            put(py, &state, &option_key(&outer).unwrap(), "3");
            assert!(!COMPARED_WITH_A_LOCK.load(Ordering::SeqCst));
            assert_eq!(REENTERED.load(Ordering::SeqCst), 1);
            // Of a copy, as Python objects are made apart from the lock.
            let options = state.lock(py).clone_ref(py);
            let tags: Vec<String> = options
                .to_objects(py)
                .unwrap()
                .into_iter()
                .map(|(key, value)| {
                    let tag: String = key.getattr("tag").unwrap().extract().unwrap();
                    let value: String = value.extract().unwrap();
                    format!("{tag}={value}")
                })
                .collect();
            assert_eq!(tags, ["first=1", "inner=2", "outer=3"]);
        });
    }

    #[test]
    fn a_key_that_is_a_str_is_set_with_no_comparison_of_python_code() {
        let _alone = crate::gc::tests_of_the_state_alone();
        Python::initialize();
        Python::attach(|py| {
            let namespace = pyo3::types::PyDict::new(py);
            py.run(
                c"class Key:\n    compared = 0\n    def __hash__(self):\n        return 1\n    def __eq__(self, other):\n        Key.compared += 1\n        return self is other\n",
                Some(&namespace),
                None,
            )
            .unwrap();
            let class = namespace.get_item("Key").unwrap().unwrap();
            let state = GilMutex::new(UnencodableOptions::default());
            // A str key of a value that UTF-8 cannot encode is kept, next to an object key; the
            // core reader unsets it.
            let key = option_key(&PyString::new(py, "a")).unwrap();
            let unencodable = OptionText::of(&py.eval(c"'\\udcff'", None, None).unwrap()).unwrap();
            let change = change_options(
                py,
                &state,
                |options| options,
                |options| reader_entry(py, options, &key, Some(&unencodable)),
                |_, _| {},
            );
            assert_eq!(change, Some(("a".to_string(), None)));
            put(
                py,
                &state,
                &option_key(&class.call0().unwrap()).unwrap(),
                "2",
            );
            assert_eq!(state.lock(py).0.len(), 2);
            // Set to a value that encodes, the str key leaves the kept options, which compare it
            // with the object key of no Python code, and the core reader takes it.
            let encodable = OptionText::literal("x");
            let change = change_options(
                py,
                &state,
                |options| options,
                |options| reader_entry(py, options, &key, Some(&encodable)),
                |_, _| {},
            );
            assert_eq!(change, Some(("a".to_string(), Some("x".to_string()))));
            assert_eq!(state.lock(py).0.len(), 1);
            let compared: usize = class.getattr("compared").unwrap().extract().unwrap();
            assert_eq!(compared, 0);
        });
    }
}

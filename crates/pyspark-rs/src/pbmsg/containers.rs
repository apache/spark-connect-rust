//! The values of repeated and map fields, as protobuf's Python containers (those of upb, the
//! implementation of a `protobuf` install) give them: a list-like container of the scalars or
//! messages of a repeated field, a `collections.abc.MutableSequence`, and a dict-like map of a map
//! field, a `collections.abc.MutableMapping` (see `pyspark.sql.connect.proto._rust`, which
//! registers them).
//!
//! A message that an unset child holds sets the child in its parent when its container may
//! change, as setting a field of the child does: as upb's containers, at the start of each method
//! that may change it, even one that then fails.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyAttributeError, PyFutureWarning, PyIndexError, PyKeyError, PyNotImplementedError,
    PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{
    PyDict, PyIterator, PyList, PySlice, PyTuple, PyType, PyWeakrefMethods, PyWeakrefReference,
};
use pyo3::IntoPyObjectExt;

use super::message::{
    content_eq, duplicate, mark_present, message_from_value, new_instance, registered_class,
    PyMessage,
};
use super::pool::{Kind, MapDesc};
use super::scalar::Scalar;

/// The message that holds a container, weakly.
pub(crate) type Owner = Option<Py<PyWeakrefReference>>;

fn owner_of(py: Python<'_>, owner: &Owner) -> Owner {
    owner.as_ref().map(|o| o.clone_ref(py))
}

/// Set the message of `owner` in its parent, if it is an unset child.
fn touch(py: Python<'_>, owner: &Owner) -> PyResult<()> {
    if let Some(owner) = owner {
        if let Some(message) = owner.bind(py).upgrade_as::<PyMessage>()? {
            mark_present(&message)?;
        }
    }
    Ok(())
}

/// The position of `index` in a sequence of `len` items, as a list's subscription takes it.
fn position(index: isize, len: usize) -> PyResult<usize> {
    let i = if index < 0 {
        index + len as isize
    } else {
        index
    };
    if i < 0 || i as usize >= len {
        Err(PyIndexError::new_err("list index out of range"))
    } else {
        Ok(i as usize)
    }
}

/// The index of a subscription that is not a slice, with upb's TypeError of a value that is not
/// an int.
fn index_of_key(key: &Bound<'_, PyAny>) -> PyResult<isize> {
    key.extract::<isize>().map_err(|err| {
        if err.is_instance_of::<PyTypeError>(key.py()) {
            PyTypeError::new_err("list indices must be integers")
        } else {
            err
        }
    })
}

/// upb's error of assigning `got` values to an extended slice of `want` items (of its message).
fn check_extended_slice(
    key: &Bound<'_, PyAny>,
    len: usize,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let slice = key.cast::<PySlice>()?;
    let indices = slice.indices(len as isize)?;
    if indices.step == 1 {
        return Ok(());
    }
    let got = value.len()?;
    if got != indices.slicelength {
        return Err(PyValueError::new_err(format!(
            "attempt to assign sequence of  {got} to extended slice of size {}",
            indices.slicelength
        )));
    }
    Ok(())
}

/// The position that `pop(index)` takes an item of, as upb's: an index out of range warns, and
/// takes the last item.
fn pop_position(py: Python<'_>, index: isize, len: usize) -> PyResult<usize> {
    let len = len as isize;
    let mut i = if index < 0 { index + len } else { index };
    if i < 0 || i >= len {
        PyErr::warn(
            py,
            py.get_type::<PyFutureWarning>().as_any(),
            c"pop index out of range",
            1,
        )?;
        i = len - 1;
    }
    if i < 0 {
        return Err(PyIndexError::new_err(format!(
            "list index ({i}) out of range"
        )));
    }
    Ok(i as usize)
}

/// The position that `list.insert` puts an item at, of `index` in a sequence of `len` items.
fn insert_position(index: isize, len: usize) -> usize {
    let len = len as isize;
    let i = if index < 0 {
        (index + len).max(0)
    } else {
        index.min(len)
    };
    i as usize
}

/// The error of pickling (or copying) a repeated field, as upb's.
fn pickle_error(py: Python<'_>) -> PyErr {
    let message = "can't pickle repeated message fields, convert to list first";
    match py
        .import("pickle")
        .and_then(|m| m.getattr("PickleError"))
        .and_then(|c| c.call1((message,)))
    {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

fn not_in_container() -> PyErr {
    PyValueError::new_err("remove(x): x not in container")
}

/// A container of `value`'s items, by `collections.abc` `name` (such as `KeysView`) of `value`.
fn view<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    value
        .py()
        .import("collections.abc")?
        .getattr(name)?
        .call1((value,))
}

/// The key and value of `pair`, as `for key, value in pairs` unpacks it.
fn unpack_pair<'py>(pair: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let items = pair.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    match <[Bound<'py, PyAny>; 2]>::try_from(items) {
        Ok([key, value]) => Ok((key, value)),
        Err(items) if items.len() < 2 => Err(PyValueError::new_err(format!(
            "not enough values to unpack (expected 2, got {})",
            items.len()
        ))),
        Err(_) => Err(PyValueError::new_err(
            "too many values to unpack (expected 2)",
        )),
    }
}

/// A message of the type `full_name` with the fields of `value`, as `append` and `insert` of upb
/// take one: a new message that merges `value`, a message of the same type.
fn message_copy<'py>(
    py: Python<'py>,
    full_name: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyMessage>> {
    let message = new_instance(py, full_name)?;
    message.call_method1("MergeFrom", (value,))?;
    Ok(message)
}

/// Whether `value` is a message, of this runtime or of protobuf's, as upb's `append` checks it.
fn verify_message(value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.cast::<PyMessage>().is_ok()
        || (value.hasattr("DESCRIPTOR")? && value.hasattr("SerializeToString")?)
    {
        return Ok(());
    }
    let repr = value.repr().map(|r| r.to_string()).unwrap_or_default();
    Err(PyTypeError::new_err(format!(
        "Expected a message object, but got {repr}."
    )))
}

#[pyclass(
    sequence,
    module = "pyspark.sql.connect.proto",
    name = "RepeatedScalarContainer"
)]
pub struct RepeatedScalar {
    pub(crate) kind: Kind,
    /// The full name of the field, which an error of a value names.
    pub(crate) field: String,
    /// The full name of the enum of an enum field.
    pub(crate) enum_type: String,
    pub(crate) items: Vec<Scalar>,
    owner: Owner,
}

impl RepeatedScalar {
    pub(crate) fn new(
        kind: Kind,
        field: String,
        enum_type: String,
        items: Vec<Scalar>,
        owner: Owner,
    ) -> Self {
        RepeatedScalar {
            kind,
            field,
            enum_type,
            items,
            owner,
        }
    }

    fn convert(&self, value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        Scalar::from_py(self.kind, &self.field, &self.enum_type, value)
    }

    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let values = self
            .items
            .iter()
            .map(|v| v.to_py(py))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, values)
    }

    /// Replace the items with the values of `list`, each checked as `append` checks it.
    fn set_from_list(&mut self, list: &Bound<'_, PyList>) -> PyResult<()> {
        self.items = list
            .iter()
            .map(|v| self.convert(&v))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(())
    }

    /// Set the message of the container in its parent, as upb's methods that may change a
    /// container do first.
    fn reify(slf: &Bound<'_, Self>) -> PyResult<()> {
        let owner = owner_of(slf.py(), &slf.borrow().owner);
        touch(slf.py(), &owner)
    }
}

#[pymethods]
impl RepeatedScalar {
    fn append(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let mut me = slf.borrow_mut();
        let v = me.convert(value)?;
        me.items.push(v);
        Ok(())
    }

    fn extend(slf: &Bound<'_, Self>, elem_seq: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let values = {
            let me = slf.borrow();
            let mut values = Vec::new();
            let items = elem_seq
                .try_iter()
                .map_err(|_| PyTypeError::new_err("must assign iterable"))?;
            for v in items {
                values.push(me.convert(&v?)?);
            }
            values
        };
        slf.borrow_mut().items.extend(values);
        Ok(())
    }

    fn insert(slf: &Bound<'_, Self>, key: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let mut me = slf.borrow_mut();
        let v = me.convert(value)?;
        let at = insert_position(key, me.items.len());
        me.items.insert(at, v);
        Ok(())
    }

    #[pyo3(signature = (key=-1))]
    fn pop(slf: &Bound<'_, Self>, key: isize) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        let at = pop_position(py, key, len)?;
        let value = slf.borrow_mut().items.remove(at);
        value.to_py(py)
    }

    fn remove(slf: &Bound<'_, Self>, elem: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        // As upb's: the first item that equals `elem`, as Python compares them.
        let values = slf.borrow().to_list(py)?;
        for (at, value) in values.iter().enumerate() {
            if value.eq(elem)? {
                slf.borrow_mut().items.remove(at);
                return Ok(());
            }
        }
        Err(not_in_container())
    }

    fn reverse(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::reify(slf)?;
        slf.borrow_mut().items.reverse();
        Ok(())
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn sort(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        // As upb's: no change of an unset container (an empty one), so no change of its message.
        let py = slf.py();
        let list = slf.borrow().to_list(py)?;
        let kwargs = sort_kwargs(py, kwargs)?;
        list.call_method("sort", args, Some(&kwargs))?;
        slf.borrow_mut().set_from_list(&list)
    }

    /// As upb's: no change of an unset container (an empty one), so no change of its message.
    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        slf.borrow_mut().items.clear();
        Ok(())
    }

    #[pyo3(name = "MergeFrom")]
    fn merge_from(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, other)
    }

    /// A copy of the container, held by no message, as upb's `copy.deepcopy` gives one.
    #[pyo3(signature = (memo=None))]
    fn __deepcopy__(&self, memo: Option<&Bound<'_, PyAny>>) -> Self {
        let _ = memo;
        RepeatedScalar::new(
            self.kind,
            self.field.clone(),
            self.enum_type.clone(),
            self.items.clone(),
            None,
        )
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(pickle_error(py))
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if key.cast::<PySlice>().is_ok() {
            return Ok(self.to_list(py)?.as_any().get_item(key)?.unbind());
        }
        let at = position(index_of_key(key)?, self.items.len())?;
        self.items[at].to_py(py)
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let mut me = slf.borrow_mut();
        if key.cast::<PySlice>().is_ok() {
            check_extended_slice(key, me.items.len(), value)?;
            let list = me.to_list(py)?;
            list.as_any().set_item(key, value)?;
            me.set_from_list(&list)
        } else {
            let at = position(index_of_key(key)?, me.items.len())?;
            let v = me.convert(value)?;
            me.items[at] = v;
            Ok(())
        }
    }

    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let mut me = slf.borrow_mut();
        if key.cast::<PySlice>().is_ok() {
            let list = me.to_list(py)?;
            list.as_any().del_item(key)?;
            me.set_from_list(&list)
        } else {
            let at = position(index_of_key(key)?, me.items.len())?;
            me.items.remove(at);
            Ok(())
        }
    }

    fn __contains__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.to_list(py)?.contains(value)
    }

    /// As upb's: the list of the items compared with `other`, or with the list of the items of
    /// `other` when it is a container of the same kind.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return Ok(py.NotImplemented());
        }
        let mine = self.to_list(py)?;
        let other = match other.cast::<RepeatedScalar>() {
            Ok(other) => other.borrow().to_list(py)?.into_any(),
            Err(_) => other.clone(),
        };
        Ok(mine.as_any().rich_compare(&other, op)?.unbind())
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.to_list(py)?.repr()?.to_string())
    }
}

/// The keyword arguments of `list.sort` of the keyword arguments of a container's `sort`: upb's
/// passes protobuf's old `sort_function` on as `cmp` (which `list.sort` then refuses).
fn sort_kwargs<'py>(
    py: Python<'py>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            if key.extract::<String>()? == "sort_function" {
                out.set_item("cmp", value)?;
            } else {
                out.set_item(key, value)?;
            }
        }
    }
    Ok(out)
}

#[pyclass(
    sequence,
    module = "pyspark.sql.connect.proto",
    name = "RepeatedCompositeContainer"
)]
pub struct RepeatedComposite {
    pub(crate) message_type: String,
    pub(crate) items: Vec<Py<PyMessage>>,
    owner: Owner,
}

impl RepeatedComposite {
    pub(crate) fn new(message_type: String, items: Vec<Py<PyMessage>>, owner: Owner) -> Self {
        RepeatedComposite {
            message_type,
            items,
            owner,
        }
    }

    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.items.iter().map(|m| m.clone_ref(py)))
    }

    fn reify(slf: &Bound<'_, Self>) -> PyResult<()> {
        let owner = owner_of(slf.py(), &slf.borrow().owner);
        touch(slf.py(), &owner)
    }

    fn set_from_list(&mut self, list: &Bound<'_, PyList>) -> PyResult<()> {
        self.items = list
            .iter()
            .map(|m| Ok(m.cast_into::<PyMessage>()?.unbind()))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(())
    }
}

#[pymethods]
impl RepeatedComposite {
    #[pyo3(signature = (**kwargs))]
    fn add<'py>(
        slf: &Bound<'py, Self>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let py = slf.py();
        Self::reify(slf)?;
        let message_type = slf.borrow().message_type.clone();
        let child = new_instance(py, &message_type)?;
        if let Some(kwargs) = kwargs {
            child.borrow_mut().init_from_mapping(py, kwargs.as_any())?;
        }
        slf.borrow_mut().items.push(child.clone().unbind());
        Ok(child)
    }

    /// As upb's: a new message, appended, then merged with `value`, so that a message of another
    /// type leaves the new message (empty) in the container.
    fn append(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        verify_message(value)?;
        let message_type = slf.borrow().message_type.clone();
        let child = new_instance(py, &message_type)?;
        slf.borrow_mut().items.push(child.clone().unbind());
        child.call_method1("MergeFrom", (value,))?;
        Ok(())
    }

    fn extend(slf: &Bound<'_, Self>, elem_seq: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        for value in elem_seq.try_iter()? {
            Self::append(slf, &value?)?;
        }
        Ok(())
    }

    fn insert(slf: &Bound<'_, Self>, key: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let message_type = slf.borrow().message_type.clone();
        let copy = message_copy(py, &message_type, value)?;
        let mut me = slf.borrow_mut();
        let at = insert_position(key, me.items.len());
        me.items.insert(at, copy.unbind());
        Ok(())
    }

    #[pyo3(signature = (key=-1))]
    fn pop(slf: &Bound<'_, Self>, key: isize) -> PyResult<Py<PyMessage>> {
        let py = slf.py();
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        let at = pop_position(py, key, len)?;
        Ok(slf.borrow_mut().items.remove(at))
    }

    fn remove(slf: &Bound<'_, Self>, elem: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let items = slf.borrow().to_list(py)?;
        for (at, item) in items.iter().enumerate() {
            if item.eq(elem)? {
                slf.borrow_mut().items.remove(at);
                return Ok(());
            }
        }
        Err(not_in_container())
    }

    fn reverse(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::reify(slf)?;
        slf.borrow_mut().items.reverse();
        Ok(())
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn sort(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        // As upb's: no change of an unset container (an empty one), so no change of its message.
        let py = slf.py();
        let list = slf.borrow().to_list(py)?;
        let kwargs = sort_kwargs(py, kwargs)?;
        list.call_method("sort", args, Some(&kwargs))?;
        slf.borrow_mut().set_from_list(&list)
    }

    /// As upb's: no change of an unset container (an empty one), so no change of its message.
    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        slf.borrow_mut().items.clear();
        Ok(())
    }

    #[pyo3(name = "MergeFrom")]
    fn merge_from(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend(slf, other)
    }

    /// A copy of the container, of copies of its messages, held by no message.
    #[pyo3(signature = (memo=None))]
    fn __deepcopy__(&self, py: Python<'_>, memo: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let _ = memo;
        let items = self
            .items
            .iter()
            .map(|m| duplicate(py, m.bind(py)).map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(RepeatedComposite::new(
            self.message_type.clone(),
            items,
            None,
        ))
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if key.cast::<PySlice>().is_ok() {
            return Ok(self.to_list(py)?.as_any().get_item(key)?.unbind());
        }
        let at = position(index_of_key(key)?, self.items.len())?;
        Ok(self.items[at].clone_ref(py).into_any())
    }

    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let _ = (key, value);
        Err(PyTypeError::new_err("does not support assignment"))
    }

    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let mut me = slf.borrow_mut();
        if key.cast::<PySlice>().is_ok() {
            let list = me.to_list(py)?;
            list.as_any().del_item(key)?;
            me.set_from_list(&list)
        } else {
            let at = position(index_of_key(key)?, me.items.len())?;
            me.items.remove(at);
            Ok(())
        }
    }

    fn __contains__(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.to_list(py)?.contains(value)
    }

    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return Ok(py.NotImplemented());
        }
        let mine = self.to_list(py)?;
        let other = match other.cast::<RepeatedComposite>() {
            Ok(other) => other.borrow().to_list(py)?.into_any(),
            Err(_) => other.clone(),
        };
        Ok(mine.as_any().rich_compare(&other, op)?.unbind())
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.to_list(py)?.repr()?.to_string())
    }
}

/// The value of an entry of a map.
pub(crate) enum MapValue {
    Scalar(Scalar),
    Message(Py<PyMessage>),
}

#[pyclass(mapping, module = "pyspark.sql.connect.proto", name = "MapContainer")]
pub struct MapContainer {
    field_name: String,
    pub(crate) map: MapDesc,
    /// The entries, in the order of their insertion.
    pub(crate) entries: Vec<(Scalar, MapValue)>,
    owner: Owner,
}

impl MapContainer {
    pub(crate) fn new(field_name: String, map: MapDesc, owner: Owner) -> Self {
        MapContainer {
            field_name,
            map,
            entries: Vec::new(),
            owner,
        }
    }

    fn is_message(&self) -> bool {
        self.map.value == Kind::Message
    }

    fn key_of(&self, key: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        Scalar::from_py(self.map.key, &format!("{}.key", self.map.entry), "", key)
    }

    fn value_of(&self, value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        Scalar::from_py(
            self.map.value,
            &format!("{}.value", self.map.entry),
            &self.map.value_type,
            value,
        )
    }

    fn find(&self, key: &Scalar) -> Option<usize> {
        self.entries.iter().position(|(k, _)| k == key)
    }

    fn reify(slf: &Bound<'_, Self>) -> PyResult<()> {
        let owner = owner_of(slf.py(), &slf.borrow().owner);
        touch(slf.py(), &owner)
    }

    /// Set the value of `key`, replacing the one it has.
    pub(crate) fn insert(&mut self, key: Scalar, value: MapValue) {
        match self.find(&key) {
            Some(at) => self.entries[at].1 = value,
            None => self.entries.push((key, value)),
        }
    }

    fn value_to_py(py: Python<'_>, value: &MapValue) -> PyResult<Py<PyAny>> {
        match value {
            MapValue::Scalar(v) => v.to_py(py),
            MapValue::Message(m) => Ok(m.clone_ref(py).into_any()),
        }
    }

    fn copy_value(py: Python<'_>, value: &MapValue) -> PyResult<MapValue> {
        Ok(match value {
            MapValue::Scalar(v) => MapValue::Scalar(v.clone()),
            MapValue::Message(m) => MapValue::Message(duplicate(py, m.bind(py))?.unbind()),
        })
    }

    /// A copy of the map, of copies of its messages, held by no message yet.
    pub(crate) fn copy(&self, py: Python<'_>) -> PyResult<MapContainer> {
        let entries = self
            .entries
            .iter()
            .map(|(k, v)| Ok((k.clone(), Self::copy_value(py, v)?)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(MapContainer {
            field_name: self.field_name.clone(),
            map: self.map.clone(),
            entries,
            owner: None,
        })
    }

    /// Set the entries of `source` in `target`, of copies of its messages.
    pub(crate) fn merge(
        py: Python<'_>,
        target: &Bound<'_, Self>,
        source: &Bound<'_, Self>,
    ) -> PyResult<()> {
        let copies = {
            let s = source.borrow();
            s.entries
                .iter()
                .map(|(k, v)| Ok((k.clone(), Self::copy_value(py, v)?)))
                .collect::<PyResult<Vec<_>>>()?
        };
        let mut t = target.borrow_mut();
        for (k, v) in copies {
            t.insert(k, v);
        }
        Ok(())
    }

    /// Whether `a` and `b` have the same keys, of equal values.
    pub(crate) fn equal(
        py: Python<'_>,
        a: &Bound<'_, Self>,
        b: &Bound<'_, Self>,
    ) -> PyResult<bool> {
        if a.is(b) {
            return Ok(true);
        }
        let (x, y) = (a.borrow(), b.borrow());
        if x.entries.len() != y.entries.len() {
            return Ok(false);
        }
        for (key, value) in &x.entries {
            let Some(at) = y.find(key) else {
                return Ok(false);
            };
            let equal = match (value, &y.entries[at].1) {
                (MapValue::Scalar(v), MapValue::Scalar(w)) => v == w,
                (MapValue::Message(m), MapValue::Message(n)) => {
                    content_eq(py, m.bind(py), n.bind(py))?
                }
                _ => false,
            };
            if !equal {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Set the items of `mapping` in `target`, as the constructor of a message takes a map field:
    /// a message value may be a message or a mapping of its fields.
    pub(crate) fn update_from(
        py: Python<'_>,
        target: &Bound<'_, Self>,
        mapping: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let items = mapping.call_method0("items")?;
        for item in items.try_iter()? {
            let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
            let (key, value) = {
                let t = target.borrow();
                let key = t.key_of(&key)?;
                let value = if t.is_message() {
                    MapValue::Message(message_from_value(py, &t.map.value_type, &value)?.unbind())
                } else {
                    MapValue::Scalar(t.value_of(&value)?)
                };
                (key, value)
            };
            target.borrow_mut().insert(key, value);
        }
        Ok(())
    }

    pub(crate) fn discard_unknown(&self, py: Python<'_>) -> PyResult<()> {
        for (_, value) in &self.entries {
            if let MapValue::Message(m) = value {
                m.bind(py).call_method0("DiscardUnknownFields")?;
            }
        }
        Ok(())
    }

    /// The value of `key`, inserting the default of the value type (a new message, of a message
    /// map) when the map has none, as upb's `map[key]` does.
    fn get_or_insert(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (key, value_type, value_kind, is_message) = {
            let me = slf.borrow();
            let key = me.key_of(key)?;
            if let Some(at) = me.find(&key) {
                return Self::value_to_py(py, &me.entries[at].1);
            }
            (
                key,
                me.map.value_type.clone(),
                me.map.value,
                me.is_message(),
            )
        };
        Self::reify(slf)?;
        let value = if is_message {
            MapValue::Message(new_instance(py, &value_type)?.unbind())
        } else {
            MapValue::Scalar(Scalar::default_of(value_kind))
        };
        let out = Self::value_to_py(py, &value)?;
        slf.borrow_mut().entries.push((key, value));
        Ok(out)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in &self.entries {
            dict.set_item(key.to_py(py)?, Self::value_to_py(py, value)?)?;
        }
        Ok(dict)
    }

    fn keys_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let keys = self
            .entries
            .iter()
            .map(|(k, _)| k.to_py(py))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, keys)
    }
}

#[pymethods]
impl MapContainer {
    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Self::get_or_insert(slf, key)
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Self::reify(slf)?;
        let mut me = slf.borrow_mut();
        let key = me.key_of(key)?;
        if me.is_message() {
            return Err(PyValueError::new_err("Message objects may not be assigned"));
        }
        let value = MapValue::Scalar(me.value_of(value)?);
        me.insert(key, value);
        Ok(())
    }

    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let mut me = slf.borrow_mut();
        let k = me.key_of(key)?;
        let at = me
            .find(&k)
            .ok_or_else(|| PyKeyError::new_err("Key not present in map"))?;
        me.entries.remove(at);
        Ok(())
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let key = self.key_of(key)?;
        Ok(self.find(&key).is_some())
    }

    fn __len__(&self) -> usize {
        self.entries.len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.keys_list(py)?.as_any().try_iter()
    }

    /// The keys, values and items, as the views of `collections.abc.Mapping` give them, as
    /// upb's maps do.
    fn keys<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        view(slf.as_any(), "KeysView")
    }

    fn values<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        view(slf.as_any(), "ValuesView")
    }

    fn items<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        view(slf.as_any(), "ItemsView")
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let k = self.key_of(key)?;
        match self.find(&k) {
            Some(at) => Self::value_to_py(py, &self.entries[at].1),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    /// The value of `key`, removed, as `MutableMapping.pop` of upb's maps gives it: `map[key]`
    /// gives the default of the value type for a missing key, so `default` is never returned.
    #[pyo3(signature = (key, default=None))]
    fn pop(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let _ = default;
        let value = Self::get_or_insert(slf, key)?;
        Self::__delitem__(slf, key)?;
        Ok(value)
    }

    fn popitem<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Py<PyAny>)> {
        let py = slf.py();
        let key = match slf.borrow().entries.first() {
            Some((key, _)) => key.to_py(py)?.into_bound(py),
            None => return Err(PyKeyError::new_err(())),
        };
        let value = Self::get_or_insert(slf, &key)?;
        Self::__delitem__(slf, &key)?;
        Ok((key, value))
    }

    #[pyo3(signature = (key, default=None))]
    fn setdefault(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.borrow().is_message() {
            return Err(PyNotImplementedError::new_err(
                "Set message map value directly is not supported.",
            ));
        }
        let Some(default) = default.filter(|d| !d.is_none()) else {
            return Err(PyValueError::new_err(
                "The value for scalar map setdefault must be set.",
            ));
        };
        {
            let me = slf.borrow();
            let k = me.key_of(key)?;
            if let Some(at) = me.find(&k) {
                return Self::value_to_py(py, &me.entries[at].1);
            }
        }
        Self::__setitem__(slf, key, default)?;
        Ok(default.clone().unbind())
    }

    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::reify(slf)?;
        slf.borrow_mut().entries.clear();
        Ok(())
    }

    /// As `MutableMapping.update`: the items of a mapping, of an object with `keys`, or of an
    /// iterable of pairs, then the keyword arguments.
    #[pyo3(signature = (other=None, /, **kwargs))]
    fn update(
        slf: &Bound<'_, Self>,
        other: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = slf.py();
        if let Some(other) = other {
            let mapping = py.import("collections.abc")?.getattr("Mapping")?;
            if other.is_instance(&mapping)? {
                for key in other.try_iter()? {
                    let key = key?;
                    Self::__setitem__(slf, &key, &other.get_item(&key)?)?;
                }
            } else if other.hasattr("keys")? {
                for key in other.call_method0("keys")?.try_iter()? {
                    let key = key?;
                    Self::__setitem__(slf, &key, &other.get_item(&key)?)?;
                }
            } else {
                for pair in other.try_iter()? {
                    let (key, value) = unpack_pair(&pair?)?;
                    Self::__setitem__(slf, &key, &value)?;
                }
            }
        }
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                Self::__setitem__(slf, &key, &value)?;
            }
        }
        Ok(())
    }

    /// As upb's: the entries of another map, or the pairs of an iterable; never a dict.
    #[pyo3(name = "MergeFrom")]
    fn merge_from(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        if other.cast::<PyDict>().is_ok() {
            return Err(PyAttributeError::new_err("Merging of dict is not allowed"));
        }
        if let Ok(other) = other.cast::<MapContainer>() {
            return Self::merge(py, slf, other);
        }
        let is_message = slf.borrow().is_message();
        for pair in other.try_iter()? {
            let (key, value) = unpack_pair(&pair?)?;
            if is_message {
                // upb converts the value as a scalar, which a message is not.
                slf.borrow().key_of(&key)?;
                return Err(PyTypeError::new_err(
                    "bad argument type for built-in operation",
                ));
            }
            Self::__setitem__(slf, &key, &value)?;
        }
        Ok(())
    }

    /// The class of the entry messages of the map, as upb's `GetEntryClass`.
    #[pyo3(name = "GetEntryClass")]
    fn get_entry_class(&self, py: Python<'_>) -> PyResult<Py<PyType>> {
        registered_class(py, &self.map.entry)
            .ok_or_else(|| PyTypeError::new_err(format!("no class for {}", self.map.entry)))
    }

    fn __richcmp__(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return Ok(py.NotImplemented());
        }
        let equal = if let Ok(other) = other.cast::<MapContainer>() {
            Self::equal(py, slf, other)?
        } else if other.cast::<PyDict>().is_ok() {
            slf.borrow().to_dict(py)?.as_any().eq(other)?
        } else {
            return Ok(py.NotImplemented());
        };
        (if matches!(op, CompareOp::Eq) {
            equal
        } else {
            !equal
        })
        .into_py_any(py)
    }

    /// The hash of the object, as `object.__hash__` gives it: upb's maps hash by identity.
    fn __hash__(slf: &Bound<'_, Self>) -> isize {
        let hash = (slf.as_ptr() as usize).rotate_right(4) as isize;
        if hash == -1 {
            -2
        } else {
            hash
        }
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(self.to_dict(py)?.repr()?.to_string())
    }
}

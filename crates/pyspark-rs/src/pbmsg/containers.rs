//! The values of repeated and map fields, as protobuf's Python containers (those of upb, the
//! implementation of a `protobuf` install) give them: a list-like container of the scalars or
//! messages of a repeated field, a `collections.abc.MutableSequence`, and a dict-like map of a map
//! field, a `collections.abc.MutableMapping` (see `pyspark.sql.connect.proto._rust`, which
//! registers them).
//!
//! A message that an unset child holds sets the child in its parent when its container may
//! change, as setting a field of the child does: as upb's containers, at the start of each method
//! that may change it, even one that then fails.
//!
//! The Python code that a method runs (the `__index__` of a value or of an index, the iteration
//! of an iterable, an `__eq__`, the key of a sort, a warning) may use the container, of this
//! thread or of another, so no borrow of a container is held while it runs: a method reads what
//! it needs of the container, runs the Python code, and then changes the container as it is by
//! then, in upb's order. upb reads the length of a container before the `__index__` of an index
//! runs, so an index is checked against the length from before. Where upb then reads or writes
//! past the end of a container that the Python code shortened, a read is upb's `IndexError`, a
//! write is lost (as upb's is) and an insertion or a deletion takes the container as it is.

use std::sync::Arc;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyAttributeError, PyFutureWarning, PyIndexError, PyKeyError, PyNotImplementedError,
    PyTypeError, PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::types::{
    PyBytes, PyDict, PyInt, PyIterator, PyList, PySlice, PyTuple, PyType, PyWeakrefMethods,
    PyWeakrefReference,
};
use pyo3::IntoPyObjectExt;

use super::message::{
    content_eq, duplicate, duplicate_with, mark_present, message_from_value, new_instance,
    registered_class, MessageClass, PyMessage,
};
use super::pool::{Kind, MapDesc};
use super::scalar::Scalar;
use crate::interrupt::ConversionBudget;

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

/// upb's error of an index past the end of a container.
fn out_of_range() -> PyErr {
    PyIndexError::new_err("list index out of range")
}

/// A subscript of a container, as upb's `PyUpb_IndexToRange` reads one of a container of `len`
/// items: an index in its range, or the start, step and length of a slice.
enum Subscript {
    Index(usize),
    Slice {
        start: isize,
        step: isize,
        count: usize,
    },
}

impl Subscript {
    /// The subscript `key` of a container of `len` items, read after `len`, as upb reads them:
    /// the `__index__` of the key or of the bounds of a slice runs then. A key that is not an
    /// int, or is not one of an index, is upb's `TypeError` "list indices must be integers".
    fn of(key: &Bound<'_, PyAny>, len: usize) -> PyResult<Subscript> {
        if let Ok(slice) = key.cast::<PySlice>() {
            let indices = slice.indices(len as isize)?;
            return Ok(Subscript::Slice {
                start: indices.start,
                step: indices.step,
                count: indices.slicelength,
            });
        }
        let index = key
            .extract::<isize>()
            .map_err(|_| PyTypeError::new_err("list indices must be integers"))?;
        let at = if index < 0 {
            index + len as isize
        } else {
            index
        };
        if at < 0 || at as usize >= len {
            return Err(out_of_range());
        }
        Ok(Subscript::Index(at as usize))
    }

    /// The positions of the subscript, in its order.
    fn positions(&self) -> impl Iterator<Item = usize> {
        let (start, step, count) = match *self {
            Subscript::Index(at) => (at as isize, 1, 1),
            Subscript::Slice { start, step, count } => (start, step, count),
        };
        (0..count as isize).map(move |k| (start + k * step) as usize)
    }

    /// The item of an index of `items`, or the list of the items of a slice, of those that
    /// `items` still has, each of `to_py`.
    fn get<T>(
        &self,
        py: Python<'_>,
        items: &[T],
        to_py: impl Fn(&T) -> PyResult<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Subscript::Index(at) = *self {
            return to_py(items.get(at).ok_or_else(out_of_range)?);
        }
        let values = self
            .positions()
            .filter_map(|at| items.get(at))
            .map(to_py)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, values)?.into_any().unbind())
    }

    /// Remove the items of `items` at the positions of the subscript that `items` still has, and
    /// give them back.
    fn remove_from<T>(&self, items: &mut Vec<T>) -> Vec<T> {
        if let Subscript::Index(at) = *self {
            return if at < items.len() {
                vec![items.remove(at)]
            } else {
                Vec::new()
            };
        }
        let mut gone = vec![false; items.len()];
        for at in self.positions() {
            if let Some(g) = gone.get_mut(at) {
                *g = true;
            }
        }
        let mut removed = Vec::new();
        let kept = std::mem::take(items)
            .into_iter()
            .zip(gone)
            .filter_map(|(item, gone)| {
                if gone {
                    removed.push(item);
                    None
                } else {
                    Some(item)
                }
            })
            .collect();
        *items = kept;
        removed
    }
}

/// The item at `at` of `items`, of `to_py`, with upb's `IndexError` of a position past their end
/// (of a container that Python code shortened).
fn item_at<T>(
    items: &[T],
    at: usize,
    to_py: impl Fn(&T) -> PyResult<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    match items.get(at) {
        Some(item) => to_py(item),
        None => Err(PyIndexError::new_err(format!(
            "list index ({at}) out of range"
        ))),
    }
}

/// The item at `at` of `items`, removed, with upb's `IndexError` of a position past their end
/// (of a container that the warning of `pop_position` shortened).
fn remove_at<T>(items: &mut Vec<T>, at: usize) -> PyResult<T> {
    if at < items.len() {
        Ok(items.remove(at))
    } else {
        Err(PyIndexError::new_err(format!(
            "list index ({at}) out of range"
        )))
    }
}

/// The values of `value`, as upb's methods that take several values read them, of CPython's
/// `PySequence_Fast`: a list or a tuple as it is (a list whose length is read again for each
/// value, so a value that a conversion appends to it is taken too), else a list of the items of
/// its iterator, with upb's `TypeError` of a value that is not iterable.
fn sequence_fast<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    unsafe {
        Bound::from_owned_ptr_or_err(
            value.py(),
            ffi::PySequence_Fast(value.as_ptr(), c"must assign iterable".as_ptr()),
        )
    }
}

/// The value at `i` of `values`, a list or a tuple of [`sequence_fast`], if it has one there by
/// now.
fn fast_item<'py>(values: &Bound<'py, PyAny>, i: usize) -> Option<Bound<'py, PyAny>> {
    match values.cast::<PyList>() {
        Ok(list) => list.get_item(i).ok(),
        Err(_) => values.cast::<PyTuple>().ok()?.get_item(i).ok(),
    }
}

/// Whether `a` equals `b`, as `PyObject_RichCompareBool` tells it, as upb's `remove` compares:
/// an object equals itself, without its `__eq__`.
fn compares_equal(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(a.is(b) || a.eq(b)?)
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

/// A message of `class` with the fields of `value`, as `append` and `insert` of upb take one: a
/// new message that merges `value`, a message of the same type.
fn message_copy<'py>(
    py: Python<'py>,
    class: &MessageClass,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyMessage>> {
    let message = class.new_message(py)?;
    PyMessage::merge_from(&message, value)?;
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

/// The field of a container of scalars, which its values are converted for: the kind of the
/// field, its full name, which the error of a value names, and the full name of its enum, of an
/// enum field.
pub(crate) struct ScalarField {
    kind: Kind,
    name: String,
    enum_type: String,
}

impl ScalarField {
    fn convert(&self, value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
        Scalar::from_py(self.kind, &self.name, &self.enum_type, value)
    }
}

/// The values of a container of scalars for its numpy array: the native bytes of the values of
/// a numeric field and the name of their dtype, or the values of a string or bytes field.
enum ArrayValues {
    Native(Vec<u8>, &'static str),
    Strings(Vec<Py<PyAny>>),
    Bytes(Vec<Py<PyAny>>),
}

/// The `dtype` that `__array__` is given, if it is given (None too), of its arguments as upb's
/// takes them: a `dtype` and a `copy`, by position or by name.
fn array_dtype_argument<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let given = args.len() + kwargs.map_or(0, |k| k.len());
    if given > 2 {
        return Err(PyTypeError::new_err(format!(
            "function takes at most 2 arguments ({given} given)"
        )));
    }
    let mut dtype = args.get_item(0).ok();
    for (key, value) in kwargs.iter().flat_map(|k| k.iter()) {
        let key = key.str()?.to_string();
        let position = match key.as_str() {
            "dtype" => 1,
            "copy" => 2,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "'{key}' is an invalid keyword argument for this function"
                )))
            }
        };
        if args.len() >= position {
            return Err(PyTypeError::new_err(format!(
                "argument for function given by name ('{key}') and position ({position})"
            )));
        }
        if position == 1 {
            dtype = Some(value);
        }
    }
    Ok(dtype)
}

/// Make the `count` items of `items` from `start` `given` items, as upb's assignment of a slice
/// of another length moves the items after the slice (`upb_Array_Resize`, then `upb_Array_Move`):
/// they follow the `given` positions, which keep the items that the container had there (the
/// default of `kind` where it had none) until the assignment sets them. A container that is
/// shorter than the slice by now (where upb's count of the items after the slice wraps around)
/// has the part of the slice that it still has made `given` items. The start of the slice, in the
/// container as it is.
fn resize_slice(
    items: &mut Vec<Scalar>,
    start: usize,
    count: usize,
    given: usize,
    kind: Kind,
) -> usize {
    let len = items.len();
    let start = start.min(len);
    let count = count.min(len - start);
    let (from, to) = (start + count, start + given);
    if to > from {
        let tail = items[from..].to_vec();
        items.truncate(to.min(len));
        items.resize(to, Scalar::default_of(kind));
        items.extend(tail);
    } else {
        items.drain(to..from);
    }
    start
}

#[pyclass(
    sequence,
    module = "pyspark.sql.connect.proto",
    name = "RepeatedScalarContainer"
)]
pub struct RepeatedScalar {
    /// The field of the container, which its methods convert values for, with no borrow of the
    /// container held.
    field: Arc<ScalarField>,
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
            field: Arc::new(ScalarField {
                kind,
                name: field,
                enum_type,
            }),
            items,
            owner,
        }
    }

    /// A copy of the container, held by no message.
    pub(crate) fn copy(&self) -> Self {
        RepeatedScalar {
            field: self.field.clone(),
            items: self.items.clone(),
            owner: None,
        }
    }

    fn field_of(slf: &Bound<'_, Self>) -> Arc<ScalarField> {
        slf.borrow().field.clone()
    }

    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let values = self
            .items
            .iter()
            .map(|v| v.to_py(py))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, values)
    }

    /// The values for the numpy array of `__array__`, of the dtype of the field.
    fn array_values(&self, py: Python<'_>) -> PyResult<ArrayValues> {
        fn native<const N: usize>(
            items: &[Scalar],
            dtype: &'static str,
            bytes: impl Fn(&Scalar) -> [u8; N],
        ) -> ArrayValues {
            let mut out = Vec::with_capacity(items.len() * N);
            for item in items {
                out.extend_from_slice(&bytes(item));
            }
            ArrayValues::Native(out, dtype)
        }
        let items = &self.items;
        Ok(match self.field.kind {
            Kind::Double => native(items, "float64", |v| match v {
                Scalar::F64(v) => v.to_ne_bytes(),
                _ => [0; 8],
            }),
            Kind::Float => native(items, "float32", |v| match v {
                Scalar::F32(v) => v.to_ne_bytes(),
                _ => [0; 4],
            }),
            Kind::Int64 | Kind::SInt64 | Kind::SFixed64 => native(items, "int64", |v| match v {
                Scalar::I64(v) => v.to_ne_bytes(),
                _ => [0; 8],
            }),
            Kind::UInt64 | Kind::Fixed64 => native(items, "uint64", |v| match v {
                Scalar::U64(v) => v.to_ne_bytes(),
                _ => [0; 8],
            }),
            Kind::Int32 | Kind::SInt32 | Kind::SFixed32 => native(items, "int32", |v| match v {
                Scalar::I32(v) => v.to_ne_bytes(),
                _ => [0; 4],
            }),
            Kind::UInt32 | Kind::Fixed32 => native(items, "uint32", |v| match v {
                Scalar::U32(v) => v.to_ne_bytes(),
                _ => [0; 4],
            }),
            Kind::Enum => native(items, "int32", |v| match v {
                Scalar::Enum(v) => v.to_ne_bytes(),
                _ => [0; 4],
            }),
            Kind::Bool => native(items, "bool", |v| [matches!(v, Scalar::Bool(true)) as u8]),
            Kind::String => {
                ArrayValues::Strings(items.iter().map(|v| v.to_py(py)).collect::<PyResult<_>>()?)
            }
            Kind::Bytes | Kind::Message => {
                ArrayValues::Bytes(items.iter().map(|v| v.to_py(py)).collect::<PyResult<_>>()?)
            }
        })
    }

    /// Set the message of the container in its parent, as upb's methods that may change a
    /// container do first.
    fn reify(slf: &Bound<'_, Self>) -> PyResult<()> {
        let owner = owner_of(slf.py(), &slf.borrow().owner);
        touch(slf.py(), &owner)
    }

    /// Append the values of `values` (of `PySequence_Fast`), each converted in turn.
    fn append_values(
        slf: &Bound<'_, Self>,
        field: &ScalarField,
        values: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let values = sequence_fast(values)?;
        let mut i = 0;
        while let Some(value) = fast_item(&values, i) {
            let v = field.convert(&value)?;
            slf.borrow_mut().items.push(v);
            i += 1;
        }
        Ok(())
    }

    /// Set the item at `at` to `value`, as upb's `upb_Array_Set`, whose write past the end of a
    /// container that the Python code of a conversion shortened is lost.
    fn set_at(slf: &Bound<'_, Self>, at: usize, value: Scalar) {
        if let Some(item) = slf.borrow_mut().items.get_mut(at) {
            *item = value;
        }
    }

    /// Assign the values of `value` to the slice of `start`, `step` and `count` (read of the
    /// container before), as upb's: the values (of `PySequence_Fast`), then for a slice of a step
    /// of 1 and of another length the items after it moved, then each value converted and set in
    /// turn, so a value that fails leaves those before it set.
    fn assign_slice(
        slf: &Bound<'_, Self>,
        field: &ScalarField,
        start: isize,
        step: isize,
        count: usize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let values = sequence_fast(value)?;
        let given = values.len()?;
        let (mut start, mut count) = (start, count);
        if given != count {
            if step != 1 {
                return Err(PyValueError::new_err(format!(
                    "attempt to assign sequence of  {given} to extended slice of size {count}"
                )));
            }
            let mut me = slf.borrow_mut();
            start = resize_slice(&mut me.items, start as usize, count, given, field.kind) as isize;
            count = given;
        }
        for k in 0..count {
            // A value of a list that a conversion shortened is upb's error of `PySequence_GetItem`.
            let value = fast_item(&values, k).ok_or_else(out_of_range)?;
            let v = field.convert(&value)?;
            Self::set_at(slf, (start + k as isize * step) as usize, v);
        }
        Ok(())
    }
}

#[pymethods]
impl RepeatedScalar {
    fn append(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let v = Self::field_of(slf).convert(value)?;
        slf.borrow_mut().items.push(v);
        Ok(())
    }

    /// As upb's: the values of `elem_seq` (of `PySequence_Fast`, so those of an iterator to its
    /// end first), each converted and appended in turn. A failure, of the iteration too, leaves
    /// the container of the length that it had: upb resizes it, so a position that Python code
    /// removed meanwhile holds the default.
    fn extend(slf: &Bound<'_, Self>, elem_seq: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let field = Self::field_of(slf);
        let start = slf.borrow().items.len();
        let extended = Self::append_values(slf, &field, elem_seq);
        if extended.is_err() {
            slf.borrow_mut()
                .items
                .resize(start, Scalar::default_of(field.kind));
        }
        extended
    }

    /// As upb's: the position of `key` in the container as it is, then the value converted. A
    /// conversion that shortened the container leaves the position past its end (where upb's
    /// insert writes out of its bounds), so the value goes at its end.
    fn insert(slf: &Bound<'_, Self>, key: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let at = insert_position(key, slf.borrow().items.len());
        let v = Self::field_of(slf).convert(value)?;
        let mut me = slf.borrow_mut();
        let at = at.min(me.items.len());
        me.items.insert(at, v);
        Ok(())
    }

    #[pyo3(signature = (key=-1))]
    fn pop(slf: &Bound<'_, Self>, key: isize) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        let at = pop_position(py, key, len)?;
        let value = remove_at(&mut slf.borrow_mut().items, at)?;
        value.to_py(py)
    }

    /// As upb's: the first item that equals `elem`, as Python compares them, among as many items
    /// as the container had, each of the container as it is when it is compared (an `__eq__` may
    /// change it).
    fn remove(slf: &Bound<'_, Self>, elem: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        for at in 0..len {
            let item = item_at(&slf.borrow().items, at, |v| v.to_py(py))?;
            if compares_equal(item.bind(py), elem)? {
                let mut me = slf.borrow_mut();
                if at < me.items.len() {
                    me.items.remove(at);
                }
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

    /// As upb's: the items, sorted as a list, then set at their positions in the container as it
    /// is by then, so items that a key appended stay after them, and positions that it removed are
    /// not written. No change of an unset container (an empty one), so no change of its message.
    #[pyo3(signature = (*args, **kwargs))]
    fn sort(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let list = slf.borrow().to_list(py)?;
        let kwargs = sort_kwargs(py, kwargs)?;
        list.call_method("sort", args, Some(&kwargs))?;
        let field = Self::field_of(slf);
        for (at, value) in list.iter().enumerate() {
            let v = field.convert(&value)?;
            Self::set_at(slf, at, v);
        }
        Ok(())
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
        self.copy()
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Err(pickle_error(py))
    }

    /// The values as a numpy array, as upb's `__array__` gives them: an array of the type of the
    /// field (`int32`, `int64`, `uint32` or `uint64` of an integer, `float64` of a double,
    /// `float32` of a float, `bool`, `int32` of an enum, `S` of bytes and `str` of strings), cast
    /// to `numpy.dtype(dtype)` when `dtype` is given (None too, which is `float64`). `copy` is
    /// taken and not read: the array is always a new one. The values of a numeric field go into
    /// the array at once, not as a Python object each.
    #[pyo3(signature = (*args, **kwargs))]
    fn __array__<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let dtype = array_dtype_argument(args, kwargs)?;
        let values = slf.borrow().array_values(py)?;
        let numpy = py.import("numpy")?;
        let dtype = match dtype {
            Some(dtype) => Some(numpy.getattr("dtype")?.call1((dtype,))?),
            None => None,
        };
        // As upb's: the values of a numeric field are cast from an array of their type, those of
        // a string or bytes field converted from their list.
        let (values, field_dtype) = match values {
            ArrayValues::Native(bytes, field_dtype) => {
                let array = numpy
                    .call_method1("frombuffer", (PyBytes::new(py, &bytes), field_dtype))?
                    .call_method0("copy")?;
                return match dtype {
                    Some(dtype) => array.call_method1("astype", (dtype,)),
                    None => Ok(array),
                };
            }
            ArrayValues::Strings(values) => {
                (values, py.get_type::<pyo3::types::PyString>().into_any())
            }
            ArrayValues::Bytes(values) => (values, "S".into_pyobject(py)?.into_any()),
        };
        numpy.call_method1("array", (values, dtype.unwrap_or(field_dtype)))
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        // The index of an int runs no Python code, so the length and the item are read at once.
        if key.is_exact_instance_of::<PyInt>() {
            let me = slf.borrow();
            return Subscript::of(key, me.items.len())?.get(py, &me.items, |v| v.to_py(py));
        }
        let len = slf.borrow().items.len();
        let subscript = Subscript::of(key, len)?;
        subscript.get(py, &slf.borrow().items, |v| v.to_py(py))
    }

    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        Self::reify(slf)?;
        let field = Self::field_of(slf);
        let len = slf.borrow().items.len();
        match Subscript::of(key, len)? {
            Subscript::Index(at) => {
                let v = field.convert(value)?;
                Self::set_at(slf, at, v);
                Ok(())
            }
            Subscript::Slice { start, step, count } => {
                Self::assign_slice(slf, &field, start, step, count, value)
            }
        }
    }

    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        let subscript = Subscript::of(key, len)?;
        subscript.remove_from(&mut slf.borrow_mut().items);
        Ok(())
    }

    /// As upb's: the list of the items compared with `other`, or with the list of the items of
    /// `other` when it is a container of the same kind. (`in` reads the items in turn, of the
    /// container as it is, as upb's, which have no `__contains__`.)
    fn __richcmp__(
        slf: &Bound<'_, Self>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return Ok(py.NotImplemented());
        }
        let mine = slf.borrow().to_list(py)?;
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
    /// The class of the messages of the container, which makes its new ones.
    pub(crate) class: MessageClass,
    pub(crate) items: Vec<Py<PyMessage>>,
    owner: Owner,
}

impl RepeatedComposite {
    pub(crate) fn new(class: MessageClass, items: Vec<Py<PyMessage>>, owner: Owner) -> Self {
        RepeatedComposite {
            class,
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
}

#[pymethods]
impl RepeatedComposite {
    /// As upb's: a new message of the fields of `kwargs`, appended; a positional argument is the
    /// error of a constructor's.
    #[pyo3(signature = (*args, **kwargs))]
    fn add<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let py = slf.py();
        if !args.is_empty() {
            return Err(PyTypeError::new_err("No positional arguments allowed"));
        }
        Self::reify(slf)?;
        let class = slf.borrow().class.clone_ref(py);
        let child = class.new_message(py)?;
        if let Some(kwargs) = kwargs {
            child.borrow_mut().init_from_mapping(py, kwargs.as_any())?;
        }
        slf.borrow_mut().items.push(child.clone().unbind());
        Ok(child)
    }

    /// As upb's: a new message, appended, then merged with `value` and returned, so that a message
    /// of another type leaves the new message (empty) in the container.
    fn append<'py>(
        slf: &Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let py = slf.py();
        Self::reify(slf)?;
        verify_message(value)?;
        let class = slf.borrow().class.clone_ref(py);
        let child = class.new_message(py)?;
        slf.borrow_mut().items.push(child.clone().unbind());
        PyMessage::merge_from(&child, value)?;
        Ok(child)
    }

    /// As upb's: each value of the iterator of `elem_seq` appended as it comes, so a failure
    /// leaves the values before it appended.
    fn extend(slf: &Bound<'_, Self>, elem_seq: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        for value in elem_seq.try_iter()? {
            Self::append(slf, &value?)?;
        }
        Ok(())
    }

    /// As upb's: the position of `key` in the container as it is, then the copy of `value`,
    /// which goes at the end of a container that is shorter by then.
    fn insert(slf: &Bound<'_, Self>, key: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let (class, at) = {
            let me = slf.borrow();
            (me.class.clone_ref(py), insert_position(key, me.items.len()))
        };
        let copy = message_copy(py, &class, value)?;
        let mut me = slf.borrow_mut();
        let at = at.min(me.items.len());
        me.items.insert(at, copy.unbind());
        Ok(())
    }

    #[pyo3(signature = (key=-1))]
    fn pop(slf: &Bound<'_, Self>, key: isize) -> PyResult<Py<PyMessage>> {
        let py = slf.py();
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        let at = pop_position(py, key, len)?;
        remove_at(&mut slf.borrow_mut().items, at)
    }

    /// As upb's: the first message that equals `elem`, among as many as the container had, each
    /// of the container as it is when it is compared.
    fn remove(slf: &Bound<'_, Self>, elem: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        for at in 0..len {
            let item = item_at(&slf.borrow().items, at, |m| Ok(m.clone_ref(py).into_any()))?;
            if compares_equal(item.bind(py), elem)? {
                // A message removed is dropped once the container is no longer borrowed.
                let _removed = {
                    let mut me = slf.borrow_mut();
                    (at < me.items.len()).then(|| me.items.remove(at))
                };
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

    /// As upb's: the messages, sorted as a list, then set at their positions in the container as
    /// it is by then (see the `sort` of the container of scalars).
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
        let sorted = list
            .iter()
            .map(|m| Ok(m.cast_into::<PyMessage>()?.unbind()))
            .collect::<PyResult<Vec<_>>>()?;
        // The list holds the messages that the assignment replaces, so none is freed here.
        for (item, message) in slf.borrow_mut().items.iter_mut().zip(sorted) {
            *item = message;
        }
        Ok(())
    }

    /// As upb's: no change of an unset container (an empty one), so no change of its message.
    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        let _items = std::mem::take(&mut slf.borrow_mut().items);
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
            self.class.clone_ref(py),
            items,
            None,
        ))
    }

    fn __len__(&self) -> usize {
        self.items.len()
    }

    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let item = |m: &Py<PyMessage>| Ok(m.clone_ref(py).into_any());
        // The index of an int runs no Python code, so the length and the item are read at once.
        if key.is_exact_instance_of::<PyInt>() {
            let me = slf.borrow();
            return Subscript::of(key, me.items.len())?.get(py, &me.items, item);
        }
        let len = slf.borrow().items.len();
        let subscript = Subscript::of(key, len)?;
        subscript.get(py, &slf.borrow().items, item)
    }

    /// As upb's: the message is set in its parent and the subscript is read, and then nothing is
    /// assigned.
    fn __setitem__(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let _ = value;
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        Subscript::of(key, len)?;
        Err(PyTypeError::new_err("does not support assignment"))
    }

    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        let subscript = Subscript::of(key, len)?;
        // The messages removed are dropped once the container is no longer borrowed.
        let _removed = subscript.remove_from(&mut slf.borrow_mut().items);
        Ok(())
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
        let mine = slf.borrow().to_list(py)?;
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

/// The key of `key` for a map of `map`, with the errors of its entry's `key` field.
fn key_of(map: &MapDesc, key: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    Scalar::from_py(map.key, &format!("{}.key", map.entry), "", key)
}

/// The value of `value` for a map of `map` of scalars, with the errors of its entry's `value`.
fn value_of(map: &MapDesc, value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    Scalar::from_py(
        map.value,
        &format!("{}.value", map.entry),
        &map.value_type,
        value,
    )
}

/// The entries of a map read for a copy, of the messages of the map, which the copy copies with
/// no borrow of the map held.
pub(crate) struct MapEntries {
    field_name: String,
    map: Arc<MapDesc>,
    entries: Vec<(Scalar, MapValue)>,
}

impl MapEntries {
    fn copy_entries(
        self,
        py: Python<'_>,
        budget: &ConversionBudget,
    ) -> PyResult<Vec<(Scalar, MapValue)>> {
        let mut copies = Vec::with_capacity(self.entries.len());
        for (key, value) in self.entries {
            let value = match value {
                MapValue::Scalar(v) => MapValue::Scalar(v),
                MapValue::Message(m) => {
                    MapValue::Message(duplicate_with(py, m.bind(py), budget)?.unbind())
                }
            };
            copies.push((key, value));
        }
        Ok(copies)
    }

    /// A map of copies of the entries, held by no message yet.
    pub(crate) fn copy(self, py: Python<'_>, budget: &ConversionBudget) -> PyResult<MapContainer> {
        let field_name = self.field_name.clone();
        let map = self.map.clone();
        Ok(MapContainer {
            field_name,
            map,
            entries: self.copy_entries(py, budget)?,
            owner: None,
        })
    }
}

#[pyclass(
    mapping,
    subclass,
    module = "pyspark.sql.connect.proto",
    name = "MapContainer"
)]
pub struct MapContainer {
    field_name: String,
    /// The key and value of the map, which its methods convert keys and values for, with no
    /// borrow of the map held.
    pub(crate) map: Arc<MapDesc>,
    /// The entries, in the order of their insertion.
    pub(crate) entries: Vec<(Scalar, MapValue)>,
    owner: Owner,
}

impl MapContainer {
    pub(crate) fn new(field_name: String, map: Arc<MapDesc>, owner: Owner) -> Self {
        MapContainer {
            field_name,
            map,
            entries: Vec::new(),
            owner,
        }
    }

    /// The Python object of `map`: a `MessageMapContainer` of a map of messages, which has
    /// `get_or_create` as upb's of the same name has, else a `MapContainer`.
    pub(crate) fn into_object(self, py: Python<'_>) -> PyResult<Py<MapContainer>> {
        if self.map.value != Kind::Message {
            return Py::new(py, self);
        }
        let init = PyClassInitializer::from(self).add_subclass(MessageMapContainer);
        Ok(Bound::new(py, init)?.into_super().unbind())
    }

    fn map_of(slf: &Bound<'_, Self>) -> Arc<MapDesc> {
        slf.borrow().map.clone()
    }

    fn find(&self, key: &Scalar) -> Option<usize> {
        self.entries.iter().position(|(k, _)| k == key)
    }

    fn reify(slf: &Bound<'_, Self>) -> PyResult<()> {
        let owner = owner_of(slf.py(), &slf.borrow().owner);
        touch(slf.py(), &owner)
    }

    /// Set the value of `key`, replacing the one it has, which it gives back (to drop once the
    /// map is no longer borrowed).
    pub(crate) fn insert(&mut self, key: Scalar, value: MapValue) -> Option<MapValue> {
        match self.find(&key) {
            Some(at) => Some(std::mem::replace(&mut self.entries[at].1, value)),
            None => {
                self.entries.push((key, value));
                None
            }
        }
    }

    fn value_to_py(py: Python<'_>, value: &MapValue) -> PyResult<Py<PyAny>> {
        match value {
            MapValue::Scalar(v) => v.to_py(py),
            MapValue::Message(m) => Ok(m.clone_ref(py).into_any()),
        }
    }

    /// The value of `key` in the map, if it has one.
    fn lookup(slf: &Bound<'_, Self>, key: &Scalar) -> PyResult<Option<Py<PyAny>>> {
        let me = slf.borrow();
        me.find(key)
            .map(|at| Self::value_to_py(slf.py(), &me.entries[at].1))
            .transpose()
    }

    /// The entries of the map, read for a copy: their keys and scalars copied, and their messages,
    /// which [`MapEntries::copy`] copies once the map is no longer borrowed.
    pub(crate) fn entries_for_copy(&self, py: Python<'_>) -> MapEntries {
        MapEntries {
            field_name: self.field_name.clone(),
            map: self.map.clone(),
            entries: self
                .entries
                .iter()
                .map(|(k, v)| {
                    let value = match v {
                        MapValue::Scalar(v) => MapValue::Scalar(v.clone()),
                        MapValue::Message(m) => MapValue::Message(m.clone_ref(py)),
                    };
                    (k.clone(), value)
                })
                .collect(),
        }
    }

    /// Set the entries of `source` in `target`, of copies of its messages.
    pub(crate) fn merge(
        py: Python<'_>,
        target: &Bound<'_, Self>,
        source: &Bound<'_, Self>,
    ) -> PyResult<()> {
        let entries = source.borrow().entries_for_copy(py);
        Self::merge_entries(py, target, entries, &ConversionBudget::full())
    }

    /// Set `entries` in `target`, of copies of their messages, a step of `budget` for each
    /// message copied.
    pub(crate) fn merge_entries(
        py: Python<'_>,
        target: &Bound<'_, Self>,
        entries: MapEntries,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        let copies = entries.copy_entries(py, budget)?;
        let _replaced: Vec<MapValue> = {
            let mut t = target.borrow_mut();
            copies
                .into_iter()
                .filter_map(|(k, v)| t.insert(k, v))
                .collect()
        };
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
        let map = Self::map_of(target);
        let items = mapping.call_method0("items")?;
        for item in items.try_iter()? {
            let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
            let key = key_of(&map, &key)?;
            let value = if map.value == Kind::Message {
                MapValue::Message(message_from_value(py, &map.value_type, &value)?.unbind())
            } else {
                MapValue::Scalar(value_of(&map, &value)?)
            };
            let _replaced = target.borrow_mut().insert(key, value);
        }
        Ok(())
    }

    /// The value of `key`, inserting the default of the value type (a new message, of a message
    /// map) when the map has none, as upb's `map[key]` does.
    fn get_or_insert(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let map = Self::map_of(slf);
        let key = key_of(&map, key)?;
        if let Some(value) = Self::lookup(slf, &key)? {
            return Ok(value);
        }
        Self::reify(slf)?;
        let value = if map.value == Kind::Message {
            MapValue::Message(new_instance(py, &map.value_type)?.unbind())
        } else {
            MapValue::Scalar(Scalar::default_of(map.value))
        };
        let mut me = slf.borrow_mut();
        // The value of the key, if Python code that making the value ran has set one.
        if let Some(at) = me.find(&key) {
            return Self::value_to_py(py, &me.entries[at].1);
        }
        let out = Self::value_to_py(py, &value)?;
        me.entries.push((key, value));
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
        let map = Self::map_of(slf);
        let key = key_of(&map, key)?;
        if map.value == Kind::Message {
            return Err(PyValueError::new_err("Message objects may not be assigned"));
        }
        let value = MapValue::Scalar(value_of(&map, value)?);
        let _replaced = slf.borrow_mut().insert(key, value);
        Ok(())
    }

    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        let key = key_of(&Self::map_of(slf), key)?;
        // The entry removed is dropped once the map is no longer borrowed.
        let _removed = {
            let mut me = slf.borrow_mut();
            let at = me
                .find(&key)
                .ok_or_else(|| PyKeyError::new_err("Key not present in map"))?;
            me.entries.remove(at)
        };
        Ok(())
    }

    fn __contains__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let key = key_of(&Self::map_of(slf), key)?;
        Ok(slf.borrow().find(&key).is_some())
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
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let key = key_of(&Self::map_of(slf), key)?;
        Ok(Self::lookup(slf, &key)?.unwrap_or_else(|| default.unwrap_or_else(|| slf.py().None())))
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
            Some((key, _)) => key.clone(),
            None => return Err(PyKeyError::new_err(())),
        };
        let key = key.to_py(py)?.into_bound(py);
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
        let map = Self::map_of(slf);
        if map.value == Kind::Message {
            return Err(PyNotImplementedError::new_err(
                "Set message map value directly is not supported.",
            ));
        }
        let Some(default) = default.filter(|d| !d.is_none()) else {
            return Err(PyValueError::new_err(
                "The value for scalar map setdefault must be set.",
            ));
        };
        if let Some(value) = Self::lookup(slf, &key_of(&map, key)?)? {
            return Ok(value);
        }
        Self::__setitem__(slf, key, default)?;
        Ok(default.clone().unbind())
    }

    fn clear(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::reify(slf)?;
        // The entries are dropped once the map is no longer borrowed.
        let _entries = std::mem::take(&mut slf.borrow_mut().entries);
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
        let map = Self::map_of(slf);
        for pair in other.try_iter()? {
            let (key, value) = unpack_pair(&pair?)?;
            if map.value == Kind::Message {
                // upb converts the value as a scalar, which a message is not.
                key_of(&map, &key)?;
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
            let mine = slf.borrow().to_dict(py)?;
            mine.as_any().eq(other)?
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

/// The map of a map field of messages, as upb's `MessageMapContainer`: a `MapContainer` with
/// `get_or_create`.
#[pyclass(
    extends = MapContainer,
    mapping,
    module = "pyspark.sql.connect.proto",
    name = "MessageMapContainer"
)]
pub struct MessageMapContainer;

#[pymethods]
impl MessageMapContainer {
    /// The value of `key`, inserted as a new message if the map has none, as `map[key]` gives it,
    /// with upb's errors of its one argument.
    #[pyo3(signature = (*args, **kwargs))]
    fn get_or_create(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        if kwargs.is_some_and(|k| !k.is_empty()) {
            return Err(PyTypeError::new_err(
                "MessageMapContainer.get_or_create() takes no keyword arguments",
            ));
        }
        if args.len() != 1 {
            return Err(PyTypeError::new_err(format!(
                "MessageMapContainer.get_or_create() takes exactly one argument ({} given)",
                args.len()
            )));
        }
        MapContainer::get_or_insert(slf.as_super(), &args.get_item(0)?)
    }
}

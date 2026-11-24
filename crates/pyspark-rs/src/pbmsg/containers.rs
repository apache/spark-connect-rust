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

use std::hash::{BuildHasher, Hash, Hasher, RandomState};
use std::sync::{Arc, OnceLock};

use hashbrown::HashTable;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyAttributeError, PyFutureWarning, PyIndexError, PyKeyError, PyNotImplementedError,
    PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PySlice, PyString, PyTuple, PyType,
    PyWeakrefMethods, PyWeakrefReference,
};
use pyo3::IntoPyObjectExt;

use super::arena::{NRef, Releases};
use super::message::{
    content_eq, duplicate_with, mark_present, new_instance, registered_class, MessageClass,
    PyMessage,
};
use super::native::{self, CopyError, NativeMap, NativeMessages, NativeValue};
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
        Ok(list) => (i < list.len()).then(|| list.get_item(i).ok()).flatten(),
        Err(_) => {
            let tuple = values.cast::<PyTuple>().ok()?;
            (i < tuple.len()).then(|| tuple.get_item(i).ok()).flatten()
        }
    }
}

/// Whether `value` converts to a scalar of no Python code: an exact `int`, `float`, `bool`,
/// `str` or `bytes`, whose conversion (or its error) is CPython's, of no method of a subclass, as
/// `__index__` and `__float__` of other objects are.
fn converts_natively(value: &Bound<'_, PyAny>) -> bool {
    value.is_exact_instance_of::<PyInt>()
        || value.is_exact_instance_of::<PyFloat>()
        || value.is_exact_instance_of::<PyBool>()
        || value.is_exact_instance_of::<PyString>()
        || value.is_exact_instance_of::<PyBytes>()
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

/// upb's errors of the arguments of a method `method` of a container of upb's class `class`
/// (`ScalarMapContainer`, `MessageMapContainer`, `RepeatedScalarContainer`, ...) that takes none
/// (`METH_NOARGS`), as CPython checks them: a keyword argument first.
fn no_arguments(
    class: &str,
    method: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    no_keywords(class, method, kwargs)?;
    if !args.is_empty() {
        return Err(PyTypeError::new_err(format!(
            "{class}.{method}() takes no arguments ({} given)",
            args.len()
        )));
    }
    Ok(())
}

/// upb's errors of the arguments of a method of a container that takes exactly one (`METH_O`), as
/// [`no_arguments`] gives them: the argument.
fn one_argument<'py>(
    class: &str,
    method: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    no_keywords(class, method, kwargs)?;
    if args.len() != 1 {
        return Err(PyTypeError::new_err(format!(
            "{class}.{method}() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    args.get_item(0)
}

/// upb's error of a keyword argument of a method of a container that takes one or none
/// (`METH_O`, `METH_NOARGS`), which names the method of its class however it is called.
fn no_keywords(class: &str, method: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
    if kwargs.is_some_and(|k| !k.is_empty()) {
        return Err(PyTypeError::new_err(format!(
            "{class}.{method}() takes no keyword arguments"
        )));
    }
    Ok(())
}

/// upb's error of a keyword argument of a method of a container that takes positional arguments
/// (`METH_VARARGS`: `setdefault`, `insert`, `pop`, `__deepcopy__`), which CPython words by the
/// way that the method is called: called of its object (`map.setdefault(key="k")`), it names the
/// method of its class (`ScalarMapContainer.setdefault() takes no keyword arguments`), and called
/// as a bound method, it names the method only (`setdefault() takes no keyword arguments`). The
/// runtime's methods cannot tell the two apart, so the error names the method of its class, but
/// where CPython binds the method of every call: before 3.12, while a trace or a profile function
/// is set on the thread (as `coverage` and debuggers set one), CPython's `trace_call_function`
/// calls a method of a native type as a bound method, so that a profiler sees it.
fn no_keywords_of_positional(
    py: Python<'_>,
    class: &str,
    method: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    if kwargs.is_none_or(|k| k.is_empty()) {
        return Ok(());
    }
    let bound = py.version_info() < (3, 12) && {
        let sys = py.import("sys")?;
        !sys.call_method0("gettrace")?.is_none() || !sys.call_method0("getprofile")?.is_none()
    };
    Err(PyTypeError::new_err(if bound {
        format!("{method}() takes no keyword arguments")
    } else {
        format!("{class}.{method}() takes no keyword arguments")
    }))
}

/// The `key` and `default` of upb's `get` of a map, which parses them as CPython's
/// `PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", {"key", "default"})` does, with its errors
/// (of a "function", as the format names none): too many arguments, a missing `key`, an argument
/// given by name and position, then a keyword of no argument.
fn get_arguments<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)> {
    const NAMES: [&str; 2] = ["key", "default"];
    let nargs = args.len();
    let nkwargs = kwargs.map_or(0, |k| k.len());
    if nargs + nkwargs > NAMES.len() {
        return Err(PyTypeError::new_err(format!(
            "function takes at most {} {}arguments ({} given)",
            NAMES.len(),
            if nargs == 0 { "keyword " } else { "" },
            nargs + nkwargs
        )));
    }
    let keyword = |name: &str| -> PyResult<Option<Bound<'py, PyAny>>> {
        match kwargs {
            Some(kwargs) => kwargs.get_item(name),
            None => Ok(None),
        }
    };
    let mut values: [Option<Bound<'py, PyAny>>; 2] = [None, None];
    for (i, name) in NAMES.iter().enumerate() {
        values[i] = if i < nargs {
            Some(args.get_item(i)?)
        } else {
            keyword(name)?
        };
        if values[i].is_none() && i == 0 {
            return Err(PyTypeError::new_err(
                "function missing required argument 'key' (pos 1)",
            ));
        }
    }
    for (i, name) in NAMES.iter().enumerate().take(nargs) {
        if keyword(name)?.is_some() {
            return Err(PyTypeError::new_err(format!(
                "argument for function given by name ('{name}') and position ({})",
                i + 1
            )));
        }
    }
    for key in kwargs.iter().flat_map(|k| k.keys()) {
        if !key
            .extract::<String>()
            .is_ok_and(|k| NAMES.contains(&k.as_str()))
        {
            return Err(PyTypeError::new_err(format!(
                "'{}' is an invalid keyword argument for this function",
                key.str()?
            )));
        }
    }
    let [key, default] = values;
    Ok((key.expect("the key, which is required"), default))
}

/// `arg` as an index of upb's `PyArg_ParseTuple` "n" format: of its `__index__`, with CPython's
/// errors ("'str' object cannot be interpreted as an integer", "Python int too large to convert to
/// C ssize_t").
fn ssize_t_of(arg: &Bound<'_, PyAny>) -> PyResult<isize> {
    let py = arg.py();
    // SAFETY: the pointers are of live objects; `PyNumber_Index` gives a new reference, owned by
    // the `Bound` that wraps it, or null with an exception set.
    unsafe {
        let index = ffi::PyNumber_Index(arg.as_ptr());
        if index.is_null() {
            return Err(PyErr::fetch(py));
        }
        let index = Bound::from_owned_ptr(py, index);
        let value = ffi::PyLong_AsSsize_t(index.as_ptr());
        match PyErr::take(py) {
            Some(error) if value == -1 => Err(error),
            _ => Ok(value),
        }
    }
}

/// The index and the value of upb's `insert` of a repeated field of `class`, as it takes them of
/// `PyArg_ParseTuple(args, "nO")`: of no keyword argument (`METH_VARARGS`), exactly two, the index
/// of [`ssize_t_of`].
fn insert_arguments<'py>(
    class: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(isize, Bound<'py, PyAny>)> {
    no_keywords_of_positional(args.py(), class, "insert", kwargs)?;
    if args.len() != 2 {
        return Err(PyTypeError::new_err(format!(
            "function takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    Ok((ssize_t_of(&args.get_item(0)?)?, args.get_item(1)?))
}

/// The index of upb's `pop` of a repeated field of `class`, as it takes it of
/// `PyArg_ParseTuple(args, "|n")`: of no keyword argument, at most one, -1 of none.
fn pop_argument(
    class: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<isize> {
    no_keywords_of_positional(args.py(), class, "pop", kwargs)?;
    match args.len() {
        0 => Ok(-1),
        1 => ssize_t_of(&args.get_item(0)?),
        given => Err(PyTypeError::new_err(format!(
            "function takes at most 1 argument ({given} given)"
        ))),
    }
}

/// The method `name` of `collections.abc`'s class `abc` (`Mapping`, `MutableMapping`) called of
/// `map` and of `args` and `kwargs`, as upb's maps have it: so its result, and the errors of its
/// arguments, are Python's (`MutableMapping.pop() missing 1 required positional argument: 'key'`).
fn abc_method(
    map: &Bound<'_, MapContainer>,
    abc: &str,
    name: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let py = map.py();
    let method = py.import("collections.abc")?.getattr(abc)?.getattr(name)?;
    let mut all = Vec::with_capacity(args.len() + 1);
    all.push(map.as_any().clone());
    all.extend(args.iter());
    Ok(method.call(PyTuple::new(py, all)?, kwargs)?.unbind())
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

/// Whether `value` is a message, of this runtime or of protobuf's (`google.protobuf.message
/// .Message`, of any type), as upb's `append` checks it: an object that only has a `DESCRIPTOR` and
/// a `SerializeToString` is not. A message of another type is the error of its merge.
fn verify_message(value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.cast::<PyMessage>().is_ok() || is_protobufs_message(value) {
        return Ok(());
    }
    let repr = value.repr().map(|r| r.to_string()).unwrap_or_default();
    Err(PyTypeError::new_err(format!(
        "Expected a message object, but got {repr}."
    )))
}

/// Whether `value` is a message of protobuf's (`google.protobuf.message.Message`), where protobuf
/// is installed.
fn is_protobufs_message(value: &Bound<'_, PyAny>) -> bool {
    value
        .py()
        .import("google.protobuf.message")
        .and_then(|module| module.getattr("Message"))
        .and_then(|message| value.is_instance(&message))
        .unwrap_or(false)
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

    /// Append the values of `values` (of `PySequence_Fast`), each converted in turn, a step of
    /// `budget` before each, when it is given. The values that convert of no Python code (of an
    /// exact `int`, `float`, `bool`, `str` or `bytes`) are appended at once, before a value whose
    /// conversion may run Python code and at the end, so that such code finds those before it in
    /// the container, as upb appends each.
    fn append_values(
        slf: &Bound<'_, Self>,
        field: &ScalarField,
        values: &Bound<'_, PyAny>,
        budget: Option<&ConversionBudget>,
    ) -> PyResult<()> {
        let values = sequence_fast(values)?;
        let mut converted = Vec::with_capacity(values.len().unwrap_or(0));
        let appended = Self::convert_values(slf, field, &values, budget, &mut converted);
        Self::take_converted(slf, &mut converted);
        appended
    }

    /// Convert the values of `values`, a list or a tuple, onto `converted` (see
    /// [`RepeatedScalar::append_values`]), of the iterator of the list or the tuple, which reads the
    /// length of a list at each value, as the loop of `PySequence_Fast` does.
    fn convert_values(
        slf: &Bound<'_, Self>,
        field: &ScalarField,
        values: &Bound<'_, PyAny>,
        budget: Option<&ConversionBudget>,
        converted: &mut Vec<Scalar>,
    ) -> PyResult<()> {
        let py = slf.py();
        for value in values.try_iter()? {
            let value = value?;
            if let Some(budget) = budget {
                budget.step(py)?;
            }
            if !converts_natively(&value) {
                Self::take_converted(slf, converted);
            }
            converted.push(Scalar::from_py(
                field.kind,
                &field.name,
                &field.enum_type,
                &value,
            )?);
        }
        Ok(())
    }

    /// Append `converted`, values converted for the container, taking them.
    #[cold]
    #[inline(never)]
    fn take_converted(slf: &Bound<'_, Self>, converted: &mut Vec<Scalar>) {
        if !converted.is_empty() {
            slf.borrow_mut().items.append(converted);
        }
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

impl RepeatedScalar {
    /// The name of upb's class of the container, which its errors of arguments name.
    const UPB_CLASS: &'static str = "RepeatedScalarContainer";

    /// As upb's: the values of `elem_seq` (of `PySequence_Fast`, so those of an iterator to its
    /// end first), each converted and appended in turn. A failure, of the iteration too, leaves
    /// the container of the length that it had: upb resizes it, so a position that Python code
    /// removed meanwhile holds the default.
    fn extend_values(slf: &Bound<'_, Self>, elem_seq: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::extend_with(slf, elem_seq, None)
    }

    /// [`RepeatedScalar::extend_values`], of a step of `budget` for each value when it is given,
    /// as the values of a repeated field of a constructor are (see `PyMessage::init_fields`).
    pub(crate) fn extend_with(
        slf: &Bound<'_, Self>,
        elem_seq: &Bound<'_, PyAny>,
        budget: Option<&ConversionBudget>,
    ) -> PyResult<()> {
        Self::reify(slf)?;
        let field = Self::field_of(slf);
        let start = slf.borrow().items.len();
        let extended = Self::append_values(slf, &field, elem_seq, budget);
        if extended.is_err() {
            slf.borrow_mut()
                .items
                .resize(start, Scalar::default_of(field.kind));
        }
        extended
    }
}

/// The methods take their arguments as upb's do, positional only, with upb's errors.
#[pymethods]
impl RepeatedScalar {
    #[pyo3(signature = (*args, **kwargs))]
    fn append(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let value = &one_argument(Self::UPB_CLASS, "append", args, kwargs)?;
        Self::reify(slf)?;
        let v = Self::field_of(slf).convert(value)?;
        slf.borrow_mut().items.push(v);
        Ok(())
    }

    /// See [`RepeatedScalar::extend_values`].
    #[pyo3(signature = (*args, **kwargs))]
    fn extend(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let elem_seq = one_argument(Self::UPB_CLASS, "extend", args, kwargs)?;
        Self::extend_values(slf, &elem_seq)
    }

    /// As upb's: the position of `key` in the container as it is, then the value converted. A
    /// conversion that shortened the container leaves the position past its end (where upb's
    /// insert writes out of its bounds), so the value goes at its end.
    #[pyo3(signature = (*args, **kwargs))]
    fn insert(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let (key, value) = insert_arguments(Self::UPB_CLASS, args, kwargs)?;
        let value = &value;
        Self::reify(slf)?;
        let at = insert_position(key, slf.borrow().items.len());
        let v = Self::field_of(slf).convert(value)?;
        let mut me = slf.borrow_mut();
        let at = at.min(me.items.len());
        me.items.insert(at, v);
        Ok(())
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn pop(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let key = pop_argument(Self::UPB_CLASS, args, kwargs)?;
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
    #[pyo3(signature = (*args, **kwargs))]
    fn remove(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let elem = &one_argument(Self::UPB_CLASS, "remove", args, kwargs)?;
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

    #[pyo3(name = "MergeFrom", signature = (*args, **kwargs))]
    fn merge_from(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let other = one_argument(Self::UPB_CLASS, "MergeFrom", args, kwargs)?;
        Self::extend_values(slf, &other)
    }

    /// A copy of the container, held by no message, as upb's `copy.deepcopy` gives one: of any
    /// positional arguments (the memo) and of no keyword argument, as upb's `METH_VARARGS`.
    #[pyo3(signature = (*args, **kwargs))]
    fn __deepcopy__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let _ = args;
        no_keywords_of_positional(slf.py(), Self::UPB_CLASS, "__deepcopy__", kwargs)?;
        Ok(slf.borrow().copy())
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
    /// The messages of the container that have their objects; those of `pending` follow them.
    pub(crate) items: Vec<Py<PyMessage>>,
    owner: Owner,
    /// The messages that follow `items`, of a native value of a repeated message field, whose
    /// objects the container makes as they are read (see [`RepeatedComposite::of_native`]).
    pending: Option<Pending>,
}

/// The messages of a container after those of its objects, as the native value of a repeated
/// message field holds them: of the value, which other messages may hold too, from the message
/// `next` on, or the messages themselves, of a value that the container alone held.
enum Pending {
    Shared(Arc<NativeMessages>, usize),
    Owned(std::vec::IntoIter<NRef>),
}

/// The messages that a read of a message of a container that holds messages natively makes the
/// objects of, at least: those that follow it too, as a read of one most often reads the next.
const CHUNK: usize = 64;

/// Release `message`, which a container or a map no longer holds: of a message that another holds
/// still, its native values moved out of the slabs of their parse first (see `PyMessage::unpin`),
/// so that a message kept of a large parse, whose other messages are released, keeps no slab of
/// them alive (of up to 64 messages each) for the part of the parse that it holds unread.
pub(crate) fn release_held(py: Python<'_>, message: Py<PyMessage>) {
    let message = message.into_bound(py);
    // SAFETY: of a live object, which `message` holds.
    if unsafe { pyo3::ffi::Py_REFCNT(message.as_ptr()) } > 1 {
        PyMessage::unpin(&message);
    }
    // Released as a `Bound`, which knows that the thread is attached, with no read of a thread
    // local.
    drop(message);
}

/// [`release_held`] of the messages of the values of the entries `released`, of a map of messages,
/// which are released with the entries.
fn release_held_entries(py: Python<'_>, released: Entries) {
    for (_, value) in released.into_entries() {
        if let MapValue::Message(message) = value {
            release_held(py, message);
        }
    }
}

/// A container released releases the messages that it held as [`release_held`] does, at once.
impl Drop for RepeatedComposite {
    fn drop(&mut self) {
        if self.items.is_empty() {
            return;
        }
        // SAFETY: a container is released with the thread attached: of the deallocation of its
        // object, or of a value that was never one, of the code that made it.
        let py = unsafe { Python::assume_attached() };
        for message in std::mem::take(&mut self.items) {
            release_held(py, message);
        }
    }
}

impl RepeatedComposite {
    pub(crate) fn new(class: MessageClass, items: Vec<Py<PyMessage>>, owner: Owner) -> Self {
        RepeatedComposite {
            class,
            items,
            owner,
            pending: None,
        }
    }

    /// A container of the messages of `value`, the native value of a repeated message field, held
    /// by no message, as the copies of the runtime were (see `native::made_of`): the container
    /// makes the objects of the messages as they are read, a chunk at a time
    /// ([`RepeatedComposite::make`]), so that a read of each message of a value of many takes its
    /// object as it is made, rather than of a pass of their own, and a read of the first ones makes
    /// theirs only. Every other operation of the container makes them all first.
    pub(crate) fn of_native(class: MessageClass, value: Arc<NativeMessages>) -> Self {
        RepeatedComposite {
            class,
            items: Vec::new(),
            owner: None,
            pending: Some(Pending::Shared(value, 0)),
        }
    }

    /// The messages of the container, of their objects and those that it holds natively.
    pub(crate) fn len(&self) -> usize {
        self.items.len() + self.pending_natives().len()
    }

    /// The messages that the container holds natively, which follow those of `items`.
    pub(crate) fn pending_natives(&self) -> &[NRef] {
        match &self.pending {
            None => &[],
            Some(Pending::Shared(value, next)) => &value.items[*next..],
            Some(Pending::Owned(items)) => items.as_slice(),
        }
    }

    /// Take the messages that the container holds natively as its own, once the field no longer
    /// holds their value (see `native::Made::fill`), if no other message holds the value: their
    /// objects then take their fields, moved.
    pub(crate) fn own_pending(&mut self) {
        let Some(Pending::Shared(value, next)) = self.pending.take() else {
            return;
        };
        self.pending = Some(match Arc::try_unwrap(value) {
            Ok(value) => {
                let mut items = value.items.into_iter();
                for _ in 0..next {
                    items.next();
                }
                Pending::Owned(items)
            }
            Err(value) => Pending::Shared(value, next),
        });
    }

    /// The native messages that the container holds, to discard their unknown fields (see
    /// `PyMessage::discard_unknown`): of its own, a value that other messages hold too cloned of
    /// one level first.
    pub(crate) fn pending_natives_mut(&mut self, py: Python<'_>) -> &mut [NRef] {
        match &mut self.pending {
            None => &mut [],
            Some(Pending::Shared(value, next)) => {
                let next = *next;
                let value = native::unique_messages(py, value);
                &mut value.items[next..]
            }
            Some(Pending::Owned(items)) => items.as_mut_slice(),
        }
    }

    /// Make the objects of the messages that the container holds natively, up to the message
    /// `upto` (all of them, of `usize::MAX`), a chunk of [`CHUNK`] at least at a time: the objects
    /// are made with no borrow of the container held, as making them may run Python code (a
    /// collection of the garbage collector, a checkpoint of its budget, which another thread may
    /// take the GIL at), and set only if the container is as it was then, else read again; then
    /// they take the fields of their messages (moved, of messages that the container alone holds,
    /// else copies of one level), running no Python code. A container that Rust code borrows
    /// meanwhile is Python's `RuntimeError`, never a panic.
    pub(crate) fn make(slf: &Bound<'_, Self>, upto: usize) -> PyResult<()> {
        let py = slf.py();
        loop {
            let (class, made, count, first) = {
                let c = slf.try_borrow()?;
                let pending = c.pending_natives();
                if pending.is_empty() || c.items.len() >= upto {
                    return Ok(());
                }
                let count = (upto - c.items.len()).max(CHUNK).min(pending.len());
                (
                    c.class.clone_ref(py),
                    c.items.len(),
                    count,
                    pending.as_ptr(),
                )
            };
            let budget = ConversionBudget::full();
            let mut objects = Vec::with_capacity(count);
            for _ in 0..count {
                budget.step(py)?;
                objects.push(class.new_message(py)?);
            }
            let mut c = slf.try_borrow_mut()?;
            if c.items.len() != made
                || c.pending_natives().len() < count
                || c.pending_natives().as_ptr() != first
            {
                // Made or changed meanwhile: these objects are dropped once the container is no
                // longer borrowed, and the container is read again.
                drop(c);
                continue;
            }
            let mut releases = Releases::default();
            for object in &objects {
                let (slots, unknown) = c.take_next(py, &mut releases);
                object.borrow_mut().take_parts(slots, unknown);
            }
            drop(releases);
            c.items.extend(objects.into_iter().map(Bound::unbind));
            if c.pending_natives().is_empty() {
                // The value is released once the container is no longer borrowed.
                let pending = c.pending.take();
                drop(c);
                drop(pending);
            }
        }
    }

    /// Make the objects of all the messages that the container holds natively (see
    /// [`RepeatedComposite::make`]).
    pub(crate) fn make_all(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::make(slf, usize::MAX)
    }

    /// The set fields and the unknown fields of the next message that the container holds
    /// natively, which it no longer holds: moved, of a message that the container alone holds (its
    /// node released of `releases`), else copied, of one level.
    fn take_next(
        &mut self,
        py: Python<'_>,
        releases: &mut Releases,
    ) -> (super::message::Slots, Vec<u8>) {
        let copy = |m: &NRef| {
            let fields = m.fields.clone_native_at(py, m.generation);
            (fields.slots, fields.unknown)
        };
        match self.pending.as_mut().expect("a message held natively") {
            Pending::Shared(value, next) => {
                let at = *next;
                *next += 1;
                copy(&value.items[at])
            }
            Pending::Owned(items) => {
                let message = items.next().expect("a message held natively");
                match NRef::into_parts_releasing(message, releases) {
                    Ok(parts) => parts,
                    Err(message) => copy(&message),
                }
            }
        }
    }

    fn to_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.items.iter().map(|m| m.clone_ref(py)))
    }

    fn reify(slf: &Bound<'_, Self>) -> PyResult<()> {
        Self::make_all(slf)?;
        let owner = owner_of(slf.py(), &slf.borrow().owner);
        touch(slf.py(), &owner)
    }

    /// `append` of `value`, each message that it merges or copies a step of `budget`.
    fn append_with<'py>(
        slf: &Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let py = slf.py();
        Self::reify(slf)?;
        verify_message(value)?;
        let class = slf.borrow().class.clone_ref(py);
        let child = class.new_message(py)?;
        slf.borrow_mut().items.push(child.clone().unbind());
        PyMessage::merge_from_with(&child, value, budget)?;
        Ok(child)
    }

    /// The name of upb's class of the container, which its errors of arguments name.
    const UPB_CLASS: &'static str = "RepeatedCompositeContainer";

    /// As upb's: each value of the iterator of `elem_seq` appended as it comes, so a failure
    /// leaves the values before it appended. The container itself is of the messages that it has
    /// first, as a list's `extend` of itself takes them, where upb's iteration of the container,
    /// which grows as it is read, ends the process. One interrupt budget counts a step for each
    /// message that the appends merge or copy, so that the extend of many messages lets another
    /// thread take the GIL, and a signal interrupt it, as a copy does.
    fn extend_values(slf: &Bound<'_, Self>, elem_seq: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        Self::reify(slf)?;
        let budget = ConversionBudget::full();
        if elem_seq.is(slf) {
            let items: Vec<Py<PyMessage>> =
                slf.borrow().items.iter().map(|m| m.clone_ref(py)).collect();
            for item in items {
                Self::append_with(slf, item.bind(py), &budget)?;
            }
            return Ok(());
        }
        for value in elem_seq.try_iter()? {
            Self::append_with(slf, &value?, &budget)?;
        }
        Ok(())
    }
}

/// The methods take their arguments as upb's do, positional only (but `add`'s keywords), with
/// upb's errors.
#[pymethods]
impl RepeatedComposite {
    /// As upb's: a new message appended, then given the fields of `kwargs` (see
    /// `PyMessage::init_fields`), so that a value that holds the container reads it with the new
    /// message in it; a failure removes the last message, as upb's does. A positional argument is
    /// the error of a constructor's.
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
        slf.borrow_mut().items.push(child.clone().unbind());
        if let Some(kwargs) = kwargs {
            if let Err(err) =
                PyMessage::init_fields(&child, kwargs.as_any(), &ConversionBudget::full())
            {
                // The message removed is dropped once the container is no longer borrowed.
                let _removed = slf.borrow_mut().items.pop();
                return Err(err);
            }
        }
        Ok(child)
    }

    /// As upb's: a new message, appended, then merged with `value` and returned, so that a message
    /// of another type leaves the new message (empty) in the container.
    #[pyo3(signature = (*args, **kwargs))]
    fn append<'py>(
        slf: &Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyMessage>> {
        let value = one_argument(Self::UPB_CLASS, "append", args, kwargs)?;
        Self::append_with(slf, &value, &ConversionBudget::full())
    }

    /// See [`RepeatedComposite::extend_values`].
    #[pyo3(signature = (*args, **kwargs))]
    fn extend(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let elem_seq = one_argument(Self::UPB_CLASS, "extend", args, kwargs)?;
        Self::extend_values(slf, &elem_seq)
    }

    /// As upb's: the position of `key` in the container as it is, then the copy of `value`,
    /// which goes at the end of a container that is shorter by then.
    #[pyo3(signature = (*args, **kwargs))]
    fn insert(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let (key, value) = insert_arguments(Self::UPB_CLASS, args, kwargs)?;
        let value = &value;
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

    #[pyo3(signature = (*args, **kwargs))]
    fn pop(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyMessage>> {
        let key = pop_argument(Self::UPB_CLASS, args, kwargs)?;
        let py = slf.py();
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        let at = pop_position(py, key, len)?;
        let removed = remove_at(&mut slf.borrow_mut().items, at)?;
        // The message outlives the container that held it.
        PyMessage::unpin(removed.bind(py));
        Ok(removed)
    }

    /// As upb's: the first message that equals `elem`, among as many as the container had, each
    /// of the container as it is when it is compared.
    #[pyo3(signature = (*args, **kwargs))]
    fn remove(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let elem = &one_argument(Self::UPB_CLASS, "remove", args, kwargs)?;
        let py = slf.py();
        Self::reify(slf)?;
        let len = slf.borrow().items.len();
        for at in 0..len {
            let item = item_at(&slf.borrow().items, at, |m| Ok(m.clone_ref(py).into_any()))?;
            if compares_equal(item.bind(py), elem)? {
                // A message removed is dropped once the container is no longer borrowed.
                let removed = {
                    let mut me = slf.borrow_mut();
                    (at < me.items.len()).then(|| me.items.remove(at))
                };
                removed.into_iter().for_each(|m| release_held(py, m));
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
        Self::make_all(slf)?;
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
        // Released once the container is no longer borrowed.
        let (held, pending) = {
            let mut c = slf.borrow_mut();
            (std::mem::take(&mut c.items), c.pending.take())
        };
        held.into_iter().for_each(|m| release_held(slf.py(), m));
        drop(pending);
        Ok(())
    }

    #[pyo3(name = "MergeFrom", signature = (*args, **kwargs))]
    fn merge_from(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let other = one_argument(Self::UPB_CLASS, "MergeFrom", args, kwargs)?;
        Self::extend_values(slf, &other)
    }

    /// A copy of the container, of copies of its messages, held by no message. The messages are
    /// read under a short borrow and copied after it, of one interrupt budget, as a copy of a
    /// message copies them.
    #[pyo3(signature = (*args, **kwargs))]
    fn __deepcopy__(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let _ = args;
        no_keywords_of_positional(slf.py(), Self::UPB_CLASS, "__deepcopy__", kwargs)?;
        let py = slf.py();
        Self::make_all(slf)?;
        let (class, items) = {
            let me = slf.borrow();
            let items: Vec<Py<PyMessage>> = me.items.iter().map(|m| m.clone_ref(py)).collect();
            (me.class.clone_ref(py), items)
        };
        let budget = ConversionBudget::full();
        let copies = items
            .iter()
            .map(|m| duplicate_with(py, m.bind(py), &budget).map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(RepeatedComposite::new(class, copies, None))
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let item = |m: &Py<PyMessage>| Ok(m.clone_ref(py).into_any());
        // The index of an int runs no Python code, so the length and the item are read at once,
        // once the container made the objects up to it.
        if key.is_exact_instance_of::<PyInt>() {
            let at = {
                let me = slf.borrow();
                match Subscript::of(key, me.len())? {
                    Subscript::Index(at) if at < me.items.len() => return item(&me.items[at]),
                    Subscript::Index(at) => at,
                    _ => unreachable!("an int is an index"),
                }
            };
            Self::make(slf, at + 1)?;
            let me = slf.borrow();
            return Subscript::of(key, me.len())?.get(py, &me.items, item);
        }
        Self::make_all(slf)?;
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
        let removed = subscript.remove_from(&mut slf.borrow_mut().items);
        removed.into_iter().for_each(|m| release_held(slf.py(), m));
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
        Self::make_all(slf)?;
        let mine = slf.borrow().to_list(py)?;
        let other = match other.cast::<RepeatedComposite>() {
            Ok(other) => {
                Self::make_all(other)?;
                other.borrow().to_list(py)?.into_any()
            }
            Err(_) => other.clone(),
        };
        Ok(mine.as_any().rich_compare(&other, op)?.unbind())
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Self::make_all(slf)?;
        Ok(slf.borrow().to_list(slf.py())?.repr()?.to_string())
    }
}

/// The value of an entry of a map.
pub(crate) enum MapValue {
    Scalar(Scalar),
    Message(Py<PyMessage>),
}

/// The hash of `key`, a key of a map, of one random state of the process.
fn key_hash(key: &Scalar) -> u64 {
    static STATE: OnceLock<RandomState> = OnceLock::new();
    let mut hasher = STATE.get_or_init(RandomState::new).build_hasher();
    match key {
        Scalar::Str(v) => v.hash(&mut hasher),
        Scalar::I64(v) => v.hash(&mut hasher),
        Scalar::U64(v) => v.hash(&mut hasher),
        Scalar::I32(v) | Scalar::Enum(v) => v.hash(&mut hasher),
        Scalar::U32(v) => v.hash(&mut hasher),
        Scalar::Bool(v) => v.hash(&mut hasher),
        Scalar::Bytes(v) => v.hash(&mut hasher),
        // A float is the key of no map (protobuf's keys are integers, bools and strings): it
        // hashes of its bits, of 0.0 for -0.0 too, which it equals.
        Scalar::F64(v) => (if *v == 0.0 { 0 } else { v.to_bits() }).hash(&mut hasher),
        Scalar::F32(v) => (if *v == 0.0 { 0 } else { v.to_bits() }).hash(&mut hasher),
    }
    hasher.finish()
}

/// The entries of a map, in the order of their insertion, with the positions of their keys in a
/// hash table, so that a key is found at once, as upb's hash table finds it, rather than by a scan
/// of the entries, which made the operations of a map of many keys (the parse, the constructor,
/// `update`, `MergeFrom`, `==`) take as long as the square of their number. A removed entry
/// leaves a hole, which the iteration passes over, so that a removal moves no entry either, until
/// the holes are more than the entries, when they are dropped at once.
pub(crate) struct Entries<V = MapValue> {
    slots: Vec<Option<(Scalar, V)>>,
    /// The hash of the key of each entry and its position in `slots`, by the hash: the table
    /// grows of the hashes that it holds, without reading the keys again.
    positions: HashTable<(u64, usize)>,
    /// The position of the first entry: the slots before it are holes.
    first: usize,
}

impl<V> Default for Entries<V> {
    fn default() -> Self {
        Entries {
            slots: Vec::new(),
            positions: HashTable::new(),
            first: 0,
        }
    }
}

impl<V> Entries<V> {
    /// The entries `entries`, of distinct keys (those of a map), in their order.
    pub(crate) fn of_distinct(entries: Vec<(Scalar, V)>) -> Entries<V> {
        let mut positions = HashTable::with_capacity(entries.len());
        for (at, (key, _)) in entries.iter().enumerate() {
            let hash = key_hash(key);
            positions.insert_unique(hash, (hash, at), |&(hash, _)| hash);
        }
        Entries {
            slots: entries.into_iter().map(Some).collect(),
            positions,
            first: 0,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.positions.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Room for `additional` more entries.
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
        self.positions.reserve(additional, |&(hash, _)| hash);
    }

    /// The entries, in the order of their insertion.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &(Scalar, V)> {
        self.slots[self.first..].iter().flatten()
    }

    /// The entries, in the order of their insertion, taken.
    pub(crate) fn into_entries(self) -> impl Iterator<Item = (Scalar, V)> {
        self.slots.into_iter().flatten()
    }

    /// The values of the entries, in the order of their insertion.
    pub(crate) fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.slots.iter_mut().flatten().map(|(_, value)| value)
    }

    /// The entries of the values of `value` of each, in the same order, of the same positions.
    pub(crate) fn map<W>(&self, mut value: impl FnMut(&V) -> W) -> Entries<W> {
        let entries = self.try_map(|v| Ok::<W, std::convert::Infallible>(value(v)));
        match entries {
            Ok(entries) => entries,
            Err(never) => match never {},
        }
    }

    /// The entries of the values of `value` of each, in the same order, of the same positions,
    /// or the first error of `value`.
    pub(crate) fn try_map<W, E>(
        &self,
        mut value: impl FnMut(&V) -> Result<W, E>,
    ) -> Result<Entries<W>, E> {
        let mut slots = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            slots.push(match slot {
                Some((key, v)) => Some((key.clone(), value(v)?)),
                None => None,
            });
        }
        Ok(Entries {
            slots,
            positions: self.positions.clone(),
            first: self.first,
        })
    }

    /// The key of the first entry at the position `at` or after it, with its position: an
    /// iteration of the entries as they are at each step, whose positions stay while the map keeps
    /// its keys (a removal, which may move them, takes a new version of the map).
    pub(crate) fn key_at_or_after(&self, at: usize) -> Option<(usize, &Scalar)> {
        let start = at.max(self.first);
        self.slots
            .get(start..)?
            .iter()
            .enumerate()
            .find_map(|(i, slot)| slot.as_ref().map(|(key, _)| (start + i, key)))
    }

    /// Whether the entry at `at`, a position of `positions`, is of `key`.
    fn is_at(slots: &[Option<(Scalar, V)>], at: usize, key: &Scalar) -> bool {
        matches!(&slots[at], Some((held, _)) if held == key)
    }

    /// The value of `key`, if the map has one.
    pub(crate) fn get(&self, key: &Scalar) -> Option<&V> {
        let hash = key_hash(key);
        let slots = &self.slots;
        let &(_, at) = self
            .positions
            .find(hash, |&(h, at)| h == hash && Self::is_at(slots, at, key))?;
        slots[at].as_ref().map(|(_, value)| value)
    }

    /// Set the value of `key`, after the other entries if the map has none of it, and give back
    /// the value that it replaces.
    pub(crate) fn insert(&mut self, key: Scalar, value: V) -> Option<V> {
        let hash = key_hash(&key);
        let Entries {
            slots, positions, ..
        } = self;
        let found = positions.entry(
            hash,
            |&(h, at)| h == hash && Self::is_at(slots, at, &key),
            |&(hash, _)| hash,
        );
        match found {
            hashbrown::hash_table::Entry::Occupied(entry) => {
                let (_, at) = *entry.get();
                let (_, held) = slots[at].as_mut().expect("a position of an entry");
                Some(std::mem::replace(held, value))
            }
            hashbrown::hash_table::Entry::Vacant(entry) => {
                entry.insert((hash, slots.len()));
                slots.push(Some((key, value)));
                None
            }
        }
    }

    /// Remove the entry of `key`, and give it back (to drop once the map is no longer borrowed).
    pub(crate) fn remove(&mut self, key: &Scalar) -> Option<(Scalar, V)> {
        let hash = key_hash(key);
        let Entries {
            slots,
            positions,
            first,
        } = self;
        let found = positions
            .find_entry(hash, |&(h, at)| h == hash && Self::is_at(slots, at, key))
            .ok()?;
        let ((_, at), _) = found.remove();
        let removed = slots[at].take();
        while slots.get(*first).is_some_and(Option::is_none) {
            *first += 1;
        }
        if slots.len() > 2 * positions.len() {
            self.drop_holes();
        }
        removed
    }

    /// Drop the holes of the removed entries, moving the entries after them.
    fn drop_holes(&mut self) {
        let mut moved_to = Vec::with_capacity(self.slots.len());
        let mut next = 0;
        for slot in &self.slots {
            moved_to.push(next);
            next += usize::from(slot.is_some());
        }
        for (_, at) in self.positions.iter_mut() {
            *at = moved_to[*at];
        }
        self.slots.retain(Option::is_some);
        self.first = 0;
    }
}

/// The key of `key` for a map of `map`, with the errors of its entry's `key` field.
fn key_of(map: &MapDesc, key: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    Scalar::from_py(map.key, &map.key_name, "", key)
}

/// The value of `value` for a map of `map` of scalars, with the errors of its entry's `value`.
fn value_of(map: &MapDesc, value: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    Scalar::from_py(map.value, &map.value_name, &map.value_type, value)
}

/// The value of an entry of a map read for a copy or a merge: a scalar, a message of a map of
/// Python (an object), or a message of a native map (see [`NativeMap`]).
pub(crate) enum EntryValue {
    Scalar(Scalar),
    Object(Py<PyMessage>),
    Native(NRef),
}

/// The entries of a map read for a copy or a merge, of the messages of the map, which the copy
/// copies with no borrow of the map held.
pub(crate) struct MapEntries {
    map: Arc<MapDesc>,
    entries: Vec<(Scalar, EntryValue)>,
}

impl MapEntries {
    /// The entries of the native map `map`, read for a merge.
    pub(crate) fn of_native(map: &NativeMap) -> MapEntries {
        MapEntries {
            map: map.map.clone(),
            entries: map
                .entries
                .iter()
                .map(|(key, value)| {
                    let value = match value {
                        NativeValue::Scalar(v) => EntryValue::Scalar(v.clone()),
                        NativeValue::Message(m) => EntryValue::Native(m.clone()),
                    };
                    (key.clone(), value)
                })
                .collect(),
        }
    }

    /// Copies of the entries, of objects of their messages, a step of `budget` for each entry and
    /// for each message copied.
    fn copy_entries(
        self,
        py: Python<'_>,
        budget: &ConversionBudget,
    ) -> PyResult<Vec<(Scalar, MapValue)>> {
        let mut copies = Vec::with_capacity(self.entries.len());
        for (key, value) in self.entries {
            budget.step(py)?;
            let value = match value {
                EntryValue::Scalar(v) => MapValue::Scalar(v),
                EntryValue::Object(m) => {
                    MapValue::Message(duplicate_with(py, m.bind(py), budget)?.unbind())
                }
                EntryValue::Native(m) => {
                    MapValue::Message(native::duplicate_native(py, &m, None, budget)?.unbind())
                }
            };
            copies.push((key, value));
        }
        Ok(copies)
    }

    /// A native copy of the entries, of the levels `levels` that may still be below the message
    /// of the map (see [`native::copy_message`]): an entry is a level below it, and the message of
    /// its value another. A step of `budget` for each entry and for each message copied.
    pub(crate) fn native_copy(
        self,
        py: Python<'_>,
        at: native::At,
        generation: u64,
        budget: &ConversionBudget,
    ) -> Result<NativeMap, CopyError> {
        let entry = if self.entries.is_empty() {
            at
        } else {
            at.below()?
        };
        let entry_levels = entry.levels;
        let mut copies = Vec::with_capacity(self.entries.len());
        for (key, value) in self.entries {
            budget.step(py)?;
            // The level of the entry, of the stack of every 16th level, as of a message's.
            let value = entry.grown(|| -> Result<NativeValue, CopyError> {
                Ok(match value {
                    EntryValue::Scalar(v) => NativeValue::Scalar(v),
                    EntryValue::Object(m) => NativeValue::Message(NRef::new(native::copy_message(
                        py,
                        m.bind(py),
                        entry.below()?,
                        generation,
                        budget,
                    )?)),
                    EntryValue::Native(m) => {
                        if entry_levels.is_some_and(|levels| m.depth >= levels) {
                            return Err(CopyError::TooDeep);
                        }
                        NativeValue::Message(native::Native::at(&m, py, generation))
                    }
                })
            })?;
            copies.push((key, value));
        }
        Ok(NativeMap::new(
            self.map,
            Entries::of_distinct(copies),
            generation,
        ))
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
    pub(crate) entries: Entries,
    owner: Owner,
    /// The version of the map, as upb's: a new one at the start of each method of the map that
    /// may change its keys, even one that then fails or changes nothing, but a set of a key that
    /// the map has (see [`MapContainer::new_version`]). An iteration of the map (a [`MapIterator`])
    /// fails at its next step once the version is not the one that it started of.
    version: u64,
}

/// A map released releases the messages of its values as [`release_held`] does, at once.
impl Drop for MapContainer {
    fn drop(&mut self) {
        if self.map.value != Kind::Message || self.entries.is_empty() {
            return;
        }
        // SAFETY: as of a repeated message container (see its `Drop`).
        let py = unsafe { Python::assume_attached() };
        release_held_entries(py, std::mem::take(&mut self.entries));
    }
}

impl MapContainer {
    pub(crate) fn new(field_name: String, map: Arc<MapDesc>, owner: Owner) -> Self {
        MapContainer {
            field_name,
            map,
            entries: Entries::default(),
            owner,
            version: 0,
        }
    }

    /// A map of the entries `entries`, held by no message, as a copy of a map is (see
    /// [`native::made_of`]).
    pub(crate) fn of_entries(field_name: String, map: Arc<MapDesc>, entries: Entries) -> Self {
        MapContainer {
            field_name,
            map,
            entries,
            owner: None,
            version: 0,
        }
    }

    /// A new version of the map, as upb's map takes one at the start of each method that may
    /// change its keys (`EnsureReified`): an iteration of the map in progress then fails at its
    /// next step. The version before, which [`MapContainer::keep_version`] restores.
    pub(crate) fn new_version(&mut self) -> u64 {
        let before = self.version;
        self.version = before.wrapping_add(1);
        before
    }

    /// The version `before` again, after a set that replaced the value of a key that the map
    /// has, of which upb's map takes back the new version: when nothing took another since.
    fn keep_version(&mut self, before: u64) {
        if self.version == before.wrapping_add(1) {
            self.version = before;
        }
    }

    /// The name of upb's class of the map, which its errors of the arguments of its methods name:
    /// `ScalarMapContainer` (the runtime's `MapContainer`) or `MessageMapContainer`.
    fn upb_class_name(slf: &Bound<'_, Self>) -> &'static str {
        if slf.borrow().map.value == Kind::Message {
            "MessageMapContainer"
        } else {
            "ScalarMapContainer"
        }
    }

    /// The full name of the map field, which upb's errors name: of the message that holds the
    /// entry type (`spark.connect.SQL.named_arguments` of `spark.connect.SQL.NamedArgumentsEntry`).
    fn full_field_name(&self) -> String {
        match self.map.entry.rsplit_once('.') {
            Some((message, _)) => format!("{message}.{}", self.field_name),
            None => self.field_name.clone(),
        }
    }

    /// The map as an object of its class, as upb's: a `ScalarMapContainer` of a map of scalars,
    /// a `MessageMapContainer` of a map of messages (which has `get_or_create`, as upb's), neither
    /// a subclass of the other.
    pub(crate) fn into_object(self, py: Python<'_>) -> PyResult<Py<MapContainer>> {
        if self.map.value != Kind::Message {
            let init = PyClassInitializer::from(self).add_subclass(ScalarMapContainer);
            return Ok(Bound::new(py, init)?.into_super().unbind());
        }
        let init = PyClassInitializer::from(self).add_subclass(MessageMapContainer);
        Ok(Bound::new(py, init)?.into_super().unbind())
    }

    fn map_of(slf: &Bound<'_, Self>) -> Arc<MapDesc> {
        slf.borrow().map.clone()
    }

    fn reify(slf: &Bound<'_, Self>) -> PyResult<()> {
        let owner = owner_of(slf.py(), &slf.borrow().owner);
        touch(slf.py(), &owner)
    }

    /// Set the value of `key`, replacing the one it has, which it gives back (to drop once the
    /// map is no longer borrowed).
    pub(crate) fn insert(&mut self, key: Scalar, value: MapValue) -> Option<MapValue> {
        self.entries.insert(key, value)
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
        me.entries
            .get(key)
            .map(|value| Self::value_to_py(slf.py(), value))
            .transpose()
    }

    /// The entries of the map, read for a copy or a merge: their keys and scalars copied, and their
    /// messages, which [`MapEntries::native_copy`] (or the merge) copies once the map is no longer
    /// borrowed.
    pub(crate) fn entries_for_copy(&self, py: Python<'_>) -> MapEntries {
        MapEntries {
            map: self.map.clone(),
            entries: self
                .entries
                .iter()
                .map(|(k, v)| {
                    let value = match v {
                        MapValue::Scalar(v) => EntryValue::Scalar(v.clone()),
                        MapValue::Message(m) => EntryValue::Object(m.clone_ref(py)),
                    };
                    (k.clone(), value)
                })
                .collect(),
        }
    }

    /// Set `entries` in `target`, of copies of their messages, a step of `budget` for each entry
    /// and for each message copied.
    pub(crate) fn merge_entries(
        py: Python<'_>,
        target: &Bound<'_, Self>,
        entries: MapEntries,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        let copies = entries.copy_entries(py, budget)?;
        target.borrow_mut().entries.reserve(copies.len());
        for (key, value) in copies {
            budget.step(py)?;
            // The value that the entry replaces is dropped once the map is no longer borrowed.
            let _replaced = target.borrow_mut().insert(key, value);
        }
        Ok(())
    }

    /// Whether the maps `a` and `b` are equal as upb's `==` of two maps tells, which is
    /// `Mapping.__eq__`: as dicts of their keys and values, of Python's `==`. So a map equals a map
    /// of other key or value types of equal keys and values (`{1: "t"}` of int32 keys and of
    /// int64 or bool keys), and a map of a NaN value equals no map, itself neither, where `==` of
    /// two messages compares their maps' values of their bits (see [`MapContainer::equal`]). Two
    /// maps of the same key and value types of no floats are compared in place.
    fn equal_as_dicts(py: Python<'_>, a: &Bound<'_, Self>, b: &Bound<'_, Self>) -> PyResult<bool> {
        let (x, y) = (Self::map_of(a), Self::map_of(b));
        let in_place = x.key == y.key
            && x.value == y.value
            && x.value_type == y.value_type
            && !matches!(x.value, Kind::Double | Kind::Float);
        if in_place {
            return Self::equal(py, a, b);
        }
        let mine = a.borrow().to_dict(py)?;
        let theirs = b.borrow().to_dict(py)?;
        mine.as_any().eq(theirs)
    }

    /// Whether `a` and `b` have the same keys, of equal values, as upb's `==` of two messages
    /// compares their maps: a float or a double of its bits.
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
        for (key, value) in x.entries.iter() {
            let Some(other) = y.entries.get(key) else {
                return Ok(false);
            };
            let equal = match (value, other) {
                (MapValue::Scalar(v), MapValue::Scalar(w)) => v.eq_as_upb(w),
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

    /// Set `value` in `target`, as upb's `InitMapAttributes` takes the value of the map field
    /// `field` (its full name) of the constructor of a message:
    ///
    /// - A map of scalars takes it as upb's `update` does, which is `MutableMapping.update`, run in
    ///   Python: the items of a mapping or of an object with `keys`, else the pairs of an
    ///   iterable (`[("a", "b")]`, `["ab"]`, `[]`), with Python's errors of a pair of another
    ///   shape (`ValueError: not enough values to unpack (expected 2, got 1)`), as its version
    ///   words them. Its loop lets other threads run, and signals interrupt it, as upb's does. A
    ///   tuple is the arguments of the call, as upb's `PyObject_CallMethod(map, "update", "O",
    ///   value)` makes it: `({"a": "b"},)` updates of the dict, and `(("a", "b"),)` of the pairs
    ///   of `("a", "b")`, which are not pairs.
    /// - A map of messages takes the keys that `value` iterates, each of `value[key]`, so a
    ///   mapping, but not pairs (`value[key]` of a list is a `TypeError`): a dict sets the fields
    ///   of the message of the key (made if the map has none), and anything else is copied into
    ///   that message as its `CopyFrom` takes it, of its errors. The error of a key that the map
    ///   refuses is raised over that of reading its value, as upb reads the value first.
    ///
    /// The items of a dict (not of a subclass, whose methods `MutableMapping.update` and upb would
    /// run) are read here, as those loops read them, in their order. Each item of a dict, and each
    /// key of a map of messages, is a step of `budget`, the budget of the constructor, as is each
    /// message that it copies or builds.
    pub(crate) fn update_from(
        py: Python<'_>,
        target: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        field: &str,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        static MUTABLE_MAPPING_UPDATE: pyo3::sync::PyOnceLock<Py<PyAny>> =
            pyo3::sync::PyOnceLock::new();
        let map = Self::map_of(target);
        if value.is_exact_instance_of::<PyDict>() {
            // The view's iteration raises Python's error of a dict that changes size meanwhile.
            for item in value.call_method0("items")?.try_iter()? {
                let (key, item): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
                budget.step(py)?;
                let key = key_of(&map, &key)?;
                if map.value == Kind::Message {
                    Self::set_message_of_key(py, target, key, item, budget)?;
                } else {
                    // As `map[key] = item` sets it, but for setting its message in a parent, which
                    // the message that the constructor makes has none of.
                    let item = MapValue::Scalar(value_of(&map, &item)?);
                    let _replaced = target.borrow_mut().insert(key, item);
                }
            }
            return Ok(());
        }
        if map.value != Kind::Message {
            let update = MUTABLE_MAPPING_UPDATE.get_or_try_init(py, || -> PyResult<_> {
                let mutable_mapping = py.import("collections.abc")?.getattr("MutableMapping")?;
                Ok(mutable_mapping.getattr("update")?.unbind())
            })?;
            let mut args = vec![target.as_any().clone()];
            match value.cast::<PyTuple>() {
                Ok(tuple) => args.extend(tuple.iter()),
                Err(_) => args.push(value.clone()),
            }
            update.bind(py).call1(PyTuple::new(py, args)?)?;
            return Ok(());
        }
        let keys = value.try_iter().map_err(|_| {
            PyTypeError::new_err(format!("Argument for field {field} is not iterable"))
        })?;
        for key in keys {
            let key = key?;
            budget.step(py)?;
            let item = value.get_item(&key);
            let key = key_of(&map, &key)?;
            Self::set_message_of_key(py, target, key, item?, budget)?;
        }
        Ok(())
    }

    /// Set `item` as the message of `key` in `target`, a map of messages, as upb's
    /// `InitMapAttributes` sets the item of a key: a dict sets the fields of the message of the key
    /// (made if the map has none), a message of the type of the map's values is copied into the
    /// message that the map has for the key, or inserted as a copy when it has none, and anything
    /// else goes to the `CopyFrom` of the message of the key, which raises its error of a value
    /// that is not a message of the type.
    fn set_message_of_key(
        py: Python<'_>,
        target: &Bound<'_, Self>,
        key: Scalar,
        item: Bound<'_, PyAny>,
        budget: &ConversionBudget,
    ) -> PyResult<()> {
        if let Ok(fields) = item.cast::<PyDict>() {
            let message = Self::message_of_key(py, target, key)?;
            PyMessage::init_fields(&message, fields.as_any(), budget)?;
            return Ok(());
        }
        if let Ok(message) = item.cast::<PyMessage>() {
            if message.borrow().desc.full_name == Self::map_of(target).value_type {
                // A message that the map has for the key takes the copy in place, as upb's
                // `CopyFrom` of it does, so that it stays the message of the key: after a
                // `MergeFrom` of the map, `x = map[key]` from before has the new fields. (The map
                // of a constructor has none.)
                let held = match target.borrow().entries.get(&key) {
                    Some(MapValue::Message(held)) => Some(held.clone_ref(py)),
                    _ => None,
                };
                if let Some(held) = held {
                    py.get_type::<PyMessage>()
                        .call_method1("CopyFrom", (held, item))?;
                    return Ok(());
                }
                let copy = duplicate_with(py, message, budget)?;
                // The value that the copy replaces is dropped once the map is no longer borrowed.
                let _replaced = target
                    .borrow_mut()
                    .insert(key, MapValue::Message(copy.unbind()));
                return Ok(());
            }
        }
        let message = Self::message_of_key(py, target, key)?;
        py.get_type::<PyMessage>()
            .call_method1("CopyFrom", (message, item))?;
        Ok(())
    }

    /// The message of `key` in `target`, a map of messages, inserted as a new message when the
    /// map has none, as upb's `map[key]` gives it.
    fn message_of_key<'py>(
        py: Python<'py>,
        target: &Bound<'py, Self>,
        key: Scalar,
    ) -> PyResult<Bound<'py, PyMessage>> {
        if let Some(MapValue::Message(message)) = target.borrow().entries.get(&key) {
            return Ok(message.bind(py).clone());
        }
        let message = new_instance(py, &Self::map_of(target).value_type)?;
        let _replaced = target
            .borrow_mut()
            .insert(key, MapValue::Message(message.clone().unbind()));
        Ok(message)
    }

    /// `update` of the positional argument `other`, if it has one, and of `kwargs`, as
    /// `MutableMapping.update` sets them. Each item is a step of an interrupt budget, as upb's
    /// `update` (`MutableMapping`'s, in Python) lets another thread run and a signal interrupt it.
    fn update_with(
        slf: &Bound<'_, Self>,
        other: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let budget = ConversionBudget::full();
        if let Some(other) = other {
            let mapping = py.import("collections.abc")?.getattr("Mapping")?;
            if other.is_instance(&mapping)? {
                for key in other.try_iter()? {
                    let key = key?;
                    budget.step(py)?;
                    Self::__setitem__(slf, &key, &other.get_item(&key)?)?;
                }
            } else if other.hasattr("keys")? {
                for key in other.call_method0("keys")?.try_iter()? {
                    let key = key?;
                    budget.step(py)?;
                    Self::__setitem__(slf, &key, &other.get_item(&key)?)?;
                }
            } else {
                for pair in other.try_iter()? {
                    let pair = pair?;
                    budget.step(py)?;
                    let (key, value) = unpack_pair(&pair)?;
                    Self::__setitem__(slf, &key, &value)?;
                }
            }
        }
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.iter() {
                budget.step(py)?;
                Self::__setitem__(slf, &key, &value)?;
            }
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
        slf.borrow_mut().new_version();
        let value = if map.value == Kind::Message {
            MapValue::Message(new_instance(py, &map.value_type)?.unbind())
        } else {
            MapValue::Scalar(Scalar::default_of(map.value))
        };
        let mut me = slf.borrow_mut();
        // The value of the key, if Python code that making the value ran has set one.
        if let Some(held) = me.entries.get(&key) {
            return Self::value_to_py(py, held);
        }
        let out = Self::value_to_py(py, &value)?;
        me.entries.insert(key, value);
        Ok(out)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.entries.iter() {
            dict.set_item(key.to_py(py)?, Self::value_to_py(py, value)?)?;
        }
        Ok(dict)
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
        let before = slf.borrow_mut().new_version();
        let map = Self::map_of(slf);
        let key = key_of(&map, key)?;
        if map.value == Kind::Message {
            return Err(PyValueError::new_err("Message objects may not be assigned"));
        }
        let value = MapValue::Scalar(value_of(&map, value)?);
        let replaced = slf.borrow_mut().insert(key, value);
        if replaced.is_some() {
            slf.borrow_mut().keep_version(before);
        }
        Ok(())
    }

    fn __delitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::reify(slf)?;
        slf.borrow_mut().new_version();
        let key = key_of(&Self::map_of(slf), key)?;
        // The entry removed is dropped once the map is no longer borrowed.
        let removed = slf.borrow_mut().entries.remove(&key);
        match removed {
            Some((_, value)) => {
                if let MapValue::Message(m) = value {
                    release_held(slf.py(), m);
                }
                Ok(())
            }
            None => Err(PyKeyError::new_err("Key not present in map")),
        }
    }

    fn __contains__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let key = key_of(&Self::map_of(slf), key)?;
        Ok(slf.borrow().entries.get(&key).is_some())
    }

    fn __len__(&self) -> usize {
        self.entries.len()
    }

    /// As upb's: an iterator of the keys of the map as it is at each step, which fails once the
    /// map takes a new version, so that it reads no key that a removal took meanwhile (which
    /// `items()` and `values()` would insert again, of the default value, by reading its value).
    fn __iter__(slf: &Bound<'_, Self>) -> MapIterator {
        MapIterator {
            map: slf.clone().unbind(),
            at: 0,
            version: slf.borrow().version,
        }
    }

    /// The keys, values and items, as upb's maps give them: `Mapping`'s methods, of its views and
    /// of the errors of their arguments.
    #[pyo3(signature = (*args, **kwargs))]
    fn keys(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        abc_method(slf, "Mapping", "keys", args, kwargs)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn values(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        abc_method(slf, "Mapping", "values", args, kwargs)
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn items(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        abc_method(slf, "Mapping", "items", args, kwargs)
    }

    /// The value of `key`, or `default`, of upb's arguments and errors (see [`get_arguments`]).
    #[pyo3(signature = (*args, **kwargs))]
    fn get(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let (key, default) = get_arguments(args, kwargs)?;
        let key = key_of(&Self::map_of(slf), &key)?;
        Ok(match Self::lookup(slf, &key)? {
            Some(value) => value,
            None => default.map_or_else(|| slf.py().None(), Bound::unbind),
        })
    }

    /// As upb's, which is `MutableMapping.pop`: `map[key]` gives the default of the value type for
    /// a missing key, so `default` is never returned.
    #[pyo3(signature = (*args, **kwargs))]
    fn pop(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        abc_method(slf, "MutableMapping", "pop", args, kwargs)
    }

    /// As upb's, which is `MutableMapping.popitem`: the first key of an iteration of the map.
    #[pyo3(signature = (*args, **kwargs))]
    fn popitem(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        abc_method(slf, "MutableMapping", "popitem", args, kwargs)
    }

    /// As upb's: of no keyword arguments, refused of a map of messages, and of a key and a value
    /// that is not None, of the errors of `PyArg_UnpackTuple` of a map of scalars.
    #[pyo3(signature = (*args, **kwargs))]
    fn setdefault(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        no_keywords_of_positional(slf.py(), Self::upb_class_name(slf), "setdefault", kwargs)?;
        let map = Self::map_of(slf);
        if map.value == Kind::Message {
            return Err(PyNotImplementedError::new_err(
                "Set message map value directly is not supported.",
            ));
        }
        match args.len() {
            0 => {
                return Err(PyTypeError::new_err(
                    "setdefault expected at least 1 argument, got 0",
                ))
            }
            1 | 2 => {}
            given => {
                return Err(PyTypeError::new_err(format!(
                    "setdefault expected at most 2 arguments, got {given}"
                )))
            }
        }
        let key = args.get_item(0)?;
        let key = &key;
        let default = args.get_item(1).ok();
        let Some(default) = default.as_ref().filter(|d| !d.is_none()) else {
            return Err(PyValueError::new_err(
                "The value for scalar map setdefault must be set.",
            ));
        };
        // As upb's, of a key that the map has, or of one that it cannot hold, too.
        Self::reify(slf)?;
        slf.borrow_mut().new_version();
        if let Some(value) = Self::lookup(slf, &key_of(&map, key)?)? {
            return Ok(value);
        }
        Self::__setitem__(slf, key, default)?;
        Ok(default.clone().unbind())
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn clear(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        no_arguments(Self::upb_class_name(slf), "clear", args, kwargs)?;
        Self::reify(slf)?;
        slf.borrow_mut().new_version();
        // The entries are dropped once the map is no longer borrowed.
        let entries = std::mem::take(&mut slf.borrow_mut().entries);
        release_held_entries(slf.py(), entries);
        Ok(())
    }

    /// As `MutableMapping.update`, which upb's maps have: of at most one positional argument, the
    /// items of a mapping, of an object with `keys`, or of an iterable of pairs (so `None`, which
    /// is not iterable, is the error of its iteration), then the keyword arguments.
    #[pyo3(signature = (*args, **kwargs))]
    fn update(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if args.len() > 1 {
            return Err(PyTypeError::new_err(format!(
                "MutableMapping.update() takes from 1 to 2 positional arguments but {} were given",
                args.len() + 1
            )));
        }
        Self::update_with(slf, args.iter().next().as_ref(), kwargs)
    }

    /// As upb's, which refuses a dict and takes any other argument as its `InitMapAttributes`
    /// takes the value of a map field of a constructor ([`MapContainer::update_from`]): a map of
    /// scalars through `MutableMapping.update` (of the arguments of a tuple, as upb calls it), so
    /// of the items of a mapping, such as `protobuf`'s map of a message of `protobuf`'s class of the
    /// type (`scan.table.options.MergeFrom(base_pb2.Table(...).options)`), or of the pairs of an
    /// iterable; a map of messages of the keys that the argument iterates, each of its value
    /// copied into the message of the key. The entries of a map of other types are converted, with
    /// upb's errors. upb's map takes a new version first, so an iteration of the map in progress
    /// fails, even of an argument that it refuses. Each entry is a step of an interrupt budget.
    /// The argument is upb's one positional argument, of the errors of `METH_O`, which come before
    /// the new version.
    #[pyo3(name = "MergeFrom", signature = (*args, **kwargs))]
    fn merge_from(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let other = &one_argument(Self::upb_class_name(slf), "MergeFrom", args, kwargs)?;
        Self::reify(slf)?;
        slf.borrow_mut().new_version();
        if other.cast::<PyDict>().is_ok() {
            return Err(PyAttributeError::new_err("Merging of dict is not allowed"));
        }
        let field = slf.borrow().full_field_name();
        Self::update_from(py, slf, other, &field, &ConversionBudget::full())
    }

    /// The class of the entry messages of the map, as upb's `GetEntryClass`, of its errors of
    /// arguments.
    #[pyo3(name = "GetEntryClass", signature = (*args, **kwargs))]
    fn get_entry_class(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyType>> {
        no_arguments(Self::upb_class_name(slf), "GetEntryClass", args, kwargs)?;
        let entry = slf.borrow().map.entry.clone();
        registered_class(slf.py(), &entry)
            .ok_or_else(|| PyTypeError::new_err(format!("no class for {entry}")))
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
            Self::equal_as_dicts(py, slf, other)?
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

/// The iterator of the keys of a map, as upb's `MapIterator`: it reads the entries of the map as
/// they are at each step, and fails with upb's `RuntimeError` at each step once the map has taken
/// a new version since it was made, as a method of the map that may change its keys takes one.
#[pyclass(module = "pyspark.sql.connect.proto", name = "MapIterator")]
pub struct MapIterator {
    map: Py<MapContainer>,
    /// The position of the next entry to read.
    at: usize,
    /// The version of the map that the iteration is of.
    version: u64,
}

#[pymethods]
impl MapIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let map = self.map.bind(py).try_borrow()?;
        if map.version != self.version {
            return Err(PyRuntimeError::new_err("Map modified during iteration."));
        }
        let Some((at, key)) = map.entries.key_at_or_after(self.at) else {
            return Ok(None);
        };
        self.at = at + 1;
        key.to_py(py).map(Some)
    }
}

/// The map of a map field of scalars, as upb's `ScalarMapContainer`, of the methods of
/// `MapContainer`, the base of the maps of the runtime.
#[pyclass(
    extends = MapContainer,
    mapping,
    module = "pyspark.sql.connect.proto",
    name = "ScalarMapContainer"
)]
pub struct ScalarMapContainer;

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
        let key = one_argument("MessageMapContainer", "get_or_create", args, kwargs)?;
        MapContainer::get_or_insert(slf.as_super(), &key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(value: &MapValue) -> &Scalar {
        match value {
            MapValue::Scalar(v) => v,
            MapValue::Message(_) => unreachable!("the tests set scalars"),
        }
    }

    fn listed(entries: &Entries) -> Vec<(Scalar, Scalar)> {
        entries
            .iter()
            .map(|(k, v)| (k.clone(), scalar(v).clone()))
            .collect()
    }

    /// The entries of a map keep the order in which their keys were inserted, as a dict does, and
    /// find each key, through removals: random operations on a few keys, which come and go, agree
    /// with a list of the entries in their order, and the holes of the removed entries are never
    /// more than the entries, so the iteration and the room of a map stay of its entries.
    #[test]
    fn entries_keep_the_order_of_insertion_through_removals() {
        let mut entries = Entries::default();
        let mut order: Vec<(Scalar, Scalar)> = Vec::new();
        // A linear congruential generator, of a fixed seed.
        let mut state = 7u64;
        let mut next = move |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };
        for step in 0..20_000i64 {
            let key = Scalar::Str(format!("k{}", next(32)));
            let value = Scalar::I64(step);
            if next(10) < 6 {
                let replaced = entries.insert(key.clone(), MapValue::Scalar(value.clone()));
                match order.iter_mut().find(|(k, _)| *k == key) {
                    Some((_, held)) => {
                        assert_eq!(replaced.as_ref().map(scalar), Some(&*held));
                        *held = value;
                    }
                    None => {
                        assert!(replaced.is_none());
                        order.push((key, value));
                    }
                }
            } else {
                let removed = entries.remove(&key);
                let at = order.iter().position(|(k, _)| *k == key);
                let removed = removed.map(|(k, v)| (k, scalar(&v).clone()));
                assert_eq!(removed, at.map(|at| order.remove(at)));
            }
            assert_eq!(listed(&entries), order, "step {step}");
            // The iteration of a map by positions, as its iterator reads it, gives the same keys.
            let mut at = 0;
            let mut keys = Vec::new();
            while let Some((found, key)) = entries.key_at_or_after(at) {
                keys.push(key.clone());
                at = found + 1;
            }
            assert!(keys.iter().eq(order.iter().map(|(k, _)| k)), "step {step}");
            assert_eq!(entries.len(), order.len());
            assert!(entries.slots.len() <= 2 * entries.len(), "step {step}");
            for (key, value) in &order {
                assert_eq!(entries.get(key).map(scalar), Some(value));
            }
        }
        // The entries of a map, of distinct keys, as a copy takes them.
        let copied = Entries::of_distinct(
            order
                .iter()
                .map(|(k, v)| (k.clone(), MapValue::Scalar(v.clone())))
                .collect(),
        );
        assert_eq!(listed(&copied), order);
        assert!(order
            .iter()
            .all(|(k, v)| copied.get(k).map(scalar) == Some(v)));
        assert!(copied.get(&Scalar::Str("missing".to_string())).is_none());
    }
}

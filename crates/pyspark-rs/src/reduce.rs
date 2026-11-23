//! The copies and pickles of the native objects of the binding that PySpark has as plain Python
//! objects, such as the Observations, the data types and the objects of `pyspark.ml`: of their
//! class, and of the state that the copy and the pickle of a plain object keep (its `__dict__` and
//! the values of the `__slots__` of its class), as PySpark's copies share what its shallow copy
//! shares.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use spark_connect::column::Column;

use crate::column::PyColumn;

/// The object of the native state `init` of the class `cls`, `T` itself or a Python subclass of
/// it, as PyO3's `tp_new` of a `#[new]` makes one of a subclass: so a copy of an object of a
/// subclass, and the object that `_of_state` gives back of its pickle, are of the subclass, as
/// the copies and pickles of PySpark's plain objects are of their class.
pub(crate) fn new_of_class<'py, T: pyo3::PyClass>(
    cls: &Bound<'py, pyo3::types::PyType>,
    init: impl Into<PyClassInitializer<T>>,
) -> PyResult<Bound<'py, T>> {
    let py = cls.py();
    if cls.is(py.get_type::<T>()) {
        return Bound::new(py, init);
    }
    if !cls.is_subclass_of::<T>()? {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{} is not a subtype of {}",
            cls.name()?,
            py.get_type::<T>().name()?
        )));
    }
    // SAFETY: `cls` is a subclass of `T`, of which PyO3's `tp_new_impl`, the `tp_new` of a
    // `#[new]`, makes an object of an initializer of `T`.
    unsafe {
        let ptr = pyo3::impl_::pymethods::tp_new_impl::<PyClassInitializer<T>, T>(
            py,
            init.into(),
            cls.as_type_ptr(),
        )?;
        Ok(Bound::from_owned_ptr(py, ptr).cast_into_unchecked())
    }
}

/// The shallow copy `copy` of `object`, with the instance attributes of `object` in its `__dict__`,
/// as the copy of a plain Python object has them: the attributes are shared by their values.
pub(crate) fn copy_instance_dict(
    object: &Bound<'_, PyAny>,
    copy: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let attributes = object.getattr("__dict__")?;
    if attributes.len()? > 0 {
        copy.getattr("__dict__")?
            .call_method1("update", (attributes,))?;
    }
    Ok(())
}

/// The reduce of `object`, which no copy but the shallow one copies, as PySpark's object of a
/// client, whose `__dict__` holds the thread-local state of the client: of the class of `object`
/// and of a `threading.local`. So a pickle, cloudpickle and a deep copy fail at that object, with
/// the error that PySpark's raise at each protocol, as pickle refuses a `_thread._local`.
pub(crate) fn thread_local_reduce<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = object.py();
    let local = py.import("threading")?.call_method0("local")?;
    let args = PyTuple::new(py, [local])?;
    Ok(PyTuple::new(py, [object.get_type().into_any(), args.into_any()])?.into_any())
}

/// The state of `object` that a pickle of a plain Python object keeps, which
/// `object.__getstate__` gives from the protocol 2: its `__dict__`, or None when that is empty,
/// paired with the values of the `__slots__` of its class that are set, when there are some
/// (`copyreg._slotnames`), and None when there is neither, as the reduce then has no state.
/// `object` has a `__dict__`.
pub(crate) fn plain_object_state<'py>(
    object: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = object.py();
    let attributes = object.getattr(pyo3::intern!(py, "__dict__"))?;
    let attributes = (attributes.len()? > 0).then_some(attributes);
    let slots = slot_values(object)?;
    if slots.is_empty() {
        return Ok(attributes);
    }
    let attributes = attributes.unwrap_or_else(|| py.None().into_bound(py));
    Ok(Some(
        PyTuple::new(py, [attributes, slots.into_any()])?.into_any(),
    ))
}

/// The values of the `__slots__` of the class of `object` that are set, by their names
/// (`copyreg._slotnames`), which the state of the pickle of a plain object pairs with its
/// `__dict__`.
pub(crate) fn slot_values<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let py = object.py();
    let slots = PyDict::new(py);
    let names = py
        .import("copyreg")?
        .call_method1("_slotnames", (object.get_type(),))?;
    for name in names.try_iter()? {
        let name = name?.cast_into::<PyString>()?;
        if let Some(value) = object.getattr_opt(&name)? {
            slots.set_item(name, value)?;
        }
    }
    Ok(slots)
}

/// The reduce of `object`, of a native class of no state of its own, as `object.__reduce_ex__`
/// gives it of PySpark's plain object from the protocol 2: `copyreg.__newobj__` of its class,
/// which a pickle names by reference, with the state of a plain object (`plain_object_state`) when
/// there is one. So it pickles and copies at every protocol: below 2, the default reduce of a class
/// of a native base fails, as copyreg's does.
pub(crate) fn plain_reduce<'py>(object: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let py = object.py();
    let newobj = py.import("copyreg")?.getattr("__newobj__")?;
    let args = PyTuple::new(py, [object.get_type()])?.into_any();
    match plain_object_state(object)? {
        Some(state) => PyTuple::new(py, [newobj, args, state]),
        None => PyTuple::new(py, [newobj, args]),
    }
}

/// The check of `object.__new__` of the arguments of `cls`, a class of PySpark's plain classes of
/// no `__new__` of their own (or a subclass): the arguments of the `__init__` of a subclass are
/// left to it, and a class without an `__init__` of its own takes none
/// (`TypeError: ChiSquareTest() takes no arguments`).
pub(crate) fn check_plain_new(
    cls: &Bound<'_, pyo3::types::PyType>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let py = cls.py();
    let given = !args.is_empty() || kwargs.is_some_and(|kwargs| !kwargs.is_empty());
    let object_init = py
        .get_type::<PyAny>()
        .getattr(pyo3::intern!(py, "__init__"))?;
    if given && cls.getattr(pyo3::intern!(py, "__init__"))?.is(&object_init) {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{}() takes no arguments",
            cls.name()?
        )));
    }
    Ok(())
}

/// The Python list of the columns, each a Column of the wrapper of `Column`, as the grouping
/// columns and the grouping sets of grouped data hold them.
pub(crate) fn column_list<'py>(
    py: Python<'py>,
    columns: &[Column],
) -> PyResult<Bound<'py, PyList>> {
    let items = columns
        .iter()
        .map(|column| Bound::new(py, PyColumn::new(column.clone())))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

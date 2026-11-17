//! The copies and pickles of the wrappers of PySpark's plain objects (readers, writers, groups,
//! catalogs and the like). As PySpark's, a copy of one shares the state that PySpark's shallow
//! copy shares, and a pickle keeps the session or DataFrame that the object holds, which
//! unpickles first, with the state of the object.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use spark_connect::column::Column;
use spark_connect::plan::AggregateGroupType;

use crate::column::PyColumn;

/// The reduce of an object that pickles as `getattr(holder, name)`, such as a catalog as the
/// attribute of its session.
pub(crate) fn getattr_reduce<'py>(
    py: Python<'py>,
    holder: Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    let getattr = py.import("builtins")?.getattr("getattr")?;
    let args = PyTuple::new(py, [holder, name.into_pyobject(py)?.into_any()])?;
    PyTuple::new(py, [getattr, args.into_any()])
}

/// The reduce of an object of its state `args`, through the reconstructor `name` of
/// `pyspark.sql.connect._state_reduce`: a function of the state, which gives back the function
/// that rebuilds the object, which cloudpickle pickles by value, and the arguments of it.
pub(crate) fn state_reduce<'py>(
    py: Python<'py>,
    name: &str,
    args: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("pyspark.sql.connect._state_reduce")?
        .getattr(name)?
        .call1(args)
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

/// The reduce `reduced`, the `(function, args)` pair of `state_reduce`, of `object`, with the
/// instance attributes of `object` as the state of its pickle when there are some, as the reduce
/// of a plain Python object gives them. Without attributes, `reduced` is returned as it is.
pub(crate) fn reduce_with_instance_dict<'py>(
    object: &Bound<'py, PyAny>,
    reduced: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let attributes = object.getattr("__dict__")?;
    if attributes.len()? == 0 {
        return Ok(reduced);
    }
    let reduced = reduced.cast::<PyTuple>()?;
    let state = [reduced.get_item(0)?, reduced.get_item(1)?, attributes];
    Ok(PyTuple::new(object.py(), state)?.into_any())
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
    if slots.is_empty() {
        return Ok(attributes);
    }
    let attributes = attributes.unwrap_or_else(|| py.None().into_bound(py));
    Ok(Some(
        PyTuple::new(py, [attributes, slots.into_any()])?.into_any(),
    ))
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

/// The Python list of the columns, each a Column of the wrapper of `Column`.
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

/// The columns of the Python iterable `items`, each a Column, as `column_list` makes them.
pub(crate) fn columns_of(items: &Bound<'_, PyAny>) -> PyResult<Vec<Column>> {
    items
        .try_iter()?
        .map(|item| Ok(item?.extract::<PyRef<'_, PyColumn>>()?.column.clone()))
        .collect()
}

/// The name of the type of a grouping, which `_of_state` of grouped data takes back.
pub(crate) fn group_type_name(group_type: AggregateGroupType) -> &'static str {
    match group_type {
        AggregateGroupType::GroupBy => "groupBy",
        AggregateGroupType::Rollup => "rollup",
        AggregateGroupType::Cube => "cube",
        AggregateGroupType::Pivot => "pivot",
        AggregateGroupType::GroupingSets => "groupingSets",
    }
}

/// The type of a grouping of its name, as `group_type_name` gives it.
pub(crate) fn group_type_of(name: &str) -> PyResult<AggregateGroupType> {
    match name {
        "groupBy" => Ok(AggregateGroupType::GroupBy),
        "rollup" => Ok(AggregateGroupType::Rollup),
        "cube" => Ok(AggregateGroupType::Cube),
        "pivot" => Ok(AggregateGroupType::Pivot),
        "groupingSets" => Ok(AggregateGroupType::GroupingSets),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown grouping type: {name}"
        ))),
    }
}

/// The Python dict of the string pairs of `map`.
pub(crate) fn str_dict<'py>(
    py: Python<'py>,
    map: &std::collections::HashMap<String, String>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    for (key, value) in map {
        dict.set_item(key, value)?;
    }
    Ok(dict)
}

/// The name of a save mode, which `DataFrameWriter.mode` takes back.
pub(crate) fn save_mode_name(mode: spark_connect::readwriter::SaveMode) -> &'static str {
    use spark_connect::readwriter::SaveMode;
    match mode {
        SaveMode::Append => "append",
        SaveMode::Overwrite => "overwrite",
        SaveMode::ErrorIfExists => "error",
        SaveMode::Ignore => "ignore",
    }
}

/// The Python object of an optional Python object, None if there is none.
pub(crate) fn optional_object<'py>(
    py: Python<'py>,
    object: Option<&Py<PyAny>>,
) -> Bound<'py, PyAny> {
    match object {
        Some(object) => object.clone_ref(py).into_bound(py),
        None => py.None().into_bound(py),
    }
}

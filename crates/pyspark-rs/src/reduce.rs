//! The copies and pickles of the wrappers of PySpark's plain objects (readers, writers, groups,
//! catalogs and the like). As PySpark's, a copy of one shares the state that PySpark's shallow
//! copy shares, and a pickle keeps the session or DataFrame that the object holds, which
//! unpickles first, with the state of the object.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
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

/// The reduce of an object of the class `T`, which pickles by the `_of_state` of `T` called
/// with `args`, the state of the object.
pub(crate) fn of_state_reduce<'py, T: pyo3::PyTypeInfo>(
    py: Python<'py>,
    args: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let of_state = py.get_type::<T>().getattr("_of_state")?;
    PyTuple::new(py, [of_state, args.into_any()])
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

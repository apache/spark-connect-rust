//! PyO3 wrappers for Spark DataTypes.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyList;
use spark_connect::types::{DataType, StructField, TreeFlag, TreeStep};
use std::collections::BTreeMap;

/// Normalize the depth used by both Python schema-tree entry points.
pub(crate) fn normalize_tree_depth(depth: &Bound<'_, PyAny>) -> PyResult<i32> {
    // Reference compares > 0 at each level, so positive fractional depths
    // behave as ceil(depth). Nonpositive depths and NaN print the full tree.
    if !depth.gt(0)? || depth.ge(i32::MAX)? {
        Ok(i32::MAX)
    } else {
        // A positive Fraction or Decimal may underflow to zero as f64. Keep
        // it at depth one: zero would request an unlimited tree instead.
        Ok((depth.extract::<f64>()?.ceil() as i32).max(1))
    }
}

/// Make a DataType picklable so it round-trips to an official-pyspark UDF worker:
/// pickle reconstructs it via `pyspark.sql.types._parse_datatype_json_string(json)`
/// (session-free, present in both our shim and the worker's official pyspark).
fn type_reduce(py: Python<'_>, json: &str) -> PyResult<(Py<PyAny>, (String,))> {
    let f = py
        .import("pyspark.sql.types")?
        .getattr("_parse_datatype_json_string")?;
    Ok((f.unbind(), (json.to_string(),)))
}

/// The `__reduce__` of the type object `obj` of `data_type`: by its JSON, as [`type_reduce`],
/// unless it has a user-defined type, whose JSON need not import its class, or a nullable flag
/// that is not a bool, which the JSON does not keep (`flagged`). Such a type pickles of the type
/// objects it is made of, `args()`, as PySpark's types do; the class is that of
/// `pyspark.sql.types`, also on the Python workers of the server. The JSON is that of PySpark's
/// `json()`, of the metadata dicts of its fields as they are (see `holds_metadata`).
fn reduce_with_udts<'py>(
    obj: &Bound<'py, PyAny>,
    data_type: &DataType,
    flagged: bool,
    args: impl FnOnce() -> PyResult<Bound<'py, pyo3::types::PyTuple>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let py = obj.py();
    if flagged || has_udt(data_type) {
        return Ok((
            obj.get_type().into_any().unbind(),
            args()?.into_any().unbind(),
        ));
    }
    let json = if holds_metadata(py, obj)? {
        type_json_text(obj)?
    } else {
        data_type.json()
    };
    let (f, args) = type_reduce(py, &json)?;
    Ok((f, args.into_pyobject(py)?.into_any().unbind()))
}

/// The user-defined types that the client lowered into core types, by the JSON of their core
/// types. PySpark's `ArrayType`, `MapType`, `StructField` and `StructType` hold the type
/// objects they are made of; so a core type gives back the user-defined type it was made of,
/// rather than one of its JSON. The `pyClass` of the JSON need not name the module of the class,
/// as for Apache Spark's `ExamplePointUDT`, of `pyspark.sql.tests.ExamplePointUDT`.
static LOWERED_UDTS: PyOnceLock<Py<pyo3::types::PyDict>> = PyOnceLock::new();

fn lowered_udts(py: Python<'_>) -> &Bound<'_, pyo3::types::PyDict> {
    LOWERED_UDTS
        .get_or_init(py, || pyo3::types::PyDict::new(py).unbind())
        .bind(py)
}

/// The `StructField` that the client lowered into the type `json`, which is its JSON, if any.
/// A StructField as a type is lowered like a user-defined type, by its JSON (see
/// `py_to_data_type`).
fn lowered_field<'py>(py: Python<'py>, json: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match lowered_udts(py).get_item(json)? {
        Some(object) if object.is_instance_of::<PyStructField>() => Ok(Some(object)),
        _ => Ok(None),
    }
}

/// The type object that the client lowered into the type `json`, which is its JSON, if any: a
/// `StructField` as a type, or a type of its class only (see `is_class_type`), which the core has
/// no type of either.
fn lowered_object<'py>(py: Python<'py>, json: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match lowered_udts(py).get_item(json)? {
        Some(object) if object.is_instance_of::<PyStructField>() || is_class_type(&object)? => {
            Ok(Some(object))
        }
        _ => Ok(None),
    }
}

/// The `typeName()` of the type of its class only that the client lowered into the type `json`,
/// which a tree string writes of it, as PySpark's `_get_jvm_type_name` does; None for another.
fn class_type_name(py: Python<'_>, json: &str) -> PyResult<Option<String>> {
    match lowered_udts(py).get_item(json)? {
        Some(object) if is_class_type(&object)? => {
            Ok(Some(object.call_method0("typeName")?.extract()?))
        }
        _ => Ok(None),
    }
}

/// Python wrapper for any DataType. This is the base of the type-class hierarchy
/// (mirroring pyspark's `DataType`); concrete type classes extend it (directly or via
/// the intermediate abstract bases below), so `isinstance(dt, DataType)` and the object
/// model (simpleString/typeName/json/...) are inherited from here via `self.inner`.
#[pyclass(
    subclass,
    name = "DataType",
    module = "pyspark.sql.types",
    weakref,
    dict
)]
pub struct PyDataType {
    /// The core type. A concrete class reads its type of its own fields (`py_to_data_type`), and
    /// the setters of its parameters keep this one the same (`sync_base`). `None` for a type
    /// whose state is its `__dict__`, as PySpark's plain classes keep it: the base `DataType`
    /// itself, the abstract classes of the hierarchy (`AtomicType`, ...), and the Python
    /// subclasses of them, such as a `UserDefinedType`.
    pub(crate) inner: Option<DataType>,
}

/// Whether `obj` is a type of its class only: a `DataType` of no core type (`PyDataType::inner`
/// is `None`) other than a `UserDefinedType` or an `UnparsedDataType`, which are of their own
/// core types. Such a type is the base `DataType`, an abstract class such as `AtomicType`, or a
/// Python subclass of them, which PySpark's methods of the base `DataType` make: its `typeName`,
/// `simpleString` and `jsonValue` are the name of its class, without `Type` and in lower case.
pub(crate) fn is_class_type(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    match obj.cast::<PyDataType>() {
        Ok(data_type) if data_type.borrow().inner.is_none() => {
            Ok(!is_udt(obj)? && !is_unparsed(obj)?)
        }
        _ => Ok(false),
    }
}

/// The type classes of the extension: a type of one of them is of its class, and a type of a
/// Python subclass of a concrete one is of the subclass (see [`is_subclass_object`]).
fn native_type_classes(py: Python<'_>) -> &[Py<pyo3::types::PyType>] {
    static CLASSES: PyOnceLock<Vec<Py<pyo3::types::PyType>>> = PyOnceLock::new();
    CLASSES.get_or_init(py, || {
        [
            py.get_type::<PyStructField>(),
            py.get_type::<PyStructType>(),
            py.get_type::<PyStringType>(),
            py.get_type::<PyIntegerType>(),
            py.get_type::<PyLongType>(),
            py.get_type::<PyDoubleType>(),
            py.get_type::<PyArrayType>(),
            py.get_type::<PyMapType>(),
            py.get_type::<PyDecimalType>(),
            py.get_type::<PyBooleanType>(),
            py.get_type::<PyDateType>(),
            py.get_type::<PyTimestampType>(),
            py.get_type::<PyTimestampNTZType>(),
            py.get_type::<PyFloatType>(),
            py.get_type::<PyShortType>(),
            py.get_type::<PyByteType>(),
            py.get_type::<PyBinaryType>(),
            py.get_type::<PyNullType>(),
            py.get_type::<PyCharType>(),
            py.get_type::<PyVarcharType>(),
            py.get_type::<PyTimeType>(),
            py.get_type::<PyCalendarIntervalType>(),
            py.get_type::<PyYearMonthIntervalType>(),
            py.get_type::<PyDayTimeIntervalType>(),
            py.get_type::<PyVariantType>(),
            py.get_type::<PyGeometryType>(),
            py.get_type::<PyGeographyType>(),
            py.get_type::<PyTimestampNTZNanosType>(),
            py.get_type::<PyTimestampLTZNanosType>(),
        ]
        .into_iter()
        .map(Bound::unbind)
        .collect()
    })
}

/// The type classes of the extension, the concrete ones of [`native_type_classes`] and the
/// abstract classes of their hierarchy, from `DataType` down.
fn type_class_hierarchy(py: Python<'_>) -> &[Py<pyo3::types::PyType>] {
    static CLASSES: PyOnceLock<Vec<Py<pyo3::types::PyType>>> = PyOnceLock::new();
    CLASSES.get_or_init(py, || {
        let abstract_classes = [
            py.get_type::<PyDataType>(),
            py.get_type::<PyAtomicType>(),
            py.get_type::<PyNumericType>(),
            py.get_type::<PyIntegralType>(),
            py.get_type::<PyFractionalType>(),
            py.get_type::<PyDatetimeType>(),
            py.get_type::<PyAnyTimeType>(),
            py.get_type::<PyAnsiIntervalType>(),
            py.get_type::<PySpatialType>(),
        ];
        abstract_classes
            .into_iter()
            .map(Bound::unbind)
            .chain(native_type_classes(py).iter().map(|cls| cls.clone_ref(py)))
            .collect()
    })
}

/// That the class `cls` of a type is of one branch of the hierarchy of the type classes: the type
/// classes of its MRO are its first one and the classes that this one extends, of which the
/// native constructor of the first makes the state of its objects. A class of type classes of two
/// branches, such as `class X(StringType, IntegerType)`, which PySpark's plain classes allow,
/// raises the `TypeError` that CPython raises of bases of conflicting layouts: its objects would
/// be of the state of one. A constructor checks it too, of a class whose `__init_subclass__` does
/// not call that of `DataType`.
fn check_one_branch(cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<()> {
    let natives = type_class_hierarchy(cls.py());
    if natives.iter().any(|native| cls.is(native)) {
        return Ok(());
    }
    let mut first: Option<Bound<'_, pyo3::types::PyType>> = None;
    for class in cls.mro().iter() {
        let class = class.cast_into::<pyo3::types::PyType>()?;
        if !natives.iter().any(|native| class.is(native)) {
            continue;
        }
        match &first {
            None => first = Some(class),
            Some(first) if !first.is_subclass(&class)? => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "multiple bases have instance lay-out conflict",
                ));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// Whether `obj` is a type of a Python subclass of a concrete type class, such as `class
/// MyString(StringType)`, as PySpark's plain classes are subclassed. Its core type is that of
/// the class, as PySpark's conversions read it by `isinstance`, and the methods of its class call
/// its own methods where PySpark's call them (`typeName`, `simpleString`, `jsonValue`, ...), of
/// which an array, a map or a struct that holds it calls its methods too. A type of no core type,
/// of an abstract class or a `UserDefinedType`, is not one (see [`is_class_type`]).
pub(crate) fn is_subclass_object(obj: &Bound<'_, PyAny>) -> bool {
    if !obj.is_instance_of::<PyDataType>() || has_state_of_dict(obj) {
        return false;
    }
    let ty = obj.get_type();
    !native_type_classes(obj.py()).iter().any(|cls| ty.is(cls))
}

/// Whether `slf` is of the class `T` itself, rather than of a Python subclass of it.
fn is_native<T: pyo3::PyClass>(slf: &Bound<'_, T>) -> bool {
    slf.as_any().get_type().is(slf.py().get_type::<T>())
}

/// As PySpark's `DataType.json`, `json.dumps(self.jsonValue(), separators=(",", ":"),
/// sort_keys=True)`, of the `jsonValue` of `obj`: its own, of a Python subclass.
fn json_of_json_value(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = obj.py();
    json_dumps(py, &obj.call_method0(pyo3::intern!(py, "jsonValue"))?)
}

/// As PySpark's `DataType.__repr__`, `self.__class__.__name__ + "()"`: the name of the class of
/// `obj`, of a Python subclass its own.
fn class_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(format!("{}()", obj.get_type().name()?))
}

/// The `typeName()` of `obj`, which PySpark's `DataType.simpleString` and `jsonValue` return, and
/// the `jsonValue` of an array, a map and a struct names their type by: of a Python subclass, the
/// name of its class without `Type`, in lower case, or its own `typeName`.
fn type_name_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    obj.call_method0(pyo3::intern!(obj.py(), "typeName"))
}

/// The reduce of `obj`, a type of a Python subclass of a type class, as PySpark's plain classes
/// reduce at every protocol from 2: `copyreg.__newobj__(cls)` and the `__dict__` of the type,
/// which holds the attributes `parameters` that PySpark's `__init__` sets, in its order, and the
/// attributes that a user set, of the `__dict__` of `obj`; no state when there are none. So a
/// copy, a deep copy and a pickle are of the subclass, which `cls.__new__(cls)` makes and
/// `__setstate__` sets, and PySpark loads it as its own subclass, also on the Python workers of
/// the server, to which cloudpickle sends a class of `__main__` by value.
fn subclass_reduce<'py>(
    obj: &Bound<'py, PyAny>,
    parameters: Vec<(&str, Bound<'py, PyAny>)>,
) -> PyResult<Py<PyAny>> {
    let py = obj.py();
    let state = pyo3::types::PyDict::new(py);
    for (name, value) in parameters {
        state.set_item(name, value)?;
    }
    let attributes = obj.getattr(pyo3::intern!(py, "__dict__"))?;
    state.call_method1(pyo3::intern!(py, "update"), (attributes,))?;
    // Of a class of `__slots__`, the state pairs the `__dict__`, None if empty, with the slots
    // that are set, as `object.__getstate__` does.
    let slots = crate::reduce::slot_values(obj)?;
    let state = match (state.is_empty(), slots.is_empty()) {
        (true, true) => py.None().into_bound(py),
        (false, true) => state.into_any(),
        (empty, false) => {
            let attributes = if empty {
                py.None().into_bound(py)
            } else {
                state.into_any()
            };
            pyo3::types::PyTuple::new(py, [attributes, slots.into_any()])?.into_any()
        }
    };
    let newobj = py.import("copyreg")?.getattr("__newobj__")?;
    Ok((newobj, (obj.get_type(),), state, py.None(), py.None())
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// The reduce of a type of the class `T` itself, of its JSON `json` (see [`type_reduce`]), or else
/// of a type of a Python subclass of it, of the attributes `parameters` (see [`subclass_reduce`]).
fn reduce_of<'py, T: pyo3::PyClass>(
    slf: &Bound<'py, T>,
    json: &str,
    parameters: impl FnOnce() -> PyResult<Vec<(&'static str, Bound<'py, PyAny>)>>,
) -> PyResult<Py<PyAny>> {
    if !is_native(slf) {
        return subclass_reduce(slf.as_any(), parameters()?);
    }
    let py = slf.py();
    Ok(type_reduce(py, json)?
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// The one instance of the type class `T` of PySpark's `DataTypeSingleton`, of `cls`: of `T`
/// itself, the one that `instance` keeps, made of `init` once, and of a Python subclass, the one
/// of that class, as `DataTypeSingleton.__call__(cls)` keeps one instance of each class in
/// `_instances`, made of `init` on the first call. Arguments raise its `TypeError`. The
/// constructor of the class makes the instance, and Python calls the `__init__` of a subclass
/// after it at each call, where PySpark's metaclass calls it on the first one only.
fn singleton_of<T: pyo3::PyClass>(
    cls: &Bound<'_, pyo3::types::PyType>,
    args: &Bound<'_, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    instance: &PyOnceLock<Py<T>>,
    init: fn() -> pyo3::PyClassInitializer<T>,
) -> PyResult<Py<T>> {
    singleton_arguments(args, kwargs)?;
    let py = cls.py();
    if cls.is(py.get_type::<T>()) {
        return instance
            .get_or_try_init(py, || Py::new(py, init()))
            .map(|instance| instance.clone_ref(py));
    }
    check_one_branch(cls)?;
    static INSTANCES: PyOnceLock<Py<pyo3::types::PyDict>> = PyOnceLock::new();
    let instances = INSTANCES
        .get_or_init(py, || pyo3::types::PyDict::new(py).unbind())
        .bind(py);
    if let Some(instance) = instances.get_item(cls)? {
        return Ok(instance.cast_into::<T>()?.unbind());
    }
    let instance = crate::reduce::new_of_class(cls, init())?;
    instances.set_item(cls, &instance)?;
    Ok(instance.unbind())
}

/// What CPython's `object.__new__(cls, *args, **kwargs)` raises of PySpark's plain classes, whose
/// `__new__` the native `__new__` of `base` stands for: arguments are those of the `__init__` of
/// a subclass, such as a `UserDefinedType` of parameters, so they raise only when `cls` takes
/// them in no `__init__` of its own (`AtomicType() takes no arguments`), or else when they come
/// from the `__new__` of a subclass, which `object.__new__` takes none of.
fn object_new_arguments(
    cls: &Bound<'_, pyo3::types::PyType>,
    base: &Bound<'_, pyo3::types::PyType>,
    args: &Bound<'_, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<()> {
    check_one_branch(cls)?;
    if args.is_empty() && kwargs.is_none_or(|kwargs| kwargs.is_empty()) {
        return Ok(());
    }
    let py = cls.py();
    let new = pyo3::intern!(py, "__new__");
    if !cls.getattr(new)?.is(base.getattr(new)?) {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "object.__new__() takes exactly one argument (the type to instantiate)",
        ));
    }
    let init = pyo3::intern!(py, "__init__");
    if cls
        .getattr(init)?
        .is(py.get_type::<pyo3::types::PyAny>().getattr(init)?)
    {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{}() takes no arguments",
            cls.name()?
        )));
    }
    Ok(())
}

/// What PySpark's `DataTypeSingleton.__call__(cls)` raises, as Python does, for the arguments
/// of a call of a type without parameters: a `TypeError` of the first keyword argument, or else
/// of the positional ones.
fn singleton_arguments(
    args: &Bound<'_, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<()> {
    if let Some((key, _)) = kwargs.and_then(|kwargs| kwargs.iter().next()) {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "DataTypeSingleton.__call__() got an unexpected keyword argument '{}'",
            key.str()?
        )));
    }
    if !args.is_empty() {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "DataTypeSingleton.__call__() takes 1 positional argument but {} were given",
            args.len() + 1
        )));
    }
    Ok(())
}

/// The fields of an interval type, as PySpark's `DayTimeIntervalType` and
/// `YearMonthIntervalType` take them: both of `default` without any, the start field for the
/// end without one, and a `PySparkRuntimeError` of `INVALID_INTERVAL_CASTING` for a field that
/// is not one of the type, of the `str` of each.
fn interval_fields(
    py: Python<'_>,
    start: Option<&Bound<'_, PyAny>>,
    end: Option<&Bound<'_, PyAny>>,
    default: (i32, i32),
    last: i32,
) -> PyResult<(i32, i32)> {
    let field = |value: Option<&Bound<'_, PyAny>>| interval_field(value?, last);
    let (start, end) = match (start, end) {
        (None, None) => return Ok(default),
        (Some(start), None) => (Some(start), Some(start)),
        (start, end) => (start, end),
    };
    match (field(start), field(end)) {
        (Some(start), Some(end)) => Ok((start, end)),
        _ => {
            let text = |value: Option<&Bound<'_, PyAny>>| -> PyResult<String> {
                Ok(match value {
                    Some(value) => value.str()?.to_string(),
                    None => "None".to_string(),
                })
            };
            Err(invalid_interval_casting(py, &text(start)?, &text(end)?))
        }
    }
}

/// The field of an interval type of the fields `0..=last` that `value` is, as PySpark's
/// `value in fields.keys()` takes it: an int of a field, or a float that equals one (`1.0`).
fn interval_field(value: &Bound<'_, PyAny>, last: i32) -> Option<i32> {
    if value.is_instance_of::<pyo3::types::PyFloat>() {
        // `1.0 in {0, 1}` of PySpark, of a float that equals a field.
        let float: f64 = value.extract().ok()?;
        return (float.fract() == 0.0 && (0.0..=f64::from(last)).contains(&float))
            .then_some(float as i32);
    }
    value
        .extract::<i64>()
        .ok()
        .filter(|field| (0..=i64::from(last)).contains(field))
        .map(|field| field as i32)
}

/// PySpark's `PySparkRuntimeError` of `INVALID_INTERVAL_CASTING`, of the `str` of each field.
fn invalid_interval_casting(py: Python<'_>, start_field: &str, end_field: &str) -> PyErr {
    crate::column::raise_pyspark(
        py,
        "PySparkRuntimeError",
        "INVALID_INTERVAL_CASTING",
        &[("start_field", start_field), ("end_field", end_field)],
    )
}

/// The field `value` that is set as the `startField` (`start`) or the `endField` of an interval
/// type of the fields `0..=last`, whose other field is `other`. PySpark's attribute is a plain
/// one, of any value, which the type then shows, as `t.startField = 1` gives
/// `DayTimeIntervalType(1, 3)`; the core type holds a field of the type only, so another value
/// raises the `INVALID_INTERVAL_CASTING` of PySpark's constructor, of the fields it would make.
fn set_interval_field(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
    other: i32,
    last: i32,
    start: bool,
) -> PyResult<i32> {
    if let Some(field) = interval_field(value, last) {
        return Ok(field);
    }
    let (value, other) = (value.str()?.to_string(), other.to_string());
    let (start_field, end_field) = if start {
        (value, other)
    } else {
        (other, value)
    };
    Err(invalid_interval_casting(py, &start_field, &end_field))
}

/// Keep the base `DataType` of the type object `obj` (`PyDataType::inner`) the core type of its
/// parameters, after one of them was set.
fn sync_base(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let data_type = py_to_data_type(obj)?;
    obj.cast::<PyDataType>()?.borrow_mut().inner = Some(data_type);
    Ok(())
}

/// Set the state `state` that PySpark pickled of a type, its `__dict__`, on the type object
/// `obj`. PySpark's types are plain classes, which pickle at the protocols 2 and up as
/// `copyreg.__newobj__(cls)` and this state, whose entries pickle sets in the `__dict__` of the
/// new object. The drop-in's class makes the object of `cls.__new__(cls)`, without arguments,
/// and sets each entry as `setattr` does: a parameter through its setter, so that the core type
/// is that of the parameters, and any other attribute in the `__dict__` of `obj`, as PySpark's
/// has it. `computed` takes the entries that PySpark keeps of the parameters, which the drop-in
/// computes of them, and returns whether it took the entry. The state of an object of a class of
/// `__slots__` is the pair of its `__dict__`, None if empty, and of its slots, which are set as
/// pickle sets them.
fn set_pickled_state<'py>(
    obj: &Bound<'py, PyAny>,
    state: &Bound<'py, PyAny>,
    mut computed: impl FnMut(&str, &Bound<'py, PyAny>) -> PyResult<bool>,
) -> PyResult<()> {
    let py = obj.py();
    let (state, slots) = match state.cast::<pyo3::types::PyTuple>() {
        Ok(pair) if pair.len() == 2 => (pair.get_item(0)?, Some(pair.get_item(1)?)),
        _ => (state.clone(), None),
    };
    if !state.is_none() {
        for (key, value) in state.cast::<pyo3::types::PyDict>()?.iter() {
            match key.cast::<pyo3::types::PyString>() {
                Ok(name) => {
                    if !computed(&name.to_cow()?, &value)? {
                        obj.setattr(name, &value)?;
                    }
                }
                // A key that is not a str is only an entry of the `__dict__`, as pickle sets it.
                Err(_) => obj
                    .getattr(pyo3::intern!(py, "__dict__"))?
                    .set_item(&key, &value)?,
            }
        }
    }
    if let Some(slots) = slots.filter(|slots| !slots.is_none()) {
        for (name, value) in slots.cast::<pyo3::types::PyDict>()?.iter() {
            obj.setattr(name.cast_into::<pyo3::types::PyString>()?, value)?;
        }
    }
    Ok(())
}

/// The abstract intermediate base classes of the type hierarchy, PySpark's `AtomicType`,
/// `NumericType`, `IntegralType`, ...: a concrete type's `__new__` builds the initializer chain
/// `PyDataType{inner} -> ... -> Concrete`, so that `isinstance(dt, NumericType)` holds with
/// PySpark's MRO. Each class makes a type of its class only too, as PySpark's plain class does
/// (see `is_class_type`): `AtomicType()` has no core type, and its `typeName` is `atomic`. The
/// `[...]` are the classes from `DataType` down to the class, of which the last is `$parent`;
/// `singleton` makes the class one of PySpark's `DataTypeSingleton`, and `{...}` holds the
/// further members of the class.
macro_rules! abstract_type {
    ($ty:ident, $name:literal, $parent:ty, [$($chain:ident),*] $(, { $($members:tt)* })?) => {
        abstract_type!(@class $ty, $name, $parent);

        #[pymethods]
        impl $ty {
            /// As the `object.__new__` of PySpark's plain class, which takes no arguments, but
            /// those that the `__init__` of a subclass takes (see `object_new_arguments`).
            #[new]
            #[classmethod]
            #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
            fn new(
                cls: &Bound<'_, pyo3::types::PyType>,
                args: &Bound<'_, pyo3::types::PyTuple>,
                kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
            ) -> PyResult<pyo3::PyClassInitializer<Self>> {
                object_new_arguments(cls, &cls.py().get_type::<Self>(), args, kwargs)?;
                Ok(class_type_chain!($ty, [$($chain),*]))
            }
            $($($members)*)?
        }
    };
    ($ty:ident, $name:literal, $parent:ty, [$($chain:ident),*], singleton) => {
        abstract_type!(@class $ty, $name, $parent);

        #[pymethods]
        impl $ty {
            /// The one instance of the class, and of each Python subclass of it, as PySpark's
            /// `DataTypeSingleton.__call__(cls)` returns, which raises its `TypeError` of an
            /// argument (see `singleton_of`).
            #[new]
            #[classmethod]
            #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
            fn new(
                cls: &Bound<'_, pyo3::types::PyType>,
                args: &Bound<'_, pyo3::types::PyTuple>,
                kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
            ) -> PyResult<Py<Self>> {
                static INSTANCE: PyOnceLock<Py<$ty>> = PyOnceLock::new();
                singleton_of(cls, args, kwargs, &INSTANCE, || {
                    class_type_chain!($ty, [$($chain),*])
                })
            }
        }
    };
    (@class $ty:ident, $name:literal, $parent:ty) => {
        // `module` matches where the drop-in re-exports these (pyspark.sql.types), so
        // `__module__` reports the reference path instead of PyO3's default `builtins`.
        #[pyclass(subclass, name = $name, module = "pyspark.sql.types", extends = $parent)]
        pub struct $ty;
    };
}

/// The `PyClassInitializer` of a type of its class only, of the class `$ty` below the classes
/// `[...]` from `DataType` down: its base has no core type (see `is_class_type`).
macro_rules! class_type_chain {
    ($ty:ident, [$($chain:ident),*]) => {
        pyo3::PyClassInitializer::from(PyDataType { inner: None })
            $(.add_subclass($chain))*
            .add_subclass($ty)
    };
}

abstract_type!(PyAtomicType, "AtomicType", PyDataType, []);
abstract_type!(PyNumericType, "NumericType", PyAtomicType, [PyAtomicType]);
abstract_type!(
    PyIntegralType,
    "IntegralType",
    PyNumericType,
    [PyAtomicType, PyNumericType],
    singleton
);
abstract_type!(
    PyFractionalType,
    "FractionalType",
    PyNumericType,
    [PyAtomicType, PyNumericType]
);
abstract_type!(PyDatetimeType, "DatetimeType", PyAtomicType, [PyAtomicType]);
abstract_type!(
    PyAnyTimeType,
    "AnyTimeType",
    PyDatetimeType,
    [PyAtomicType, PyDatetimeType]
);
abstract_type!(
    PyAnsiIntervalType,
    "AnsiIntervalType",
    PyAtomicType,
    [PyAtomicType]
);
abstract_type!(
    PySpatialType,
    "SpatialType",
    PyAtomicType,
    [PyAtomicType],
    {
        // The constants of the spatial types, PySpark's `SpatialType.MIXED_SRID` and `MIXED_CRS`.
        #[classattr]
        const MIXED_CRS: &'static str = "SRID:ANY";
        #[classattr]
        const MIXED_SRID: i32 = -1;
    }
);

/// Build the `PyClassInitializer` chain for a concrete type: the base `PyDataType`
/// carries `inner`; each intermediate abstract base and finally the concrete unit
/// value are stacked on top. Usage: `init_chain!(inner_expr, Concrete, [Mid1, Mid2, ...])`.
macro_rules! init_chain {
    ($inner:expr, $concrete:ident, [$($mid:ident),*]) => {
        pyo3::PyClassInitializer::from(PyDataType { inner: Some($inner) })
            $(.add_subclass($mid))*
            .add_subclass($concrete)
    };
    // Variant carrying a concrete value (parameterized types).
    ($inner:expr, $concrete_val:expr, [$($mid:ident),*], value) => {
        pyo3::PyClassInitializer::from(PyDataType { inner: Some($inner) })
            $(.add_subclass($mid))*
            .add_subclass($concrete_val)
    };
}

/// A child type that a type keeps as a Python object, as PySpark keeps its attributes
/// (`StructField.dataType`, `ArrayType.elementType`, `MapType.keyType` and `valueType`): the
/// object given to the constructor or setter, or else one made from the core type on first
/// access. Once there, the object is the child, so its edits, in place or as a whole, are the
/// child's, and the core type of the child is read from the object (`live`).
#[derive(Default)]
pub(crate) struct KeptType(std::sync::Mutex<Option<Py<PyAny>>>);

impl KeptType {
    /// Keep `obj` if it is a data type object: a DDL string, or another value, is not kept.
    fn of(obj: &Bound<'_, PyAny>) -> Self {
        if obj.is_instance_of::<PyDataType>() {
            KeptType(std::sync::Mutex::new(Some(obj.clone().unbind())))
        } else {
            Self::default()
        }
    }

    fn kept(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|object| object.clone_ref(py))
    }

    /// The object of the child, made from `snapshot` and kept on the first access.
    fn object(&self, py: Python<'_>, snapshot: &DataType) -> PyResult<Py<PyAny>> {
        if let Some(object) = self.kept(py) {
            return Ok(object);
        }
        let made = data_type_to_py(py, snapshot)?;
        let mut kept = self.0.lock().unwrap_or_else(|e| e.into_inner());
        Ok(kept.get_or_insert(made).clone_ref(py))
    }

    /// The core type of the child, as it is now: read from its object when it has one.
    fn live(&self, py: Python<'_>, snapshot: &DataType) -> PyResult<DataType> {
        match self.kept(py) {
            Some(object) => py_to_data_type(object.bind(py)),
            None => Ok(snapshot.clone()),
        }
    }
}

/// Construct a `StructType` Python object (with its full base chain) from core fields.
fn py_new_struct(py: Python<'_>, fields: Vec<StructField>) -> PyResult<Py<PyStructType>> {
    let objects = fields
        .into_iter()
        .map(|field| field_object(py, PyStructField::from_field(py, field)?))
        .collect::<PyResult<Vec<_>>>()?;
    struct_of_list(py, PyList::new(py, objects)?)
}

/// The `StructType` of the list `fields` of `StructField` objects, which the struct holds as its
/// `fields`, as PySpark's `__init__` does: the list is the struct's own, not a copy of it.
fn struct_of_list(py: Python<'_>, fields: Bound<'_, PyList>) -> PyResult<Py<PyStructType>> {
    let (base, parts) = struct_parts(py, fields)?;
    Py::new(py, init_chain!(base, parts, [], value))
}

/// The base core type and the parts of a `StructType` of the list `fields` of `StructField`
/// objects. Its `names` are the names of the fields as they are now.
fn struct_parts(py: Python<'_>, fields: Bound<'_, PyList>) -> PyResult<(DataType, PyStructType)> {
    let names = PyList::empty(py);
    let mut base_fields = Vec::with_capacity(fields.len());
    for item in fields.iter() {
        let object = as_struct_field(&item)?;
        names.append(object.borrow().field.name.as_str())?;
        base_fields.push(object.borrow().current_field(py)?);
    }
    let need_conversion = need_conversion_of_fields(py, &fields)?;
    let parts = PyStructType {
        fields: fields.unbind(),
        names: names.unbind(),
        need_conversion,
    };
    Ok((
        DataType::Struct {
            fields: base_fields,
        },
        parts,
    ))
}

/// The `needConversion` of each of the `StructField` objects of `fields`, as they are now, as
/// PySpark's `_needConversion`, `[f.needConversion() for f in self]`: of a field of a Python
/// subclass, of its own method.
fn need_conversion_of_fields(py: Python<'_>, fields: &Bound<'_, PyList>) -> PyResult<Vec<bool>> {
    fields
        .iter()
        .map(|item| {
            if is_subclass_object(&item) {
                return item
                    .call_method0(pyo3::intern!(py, "needConversion"))?
                    .is_truthy();
            }
            as_struct_field(&item)?.borrow().need_conversion_now(py)
        })
        .collect()
}

/// The `needConversion` of the data type object `obj`, as PySpark's: a struct and a user-defined
/// type always need a conversion, and an array or a map when one of its element types does. The
/// fields of a struct are not read, nor their metadata, so this does not grow with the struct.
/// The types that `obj` holds are read as PySpark's call of their `needConversion` reads them
/// (see `child_need_conversion`).
fn need_conversion_of_object(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    fn of_child(py: Python<'_>, kept: &KeptType, snapshot: &DataType) -> PyResult<bool> {
        match kept.kept(py) {
            Some(object) => child_need_conversion(py, object.bind(py)),
            None => need_conversion(py, snapshot),
        }
    }
    if obj.is_instance_of::<PyStructType>() {
        return Ok(true);
    }
    // A StructField, also as a type, needs a conversion as its dataType does, as PySpark's does.
    if let Ok(field) = obj.cast::<PyStructField>() {
        let field = field.borrow();
        return match field.data_type_object.kept(py) {
            Some(object) => {
                drop(field);
                child_need_conversion(py, object.bind(py))
            }
            None => need_conversion(py, &field.field.data_type),
        };
    }
    if let Ok(array) = obj.cast::<PyArrayType>() {
        let array = array.borrow();
        return of_child(py, &array.element_object, &array.element_type);
    }
    if let Ok(map) = obj.cast::<PyMapType>() {
        let map = map.borrow();
        return Ok(of_child(py, &map.key_object, &map.key_type)?
            || of_child(py, &map.value_object, &map.value_type)?);
    }
    if is_udt(obj)? {
        return Ok(true);
    }
    need_conversion(py, &py_to_data_type(obj)?)
}

/// The `needConversion` of the type object `child` that a type holds, as PySpark's call of its
/// method reads it: of a Python subclass, of its own method, and else as the class's (see
/// [`need_conversion_of_object`]).
fn child_need_conversion(py: Python<'_>, child: &Bound<'_, PyAny>) -> PyResult<bool> {
    if is_subclass_object(child) {
        return child
            .call_method0(pyo3::intern!(py, "needConversion"))?
            .is_truthy();
    }
    need_conversion_of_object(py, child)
}

/// The `TypeError` of an iteration of `obj`, which is not iterable, as CPython words it: with the
/// `__name__` of its type, where the error of PyO3's class names the type by its module too.
fn not_iterable(obj: &Bound<'_, PyAny>) -> PyErr {
    let name = obj
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_default();
    PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("'{name}' object is not iterable"))
}

/// The `AttributeError` of the attribute `attr` that `obj` does not have, as CPython words it
/// (see [`not_iterable`]).
fn no_attribute(obj: &Bound<'_, PyAny>, attr: &str) -> PyErr {
    let name = obj
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_default();
    PyErr::new::<pyo3::exceptions::PyAttributeError, _>(format!(
        "'{name}' object has no attribute '{attr}'"
    ))
}

/// The object of the type below `obj` by `step`, when it is kept as an object (see `KeptType`).
/// A type that is not kept has only bool flags, which the core prints.
fn kept_child<'py>(obj: &Bound<'py, PyAny>, step: TreeStep) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = obj.py();
    let kept = if let Ok(st) = obj.cast::<PyStructType>() {
        let TreeStep::Field(i) = step else {
            return Ok(None);
        };
        let item = st.borrow().fields.bind(py).get_item(i)?;
        let field = as_struct_field(&item)?;
        let kept = field.borrow().data_type_object.kept(py);
        kept
    } else if let Ok(array) = obj.cast::<PyArrayType>() {
        match step {
            TreeStep::Element => array.borrow().element_object.kept(py),
            _ => None,
        }
    } else if let Ok(map) = obj.cast::<PyMapType>() {
        match step {
            TreeStep::Key => map.borrow().key_object.kept(py),
            TreeStep::Value => map.borrow().value_object.kept(py),
            TreeStep::Field(_) | TreeStep::Element => None,
        }
    } else {
        None
    };
    Ok(kept.map(|object| object.into_bound(py)))
}

/// The type object at `path` below the type `root`, when each type on the way is kept as an
/// object (see `kept_child`).
fn tree_node<'py>(
    root: &Bound<'py, PyAny>,
    path: &[TreeStep],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let mut node = Some(root.clone());
    for step in path {
        node = match node {
            Some(current) => kept_child(&current, *step)?,
            None => return Ok(None),
        };
    }
    Ok(node)
}

/// The name that a tree string writes of the type at `path` below the type `root`, whose core
/// type is the JSON `json`: of a type of its class only, its `typeName()`, as PySpark's
/// `_get_jvm_type_name` writes it, of the object that is kept there, or else of the object that
/// the client lowered of the JSON; None for another type.
fn tree_type_name_text(
    root: &Bound<'_, PyAny>,
    path: &[TreeStep],
    json: &str,
) -> PyResult<Option<String>> {
    match tree_node(root, path)? {
        Some(node) if is_class_type(&node)? => Ok(Some(
            node.call_method0(pyo3::intern!(root.py(), "typeName"))?
                .extract()?,
        )),
        Some(_) => Ok(None),
        None => class_type_name(root.py(), json),
    }
}

/// The name that a tree string writes of the type at `path` below the type `root` when it is of a
/// Python subclass (see `is_subclass_object`), as PySpark's `_get_jvm_type_name` writes it: the
/// `simpleString()` of a decimal, a char, a varchar, an interval of fields or a time, and else the
/// `typeName()`, of its own methods; None for another type.
fn subclass_tree_type_name(root: &Bound<'_, PyAny>, path: &[TreeStep]) -> PyResult<Option<String>> {
    let Some(node) = tree_node(root, path)? else {
        return Ok(None);
    };
    if !is_subclass_object(&node) {
        return Ok(None);
    }
    let of_simple_string = node.is_instance_of::<PyDecimalType>()
        || node.is_instance_of::<PyCharType>()
        || node.is_instance_of::<PyVarcharType>()
        || node.is_instance_of::<PyDayTimeIntervalType>()
        || node.is_instance_of::<PyYearMonthIntervalType>()
        || node.is_instance_of::<PyTimeType>();
    let py = root.py();
    let name = if of_simple_string {
        node.call_method0(pyo3::intern!(py, "simpleString"))?
    } else {
        type_name_of(&node)?
    };
    Ok(Some(name.str()?.to_cow()?.into_owned()))
}

/// The text of the nullable flag `flag` of the type at `path` below the type `root`, as PySpark's
/// `treeString` writes it: `str(flag).lower()` of the flag as it was given. `None` when the type
/// is not kept as an object, whose flags are the core's bools.
fn tree_flag_text(
    root: &Bound<'_, PyAny>,
    path: &[TreeStep],
    flag: TreeFlag,
) -> PyResult<Option<String>> {
    let py = root.py();
    let Some(node) = tree_node(root, path)? else {
        return Ok(None);
    };
    let flag_object = match flag {
        TreeFlag::Nullable(i) => {
            let Ok(st) = node.cast::<PyStructType>() else {
                return Ok(None);
            };
            let item = st.borrow().fields.bind(py).get_item(i)?;
            let field = as_struct_field(&item)?;
            let nullable = field.borrow().nullable.clone_ref(py);
            nullable.into_bound(py)
        }
        TreeFlag::ContainsNull => match node.cast::<PyArrayType>() {
            Ok(array) => array
                .borrow()
                .contains_null_object
                .clone_ref(py)
                .into_bound(py),
            Err(_) => return Ok(None),
        },
        TreeFlag::ValueContainsNull => match node.cast::<PyMapType>() {
            Ok(map) => map
                .borrow()
                .value_contains_null_object
                .clone_ref(py)
                .into_bound(py),
            Err(_) => return Ok(None),
        },
    };
    Ok(Some(flag_object.str()?.to_cow()?.to_lowercase()))
}

/// The base of a `StructField` object. Its core type is not read: the type of a field is read
/// from the field (see `py_to_data_type`), which is a lowered type of the field's JSON.
fn field_base(field: PyStructField) -> pyo3::PyClassInitializer<PyStructField> {
    init_chain!(
        DataType::Unparsed {
            data_type_string: String::new(),
        },
        field,
        [],
        value
    )
}

/// The Python object of the `StructField` `field`.
fn field_object(py: Python<'_>, field: PyStructField) -> PyResult<Py<PyStructField>> {
    Py::new(py, field_base(field))
}

/// The core type of `obj` for a proto that the client sends, as PySpark's
/// `pyspark_types_to_proto_types`, which has no proto of a `StructField` as a type, nor of a
/// type of its class only, such as `AtomicType()`: it is the `UNSUPPORTED_OPERATION` of the
/// first one, `obj` itself or else the first in its fields.
pub(crate) fn py_to_proto_data_type(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let data_type = py_to_data_type(obj)?;
    if let Some(field) = unsendable_field_repr(obj, &data_type)? {
        return Err(crate::column::raise_pyspark(
            obj.py(),
            "PySparkValueError",
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("data type {field}"))],
        ));
    }
    Ok(data_type)
}

/// The core type of `value`, the return type of a UDTF, as PySpark's `PythonUDTF.to_plan` converts
/// it with `pyspark_types_to_proto_types`, of the `UnparsedDataType` that it makes of a str: a str
/// is a DDL string that the server parses, which protobuf's string field takes as UTF-8, so one
/// that UTF-8 cannot encode raises its `UnicodeEncodeError`; and a DataType is converted as
/// [`py_to_proto_data_type`] converts it. None for another value, such as an int, of which
/// PySpark raises [`not_a_type_error`].
pub(crate) fn udtf_return_type(value: &Bound<'_, PyAny>) -> PyResult<Option<DataType>> {
    if value.is_instance_of::<pyo3::types::PyString>() {
        return Ok(Some(DataType::Unparsed {
            data_type_string: crate::str_argument(value)?,
        }));
    }
    if value.is_instance_of::<PyDataType>() {
        return py_to_proto_data_type(value).map(Some);
    }
    Ok(None)
}

/// The error of PySpark's `pyspark_types_to_proto_types` of `value`, which is not a DataType: the
/// `UNSUPPORTED_OPERATION` of `data type {value}`, of the str of the value as it is, a lone
/// surrogate too, or the error of that str.
pub(crate) fn not_a_type_error(value: &Bound<'_, PyAny>) -> PyErr {
    let py = value.py();
    let operation = match value
        .str()
        .and_then(|text| pyo3::types::PyString::new(py, "data type ").add(text))
    {
        Ok(operation) => operation,
        Err(err) => return err,
    };
    crate::errors::raise_pyspark_of(
        py,
        "PySparkValueError",
        "UNSUPPORTED_OPERATION",
        &[("operation", &operation)],
    )
}

/// The repr of the first type of the type object `obj` of `data_type`, `obj` itself or one
/// nested in it, which PySpark's conversion of a plan's type (`pyspark_types_to_proto_types`)
/// does not take: a `StructField` as a type, or a type of its class only; None if there is none.
pub(crate) fn unsendable_field_repr(
    obj: &Bound<'_, PyAny>,
    data_type: &DataType,
) -> PyResult<Option<String>> {
    let nested = if obj.is_instance_of::<PyStructField>() || is_class_type(obj)? {
        Some(obj.clone())
    } else {
        nested_python_type_in(obj.py(), data_type)?
    };
    nested
        .map(|field| field.repr().and_then(|text| text.extract::<String>()))
        .transpose()
}

/// The first `StructField` that is a type of `data_type`, in the depth-first order of its fields,
/// down to `depth` levels (as `treeString` prints them), as its Python object.
fn nested_field_in<'py>(
    py: Python<'py>,
    data_type: &DataType,
    depth: i32,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    nested_lowered_in(py, data_type, depth, true)
}

/// The first type of `data_type` that the core has no type of, in the depth-first order of its
/// fields, as its Python object: a `StructField` as a type, or a type of its class only, of which
/// PySpark has no proto (`pyspark_types_to_proto_types`), nor an Arrow type (`to_arrow_type`).
pub(crate) fn nested_python_type_in<'py>(
    py: Python<'py>,
    data_type: &DataType,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    nested_lowered_in(py, data_type, i32::MAX, false)
}

/// The first type of `data_type` that the client lowered by its JSON, a `StructField` as a type
/// (only, if `fields_only`) or a type of its class only, in the depth-first order of its fields,
/// down to `depth` levels, as its Python object.
fn nested_lowered_in<'py>(
    py: Python<'py>,
    data_type: &DataType,
    depth: i32,
    fields_only: bool,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if depth <= 0 {
        return Ok(None);
    }
    let children: Vec<&DataType> = match data_type {
        DataType::Struct { fields } => fields.iter().map(|field| &field.data_type).collect(),
        DataType::Array { element_type, .. } => vec![element_type],
        DataType::Map {
            key_type,
            value_type,
            ..
        } => vec![key_type, value_type],
        _ => Vec::new(),
    };
    for child in children {
        if let DataType::Unparsed { data_type_string } = child {
            let lowered = if fields_only {
                lowered_field(py, data_type_string)?
            } else {
                lowered_object(py, data_type_string)?
            };
            if lowered.is_some() {
                return Ok(lowered);
            }
        }
        if let Some(lowered) = nested_lowered_in(py, child, depth - 1, fields_only)? {
            return Ok(Some(lowered));
        }
    }
    Ok(None)
}

/// `value` as a list: the list itself, or else a new list of its items.
fn list_of<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    match value.cast::<PyList>() {
        Ok(list) => Ok(list.clone()),
        Err(_) => PyList::new(py, value.try_iter()?.collect::<PyResult<Vec<_>>>()?),
    }
}

/// The `StructField` object of `item`, which a `StructType` holds in its `fields`.
fn as_struct_field<'py>(item: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyStructField>> {
    item.cast::<PyStructField>().cloned().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "StructType fields must be StructField objects",
        )
    })
}

/// The default of a nullable argument (`nullable`, `containsNull`, `valueContainsNull`): a `True`.
fn default_flag() -> Py<PyAny> {
    Python::attach(|py| flag_of_bool(py, true))
}

/// A nullable flag of the bool `flag`, as the Python object PySpark keeps.
fn flag_of_bool(py: Python<'_>, flag: bool) -> Py<PyAny> {
    pyo3::types::PyBool::new(py, flag)
        .to_owned()
        .into_any()
        .unbind()
}

/// Whether the nullable flag `flag` is a bool. The core keeps only the truth of any other value.
fn is_bool_flag(flag: &Bound<'_, PyAny>) -> bool {
    flag.is_instance_of::<pyo3::types::PyBool>()
}

/// The name of a `StructField`, which PySpark asserts is a str.
fn field_name(name: &Bound<'_, PyAny>) -> PyResult<String> {
    if !name.is_instance_of::<pyo3::types::PyString>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
            "field name {} should be a string",
            name.str()?
        )));
    }
    name.extract()
}

/// Whether PySpark's converters of local rows assert a nullable flag of the struct `obj` to be
/// other than a bool, as `LocalDataToArrowConversion._create_converter` does: the `nullable` of
/// each field, and the `containsNull` of an array where its converter is made, and the
/// `valueContainsNull` of a map. A converter is made for the children of a type only where the
/// type needs one (`infer::needs_converter`), so that a flag of an atomic type is not asserted.
pub(crate) fn has_non_bool_converter_flag(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let data_type = py_to_data_type(obj)?;
    converter_flag_is_other(py, &data_type, Some(obj), true)
}

/// The walk of `has_non_bool_converter_flag` for a type, of its object `object` if it is kept, and
/// of its `nullable` as a converter is made for it.
fn converter_flag_is_other(
    py: Python<'_>,
    data_type: &DataType,
    object: Option<&Bound<'_, PyAny>>,
    nullable: bool,
) -> PyResult<bool> {
    if !crate::infer::needs_converter(data_type, nullable) {
        return Ok(false);
    }
    // The flag as it was given and the object of a child, which the core type has only as a bool
    // and a snapshot of the child.
    let child = |kept: &KeptType| -> Option<Bound<'_, PyAny>> {
        kept.kept(py).map(|object| object.into_bound(py))
    };
    match data_type {
        DataType::Struct { fields } => {
            for (i, field) in fields.iter().enumerate() {
                let (flag, kept) = match object.and_then(|o| o.cast::<PyStructType>().ok()) {
                    Some(st) => {
                        let item = st.borrow().fields.bind(py).get_item(i)?;
                        let object = as_struct_field(&item)?;
                        let object = object.borrow();
                        let kept = object.data_type_object.kept(py).map(|c| c.into_bound(py));
                        (Some(object.nullable.clone_ref(py).into_bound(py)), kept)
                    }
                    None => (None, None),
                };
                if flag.is_some_and(|flag| !is_bool_flag(&flag)) {
                    return Ok(true);
                }
                if converter_flag_is_other(py, &field.data_type, kept.as_ref(), field.nullable)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        DataType::Array {
            element_type,
            contains_null,
        } => {
            let array = object.and_then(|o| o.cast::<PyArrayType>().ok());
            let (flag, kept) = match &array {
                Some(array) => {
                    let array = array.borrow();
                    (
                        Some(array.contains_null_object.clone_ref(py).into_bound(py)),
                        child(&array.element_object),
                    )
                }
                None => (None, None),
            };
            if flag.is_some_and(|flag| !is_bool_flag(&flag)) {
                return Ok(true);
            }
            converter_flag_is_other(py, element_type, kept.as_ref(), *contains_null)
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let map = object.and_then(|o| o.cast::<PyMapType>().ok());
            let (flag, key, value) = match &map {
                Some(map) => {
                    let map = map.borrow();
                    (
                        Some(map.value_contains_null_object.clone_ref(py).into_bound(py)),
                        child(&map.key_object),
                        child(&map.value_object),
                    )
                }
                None => (None, None, None),
            };
            if flag.is_some_and(|flag| !is_bool_flag(&flag)) {
                return Ok(true);
            }
            Ok(converter_flag_is_other(py, key_type, key.as_ref(), false)?
                || converter_flag_is_other(py, value_type, value.as_ref(), *value_contains_null)?)
        }
        _ => Ok(false),
    }
}

/// Whether the JSON of the type object `obj` is written from its objects, rather than from the
/// core type: a nullable flag of it, or of a field in it, is other than a bool, a name is not a
/// str, a type is a `StructField`, or a type, `obj` itself or one in it, is of a Python subclass,
/// whose JSON is of its own methods.
pub(crate) fn has_other_flag(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if is_subclass_object(obj) {
        return Ok(true);
    }
    if let Ok(field) = obj.cast::<PyStructField>() {
        let field = field.borrow();
        if field.name_object.is_some() || !is_bool_flag(field.nullable.bind(py)) {
            return Ok(true);
        }
        return match field.data_type_object.kept(py) {
            Some(data_type) => child_has_other_flag(py, data_type.bind(py)),
            None => Ok(false),
        };
    }
    if let Ok(array) = obj.cast::<PyArrayType>() {
        let array = array.borrow();
        if !is_bool_flag(array.contains_null_object.bind(py)) {
            return Ok(true);
        }
        return match array.element_object.kept(py) {
            Some(element) => child_has_other_flag(py, element.bind(py)),
            None => Ok(false),
        };
    }
    if let Ok(map) = obj.cast::<PyMapType>() {
        let map = map.borrow();
        if !is_bool_flag(map.value_contains_null_object.bind(py)) {
            return Ok(true);
        }
        for child in [map.key_object.kept(py), map.value_object.kept(py)]
            .into_iter()
            .flatten()
        {
            if child_has_other_flag(py, child.bind(py))? {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    if let Ok(st) = obj.cast::<PyStructType>() {
        for item in st.borrow().fields.bind(py).iter() {
            if has_other_flag(py, &item)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// `has_other_flag` of a type in a type position: a `StructField` or a type of its class only
/// there is written from its own object, as its JSON is not the core's.
fn child_has_other_flag(py: Python<'_>, child: &Bound<'_, PyAny>) -> PyResult<bool> {
    if child.is_instance_of::<PyStructField>() || is_class_type(child)? {
        return Ok(true);
    }
    has_other_flag(py, child)
}

/// The key of the metadata of a struct field that holds the collations of its strings.
const COLLATIONS_METADATA_KEY: &str = "__COLLATIONS";

/// Whether the type `data_type` of a struct field has a string of a collation, also in its arrays
/// and maps, but not in a struct, whose fields are of their own: PySpark's `jsonValue` of the
/// field then writes the types of `removeCollations(self.dataType)`, of the base classes of its
/// strings, arrays and maps, and the collations in its metadata, as the core's JSON does.
fn has_collations(data_type: &DataType) -> bool {
    match data_type {
        DataType::String { collation } => collation != "UTF8_BINARY",
        DataType::Array { element_type, .. } => has_collations(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_collations(key_type) || has_collations(value_type),
        _ => false,
    }
}

/// Set the nullable flags in the JSON value `json` of `obj` to the flags as they were given, as
/// PySpark's `jsonValue` has them, and the JSON of the types of Python subclasses to their own.
/// Returns whether one of them is other than a bool, or one type is written of its own object.
/// `collated` is of a type in a field that has collations (see `has_collations`).
fn keep_flags_in_json(
    py: Python<'_>,
    json: &Bound<'_, PyAny>,
    obj: &Bound<'_, PyAny>,
    collated: bool,
) -> PyResult<bool> {
    if let Ok(field) = obj.cast::<PyStructField>() {
        let field = field.borrow();
        let mut other = false;
        if let Some(name) = &field.name_object {
            json.set_item("name", name.bind(py))?;
            other = true;
        }
        json.set_item("nullable", field.nullable.bind(py))?;
        other |= !is_bool_flag(field.nullable.bind(py));
        // As PySpark's, the metadata dict of the field, of the values as they were given, which
        // the core's JSON values do not keep, and the collations of its strings, if it has some.
        let metadata = field.metadata.bind(py);
        if !metadata.is_empty() {
            let collated = match field.data_type_object.kept(py) {
                Some(object) => has_collations(&py_to_data_type(object.bind(py))?),
                None => has_collations(&field.field.data_type),
            };
            if collated {
                let with_collations = metadata.copy()?;
                let collations = json
                    .get_item("metadata")?
                    .get_item(COLLATIONS_METADATA_KEY)?;
                with_collations.set_item(COLLATIONS_METADATA_KEY, collations)?;
                json.set_item("metadata", with_collations)?;
            } else {
                json.set_item("metadata", metadata)?;
            }
            other = true;
        }
        if let Some(data_type) = field.data_type_object.kept(py) {
            let data_type = data_type.bind(py);
            // The collations matter only to the types of Python subclasses in the field.
            let collated = holds_subclass_object(py, data_type)?
                && has_collations(&py_to_data_type(data_type)?);
            other |= keep_child_json(py, json, "type", data_type, collated)?;
        }
        return Ok(other);
    }
    if let Ok(array) = obj.cast::<PyArrayType>() {
        let array = array.borrow();
        json.set_item("containsNull", array.contains_null_object.bind(py))?;
        let mut other = !is_bool_flag(array.contains_null_object.bind(py));
        if let Some(element) = array.element_object.kept(py) {
            other |= keep_child_json(py, json, "elementType", element.bind(py), collated)?;
        }
        return Ok(other);
    }
    if let Ok(map) = obj.cast::<PyMapType>() {
        let map = map.borrow();
        json.set_item("valueContainsNull", map.value_contains_null_object.bind(py))?;
        let mut other = !is_bool_flag(map.value_contains_null_object.bind(py));
        if let Some(key) = map.key_object.kept(py) {
            other |= keep_child_json(py, json, "keyType", key.bind(py), collated)?;
        }
        if let Some(value) = map.value_object.kept(py) {
            other |= keep_child_json(py, json, "valueType", value.bind(py), collated)?;
        }
        return Ok(other);
    }
    if let Ok(st) = obj.cast::<PyStructType>() {
        let items = st.borrow().fields.clone_ref(py);
        let fields = json.get_item("fields")?;
        let mut other = false;
        for (i, item) in items.bind(py).iter().enumerate() {
            // As PySpark's `[f.jsonValue() for f in self]`, of the own method of a field of a
            // Python subclass.
            if is_subclass_object(&item) {
                fields.set_item(i, item.call_method0(pyo3::intern!(py, "jsonValue"))?)?;
                other = true;
                continue;
            }
            other |= keep_flags_in_json(py, &fields.get_item(i)?, &item, false)?;
        }
        return Ok(other);
    }
    Ok(false)
}

/// Set the type at `key` of the JSON object `json` to the JSON of `child`, with its flags as given.
/// A `StructField` as a type is its own JSON object, and a type of its class only, or of a Python
/// subclass, is its own `jsonValue()`, as PySpark's `jsonValue` has them, but a string, an array
/// and a map of a Python subclass in a field of collations, which PySpark writes of their base
/// classes (`removeCollations`), as the core's JSON has them.
fn keep_child_json(
    py: Python<'_>,
    json: &Bound<'_, PyAny>,
    key: &str,
    child: &Bound<'_, PyAny>,
    collated: bool,
) -> PyResult<bool> {
    if let Ok(field) = child.cast::<PyStructField>() {
        let (nested, _) = field_json_value(py, field)?;
        json.set_item(key, nested)?;
        return Ok(true);
    }
    let of_base_class = collated
        && (child.is_instance_of::<PyStringType>()
            || child.is_instance_of::<PyCharType>()
            || child.is_instance_of::<PyVarcharType>()
            || child.is_instance_of::<PyArrayType>()
            || child.is_instance_of::<PyMapType>());
    if is_class_type(child)? || (is_subclass_object(child) && !of_base_class) {
        json.set_item(key, child.call_method0(pyo3::intern!(py, "jsonValue"))?)?;
        return Ok(true);
    }
    keep_flags_in_json(py, &json.get_item(key)?, child, collated)
}

/// The `jsonValue` of the type object `obj`, with its nullable flags as given, and whether one
/// of them is other than a bool.
fn type_json_value<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let json = py_json_value(py, &py_to_data_type(obj)?.json_value(), false)?;
    let mut other = keep_flags_in_json(py, &json, obj, false)?;
    // As PySpark's `jsonValue` of an array, a map and a struct, of `self.typeName()`: of a Python
    // subclass, the name of its class, or its own `typeName`.
    if is_subclass_object(obj)
        && (obj.is_instance_of::<PyArrayType>()
            || obj.is_instance_of::<PyMapType>()
            || obj.is_instance_of::<PyStructType>())
    {
        json.set_item("type", type_name_of(obj)?)?;
        other = true;
    }
    Ok((json, other))
}

/// The `json()` of the type object `obj`: the core's text, or else `json.dumps` of its
/// `jsonValue` when a flag is other than a bool, as PySpark's `json()` writes it.
fn type_json_text(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = obj.py();
    let (json, other) = type_json_value(py, obj)?;
    if !other {
        return Ok(py_to_data_type(obj)?.json());
    }
    json_dumps(py, &json)
}

/// `json.dumps` of the JSON value `json`, with the separators and sorted keys of `json()`.
fn json_dumps(py: Python<'_>, json: &Bound<'_, PyAny>) -> PyResult<String> {
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("separators", (",", ":"))?;
    kwargs.set_item("sort_keys", true)?;
    py.import("json")?
        .call_method("dumps", (json,), Some(&kwargs))?
        .extract()
}

/// The `json()` text of the type object `schema`, as PySpark's `json()` writes it, when the core's
/// text is not it: of a type of a Python subclass, of its own method, and of a struct of a nullable
/// flag that is not a bool or of metadata (see `type_json_value`). None for no schema, or another
/// one.
pub(crate) fn raw_schema_json(schema: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(schema) = schema else {
        return Ok(None);
    };
    let py = schema.py();
    if is_subclass_object(schema) {
        return Ok(Some(
            schema.call_method0(pyo3::intern!(py, "json"))?.extract()?,
        ));
    }
    if !schema.is_instance_of::<PyStructType>() {
        return Ok(None);
    }
    let (json, other) = type_json_value(py, schema)?;
    if !other {
        return Ok(None);
    }
    json_dumps(py, &json).map(Some)
}

/// The `jsonValue` of the field `field`, with its nullable flags as given, and whether one of
/// them is other than a bool.
fn field_json_value<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyStructField>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let v = field.borrow().current_field(py)?.json_value();
    let json = py_json_value(py, &v, true)?;
    let other = keep_flags_in_json(py, &json, field.as_any(), false)?;
    Ok((json, other))
}

/// The `fromJson` helper `name` of the shim `pyspark.sql.types`, which parses the nested JSON
/// forms of the types as PySpark's: the collations of a struct field by their paths, and the
/// Python class of a `udt`. The core's grammar parses the strings of the atomic types.
fn shim_from_json<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("pyspark.sql.types")?.getattr(name)
}

/// Convert a Python `StructField` metadata dict to the core JSON-value map. Values may be any
/// JSON-serializable Python object (mirrors pyspark's `Dict[str, Any]` field metadata), so we
/// route through `json.dumps` -> serde_json.
fn py_metadata_to_map(
    py: Python<'_>,
    d: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<BTreeMap<String, serde_json::Value>> {
    let s: String = py
        .import("json")?
        .getattr("dumps")?
        .call1((d,))?
        .extract()?;
    serde_json::from_str(&s).or_else(|e| {
        // The metadata of the core is JSON of UTF-8, which `json.dumps` escapes a lone surrogate
        // of: the error is that of the first str that UTF-8 cannot encode, as of a name.
        check_metadata_strs(d.as_any())?;
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            e.to_string(),
        ))
    })
}

/// The `UnicodeEncodeError` of the first str of `value`, of metadata, that UTF-8 cannot encode:
/// of the keys and the values of its dicts and the items of its lists and tuples, in their order.
fn check_metadata_strs(value: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(text) = value.cast::<pyo3::types::PyString>() {
        text.to_cow()?;
    } else if let Ok(dict) = value.cast::<pyo3::types::PyDict>() {
        for (key, item) in dict.iter() {
            check_metadata_strs(&key)?;
            check_metadata_strs(&item)?;
        }
    } else if value.is_instance_of::<pyo3::types::PyList>()
        || value.is_instance_of::<pyo3::types::PyTuple>()
    {
        for item in value.try_iter()? {
            check_metadata_strs(&item?)?;
        }
    }
    Ok(())
}

/// The keys of the JSON objects of the nested data types, in the order of PySpark's
/// `jsonValue`: struct, array, map, and the UDTs of a Scala class or of a pickled Python class.
const TYPE_JSON_KEYS: [&str; 11] = [
    "type",
    "fields",
    "elementType",
    "containsNull",
    "keyType",
    "valueType",
    "valueContainsNull",
    "class",
    "pyClass",
    "serializedClass",
    "sqlType",
];

/// The keys of the JSON object of a struct field, in the order of PySpark's `jsonValue`.
const FIELD_JSON_KEYS: [&str; 4] = ["name", "type", "nullable", "metadata"];

/// The JSON text of the JSON value `v` of a data type (of a struct field, if `field`), with
/// the keys of its objects in the order of PySpark's `jsonValue`. The core's JSON objects
/// sort their keys, as `json()` does; the metadata of a field stays in that order.
fn ordered_json_text(v: &serde_json::Value, field: bool) -> String {
    let Some(object) = v.as_object() else {
        return v.to_string();
    };
    let order: &[&str] = if field {
        &FIELD_JSON_KEYS
    } else {
        &TYPE_JSON_KEYS
    };
    let keys = order
        .iter()
        .copied()
        .filter(|key| object.contains_key(*key))
        .chain(
            object
                .keys()
                .map(String::as_str)
                .filter(|key| !order.contains(key)),
        );
    let entries: Vec<String> = keys
        .map(|key| {
            let value = &object[key];
            let text = match (field, key) {
                (true, "type") | (false, "elementType" | "keyType" | "valueType" | "sqlType") => {
                    ordered_json_text(value, false)
                }
                (false, "fields") => match value.as_array() {
                    Some(fields) => {
                        let fields: Vec<String> =
                            fields.iter().map(|f| ordered_json_text(f, true)).collect();
                        format!("[{}]", fields.join(","))
                    }
                    None => value.to_string(),
                },
                _ => value.to_string(),
            };
            format!("{}:{}", serde_json::Value::from(key), text)
        })
        .collect();
    format!("{{{}}}", entries.join(","))
}

/// The `jsonValue` of the JSON value `v` of a data type (of a struct field, if `field`): its
/// dicts have the keys in the order of PySpark's, as `json.dumps(t.jsonValue())` shows.
fn py_json_value<'py>(
    py: Python<'py>,
    v: &serde_json::Value,
    field: bool,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?
        .getattr("loads")?
        .call1((ordered_json_text(v, field),))
}

/// The `toInternal` or `fromInternal` (the `_to_internal` or `_from_internal` of the shim
/// `pyspark.sql.types`) of `data_type`, which convert values as those of PySpark's types do.
fn shim_conversion<'py>(
    data_type: &Bound<'py, PyAny>,
    conversion: &str,
    obj: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    data_type
        .py()
        .import("pyspark.sql.types")?
        .getattr(conversion)?
        .call1((data_type, obj))
}

/// `needConversion` of `data_type`, as PySpark's: that of a year-month interval is false with
/// `PYSPARK_YM_INTERVAL_LEGACY=1`, with which `df.collect` returns the internal integers.
fn need_conversion(py: Python<'_>, data_type: &DataType) -> PyResult<bool> {
    Ok(match data_type {
        DataType::YearMonthInterval { .. } => !py
            .import("os")?
            .getattr("environ")?
            .call_method1("get", ("PYSPARK_YM_INTERVAL_LEGACY",))?
            .eq("1")?,
        DataType::Array { element_type, .. } => need_conversion(py, element_type)?,
        DataType::Map {
            key_type,
            value_type,
            ..
        } => need_conversion(py, key_type)? || need_conversion(py, value_type)?,
        other => other.need_conversion(),
    })
}

/// Convert the core JSON-value metadata map back to a Python dict.
fn metadata_map_to_py<'py>(
    py: Python<'py>,
    m: &BTreeMap<String, serde_json::Value>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let s = serde_json::to_string(m).unwrap_or_else(|_| "{}".to_string());
    py.import("json")?
        .getattr("loads")?
        .call1((s,))?
        .cast_into::<pyo3::types::PyDict>()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("metadata is not a dict"))
}

#[pymethods]
impl PyDataType {
    /// As the `object.__new__` of PySpark's plain `DataType`: `DataType()` is a type of its class
    /// only (see `is_class_type`), and a Python subclass, such as a `UserDefinedType` of
    /// parameters, takes the arguments of its `__init__` (see `object_new_arguments`). Its state
    /// is its `__dict__`, as PySpark's, so it has no core type.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn __new__(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        object_new_arguments(cls, &cls.py().get_type::<Self>(), args, kwargs)?;
        Ok(PyDataType { inner: None })
    }

    /// A Python subclass of the type classes is of one branch of their hierarchy (see
    /// `check_one_branch`), which the class raises of when it is made, as CPython raises of bases
    /// of conflicting layouts. As `object.__init_subclass__` then, which takes no keyword
    /// arguments.
    #[classmethod]
    #[pyo3(signature = (**kwargs))]
    fn __init_subclass__(
        cls: &Bound<'_, pyo3::types::PyType>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        check_one_branch(cls)?;
        let py = cls.py();
        py.import("builtins")?
            .getattr("super")?
            .call1((py.get_type::<PyDataType>(), cls))?
            .call_method("__init_subclass__", (), kwargs)?;
        Ok(())
    }

    /// A type of its class only pickles as PySpark's plain class does, as its class and its
    /// `__dict__`, of `copyreg.__newobj__(cls)`, which both clients load at every protocol: this
    /// is what `object.__reduce_ex__` gives of PySpark's at the protocols 2 and up, whose pickle
    /// is PySpark's own. A class of a core type without a `__reduce__` of its own pickles as
    /// `_parse_datatype_json_string` of its JSON.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.borrow().inner.is_none() {
            let state = slf.getattr(pyo3::intern!(py, "__dict__"))?;
            let state = if state.is_truthy()? {
                state
            } else {
                py.None().into_bound(py)
            };
            let newobj = py.import("copyreg")?.getattr("__newobj__")?;
            return Ok((newobj, (slf.get_type(),), state, py.None(), py.None())
                .into_pyobject(py)?
                .into_any()
                .unbind());
        }
        let json: String = slf.call_method0("json")?.extract()?;
        Ok(type_reduce(py, &json)?
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, also of a pure-Python
    /// subclass, such as an `UnparsedDataType`, whose `jsonValue` raises.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of a type of its class only.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if slf.borrow().inner.is_none() {
            return slf.call_method0(pyo3::intern!(py, "typeName"));
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    /// As PySpark's `DataType.needConversion`, false of a type of its class only.
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        if slf.borrow().inner.is_none() {
            return Ok(false);
        }
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    /// As PySpark's `DataType.fromInternal`, of which a type of its class only returns `obj`.
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if slf.borrow().inner.is_none() {
            return Ok(obj);
        }
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    /// As PySpark's `DataType.toInternal`, of which a type of its class only returns `obj`.
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if slf.borrow().inner.is_none() {
            return Ok(obj);
        }
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// The type of the DDL string `ddl`, as PySpark's `DataType.fromDDL`, which the server of
    /// the active session parses (`_parse_datatype_string`): the concrete type, a StructType
    /// for a list of fields. The subclasses inherit this classmethod, so that any of them
    /// parses DDL the same way.
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __obj_from_ddl(cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<Py<PyAny>> {
        Ok(cls
            .py()
            .import("pyspark.sql.types")?
            .call_method1("_parse_datatype_string", (ddl,))?
            .unbind())
    }
    /// The name of the type, as PySpark's classmethod `DataType.typeName`: the name of the class
    /// without its `Type`, lowercased, such as `integer` of `IntegerType`, also of a subclass.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn __dt_type_name(cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<String> {
        let name = cls.name()?.to_string();
        let stem: String = name
            .chars()
            .take(name.chars().count().saturating_sub(4))
            .collect();
        Ok(stem.to_lowercase())
    }

    /// As pyspark's `DataType.__repr__`: the class name, for the base class and for
    /// pure-Python subclasses (such as a `UserDefinedType`) that do not define their own;
    /// the concrete types override it. There is no `__str__`, so `str()` is `repr()`, as
    /// in pyspark.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        class_repr(slf.as_any())
    }

    /// As pyspark's `simpleString()`, the `typeName()` of a type of its class only. An array, map
    /// or struct of a user-defined type names it by the `simpleString()` of its Python class,
    /// such as `array<vector>`.
    fn simpleString(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        // A struct, array or map reads its live type, as its own `simpleString` does: the base
        // method is also called on them directly, as `DataType.simpleString(st)`.
        if slf.is_instance_of::<PyStructType>()
            || slf.is_instance_of::<PyArrayType>()
            || slf.is_instance_of::<PyMapType>()
        {
            return simple_string_with_udts(py, &py_to_data_type(slf.as_any())?);
        }
        let this = slf.borrow();
        let Some(inner) = &this.inner else {
            drop(this);
            return slf.call_method0(pyo3::intern!(py, "typeName"))?.extract();
        };
        match inner {
            DataType::Array { .. } | DataType::Map { .. } | DataType::Struct { .. }
                if has_udt(inner) =>
            {
                simple_string_with_udts(py, inner)
            }
            _ => Ok(inner.simple_string()),
        }
    }

    // Equality/hashing are defined once here and inherited by every concrete type class
    // (and by pure-Python subclasses such as `UserDefinedType`). Mirrors pyspark's
    // `DataType.__eq__`/`__ne__`/`__hash__`.
    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        data_type_eq(slf.as_any(), other)
    }

    fn __ne__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(!data_type_eq(slf.as_any(), other)?)
    }

    /// pyspark hashes `str(self)`. Hash the simple string of the same live core type that
    /// `__eq__` compares, so equal types always hash equal (`str()` can lag behind the
    /// live fields of a mutated `StructType`). A `UserDefinedType` is compared by class
    /// and `__dict__` rather than by core type, so it keeps pyspark's `hash(str(self))`.
    /// As PySpark's `hash(str(self))`: of the repr, which keeps a flag as it was given, so that
    /// `ArrayType(IntegerType(), 1)` and `ArrayType(IntegerType(), True)` hash differently, as in
    /// PySpark, though they are equal.
    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<isize> {
        slf.as_any().str()?.hash()
    }
}

/// Whether `obj` is an instance of the pure-Python `pyspark.sql.types.UserDefinedType`.
fn is_udt(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    match obj
        .py()
        .import("pyspark.sql.types")
        .and_then(|m| m.getattr("UserDefinedType"))
    {
        Ok(cls) => obj.is_instance(&cls),
        Err(_) => Ok(false),
    }
}

/// Whether `obj` is an instance of the pure-Python `UnparsedDataType` of
/// `pyspark.sql.connect.types`, a DDL string of a type that the server parses.
fn is_unparsed(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    match obj
        .py()
        .import("pyspark.sql.connect.types")
        .and_then(|m| m.getattr("UnparsedDataType"))
    {
        Ok(cls) => obj.is_instance(&cls),
        Err(_) => Ok(false),
    }
}

/// Whether the type object `obj` keeps its state in its `__dict__`, as PySpark's plain classes
/// do: a `DataType` of no core type (see `PyDataType::inner`).
fn has_state_of_dict(obj: &Bound<'_, PyAny>) -> bool {
    obj.cast::<PyDataType>()
        .is_ok_and(|data_type| data_type.borrow().inner.is_none())
}

/// Whether PySpark's type of the core type `data_type` has attributes in its `__dict__`: its
/// parameters, which the drop-in's type keeps in Rust. The types without parameters have none.
fn has_pyspark_attributes(data_type: &DataType) -> bool {
    !matches!(
        data_type,
        DataType::Null
            | DataType::Boolean
            | DataType::Byte
            | DataType::Short
            | DataType::Integer
            | DataType::Long
            | DataType::Float
            | DataType::Double
            | DataType::Binary
            | DataType::Date
            | DataType::Timestamp
            | DataType::TimestampNtz
            | DataType::CalendarInterval
            | DataType::Variant
    )
}

/// `DataType.__eq__` with pyspark semantics. pyspark compares
/// `isinstance(other, self.__class__) and self.__dict__ == other.__dict__`; the Rust-backed
/// classes keep their state in Rust rather than `__dict__`, so compare the core `DataType`
/// built from each side's live fields instead (a mutated `StructType` compares by its
/// current fields). A type whose state is its `__dict__` keeps PySpark's formula verbatim:
/// the base `DataType`, an abstract class such as `AtomicType`, and a Python subclass, such as
/// a `UserDefinedType`. Of such a `self` and a type of a core type, PySpark's `__dict__` of the
/// latter holds its parameters. `==` calls the `__eq__` of a subclass first, as Python does, so
/// that `DataType() == AtomicType()` is `AtomicType().__eq__(DataType())`, false. So it is for a
/// Python subclass of a concrete class, whose type is equal only to a type of its class or of a
/// subclass of it, of the same parameters and of the same attributes of its `__dict__`, which
/// holds the attributes that a user set: `StringType() == MyString()` is
/// `MyString().__eq__(StringType())`, false in both orders.
fn data_type_eq(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !b.is_instance_of::<PyDataType>() {
        return Ok(false);
    }
    if has_state_of_dict(a) {
        if !b.is_instance(&a.get_type())? {
            return Ok(false);
        }
        if !has_state_of_dict(b)
            && (b.is_instance_of::<PyStructField>() || has_pyspark_attributes(&py_to_data_type(b)?))
        {
            return Ok(false);
        }
        let dict = pyo3::intern!(a.py(), "__dict__");
        return a.getattr(dict)?.eq(b.getattr(dict)?);
    }
    // A type of no core type is not of the class of `a`, nor of a subclass of it.
    if has_state_of_dict(b) || !b.is_instance(&a.get_type())? {
        return Ok(false);
    }
    // A struct also compares its `names`, which PySpark's `__dict__` comparison does too.
    if let (Ok(sa), Ok(sb)) = (a.cast::<PyStructType>(), b.cast::<PyStructType>()) {
        if !sa
            .borrow()
            .names
            .bind(a.py())
            .eq(sb.borrow().names.bind(b.py()))?
        {
            return Ok(false);
        }
    }
    // The metadata of the fields are compared as Python's `==` compares the dicts of PySpark.
    let core = py_to_data_type(a)?;
    if !core.eq_as_python(&py_to_data_type(b)?) {
        return Ok(false);
    }
    // The `hasPrecisionInfo` of a decimal is an attribute of PySpark's `__dict__` too.
    if let (Ok(da), Ok(db)) = (a.cast::<PyDecimalType>(), b.cast::<PyDecimalType>()) {
        let py = a.py();
        let ia = da.borrow().has_precision_info.clone_ref(py);
        let ib = db.borrow().has_precision_info.clone_ref(py);
        if !ia.bind(py).eq(ib.bind(py))? {
            return Ok(false);
        }
    }
    // The flags as they were given are compared as PySpark's `__dict__` does: `1` equals `True`,
    // and so are the metadata dicts of the fields, as Python compares them (`(1, 2)` is not
    // `[1, 2]`, nor `{1: "a"}` `{"1": "a"}`), which the JSON values of the types hold.
    if has_other_flag(a.py(), a)?
        || has_other_flag(b.py(), b)?
        || holds_metadata(a.py(), a)?
        || holds_metadata(b.py(), b)?
    {
        let (ja, _) = type_json_value(a.py(), a)?;
        let (jb, _) = type_json_value(b.py(), b)?;
        if !ja.eq(jb)? {
            return Ok(false);
        }
    }
    // The types that the client keeps as their objects, whose core types are their JSON, compare
    // by those objects, as PySpark's `__dict__` holds them: `ArrayType(t)` is not equal to
    // `ArrayType(u)` of a `u` of another `__dict__` or class, of the same JSON. So do the types
    // of Python subclasses, of the core types of their classes, and the types that keep them.
    let py = a.py();
    if (holds_objects(&core) || keeps_subclass_objects(py, a)? || keeps_subclass_objects(py, b)?)
        && !held_objects_eq(a, b, &core)?
    {
        return Ok(false);
    }
    // The attributes that a user set on a type of a Python subclass, of its `__dict__`, which
    // PySpark's `__dict__` holds after the parameters.
    if is_subclass_object(a) || is_subclass_object(b) {
        let dict = pyo3::intern!(py, "__dict__");
        return a.getattr(dict)?.eq(b.getattr(dict)?);
    }
    Ok(true)
}

// Define concrete type classes
#[pyclass(subclass, name = "NullType", module = "pyspark.sql.types", extends = PyDataType)]
pub struct PyNullType;

#[pymethods]
impl PyNullType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"void\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyNullType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(DataType::Null, PyNullType, [])
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("NullType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "void").clone().into_any())
    }
    /// As PySpark's, which names this type `void`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "void"
    }
}

#[pyclass(subclass, name = "BooleanType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyBooleanType;

#[pymethods]
impl PyBooleanType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"boolean\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyBooleanType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(DataType::Boolean, PyBooleanType, [PyAtomicType])
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("BooleanType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "boolean").clone().into_any())
    }
}

#[pyclass(subclass, name = "ByteType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyByteType;

#[pymethods]
impl PyByteType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"byte\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyByteType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(
                DataType::Byte,
                PyByteType,
                [PyAtomicType, PyNumericType, PyIntegralType]
            )
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("ByteType()".to_string())
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "tinyint"
    }
}

#[pyclass(subclass, name = "ShortType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyShortType;

#[pymethods]
impl PyShortType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"short\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyShortType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(
                DataType::Short,
                PyShortType,
                [PyAtomicType, PyNumericType, PyIntegralType]
            )
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("ShortType()".to_string())
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "smallint"
    }
}

#[pyclass(subclass, name = "IntegerType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyIntegerType;

#[pymethods]
impl PyIntegerType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"integer\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyIntegerType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(
                DataType::Integer,
                PyIntegerType,
                [PyAtomicType, PyNumericType, PyIntegralType]
            )
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("IntegerType()".to_string())
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "int"
    }
}

#[pyclass(subclass, name = "LongType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyLongType;

#[pymethods]
impl PyLongType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"long\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyLongType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(
                DataType::Long,
                PyLongType,
                [PyAtomicType, PyNumericType, PyIntegralType]
            )
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("LongType()".to_string())
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "bigint"
    }
}

#[pyclass(subclass, name = "FloatType", module = "pyspark.sql.types", extends = PyFractionalType)]
pub struct PyFloatType;

#[pymethods]
impl PyFloatType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"float\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyFloatType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(
                DataType::Float,
                PyFloatType,
                [PyAtomicType, PyNumericType, PyFractionalType]
            )
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("FloatType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "float").clone().into_any())
    }
}

#[pyclass(subclass, name = "DoubleType", module = "pyspark.sql.types", extends = PyFractionalType)]
pub struct PyDoubleType;

#[pymethods]
impl PyDoubleType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"double\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyDoubleType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(
                DataType::Double,
                PyDoubleType,
                [PyAtomicType, PyNumericType, PyFractionalType]
            )
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("DoubleType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "double").clone().into_any())
    }
}

#[pyclass(subclass, name = "DecimalType", module = "pyspark.sql.types", extends = PyFractionalType)]
pub struct PyDecimalType {
    pub precision: i32,
    pub scale: i32,
    /// `hasPrecisionInfo` as it was set: PySpark's `__init__` sets it true, and it is a plain
    /// attribute, of any value, which `==` compares as PySpark's `__dict__` does.
    has_precision_info: Py<PyAny>,
}

impl PyDecimalType {
    /// The parts of a `DecimalType(precision, scale)`, as PySpark's `__init__` sets them.
    fn of(py: Python<'_>, precision: i32, scale: i32) -> Self {
        PyDecimalType {
            precision,
            scale,
            has_precision_info: flag_of_bool(py, true),
        }
    }
}

#[pymethods]
impl PyDecimalType {
    /// As PySpark's `DecimalType.hasPrecisionInfo`, a public attribute that `__init__` sets true,
    /// and that may be set, as a plain attribute is. It is not of the core type, nor its JSON.
    #[getter(hasPrecisionInfo)]
    fn has_precision_info(&self, py: Python<'_>) -> Py<PyAny> {
        self.has_precision_info.clone_ref(py)
    }
    #[setter(hasPrecisionInfo)]
    fn set_has_precision_info(&mut self, value: Py<PyAny>) {
        self.has_precision_info = value;
    }
    /// The type of the state that PySpark pickled, of `precision`, `scale` and
    /// `hasPrecisionInfo`, which pickle sets as attributes (see `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let this = slf.borrow();
        let dt = DataType::Decimal {
            precision: this.precision,
            scale: this.scale,
        };
        reduce_of(slf, &dt.json(), || {
            Ok(vec![
                ("precision", this.precision.into_pyobject(py)?.into_any()),
                ("scale", this.scale.into_pyobject(py)?.into_any()),
                ("hasPrecisionInfo", this.has_precision_info.bind(py).clone()),
            ])
        })
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `DecimalType.__new__(DecimalType)`, of which pickle's `copyreg.__newobj__` makes the type
    /// that PySpark pickled, is a `DecimalType(10, 0)` until `__setstate__` sets it. The class
    /// shows the signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(precision=10, scale=0)"
    )]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let py = cls.py();
        let (precision, scale) = (10, 0);
        Ok(init_chain!(
            DataType::Decimal { precision, scale },
            PyDecimalType::of(py, precision, scale),
            [PyAtomicType, PyNumericType, PyFractionalType],
            value
        ))
    }
    /// As PySpark's `__init__(self, precision=10, scale=0)`, which sets `hasPrecisionInfo` true.
    #[pyo3(signature = (precision=10, scale=0))]
    fn __init__(slf: &Bound<'_, Self>, precision: i32, scale: i32) -> PyResult<()> {
        let py = slf.py();
        let parts = PyDecimalType::of(py, precision, scale);
        {
            let mut this = slf.borrow_mut();
            this.precision = parts.precision;
            this.scale = parts.scale;
            this.has_precision_info = parts.has_precision_info;
        }
        sync_base(slf.as_any())
    }
    // The parameters are plain attributes, as PySpark's: an assignment of one makes the type of
    // it, as its repr, JSON, `==` and hash show.
    #[getter]
    fn precision(&self) -> i32 {
        self.precision
    }
    #[setter]
    fn set_precision(slf: &Bound<'_, Self>, precision: i32) -> PyResult<()> {
        slf.borrow_mut().precision = precision;
        sync_base(slf.as_any())
    }
    #[getter]
    fn scale(&self) -> i32 {
        self.scale
    }
    #[setter]
    fn set_scale(slf: &Bound<'_, Self>, scale: i32) -> PyResult<()> {
        slf.borrow_mut().scale = scale;
        sync_base(slf.as_any())
    }
    fn __repr__(&self) -> String {
        format!("DecimalType({},{})", self.precision, self.scale)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("decimal({},{})", self.precision, self.scale)
    }
}

#[pyclass(subclass, name = "StringType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyStringType {
    pub collation: String,
}

impl PyStringType {
    /// PySpark's `self.isUTF8BinaryCollation()`, which its `simpleString` and `__repr__` call: of a
    /// Python subclass, its own.
    fn of_binary_collation(slf: &Bound<'_, Self>) -> PyResult<bool> {
        if !is_native(slf) {
            return slf
                .call_method0(pyo3::intern!(slf.py(), "isUTF8BinaryCollation"))?
                .is_truthy();
        }
        Ok(slf.borrow().collation == "UTF8_BINARY")
    }
}

#[pymethods]
impl PyStringType {
    /// The collation of the strings, as PySpark's `StringType.collation`: a plain attribute, of
    /// which an assignment makes the type of the collation.
    #[getter]
    fn collation(&self) -> String {
        self.collation.clone()
    }
    #[setter]
    fn set_collation(slf: &Bound<'_, Self>, collation: String) -> PyResult<()> {
        slf.borrow_mut().collation = collation;
        sync_base(slf.as_any())
    }
    /// The type of the state that PySpark pickled, of `collation` (see `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    // Collation providers, mirroring `StringType.providerSpark/providerICU/providers`.
    #[classattr]
    #[allow(non_upper_case_globals)]
    const providerSpark: &'static str = "spark";
    #[classattr]
    #[allow(non_upper_case_globals)]
    const providerICU: &'static str = "icu";
    #[classattr]
    fn providers() -> Vec<&'static str> {
        vec!["spark", "icu"]
    }

    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `StringType.jsonValue`, the `simpleString()` of the type, of a Python
    /// subclass its own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return slf.call_method0(pyo3::intern!(py, "simpleString"));
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let collation = slf.borrow().collation.clone();
        let dt = DataType::String {
            collation: collation.clone(),
        };
        reduce_of(slf, &dt.json(), || {
            Ok(vec![("collation", collation.into_pyobject(py)?.into_any())])
        })
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `StringType.__new__(StringType)`, of which pickle's `copyreg.__newobj__` makes the type that
    /// PySpark pickled, is a `StringType()` until `__setstate__` sets it. The class shows the
    /// signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(collation='UTF8_BINARY')"
    )]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let collation = "UTF8_BINARY".to_string();
        Ok(init_chain!(
            DataType::String {
                collation: collation.clone()
            },
            PyStringType { collation },
            [PyAtomicType],
            value
        ))
    }
    /// As PySpark's `__init__(self, collation="UTF8_BINARY")`.
    #[pyo3(signature = (collation="UTF8_BINARY".to_string()))]
    fn __init__(slf: &Bound<'_, Self>, collation: String) -> PyResult<()> {
        Self::set_collation(slf, collation)
    }
    /// As PySpark's, of `isUTF8BinaryCollation()`, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let collation = slf.borrow().collation.clone();
        Ok(if Self::of_binary_collation(slf)? {
            "StringType()".to_string()
        } else {
            format!("StringType('{collation}')")
        })
    }
    /// As PySpark's, of `isUTF8BinaryCollation()`, of a Python subclass its own.
    #[pyo3(name = "simpleString")]
    fn simple_string(slf: &Bound<'_, Self>) -> PyResult<String> {
        let collation = slf.borrow().collation.clone();
        Ok(if Self::of_binary_collation(slf)? {
            "string".to_string()
        } else {
            format!("string collate {collation}")
        })
    }

    /// Provider ("spark" for UTF8*, else "icu") for a collation name.
    /// Mirrors `StringType.collationProvider`.
    #[classmethod]
    #[pyo3(name = "collationProvider")]
    fn collation_provider(_cls: &Bound<'_, pyo3::types::PyType>, collationName: &str) -> String {
        if collationName.starts_with("UTF8") {
            "spark".to_string()
        } else {
            "icu".to_string()
        }
    }

    /// Whether this is the default UTF8_BINARY collation. Mirrors `StringType.isUTF8BinaryCollation`.
    #[pyo3(name = "isUTF8BinaryCollation")]
    fn is_utf8_binary_collation(&self) -> bool {
        self.collation == "UTF8_BINARY"
    }
}

#[pyclass(subclass, name = "BinaryType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyBinaryType;

#[pymethods]
impl PyBinaryType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"binary\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyBinaryType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(DataType::Binary, PyBinaryType, [PyAtomicType])
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("BinaryType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "binary").clone().into_any())
    }
}

#[pyclass(subclass, name = "DateType", module = "pyspark.sql.types", extends = PyDatetimeType)]
pub struct PyDateType;

#[pymethods]
impl PyDateType {
    /// Number of days from 0001-01-01 to 1970-01-01 (mirrors pyspark's DateType.EPOCH_ORDINAL).
    #[classattr]
    const EPOCH_ORDINAL: i32 = 719163;
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        v: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", v)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        d: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", d)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"date\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyDateType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(DataType::Date, PyDateType, [PyAtomicType, PyDatetimeType])
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("DateType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "date").clone().into_any())
    }
}

#[pyclass(subclass, name = "TimestampType", module = "pyspark.sql.types", extends = PyDatetimeType)]
pub struct PyTimestampType;

#[pymethods]
impl PyTimestampType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        ts: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", ts)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        dt: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", dt)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"timestamp\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyTimestampType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(
                DataType::Timestamp,
                PyTimestampType,
                [PyAtomicType, PyDatetimeType]
            )
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("TimestampType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "timestamp").clone().into_any())
    }
}

#[pyclass(
    subclass,
    name = "TimestampNTZType",
    module = "pyspark.sql.types",
    extends = PyDatetimeType
)]
pub struct PyTimestampNTZType;

#[pymethods]
impl PyTimestampNTZType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        ts: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", ts)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        dt: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", dt)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"timestamp_ntz\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyTimestampNTZType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(
                DataType::TimestampNtz,
                PyTimestampNTZType,
                [PyAtomicType, PyDatetimeType]
            )
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("TimestampNTZType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "timestamp_ntz").clone().into_any())
    }
    /// As PySpark's, which names this type `timestamp_ntz`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "timestamp_ntz"
    }
}

#[pyclass(subclass, name = "ArrayType", module = "pyspark.sql.types", extends = PyDataType)]
pub struct PyArrayType {
    pub element_type: DataType,
    pub contains_null: bool,
    element_object: KeptType,
    /// The `containsNull` as it was given (mirrors PySpark, which keeps any value).
    contains_null_object: Py<PyAny>,
}

#[pymethods]
impl PyArrayType {
    /// `ArrayType.elementType` attribute (the element DataType).
    #[getter]
    fn elementType(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.element_object.object(py, &self.element_type)
    }
    /// `ArrayType.containsNull` attribute, as it was given.
    #[getter]
    fn containsNull(&self, py: Python<'_>) -> Py<PyAny> {
        self.contains_null_object.clone_ref(py)
    }
    #[setter]
    fn set_containsNull(&mut self, py: Python<'_>, containsNull: Py<PyAny>) -> PyResult<()> {
        self.contains_null = containsNull.bind(py).is_truthy()?;
        self.contains_null_object = containsNull;
        Ok(())
    }
    #[setter]
    fn set_elementType(&mut self, elementType: &Bound<'_, PyAny>) -> PyResult<()> {
        self.element_type = py_to_data_type(elementType)?;
        self.element_object = KeptType::of(elementType);
        Ok(())
    }

    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        type_json_text(slf.as_any())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(type_json_value(py, slf.as_any())?.0)
    }
    /// As PySpark's, of the `needConversion()` of the types that it holds (see
    /// `need_conversion_of_object`).
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion_of_object(slf.py(), slf.as_any())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, or of the types it is made of (see `reduce_with_udts`), and of a Python
    /// subclass, of its class and its `__dict__` (see `subclass_reduce`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if !is_native(slf) {
            let (element_type, contains_null) = {
                let this = slf.borrow();
                (
                    this.element_object.object(py, &this.element_type)?,
                    this.contains_null_object.clone_ref(py),
                )
            };
            return subclass_reduce(
                slf.as_any(),
                vec![
                    ("elementType", element_type.into_bound(py)),
                    ("containsNull", contains_null.into_bound(py)),
                ],
            );
        }
        let this = slf.borrow();
        let dt = DataType::Array {
            element_type: Box::new(this.element_object.live(py, &this.element_type)?),
            contains_null: this.contains_null,
        };
        let flagged = has_other_flag(slf.py(), slf.as_any())?;
        let (f, args) = reduce_with_udts(slf.as_any(), &dt, flagged, || {
            let element_type = this.element_object.object(py, &this.element_type)?;
            (element_type, this.contains_null_object.clone_ref(py)).into_pyobject(py)
        })?;
        Ok((f, args).into_pyobject(py)?.into_any().unbind())
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `ArrayType.__new__(ArrayType)`, of which pickle's `copyreg.__newobj__` makes the type that
    /// PySpark pickled, is an array of `NullType()` until `__setstate__` sets its attributes. The
    /// class shows the signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(elementType, containsNull=True)"
    )]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let py = cls.py();
        Ok(init_chain!(
            DataType::Array {
                element_type: Box::new(DataType::Null),
                contains_null: true,
            },
            PyArrayType {
                element_type: DataType::Null,
                contains_null: true,
                element_object: KeptType::default(),
                contains_null_object: flag_of_bool(py, true),
            },
            [],
            value
        ))
    }
    /// As PySpark's `__init__(self, elementType, containsNull=True)`, which asserts that the
    /// element type is a `DataType`.
    #[pyo3(signature = (elementType, containsNull=default_flag()))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        elementType: &Bound<'_, PyAny>,
        containsNull: Py<PyAny>,
    ) -> PyResult<()> {
        let et = nested_data_type(elementType, "elementType")?;
        let contains_null = containsNull.bind(slf.py()).is_truthy()?;
        let mut this = slf.borrow_mut();
        this.as_super().inner = Some(DataType::Array {
            element_type: Box::new(et.clone()),
            contains_null,
        });
        this.element_type = et;
        this.contains_null = contains_null;
        this.element_object = KeptType::of(elementType);
        this.contains_null_object = containsNull;
        Ok(())
    }
    /// The type of the state that PySpark pickled, of `elementType` and `containsNull`, which
    /// pickle sets as attributes (see `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Mirror pyspark: ArrayType(<elementType repr>, <containsNull>).
        let el = self.element_object.object(py, &self.element_type)?;
        let el_repr: String = el.bind(py).repr()?.extract()?;
        let contains_null: String = self.contains_null_object.bind(py).str()?.extract()?;
        Ok(format!("ArrayType({}, {})", el_repr, contains_null))
    }
    /// As PySpark's, of the `simpleString()` of the element type object (see
    /// `child_simple_string`).
    #[pyo3(name = "simpleString")]
    fn simple_string(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "array<{}>",
            child_simple_string(py, &self.element_object, &self.element_type)?
        ))
    }

    /// A copy with all nullability fields set true (recursively). Mirrors `ArrayType.toNullable()`.
    #[pyo3(name = "toNullable")]
    fn __at_to_nullable(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let this = slf.borrow();
        let element_type = this.element_object.live(py, &this.element_type)?;
        // Of the type objects that it holds, as PySpark's `_as_nullable`, when one is a type that
        // the client keeps as its object, which its JSON does not tell apart, or a type of a
        // Python subclass, which it keeps.
        if holds_objects(&element_type) || kept_holds_subclass_object(py, &this.element_object)? {
            drop(this);
            return Ok(as_nullable(slf.as_any())?.unbind());
        }
        let dt = (DataType::Array {
            element_type: Box::new(element_type),
            contains_null: this.contains_null,
        })
        .to_nullable();
        data_type_to_py(py, &dt)
    }

    /// Build an ArrayType from its JSON value. Mirrors `ArrayType.fromJson`.
    #[classmethod]
    #[pyo3(name = "fromJson")]
    #[pyo3(
        signature = (json, fieldPath="".to_string(), collationsMap=None),
        text_signature = "($cls, json, fieldPath='', collationsMap=None)"
    )]
    fn __at_from_json(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        json: &Bound<'_, PyAny>,
        fieldPath: String,
        collationsMap: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(shim_from_json(py, "_array_type_from_json")?
            .call1((json, fieldPath, collationsMap))?
            .unbind())
    }
}

#[pyclass(subclass, name = "MapType", module = "pyspark.sql.types", extends = PyDataType)]
pub struct PyMapType {
    pub key_type: DataType,
    pub value_type: DataType,
    pub value_contains_null: bool,
    key_object: KeptType,
    value_object: KeptType,
    /// The `valueContainsNull` as it was given (mirrors PySpark, which keeps any value).
    value_contains_null_object: Py<PyAny>,
}

#[pymethods]
impl PyMapType {
    /// `MapType.keyType` attribute.
    #[getter]
    fn keyType(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.key_object.object(py, &self.key_type)
    }
    /// `MapType.valueType` attribute.
    #[getter]
    fn valueType(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.value_object.object(py, &self.value_type)
    }
    /// `MapType.valueContainsNull` attribute, as it was given.
    #[getter]
    fn valueContainsNull(&self, py: Python<'_>) -> Py<PyAny> {
        self.value_contains_null_object.clone_ref(py)
    }
    #[setter]
    fn set_keyType(&mut self, keyType: &Bound<'_, PyAny>) -> PyResult<()> {
        self.key_type = py_to_data_type(keyType)?;
        self.key_object = KeptType::of(keyType);
        Ok(())
    }
    #[setter]
    fn set_valueType(&mut self, valueType: &Bound<'_, PyAny>) -> PyResult<()> {
        self.value_type = py_to_data_type(valueType)?;
        self.value_object = KeptType::of(valueType);
        Ok(())
    }
    #[setter]
    fn set_valueContainsNull(
        &mut self,
        py: Python<'_>,
        valueContainsNull: Py<PyAny>,
    ) -> PyResult<()> {
        self.value_contains_null = valueContainsNull.bind(py).is_truthy()?;
        self.value_contains_null_object = valueContainsNull;
        Ok(())
    }

    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        type_json_text(slf.as_any())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(type_json_value(py, slf.as_any())?.0)
    }
    /// As PySpark's, of the `needConversion()` of the types that it holds (see
    /// `need_conversion_of_object`).
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion_of_object(slf.py(), slf.as_any())
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, or of the types it is made of (see `reduce_with_udts`), and of a Python
    /// subclass, of its class and its `__dict__` (see `subclass_reduce`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if !is_native(slf) {
            let (key_type, value_type, value_contains_null) = {
                let this = slf.borrow();
                (
                    this.key_object.object(py, &this.key_type)?,
                    this.value_object.object(py, &this.value_type)?,
                    this.value_contains_null_object.clone_ref(py),
                )
            };
            return subclass_reduce(
                slf.as_any(),
                vec![
                    ("keyType", key_type.into_bound(py)),
                    ("valueType", value_type.into_bound(py)),
                    ("valueContainsNull", value_contains_null.into_bound(py)),
                ],
            );
        }
        let this = slf.borrow();
        let dt = DataType::Map {
            key_type: Box::new(this.key_object.live(py, &this.key_type)?),
            value_type: Box::new(this.value_object.live(py, &this.value_type)?),
            value_contains_null: this.value_contains_null,
        };
        let flagged = has_other_flag(slf.py(), slf.as_any())?;
        let (f, args) = reduce_with_udts(slf.as_any(), &dt, flagged, || {
            let key_type = this.key_object.object(py, &this.key_type)?;
            let value_type = this.value_object.object(py, &this.value_type)?;
            (
                key_type,
                value_type,
                this.value_contains_null_object.clone_ref(py),
            )
                .into_pyobject(py)
        })?;
        Ok((f, args).into_pyobject(py)?.into_any().unbind())
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `MapType.__new__(MapType)`, of which pickle's `copyreg.__newobj__` makes the type that
    /// PySpark pickled, is a map of `NullType()` keys and values until `__setstate__` sets its
    /// attributes. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(keyType, valueType, valueContainsNull=True)"
    )]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let py = cls.py();
        Ok(init_chain!(
            DataType::Map {
                key_type: Box::new(DataType::Null),
                value_type: Box::new(DataType::Null),
                value_contains_null: true,
            },
            PyMapType {
                key_type: DataType::Null,
                value_type: DataType::Null,
                value_contains_null: true,
                key_object: KeptType::default(),
                value_object: KeptType::default(),
                value_contains_null_object: flag_of_bool(py, true),
            },
            [],
            value
        ))
    }
    /// As PySpark's `__init__(self, keyType, valueType, valueContainsNull=True)`, which asserts
    /// that the key and the value types are `DataType`s.
    #[pyo3(signature = (keyType, valueType, valueContainsNull=default_flag()))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        keyType: &Bound<'_, PyAny>,
        valueType: &Bound<'_, PyAny>,
        valueContainsNull: Py<PyAny>,
    ) -> PyResult<()> {
        let kt = nested_data_type(keyType, "keyType")?;
        let vt = nested_data_type(valueType, "valueType")?;
        let value_contains_null = valueContainsNull.bind(slf.py()).is_truthy()?;
        let mut this = slf.borrow_mut();
        this.as_super().inner = Some(DataType::Map {
            key_type: Box::new(kt.clone()),
            value_type: Box::new(vt.clone()),
            value_contains_null,
        });
        this.key_type = kt;
        this.value_type = vt;
        this.value_contains_null = value_contains_null;
        this.key_object = KeptType::of(keyType);
        this.value_object = KeptType::of(valueType);
        this.value_contains_null_object = valueContainsNull;
        Ok(())
    }
    /// The type of the state that PySpark pickled, of `keyType`, `valueType` and
    /// `valueContainsNull`, which pickle sets as attributes (see `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Mirror pyspark: MapType(<keyType repr>, <valueType repr>, <valueContainsNull>).
        let k: String = self
            .key_object
            .object(py, &self.key_type)?
            .bind(py)
            .repr()?
            .extract()?;
        let v: String = self
            .value_object
            .object(py, &self.value_type)?
            .bind(py)
            .repr()?
            .extract()?;
        let value_contains_null: String =
            self.value_contains_null_object.bind(py).str()?.extract()?;
        Ok(format!("MapType({}, {}, {})", k, v, value_contains_null))
    }
    /// As PySpark's, of the `simpleString()` of the key and value type objects (see
    /// `child_simple_string`).
    #[pyo3(name = "simpleString")]
    fn simple_string(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "map<{},{}>",
            child_simple_string(py, &self.key_object, &self.key_type)?,
            child_simple_string(py, &self.value_object, &self.value_type)?
        ))
    }

    /// A copy with all nullability fields set true (recursively). Mirrors `MapType.toNullable()`.
    #[pyo3(name = "toNullable")]
    fn __mt_to_nullable(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let this = slf.borrow();
        let key_type = this.key_object.live(py, &this.key_type)?;
        let value_type = this.value_object.live(py, &this.value_type)?;
        // Of the type objects that it holds, as PySpark's `_as_nullable` (see `ArrayType`'s).
        if holds_objects(&key_type)
            || holds_objects(&value_type)
            || kept_holds_subclass_object(py, &this.key_object)?
            || kept_holds_subclass_object(py, &this.value_object)?
        {
            drop(this);
            return Ok(as_nullable(slf.as_any())?.unbind());
        }
        let dt = (DataType::Map {
            key_type: Box::new(key_type),
            value_type: Box::new(value_type),
            value_contains_null: this.value_contains_null,
        })
        .to_nullable();
        data_type_to_py(py, &dt)
    }

    /// Build a MapType from its JSON value. Mirrors `MapType.fromJson`.
    #[classmethod]
    #[pyo3(name = "fromJson")]
    #[pyo3(
        signature = (json, fieldPath="".to_string(), collationsMap=None),
        text_signature = "($cls, json, fieldPath='', collationsMap=None)"
    )]
    fn __mt_from_json(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        json: &Bound<'_, PyAny>,
        fieldPath: String,
        collationsMap: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(shim_from_json(py, "_map_type_from_json")?
            .call1((json, fieldPath, collationsMap))?
            .unbind())
    }
}

#[pyclass(subclass, name = "StructField", module = "pyspark.sql.types", extends = PyDataType)]
pub struct PyStructField {
    /// The data type object, which is the field's `dataType` once it is kept (see `KeptType`).
    data_type_object: KeptType,
    pub(crate) field: StructField,
    /// The name as it was given, when it is not a str (PySpark's setter takes any value); the
    /// name of `field` is its `str`.
    name_object: Option<Py<PyAny>>,
    /// The nullable flag as it was given, which PySpark's `StructField` keeps as it is (any
    /// value, not only a bool). `field.nullable` is its truth.
    nullable: Py<PyAny>,
    /// The metadata dict, which PySpark's `StructField` keeps as it is: an edit of it in place
    /// is the metadata of the field. `field.metadata` is only its snapshot at the time it was
    /// made, so the metadata is read through `current_field`.
    metadata: Py<pyo3::types::PyDict>,
    /// The `needConversion` of the type, kept while it cannot change: that of an atomic type does
    /// not change with its parameters, which may be set, so `add` need not read it again for each
    /// field. Reset by a set `dataType`.
    need_conversion_of_type: std::sync::OnceLock<bool>,
}

impl PyStructField {
    /// A field of a core field, with a metadata dict of its metadata.
    pub(crate) fn from_field(py: Python<'_>, field: StructField) -> PyResult<Self> {
        let metadata = metadata_map_to_py(py, &field.metadata)?.unbind();
        Ok(PyStructField {
            nullable: flag_of_bool(py, field.nullable),
            name_object: None,
            field,
            metadata,
            data_type_object: KeptType::default(),
            need_conversion_of_type: std::sync::OnceLock::new(),
        })
    }

    /// A field of the given parts, as PySpark's `StructField(name, dataType, nullable, metadata)`:
    /// a non-empty metadata dict is kept as it is, so that an edit of it shows in the field; an
    /// empty or no dict gives a new one (`metadata or {}`).
    fn build(
        py: Python<'_>,
        field: StructField,
        metadata: Option<&Bound<'_, pyo3::types::PyDict>>,
        nullable: Py<PyAny>,
        data_type_object: KeptType,
    ) -> PyResult<Self> {
        let metadata = match metadata {
            Some(d) if !d.is_empty() => d.clone().unbind(),
            _ => pyo3::types::PyDict::new(py).unbind(),
        };
        Ok(PyStructField {
            field,
            name_object: None,
            nullable,
            metadata,
            data_type_object,
            need_conversion_of_type: std::sync::OnceLock::new(),
        })
    }

    /// The `needConversion` of the type as it is now (see `need_conversion_of_type`).
    fn need_conversion_now(&self, py: Python<'_>) -> PyResult<bool> {
        if let Some(&need) = self.need_conversion_of_type.get() {
            return Ok(need);
        }
        // The type is read from its object when the field has one, without reading the fields
        // of a nested struct, or else from the snapshot. The value is kept only for a type whose
        // value does not change: not a container, whose element types can be edited in place,
        // nor a year-month interval, whose value follows `PYSPARK_YM_INTERVAL_LEGACY`.
        let (need, fixed) = match self.data_type_object.kept(py) {
            Some(object) => {
                let object = object.bind(py);
                // Nor a type of a Python subclass, whose method is its own.
                let fixed = !(object.is_instance_of::<PyStructType>()
                    || object.is_instance_of::<PyArrayType>()
                    || object.is_instance_of::<PyMapType>()
                    || object.is_instance_of::<PyYearMonthIntervalType>()
                    || is_udt(object)?
                    || is_subclass_object(object));
                (child_need_conversion(py, object)?, fixed)
            }
            None => {
                let data_type = &self.field.data_type;
                let fixed = !(matches!(
                    data_type,
                    DataType::Array { .. }
                        | DataType::Map { .. }
                        | DataType::Struct { .. }
                        | DataType::YearMonthInterval { .. }
                ) || has_udt(data_type));
                (need_conversion(py, data_type)?, fixed)
            }
        };
        if fixed {
            let _ = self.need_conversion_of_type.set(need);
        }
        Ok(need)
    }

    /// The core field, with the metadata of the dict as it is now.
    pub(crate) fn current_field(&self, py: Python<'_>) -> PyResult<StructField> {
        let mut field = self.field.clone();
        field.data_type = self.data_type_object.live(py, &self.field.data_type)?;
        let metadata = self.metadata.bind(py);
        field.metadata = if metadata.is_empty() {
            BTreeMap::new()
        } else {
            py_metadata_to_map(py, metadata)?
        };
        Ok(field)
    }
}

#[pymethods]
impl PyStructField {
    // --- StructField object-model methods (v4.2.0 parity), operating on the field
    // itself (a StructField is not a DataType, so these do not go through
    // py_to_data_type). ---
    /// As PySpark's `name:` and the `simpleString()` of the type object of the field (see
    /// `child_simple_string`).
    #[pyo3(name = "simpleString")]
    fn __sf_simple_string(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "{}:{}",
            self.field.name,
            child_simple_string(py, &self.data_type_object, &self.field.data_type)?
        ))
    }
    /// As PySpark's, `StructField.typeName` raises `INVALID_TYPENAME_CALL`: a field has no type
    /// name, but its `dataType` has.
    #[pyo3(name = "typeName")]
    fn __sf_type_name(&self, py: Python<'_>) -> PyResult<String> {
        Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "INVALID_TYPENAME_CALL",
            &[],
        ))
    }
    /// As PySpark's `DataType.json`, of the `jsonValue` of the field, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __sf_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        let (json, other) = field_json_value(py, slf)?;
        if other {
            return json_dumps(py, &json);
        }
        Ok(slf.borrow().current_field(py)?.json_value().to_string())
    }
    #[pyo3(name = "jsonValue")]
    fn __sf_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(field_json_value(slf.py(), slf)?.0)
    }
    /// As PySpark's, of the `needConversion()` of the type of the field (see
    /// `need_conversion_of_object`).
    #[pyo3(name = "needConversion")]
    fn __sf_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion_of_object(slf.py(), slf.as_any())
    }
    #[pyo3(name = "fromInternal")]
    fn __sf_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __sf_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Collation metadata for the field (empty when the field has no collations).
    #[pyo3(name = "getCollationMetadata")]
    fn __sf_get_collation_metadata<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        Ok(pyo3::types::PyDict::new(py))
    }

    /// Map of collated-field path -> collation name, parsed from the field's
    /// `__COLLATIONS` metadata (empty when absent). Mirrors `StructField.getCollationsMap`.
    #[pyo3(name = "getCollationsMap")]
    fn __sf_get_collations_map<'py>(
        &self,
        py: Python<'py>,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let out = pyo3::types::PyDict::new(py);
        if let Ok(dict) = metadata.cast::<pyo3::types::PyDict>() {
            if let Some(coll) = dict.get_item("__COLLATIONS")? {
                if let Ok(cdict) = coll.cast::<pyo3::types::PyDict>() {
                    for (k, v) in cdict.iter() {
                        let val: String = v.extract()?;
                        // Value is "provider.name"; the map holds just the name.
                        let name = val.rsplit('.').next().unwrap_or(&val).to_string();
                        out.set_item(k, name)?;
                    }
                }
            }
        }
        Ok(out)
    }

    /// The "provider.collation" value for a collated string type. Mirrors
    /// `StructField.schemaCollationValue` (UTF8_BINARY/UTF8_LCASE are the "spark"
    /// provider, everything else "icu").
    #[pyo3(name = "schemaCollationValue")]
    fn __sf_schema_collation_value(&self, dt: &Bound<'_, PyAny>) -> PyResult<String> {
        let collation = match py_to_data_type(dt)? {
            DataType::String { collation } => collation,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "schemaCollationValue expects a StringType",
                ));
            }
        };
        let provider = if collation == "UTF8_BINARY" || collation == "UTF8_LCASE" {
            "spark"
        } else {
            "icu"
        };
        Ok(format!("{provider}.{collation}"))
    }
    /// The type of the DDL string `ddl`, as PySpark's `DataType.fromDDL`, which `StructField`
    /// inherits: the parsed type (a StructType for "a int"), not a StructField.
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    fn __sf_from_ddl(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        ddl: &str,
    ) -> PyResult<Py<PyAny>> {
        Ok(py
            .import("pyspark.sql.types")?
            .call_method1("_parse_datatype_string", (ddl,))?
            .unbind())
    }

    #[classmethod]
    #[pyo3(name = "fromJson")]
    fn __sf_from_json(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        json: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(shim_from_json(py, "_struct_field_from_json")?
            .call1((json,))?
            .unbind())
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `StructField.__new__(StructField)`, of which pickle's `copyreg.__newobj__` makes the field
    /// that PySpark pickled, is a field of an empty name and of `NullType()` until `__setstate__`
    /// sets its attributes. The class shows the signature of `__init__`, as `inspect.signature`
    /// of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(name, dataType, nullable=True, metadata=None)"
    )]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let py = cls.py();
        let field = StructField {
            name: String::new(),
            data_type: DataType::Null,
            nullable: true,
            metadata: BTreeMap::new(),
        };
        Ok(field_base(Self::from_field(py, field)?))
    }
    /// As PySpark's `__init__(self, name, dataType, nullable=True, metadata=None)`, which asserts
    /// the data type first, then the name.
    #[pyo3(signature = (name, dataType, nullable=default_flag(), metadata=None))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        name: &Bound<'_, PyAny>,
        dataType: &Bound<'_, PyAny>,
        nullable: Py<PyAny>,
        metadata: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        let py = slf.py();
        let data_type = nested_data_type(dataType, "dataType")?;
        let name = field_name(name)?;
        let md = match metadata {
            Some(d) => py_metadata_to_map(py, d)?,
            None => BTreeMap::new(),
        };
        let field = StructField {
            name,
            data_type,
            nullable: nullable.bind(py).is_truthy()?,
            metadata: md,
        };
        let this = Self::build(py, field, metadata, nullable, KeptType::of(dataType))?;
        // The parts that this replaces are freed after the borrow ends.
        let replaced = std::mem::replace(&mut *slf.borrow_mut(), this);
        drop(replaced);
        Ok(())
    }
    /// The field of the state that PySpark pickled, of `name`, `dataType`, `nullable` and
    /// `metadata`, which pickle sets as attributes (see `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    // The attributes are plain ones, as PySpark's `StructField`: an assignment of them is the
    // field's, and of the `StructType` the field is in.
    #[getter]
    fn name(&self, py: Python<'_>) -> Py<PyAny> {
        match &self.name_object {
            Some(name) => name.clone_ref(py),
            None => pyo3::types::PyString::new(py, &self.field.name)
                .into_any()
                .unbind(),
        }
    }
    /// A name that is not a str is kept as it was given, and its `str` is the name of the field.
    #[setter]
    fn set_name(&mut self, py: Python<'_>, name: Py<PyAny>) -> PyResult<()> {
        if name.bind(py).is_instance_of::<pyo3::types::PyString>() {
            self.field.name = name.extract(py)?;
            self.name_object = None;
        } else {
            self.field.name = name.bind(py).str()?.to_cow()?.into_owned();
            self.name_object = Some(name);
        }
        Ok(())
    }
    /// The nullable flag as it was given (mirrors `StructField.nullable`, any value).
    #[getter]
    fn nullable(&self, py: Python<'_>) -> Py<PyAny> {
        self.nullable.clone_ref(py)
    }
    #[setter]
    fn set_nullable(&mut self, py: Python<'_>, nullable: Py<PyAny>) -> PyResult<()> {
        self.field.nullable = nullable.bind(py).is_truthy()?;
        self.nullable = nullable;
        Ok(())
    }
    /// The field's data type as its concrete Python type object (mirrors `StructField.dataType`).
    #[getter]
    #[pyo3(name = "dataType")]
    fn data_type(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.data_type_object.object(py, &self.field.data_type)
    }
    #[setter]
    #[pyo3(name = "dataType")]
    fn set_data_type(&mut self, dataType: &Bound<'_, PyAny>) -> PyResult<()> {
        self.field.data_type = py_to_data_type(dataType)?;
        self.data_type_object = KeptType::of(dataType);
        self.need_conversion_of_type = std::sync::OnceLock::new();
        Ok(())
    }
    /// The field's metadata dict (mirrors `StructField.metadata`; values may be any type).
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyDict> {
        self.metadata.bind(py).clone()
    }
    #[setter]
    fn set_metadata(
        &mut self,
        py: Python<'_>,
        metadata: &Bound<'_, pyo3::types::PyDict>,
    ) -> PyResult<()> {
        self.field.metadata = py_metadata_to_map(py, metadata)?;
        self.metadata = metadata.clone().unbind();
        Ok(())
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Mirror pyspark: StructField('name', <DataType repr>, nullable).
        let dt = self.data_type_object.object(py, &self.field.data_type)?;
        let dt_repr: String = dt.bind(py).repr()?.extract()?;
        let nullable: String = self.nullable.bind(py).str()?.extract()?;
        Ok(format!(
            "StructField('{}', {}, {})",
            self.field.name, dt_repr, nullable
        ))
    }
    /// As PySpark's `DataType.__eq__`: of a field of the class of `self` or a subclass of it, as
    /// `isinstance(other, self.__class__)`, and of its `__dict__`, of the parameters and of the
    /// attributes that a user set on a field of a Python subclass (see `data_type_eq`).
    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let Ok(o) = other.cast::<PyStructField>() else {
            return Ok(false);
        };
        if !other.is_instance(&slf.get_type())? {
            return Ok(false);
        }
        // Of the metadata as Python's `==` compares the dict of PySpark's field.
        let field = slf.borrow().current_field(py)?;
        if !field.eq_as_python(&o.borrow().current_field(py)?) {
            return Ok(false);
        }
        // The flags as they were given are compared as PySpark's `__dict__` does, and so are the
        // metadata dicts, as Python compares them (see `data_type_eq`).
        if has_other_flag(py, slf.as_any())?
            || has_other_flag(py, other)?
            || holds_metadata(py, slf.as_any())?
            || holds_metadata(py, other)?
        {
            let (a, _) = field_json_value(py, slf)?;
            let (b, _) = field_json_value(py, o)?;
            if !a.eq(b)? {
                return Ok(false);
            }
        }
        // A type that the client keeps as its object, or one that holds one, compares by the
        // objects, as PySpark's `__dict__` holds them, which their JSON does not tell apart, and
        // so does a type of a Python subclass, or one that keeps one.
        let held = {
            let (this, that) = (slf.borrow(), o.borrow());
            holds_objects(&field.data_type)
                || kept_holds_subclass_object(py, &this.data_type_object)?
                || kept_holds_subclass_object(py, &that.data_type_object)?
        };
        if held {
            let (a, b) = {
                let (this, that) = (slf.borrow(), o.borrow());
                (
                    this.data_type_object.object(py, &this.field.data_type)?,
                    that.data_type_object.object(py, &that.field.data_type)?,
                )
            };
            if !a.bind(py).eq(b.bind(py))? {
                return Ok(false);
            }
        }
        // The attributes that a user set on a field of a Python subclass, of its `__dict__`.
        if is_subclass_object(slf.as_any()) || is_subclass_object(other) {
            let dict = pyo3::intern!(py, "__dict__");
            return slf.getattr(dict)?.eq(other.getattr(dict)?);
        }
        Ok(true)
    }
    /// pyspark: `hash(str(self))` (inherited from `DataType`). Defining `__eq__` alone
    /// would leave StructField unhashable.
    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<isize> {
        slf.str()?.hash()
    }
    /// A field pickles, as PySpark's does: of its name, type, nullability and metadata, with
    /// the class of `pyspark.sql.types`, also on the Python workers of the server. A field of a
    /// Python subclass pickles of its class and its `__dict__` (see `subclass_reduce`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (name, data_type, nullable, metadata) = {
            let this = slf.borrow();
            (
                this.name(py),
                this.data_type_object.object(py, &this.field.data_type)?,
                this.nullable.clone_ref(py),
                this.metadata.clone_ref(py),
            )
        };
        if !is_native(slf) {
            return subclass_reduce(
                slf.as_any(),
                vec![
                    ("name", name.into_bound(py)),
                    ("dataType", data_type.into_bound(py)),
                    ("nullable", nullable.into_bound(py)),
                    ("metadata", metadata.into_bound(py).into_any()),
                ],
            );
        }
        let name = slf.borrow().field.name.clone();
        let args = (name, data_type, nullable, metadata).into_pyobject(py)?;
        Ok((slf.get_type(), args)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }
}

#[pyclass(subclass, name = "StructType", module = "pyspark.sql.types", extends = PyDataType)]
pub struct PyStructType {
    /// The `StructField` objects, as PySpark's `StructType.fields` list holds them. The list is
    /// the struct's own: an edit of it (append, `[i] = `, pop, sort, ...) is an edit of the
    /// struct, so the fields are read from it each time.
    fields: Py<PyList>,
    /// The names, as PySpark's `StructType.names` list holds them, which is kept apart from
    /// `fields`: `__init__` and `add` set it, and nothing else does.
    names: Py<PyList>,
    /// The `needConversion` of each field, as PySpark's `_needConversion`. Like PySpark's, it is
    /// computed by `__init__` and `add`, so an edit of a field's type does not change it.
    need_conversion: Vec<bool>,
}

impl PyStructType {
    /// The core fields, as they are now: each field's metadata is read from its object, where an
    /// edit in place is kept, as in PySpark.
    pub(crate) fn current_fields(&self, py: Python<'_>) -> PyResult<Vec<StructField>> {
        self.fields
            .bind(py)
            .iter()
            .map(|item| as_struct_field(&item)?.borrow().current_field(py))
            .collect()
    }

    /// The core type of the struct, as it is now.
    fn current_data_type(&self, py: Python<'_>) -> PyResult<DataType> {
        Ok(DataType::Struct {
            fields: self.current_fields(py)?,
        })
    }
}

impl PyStructType {
    /// The base core type and the parts of `StructType(fields)`, as PySpark's `__init__` takes
    /// the fields: no fields, or none given, is a new empty list (`if not fields`), and a list of
    /// fields is the struct's own list.
    fn parts_of(
        py: Python<'_>,
        fields: Option<Bound<'_, PyAny>>,
    ) -> PyResult<(DataType, PyStructType)> {
        let Some(fields) = fields.filter(|fs| fs.is_truthy().unwrap_or(false)) else {
            return struct_parts(py, PyList::empty(py));
        };
        // As PySpark's `[f.name for f in fields]`, which reads the name of each item first: an
        // item without one, such as a character of a str or a tuple, is an AttributeError.
        let items = fields
            .try_iter()
            .map_err(|_| not_iterable(&fields))?
            .collect::<PyResult<Vec<_>>>()?;
        for item in &items {
            match item.getattr("name") {
                Ok(_) => {}
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
                    return Err(no_attribute(item, "name"));
                }
                Err(e) => return Err(e),
            }
        }
        if !items
            .iter()
            .all(|item| item.is_instance_of::<PyStructField>())
        {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "fields should be a list of StructField",
            ));
        }
        // A list is the struct's own list, as in PySpark's `__init__`.
        let list = match fields.cast::<PyList>() {
            Ok(list) => list.clone(),
            Err(_) => PyList::new(py, items.iter())?,
        };
        struct_parts(py, list)
    }
}

#[pymethods]
impl PyStructType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        type_json_text(slf.as_any())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(type_json_value(py, slf.as_any())?.0)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, or of the types it is made of (see `reduce_with_udts`), and of a Python
    /// subclass, of its class and its `__dict__`, of the `fields`, `names`, `_needConversion` and
    /// `_needSerializeAnyField` that PySpark's `__init__` sets (see `subclass_reduce`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if !is_native(slf) {
            let (fields, names, need_conversion) = {
                let this = slf.borrow();
                (
                    this.fields.clone_ref(py),
                    this.names.clone_ref(py),
                    this.need_conversion.clone(),
                )
            };
            let any = need_conversion.iter().any(|&need| need);
            return subclass_reduce(
                slf.as_any(),
                vec![
                    ("fields", fields.into_bound(py).into_any()),
                    ("names", names.into_bound(py).into_any()),
                    (
                        "_needConversion",
                        need_conversion.into_pyobject(py)?.into_any(),
                    ),
                    (
                        "_needSerializeAnyField",
                        any.into_pyobject(py)?.to_owned().into_any(),
                    ),
                ],
            );
        }
        let this = slf.borrow();
        let dt = this.current_data_type(py)?;
        let flagged = has_other_flag(slf.py(), slf.as_any())?;
        let (f, args) = reduce_with_udts(slf.as_any(), &dt, flagged, || {
            let objects = PyList::new(py, this.fields.bind(py).iter())?;
            (objects,).into_pyobject(py)
        })?;
        Ok((f, args).into_pyobject(py)?.into_any().unbind())
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `StructType.__new__(StructType)`, of which pickle's `copyreg.__newobj__` makes the type that
    /// PySpark pickled, is a struct of no fields until `__setstate__` sets them. The class shows
    /// the signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(fields=None)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let py = cls.py();
        let (base, parts) = struct_parts(py, PyList::empty(py))?;
        Ok(init_chain!(base, parts, [], value))
    }
    /// `StructType(fields=None)` where each field is a `StructField`, as PySpark's `__init__`,
    /// which sets the `fields`, the `names` and the `_needConversion` of the struct.
    #[pyo3(signature = (fields=None))]
    fn __init__(slf: &Bound<'_, Self>, fields: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let py = slf.py();
        let (base, parts) = Self::parts_of(py, fields)?;
        let mut this = slf.borrow_mut();
        this.as_super().inner = Some(base);
        // The parts that this replaces are freed after the borrow ends.
        let replaced = std::mem::replace(&mut *this, parts);
        drop(this);
        drop(replaced);
        Ok(())
    }

    /// The struct of the state that PySpark pickled, of `fields`, `names`, `_needConversion` and
    /// `_needSerializeAnyField`, which pickle sets as attributes (see `set_pickled_state`): the
    /// fields and the names as they were, also when an edit of the fields made them differ, and
    /// `_needConversion` as PySpark kept it, of which `_needSerializeAnyField` is. A state without
    /// it has the `needConversion` of its fields.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let mut kept: Option<Vec<bool>> = None;
        set_pickled_state(slf.as_any(), state, |key, value| match key {
            "_needConversion" => {
                let need = value
                    .try_iter()?
                    .map(|need| need?.is_truthy())
                    .collect::<PyResult<Vec<bool>>>()?;
                kept = Some(need);
                Ok(true)
            }
            "_needSerializeAnyField" => Ok(true),
            _ => Ok(false),
        })?;
        let need_conversion = match kept {
            Some(need) => need,
            None => {
                let fields = slf.borrow().fields.clone_ref(py);
                need_conversion_of_fields(py, fields.bind(py))?
            }
        };
        slf.borrow_mut().need_conversion = need_conversion;
        Ok(())
    }

    /// Append a field and return `self` (chainable, mirroring `StructType.add`):
    /// `StructType().add("a", "integer").add("b", StringType())`. The field goes to `fields` and
    /// its name to `names`, as PySpark's `add` does them.
    #[pyo3(
        signature = (field, data_type=None, nullable=default_flag(), metadata=None),
        text_signature = "($self, field, data_type=None, nullable=True, metadata=None)"
    )]
    fn add<'py>(
        slf: Bound<'py, Self>,
        field: &Bound<'_, PyAny>,
        data_type: Option<&Bound<'_, PyAny>>,
        nullable: Py<PyAny>,
        metadata: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = field.py();
        let (object, name) = if let Ok(sf) = field.cast::<PyStructField>() {
            (sf.clone().into_any(), sf.borrow().field.name.clone())
        } else {
            // As PySpark's: a name takes a data type, of which a str is the JSON value of one. A
            // name without one is an error, but any other field is a `StructField` of the None.
            let type_object = match data_type {
                Some(data_type) if data_type.is_instance_of::<pyo3::types::PyString>() => py
                    .import("pyspark.sql.types")?
                    .getattr("_parse_datatype_json_value")?
                    .call1((data_type,))?,
                Some(data_type) => data_type.clone(),
                None if field.is_instance_of::<pyo3::types::PyString>() => {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "PySparkValueError",
                        "ARGUMENT_REQUIRED",
                        &[
                            ("arg_name", "data_type"),
                            ("condition", "passing name of struct_field to create"),
                        ],
                    ));
                }
                None => py.None().into_bound(py),
            };
            // The data type is asserted, then the name, by the `StructField` they make.
            let dt = nested_data_type(&type_object, "dataType")?;
            let name = field_name(field)?;
            let md = match metadata {
                Some(d) => py_metadata_to_map(py, d)?,
                None => BTreeMap::new(),
            };
            let new_field = StructField {
                name: name.clone(),
                data_type: dt,
                nullable: nullable.bind(py).is_truthy()?,
                metadata: md,
            };
            let object = field_object(
                py,
                PyStructField::build(
                    py,
                    new_field,
                    metadata,
                    nullable,
                    KeptType::of(&type_object),
                )?,
            )?;
            (object.into_bound(py).into_any(), name)
        };
        // Only the types are read here, not the metadata: `add` is then linear in the fields
        // (a loop of adds with metadata would otherwise encode every field's metadata each time).
        let new_field = as_struct_field(&object)?.borrow().current_field(py)?;
        let need_conversion = {
            let me = slf.borrow();
            me.fields.bind(py).append(object)?;
            me.names.bind(py).append(name)?;
            need_conversion_of_fields(py, me.fields.bind(py))?
        };
        // The base `DataType` is a snapshot, which only the bare-base fallback of
        // `py_to_data_type` reads (a struct is read from its live fields), so it takes the new
        // field, as the fields of `add` do.
        let mut me = slf.borrow_mut();
        if let Some(DataType::Struct { fields }) = &mut me.as_super().inner {
            fields.push(new_field);
        }
        me.need_conversion = need_conversion;
        drop(me);
        Ok(slf)
    }

    /// The fields as `StructField` objects, in the list that is the struct's own (mirrors
    /// `StructType.fields`): the same list each time, so an edit of it is an edit of the struct.
    #[getter]
    fn fields(&self, py: Python<'_>) -> Py<PyList> {
        self.fields.clone_ref(py)
    }
    /// An assignment of `fields` is the struct's, as PySpark's: `names` and `needConversion` are
    /// not recomputed by it. A value that is not a list is a list of its items.
    #[setter]
    fn set_fields(&mut self, py: Python<'_>, fields: &Bound<'_, PyAny>) -> PyResult<()> {
        let list = list_of(py, fields)?;
        for item in list.iter() {
            as_struct_field(&item)?;
        }
        self.fields = list.unbind();
        Ok(())
    }

    /// The field names, in the list that is the struct's own (mirrors `StructType.names`).
    #[getter]
    fn names(&self, py: Python<'_>) -> Py<PyList> {
        self.names.clone_ref(py)
    }
    #[setter]
    fn set_names(&mut self, py: Python<'_>, names: &Bound<'_, PyAny>) -> PyResult<()> {
        self.names = list_of(py, names)?.unbind();
        Ok(())
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.fields.bind(py).len()
    }

    /// Index by position (`st[0]`) or by name (`st["a"]`), returning a `StructField`
    /// (mirrors `StructType.__getitem__`; a slice yields a new `StructType`), with the errors of
    /// PySpark's: `KEY_NOT_EXISTS` of a name, `INDEX_OUT_OF_RANGE` of an index, and
    /// `NOT_EXPECTED_TYPE` of another key. A name is looked up in the fields, as PySpark does.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use crate::column::raise_pyspark;
        let fields = self.fields.bind(py);
        if let Ok(name) = key.cast::<pyo3::types::PyString>() {
            let name = name.to_cow()?;
            for item in fields.iter() {
                if as_struct_field(&item)?.borrow().field.name == *name {
                    return Ok(item.unbind());
                }
            }
            return Err(raise_pyspark(
                py,
                "PySparkKeyError",
                "KEY_NOT_EXISTS",
                &[("key", &name)],
            ));
        }
        if key.is_instance_of::<pyo3::types::PySlice>() {
            // As PySpark's `StructType(self.fields[key])`: a new struct of the same fields.
            let sliced = fields.as_any().get_item(key)?.cast_into::<PyList>()?;
            return Ok(struct_of_list(py, sliced)?.into_any());
        }
        if !key.is_instance_of::<pyo3::types::PyInt>() {
            return Err(crate::stat_arguments::not_expected_type(
                key,
                "key",
                "int, slice or str",
            ));
        }
        let n = fields.len() as isize;
        let index = key.extract::<isize>().ok();
        match index.map(|i| if i < 0 { i + n } else { i }) {
            Some(i) if (0..n).contains(&i) => Ok(fields.get_item(i as usize)?.unbind()),
            _ => Err(raise_pyspark(
                py,
                "PySparkIndexError",
                "INDEX_OUT_OF_RANGE",
                &[("arg_name", "StructType"), ("index", &key.str()?.to_cow()?)],
            )),
        }
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.fields.bind(py).call_method0("__iter__")?.unbind())
    }

    /// As PySpark's, of the `simpleString()` of each field object, which names its type by the
    /// type object that it holds (see `child_simple_string`), of a field of a Python subclass its
    /// own.
    #[pyo3(name = "simpleString")]
    fn simple_string(&self, py: Python<'_>) -> PyResult<String> {
        let mut parts = Vec::with_capacity(self.fields.bind(py).len());
        for item in self.fields.bind(py).iter() {
            if is_subclass_object(&item) {
                parts.push(
                    item.call_method0(pyo3::intern!(py, "simpleString"))?
                        .extract()?,
                );
                continue;
            }
            parts.push(as_struct_field(&item)?.borrow().__sf_simple_string(py)?);
        }
        Ok(format!("struct<{}>", parts.join(",")))
    }

    /// Field names, as a copy of the `names` list. Mirrors `StructType.fieldNames()`, which is
    /// `list(self.names)`: it does not follow an edit of `fields`.
    #[pyo3(name = "fieldNames")]
    fn __st_field_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.names.bind(py).call_method0("copy")
    }

    /// Whether a field needs conversion, as PySpark's `_needSerializeAnyField`.
    #[getter(_needSerializeAnyField)]
    fn __st_need_serialize_any_field(&self) -> bool {
        self.need_conversion.iter().any(|&need| need)
    }

    /// The `needConversion` of each field, as PySpark's `_needConversion`.
    #[getter(_needConversion)]
    fn __st_need_conversion_of_fields(&self) -> Vec<bool> {
        self.need_conversion.clone()
    }

    /// DDL string. Mirrors `StructType.toDDL()`, which asks the server of the active session to
    /// convert the JSON of the schema, as Spark writes DDL.
    #[pyo3(name = "toDDL")]
    fn __st_to_ddl(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let session = py
            .import("pyspark.sql")?
            .getattr("SparkSession")?
            .call_method0("getActiveSession")?;
        if session.is_none() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        session.call_method1("_to_ddl", (slf,))?.extract()
    }

    /// Tree string. Mirrors `StructType.treeString(maxDepth)`.
    // Use an integer default so explicit None raises TypeError, matching PySpark.
    #[allow(non_snake_case)]
    #[pyo3(
        name = "treeString",
        signature = (maxDepth = Python::attach(|py| pyo3::types::PyInt::new(py, i32::MAX).into_any().unbind())),
        text_signature = "($self, maxDepth=2147483647)"
    )]
    fn __st_tree_string(slf: &Bound<'_, Self>, maxDepth: Py<PyAny>) -> PyResult<String> {
        let py = slf.py();
        let depth = normalize_tree_depth(maxDepth.bind(py))?;
        let data_type = slf.borrow().current_data_type(py)?;
        // As PySpark's, which takes the type name of each type it prints: a StructField as a type
        // has none (`StructField.typeName`).
        if nested_field_in(py, &data_type, depth)?.is_some() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "INVALID_TYPENAME_CALL",
                &[],
            ));
        }
        // The flags are written as they were given, which the core's bools are not, and a type of
        // its class only, or of a Python subclass, by its `typeName()`, which PySpark's
        // `_get_jvm_type_name` writes of it.
        let other_flags = has_other_flag(py, slf.as_any())?;
        let subclasses = keeps_subclass_objects(py, slf.as_any())?;
        let error: std::cell::RefCell<Option<PyErr>> = std::cell::RefCell::new(None);
        let keep = |result: PyResult<Option<String>>| match result {
            Ok(text) => text,
            Err(e) => {
                error.borrow_mut().get_or_insert(e);
                None
            }
        };
        let text = data_type.tree_string_with_names(
            depth,
            |path, flag, nullable| {
                let text = if other_flags {
                    keep(tree_flag_text(slf.as_any(), path, flag))
                } else {
                    None
                };
                text.unwrap_or_else(|| nullable.to_string())
            },
            |path, data_type| match data_type {
                DataType::Unparsed { data_type_string } => {
                    keep(tree_type_name_text(slf.as_any(), path, data_type_string))
                }
                _ if subclasses => keep(subclass_tree_type_name(slf.as_any(), path)),
                _ => None,
            },
        );
        if let Some(e) = error.into_inner() {
            return Err(e);
        }
        text.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// A copy with all fields made nullable (recursively). Mirrors `StructType.toNullable()`.
    #[pyo3(name = "toNullable")]
    fn __st_to_nullable(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let data_type = slf.borrow().current_data_type(py)?;
        // Of the type objects that it holds, as PySpark's `_as_nullable` (see `ArrayType`'s).
        if holds_objects(&data_type) || keeps_subclass_objects(py, slf.as_any())? {
            return Ok(as_nullable(slf.as_any())?.unbind());
        }
        let fields = match data_type.to_nullable() {
            DataType::Struct { fields } => fields,
            _ => slf.borrow().current_fields(py)?,
        };
        Ok(py_new_struct(py, fields)?.into_any())
    }

    /// Build a StructType from its JSON value. Mirrors `StructType.fromJson`.
    #[classmethod]
    #[pyo3(name = "fromJson")]
    fn __st_from_json(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        json: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(shim_from_json(py, "_struct_type_from_json")?
            .call1((json,))?
            .unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Mirror pyspark: StructType([str(field), ...]), each field as its own repr shows it.
        let mut parts = Vec::with_capacity(self.fields.bind(py).len());
        for field in self.fields.bind(py).iter() {
            parts.push(field.str()?.extract::<String>()?);
        }
        Ok(format!("StructType([{}])", parts.join(", ")))
    }
}

#[pyclass(subclass, name = "CharType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyCharType {
    pub length: i32,
}

#[pymethods]
impl PyCharType {
    /// The length of the strings, as PySpark's `CharType.length`: a plain attribute, of which an
    /// assignment makes the type of the length.
    #[getter]
    fn length(&self) -> i32 {
        self.length
    }
    #[setter]
    fn set_length(slf: &Bound<'_, Self>, length: i32) -> PyResult<()> {
        slf.borrow_mut().length = length;
        sync_base(slf.as_any())
    }
    /// The type of the state that PySpark pickled, of `length` (see `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let length = slf.borrow().length;
        let dt = DataType::Char { length };
        reduce_of(slf, &dt.json(), || {
            Ok(vec![("length", length.into_pyobject(py)?.into_any())])
        })
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `CharType.__new__(CharType)`, of which pickle's `copyreg.__newobj__` makes the type that
    /// PySpark pickled, is a type of length 0 until `__setstate__` sets it. The class shows the
    /// signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(length)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        Ok(init_chain!(
            DataType::Char { length: 0 },
            PyCharType { length: 0 },
            [PyAtomicType],
            value
        ))
    }
    /// As PySpark's `__init__(self, length)`.
    fn __init__(slf: &Bound<'_, Self>, length: i32) -> PyResult<()> {
        Self::set_length(slf, length)
    }
    fn __repr__(&self) -> String {
        format!("CharType({})", self.length)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("char({})", self.length)
    }
}

#[pyclass(subclass, name = "VarcharType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyVarcharType {
    pub length: i32,
}

#[pymethods]
impl PyVarcharType {
    /// The maximum length of the strings, as PySpark's `VarcharType.length`: a plain attribute,
    /// of which an assignment makes the type of the length.
    #[getter]
    fn length(&self) -> i32 {
        self.length
    }
    #[setter]
    fn set_length(slf: &Bound<'_, Self>, length: i32) -> PyResult<()> {
        slf.borrow_mut().length = length;
        sync_base(slf.as_any())
    }
    /// The type of the state that PySpark pickled, of `length` (see `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let length = slf.borrow().length;
        let dt = DataType::Varchar { length };
        reduce_of(slf, &dt.json(), || {
            Ok(vec![("length", length.into_pyobject(py)?.into_any())])
        })
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `VarcharType.__new__(VarcharType)`, of which pickle's `copyreg.__newobj__` makes the type
    /// that PySpark pickled, is a type of length 0 until `__setstate__` sets it. The class shows
    /// the signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(length)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        Ok(init_chain!(
            DataType::Varchar { length: 0 },
            PyVarcharType { length: 0 },
            [PyAtomicType],
            value
        ))
    }
    /// As PySpark's `__init__(self, length)`.
    fn __init__(slf: &Bound<'_, Self>, length: i32) -> PyResult<()> {
        Self::set_length(slf, length)
    }
    fn __repr__(&self) -> String {
        format!("VarcharType({})", self.length)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("varchar({})", self.length)
    }
}

#[pyclass(subclass, name = "TimeType", module = "pyspark.sql.types", extends = PyAnyTimeType)]
pub struct PyTimeType {
    pub precision: i32,
}

#[pymethods]
impl PyTimeType {
    /// The precision of the times, as PySpark's `TimeType.precision`: a plain attribute, of which
    /// an assignment makes the type of the precision.
    #[getter]
    fn precision(&self) -> i32 {
        self.precision
    }
    #[setter]
    fn set_precision(slf: &Bound<'_, Self>, precision: i32) -> PyResult<()> {
        slf.borrow_mut().precision = precision;
        sync_base(slf.as_any())
    }
    /// The type of the state that PySpark pickled, of `precision` (see `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        nano: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", nano)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        t: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", t)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let precision = slf.borrow().precision;
        let dt = DataType::Time { precision };
        reduce_of(slf, &dt.json(), || {
            Ok(vec![("precision", precision.into_pyobject(py)?.into_any())])
        })
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `TimeType.__new__(TimeType)`, of which pickle's `copyreg.__newobj__` makes the type that
    /// PySpark pickled, is a `TimeType(6)` until `__setstate__` sets it. The class shows the
    /// signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(precision=6)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let precision = 6;
        Ok(init_chain!(
            DataType::Time { precision },
            PyTimeType { precision },
            [PyAtomicType, PyDatetimeType, PyAnyTimeType],
            value
        ))
    }
    /// As PySpark's `__init__(self, precision=6)`, of microseconds.
    #[pyo3(signature = (precision=6))]
    fn __init__(slf: &Bound<'_, Self>, precision: i32) -> PyResult<()> {
        Self::set_precision(slf, precision)
    }
    fn __repr__(&self) -> String {
        format!("TimeType({})", self.precision)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("time({})", self.precision)
    }
}

/// Spark's `TimestampNTZNanosType(precision)` (precision 7..=9; default 9). PySpark 4.3.0 has no
/// Python class for the nanosecond timestamp types, but the server reports them in
/// schemas, so they materialize as this class instead of being mapped to a lossy type. As
/// PySpark has no such name, it is not one of `pyspark.sql.types`.
#[pyclass(name = "TimestampNTZNanosType", module = "pyspark._pyspark", extends = PyDatetimeType)]
pub struct PyTimestampNTZNanosType {
    pub precision: i32,
}

#[pymethods]
impl PyTimestampNTZNanosType {
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::TimestampNtzNanos {
            precision: self.precision,
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    #[pyo3(signature = (precision=9))]
    fn new(precision: i32) -> PyResult<pyo3::PyClassInitializer<Self>> {
        if !(7..=9).contains(&precision) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "TimestampNTZNanosType precision must be in 7..9, got {}",
                precision
            )));
        }
        Ok(init_chain!(
            DataType::TimestampNtzNanos { precision },
            PyTimestampNTZNanosType { precision },
            [PyAtomicType, PyDatetimeType],
            value
        ))
    }
    #[getter]
    fn precision(&self) -> i32 {
        self.precision
    }
    fn __repr__(&self) -> String {
        format!("TimestampNTZNanosType({})", self.precision)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("timestamp_ntz({})", self.precision)
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> String {
        format!("timestamp_ntz({})", self.precision)
    }
}

/// Spark's `TimestampLTZNanosType(precision)` (precision 7..=9; default 9). PySpark 4.3.0 has no
/// Python class for the nanosecond timestamp types, but the server reports them in
/// schemas, so they materialize as this class instead of being mapped to a lossy type. As
/// PySpark has no such name, it is not one of `pyspark.sql.types`.
#[pyclass(name = "TimestampLTZNanosType", module = "pyspark._pyspark", extends = PyDatetimeType)]
pub struct PyTimestampLTZNanosType {
    pub precision: i32,
}

#[pymethods]
impl PyTimestampLTZNanosType {
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        let dt = DataType::TimestampLtzNanos {
            precision: self.precision,
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    #[pyo3(signature = (precision=9))]
    fn new(precision: i32) -> PyResult<pyo3::PyClassInitializer<Self>> {
        if !(7..=9).contains(&precision) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "TimestampLTZNanosType precision must be in 7..9, got {}",
                precision
            )));
        }
        Ok(init_chain!(
            DataType::TimestampLtzNanos { precision },
            PyTimestampLTZNanosType { precision },
            [PyAtomicType, PyDatetimeType],
            value
        ))
    }
    #[getter]
    fn precision(&self) -> i32 {
        self.precision
    }
    fn __repr__(&self) -> String {
        format!("TimestampLTZNanosType({})", self.precision)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("timestamp_ltz({})", self.precision)
    }
    #[pyo3(name = "typeName")]
    fn type_name(&self) -> String {
        format!("timestamp_ltz({})", self.precision)
    }
}

#[pyclass(
    subclass,
    name = "CalendarIntervalType",
    module = "pyspark.sql.types",
    extends = PyDataType
)]
pub struct PyCalendarIntervalType;

#[pymethods]
impl PyCalendarIntervalType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"interval\"", || Ok(Vec::new()))
    }
    /// The one instance of the type, and of each Python subclass of it, as PySpark's
    /// `DataTypeSingleton`: every call of the class returns it, and raises its `TypeError` of an
    /// argument (see `singleton_of`). The class shows the signature of
    /// `DataTypeSingleton.__call__(cls)`, `()`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        static INSTANCE: PyOnceLock<Py<PyCalendarIntervalType>> = PyOnceLock::new();
        singleton_of(cls, args, kwargs, &INSTANCE, || {
            init_chain!(DataType::CalendarInterval, PyCalendarIntervalType, [])
        })
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("CalendarIntervalType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "interval").clone().into_any())
    }
    /// As PySpark's, which names this type `interval`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "interval"
    }
}

#[pyclass(
    subclass,
    name = "YearMonthIntervalType",
    module = "pyspark.sql.types",
    extends = PyAnsiIntervalType
)]
pub struct PyYearMonthIntervalType {
    pub start_field: i32,
    pub end_field: i32,
}

#[pymethods]
impl PyYearMonthIntervalType {
    /// The first field, as PySpark's `YearMonthIntervalType.startField`: a plain attribute, of
    /// which an assignment of a field makes the type of it (see `set_interval_field`).
    #[getter(startField)]
    fn start_field(&self) -> i32 {
        self.start_field
    }
    #[setter(startField)]
    fn set_start_field(slf: &Bound<'_, Self>, field: &Bound<'_, PyAny>) -> PyResult<()> {
        let end = slf.borrow().end_field;
        let start = set_interval_field(slf.py(), field, end, 1, true)?;
        slf.borrow_mut().start_field = start;
        sync_base(slf.as_any())
    }
    /// The last field, as PySpark's `YearMonthIntervalType.endField`, a plain attribute too.
    #[getter(endField)]
    fn end_field(&self) -> i32 {
        self.end_field
    }
    #[setter(endField)]
    fn set_end_field(slf: &Bound<'_, Self>, field: &Bound<'_, PyAny>) -> PyResult<()> {
        let start = slf.borrow().start_field;
        let end = set_interval_field(slf.py(), field, start, 1, false)?;
        slf.borrow_mut().end_field = end;
        sync_base(slf.as_any())
    }
    /// The type of the state that PySpark pickled, of `startField` and `endField` (see
    /// `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    #[classattr]
    const YEAR: i32 = 0;
    #[classattr]
    const MONTH: i32 = 1;
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (start_field, end_field) = {
            let this = slf.borrow();
            (this.start_field, this.end_field)
        };
        let dt = DataType::YearMonthInterval {
            start_field,
            end_field,
        };
        reduce_of(slf, &dt.json(), || {
            Ok(vec![
                ("startField", start_field.into_pyobject(py)?.into_any()),
                ("endField", end_field.into_pyobject(py)?.into_any()),
            ])
        })
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `YearMonthIntervalType.__new__(YearMonthIntervalType)`, of which pickle's
    /// `copyreg.__newobj__` makes the type that PySpark pickled, is a `YearMonthIntervalType(0, 1)`
    /// until `__setstate__` sets it. The class shows the signature of `__init__`, as
    /// `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(startField=None, endField=None)"
    )]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let (start_field, end_field) = (0, 1);
        Ok(init_chain!(
            DataType::YearMonthInterval {
                start_field,
                end_field,
            },
            PyYearMonthIntervalType {
                start_field,
                end_field,
            },
            [PyAtomicType, PyAnsiIntervalType],
            value
        ))
    }
    /// Fields: YEAR=0, MONTH=1. Defaults to the full YEAR..MONTH range.
    /// As PySpark's `__init__(self, startField=None, endField=None)`, of the fields
    /// YEAR and MONTH: `INVALID_INTERVAL_CASTING` for another.
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        startField: Option<&Bound<'_, PyAny>>,
        endField: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let (start, end) = interval_fields(slf.py(), startField, endField, (0, 1), 1)?;
        {
            let mut this = slf.borrow_mut();
            this.start_field = start;
            this.end_field = end;
        }
        sync_base(slf.as_any())
    }
    /// As PySpark's, `"%s(%d, %d)" % (type(self).__name__, ...)`: of a Python subclass, of the
    /// name of its class.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let (start_field, end_field) = {
            let this = slf.borrow();
            (this.start_field, this.end_field)
        };
        if !is_native(slf) {
            let name = slf.get_type().name()?;
            return Ok(format!("{name}({start_field}, {end_field})"));
        }
        Ok(format!("YearMonthIntervalType({start_field}, {end_field})"))
    }
}

#[pyclass(
    subclass,
    name = "DayTimeIntervalType",
    module = "pyspark.sql.types",
    extends = PyAnsiIntervalType
)]
pub struct PyDayTimeIntervalType {
    pub start_field: i32,
    pub end_field: i32,
}

#[pymethods]
impl PyDayTimeIntervalType {
    /// The first field, as PySpark's `DayTimeIntervalType.startField`: a plain attribute, of
    /// which an assignment of a field makes the type of it (see `set_interval_field`).
    #[getter(startField)]
    fn start_field(&self) -> i32 {
        self.start_field
    }
    #[setter(startField)]
    fn set_start_field(slf: &Bound<'_, Self>, field: &Bound<'_, PyAny>) -> PyResult<()> {
        let end = slf.borrow().end_field;
        let start = set_interval_field(slf.py(), field, end, 3, true)?;
        slf.borrow_mut().start_field = start;
        sync_base(slf.as_any())
    }
    /// The last field, as PySpark's `DayTimeIntervalType.endField`, a plain attribute too.
    #[getter(endField)]
    fn end_field(&self) -> i32 {
        self.end_field
    }
    #[setter(endField)]
    fn set_end_field(slf: &Bound<'_, Self>, field: &Bound<'_, PyAny>) -> PyResult<()> {
        let start = slf.borrow().start_field;
        let end = set_interval_field(slf.py(), field, start, 3, false)?;
        slf.borrow_mut().end_field = end;
        sync_base(slf.as_any())
    }
    /// The type of the state that PySpark pickled, of `startField` and `endField` (see
    /// `set_pickled_state`).
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |_, _| Ok(false))
    }
    #[classattr]
    const DAY: i32 = 0;
    #[classattr]
    const HOUR: i32 = 1;
    #[classattr]
    const MINUTE: i32 = 2;
    #[classattr]
    const SECOND: i32 = 3;
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        micros: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", micros)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        dt: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", dt)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let (start_field, end_field) = {
            let this = slf.borrow();
            (this.start_field, this.end_field)
        };
        let dt = DataType::DayTimeInterval {
            start_field,
            end_field,
        };
        reduce_of(slf, &dt.json(), || {
            Ok(vec![
                ("startField", start_field.into_pyobject(py)?.into_any()),
                ("endField", end_field.into_pyobject(py)?.into_any()),
            ])
        })
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `DayTimeIntervalType.__new__(DayTimeIntervalType)`, of which pickle's
    /// `copyreg.__newobj__` makes the type that PySpark pickled, is a `DayTimeIntervalType(0, 3)`
    /// until `__setstate__` sets it. The class shows the signature of `__init__`, as
    /// `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(startField=None, endField=None)"
    )]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let (start_field, end_field) = (0, 3);
        Ok(init_chain!(
            DataType::DayTimeInterval {
                start_field,
                end_field,
            },
            PyDayTimeIntervalType {
                start_field,
                end_field,
            },
            [PyAtomicType, PyAnsiIntervalType],
            value
        ))
    }
    /// Fields: DAY=0, HOUR=1, MINUTE=2, SECOND=3. Defaults to the full DAY..SECOND range.
    /// As PySpark's `__init__(self, startField=None, endField=None)`, of the fields
    /// DAY, HOUR, MINUTE and SECOND: `INVALID_INTERVAL_CASTING` for another.
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        startField: Option<&Bound<'_, PyAny>>,
        endField: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let (start, end) = interval_fields(slf.py(), startField, endField, (0, 3), 3)?;
        {
            let mut this = slf.borrow_mut();
            this.start_field = start;
            this.end_field = end;
        }
        sync_base(slf.as_any())
    }
    /// As PySpark's, `"%s(%d, %d)" % (type(self).__name__, ...)`: of a Python subclass, of the
    /// name of its class.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let (start_field, end_field) = {
            let this = slf.borrow();
            (this.start_field, this.end_field)
        };
        if !is_native(slf) {
            let name = slf.get_type().name()?;
            return Ok(format!("{name}({start_field}, {end_field})"));
        }
        Ok(format!("DayTimeIntervalType({start_field}, {end_field})"))
    }
}

#[pyclass(subclass, name = "VariantType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyVariantType;

#[pymethods]
impl PyVariantType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    /// As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    /// own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, &v, false)
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        variant: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", variant)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__` (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        reduce_of(slf, "\"variant\"", || Ok(Vec::new()))
    }
    /// As the `object.__new__` of PySpark's plain class, which takes no arguments (see
    /// `object_new_arguments`).
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = "()")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        object_new_arguments(cls, &cls.py().get_type::<Self>(), args, kwargs)?;
        Ok(init_chain!(
            DataType::Variant,
            PyVariantType,
            [PyAtomicType]
        ))
    }
    /// As PySpark's `DataType.__repr__`, of the name of the class, of a Python subclass its own.
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return class_repr(slf.as_any());
        }
        Ok("VariantType()".to_string())
    }
    /// As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    /// its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "variant").clone().into_any())
    }
}

#[pyclass(subclass, name = "GeometryType", module = "pyspark.sql.types", extends = PySpatialType)]
pub struct PyGeometryType {
    pub srid: i32,
}

#[pymethods]
impl PyGeometryType {
    #[classattr]
    const DEFAULT_CRS: &'static str = "OGC:CRS84";
    #[classattr]
    const DEFAULT_SRID: i32 = 4326;
    #[classattr]
    const MIXED_CRS: &'static str = "SRID:ANY";
    #[classattr]
    const MIXED_SRID: i32 = -1;
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `GeometryType.__new__(GeometryType)`, of which pickle's `copyreg.__newobj__` makes the type
    /// that PySpark pickled, is a type of the default SRID until `__setstate__` sets it. The class
    /// shows the signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(srid)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let srid = Self::DEFAULT_SRID;
        Ok(init_chain!(
            DataType::Geometry { srid },
            PyGeometryType { srid },
            [PyAtomicType, PySpatialType],
            value
        ))
    }
    /// As PySpark's `__init__(self, srid)`: `"ANY"` for a mixed SRID, or an SRID of a
    /// coordinate reference system that PySpark knows.
    fn __init__(slf: &Bound<'_, Self>, srid: &Bound<'_, PyAny>) -> PyResult<()> {
        let srid = spatial_type_srid(srid, false)?;
        slf.borrow_mut().srid = srid;
        sync_base(slf.as_any())
    }
    /// The SRID, as PySpark's `GeometryType.srid`: a plain attribute, of which an assignment
    /// makes the type of the SRID (see `attribute_srid`), also its JSON, which PySpark's writes
    /// of the private `_crs` that its constructor sets.
    #[getter]
    fn srid(&self) -> i32 {
        self.srid
    }
    #[setter]
    fn set_srid(slf: &Bound<'_, Self>, srid: &Bound<'_, PyAny>) -> PyResult<()> {
        let srid = attribute_srid(srid, false)?;
        slf.borrow_mut().srid = srid;
        sync_base(slf.as_any())
    }
    /// The type of the state that PySpark pickled, of `srid` and `_crs`, which pickle sets as
    /// attributes (see `set_pickled_state`): the type is of the `srid`, whose CRS the JSON
    /// names, as PySpark's constructor sets `_crs` of it.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |key, _| Ok(key == "_crs"))
    }
    /// As PySpark's `GeometryType.toInternal(geometry)`: the dict of the SRID and the WKB of a
    /// `Geometry` (see `_to_internal`).
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        geometry: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", geometry)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__`, which holds the
    /// `_crs` of the SRID that PySpark's constructor sets and its `jsonValue` reads (see
    /// `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let srid = slf.borrow().srid;
        let dt = DataType::Geometry { srid };
        reduce_of(slf, &dt.json(), || {
            let (crs, _) = spatial_json_parts(&dt, "geometry");
            Ok(vec![
                ("srid", srid.into_pyobject(py)?.into_any()),
                ("_crs", crs.into_pyobject(py)?.into_any()),
            ])
        })
    }
    /// As PySpark's `GeometryType._from_crs(crs)`, of the JSON of a geometry type: the mixed
    /// SRID of `SRID:ANY` of any case, or else the SRID of a CRS of `geo_utils`, and
    /// `ST_INVALID_CRS_VALUE` of another.
    #[classmethod]
    fn _from_crs<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        crs: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        py.import("pyspark.sql.types")?
            .getattr("_geometry_type_from_crs")?
            .call1((crs,))
    }
    fn __repr__(&self) -> String {
        match self.srid {
            -1 => "GeometryType(ANY)".to_string(),
            srid => format!("GeometryType({srid})"),
        }
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        DataType::Geometry { srid: self.srid }.simple_string()
    }
}

#[pyclass(subclass, name = "GeographyType", module = "pyspark.sql.types", extends = PySpatialType)]
pub struct PyGeographyType {
    pub srid: i32,
}

#[pymethods]
impl PyGeographyType {
    #[classattr]
    const DEFAULT_ALG: &'static str = "SPHERICAL";
    #[classattr]
    const DEFAULT_CRS: &'static str = "OGC:CRS84";
    #[classattr]
    const DEFAULT_SRID: i32 = 4326;
    #[classattr]
    const MIXED_CRS: &'static str = "SRID:ANY";
    #[classattr]
    const MIXED_SRID: i32 = -1;
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// `GeographyType.__new__(GeographyType)`, of which pickle's `copyreg.__newobj__` makes the
    /// type that PySpark pickled, is a type of the default SRID until `__setstate__` sets it. The
    /// class shows the signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(srid)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        let srid = Self::DEFAULT_SRID;
        Ok(init_chain!(
            DataType::Geography { srid },
            PyGeographyType { srid },
            [PyAtomicType, PySpatialType],
            value
        ))
    }
    /// As PySpark's `__init__(self, srid)`: `"ANY"` for a mixed SRID, or an SRID of a
    /// geographic coordinate reference system that PySpark knows.
    fn __init__(slf: &Bound<'_, Self>, srid: &Bound<'_, PyAny>) -> PyResult<()> {
        let srid = spatial_type_srid(srid, true)?;
        slf.borrow_mut().srid = srid;
        sync_base(slf.as_any())
    }
    /// The SRID, as PySpark's `GeographyType.srid`: a plain attribute, of which an assignment
    /// makes the type of the SRID (see `attribute_srid`), also its JSON, which PySpark's writes
    /// of the private `_crs` and `_alg` that its constructor sets.
    #[getter]
    fn srid(&self) -> i32 {
        self.srid
    }
    #[setter]
    fn set_srid(slf: &Bound<'_, Self>, srid: &Bound<'_, PyAny>) -> PyResult<()> {
        let srid = attribute_srid(srid, true)?;
        slf.borrow_mut().srid = srid;
        sync_base(slf.as_any())
    }
    /// The type of the state that PySpark pickled, of `srid`, `_crs` and `_alg`, which pickle
    /// sets as attributes (see `set_pickled_state`): the type is of the `srid`, whose CRS and
    /// algorithm the JSON names, as PySpark's constructor sets `_crs` and `_alg` of it.
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        set_pickled_state(slf.as_any(), state, |key, _| {
            Ok(matches!(key, "_crs" | "_alg"))
        })
    }
    /// As PySpark's `GeographyType.toInternal(geography)`: the dict of the SRID and the WKB of a
    /// `Geography` (see `_to_internal`).
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        geography: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", geography)
    }
    /// Of its JSON, and of a Python subclass, of its class and its `__dict__`, which holds the
    /// `_crs` of the SRID and the `_alg` that PySpark's constructor sets and its `jsonValue` reads
    /// (see `reduce_of`).
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let srid = slf.borrow().srid;
        let dt = DataType::Geography { srid };
        reduce_of(slf, &dt.json(), || {
            let (crs, alg) = spatial_json_parts(&dt, "geography");
            Ok(vec![
                ("srid", srid.into_pyobject(py)?.into_any()),
                ("_crs", crs.into_pyobject(py)?.into_any()),
                ("_alg", alg.into_pyobject(py)?.into_any()),
            ])
        })
    }
    /// As PySpark's `GeographyType._from_crs(crs, alg)`, of the JSON of a geography type:
    /// `ST_INVALID_ALGORITHM_VALUE` of another algorithm than `SPHERICAL`, the mixed SRID of
    /// `SRID:ANY` of any case, or else the SRID of a geographic CRS of `geo_utils`, and
    /// `ST_INVALID_CRS_VALUE` of another.
    #[classmethod]
    fn _from_crs<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        crs: &Bound<'py, PyAny>,
        alg: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        py.import("pyspark.sql.types")?
            .getattr("_geography_type_from_crs")?
            .call1((crs, alg))
    }
    fn __repr__(&self) -> String {
        match self.srid {
            -1 => "GeographyType(ANY)".to_string(),
            srid => format!("GeographyType({srid})"),
        }
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        DataType::Geography { srid: self.srid }.simple_string()
    }
}

/// The CRS and, of a geography, the algorithm that the JSON of the spatial type `data_type`, of
/// the name `name`, holds: `geometry(<crs>)` and `geography(<crs>, <alg>)`, as PySpark writes them
/// of its private `_crs` and `_alg`.
fn spatial_json_parts(data_type: &DataType, name: &str) -> (String, String) {
    let json = data_type.json_value();
    let text = json.as_str().unwrap_or_default();
    let inner = text
        .strip_prefix(name)
        .and_then(|rest| rest.strip_prefix('('))
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or_default();
    match inner.split_once(", ") {
        Some((crs, alg)) => (crs.to_string(), alg.to_string()),
        None => (inner.to_string(), String::new()),
    }
}

/// The SRID of a spatial type that PySpark constructs from `srid`: `"ANY"` gives the mixed
/// SRID, and an integer must be an SRID that PySpark's spatial reference system mappers know,
/// 0, 3857 and 4326 for a geometry and 4326 for a geography in PySpark 4.3.0. Anything else
/// raises `ST_INVALID_SRID_VALUE`, as in PySpark.
pub(crate) fn spatial_type_srid(srid: &Bound<'_, PyAny>, geographic: bool) -> PyResult<i32> {
    use pyo3::types::{PyInt, PyString};
    if let Ok(name) = srid.cast::<PyString>() {
        if name.to_cow()? == "ANY" {
            return Ok(spark_connect::types::SpatialType::MIXED_SRID);
        }
    }
    if srid.is_instance_of::<PyInt>() {
        if let Ok(value) = srid.extract::<i64>() {
            if pyspark_takes_srid(value, geographic) {
                return Ok(value as i32);
            }
        }
    }
    let srid_str = srid.str()?.to_string();
    Err(crate::column::raise_pyspark(
        srid.py(),
        "IllegalArgumentException",
        "ST_INVALID_SRID_VALUE",
        &[("srid", &srid_str)],
    ))
}

/// The SRID `srid` that is set as the `srid` of a spatial type of the given kind, a plain
/// attribute in PySpark, of any value. The core type holds an SRID that PySpark's types hold: the
/// mixed SRID -1, of which PySpark's `GeometryType("ANY")` is, or a value that the constructor
/// takes, of which another raises its `ST_INVALID_SRID_VALUE` (see [`spatial_type_srid`]).
fn attribute_srid(srid: &Bound<'_, PyAny>, geographic: bool) -> PyResult<i32> {
    use spark_connect::types::SpatialType;
    if srid.is_instance_of::<pyo3::types::PyInt>()
        && srid.extract::<i64>().ok() == Some(i64::from(SpatialType::MIXED_SRID))
    {
        return Ok(SpatialType::MIXED_SRID);
    }
    spatial_type_srid(srid, geographic)
}

/// Whether PySpark 4.3.0 takes the SRID `srid` in a spatial type of the given kind: its
/// `geo_utils` maps 0, 3857 and 4326 of a geometry, and 4326 of a geography, and no other SRID.
fn pyspark_takes_srid(srid: i64, geographic: bool) -> bool {
    if geographic {
        srid == 4326
    } else {
        matches!(srid, 0 | 3857 | 4326)
    }
}

/// Whether PySpark's conversion of a schema from its proto builds the spatial type of the given
/// kind of the SRID `srid` that the server sent, with `GeometryType(srid)` or
/// `GeographyType(srid)`: the mixed SRID, or an SRID that PySpark takes.
pub(crate) fn pyspark_converts_srid(srid: i32, geographic: bool) -> bool {
    srid == spark_connect::types::SpatialType::MIXED_SRID
        || pyspark_takes_srid(i64::from(srid), geographic)
}

/// The `ST_INVALID_SRID_VALUE` that PySpark raises for the SRID `srid` of a spatial type of a
/// schema that it does not convert (see [`pyspark_converts_srid`]).
pub(crate) fn invalid_schema_srid(py: Python<'_>, srid: i32) -> PyErr {
    crate::column::raise_pyspark(
        py,
        "IllegalArgumentException",
        "ST_INVALID_SRID_VALUE",
        &[("srid", &srid.to_string())],
    )
}

/// The SRID of a spatial type that the server sent in a schema, as PySpark's schema conversion
/// builds it: the mixed SRID, or an SRID that PySpark takes, and `ST_INVALID_SRID_VALUE` for
/// another one, which PySpark raises when it reads the schema.
fn schema_spatial_srid(py: Python<'_>, srid: i32, geographic: bool) -> PyResult<i32> {
    if pyspark_converts_srid(srid, geographic) {
        return Ok(srid);
    }
    Err(invalid_schema_srid(py, srid))
}

/// Parse a non-UDT DataType's Spark JSON (e.g. `"string collate UTF8_LCASE"`,
/// `"geography(OGC:CRS84, SPHERICAL)"`, or a nested struct/array/map object) with the core
/// grammar, into its concrete Python type object. `pyspark.sql.types._parse_datatype_json_value`
/// delegates its atomic and parameterized string forms here, so the drop-in keeps one JSON
/// grammar (the core's) instead of a second parser in Python. A nanosecond timestamp
/// (`timestamp_ntz(9)`) is not a JSON name of PySpark, whose parser has no such type: a
/// `ValueError`, which the shim raises as `CANNOT_PARSE_DATATYPE`.
#[pyfunction]
pub fn parse_datatype_json(py: Python<'_>, json: &str) -> PyResult<Py<PyAny>> {
    let dt = DataType::from_json_str(json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    if crate::dataframe::nanosecond_timestamp_text(&dt).is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "not a data type of PySpark: {json}"
        )));
    }
    data_type_to_py(py, &dt)
}

/// Convert a Python DataType object (any of the type classes) or a DDL string into
/// the core `DataType`. Mirrors pyspark accepting `Union[DataType, str]` everywhere.
/// Materialize a core [`DataType`] into its concrete Python type object — a real
/// `StructType`/`IntegerType`/... with the correct MRO — the inverse of
/// [`py_to_data_type`]. Used by `df.schema`, `StructField.dataType`, and `StructType`
/// field access so schema introspection yields the proper classes (not a bare `DataType`).
/// Whether `data_type` has a user-defined type.
pub(crate) fn has_udt(data_type: &DataType) -> bool {
    match data_type {
        DataType::Udt { .. } => true,
        DataType::Array { element_type, .. } => has_udt(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_udt(key_type) || has_udt(value_type),
        DataType::Struct { fields } => fields.iter().any(|field| has_udt(&field.data_type)),
        _ => false,
    }
}

/// Whether `data_type` holds a type that the client keeps as its Python object, whose core type is
/// only its JSON: a user-defined type, or a type that it lowered by its JSON (`Unparsed`), a
/// `StructField` as a type or a type of its class only. Two such types of the same JSON, such as
/// `UserDefinedType`s of other parameters, are told apart only by their objects.
fn holds_objects(data_type: &DataType) -> bool {
    match data_type {
        DataType::Udt { .. } | DataType::Unparsed { .. } => true,
        DataType::Array { element_type, .. } => holds_objects(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => holds_objects(key_type) || holds_objects(value_type),
        DataType::Struct { fields } => fields.iter().any(|field| holds_objects(&field.data_type)),
        _ => false,
    }
}

/// Whether a type that the type object `obj` keeps as its object (see `KeptType`), at any depth,
/// holds of `holds`: the field of a struct, the type of a field, the element of an array, or the
/// key or the value of a map, which `holds` reads, as their own children.
fn keeps_object_of(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    holds: fn(Python<'_>, &Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<bool> {
    let kept_holds = |kept: Option<Py<PyAny>>| -> PyResult<bool> {
        match kept {
            Some(object) => holds(py, object.bind(py)),
            None => Ok(false),
        }
    };
    if let Ok(field) = obj.cast::<PyStructField>() {
        let kept = field.borrow().data_type_object.kept(py);
        return kept_holds(kept);
    }
    if let Ok(array) = obj.cast::<PyArrayType>() {
        let kept = array.borrow().element_object.kept(py);
        return kept_holds(kept);
    }
    if let Ok(map) = obj.cast::<PyMapType>() {
        let (key, value) = {
            let map = map.borrow();
            (map.key_object.kept(py), map.value_object.kept(py))
        };
        return Ok(kept_holds(key)? || kept_holds(value)?);
    }
    if let Ok(st) = obj.cast::<PyStructType>() {
        let fields = st.borrow().fields.clone_ref(py);
        for item in fields.bind(py).iter() {
            if holds(py, &item)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Whether the type object `obj` keeps, at any depth of the types that it keeps as their objects
/// (see `keeps_object_of`), a type of a Python subclass of a type class (see
/// [`is_subclass_object`]). Its core type is that of the class, so such a type is compared,
/// written and copied of its object, as PySpark's `__dict__` holds it.
fn keeps_subclass_objects(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    keeps_object_of(py, obj, holds_subclass_object)
}

/// Whether `obj` is a type of a Python subclass of a type class, or keeps one (see
/// [`keeps_subclass_objects`]).
fn holds_subclass_object(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_subclass_object(obj) || keeps_subclass_objects(py, obj)?)
}

/// Whether the child kept as `kept`, if it is, is or keeps a type of a Python subclass of a type
/// class.
fn kept_holds_subclass_object(py: Python<'_>, kept: &KeptType) -> PyResult<bool> {
    match kept.kept(py) {
        Some(object) => holds_subclass_object(py, object.bind(py)),
        None => Ok(false),
    }
}

/// Whether the type object `obj` is, or keeps at any depth of the types that it keeps as their
/// objects (see `keeps_object_of`), a field of metadata. PySpark's field keeps its metadata dict
/// as it was given, which `==` compares as Python compares the dicts, and `json()` writes of
/// `json.dumps`: so does the drop-in's of the dict (see `keep_flags_in_json`), whose values the
/// core's JSON does not hold as they are (a tuple, a key that is not a str, an int of more than
/// 64 bits, the text of a float).
fn holds_metadata(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(field) = obj.cast::<PyStructField>() {
        if !field.borrow().metadata.bind(py).is_empty() {
            return Ok(true);
        }
    }
    keeps_object_of(py, obj, holds_metadata)
}

/// The `simpleString()` of the type of a field, of the element of an array, or of the key or the
/// value of a map, as PySpark's of the type object that it holds: the object's own when it is
/// kept, so that a type that the client keeps as its object is named by it, which its JSON does
/// not tell apart from another of the same JSON; else of its core type.
fn child_simple_string(py: Python<'_>, kept: &KeptType, snapshot: &DataType) -> PyResult<String> {
    match kept.kept(py) {
        Some(object) => object
            .bind(py)
            .call_method0(pyo3::intern!(py, "simpleString"))?
            .extract(),
        None => simple_string_with_udts(py, snapshot),
    }
}

/// As PySpark's `_as_nullable` of the type object `obj`: of an array, a map or a struct, one of
/// the same types made nullable in turn and of its nullable flags set true, as `toNullable`
/// returns it; any other type is the object itself, as PySpark's base `_as_nullable` returns
/// `self`, so that a `UserDefinedType` or a type of its class only stays the object it was.
fn as_nullable<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    if obj.is_instance_of::<PyArrayType>() {
        let element = as_nullable(&obj.getattr(pyo3::intern!(py, "elementType"))?)?;
        return py.get_type::<PyArrayType>().call1((element, true));
    }
    if obj.is_instance_of::<PyMapType>() {
        let key = as_nullable(&obj.getattr(pyo3::intern!(py, "keyType"))?)?;
        let value = as_nullable(&obj.getattr(pyo3::intern!(py, "valueType"))?)?;
        return py.get_type::<PyMapType>().call1((key, value, true));
    }
    if obj.is_instance_of::<PyStructType>() {
        let fields = PyList::empty(py);
        for field in obj.getattr(pyo3::intern!(py, "fields"))?.try_iter()? {
            let field = field?;
            let data_type = as_nullable(&field.getattr(pyo3::intern!(py, "dataType"))?)?;
            fields.append(py.get_type::<PyStructField>().call1((
                field.getattr(pyo3::intern!(py, "name"))?,
                data_type,
                true,
                field.getattr(pyo3::intern!(py, "metadata"))?,
            ))?)?;
        }
        return py.get_type::<PyStructType>().call1((fields,));
    }
    Ok(obj.clone())
}

/// Whether the objects that the type objects `a` and `b`, of one class and of the equal core
/// types `core`, hold at each place whose type holds an object (see `holds_objects`), or whose
/// object of either is or keeps a type of a Python subclass (see `holds_subclass_object`), are
/// equal, as PySpark's comparison of the `__dict__`s compares the type objects that they hold: a
/// field of a struct by the field, and the type of a field, the element of an array and the key
/// and the value of a map by its object, kept, or else made of the core type. A `StructField`
/// compares its type so itself (see its `__eq__`).
fn held_objects_eq(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, core: &DataType) -> PyResult<bool> {
    let py = a.py();
    let mut pairs: Vec<(Bound<'_, PyAny>, Bound<'_, PyAny>)> = Vec::new();
    match core {
        DataType::Struct { fields } => {
            let (Ok(x), Ok(y)) = (a.cast::<PyStructType>(), b.cast::<PyStructType>()) else {
                return Ok(true);
            };
            let (x, y) = (
                x.borrow().fields.clone_ref(py),
                y.borrow().fields.clone_ref(py),
            );
            let (x, y) = (x.bind(py), y.bind(py));
            for (i, field) in fields.iter().enumerate() {
                let (fx, fy) = (x.get_item(i)?, y.get_item(i)?);
                if holds_objects(&field.data_type)
                    || holds_subclass_object(py, &fx)?
                    || holds_subclass_object(py, &fy)?
                {
                    pairs.push((fx, fy));
                }
            }
        }
        DataType::Array { element_type, .. } => {
            let (Ok(x), Ok(y)) = (a.cast::<PyArrayType>(), b.cast::<PyArrayType>()) else {
                return Ok(true);
            };
            let (x, y) = (x.borrow(), y.borrow());
            if holds_objects(element_type)
                || kept_holds_subclass_object(py, &x.element_object)?
                || kept_holds_subclass_object(py, &y.element_object)?
            {
                pairs.push((
                    x.element_object.object(py, &x.element_type)?.into_bound(py),
                    y.element_object.object(py, &y.element_type)?.into_bound(py),
                ));
            }
        }
        DataType::Map {
            key_type,
            value_type,
            ..
        } => {
            let (Ok(x), Ok(y)) = (a.cast::<PyMapType>(), b.cast::<PyMapType>()) else {
                return Ok(true);
            };
            let (x, y) = (x.borrow(), y.borrow());
            if holds_objects(key_type)
                || kept_holds_subclass_object(py, &x.key_object)?
                || kept_holds_subclass_object(py, &y.key_object)?
            {
                pairs.push((
                    x.key_object.object(py, &x.key_type)?.into_bound(py),
                    y.key_object.object(py, &y.key_type)?.into_bound(py),
                ));
            }
            if holds_objects(value_type)
                || kept_holds_subclass_object(py, &x.value_object)?
                || kept_holds_subclass_object(py, &y.value_object)?
            {
                pairs.push((
                    x.value_object.object(py, &x.value_type)?.into_bound(py),
                    y.value_object.object(py, &y.value_type)?.into_bound(py),
                ));
            }
        }
        _ => {}
    }
    for (x, y) in pairs {
        if !x.eq(y)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether the collected values of `data_type` need the converters of
/// `pyspark.sql.types._udt_field_converters`: of a user-defined type, which deserializes them,
/// or of a GEOMETRY or a GEOGRAPHY, which makes their Python objects.
pub(crate) fn has_converted_values(data_type: &DataType) -> bool {
    match data_type {
        DataType::Udt { .. } | DataType::Geometry { .. } | DataType::Geography { .. } => true,
        DataType::Array { element_type, .. } => has_converted_values(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_converted_values(key_type) || has_converted_values(value_type),
        DataType::Struct { fields } => fields
            .iter()
            .any(|field| has_converted_values(&field.data_type)),
        _ => false,
    }
}

/// The `simpleString()` of `data_type` as PySpark's: of the Python class of each of its
/// user-defined types, such as `vector` of a `VectorUDT`, which the Rust core does not know.
pub(crate) fn simple_string_with_udts(py: Python<'_>, data_type: &DataType) -> PyResult<String> {
    Ok(match data_type {
        DataType::Udt { .. } => data_type_to_py(py, data_type)?
            .bind(py)
            .call_method0("simpleString")?
            .extract()?,
        DataType::Unparsed { data_type_string } => match lowered_object(py, data_type_string)? {
            Some(object) => object.call_method0("simpleString")?.extract()?,
            None => data_type.simple_string(),
        },
        DataType::Array { element_type, .. } => {
            format!("array<{}>", simple_string_with_udts(py, element_type)?)
        }
        DataType::Map {
            key_type,
            value_type,
            ..
        } => format!(
            "map<{},{}>",
            simple_string_with_udts(py, key_type)?,
            simple_string_with_udts(py, value_type)?
        ),
        DataType::Struct { fields } => {
            let fields = fields
                .iter()
                .map(|field| {
                    Ok(format!(
                        "{}:{}",
                        field.name,
                        simple_string_with_udts(py, &field.data_type)?
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?;
            format!("struct<{}>", fields.join(","))
        }
        _ => data_type.simple_string(),
    })
}

pub(crate) fn data_type_to_py(py: Python<'_>, dt: &DataType) -> PyResult<Py<PyAny>> {
    if matches!(
        dt,
        DataType::Struct { .. } | DataType::Array { .. } | DataType::Map { .. }
    ) {
        if let Some(filled) = with_udt_sql_types(py, dt) {
            return data_type_to_py(py, &filled);
        }
    }
    macro_rules! obj {
        ($init:expr) => {
            Py::new(py, $init)?.into_any()
        };
    }
    // The one instance of a type of PySpark's `DataTypeSingleton`.
    macro_rules! singleton {
        ($ty:ty) => {
            <$ty>::new(
                &py.get_type::<$ty>(),
                &pyo3::types::PyTuple::empty(py),
                None,
            )?
            .into_any()
        };
    }
    let o = match dt {
        DataType::Null => singleton!(PyNullType),
        DataType::Boolean => {
            singleton!(PyBooleanType)
        }
        DataType::Byte => singleton!(PyByteType),
        DataType::Short => singleton!(PyShortType),
        DataType::Integer => {
            singleton!(PyIntegerType)
        }
        DataType::Long => singleton!(PyLongType),
        DataType::Float => singleton!(PyFloatType),
        DataType::Double => {
            singleton!(PyDoubleType)
        }
        DataType::Decimal { precision, scale } => {
            let (precision, scale) = (*precision, *scale);
            obj!(init_chain!(
                DataType::Decimal { precision, scale },
                PyDecimalType::of(py, precision, scale),
                [PyAtomicType, PyNumericType, PyFractionalType],
                value
            ))
        }
        DataType::String { collation } => obj!(init_chain!(
            DataType::String {
                collation: collation.clone()
            },
            PyStringType {
                collation: collation.clone()
            },
            [PyAtomicType],
            value
        )),
        DataType::Binary => {
            singleton!(PyBinaryType)
        }
        DataType::Date => singleton!(PyDateType),
        DataType::Timestamp => {
            singleton!(PyTimestampType)
        }
        DataType::TimestampNtz => {
            singleton!(PyTimestampNTZType)
        }
        DataType::Array {
            element_type,
            contains_null,
        } => {
            let (et, contains_null) = ((**element_type).clone(), *contains_null);
            obj!(init_chain!(
                DataType::Array {
                    element_type: Box::new(et.clone()),
                    contains_null,
                },
                PyArrayType {
                    element_type: et,
                    contains_null,
                    element_object: KeptType::default(),
                    contains_null_object: flag_of_bool(py, contains_null),
                },
                [],
                value
            ))
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let (kt, vt, value_contains_null) = (
                (**key_type).clone(),
                (**value_type).clone(),
                *value_contains_null,
            );
            obj!(init_chain!(
                DataType::Map {
                    key_type: Box::new(kt.clone()),
                    value_type: Box::new(vt.clone()),
                    value_contains_null,
                },
                PyMapType {
                    key_type: kt,
                    value_type: vt,
                    value_contains_null,
                    key_object: KeptType::default(),
                    value_object: KeptType::default(),
                    value_contains_null_object: flag_of_bool(py, value_contains_null),
                },
                [],
                value
            ))
        }
        DataType::Struct { fields } => py_new_struct(py, fields.clone())?.into_any(),
        DataType::Char { length } => {
            let length = *length;
            obj!(init_chain!(
                DataType::Char { length },
                PyCharType { length },
                [PyAtomicType],
                value
            ))
        }
        DataType::Varchar { length } => {
            let length = *length;
            obj!(init_chain!(
                DataType::Varchar { length },
                PyVarcharType { length },
                [PyAtomicType],
                value
            ))
        }
        DataType::Time { precision } => {
            let precision = *precision;
            obj!(init_chain!(
                DataType::Time { precision },
                PyTimeType { precision },
                [PyAtomicType, PyDatetimeType, PyAnyTimeType],
                value
            ))
        }
        DataType::TimestampNtzNanos { precision } => {
            let precision = *precision;
            obj!(init_chain!(
                DataType::TimestampNtzNanos { precision },
                PyTimestampNTZNanosType { precision },
                [PyAtomicType, PyDatetimeType],
                value
            ))
        }
        DataType::TimestampLtzNanos { precision } => {
            let precision = *precision;
            obj!(init_chain!(
                DataType::TimestampLtzNanos { precision },
                PyTimestampLTZNanosType { precision },
                [PyAtomicType, PyDatetimeType],
                value
            ))
        }
        DataType::CalendarInterval => {
            singleton!(PyCalendarIntervalType)
        }
        DataType::YearMonthInterval {
            start_field,
            end_field,
        } => {
            let (start_field, end_field) = (*start_field, *end_field);
            obj!(init_chain!(
                DataType::YearMonthInterval {
                    start_field,
                    end_field,
                },
                PyYearMonthIntervalType {
                    start_field,
                    end_field,
                },
                [PyAtomicType, PyAnsiIntervalType],
                value
            ))
        }
        DataType::DayTimeInterval {
            start_field,
            end_field,
        } => {
            let (start_field, end_field) = (*start_field, *end_field);
            obj!(init_chain!(
                DataType::DayTimeInterval {
                    start_field,
                    end_field,
                },
                PyDayTimeIntervalType {
                    start_field,
                    end_field,
                },
                [PyAtomicType, PyAnsiIntervalType],
                value
            ))
        }
        DataType::Variant => obj!(init_chain!(
            DataType::Variant,
            PyVariantType,
            [PyAtomicType]
        )),
        DataType::Geometry { srid } => {
            let srid = schema_spatial_srid(py, *srid, false)?;
            obj!(init_chain!(
                DataType::Geometry { srid },
                PyGeometryType { srid },
                [PyAtomicType, PySpatialType],
                value
            ))
        }
        DataType::Geography { srid } => {
            let srid = schema_spatial_srid(py, *srid, true)?;
            obj!(init_chain!(
                DataType::Geography { srid },
                PyGeographyType { srid },
                [PyAtomicType, PySpatialType],
                value
            ))
        }
        // An unparsed DDL string has no concrete Python class; keep it as a bare `DataType`.
        // (The Python parser delegates JSON strings back here, so routing it there would loop.)
        // As PySpark Connect's, a DDL string that the server parses.
        DataType::Unparsed { data_type_string } => match lowered_object(py, data_type_string)? {
            Some(object) => object.unbind(),
            None => py
                .import("pyspark.sql.connect.types")?
                .getattr("UnparsedDataType")?
                .call1((data_type_string,))?
                .unbind(),
        },
        DataType::Udt { .. } => {
            // The user-defined type that the client lowered into `dt`; or else, as of the
            // schema of the server, one that re-imports its Python class, of the Python parser.
            let json = dt.json();
            match lowered_udts(py).get_item(&json)? {
                Some(udt) => udt.unbind(),
                None => py
                    .import("pyspark.sql.types")?
                    .getattr("_parse_datatype_json_string")?
                    .call1((json,))?
                    .unbind(),
            }
        }
    };
    Ok(o)
}

/// `dt` with the SQL type of each user-defined type of a Python class that lacks one, from the
/// class's `sqlType()`, or `None` if it has none to fill. The server sends no SQL type of a JVM type
/// with a Python class, such as `VectorUDT`, and PySpark's conversion of a schema of the server
/// makes the type of its class (`UserDefinedType.fromJson`), whose JSON has its `sqlType`: so the
/// JSON of the schema, and a schema made of it, have the `sqlType` as PySpark's have. A class that
/// does not import keeps its type as it is, as before.
fn with_udt_sql_types(py: Python<'_>, dt: &DataType) -> Option<DataType> {
    match dt {
        DataType::Udt {
            type_str,
            jvm_class,
            python_class: Some(python_class),
            serialized_python_class,
            sql_type: None,
        } => {
            let udt = py
                .import("pyspark.sql.types")
                .and_then(|types| types.getattr("_parse_datatype_json_string"))
                .and_then(|parse| parse.call1((dt.json(),)))
                .ok()?;
            let sql_type = udt
                .call_method0("sqlType")
                .and_then(|sql_type| py_to_data_type(&sql_type))
                .ok()?;
            Some(DataType::Udt {
                type_str: type_str.clone(),
                jvm_class: jvm_class.clone(),
                python_class: Some(python_class.clone()),
                serialized_python_class: serialized_python_class.clone(),
                sql_type: Some(Box::new(sql_type)),
            })
        }
        DataType::Array {
            element_type,
            contains_null,
        } => with_udt_sql_types(py, element_type).map(|element_type| DataType::Array {
            element_type: Box::new(element_type),
            contains_null: *contains_null,
        }),
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let (key, value) = (
                with_udt_sql_types(py, key_type),
                with_udt_sql_types(py, value_type),
            );
            if key.is_none() && value.is_none() {
                return None;
            }
            Some(DataType::Map {
                key_type: Box::new(key.unwrap_or_else(|| (**key_type).clone())),
                value_type: Box::new(value.unwrap_or_else(|| (**value_type).clone())),
                value_contains_null: *value_contains_null,
            })
        }
        DataType::Struct { fields } => {
            let mut filled = false;
            let fields = fields
                .iter()
                .map(|field| match with_udt_sql_types(py, &field.data_type) {
                    Some(data_type) => {
                        filled = true;
                        StructField {
                            data_type,
                            ..field.clone()
                        }
                    }
                    None => field.clone(),
                })
                .collect();
            filled.then_some(DataType::Struct { fields })
        }
        _ => None,
    }
}

/// The type of `schema`, a DataType or a DDL string, which the server parses, as PySpark
/// Connect's `_parse_ddl` does: so any DDL of Spark, with nested types, quoted names,
/// collations and the NOT NULL and COMMENT of fields, parses as Spark parses it.
pub(crate) fn schema_data_type(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    schema: &Bound<'_, PyAny>,
) -> PyResult<DataType> {
    if schema.is_instance_of::<pyo3::types::PyString>() {
        let ddl: String = schema.extract()?;
        let parsed = crate::interrupt::interruptible(py, session, || session.parse_ddl(&ddl))?;
        // As PySpark converts the parsed type from its proto, which fails for a nanosecond timestamp
        // and for a spatial type of an SRID that PySpark does not take.
        crate::dataframe::check_schema_conversion(py, Some(&parsed))?;
        return Ok(parsed);
    }
    py_to_data_type(schema)
}

/// The type of a constructor of a nested type (`ArrayType`, `MapType`, `StructField`), which
/// PySpark asserts to be a `DataType` before it reads it: a type string, or any other value, is an
/// `AssertionError` with PySpark's message, for the argument `name`.
fn nested_data_type(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<DataType> {
    let data_type_class = obj.py().import("pyspark.sql.types")?.getattr("DataType")?;
    if !obj.is_instance(&data_type_class)? {
        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
            "{name} {} should be an instance of {}",
            obj.str()?,
            data_type_class.str()?
        )));
    }
    py_to_data_type(obj)
}

pub(crate) fn py_to_data_type(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    if let Ok(g) = obj.extract::<PyRef<PyGeometryType>>() {
        return Ok(DataType::Geometry { srid: g.srid });
    }
    if let Ok(g) = obj.extract::<PyRef<PyGeographyType>>() {
        return Ok(DataType::Geography { srid: g.srid });
    }
    if obj.extract::<PyRef<PyNullType>>().is_ok() {
        return Ok(DataType::Null);
    }
    if obj.extract::<PyRef<PyBooleanType>>().is_ok() {
        return Ok(DataType::Boolean);
    }
    if obj.extract::<PyRef<PyByteType>>().is_ok() {
        return Ok(DataType::Byte);
    }
    if obj.extract::<PyRef<PyShortType>>().is_ok() {
        return Ok(DataType::Short);
    }
    if obj.extract::<PyRef<PyIntegerType>>().is_ok() {
        return Ok(DataType::Integer);
    }
    if obj.extract::<PyRef<PyLongType>>().is_ok() {
        return Ok(DataType::Long);
    }
    if obj.extract::<PyRef<PyFloatType>>().is_ok() {
        return Ok(DataType::Float);
    }
    if obj.extract::<PyRef<PyDoubleType>>().is_ok() {
        return Ok(DataType::Double);
    }
    if let Ok(s) = obj.extract::<PyRef<PyStringType>>() {
        return Ok(DataType::String {
            collation: s.collation.clone(),
        });
    }
    if obj.extract::<PyRef<PyBinaryType>>().is_ok() {
        return Ok(DataType::Binary);
    }
    if obj.extract::<PyRef<PyDateType>>().is_ok() {
        return Ok(DataType::Date);
    }
    if obj.extract::<PyRef<PyTimestampType>>().is_ok() {
        return Ok(DataType::Timestamp);
    }
    if obj.extract::<PyRef<PyTimestampNTZType>>().is_ok() {
        return Ok(DataType::TimestampNtz);
    }
    if let Ok(t) = obj.extract::<PyRef<PyTimestampNTZNanosType>>() {
        return Ok(DataType::TimestampNtzNanos {
            precision: t.precision,
        });
    }
    if let Ok(t) = obj.extract::<PyRef<PyTimestampLTZNanosType>>() {
        return Ok(DataType::TimestampLtzNanos {
            precision: t.precision,
        });
    }
    if let Ok(d) = obj.extract::<PyRef<PyDecimalType>>() {
        return Ok(DataType::Decimal {
            precision: d.precision,
            scale: d.scale,
        });
    }
    if let Ok(a) = obj.extract::<PyRef<PyArrayType>>() {
        return Ok(DataType::Array {
            element_type: Box::new(a.element_object.live(obj.py(), &a.element_type)?),
            contains_null: a.contains_null,
        });
    }
    if let Ok(m) = obj.extract::<PyRef<PyMapType>>() {
        return Ok(DataType::Map {
            key_type: Box::new(m.key_object.live(obj.py(), &m.key_type)?),
            value_type: Box::new(m.value_object.live(obj.py(), &m.value_type)?),
            value_contains_null: m.value_contains_null,
        });
    }
    if let Ok(st) = obj.extract::<PyRef<PyStructType>>() {
        return Ok(DataType::Struct {
            fields: st.current_fields(obj.py())?,
        });
    }
    if let Ok(c) = obj.extract::<PyRef<PyCharType>>() {
        return Ok(DataType::Char { length: c.length });
    }
    if let Ok(v) = obj.extract::<PyRef<PyVarcharType>>() {
        return Ok(DataType::Varchar { length: v.length });
    }
    if let Ok(t) = obj.extract::<PyRef<PyTimeType>>() {
        return Ok(DataType::Time {
            precision: t.precision,
        });
    }
    if obj.extract::<PyRef<PyCalendarIntervalType>>().is_ok() {
        return Ok(DataType::CalendarInterval);
    }
    if let Ok(i) = obj.extract::<PyRef<PyYearMonthIntervalType>>() {
        return Ok(DataType::YearMonthInterval {
            start_field: i.start_field,
            end_field: i.end_field,
        });
    }
    if let Ok(i) = obj.extract::<PyRef<PyDayTimeIntervalType>>() {
        return Ok(DataType::DayTimeInterval {
            start_field: i.start_field,
            end_field: i.end_field,
        });
    }
    if obj.extract::<PyRef<PyVariantType>>().is_ok() {
        return Ok(DataType::Variant);
    }
    // `UserDefinedType` is a pure-Python subclass of the base `DataType` (its jsonValue
    // cloudpickles the concrete class, so it can't be lowered into Rust). Detect it before
    // the bare-base fallback and build the core `Udt` from its jsonValue + sqlType().
    if let Ok(udt_cls) = obj
        .py()
        .import("pyspark.sql.types")
        .and_then(|m| m.getattr("UserDefinedType"))
    {
        if obj.is_instance(&udt_cls)? {
            let jv = obj.call_method0("jsonValue")?;
            let get = |k: &str| -> PyResult<Option<String>> {
                let v = jv.call_method1("get", (k,))?;
                if v.is_none() {
                    Ok(None)
                } else {
                    Ok(Some(v.extract()?))
                }
            };
            let sql_type = py_to_data_type(&obj.call_method0("sqlType")?)?;
            let udt = DataType::Udt {
                type_str: "udt".to_string(),
                jvm_class: get("class")?,
                python_class: get("pyClass")?,
                serialized_python_class: get("serializedClass")?,
                sql_type: Some(Box::new(sql_type)),
            };
            lowered_udts(obj.py()).set_item(udt.json(), obj)?;
            return Ok(udt);
        }
    }
    // A `StructField` as a type, which PySpark allows in a nested position: its JSON is the
    // type, and the object is kept by the JSON, as a user-defined type is (see `lowered_udts`).
    if let Ok(field) = obj.cast::<PyStructField>() {
        let json: String = field.call_method0("json")?.extract()?;
        lowered_udts(obj.py()).set_item(&json, obj)?;
        return Ok(DataType::Unparsed {
            data_type_string: json,
        });
    }
    // `UnparsedDataType` of `pyspark.sql.connect.types`, a pure-Python subclass of the base, is
    // a DDL string that the server parses, which PySpark sets in the string field of the proto of
    // the type as protobuf converts it: bytes of UTF-8 too, and protobuf's `TypeError` of another
    // value.
    if is_unparsed(obj)? {
        return Ok(DataType::Unparsed {
            data_type_string: crate::str_argument(&obj.getattr("data_type_string")?)?,
        });
    }
    // The bare base `DataType` is matched LAST: concrete subclasses (which all share the
    // PyDataType base) are handled above via their own fields, so a mutated StructType reflects
    // its live fields rather than the base's snapshot.
    if let Ok(d) = obj.cast::<PyDataType>() {
        if let Some(inner) = &d.borrow().inner {
            return Ok(inner.clone());
        }
        // A type of its class only (`DataType()`, `AtomicType()`, ...), which the core has no
        // type of: it is lowered by its JSON, as a StructField as a type is, and kept by it, so
        // that the array, map or struct that holds it names it as PySpark does, and the proto of
        // a plan refuses it, as PySpark's `pyspark_types_to_proto_types` does.
        let json: String = obj
            .call_method0(pyo3::intern!(obj.py(), "json"))?
            .extract()?;
        lowered_udts(obj.py()).set_item(&json, obj)?;
        return Ok(DataType::Unparsed {
            data_type_string: json,
        });
    }
    if let Ok(s) = obj.extract::<String>() {
        return DataType::from_ddl(&s).map_err(|e| {
            // The registry has no message for the class of a parse error, so its detail is kept.
            let message = match e.params.get("detail") {
                Some(detail) => format!("[{}] {detail}", e.error_class),
                None => e.message(),
            };
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "invalid type string '{s}': {message}"
            ))
        });
    }
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "expected a DataType or a DDL type string",
    ))
}

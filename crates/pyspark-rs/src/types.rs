//! PyO3 wrappers for Spark DataTypes.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyList;
use spark_connect::types::{DataType, Metadata, StructField, TreeFlag, TreeStep};

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

// Python wrapper for any DataType. This is the base of the type-class hierarchy
// (mirroring pyspark's `DataType`); concrete type classes extend it (directly or via
// the intermediate abstract bases below), so `isinstance(dt, DataType)` and the object
// model (simpleString/typeName/json/...) are inherited from here via `self.inner`.
#[doc = connect_doc!(DataType)]
#[pyclass(
    subclass,
    name = "DataType",
    module = "pyspark.sql.types",
    weakref,
    dict
)]
pub struct PyDataType {
    /// The core type of the class of the type, as it was made. A concrete class reads its type of
    /// its own fields, or, of a type of parameters, of the parameters of its `__dict__`
    /// (`py_to_data_type`), and not of this one, which stands for the kind of the class. `None`
    /// for a type of its class only, whose state is its `__dict__`, as PySpark's plain classes
    /// keep it: the base `DataType` itself, the abstract classes of the hierarchy
    /// (`AtomicType`, ...), and the Python subclasses of them, such as a `UserDefinedType`.
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
    // Of a type of a core type (`PyDataType::inner`), and whether it is of parameters, of one
    // read of the kind of its class.
    let parameters = match obj.cast::<PyDataType>() {
        Ok(data_type) => match data_type.borrow().inner.as_ref() {
            None => return false,
            Some(inner) => matches!(
                inner,
                DataType::Decimal { .. }
                    | DataType::String { .. }
                    | DataType::Char { .. }
                    | DataType::Varchar { .. }
                    | DataType::Time { .. }
                    | DataType::DayTimeInterval { .. }
                    | DataType::YearMonthInterval { .. }
                    | DataType::Geometry { .. }
                    | DataType::Geography { .. }
            ),
        },
        Err(_) => return false,
    };
    let ty = obj.get_type();
    if !native_type_classes(obj.py()).iter().any(|cls| ty.is(cls)) {
        return true;
    }
    // A type of parameters of its class whose `__dict__` is not that of its core type stands as
    // one of a subclass: its uses read its own methods, as PySpark's read its attributes.
    parameters && !has_canonical_parameters(obj)
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

/// The reduce of `obj`, a type of a Python subclass of a type class, or of a state that its JSON
/// does not keep (see `has_dict_state`), as PySpark's plain classes reduce at every protocol from
/// 2: `copyreg.__newobj__(cls)` and the `__dict__` of the type, which holds the attributes
/// `parameters` that PySpark's `__init__` sets, in its order, and the attributes that a user set,
/// of the `__dict__` of `obj`; no state when there are none. So a copy, a deep copy and a pickle
/// are of the class, which `cls.__new__(cls)` makes and `__setstate__` sets, and PySpark loads it
/// as its own class, also on the Python workers of the server, to which cloudpickle sends a class
/// of `__main__` by value.
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

/// Whether the type object `obj` is of a state that its JSON does not keep, and PySpark's
/// `__dict__` holds: attributes that a user set, in its `__dict__`, which PySpark's holds after
/// its parameters, or a `hasPrecisionInfo` other than the `True` of PySpark's `__init__`.
fn has_dict_state(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    // Of a type of parameters, whose `__dict__` holds them, one that is not that of its core type,
    // and of a field, an array or a map, one of other entries than its parameters.
    if is_parameter_type(obj) {
        return Ok(!has_canonical_parameters(obj));
    }
    if let Some(names) = container_parameters(obj) {
        return Ok(!has_canonical_entries(obj, names)?);
    }
    obj.getattr(pyo3::intern!(py, "__dict__"))?.is_truthy()
}

/// The reduce of a type of the class `T` itself, of its JSON `json` (see [`type_reduce`]), or else
/// of its class and PySpark's `__dict__` of it, of the attributes `parameters` (see
/// [`subclass_reduce`]): of a type of a Python subclass of `T`, and of a type of a state that its
/// JSON does not keep (see [`has_dict_state`]), such as attributes that a user set, so that its
/// copies, deep copies and pickles keep it, as PySpark's pickle of the `__dict__` does, also on the
/// Python workers of the server.
fn reduce_of<'py, T: pyo3::PyClass>(
    slf: &Bound<'py, T>,
    json: &str,
    parameters: impl FnOnce() -> PyResult<Vec<(&'static str, Bound<'py, PyAny>)>>,
) -> PyResult<Py<PyAny>> {
    if !is_native(slf) || has_dict_state(slf.as_any())? {
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

/// PySpark's `PySparkRuntimeError` of `INVALID_INTERVAL_CASTING`, of the `str` of each field.
fn invalid_interval_casting(py: Python<'_>, start_field: &str, end_field: &str) -> PyErr {
    crate::column::raise_pyspark(
        py,
        "PySparkRuntimeError",
        "INVALID_INTERVAL_CASTING",
        &[("start_field", start_field), ("end_field", end_field)],
    )
}

/// The kinds of the classes of parameters, whose state is their `__dict__`, as PySpark's plain
/// classes keep it (see [`parameter_kind`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParameterKind {
    Decimal,
    String,
    Char,
    Varchar,
    Time,
    DayTimeInterval,
    YearMonthInterval,
    Geometry,
    Geography,
}

/// The kind of the class of parameters of the type object `obj`, or None of another type: a
/// decimal, a string, a char, a varchar, a time, an interval of fields or a spatial type, of the
/// kind of its class (`PyDataType::inner`), of no check of its classes one at a time.
fn parameter_kind(obj: &Bound<'_, PyAny>) -> Option<ParameterKind> {
    let data_type = obj.cast::<PyDataType>().ok()?.try_borrow().ok()?;
    Some(match data_type.inner.as_ref()? {
        DataType::Decimal { .. } => ParameterKind::Decimal,
        DataType::String { .. } => ParameterKind::String,
        DataType::Char { .. } => ParameterKind::Char,
        DataType::Varchar { .. } => ParameterKind::Varchar,
        DataType::Time { .. } => ParameterKind::Time,
        DataType::DayTimeInterval { .. } => ParameterKind::DayTimeInterval,
        DataType::YearMonthInterval { .. } => ParameterKind::YearMonthInterval,
        DataType::Geometry { .. } => ParameterKind::Geometry,
        DataType::Geography { .. } => ParameterKind::Geography,
        _ => return None,
    })
}

/// The kind of an atomic type object of a concrete class, of the core type of its class
/// (`PyDataType::inner`): of no parameters, that type, or of parameters, their kind.
enum AtomicKind {
    Fixed(DataType),
    Parameters(ParameterKind),
}

/// The kind of the atomic type object `obj` of a concrete class (see [`AtomicKind`]), of one
/// read of its class's core type, or None of another type, such as a container, a type of its
/// class only or a type of a native state of its own (the timestamps of nanoseconds).
fn atomic_kind(obj: &Bound<'_, PyAny>) -> Option<AtomicKind> {
    let data_type = obj.cast::<PyDataType>().ok()?.try_borrow().ok()?;
    Some(match data_type.inner.as_ref()? {
        fixed @ (DataType::Null
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
        | DataType::Variant) => AtomicKind::Fixed(fixed.clone()),
        DataType::Decimal { .. } => AtomicKind::Parameters(ParameterKind::Decimal),
        DataType::String { .. } => AtomicKind::Parameters(ParameterKind::String),
        DataType::Char { .. } => AtomicKind::Parameters(ParameterKind::Char),
        DataType::Varchar { .. } => AtomicKind::Parameters(ParameterKind::Varchar),
        DataType::Time { .. } => AtomicKind::Parameters(ParameterKind::Time),
        DataType::DayTimeInterval { .. } => AtomicKind::Parameters(ParameterKind::DayTimeInterval),
        DataType::YearMonthInterval { .. } => {
            AtomicKind::Parameters(ParameterKind::YearMonthInterval)
        }
        DataType::Geometry { .. } => AtomicKind::Parameters(ParameterKind::Geometry),
        DataType::Geography { .. } => AtomicKind::Parameters(ParameterKind::Geography),
        _ => return None,
    })
}

/// The native class of the kind `kind` of parameters.
fn parameter_class(py: Python<'_>, kind: ParameterKind) -> Bound<'_, pyo3::types::PyType> {
    match kind {
        ParameterKind::Decimal => py.get_type::<PyDecimalType>(),
        ParameterKind::String => py.get_type::<PyStringType>(),
        ParameterKind::Char => py.get_type::<PyCharType>(),
        ParameterKind::Varchar => py.get_type::<PyVarcharType>(),
        ParameterKind::Time => py.get_type::<PyTimeType>(),
        ParameterKind::DayTimeInterval => py.get_type::<PyDayTimeIntervalType>(),
        ParameterKind::YearMonthInterval => py.get_type::<PyYearMonthIntervalType>(),
        ParameterKind::Geometry => py.get_type::<PyGeometryType>(),
        ParameterKind::Geography => py.get_type::<PyGeographyType>(),
    }
}

/// The parameter `name` of the type object `obj` of a class of parameters, a field, an array or a
/// map, as PySpark's methods read `self.<name>`: the entry of its `__dict__`, which holds the
/// parameters as PySpark's plain classes hold them, or a property or a `__getattr__` of a Python
/// subclass, of the `AttributeError` of a type whose `__init__` did not set it (see
/// [`crate::state::attr`]).
fn parameter<'py>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, pyo3::types::PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    // Of a type of its class itself, whose attribute is the entry of its `__dict__`, of no code of
    // a user that the check of the recursion stands for: a field, an array, a map, or a type of
    // parameters of the class of its kind.
    let exact = obj.is_exact_instance_of::<PyStructField>()
        || obj.is_exact_instance_of::<PyStructType>()
        || obj.is_exact_instance_of::<PyArrayType>()
        || obj.is_exact_instance_of::<PyMapType>()
        || parameter_kind(obj)
            .is_some_and(|kind| obj.get_type().is(parameter_class(obj.py(), kind)));
    if exact {
        return obj.getattr(name);
    }
    crate::state::attr(obj, name)
}

/// Python's `format % args`, of which PySpark's methods make the texts of the types of
/// parameters, as `"decimal(%d,%d)" % (self.precision, self.scale)`, of its errors.
fn percent<'py, A: IntoPyObject<'py>>(
    py: Python<'py>,
    format: &str,
    args: A,
) -> PyResult<Bound<'py, pyo3::types::PyString>> {
    Ok(pyo3::types::PyString::new(py, format)
        .rem(args)?
        .cast_into::<pyo3::types::PyString>()?)
}

/// Set the parameters `entries` of the type object `obj`, in their order, as PySpark's
/// `__init__` assigns them (`self.<name> = value`): through `setattr`, so to the `__dict__`, or
/// through a property or a `__setattr__` of a Python subclass (see [`crate::state::assign`]).
fn assign_parameters<'py>(
    obj: &Bound<'py, PyAny>,
    entries: &[(&str, Bound<'py, PyAny>)],
) -> PyResult<()> {
    let py = obj.py();
    for (name, value) in entries {
        obj.setattr(pyo3::types::PyString::intern(py, name), value)?;
    }
    Ok(())
}

/// The parameters of a type of parameters, their names and values in the order of PySpark's
/// `__init__` (see [`parameters_of`]).
type Parameters<'py> = Vec<(&'static str, Bound<'py, PyAny>)>;

/// The parameters of the core type `data_type` of a class of parameters, as PySpark's `__init__`
/// sets them of the arguments that make the type: their names and values in its order, as the
/// `__dict__` of the type holds them; None of a type of no parameters, or of a container.
fn parameters_of<'py>(py: Python<'py>, data_type: &DataType) -> PyResult<Option<Parameters<'py>>> {
    let int =
        |value: i32| -> PyResult<Bound<'py, PyAny>> { Ok(value.into_pyobject(py)?.into_any()) };
    let text =
        |value: &str| -> Bound<'py, PyAny> { pyo3::types::PyString::new(py, value).into_any() };
    Ok(Some(match data_type {
        DataType::Decimal { precision, scale } => vec![
            ("precision", int(*precision)?),
            ("scale", int(*scale)?),
            (
                "hasPrecisionInfo",
                pyo3::types::PyBool::new(py, true).to_owned().into_any(),
            ),
        ],
        DataType::String { collation } => vec![("collation", text(collation))],
        DataType::Char { length } | DataType::Varchar { length } => {
            vec![("length", int(*length)?)]
        }
        DataType::Time { precision } => vec![("precision", int(*precision)?)],
        DataType::DayTimeInterval {
            start_field,
            end_field,
        }
        | DataType::YearMonthInterval {
            start_field,
            end_field,
        } => vec![
            ("startField", int(*start_field)?),
            ("endField", int(*end_field)?),
        ],
        DataType::Geometry { srid } => {
            let (crs, _) = spatial_json_parts(data_type, "geometry");
            vec![("srid", int(*srid)?), ("_crs", text(&crs))]
        }
        DataType::Geography { srid } => {
            let (crs, alg) = spatial_json_parts(data_type, "geography");
            vec![
                ("srid", int(*srid)?),
                ("_crs", text(&crs)),
                ("_alg", text(&alg)),
            ]
        }
        _ => return Ok(None),
    }))
}

/// Whether the type object `obj` is of a class of parameters whose state is its `__dict__` (see
/// [`parameter_kind`]).
fn is_parameter_type(obj: &Bound<'_, PyAny>) -> bool {
    parameter_kind(obj).is_some()
}

/// The names of the parameters of the struct, the field, the array or the map `obj`, in the order
/// of PySpark's `__init__`, which its `__dict__` holds; None of another type.
fn container_parameters(obj: &Bound<'_, PyAny>) -> Option<&'static [&'static str]> {
    if obj.is_instance_of::<PyStructType>() {
        Some(&[
            "fields",
            "names",
            "_needConversion",
            "_needSerializeAnyField",
        ])
    } else if obj.is_instance_of::<PyStructField>() {
        Some(&["name", "dataType", "nullable", "metadata"])
    } else if obj.is_instance_of::<PyArrayType>() {
        Some(&["elementType", "containsNull"])
    } else if obj.is_instance_of::<PyMapType>() {
        Some(&["keyType", "valueType", "valueContainsNull"])
    } else {
        None
    }
}

/// Whether the type object `obj` keeps its state in its `__dict__`, as PySpark's plain classes do:
/// a type of parameters, a field, an array or a map, which pickle and copy as `copyreg.__newobj__`
/// of the class and the `__dict__` (see `type_reduce_ex`).
fn keeps_dict_state(obj: &Bound<'_, PyAny>) -> bool {
    container_parameters(obj).is_some() || is_parameter_type(obj)
}

/// Whether the `__dict__` of `obj`, a field, an array or a map, holds the parameters of `names`
/// alone, in their order, as PySpark's `__init__` makes it: a type of a value that a user set on
/// it, or of a parameter that was deleted, is of a state that its JSON does not keep (see
/// `has_dict_state`). The values are not read: the flags and the names that are not of the core's
/// kinds are written of their objects (see `has_other_flag`).
fn has_canonical_entries(obj: &Bound<'_, PyAny>, names: &[&str]) -> PyResult<bool> {
    let dict = obj.getattr(pyo3::intern!(obj.py(), "__dict__"))?;
    let Ok(dict) = dict.cast::<pyo3::types::PyDict>() else {
        return Ok(false);
    };
    if dict.len() != names.len() {
        return Ok(false);
    }
    for (key, name) in dict.keys().iter().zip(names) {
        match key.cast::<pyo3::types::PyString>() {
            Ok(key) if key.to_cow().is_ok_and(|key| key == *name) => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// Whether the `__dict__` of `obj`, a type of a class of parameters, is the one that PySpark's
/// `__init__` makes of the arguments of its core type (see [`parameters_of`]): the same entries in
/// the same order, of exact ints and strs of the values that the core type holds, so that the type
/// that its JSON makes is equal to it, as PySpark's `__dict__` comparison tells, and pickles as it.
/// Another, such as of a value that a user set (`t.precision = 10.5`), of a `hasPrecisionInfo`
/// that is not `True` or of an attribute of its own, is of its own methods (see
/// [`is_subclass_object`]).
fn has_canonical_parameters(obj: &Bound<'_, PyAny>) -> bool {
    use pyo3::intern;
    use pyo3::types::{PyDict, PyInt, PyString};
    let py = obj.py();
    let Some(kind) = parameter_kind(obj) else {
        return false;
    };
    let Ok(dict) = obj.getattr(intern!(py, "__dict__")) else {
        return false;
    };
    let Ok(dict) = dict.cast::<PyDict>() else {
        return false;
    };
    if matches!(kind, ParameterKind::Geometry | ParameterKind::Geography) {
        return has_canonical_spatial_parameters(dict, kind == ParameterKind::Geography);
    }
    // The names of the parameters of the class, in the order of its `__init__`, and the check of
    // the value of each, of the values that the core type holds as they are written.
    let exact_int = |value: &Bound<'_, PyAny>| {
        value.is_exact_instance_of::<PyInt>() && value.extract::<i32>().is_ok()
    };
    let field_of = |last: i32| {
        move |value: &Bound<'_, PyAny>| {
            value.is_exact_instance_of::<PyInt>()
                && value
                    .extract::<i32>()
                    .is_ok_and(|field| (0..=last).contains(&field))
        }
    };
    // Of an exact str; one that UTF-8 cannot encode fails the lowering of the type, of which the
    // uses that read the core type read the object instead (see `type_json_value`).
    let exact_str = |value: &Bound<'_, PyAny>| value.is_exact_instance_of::<PyString>();
    let true_flag = |value: &Bound<'_, PyAny>| value.is(pyo3::types::PyBool::new(py, true));
    type Check<'a> = &'a dyn Fn(&Bound<'_, PyAny>) -> bool;
    let (day_time, year_month) = (field_of(3), field_of(1));
    let checks: &[(&Bound<'_, PyString>, Check<'_>)] = match kind {
        ParameterKind::Decimal => &[
            (intern!(py, "precision"), &exact_int),
            (intern!(py, "scale"), &exact_int),
            (intern!(py, "hasPrecisionInfo"), &true_flag),
        ],
        ParameterKind::String => &[(intern!(py, "collation"), &exact_str)],
        ParameterKind::Char | ParameterKind::Varchar => &[(intern!(py, "length"), &exact_int)],
        ParameterKind::Time => &[(intern!(py, "precision"), &exact_int)],
        ParameterKind::DayTimeInterval => &[
            (intern!(py, "startField"), &day_time),
            (intern!(py, "endField"), &day_time),
        ],
        ParameterKind::YearMonthInterval => &[
            (intern!(py, "startField"), &year_month),
            (intern!(py, "endField"), &year_month),
        ],
        ParameterKind::Geometry | ParameterKind::Geography => return false,
    };
    if dict.len() != checks.len() {
        return false;
    }
    dict.iter()
        .zip(checks)
        .all(|((key, value), (name, check))| is_key(&key, name) && check(&value))
}

/// Whether the key `key` of a `__dict__` is the str `name`: the interned str of the name, which
/// `setattr` and pickle's state set, or else an equal str.
fn is_key(key: &Bound<'_, PyAny>, name: &Bound<'_, pyo3::types::PyString>) -> bool {
    key.is(name)
        || (key.is_exact_instance_of::<pyo3::types::PyString>() && key.eq(name).unwrap_or(false))
}

/// [`has_canonical_parameters`] of a spatial type, of the dict `dict` of its parameters: an SRID
/// that PySpark's constructor takes, of the `_crs` that it sets of it, and of a geography, of the
/// `_alg` that it sets.
fn has_canonical_spatial_parameters(
    dict: &Bound<'_, pyo3::types::PyDict>,
    geographic: bool,
) -> bool {
    use pyo3::intern;
    use pyo3::types::{PyInt, PyString};
    let py = dict.py();
    let names = [
        intern!(py, "srid"),
        intern!(py, "_crs"),
        intern!(py, "_alg"),
    ];
    let names = if geographic { &names[..] } else { &names[..2] };
    if dict.len() != names.len() {
        return false;
    }
    let mut values = Vec::with_capacity(names.len());
    for ((key, value), name) in dict.iter().zip(names) {
        if !is_key(&key, name) {
            return false;
        }
        values.push(value);
    }
    let Some(srid) = values[0]
        .is_exact_instance_of::<PyInt>()
        .then(|| values[0].extract::<i32>().ok())
        .flatten()
    else {
        return false;
    };
    if !pyspark_converts_srid(srid, geographic) {
        return false;
    }
    let (data_type, name) = if geographic {
        (DataType::Geography { srid }, "geography")
    } else {
        (DataType::Geometry { srid }, "geometry")
    };
    let (crs, alg) = spatial_json_parts(&data_type, name);
    let text_is = |value: &Bound<'_, PyAny>, expected: &str| {
        value.is_exact_instance_of::<PyString>()
            && value
                .cast::<PyString>()
                .is_ok_and(|text| text.to_cow().is_ok_and(|text| text == expected))
    };
    text_is(&values[1], &crs) && (!geographic || text_is(&values[2], &alg))
}

/// The core type of the type object `obj` of a class of parameters, of the parameters that its
/// `__dict__` holds (see [`parameter`]), as PySpark's `pyspark_types_to_proto_types` reads them
/// into the int and string fields of the proto of the type: of the errors of their conversion,
/// such as of a float (`'float' object cannot be interpreted as an integer`), and of the
/// `AttributeError` of a type whose `__init__` did not set them.
fn parameter_type_of<'py>(obj: &Bound<'py, PyAny>, kind: ParameterKind) -> PyResult<DataType> {
    use pyo3::intern;
    let py = obj.py();
    // As `parameter` reads them, of one check of the class of the type for all of them: of a type
    // of the class of its kind itself, the entries of its `__dict__`, and else as Python reads
    // them, of the check of the recursion.
    let exact = obj.get_type().is(parameter_class(py, kind));
    let read = |name: &Bound<'py, pyo3::types::PyString>| -> PyResult<Bound<'py, PyAny>> {
        if exact {
            obj.getattr(name)
        } else {
            crate::state::attr(obj, name)
        }
    };
    // As protobuf's (upb's) int32 field of the proto, of its `TypeError` of a bool and its
    // `ValueError` of an int out of the range (see `crate::stat_arguments::proto_int`).
    let int = |name: &Bound<'py, pyo3::types::PyString>, field: &str| -> PyResult<i32> {
        crate::stat_arguments::proto_int(&read(name)?, field)
    };
    Ok(match kind {
        ParameterKind::Decimal => {
            let scale = int(intern!(py, "scale"), "spark.connect.DataType.Decimal.scale")?;
            let precision = int(
                intern!(py, "precision"),
                "spark.connect.DataType.Decimal.precision",
            )?;
            DataType::Decimal { precision, scale }
        }
        ParameterKind::String => DataType::String {
            collation: crate::str_argument(&read(intern!(py, "collation"))?)?,
        },
        ParameterKind::Char => DataType::Char {
            length: int(intern!(py, "length"), "spark.connect.DataType.Char.length")?,
        },
        ParameterKind::Varchar => DataType::Varchar {
            length: int(
                intern!(py, "length"),
                "spark.connect.DataType.VarChar.length",
            )?,
        },
        ParameterKind::Time => DataType::Time {
            precision: int(
                intern!(py, "precision"),
                "spark.connect.DataType.Time.precision",
            )?,
        },
        ParameterKind::DayTimeInterval => {
            let start_field = int(
                intern!(py, "startField"),
                "spark.connect.DataType.DayTimeInterval.start_field",
            )?;
            let end_field = int(
                intern!(py, "endField"),
                "spark.connect.DataType.DayTimeInterval.end_field",
            )?;
            DataType::DayTimeInterval {
                start_field,
                end_field,
            }
        }
        ParameterKind::YearMonthInterval => {
            let start_field = int(
                intern!(py, "startField"),
                "spark.connect.DataType.YearMonthInterval.start_field",
            )?;
            let end_field = int(
                intern!(py, "endField"),
                "spark.connect.DataType.YearMonthInterval.end_field",
            )?;
            DataType::YearMonthInterval {
                start_field,
                end_field,
            }
        }
        ParameterKind::Geometry => DataType::Geometry {
            srid: int(intern!(py, "srid"), "spark.connect.DataType.Geometry.srid")?,
        },
        ParameterKind::Geography => DataType::Geography {
            srid: int(intern!(py, "srid"), "spark.connect.DataType.Geography.srid")?,
        },
    })
}

/// A type object of the class `cls` of parameters, of the core type `data_type`, as PySpark's
/// `__init__` makes it of the arguments of that type: of the parameters of [`parameters_of`],
/// set in its `__dict__`.
fn with_parameters<'py>(
    object: Bound<'py, PyAny>,
    data_type: &DataType,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(parameters) = parameters_of(object.py(), data_type)? {
        assign_parameters(&object, &parameters)?;
    }
    Ok(object)
}

/// The `__reduce_ex__` of the type object `obj`, as `object.__reduce_ex__` gives it of PySpark's
/// plain class: the `__reduce__` of a Python subclass of its own, which it calls; a `TypeError` of
/// a class of `__slots__` of no `__getstate__` of its own at the protocols 0 and 1, as copyreg's
/// `_reduce_ex` raises it; of a type whose state is its `__dict__`, a type of parameters, a field,
/// an array or a map (see `keeps_dict_state`), `copyreg.__newobj__` of the class and the state of
/// its `__getstate__`, also at the protocols 0 and 1, of which copyreg would make a pickle of a
/// class of a native base that does not load (see `crate::state::reduce_ex`); and else the
/// `__reduce__` of its class.
fn type_reduce_ex<'py>(obj: &Bound<'py, PyAny>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    let cls = obj.get_type();
    let natives = type_class_hierarchy(py);
    let mut native = py.get_type::<PyDataType>();
    for class in cls.mro().iter() {
        let class = class.cast_into::<pyo3::types::PyType>()?;
        if natives.iter().any(|known| class.is(known)) {
            native = class;
            break;
        }
    }
    let reduce = pyo3::intern!(py, "__reduce__");
    if !cls.getattr(reduce)?.is(native.getattr(reduce)?) {
        return obj.call_method0(reduce);
    }
    let getstate = pyo3::intern!(py, "__getstate__");
    if protocol < 2 && cls.getattr(getstate)?.is(native.getattr(getstate)?) {
        let slots = obj.getattr_opt(pyo3::intern!(py, "__slots__"))?;
        if slots.map(|slots| slots.is_truthy()).transpose()? == Some(true) {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "a class that defines __slots__ without defining __getstate__ cannot be pickled",
            ));
        }
    }
    if !keeps_dict_state(obj) {
        return obj.call_method0(reduce);
    }
    // As `object.__reduce_ex__` gives it, of five items, of no list or dict items.
    let newobj = py.import("copyreg")?.getattr("__newobj__")?;
    let args = pyo3::types::PyTuple::new(py, [cls.into_any()])?.into_any();
    let state = obj.call_method0(getstate)?;
    let none = py.None().into_bound(py);
    Ok(pyo3::types::PyTuple::new(py, [newobj, args, state, none.clone(), none])?.into_any())
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
    ($ty:ident, $name:tt, $parent:ty, [$($chain:ident),*] $(, { $($members:tt)* })?) => {
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
    ($ty:ident, $name:tt, $parent:ty, [$($chain:ident),*], singleton) => {
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
    (@class $ty:ident, $name:tt, $parent:ty) => {
        // `module` matches where the drop-in re-exports these (pyspark.sql.types), so
        // `__module__` reports the reference path instead of PyO3's default `builtins`. The
        // docstring is PySpark's of the class, of the name of the class, a string forwarded as a
        // `tt`, which `connect_doc!` matches.
        #[doc = connect_doc!($name)]
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

/// The core type of the type object `obj`, as [`py_to_data_type`] lowers it, or else a type that
/// stands for it, for a use that does not read its JSON. PySpark's `ArrayType`, `MapType`,
/// `StructField` and `StructType` keep the type objects that they are given as they are, and read
/// one only for a use that needs its JSON, such as `json()` or a plan, or its conversion, such
/// as `toInternal`. So a `UserDefinedType` whose `module()` or `sqlType()` is not implemented
/// stands as the null type, and an array, a map, a field or a struct that holds one stands as
/// the types that it holds, each so. Such a use reads the object again, which raises its error
/// then (see [`lower_type`]). An error of another object, or one that is not an `Exception`,
/// such as a `KeyboardInterrupt`, is raised. A type of more levels than Python's recursion limit,
/// such as one that holds itself (see [`type_level`]), stands as the null type too: each use
/// that reads its levels raises its `RecursionError`, as PySpark's do.
pub(crate) fn lenient_type(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    match stand_in_type(obj) {
        Err(error)
            if error.is_instance_of::<pyo3::exceptions::PyRecursionError>(obj.py())
                && obj.is_instance_of::<PyDataType>() =>
        {
            Ok(DataType::Null)
        }
        result => result,
    }
}

std::thread_local! {
    /// Whether the thread walks the types that stand for a type that does not lower (see
    /// [`stand_in_type`]).
    static STANDING_IN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The core type of the type object `obj`, or the type that stands for it, as [`lenient_type`]
/// gives them, of the `RecursionError` of a type of more levels than Python's recursion limit:
/// of its lowering, or of the levels of the types that stand for it, which are walks of its type
/// objects too (see [`type_level`]), as a type that holds itself and a type that does not lower
/// now has levels that no walk ends.
fn stand_in_type(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let py = obj.py();
    // Below a type that does not lower, its structs, arrays and maps stand of the types that they
    // hold, of no lowering of each again, which would read the levels below each as many times as
    // there are levels above them (cubic in the levels of a type made a level at a time); the other
    // types are lowered, of a stand-in if they do not lower now.
    let container = obj.is_instance_of::<PyStructType>()
        || obj.is_instance_of::<PyArrayType>()
        || obj.is_instance_of::<PyMapType>();
    if !(container && STANDING_IN.with(std::cell::Cell::get)) {
        let error = match py_to_data_type(obj) {
            Ok(data_type) => return Ok(data_type),
            Err(error) => error,
        };
        if !is_exception(py, &error)
            || !obj.is_instance_of::<PyDataType>()
            || error.is_instance_of::<pyo3::exceptions::PyRecursionError>(py)
        {
            return Err(error);
        }
    }
    /// The walk of the types that stand for a type, until it is dropped (see [`STANDING_IN`]).
    struct StandingIn(bool);
    impl Drop for StandingIn {
        fn drop(&mut self) {
            STANDING_IN.with(|standing_in| standing_in.set(self.0));
        }
    }
    let _standing_in = StandingIn(STANDING_IN.with(|standing_in| standing_in.replace(true)));
    type_level(py, || {
        if obj.is_instance_of::<PyArrayType>() {
            let element = parameter(obj, pyo3::intern!(py, "elementType"))?;
            return Ok(DataType::Array {
                element_type: Box::new(child_stand_in(&element)?),
                contains_null: flag_parameter(obj, pyo3::intern!(py, "containsNull"))?,
            });
        }
        if obj.is_instance_of::<PyMapType>() {
            let key = parameter(obj, pyo3::intern!(py, "keyType"))?;
            let value = parameter(obj, pyo3::intern!(py, "valueType"))?;
            return Ok(DataType::Map {
                key_type: Box::new(child_stand_in(&key)?),
                value_type: Box::new(child_stand_in(&value)?),
                value_contains_null: flag_parameter(obj, pyo3::intern!(py, "valueContainsNull"))?,
            });
        }
        if obj.is_instance_of::<PyStructType>() {
            return struct_stand_in(obj);
        }
        Ok(DataType::Null)
    })
}

/// The base core type of an array, which stands for the kind of its class (see
/// `PyDataType::inner`): its parameters are entries of its `__dict__`, which its lowering reads.
fn array_kind() -> DataType {
    DataType::Array {
        element_type: Box::new(DataType::Null),
        contains_null: true,
    }
}

/// The base core type of a map, as [`array_kind`] of an array.
fn map_kind() -> DataType {
    DataType::Map {
        key_type: Box::new(DataType::Null),
        value_type: Box::new(DataType::Null),
        value_contains_null: true,
    }
}

/// The type that stands for the child type object `child` of an array or a map, as
/// [`stand_in_type`] makes it, of a level of the types that stand for a type: of a value that is
/// not a data type, which PySpark's attribute keeps, the null type, of which the uses read the
/// object, as PySpark's do.
fn child_stand_in(child: &Bound<'_, PyAny>) -> PyResult<DataType> {
    if !child.is_instance_of::<PyDataType>() {
        return Ok(DataType::Null);
    }
    stand_in_type(child)
}

/// The core type of the type object `obj` for a tree string of `levels` levels of the types that
/// it holds, of a type of more levels than Python's recursion limit, such as one that holds
/// itself, whose lowering raises its `RecursionError`: of each struct, array and map to `levels`
/// levels, below which a struct is of no fields, and an array and a map of the null type, whose
/// kinds the string names and whose levels it does not hold, and of each other type as
/// [`lenient_type`] lowers it. A walk of the type objects (see [`type_level`]), which raises the
/// `RecursionError` of more levels than the limit, as PySpark's walk of a tree string does.
fn tree_stand_in(obj: &Bound<'_, PyAny>, levels: usize) -> PyResult<DataType> {
    let py = obj.py();
    let below = levels.saturating_sub(1);
    // The child `child` of the levels below.
    let child_object = |child: Bound<'_, PyAny>| {
        if levels == 0 {
            Ok(DataType::Null)
        } else {
            tree_stand_in(&child, below)
        }
    };
    type_level(py, || {
        if obj.is_instance_of::<PyStructType>() {
            let items = struct_fields(obj)?;
            let mut fields = Vec::new();
            if levels > 0 {
                for item in items.try_iter()? {
                    let field = as_struct_field(&item?)?;
                    let name = parameter(&field, pyo3::intern!(py, "name"))?;
                    let name = field_core_name(&name, false)?;
                    let data_type =
                        child_object(parameter(&field, pyo3::intern!(py, "dataType"))?)?;
                    fields.push(field_of_type(&field, name, data_type, true)?);
                }
            }
            return Ok(DataType::Struct { fields });
        }
        if obj.is_instance_of::<PyArrayType>() {
            let element = parameter(obj, pyo3::intern!(py, "elementType"))?;
            return Ok(DataType::Array {
                element_type: Box::new(child_object(element)?),
                contains_null: flag_parameter(obj, pyo3::intern!(py, "containsNull"))?,
            });
        }
        if obj.is_instance_of::<PyMapType>() {
            let key = parameter(obj, pyo3::intern!(py, "keyType"))?;
            let value = parameter(obj, pyo3::intern!(py, "valueType"))?;
            return Ok(DataType::Map {
                key_type: Box::new(child_object(key)?),
                value_type: Box::new(child_object(value)?),
                value_contains_null: flag_parameter(obj, pyo3::intern!(py, "valueContainsNull"))?,
            });
        }
        lenient_type(obj)
    })
}

/// Whether `error`, of the lowering of a type object, is an `Exception`, which a use that does
/// not read the type's JSON leaves to a use that does (see [`lenient_type`]): not a
/// `KeyboardInterrupt` nor a `SystemExit`.
fn is_exception(py: Python<'_>, error: &PyErr) -> bool {
    error.is_instance_of::<pyo3::exceptions::PyException>(py)
}

/// Construct a `StructType` Python object (with its full base chain) from core fields.
fn py_new_struct(py: Python<'_>, fields: &[StructField]) -> PyResult<Py<PyStructType>> {
    let objects = fields
        .iter()
        .map(|field| field_object(py, field, data_type_to_py(py, &field.data_type)?))
        .collect::<PyResult<Vec<_>>>()?;
    struct_of_list(py, PyList::new(py, objects)?)
}

/// The `StructType` of the list `fields` of `StructField` objects, as PySpark's
/// `StructType(fields)` makes it: the list is the struct's own, not a copy of it (see
/// `init_struct`).
fn struct_of_list(py: Python<'_>, fields: Bound<'_, PyList>) -> PyResult<Py<PyStructType>> {
    let object = Py::new(py, init_chain!(struct_kind(), PyStructType, [], value))?;
    init_struct(object.bind(py).as_any(), Some(fields.into_any()))?;
    Ok(object)
}

/// The `needConversion` of the data type object `obj`, as PySpark's: a struct and a user-defined
/// type always need a conversion, and an array or a map when one of its element types does. The
/// fields of a struct are not read, nor their metadata, so this does not grow with the struct.
/// The types that `obj` holds are read as PySpark's call of their `needConversion` reads them
/// (see `child_need_conversion`).
fn need_conversion_of_object(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    // A level of the type objects of a type of many levels, on a stack with room.
    type_level(py, || need_conversion_of_object_level(py, obj))
}

/// [`need_conversion_of_object`] of one level of a type object.
fn need_conversion_of_object_level(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<PyStructType>() {
        return Ok(true);
    }
    // A StructField, also as a type, needs a conversion as its dataType does, as PySpark's does.
    if obj.is_instance_of::<PyStructField>() {
        return field_need_conversion(obj);
    }
    // As PySpark's `self.elementType.needConversion()`, and of a map, of its key and its value.
    if obj.is_instance_of::<PyArrayType>() {
        let element = parameter(obj, pyo3::intern!(py, "elementType"))?;
        return child_need_conversion(py, &element);
    }
    if obj.is_instance_of::<PyMapType>() {
        let key = parameter(obj, pyo3::intern!(py, "keyType"))?;
        if child_need_conversion(py, &key)? {
            return Ok(true);
        }
        let value = parameter(obj, pyo3::intern!(py, "valueType"))?;
        return child_need_conversion(py, &value);
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
    // Of a value that is not a type too, which PySpark's attribute keeps, of the `AttributeError`
    // of one of no such method.
    if is_subclass_object(child) || !child.is_instance_of::<PyDataType>() {
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

/// The object of the type below `obj` by `step`: the type of a field of a struct, the element of
/// an array, or the key or the value of a map, of the entries of their `__dict__` (see
/// `parameter`).
fn kept_child<'py>(obj: &Bound<'py, PyAny>, step: TreeStep) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = obj.py();
    if obj.is_instance_of::<PyStructType>() {
        let TreeStep::Field(i) = step else {
            return Ok(None);
        };
        let item = struct_fields(obj)?.get_item(i)?;
        let field = as_struct_field(&item)?;
        return Ok(Some(parameter(&field, pyo3::intern!(py, "dataType"))?));
    }
    let name = match step {
        TreeStep::Element if obj.is_instance_of::<PyArrayType>() => {
            pyo3::intern!(py, "elementType")
        }
        TreeStep::Key if obj.is_instance_of::<PyMapType>() => pyo3::intern!(py, "keyType"),
        TreeStep::Value if obj.is_instance_of::<PyMapType>() => pyo3::intern!(py, "valueType"),
        _ => return Ok(None),
    };
    Ok(Some(parameter(obj, name)?))
}

/// The type object at `path` below the type `root`, of the types on the way (see `kept_child`);
/// None of a path that a type on the way has no child of.
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
/// `_get_jvm_type_name` writes it, of the object there, or else of the object that the client
/// lowered of the JSON; None for another type.
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

/// The name that a tree string writes of the type at `path` below the type `root` that stands as
/// the null type (see `lenient_type`): of a user-defined type, its `typeName()`, as PySpark's
/// `_get_jvm_type_name` writes it, and else that of a type of a Python subclass (see
/// [`subclass_tree_type_name`]), if `subclasses`; None for another type, such as a `NullType()`.
fn stand_in_tree_type_name(
    root: &Bound<'_, PyAny>,
    path: &[TreeStep],
    subclasses: bool,
) -> PyResult<Option<String>> {
    match tree_node(root, path)? {
        Some(node) if is_udt(&node)? => {
            Ok(Some(type_name_of(&node)?.str()?.to_cow()?.into_owned()))
        }
        _ if subclasses => subclass_tree_type_name(root, path),
        _ => Ok(None),
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
/// `treeString` writes it: `str(flag).lower()` of the flag as it was given. `None` of a path that
/// is not of such a type, whose flags are the core's bools.
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
            if !node.is_instance_of::<PyStructType>() {
                return Ok(None);
            }
            let item = struct_fields(&node)?.get_item(i)?;
            let field = as_struct_field(&item)?;
            parameter(&field, pyo3::intern!(py, "nullable"))?
        }
        TreeFlag::ContainsNull if node.is_instance_of::<PyArrayType>() => {
            parameter(&node, pyo3::intern!(py, "containsNull"))?
        }
        TreeFlag::ValueContainsNull if node.is_instance_of::<PyMapType>() => {
            parameter(&node, pyo3::intern!(py, "valueContainsNull"))?
        }
        TreeFlag::ContainsNull | TreeFlag::ValueContainsNull => return Ok(None),
    };
    Ok(Some(flag_object.str()?.to_cow()?.to_lowercase()))
}

/// The base of a `StructField` object, of no parameters. Its core type is not read: the type of
/// a field is read from the field (see `py_to_data_type`), which is a lowered type of the field's
/// JSON.
fn field_base() -> pyo3::PyClassInitializer<PyStructField> {
    init_chain!(
        DataType::Unparsed {
            data_type_string: String::new(),
        },
        PyStructField::empty(),
        [],
        value
    )
}

/// The Python object of the core field `field`, of the type object `data_type`, as PySpark's
/// `__init__` sets its parameters: its name, the type, its nullable flag and a dict of its
/// metadata.
fn field_object(
    py: Python<'_>,
    field: &StructField,
    data_type: Py<PyAny>,
) -> PyResult<Py<PyStructField>> {
    let object = Py::new(py, field_base())?;
    let metadata = metadata_map_to_py(py, &field.metadata)?;
    assign_parameters(
        object.bind(py).as_any(),
        &[
            (
                "name",
                pyo3::types::PyString::new(py, &field.name).into_any(),
            ),
            ("dataType", data_type.into_bound(py)),
            ("nullable", flag_of_bool(py, field.nullable).into_bound(py)),
            ("metadata", metadata),
        ],
    )?;
    Ok(object)
}

/// The core type of `obj` for a proto that the client sends, as PySpark's
/// `pyspark_types_to_proto_types`, which has no proto of a `StructField` as a type, nor of a
/// type of its class only, such as `AtomicType()`: it is the `UNSUPPORTED_OPERATION` of the
/// first one, `obj` itself or else the first in its fields.
pub(crate) fn py_to_proto_data_type(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let (lowered, maybe_foreign) = lower_noting_subclasses(obj, MetadataOf::Plan);
    if maybe_foreign {
        let mut foreign = ForeignFields::default();
        if foreign.holds(obj)? {
            return foreign_proto_type(obj, &mut foreign);
        }
    }
    let data_type = lowered?;
    // Of the str of the type, as PySpark's `f"data type {data_type}"`, a lone surrogate too.
    if let Some(unsendable) = unsendable_type(obj, &data_type)? {
        return Err(not_a_type_error(&unsendable));
    }
    Ok(data_type)
}

/// The core type that PySpark's `pyspark_types_to_proto_types` sends of the type object `obj`,
/// which holds a struct whose fields hold an element that is not a `StructField` (see
/// `holds_foreign_field`): of each element of the `fields` of a struct, its `name`, as protobuf's
/// string field takes it, its `dataType`, in turn, its `nullable`, as protobuf's bool field takes
/// it, and `json.dumps` of its `metadata` when there is one, in that order, of the errors of each;
/// of a field that holds no such struct, as it lowers; and of an array and a map, of their types
/// in turn and their flags.
fn foreign_proto_type(obj: &Bound<'_, PyAny>, foreign: &mut ForeignFields) -> PyResult<DataType> {
    // A level of a type of many levels, on a stack with room, of the bound of the walks.
    type_level(obj.py(), || foreign_proto_type_level(obj, foreign))
}

/// [`foreign_proto_type`] of one level: of a type that holds no such struct, of
/// [`py_to_proto_data_type`], and of a value that is not a data type, of PySpark's
/// `UNSUPPORTED_OPERATION`.
fn foreign_proto_type_level(
    obj: &Bound<'_, PyAny>,
    foreign: &mut ForeignFields,
) -> PyResult<DataType> {
    let py = obj.py();
    if !obj.is_instance_of::<PyDataType>() {
        return Err(not_a_type_error(obj));
    }
    // A user-defined type, of its JSON and of the proto of its `sqlType()`, as PySpark's
    // conversion writes it, of no look into its `sqlType()` first, which would call it once more.
    // An atomic type, which no user-defined type is, is not looked up as one.
    if !obj.is_instance_of::<PyAtomicType>() && is_udt(obj)? {
        return udt_of(obj, |sql_type| foreign_proto_type(sql_type, foreign));
    }
    if !foreign.holds(obj)? {
        return py_to_proto_data_type(obj);
    }
    if obj.is_instance_of::<PyStructType>() {
        let mut fields = Vec::new();
        // Of the attributes of each element, a field of the native class too, whose attributes are
        // the entries of its `__dict__`: of the type that its `dataType` converts to, as PySpark's
        // conversion reads it, of no look into it first, which may call the code of a user once
        // more (a type that holds no such element converts as a plan's type, see
        // `foreign_proto_type_level`).
        for element in obj.getattr(pyo3::intern!(py, "fields"))?.try_iter()? {
            let element = element?;
            let name = crate::options::str_argument(&element.getattr(pyo3::intern!(py, "name"))?)?;
            let data_type =
                foreign_proto_type(&element.getattr(pyo3::intern!(py, "dataType"))?, foreign)?;
            let nullable =
                crate::streaming::proto_bool(&element.getattr(pyo3::intern!(py, "nullable"))?)?;
            // As PySpark's `json.dumps(field.metadata)` of metadata of a length, of any value.
            let metadata =
                py_metadata_to_map(py, &element.getattr(pyo3::intern!(py, "metadata"))?)?;
            fields.push(StructField {
                name,
                data_type,
                nullable,
                metadata,
            });
        }
        return Ok(DataType::Struct { fields });
    }
    if obj.is_instance_of::<PyArrayType>() {
        let element_type =
            foreign_proto_type(&obj.getattr(pyo3::intern!(py, "elementType"))?, foreign)?;
        let contains_null =
            crate::streaming::proto_bool(&obj.getattr(pyo3::intern!(py, "containsNull"))?)?;
        return Ok(DataType::Array {
            element_type: Box::new(element_type),
            contains_null,
        });
    }
    if obj.is_instance_of::<PyMapType>() {
        let key_type = foreign_proto_type(&obj.getattr(pyo3::intern!(py, "keyType"))?, foreign)?;
        let value_type =
            foreign_proto_type(&obj.getattr(pyo3::intern!(py, "valueType"))?, foreign)?;
        let value_contains_null =
            crate::streaming::proto_bool(&obj.getattr(pyo3::intern!(py, "valueContainsNull"))?)?;
        return Ok(DataType::Map {
            key_type: Box::new(key_type),
            value_type: Box::new(value_type),
            value_contains_null,
        });
    }
    // A `StructField` as a type, which PySpark's conversion does not take.
    Err(not_a_type_error(obj))
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

/// The first type of the type object `obj` of `data_type`, `obj` itself or one nested in it,
/// which PySpark's conversion of a plan's type (`pyspark_types_to_proto_types`) does not take: a
/// `StructField` as a type, or a type of its class only; None if there is none.
fn unsendable_type<'py>(
    obj: &Bound<'py, PyAny>,
    data_type: &DataType,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if obj.is_instance_of::<PyStructField>() || is_class_type(obj)? {
        return Ok(Some(obj.clone()));
    }
    nested_python_type_in(obj.py(), data_type)
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
/// fields and of the SQL types of its user-defined types, as its Python object: a `StructField`
/// as a type, or a type of its class only, of which PySpark has no proto
/// (`pyspark_types_to_proto_types`), nor an Arrow type (`to_arrow_type`).
pub(crate) fn nested_python_type_in<'py>(
    py: Python<'py>,
    data_type: &DataType,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    nested_lowered_in(py, data_type, i32::MAX, false)
}

/// The first type of `data_type` that the client lowered by its JSON, a `StructField` as a type
/// (only, if `fields_only`) or a type of its class only, in the depth-first order of its fields,
/// down to `depth` levels, as its Python object; unless `fields_only`, also of the SQL type of a
/// user-defined type.
fn nested_lowered_in<'py>(
    py: Python<'py>,
    data_type: &DataType,
    depth: i32,
    fields_only: bool,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // A level of a type of many levels, on a stack with room.
    type_level(py, || {
        nested_lowered_in_level(py, data_type, depth, fields_only)
    })
}

/// [`nested_lowered_in`] of one level.
fn nested_lowered_in_level<'py>(
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
        // PySpark's `pyspark_types_to_proto_types` and `to_arrow_type` convert the `sqlType()` of
        // a user-defined type too, which a tree string does not print.
        DataType::Udt {
            sql_type: Some(sql_type),
            ..
        } if !fields_only => vec![sql_type],
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

/// Whether PySpark's converters of local rows assert a nullable flag of the struct `obj` to be
/// other than a bool, as `LocalDataToArrowConversion._create_converter` does: the `nullable` of
/// each field, and the `containsNull` of an array where its converter is made, and the
/// `valueContainsNull` of a map. A converter is made for the children of a type only where the
/// type needs one (`infer::needs_converter`), so that a flag of an atomic type is not asserted.
pub(crate) fn has_non_bool_converter_flag(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let data_type = py_to_data_type_as_given(obj)?;
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
    // A level of a type of many levels, on a stack with room.
    type_level(py, || {
        converter_flag_is_other_level(py, data_type, object, nullable)
    })
}

/// [`converter_flag_is_other`] of one level.
fn converter_flag_is_other_level(
    py: Python<'_>,
    data_type: &DataType,
    object: Option<&Bound<'_, PyAny>>,
    nullable: bool,
) -> PyResult<bool> {
    if !crate::infer::needs_converter(data_type, nullable) {
        return Ok(false);
    }
    // The flag as it was given and the object of a child, which the core type has only as a bool
    // and the core type of the child.
    match data_type {
        DataType::Struct { fields } => {
            for (i, field) in fields.iter().enumerate() {
                let (flag, kept) = match object.filter(|o| o.is_instance_of::<PyStructType>()) {
                    Some(st) => {
                        let item = struct_fields(st)?.get_item(i)?;
                        let object = as_struct_field(&item)?;
                        (
                            Some(parameter(&object, pyo3::intern!(py, "nullable"))?),
                            Some(parameter(&object, pyo3::intern!(py, "dataType"))?),
                        )
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
            let (flag, kept) = match object.filter(|o| o.is_instance_of::<PyArrayType>()) {
                Some(array) => (
                    Some(parameter(array, pyo3::intern!(py, "containsNull"))?),
                    Some(parameter(array, pyo3::intern!(py, "elementType"))?),
                ),
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
            let (flag, key, value) = match object.filter(|o| o.is_instance_of::<PyMapType>()) {
                Some(map) => (
                    Some(parameter(map, pyo3::intern!(py, "valueContainsNull"))?),
                    Some(parameter(map, pyo3::intern!(py, "keyType"))?),
                    Some(parameter(map, pyo3::intern!(py, "valueType"))?),
                ),
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
    // A level of a type of many levels, on a stack with room.
    type_level(py, || has_other_flag_level(py, obj))
}

/// [`has_other_flag`] of one level.
fn has_other_flag_level(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if is_subclass_object(obj) {
        return Ok(true);
    }
    if obj.is_instance_of::<PyStructField>() {
        let name = parameter(obj, pyo3::intern!(py, "name"))?;
        if !name.is_instance_of::<pyo3::types::PyString>()
            || !is_bool_flag(&parameter(obj, pyo3::intern!(py, "nullable"))?)
        {
            return Ok(true);
        }
        return child_has_other_flag(py, &parameter(obj, pyo3::intern!(py, "dataType"))?);
    }
    if obj.is_instance_of::<PyArrayType>() {
        if !is_bool_flag(&parameter(obj, pyo3::intern!(py, "containsNull"))?) {
            return Ok(true);
        }
        let element = parameter(obj, pyo3::intern!(py, "elementType"))?;
        return child_has_other_flag(py, &element);
    }
    if obj.is_instance_of::<PyMapType>() {
        if !is_bool_flag(&parameter(obj, pyo3::intern!(py, "valueContainsNull"))?) {
            return Ok(true);
        }
        for name in [pyo3::intern!(py, "keyType"), pyo3::intern!(py, "valueType")] {
            if child_has_other_flag(py, &parameter(obj, name)?)? {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    if obj.is_instance_of::<PyStructType>() {
        for item in struct_fields(obj)?.try_iter()? {
            if has_other_flag(py, &item?)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// `has_other_flag` of a type in a type position: a `StructField` or a type of its class only
/// there is written from its own object, as its JSON is not the core's.
fn child_has_other_flag(py: Python<'_>, child: &Bound<'_, PyAny>) -> PyResult<bool> {
    // Of a value that is not a data type, which PySpark's attribute keeps, too.
    if !child.is_instance_of::<PyDataType>()
        || child.is_instance_of::<PyStructField>()
        || is_class_type(child)?
    {
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
    // A level of a type of many levels, on a stack with room.
    crate::pbmsg::grow(|| has_collations_level(data_type))
}

/// [`has_collations`] of one level.
fn has_collations_level(data_type: &DataType) -> bool {
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
    // A level of the JSON of a type of many levels, on a stack with room.
    type_level(py, || keep_flags_in_json_level(py, json, obj, collated))
}

/// [`keep_flags_in_json`] of one level of a type object.
fn keep_flags_in_json_level(
    py: Python<'_>,
    json: &Bound<'_, PyAny>,
    obj: &Bound<'_, PyAny>,
    collated: bool,
) -> PyResult<bool> {
    if obj.is_instance_of::<PyStructField>() {
        let mut other = false;
        let name = parameter(obj, pyo3::intern!(py, "name"))?;
        if !name.is_instance_of::<pyo3::types::PyString>() {
            json.set_item("name", &name)?;
            other = true;
        }
        let nullable = parameter(obj, pyo3::intern!(py, "nullable"))?;
        json.set_item("nullable", &nullable)?;
        other |= !is_bool_flag(&nullable);
        let data_type = parameter(obj, pyo3::intern!(py, "dataType"))?;
        // As PySpark's, the metadata of the field, of the values as they were given, which the
        // core's JSON values do not keep, and the collations of its strings, if it has some; of
        // metadata that is not a dict too, which PySpark's field keeps as it is.
        let metadata = parameter(obj, pyo3::intern!(py, "metadata"))?;
        let empty_dict = metadata
            .cast::<pyo3::types::PyDict>()
            .is_ok_and(|dict| dict.is_empty());
        if !empty_dict {
            if has_collations(&py_to_data_type(&data_type)?) {
                let with_collations = unpacked_metadata(py, &metadata)?;
                let collations = json
                    .get_item("metadata")?
                    .get_item(COLLATIONS_METADATA_KEY)?;
                with_collations.set_item(COLLATIONS_METADATA_KEY, collations)?;
                json.set_item("metadata", with_collations)?;
            } else {
                json.set_item("metadata", &metadata)?;
            }
            other = true;
        }
        // The collations matter only to the types of Python subclasses in the field.
        let collated =
            holds_subclass_object(py, &data_type)? && has_collations(&py_to_data_type(&data_type)?);
        other |= keep_child_json(py, json, "type", &data_type, collated)?;
        return Ok(other);
    }
    if obj.is_instance_of::<PyArrayType>() {
        let contains_null = parameter(obj, pyo3::intern!(py, "containsNull"))?;
        json.set_item("containsNull", &contains_null)?;
        let mut other = !is_bool_flag(&contains_null);
        let element = parameter(obj, pyo3::intern!(py, "elementType"))?;
        other |= keep_child_json(py, json, "elementType", &element, collated)?;
        return Ok(other);
    }
    if obj.is_instance_of::<PyMapType>() {
        let value_contains_null = parameter(obj, pyo3::intern!(py, "valueContainsNull"))?;
        json.set_item("valueContainsNull", &value_contains_null)?;
        let mut other = !is_bool_flag(&value_contains_null);
        let key = parameter(obj, pyo3::intern!(py, "keyType"))?;
        other |= keep_child_json(py, json, "keyType", &key, collated)?;
        let value = parameter(obj, pyo3::intern!(py, "valueType"))?;
        other |= keep_child_json(py, json, "valueType", &value, collated)?;
        return Ok(other);
    }
    if obj.is_instance_of::<PyStructType>() {
        let items = struct_fields(obj)?;
        let fields = json.get_item("fields")?;
        let mut other = false;
        for (i, item) in items.try_iter()?.enumerate() {
            let item = item?;
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
    // A `UserDefinedType` is its own `jsonValue()`, of the `jsonValue()` of its `sqlType()`, whose
    // fields hold their metadata dicts as they are, which the JSON of the core type does not.
    if is_udt(child)? {
        json.set_item(key, child.call_method0(pyo3::intern!(py, "jsonValue"))?)?;
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
    // Of a type of many levels, whose JSON values and copies recurse over its levels as they
    // are made, on a stack with room for a level (the recursions below grow on their own).
    type_level(py, || type_json_value_level(py, obj))
}

/// [`type_json_value`] of one level.
fn type_json_value_level<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    // As PySpark's `jsonValue`, of the attributes as Python reads them, of an object of a Python
    // subclass that reads one of them of its own, or of a class patched over one, which the core's
    // JSON of the native values does not hold (see `json_reads_otherwise`).
    if json_reads_otherwise(obj)? {
        return Ok((
            ForeignFields::default().json_value_of_native(py, obj)?,
            true,
        ));
    }
    // Of a struct whose fields hold an element that is not a `StructField`, at any level, as
    // PySpark's `jsonValue` writes it, of the `jsonValue` of each element.
    let (lowered, maybe_foreign) = lower_noting_subclasses(obj, MetadataOf::AsGiven);
    if maybe_foreign {
        let mut foreign = ForeignFields::default();
        if foreign.holds(obj)? {
            return Ok((foreign.json_value_of_native(py, obj)?, true));
        }
    }
    // Of a type that does not lower, such as one that holds a type of parameters of values that
    // the core type does not hold (`DecimalType(10.5)`), as PySpark's `jsonValue` writes it, of
    // the `jsonValue` of each type that it holds, of their own errors.
    let lowered = match lowered {
        Ok(lowered) => lowered,
        Err(error)
            if is_exception(py, &error)
                && !error.is_instance_of::<pyo3::exceptions::PyRecursionError>(py)
                && (obj.is_instance_of::<PyStructType>()
                    || obj.is_instance_of::<PyStructField>()
                    || obj.is_instance_of::<PyArrayType>()
                    || obj.is_instance_of::<PyMapType>()) =>
        {
            return Ok((
                ForeignFields::default().json_value_of_native(py, obj)?,
                true,
            ));
        }
        Err(error) => return Err(error),
    };
    let json = py_json_value(py, lowered.json_value(), false)?;
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
    // Of a type of many levels, whose JSON values and copies recurse over its levels as they
    // are made, on a stack with room for a level (the recursions below grow on their own).
    type_level(obj.py(), || type_json_text_level(obj))
}

/// [`type_json_text`] of one level.
fn type_json_text_level(obj: &Bound<'_, PyAny>) -> PyResult<String> {
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
    // Of the JSON of a type of many levels, which the encoder of Python recurses over in C.
    grow_for_json(|| {
        py.import("json")?
            .call_method("dumps", (json,), Some(&kwargs))?
            .extract()
    })
}

/// The `json()` text of the type object `schema`, as PySpark's `json()` writes it, when the core's
/// text is not it: of a type of a Python subclass, of its own method, and of a struct of a nullable
/// flag that is not a bool or of metadata (see `type_json_value`). None for no schema, or another
/// one. Of the attributes of the types as Python reads them, as PySpark's `json()` reads them, of
/// a Python subclass of its own or of a patched class too (see `json_reads_otherwise`), of a
/// relation of no rows.
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

/// The `json()` that PySpark's `createDataFrame` of no rows sends of `schema`, a struct whose
/// fields hold an element that is not a `StructField`, at any level (a list that a str was appended
/// to in place, `s.fields.append("x")`), which the core's struct type does not take (see
/// `ForeignFields`): as PySpark's `json()` of the object writes it, of the `jsonValue` of each
/// element of the fields of each struct in their order, an element's own, so of the
/// `AttributeError` of one that has none, such as `'str' object has no attribute 'jsonValue'`. The
/// `json` or `jsonValue` of a Python subclass is its own. None of another schema.
pub(crate) fn foreign_schema_json(schema: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let py = schema.py();
    let mut foreign = ForeignFields::default();
    if !schema.is_instance_of::<PyStructType>() || !foreign.holds(schema)? {
        return Ok(None);
    }
    if overrides(schema, "json", &py.get_type::<PyStructType>())? {
        return Ok(Some(
            schema.call_method0(pyo3::intern!(py, "json"))?.extract()?,
        ));
    }
    json_dumps(py, &foreign.json_value(py, schema)?).map(Some)
}

/// Which type objects hold an element that is not a `StructField` in the `fields` of one of their
/// structs, at any level: of the fields' types, the elements of the fields themselves (a struct, an
/// array or a map in `fields` is such an element, and may hold more), the element of an array, and
/// the key and the value of a map. Of each object, the children that it keeps, which the lowering
/// into the core's types reads, and those that its attributes give, which PySpark's methods read
/// (see `ForeignFields::json_value`), of a Python subclass of an attribute of its own, such as a
/// property that makes a new type at each read. A child that an object does not keep as an object
/// is of the core's types, which hold fields only. Each object is looked at once, and held with its
/// answer, so that no object takes the address of one that was freed and gets its answer.
#[derive(Default)]
struct ForeignFields(std::collections::HashMap<usize, (Py<PyAny>, bool)>);

impl ForeignFields {
    /// Whether the type object `obj` holds such an element.
    fn holds(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        let key = obj.as_ptr() as usize;
        if let Some((_, holds)) = self.0.get(&key) {
            return Ok(*holds);
        }
        // Of a type of many levels, on a stack with room for a level. An object is marked as
        // holding none while it is looked at: a type that holds itself reaches it again, which
        // adds nothing to the answer of the walk that is looking at it.
        self.0.insert(key, (obj.clone().unbind(), false));
        let holds = type_level(obj.py(), || self.holds_level(obj))?;
        if let Some(entry) = self.0.get_mut(&key) {
            entry.1 = holds;
        }
        Ok(holds)
    }

    /// [`ForeignFields::holds`] of one level.
    fn holds_level(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = obj.py();
        if obj.is_instance_of::<PyStructType>() {
            // Of an element of its own fields that is not a `StructField`, before the looks into
            // the types of the others, which may call the code of a user, such as the `sqlType()`
            // of a user-defined type.
            if holds_foreign_element(obj)? {
                return Ok(true);
            }
            if self.holds_element(struct_fields(obj)?.try_iter()?)? {
                return Ok(true);
            }
            let native = py.get_type::<PyStructType>();
            if reads_own(obj, "__iter__", &native)? {
                return self.holds_element(struct_elements(obj)?);
            }
            Ok(false)
        } else if obj.is_instance_of::<PyStructField>() {
            // Of the type of the `__dict__` of the field, or of the attribute of a Python
            // subclass of its own (see `parameter`).
            self.holds(&parameter(obj, pyo3::intern!(py, "dataType"))?)
        } else if obj.is_instance_of::<PyArrayType>() {
            // Of the children of the `__dict__` of the type, or of the attributes of a Python
            // subclass of its own (see `parameter`).
            self.holds(&parameter(obj, pyo3::intern!(py, "elementType"))?)
        } else if obj.is_instance_of::<PyMapType>() {
            Ok(self.holds(&parameter(obj, pyo3::intern!(py, "keyType"))?)?
                || self.holds(&parameter(obj, pyo3::intern!(py, "valueType"))?)?)
        } else if !obj.is_instance_of::<PyAtomicType>() && is_udt(obj)? {
            // Of the `sqlType()` of a user-defined type, whose JSON and proto PySpark writes of
            // it; of one that is not implemented, none, whose uses raise its error. An atomic type,
            // which no user-defined type is, is not looked up as one.
            match obj.call_method0(pyo3::intern!(py, "sqlType")) {
                Ok(sql_type) => self.holds(&sql_type),
                Err(error) if is_exception(py, &error) => Ok(false),
                Err(error) => Err(error),
            }
        } else {
            Ok(false)
        }
    }

    /// Whether one of `elements`, the elements of the fields of a struct, is not a `StructField`,
    /// or is one whose type holds such an element.
    fn holds_element(&mut self, elements: Bound<'_, pyo3::types::PyIterator>) -> PyResult<bool> {
        for element in elements {
            let element = element?;
            if !element.is_instance_of::<PyStructField>() || self.holds(&element)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The `jsonValue` of the type object `obj`, as PySpark's `jsonValue` makes it, of the
    /// attributes that it reads in their order: of a struct, the `jsonValue` of each element of
    /// its fields in their order (`for f in self`), the element's own for one that is not a type;
    /// of a field, PySpark's `StructField.jsonValue`, of its `dataType` and `name`, of the
    /// collations of the strings of its arrays and maps, which go to its metadata, then of its
    /// `metadata`, `name`, `dataType` (written without the collations) and `nullable`; and of an
    /// array and a map, of their `typeName`, their types and their flags. A Python subclass of its
    /// own `jsonValue`, and a type of another kind, are of their `jsonValue`.
    fn json_value<'py>(
        &mut self,
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Of a type of many levels, whose JSON values recurse over its levels as they are made, on
        // a stack with room for a level.
        type_level(py, || self.json_value_level(py, obj, true))
    }

    /// [`ForeignFields::json_value`] of `obj` itself, which the native `jsonValue` of its class
    /// writes: not of a `jsonValue` of a Python subclass of its own, or of a patch of the native
    /// class, which may call the native one (`super().jsonValue()`).
    fn json_value_of_native<'py>(
        &mut self,
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        type_level(py, || self.json_value_level(py, obj, false))
    }

    /// [`ForeignFields::json_value`] of one level, of the `jsonValue` of a Python subclass of its
    /// own, or of a patch of the native class, if `of_its_own`.
    fn json_value_level<'py>(
        &mut self,
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
        of_its_own: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let json_value = pyo3::intern!(py, "jsonValue");
        let own = |native: Bound<'_, pyo3::types::PyType>| {
            Ok::<bool, PyErr>(of_its_own && reads_own(obj, "jsonValue", &native)?)
        };
        let json = pyo3::types::PyDict::new(py);
        if obj.is_instance_of::<PyStructType>() {
            if own(py.get_type::<PyStructType>())? {
                return obj.call_method0(json_value);
            }
            json.set_item("type", type_name_of(obj)?)?;
            let fields = PyList::empty(py);
            for element in struct_elements(obj)? {
                fields.append(self.json_value(py, &element?)?)?;
            }
            json.set_item("fields", fields)?;
        } else if obj.is_instance_of::<PyStructField>() {
            if own(py.get_type::<PyStructField>())? {
                return obj.call_method0(json_value);
            }
            // As PySpark's `StructField.jsonValue`: `getCollationMetadata` reads the type and the
            // name, of the collations of the strings of the arrays and maps of the type, which go
            // to the metadata, and the type is written without them (`_dataTypeJsonValue`).
            let data_type = obj.getattr(pyo3::intern!(py, "dataType"))?;
            let name = obj.getattr(pyo3::intern!(py, "name"))?;
            let collations = pyo3::types::PyDict::new(py);
            visit_collations(py, &data_type, &name, &collations)?;
            let metadata = obj.getattr(pyo3::intern!(py, "metadata"))?;
            let metadata = if collations.is_empty() {
                metadata
            } else {
                let merged = unpacked_metadata(py, &metadata)?;
                merged.set_item("__COLLATIONS", &collations)?;
                merged.into_any()
            };
            json.set_item("name", obj.getattr(pyo3::intern!(py, "name"))?)?;
            let data_type = obj.getattr(pyo3::intern!(py, "dataType"))?;
            let type_json = if collations.is_empty() {
                self.json_value(py, &data_type)?
            } else {
                self.json_value_without_collations(py, &data_type)?
            };
            json.set_item("type", type_json)?;
            json.set_item("nullable", obj.getattr(pyo3::intern!(py, "nullable"))?)?;
            json.set_item("metadata", metadata)?;
        } else if obj.is_instance_of::<PyArrayType>() {
            if own(py.get_type::<PyArrayType>())? {
                return obj.call_method0(json_value);
            }
            json.set_item("type", type_name_of(obj)?)?;
            let element = obj.getattr(pyo3::intern!(py, "elementType"))?;
            json.set_item("elementType", self.json_value(py, &element)?)?;
            json.set_item(
                "containsNull",
                obj.getattr(pyo3::intern!(py, "containsNull"))?,
            )?;
        } else if obj.is_instance_of::<PyMapType>() {
            if own(py.get_type::<PyMapType>())? {
                return obj.call_method0(json_value);
            }
            json.set_item("type", type_name_of(obj)?)?;
            let key = obj.getattr(pyo3::intern!(py, "keyType"))?;
            json.set_item("keyType", self.json_value(py, &key)?)?;
            let value = obj.getattr(pyo3::intern!(py, "valueType"))?;
            json.set_item("valueType", self.json_value(py, &value)?)?;
            json.set_item(
                "valueContainsNull",
                obj.getattr(pyo3::intern!(py, "valueContainsNull"))?,
            )?;
        } else {
            return obj.call_method0(json_value);
        }
        Ok(json.into_any())
    }

    /// The JSON value of PySpark's `removeCollations(data_type).jsonValue()` of a field whose
    /// strings are collated (`StructField._dataTypeJsonValue`): of new arrays and maps of the
    /// types and the flags of `data_type`, read in their order, of a string, a varchar and a char
    /// of no collation, and of any other type, a struct too, its own JSON value.
    fn json_value_without_collations<'py>(
        &mut self,
        py: Python<'py>,
        data_type: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        type_level(py, || {
            let json = pyo3::types::PyDict::new(py);
            if data_type.is_instance_of::<PyArrayType>() {
                let element = data_type.getattr(pyo3::intern!(py, "elementType"))?;
                json.set_item("type", "array")?;
                json.set_item(
                    "elementType",
                    self.json_value_without_collations(py, &element)?,
                )?;
                json.set_item(
                    "containsNull",
                    data_type.getattr(pyo3::intern!(py, "containsNull"))?,
                )?;
                return Ok(json.into_any());
            }
            if data_type.is_instance_of::<PyMapType>() {
                let key = data_type.getattr(pyo3::intern!(py, "keyType"))?;
                let key = self.json_value_without_collations(py, &key)?;
                let value = data_type.getattr(pyo3::intern!(py, "valueType"))?;
                let value = self.json_value_without_collations(py, &value)?;
                json.set_item("type", "map")?;
                json.set_item("keyType", key)?;
                json.set_item("valueType", value)?;
                json.set_item(
                    "valueContainsNull",
                    data_type.getattr(pyo3::intern!(py, "valueContainsNull"))?,
                )?;
                return Ok(json.into_any());
            }
            let plain = if data_type.is_instance_of::<PyStringType>() {
                Some(py.get_type::<PyStringType>().call0()?)
            } else if data_type.is_instance_of::<PyVarcharType>() {
                let length = data_type.getattr(pyo3::intern!(py, "length"))?;
                Some(py.get_type::<PyVarcharType>().call1((length,))?)
            } else if data_type.is_instance_of::<PyCharType>() {
                let length = data_type.getattr(pyo3::intern!(py, "length"))?;
                Some(py.get_type::<PyCharType>().call1((length,))?)
            } else {
                None
            };
            match plain {
                Some(plain) => plain.call_method0(pyo3::intern!(py, "jsonValue")),
                None => self.json_value(py, data_type),
            }
        })
    }
}

/// The elements of the fields of the struct `obj`, as PySpark's `for f in self` iterates them (see
/// `struct_iter`).
fn struct_elements<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
    struct_iter(obj)
}

/// PySpark's `StructField.getCollationMetadata` of the type `data_type` of a field at `path`, its
/// name: the `provider.collation` of each collated string of its arrays and maps, of itself too,
/// by its path (`s.element`, `s.key`, `s.value`), into `out`, but not of the structs in it, whose
/// fields write their own (`visitRecursively`).
fn visit_collations(
    py: Python<'_>,
    data_type: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    out: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<()> {
    type_level(py, || {
        if data_type.is_instance_of::<PyArrayType>() {
            let element = data_type.getattr(pyo3::intern!(py, "elementType"))?;
            process_collations(py, &element, &path.add(".element")?, out)
        } else if data_type.is_instance_of::<PyMapType>() {
            let key = data_type.getattr(pyo3::intern!(py, "keyType"))?;
            process_collations(py, &key, &path.add(".key")?, out)?;
            let value = data_type.getattr(pyo3::intern!(py, "valueType"))?;
            process_collations(py, &value, &path.add(".value")?, out)
        } else if is_collated_string(py, data_type)? {
            out.set_item(path, collation_value(py, data_type)?)
        } else {
            Ok(())
        }
    })
}

/// PySpark's `processDataType` of `getCollationMetadata`: a collated string at `path`, or the
/// levels of another type.
fn process_collations(
    py: Python<'_>,
    data_type: &Bound<'_, PyAny>,
    path: &Bound<'_, PyAny>,
    out: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<()> {
    if is_collated_string(py, data_type)? {
        return out.set_item(path, collation_value(py, data_type)?);
    }
    visit_collations(py, data_type, path, out)
}

/// PySpark's `StructField._isCollatedString`: a `StringType` of a collation other than UTF-8
/// binary.
fn is_collated_string(py: Python<'_>, data_type: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(data_type.is_instance_of::<PyStringType>()
        && !data_type
            .call_method0(pyo3::intern!(py, "isUTF8BinaryCollation"))?
            .is_truthy()?)
}

/// PySpark's `StructField.schemaCollationValue`: `{provider}.{collation}` of the string type.
fn collation_value<'py>(
    py: Python<'py>,
    data_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let collation = data_type.getattr(pyo3::intern!(py, "collation"))?;
    let provider = py
        .get_type::<PyStringType>()
        .call_method1(pyo3::intern!(py, "collationProvider"), (&collation,))?;
    Ok(
        pyo3::types::PyString::new(py, &format!("{}.{}", provider.str()?, collation.str()?))
            .into_any(),
    )
}

/// Whether the type object `obj` holds an element that is not a `StructField` in the `fields` of
/// one of its structs, at any level (see `ForeignFields`).
pub(crate) fn holds_foreign_field(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    ForeignFields::default().holds(obj)
}

/// Whether the `fields` of the struct `obj` themselves hold an element that is not a
/// `StructField`, as PySpark's struct holds what its list holds.
fn holds_foreign_element(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !obj.is_instance_of::<PyStructType>() {
        return Ok(false);
    }
    // Of `fields` that do not read or iterate, the uses read the attribute as PySpark's do, of its
    // errors.
    let py = obj.py();
    let fields = match struct_fields(obj).and_then(|fields| fields.try_iter()) {
        Ok(fields) => fields,
        Err(error) if is_exception(py, &error) => return Ok(true),
        Err(error) => return Err(error),
    };
    for element in fields {
        if !element?.is_instance_of::<PyStructField>() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The native classes of types that PySpark's `jsonValue` of a type reads the attributes of, and
/// those attributes of the classes, which a Python subclass may read of its own, or a patch of the
/// class replace: of a struct, the iteration; of a string, `simpleString` and
/// `isUTF8BinaryCollation`; and `jsonValue` of each. The core's JSON of the other types is that of
/// their native values. The parameters of a struct, a field, an array, a map and a string are
/// entries of their `__dict__` (see `container_parameters`), which their lowering reads as Python
/// reads them, of a property that a patch of the class sets too (see `parameter`), and which a
/// subclass may read of its own (see `json_reads_of_its_own`).
fn json_reads(py: Python<'_>) -> [(Bound<'_, pyo3::types::PyType>, &'static [&'static str]); 5] {
    [
        (py.get_type::<PyStructType>(), &["__iter__", "jsonValue"]),
        (py.get_type::<PyStructField>(), &["jsonValue"]),
        (py.get_type::<PyArrayType>(), &["jsonValue"]),
        (py.get_type::<PyMapType>(), &["jsonValue"]),
        (
            py.get_type::<PyStringType>(),
            &["simpleString", "isUTF8BinaryCollation", "jsonValue"],
        ),
    ]
}

/// The attributes of `json_reads` of the native classes, as the module makes them, before any
/// patch of them, which `native_attribute` compares a class's attribute with.
type NativeJsonReads = Vec<(
    Py<pyo3::types::PyType>,
    &'static str,
    Py<pyo3::types::PyString>,
    Py<PyAny>,
)>;
static NATIVE_JSON_READS: PyOnceLock<NativeJsonReads> = PyOnceLock::new();

/// Keep the attributes of `json_reads` of the native classes, as the module makes them (see
/// `NATIVE_JSON_READS`).
pub(crate) fn keep_native_json_reads(py: Python<'_>) -> PyResult<()> {
    NATIVE_JSON_READS.get_or_try_init(py, || -> PyResult<NativeJsonReads> {
        let mut kept = Vec::new();
        for (class, names) in json_reads(py) {
            for &name in names {
                let interned = pyo3::types::PyString::intern(py, name);
                let attribute = class.getattr(&interned)?;
                kept.push((
                    class.clone().unbind(),
                    name,
                    interned.unbind(),
                    attribute.unbind(),
                ));
            }
        }
        Ok(kept)
    })?;
    Ok(())
}

/// Whether the attribute `name` of the native class `class`, if the class has one of its own, is
/// its native getter, method or slot, and not what a patch of the class set, as
/// `StringType.isUTF8BinaryCollation = lambda self: False` sets it. True of an attribute that the
/// class does not have of its own, which it inherits. Of an attribute of `json_reads`, the one
/// that the module made (`NATIVE_JSON_READS`), of one lookup.
fn native_attribute(class: &Bound<'_, pyo3::types::PyType>, name: &str) -> PyResult<bool> {
    let py = class.py();
    if let Some((_, _, interned, native)) = NATIVE_JSON_READS.get(py).and_then(|kept| {
        kept.iter()
            .find(|(owner, kept_name, _, _)| *kept_name == name && owner.bind(py).is(class))
    }) {
        return Ok(class
            .getattr(interned.bind(py))
            .is_ok_and(|attribute| attribute.is(native.bind(py))));
    }
    let Ok(attribute) = class.getattr(pyo3::intern!(py, "__dict__"))?.get_item(name) else {
        return Ok(true);
    };
    let kind = attribute.get_type().name()?;
    Ok(matches!(
        kind.to_cow()?.as_ref(),
        "getset_descriptor" | "method_descriptor" | "wrapper_descriptor" | "classmethod_descriptor"
    ) && attribute
        .getattr(pyo3::intern!(py, "__objclass__"))
        .is_ok_and(|owner| owner.is(class)))
}

/// Whether a native class of types is patched over an attribute that PySpark's `jsonValue` reads
/// (see `json_reads`): PySpark's `json()` of the types then reads the patch, as
/// `ForeignFields::json_value` does, where the core's JSON is of the native values. Not while a
/// JSON of the native values is written (see `of_native_values`).
fn json_reads_patched(py: Python<'_>) -> PyResult<bool> {
    if json_of_native_values() {
        return Ok(false);
    }
    if let Some(kept) = NATIVE_JSON_READS.get(py) {
        return Ok(kept.iter().any(|(class, _, name, native)| {
            !class
                .bind(py)
                .getattr(name.bind(py))
                .is_ok_and(|attribute| attribute.is(native.bind(py)))
        }));
    }
    for (class, names) in json_reads(py) {
        for name in names {
            if !native_attribute(&class, name)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Whether the type object `obj` is of a Python subclass that reads one of the attributes that
/// PySpark's `jsonValue` of it reads (see `json_reads`) of its own, or one of the parameters of a
/// field, an array or a map, which PySpark's `jsonValue` reads too (see `container_parameters`).
fn json_reads_of_its_own(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    for (native, names) in json_reads(obj.py()) {
        if obj.is_instance(&native)? {
            if obj.get_type().is(&native) {
                return Ok(false);
            }
            let parameters = container_parameters(obj).unwrap_or(&[]);
            for name in names.iter().chain(parameters) {
                if overrides(obj, name, &native)? {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
    }
    Ok(false)
}

/// Whether PySpark's `jsonValue` of the type object `obj` reads, of its level, an attribute that
/// the native class does not give: of a Python subclass of the attribute of its own, or of a
/// native class patched over one (see `json_reads`). `ForeignFields::json_value` reads them as
/// PySpark's does, of the levels below too, where the core's JSON is of the native values. Not
/// while a JSON of the native values is written (see `of_native_values`).
fn json_reads_otherwise(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(!json_of_native_values() && (json_reads_of_its_own(obj)? || json_reads_patched(obj.py())?))
}

thread_local! {
    /// Whether this thread writes the JSON of types of their native values now (see
    /// `of_native_values`).
    static OF_NATIVE_VALUES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run `write`, which writes the JSON of a type, of the native values of the types, of the
/// attributes that the native classes keep, and not of those that a Python subclass, or a patch of
/// a native class, reads otherwise (see `json_reads_otherwise`): the schema of a relation of rows
/// that the native types convert, which is of the types of its rows (see `schema_json_text`).
fn of_native_values<T>(write: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            OF_NATIVE_VALUES.with(|of| of.set(self.0));
        }
    }
    let _restore = Restore(OF_NATIVE_VALUES.with(|of| of.replace(true)));
    write()
}

/// Whether this thread writes the JSON of types of their native values now (see
/// `of_native_values`).
fn json_of_native_values() -> bool {
    OF_NATIVE_VALUES.with(std::cell::Cell::get)
}

/// Whether PySpark's `jsonValue` of the type object `obj` reads otherwise (see
/// `json_reads_otherwise`) at any of its levels, of the types that it keeps as their objects (see
/// `keeps_object_of`).
pub(crate) fn holds_json_reads_otherwise(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    fn holds_own(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(json_reads_of_its_own(obj)? || keeps_object_of(py, obj, holds_own)?)
    }
    Ok(json_reads_patched(py)? || holds_own(py, obj)?)
}

/// Whether the attribute `name` of `obj`, of the native class `native` or a Python subclass of it,
/// is not the native one: of a Python subclass that overrides it (see `overrides`), or of the
/// native class patched over it (see `native_attribute`), as `StructType.__iter__ = ...` patches
/// it, whose read is the patch.
fn reads_own(
    obj: &Bound<'_, PyAny>,
    name: &str,
    native: &Bound<'_, pyo3::types::PyType>,
) -> PyResult<bool> {
    Ok(overrides(obj, name, native)? || !native_attribute(native, name)?)
}

/// Whether the class of `obj` overrides the attribute `name` of the native class `native`, as a
/// Python subclass of its own method does.
fn overrides(
    obj: &Bound<'_, PyAny>,
    name: &str,
    native: &Bound<'_, pyo3::types::PyType>,
) -> PyResult<bool> {
    // Of the native class itself, whose attribute is its own, of no lookup.
    let class = obj.get_type();
    if class.is(native) {
        return Ok(false);
    }
    match native.getattr_opt(name)? {
        Some(attribute) => Ok(!class.getattr(name)?.is(&attribute)),
        // An attribute that the native class does not have, such as a parameter of a type, which
        // its `__dict__` holds (see `parameter`): of a class that has one of its own.
        None => Ok(class.getattr_opt(name)?.is_some()),
    }
}

/// The `jsonValue` of the field `field`, with its nullable flags as given, and whether one of
/// them is other than a bool.
fn field_json_value<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyStructField>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    // Of a type of many levels, whose JSON values and copies recurse over its levels as they
    // are made, on a stack with room for a level (the recursions below grow on their own).
    type_level(py, || field_json_value_level(py, field))
}

/// [`field_json_value`] of one level.
fn field_json_value_level<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyStructField>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    // As PySpark's `StructField.jsonValue`, of its `name`, `dataType`, `nullable` and `metadata`
    // as Python reads them, of a Python subclass of one of its own, or of the class patched over
    // one (see `json_reads_otherwise`).
    if json_reads_otherwise(field.as_any())? {
        return Ok((
            ForeignFields::default().json_value_of_native(py, field.as_any())?,
            true,
        ));
    }
    // Of a type that holds a struct whose fields hold an element that is not a `StructField`, of
    // the walk of the foreign fields only of a field that may hold one, as the lowering of its type
    // tells (see `lower_noting_subclasses`).
    let before = LOWERED_SUBCLASS.with(|lowered| lowered.replace(false));
    note_subclass(field.as_any(), &py.get_type::<PyStructField>());
    let current = core_field(field.as_any(), MetadataOf::AsGiven);
    let subclass = LOWERED_SUBCLASS.with(|lowered| lowered.replace(before || lowered.get()));
    if subclass
        || matches!(&current, Err(error) if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py))
    {
        let mut foreign = ForeignFields::default();
        if foreign.holds(field.as_any())? {
            return Ok((foreign.json_value_of_native(py, field.as_any())?, true));
        }
    }
    // Of a field whose `__dict__` the core field does not hold, such as of a type that does not
    // lower (a value that is not a type, `DecimalType(10.5)`) or of no parameters, as PySpark's
    // `jsonValue` writes it, of the attributes in its order and of the `jsonValue` of its type, of
    // their own errors.
    let current = match current {
        Ok(current) => current,
        Err(error)
            if is_exception(py, &error)
                && !error.is_instance_of::<pyo3::exceptions::PyRecursionError>(py) =>
        {
            return Ok((
                ForeignFields::default().json_value_of_native(py, field.as_any())?,
                true,
            ));
        }
        Err(error) => return Err(error),
    };
    let json = py_json_value(py, current.json_value(), true)?;
    let other = keep_flags_in_json(py, &json, field.as_any(), false)?;
    Ok((json, other))
}

/// PySpark's `_schema.json()` of the struct `schema`, if it holds metadata or a `UserDefinedType`,
/// whose `sqlType()` may hold some, which its `createDataFrame` of data writes after it converts
/// the data, and sends as the schema of the relation: of the error of `json.dumps` of the metadata
/// of a field, also of the keys of the sorted JSON (`'<' not supported between instances of 'str'
/// and 'int'`), and of the keys of a dict in the order of Python's sort of them, of an int key by
/// its value (`{10: "a", 9: "b"}` is `{"9":"b","10":"a"}`), which the sort of the strs of the
/// keys of the core's JSON does not keep. None of another schema, whose JSON the core writes as
/// PySpark does. Of the native values of the types (see `of_native_values`), of the relation of
/// rows that the native types convert, which are of those types: PySpark's conversion of the
/// rows reads the attributes that a Python subclass or a patch of a class reads otherwise, as its
/// `json()` does (see `pyspark_schema_json_text`).
pub(crate) fn schema_json_text(schema: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(schema) = schema.filter(|schema| schema.is_instance_of::<PyStructType>()) else {
        return Ok(None);
    };
    let py = schema.py();
    of_native_values(|| {
        if !(holds_metadata(py, schema)? || holds_udt(py, schema)?) {
            return Ok(None);
        }
        Ok(Some(
            schema.call_method0(pyo3::intern!(py, "json"))?.extract()?,
        ))
    })
}

/// [`schema_json_text`] of the attributes of the types as Python reads them, as PySpark's
/// `json()` reads them, of a relation of rows that PySpark's conversion converted, which reads them
/// too (`prepare_pandas_table`), or of no rows: also of a struct that holds, at any level, a type
/// of a Python subclass that reads one of them of its own, or of a native class patched over one
/// (see `json_reads_otherwise`), whose `json()` reads them.
pub(crate) fn pyspark_schema_json_text(
    schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    let Some(schema) = schema.filter(|schema| schema.is_instance_of::<PyStructType>()) else {
        return Ok(None);
    };
    let py = schema.py();
    if !holds_json_reads_otherwise(py, schema)? {
        return schema_json_text(Some(schema));
    }
    Ok(Some(
        schema.call_method0(pyo3::intern!(py, "json"))?.extract()?,
    ))
}

/// Whether the type object `obj` is, or keeps at any depth of the types that it keeps as their
/// objects (see `keeps_object_of`), a `UserDefinedType`.
pub(crate) fn holds_udt(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_udt(obj)? || keeps_object_of(py, obj, holds_udt)?)
}

/// The `fromJson` helper `name` of the shim `pyspark.sql.types`, which parses the nested JSON
/// forms of the types as PySpark's: the collations of a struct field by their paths, and the
/// Python class of a `udt`. The core's grammar parses the strings of the atomic types.
fn shim_from_json<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("pyspark.sql.types")?.getattr(name)
}

/// The core metadata of the metadata `d` of a `StructField`, as PySpark's plans send it: of the
/// text of `json.dumps(d)` if it has a length (`if field.metadata is not None and
/// len(field.metadata) > 0`), whose values the core keeps as it writes them: an int of any size,
/// and the `NaN` and `Infinity` of a float that is not finite. PySpark's field keeps any metadata,
/// so `d` may be another value than a dict, such as a list or a str, whose JSON the core keeps as
/// it is (see `Metadata::from_json_value`), and which the server refuses. The errors are those of
/// the length and of `json.dumps` of `d`, as PySpark's plans raise them (`object of type 'int'
/// has no len()`), and that of a str that UTF-8 cannot encode.
fn py_metadata_to_map(py: Python<'_>, d: &Bound<'_, PyAny>) -> PyResult<Metadata> {
    if !metadata_is_sent(d)? {
        return Ok(Metadata::new());
    }
    let s: String = py
        .import("json")?
        .getattr("dumps")?
        .call1((d,))?
        .extract()?;
    metadata_of_dumps(d, &s)
}

/// Whether a plan sends the metadata `d` of a field, as PySpark's `field.metadata is not None and
/// len(field.metadata) > 0`.
fn metadata_is_sent(d: &Bound<'_, PyAny>) -> PyResult<bool> {
    match d.cast::<pyo3::types::PyDict>() {
        Ok(dict) => Ok(!dict.is_empty()),
        Err(_) => Ok(!d.is_none() && d.len()? > 0),
    }
}

/// The core metadata of the text `s` of `json.dumps` of the metadata `d`.
fn metadata_of_dumps(d: &Bound<'_, PyAny>, s: &str) -> PyResult<Metadata> {
    Metadata::from_json_value(s).or_else(|e| {
        // The metadata of the core is JSON of UTF-8, which `json.dumps` escapes a lone surrogate
        // of: the error is that of the first str that UTF-8 cannot encode, as of a name.
        check_metadata_strs(d)?;
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            e.to_string(),
        ))
    })
}

/// The core metadata of the dict `d` of a field of a type that stands for its JSON (see
/// `stand_in_field_level`), or of the JSON value of the field: PySpark's field keeps any dict, and
/// only the uses that write its JSON raise the error of `json.dumps` of it, `json()` and the plans
/// of the type (see `core_field`). So metadata of a value that `json.dumps` refuses, such as
/// a set, bytes or a key of a tuple, stands as none here, and so does metadata of another value
/// whose length raises an `Exception`, such as an int (see `metadata_is_sent`), and metadata of a
/// str that UTF-8 cannot encode, which the JSON of the field writes of the dict itself, as PySpark's
/// `json.dumps` escapes it (see `keep_flags_in_json`), and a plan raises the `UnicodeEncodeError`
/// of (see `py_metadata_to_map`).
fn kept_metadata(py: Python<'_>, d: &Bound<'_, PyAny>) -> PyResult<Metadata> {
    match metadata_is_sent(d) {
        Ok(true) => {}
        Ok(false) => return Ok(Metadata::new()),
        Err(error) if is_exception(py, &error) => return Ok(Metadata::new()),
        Err(error) => return Err(error),
    }
    let dumps = py.import("json")?.getattr("dumps")?.call1((d,));
    let s: String = match dumps {
        Ok(s) => s.extract()?,
        Err(error) if is_exception(py, &error) => return Ok(Metadata::new()),
        Err(error) => return Err(error),
    };
    match metadata_of_dumps(d, &s) {
        // Of JSON nested deeper than the core reads, or of a str that UTF-8 cannot encode (a
        // `UnicodeEncodeError` is a `ValueError`).
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {
            Ok(Metadata::new())
        }
        other => other,
    }
}

/// How a lowering of type objects takes the metadata of their fields.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataOf {
    /// Of the error of `json.dumps` of a dict, as the plans of PySpark raise it.
    Strict,
    /// As `Strict`, for the proto of a plan (see [`py_to_proto_data_type`]), which also takes the
    /// names and the nullable flags of the types as protobuf's string and bool fields take them,
    /// which PySpark's conversion sets them in: of the `TypeError` of a name that is not a str
    /// (see `field_core_name`) and of a flag that is not an int (see `lowered_flag`).
    Plan,
    /// As it was given, for a use that does not write the JSON of the metadata, such as the JSON
    /// value of a type: PySpark's `jsonValue` holds each metadata dict as it is, which only
    /// `json.dumps` of it may refuse, so a dict that `json.dumps` refuses stands as none (see
    /// `kept_metadata`), which the dict of the field replaces in the JSON value (see
    /// `keep_flags_in_json`). The error of a type, as that of a `UserDefinedType` of no JSON,
    /// stays. The mode goes down the lowering, also to the SQL type of a `UserDefinedType`, but
    /// not to the lowerings of the code of a user that it calls, such as its `sqlType()`, which
    /// are of their own.
    AsGiven,
}

/// The `UnicodeEncodeError` of the first str of `value`, of metadata, that UTF-8 cannot encode:
/// of the keys and the values of its dicts and the items of its lists and tuples, in their order.
fn check_metadata_strs(value: &Bound<'_, PyAny>) -> PyResult<()> {
    // A level of a type of many levels, on a stack with room.
    crate::pbmsg::grow(|| check_metadata_strs_level(value))
}

/// [`check_metadata_strs`] of one level.
fn check_metadata_strs_level(value: &Bound<'_, PyAny>) -> PyResult<()> {
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
    // A level of the JSON of a type of many levels, on a stack with room.
    crate::pbmsg::grow(|| ordered_json_text_level(v, field))
}

/// [`ordered_json_text`] of one level of a JSON value.
fn ordered_json_text_level(v: &serde_json::Value, field: bool) -> String {
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
    v: serde_json::Value,
    field: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let text = ordered_json_text(&v, field);
    drop_json_value(v);
    // Of the JSON of a type of many levels, which the decoder of Python recurses over in C, once
    // per level, on a stack with room for them; of a shallow one, as it is.
    let loads = || py.import("json")?.getattr("loads")?.call1((text.as_str(),));
    if json_text_depth(&text) > SHALLOW_JSON_LEVELS {
        grow_for_json(loads)
    } else {
        loads()
    }
}

/// The levels of JSON that Python's decoder of `json` reads on any stack that holds a level of a
/// conversion (of `pbmsg::grow`), of its frames of a few hundred bytes a level.
const SHALLOW_JSON_LEVELS: usize = 64;

/// The levels of the arrays and objects of the JSON text `text`, nested in each other, at their
/// deepest.
fn json_text_depth(text: &str) -> usize {
    let (mut depth, mut deepest, mut in_string, mut escaped) = (0usize, 0usize, false, false);
    for byte in text.bytes() {
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'[' | b'{' => {
                depth += 1;
                deepest = deepest.max(depth);
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    deepest
}

/// Drop `value`, a JSON value of any depth, of no recursion: the drop of `serde_json` recurses
/// once per level of arrays and objects.
fn drop_json_value(value: serde_json::Value) {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            serde_json::Value::Array(items) => pending.extend(items),
            serde_json::Value::Object(entries) => {
                pending.extend(entries.into_iter().map(|(_, item)| item))
            }
            _ => {}
        }
    }
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
    // A level of a type of many levels, such as the schema of a response, on a stack with room.
    crate::pbmsg::grow(|| need_conversion_level(py, data_type))
}

/// [`need_conversion`] of one level of a type.
fn need_conversion_level(py: Python<'_>, data_type: &DataType) -> PyResult<bool> {
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

/// The Python value of core metadata, as PySpark's `json.loads` of the metadata of a field reads
/// it: a dict, or the value of metadata that is not an object (see `Metadata::from_json_value`).
fn metadata_map_to_py<'py>(py: Python<'py>, m: &Metadata) -> PyResult<Bound<'py, PyAny>> {
    // Of no metadata, a new dict, as `json.loads("{}")` makes.
    if m.is_empty() {
        return Ok(pyo3::types::PyDict::new(py).into_any());
    }
    py.import("json")?.getattr("loads")?.call1((m.json(),))
}

/// The metadata of a field made of the metadata `metadata`, as PySpark's `metadata or {}`: the
/// object if it is true, else none, for which the field makes an empty dict.
fn given_metadata<'py>(
    metadata: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match metadata {
        Some(metadata) if metadata.is_truthy()? => Ok(Some(metadata.clone())),
        _ => Ok(None),
    }
}

/// `{**metadata}`, of which PySpark's `StructField.jsonValue` makes the metadata of the collations
/// of the strings of a field (`{**self.metadata, "__COLLATIONS": ...}`): of the `TypeError` of
/// metadata that is not a mapping, as Python's unpacking raises it.
fn unpacked_metadata<'py>(
    py: Python<'py>,
    metadata: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let merged = pyo3::types::PyDict::new(py);
    // SAFETY: both are valid objects of the thread that holds the GIL (`py`).
    if unsafe { pyo3::ffi::PyDict_Update(merged.as_ptr(), metadata.as_ptr()) } < 0 {
        let error = PyErr::fetch(py);
        if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) {
            let class = metadata.get_type().name()?;
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "'{class}' object is not a mapping"
            )));
        }
        return Err(error);
    }
    Ok(merged)
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

    /// As `object.__getstate__` of PySpark's plain class: the `__dict__` of the type, with the
    /// values of the `__slots__` of a subclass that are set, or None when there is neither, which
    /// CPython's own refuses of an object of a native class (see `crate::state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// As `object.__reduce_ex__` of PySpark's plain class (see `type_reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        type_reduce_ex(slf.as_any(), protocol)
    }

    /// A type of its class only pickles as PySpark's plain class does, as its class and its
    /// `__dict__`, of `copyreg.__newobj__(cls)`, which both clients load at every protocol: this
    /// is what `object.__reduce_ex__` gives of PySpark's at the protocols 2 and up, whose pickle
    /// is PySpark's own. So does a type whose state is its `__dict__`, a type of parameters, a
    /// field, an array or a map (see `type_reduce_ex`), where PySpark's `__reduce__` is copyreg's
    /// `_reconstructor` of `object.__new__`, which CPython refuses of a native class. A class of a
    /// core type without a `__reduce__` of its own pickles as `_parse_datatype_json_string` of its
    /// JSON.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if keeps_dict_state(slf.as_any()) {
            return Ok(type_reduce_ex(slf.as_any(), 2)?.unbind());
        }
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
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, also of a pure-Python
    // subclass, such as an `UnparsedDataType`, whose `jsonValue` raises.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of a type of its class only.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if slf.borrow().inner.is_none() {
            return slf.call_method0(pyo3::intern!(py, "typeName"));
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    // As PySpark's `DataType.needConversion`, false of a type of its class only.
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(DataType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        if slf.borrow().inner.is_none() {
            return Ok(false);
        }
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    // As PySpark's `DataType.fromInternal`, of which a type of its class only returns `obj`.
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(DataType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if slf.borrow().inner.is_none() {
            return Ok(obj);
        }
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    // As PySpark's `DataType.toInternal`, of which a type of its class only returns `obj`.
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(DataType.toInternal)]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if slf.borrow().inner.is_none() {
            return Ok(obj);
        }
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    // The type of the DDL string `ddl`, as PySpark's `DataType.fromDDL`, which the server of
    // the active session parses (`_parse_datatype_string`): the concrete type, a StructType
    // for a list of fields. The subclasses inherit this classmethod, so that any of them
    // parses DDL the same way.
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    #[doc = connect_doc!(DataType.fromDDL)]
    fn __obj_from_ddl(cls: &Bound<'_, pyo3::types::PyType>, ddl: &str) -> PyResult<Py<PyAny>> {
        Ok(cls
            .py()
            .import("pyspark.sql.types")?
            .call_method1("_parse_datatype_string", (ddl,))?
            .unbind())
    }
    // The name of the type, as PySpark's classmethod `DataType.typeName`: the name of the class
    // without its `Type`, lowercased, such as `integer` of `IntegerType`, also of a subclass.
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

    // As PySpark's `DataType.simpleString`, `self.typeName()`. The types of a string of their
    // own, such as `bigint`, `decimal(10,2)` or `struct<a:bigint>`, override it, as PySpark's
    // do, so the base method called on one of them directly, as `DataType.simpleString(st)`, is
    // its `typeName()`, `struct`, as in PySpark.
    fn simpleString<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        type_name_of(slf.as_any())
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
    // Of a type of many levels, whose JSON values and copies recurse over its levels as they
    // are made, on a stack with room for a level (the recursions below grow on their own).
    type_level(a.py(), || data_type_eq_level(a, b))
}

/// [`data_type_eq`] of one level: of the same object, true, as Python compares the `__dict__` of
/// PySpark's type, of the same objects in it, without reading them.
fn data_type_eq_level(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    if a.is(b) {
        return Ok(true);
    }
    if !b.is_instance_of::<PyDataType>() {
        return Ok(false);
    }
    if has_state_of_dict(a) {
        if !b.is_instance(&a.get_type())? {
            return Ok(false);
        }
        // Of a struct, whose state the drop-in keeps in Rust, which PySpark's `__dict__` holds.
        if !has_state_of_dict(b) && b.is_instance_of::<PyStructType>() {
            return Ok(false);
        }
        let dict = pyo3::intern!(a.py(), "__dict__");
        return a.getattr(dict)?.eq(b.getattr(dict)?);
    }
    // A type of no core type is not of the class of `a`, nor of a subclass of it.
    if has_state_of_dict(b) || !b.is_instance(&a.get_type())? {
        return Ok(false);
    }
    // A type of parameters, an array and a map, as PySpark's, of the `__dict__` that holds them.
    if keeps_dict_state(a) {
        let dict = pyo3::intern!(a.py(), "__dict__");
        return a.getattr(dict)?.eq(b.getattr(dict)?);
    }
    // The metadata of the fields are compared as Python's `==` compares the dicts of PySpark. A
    // type that holds one that does not lower now compares by its attributes, as PySpark's
    // `__dict__` comparison does (see `lenient_type`).
    let py = a.py();
    let core = match py_to_data_type(a) {
        Ok(core) => core,
        Err(error) if is_exception(py, &error) => return attributes_eq(a, b),
        Err(error) => return Err(error),
    };
    let other = match py_to_data_type(b) {
        Ok(other) => other,
        Err(error) if is_exception(py, &error) => return attributes_eq(a, b),
        Err(error) => return Err(error),
    };
    if !core.eq_as_python(&other) {
        return Ok(false);
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

/// As PySpark's `self.__dict__ == other.__dict__` of the type objects `a` and `b`, of one class
/// (as `data_type_eq` checks), when one holds a type that does not lower now (see
/// `lenient_type`): of the attributes that PySpark's `__init__` sets, in their order, which
/// Python's `==` compares as the values of PySpark's `__dict__`, a type that they hold by its own
/// `__eq__`, and then of the attributes that a user set on a type of a Python subclass.
fn attributes_eq(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let names: &[&str] = if a.is_instance_of::<PyArrayType>() {
        &["elementType", "containsNull"]
    } else if a.is_instance_of::<PyMapType>() {
        &["keyType", "valueType", "valueContainsNull"]
    } else if a.is_instance_of::<PyStructField>() {
        &["name", "dataType", "nullable", "metadata"]
    } else if a.is_instance_of::<PyStructType>() {
        &[
            "fields",
            "names",
            "_needConversion",
            "_needSerializeAnyField",
        ]
    } else {
        &[]
    };
    // Of each attribute as Python's comparison of dicts compares their values: the same object is
    // equal, of no `==` of it.
    for name in names {
        let (x, y) = (a.getattr(*name)?, b.getattr(*name)?);
        if !(x.is(&y) || x.eq(&y)?) {
            return Ok(false);
        }
    }
    if is_subclass_object(a) || is_subclass_object(b) {
        let dict = pyo3::intern!(a.py(), "__dict__");
        return a.getattr(dict)?.eq(b.getattr(dict)?);
    }
    Ok(true)
}

// Define concrete type classes
#[doc = connect_doc!(NullType)]
#[pyclass(subclass, name = "NullType", module = "pyspark.sql.types", extends = PyDataType)]
pub struct PyNullType;

#[pymethods]
impl PyNullType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(NullType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(NullType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(NullType.toInternal)]
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "void").clone().into_any())
    }
    // As PySpark's, which names this type `void`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "void"
    }
}

#[doc = connect_doc!(BooleanType)]
#[pyclass(subclass, name = "BooleanType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyBooleanType;

#[pymethods]
impl PyBooleanType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(BooleanType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(BooleanType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(BooleanType.toInternal)]
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "boolean").clone().into_any())
    }
}

#[doc = connect_doc!(ByteType)]
#[pyclass(subclass, name = "ByteType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyByteType;

#[pymethods]
impl PyByteType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(ByteType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(ByteType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(ByteType.toInternal)]
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

#[doc = connect_doc!(ShortType)]
#[pyclass(subclass, name = "ShortType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyShortType;

#[pymethods]
impl PyShortType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(ShortType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(ShortType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(ShortType.toInternal)]
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

#[doc = connect_doc!(IntegerType)]
#[pyclass(subclass, name = "IntegerType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyIntegerType;

#[pymethods]
impl PyIntegerType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(IntegerType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(IntegerType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(IntegerType.toInternal)]
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

#[doc = connect_doc!(LongType)]
#[pyclass(subclass, name = "LongType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyLongType;

#[pymethods]
impl PyLongType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(LongType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(LongType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(LongType.toInternal)]
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

#[doc = connect_doc!(FloatType)]
#[pyclass(subclass, name = "FloatType", module = "pyspark.sql.types", extends = PyFractionalType)]
pub struct PyFloatType;

#[pymethods]
impl PyFloatType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(FloatType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(FloatType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(FloatType.toInternal)]
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "float").clone().into_any())
    }
}

#[doc = connect_doc!(DoubleType)]
#[pyclass(subclass, name = "DoubleType", module = "pyspark.sql.types", extends = PyFractionalType)]
pub struct PyDoubleType;

#[pymethods]
impl PyDoubleType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(DoubleType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(DoubleType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(DoubleType.toInternal)]
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "double").clone().into_any())
    }
}

#[doc = connect_doc!(DecimalType)]
#[pyclass(subclass, name = "DecimalType", module = "pyspark.sql.types", extends = PyFractionalType)]
// The parameters `precision`, `scale` and `hasPrecisionInfo` are entries of the `__dict__` of the
// type, as PySpark's `__init__` sets them, of any values, which its methods read as PySpark's do
// (see `parameter`), and of which its core type is made (see `parameter_type_of`).
pub struct PyDecimalType;

#[pymethods]
impl PyDecimalType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    // As PySpark's, `"decimal(%d,%d)" % (self.precision, self.scale)`.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Self::simple_string(slf)?.into_any())
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(DecimalType.needConversion)]
    fn __obj_need_conversion(&self) -> bool {
        false
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(DecimalType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(DecimalType.toInternal)]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no parameters until `__init__` sets them, as `DecimalType.__new__(DecimalType)`,
    /// of which pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__`
    /// pickle sets. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
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
        Ok(init_chain!(
            DataType::Decimal {
                precision: 10,
                scale: 0
            },
            PyDecimalType,
            [PyAtomicType, PyNumericType, PyFractionalType],
            value
        ))
    }
    /// As PySpark's `__init__(self, precision=10, scale=0)`, of any values, which sets
    /// `hasPrecisionInfo` true.
    #[pyo3(signature = (precision=py_int(10), scale=py_int(0)))]
    fn __init__(slf: &Bound<'_, Self>, precision: Py<PyAny>, scale: Py<PyAny>) -> PyResult<()> {
        let py = slf.py();
        assign_parameters(
            slf.as_any(),
            &[
                ("precision", precision.into_bound(py)),
                ("scale", scale.into_bound(py)),
                (
                    "hasPrecisionInfo",
                    pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                ),
            ],
        )
    }
    // As PySpark's, `"DecimalType(%d,%d)" % (self.precision, self.scale)`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        let obj = slf.as_any();
        let args = (
            parameter(obj, pyo3::intern!(obj.py(), "precision"))?,
            parameter(obj, pyo3::intern!(obj.py(), "scale"))?,
        );
        percent(slf.py(), "DecimalType(%d,%d)", args)
    }
    // As PySpark's, `"decimal(%d,%d)" % (self.precision, self.scale)`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        let obj = slf.as_any();
        let args = (
            parameter(obj, pyo3::intern!(obj.py(), "precision"))?,
            parameter(obj, pyo3::intern!(obj.py(), "scale"))?,
        );
        percent(slf.py(), "decimal(%d,%d)", args)
    }
}

/// The Python int `value`, a default of a parameter of a type.
fn py_int(value: i64) -> Py<PyAny> {
    Python::attach(|py| {
        value
            .into_pyobject(py)
            .map(|value| value.into_any().unbind())
            .unwrap_or_else(|_| py.None())
    })
}

#[doc = connect_doc!(StringType)]
#[pyclass(subclass, name = "StringType", module = "pyspark.sql.types", extends = PyAtomicType)]
// The parameter `collation` is an entry of the `__dict__` of the type, as PySpark's `__init__`
// sets it (see `parameter`).
pub struct PyStringType;

#[pymethods]
impl PyStringType {
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
    // As PySpark's `DataType.json`, of the `jsonValue` of the type.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    // As PySpark's `StringType.jsonValue`, `self.simpleString()`.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.call_method0(pyo3::intern!(slf.py(), "simpleString"))
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(StringType.needConversion)]
    fn __obj_need_conversion(&self) -> bool {
        false
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(StringType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(StringType.toInternal)]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no collation until `__init__` sets it, as `StringType.__new__(StringType)`, of
    /// which pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__`
    /// pickle sets. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
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
        Ok(init_chain!(
            DataType::String {
                collation: "UTF8_BINARY".to_string()
            },
            PyStringType,
            [PyAtomicType],
            value
        ))
    }
    /// As PySpark's `__init__(self, collation="UTF8_BINARY")`, of any value.
    #[pyo3(signature = (collation=py_str("UTF8_BINARY")))]
    fn __init__(slf: &Bound<'_, Self>, collation: Py<PyAny>) -> PyResult<()> {
        assign_parameters(
            slf.as_any(),
            &[("collation", collation.into_bound(slf.py()))],
        )
    }
    /// As PySpark's, `"StringType()"` of `self.isUTF8BinaryCollation()`, else
    /// `"StringType('%s')" % self.collation`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        let py = slf.py();
        if Self::of_binary_collation(slf)? {
            return Ok(pyo3::types::PyString::new(py, "StringType()"));
        }
        percent(
            py,
            "StringType('%s')",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "collation"),
            )?,),
        )
    }
    // As PySpark's, `"string"` of `self.isUTF8BinaryCollation()`, else
    // `f"string collate {self.collation}"`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        let py = slf.py();
        if Self::of_binary_collation(slf)? {
            return Ok(pyo3::types::PyString::new(py, "string"));
        }
        let collation = parameter(slf.as_any(), pyo3::intern!(slf.as_any().py(), "collation"))?;
        Ok(pyo3::types::PyString::new(py, "string collate {}")
            .call_method1(pyo3::intern!(py, "format"), (collation,))?
            .cast_into::<pyo3::types::PyString>()?)
    }

    // Provider ("spark" for UTF8*, else "icu") for a collation name.
    // Mirrors `StringType.collationProvider`.
    // As PySpark's, of `collationName.startswith("UTF8")`, of the error of another value.
    #[classmethod]
    #[pyo3(name = "collationProvider")]
    fn collation_provider(
        _cls: &Bound<'_, pyo3::types::PyType>,
        collationName: &Bound<'_, PyAny>,
    ) -> PyResult<&'static str> {
        let spark = collationName
            .call_method1(pyo3::intern!(collationName.py(), "startswith"), ("UTF8",))?
            .is_truthy()?;
        Ok(if spark { "spark" } else { "icu" })
    }

    // As PySpark's `StringType.isUTF8BinaryCollation`, `self.collation == "UTF8_BINARY"`.
    #[pyo3(name = "isUTF8BinaryCollation")]
    fn is_utf8_binary_collation<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        parameter(slf.as_any(), pyo3::intern!(slf.as_any().py(), "collation"))?
            .rich_compare("UTF8_BINARY", pyo3::basic::CompareOp::Eq)
    }
}

impl PyStringType {
    /// PySpark's `self.isUTF8BinaryCollation()`, which its `simpleString` and `__repr__` call: of a
    /// Python subclass, or of the class patched over the method, its own; and of the native class,
    /// of its native method, of no call of it.
    fn of_binary_collation(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let py = slf.py();
        if is_native(slf) && native_attribute(&py.get_type::<Self>(), "isUTF8BinaryCollation")? {
            return Self::is_utf8_binary_collation(slf)?.is_truthy();
        }
        slf.call_method0(pyo3::intern!(py, "isUTF8BinaryCollation"))?
            .is_truthy()
    }
}

/// The Python str `value`, a default of a parameter of a type.
fn py_str(value: &str) -> Py<PyAny> {
    Python::attach(|py| pyo3::types::PyString::new(py, value).into_any().unbind())
}

#[doc = connect_doc!(BinaryType)]
#[pyclass(subclass, name = "BinaryType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyBinaryType;

#[pymethods]
impl PyBinaryType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(BinaryType.needConversion)]
    fn __obj_need_conversion(slf: &Bound<'_, Self>) -> PyResult<bool> {
        need_conversion(slf.py(), &py_to_data_type(slf.as_any())?)
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(BinaryType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(BinaryType.toInternal)]
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "binary").clone().into_any())
    }
}

#[doc = connect_doc!(DateType)]
#[pyclass(subclass, name = "DateType", module = "pyspark.sql.types", extends = PyDatetimeType)]
pub struct PyDateType;

#[pymethods]
impl PyDateType {
    /// Number of days from 0001-01-01 to 1970-01-01 (mirrors pyspark's DateType.EPOCH_ORDINAL).
    #[classattr]
    const EPOCH_ORDINAL: i32 = 719163;
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "date").clone().into_any())
    }
}

#[doc = connect_doc!(TimestampType)]
#[pyclass(subclass, name = "TimestampType", module = "pyspark.sql.types", extends = PyDatetimeType)]
pub struct PyTimestampType;

#[pymethods]
impl PyTimestampType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "timestamp").clone().into_any())
    }
}

#[doc = connect_doc!(TimestampNTZType)]
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
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "timestamp_ntz").clone().into_any())
    }
    // As PySpark's, which names this type `timestamp_ntz`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "timestamp_ntz"
    }
}

#[doc = connect_doc!(ArrayType)]
#[pyclass(subclass, name = "ArrayType", module = "pyspark.sql.types", extends = PyDataType)]
// The parameters `elementType` and `containsNull` are entries of the `__dict__` of the type, as
// PySpark's `__init__` sets them, of any values (see `parameter`), and its core type is made of
// them (see `lower_type`).
pub struct PyArrayType;

#[pymethods]
impl PyArrayType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass, or of
    // the class patched over it, its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) || !native_attribute(&slf.py().get_type::<Self>(), "jsonValue")? {
            return json_of_json_value(slf.as_any());
        }
        // Of an array of more levels than the recursion limit, PySpark's text, of the calls of its
        // `jsonValue` of each element, at any depth of its caller.
        type_json_text(slf.as_any()).map_err(|error| {
            if error.is_instance_of::<pyo3::exceptions::PyRecursionError>(slf.py()) {
                pyo3::exceptions::PyRecursionError::new_err(
                    "maximum recursion depth exceeded while calling a Python object",
                )
            } else {
                error
            }
        })
    }
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        Ok(type_json_value(py, slf.as_any())?.0)
    }
    // As PySpark's, of the `needConversion()` of the types that it holds (see
    // `need_conversion_of_object`).
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
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no parameters until `__init__` sets them, as `ArrayType.__new__(ArrayType)`, of
    /// which pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__`
    /// pickle sets. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
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
        Ok(init_chain!(array_kind(), PyArrayType, [], value))
    }
    /// As PySpark's `__init__(self, elementType, containsNull=True)`, which asserts that the
    /// element type is a `DataType`, and sets the parameters as they are given.
    #[pyo3(signature = (elementType, containsNull=default_flag()))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        elementType: &Bound<'_, PyAny>,
        containsNull: Py<PyAny>,
    ) -> PyResult<()> {
        assert_data_type(elementType, "elementType")?;
        assign_parameters(
            slf.as_any(),
            &[
                ("elementType", elementType.clone()),
                ("containsNull", containsNull.into_bound(slf.py())),
            ],
        )
    }
    // As PySpark's, `"ArrayType(%s, %s)" % (self.elementType, str(self.containsNull))`. A level
    // of the repr of a type of many levels, on a stack with room.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        crate::pbmsg::grow(|| {
            let py = slf.py();
            let obj = slf.as_any();
            let element = parameter(obj, pyo3::intern!(py, "elementType"))?;
            let contains_null = parameter(obj, pyo3::intern!(py, "containsNull"))?.str()?;
            percent(py, "ArrayType(%s, %s)", (element, contains_null))
        })
    }
    // As PySpark's, `"array<%s>" % self.elementType.simpleString()` (see `child_simple_string`).
    #[pyo3(name = "simpleString")]
    fn simple_string(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let element = parameter(slf.as_any(), pyo3::intern!(py, "elementType"))?;
        Ok(format!("array<{}>", child_simple_string(py, &element)?))
    }

    // A copy with all nullability fields set true (recursively), as PySpark's
    // `ArrayType.toNullable()`, its `_as_nullable` of the type objects that it holds (see
    // `as_nullable`): the metadata dicts of the fields as they are, and a type that the client
    // keeps as its object, which its JSON does not tell apart, or of a Python subclass, as it is.
    #[pyo3(name = "toNullable")]
    #[doc = connect_doc!(ArrayType.toNullable)]
    fn __at_to_nullable(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(as_nullable(slf.as_any())?.unbind())
    }

    // Build an ArrayType from its JSON value. Mirrors `ArrayType.fromJson`.
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

#[doc = connect_doc!(MapType)]
#[pyclass(subclass, name = "MapType", module = "pyspark.sql.types", extends = PyDataType)]
// The parameters `keyType`, `valueType` and `valueContainsNull` are entries of the `__dict__` of
// the type, as PySpark's `__init__` sets them, of any values (see `parameter`).
pub struct PyMapType;

#[pymethods]
impl PyMapType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass, or of
    // the class patched over it, its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) || !native_attribute(&slf.py().get_type::<Self>(), "jsonValue")? {
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
    // As PySpark's, of the `needConversion()` of the types that it holds (see
    // `need_conversion_of_object`).
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
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no parameters until `__init__` sets them, as `MapType.__new__(MapType)`, of which
    /// pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__` pickle
    /// sets. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
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
        Ok(init_chain!(map_kind(), PyMapType, [], value))
    }
    /// As PySpark's `__init__(self, keyType, valueType, valueContainsNull=True)`, which asserts
    /// that the key and the value types are `DataType`s, and sets the parameters as they are given.
    #[pyo3(signature = (keyType, valueType, valueContainsNull=default_flag()))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        keyType: &Bound<'_, PyAny>,
        valueType: &Bound<'_, PyAny>,
        valueContainsNull: Py<PyAny>,
    ) -> PyResult<()> {
        assert_data_type(keyType, "keyType")?;
        assert_data_type(valueType, "valueType")?;
        assign_parameters(
            slf.as_any(),
            &[
                ("keyType", keyType.clone()),
                ("valueType", valueType.clone()),
                ("valueContainsNull", valueContainsNull.into_bound(slf.py())),
            ],
        )
    }
    // As PySpark's, `"MapType(%s, %s, %s)" % (self.keyType, self.valueType,
    // str(self.valueContainsNull))`. A level of the repr of a type of many levels, on a stack
    // with room.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        crate::pbmsg::grow(|| {
            let py = slf.py();
            let obj = slf.as_any();
            let key = parameter(obj, pyo3::intern!(py, "keyType"))?;
            let value = parameter(obj, pyo3::intern!(py, "valueType"))?;
            let contains_null = parameter(obj, pyo3::intern!(py, "valueContainsNull"))?.str()?;
            percent(py, "MapType(%s, %s, %s)", (key, value, contains_null))
        })
    }
    // As PySpark's, `"map<%s,%s>" % (self.keyType.simpleString(),
    // self.valueType.simpleString())` (see `child_simple_string`).
    #[pyo3(name = "simpleString")]
    fn simple_string(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let obj = slf.as_any();
        let key = parameter(obj, pyo3::intern!(py, "keyType"))?;
        let value = parameter(obj, pyo3::intern!(py, "valueType"))?;
        let key = child_simple_string(py, &key)?;
        Ok(format!("map<{},{}>", key, child_simple_string(py, &value)?))
    }

    // A copy with all nullability fields set true (recursively), as PySpark's
    // `MapType.toNullable()`, its `_as_nullable` of the type objects that it holds (see
    // `as_nullable`).
    #[pyo3(name = "toNullable")]
    #[doc = connect_doc!(MapType.toNullable)]
    fn __mt_to_nullable(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(as_nullable(slf.as_any())?.unbind())
    }

    // Build a MapType from its JSON value. Mirrors `MapType.fromJson`.
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

#[doc = connect_doc!(StructField)]
#[pyclass(
    subclass,
    frozen,
    name = "StructField",
    module = "pyspark.sql.types",
    extends = PyDataType
)]
// The parameters `name`, `dataType`, `nullable` and `metadata` are entries of the `__dict__` of
// the field, as PySpark's `__init__` sets them, of any values (see `parameter`), and its core
// field is made of them (see `core_field`).
pub struct PyStructField {
    /// The UTF-8 text of the name of the field that a lowering last read, of the str object of
    /// the name, as a str is immutable: the lowerings of a field encode its name once, as the
    /// stable ABI of Python 3.9 has no read of the UTF-8 of a str of no copy (see
    /// `field_core_name_of`). An exact str holds no other object, so this holds no cycle.
    name_text: std::sync::Mutex<Option<(Py<pyo3::types::PyString>, String)>>,
}

impl PyStructField {
    /// A field of no parameters, of no name text yet.
    fn empty() -> Self {
        PyStructField {
            name_text: std::sync::Mutex::new(None),
        }
    }
}

/// [`field_core_name`] of the name object `name` of the field `field`: of a str of the field's
/// kept text when it is the str of that text, of no encoding of it (see `PyStructField::name_text`).
fn field_core_name_of(
    field: &Bound<'_, PyStructField>,
    name: &Bound<'_, PyAny>,
    strict: bool,
) -> PyResult<String> {
    let Ok(text) = name.cast_exact::<pyo3::types::PyString>() else {
        return field_core_name(name, strict);
    };
    let this = field.get();
    if let Ok(kept) = this.name_text.lock() {
        if let Some((object, value)) = kept.as_ref() {
            if object.is(text) {
                return Ok(value.clone());
            }
        }
    }
    let value = text.to_cow()?.into_owned();
    let replaced = match this.name_text.lock() {
        Ok(mut kept) => kept.replace((text.clone().unbind(), value.clone())),
        Err(_) => None,
    };
    drop(replaced);
    Ok(value)
}

/// The core field of the field object `field`, of the entries of its `__dict__` as PySpark's
/// conversion of a plan reads them, in their order: its name, as protobuf's string field takes it
/// (see `field_core_name`), its type, as [`lower_child`] lowers it, the truth of its `nullable`,
/// and its metadata, of the error of `json.dumps` of it, or as it stands of a use that writes the
/// JSON of the field (`MetadataOf::AsGiven`, see `kept_metadata`).
pub(crate) fn core_field(field: &Bound<'_, PyAny>, metadata: MetadataOf) -> PyResult<StructField> {
    let py = field.py();
    let plan = metadata == MetadataOf::Plan;
    let name = field_core_name(&parameter(field, pyo3::intern!(py, "name"))?, plan)?;
    let data_type = lower_child(&parameter(field, pyo3::intern!(py, "dataType"))?, metadata)?;
    let nullable = lowered_flag(field, pyo3::intern!(py, "nullable"), metadata)?;
    let lenient = metadata == MetadataOf::AsGiven;
    field_with(field, name, data_type, nullable, lenient)
}

/// Whether `StructField` has no attribute of the class over a parameter of its objects (`name`,
/// `dataType`, `nullable`, `metadata`), which a patch of the class would set: an attribute of a
/// field of the class is then the entry of its `__dict__`, which a lowering of many fields reads
/// once of each (see [`core_field_of_dict`]).
fn fields_read_their_dicts(py: Python<'_>) -> PyResult<bool> {
    use pyo3::intern;
    // The view of the namespace of the class, which shows each patch of the class as it is made,
    // of one read of its `__dict__`, of no view made at each lowering.
    static NAMESPACE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let namespace = NAMESPACE
        .get_or_try_init(py, || {
            py.get_type::<PyStructField>()
                .getattr(intern!(py, "__dict__"))
                .map(Bound::unbind)
        })?
        .bind(py);
    for name in [
        intern!(py, "name"),
        intern!(py, "dataType"),
        intern!(py, "nullable"),
        intern!(py, "metadata"),
    ] {
        if namespace.contains(name)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The `__dict__` of the field `field` of the native class, of the slot of its object, of no call
/// of the getter of `__dict__`: a `#[pyclass(dict)]` keeps it at the positive `__dictoffset__` of
/// the class. None of none yet, which the getter makes.
fn native_field_dict<'py>(
    field: &Bound<'py, PyStructField>,
) -> Option<Bound<'py, pyo3::types::PyDict>> {
    static OFFSET: std::sync::OnceLock<isize> = std::sync::OnceLock::new();
    let py = field.py();
    let offset = *OFFSET.get_or_init(|| {
        py.get_type::<PyStructField>()
            .getattr(pyo3::intern!(py, "__dictoffset__"))
            .and_then(|offset| offset.extract::<isize>())
            .unwrap_or(0)
    });
    if offset <= 0 {
        return None;
    }
    // SAFETY: `field` is an object of the native class itself, of the GIL (`py`), whose layout
    // holds the pointer of its `__dict__`, or null, at `offset`, as PyO3 makes it; the dict is
    // taken as a new reference, which stays valid while the code below runs.
    unsafe {
        let slot = field
            .as_ptr()
            .cast::<u8>()
            .offset(offset)
            .cast::<*mut pyo3::ffi::PyObject>();
        let dict = *slot;
        if dict.is_null() {
            return None;
        }
        Bound::from_borrowed_ptr(py, dict)
            .cast_into::<pyo3::types::PyDict>()
            .ok()
    }
}

/// The entries of the `__dict__` `dict` of a field of the native class, `name`, `dataType`,
/// `nullable` and `metadata`, when it holds those four alone, in that order, as the field's
/// `__init__` makes it: of one walk of the dict, of the keys by the identity of the interned names
/// of the attributes, which `setattr` interns, of the values as new references, which stay valid
/// while the code that reads them runs. None of another dict, which the reads of the attributes
/// take (see `core_field`).
fn canonical_field_entries<'py>(
    dict: &Bound<'py, pyo3::types::PyDict>,
) -> Option<[Bound<'py, PyAny>; 4]> {
    use pyo3::intern;
    let py = dict.py();
    let keys = [
        intern!(py, "name"),
        intern!(py, "dataType"),
        intern!(py, "nullable"),
        intern!(py, "metadata"),
    ];
    // SAFETY: `dict` is a valid dict of the GIL; `PyDict_Next` gives borrowed references, of
    // which each value is taken as a new reference before any code runs that may change the dict.
    unsafe {
        if pyo3::ffi::PyDict_Size(dict.as_ptr()) != 4 {
            return None;
        }
        let mut position: pyo3::ffi::Py_ssize_t = 0;
        let mut key: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
        let mut value: *mut pyo3::ffi::PyObject = std::ptr::null_mut();
        let mut values: [*mut pyo3::ffi::PyObject; 4] = [std::ptr::null_mut(); 4];
        for (expected, slot) in keys.iter().zip(values.iter_mut()) {
            if pyo3::ffi::PyDict_Next(dict.as_ptr(), &mut position, &mut key, &mut value) == 0
                || key != expected.as_ptr()
            {
                return None;
            }
            *slot = value;
        }
        Some(values.map(|value| Bound::from_borrowed_ptr(py, value)))
    }
}

/// [`core_field`] of the field `field` of the native class, of its `__dict__`, which a class of
/// no attribute over a parameter reads as its attributes (see [`fields_read_their_dicts`]), of no
/// call of an attribute: of a dict of the four parameters alone (see `canonical_field_entries`), of
/// an exact str name and an exact bool flag, converted in the order of `core_field`. None of
/// another field, such as of an entry missing, of which `core_field` raises the `AttributeError`.
fn core_field_of_dict(
    field: &Bound<'_, PyStructField>,
    metadata: MetadataOf,
) -> PyResult<Option<StructField>> {
    let py = field.py();
    let Some(dict) = native_field_dict(field) else {
        return Ok(None);
    };
    let Some([name, data_type, nullable, entry]) = canonical_field_entries(&dict) else {
        return Ok(None);
    };
    drop(dict);
    let (Ok(nullable), true) = (
        nullable.cast_exact::<pyo3::types::PyBool>(),
        name.is_exact_instance_of::<pyo3::types::PyString>(),
    ) else {
        return Ok(None);
    };
    let plan = metadata == MetadataOf::Plan;
    let name = field_core_name_of(field, &name, plan)?;
    let data_type = lower_child(&data_type, metadata)?;
    Ok(Some(StructField {
        name,
        data_type,
        nullable: nullable.is_true(),
        metadata: if metadata == MetadataOf::AsGiven {
            kept_metadata(py, &entry)?
        } else {
            py_metadata_to_map(py, &entry)?
        },
    }))
}

/// The core field of the field object `field`, as [`core_field`] reads it, of a level of the types
/// that stand for a type (see [`stand_in_type`]), for a use that does not read its JSON, of the
/// `RecursionError` of a type of more levels than Python's recursion limit, and of no metadata of a
/// dict that `json.dumps` refuses (see `kept_metadata`).
fn stand_in_field_level(field: &Bound<'_, PyAny>) -> PyResult<StructField> {
    let py = field.py();
    let name = field_core_name(&parameter(field, pyo3::intern!(py, "name"))?, false)?;
    let data_type = child_stand_in(&parameter(field, pyo3::intern!(py, "dataType"))?)?;
    field_of_type(field, name, data_type, true)
}

/// The core field of the name `name` and the type `data_type` of the field object `field`, of the
/// truth of its `nullable` and of its metadata, as it stands if `lenient` (see `kept_metadata`).
fn field_of_type(
    field: &Bound<'_, PyAny>,
    name: String,
    data_type: DataType,
    lenient: bool,
) -> PyResult<StructField> {
    let nullable = flag_parameter(field, pyo3::intern!(field.py(), "nullable"))?;
    field_with(field, name, data_type, nullable, lenient)
}

/// The core field of the name `name`, the type `data_type` and the flag `nullable` of the field
/// object `field`, and of its metadata, as it stands if `lenient` (see `kept_metadata`).
fn field_with(
    field: &Bound<'_, PyAny>,
    name: String,
    data_type: DataType,
    nullable: bool,
    lenient: bool,
) -> PyResult<StructField> {
    let py = field.py();
    let metadata = parameter(field, pyo3::intern!(py, "metadata"))?;
    Ok(StructField {
        name,
        data_type,
        nullable,
        metadata: if lenient {
            kept_metadata(py, &metadata)?
        } else {
            py_metadata_to_map(py, &metadata)?
        },
    })
}

/// The core name of a field of the name object `name`: the text of a str, and of another value,
/// for a plan (`strict`), as protobuf's string field takes it, which PySpark's conversion sets it
/// in: bytes of UTF-8 too, and else its `TypeError` (see [`crate::options::str_argument`]). Of
/// another use, the text of its str, which the uses that write the JSON of the field replace with
/// the object (see `keep_flags_in_json`).
fn field_core_name(name: &Bound<'_, PyAny>, strict: bool) -> PyResult<String> {
    if let Ok(text) = name.cast::<pyo3::types::PyString>() {
        return Ok(text.to_cow()?.into_owned());
    }
    if strict {
        return crate::options::str_argument(name);
    }
    Ok(name.str()?.to_cow()?.into_owned())
}

/// The `needConversion` of the field object `field` of the native class, as PySpark's
/// `self.dataType.needConversion()`, of the type object of its `__dict__` (see
/// [`child_need_conversion`]), of the `AttributeError` of a value that has no such method.
fn field_need_conversion(field: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = field.py();
    child_need_conversion(py, &parameter(field, pyo3::intern!(py, "dataType"))?)
}

#[pymethods]
impl PyStructField {
    // --- StructField object-model methods (v4.2.0 parity), operating on the field
    // itself (a StructField is not a DataType, so these do not go through
    // py_to_data_type). ---
    // As PySpark's `"%s:%s" % (self.name, self.dataType.simpleString())` (see
    // `child_simple_string`).
    #[pyo3(name = "simpleString")]
    fn __sf_simple_string(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let obj = slf.as_any();
        let name = parameter(obj, pyo3::intern!(py, "name"))?;
        let data_type = parameter(obj, pyo3::intern!(py, "dataType"))?;
        let simple = child_simple_string(py, &data_type)?;
        Ok(format!("{}:{simple}", name.str()?.to_cow()?))
    }
    // As PySpark's, `StructField.typeName` raises `INVALID_TYPENAME_CALL`: a field has no type
    // name, but its `dataType` has.
    #[pyo3(name = "typeName")]
    fn __sf_type_name(&self, py: Python<'_>) -> PyResult<String> {
        Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "INVALID_TYPENAME_CALL",
            &[],
        ))
    }
    // As PySpark's `DataType.json`, of the `jsonValue` of the field, of a Python subclass, or of
    // the class patched over it, its own.
    #[pyo3(name = "json")]
    fn __sf_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        if !is_native(slf) || !native_attribute(&py.get_type::<Self>(), "jsonValue")? {
            return json_of_json_value(slf.as_any());
        }
        let (json, other) = field_json_value(py, slf)?;
        if other {
            return json_dumps(py, &json);
        }
        Ok(core_field(slf.as_any(), MetadataOf::Strict)?.json())
    }
    #[pyo3(name = "jsonValue")]
    fn __sf_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(field_json_value(slf.py(), slf)?.0)
    }
    // As PySpark's, of the `needConversion()` of the type of the field (see
    // `need_conversion_of_object`).
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
    // Collation metadata for the field (empty when the field has no collations).
    #[pyo3(name = "getCollationMetadata")]
    fn __sf_get_collation_metadata<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        Ok(pyo3::types::PyDict::new(py))
    }

    // Map of collated-field path -> collation name, parsed from the field's
    // `__COLLATIONS` metadata (empty when absent). Mirrors `StructField.getCollationsMap`.
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

    // The "provider.collation" value for a collated string type. Mirrors
    // `StructField.schemaCollationValue` (UTF8_BINARY/UTF8_LCASE are the "spark"
    // provider, everything else "icu").
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
    // The type of the DDL string `ddl`, as PySpark's `DataType.fromDDL`, which `StructField`
    // inherits: the parsed type (a StructType for "a int"), not a StructField.
    #[classmethod]
    #[pyo3(name = "fromDDL")]
    #[doc = connect_doc!(StructField.fromDDL)]
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
    /// a field of no parameters until `__init__` sets them, as `StructField.__new__(StructField)`,
    /// of which pickle's `copyreg.__newobj__` makes the field that PySpark pickled, whose
    /// `__dict__` pickle sets. The class shows the signature of `__init__`, as `inspect.signature`
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
        Ok(field_base())
    }
    /// As PySpark's `__init__(self, name, dataType, nullable=True, metadata=None)`, which asserts
    /// the data type first, then the name, and sets the parameters as they are given, of
    /// `metadata or {}`.
    #[pyo3(signature = (name, dataType, nullable=default_flag(), metadata=None))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        name: &Bound<'_, PyAny>,
        dataType: &Bound<'_, PyAny>,
        nullable: Py<PyAny>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let py = slf.py();
        assert_data_type(dataType, "dataType")?;
        // As `"field name %s should be a string" % (name)`, of the name itself as the arguments.
        if !name.is_instance_of::<pyo3::types::PyString>() {
            let message = percent(py, "field name %s should be a string", name)?;
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                message.unbind(),
            ));
        }
        let metadata = match given_metadata(metadata)? {
            Some(metadata) => metadata,
            None => pyo3::types::PyDict::new(py).into_any(),
        };
        assign_parameters(
            slf.as_any(),
            &[
                ("name", name.clone()),
                ("dataType", dataType.clone()),
                ("nullable", nullable.into_bound(py)),
                ("metadata", metadata),
            ],
        )
    }
    // As PySpark's, `"StructField('%s', %s, %s)" % (self.name, self.dataType,
    // str(self.nullable))`. A level of the repr of a type of many levels, on a stack with room.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        crate::pbmsg::grow(|| {
            let py = slf.py();
            let obj = slf.as_any();
            let name = parameter(obj, pyo3::intern!(py, "name"))?;
            let data_type = parameter(obj, pyo3::intern!(py, "dataType"))?;
            let nullable = parameter(obj, pyo3::intern!(py, "nullable"))?.str()?;
            percent(py, "StructField('%s', %s, %s)", (name, data_type, nullable))
        })
    }
    /// As PySpark's `DataType.__eq__`, `isinstance(other, self.__class__) and self.__dict__ ==
    /// other.__dict__`: of the field itself, true, as Python's comparison of the same `__dict__`,
    /// of the same objects in it, is, of no comparison of them. A level of the comparison of a type
    /// of many levels, on a stack with room.
    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if slf.is(other) {
            return Ok(true);
        }
        if !other.is_instance(&slf.get_type())? {
            return Ok(false);
        }
        let dict = pyo3::intern!(slf.py(), "__dict__");
        type_level(slf.py(), || slf.getattr(dict)?.eq(other.getattr(dict)?))
    }
    /// pyspark: `hash(str(self))` (inherited from `DataType`). Defining `__eq__` alone
    /// would leave StructField unhashable.
    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<isize> {
        slf.str()?.hash()
    }
}

#[doc = connect_doc!(StructType)]
#[pyclass(subclass, name = "StructType", module = "pyspark.sql.types", extends = PyDataType)]
// The parameters `fields`, `names`, `_needConversion` and `_needSerializeAnyField` are entries of
// the `__dict__` of the struct, as PySpark's `__init__` sets them, of any values (see
// `parameter`), and its core type is made of the elements of its `fields` (see `struct_core_fields`).
pub struct PyStructType;

/// The base core type of a struct, which stands for the kind of its class (see
/// `PyDataType::inner`): its fields are an entry of its `__dict__`, which its lowering reads.
fn struct_kind() -> DataType {
    DataType::Struct { fields: Vec::new() }
}

/// The `fields` of the struct `obj`, as PySpark's methods read `self.fields` (see `parameter`).
fn struct_fields<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    parameter(obj, pyo3::intern!(obj.py(), "fields"))
}

/// The elements of the struct `obj` as PySpark's `for f in self` iterates them: `iter(self.fields)`
/// of the native `__iter__`, or the `__iter__` of a Python subclass of its own or of a patch of
/// the class.
fn struct_iter<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
    let py = obj.py();
    if reads_own(obj, "__iter__", &py.get_type::<PyStructType>())? {
        return obj.try_iter();
    }
    struct_fields(obj)?.try_iter()
}

/// The core fields of the struct `obj`, of the elements of its `fields` as they are now, each a
/// `StructField` (see `core_field`), of the metadata as `metadata` takes it.
pub(crate) fn struct_core_fields(
    obj: &Bound<'_, PyAny>,
    metadata: MetadataOf,
) -> PyResult<Vec<StructField>> {
    let py = obj.py();
    let native = py.get_type::<PyStructField>();
    // Of the `__dict__` of each field of the native class, of one read of it, while the class has
    // no attribute over a parameter (see `fields_read_their_dicts`).
    let of_dicts = fields_read_their_dicts(py)?;
    let core = |item: Bound<'_, PyAny>| -> PyResult<StructField> {
        let field = as_struct_field(&item)?;
        note_subclass(&item, &native);
        if of_dicts && item.is_exact_instance_of::<PyStructField>() {
            if let Some(core) = core_field_of_dict(&field, metadata)? {
                return Ok(core);
            }
        }
        core_field(&field, metadata)
    };
    let fields = struct_fields(obj)?;
    // Of a list, of its elements as its iterator gives them, of its length at each step, which
    // sees an element that the code of an element appends, of no iterator object.
    if let Ok(list) = fields.cast_exact::<PyList>() {
        let mut lowered = Vec::with_capacity(list.len());
        let mut index = 0;
        while index < list.len() {
            let item = list.get_item(index)?;
            index += 1;
            lowered.push(core(item)?);
        }
        return Ok(lowered);
    }
    fields.try_iter()?.map(|item| core(item?)).collect()
}

/// The core type of the struct `obj`, as [`struct_core_fields`] reads it, of the types that stand
/// for the types that do not lower now (see [`lenient_type`]), for a use that does not read its
/// JSON, of the `RecursionError` of a type of more levels than Python's recursion limit in it (see
/// [`stand_in_type`]).
fn struct_stand_in(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let fields = struct_fields(obj)?
        .try_iter()?
        .map(|item| {
            let item = item?;
            match item.cast::<PyStructField>() {
                Ok(field) => stand_in_field_level(field),
                // An element that is not a `StructField`, of which the uses read the object, as
                // PySpark's, stands as a field of the null type.
                Err(_) => Ok(StructField {
                    name: String::new(),
                    data_type: DataType::Null,
                    nullable: true,
                    metadata: Metadata::new(),
                }),
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(DataType::Struct { fields })
}

/// As PySpark's `StructType.__init__(self, fields=None)` of the struct `obj`: `fields` and `names`
/// of new empty lists of no fields (`if not fields`), else the fields as they are given and the
/// list of their names (`[f.name for f in fields]`), which asserts that each is a `StructField`
/// then; and `_needConversion` and `_needSerializeAnyField` of the fields as the struct iterates
/// them (see `assign_need_conversion`). Each is assigned through `setattr`, as PySpark's
/// `self.<name> = value`, so of the setters of a subclass and their errors (SPARK-62231).
fn init_struct(obj: &Bound<'_, PyAny>, fields: Option<Bound<'_, PyAny>>) -> PyResult<()> {
    let py = obj.py();
    let given = match fields {
        Some(fields) if fields.is_truthy()? => Some(fields),
        _ => None,
    };
    match given {
        None => assign_parameters(
            obj,
            &[
                ("fields", PyList::empty(py).into_any()),
                ("names", PyList::empty(py).into_any()),
            ],
        )?,
        Some(fields) => {
            obj.setattr(pyo3::intern!(py, "fields"), &fields)?;
            // As `[f.name for f in fields]`, which reads the name of each item first: an item
            // without one, such as a character of a str or a tuple, is an AttributeError.
            let names = PyList::empty(py);
            for item in fields.try_iter().map_err(|_| not_iterable(&fields))? {
                let item = item?;
                match item.getattr(pyo3::intern!(py, "name")) {
                    Ok(name) => names.append(name)?,
                    Err(e) if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
                        return Err(no_attribute(&item, "name"));
                    }
                    Err(e) => return Err(e),
                }
            }
            obj.setattr(pyo3::intern!(py, "names"), names)?;
            for item in fields.try_iter()? {
                if !item?.is_instance_of::<PyStructField>() {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "fields should be a list of StructField",
                    ));
                }
            }
        }
    }
    assign_need_conversion(obj)
}

/// As PySpark's `self._needConversion = [f.needConversion() for f in self]` and
/// `self._needSerializeAnyField = any(self._needConversion)` of the struct `obj`, of the elements
/// as it iterates them (see `struct_iter`): of a field of the native class, the `needConversion` of
/// its type (see `field_need_conversion`), and of another element, its own method, of what it
/// returns.
fn assign_need_conversion(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = obj.py();
    let need = PyList::empty(py);
    for item in struct_iter(obj)? {
        let item = item?;
        let value = match item.cast::<PyStructField>() {
            Ok(field) if !is_subclass_object(&item) => {
                pyo3::types::PyBool::new(py, field_need_conversion(field.as_any())?)
                    .to_owned()
                    .into_any()
            }
            _ => item.call_method0(pyo3::intern!(py, "needConversion"))?,
        };
        need.append(value)?;
    }
    let mut any = false;
    for value in need.iter() {
        if value.is_truthy()? {
            any = true;
            break;
        }
    }
    obj.setattr(pyo3::intern!(py, "_needConversion"), need)?;
    obj.setattr(
        pyo3::intern!(py, "_needSerializeAnyField"),
        pyo3::types::PyBool::new(py, any),
    )
}

#[pymethods]
impl PyStructType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass, or of
    // the class patched over it, its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) || !native_attribute(&slf.py().get_type::<Self>(), "jsonValue")? {
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
    // As PySpark's, true, without reading the types of the fields (see
    // `need_conversion_of_object`).
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
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a struct of no parameters until `__init__` sets them, as `StructType.__new__(StructType)`,
    /// of which pickle's `copyreg.__newobj__` makes the struct that PySpark pickled, whose
    /// `__dict__` pickle sets. The class shows the signature of `__init__`, as `inspect.signature`
    /// of PySpark's class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(fields=None)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        Ok(init_chain!(struct_kind(), PyStructType, [], value))
    }
    /// As PySpark's `__init__(self, fields=None)` (see `init_struct`).
    #[pyo3(signature = (fields=None))]
    fn __init__(slf: &Bound<'_, Self>, fields: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        // Of a type of many levels, whose lowering and copies recurse over its levels, on a
        // stack with room for a level (the recursions below grow on their own).
        crate::pbmsg::grow(|| init_struct(slf.as_any(), fields))
    }

    // Append a field and return `self` (chainable, mirroring `StructType.add`):
    // `StructType().add("a", "integer").add("b", StringType())`. As PySpark's, the field goes to
    // `self.fields` and its name to `self.names`, of their `append`, and `_needConversion` and
    // `_needSerializeAnyField` are assigned again.
    #[pyo3(
        signature = (field, data_type=None, nullable=default_flag(), metadata=None),
        text_signature = "($self, field, data_type=None, nullable=True, metadata=None)"
    )]
    #[doc = connect_doc!(StructType.add)]
    fn add<'py>(
        slf: Bound<'py, Self>,
        field: &Bound<'_, PyAny>,
        data_type: Option<&Bound<'_, PyAny>>,
        nullable: Py<PyAny>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = field.py();
        let obj = slf.as_any();
        let append = pyo3::intern!(py, "append");
        let names = pyo3::intern!(py, "names");
        if field.is_instance_of::<PyStructField>() {
            struct_fields(obj)?.call_method1(append, (field,))?;
            let name = field.getattr(pyo3::intern!(py, "name"))?;
            parameter(obj, names)?.call_method1(append, (name,))?;
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
            // As PySpark's `StructField(field, data_type_f, nullable, metadata)`, of the asserts of
            // its `__init__`, and the name `field` itself.
            let object =
                py.get_type::<PyStructField>()
                    .call1((field, type_object, nullable, metadata))?;
            struct_fields(obj)?.call_method1(append, (object,))?;
            parameter(obj, names)?.call_method1(append, (field,))?;
        }
        assign_need_conversion(obj)?;
        Ok(slf)
    }

    // As PySpark's, `len(self.fields)`.
    fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
        struct_fields(slf.as_any())?.len()
    }

    /// Index by position (`st[0]`) or by name (`st["a"]`), returning a `StructField`
    /// (mirrors `StructType.__getitem__`; a slice yields a new `StructType`), with the errors of
    /// PySpark's: `KEY_NOT_EXISTS` of a name, `INDEX_OUT_OF_RANGE` of an index, and
    /// `NOT_EXPECTED_TYPE` of another key. A name is looked up in the elements as the struct
    /// iterates them (`for field in self`), and an index and a slice in `self.fields`.
    fn __getitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        use crate::column::raise_pyspark;
        let py = slf.py();
        let obj = slf.as_any();
        if key.is_instance_of::<pyo3::types::PyString>() {
            for item in struct_iter(obj)? {
                let item = item?;
                // As PySpark's `field.name == key`, of the `AttributeError` of none (see
                // `parameter`).
                let found = match item.cast::<PyStructField>() {
                    Ok(field) => parameter(field, pyo3::intern!(py, "name"))?.eq(key)?,
                    Err(_) => item.getattr(pyo3::intern!(py, "name"))?.eq(key)?,
                };
                if found {
                    return Ok(item.unbind());
                }
            }
            return Err(raise_pyspark(
                py,
                "PySparkKeyError",
                "KEY_NOT_EXISTS",
                &[("key", &key.str()?.to_cow()?)],
            ));
        }
        if key.is_instance_of::<pyo3::types::PyInt>() {
            // As PySpark's `self.fields[key]`, of the `INDEX_OUT_OF_RANGE` of its `IndexError`.
            return match struct_fields(obj)?.get_item(key) {
                Ok(item) => Ok(item.unbind()),
                Err(error) if error.is_instance_of::<pyo3::exceptions::PyIndexError>(py) => {
                    Err(raise_pyspark(
                        py,
                        "PySparkIndexError",
                        "INDEX_OUT_OF_RANGE",
                        &[("arg_name", "StructType"), ("index", &key.str()?.to_cow()?)],
                    ))
                }
                Err(error) => Err(error),
            };
        }
        if key.is_instance_of::<pyo3::types::PySlice>() {
            // As PySpark's `StructType(self.fields[key])`.
            let sliced = struct_fields(obj)?.get_item(key)?;
            return Ok(py.get_type::<PyStructType>().call1((sliced,))?.unbind());
        }
        Err(crate::stat_arguments::not_expected_type(
            key,
            "key",
            "int, slice or str",
        ))
    }

    // As PySpark's, `iter(self.fields)`.
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        struct_fields(slf.as_any())?.try_iter()
    }

    // As PySpark's `"struct<%s>" % (",".join(f.simpleString() for f in self))`: of the
    // `simpleString()` of each element as the struct iterates them, of a field of the native class
    // of its own simple string (see `child_simple_string`), and of Python's `",".join` of a text
    // that is not a str, of its `TypeError`.
    #[pyo3(name = "simpleString")]
    fn simple_string(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let mut texts: Vec<String> = Vec::new();
        let mut joined: Option<Bound<'_, PyList>> = None;
        for item in struct_iter(slf.as_any())? {
            let item = item?;
            match item.cast::<PyStructField>() {
                Ok(field) if !is_subclass_object(&item) => {
                    let text = PyStructField::__sf_simple_string(field)?;
                    match &joined {
                        Some(parts) => parts.append(text)?,
                        None => texts.push(text),
                    }
                }
                // Of the element's own method, of its `AttributeError` of none: of a text that
                // is not a str, the texts are joined as PySpark's are, after every element gave
                // its own.
                _ => {
                    let text = item.call_method0(pyo3::intern!(py, "simpleString"))?;
                    match (&joined, text.cast::<pyo3::types::PyString>()) {
                        (None, Ok(text)) => texts.push(text.to_cow()?.into_owned()),
                        (Some(parts), _) => parts.append(text)?,
                        (None, Err(_)) => {
                            let parts = PyList::new(py, &texts)?;
                            parts.append(text)?;
                            joined = Some(parts);
                        }
                    }
                }
            }
        }
        match joined {
            Some(parts) => {
                let text = pyo3::types::PyString::new(py, ",").call_method1("join", (parts,))?;
                Ok(format!("struct<{}>", text.extract::<String>()?))
            }
            None => Ok(format!("struct<{}>", texts.join(","))),
        }
    }

    // As PySpark's `list(self.names)`.
    #[pyo3(name = "fieldNames")]
    #[doc = connect_doc!(StructType.fieldNames)]
    fn __st_field_names<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let names = parameter(slf.as_any(), pyo3::intern!(py, "names"))?;
        py.get_type::<PyList>().call1((names,))
    }

    // DDL string. Mirrors `StructType.toDDL()`, which asks the server of the active session to
    // convert the JSON of the schema, as Spark writes DDL.
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

    // Tree string. Mirrors `StructType.treeString(maxDepth)`, of no docstring, as PySpark's.
    // Use an integer default so explicit None raises TypeError, matching PySpark.
    #[allow(non_snake_case)]
    #[pyo3(
        name = "treeString",
        signature = (maxDepth = Python::attach(|py| pyo3::types::PyInt::new(py, i32::MAX).into_any().unbind())),
        text_signature = "($self, maxDepth=2147483647)"
    )]
    fn __st_tree_string<'py>(
        slf: &Bound<'py, Self>,
        maxDepth: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let depth = normalize_tree_depth(maxDepth.bind(py))?;
        // Of a struct whose fields hold an element that is not a `StructField`, at any level, as
        // PySpark's, of the `_build_formatted_string` of each element, of its `AttributeError`
        // of none (`_foreign_tree_string` of the shim).
        // Of the lowering that the tree string reads, of the walk of the foreign fields only of a
        // type that may hold them (see `lower_noting_subclasses`).
        let (lowered, maybe_foreign) = lower_noting_subclasses(slf.as_any(), MetadataOf::Strict);
        if foreign_of_lowering(slf.as_any(), maybe_foreign)? {
            return py
                .import("pyspark.sql.types")?
                .getattr("_foreign_tree_string")?
                .call1((slf, depth));
        }
        // As PySpark's, which reads the type objects, not their JSON: a type that does not lower
        // now stands as the null type, and its name is its object's (see `lenient_type`). A type
        // of more levels than Python's recursion limit, such as one that holds itself, is read to
        // the levels that the string holds, as PySpark's walk of the objects reads them, which
        // raises its `RecursionError` of more (see `tree_stand_in`): its flags and names are
        // those of its objects, as the walks of them would not end.
        let (data_type, stand_ins, walked) = match lowered {
            Ok(data_type) => (data_type, false, false),
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyRecursionError>(py) => {
                let levels = usize::try_from(depth).unwrap_or(usize::MAX);
                (tree_stand_in(slf.as_any(), levels)?, true, true)
            }
            Err(error) if is_exception(py, &error) => {
                // Of a field of a name that UTF-8 cannot encode, at any depth, as PySpark's, of the
                // `_build_formatted_string` of each element (see below).
                if holds_other_name(py, slf.as_any())? {
                    return py
                        .import("pyspark.sql.types")?
                        .getattr("_foreign_tree_string")?
                        .call1((slf, depth));
                }
                let stand_in = struct_stand_in(slf.as_any());
                match stand_in {
                    Ok(data_type) => (data_type, true, false),
                    // Of a type that does not lower now that holds one of more levels than the
                    // limit, such as itself.
                    Err(error)
                        if error.is_instance_of::<pyo3::exceptions::PyRecursionError>(py) =>
                    {
                        let levels = usize::try_from(depth).unwrap_or(usize::MAX);
                        (tree_stand_in(slf.as_any(), levels)?, true, true)
                    }
                    Err(error) => return Err(error),
                }
            }
            Err(error) => return Err(error),
        };
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
        let other_flags = walked || has_other_flag(py, slf.as_any())?;
        // A field of a name that is not a str, at any depth, which PySpark writes of
        // `escape_meta_characters(name)`, of the error of a value that has no `replace` of strs, as
        // PySpark's `_build_formatted_string` of each element (`_foreign_tree_string` of the shim).
        if other_flags && !walked && holds_other_name(py, slf.as_any())? {
            return py
                .import("pyspark.sql.types")?
                .getattr("_foreign_tree_string")?
                .call1((slf, depth));
        }
        let subclasses = walked || keeps_subclass_objects(py, slf.as_any())?;
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
                DataType::Null if stand_ins => {
                    keep(stand_in_tree_type_name(slf.as_any(), path, subclasses))
                }
                _ if subclasses => keep(subclass_tree_type_name(slf.as_any(), path)),
                _ => None,
            },
        );
        if let Some(e) = error.into_inner() {
            return Err(e);
        }
        let text =
            text.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(pyo3::types::PyString::new(py, &text).into_any())
    }

    // A copy with all fields made nullable (recursively), as PySpark's `StructType.toNullable()`,
    // of the type objects that it holds (see `ArrayType`'s): each field of the metadata dict of
    // the field, as PySpark's `_as_nullable` passes `metadata=field.metadata`.
    #[pyo3(name = "toNullable")]
    #[doc = connect_doc!(StructType.toNullable)]
    fn __st_to_nullable(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(as_nullable(slf.as_any())?.unbind())
    }

    // Build a StructType from its JSON value. Mirrors `StructType.fromJson`.
    #[classmethod]
    #[pyo3(name = "fromJson")]
    #[doc = connect_doc!(StructType.fromJson)]
    fn __st_from_json(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        json: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        Ok(shim_from_json(py, "_struct_type_from_json")?
            .call1((json,))?
            .unbind())
    }

    // As PySpark's, `"StructType([%s])" % ", ".join(str(field) for field in self)`. A level of
    // the repr of a type of many levels, on a stack with room.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        crate::pbmsg::grow(|| {
            let py = slf.py();
            let parts = PyList::empty(py);
            for field in struct_iter(slf.as_any())? {
                parts.append(field?.str()?)?;
            }
            let joined = pyo3::types::PyString::new(py, ", ").call_method1("join", (parts,))?;
            percent(py, "StructType([%s])", (joined,))
        })
    }
}

#[doc = connect_doc!(CharType)]
#[pyclass(subclass, name = "CharType", module = "pyspark.sql.types", extends = PyAtomicType)]
// The parameter `length` is an entry of the `__dict__` of the type, as PySpark's `__init__` sets
// it (see `parameter`).
pub struct PyCharType;

#[pymethods]
impl PyCharType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    // As PySpark's, `"char(%d)" % (self.length)`.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(percent(
            slf.py(),
            "char(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "length"),
            )?,),
        )?
        .into_any())
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(CharType.needConversion)]
    fn __obj_need_conversion(&self) -> bool {
        false
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(CharType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(CharType.toInternal)]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no `length` until `__init__` sets it, as `CharType.__new__(CharType)`, of which
    /// pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__` pickle
    /// sets. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
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
            PyCharType,
            [PyAtomicType],
            value
        ))
    }
    /// As PySpark's `__init__`, of any value.
    fn __init__(slf: &Bound<'_, Self>, length: Py<PyAny>) -> PyResult<()> {
        assign_parameters(slf.as_any(), &[("length", length.into_bound(slf.py()))])
    }
    // As PySpark's, `"CharType(%d)" % (self.length)`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        percent(
            slf.py(),
            "CharType(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "length"),
            )?,),
        )
    }
    // As PySpark's, `"char(%d)" % (self.length)`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        percent(
            slf.py(),
            "char(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "length"),
            )?,),
        )
    }
}

#[doc = connect_doc!(VarcharType)]
#[pyclass(subclass, name = "VarcharType", module = "pyspark.sql.types", extends = PyAtomicType)]
// The parameter `length` is an entry of the `__dict__` of the type, as PySpark's `__init__` sets
// it (see `parameter`).
pub struct PyVarcharType;

#[pymethods]
impl PyVarcharType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    // As PySpark's, `"varchar(%d)" % (self.length)`.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(percent(
            slf.py(),
            "varchar(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "length"),
            )?,),
        )?
        .into_any())
    }
    #[pyo3(name = "needConversion")]
    #[doc = connect_doc!(VarcharType.needConversion)]
    fn __obj_need_conversion(&self) -> bool {
        false
    }
    #[pyo3(name = "fromInternal")]
    #[doc = connect_doc!(VarcharType.fromInternal)]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    #[pyo3(name = "toInternal")]
    #[doc = connect_doc!(VarcharType.toInternal)]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", obj)
    }
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no `length` until `__init__` sets it, as `VarcharType.__new__(VarcharType)`, of which
    /// pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__` pickle
    /// sets. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
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
            PyVarcharType,
            [PyAtomicType],
            value
        ))
    }
    /// As PySpark's `__init__`, of any value.
    fn __init__(slf: &Bound<'_, Self>, length: Py<PyAny>) -> PyResult<()> {
        assign_parameters(slf.as_any(), &[("length", length.into_bound(slf.py()))])
    }
    // As PySpark's, `"VarcharType(%d)" % (self.length)`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        percent(
            slf.py(),
            "VarcharType(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "length"),
            )?,),
        )
    }
    // As PySpark's, `"varchar(%d)" % (self.length)`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        percent(
            slf.py(),
            "varchar(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "length"),
            )?,),
        )
    }
}

#[doc = connect_doc!(TimeType)]
#[pyclass(subclass, name = "TimeType", module = "pyspark.sql.types", extends = PyAnyTimeType)]
// The parameter `precision` is an entry of the `__dict__` of the type, as PySpark's `__init__` sets
// it (see `parameter`).
pub struct PyTimeType;

#[pymethods]
impl PyTimeType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    // As PySpark's, `"time(%d)" % (self.precision)`.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(percent(
            slf.py(),
            "time(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "precision"),
            )?,),
        )?
        .into_any())
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(&self) -> bool {
        true
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
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no `precision` until `__init__` sets it, as `TimeType.__new__(TimeType)`, of which
    /// pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__` pickle
    /// sets. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(precision=6)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        Ok(init_chain!(
            DataType::Time { precision: 6 },
            PyTimeType,
            [PyAtomicType, PyDatetimeType, PyAnyTimeType],
            value
        ))
    }
    /// As PySpark's `__init__`, of any value.
    #[pyo3(signature = (precision=py_int(6)))]
    fn __init__(slf: &Bound<'_, Self>, precision: Py<PyAny>) -> PyResult<()> {
        assign_parameters(
            slf.as_any(),
            &[("precision", precision.into_bound(slf.py()))],
        )
    }
    // As PySpark's, `"TimeType(%d)" % (self.precision)`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        percent(
            slf.py(),
            "TimeType(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "precision"),
            )?,),
        )
    }
    // As PySpark's, `"time(%d)" % (self.precision)`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        percent(
            slf.py(),
            "time(%d)",
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "precision"),
            )?,),
        )
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
        py_json_value(py, v, false)
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
        py_json_value(py, v, false)
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

#[doc = connect_doc!(CalendarIntervalType)]
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
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "interval").clone().into_any())
    }
    // As PySpark's, which names this type `interval`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "interval"
    }
}

/// As PySpark's `_str_repr` of an interval type, its `simpleString` and `jsonValue`: the names of
/// its fields of `cls._fields`, of the class of PySpark's method (`DayTimeIntervalType._fields`,
/// also of a subclass), `interval day to second`, or `interval day` of one, of the `KeyError` of
/// a field that is not one of them.
fn interval_str_repr<'py>(
    obj: &Bound<'py, PyAny>,
    cls: &Bound<'py, pyo3::types::PyType>,
) -> PyResult<Bound<'py, pyo3::types::PyString>> {
    let py = obj.py();
    let fields = cls.getattr(pyo3::intern!(py, "_fields"))?;
    let start = fields.get_item(parameter(obj, pyo3::intern!(obj.py(), "startField"))?)?;
    let end = fields.get_item(parameter(obj, pyo3::intern!(obj.py(), "endField"))?)?;
    if start.eq(&end)? {
        return percent(py, "interval %s", (start,));
    }
    percent(py, "interval %s to %s", (start, end))
}

/// As PySpark's `__init__(self, startField=None, endField=None)` of an interval type of the
/// fields of `cls._fields`: both of `default` of none, the start field for the end without one,
/// and a `PySparkRuntimeError` of `INVALID_INTERVAL_CASTING`, of the `str` of each, of a field that
/// is not a key of the fields (`startField not in fields.keys()`, of the `TypeError` of an
/// unhashable one); else the fields as they were given, which it sets.
fn init_interval<'py>(
    obj: &Bound<'py, PyAny>,
    cls: &Bound<'py, pyo3::types::PyType>,
    start: Option<Bound<'py, PyAny>>,
    end: Option<Bound<'py, PyAny>>,
    default: (i32, i32),
) -> PyResult<()> {
    let py = obj.py();
    let none = || py.None().into_bound(py);
    let given = |value: Option<Bound<'py, PyAny>>| value.filter(|value| !value.is_none());
    let (start, end) = match (given(start), given(end)) {
        (None, None) => (
            default.0.into_pyobject(py)?.into_any(),
            default.1.into_pyobject(py)?.into_any(),
        ),
        (Some(start), None) => (start.clone(), start),
        (start, end) => (start.unwrap_or_else(none), end.unwrap_or_else(none)),
    };
    let keys = cls
        .getattr(pyo3::intern!(py, "_fields"))?
        .call_method0(pyo3::intern!(py, "keys"))?;
    if !keys.contains(&start)? || !keys.contains(&end)? {
        return Err(invalid_interval_casting(
            py,
            &start.str()?.to_string(),
            &end.str()?.to_string(),
        ));
    }
    assign_parameters(obj, &[("startField", start), ("endField", end)])
}

/// As PySpark's `__repr__` of an interval type, `"%s(%d, %d)" % (type(self).__name__,
/// self.startField, self.endField)`.
fn interval_repr<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
    let args = (
        obj.get_type().name()?,
        parameter(obj, pyo3::intern!(obj.py(), "startField"))?,
        parameter(obj, pyo3::intern!(obj.py(), "endField"))?,
    );
    percent(obj.py(), "%s(%d, %d)", args)
}

/// The dict of the fields of an interval type, as PySpark's class attribute `_fields`, of their
/// numbers and names, and its inverse, `_inverted_fields`, if `inverted`.
fn interval_field_names(
    py: Python<'_>,
    names: &[&str],
    inverted: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    for (number, name) in names.iter().enumerate() {
        if inverted {
            dict.set_item(*name, number)?;
        } else {
            dict.set_item(number, *name)?;
        }
    }
    Ok(dict.unbind())
}

#[doc = connect_doc!(YearMonthIntervalType)]
#[pyclass(
    subclass,
    name = "YearMonthIntervalType",
    module = "pyspark.sql.types",
    extends = PyAnsiIntervalType
)]
// The parameters `startField` and `endField` are entries of the `__dict__` of the type, as
// PySpark's `__init__` sets them (see `parameter`).
pub struct PyYearMonthIntervalType;

#[pymethods]
impl PyYearMonthIntervalType {
    #[classattr]
    const YEAR: i32 = 0;
    #[classattr]
    const MONTH: i32 = 1;
    #[classattr]
    fn _fields(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        interval_field_names(py, &["year", "month"], false)
    }
    #[classattr]
    fn _inverted_fields(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        interval_field_names(py, &["year", "month"], true)
    }
    // As PySpark's `_str_repr`, of the fields of `YearMonthIntervalType._fields`.
    fn _str_repr<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        interval_str_repr(slf.as_any(), &slf.py().get_type::<Self>())
    }
    // As PySpark's `simpleString = _str_repr`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        interval_str_repr(slf.as_any(), &slf.py().get_type::<Self>())
    }
    // As PySpark's `jsonValue = _str_repr`.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(interval_str_repr(slf.as_any(), &slf.py().get_type::<Self>())?.into_any())
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(&self) -> bool {
        // As PySpark's, of `PYSPARK_YM_INTERVAL_LEGACY`, read at each call.
        std::env::var("PYSPARK_YM_INTERVAL_LEGACY").ok().as_deref() != Some("1")
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
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no fields until `__init__` sets them, as `YearMonthIntervalType.__new__(YearMonthIntervalType)`, of which
    /// pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__` pickle
    /// sets. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
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
        Ok(init_chain!(
            DataType::YearMonthInterval {
                start_field: 0,
                end_field: 1,
            },
            PyYearMonthIntervalType,
            [PyAtomicType, PyAnsiIntervalType],
            value
        ))
    }
    /// As PySpark's `__init__(self, startField=None, endField=None)`, of the fields of
    /// `YearMonthIntervalType._fields`, YEAR and MONTH.
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        startField: Option<Bound<'_, PyAny>>,
        endField: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        init_interval(
            slf.as_any(),
            &slf.py().get_type::<Self>(),
            startField,
            endField,
            (0, 1),
        )
    }
    /// As PySpark's, `"%s(%d, %d)" % (type(self).__name__, self.startField, self.endField)`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        interval_repr(slf.as_any())
    }
}

#[doc = connect_doc!(DayTimeIntervalType)]
#[pyclass(
    subclass,
    name = "DayTimeIntervalType",
    module = "pyspark.sql.types",
    extends = PyAnsiIntervalType
)]
// The parameters `startField` and `endField` are entries of the `__dict__` of the type, as
// PySpark's `__init__` sets them (see `parameter`).
pub struct PyDayTimeIntervalType;

#[pymethods]
impl PyDayTimeIntervalType {
    #[classattr]
    const DAY: i32 = 0;
    #[classattr]
    const HOUR: i32 = 1;
    #[classattr]
    const MINUTE: i32 = 2;
    #[classattr]
    const SECOND: i32 = 3;
    #[classattr]
    fn _fields(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        interval_field_names(py, &["day", "hour", "minute", "second"], false)
    }
    #[classattr]
    fn _inverted_fields(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        interval_field_names(py, &["day", "hour", "minute", "second"], true)
    }
    // As PySpark's `_str_repr`, of the fields of `DayTimeIntervalType._fields`.
    fn _str_repr<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        interval_str_repr(slf.as_any(), &slf.py().get_type::<Self>())
    }
    // As PySpark's `simpleString = _str_repr`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        interval_str_repr(slf.as_any(), &slf.py().get_type::<Self>())
    }
    // As PySpark's `jsonValue = _str_repr`.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(interval_str_repr(slf.as_any(), &slf.py().get_type::<Self>())?.into_any())
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        json_of_json_value(slf.as_any())
    }
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(&self) -> bool {
        true
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
    /// As the `object.__new__` of PySpark's plain class, of any arguments, which `__init__` takes:
    /// a type of no fields until `__init__` sets them, as `DayTimeIntervalType.__new__(DayTimeIntervalType)`, of which
    /// pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__` pickle
    /// sets. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
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
        Ok(init_chain!(
            DataType::DayTimeInterval {
                start_field: 0,
                end_field: 3,
            },
            PyDayTimeIntervalType,
            [PyAtomicType, PyAnsiIntervalType],
            value
        ))
    }
    /// As PySpark's `__init__(self, startField=None, endField=None)`, of the fields of
    /// `DayTimeIntervalType._fields`, DAY, HOUR, MINUTE and SECOND.
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn __init__(
        slf: &Bound<'_, Self>,
        startField: Option<Bound<'_, PyAny>>,
        endField: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        init_interval(
            slf.as_any(),
            &slf.py().get_type::<Self>(),
            startField,
            endField,
            (0, 3),
        )
    }
    /// As PySpark's, `"%s(%d, %d)" % (type(self).__name__, self.startField, self.endField)`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyString>> {
        interval_repr(slf.as_any())
    }
}

#[doc = connect_doc!(VariantType)]
#[pyclass(subclass, name = "VariantType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyVariantType;

#[pymethods]
impl PyVariantType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    // As PySpark's `DataType.json`, of the `jsonValue` of the type, of a Python subclass its own.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        if !is_native(slf) {
            return json_of_json_value(slf.as_any());
        }
        Ok(py_to_data_type(slf.as_any())?.json())
    }
    // As PySpark's `DataType.jsonValue`, the `typeName()` of the type, of a Python subclass its
    // own.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        let v = py_to_data_type(slf.as_any())?.json_value();
        py_json_value(py, v, false)
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
    // As PySpark's `DataType.simpleString`, the `typeName()` of the type, of a Python subclass
    // its own.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        if !is_native(slf) {
            return type_name_of(slf.as_any());
        }
        Ok(pyo3::intern!(slf.py(), "variant").clone().into_any())
    }
}

#[doc = connect_doc!(GeometryType)]
#[pyclass(subclass, name = "GeometryType", module = "pyspark.sql.types", extends = PySpatialType)]
// The parameters `srid` and `_crs` are entries of the `__dict__` of the type, as PySpark's
// `__init__` sets them (see `parameter`).
pub struct PyGeometryType;

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
    /// a type of no SRID until `__init__` sets it, as `GeometryType.__new__(GeometryType)`, of which
    /// pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__` pickle
    /// sets. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(srid)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        Ok(init_chain!(
            DataType::Geometry {
                srid: Self::DEFAULT_SRID
            },
            PyGeometryType,
            [PyAtomicType, PySpatialType],
            value
        ))
    }
    /// As PySpark's `__init__(self, srid)`: of `srid == "ANY"`, the mixed SRID and its CRS; else
    /// an int of a coordinate reference system that PySpark knows, as it is, and its CRS, or the
    /// `ST_INVALID_SRID_VALUE` of another value.
    fn __init__(slf: &Bound<'_, Self>, srid: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let entries = if srid.eq("ANY")? {
            vec![
                ("srid", Self::MIXED_SRID.into_pyobject(py)?.into_any()),
                (
                    "_crs",
                    pyo3::types::PyString::new(py, Self::MIXED_CRS).into_any(),
                ),
            ]
        } else {
            let checked = spatial_type_srid(srid, false)?;
            let (crs, _) = spatial_json_parts(&DataType::Geometry { srid: checked }, "geometry");
            vec![
                ("srid", srid.clone()),
                ("_crs", pyo3::types::PyString::new(py, &crs).into_any()),
            ]
        };
        assign_parameters(slf.as_any(), &entries)
    }
    // PySpark's `GeometryType` defines its own `needConversion` and `fromInternal`, of no docstring,
    // which convert as those of `DataType` (see `_from_internal`): the class has them as PySpark's.
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(&self) -> bool {
        true
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    // As PySpark's `GeometryType.toInternal(geometry)`: the dict of the SRID and the WKB of a
    // `Geometry` (see `_to_internal`).
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        geometry: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", geometry)
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
    // As PySpark's, `f"geometry({self._crs})"`, of the CRS.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        pyo3::types::PyString::new(py, "geometry({})").call_method1(
            pyo3::intern!(py, "format"),
            (parameter(
                slf.as_any(),
                pyo3::intern!(slf.as_any().py(), "_crs"),
            )?,),
        )
    }
    // As PySpark's, `"GeometryType(ANY)"` of the mixed SRID, else `f"GeometryType({self.srid})"`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let srid = parameter(slf.as_any(), pyo3::intern!(slf.as_any().py(), "srid"))?;
        if srid.eq(Self::MIXED_SRID)? {
            return Ok(pyo3::types::PyString::new(py, "GeometryType(ANY)").into_any());
        }
        pyo3::types::PyString::new(py, "GeometryType({})")
            .call_method1(pyo3::intern!(py, "format"), (srid,))
    }
    // As PySpark's, `"geometry(any)"` of the mixed SRID, else `f"geometry({self.srid})"`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let srid = parameter(slf.as_any(), pyo3::intern!(slf.as_any().py(), "srid"))?;
        if srid.eq(Self::MIXED_SRID)? {
            return Ok(pyo3::types::PyString::new(py, "geometry(any)").into_any());
        }
        pyo3::types::PyString::new(py, "geometry({})")
            .call_method1(pyo3::intern!(py, "format"), (srid,))
    }
}

#[doc = connect_doc!(GeographyType)]
#[pyclass(subclass, name = "GeographyType", module = "pyspark.sql.types", extends = PySpatialType)]
// The parameters `srid`, `_crs` and `_alg` are entries of the `__dict__` of the type, as PySpark's
// `__init__` sets them (see `parameter`).
pub struct PyGeographyType;

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
    /// a type of no SRID until `__init__` sets it, as `GeographyType.__new__(GeographyType)`, of which
    /// pickle's `copyreg.__newobj__` makes the type that PySpark pickled, whose `__dict__` pickle
    /// sets. The class shows the signature of `__init__`, as `inspect.signature` of PySpark's
    /// class does.
    #[new]
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(srid)")]
    fn new(
        cls: &Bound<'_, pyo3::types::PyType>,
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        check_one_branch(cls)?;
        Ok(init_chain!(
            DataType::Geography {
                srid: Self::DEFAULT_SRID
            },
            PyGeographyType,
            [PyAtomicType, PySpatialType],
            value
        ))
    }
    /// As PySpark's `__init__(self, srid)`: of `srid == "ANY"`, the mixed SRID and its CRS; else
    /// an int of a coordinate reference system that PySpark knows, as it is, and its CRS, or the
    /// `ST_INVALID_SRID_VALUE` of another value.
    fn __init__(slf: &Bound<'_, Self>, srid: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let mut entries = if srid.eq("ANY")? {
            vec![
                ("srid", Self::MIXED_SRID.into_pyobject(py)?.into_any()),
                (
                    "_crs",
                    pyo3::types::PyString::new(py, Self::MIXED_CRS).into_any(),
                ),
            ]
        } else {
            let checked = spatial_type_srid(srid, true)?;
            let (crs, _) = spatial_json_parts(&DataType::Geography { srid: checked }, "geography");
            vec![
                ("srid", srid.clone()),
                ("_crs", pyo3::types::PyString::new(py, &crs).into_any()),
            ]
        };
        // As PySpark's, `self._alg = GeographyType.DEFAULT_ALG`, after either.
        entries.push((
            "_alg",
            pyo3::types::PyString::new(py, Self::DEFAULT_ALG).into_any(),
        ));
        assign_parameters(slf.as_any(), &entries)
    }
    // PySpark's `GeographyType` defines its own `needConversion` and `fromInternal`, of no docstring,
    // which convert as those of `DataType` (see `_from_internal`): the class has them as PySpark's.
    #[pyo3(name = "needConversion")]
    fn __obj_need_conversion(&self) -> bool {
        true
    }
    #[pyo3(name = "fromInternal")]
    fn __obj_from_internal<'py>(
        slf: &Bound<'py, Self>,
        obj: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_from_internal", obj)
    }
    // As PySpark's `GeographyType.toInternal(geography)`: the dict of the SRID and the WKB of a
    // `Geography` (see `_to_internal`).
    #[pyo3(name = "toInternal")]
    fn __obj_to_internal<'py>(
        slf: &Bound<'py, Self>,
        geography: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", geography)
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
    // As PySpark's, `f"geography({self._crs}, {self._alg})"`, of the CRS and the algorithm.
    #[pyo3(name = "jsonValue")]
    fn __obj_json_value<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let (crs, alg) = (
            parameter(slf.as_any(), pyo3::intern!(slf.as_any().py(), "_crs"))?,
            parameter(slf.as_any(), pyo3::intern!(slf.as_any().py(), "_alg"))?,
        );
        pyo3::types::PyString::new(py, "geography({}, {})")
            .call_method1(pyo3::intern!(py, "format"), (crs, alg))
    }
    // As PySpark's, `"GeographyType(ANY)"` of the mixed SRID, else `f"GeographyType({self.srid})"`.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let srid = parameter(slf.as_any(), pyo3::intern!(slf.as_any().py(), "srid"))?;
        if srid.eq(Self::MIXED_SRID)? {
            return Ok(pyo3::types::PyString::new(py, "GeographyType(ANY)").into_any());
        }
        pyo3::types::PyString::new(py, "GeographyType({})")
            .call_method1(pyo3::intern!(py, "format"), (srid,))
    }
    // As PySpark's, `"geography(any)"` of the mixed SRID, else `f"geography({self.srid})"`.
    #[pyo3(name = "simpleString")]
    fn simple_string<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let srid = parameter(slf.as_any(), pyo3::intern!(slf.as_any().py(), "srid"))?;
        if srid.eq(Self::MIXED_SRID)? {
            return Ok(pyo3::types::PyString::new(py, "geography(any)").into_any());
        }
        pyo3::types::PyString::new(py, "geography({})")
            .call_method1(pyo3::intern!(py, "format"), (srid,))
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
    // A level of a type of many levels, such as the schema of a response, on a stack with room.
    crate::pbmsg::grow(|| has_udt_level(data_type))
}

/// [`has_udt`] of one level.
fn has_udt_level(data_type: &DataType) -> bool {
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
    // A level of a type of many levels, on a stack with room.
    crate::pbmsg::grow(|| holds_objects_level(data_type))
}

/// [`holds_objects`] of one level.
fn holds_objects_level(data_type: &DataType) -> bool {
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

/// Whether a type that the type object `obj` holds, at any depth, holds of `holds`: the field of a
/// struct, the type of a field, the element of an array, or the key or the value of a map, of the
/// entries of their `__dict__` (see `parameter`), which `holds` reads, as their own children.
fn keeps_object_of(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    holds: fn(Python<'_>, &Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<bool> {
    // A level of the type objects of a type of many levels, on a stack with room.
    type_level(py, || keeps_object_of_level(py, obj, holds))
}

/// [`keeps_object_of`] of one level of a type object.
fn keeps_object_of_level(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    holds: fn(Python<'_>, &Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<bool> {
    if obj.is_instance_of::<PyStructField>() {
        return holds(py, &parameter(obj, pyo3::intern!(py, "dataType"))?);
    }
    if obj.is_instance_of::<PyArrayType>() {
        return holds(py, &parameter(obj, pyo3::intern!(py, "elementType"))?);
    }
    if obj.is_instance_of::<PyMapType>() {
        return Ok(holds(py, &parameter(obj, pyo3::intern!(py, "keyType"))?)?
            || holds(py, &parameter(obj, pyo3::intern!(py, "valueType"))?)?);
    }
    if obj.is_instance_of::<PyStructType>() {
        for item in struct_fields(obj)?.try_iter()? {
            if holds(py, &item?)? {
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

/// Whether the type object `obj` is, or holds at any depth (see `keeps_object_of`), a field of a
/// name that is not a str, or of a str that UTF-8 cannot encode, such as one of a lone surrogate,
/// which the core's types do not hold.
fn holds_other_name(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<PyStructField>() {
        let name = parameter(obj, pyo3::intern!(py, "name"))?;
        match name.cast::<pyo3::types::PyString>() {
            Ok(text) if text.to_cow().is_ok() => {}
            _ => return Ok(true),
        }
    }
    keeps_object_of(py, obj, holds_other_name)
}

/// Whether the type object `obj` is, or keeps at any depth of the types that it keeps as their
/// objects (see `keeps_object_of`), a field of metadata. PySpark's field keeps its metadata dict
/// as it was given, which `==` compares as Python compares the dicts, and `json()` writes of
/// `json.dumps`: so does the drop-in's of the dict (see `keep_flags_in_json`), whose values the
/// core's JSON does not hold as they are (a tuple, a key that is not a str, an int of more than
/// 64 bits, the text of a float).
fn holds_metadata(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<PyStructField>() {
        // Of a dict that is not empty, or of metadata that is not a dict.
        let metadata = parameter(obj, pyo3::intern!(py, "metadata"))?;
        if !metadata
            .cast::<pyo3::types::PyDict>()
            .is_ok_and(|dict| dict.is_empty())
        {
            return Ok(true);
        }
    }
    keeps_object_of(py, obj, holds_metadata)
}

/// The `simpleString()` of the type object `child` of a field, an array or a map, as PySpark's
/// `self.<child>.simpleString()` reads it, as a str: of a struct, an array, a map or a field of the
/// native class, of its simple string, of no call of its method, so that the walk of a type of
/// more levels than the recursion limit raises its own `RecursionError`, of PySpark's text, rather
/// than the one of the calls of Python, of " while calling a Python object"; of another type, of
/// its own `simpleString()`, of the str of what it returns, as PySpark's `%s` formats it.
fn child_simple_string(py: Python<'_>, child: &Bound<'_, PyAny>) -> PyResult<String> {
    // A level of the objects of a type of many levels, such as the schema of a response, on a
    // stack with room.
    type_level(py, || child_simple_string_level(py, child))
}

/// [`child_simple_string`] of one level.
fn child_simple_string_level(py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<String> {
    let class = object.get_type();
    if class.is(py.get_type::<PyStructType>()) {
        return PyStructType::simple_string(object.cast::<PyStructType>()?);
    }
    if class.is(py.get_type::<PyArrayType>()) {
        return PyArrayType::simple_string(object.cast::<PyArrayType>()?);
    }
    if class.is(py.get_type::<PyMapType>()) {
        return PyMapType::simple_string(object.cast::<PyMapType>()?);
    }
    if class.is(py.get_type::<PyStructField>()) {
        return PyStructField::__sf_simple_string(object.cast::<PyStructField>()?);
    }
    Ok(object
        .call_method0(pyo3::intern!(py, "simpleString"))?
        .str()?
        .to_cow()?
        .into_owned())
}

/// As PySpark's constructors of an array, a map and a field, `assert isinstance(obj, DataType)`,
/// of its message, of the parameter `name`.
fn assert_data_type(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<()> {
    let data_type_class = obj.py().import("pyspark.sql.types")?.getattr("DataType")?;
    if !obj.is_instance(&data_type_class)? {
        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
            "{name} {} should be an instance of {}",
            obj.str()?,
            data_type_class.str()?
        )));
    }
    Ok(())
}

/// The core type of the child type object `child` of a field, an array or a map, as
/// [`lower_type`] lowers it, of the metadata as `metadata` takes it: of a value that is not a data
/// type, such as a DDL string that a user set, which PySpark's attribute keeps, the
/// `UNSUPPORTED_OPERATION` of PySpark's conversion of the plan of it.
fn lower_child(child: &Bound<'_, PyAny>, metadata: MetadataOf) -> PyResult<DataType> {
    // An atomic type, of the kind of its class, of one read of it, as the lowering of a level of
    // it takes it (see `lower_type_level`): of no parameters, its type, and of parameters, of the
    // parameters of its `__dict__`.
    match atomic_kind(child) {
        Some(AtomicKind::Fixed(data_type)) => return Ok(data_type),
        Some(AtomicKind::Parameters(kind)) => return parameter_type_of(child, kind),
        None => {}
    }
    if !child.is_instance_of::<PyDataType>() {
        return Err(not_a_type_error(child));
    }
    lower_type(child, metadata)
}

/// The truth of the flag `name` of the type object `obj` (`containsNull`, `valueContainsNull`,
/// `nullable`), of its `__bool__`, as a core type that stands for the type takes it.
fn flag_parameter(
    obj: &Bound<'_, PyAny>,
    name: &Bound<'_, pyo3::types::PyString>,
) -> PyResult<bool> {
    parameter(obj, name)?.is_truthy()
}

/// The flag `name` of the type object `obj` as its lowering takes it: of a plan, as protobuf's
/// bool field takes it, which PySpark's conversion sets it in (`ret.array.contains_null =
/// data_type.containsNull`), of the `TypeError` of a value that is not an int (see
/// [`crate::streaming::proto_bool`]), and else its truth (see [`flag_parameter`]).
fn lowered_flag(
    obj: &Bound<'_, PyAny>,
    name: &Bound<'_, pyo3::types::PyString>,
    metadata: MetadataOf,
) -> PyResult<bool> {
    let flag = parameter(obj, name)?;
    // A bool, of no look at its class, as protobuf takes it.
    if let Ok(flag) = flag.cast_exact::<pyo3::types::PyBool>() {
        return Ok(flag.is_true());
    }
    if metadata == MetadataOf::Plan {
        return crate::streaming::proto_bool(&flag);
    }
    flag.is_truthy()
}

/// As PySpark's `_as_nullable` of the type object `obj`: of an array, a map or a struct, one of
/// the same types made nullable in turn and of its nullable flags set true, as `toNullable`
/// returns it, whose fields hold the metadata dicts of the fields as they are; any other type is
/// the object itself, as PySpark's base `_as_nullable` returns `self`, so that a
/// `UserDefinedType`, a `DecimalType` or a type of its class only stays the object it was.
fn as_nullable<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // A level of a type of many levels, on a stack with room.
    type_level(obj.py(), || as_nullable_level(obj))
}

/// [`as_nullable`] of one level.
fn as_nullable_level<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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
            // As PySpark's `StructField(field.name, field.dataType._as_nullable(), nullable=True,
            // metadata=field.metadata)`, in its order, of the `_as_nullable` of a type that is
            // not a data type of its own.
            let name = field.getattr(pyo3::intern!(py, "name"))?;
            let data_type = field.getattr(pyo3::intern!(py, "dataType"))?;
            let data_type = if data_type.is_instance_of::<PyDataType>() {
                as_nullable(&data_type)?
            } else {
                data_type.call_method0(pyo3::intern!(py, "_as_nullable"))?
            };
            fields.append(py.get_type::<PyStructField>().call1((
                name,
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
            if !(a.is_instance_of::<PyStructType>() && b.is_instance_of::<PyStructType>()) {
                return Ok(true);
            }
            let (x, y) = (struct_fields(a)?, struct_fields(b)?);
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
            if !(a.is_instance_of::<PyArrayType>() && b.is_instance_of::<PyArrayType>()) {
                return Ok(true);
            }
            let name = pyo3::intern!(py, "elementType");
            let (x, y) = (parameter(a, name)?, parameter(b, name)?);
            if holds_objects(element_type)
                || holds_subclass_object(py, &x)?
                || holds_subclass_object(py, &y)?
            {
                pairs.push((x, y));
            }
        }
        DataType::Map {
            key_type,
            value_type,
            ..
        } => {
            if !(a.is_instance_of::<PyMapType>() && b.is_instance_of::<PyMapType>()) {
                return Ok(true);
            }
            for (name, child_type) in [
                (pyo3::intern!(py, "keyType"), key_type),
                (pyo3::intern!(py, "valueType"), value_type),
            ] {
                let (x, y) = (parameter(a, name)?, parameter(b, name)?);
                if holds_objects(child_type)
                    || holds_subclass_object(py, &x)?
                    || holds_subclass_object(py, &y)?
                {
                    pairs.push((x, y));
                }
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
    // A level of a type of many levels, on a stack with room.
    crate::pbmsg::grow(|| has_converted_values_level(data_type))
}

/// [`has_converted_values`] of one level.
fn has_converted_values_level(data_type: &DataType) -> bool {
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

/// The stack that the decoder and the encoder of Python's `json` take of the JSON of a type of
/// many levels, which they recurse over in C, once per level, up to Python's recursion limit
/// (about 1,000 levels, of a few hundred KiB), and the segment that a call with less room left
/// takes.
const JSON_STACK_RED_ZONE: usize = if cfg!(debug_assertions) {
    4 << 20
} else {
    512 << 10
};
const JSON_STACK_SEGMENT: usize = if cfg!(debug_assertions) {
    16 << 20
} else {
    4 << 20
};

/// Run `f`, a call of Python's `json` of the JSON of a type of many levels, on a stack with room
/// for its recursion, of a thread of a small stack too.
fn grow_for_json<T>(f: impl FnOnce() -> T) -> T {
    stacker::maybe_grow(JSON_STACK_RED_ZONE, JSON_STACK_SEGMENT, f)
}

pub(crate) fn data_type_to_py(py: Python<'_>, dt: &DataType) -> PyResult<Py<PyAny>> {
    // A level of a type of many levels, such as the schema of a response, on a stack with room (its
    // copies of the types below it grow on their own).
    crate::pbmsg::grow(|| data_type_to_py_level(py, dt))
}

/// [`data_type_to_py`] of one level of a type.
fn data_type_to_py_level(py: Python<'_>, dt: &DataType) -> PyResult<Py<PyAny>> {
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
        // A type of parameters, of the `__dict__` of PySpark's `__init__` of them.
        DataType::Decimal { .. } => with_parameters(
            obj!(init_chain!(
                dt.clone(),
                PyDecimalType,
                [PyAtomicType, PyNumericType, PyFractionalType],
                value
            ))
            .into_bound(py),
            dt,
        )?
        .unbind(),
        DataType::String { .. } => with_parameters(
            obj!(init_chain!(dt.clone(), PyStringType, [PyAtomicType], value)).into_bound(py),
            dt,
        )?
        .unbind(),
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
        // An array and a map, of the objects of their types, as PySpark's `__init__` sets them.
        DataType::Array {
            element_type,
            contains_null,
        } => {
            let array = obj!(init_chain!(array_kind(), PyArrayType, [], value)).into_bound(py);
            assign_parameters(
                &array,
                &[
                    (
                        "elementType",
                        data_type_to_py(py, element_type)?.into_bound(py),
                    ),
                    (
                        "containsNull",
                        flag_of_bool(py, *contains_null).into_bound(py),
                    ),
                ],
            )?;
            array.unbind()
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let map = obj!(init_chain!(map_kind(), PyMapType, [], value)).into_bound(py);
            assign_parameters(
                &map,
                &[
                    ("keyType", data_type_to_py(py, key_type)?.into_bound(py)),
                    ("valueType", data_type_to_py(py, value_type)?.into_bound(py)),
                    (
                        "valueContainsNull",
                        flag_of_bool(py, *value_contains_null).into_bound(py),
                    ),
                ],
            )?;
            map.unbind()
        }
        DataType::Struct { fields } => py_new_struct(py, fields)?.into_any(),
        DataType::Char { .. } => with_parameters(
            obj!(init_chain!(dt.clone(), PyCharType, [PyAtomicType], value)).into_bound(py),
            dt,
        )?
        .unbind(),
        DataType::Varchar { .. } => with_parameters(
            obj!(init_chain!(
                dt.clone(),
                PyVarcharType,
                [PyAtomicType],
                value
            ))
            .into_bound(py),
            dt,
        )?
        .unbind(),
        DataType::Time { .. } => with_parameters(
            obj!(init_chain!(
                dt.clone(),
                PyTimeType,
                [PyAtomicType, PyDatetimeType, PyAnyTimeType],
                value
            ))
            .into_bound(py),
            dt,
        )?
        .unbind(),
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
        DataType::YearMonthInterval { .. } => with_parameters(
            obj!(init_chain!(
                dt.clone(),
                PyYearMonthIntervalType,
                [PyAtomicType, PyAnsiIntervalType],
                value
            ))
            .into_bound(py),
            dt,
        )?
        .unbind(),
        DataType::DayTimeInterval { .. } => with_parameters(
            obj!(init_chain!(
                dt.clone(),
                PyDayTimeIntervalType,
                [PyAtomicType, PyAnsiIntervalType],
                value
            ))
            .into_bound(py),
            dt,
        )?
        .unbind(),
        DataType::Variant => obj!(init_chain!(
            DataType::Variant,
            PyVariantType,
            [PyAtomicType]
        )),
        DataType::Geometry { srid } => {
            let data_type = DataType::Geometry {
                srid: schema_spatial_srid(py, *srid, false)?,
            };
            with_parameters(
                obj!(init_chain!(
                    data_type.clone(),
                    PyGeometryType,
                    [PyAtomicType, PySpatialType],
                    value
                ))
                .into_bound(py),
                &data_type,
            )?
            .unbind()
        }
        DataType::Geography { srid } => {
            let data_type = DataType::Geography {
                srid: schema_spatial_srid(py, *srid, true)?,
            };
            with_parameters(
                obj!(init_chain!(
                    data_type.clone(),
                    PyGeographyType,
                    [PyAtomicType, PySpatialType],
                    value
                ))
                .into_bound(py),
                &data_type,
            )?
            .unbind()
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
                None => udt_of_json(py, &json)?,
            }
        }
    };
    Ok(o)
}

/// A new user-defined type object of the JSON `json` of a user-defined type, as PySpark's
/// `UserDefinedType.fromJson` makes it of a type of the server: `UDT()` of the class that its
/// `pyClass` names, which raises for a class that it does not import, or whose `__init__` takes
/// arguments.
fn udt_of_json(py: Python<'_>, json: &str) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("pyspark.sql.types")?
        .getattr("_parse_datatype_json_string")?
        .call1((json,))?
        .unbind())
}

/// The type object of `dt`, a type that the server gave, such as the schema of a DataFrame or of
/// its result, as PySpark's conversion of the types of the server makes it: each user-defined
/// type a new object of its class (see [`udt_of_json`]), which the array, the map, the field or
/// the struct that holds it keeps. [`data_type_to_py`] gives instead the object of its JSON that
/// the client lowered last, for the types that the client makes of core types: one of a user
/// would be shared by every schema of its JSON, of its attributes and identity, and a class whose
/// `__init__` takes arguments would not raise. A type of no user-defined type is made as
/// `data_type_to_py` makes it.
pub(crate) fn schema_to_py(py: Python<'_>, dt: &DataType) -> PyResult<Py<PyAny>> {
    // A level of a type of many levels, such as the schema of a response, on a stack with room (its
    // copies of the types below it grow on their own).
    crate::pbmsg::grow(|| schema_to_py_level(py, dt))
}

/// [`schema_to_py`] of one level of a type.
fn schema_to_py_level(py: Python<'_>, dt: &DataType) -> PyResult<Py<PyAny>> {
    if !has_udt(dt) {
        return data_type_to_py(py, dt);
    }
    // As `data_type_to_py`, with the SQL types of the JVM types of Python classes.
    let filled = with_udt_sql_types(py, dt);
    let dt = filled.as_ref().unwrap_or(dt);
    Ok(match dt {
        DataType::Udt { .. } => udt_of_json(py, &dt.json())?,
        DataType::Array {
            element_type,
            contains_null,
        } => {
            let element = schema_to_py(py, element_type)?;
            let array = Py::new(py, init_chain!(array_kind(), PyArrayType, [], value))?.into_any();
            assign_parameters(
                array.bind(py),
                &[
                    ("elementType", element.into_bound(py)),
                    (
                        "containsNull",
                        flag_of_bool(py, *contains_null).into_bound(py),
                    ),
                ],
            )?;
            array
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let key = schema_to_py(py, key_type)?;
            let value = schema_to_py(py, value_type)?;
            let map = Py::new(py, init_chain!(map_kind(), PyMapType, [], value))?.into_any();
            assign_parameters(
                map.bind(py),
                &[
                    ("keyType", key.into_bound(py)),
                    ("valueType", value.into_bound(py)),
                    (
                        "valueContainsNull",
                        flag_of_bool(py, *value_contains_null).into_bound(py),
                    ),
                ],
            )?;
            map
        }
        DataType::Struct { fields } => {
            let objects = PyList::empty(py);
            for field in fields {
                let data_type = schema_to_py(py, &field.data_type)?;
                objects.append(field_object(py, field, data_type)?)?;
            }
            struct_of_list(py, objects)?.into_any()
        }
        _ => data_type_to_py(py, dt)?,
    })
}

/// `dt` with the SQL type of each user-defined type of a Python class that lacks one, from the
/// class's `sqlType()`, or `None` if it has none to fill. The server sends no SQL type of a JVM type
/// with a Python class, such as `VectorUDT`, and PySpark's conversion of a schema of the server
/// makes the type of its class (`UserDefinedType.fromJson`), whose JSON has its `sqlType`: so the
/// JSON of the schema, and a schema made of it, have the `sqlType` as PySpark's have. A class that
/// does not import keeps its type as it is, as before.
fn with_udt_sql_types(py: Python<'_>, dt: &DataType) -> Option<DataType> {
    // A level of a type of many levels, such as the schema of a response, on a stack with room (its
    // copies of the types beside a filled one grow on their own).
    crate::pbmsg::grow(|| with_udt_sql_types_level(py, dt))
}

/// [`with_udt_sql_types`] of one level of a type.
fn with_udt_sql_types_level(py: Python<'_>, dt: &DataType) -> Option<DataType> {
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
            // Of no copy of the fields before one of them is filled, nor of a type that a filled
            // one replaces.
            let (first, filled) = fields.iter().enumerate().find_map(|(index, field)| {
                with_udt_sql_types(py, &field.data_type).map(|filled| (index, filled))
            })?;
            let field_of = |field: &StructField, data_type| StructField {
                name: field.name.clone(),
                data_type,
                nullable: field.nullable,
                metadata: field.metadata.clone(),
            };
            let mut filled_fields = Vec::with_capacity(fields.len());
            filled_fields.extend(
                fields[..first]
                    .iter()
                    .map(|field| field_of(field, field.data_type.clone())),
            );
            filled_fields.push(field_of(&fields[first], filled));
            filled_fields.extend(fields[first + 1..].iter().map(|field| {
                let data_type = with_udt_sql_types(py, &field.data_type)
                    .unwrap_or_else(|| field.data_type.clone());
                field_of(field, data_type)
            }));
            Some(DataType::Struct {
                fields: filled_fields,
            })
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
        // Of the text of the type, as `_parse_ddl`.
        let parsed =
            crate::interrupt::interruptible_without_parts(py, session, || session.parse_ddl(&ddl))?;
        // As PySpark converts the parsed type from its proto, which fails for a nanosecond timestamp
        // and for a spatial type of an SRID that PySpark does not take.
        crate::dataframe::check_schema_conversion(py, Some(&parsed))?;
        return Ok(parsed);
    }
    py_to_data_type(schema)
}

std::thread_local! {
    /// The levels of the walks of type objects that the thread is in (see [`type_level`]).
    static TYPE_LEVELS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `f`, a level of a walk of type objects, of their fields, elements, keys and values, on a stack
/// with room for it (see `crate::pbmsg::grow`), of no more levels of the walks of the thread than
/// Python's recursion limit (`sys.getrecursionlimit()`), past which it raises PySpark's
/// `RecursionError`: PySpark's methods walk the levels of a type in Python, of a frame of Python
/// for each level at least, so they raise it of a type of more levels, and of a type that holds
/// itself, as a struct that a field of it holds, whose levels no walk ends.
#[inline]
fn type_level<T>(_py: Python<'_>, f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    /// A level of [`TYPE_LEVELS`], of the cell of the thread, until it is dropped.
    struct Level<'a>(&'a std::cell::Cell<usize>);
    impl Drop for Level<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get().saturating_sub(1));
        }
    }
    // Of one access of the thread-local, which the walk of a wide type makes at each level.
    TYPE_LEVELS.with(|levels| {
        let depth = levels.get() + 1;
        levels.set(depth);
        let _level = Level(levels);
        // The limit is read only past the levels of the walks of ordinary types, which a wide
        // type makes at each of its levels: a limit below them bounds the walks a little later.
        if depth > TYPE_LEVELS_UNCHECKED {
            // SAFETY: reads the recursion limit of the interpreter, of the thread that holds the
            // GIL (`_py`), as `sys.getrecursionlimit()` does.
            let limit = unsafe { pyo3::ffi::Py_GetRecursionLimit() };
            if depth > usize::try_from(limit).unwrap_or(0) {
                return Err(pyo3::exceptions::PyRecursionError::new_err(
                    "maximum recursion depth exceeded",
                ));
            }
        }
        crate::pbmsg::grow(f)
    })
}

/// The levels of the walks of type objects that a thread goes in without reading Python's
/// recursion limit (see [`type_level`]).
const TYPE_LEVELS_UNCHECKED: usize = 32;

/// The core type of the `UserDefinedType` object `obj`, of its `jsonValue()` and of the core type
/// that `lower` makes of its `sqlType()`, in that order, as PySpark's conversion reads them; the
/// object is kept by the JSON of the type (see `lowered_udts`).
fn udt_of(
    obj: &Bound<'_, PyAny>,
    lower: impl FnOnce(&Bound<'_, PyAny>) -> PyResult<DataType>,
) -> PyResult<DataType> {
    let jv = obj.call_method0("jsonValue")?;
    let get = |k: &str| -> PyResult<Option<String>> {
        let v = jv.call_method1("get", (k,))?;
        if v.is_none() {
            Ok(None)
        } else {
            Ok(Some(v.extract()?))
        }
    };
    let sql_type = lower(&obj.call_method0("sqlType")?)?;
    let udt = DataType::Udt {
        type_str: "udt".to_string(),
        jvm_class: get("class")?,
        python_class: get("pyClass")?,
        serialized_python_class: get("serializedClass")?,
        sql_type: Some(Box::new(sql_type)),
    };
    lowered_udts(obj.py()).set_item(udt.json(), obj)?;
    Ok(udt)
}

pub(crate) fn py_to_data_type(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    lower_type(obj, MetadataOf::Strict)
}

std::thread_local! {
    /// Whether a lowering of the thread read a struct, a field, an array or a map of a Python
    /// subclass, whose own reads may give elements that the lowering does not read (see
    /// [`lower_noting_subclasses`]).
    static LOWERED_SUBCLASS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Mark that the lowering reads an object of a class of `obj` other than `native`, a Python
/// subclass (see [`lower_noting_subclasses`]).
fn note_subclass(obj: &Bound<'_, PyAny>, native: &Bound<'_, pyo3::types::PyType>) {
    if !obj.get_type().is(native) {
        LOWERED_SUBCLASS.with(|lowered| lowered.set(true));
    }
}

/// The lowering of `obj`, of the metadata as `metadata` takes it, and whether the type may hold a
/// struct whose fields hold an element that is not a `StructField` (see `ForeignFields`): of its
/// `TypeError`, such as that of such an element of a struct, or of a struct, a field, an array or
/// a map of a Python subclass in it, whose own reads may give such elements. Of an ordinary type,
/// the walk of the foreign fields is not made, and its lowering is the one that the use makes.
pub(crate) fn lower_noting_subclasses(
    obj: &Bound<'_, PyAny>,
    metadata: MetadataOf,
) -> (PyResult<DataType>, bool) {
    // A struct whose own fields hold such an element is not lowered, whose uses read its elements
    // (see `ForeignFields`): the lowering would read the types of the fields before the element,
    // whose reads may call the code of a user, such as the `sqlType()` of a user-defined type,
    // which the uses then read once more.
    match holds_foreign_element(obj) {
        Ok(true) => {
            return (
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "StructType fields must be StructField objects",
                )),
                true,
            )
        }
        Ok(false) => {}
        Err(error) => return (Err(error), false),
    }
    let before = LOWERED_SUBCLASS.with(|lowered| lowered.replace(false));
    let lowered = lower_type(obj, metadata);
    let subclass = LOWERED_SUBCLASS.with(|lowered| lowered.replace(before || lowered.get()));
    let maybe_foreign = subclass
        || matches!(&lowered, Err(error) if error.is_instance_of::<pyo3::exceptions::PyTypeError>(obj.py()));
    (lowered, maybe_foreign)
}

/// Whether the type object `obj`, of the lowering `lowered` of [`lower_noting_subclasses`], holds a
/// struct whose fields hold an element that is not a `StructField`, as
/// [`holds_foreign_field`] tells, of no walk of an ordinary type.
pub(crate) fn foreign_of_lowering(obj: &Bound<'_, PyAny>, maybe_foreign: bool) -> PyResult<bool> {
    Ok(maybe_foreign && holds_foreign_field(obj)?)
}

/// [`py_to_data_type`] of the metadata of the fields as they were given (see
/// [`MetadataOf::AsGiven`]).
pub(crate) fn py_to_data_type_as_given(obj: &Bound<'_, PyAny>) -> PyResult<DataType> {
    lower_type(obj, MetadataOf::AsGiven)
}

/// The lowering of [`py_to_data_type`], of the metadata as `metadata` takes it.
fn lower_type(obj: &Bound<'_, PyAny>, metadata: MetadataOf) -> PyResult<DataType> {
    // A level of a type object of many levels, of a response or that the user makes, whose
    // lowering recurses through the objects of its fields, elements and values: on a stack with
    // room, of a thread of a small stack too (its copies of the core types that they keep grow on
    // their own).
    type_level(obj.py(), || lower_type_level(obj, metadata))
}

/// [`lower_type`] of one level of a type object.
fn lower_type_level(obj: &Bound<'_, PyAny>, metadata: MetadataOf) -> PyResult<DataType> {
    // Of the kind of the class of the type (`PyDataType::inner`): a type of no parameters, of its
    // class, also of a Python subclass, which is of the core type of its class; and a type of
    // parameters, of the parameters of its `__dict__`.
    if let Some(kind) = atomic_kind(obj) {
        return match kind {
            AtomicKind::Fixed(data_type) => Ok(data_type),
            AtomicKind::Parameters(kind) => parameter_type_of(obj, kind),
        };
    }
    // A container, before the checks of the classes of no parameters, which it is of none of: an
    // array and a map, of the parameters of their `__dict__`, as PySpark's conversion of a plan
    // reads them, the types in turn, then the flags, and a struct, of its fields.
    if obj.is_instance_of::<PyArrayType>() {
        let py = obj.py();
        note_subclass(obj, &py.get_type::<PyArrayType>());
        let element = parameter(obj, pyo3::intern!(py, "elementType"))?;
        let element_type = lower_child(&element, metadata)?;
        return Ok(DataType::Array {
            element_type: Box::new(element_type),
            contains_null: lowered_flag(obj, pyo3::intern!(py, "containsNull"), metadata)?,
        });
    }
    if obj.is_instance_of::<PyMapType>() {
        let py = obj.py();
        note_subclass(obj, &py.get_type::<PyMapType>());
        let key_type = lower_child(&parameter(obj, pyo3::intern!(py, "keyType"))?, metadata)?;
        let value = parameter(obj, pyo3::intern!(py, "valueType"))?;
        let value_type = lower_child(&value, metadata)?;
        let value_contains_null = pyo3::intern!(py, "valueContainsNull");
        return Ok(DataType::Map {
            key_type: Box::new(key_type),
            value_type: Box::new(value_type),
            value_contains_null: lowered_flag(obj, value_contains_null, metadata)?,
        });
    }
    if obj.is_instance_of::<PyStructType>() {
        note_subclass(obj, &obj.py().get_type::<PyStructType>());
        return Ok(DataType::Struct {
            fields: struct_core_fields(obj, metadata)?,
        });
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
    if obj.extract::<PyRef<PyCalendarIntervalType>>().is_ok() {
        return Ok(DataType::CalendarInterval);
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
            return udt_of(obj, |sql_type| lower_type(sql_type, metadata));
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

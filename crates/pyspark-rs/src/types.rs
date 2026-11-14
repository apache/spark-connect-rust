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

/// The `__reduce__` of a type of `data_type`, whose class is `cls`: by its JSON, as
/// [`type_reduce`], unless it has a user-defined type, whose JSON need not import its class, or
/// a nullable flag that is not a bool, which the JSON does not keep (`flagged`). Such a type
/// pickles of the type objects it is made of, `args()`, as PySpark's types do; the class is that
/// of `pyspark.sql.types`, also on the Python workers of the server.
fn reduce_with_udts<'py>(
    cls: Bound<'py, pyo3::types::PyType>,
    data_type: &DataType,
    flagged: bool,
    args: impl FnOnce() -> PyResult<Bound<'py, pyo3::types::PyTuple>>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let py = cls.py();
    if flagged || has_udt(data_type) {
        return Ok((cls.into_any().unbind(), args()?.into_any().unbind()));
    }
    let (f, args) = type_reduce(py, &data_type.json())?;
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
    pub(crate) inner: DataType,
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
    let field = |value: Option<&Bound<'_, PyAny>>| -> Option<i32> {
        let value = value?;
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
    };
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
            Err(crate::column::raise_pyspark(
                py,
                "PySparkRuntimeError",
                "INVALID_INTERVAL_CASTING",
                &[
                    ("start_field", text(start)?.as_str()),
                    ("end_field", text(end)?.as_str()),
                ],
            ))
        }
    }
}

/// The abstract intermediate base classes of the type hierarchy (mirror pyspark's
/// AtomicType/NumericType/IntegralType/... ). They carry no data — a concrete type's
/// `__new__` builds the initializer chain `PyDataType{inner} -> ... -> Concrete` — but
/// existing so `isinstance(dt, NumericType)` etc. work with the reference MRO.
macro_rules! abstract_type {
    ($ty:ident, $name:literal, $parent:ty) => {
        // `module` matches where the drop-in re-exports these (pyspark.sql.types), so
        // `__module__` reports the reference path instead of PyO3's default `builtins`.
        #[pyclass(subclass, name = $name, module = "pyspark.sql.types", extends = $parent)]
        pub struct $ty;
    };
}
abstract_type!(PyAtomicType, "AtomicType", PyDataType);
abstract_type!(PyNumericType, "NumericType", PyAtomicType);
abstract_type!(PyIntegralType, "IntegralType", PyNumericType);
abstract_type!(PyFractionalType, "FractionalType", PyNumericType);
abstract_type!(PyDatetimeType, "DatetimeType", PyAtomicType);
abstract_type!(PyAnyTimeType, "AnyTimeType", PyDatetimeType);
abstract_type!(PyAnsiIntervalType, "AnsiIntervalType", PyAtomicType);
abstract_type!(PySpatialType, "SpatialType", PyAtomicType);

/// Build the `PyClassInitializer` chain for a concrete type: the base `PyDataType`
/// carries `inner`; each intermediate abstract base and finally the concrete unit
/// value are stacked on top. Usage: `init_chain!(inner_expr, Concrete, [Mid1, Mid2, ...])`.
macro_rules! init_chain {
    ($inner:expr, $concrete:ident, [$($mid:ident),*]) => {
        pyo3::PyClassInitializer::from(PyDataType { inner: $inner })
            $(.add_subclass($mid))*
            .add_subclass($concrete)
    };
    // Variant carrying a concrete value (parameterized types).
    ($inner:expr, $concrete_val:expr, [$($mid:ident),*], value) => {
        pyo3::PyClassInitializer::from(PyDataType { inner: $inner })
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
    let need_conversion = need_conversion_of(py, &base_fields)?;
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

/// The `needConversion` of each of `fields`, as PySpark's `_needConversion`.
fn need_conversion_of(py: Python<'_>, fields: &[StructField]) -> PyResult<Vec<bool>> {
    fields
        .iter()
        .map(|field| need_conversion(py, &field.data_type))
        .collect()
}

/// The `needConversion` of each of the `StructField` objects of `fields`, as they are now.
fn need_conversion_of_fields(py: Python<'_>, fields: &Bound<'_, PyList>) -> PyResult<Vec<bool>> {
    fields
        .iter()
        .map(|item| as_struct_field(&item)?.borrow().need_conversion_now(py))
        .collect()
}

/// The `needConversion` of the data type object `obj`, as PySpark's: a struct and a user-defined
/// type always need a conversion, and an array or a map when one of its element types does. The
/// fields of a struct are not read, nor their metadata, so this does not grow with the struct.
fn need_conversion_of_object(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    fn of_child(py: Python<'_>, kept: &KeptType, snapshot: &DataType) -> PyResult<bool> {
        match kept.kept(py) {
            Some(object) => need_conversion_of_object(py, object.bind(py)),
            None => need_conversion(py, snapshot),
        }
    }
    if obj.is_instance_of::<PyStructType>() {
        return Ok(true);
    }
    // A StructField as a type needs a conversion as its dataType does, as PySpark's does.
    if let Ok(field) = obj.cast::<PyStructField>() {
        let kept = field.borrow().data_type_object.kept(py);
        return match kept {
            Some(object) => need_conversion_of_object(py, object.bind(py)),
            None => need_conversion(py, &py_to_data_type(obj)?),
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

/// The text of the nullable flag `flag` of the type at `path` below the type `root`, as PySpark's
/// `treeString` writes it: `str(flag).lower()` of the flag as it was given. `None` when the type
/// is not kept as an object, whose flags are the core's bools.
fn tree_flag_text(
    root: &Bound<'_, PyAny>,
    path: &[TreeStep],
    flag: TreeFlag,
) -> PyResult<Option<String>> {
    let py = root.py();
    let mut node = Some(root.clone());
    for step in path {
        node = match node {
            Some(current) => kept_child(&current, *step)?,
            None => return Ok(None),
        };
    }
    let Some(node) = node else {
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
/// `pyspark_types_to_proto_types`, which has no proto of a `StructField` as a type: it is the
/// `UNSUPPORTED_OPERATION` of the first one, `obj` itself or else the first in its fields.
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

/// The repr of the first `StructField` that is a type of the type object `obj` of `data_type`,
/// `obj` itself or one nested in it, which PySpark's conversion of a plan's type
/// (`pyspark_types_to_proto_types`) does not take; None if there is none.
pub(crate) fn unsendable_field_repr(
    obj: &Bound<'_, PyAny>,
    data_type: &DataType,
) -> PyResult<Option<String>> {
    let nested = if obj.is_instance_of::<PyStructField>() {
        Some(obj.clone())
    } else {
        nested_field_in(obj.py(), data_type, i32::MAX)?
    };
    nested
        .map(|field| field.repr().and_then(|text| text.extract::<String>()))
        .transpose()
}

/// The first `StructField` that is a type of `data_type`, in the depth-first order of its fields,
/// down to `depth` levels (as `treeString` prints them), as its Python object.
pub(crate) fn nested_field_in<'py>(
    py: Python<'py>,
    data_type: &DataType,
    depth: i32,
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
            if let Some(field) = lowered_field(py, data_type_string)? {
                return Ok(Some(field));
            }
        }
        if let Some(field) = nested_field_in(py, child, depth - 1)? {
            return Ok(Some(field));
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
/// str, or a type is a `StructField`.
pub(crate) fn has_other_flag(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
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

/// `has_other_flag` of a type in a type position: a `StructField` there is written from its own
/// object, as its JSON is not the core's.
fn child_has_other_flag(py: Python<'_>, child: &Bound<'_, PyAny>) -> PyResult<bool> {
    if child.is_instance_of::<PyStructField>() {
        return Ok(true);
    }
    has_other_flag(py, child)
}

/// Set the nullable flags in the JSON value `json` of `obj` to the flags as they were given, as
/// PySpark's `jsonValue` has them. Returns whether one of them is other than a bool.
fn keep_flags_in_json(
    py: Python<'_>,
    json: &Bound<'_, PyAny>,
    obj: &Bound<'_, PyAny>,
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
        if let Some(data_type) = field.data_type_object.kept(py) {
            other |= keep_child_json(py, json, "type", data_type.bind(py))?;
        }
        return Ok(other);
    }
    if let Ok(array) = obj.cast::<PyArrayType>() {
        let array = array.borrow();
        json.set_item("containsNull", array.contains_null_object.bind(py))?;
        let mut other = !is_bool_flag(array.contains_null_object.bind(py));
        if let Some(element) = array.element_object.kept(py) {
            other |= keep_child_json(py, json, "elementType", element.bind(py))?;
        }
        return Ok(other);
    }
    if let Ok(map) = obj.cast::<PyMapType>() {
        let map = map.borrow();
        json.set_item("valueContainsNull", map.value_contains_null_object.bind(py))?;
        let mut other = !is_bool_flag(map.value_contains_null_object.bind(py));
        if let Some(key) = map.key_object.kept(py) {
            other |= keep_child_json(py, json, "keyType", key.bind(py))?;
        }
        if let Some(value) = map.value_object.kept(py) {
            other |= keep_child_json(py, json, "valueType", value.bind(py))?;
        }
        return Ok(other);
    }
    if let Ok(st) = obj.cast::<PyStructType>() {
        let st = st.borrow();
        let fields = json.get_item("fields")?;
        let mut other = false;
        for (i, item) in st.fields.bind(py).iter().enumerate() {
            other |= keep_flags_in_json(py, &fields.get_item(i)?, &item)?;
        }
        return Ok(other);
    }
    Ok(false)
}

/// Set the type at `key` of the JSON object `json` to the JSON of `child`, with its flags as given.
/// A `StructField` as a type is its own JSON object, as PySpark's `jsonValue` has it.
fn keep_child_json(
    py: Python<'_>,
    json: &Bound<'_, PyAny>,
    key: &str,
    child: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if let Ok(field) = child.cast::<PyStructField>() {
        let (nested, _) = field_json_value(py, field)?;
        json.set_item(key, nested)?;
        return Ok(true);
    }
    keep_flags_in_json(py, &json.get_item(key)?, child)
}

/// The `jsonValue` of the type object `obj`, with its nullable flags as given, and whether one
/// of them is other than a bool.
fn type_json_value<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let json = py_json_value(py, &py_to_data_type(obj)?.json_value(), false)?;
    let other = keep_flags_in_json(py, &json, obj)?;
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

/// The `jsonValue` of the field `field`, with its nullable flags as given, and whether one of
/// them is other than a bool.
fn field_json_value<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyStructField>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let v = field.borrow().current_field(py)?.json_value();
    let json = py_json_value(py, &v, true)?;
    let other = keep_flags_in_json(py, &json, field.as_any())?;
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
    serde_json::from_str(&s)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
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
    /// Allow instantiating the base and, crucially, pure-Python subclasses of it
    /// (e.g. a user's `UserDefinedType`). The sentinel `inner` is never consulted for
    /// such subclasses — they override the object-model methods in Python, and
    /// `py_to_data_type` special-cases `UserDefinedType` before reading `inner`.
    /// As `object.__new__` of PySpark's pure-Python `DataType`: the arguments are those of the
    /// `__init__` of a subclass, such as a `UserDefinedType` of parameters.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn __new__(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        PyDataType {
            inner: DataType::Null,
        }
    }

    /// The fallback for the classes without their own `__reduce__`: a bare `DataType`
    /// (an unparsed DDL type) and a pure-Python `UserDefinedType`, whose `json()` names
    /// its Python class.
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (String,))> {
        let json: String = slf.call_method0("json")?.extract()?;
        type_reduce(slf.py(), &json)
    }

    // --- DataType object-model methods (v4.2.0 parity) ---
    /// As PySpark's `DataType.json`, of the `jsonValue` of the type, also of a pure-Python
    /// subclass, such as an `UnparsedDataType`, whose `jsonValue` raises.
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("separators", (",", ":"))?;
        kwargs.set_item("sort_keys", true)?;
        py.import("json")?
            .getattr("dumps")?
            .call((slf.call_method0("jsonValue")?,), Some(&kwargs))?
            .extract()
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
        Ok(format!("{}()", slf.get_type().name()?))
    }

    /// As pyspark's `simpleString()`. An array, map or struct of a user-defined type names it
    /// by the `simpleString()` of its Python class, such as `array<vector>`.
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
        match &this.inner {
            DataType::Array { .. } | DataType::Map { .. } | DataType::Struct { .. }
                if has_udt(&this.inner) =>
            {
                simple_string_with_udts(py, &this.inner)
            }
            _ => Ok(this.inner.simple_string()),
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
    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<isize> {
        let any = slf.as_any();
        if is_udt(any)? {
            return any.str()?.hash();
        }
        let s = py_to_data_type(any)?.simple_string();
        pyo3::types::PyString::new(any.py(), &s).hash()
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

/// `DataType.__eq__` with pyspark semantics. pyspark compares
/// `isinstance(other, self.__class__) and self.__dict__ == other.__dict__`; the Rust-backed
/// classes keep their state in Rust rather than `__dict__`, so compare the core `DataType`
/// built from each side's live fields instead (a mutated `StructType` compares by its
/// current fields, and a bare `DataType` returned by `fromDDL` equals the concrete class).
/// `UserDefinedType` is a pure-Python subclass whose state does live in `__dict__`, so it
/// keeps pyspark's formula verbatim.
fn data_type_eq(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !b.is_instance_of::<PyDataType>() {
        return Ok(false);
    }
    let a_udt = is_udt(a)?;
    if a_udt || is_udt(b)? {
        // A non-UDT `self` is never a superclass of a UDT other than the bare base,
        // whose (empty) state pyspark would compare; treat that as unequal too.
        return Ok(a_udt
            && b.is_instance(&a.get_type())?
            && a.getattr("__dict__")?.eq(b.getattr("__dict__")?)?);
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
    if py_to_data_type(a)? != py_to_data_type(b)? {
        return Ok(false);
    }
    // The flags as they were given are compared as PySpark's `__dict__` does: `1` equals `True`.
    if has_other_flag(a.py(), a)? || has_other_flag(b.py(), b)? {
        let (a, _) = type_json_value(a.py(), a)?;
        let (b, _) = type_json_value(b.py(), b)?;
        return a.eq(b);
    }
    Ok(true)
}

// Define concrete type classes
#[pyclass(name = "NullType", module = "pyspark.sql.types", extends = PyDataType)]
pub struct PyNullType;

#[pymethods]
impl PyNullType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"void\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyNullType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(py, init_chain!(DataType::Null, PyNullType, []))
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "NullType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "void"
    }
    /// As PySpark's, which names this type `void`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "void"
    }
}

#[pyclass(name = "BooleanType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyBooleanType;

#[pymethods]
impl PyBooleanType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"boolean\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyBooleanType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(DataType::Boolean, PyBooleanType, [PyAtomicType]),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "BooleanType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "boolean"
    }
}

#[pyclass(name = "ByteType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyByteType;

#[pymethods]
impl PyByteType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"byte\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyByteType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(
                        DataType::Byte,
                        PyByteType,
                        [PyAtomicType, PyNumericType, PyIntegralType]
                    ),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "ByteType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "tinyint"
    }
}

#[pyclass(name = "ShortType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyShortType;

#[pymethods]
impl PyShortType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"short\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyShortType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(
                        DataType::Short,
                        PyShortType,
                        [PyAtomicType, PyNumericType, PyIntegralType]
                    ),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "ShortType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "smallint"
    }
}

#[pyclass(name = "IntegerType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyIntegerType;

#[pymethods]
impl PyIntegerType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"integer\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyIntegerType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(
                        DataType::Integer,
                        PyIntegerType,
                        [PyAtomicType, PyNumericType, PyIntegralType]
                    ),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "IntegerType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "int"
    }
}

#[pyclass(name = "LongType", module = "pyspark.sql.types", extends = PyIntegralType)]
pub struct PyLongType;

#[pymethods]
impl PyLongType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"long\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyLongType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(
                        DataType::Long,
                        PyLongType,
                        [PyAtomicType, PyNumericType, PyIntegralType]
                    ),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "LongType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "bigint"
    }
}

#[pyclass(name = "FloatType", module = "pyspark.sql.types", extends = PyFractionalType)]
pub struct PyFloatType;

#[pymethods]
impl PyFloatType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"float\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyFloatType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(
                        DataType::Float,
                        PyFloatType,
                        [PyAtomicType, PyNumericType, PyFractionalType]
                    ),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "FloatType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "float"
    }
}

#[pyclass(name = "DoubleType", module = "pyspark.sql.types", extends = PyFractionalType)]
pub struct PyDoubleType;

#[pymethods]
impl PyDoubleType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"double\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyDoubleType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(
                        DataType::Double,
                        PyDoubleType,
                        [PyAtomicType, PyNumericType, PyFractionalType]
                    ),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "DoubleType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "double"
    }
}

#[pyclass(name = "DecimalType", module = "pyspark.sql.types", extends = PyFractionalType)]
pub struct PyDecimalType {
    pub precision: i32,
    pub scale: i32,
}

#[pymethods]
impl PyDecimalType {
    /// As PySpark's `DecimalType.hasPrecisionInfo`, a public attribute that is always true.
    #[getter(hasPrecisionInfo)]
    fn has_precision_info(&self) -> bool {
        true
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        let dt = DataType::Decimal {
            precision: self.precision,
            scale: self.scale,
        };
        type_reduce(py, &dt.json())
    }
    /// `DecimalType(precision=10, scale=0)` - defaults match pyspark.
    #[new]
    #[pyo3(signature = (precision=10, scale=0))]
    fn new(precision: i32, scale: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Decimal { precision, scale },
            PyDecimalType { precision, scale },
            [PyAtomicType, PyNumericType, PyFractionalType],
            value
        )
    }
    #[getter]
    fn precision(&self) -> i32 {
        self.precision
    }
    #[getter]
    fn scale(&self) -> i32 {
        self.scale
    }
    fn __repr__(&self) -> String {
        format!("DecimalType({},{})", self.precision, self.scale)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("decimal({},{})", self.precision, self.scale)
    }
}

#[pyclass(name = "StringType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyStringType {
    pub collation: String,
}

#[pymethods]
impl PyStringType {
    /// The collation of the strings, as PySpark's `StringType.collation`.
    #[getter]
    fn collation(&self) -> String {
        self.collation.clone()
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
        let dt = DataType::String {
            collation: self.collation.clone(),
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    #[pyo3(signature = (collation="UTF8_BINARY".to_string()))]
    fn new(collation: String) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::String {
                collation: collation.clone()
            },
            PyStringType { collation },
            [PyAtomicType],
            value
        )
    }
    fn __repr__(&self) -> String {
        if self.collation == "UTF8_BINARY" {
            "StringType()".to_string()
        } else {
            format!("StringType('{}')", self.collation)
        }
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        if self.collation == "UTF8_BINARY" {
            "string".to_string()
        } else {
            format!("string collate {}", self.collation)
        }
    }

    /// Provider ("spark" for UTF8*, else "icu") for a collation name.
    /// Mirrors `StringType.collationProvider`.
    #[classmethod]
    #[pyo3(name = "collationProvider")]
    fn collation_provider(_cls: &Bound<'_, pyo3::types::PyType>, collation_name: &str) -> String {
        if collation_name.starts_with("UTF8") {
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

#[pyclass(name = "BinaryType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyBinaryType;

#[pymethods]
impl PyBinaryType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"binary\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyBinaryType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(DataType::Binary, PyBinaryType, [PyAtomicType]),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "BinaryType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "binary"
    }
}

#[pyclass(name = "DateType", module = "pyspark.sql.types", extends = PyDatetimeType)]
pub struct PyDateType;

#[pymethods]
impl PyDateType {
    /// Number of days from 0001-01-01 to 1970-01-01 (mirrors pyspark's DateType.EPOCH_ORDINAL).
    #[classattr]
    const EPOCH_ORDINAL: i32 = 719163;
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"date\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyDateType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(DataType::Date, PyDateType, [PyAtomicType, PyDatetimeType]),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "DateType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "date"
    }
}

#[pyclass(name = "TimestampType", module = "pyspark.sql.types", extends = PyDatetimeType)]
pub struct PyTimestampType;

#[pymethods]
impl PyTimestampType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"timestamp\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyTimestampType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(
                        DataType::Timestamp,
                        PyTimestampType,
                        [PyAtomicType, PyDatetimeType]
                    ),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "TimestampType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "timestamp"
    }
}

#[pyclass(name = "TimestampNTZType", module = "pyspark.sql.types", extends = PyDatetimeType)]
pub struct PyTimestampNTZType;

#[pymethods]
impl PyTimestampNTZType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"timestamp_ntz\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyTimestampNTZType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(
                        DataType::TimestampNtz,
                        PyTimestampNTZType,
                        [PyAtomicType, PyDatetimeType]
                    ),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "TimestampNTZType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "timestamp_ntz"
    }
    /// As PySpark's, which names this type `timestamp_ntz`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "timestamp_ntz"
    }
}

#[pyclass(name = "ArrayType", module = "pyspark.sql.types", extends = PyDataType)]
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
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
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
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let py = slf.py();
        let this = slf.borrow();
        let dt = DataType::Array {
            element_type: Box::new(this.element_object.live(py, &this.element_type)?),
            contains_null: this.contains_null,
        };
        let flagged = has_other_flag(slf.py(), slf.as_any())?;
        reduce_with_udts(slf.get_type(), &dt, flagged, || {
            let element_type = this.element_object.object(py, &this.element_type)?;
            (element_type, this.contains_null_object.clone_ref(py)).into_pyobject(py)
        })
    }
    // Kwargs mirror the reference: ArrayType(elementType, containsNull=True).
    #[new]
    #[pyo3(signature = (elementType, containsNull=default_flag()))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        elementType: &Bound<'_, PyAny>,
        containsNull: Py<PyAny>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let et = nested_data_type(elementType, "elementType")?;
        let contains_null = containsNull.bind(py).is_truthy()?;
        Ok(init_chain!(
            DataType::Array {
                element_type: Box::new(et.clone()),
                contains_null,
            },
            PyArrayType {
                element_type: et,
                contains_null,
                element_object: KeptType::of(elementType),
                contains_null_object: containsNull,
            },
            [],
            value
        ))
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        // Mirror pyspark: ArrayType(<elementType repr>, <containsNull>).
        let el = self.element_object.object(py, &self.element_type)?;
        let el_repr: String = el.bind(py).repr()?.extract()?;
        let contains_null: String = self.contains_null_object.bind(py).str()?.extract()?;
        Ok(format!("ArrayType({}, {})", el_repr, contains_null))
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "array<{}>",
            simple_string_with_udts(py, &self.element_object.live(py, &self.element_type)?)?
        ))
    }

    /// A copy with all nullability fields set true (recursively). Mirrors `ArrayType.toNullable()`.
    #[pyo3(name = "toNullable")]
    fn __at_to_nullable(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dt = (DataType::Array {
            element_type: Box::new(self.element_object.live(py, &self.element_type)?),
            contains_null: self.contains_null,
        })
        .to_nullable();
        data_type_to_py(py, &dt)
    }

    /// Build an ArrayType from its JSON value. Mirrors `ArrayType.fromJson`.
    #[classmethod]
    #[pyo3(name = "fromJson")]
    #[pyo3(signature = (json, fieldPath="".to_string(), collationsMap=None))]
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

#[pyclass(name = "MapType", module = "pyspark.sql.types", extends = PyDataType)]
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
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
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
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let py = slf.py();
        let this = slf.borrow();
        let dt = DataType::Map {
            key_type: Box::new(this.key_object.live(py, &this.key_type)?),
            value_type: Box::new(this.value_object.live(py, &this.value_type)?),
            value_contains_null: this.value_contains_null,
        };
        let flagged = has_other_flag(slf.py(), slf.as_any())?;
        reduce_with_udts(slf.get_type(), &dt, flagged, || {
            let key_type = this.key_object.object(py, &this.key_type)?;
            let value_type = this.value_object.object(py, &this.value_type)?;
            (
                key_type,
                value_type,
                this.value_contains_null_object.clone_ref(py),
            )
                .into_pyobject(py)
        })
    }
    // Kwargs mirror the reference: MapType(keyType, valueType, valueContainsNull=True).
    #[new]
    #[pyo3(signature = (keyType, valueType, valueContainsNull=default_flag()))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        keyType: &Bound<'_, PyAny>,
        valueType: &Bound<'_, PyAny>,
        valueContainsNull: Py<PyAny>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let kt = nested_data_type(keyType, "keyType")?;
        let vt = nested_data_type(valueType, "valueType")?;
        let value_contains_null = valueContainsNull.bind(py).is_truthy()?;
        Ok(init_chain!(
            DataType::Map {
                key_type: Box::new(kt.clone()),
                value_type: Box::new(vt.clone()),
                value_contains_null,
            },
            PyMapType {
                key_type: kt,
                value_type: vt,
                value_contains_null,
                key_object: KeptType::of(keyType),
                value_object: KeptType::of(valueType),
                value_contains_null_object: valueContainsNull,
            },
            [],
            value
        ))
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
    #[pyo3(name = "simpleString")]
    fn simple_string(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "map<{},{}>",
            simple_string_with_udts(py, &self.key_object.live(py, &self.key_type)?)?,
            simple_string_with_udts(py, &self.value_object.live(py, &self.value_type)?)?
        ))
    }

    /// A copy with all nullability fields set true (recursively). Mirrors `MapType.toNullable()`.
    #[pyo3(name = "toNullable")]
    fn __mt_to_nullable(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let dt = (DataType::Map {
            key_type: Box::new(self.key_object.live(py, &self.key_type)?),
            value_type: Box::new(self.value_object.live(py, &self.value_type)?),
            value_contains_null: self.value_contains_null,
        })
        .to_nullable();
        data_type_to_py(py, &dt)
    }

    /// Build a MapType from its JSON value. Mirrors `MapType.fromJson`.
    #[classmethod]
    #[pyo3(name = "fromJson")]
    #[pyo3(signature = (json, fieldPath="".to_string(), collationsMap=None))]
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

#[pyclass(name = "StructField", module = "pyspark.sql.types", extends = PyDataType)]
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
    /// The `needConversion` of the type, kept while the type cannot change: an atomic type
    /// object is not changed in place, so `add` need not read it again for each field. Reset by
    /// a set `dataType`.
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
        // of a nested struct, or else from the snapshot. The value is kept only for a type that
        // does not change: not a container, which can be edited in place, nor a year-month
        // interval, whose value follows `PYSPARK_YM_INTERVAL_LEGACY`.
        let (need, fixed) = match self.data_type_object.kept(py) {
            Some(object) => {
                let object = object.bind(py);
                let fixed = !(object.is_instance_of::<PyStructType>()
                    || object.is_instance_of::<PyArrayType>()
                    || object.is_instance_of::<PyMapType>()
                    || object.is_instance_of::<PyYearMonthIntervalType>()
                    || is_udt(object)?);
                (need_conversion_of_object(py, object)?, fixed)
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
    #[pyo3(name = "simpleString")]
    fn __sf_simple_string(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "{}:{}",
            self.field.name,
            simple_string_with_udts(py, &self.data_type_object.live(py, &self.field.data_type)?)?
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
    #[pyo3(name = "json")]
    fn __sf_json(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
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
    #[pyo3(name = "needConversion")]
    fn __sf_need_conversion(&self, py: Python<'_>) -> PyResult<bool> {
        need_conversion(py, &self.data_type_object.live(py, &self.field.data_type)?)
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
    #[new]
    #[pyo3(signature = (name, dataType, nullable=default_flag(), metadata=None))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        name: &Bound<'_, PyAny>,
        dataType: &Bound<'_, PyAny>,
        nullable: Py<PyAny>,
        metadata: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        // As PySpark's: the data type is asserted first, then the name.
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
        Ok(field_base(this))
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
    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let Ok(o) = other.cast::<PyStructField>() else {
            return Ok(false);
        };
        if slf.borrow().current_field(py)? != o.borrow().current_field(py)? {
            return Ok(false);
        }
        // The flags as they were given are compared as PySpark's `__dict__` does.
        if has_other_flag(py, slf.as_any())? || has_other_flag(py, other)? {
            let (a, _) = field_json_value(py, slf)?;
            let (b, _) = field_json_value(py, o)?;
            return a.eq(b);
        }
        Ok(true)
    }
    /// pyspark: `hash(str(self))` (inherited from `DataType`). Defining `__eq__` alone
    /// would leave StructField unhashable.
    fn __hash__(slf: &Bound<'_, Self>) -> PyResult<isize> {
        slf.str()?.hash()
    }
    /// A field pickles, as PySpark's does: of its name, type, nullability and metadata, with
    /// the class of `pyspark.sql.types`, also on the Python workers of the server.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(
        Bound<'py, pyo3::types::PyType>,
        Bound<'py, pyo3::types::PyTuple>,
    )> {
        let py = slf.py();
        let this = slf.borrow();
        let args = (
            this.field.name.clone(),
            this.data_type_object.object(py, &this.field.data_type)?,
            this.nullable.clone_ref(py),
            this.metadata.clone_ref(py),
        )
            .into_pyobject(py)?;
        Ok((slf.get_type(), args))
    }
}

#[pyclass(name = "StructType", module = "pyspark.sql.types", extends = PyDataType)]
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

#[pymethods]
impl PyStructType {
    // --- DataType object-model methods (v4.2.0 parity) ---
    #[pyo3(name = "json")]
    fn __obj_json(slf: &Bound<'_, Self>) -> PyResult<String> {
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
    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let py = slf.py();
        let this = slf.borrow();
        let dt = this.current_data_type(py)?;
        let flagged = has_other_flag(slf.py(), slf.as_any())?;
        reduce_with_udts(slf.get_type(), &dt, flagged, || {
            let objects = PyList::new(py, this.fields.bind(py).iter())?;
            (objects,).into_pyobject(py)
        })
    }
    /// `StructType(fields=None)` where each field is a `StructField`, as PySpark's `__init__`.
    #[new]
    #[pyo3(signature = (fields=None))]
    fn new(
        py: Python<'_>,
        fields: Option<Bound<'_, PyAny>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        // As PySpark's `if not fields`: no fields, or none given, is a new empty list.
        let Some(fields) = fields.filter(|fs| fs.is_truthy().unwrap_or(false)) else {
            let (base, parts) = struct_parts(py, PyList::empty(py))?;
            return Ok(init_chain!(base, parts, [], value));
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
        let (base, parts) = struct_parts(py, list)?;
        Ok(init_chain!(base, parts, [], value))
    }

    /// Append a field and return `self` (chainable, mirroring `StructType.add`):
    /// `StructType().add("a", "integer").add("b", StringType())`. The field goes to `fields` and
    /// its name to `names`, as PySpark's `add` does them.
    #[pyo3(signature = (field, data_type=None, nullable=default_flag(), metadata=None))]
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
        if let DataType::Struct { fields } = &mut me.as_super().inner {
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

    #[pyo3(name = "simpleString")]
    fn simple_string(&self, py: Python<'_>) -> PyResult<String> {
        simple_string_with_udts(py, &self.current_data_type(py)?)
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
        // The flags are written as they were given, which the core's bools are not.
        let text =
            if has_other_flag(py, slf.as_any())? {
                let mut error: Option<PyErr> = None;
                let text = data_type.tree_string_with_flags(depth, |path, flag, nullable| {
                    match tree_flag_text(slf.as_any(), path, flag) {
                        Ok(Some(text)) => text,
                        Ok(None) => nullable.to_string(),
                        Err(e) => {
                            error.get_or_insert(e);
                            nullable.to_string()
                        }
                    }
                });
                if let Some(e) = error {
                    return Err(e);
                }
                text
            } else {
                data_type.tree_string_with_depth(depth)
            };
        text.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// A copy with all fields made nullable (recursively). Mirrors `StructType.toNullable()`.
    #[pyo3(name = "toNullable")]
    fn __st_to_nullable(&self, py: Python<'_>) -> PyResult<Py<PyStructType>> {
        let fields = match self.current_data_type(py)?.to_nullable() {
            DataType::Struct { fields } => fields,
            _ => self.current_fields(py)?,
        };
        py_new_struct(py, fields)
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

#[pyclass(name = "CharType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyCharType {
    pub length: i32,
}

#[pymethods]
impl PyCharType {
    /// The length of the strings, as PySpark's `CharType.length`.
    #[getter]
    fn length(&self) -> i32 {
        self.length
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        let dt = DataType::Char {
            length: self.length,
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    fn new(length: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Char { length },
            PyCharType { length },
            [PyAtomicType],
            value
        )
    }
    fn __repr__(&self) -> String {
        format!("CharType({})", self.length)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("char({})", self.length)
    }
}

#[pyclass(name = "VarcharType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyVarcharType {
    pub length: i32,
}

#[pymethods]
impl PyVarcharType {
    /// The maximum length of the strings, as PySpark's `VarcharType.length`.
    #[getter]
    fn length(&self) -> i32 {
        self.length
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        let dt = DataType::Varchar {
            length: self.length,
        };
        type_reduce(py, &dt.json())
    }
    #[new]
    fn new(length: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Varchar { length },
            PyVarcharType { length },
            [PyAtomicType],
            value
        )
    }
    fn __repr__(&self) -> String {
        format!("VarcharType({})", self.length)
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> String {
        format!("varchar({})", self.length)
    }
}

#[pyclass(name = "TimeType", module = "pyspark.sql.types", extends = PyAnyTimeType)]
pub struct PyTimeType {
    pub precision: i32,
}

#[pymethods]
impl PyTimeType {
    /// The precision of the times, as PySpark's `TimeType.precision`.
    #[getter]
    fn precision(&self) -> i32 {
        self.precision
    }
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        let dt = DataType::Time {
            precision: self.precision,
        };
        type_reduce(py, &dt.json())
    }
    /// `TimeType(precision=6)` (microsecond precision default), matching pyspark 4.2.
    #[new]
    #[pyo3(signature = (precision=6))]
    fn new(precision: i32) -> pyo3::PyClassInitializer<Self> {
        init_chain!(
            DataType::Time { precision },
            PyTimeType { precision },
            [PyAtomicType, PyDatetimeType, PyAnyTimeType],
            value
        )
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

#[pyclass(name = "CalendarIntervalType", module = "pyspark.sql.types", extends = PyDataType)]
pub struct PyCalendarIntervalType;

#[pymethods]
impl PyCalendarIntervalType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        type_reduce(py, "\"calendarinterval\"")
    }
    /// The one instance of the type, as PySpark's `DataTypeSingleton`: every call of the class
    /// returns it, and raises its `TypeError` of an argument.
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Py<Self>> {
        singleton_arguments(args, kwargs)?;
        static INSTANCE: PyOnceLock<Py<PyCalendarIntervalType>> = PyOnceLock::new();
        INSTANCE
            .get_or_try_init(py, || {
                Py::new(
                    py,
                    init_chain!(DataType::CalendarInterval, PyCalendarIntervalType, []),
                )
            })
            .map(|instance| instance.clone_ref(py))
    }
    fn __repr__(&self) -> String {
        "CalendarIntervalType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "interval"
    }
    /// As PySpark's, which names this type `interval`.
    #[classmethod]
    #[pyo3(name = "typeName")]
    fn type_name(_cls: &Bound<'_, pyo3::types::PyType>) -> &'static str {
        "interval"
    }
}

#[pyclass(name = "YearMonthIntervalType", module = "pyspark.sql.types", extends = PyAnsiIntervalType)]
pub struct PyYearMonthIntervalType {
    pub start_field: i32,
    pub end_field: i32,
}

#[pymethods]
impl PyYearMonthIntervalType {
    /// The first field, as PySpark's `YearMonthIntervalType.startField`.
    #[getter(startField)]
    fn start_field(&self) -> i32 {
        self.start_field
    }
    /// The last field, as PySpark's `YearMonthIntervalType.endField`.
    #[getter(endField)]
    fn end_field(&self) -> i32 {
        self.end_field
    }
    #[classattr]
    const YEAR: i32 = 0;
    #[classattr]
    const MONTH: i32 = 1;
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        let dt = DataType::YearMonthInterval {
            start_field: self.start_field,
            end_field: self.end_field,
        };
        type_reduce(py, &dt.json())
    }
    /// Fields: YEAR=0, MONTH=1. Defaults to the full YEAR..MONTH range.
    /// As PySpark's `YearMonthIntervalType(startField=None, endField=None)`, of the fields YEAR
    /// and MONTH: `INVALID_INTERVAL_CASTING` for another.
    #[new]
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        startField: Option<&Bound<'_, PyAny>>,
        endField: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let (start, end) = interval_fields(py, startField, endField, (0, 1), 1)?;
        Ok(init_chain!(
            DataType::YearMonthInterval {
                start_field: start,
                end_field: end,
            },
            PyYearMonthIntervalType {
                start_field: start,
                end_field: end,
            },
            [PyAtomicType, PyAnsiIntervalType],
            value
        ))
    }
    fn __repr__(&self) -> String {
        format!(
            "YearMonthIntervalType({}, {})",
            self.start_field, self.end_field
        )
    }
}

#[pyclass(name = "DayTimeIntervalType", module = "pyspark.sql.types", extends = PyAnsiIntervalType)]
pub struct PyDayTimeIntervalType {
    pub start_field: i32,
    pub end_field: i32,
}

#[pymethods]
impl PyDayTimeIntervalType {
    /// The first field, as PySpark's `DayTimeIntervalType.startField`.
    #[getter(startField)]
    fn start_field(&self) -> i32 {
        self.start_field
    }
    /// The last field, as PySpark's `DayTimeIntervalType.endField`.
    #[getter(endField)]
    fn end_field(&self) -> i32 {
        self.end_field
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
        let dt = DataType::DayTimeInterval {
            start_field: self.start_field,
            end_field: self.end_field,
        };
        type_reduce(py, &dt.json())
    }
    /// Fields: DAY=0, HOUR=1, MINUTE=2, SECOND=3. Defaults to the full DAY..SECOND range.
    /// As PySpark's `DayTimeIntervalType(startField=None, endField=None)`, of the fields DAY,
    /// HOUR, MINUTE and SECOND: `INVALID_INTERVAL_CASTING` for another.
    #[new]
    #[pyo3(signature = (startField=None, endField=None))]
    #[allow(non_snake_case)]
    fn new(
        py: Python<'_>,
        startField: Option<&Bound<'_, PyAny>>,
        endField: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let (start, end) = interval_fields(py, startField, endField, (0, 3), 3)?;
        Ok(init_chain!(
            DataType::DayTimeInterval {
                start_field: start,
                end_field: end,
            },
            PyDayTimeIntervalType {
                start_field: start,
                end_field: end,
            },
            [PyAtomicType, PyAnsiIntervalType],
            value
        ))
    }
    fn __repr__(&self) -> String {
        format!(
            "DayTimeIntervalType({}, {})",
            self.start_field, self.end_field
        )
    }
}

#[pyclass(name = "VariantType", module = "pyspark.sql.types", extends = PyAtomicType)]
pub struct PyVariantType;

#[pymethods]
impl PyVariantType {
    // --- DataType object-model methods (v4.2.0 parity) ---
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
        variant: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        shim_conversion(slf.as_any(), "_to_internal", variant)
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, "\"variant\"")
    }
    #[new]
    fn new() -> pyo3::PyClassInitializer<Self> {
        init_chain!(DataType::Variant, PyVariantType, [PyAtomicType])
    }
    fn __repr__(&self) -> String {
        "VariantType()".to_string()
    }
    #[pyo3(name = "simpleString")]
    fn simple_string(&self) -> &'static str {
        "variant"
    }
}

// Class constants shared by all spatial types (SpatialType is macro-generated with no
// #[pymethods], so this companion block adds the reference SpatialType.MIXED_* constants).
#[pymethods]
impl PySpatialType {
    #[classattr]
    const MIXED_CRS: &'static str = "SRID:ANY";
    #[classattr]
    const MIXED_SRID: i32 = -1;
}

#[pyclass(name = "GeometryType", module = "pyspark.sql.types", extends = PySpatialType)]
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
    /// As PySpark's `GeometryType(srid)`: `"ANY"` for a mixed SRID, or an SRID of a
    /// coordinate reference system that PySpark knows.
    #[new]
    fn new(srid: &Bound<'_, PyAny>) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let srid = spatial_type_srid(srid, false)?;
        Ok(init_chain!(
            DataType::Geometry { srid },
            PyGeometryType { srid },
            [PyAtomicType, PySpatialType],
            value
        ))
    }
    #[getter]
    fn srid(&self) -> i32 {
        self.srid
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, &DataType::Geometry { srid: self.srid }.json())
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

#[pyclass(name = "GeographyType", module = "pyspark.sql.types", extends = PySpatialType)]
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
    /// As PySpark's `GeographyType(srid)`: `"ANY"` for a mixed SRID, or an SRID of a
    /// geographic coordinate reference system that PySpark knows.
    #[new]
    fn new(srid: &Bound<'_, PyAny>) -> PyResult<pyo3::PyClassInitializer<Self>> {
        let srid = spatial_type_srid(srid, true)?;
        Ok(init_chain!(
            DataType::Geography { srid },
            PyGeographyType { srid },
            [PyAtomicType, PySpatialType],
            value
        ))
    }
    #[getter]
    fn srid(&self) -> i32 {
        self.srid
    }
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        type_reduce(py, &DataType::Geography { srid: self.srid }.json())
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
        DataType::Unparsed { data_type_string } => match lowered_field(py, data_type_string)? {
            Some(field) => field.call_method0("simpleString")?.extract()?,
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
    macro_rules! obj {
        ($init:expr) => {
            Py::new(py, $init)?.into_any()
        };
    }
    let o = match dt {
        DataType::Null => PyNullType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any(),
        DataType::Boolean => {
            PyBooleanType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any()
        }
        DataType::Byte => PyByteType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any(),
        DataType::Short => PyShortType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any(),
        DataType::Integer => {
            PyIntegerType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any()
        }
        DataType::Long => PyLongType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any(),
        DataType::Float => PyFloatType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any(),
        DataType::Double => {
            PyDoubleType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any()
        }
        DataType::Decimal { precision, scale } => {
            let (precision, scale) = (*precision, *scale);
            obj!(init_chain!(
                DataType::Decimal { precision, scale },
                PyDecimalType { precision, scale },
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
            PyBinaryType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any()
        }
        DataType::Date => PyDateType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any(),
        DataType::Timestamp => {
            PyTimestampType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any()
        }
        DataType::TimestampNtz => {
            PyTimestampNTZType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any()
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
            PyCalendarIntervalType::new(py, &pyo3::types::PyTuple::empty(py), None)?.into_any()
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
        DataType::Unparsed { data_type_string } => match lowered_field(py, data_type_string)? {
            Some(field) => field.unbind(),
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
    // a DDL string that the server parses.
    if let Ok(unparsed_cls) = obj
        .py()
        .import("pyspark.sql.connect.types")
        .and_then(|m| m.getattr("UnparsedDataType"))
    {
        if obj.is_instance(&unparsed_cls)? {
            return Ok(DataType::Unparsed {
                data_type_string: obj.getattr("data_type_string")?.extract()?,
            });
        }
    }
    // The bare base `DataType`, which only `DataType()` and pure-Python subclasses are, is
    // matched LAST: concrete subclasses (which all share the PyDataType base) are handled
    // above via their own fields, so a mutated StructType reflects its live fields rather
    // than the base's snapshot.
    if let Ok(d) = obj.extract::<PyRef<PyDataType>>() {
        return Ok(d.inner.clone());
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

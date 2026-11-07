//! Schema inference and conversion of local data for `createDataFrame`, as PySpark 4.3.0
//! does them: `_infer_schema`, `_infer_type`, `_merge_type` and `_has_nulltype` of
//! `pyspark.sql.types`, and `LocalDataToArrowConversion` of `pyspark.sql.conversion`, with
//! which Spark Connect converts local data. The checks run in PySpark's order and raise its
//! errors, so that the same data gives the same schema, values and errors.

use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{PyAssertionError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    IntoPyDict, PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple,
    PyType,
};
use spark_connect::conf::RuntimeConf;
use spark_connect::row::{StructValue, Value};
use spark_connect::types::{DataType, GeographyType, GeometryType, StructField};

use crate::column::raise_pyspark;
use crate::errors::ResultExt;
use crate::row::PyRow;
use crate::values::{PyGeography, PyGeometry, PyVariantVal};

static DECIMAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static DATE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static DATETIME: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static TIME: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static TIMEDELTA: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static ARRAY: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static SORTED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The session configurations that inference reads, as PySpark reads them for
/// `createDataFrame`.
struct Confs {
    dict_as_struct: bool,
    array_from_first_element: bool,
    map_from_first_pair: bool,
    prefer_timestamp_ntz: bool,
}

/// The configurations, read from the server, in one request, the first time that inference
/// needs one. Only dicts, lists of more than one element and naive datetimes depend on them,
/// so other data is inferred without a request, as without a server.
pub(crate) struct LazyConfs {
    conf: RuntimeConf,
    confs: OnceCell<Confs>,
}

impl LazyConfs {
    pub(crate) fn new(conf: RuntimeConf) -> Self {
        LazyConfs {
            conf,
            confs: OnceCell::new(),
        }
    }

    fn get(&self, py: Python<'_>) -> PyResult<&Confs> {
        if let Some(confs) = self.confs.get() {
            return Ok(confs);
        }
        let keys = [
            "spark.sql.pyspark.inferNestedDictAsStruct.enabled",
            "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
            "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
            "spark.sql.timestampType",
        ];
        let conf = &self.conf;
        let values = py
            .detach(|| conf.get_many(&keys).map_err(Box::new))
            .to_pyerr()?;
        let is =
            |i: usize, expected: &str| values.get(i).and_then(|v| v.as_deref()) == Some(expected);
        Ok(self.confs.get_or_init(|| Confs {
            dict_as_struct: is(0, "true"),
            array_from_first_element: is(1, "true"),
            map_from_first_pair: is(2, "true"),
            prefer_timestamp_ntz: is(3, "TIMESTAMP_NTZ"),
        }))
    }
}

/// Where an option of inference takes its value from.
#[derive(Clone, Copy)]
enum Source {
    /// False, where PySpark leaves the option at its default.
    Off,
    /// `spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled`.
    MapFromFirstPair,
    /// `spark.sql.timestampType` being `TIMESTAMP_NTZ`.
    PreferTimestampNtz,
}

/// The options that inference passes down, as PySpark does. `infer_dict_as_struct` and
/// `infer_array_from_first_element` always come from the configurations, and the other two
/// shift where PySpark passes them on.
#[derive(Clone, Copy)]
pub(crate) struct Options<'c> {
    confs: &'c LazyConfs,
    map_from_first_pair: Source,
    prefer_timestamp_ntz: Source,
}

impl<'c> Options<'c> {
    pub(crate) fn new(confs: &'c LazyConfs) -> Self {
        Options {
            confs,
            map_from_first_pair: Source::MapFromFirstPair,
            prefer_timestamp_ntz: Source::PreferTimestampNtz,
        }
    }

    /// For the elements of a list. PySpark passes the options to `_infer_type` by position
    /// without `infer_map_from_first_pair`, so that it takes the value of
    /// `prefer_timestamp_ntz`, which is left false.
    fn for_elements(self) -> Self {
        Options {
            map_from_first_pair: self.prefer_timestamp_ntz,
            prefer_timestamp_ntz: Source::Off,
            ..self
        }
    }

    /// For a row in a value, such as a tuple, a Row or an object, which PySpark infers
    /// without `infer_map_from_first_pair`.
    fn for_nested_rows(self) -> Self {
        Options {
            map_from_first_pair: Source::Off,
            ..self
        }
    }

    fn value(&self, py: Python<'_>, source: Source) -> PyResult<bool> {
        Ok(match source {
            Source::Off => false,
            Source::MapFromFirstPair => self.confs.get(py)?.map_from_first_pair,
            Source::PreferTimestampNtz => self.confs.get(py)?.prefer_timestamp_ntz,
        })
    }
}

pub(crate) fn string_type() -> DataType {
    DataType::String {
        collation: "UTF8_BINARY".to_string(),
    }
}

fn nullable_field(name: String, data_type: DataType) -> StructField {
    StructField {
        name,
        data_type,
        nullable: true,
        metadata: Default::default(),
    }
}

fn map_type(key_type: DataType, value_type: DataType) -> DataType {
    DataType::Map {
        key_type: Box::new(key_type),
        value_type: Box::new(value_type),
        value_contains_null: true,
    }
}

pub(crate) fn type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.get_type().name()?.to_string())
}

/// A nullable field of `name`, which PySpark's `StructField` asserts is a string.
fn field_of(name: &Bound<'_, PyAny>, data_type: DataType) -> PyResult<StructField> {
    match name.cast::<PyString>() {
        Ok(name) => Ok(nullable_field(name.to_cow()?.into_owned(), data_type)),
        Err(_) => Err(PyAssertionError::new_err(format!(
            "field name {} should be a string",
            name.str()?
        ))),
    }
}

/// How deep values may nest in inference. PySpark recurses in Python, so a value nested
/// deeper than its recursion limit allows, as an object that refers to itself, raises
/// `RecursionError`. Inference here recurses in Rust frames, which that limit does not count
/// and which could overflow the thread's stack, so it keeps a depth of its own. A level takes
/// about 5 KiB of stack, so 64 levels fit a thread of 512 KiB, and they are deeper than the
/// server can analyze nested types in practice.
const MAX_DEPTH: usize = 64;

thread_local! {
    static DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One more level of nesting for as long as it lives.
struct Nesting;

impl Nesting {
    fn enter() -> PyResult<Nesting> {
        DEPTH.with(|depth| {
            if depth.get() >= MAX_DEPTH {
                return Err(pyo3::exceptions::PyRecursionError::new_err(
                    "maximum recursion depth exceeded while inferring the type of a value",
                ));
            }
            depth.set(depth.get() + 1);
            Ok(Nesting)
        })
    }
}

impl Drop for Nesting {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// The type of `obj`, as PySpark's `_infer_type`.
pub(crate) fn infer_type(obj: &Bound<'_, PyAny>, o: Options<'_>) -> PyResult<DataType> {
    let _nesting = Nesting::enter()?;
    let py = obj.py();
    if obj.is_none() {
        return Ok(DataType::Null);
    }
    // The types that PySpark maps by `type(obj)`, before it looks for `__UDT__`, which none
    // of them can have: neither these classes nor their instances take new attributes.
    if obj.is_exact_instance_of::<PyBool>() {
        return Ok(DataType::Boolean);
    }
    if obj.is_exact_instance_of::<PyInt>() {
        return Ok(DataType::Long);
    }
    if obj.is_exact_instance_of::<PyFloat>() {
        return Ok(DataType::Double);
    }
    if obj.is_exact_instance_of::<PyString>() {
        return Ok(string_type());
    }
    if obj.is_exact_instance_of::<PyBytes>() || obj.is_exact_instance_of::<PyByteArray>() {
        return Ok(DataType::Binary);
    }
    let ty = obj.get_type();
    if ty.is(DECIMAL.import(py, "decimal", "Decimal")?) {
        // As PySpark: the precision and scale of a value may differ from row to row.
        return Ok(DataType::Decimal {
            precision: 38,
            scale: 18,
        });
    }
    if ty.is(DATETIME.import(py, "datetime", "datetime")?) {
        let naive = obj.getattr(intern!(py, "tzinfo"))?.is_none();
        return Ok(if naive && o.value(py, o.prefer_timestamp_ntz)? {
            DataType::TimestampNtz
        } else {
            DataType::Timestamp
        });
    }
    if ty.is(DATE.import(py, "datetime", "date")?) {
        return Ok(DataType::Date);
    }
    if ty.is(TIME.import(py, "datetime", "time")?) {
        return Ok(DataType::Time { precision: 6 });
    }
    if ty.is(TIMEDELTA.import(py, "datetime", "timedelta")?) {
        // `DayTimeIntervalType()`, from days to seconds.
        return Ok(DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        });
    }
    if obj.is_exact_instance_of::<PyGeometry>() {
        let srid = obj.call_method0(intern!(py, "getSrid"))?.extract()?;
        return Ok(DataType::Geometry { srid });
    }
    if obj.is_exact_instance_of::<PyGeography>() {
        let srid = obj.call_method0(intern!(py, "getSrid"))?.extract()?;
        return Ok(DataType::Geography { srid });
    }
    if obj.hasattr(intern!(py, "__UDT__"))? {
        return crate::types::py_to_data_type(&obj.getattr(intern!(py, "__UDT__"))?);
    }
    if let Ok(dict) = obj.cast::<PyDict>() {
        return infer_dict(dict, o);
    }
    if let Ok(list) = obj.cast::<PyList>() {
        return infer_list(list, o);
    }
    if obj.is_instance(ARRAY.import(py, "array", "array")?.as_any())? {
        let typecode: String = obj.getattr(intern!(py, "typecode"))?.extract()?;
        return match array_element_type(&typecode) {
            Some(element_type) => Ok(DataType::Array {
                element_type: Box::new(element_type),
                contains_null: false,
            }),
            None => Err(raise_pyspark(
                py,
                "PySparkTypeError",
                "UNSUPPORTED_DATA_TYPE",
                &[("data_type", &format!("array({typecode})"))],
            )),
        };
    }
    if obj.is_instance_of::<PyVariantVal>() {
        return Ok(DataType::Variant);
    }
    match infer_schema(obj, None, o.for_nested_rows()) {
        Err(e) if e.is_instance_of::<PyTypeError>(py) => Err(raise_pyspark(
            py,
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE",
            &[("data_type", &type_name(obj)?)],
        )),
        result => result,
    }
}

/// A dict: a struct of its keys with non-null values, the map type of its first pair with
/// neither null, or, by default, the merged types of its keys and of its values.
fn infer_dict(dict: &Bound<'_, PyDict>, o: Options<'_>) -> PyResult<DataType> {
    let py = dict.py();
    if o.confs.get(py)?.dict_as_struct {
        let mut fields = Vec::new();
        for (key, value) in dict.iter() {
            if !key.is_none() && !value.is_none() {
                let data_type = infer_type(&value, o)?;
                fields.push(field_of(&key, data_type)?);
            }
        }
        return Ok(DataType::Struct { fields });
    }
    if o.value(py, o.map_from_first_pair)? {
        for (key, value) in dict.iter() {
            if !key.is_none() && !value.is_none() {
                return Ok(map_type(infer_type(&key, o)?, infer_type(&value, o)?));
            }
        }
        return Ok(map_type(DataType::Null, DataType::Null));
    }
    let (mut key_type, mut value_type) = (DataType::Null, DataType::Null);
    for (key, value) in dict.iter() {
        if !key.is_none() {
            key_type = merge_type(py, key_type, infer_type(&key, o)?)?;
        }
        if !value.is_none() {
            value_type = merge_type(py, value_type, infer_type(&value, o)?)?;
        }
    }
    Ok(map_type(key_type, value_type))
}

/// A list: an array of the merged types of its elements, or of its first element's type.
fn infer_list(list: &Bound<'_, PyList>, o: Options<'_>) -> PyResult<DataType> {
    let py = list.py();
    let elements = o.for_elements();
    let element_type = match list.len() {
        0 => DataType::Null,
        // One element has its own type either way, without reading the configuration.
        1 => infer_type(&list.get_item(0)?, elements)?,
        _ if o.confs.get(py)?.array_from_first_element => infer_type(&list.get_item(0)?, elements)?,
        _ => {
            let mut merged = DataType::Null;
            for (i, value) in list.iter().enumerate() {
                let data_type = infer_type(&value, elements)?;
                merged = if i == 0 {
                    data_type
                } else {
                    merge_type(py, merged, data_type)?
                };
            }
            merged
        }
    };
    Ok(DataType::Array {
        element_type: Box::new(element_type),
        contains_null: true,
    })
}

/// The element type of an `array.array` of `typecode`, as PySpark maps the typecodes by the
/// sizes of their C types; the unsigned ones take one more bit, and none needs more than 64.
fn array_element_type(typecode: &str) -> Option<DataType> {
    use std::ffi::{c_int, c_long, c_schar, c_short, c_uchar, c_uint, c_ulong, c_ushort};
    use std::mem::size_of;
    let integral = |size: usize, unsigned: bool| match size * 8 + usize::from(unsigned) {
        0..=8 => Some(DataType::Byte),
        9..=16 => Some(DataType::Short),
        17..=32 => Some(DataType::Integer),
        33..=64 => Some(DataType::Long),
        _ => None,
    };
    match typecode {
        "f" => Some(DataType::Float),
        "d" => Some(DataType::Double),
        "u" => Some(string_type()),
        "b" => integral(size_of::<c_schar>(), false),
        "h" => integral(size_of::<c_short>(), false),
        "i" => integral(size_of::<c_int>(), false),
        "l" => integral(size_of::<c_long>(), false),
        "B" => integral(size_of::<c_uchar>(), true),
        "H" => integral(size_of::<c_ushort>(), true),
        "I" => integral(size_of::<c_uint>(), true),
        "L" => integral(size_of::<c_ulong>(), true),
        _ => None,
    }
}

type Items<'py> = Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>;

/// `sorted(mapping.items())`, as PySpark orders the fields of a dict or of an object.
fn sorted_items<'py>(mapping: &Bound<'py, PyAny>) -> PyResult<Items<'py>> {
    let py = mapping.py();
    let sorted = SORTED.import(py, "builtins", "sorted")?;
    let items = sorted.call1((mapping.call_method0(intern!(py, "items"))?,))?;
    items
        .try_iter()?
        .map(|item| {
            let item = item?;
            Ok((item.get_item(0)?, item.get_item(1)?))
        })
        .collect()
}

/// The values of a tuple or list, named by `names` as `_infer_schema` zips them. Without
/// names, they are `_1`, `_2`, and so on; a shorter list of names is extended so, in place.
pub(crate) fn positional<'py>(
    py: Python<'py>,
    values: Vec<Bound<'py, PyAny>>,
    names: Option<&mut Vec<String>>,
) -> Items<'py> {
    let defaults: Vec<String>;
    let names: &[String] = match names {
        Some(names) => {
            let width = values.len();
            if names.len() < width {
                let start = names.len() + 1;
                names.extend((start..=width).map(|i| format!("_{i}")));
            }
            names
        }
        None => {
            defaults = (1..=values.len()).map(|i| format!("_{i}")).collect();
            &defaults
        }
    };
    names
        .iter()
        .map(|name| PyString::new(py, name).into_any())
        .zip(values)
        .collect()
}

/// The (name, value) pairs of a row, as `_infer_schema` takes them: a dict by its sorted
/// keys, a Row or a namedtuple by its field names, another tuple or a list by `names`, and
/// another object by the sorted names of its `__dict__`.
fn row_items<'py>(
    row: &Bound<'py, PyAny>,
    names: Option<&mut Vec<String>>,
) -> PyResult<Items<'py>> {
    let py = row.py();
    if row.is_instance_of::<PyDict>() {
        return sorted_items(row);
    }
    if let Ok(pyrow) = row.cast::<PyRow>() {
        let pyrow = pyrow.borrow();
        let (fields, values) = pyrow.parts(py)?;
        return Ok(match fields {
            Some(fields) => fields
                .iter()
                .map(|f| PyString::new(py, f).into_any())
                .zip(values.iter())
                .collect(),
            None => positional(py, values.iter().collect(), names),
        });
    }
    if row.is_instance_of::<PyTuple>() || row.is_instance_of::<PyList>() {
        let fields = if row.hasattr(intern!(py, "__fields__"))? {
            Some(row.getattr(intern!(py, "__fields__"))?)
        } else if row.hasattr(intern!(py, "_fields"))? {
            Some(row.getattr(intern!(py, "_fields"))?)
        } else {
            None
        };
        let values = row.try_iter()?.collect::<PyResult<Vec<_>>>()?;
        return match fields {
            Some(fields) => Ok(fields
                .try_iter()?
                .collect::<PyResult<Vec<_>>>()?
                .into_iter()
                .zip(values)
                .collect()),
            None => Ok(positional(py, values, names)),
        };
    }
    if row.hasattr(intern!(py, "__dict__"))? {
        return sorted_items(&row.getattr(intern!(py, "__dict__"))?);
    }
    Err(raise_pyspark(
        py,
        "PySparkTypeError",
        "CANNOT_INFER_SCHEMA_FOR_TYPE",
        &[("data_type", &type_name(row)?)],
    ))
}

/// The struct type of (name, value) pairs. A `TypeError` in inferring a value, as from a
/// type that cannot be inferred or merged, becomes `CANNOT_INFER_TYPE_FOR_FIELD`.
pub(crate) fn infer_fields(py: Python<'_>, items: Items<'_>, o: Options<'_>) -> PyResult<DataType> {
    let mut fields = Vec::with_capacity(items.len());
    for (name, value) in items {
        let data_type = match infer_type(&value, o) {
            Err(e) if e.is_instance_of::<PyTypeError>(py) => {
                return Err(raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "CANNOT_INFER_TYPE_FOR_FIELD",
                    &[("field_name", &name.str()?.to_cow()?)],
                ));
            }
            result => result?,
        };
        fields.push(field_of(&name, data_type)?);
    }
    Ok(DataType::Struct { fields })
}

/// The struct type of a row, as PySpark's `_infer_schema`.
pub(crate) fn infer_schema(
    row: &Bound<'_, PyAny>,
    names: Option<&mut Vec<String>>,
    o: Options<'_>,
) -> PyResult<DataType> {
    let items = row_items(row, names)?;
    infer_fields(row.py(), items, o)
}

/// Whether PySpark's class of `dt` is an `AtomicType`.
pub(crate) fn is_atomic(dt: &DataType) -> bool {
    !matches!(
        dt,
        DataType::Null
            | DataType::CalendarInterval
            | DataType::Array { .. }
            | DataType::Map { .. }
            | DataType::Struct { .. }
            | DataType::Udt { .. }
            | DataType::Unparsed { .. }
    )
}

/// The name of PySpark's class of `dt`, by which its errors name a type.
pub(crate) fn class_name(dt: &DataType) -> String {
    let name = match dt {
        DataType::Null => "NullType",
        DataType::Boolean => "BooleanType",
        DataType::Byte => "ByteType",
        DataType::Short => "ShortType",
        DataType::Integer => "IntegerType",
        DataType::Long => "LongType",
        DataType::Float => "FloatType",
        DataType::Double => "DoubleType",
        DataType::Decimal { .. } => "DecimalType",
        DataType::String { .. } => "StringType",
        DataType::Char { .. } => "CharType",
        DataType::Varchar { .. } => "VarcharType",
        DataType::Binary => "BinaryType",
        DataType::Date => "DateType",
        DataType::Timestamp => "TimestampType",
        DataType::TimestampNtz => "TimestampNTZType",
        DataType::Time { .. } => "TimeType",
        DataType::TimestampNtzNanos { .. } => "TimestampNTZNanosType",
        DataType::TimestampLtzNanos { .. } => "TimestampLTZNanosType",
        DataType::CalendarInterval => "CalendarIntervalType",
        DataType::YearMonthInterval { .. } => "YearMonthIntervalType",
        DataType::DayTimeInterval { .. } => "DayTimeIntervalType",
        DataType::Array { .. } => "ArrayType",
        DataType::Map { .. } => "MapType",
        DataType::Struct { .. } => "StructType",
        DataType::Variant => "VariantType",
        DataType::Geometry { .. } => "GeometryType",
        DataType::Geography { .. } => "GeographyType",
        DataType::Unparsed { .. } => "UnparsedDataType",
        DataType::Udt { python_class, .. } => {
            // A user-defined type is its own class, which `pyClass` names in full.
            let class = python_class.as_deref().unwrap_or("UserDefinedType");
            return class.rsplit('.').next().unwrap_or(class).to_string();
        }
    };
    name.to_string()
}

/// Whether two types are of the same PySpark class, as `_merge_type` compares them.
fn same_class(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::Udt { .. }, DataType::Udt { .. }) => class_name(a) == class_name(b),
        _ => std::mem::discriminant(a) == std::mem::discriminant(b),
    }
}

/// The merge of two inferred types, as PySpark's `_merge_type`: a null type takes the
/// other, a string absorbs another atomic type, and structs, arrays and maps merge by
/// their parts. Other types must be of the same class, and the first one is taken.
pub(crate) fn merge_type(py: Python<'_>, a: DataType, b: DataType) -> PyResult<DataType> {
    use DataType as D;
    match (&a, &b) {
        (D::Null, _) => return Ok(b),
        (_, D::Null) => return Ok(a),
        (D::Timestamp, D::TimestampNtz) => return Ok(a),
        (D::TimestampNtz, D::Timestamp) => return Ok(b),
        (D::Geometry { srid: x }, D::Geometry { srid: y }) if x != y => {
            return Ok(D::Geometry {
                srid: GeometryType::MIXED_SRID,
            });
        }
        (D::Geography { srid: x }, D::Geography { srid: y }) if x != y => {
            return Ok(D::Geography {
                srid: GeographyType::MIXED_SRID,
            });
        }
        _ => {}
    }
    if is_atomic(&a) && matches!(b, D::String { .. }) {
        return Ok(b);
    }
    if matches!(a, D::String { .. }) && is_atomic(&b) {
        return Ok(a);
    }
    if !same_class(&a, &b) {
        return Err(raise_pyspark(
            py,
            "PySparkTypeError",
            "CANNOT_MERGE_TYPE",
            &[
                ("data_type1", &class_name(&a)),
                ("data_type2", &class_name(&b)),
            ],
        ));
    }
    Ok(match (a, b) {
        (D::Struct { fields: a }, D::Struct { fields: b }) => {
            // `dict((f.name, f.dataType) for f in b.fields)`: each name at its first place,
            // with the type of its last field.
            let mut index: HashMap<String, usize> = HashMap::with_capacity(b.len());
            let mut others: Vec<(String, DataType)> = Vec::with_capacity(b.len());
            for f in b {
                match index.get(&f.name) {
                    Some(&i) => others[i].1 = f.data_type,
                    None => {
                        index.insert(f.name.clone(), others.len());
                        others.push((f.name, f.data_type));
                    }
                }
            }
            let mut names: HashSet<String> = HashSet::with_capacity(a.len());
            let mut fields = Vec::with_capacity(a.len() + others.len());
            for f in a {
                let other = match index.get(&f.name) {
                    Some(&i) => others[i].1.clone(),
                    None => D::Null,
                };
                names.insert(f.name.clone());
                fields.push(nullable_field(f.name, merge_type(py, f.data_type, other)?));
            }
            for (name, data_type) in others {
                if !names.contains(&name) {
                    fields.push(nullable_field(name, data_type));
                }
            }
            D::Struct { fields }
        }
        (
            D::Array {
                element_type: x, ..
            },
            D::Array {
                element_type: y, ..
            },
        ) => D::Array {
            element_type: Box::new(merge_type(py, *x, *y)?),
            contains_null: true,
        },
        (
            D::Map {
                key_type: ka,
                value_type: va,
                ..
            },
            D::Map {
                key_type: kb,
                value_type: vb,
                ..
            },
        ) => map_type(merge_type(py, *ka, *kb)?, merge_type(py, *va, *vb)?),
        (a, _) => a,
    })
}

/// Whether `dt` has a null type in it, which PySpark cannot infer data of.
pub(crate) fn has_null_type(dt: &DataType) -> bool {
    match dt {
        DataType::Struct { fields } => fields.iter().any(|f| has_null_type(&f.data_type)),
        DataType::Array { element_type, .. } => has_null_type(element_type),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_null_type(key_type) || has_null_type(value_type),
        other => matches!(other, DataType::Null),
    }
}

/// A PySpark error of `class` with a message alone, as PySpark raises some of them.
fn pyspark_message(py: Python<'_>, class: &str, message: &str) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py.import("pyspark.errors")?.getattr(class)?;
        Ok(PyErr::from_value(cls.call1((message,))?))
    };
    build().unwrap_or_else(|e| e)
}

fn data_type_str(py: Python<'_>, dt: &DataType) -> PyResult<String> {
    Ok(crate::types::data_type_to_py(py, dt)?
        .bind(py)
        .str()?
        .to_cow()?
        .into_owned())
}

fn must_not_be_none(py: Python<'_>, dt: &DataType) -> PyResult<PyErr> {
    Ok(pyspark_message(
        py,
        "PySparkValueError",
        &format!("input for {} must not be None", data_type_str(py, dt)?),
    ))
}

pub(crate) fn axis_length_mismatch(py: Python<'_>, expected: usize, actual: usize) -> PyErr {
    raise_pyspark(
        py,
        "PySparkValueError",
        "AXIS_LENGTH_MISMATCH",
        &[
            ("expected_length", &expected.to_string()),
            ("actual_length", &actual.to_string()),
        ],
    )
}

/// What PySpark's `assert` raises, with no message.
fn assertion() -> PyErr {
    PyAssertionError::new_err(())
}

/// The error of a value that this client cannot send in a column of its type yet.
fn unsupported(obj: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    Ok(PyTypeError::new_err(format!(
        "unsupported value type for createDataFrame: {}",
        type_name(obj)?
    )))
}

/// The name of the type of `obj` as PyArrow writes it in its errors: with its module, unless
/// it is a builtin type, such as `int` or `numpy.int64`.
fn arrow_type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let ty = obj.get_type();
    let (module, name) = (ty.module()?.to_string(), ty.qualname()?.to_string());
    Ok(if module == "builtins" {
        name
    } else {
        format!("{module}.{name}")
    })
}

/// An error of PyArrow, of `class` with `message`, as its conversions raise it.
fn pyarrow_error(py: Python<'_>, class: &str, message: String) -> PyResult<PyErr> {
    let error = py
        .import(intern!(py, "pyarrow"))?
        .getattr(class)?
        .call1((message,))?;
    Ok(PyErr::from_value(error))
}

/// The error of a value that PyArrow cannot convert to the type it is named by.
fn arrow_invalid(obj: &Bound<'_, PyAny>, arrow_type: &str) -> PyResult<PyErr> {
    Ok(PyValueError::new_err(format!(
        "Could not convert {} with type {}: tried to convert to {arrow_type}",
        obj.repr()?,
        arrow_type_name(obj)?
    )))
}

/// The kind of a NumPy scalar of a bool or a number, which PyArrow converts as the Python
/// value that it stands for.
#[derive(Clone, Copy, PartialEq)]
enum NumpyScalar {
    Bool,
    Integer,
    Floating,
}

/// The kind of `obj` if it is a NumPy scalar of a bool or a number. NumPy is not imported
/// when it is not loaded, as such a value can only exist once it was.
fn numpy_scalar(obj: &Bound<'_, PyAny>) -> PyResult<Option<NumpyScalar>> {
    let py = obj.py();
    let modules = py
        .import(intern!(py, "sys"))?
        .getattr(intern!(py, "modules"))?;
    let numpy = match modules.cast::<PyDict>()?.get_item(intern!(py, "numpy"))? {
        Some(numpy) if !numpy.is_none() => numpy,
        _ => return Ok(None),
    };
    for (class, kind) in [
        ("bool_", NumpyScalar::Bool),
        ("integer", NumpyScalar::Integer),
        ("floating", NumpyScalar::Floating),
    ] {
        if obj.is_instance(&numpy.getattr(class)?)? {
            return Ok(Some(kind));
        }
    }
    Ok(None)
}

/// An integer as a floating-point value, which PyArrow takes only when it is exact: within
/// 2^53 for a double and 2^24 for a float.
fn exact_float(obj: &Bound<'_, PyAny>, dt: &DataType) -> PyResult<f64> {
    let (bits, precision) = match dt {
        DataType::Float => (24, "single"),
        _ => (53, "double"),
    };
    let out_of_range = || -> PyResult<PyErr> {
        Ok(PyValueError::new_err(format!(
            "Integer value {} is outside of the range exactly representable by a IEEE 754 \
             {precision} precision value",
            obj.str()?
        )))
    };
    let value: i64 = match obj.extract() {
        Ok(value) => value,
        Err(_) => return Err(out_of_range()?),
    };
    if value.unsigned_abs() > 1u64 << bits {
        return Err(out_of_range()?);
    }
    Ok(value as f64)
}

/// Microseconds of a `timedelta`, exactly.
fn delta_micros(delta: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = delta.py();
    let days: i64 = delta.getattr(intern!(py, "days"))?.extract()?;
    let seconds: i64 = delta.getattr(intern!(py, "seconds"))?.extract()?;
    let micros: i64 = delta.getattr(intern!(py, "microseconds"))?.extract()?;
    Ok(days * 86_400_000_000 + seconds * 1_000_000 + micros)
}

/// Microseconds since the epoch of the instant of a datetime, which PySpark takes with
/// `astimezone(timezone.utc)`: a naive datetime is in the local time zone.
fn instant_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    let datetime = py.import(intern!(py, "datetime"))?;
    let utc = datetime
        .getattr(intern!(py, "timezone"))?
        .getattr(intern!(py, "utc"))?;
    let aware = obj.call_method1(intern!(py, "astimezone"), (&utc,))?;
    let epoch = DATETIME
        .import(py, "datetime", "datetime")?
        .call((1970, 1, 1), Some(&[("tzinfo", &utc)].into_py_dict(py)?))?;
    delta_micros(&aware.sub(epoch)?)
}

/// Microseconds since 1970-01-01 00:00 of the wall-clock time of a naive datetime, as
/// PyArrow takes a TIMESTAMP_NTZ value.
fn wall_clock_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    let epoch = DATETIME
        .import(py, "datetime", "datetime")?
        .call1((1970, 1, 1))?;
    delta_micros(&obj.sub(epoch)?)
}

/// A field value of `dt` from `obj`, as PySpark's converters and then PyArrow convert it.
pub(crate) fn convert(obj: &Bound<'_, PyAny>, dt: &DataType, nullable: bool) -> PyResult<Value> {
    let py = obj.py();
    if obj.is_none() {
        // A null type takes None either way.
        if !nullable && !matches!(dt, DataType::Null) {
            return Err(must_not_be_none(py, dt)?);
        }
        return Ok(Value::Null);
    }
    match dt {
        DataType::Null => Err(pyspark_message(
            py,
            "PySparkValueError",
            &format!(
                "input for {} must be None, but got {}",
                data_type_str(py, dt)?,
                obj.str()?
            ),
        )),
        DataType::Struct { fields } => convert_struct(obj, fields),
        DataType::Array {
            element_type,
            contains_null,
        } => {
            // As PySpark: an array whose elements need a converter takes a list or an
            // array.array, and another array is taken as it is, by PyArrow, which takes any
            // sequence, such as a tuple or a NumPy array of numbers or bools.
            if needs_converter(dt, nullable) {
                if !obj.is_instance_of::<PyList>()
                    && !obj.is_instance(ARRAY.import(py, "array", "array")?.as_any())?
                {
                    return Err(assertion());
                }
            } else if obj.is_instance_of::<PyDict>() || obj.try_iter().is_err() {
                return Err(pyarrow_error(
                    py,
                    "ArrowTypeError",
                    format!(
                        "Could not convert {} with type {}: was not a sequence or recognized \
                         null for conversion to list type",
                        obj.repr()?,
                        arrow_type_name(obj)?
                    ),
                )?);
            }
            let mut items = Vec::with_capacity(obj.len().unwrap_or(0));
            for item in obj.try_iter()? {
                items.push(convert(&item?, element_type, *contains_null)?);
            }
            Ok(Value::List(items))
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let dict = obj.cast::<PyDict>().map_err(|_| assertion())?;
            let mut entries = Vec::with_capacity(dict.len());
            for (key, value) in dict.iter() {
                entries.push((
                    convert(&key, key_type, false)?,
                    convert(&value, value_type, *value_contains_null)?,
                ));
            }
            Ok(Value::Map(entries))
        }
        DataType::Binary => {
            if let Ok(bytes) = obj.cast::<PyBytes>() {
                Ok(Value::Binary(bytes.as_bytes().to_vec()))
            } else if let Ok(bytes) = obj.cast::<PyByteArray>() {
                Ok(Value::Binary(bytes.to_vec()))
            } else {
                Err(assertion())
            }
        }
        DataType::Timestamp | DataType::TimestampLtzNanos { .. } => {
            if !obj.is_instance(DATETIME.import(py, "datetime", "datetime")?.as_any())? {
                return Err(assertion());
            }
            Ok(Value::Timestamp(instant_micros(obj)?))
        }
        DataType::TimestampNtz | DataType::TimestampNtzNanos { .. } => {
            if !obj.is_instance(DATETIME.import(py, "datetime", "datetime")?.as_any())?
                || !obj.getattr(intern!(py, "tzinfo"))?.is_none()
            {
                return Err(assertion());
            }
            Ok(Value::Timestamp(wall_clock_micros(obj)?))
        }
        DataType::Decimal { precision, scale } => {
            let decimal = DECIMAL.import(py, "decimal", "Decimal")?;
            if !obj.is_instance(decimal.as_any())? {
                return Err(assertion());
            }
            if obj.call_method0(intern!(py, "is_nan"))?.is_truthy()? {
                if !nullable {
                    return Err(must_not_be_none(py, dt)?);
                }
                return Ok(Value::Null);
            }
            // `value.quantize(Decimal("1E-<scale>"), context=Context(prec=<precision>,
            // rounding=ROUND_HALF_EVEN))`, written without an exponent.
            let module = py.import(intern!(py, "decimal"))?;
            let exp = decimal.call1((format!("1E-{scale}"),))?;
            let context = module.getattr(intern!(py, "Context"))?.call(
                (),
                Some(
                    &[
                        ("prec", precision.into_pyobject(py)?.into_any()),
                        ("rounding", module.getattr(intern!(py, "ROUND_HALF_EVEN"))?),
                    ]
                    .into_py_dict(py)?,
                ),
            )?;
            let quantized = obj.call_method(
                intern!(py, "quantize"),
                (exp,),
                Some(&[("context", context)].into_py_dict(py)?),
            )?;
            let text =
                PyString::new(py, "{:f}").call_method1(intern!(py, "format"), (quantized,))?;
            Ok(Value::Decimal {
                value: text.extract()?,
                precision: Some(*precision),
                scale: Some(*scale),
            })
        }
        DataType::String { .. } => {
            let text = if obj.is_exact_instance_of::<PyString>() {
                obj.cast::<PyString>()?.to_cow()?.into_owned()
            } else if obj.is_instance_of::<PyBool>() {
                // As PySpark Classic, which converts a bool to a string in the JVM.
                (if obj.is_truthy()? { "true" } else { "false" }).to_string()
            } else {
                obj.str()?.to_cow()?.into_owned()
            };
            Ok(Value::String(text))
        }
        DataType::Char { .. } | DataType::Varchar { .. } => {
            if let Ok(text) = obj.cast::<PyString>() {
                Ok(Value::String(text.to_cow()?.into_owned()))
            } else if let Ok(bytes) = obj.cast::<PyBytes>() {
                Ok(Value::String(
                    String::from_utf8(bytes.as_bytes().to_vec())
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                ))
            } else {
                Err(PyTypeError::new_err(format!(
                    "Expected bytes, got a '{}' object",
                    type_name(obj)?
                )))
            }
        }
        DataType::Boolean => {
            if obj.is_instance_of::<PyBool>() || numpy_scalar(obj)? == Some(NumpyScalar::Bool) {
                Ok(Value::Bool(obj.is_truthy()?))
            } else {
                Err(arrow_invalid(obj, "boolean")?)
            }
        }
        DataType::Byte | DataType::Short | DataType::Integer | DataType::Long => {
            convert_integral(obj, dt)
        }
        DataType::Float | DataType::Double => {
            // PyArrow takes a float, a bool, an int that it represents exactly, and a NumPy
            // float or integer, but not a NumPy bool.
            let value = if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyBool>() {
                obj.extract::<f64>()?
            } else if obj.is_instance_of::<PyInt>() {
                exact_float(obj, dt)?
            } else {
                match numpy_scalar(obj)? {
                    Some(NumpyScalar::Floating) => obj.extract::<f64>()?,
                    Some(NumpyScalar::Integer) => exact_float(obj, dt)?,
                    _ => {
                        let arrow_type = if matches!(dt, DataType::Float) {
                            "float"
                        } else {
                            "double"
                        };
                        return Err(arrow_invalid(obj, arrow_type)?);
                    }
                }
            };
            Ok(match dt {
                DataType::Float => Value::Float(value as f32),
                _ => Value::Double(value),
            })
        }
        DataType::DayTimeInterval { .. } => Ok(Value::DayTimeInterval(duration_micros(obj)?)),
        DataType::Date => Ok(Value::Date(date_days(obj)?)),
        // As PyArrow's `time64[ns]`, to which PySpark passes the value as it is: a time, and
        // the nanoseconds of an int, or of a float, truncated.
        DataType::Time { .. } => {
            if obj.is_instance(TIME.import(py, "datetime", "time")?.as_any())? {
                Ok(Value::Time(time_nanos(obj)?))
            } else if obj.is_instance_of::<PyInt>() {
                Ok(Value::Time(obj.extract()?))
            } else if obj.is_instance_of::<PyFloat>() {
                Ok(Value::Time(obj.extract::<f64>()? as i64))
            } else {
                Err(PyTypeError::new_err(format!(
                    "object of type {} cannot be converted to int",
                    obj.get_type().repr()?
                )))
            }
        }
        // As PySpark's converters of these types: the value and metadata of a VariantVal, and
        // the srid and WKB of a Geometry or a Geography, the structs of their Arrow types.
        DataType::Variant => match obj.cast::<PyVariantVal>() {
            Ok(variant) => {
                let variant = variant.borrow();
                Ok(Value::Variant {
                    value: variant.value.clone(),
                    metadata: variant.metadata.clone(),
                })
            }
            Err(_) => Err(raise_pyspark(
                py,
                "PySparkValueError",
                "MALFORMED_VARIANT",
                &[],
            )),
        },
        DataType::Geometry { .. } => match obj.cast::<PyGeometry>() {
            Ok(geometry) => {
                let geometry = geometry.borrow();
                Ok(spatial_value(geometry.srid, &geometry.wkb))
            }
            Err(_) => Err(raise_pyspark(
                py,
                "PySparkValueError",
                "MALFORMED_GEOMETRY",
                &[],
            )),
        },
        DataType::Geography { .. } => match obj.cast::<PyGeography>() {
            Ok(geography) => {
                let geography = geography.borrow();
                Ok(spatial_value(geography.srid, &geography.wkb))
            }
            Err(_) => Err(raise_pyspark(
                py,
                "PySparkValueError",
                "MALFORMED_GEOGRAPHY",
                &[],
            )),
        },
        // As PySpark's converter of a user-defined type: the value serialized by the UDT, and
        // converted as its SQL type.
        DataType::Udt {
            sql_type: Some(sql_type),
            python_class,
            serialized_python_class,
            ..
        } => {
            let udt = udt_of(
                obj,
                python_class.as_deref(),
                serialized_python_class.as_deref(),
            )?;
            let serialized = udt.call_method1(intern!(py, "serialize"), (obj,))?;
            convert(&serialized, sql_type, nullable)
        }
        _ => Err(unsupported(obj)?),
    }
}

/// The days since the epoch of a value of a date column, as PyArrow converts it to date32: a
/// date, or the date of a datetime; an int, or a float truncated, as days; and a NumPy bool or
/// number as the value it stands for. Another value raises PyArrow's error.
fn date_days(obj: &Bound<'_, PyAny>) -> PyResult<i32> {
    let py = obj.py();
    if obj.is_instance(DATE.import(py, "datetime", "date")?.as_any())? {
        let ordinal: i64 = obj.call_method0(intern!(py, "toordinal"))?.extract()?;
        return Ok((ordinal - 719_163) as i32);
    }
    if obj.is_instance_of::<PyBool>() {
        return Err(pyarrow_error(
            py,
            "ArrowTypeError",
            "Expected integer, got bool".into(),
        )?);
    }
    let kind = if obj.is_instance_of::<PyInt>() {
        Some(NumpyScalar::Integer)
    } else if obj.is_instance_of::<PyFloat>() {
        Some(NumpyScalar::Floating)
    } else {
        numpy_scalar(obj)?
    };
    let days: i64 = match kind {
        Some(NumpyScalar::Bool) => i64::from(obj.is_truthy()?),
        Some(NumpyScalar::Integer) => obj.extract()?,
        Some(NumpyScalar::Floating) => {
            let value: f64 = obj.extract()?;
            if value.is_nan() {
                return Err(PyValueError::new_err("cannot convert float NaN to integer"));
            }
            if value.is_infinite() {
                return Err(PyOverflowError::new_err(
                    "cannot convert float infinity to integer",
                ));
            }
            let truncated = value.trunc();
            if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&truncated) {
                return Err(PyOverflowError::new_err(
                    "Python int too large to convert to C long",
                ));
            }
            truncated as i64
        }
        None => {
            return Err(pyarrow_error(
                py,
                "ArrowTypeError",
                format!(
                    "object of type {} cannot be converted to int",
                    obj.get_type().repr()?
                ),
            )?)
        }
    };
    // The days of PyArrow's dates: from 0001-01-01 to 9999-12-31, and a larger magnitude is
    // out of the range of a Python timedelta.
    if days.unsigned_abs() > 999_999_999 {
        return Err(PyOverflowError::new_err(format!(
            "days={days}; must have magnitude <= 999999999"
        )));
    }
    if !(-719_162..=2_932_896).contains(&days) {
        return Err(PyOverflowError::new_err("date value out of range"));
    }
    Ok(days as i32)
}

/// Whether PySpark's `LocalDataToArrowConversion._need_converter` gives a value of `dt` a
/// converter, for a value of `nullable`: the values that PyArrow takes as they are do not.
fn needs_converter(dt: &DataType, nullable: bool) -> bool {
    if !nullable {
        return true;
    }
    match dt {
        DataType::Null
        | DataType::Struct { .. }
        | DataType::Map { .. }
        | DataType::Binary
        | DataType::Timestamp
        | DataType::TimestampNtz
        | DataType::Decimal { .. }
        | DataType::String { .. }
        | DataType::Udt { .. }
        | DataType::Variant
        | DataType::Geometry { .. }
        | DataType::Geography { .. } => true,
        DataType::Array {
            element_type,
            contains_null,
        } => needs_converter(element_type, *contains_null),
        _ => false,
    }
}

/// The nanoseconds since midnight of a `datetime.time`, as PySpark's `TimeType.toInternal`.
pub(crate) fn time_nanos(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    let field = |name: &Bound<'_, PyString>| -> PyResult<i64> { obj.getattr(name)?.extract() };
    let seconds = field(intern!(py, "hour"))? * 3600
        + field(intern!(py, "minute"))? * 60
        + field(intern!(py, "second"))?;
    Ok(seconds * 1_000_000_000 + field(intern!(py, "microsecond"))? * 1_000)
}

/// The value of a GEOMETRY or a GEOGRAPHY, as their `toInternal`: its srid and WKB.
fn spatial_value(srid: i32, wkb: &[u8]) -> Value {
    Value::Struct(
        spark_connect::row::Row::new(
            vec!["srid".to_string(), "wkb".to_string()],
            vec![Value::Integer(srid), Value::Binary(wkb.to_vec())],
        )
        .into(),
    )
}

/// The user-defined type of a column whose class is `python_class`, to serialize `obj`: that of
/// the value, as PySpark infers a UDT by `__UDT__`, when it is of the class; else an instance
/// of the class, which the type sends cloudpickled for a class of `__main__`.
fn udt_of<'py>(
    obj: &Bound<'py, PyAny>,
    python_class: Option<&str>,
    serialized_python_class: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    let name_of = |class: &Bound<'py, PyAny>| -> PyResult<String> {
        let module: String = class.getattr(intern!(py, "__module__"))?.extract()?;
        let name: String = class.getattr(intern!(py, "__name__"))?.extract()?;
        Ok(format!("{module}.{name}"))
    };
    if let Ok(udt) = obj.getattr(intern!(py, "__UDT__")) {
        let class = udt.get_type();
        let module_name: String = udt.call_method0(intern!(py, "module"))?.extract()?;
        let name: String = class.getattr(intern!(py, "__name__"))?.extract()?;
        let matches = python_class.is_none_or(|python_class| {
            python_class == format!("{module_name}.{name}")
                || name_of(class.as_any()).is_ok_and(|n| n == python_class)
        });
        if matches {
            return Ok(udt);
        }
    }
    let class = match (serialized_python_class, python_class) {
        (Some(serialized), _) => {
            let pickled = py
                .import(intern!(py, "base64"))?
                .call_method1(intern!(py, "b64decode"), (serialized,))?;
            py.import(intern!(py, "pyspark.serializers"))?
                .getattr(intern!(py, "CloudPickleSerializer"))?
                .call0()?
                .call_method1(intern!(py, "loads"), (pickled,))?
        }
        (None, Some(python_class)) => {
            let (module, name) = python_class.rsplit_once('.').ok_or_else(|| {
                PyValueError::new_err(format!("invalid UDT class {python_class}"))
            })?;
            py.import(module)?.getattr(name)?
        }
        (None, None) => {
            return Err(PyValueError::new_err(
                "a user-defined type without its Python class cannot serialize values",
            ))
        }
    };
    class.call0()
}

/// A value of a day-time interval column, as PyArrow converts one to `duration[us]`: a
/// timedelta to its microseconds, as `DayTimeIntervalType.toInternal`, and an int, or a
/// float truncated, as microseconds. Another value raises PyArrow's error.
fn duration_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    if obj.is_instance(TIMEDELTA.import(py, "datetime", "timedelta")?.as_any())? {
        return timedelta_micros(obj);
    }
    if obj.is_instance_of::<PyBool>() {
        return Err(pyarrow_error(
            py,
            "ArrowTypeError",
            "Expected integer, got bool".into(),
        )?);
    }
    if numpy_scalar(obj)?.is_some() {
        let arrow_type = py.import(intern!(py, "pyarrow"))?.call_method1(
            intern!(py, "from_numpy_dtype"),
            (obj.getattr(intern!(py, "dtype"))?,),
        )?;
        return Err(pyarrow_error(
            py,
            "ArrowNotImplementedError",
            format!("Expected np.timedelta64 but got: {}", arrow_type.str()?),
        )?);
    }
    let too_large = || PyOverflowError::new_err("Python int too large to convert to C long");
    if obj.is_instance_of::<PyInt>() {
        return obj.extract::<i64>().map_err(|_| too_large());
    }
    if let Ok(value) = obj.cast::<PyFloat>() {
        let value = value.value();
        if value.is_nan() {
            return Err(PyValueError::new_err("cannot convert float NaN to integer"));
        }
        if value.is_infinite() {
            return Err(PyOverflowError::new_err(
                "cannot convert float infinity to integer",
            ));
        }
        let truncated = value.trunc();
        if !(-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&truncated) {
            return Err(too_large());
        }
        return Ok(truncated as i64);
    }
    Err(pyarrow_error(
        py,
        "ArrowTypeError",
        format!(
            "object of type {} cannot be converted to int",
            obj.get_type().repr()?
        ),
    )?)
}

/// The microseconds of a `datetime.timedelta`, as `DayTimeIntervalType.toInternal`:/// The microseconds of a `datetime.timedelta`, as `DayTimeIntervalType.toInternal`:
/// `((days * 86400) + seconds) * 1_000_000 + microseconds`.
pub(crate) fn timedelta_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    let days: i64 = obj.getattr(intern!(py, "days"))?.extract()?;
    let seconds: i64 = obj.getattr(intern!(py, "seconds"))?.extract()?;
    let micros: i64 = obj.getattr(intern!(py, "microseconds"))?.extract()?;
    days.checked_mul(86_400)
        .and_then(|s| s.checked_add(seconds))
        .and_then(|s| s.checked_mul(1_000_000))
        .and_then(|us| us.checked_add(micros))
        .ok_or_else(|| PyOverflowError::new_err("Python int too large to convert to C long"))
}

/// An integral value, as PyArrow converts one: an int in range, or a float truncated.
fn convert_integral(obj: &Bound<'_, PyAny>, dt: &DataType) -> PyResult<Value> {
    let (bits, arrow_type) = match dt {
        DataType::Byte => (8, "int8"),
        DataType::Short => (16, "int16"),
        DataType::Integer => (32, "int32"),
        _ => (64, "int64"),
    };
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err("Expected integer, got bool"));
    }
    // A float, also of NumPy, is truncated, as PyArrow does.
    let truncated = |obj: &Bound<'_, PyAny>| -> PyResult<i64> {
        let value: f64 = obj.extract()?;
        // The range of i64 is [-2^63, 2^63), both bounds exact as f64.
        if !(i64::MIN as f64..i64::MAX as f64).contains(&value.trunc()) {
            return Err(arrow_invalid(obj, arrow_type)?);
        }
        Ok(value.trunc() as i64)
    };
    let value: i64 = if obj.is_instance_of::<PyInt>() {
        obj.extract()?
    } else if obj.is_instance_of::<PyFloat>() {
        truncated(obj)?
    } else {
        match numpy_scalar(obj)? {
            Some(NumpyScalar::Integer) => obj.extract()?,
            Some(NumpyScalar::Floating) => truncated(obj)?,
            Some(NumpyScalar::Bool) => i64::from(obj.is_truthy()?),
            None => return Err(arrow_invalid(obj, arrow_type)?),
        }
    };
    if bits < 64 && !(-(1i64 << (bits - 1))..(1i64 << (bits - 1))).contains(&value) {
        return Err(PyValueError::new_err(format!(
            "Value {value} too large to fit in C integer type"
        )));
    }
    Ok(match dt {
        DataType::Byte => Value::Byte(value as i8),
        DataType::Short => Value::Short(value as i16),
        DataType::Integer => Value::Integer(value as i32),
        _ => Value::Long(value),
    })
}

/// The values of `fields` from a mapping, by their names, as `mapping.get(name)`.
fn values_by_name<'py>(
    mapping: &Bound<'py, PyAny>,
    fields: &[StructField],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = mapping.py();
    if let Ok(dict) = mapping.cast::<PyDict>() {
        return fields
            .iter()
            .map(|f| {
                Ok(dict
                    .get_item(&f.name)?
                    .unwrap_or_else(|| py.None().into_bound(py)))
            })
            .collect();
    }
    fields
        .iter()
        .map(|f| mapping.call_method1(intern!(py, "get"), (&f.name,)))
        .collect()
}

/// The values of a tuple or a Row, which must be as many as `fields`.
fn values_by_position<'py>(
    py: Python<'py>,
    values: Vec<Bound<'py, PyAny>>,
    fields: &[StructField],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if values.len() == fields.len() {
        Ok(values)
    } else {
        Err(axis_length_mismatch(py, fields.len(), values.len()))
    }
}

/// A struct value, as PySpark converts one: a tuple or a Row by position, a dict by name,
/// and another object by the names of its `__dict__`.
fn convert_struct(obj: &Bound<'_, PyAny>, fields: &[StructField]) -> PyResult<Value> {
    let py = obj.py();
    let values = if let Ok(row) = obj.cast::<PyRow>() {
        let (_, values) = row.borrow().parts(py)?;
        values_by_position(py, values.iter().collect(), fields)?
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        values_by_position(py, tuple.iter().collect(), fields)?
    } else if obj.is_instance_of::<PyDict>() {
        values_by_name(obj, fields)?
    } else if let Ok(attributes) = obj.getattr(intern!(py, "__dict__")) {
        values_by_name(&attributes, fields)?
    } else {
        // PySpark asserts that another value has a `__dict__`.
        return Err(PyAssertionError::new_err(format!(
            "{} {}",
            obj.get_type().str()?,
            obj.str()?
        )));
    };
    struct_of(&values, fields)
}

fn struct_of(values: &[Bound<'_, PyAny>], fields: &[StructField]) -> PyResult<Value> {
    let names: std::sync::Arc<[String]> = fields.iter().map(|f| f.name.clone()).collect();
    let values = values
        .iter()
        .zip(fields)
        .map(|(value, f)| convert(value, &f.data_type, f.nullable))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Value::Struct(StructValue::new(names, values)))
}

/// `dict(sorted(row.items()))`, as PySpark sorts dict rows to respect their inferred schema.
pub(crate) fn sorted_dict<'py>(row: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = row.py();
    let sorted = SORTED.import(py, "builtins", "sorted")?;
    let items = sorted.call1((row.call_method0(intern!(py, "items"))?,))?;
    Ok(py.get_type::<PyDict>().call1((items,))?.into_any())
}

/// The field of the single column that data of an atomic schema has, as PySpark names it.
pub(crate) fn value_field(data_type: DataType) -> StructField {
    nullable_field("value".to_string(), data_type)
}

/// The values of a row in the order of `fields`, as PySpark's `to_row` takes them: nulls for
/// None, a tuple or a Row by position, a dict by name, another object by its `__dict__`, and
/// another value, such as a list, by position.
pub(crate) fn row_values<'py>(
    item: &Bound<'py, PyAny>,
    fields: &[StructField],
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = item.py();
    if item.is_none() {
        return Ok(vec![py.None().into_bound(py); fields.len()]);
    }
    if let Ok(row) = item.cast::<PyRow>() {
        let (_, values) = row.borrow().parts(py)?;
        return values_by_position(py, values.iter().collect(), fields);
    }
    if let Ok(tuple) = item.cast::<PyTuple>() {
        return values_by_position(py, tuple.iter().collect(), fields);
    }
    if item.is_instance_of::<PyDict>() {
        return values_by_name(item, fields);
    }
    if item.is_instance_of::<PyVariantVal>() {
        return Err(pyspark_message(
            py,
            "PySparkValueError",
            "Rows cannot be of type VariantVal",
        ));
    }
    if item.hasattr(intern!(py, "__dict__"))? {
        return values_by_name(&item.getattr(intern!(py, "__dict__"))?, fields);
    }
    let length = item.len()?;
    if length != fields.len() {
        return Err(axis_length_mismatch(py, fields.len(), length));
    }
    item.try_iter()?.collect()
}

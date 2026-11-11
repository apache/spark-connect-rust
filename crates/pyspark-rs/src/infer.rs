//! Schema inference and conversion of local data for `createDataFrame`, as PySpark 4.3.0
//! does them: `_infer_schema`, `_infer_type`, `_merge_type` and `_has_nulltype` of
//! `pyspark.sql.types`, and `LocalDataToArrowConversion` of `pyspark.sql.conversion`, with
//! which Spark Connect converts local data. The checks run in PySpark's order and raise its
//! errors, so that the same data gives the same schema, values and errors.

use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{
    PyAssertionError, PyNotImplementedError, PyOverflowError, PyTypeError, PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    IntoPyDict, PyBool, PyByteArray, PyBytes, PyDict, PyFloat, PyInt, PyList, PyMemoryView,
    PyString, PyTuple, PyType,
};
use spark_connect::conf::RuntimeConf;
use spark_connect::row::{StructValue, Value};
use spark_connect::types::{DataType, GeographyType, GeometryType, StructField};

use crate::column::raise_pyspark;
use crate::errors::ResultExt;
use crate::values::{geography_class, geometry_class, variant_val_class};

static DECIMAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static C_DECIMAL: PyOnceLock<Option<Py<PyType>>> = PyOnceLock::new();
static DATE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static DATETIME: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static TIME: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static TIMEDELTA: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static ARRAY: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static SORTED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Cache the accelerator class, including its absence, independently of the public alias.
pub(crate) fn c_decimal_type(py: Python<'_>) -> PyResult<Option<&Bound<'_, PyType>>> {
    let decimal = C_DECIMAL.get_or_try_init(py, || match py.import("_decimal") {
        Ok(module) => Ok(Some(
            module.getattr("Decimal")?.cast_into::<PyType>()?.unbind(),
        )),
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyImportError>(py) => Ok(None),
        Err(error) => Err(error),
    })?;
    Ok(decimal.as_ref().map(|decimal| decimal.bind(py)))
}

/// The effective Decimal class from [`decimal_classes`]: C `_decimal.Decimal` when
/// available, otherwise the cached public alias.
fn decimal_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    Ok(decimal_classes(py)?.1)
}

/// Resolve the accelerator once and return it with the effective conversion class.
pub(crate) fn decimal_classes(
    py: Python<'_>,
) -> PyResult<(Option<&Bound<'_, PyType>>, &Bound<'_, PyType>)> {
    let c_decimal = c_decimal_type(py)?;
    let decimal = match c_decimal {
        Some(decimal) => decimal,
        None => DECIMAL.import(py, "decimal", "Decimal")?,
    };
    Ok((c_decimal, decimal))
}

/// Whether an object has the exact accelerator Decimal type. The shortcuts assume
/// this type is immutable, as on CPython; subclasses and pure-Python values keep hooks.
pub(crate) fn is_exact_c_decimal(
    obj: &Bound<'_, PyAny>,
    decimal: Option<&Bound<'_, PyType>>,
) -> bool {
    decimal.is_some_and(|decimal| obj.get_type().is(decimal))
}

/// Cached per Decimal type: a quantum, its context and lazily built context kwargs.
struct QuantizeArgs {
    exp: Py<PyAny>,
    context: Py<PyAny>,
    kwargs: Option<Py<PyDict>>,
}

/// Owned arguments for one call, released from the cache borrow before Python runs.
enum QuantizeCallArgs<'py> {
    Positional {
        exp: Bound<'py, PyAny>,
        context: Bound<'py, PyAny>,
    },
    Keyword {
        exp: Bound<'py, PyAny>,
        kwargs: Bound<'py, PyDict>,
    },
}

/// A cache entry's arguments: a ready call, or a keyword dict to build outside the borrow.
enum QuantizeLookup<'py> {
    Call(QuantizeCallArgs<'py>),
    Keywords {
        exp: Bound<'py, PyAny>,
        context: Bound<'py, PyAny>,
    },
}

impl QuantizeArgs {
    /// Clone only the pair needed by the selected call form, without running Python.
    fn call_args<'py>(&self, py: Python<'py>, exact_c: bool) -> QuantizeLookup<'py> {
        if exact_c {
            QuantizeLookup::Call(QuantizeCallArgs::Positional {
                exp: self.exp.bind(py).clone(),
                context: self.context.bind(py).clone(),
            })
        } else if let Some(kwargs) = &self.kwargs {
            QuantizeLookup::Call(QuantizeCallArgs::Keyword {
                exp: self.exp.bind(py).clone(),
                kwargs: kwargs.bind(py).clone(),
            })
        } else {
            QuantizeLookup::Keywords {
                exp: self.exp.bind(py).clone(),
                context: self.context.bind(py).clone(),
            }
        }
    }
}

type Quanta = HashMap<(i32, i32), QuantizeArgs>;

thread_local! {
    /// Quantum, context and cached context kwargs per Decimal type, while a [`DecimalQuanta`]
    /// lives. Both call forms share one context, built once per type rather than per value.
    static DECIMAL_QUANTA: std::cell::RefCell<Option<Quanta>> =
        const { std::cell::RefCell::new(None) };
}

/// While alive, this thread's conversions build each Decimal type's quantum and context
/// once, and its context keyword dict only if needed. Each conversion of rows has its own,
/// as each `createDataFrame` of PySpark has its converters. A type's context is built lazily
/// at its first non-null, non-NaN value, using `decimal.DefaultContext` at that time.
#[must_use = "bind the guard for the duration of the conversion"]
pub(crate) struct DecimalQuanta(Option<Quanta>);

impl DecimalQuanta {
    pub(crate) fn new() -> Self {
        let previous = DECIMAL_QUANTA.with(|quanta| quanta.replace(Some(HashMap::new())));
        DecimalQuanta(previous)
    }
}

impl Drop for DecimalQuanta {
    fn drop(&mut self) {
        DECIMAL_QUANTA.with(|quanta| quanta.replace(self.0.take()));
    }
}

/// Cache the quantum, context and context keyword dict for `decimal(precision, scale)`.
/// Return only the arguments needed by the selected call form, sharing the same context.
/// `decimal` must be the effective class from [`decimal_classes`]: the C class when
/// available, otherwise the cached public alias.
fn decimal_quantum<'py>(
    py: Python<'py>,
    decimal: &Bound<'py, PyType>,
    precision: i32,
    scale: i32,
    exact_c: bool,
) -> PyResult<QuantizeCallArgs<'py>> {
    let cached = DECIMAL_QUANTA.with(|quanta| {
        quanta.borrow().as_ref().and_then(|quanta| {
            quanta
                .get(&(precision, scale))
                .map(|args| args.call_args(py, exact_c))
        })
    });
    let cached = match cached {
        Some(cached) => cached,
        None => {
            let module = py.import(intern!(py, "decimal"))?;
            let exp = decimal.call1((format!("1E-{scale}"),))?;
            // Read Context at construction time, so each conversion observes the current factory.
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
            let args = QuantizeArgs {
                exp: exp.unbind(),
                context: context.unbind(),
                kwargs: None,
            };
            let cached = args.call_args(py, exact_c);
            let discarded = DECIMAL_QUANTA.with(|quanta| match quanta.borrow_mut().as_mut() {
                Some(quanta) => quanta.insert((precision, scale), args),
                None => Some(args),
            });
            // Release displaced or uncached Python objects only after the borrow ends.
            drop(discarded);
            cached
        }
    };
    match cached {
        QuantizeLookup::Call(args) => Ok(args),
        QuantizeLookup::Keywords { exp, context } => {
            // Dict allocation may run GC callbacks: release the cache borrow first, and
            // share the context already used by positional calls of this Decimal type.
            let kwargs = [("context", context)].into_py_dict(py)?;
            DECIMAL_QUANTA.with(|quanta| {
                if let Some(args) = quanta
                    .borrow_mut()
                    .as_mut()
                    .and_then(|quanta| quanta.get_mut(&(precision, scale)))
                {
                    if args.kwargs.is_none() {
                        args.kwargs = Some(kwargs.clone().unbind());
                    }
                }
            });
            Ok(QuantizeCallArgs::Keyword { exp, kwargs })
        }
    }
}

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
    if ty.is(decimal_type(py)?) {
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
    // As PySpark's `_infer_type`, the type of a value is `GeometryType(srid)` or
    // `GeographyType(srid)`, which raises `ST_INVALID_SRID_VALUE` for an SRID that the type does
    // not take (the explicit schema is not checked here, the server checks the values of it).
    if obj.get_type().is(geometry_class(py)?) {
        let srid = obj.call_method0(intern!(py, "getSrid"))?;
        let srid = crate::types::spatial_type_srid(&srid, false)?;
        return Ok(DataType::Geometry { srid });
    }
    if obj.get_type().is(geography_class(py)?) {
        let srid = obj.call_method0(intern!(py, "getSrid"))?;
        let srid = crate::types::spatial_type_srid(&srid, true)?;
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
    if obj.is_instance(variant_val_class(py)?)? {
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

/// An error of PyArrow, of `class` with `message`, as its conversions raise it: `ArrowInvalid`,
/// a `ValueError`, `ArrowTypeError`, a `TypeError`, or `ArrowNotImplementedError`, a
/// `NotImplementedError`. Without PyArrow, the built-in class that it derives from.
fn pyarrow_error(py: Python<'_>, class: &str, message: String) -> PyResult<PyErr> {
    match py
        .import(intern!(py, "pyarrow"))
        .and_then(|pyarrow| pyarrow.getattr(class))
    {
        Ok(class) => Ok(PyErr::from_value(class.call1((message,))?)),
        Err(_) => Ok(match class {
            "ArrowTypeError" => PyTypeError::new_err(message),
            "ArrowNotImplementedError" => PyNotImplementedError::new_err(message),
            _ => PyValueError::new_err(message),
        }),
    }
}

/// The `ArrowInvalid` of a value that PyArrow cannot convert to the type it is named by.
fn arrow_invalid(obj: &Bound<'_, PyAny>, arrow_type: &str) -> PyResult<PyErr> {
    pyarrow_error(
        obj.py(),
        "ArrowInvalid",
        format!(
            "Could not convert {} with type {}: tried to convert to {arrow_type}",
            obj.repr()?,
            arrow_type_name(obj)?
        ),
    )
}

/// `obj` as an int, as PyArrow's `PyObjectToPyInt` takes it: by `__index__`, else by the
/// `__int__` of its type, which may raise, such as for a NaN; else the `ArrowTypeError` "object
/// of type <class 'T'> cannot be converted to int". A float, a `Decimal` and a NumPy number or
/// bool are taken by their `__int__`.
fn py_int<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    // SAFETY: PyNumber_Index returns a new reference, or NULL with an exception set.
    if let Ok(index) =
        unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyNumber_Index(obj.as_ptr())) }
    {
        return Ok(index);
    }
    let ty = obj.get_type();
    if ty.hasattr(intern!(py, "__int__"))? {
        return ty.getattr(intern!(py, "__int__"))?.call1((obj,));
    }
    Err(pyarrow_error(
        py,
        "ArrowTypeError",
        format!("object of type {} cannot be converted to int", ty.repr()?),
    )?)
}

/// `obj` as an integer of `bits` bits, as PyArrow's `CIntFromPython` takes it: not a bool (an
/// `ArrowTypeError`), an int, else by `py_int`. A value out of the range of the bits raises an
/// `ArrowInvalid` of `overflow`, else "Value X too large to fit in C integer type", and one out
/// of the range of a C long Python's `OverflowError`.
fn c_int_from_python(obj: &Bound<'_, PyAny>, bits: u32, overflow: Option<&str>) -> PyResult<i64> {
    let py = obj.py();
    if obj.is_instance_of::<PyBool>() {
        return Err(pyarrow_error(
            py,
            "ArrowTypeError",
            "Expected integer, got bool".into(),
        )?);
    }
    let int = if obj.is_instance_of::<PyInt>() {
        obj.clone()
    } else {
        py_int(obj)?
    };
    let value: i64 = int.extract()?;
    if bits < 64 && !(-(1i64 << (bits - 1))..(1i64 << (bits - 1))).contains(&value) {
        let message = match overflow {
            Some(message) => message.to_string(),
            None => format!("Value {} too large to fit in C integer type", int.str()?),
        };
        return Err(pyarrow_error(py, "ArrowInvalid", message)?);
    }
    Ok(value)
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

/// Whether `obj` is a NumPy scalar of any kind. NumPy is not imported when it is not loaded.
fn numpy_generic(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    let modules = py
        .import(intern!(py, "sys"))?
        .getattr(intern!(py, "modules"))?;
    match modules.cast::<PyDict>()?.get_item(intern!(py, "numpy"))? {
        Some(numpy) if !numpy.is_none() => obj.is_instance(&numpy.getattr(intern!(py, "generic"))?),
        _ => Ok(false),
    }
}

/// An integer as a floating-point value, which PyArrow takes only when it is exact: within
/// 2^53 for a double and 2^24 for a float.
fn exact_float(obj: &Bound<'_, PyAny>, dt: &DataType) -> PyResult<f64> {
    let (bits, precision) = match dt {
        DataType::Float => (24, "single"),
        _ => (53, "double"),
    };
    let out_of_range = || -> PyResult<PyErr> {
        pyarrow_error(
            obj.py(),
            "ArrowInvalid",
            format!(
                "Integer value {} is outside of the range exactly representable by a IEEE 754 \
                 {precision} precision value",
                obj.str()?
            ),
        )
    };
    let value: i64 = match obj.extract() {
        Ok(value) => value,
        Err(_) => {
            return Err(pyarrow_error(
                obj.py(),
                "ArrowInvalid",
                "PyLong is too large to fit int64".into(),
            )?)
        }
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

/// The microseconds of the wall clock of a datetime, naive or aware, as PySpark's
/// `TimestampNTZType.toInternal` takes them: `calendar.timegm` of its own time tuple, whatever
/// its zone. The fields of a pandas Timestamp are read from it, as PySpark's, so its nanoseconds
/// are dropped silently. The naive values are those of a TIMESTAMP_NTZ, as PyArrow takes them.
pub(crate) fn wall_clock_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    let seconds: i64 = py
        .import(intern!(py, "calendar"))?
        .call_method1(
            intern!(py, "timegm"),
            (obj.call_method0(intern!(py, "timetuple"))?,),
        )?
        .extract()?;
    let micros: i64 = obj.getattr(intern!(py, "microsecond"))?.extract()?;
    Ok(seconds * 1_000_000 + micros)
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
            if let Ok(bytes) = obj.cast_exact::<PyBytes>() {
                Ok(Value::Binary(bytes.as_bytes().to_vec()))
            } else if let Ok(bytes) = obj.cast_exact::<PyByteArray>() {
                Ok(Value::Binary(bytes.to_vec()))
            } else if obj.is_instance_of::<PyBytes>() || obj.is_instance_of::<PyByteArray>() {
                // bytes(obj) honors a __bytes__ of the subclass, which its buffer does not.
                let bytes = py.get_type::<PyBytes>().call1((obj,))?;
                Ok(Value::Binary(bytes.cast::<PyBytes>()?.as_bytes().to_vec()))
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
            let (c_decimal, decimal) = decimal_classes(py)?;
            if !obj.is_instance(decimal.as_any())? {
                return Err(assertion());
            }
            if obj.call_method0(intern!(py, "is_nan"))?.is_truthy()? {
                if !nullable {
                    return Err(must_not_be_none(py, dt)?);
                }
                return Ok(Value::Null);
            }
            // Exact C Decimals accept positional context; other values keep PySpark's
            // keyword call so subclass overrides and the pure-Python fallback retain it.
            let exact_c = is_exact_c_decimal(obj, c_decimal);
            let quantized = match decimal_quantum(py, decimal, *precision, *scale, exact_c)? {
                QuantizeCallArgs::Positional { exp, context } => {
                    obj.call_method1(intern!(py, "quantize"), (exp, py.None(), context))?
                }
                QuantizeCallArgs::Keyword { exp, kwargs } => {
                    obj.call_method(intern!(py, "quantize"), (exp,), Some(&kwargs))?
                }
            };
            Ok(Value::Decimal {
                value: quantized.str()?.extract()?,
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
        // As PySpark, which has no converter of these types: the value goes as it is to
        // `to_arrow_schema`, which raises for the type (`unsupported_arrow_type`).
        DataType::Char { .. }
        | DataType::Varchar { .. }
        | DataType::YearMonthInterval { .. }
        | DataType::CalendarInterval => Ok(Value::Null),
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
                            "float32"
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
        // As PyArrow's `time64[ns]`, to which PySpark passes the value as it is: a time, else
        // the nanoseconds of the value as `CIntFromPython` takes it.
        DataType::Time { .. } => {
            if obj.is_instance(TIME.import(py, "datetime", "time")?.as_any())? {
                Ok(Value::Time(time_nanos(obj)?))
            } else {
                Ok(Value::Time(c_int_from_python(
                    obj,
                    64,
                    Some("Integer too large for int64"),
                )?))
            }
        }
        // As PySpark's converters of these types, their `toInternal`: the `value` and
        // `metadata` attributes of a VariantVal, and the `srid` and `wkb` of a Geometry or a
        // Geography, which PyArrow converts as the fields of the structs of their Arrow types.
        DataType::Variant => {
            if obj.is_instance(variant_val_class(py)?)? {
                let value = obj.getattr(intern!(py, "value"))?;
                let metadata = obj.getattr(intern!(py, "metadata"))?;
                Ok(Value::Variant {
                    value: arrow_binary(&value)?,
                    metadata: arrow_binary(&metadata)?,
                })
            } else {
                Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "MALFORMED_VARIANT",
                    &[],
                ))
            }
        }
        DataType::Geometry { .. } => {
            if obj.is_instance(geometry_class(py)?)? {
                spatial_value(obj)
            } else {
                Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "MALFORMED_GEOMETRY",
                    &[],
                ))
            }
        }
        DataType::Geography { .. } => {
            if obj.is_instance(geography_class(py)?)? {
                spatial_value(obj)
            } else {
                Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "MALFORMED_GEOGRAPHY",
                    &[],
                ))
            }
        }
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
/// date, or the date of a datetime, else the days of the value as `CIntFromPython` takes it, in
/// the range of an int32, else the `ArrowInvalid` "Integer too large for date32".
fn date_days(obj: &Bound<'_, PyAny>) -> PyResult<i32> {
    let py = obj.py();
    if obj.is_instance(DATE.import(py, "datetime", "date")?.as_any())? {
        let ordinal: i64 = obj.call_method0(intern!(py, "toordinal"))?.extract()?;
        return Ok((ordinal - 719_163) as i32);
    }
    Ok(c_int_from_python(obj, 32, Some("Integer too large for date32"))? as i32)
}

/// Whether a field of `fields`, or a type in one, is a null type that is not nullable. PyArrow
/// rejects such a field when it builds it, `pa.field(name, pa.null(), nullable=False)`, which
/// PySpark's `to_arrow_schema` does. A map key is never nullable.
pub(crate) fn has_non_nullable_null_type(fields: &[StructField]) -> bool {
    fn in_type(dt: &DataType) -> bool {
        match dt {
            DataType::Struct { fields } => has_non_nullable_null_type(fields),
            DataType::Array {
                element_type,
                contains_null,
            } => {
                (!contains_null && matches!(**element_type, DataType::Null))
                    || in_type(element_type)
            }
            DataType::Map {
                key_type,
                value_type,
                value_contains_null,
            } => {
                matches!(**key_type, DataType::Null)
                    || (!value_contains_null && matches!(**value_type, DataType::Null))
                    || in_type(key_type)
                    || in_type(value_type)
            }
            _ => false,
        }
    }
    fields
        .iter()
        .any(|f| (!f.nullable && matches!(f.data_type, DataType::Null)) || in_type(&f.data_type))
}

/// The `UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION` of the first char, varchar, year-month or
/// calendar interval type of `dt`, depth first, which PySpark's `to_arrow_schema` raises when it
/// converts the schema of local rows to Arrow: it has no Arrow type of these types.
pub(crate) fn unsupported_arrow_type(py: Python<'_>, dt: &DataType) -> PyResult<Option<PyErr>> {
    fn first_unsupported(dt: &DataType) -> Option<&DataType> {
        match dt {
            DataType::Char { .. }
            | DataType::Varchar { .. }
            | DataType::YearMonthInterval { .. }
            | DataType::CalendarInterval => Some(dt),
            DataType::Struct { fields } => {
                fields.iter().find_map(|f| first_unsupported(&f.data_type))
            }
            DataType::Array { element_type, .. } => first_unsupported(element_type),
            DataType::Map {
                key_type,
                value_type,
                ..
            } => first_unsupported(key_type).or_else(|| first_unsupported(value_type)),
            _ => None,
        }
    }
    Ok(match first_unsupported(dt) {
        Some(unsupported) => Some(raise_pyspark(
            py,
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION",
            &[("data_type", &data_type_str(py, unsupported)?)],
        )),
        None => None,
    })
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

/// The value of a Geometry or a Geography `obj`, as their `toInternal`, the dict of its `srid`
/// and `wkb` attributes, which PyArrow converts to the int32 and the binary of their struct.
fn spatial_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = obj.py();
    let srid = obj.getattr(intern!(py, "srid"))?;
    let wkb = obj.getattr(intern!(py, "wkb"))?;
    Ok(Value::Struct(
        spark_connect::row::Row::new(
            vec!["srid".to_string(), "wkb".to_string()],
            vec![
                convert_integral(&srid, &DataType::Integer)?,
                Value::Binary(arrow_binary(&wkb)?),
            ],
        )
        .into(),
    ))
}

/// The bytes of `obj` as PyArrow converts a value to binary: of a bytes, a bytearray or a
/// memoryview, or the UTF-8 of a str; else the `ArrowTypeError` "Expected bytes, got a 'T'
/// object". So are the attributes of a VariantVal, a Geometry and a Geography, which PySpark
/// stores as they are given, converted. A None, which PyArrow takes as a null that the server
/// then rejects, is no value here either, as these structs have no null fields here.
fn arrow_binary(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let py = obj.py();
    if let Ok(bytes) = obj.cast::<PyBytes>() {
        Ok(bytes.as_bytes().to_vec())
    } else if let Ok(text) = obj.cast::<PyString>() {
        Ok(text.to_cow()?.as_bytes().to_vec())
    } else if let Ok(bytes) = obj.cast::<PyByteArray>() {
        Ok(bytes.to_vec())
    } else if obj.is_instance_of::<PyMemoryView>() {
        // Its bytes in C order, as PyArrow takes them, also of a view that is not contiguous.
        Ok(obj
            .call_method0(intern!(py, "tobytes"))?
            .cast_into::<PyBytes>()?
            .as_bytes()
            .to_vec())
    } else {
        Err(pyarrow_error(
            py,
            "ArrowTypeError",
            format!("Expected bytes, got a '{}' object", arrow_type_name(obj)?),
        )?)
    }
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

/// The microseconds of a value of a day-time interval column, as PyArrow converts it to
/// `duration[us]`: a timedelta; a NumPy scalar of `timedelta64[us]`, as another NumPy scalar
/// raises `ArrowNotImplementedError`; else the value as `CIntFromPython` takes it.
fn duration_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    if obj.is_instance(TIMEDELTA.import(py, "datetime", "timedelta")?.as_any())? {
        return timedelta_micros(obj);
    }
    if numpy_generic(obj)? {
        let arrow_type = py.import(intern!(py, "pyarrow"))?.call_method1(
            intern!(py, "from_numpy_dtype"),
            (obj.getattr(intern!(py, "dtype"))?,),
        )?;
        let arrow_type = arrow_type.str()?.to_cow()?.into_owned();
        if arrow_type == "duration[us]" {
            // The microseconds that the scalar holds, as PyArrow reads them.
            return obj
                .call_method1(intern!(py, "astype"), (intern!(py, "int64"),))?
                .extract();
        }
        return Err(pyarrow_error(
            py,
            "ArrowNotImplementedError",
            format!("Expected np.timedelta64 but got: {arrow_type}"),
        )?);
    }
    c_int_from_python(obj, 64, None)
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

/// An integral value, as PyArrow converts one: the value as `CIntFromPython` takes it, such as an
/// int in range, a float truncated or a `Decimal`. The error of a value that is not an int, or a
/// NumPy integer, is the `ArrowInvalid` "Could not convert".
fn convert_integral(obj: &Bound<'_, PyAny>, dt: &DataType) -> PyResult<Value> {
    let (bits, arrow_type) = match dt {
        DataType::Byte => (8, "int8"),
        DataType::Short => (16, "int16"),
        DataType::Integer => (32, "int32"),
        _ => (64, "int64"),
    };
    let value = match c_int_from_python(obj, bits, None) {
        Ok(value) => value,
        Err(err) => {
            let int_scalar =
                obj.is_instance_of::<PyInt>() || numpy_scalar(obj)? == Some(NumpyScalar::Integer);
            return Err(if int_scalar {
                err
            } else {
                arrow_invalid(obj, arrow_type)?
            });
        }
    };
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
    let values = if let Ok(tuple) = obj.cast::<PyTuple>() {
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

/// What PySpark's `to_row` raises for a row that is a VariantVal.
pub(crate) fn variant_row(py: Python<'_>) -> PyErr {
    pyspark_message(py, "PySparkValueError", "Rows cannot be of type VariantVal")
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
    if let Ok(tuple) = item.cast::<PyTuple>() {
        return values_by_position(py, tuple.iter().collect(), fields);
    }
    if item.is_instance_of::<PyDict>() {
        return values_by_name(item, fields);
    }
    if item.is_instance(variant_val_class(py)?)? {
        return Err(variant_row(py));
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

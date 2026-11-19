//! Schema inference and conversion of local data for `createDataFrame`, as PySpark 4.3.0
//! does them: `_infer_schema`, `_infer_type`, `_merge_type` and `_has_nulltype` of
//! `pyspark.sql.types`, and `LocalDataToArrowConversion` of `pyspark.sql.conversion`, with
//! which Spark Connect converts local data. The checks run in PySpark's order and raise its
//! errors, so that the same data gives the same schema, values and errors.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{
    PyAssertionError, PyIndexError, PyNotImplementedError, PyOverflowError, PyTypeError,
    PyValueError,
};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    IntoPyDict, PyBool, PyByteArray, PyBytes, PyDict, PyDictValues, PyFloat, PyInt, PyList,
    PyMemoryView, PySet, PyString, PyTuple, PyType,
};
use spark_connect::row::{StructValue, Value};
use spark_connect::types::{DataType, GeographyType, GeometryType, StructField};

use crate::column::raise_pyspark;
use crate::interrupt::ConversionBudget;
use crate::values::{geography_class, geometry_class, variant_val_class};

static DECIMAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static C_DECIMAL: PyOnceLock<Option<Py<PyType>>> = PyOnceLock::new();
static DATE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static DATETIME: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static UTC_EPOCH: PyOnceLock<UtcEpoch> = PyOnceLock::new();
static TIME: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static TIMEDELTA: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static ARRAY: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static SORTED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static SYS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

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

/// The configurations of inference, of the settings that PySpark's `createDataFrame` reads with
/// the others of its conversions, in one request, before it converts any data.
pub(crate) struct InferenceConfs {
    confs: Confs,
}

impl InferenceConfs {
    pub(crate) fn of_options(options: &spark_connect::ArrowInputOptions) -> Self {
        InferenceConfs {
            confs: Confs {
                dict_as_struct: options.infer_dict_as_struct(),
                array_from_first_element: options.infer_array_from_first_element(),
                map_from_first_pair: options.infer_map_from_first_pair(),
                prefer_timestamp_ntz: options.timestamp_type() == Some("TIMESTAMP_NTZ"),
            },
        }
    }

    fn get(&self, _py: Python<'_>) -> PyResult<&Confs> {
        Ok(&self.confs)
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
/// shift where PySpark passes them on. The budget is that of the conversion, of a step for each
/// value that inference types.
#[derive(Clone, Copy)]
pub(crate) struct Options<'c> {
    confs: &'c InferenceConfs,
    budget: &'c ConversionBudget,
    map_from_first_pair: Source,
    prefer_timestamp_ntz: Source,
}

impl<'c> Options<'c> {
    pub(crate) fn new(confs: &'c InferenceConfs, budget: &'c ConversionBudget) -> Self {
        Options {
            confs,
            budget,
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

/// A type that inference gives, with the type objects of the user-defined types in it.
pub(crate) type Inferred<'py> = (DataType, TypeObjects<'py>);

/// An inferred type that holds no user-defined type.
fn inferred<'py>(data_type: DataType) -> Inferred<'py> {
    (data_type, TypeObjects::default())
}

/// The type of `obj`, as PySpark's `_infer_type`, with the `__UDT__` of a value of a user-defined
/// type, which PySpark's inferred schema holds.
pub(crate) fn infer_type<'py>(obj: &Bound<'py, PyAny>, o: Options<'_>) -> PyResult<Inferred<'py>> {
    let _nesting = Nesting::enter()?;
    let py = obj.py();
    o.budget.step(py)?;
    if obj.is_none() {
        return Ok(inferred(DataType::Null));
    }
    // The types that PySpark maps by `type(obj)`, before it looks for `__UDT__`, which none
    // of them can have: neither these classes nor their instances take new attributes.
    if obj.is_exact_instance_of::<PyBool>() {
        return Ok(inferred(DataType::Boolean));
    }
    if obj.is_exact_instance_of::<PyInt>() {
        return Ok(inferred(DataType::Long));
    }
    if obj.is_exact_instance_of::<PyFloat>() {
        return Ok(inferred(DataType::Double));
    }
    if obj.is_exact_instance_of::<PyString>() {
        return Ok(inferred(string_type()));
    }
    if obj.is_exact_instance_of::<PyBytes>() || obj.is_exact_instance_of::<PyByteArray>() {
        return Ok(inferred(DataType::Binary));
    }
    let ty = obj.get_type();
    if ty.is(decimal_type(py)?) {
        // As PySpark: the precision and scale of a value may differ from row to row.
        return Ok(inferred(DataType::Decimal {
            precision: 38,
            scale: 18,
        }));
    }
    if ty.is(DATETIME.import(py, "datetime", "datetime")?) {
        let naive = obj.getattr(intern!(py, "tzinfo"))?.is_none();
        return Ok(inferred(if naive && o.value(py, o.prefer_timestamp_ntz)? {
            DataType::TimestampNtz
        } else {
            DataType::Timestamp
        }));
    }
    if ty.is(DATE.import(py, "datetime", "date")?) {
        return Ok(inferred(DataType::Date));
    }
    if ty.is(TIME.import(py, "datetime", "time")?) {
        return Ok(inferred(DataType::Time { precision: 6 }));
    }
    if ty.is(TIMEDELTA.import(py, "datetime", "timedelta")?) {
        // `DayTimeIntervalType()`, from days to seconds.
        return Ok(inferred(DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        }));
    }
    // As PySpark's `_infer_type`, the type of a value is `GeometryType(srid)` or
    // `GeographyType(srid)`, which raises `ST_INVALID_SRID_VALUE` for an SRID that the type does
    // not take (the explicit schema is not checked here, the server checks the values of it).
    if obj.get_type().is(geometry_class(py)?) {
        let srid = obj.call_method0(intern!(py, "getSrid"))?;
        let srid = crate::types::spatial_type_srid(&srid, false)?;
        return Ok(inferred(DataType::Geometry { srid }));
    }
    if obj.get_type().is(geography_class(py)?) {
        let srid = obj.call_method0(intern!(py, "getSrid"))?;
        let srid = crate::types::spatial_type_srid(&srid, true)?;
        return Ok(inferred(DataType::Geography { srid }));
    }
    if obj.hasattr(intern!(py, "__UDT__"))? {
        let udt = obj.getattr(intern!(py, "__UDT__"))?;
        let data_type = crate::types::py_to_data_type(&udt)?;
        return Ok((data_type, TypeObjects::of_udt(udt)));
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
            Some(element_type) => Ok(inferred(DataType::Array {
                element_type: Box::new(element_type),
                contains_null: false,
            })),
            None => Err(raise_pyspark(
                py,
                "PySparkTypeError",
                "UNSUPPORTED_DATA_TYPE",
                &[("data_type", &format!("array({typecode})"))],
            )),
        };
    }
    if obj.is_instance(variant_val_class(py)?)? {
        return Ok(inferred(DataType::Variant));
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
fn infer_dict<'py>(dict: &Bound<'py, PyDict>, o: Options<'_>) -> PyResult<Inferred<'py>> {
    let py = dict.py();
    if o.confs.get(py)?.dict_as_struct {
        let mut fields = Vec::new();
        let mut objects = FieldObjects::default();
        for (key, value) in dict.iter() {
            if !key.is_none() && !value.is_none() {
                let (data_type, value_objects) = infer_type(&value, o)?;
                objects.push(fields.len(), value_objects);
                fields.push(field_of(&key, data_type)?);
            }
        }
        let objects = objects.finish(fields.len());
        return Ok((DataType::Struct { fields }, objects));
    }
    if o.value(py, o.map_from_first_pair)? {
        for (key, value) in dict.iter() {
            if !key.is_none() && !value.is_none() {
                return Ok(map_of(infer_type(&key, o)?, infer_type(&value, o)?));
            }
        }
        return Ok(inferred(map_type(DataType::Null, DataType::Null)));
    }
    let (mut key_type, mut value_type) = (inferred(DataType::Null), inferred(DataType::Null));
    for (key, value) in dict.iter() {
        if !key.is_none() {
            key_type = merge_type(py, key_type, infer_type(&key, o)?)?;
        }
        if !value.is_none() {
            value_type = merge_type(py, value_type, infer_type(&value, o)?)?;
        }
    }
    Ok(map_of(key_type, value_type))
}

/// The inferred map type of an inferred key and value type.
fn map_of<'py>(
    (key_type, key): Inferred<'py>,
    (value_type, value): Inferred<'py>,
) -> Inferred<'py> {
    (
        map_type(key_type, value_type),
        TypeObjects::of_map(key, value),
    )
}

/// A list: an array of the merged types of its elements, or of its first element's type.
fn infer_list<'py>(list: &Bound<'py, PyList>, o: Options<'_>) -> PyResult<Inferred<'py>> {
    let py = list.py();
    let elements = o.for_elements();
    let (element_type, element) = match list.len() {
        0 => inferred(DataType::Null),
        // One element has its own type either way, without reading the configuration.
        1 => infer_type(&list.get_item(0)?, elements)?,
        _ if o.confs.get(py)?.array_from_first_element => infer_type(&list.get_item(0)?, elements)?,
        _ => {
            let mut merged = inferred(DataType::Null);
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
    let data_type = DataType::Array {
        element_type: Box::new(element_type),
        contains_null: true,
    };
    Ok((data_type, TypeObjects::of_array(element)))
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
/// another object by the sorted names of its `__dict__`. The values of a tuple or a list are
/// those of the Python protocol, of the `__len__` and `__iter__` of a subclass too.
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
        return match fields {
            // `zip(row.__fields__, tuple(row))`.
            Some(fields) => {
                let values = match row.cast::<PyTuple>() {
                    Ok(tuple) if reads_as_stored(tuple, false)? => tuple.clone(),
                    _ => sequence_tuple(row)?,
                };
                fields
                    .try_iter()?
                    .zip(values.iter())
                    .map(|(name, value)| Ok((name?, value)))
                    .collect()
            }
            None => positional_items(py, row, names),
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

/// The (name, value) pairs of a tuple or a list `row` of no field names, as `_infer_schema`
/// takes `zip(names, row)`: names as many as `len(row)`, `_1`, `_2`, and so on, or `names`,
/// extended so, in place, when they are fewer, and the values of `iter(row)`, up to the last
/// name.
fn positional_items<'py>(
    py: Python<'py>,
    row: &Bound<'py, PyAny>,
    names: Option<&mut Vec<String>>,
) -> PyResult<Items<'py>> {
    let defaults: Vec<String>;
    let names: &[String] = match names {
        None => {
            defaults = (1..=row.len()?).map(|i| format!("_{i}")).collect();
            &defaults
        }
        Some(names) => {
            // `len(row)` twice, as `elif len(names) < len(row): names.extend(... len(row) ...)`.
            if names.len() < row.len()? {
                let (start, width) = (names.len() + 1, row.len()?);
                names.extend((start..=width).map(|i| format!("_{i}")));
            }
            names
        }
    };
    let mut values = row.try_iter()?;
    let mut items = Vec::with_capacity(names.len());
    for name in names {
        match values.next() {
            Some(value) => items.push((PyString::new(py, name).into_any(), value?)),
            None => break,
        }
    }
    Ok(items)
}

/// The struct type of (name, value) pairs. A `TypeError` in inferring a value, as from a
/// type that cannot be inferred or merged, becomes `CANNOT_INFER_TYPE_FOR_FIELD`.
pub(crate) fn infer_fields<'py>(
    py: Python<'py>,
    items: Items<'py>,
    o: Options<'_>,
) -> PyResult<Inferred<'py>> {
    let mut fields = Vec::with_capacity(items.len());
    let mut objects = FieldObjects::default();
    for (name, value) in items {
        let (data_type, value_objects) = match infer_type(&value, o) {
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
        objects.push(fields.len(), value_objects);
        fields.push(field_of(&name, data_type)?);
    }
    let objects = objects.finish(fields.len());
    Ok((DataType::Struct { fields }, objects))
}

/// The struct type of a row, as PySpark's `_infer_schema`.
pub(crate) fn infer_schema<'py>(
    row: &Bound<'py, PyAny>,
    names: Option<&mut Vec<String>>,
    o: Options<'_>,
) -> PyResult<Inferred<'py>> {
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
/// their parts. Other types must be of the same class, and the first one is taken, with its type
/// objects: so a user-defined type is the `__UDT__` of its first value.
pub(crate) fn merge_type<'py>(
    py: Python<'_>,
    a: Inferred<'py>,
    b: Inferred<'py>,
) -> PyResult<Inferred<'py>> {
    use DataType as D;
    let ((a, mut a_objects), (b, mut b_objects)) = (a, b);
    match (&a, &b) {
        (D::Null, _) => return Ok((b, b_objects)),
        (_, D::Null) => return Ok((a, a_objects)),
        (D::Timestamp, D::TimestampNtz) => return Ok((a, a_objects)),
        (D::TimestampNtz, D::Timestamp) => return Ok((b, b_objects)),
        (D::Geometry { srid: x }, D::Geometry { srid: y }) if x != y => {
            return Ok(inferred(D::Geometry {
                srid: GeometryType::MIXED_SRID,
            }));
        }
        (D::Geography { srid: x }, D::Geography { srid: y }) if x != y => {
            return Ok(inferred(D::Geography {
                srid: GeographyType::MIXED_SRID,
            }));
        }
        _ => {}
    }
    if is_atomic(&a) && matches!(b, D::String { .. }) {
        return Ok((b, b_objects));
    }
    if matches!(a, D::String { .. }) && is_atomic(&b) {
        return Ok((a, a_objects));
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
            let mut others: Vec<(String, Inferred<'py>)> = Vec::with_capacity(b.len());
            for (i, f) in b.into_iter().enumerate() {
                let other = (f.data_type, b_objects.take_child(i));
                match index.get(&f.name) {
                    Some(&j) => others[j].1 = other,
                    None => {
                        index.insert(f.name.clone(), others.len());
                        others.push((f.name, other));
                    }
                }
            }
            let mut names: HashSet<String> = HashSet::with_capacity(a.len());
            let mut fields = Vec::with_capacity(a.len() + others.len());
            let mut objects = FieldObjects::default();
            for (i, f) in a.into_iter().enumerate() {
                let other = match index.get(&f.name) {
                    Some(&j) => others[j].1.clone(),
                    None => inferred(D::Null),
                };
                names.insert(f.name.clone());
                let (data_type, merged) =
                    merge_type(py, (f.data_type, a_objects.take_child(i)), other)?;
                objects.push(fields.len(), merged);
                fields.push(nullable_field(f.name, data_type));
            }
            for (name, (data_type, other)) in others {
                if !names.contains(&name) {
                    objects.push(fields.len(), other);
                    fields.push(nullable_field(name, data_type));
                }
            }
            let objects = objects.finish(fields.len());
            (D::Struct { fields }, objects)
        }
        (
            D::Array {
                element_type: x, ..
            },
            D::Array {
                element_type: y, ..
            },
        ) => {
            let (element_type, element) = merge_type(
                py,
                (*x, a_objects.take_child(0)),
                (*y, b_objects.take_child(0)),
            )?;
            let data_type = D::Array {
                element_type: Box::new(element_type),
                contains_null: true,
            };
            (data_type, TypeObjects::of_array(element))
        }
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
        ) => map_of(
            merge_type(
                py,
                (*ka, a_objects.take_child(0)),
                (*kb, b_objects.take_child(0)),
            )?,
            merge_type(
                py,
                (*va, a_objects.take_child(1)),
                (*vb, b_objects.take_child(1)),
            )?,
        ),
        (a, _) => (a, a_objects),
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

/// The `str()` of the type of a place of local rows, which PySpark's converters name in their
/// errors: of the type object of the schema there, as of a subclass, else of the core type `dt`.
fn type_str(py: Python<'_>, dt: &DataType, objects: Option<&TypeObjects<'_>>) -> PyResult<String> {
    match objects.and_then(TypeObjects::object) {
        Some(object) => Ok(object.str()?.to_cow()?.into_owned()),
        None => data_type_str(py, dt),
    }
}

fn must_not_be_none(
    py: Python<'_>,
    dt: &DataType,
    objects: Option<&TypeObjects<'_>>,
) -> PyResult<PyErr> {
    Ok(pyspark_message(
        py,
        "PySparkValueError",
        &format!("input for {} must not be None", type_str(py, dt, objects)?),
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

/// The name of the type of `obj` as PyArrow writes it in its errors, which is the `tp_name` of
/// the type: the bare name of a heap type, a class defined in Python, such as `Counter`; else
/// the name with its module, unless it is a builtin type, such as `int` or `numpy.int64`.
fn arrow_type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let ty = obj.get_type();
    // SAFETY: `ty` is a live type object, and PyType_HasFeature only reads its flags.
    let heap_type = unsafe {
        pyo3::ffi::PyType_HasFeature(ty.as_type_ptr(), pyo3::ffi::Py_TPFLAGS_HEAPTYPE) != 0
    };
    if heap_type {
        return Ok(ty.name()?.to_string());
    }
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
/// `nb_int` slot of its type, which may raise, such as for a NaN; else the `ArrowTypeError`
/// "object of type <class 'T'> cannot be converted to int". A float, a `Decimal` and a NumPy
/// number or bool are taken by their `nb_int`. A NumPy `str_` or `bytes_` has none, although
/// its type has the `__int__` of `numpy.generic`.
fn py_int<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    // SAFETY: PyNumber_Index returns a new reference, or NULL with an exception set.
    if let Ok(index) =
        unsafe { Bound::from_owned_ptr_or_err(py, pyo3::ffi::PyNumber_Index(obj.as_ptr())) }
    {
        return Ok(index);
    }
    let ty = obj.get_type();
    // SAFETY: `ty` is a live type object. PyType_GetSlot gives NULL for a slot that is not set,
    // and, before Python 3.10, which reads the slots of heap types only, also with an exception
    // set for a static type.
    let slot = unsafe { pyo3::ffi::PyType_GetSlot(ty.as_type_ptr(), pyo3::ffi::Py_nb_int) };
    if !slot.is_null() {
        // SAFETY: the nb_int slot is a unaryfunc, which returns a new reference, or NULL with
        // an exception set.
        return unsafe {
            let nb_int = std::mem::transmute::<*mut std::ffi::c_void, pyo3::ffi::unaryfunc>(slot);
            Bound::from_owned_ptr_or_err(py, nb_int(obj.as_ptr()))
        };
    }
    // On Python 3.9, the `__int__` of a static type, which wraps its nb_int, stands for it.
    if PyErr::take(py).is_some() && ty.hasattr(intern!(py, "__int__"))? {
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

/// Immutable arguments shared by local timestamp conversions. The input's local timezone
/// and datetime subclass hooks are still resolved for every value.
struct UtcEpoch {
    timezone: Py<PyAny>,
    datetime: Py<PyAny>,
}

/// Microseconds since the epoch of the instant of a datetime, which PySpark takes with
/// `astimezone(timezone.utc)`: a naive datetime is in the local time zone.
fn instant_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let py = obj.py();
    let epoch = UTC_EPOCH.get_or_try_init(py, || -> PyResult<_> {
        let utc = py
            .import(intern!(py, "datetime"))?
            .getattr(intern!(py, "timezone"))?
            .getattr(intern!(py, "utc"))?;
        let datetime = DATETIME
            .import(py, "datetime", "datetime")?
            .call((1970, 1, 1), Some(&[("tzinfo", &utc)].into_py_dict(py)?))?;
        Ok(UtcEpoch {
            timezone: utc.unbind(),
            datetime: datetime.unbind(),
        })
    })?;
    let aware = obj.call_method1(intern!(py, "astimezone"), (epoch.timezone.bind(py),))?;
    delta_micros(&aware.sub(epoch.datetime.bind(py))?)
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

/// The first error of PyArrow in the conversion of the values of a table. PySpark converts all
/// the values with its converters before PyArrow builds any column, so that an error of a
/// converter is raised before one of PyArrow, wherever the values are.
#[derive(Default)]
pub(crate) struct Late(Option<PyErr>);

impl Late {
    /// The value of `convert`, or a null in its place once its error is kept. Once one is
    /// kept, `convert` is not called: PySpark stops at the first error of PyArrow.
    fn defer(&mut self, convert: impl FnOnce() -> PyResult<Value>) -> Value {
        if self.0.is_some() {
            return Value::Null;
        }
        convert().unwrap_or_else(|err| {
            self.0 = Some(err);
            Value::Null
        })
    }

    /// The error to raise once every value is converted, if any.
    pub(crate) fn finish(self) -> PyResult<()> {
        self.0.map_or(Ok(()), Err)
    }
}

/// The text of `text` as PyArrow encodes it, which fails for a lone surrogate.
fn utf8_text(text: &Bound<'_, PyString>) -> PyResult<Value> {
    Ok(Value::String(text.to_cow()?.into_owned()))
}

/// The bytes of a bytes or a bytearray, as PySpark sends `bytes(value)` of one, in a binary
/// column of createDataFrame and in a binary literal: an exact one is read as it is, and of a
/// subclass, `bytes()` honors its `__bytes__`, which its buffer does not. `None` for another
/// value.
pub(crate) fn bytes_of(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<u8>>> {
    if let Ok(bytes) = obj.cast_exact::<PyBytes>() {
        Ok(Some(bytes.as_bytes().to_vec()))
    } else if let Ok(bytes) = obj.cast_exact::<PyByteArray>() {
        Ok(Some(bytes.to_vec()))
    } else if obj.is_instance_of::<PyBytes>() || obj.is_instance_of::<PyByteArray>() {
        let bytes = obj.py().get_type::<PyBytes>().call1((obj,))?;
        Ok(Some(bytes.cast::<PyBytes>()?.as_bytes().to_vec()))
    } else {
        Ok(None)
    }
}

/// A value of an array whose elements need no converter, which PySpark passes to PyArrow as
/// it is, as PyArrow's list converter reads it:
/// - A NumPy array that is not one-dimensional is the `ArrowInvalid` "Can only convert
///   1-dimensional array values".
/// - A list or a tuple is read as it is stored, also of a subclass that overrides `__iter__`
///   or `__getitem__`.
/// - Another sequence by `PySequence_Check`, such as a range or a NumPy array, is read by
///   `PySequence_Size` and then `PySequence_GetItem` of each index, whose errors PyArrow raises:
///   a `__getitem__` without a `__len__` "has no len()", and a `__len__` longer than the items
///   gives the `IndexError` of the first index past them. A shorter one ends the list there.
/// - A set and a `dict_values` are iterated.
/// - Any other value, a dict too, is "not a sequence": an iterator, a generator, a frozenset,
///   a `dict_keys` or a `dict_items`.
///
/// Its errors, as those of its elements, wait for the converters of the other columns. Its
/// elements need no converter, so they hold no user-defined or spatial type, and raise no error
/// that names their type object: they take no [`TypeObjects`].
fn arrow_list(
    obj: &Bound<'_, PyAny>,
    dt: &DataType,
    element_type: &DataType,
    contains_null: bool,
    late: &mut Late,
    budget: &ConversionBudget,
) -> PyResult<Value> {
    let py = obj.py();
    let mut items = Vec::new();
    // A list and a tuple, the values of most rows, are read first: neither is a NumPy array.
    if let Ok(list) = obj.cast::<PyList>() {
        for item in list.iter() {
            items.push(convert(
                &item,
                element_type,
                None,
                contains_null,
                late,
                budget,
            )?);
        }
        return Ok(Value::List(items));
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        for item in tuple.iter() {
            items.push(convert(
                &item,
                element_type,
                None,
                contains_null,
                late,
                budget,
            )?);
        }
        return Ok(Value::List(items));
    }
    if numpy_array_dimensions(obj)?.is_some_and(|dimensions| dimensions != 1) {
        let message = match arrow_type_text(dt) {
            Some(list) => format!(
                "Can only convert 1-dimensional array values of {list} to a variable-sized list"
            ),
            // A type that PyArrow has none of, which `arrow_schema_error` raises first.
            None => "Can only convert 1-dimensional array values".to_string(),
        };
        return Ok(late.defer(|| Err(pyarrow_error(py, "ArrowInvalid", message)?)));
    }
    // SAFETY: `obj` is a live object, and PySequence_Check always succeeds without calling
    // into Python.
    let is_sequence = unsafe { pyo3::ffi::PySequence_Check(obj.as_ptr()) } == 1;
    if is_sequence {
        // SAFETY: `obj` is a live object. PySequence_Size returns -1 with an exception set
        // when it fails.
        let len = unsafe { pyo3::ffi::PySequence_Size(obj.as_ptr()) };
        if len < 0 {
            let err = PyErr::fetch(py);
            return Ok(late.defer(|| Err(err)));
        }
        // The length is the object's own, so it reserves nothing: a wrong one is read up to
        // the first index that fails.
        for index in 0..len {
            // SAFETY: PySequence_GetItem returns a new reference, or NULL with an exception
            // set.
            let item = unsafe {
                Bound::from_owned_ptr_or_err(py, pyo3::ffi::PySequence_GetItem(obj.as_ptr(), index))
            };
            match item {
                Ok(item) => items.push(convert(
                    &item,
                    element_type,
                    None,
                    contains_null,
                    late,
                    budget,
                )?),
                Err(err) => return Ok(late.defer(|| Err(err))),
            }
        }
    } else if obj.is_instance_of::<PySet>() || obj.is_instance_of::<PyDictValues>() {
        for item in obj.try_iter()? {
            items.push(convert(
                &item?,
                element_type,
                None,
                contains_null,
                late,
                budget,
            )?);
        }
    } else {
        return Ok(late.defer(|| {
            Err(pyarrow_error(
                py,
                "ArrowTypeError",
                format!(
                    "Could not convert {} with type {}: was not a sequence or recognized null \
                     for conversion to list type",
                    obj.repr()?,
                    arrow_type_name(obj)?
                ),
            )?)
        }));
    }
    Ok(Value::List(items))
}

/// The number of dimensions of `obj` if it is a NumPy array. NumPy is not imported when it is
/// not loaded, as such a value can only exist once it was.
fn numpy_array_dimensions(obj: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    let py = obj.py();
    let modules = SYS
        .get_or_try_init(py, || py.import("sys").map(Bound::unbind))?
        .bind(py)
        .getattr(intern!(py, "modules"))?;
    match modules.cast::<PyDict>()?.get_item(intern!(py, "numpy"))? {
        Some(numpy) if !numpy.is_none() => {
            if obj.is_instance(&numpy.getattr(intern!(py, "ndarray"))?)? {
                Ok(Some(obj.getattr(intern!(py, "ndim"))?.extract()?))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

/// The Arrow type of `dt` as PyArrow writes it, for the types that PySpark has no converter
/// of, whose values PyArrow takes as they are (`arrow_list`). `None` for a type that PySpark's
/// `to_arrow_type` has no Arrow type of, such as a char or a year-month interval.
fn arrow_type_text(dt: &DataType) -> Option<String> {
    Some(match dt {
        DataType::Boolean => "bool".to_string(),
        DataType::Byte => "int8".to_string(),
        DataType::Short => "int16".to_string(),
        DataType::Integer => "int32".to_string(),
        DataType::Long => "int64".to_string(),
        DataType::Float => "float".to_string(),
        DataType::Double => "double".to_string(),
        DataType::Date => "date32[day]".to_string(),
        DataType::DayTimeInterval { .. } => "duration[us]".to_string(),
        DataType::Time { .. } => "time64[ns]".to_string(),
        DataType::Array {
            element_type,
            contains_null,
        } => format!(
            "list<element: {}{}>",
            arrow_type_text(element_type)?,
            if *contains_null { "" } else { " not null" }
        ),
        _ => return None,
    })
}

/// The type objects of the schema of local rows, at the places of its core type, which PySpark's
/// converters hold (`LocalDataToArrowConversion._create_converter`) and use:
/// - a user-defined type serializes each value with its object (`udt = dataType`), of its own
///   parameters, which the JSON of the core type does not keep;
/// - a spatial type converts each value with its `toInternal`, which a subclass may override;
/// - the errors of a value name the `str()` of the type object, as of a subclass.
///
/// A schema that the client was given has its object at each place. One that it inferred has
/// those of its user-defined types alone, as its other types are PySpark's own classes: the
/// `__UDT__` of the first value at each place, which `_merge_type` keeps of two of one class
/// (see [`merge_type`]). The empty default holds no object, at no place below either.
#[derive(Clone, Default)]
pub(crate) struct TypeObjects<'py> {
    /// The type object of this place.
    object: Option<Bound<'py, PyAny>>,
    /// The objects of the fields of a struct, in their order; of the element of an array; of the
    /// key and the value of a map; or of the `sqlType()` of a user-defined type.
    children: Vec<TypeObjects<'py>>,
    /// The names by which PySpark's converter of a struct below the top of the schema takes its
    /// values apart, `fieldNames()` of its type object, when they are not those of its fields as
    /// they are (see [`cached_names`]).
    names: Option<Vec<Bound<'py, PyAny>>>,
}

impl<'py> TypeObjects<'py> {
    /// The objects of the type object `obj` of a given schema, of the core type `dt`, as PySpark
    /// makes its converters of them: of the `dataType` of each of its `fields`, its `elementType`,
    /// its `keyType` and `valueType`, and of the `sqlType()` of a user-defined type.
    pub(crate) fn of(obj: &Bound<'py, PyAny>, dt: &DataType) -> PyResult<Self> {
        Self::of_place(obj, dt, false)
    }

    /// As [`Self::of`], `nested` below the top of the schema, where PySpark's converter of a
    /// struct reads its `fieldNames()`, which the top level takes of the rows (see
    /// [`cached_names`]).
    fn of_place(obj: &Bound<'py, PyAny>, dt: &DataType, nested: bool) -> PyResult<Self> {
        let py = obj.py();
        let mut names = None;
        let children = match dt {
            DataType::Udt {
                sql_type: Some(sql_type),
                ..
            } => vec![Self::of_place(
                &obj.call_method0(intern!(py, "sqlType"))?,
                sql_type,
                true,
            )?],
            DataType::Struct { fields } => {
                if nested {
                    names = cached_names(obj, fields)?;
                }
                let objects = obj.getattr(intern!(py, "fields"))?;
                let mut children = Vec::with_capacity(fields.len());
                for (field, object) in fields.iter().zip(objects.try_iter()?) {
                    let data_type = object?.getattr(intern!(py, "dataType"))?;
                    children.push(Self::of_place(&data_type, &field.data_type, true)?);
                }
                children
            }
            DataType::Array { element_type, .. } => vec![Self::of_place(
                &obj.getattr(intern!(py, "elementType"))?,
                element_type,
                true,
            )?],
            DataType::Map {
                key_type,
                value_type,
                ..
            } => vec![
                Self::of_place(&obj.getattr(intern!(py, "keyType"))?, key_type, true)?,
                Self::of_place(&obj.getattr(intern!(py, "valueType"))?, value_type, true)?,
            ],
            _ => Vec::new(),
        };
        Ok(Self {
            object: Some(obj.clone()),
            children,
            names,
        })
    }

    /// The objects of a user-defined type that inference took of a value, its `__UDT__`; those
    /// of its `sqlType()` are read once the schema is inferred ([`Self::with_sql_types`]).
    fn of_udt(udt: Bound<'py, PyAny>) -> Self {
        Self {
            object: Some(udt),
            children: Vec::new(),
            names: None,
        }
    }

    /// The objects of a struct of the objects of its fields, of no type object of its own, as
    /// the struct of the single column of an atomic schema.
    pub(crate) fn of_fields(fields: Vec<Self>) -> Self {
        Self {
            object: None,
            children: fields,
            names: None,
        }
    }

    /// The objects of an array of inferred elements, empty if they are.
    fn of_array(element: Self) -> Self {
        Self::of_children(vec![element])
    }

    /// The objects of a map of inferred keys and values, empty if both are.
    fn of_map(key: Self, value: Self) -> Self {
        Self::of_children(vec![key, value])
    }

    fn of_children(children: Vec<Self>) -> Self {
        if children.iter().all(Self::is_empty) {
            return Self::default();
        }
        Self {
            object: None,
            children,
            names: None,
        }
    }

    fn is_empty(&self) -> bool {
        self.object.is_none() && self.children.is_empty()
    }

    /// The objects of an inferred schema `dt`, with those of the `sqlType()` of each of its
    /// user-defined types, as PySpark makes the converter of each of them of its `sqlType()`.
    pub(crate) fn with_sql_types(mut self, dt: &DataType) -> PyResult<Self> {
        match dt {
            DataType::Udt {
                sql_type: Some(sql_type),
                ..
            } => {
                if let Some(udt) = &self.object {
                    let sql = udt.call_method0(intern!(udt.py(), "sqlType"))?;
                    self.children = vec![Self::of_place(&sql, sql_type, true)?];
                }
            }
            DataType::Struct { fields } => {
                for (child, field) in self.children.iter_mut().zip(fields) {
                    *child = std::mem::take(child).with_sql_types(&field.data_type)?;
                }
            }
            DataType::Array { element_type, .. } => {
                if let Some(child) = self.children.get_mut(0) {
                    *child = std::mem::take(child).with_sql_types(element_type)?;
                }
            }
            DataType::Map {
                key_type,
                value_type,
                ..
            } => {
                for (child, data_type) in self.children.iter_mut().zip([key_type, value_type]) {
                    *child = std::mem::take(child).with_sql_types(data_type)?;
                }
            }
            _ => {}
        }
        Ok(self)
    }

    /// The type object of this place, if the schema holds it.
    fn object(&self) -> Option<&Bound<'py, PyAny>> {
        self.object.as_ref()
    }

    /// The names by which a value of the struct of this place is taken apart, when they are not
    /// those of its fields as they are (see [`cached_names`]).
    fn names(&self) -> Option<&[Bound<'py, PyAny>]> {
        self.names.as_deref()
    }

    /// The objects of the field `index` of a struct.
    pub(crate) fn field(&self, index: usize) -> Option<&Self> {
        self.children.get(index)
    }

    fn element(&self) -> Option<&Self> {
        self.children.first()
    }

    fn key_and_value(&self) -> (Option<&Self>, Option<&Self>) {
        (self.children.first(), self.children.get(1))
    }

    /// The objects of the `sqlType()` of a user-defined type.
    fn sql_type(&self) -> Option<&Self> {
        self.children.first()
    }

    /// The objects of the child `index`, taken out, empty if there are none.
    fn take_child(&mut self, index: usize) -> Self {
        self.children
            .get_mut(index)
            .map(std::mem::take)
            .unwrap_or_default()
    }
}

/// The objects of the fields of an inferred struct, a field at a time, which allocate only once a
/// field has some, as most data has no user-defined type.
#[derive(Default)]
struct FieldObjects<'py>(Vec<TypeObjects<'py>>);

impl<'py> FieldObjects<'py> {
    /// The objects of the field `index`, after those of the fields before it.
    fn push(&mut self, index: usize, objects: TypeObjects<'py>) {
        if !objects.is_empty() {
            self.0.resize_with(index, TypeObjects::default);
            self.0.push(objects);
        }
    }

    /// The objects of the struct of `len` fields.
    fn finish(mut self, len: usize) -> TypeObjects<'py> {
        if self.0.is_empty() {
            return TypeObjects::default();
        }
        self.0.resize_with(len, TypeObjects::default);
        TypeObjects {
            object: None,
            children: self.0,
            names: None,
        }
    }
}

/// A field value of `dt` from `obj`, as PySpark's converters and then PyArrow convert it, with the
/// type objects `objects` of `dt` (see [`TypeObjects`]). Each value, also of an array, a map or a
/// struct, is a step of `budget`.
pub(crate) fn convert<'py>(
    obj: &Bound<'py, PyAny>,
    dt: &DataType,
    objects: Option<&TypeObjects<'py>>,
    nullable: bool,
    late: &mut Late,
    budget: &ConversionBudget,
) -> PyResult<Value> {
    let py = obj.py();
    budget.step(py)?;
    if obj.is_none() {
        // A null type takes None either way.
        if !nullable && !matches!(dt, DataType::Null) {
            return Err(must_not_be_none(py, dt, objects)?);
        }
        return Ok(Value::Null);
    }
    match dt {
        DataType::Null => Err(pyspark_message(
            py,
            "PySparkValueError",
            &format!(
                "input for {} must be None, but got {}",
                type_str(py, dt, objects)?,
                obj.str()?
            ),
        )),
        DataType::Struct { fields } => convert_struct(obj, fields, objects, late, budget),
        DataType::Array {
            element_type,
            contains_null,
        } => {
            // As PySpark: an array whose elements need a converter takes a list or an
            // array.array, which the converter iterates, and another array is taken as it is,
            // by PyArrow (`arrow_list`).
            if !needs_converter(dt, nullable) {
                return arrow_list(obj, dt, element_type, *contains_null, late, budget);
            }
            if !obj.is_instance_of::<PyList>()
                && !obj.is_instance(ARRAY.import(py, "array", "array")?.as_any())?
            {
                return Err(assertion());
            }
            let element = objects.and_then(TypeObjects::element);
            let mut items = Vec::with_capacity(obj.len().unwrap_or(0));
            for item in obj.try_iter()? {
                items.push(convert(
                    &item?,
                    element_type,
                    element,
                    *contains_null,
                    late,
                    budget,
                )?);
            }
            Ok(Value::List(items))
        }
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => {
            let dict = obj.cast::<PyDict>().map_err(|_| assertion())?;
            let (key_objects, value_objects) = objects
                .map(TypeObjects::key_and_value)
                .unwrap_or((None, None));
            let mut entries = Vec::with_capacity(dict.len());
            for (key, value) in dict.iter() {
                entries.push((
                    convert(&key, key_type, key_objects, false, late, budget)?,
                    convert(
                        &value,
                        value_type,
                        value_objects,
                        *value_contains_null,
                        late,
                        budget,
                    )?,
                ));
            }
            Ok(Value::Map(entries))
        }
        DataType::Binary => match bytes_of(obj)? {
            Some(bytes) => Ok(Value::Binary(bytes)),
            None => Err(assertion()),
        },
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
                    return Err(must_not_be_none(py, dt, objects)?);
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
            if obj.is_exact_instance_of::<PyString>() {
                return Ok(late.defer(|| utf8_text(obj.cast::<PyString>()?)));
            }
            if obj.is_instance_of::<PyBool>() {
                // As PySpark Classic, which converts a bool to a string in the JVM.
                return Ok(Value::String(
                    (if obj.is_truthy()? { "true" } else { "false" }).to_string(),
                ));
            }
            let text = obj.str()?;
            Ok(late.defer(|| utf8_text(&text)))
        }
        // As PySpark, which has no converter of these types, nor of a type of its class only
        // such as `AtomicType()` or a `StructField` as a type, which the client lowers as
        // unparsed types: the value goes as it is to `to_arrow_schema`, which raises for the type
        // (`arrow_schema_error`).
        DataType::Char { .. }
        | DataType::Varchar { .. }
        | DataType::YearMonthInterval { .. }
        | DataType::CalendarInterval
        | DataType::Unparsed { .. } => Ok(Value::Null),
        DataType::Boolean
        | DataType::Byte
        | DataType::Short
        | DataType::Integer
        | DataType::Long
        | DataType::Float
        | DataType::Double
        | DataType::DayTimeInterval { .. }
        | DataType::Date
        | DataType::Time { .. } => Ok(late.defer(|| arrow_scalar(obj, dt))),
        // As PySpark's converters of these types, their `toInternal`: the `value` and
        // `metadata` attributes of a VariantVal, and the `srid` and `wkb` of a Geometry or a
        // Geography, which PyArrow converts as the fields of the structs of their Arrow types. A
        // spatial type of the schema of a Python subclass converts with its own (`spatial_value`).
        // Another value raises the `MALFORMED_*` condition of no message parameters, as PySpark's
        // `PySparkValueError(errorClass=...)` of its converter.
        DataType::Variant => {
            if obj.is_instance(variant_val_class(py)?)? {
                let value = obj.getattr(intern!(py, "value"))?;
                let metadata = obj.getattr(intern!(py, "metadata"))?;
                Ok(late.defer(|| {
                    Ok(Value::Variant {
                        value: arrow_binary(&value)?,
                        metadata: arrow_binary(&metadata)?,
                    })
                }))
            } else {
                Err(crate::errors::raise_pyspark_condition(
                    py,
                    "PySparkValueError",
                    "MALFORMED_VARIANT",
                ))
            }
        }
        DataType::Geometry { .. } => {
            if obj.is_instance(geometry_class(py)?)? {
                let native = py.get_type::<crate::types::PyGeometryType>();
                spatial_value(obj, objects.and_then(TypeObjects::object), &native, late)
            } else {
                Err(crate::errors::raise_pyspark_condition(
                    py,
                    "PySparkValueError",
                    "MALFORMED_GEOMETRY",
                ))
            }
        }
        DataType::Geography { .. } => {
            if obj.is_instance(geography_class(py)?)? {
                let native = py.get_type::<crate::types::PyGeographyType>();
                spatial_value(obj, objects.and_then(TypeObjects::object), &native, late)
            } else {
                Err(crate::errors::raise_pyspark_condition(
                    py,
                    "PySparkValueError",
                    "MALFORMED_GEOGRAPHY",
                ))
            }
        }
        // As PySpark's converter of a user-defined type: the value serialized by the UDT, and
        // converted as its SQL type. The UDT is the object of the schema, given or inferred, as
        // PySpark's.
        DataType::Udt {
            sql_type: Some(sql_type),
            python_class,
            serialized_python_class,
            ..
        } => {
            let (udt, sql_objects) = match objects.and_then(|o| Some((o.object()?, o))) {
                Some((udt, objects)) => (udt.clone(), objects.sql_type()),
                None => (
                    udt_of(
                        obj,
                        python_class.as_deref(),
                        serialized_python_class.as_deref(),
                    )?,
                    None,
                ),
            };
            let serialized = udt.call_method1(intern!(py, "serialize"), (obj,))?;
            convert(&serialized, sql_type, sql_objects, nullable, late, budget)
        }
        _ => Err(unsupported(obj)?),
    }
}

/// The value of a type that PySpark has no converter of, as PyArrow converts it.
fn arrow_scalar(obj: &Bound<'_, PyAny>, dt: &DataType) -> PyResult<Value> {
    let py = obj.py();
    match dt {
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
        DataType::DayTimeInterval { .. } => {
            Ok(duration_micros(obj)?.map_or(Value::Null, Value::DayTimeInterval))
        }
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
        // `convert` calls this for the types above only, those that PySpark has no converter of.
        _ => unreachable!("arrow_scalar of {dt:?}, a type that convert does not defer to it"),
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

/// The first error of PySpark's `to_arrow_schema` of the schema of local rows, which builds the
/// Arrow fields one at a time, depth first, each once its type is built: a char, varchar,
/// year-month or calendar interval type, of which Arrow has no type, and a type of its class
/// only or a `StructField` as a type, of which `to_arrow_type` has none either, is
/// `UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION`, and a field of the null type that is not
/// nullable, as a map key never is, is the `ValueError` of PyArrow's
/// `pa.field(name, pa.null(), nullable=False)`. A user-defined type is built as its SQL type.
pub(crate) fn arrow_schema_error(
    py: Python<'_>,
    fields: &[StructField],
) -> PyResult<Option<PyErr>> {
    enum Error<'a> {
        Unsupported(&'a DataType),
        NonNullableNull,
    }
    fn of_type(dt: &DataType) -> Option<Error<'_>> {
        match dt {
            DataType::Char { .. }
            | DataType::Varchar { .. }
            | DataType::YearMonthInterval { .. }
            | DataType::CalendarInterval
            | DataType::Unparsed { .. } => Some(Error::Unsupported(dt)),
            DataType::Struct { fields } => fields
                .iter()
                .find_map(|f| of_field(&f.data_type, f.nullable)),
            DataType::Array {
                element_type,
                contains_null,
            } => of_field(element_type, *contains_null),
            DataType::Map {
                key_type,
                value_type,
                value_contains_null,
            } => of_field(key_type, false).or_else(|| of_field(value_type, *value_contains_null)),
            DataType::Udt {
                sql_type: Some(sql_type),
                ..
            } => of_type(sql_type),
            _ => None,
        }
    }
    fn is_null(dt: &DataType) -> bool {
        match dt {
            DataType::Null => true,
            DataType::Udt {
                sql_type: Some(sql_type),
                ..
            } => is_null(sql_type),
            _ => false,
        }
    }
    fn of_field(dt: &DataType, nullable: bool) -> Option<Error<'_>> {
        of_type(dt).or_else(|| (!nullable && is_null(dt)).then_some(Error::NonNullableNull))
    }
    let first = fields
        .iter()
        .find_map(|f| of_field(&f.data_type, f.nullable));
    Ok(match first {
        Some(Error::Unsupported(unsupported)) => Some(raise_pyspark(
            py,
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION",
            &[("data_type", &data_type_str(py, unsupported)?)],
        )),
        Some(Error::NonNullableNull) => Some(PyValueError::new_err(
            "A null type field may not be non-nullable",
        )),
        None => None,
    })
}

/// Whether PySpark's `LocalDataToArrowConversion._need_converter` gives a value of `dt` a
/// converter, for a value of `nullable`: the values that PyArrow takes as they are do not.
pub(crate) fn needs_converter(dt: &DataType, nullable: bool) -> bool {
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

/// The value of a Geometry or a Geography `obj`, as the `toInternal` of the type of its column: of
/// the class `native` of the type, the dict of its `srid` and `wkb` attributes, which PyArrow
/// converts to the int32 and the binary of their struct. The type object of a schema of a Python
/// subclass converts the value with its own `toInternal`, which the subclass may override, as
/// PySpark's converter calls `dataType.toInternal(value)`; PyArrow then converts what it returns.
fn spatial_value<'py>(
    obj: &Bound<'py, PyAny>,
    type_object: Option<&Bound<'py, PyAny>>,
    native: &Bound<'py, PyType>,
    late: &mut Late,
) -> PyResult<Value> {
    let py = obj.py();
    if let Some(type_object) = type_object.filter(|t| !t.get_type().is(native)) {
        let internal = type_object.call_method1(intern!(py, "toInternal"), (obj,))?;
        return Ok(late.defer(|| spatial_internal(&internal)));
    }
    let srid = obj.getattr(intern!(py, "srid"))?;
    let wkb = obj.getattr(intern!(py, "wkb"))?;
    Ok(late.defer(|| spatial_struct(&srid, &wkb)))
}

/// The struct of a spatial value of what the `toInternal` of its type returned, as PyArrow
/// converts it: None is a null; a dict of the `srid` and the `wkb`, as the type's `toInternal`
/// returns, and a tuple of two values are read as they are stored; another value, such as a
/// sequence of (name, value) pairs, is converted by PyArrow alone, which raises its error of it.
/// PyArrow also keeps the kind of the first value of a column for the others, which this does
/// not.
fn spatial_internal(internal: &Bound<'_, PyAny>) -> PyResult<Value> {
    let py = internal.py();
    if internal.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(dict) = internal.cast::<PyDict>() {
        if let (Some(srid), Some(wkb)) = (
            dict.get_item(intern!(py, "srid"))?,
            dict.get_item(intern!(py, "wkb"))?,
        ) {
            return spatial_struct(&srid, &wkb);
        }
    } else if let Ok(tuple) = internal.cast::<PyTuple>() {
        if tuple.len() == 2 {
            return spatial_struct(&tuple.get_item(0)?, &tuple.get_item(1)?);
        }
    }
    let Ok(pyarrow) = py.import(intern!(py, "pyarrow")) else {
        return Err(pyarrow_error(
            py,
            "ArrowTypeError",
            format!(
                "Could not convert {} with type {}: was not a dict, tuple, or recognized null \
                 value for conversion to struct type",
                internal.repr()?,
                arrow_type_name(internal)?
            ),
        )?);
    };
    let not_null = [("nullable", false)].into_py_dict(py)?;
    let field = |name: &str, arrow_type: &str| {
        pyarrow.call_method(
            intern!(py, "field"),
            (name, pyarrow.call_method0(arrow_type)?),
            Some(&not_null),
        )
    };
    let struct_type = pyarrow.call_method1(
        intern!(py, "struct"),
        ([field("srid", "int32")?, field("wkb", "binary")?],),
    )?;
    let value = pyarrow
        .call_method1(intern!(py, "array"), ([internal], struct_type))?
        .call_method0(intern!(py, "to_pylist"))?
        .get_item(0)?;
    if value.is_none() {
        return Ok(Value::Null);
    }
    spatial_struct(&value.get_item("srid")?, &value.get_item("wkb")?)
}

/// The struct of a Geometry or a Geography, from its `srid` and `wkb` as PyArrow converts them.
fn spatial_struct(srid: &Bound<'_, PyAny>, wkb: &Bound<'_, PyAny>) -> PyResult<Value> {
    Ok(Value::Struct(
        spark_connect::row::Row::new(
            vec!["srid".to_string(), "wkb".to_string()],
            vec![
                convert_integral(srid, &DataType::Integer)?,
                Value::Binary(arrow_binary(wkb)?),
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

/// The user-defined type of a column whose class is `python_class`, to serialize `obj`, where the
/// conversion has no type object of the schema at its place ([`TypeObjects`]), as of a struct
/// object whose `fields` give fewer than its core type: that of the value, as PySpark infers a UDT
/// by `__UDT__`, when it is of the class; else an instance of the class, which the type sends
/// cloudpickled for a class of `__main__`.
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
/// `duration[us]`: a timedelta; a NumPy scalar of `timedelta64[us]`, or None for its NaT, which
/// PyArrow makes null, as another NumPy scalar raises `ArrowNotImplementedError`; else the value
/// as `CIntFromPython` takes it.
fn duration_micros(obj: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    let py = obj.py();
    if obj.is_instance(TIMEDELTA.import(py, "datetime", "timedelta")?.as_any())? {
        return timedelta_micros(obj).map(Some);
    }
    if numpy_generic(obj)? {
        let arrow_type = py.import(intern!(py, "pyarrow"))?.call_method1(
            intern!(py, "from_numpy_dtype"),
            (obj.getattr(intern!(py, "dtype"))?,),
        )?;
        let arrow_type = arrow_type.str()?.to_cow()?.into_owned();
        if arrow_type == "duration[us]" {
            // The microseconds that the scalar holds, as PyArrow reads them. A NaT, which NumPy
            // holds as the least int64, is null, as PyArrow makes it.
            let micros: i64 = obj
                .call_method1(intern!(py, "astype"), (intern!(py, "int64"),))?
                .extract()?;
            return Ok((micros != i64::MIN).then_some(micros));
        }
        return Err(pyarrow_error(
            py,
            "ArrowNotImplementedError",
            format!("Expected np.timedelta64 but got: {arrow_type}"),
        )?);
    }
    c_int_from_python(obj, 64, None).map(Some)
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

/// The values of a struct value from a mapping, as `mapping.get(name)` of each of `names`, or of
/// the names of `fields` without them.
fn values_by_name<'py>(
    mapping: &Bound<'py, PyAny>,
    fields: &[StructField],
    names: Option<&[Bound<'py, PyAny>]>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = mapping.py();
    let Some(names) = names else {
        let names = fields
            .iter()
            .map(|f| PyString::new(py, &f.name).into_any())
            .collect::<Vec<_>>();
        return values_by_name(mapping, fields, Some(&names));
    };
    if let Ok(dict) = mapping.cast::<PyDict>() {
        return names
            .iter()
            .map(|name| {
                Ok(dict
                    .get_item(name)?
                    .unwrap_or_else(|| py.None().into_bound(py)))
            })
            .collect();
    }
    names
        .iter()
        .map(|name| mapping.call_method1(intern!(py, "get"), (name,)))
        .collect()
}

/// The values of a tuple or a Row, which must be as many as the names that the struct value is
/// taken apart by, `len`.
fn values_by_position<'py>(
    py: Python<'py>,
    values: Vec<Bound<'py, PyAny>>,
    len: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if values.len() == len {
        Ok(values)
    } else {
        Err(axis_length_mismatch(py, len, values.len()))
    }
}

/// The names by which PySpark's converters take a value of the struct `obj` of the given schema
/// apart, `obj.fieldNames()`, which `to_row` reads of the schema and the converter of a struct of
/// its type object, when they are not the names of `fields`, the fields of `obj` as they are:
/// PySpark's `StructType` caches its `names` as it is made (and as `add` adds a field), so of a
/// `fields` list changed in place, such as of a `StructField` appended, they are the names before
/// the change, while the converters are of the fields as they are. None of the type of another
/// class, or of names that are those of the fields.
pub(crate) fn cached_names<'py>(
    obj: &Bound<'py, PyAny>,
    fields: &[StructField],
) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    if !obj.is_instance_of::<crate::types::PyStructType>() {
        return Ok(None);
    }
    let names = obj
        .call_method0(intern!(obj.py(), "fieldNames"))?
        .try_iter()?
        .collect::<PyResult<Vec<_>>>()?;
    let same = names.len() == fields.len()
        && names.iter().zip(fields).all(|(name, field)| {
            name.cast::<PyString>()
                .is_ok_and(|name| name.to_cow().is_ok_and(|name| name == field.name))
        });
    Ok((!same).then_some(names))
}

/// The `IndexError` that PySpark's conversion of local rows raises once their values are
/// converted, as it makes the Arrow schema of the fields of the schema, whose
/// `_deduplicate_field_names` names the fields of each struct below the top level, in an array
/// or a map too, by the cached names of its type object (see [`cached_names`]): of fewer names
/// than fields, of a `fields` list that a field was added to in place. Its converters stop at a
/// user-defined type, of whose SQL type the schema has no struct.
pub(crate) fn dedup_names_error(
    objects: Option<&TypeObjects<'_>>,
    fields: &[StructField],
) -> Option<PyErr> {
    fn short<'py>(objects: Option<&TypeObjects<'py>>, dt: &DataType) -> bool {
        let Some(objects) = objects else {
            return false;
        };
        match dt {
            DataType::Struct { fields } => {
                objects
                    .names()
                    .is_some_and(|names| names.len() < fields.len())
                    || fields
                        .iter()
                        .enumerate()
                        .any(|(i, field)| short(objects.field(i), &field.data_type))
            }
            DataType::Array { element_type, .. } => short(objects.element(), element_type),
            DataType::Map {
                key_type,
                value_type,
                ..
            } => {
                let (key, value) = objects.key_and_value();
                short(key, key_type) || short(value, value_type)
            }
            _ => false,
        }
    }
    let objects = objects?;
    fields
        .iter()
        .enumerate()
        .any(|(i, field)| short(objects.field(i), &field.data_type))
        .then(|| PyIndexError::new_err("list index out of range"))
}

/// A struct value, as PySpark converts one: a tuple or a Row by position, a dict by name,
/// and another object by the names of its `__dict__`. A tuple is read by `len(value)` and
/// `value[i]`, of the `__len__` and `__getitem__` of a subclass too (`struct_by_index`).
fn convert_struct<'py>(
    obj: &Bound<'py, PyAny>,
    fields: &[StructField],
    objects: Option<&TypeObjects<'py>>,
    late: &mut Late,
    budget: &ConversionBudget,
) -> PyResult<Value> {
    let py = obj.py();
    // Of the cached names of a `fields` list changed in place, if they differ (see
    // `cached_names`).
    let names = objects.and_then(TypeObjects::names);
    let len = names.map_or(fields.len(), <[_]>::len);
    let values = if let Ok(tuple) = obj.cast::<PyTuple>() {
        if !reads_as_stored(tuple, true)? {
            return struct_by_index(obj, fields, len, objects, late, budget);
        }
        values_by_position(py, tuple.iter().collect(), len)?
    } else if obj.is_instance_of::<PyDict>() {
        values_by_name(obj, fields, names)?
    } else if let Ok(attributes) = obj.getattr(intern!(py, "__dict__")) {
        values_by_name(&attributes, fields, names)?
    } else {
        // PySpark asserts that another value has a `__dict__`.
        return Err(PyAssertionError::new_err(format!(
            "{} {}",
            obj.get_type().str()?,
            obj.str()?
        )));
    };
    struct_of(&values, fields, objects, late, budget)
}

fn struct_of<'py>(
    values: &[Bound<'py, PyAny>],
    fields: &[StructField],
    objects: Option<&TypeObjects<'py>>,
    late: &mut Late,
    budget: &ConversionBudget,
) -> PyResult<Value> {
    let names: std::sync::Arc<[String]> = fields.iter().map(|f| f.name.clone()).collect();
    // Each value is converted with the field of its position, as PySpark's `field_convs[i]` of
    // the values of the cached names of a `fields` list changed in place (see `cached_names`):
    // past the fields, its `IndexError`, once those before are converted; short of them, the
    // fields after are null, and the Arrow schema of the conversion raises once every value is
    // converted (`dedup_names_error`).
    let mut converted = Vec::with_capacity(fields.len());
    for (index, value) in values.iter().enumerate() {
        let Some(f) = fields.get(index) else {
            return Err(PyIndexError::new_err("list index out of range"));
        };
        let field_objects = objects.and_then(|objects| objects.field(index));
        converted.push(convert(
            value,
            &f.data_type,
            field_objects,
            f.nullable,
            late,
            budget,
        )?);
    }
    converted.resize_with(fields.len(), || Value::Null);
    Ok(Value::Struct(StructValue::new(names, converted)))
}

/// A struct value of a tuple of a subclass of its own `__len__` or `__getitem__`, as PySpark's
/// converter reads one: `len(value)` must be `len`, the number of the names that it is taken
/// apart by (see `cached_names`), and `value[i]` of each is read as the value before it is
/// converted with the field of its position, of which PySpark reads the converter first: past
/// the fields, its `IndexError`, and short of them, the fields after are null.
fn struct_by_index<'py>(
    obj: &Bound<'py, PyAny>,
    fields: &[StructField],
    len: usize,
    objects: Option<&TypeObjects<'py>>,
    late: &mut Late,
    budget: &ConversionBudget,
) -> PyResult<Value> {
    let py = obj.py();
    if obj.len()? != len {
        // PySpark's message reads `len(value)` again.
        return Err(axis_length_mismatch(py, len, obj.len()?));
    }
    let names: std::sync::Arc<[String]> = fields.iter().map(|f| f.name.clone()).collect();
    let mut values = Vec::with_capacity(fields.len());
    for index in 0..len {
        let Some(f) = fields.get(index) else {
            return Err(PyIndexError::new_err("list index out of range"));
        };
        let field_objects = objects.and_then(|objects| objects.field(index));
        let value = obj.get_item(index)?;
        values.push(convert(
            &value,
            &f.data_type,
            field_objects,
            f.nullable,
            late,
            budget,
        )?);
    }
    values.resize_with(fields.len(), || Value::Null);
    Ok(Value::Struct(StructValue::new(names, values)))
}

/// The methods of `tuple` that read a tuple as it is stored, `__len__`, `__iter__` and
/// `__getitem__`, and the `__getitem__` of the Row class, which reads an int index so.
static TUPLE_METHODS: PyOnceLock<[Py<PyAny>; 4]> = PyOnceLock::new();

/// Whether the type of the tuple `value` has the `__len__` of `tuple`, and its `__iter__`, or,
/// `by_index`, the `__getitem__` of `tuple` or of the Row class, which reads an int index of the
/// storage: so that `len(value)` and `tuple(value)`, or `value[i]`, give the values as they are
/// stored, which the native reads take without the calls. So do an exact tuple, a Row and a
/// namedtuple.
fn reads_as_stored(value: &Bound<'_, PyTuple>, by_index: bool) -> PyResult<bool> {
    let py = value.py();
    if value.is_exact_instance_of::<PyTuple>() {
        return Ok(true);
    }
    let [len, iter, item, row_item] = TUPLE_METHODS.get_or_try_init(py, || -> PyResult<_> {
        let tuple = py.get_type::<PyTuple>();
        let row = py.import("pyspark.sql.types")?.getattr("Row")?;
        Ok([
            tuple.getattr(intern!(py, "__len__"))?.unbind(),
            tuple.getattr(intern!(py, "__iter__"))?.unbind(),
            tuple.getattr(intern!(py, "__getitem__"))?.unbind(),
            row.getattr(intern!(py, "__getitem__"))?.unbind(),
        ])
    })?;
    let ty = value.get_type();
    if !ty.getattr(intern!(py, "__len__"))?.is(len.bind(py)) {
        return Ok(false);
    }
    if !by_index {
        return Ok(ty.getattr(intern!(py, "__iter__"))?.is(iter.bind(py)));
    }
    let getitem = ty.getattr(intern!(py, "__getitem__"))?;
    Ok(getitem.is(item.bind(py)) || getitem.is(row_item.bind(py)))
}

/// `tuple(obj)`, as `PySequence_Tuple` makes it: `obj` itself of an exact tuple, else of the
/// items of `iter(obj)`, after its length hint, which calls `__len__`.
fn sequence_tuple<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    // SAFETY: `obj` is a live object, and PySequence_Tuple returns a new reference to a tuple,
    // or NULL with an exception set.
    let tuple = unsafe {
        Bound::from_owned_ptr_or_err(obj.py(), pyo3::ffi::PySequence_Tuple(obj.as_ptr()))
    }?;
    Ok(tuple.cast_into::<PyTuple>()?)
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
/// another value, such as a list, by position. A tuple of a subclass and another value are taken
/// by `len(item)`, which must be the number of fields, and `tuple(item)`, of the `__len__` and
/// `__iter__` of a subclass too, which may give fewer or more values than the fields: PySpark's
/// `row[i]` of a missing one raises `IndexError` as its column converts, and those after the
/// fields are not read (`create_from_local`). With `names`, the cached names of the schema that
/// differ from its fields (see [`cached_names`]), the row is taken apart by them, as `to_row`
/// reads `schema.fieldNames()`, and its values are converted with the fields of their positions.
pub(crate) fn row_values<'py>(
    item: &Bound<'py, PyAny>,
    fields: &[StructField],
    names: Option<&[Bound<'py, PyAny>]>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let py = item.py();
    let len = names.map_or(fields.len(), <[_]>::len);
    if item.is_none() {
        return Ok(vec![py.None().into_bound(py); len]);
    }
    if let Ok(tuple) = item.cast::<PyTuple>() {
        if !reads_as_stored(tuple, false)? {
            return sequence_values(item, len);
        }
        return values_by_position(py, tuple.iter().collect(), len);
    }
    if item.is_instance_of::<PyDict>() {
        return values_by_name(item, fields, names);
    }
    if item.is_instance(variant_val_class(py)?)? {
        return Err(variant_row(py));
    }
    if item.hasattr(intern!(py, "__dict__"))? {
        return values_by_name(&item.getattr(intern!(py, "__dict__"))?, fields, names);
    }
    sequence_values(item, len)
}

/// The values of `tuple(item)` of a row whose `len(item)` must be `len`, the number of the names
/// that the row is taken apart by.
fn sequence_values<'py>(item: &Bound<'py, PyAny>, len: usize) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if item.len()? != len {
        // PySpark's message reads `len(item)` again.
        return Err(axis_length_mismatch(item.py(), len, item.len()?));
    }
    Ok(sequence_tuple(item)?.iter().collect())
}

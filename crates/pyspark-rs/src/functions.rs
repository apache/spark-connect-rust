//! PyO3 wrappers for Spark SQL functions.

use pyo3::prelude::*;
use spark_connect::column::Column;
use spark_connect::expression::{
    Expression, ExpressionKind, ExpressionSpec, LiteralExpression, NamedArgument,
};
use spark_connect::functions as spark_funcs;
use spark_connect::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::column::PyColumn;
use crate::dataframe::PyDataFrame;
use crate::session::PySparkSession;

// Include the auto-generated dispatch function
mod dispatch {
    use pyo3::prelude::*;
    include!("dispatch_generated.rs");
}

/// PyO3 wrapper for building a Python UDF expression.
/// Called from Python as _make_udf to create a UDF column.
///
/// Arguments:
/// - name: str - name of the UDF
/// - return_type: DataType - return type of the UDF
/// - eval_type: int - evaluation type (from PythonEvalType)
/// - command_bytes: bytes - cloudpickled command (usually (func, output_type))
/// - python_ver: str - Python version (e.g., "3.9")
/// - *args: Column - argument columns
/// - named_args: (str, Column) pairs - the keyword arguments, in order
#[pyfunction]
#[pyo3(signature = (name, return_type, eval_type, command_bytes, python_ver, *args, deterministic=true, named_args=Vec::new()))]
#[allow(clippy::too_many_arguments)]
fn pyfunc_make_udf(
    name: String,
    return_type: &Bound<'_, PyAny>,
    eval_type: i32,
    command_bytes: Vec<u8>,
    python_ver: String,
    args: Vec<Bound<'_, PyAny>>,
    deterministic: bool,
    named_args: Vec<(String, Bound<'_, PyAny>)>,
) -> PyResult<PyColumn> {
    // Accept any DataType object (our type classes / DataType) or a DDL string.
    let return_data_type = crate::types::py_to_data_type(return_type)?;

    // The positional arguments, then the named ones, as PySpark's UDF call.
    let arg_exprs: Vec<Expression> = udf_arguments(args, named_args, false)?
        .into_iter()
        .map(|c| c.expression().clone())
        .collect();

    // Create the Python UDF payload
    let python_udf = PythonUDFPayload::new(return_data_type, eval_type, command_bytes, python_ver);

    // Create the UDF expression
    let udf_expr =
        CommonInlineUserDefinedFunctionExpression::new(name, deterministic, arg_exprs, python_udf);

    Ok(PyColumn::new(Column::new(
        ExpressionSpec::CommonInlineUserDefinedFunction(Box::new(udf_expr)).into(),
    )))
}

/// The arguments of a UDF or UDTF call: the positional ones, then a `NamedArgumentExpression`
/// of each keyword argument, as PySpark's `UserDefinedFunction` and `UserDefinedTableFunction`
/// build them. As PySpark's, a UDTF call also takes a `TableArg`, which passes the rows of
/// its DataFrame.
fn udf_arguments(
    args: Vec<Bound<'_, PyAny>>,
    named_args: Vec<(String, Bound<'_, PyAny>)>,
    table_args: bool,
) -> PyResult<Vec<Column>> {
    let argument = |arg: &Bound<'_, PyAny>| -> PyResult<Column> {
        if table_args {
            if let Ok(table_arg) = arg.extract::<PyRef<'_, crate::tablearg::PyTableArg>>() {
                return Ok(table_arg.inner.to_column());
            }
        }
        // As PySpark's `to_expr` of a UDF call: a column name is a ColumnReference of it,
        // without the origin of `col`.
        if arg.is_instance_of::<pyo3::types::PyString>() {
            return Ok(Column::new(Expression::column(arg.extract::<String>()?)));
        }
        to_column(arg)
    };
    let mut columns = Vec::with_capacity(args.len() + named_args.len());
    for arg in args {
        columns.push(argument(&arg)?);
    }
    for (key, value) in named_args {
        let value = argument(&value)?.expression().clone();
        columns.push(Column::new(
            ExpressionSpec::NamedArgument(Box::new(NamedArgument::new(key, value))).into(),
        ));
    }
    Ok(columns)
}

/// PyO3 wrapper for building a Python UDTF (table function) call.
/// Called from Python as `make_udtf`; mirrors `make_udf` but yields a DataFrame
/// (a `CommonInlineUserDefinedTableFunction` relation) via the session's TVF.
///
/// - session: the SparkSession the UDTF is evaluated against
/// - name/return_type/eval_type/command_bytes/python_ver: as for `make_udf`
///   (eval_type ∈ {300 SQL_TABLE_UDF, 301 SQL_ARROW_TABLE_UDF, 302 SQL_ARROW_UDTF})
/// - *args: Column arguments
/// - deterministic: whether the UDTF is deterministic, which a PySpark UDTF is not unless
///   `asDeterministic()` makes it so
/// - named_args: (str, Column) pairs - the keyword arguments, in order
#[pyfunction]
#[pyo3(signature = (session, name, return_type, eval_type, command_bytes, python_ver, *args, deterministic=false, named_args=Vec::new()))]
#[allow(clippy::too_many_arguments)]
fn pyfunc_make_udtf(
    session: PyRef<'_, PySparkSession>,
    name: String,
    return_type: Option<&Bound<'_, PyAny>>,
    eval_type: i32,
    command_bytes: Vec<u8>,
    python_ver: String,
    args: Vec<Bound<'_, PyAny>>,
    deterministic: bool,
    named_args: Vec<(String, Bound<'_, PyAny>)>,
) -> PyResult<PyDataFrame> {
    let col_args = udf_arguments(args, named_args, true)?;
    // Connect leaves DDL parsing and validation to the server. In particular,
    // do not use py_to_data_type's eager DDL parser for string return types.
    let rt = return_type
        .map(|value| {
            if let Ok(ddl) = value.extract::<String>() {
                Ok(spark_connect::types::DataType::Unparsed {
                    data_type_string: ddl,
                })
            } else {
                crate::types::py_to_data_type(value)
            }
        })
        .transpose()?;
    let df = session.session.tvf().udtf(
        &name,
        col_args,
        rt,
        eval_type,
        command_bytes,
        python_ver,
        deterministic,
    );
    Ok(PyDataFrame::new(df))
}

/// Register all SQL functions into the Python functions module.
pub fn register_functions(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pyfunc_col, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_lit, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_expr, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_sum, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_count, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_avg, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_max, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_min, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_when, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_call_function, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_make_udf, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_make_udtf, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_named_lambda_variable, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_lambda_function, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_call_named_function, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_invoke_function, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_sha2, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_variant_strip_nulls, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_window, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_window_with_slide_and_start, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_from_avro, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_from_avro_with_options, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_to_avro_with_schema, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_from_protobuf, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_from_protobuf_with_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_from_protobuf_with_options, m)?)?;
    m.add_function(wrap_pyfunction!(
        pyfunc_from_protobuf_with_descriptor_and_options,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(pyfunc_to_protobuf, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_to_protobuf_with_descriptor, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_to_protobuf_with_options, m)?)?;
    m.add_function(wrap_pyfunction!(
        pyfunc_to_protobuf_with_descriptor_and_options,
        m
    )?)?;
    Ok(())
}

/// `enum.Enum`, of whose instances a literal is of the value.
static ENUM: pyo3::sync::PyOnceLock<Py<pyo3::types::PyType>> = pyo3::sync::PyOnceLock::new();

/// The value of `obj`, the value of the value of an `Enum` while it is one, else `obj`, as
/// PySpark's `pyspark.sql.utils.enum_to_value`, which its `lit`, Column operators and functions
/// apply to the values that they take.
pub(crate) fn enum_to_value<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    let enum_type = ENUM.import(py, "enum", "Enum")?;
    let mut value = obj.clone();
    while value.is_instance(enum_type)? {
        value = value.getattr("value")?;
    }
    Ok(value)
}

/// The error of a Python int beyond the range of a long, as PySpark's `LiteralExpression`
/// raises it: `VALUE_NOT_BETWEEN`, whose maximum is `JVM_SHORT_MAX`, as PySpark writes it.
pub(crate) fn value_not_between(py: Python<'_>) -> PyErr {
    crate::column::raise_pyspark(
        py,
        "PySparkValueError",
        "VALUE_NOT_BETWEEN",
        &[
            ("arg_name", "value"),
            ("min", "-9223372036854775808"),
            ("max", "32767"),
        ],
    )
}

/// An int32 literal within its range, otherwise a long, as PySpark's `_infer_type`.
fn integer_literal(value: i64) -> LiteralExpression {
    if let Ok(value) = i32::try_from(value) {
        LiteralExpression::int(value)
    } else {
        LiteralExpression::long(value)
    }
}

/// Exact built-in primitives need no Enum, NumPy or non-primitive scalar probing.
/// Subclasses and out-of-range ints retain the generic conversion order in the callers.
fn exact_primitive_literal(obj: &Bound<'_, PyAny>) -> PyResult<Option<LiteralExpression>> {
    if let Ok(boolean) = obj.cast_exact::<pyo3::types::PyBool>() {
        return Ok(Some(LiteralExpression::boolean(boolean.extract()?)));
    }
    if let Ok(integer) = obj.cast_exact::<pyo3::types::PyInt>() {
        if let Ok(value) = integer.extract::<i64>() {
            return Ok(Some(integer_literal(value)));
        }
    }
    if let Ok(float) = obj.cast_exact::<pyo3::types::PyFloat>() {
        return Ok(Some(LiteralExpression::double(float.extract()?)));
    }
    if let Ok(string) = obj.cast_exact::<pyo3::types::PyString>() {
        // A str that UTF-8 cannot encode raises the `UnicodeEncodeError` of its encoding, as
        // the str of a subclass below.
        return Ok(Some(LiteralExpression::string(string.extract::<String>()?)));
    }
    if obj.is_exact_instance_of::<pyo3::types::PyBytes>()
        || obj.is_exact_instance_of::<pyo3::types::PyByteArray>()
    {
        return Ok(Some(LiteralExpression::Binary(obj.extract()?)));
    }
    Ok(None)
}

/// Convert a Python object to a Rust Column (handles Column or scalars that become literals).
pub fn to_column(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
    // Try to get as PyColumn first
    if let Ok(col) = obj.extract::<Bound<'_, PyColumn>>() {
        return Ok(col.borrow().column.clone());
    }
    if let Some(lit) = exact_primitive_literal(obj)? {
        return Ok(Column::new(Expression::literal(lit)));
    }
    // An Enum is the literal of its value, as PySpark's.
    let obj = &enum_to_value(obj)?;

    // Check for None
    if obj.is_none() {
        return Ok(Column::new(Expression::literal(LiteralExpression::null(
            spark_connect::types::DataType::Null,
        ))));
    }

    // A list, or a one-dimensional NumPy array, is an array, as PySpark's functions take a
    // value with `lit`: `pyspark.sql.functions.lit` builds it. Another NumPy array is not a
    // literal value; NumPy would convert one of a single element to a scalar below.
    let ndarray = numpy_ndarray_ndim(obj)?;
    if obj.is_instance_of::<pyo3::types::PyList>() || ndarray == Some(1) {
        let column = obj
            .py()
            .import("pyspark.sql.functions")?
            .getattr("lit")?
            .call1((obj,))?;
        return Ok(column.extract::<PyRef<'_, PyColumn>>()?.column.clone());
    }
    if ndarray.is_some() {
        return Err(crate::column::raise_pyspark(
            obj.py(),
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE",
            &[("data_type", "ndarray")],
        ));
    }

    // A NumPy scalar is typed by its dtype, as PySpark's `LiteralExpression._infer_type`.
    if let Some(lit) = numpy_scalar_literal(obj)? {
        return Ok(Column::new(Expression::literal(lit)));
    }

    // Non-primitive scalars must be checked BEFORE int/float: Decimal defines
    // __float__/__int__, and a pandas Timestamp (a datetime subclass) defines __int__
    // (nanoseconds), so an early int/float extract would silently mis-encode them.
    // Mirrors `py_to_value` in session.rs so `lit(..)` and `createDataFrame(..)` agree.
    if let Some(lit) = scalar_literal(obj)? {
        return Ok(Column::new(Expression::literal(lit)));
    }

    // Try to convert scalars to literals
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Column::new(Expression::literal(
            LiteralExpression::boolean(b),
        )));
    }

    // An int or a float by its type, as PySpark's `_infer_type` takes them: an object that
    // only converts to one (`__index__`, `__float__`) is not a literal.
    if obj.is_instance_of::<pyo3::types::PyInt>() {
        return match obj.extract::<i64>() {
            Ok(i) => Ok(Column::new(Expression::literal(integer_literal(i)))),
            // An int beyond a long is not a double, as PySpark's literal of it: it is an error.
            Err(_) => Err(value_not_between(obj.py())),
        };
    }

    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return Ok(Column::new(Expression::literal(LiteralExpression::double(
            obj.extract::<f64>()?,
        ))));
    }

    if obj.is_instance_of::<pyo3::types::PyString>() {
        // A str that UTF-8 cannot encode (a lone surrogate) raises the `UnicodeEncodeError` of
        // its encoding, as protobuf's string field does when PySpark sends the literal.
        return Ok(Column::new(Expression::literal(LiteralExpression::string(
            obj.extract::<String>()?,
        ))));
    }

    // As PySpark's literal conversion for a value of another type.
    let type_name = obj
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    Err(crate::column::raise_pyspark(
        obj.py(),
        "PySparkTypeError",
        "UNSUPPORTED_DATA_TYPE",
        &[("data_type", &type_name)],
    ))
}

/// Convert a Python object that a Column method takes as a value, such as `when`, `otherwise`
/// or `getItem`, to a Column. A list is a typed array literal, as PySpark's
/// `LiteralExpression._from_value` makes it, so an empty list or one that starts with None is
/// CANNOT_INFER_ARRAY_ELEMENT_TYPE. `to_column` builds a list as `lit` does, for the functions.
/// A NumPy array is not a value, as PySpark's `_from_value` does not take it, though `lit` does.
pub(crate) fn to_value_column(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Some(lit) = exact_primitive_literal(obj)? {
        return Ok(Column::new(Expression::literal(lit)));
    }
    let obj = &enum_to_value(obj)?;
    if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
        return Ok(Column::new(Expression::literal(typed_array_literal(list)?)));
    }
    if numpy_ndarray_ndim(obj)?.is_some() {
        return Err(crate::column::raise_pyspark(
            obj.py(),
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE",
            &[("data_type", "ndarray")],
        ));
    }
    to_column(obj)
}

/// How deep the lists of a Column method may nest. It protects this client: a deeper list is a
/// `RecursionError` here, rather than a native stack overflow. PySpark's `LiteralExpression`
/// recurses in Python for each level, so where its `RecursionError` starts depends on the Python
/// recursion limit and on the protobuf backend, and this bound is not a copy of its point (a
/// little less below a deeper caller).
const MAX_LIST_DEPTH: usize = 988;

/// The stack that a level of nested lists keeps free before it goes down a level, and the size of
/// a stack segment that it adds when less is left, as for the levels of an expression or a plan:
/// a thread's stack may be smaller than the depth of a list needs, so a level grows it.
const LIST_RED_ZONE: usize = if cfg!(debug_assertions) {
    1 << 20
} else {
    128 << 10
};
const LIST_STACK_SEGMENT: usize = if cfg!(debug_assertions) {
    16 << 20
} else {
    4 << 20
};

/// Run `f`, which goes down a level of nested lists, with at least [`LIST_RED_ZONE`] of stack left.
fn with_stack_for_a_list_level<T>(f: impl FnOnce() -> T) -> T {
    stacker::maybe_grow(LIST_RED_ZONE, LIST_STACK_SEGMENT, f)
}

thread_local! {
    static LIST_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One more level of nested lists for as long as it lives.
struct ListLevel;

impl ListLevel {
    fn enter() -> PyResult<ListLevel> {
        LIST_DEPTH.with(|depth| {
            if depth.get() >= MAX_LIST_DEPTH {
                return Err(pyo3::exceptions::PyRecursionError::new_err(
                    "maximum recursion depth exceeded while converting a list to a literal",
                ));
            }
            depth.set(depth.get() + 1);
            Ok(ListLevel)
        })
    }
}

impl Drop for ListLevel {
    fn drop(&mut self) {
        LIST_DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}

/// The array literal of a list, of the type of its first element, as PySpark's
/// `LiteralExpression` infers it. The first element's literal gives that type, and each other
/// element is converted to it once, by `literal_as`.
fn typed_array_literal(list: &Bound<'_, pyo3::types::PyList>) -> PyResult<LiteralExpression> {
    let _level = ListLevel::enter()?;
    let Some(first) = list.iter().next().filter(|first| !first.is_none()) else {
        return Err(crate::column::raise_pyspark(
            list.py(),
            "PySparkTypeError",
            "CANNOT_INFER_ARRAY_ELEMENT_TYPE",
            &[],
        ));
    };
    let first = match first.cast::<pyo3::types::PyList>() {
        Ok(inner) => with_stack_for_a_list_level(|| typed_array_literal(inner))?,
        Err(_) => scalar_literal_of(&first)?,
    };
    let element_type = literal_data_type(&first);
    let mut elements = vec![first];
    for item in list.iter().skip(1) {
        elements.push(literal_as(&item, &element_type)?);
    }
    Ok(LiteralExpression::Array {
        element_type: Box::new(element_type),
        elements,
    })
}

/// The literal of `obj` as a value of `data_type`, as PySpark's `LiteralExpression(obj, type)`:
/// None is a null, a list is an array of the element type, and another value must be accepted
/// for the type, by the Python or NumPy classes that PySpark's asserts take, else it is a bare
/// AssertionError. So an int is a byte, a short, an integer or a long only within its range,
/// and a bool is an int, but a float is none of them.
fn literal_as(
    obj: &Bound<'_, PyAny>,
    data_type: &spark_connect::types::DataType,
) -> PyResult<LiteralExpression> {
    use spark_connect::types::DataType;
    let obj = &enum_to_value(obj)?;
    if obj.is_none() {
        return Ok(LiteralExpression::null(data_type.clone()));
    }
    let assertion = || pyo3::exceptions::PyAssertionError::new_err(());
    let literal = match data_type {
        DataType::Array { element_type, .. } => {
            let list = obj.cast::<pyo3::types::PyList>().map_err(|_| assertion())?;
            let _level = ListLevel::enter()?;
            let elements = with_stack_for_a_list_level(|| {
                list.iter()
                    .map(|item| literal_as(&item, element_type))
                    .collect::<PyResult<Vec<_>>>()
            })?;
            LiteralExpression::Array {
                element_type: element_type.clone(),
                elements,
            }
        }
        DataType::Boolean => {
            if !(obj.is_instance_of::<pyo3::types::PyBool>() || is_numpy_scalar(obj, &["bool_"])?) {
                return Err(assertion());
            }
            LiteralExpression::Boolean(obj.is_truthy()?)
        }
        DataType::Byte => {
            LiteralExpression::Byte(
                integral_value(obj, &["int8"], i8::MIN.into(), i8::MAX.into())? as i32,
            )
        }
        DataType::Short => LiteralExpression::Short(integral_value(
            obj,
            &["int8", "int16"],
            i16::MIN.into(),
            i16::MAX.into(),
        )? as i32),
        DataType::Integer => LiteralExpression::Integer(integral_value(
            obj,
            &["int8", "int16", "int32"],
            i32::MIN.into(),
            i32::MAX.into(),
        )? as i32),
        DataType::Long => LiteralExpression::Long(integral_value(
            obj,
            &["int8", "int16", "int32", "int64"],
            i64::MIN,
            i64::MAX,
        )?),
        DataType::Float => {
            if !(obj.is_instance_of::<pyo3::types::PyFloat>()
                || is_numpy_scalar(obj, &["float32"])?)
            {
                return Err(assertion());
            }
            LiteralExpression::Float(obj.extract::<f64>()? as f32)
        }
        DataType::Double => {
            if !(obj.is_instance_of::<pyo3::types::PyFloat>()
                || is_numpy_scalar(obj, &["float32", "float64"])?)
            {
                return Err(assertion());
            }
            LiteralExpression::Double(obj.extract::<f64>()?)
        }
        DataType::Decimal { precision, scale } => {
            if !is_instance_of_class(obj, "decimal", "Decimal")? {
                return Err(assertion());
            }
            LiteralExpression::Decimal {
                value: obj.str()?.extract()?,
                precision: *precision,
                scale: *scale,
            }
        }
        DataType::String { .. } => {
            if !obj.is_instance_of::<pyo3::types::PyString>() {
                return Err(assertion());
            }
            // A str that UTF-8 cannot encode raises, as in `to_column`.
            LiteralExpression::String(obj.extract::<String>()?)
        }
        // As PySpark's `to_plan`, which sends `bytes(value)`.
        DataType::Binary => match crate::infer::bytes_of(obj)? {
            Some(bytes) => LiteralExpression::Binary(bytes),
            None => return Err(assertion()),
        },
        DataType::Date => {
            // A datetime is a date too, and its date is the one of the literal.
            if !is_instance_of_class(obj, "datetime", "date")? {
                return Err(assertion());
            }
            let ordinal: i64 = obj.call_method0("toordinal")?.extract()?;
            // 719163 is the ordinal of 1970-01-01.
            LiteralExpression::Date((ordinal - 719163i64) as i32)
        }
        DataType::Time { precision } => {
            if !is_instance_of_class(obj, "datetime", "time")? {
                return Err(assertion());
            }
            LiteralExpression::Time {
                nano: crate::infer::time_nanos(obj)?,
                precision: *precision,
            }
        }
        DataType::Timestamp => {
            if !is_instance_of_class(obj, "datetime", "datetime")? {
                return Err(assertion());
            }
            LiteralExpression::Timestamp(crate::session::datetime_to_micros(obj)?)
        }
        DataType::TimestampNtz => {
            if !is_instance_of_class(obj, "datetime", "datetime")? {
                return Err(assertion());
            }
            LiteralExpression::TimestampNtz(crate::infer::wall_clock_micros(obj)?)
        }
        DataType::DayTimeInterval { .. } => {
            if !is_instance_of_class(obj, "datetime", "timedelta")? {
                return Err(assertion());
            }
            LiteralExpression::DayTimeInterval(crate::infer::timedelta_micros(obj)?)
        }
        // Another type is not a value of a list's elements, as PySpark's asserts take it.
        _ => return Err(assertion()),
    };
    Ok(literal)
}

/// The value of `obj` as an integer within `min..=max`, for an integer type of PySpark's
/// `LiteralExpression`: a Python int (a bool included), or a NumPy int of one of `numpy`.
fn integral_value(obj: &Bound<'_, PyAny>, numpy: &[&str], min: i64, max: i64) -> PyResult<i64> {
    let assertion = || pyo3::exceptions::PyAssertionError::new_err(());
    if !(obj.is_instance_of::<pyo3::types::PyInt>() || is_numpy_scalar(obj, numpy)?) {
        return Err(assertion());
    }
    match obj.extract::<i64>() {
        Ok(value) if (min..=max).contains(&value) => Ok(value),
        _ => Err(assertion()),
    }
}

/// Whether `obj` is an instance of the class `name` of the module `module`.
fn is_instance_of_class(obj: &Bound<'_, PyAny>, module: &str, name: &str) -> PyResult<bool> {
    let class = obj.py().import(module)?.getattr(name)?;
    obj.is_instance(&class)
}

/// Whether `obj` is an instance of one of the NumPy scalar classes `classes`, such as `int8`.
/// NumPy is looked up in `sys.modules`, as a NumPy scalar exists only when NumPy is imported.
fn is_numpy_scalar(obj: &Bound<'_, PyAny>, classes: &[&str]) -> PyResult<bool> {
    let modules = obj.py().import("sys")?.getattr("modules")?;
    let numpy = match modules.cast::<pyo3::types::PyDict>()?.get_item("numpy")? {
        Some(numpy) if !numpy.is_none() => numpy,
        _ => return Ok(false),
    };
    for class in classes {
        if obj.is_instance(&numpy.getattr(*class)?)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The literal of a value that is not a list, as `to_column` builds it.
fn scalar_literal_of(obj: &Bound<'_, PyAny>) -> PyResult<LiteralExpression> {
    match to_column(obj)?.expression().kind() {
        ExpressionKind::Literal(literal) => Ok(literal.clone()),
        _ => {
            let type_name = obj
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<unknown>".to_string());
            Err(crate::column::raise_pyspark(
                obj.py(),
                "PySparkTypeError",
                "UNSUPPORTED_DATA_TYPE",
                &[("data_type", &type_name)],
            ))
        }
    }
}

/// The data type of a literal, as the type of PySpark's `LiteralExpression` of it.
fn literal_data_type(literal: &LiteralExpression) -> spark_connect::types::DataType {
    use spark_connect::types::DataType;
    match literal {
        LiteralExpression::Null(data_type) => data_type.as_ref().clone(),
        LiteralExpression::Boolean(_) => DataType::Boolean,
        LiteralExpression::Byte(_) => DataType::Byte,
        LiteralExpression::Short(_) => DataType::Short,
        LiteralExpression::Integer(_) => DataType::Integer,
        LiteralExpression::Long(_) => DataType::Long,
        LiteralExpression::Float(_) => DataType::Float,
        LiteralExpression::Double(_) => DataType::Double,
        LiteralExpression::Decimal {
            precision, scale, ..
        } => DataType::Decimal {
            precision: *precision,
            scale: *scale,
        },
        LiteralExpression::String(_) => crate::infer::string_type(),
        LiteralExpression::Binary(_) => DataType::Binary,
        LiteralExpression::Date(_) => DataType::Date,
        LiteralExpression::Timestamp(_) => DataType::Timestamp,
        LiteralExpression::TimestampNtz(_) => DataType::TimestampNtz,
        LiteralExpression::DayTimeInterval(_) => DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        },
        // Not of a Python value, which has no year-month interval type.
        LiteralExpression::YearMonthInterval(_) => DataType::YearMonthInterval {
            start_field: 0,
            end_field: 1,
        },
        LiteralExpression::Time { precision, .. } => DataType::Time {
            precision: *precision,
        },
        LiteralExpression::Array { element_type, .. } => DataType::Array {
            element_type: element_type.clone(),
            contains_null: true,
        },
    }
}

/// The number of dimensions of `obj` if it is a `numpy.ndarray`. NumPy is looked up in
/// `sys.modules`, as an array exists only when it is imported, so this does not import it.
fn numpy_ndarray_ndim(obj: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    let py = obj.py();
    let modules = py.import("sys")?.getattr("modules")?;
    match modules.cast::<pyo3::types::PyDict>()?.get_item("numpy")? {
        Some(numpy) if !numpy.is_none() && obj.is_instance(&numpy.getattr("ndarray")?)? => {
            Ok(Some(obj.getattr("ndim")?.extract()?))
        }
        _ => Ok(None),
    }
}

/// The literal of a NumPy scalar, typed as PySpark's `LiteralExpression._infer_type` types it: a
/// bool, a signed int of 8 to 64 bits (tinyint to bigint), or a float32 or float64. A str or bytes
/// scalar is `None`, for the string and binary conversions to take. Any other NumPy scalar (an
/// unsigned int, float16, complex, datetime64 or timedelta64) is UNSUPPORTED_DATA_TYPE,
/// named by its Python type. A supported dtype with a scalar class outside PySpark's accepted
/// classes raises a bare `AssertionError`, such as a distinct `longlong`, or an eight-byte
/// `longdouble` on macOS arm64 and Windows. Wider `longdouble` dtypes are unsupported.
/// NumPy is looked up in `sys.modules`, as a NumPy scalar exists only when NumPy is imported.
fn numpy_scalar_literal(obj: &Bound<'_, PyAny>) -> PyResult<Option<LiteralExpression>> {
    let py = obj.py();
    let modules = py.import("sys")?.getattr("modules")?;
    let numpy = match modules.cast::<pyo3::types::PyDict>()?.get_item("numpy")? {
        Some(numpy) if !numpy.is_none() => numpy,
        _ => return Ok(None),
    };
    if !obj.is_instance(&numpy.getattr("generic")?)? {
        return Ok(None);
    }
    if obj.is_instance(&numpy.getattr("bool_")?)? {
        return Ok(Some(LiteralExpression::Boolean(obj.is_truthy()?)));
    }
    let dtype = obj.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    // Mirror LiteralExpression's accepted scalar classes after dtype inference. Aliases can
    // share a supported dtype without being instances of its canonical NumPy scalar class.
    // Check the matching width first; the remaining classes preserve the reference's full set.
    let scalar_classes: &[&str] = match (kind.as_str(), itemsize) {
        ("i", 1) => &["int8"],
        ("i", 2) => &["int16", "int8"],
        ("i", 4) => &["int32", "int16", "int8"],
        ("i", 8) => &["int64", "int32", "int16", "int8"],
        ("f", 4) => &["float32"],
        ("f", 8) => &["float64", "float32"],
        _ => &[],
    };
    if !scalar_classes.is_empty() {
        let mut accepted = match kind.as_str() {
            "i" => obj.is_instance_of::<pyo3::types::PyInt>(),
            "f" => obj.is_instance_of::<pyo3::types::PyFloat>(),
            _ => false,
        };
        if !accepted {
            for class in scalar_classes {
                if obj.is_instance(&numpy.getattr(*class)?)? {
                    accepted = true;
                    break;
                }
            }
        }
        if !accepted {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
    }
    let literal = match (kind.as_str(), itemsize) {
        ("U", _) | ("S", _) => return Ok(None),
        ("i", 1) => LiteralExpression::Byte(obj.extract::<i64>()? as i32),
        ("i", 2) => LiteralExpression::Short(obj.extract::<i64>()? as i32),
        ("i", 4) => LiteralExpression::Integer(obj.extract::<i64>()? as i32),
        ("i", 8) => LiteralExpression::Long(obj.extract::<i64>()?),
        ("f", 4) => LiteralExpression::Float(obj.extract::<f64>()? as f32),
        ("f", 8) => LiteralExpression::Double(obj.extract::<f64>()?),
        _ => {
            let type_name = obj
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<unknown>".to_string());
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "UNSUPPORTED_DATA_TYPE",
                &[("data_type", &type_name)],
            ));
        }
    };
    Ok(Some(literal))
}

/// Build a literal for the non-primitive scalar Python types (`datetime`, `date`,
/// `decimal.Decimal`, `bytes`/`bytearray`). Returns `None` if `obj` is not one of them,
/// so the caller falls through to the primitive int/float/str/bool checks. Kept in sync
/// with `session::py_to_value` so `lit(..)` and `createDataFrame(..)` encode identically.
fn scalar_literal(obj: &Bound<'_, PyAny>) -> PyResult<Option<LiteralExpression>> {
    let py = obj.py();

    // decimal.Decimal -> Decimal literal (before int/float: Decimal has __float__/__int__).
    // As PySpark's LiteralExpression of a Decimal, of DecimalType(): precision 10 and scale
    // 0, which the server widens to those of the value, so lit(Decimal("1.5")) is a
    // decimal(10,1).
    if let Ok(dec_mod) = py.import("decimal") {
        let decimal_cls = dec_mod.getattr("Decimal")?;
        if obj.is_instance(&decimal_cls)? {
            return Ok(Some(LiteralExpression::Decimal {
                value: obj.str()?.extract()?,
                precision: 10,
                scale: 0,
            }));
        }
    }

    // datetime.datetime -> Timestamp (checked before datetime.date, which it subclasses,
    // and before int, since a pandas Timestamp's __int__ returns nanoseconds).
    if let Ok(dt_mod) = py.import("datetime") {
        let datetime_cls = dt_mod.getattr("datetime")?;
        if obj.is_instance(&datetime_cls)? {
            // As PySpark's `LiteralExpression._infer_type`: a TIMESTAMP_NTZ of the wall clock
            // when the active session prefers it (`spark.sql.timestampType`), which PySpark reads
            // for each datetime literal, else a TIMESTAMP of the instant.
            if py
                .import("pyspark.sql.utils")?
                .call_method0("is_timestamp_ntz_preferred")?
                .is_truthy()?
            {
                return Ok(Some(LiteralExpression::TimestampNtz(
                    crate::infer::wall_clock_micros(obj)?,
                )));
            }
            // Shared with session::py_to_value so lit() and createDataFrame() agree.
            let micros = crate::session::datetime_to_micros(obj)?;
            return Ok(Some(LiteralExpression::Timestamp(micros)));
        }
        // datetime.time -> TimeType(), as PySpark's literal of one: its nanoseconds since
        // midnight, of the default precision.
        if obj.is_instance(&dt_mod.getattr("time")?)? {
            return Ok(Some(LiteralExpression::Time {
                nano: crate::infer::time_nanos(obj)?,
                precision: 6,
            }));
        }
        // datetime.timedelta -> DayTimeIntervalType, as PySpark's literal of one.
        if obj.is_instance(&dt_mod.getattr("timedelta")?)? {
            let micros = crate::infer::timedelta_micros(obj)?;
            return Ok(Some(LiteralExpression::DayTimeInterval(micros)));
        }
        let date_cls = dt_mod.getattr("date")?;
        if obj.is_instance(&date_cls)? {
            let ordinal: i64 = obj.call_method0("toordinal")?.extract()?;
            // 719163 is the ordinal of 1970-01-01.
            let days = (ordinal - 719163i64) as i32;
            return Ok(Some(LiteralExpression::Date(days)));
        }
    }

    // bytes / bytearray -> Binary. Checked explicitly so it is not mistaken for anything else.
    // PySpark's `to_plan` sends `bytes(value)`, which honors the `__bytes__` of a subclass.
    if let Some(bytes) = crate::infer::bytes_of(obj)? {
        return Ok(Some(LiteralExpression::Binary(bytes)));
    }

    Ok(None)
}

#[pyfunction]
fn pyfunc_col(name: &str) -> PyResult<PyColumn> {
    // As PySpark's `col`, of `_with_origin`.
    let _origin = crate::origin::with_origin("col")?;
    Ok(PyColumn::new(spark_funcs::col(name)))
}

#[pyfunction]
fn pyfunc_lit(_py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
    let col = to_column(value)?;
    Ok(PyColumn::new(col))
}

#[pyfunction]
fn pyfunc_expr(expr_str: &str) -> PyColumn {
    PyColumn::new(spark_funcs::expr(expr_str))
}

#[pyfunction]
fn pyfunc_sum(col: &PyColumn) -> PyColumn {
    PyColumn::new(spark_funcs::sum(col.column.clone()))
}

/// `count(col)`, as PySpark Connect's: of a Column or a column name, else NOT_EXPECTED_TYPE,
/// as its `_to_col`; a star Column, as `col("*")` or `df["*"]`, counts the rows, as
/// `count(1)`, while a name is the column of it, so that `count("*")` is the count of a star.
#[pyfunction]
fn pyfunc_count(col: &Bound<'_, PyAny>) -> PyResult<PyColumn> {
    let column = if let Ok(column) = col.extract::<PyRef<'_, PyColumn>>() {
        if matches!(
            column.column.expression().kind(),
            ExpressionKind::UnresolvedStar(..)
        ) {
            spark_connect::column::lit(1)
        } else {
            column.column.clone()
        }
    } else if col.is_instance_of::<pyo3::types::PyString>() {
        // As PySpark's `_to_col`, by `col`, of its origin.
        let _origin = crate::origin::with_origin("col")?;
        spark_funcs::col(&col.extract::<String>()?)
    } else {
        let type_name = col.get_type().name()?.to_string();
        return Err(crate::column::raise_pyspark(
            col.py(),
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("expected_type", "Column or str"),
                ("arg_name", "col"),
                ("arg_type", &type_name),
            ],
        ));
    };
    Ok(PyColumn::new(spark_funcs::count(column)))
}

#[pyfunction]
fn pyfunc_avg(col: &PyColumn) -> PyColumn {
    PyColumn::new(spark_funcs::avg(col.column.clone()))
}

#[pyfunction]
fn pyfunc_max(col: &PyColumn) -> PyColumn {
    PyColumn::new(spark_funcs::max(col.column.clone()))
}

#[pyfunction]
fn pyfunc_min(col: &PyColumn) -> PyColumn {
    PyColumn::new(spark_funcs::min(col.column.clone()))
}

#[pyfunction]
fn pyfunc_when(condition: &PyColumn, value: &PyColumn) -> PyColumn {
    PyColumn::new(spark_funcs::when(
        condition.column.clone(),
        value.column.clone(),
    ))
}

/// Generic dispatch function for all 440 SQL functions.
/// Takes a function name and variadic arguments, calls the appropriate Rust function.
#[pyfunction]
#[pyo3(signature = (name, *args))]
fn pyfunc_call_function(name: String, args: Vec<Bound<'_, PyAny>>) -> PyResult<PyColumn> {
    // Convert all Python arguments to Columns
    let mut col_args = Vec::new();
    for arg in args {
        col_args.push(to_column(&arg)?);
    }

    // Call the dispatch function
    let result_col = dispatch::call_builtin(&name, col_args)?;
    Ok(PyColumn::new(result_col))
}

// ============================================================================
// Higher-Order Function (HOF) primitives
//
// The lambda machinery mirrors pyspark.sql.connect.functions._create_lambda /
// _invoke_higher_order_function: the Python wrapper (functions.py) generates fresh
// variable names, builds placeholder Columns via `pyfunc_named_lambda_variable`,
// invokes the user's lambda to obtain the body Column, then wraps it with
// `pyfunc_lambda_function` and passes it as an argument to `pyfunc_call_function`.
// ============================================================================

use spark_connect::expression::{
    CallFunctionWrapper, LambdaFunction, UnresolvedNamedLambdaVariable,
};

/// Build a Column wrapping an `UnresolvedNamedLambdaVariable` with the given name.
/// Mirrors `UnresolvedNamedLambdaVariable([name])` on the Python side.
#[pyfunction]
fn pyfunc_named_lambda_variable(name: String) -> PyColumn {
    PyColumn::new(Column::new(
        ExpressionSpec::UnresolvedNamedLambdaVariable(UnresolvedNamedLambdaVariable::new(name))
            .into(),
    ))
}

/// Build a Column wrapping a `LambdaFunction` from a body Column and its argument
/// variable names. Mirrors `LambdaFunction(body._expr, arg_exprs)`.
#[pyfunction]
fn pyfunc_lambda_function(body: &PyColumn, arg_names: Vec<String>) -> PyColumn {
    let args = arg_names
        .into_iter()
        .map(UnresolvedNamedLambdaVariable::new)
        .collect();
    let lambda = LambdaFunction::new(body.column.expression().clone(), args);
    PyColumn::new(Column::new(
        ExpressionSpec::LambdaFunction(Box::new(lambda)).into(),
    ))
}

/// Build a Column wrapping a `CallFunction` expression (a named function call
/// carrying its argument columns). Mirrors `functions.call_function`.
#[pyfunction]
#[pyo3(signature = (name, *args))]
fn pyfunc_call_named_function(name: String, args: Vec<Bound<'_, PyAny>>) -> PyResult<PyColumn> {
    let mut arg_exprs = Vec::with_capacity(args.len());
    for arg in args {
        arg_exprs.push(to_column(&arg)?.expression().clone());
    }
    let result = Column::new(
        ExpressionSpec::CallFunction(Box::new(CallFunctionWrapper::new(name, arg_exprs))).into(),
    );
    Ok(PyColumn::new(result))
}

/// Build a Column wrapping an `UnresolvedFunction` for ANY name, unconditionally
/// (not gated by the builtin dispatch allowlist). Mirrors
/// `pyspark.sql.connect.functions._invoke_function(name, *args)` — used for
/// higher-order functions, `call_udf`, and functions like `cume_dist` that are
/// not part of the generated dispatch.
#[pyfunction]
#[pyo3(signature = (name, *args))]
fn pyfunc_invoke_function(name: String, args: Vec<Bound<'_, PyAny>>) -> PyResult<PyColumn> {
    let mut arg_exprs = Vec::with_capacity(args.len());
    for arg in args {
        arg_exprs.push(to_column(&arg)?.expression().clone());
    }
    let result = Column::new(Expression::function(name, arg_exprs));
    Ok(PyColumn::new(result))
}

// ============================================================================
// Mixed/special functions - dedicated bindings for non-generic dispatch
// ============================================================================

/// Wrapper for sha2(col, numbits).
#[pyfunction]
fn pyfunc_sha2(col: &PyColumn, numbits: i32) -> PyColumn {
    PyColumn::new(spark_funcs::sha2(col.column.clone(), numbits))
}

/// Wrapper for variant_strip_nulls(v, include_arrays) - bool second arg.
#[pyfunction]
fn pyfunc_variant_strip_nulls(v: &PyColumn, include_arrays: bool) -> PyColumn {
    PyColumn::new(spark_funcs::variant_strip_nulls(
        v.column.clone(),
        include_arrays,
    ))
}

/// Wrapper for window(col, window_duration).
#[pyfunction]
fn pyfunc_window(col: &PyColumn, window_duration: &str) -> PyColumn {
    PyColumn::new(spark_funcs::window(col.column.clone(), window_duration))
}

/// Wrapper for window(col, window_duration, slide_duration, start_time).
#[pyfunction]
fn pyfunc_window_with_slide_and_start(
    col: &PyColumn,
    window_duration: &str,
    slide_duration: &str,
    start_time: &str,
) -> PyColumn {
    PyColumn::new(spark_funcs::window_with_slide_and_start(
        col.column.clone(),
        window_duration,
        slide_duration,
        start_time,
    ))
}

/// Wrapper for from_avro(data, jsonFormatSchema).
#[pyfunction]
fn pyfunc_from_avro(data: &PyColumn, json_format_schema: &str) -> PyColumn {
    PyColumn::new(spark_funcs::from_avro(
        data.column.clone(),
        json_format_schema,
    ))
}

/// Wrapper for from_avro_with_options(data, jsonFormatSchema, options).
#[pyfunction]
fn pyfunc_from_avro_with_options(
    data: &PyColumn,
    json_format_schema: &str,
    options: &PyColumn,
) -> PyColumn {
    PyColumn::new(spark_funcs::from_avro_with_options(
        data.column.clone(),
        json_format_schema,
        options.column.clone(),
    ))
}

/// Wrapper for to_avro_with_schema(data, jsonFormatSchema).
#[pyfunction]
fn pyfunc_to_avro_with_schema(data: &PyColumn, json_format_schema: &str) -> PyColumn {
    PyColumn::new(spark_funcs::to_avro_with_schema(
        data.column.clone(),
        json_format_schema,
    ))
}

/// Wrapper for from_protobuf(data, messageName).
#[pyfunction]
fn pyfunc_from_protobuf(data: &PyColumn, message_name: &str) -> PyColumn {
    PyColumn::new(spark_funcs::from_protobuf(
        data.column.clone(),
        message_name,
    ))
}

/// Wrapper for from_protobuf_with_descriptor(data, messageName, binaryDescriptorSet).
#[pyfunction]
fn pyfunc_from_protobuf_with_descriptor(
    data: &PyColumn,
    message_name: &str,
    binary_descriptor_set: Vec<u8>,
) -> PyColumn {
    PyColumn::new(spark_funcs::from_protobuf_with_descriptor(
        data.column.clone(),
        message_name,
        binary_descriptor_set,
    ))
}

/// Wrapper for from_protobuf_with_options(data, messageName, options).
#[pyfunction]
fn pyfunc_from_protobuf_with_options(
    data: &PyColumn,
    message_name: &str,
    options: &PyColumn,
) -> PyColumn {
    PyColumn::new(spark_funcs::from_protobuf_with_options(
        data.column.clone(),
        message_name,
        options.column.clone(),
    ))
}

/// Wrapper for from_protobuf_with_descriptor_and_options(data, messageName, binaryDescriptorSet, options).
#[pyfunction]
fn pyfunc_from_protobuf_with_descriptor_and_options(
    data: &PyColumn,
    message_name: &str,
    binary_descriptor_set: Vec<u8>,
    options: &PyColumn,
) -> PyColumn {
    PyColumn::new(spark_funcs::from_protobuf_with_descriptor_and_options(
        data.column.clone(),
        message_name,
        binary_descriptor_set,
        options.column.clone(),
    ))
}

/// Wrapper for to_protobuf(data, messageName).
#[pyfunction]
fn pyfunc_to_protobuf(data: &PyColumn, message_name: &str) -> PyColumn {
    PyColumn::new(spark_funcs::to_protobuf(data.column.clone(), message_name))
}

/// Wrapper for to_protobuf_with_descriptor(data, messageName, binaryDescriptorSet).
#[pyfunction]
fn pyfunc_to_protobuf_with_descriptor(
    data: &PyColumn,
    message_name: &str,
    binary_descriptor_set: Vec<u8>,
) -> PyColumn {
    PyColumn::new(spark_funcs::to_protobuf_with_descriptor(
        data.column.clone(),
        message_name,
        binary_descriptor_set,
    ))
}

/// Wrapper for to_protobuf_with_options(data, messageName, options).
#[pyfunction]
fn pyfunc_to_protobuf_with_options(
    data: &PyColumn,
    message_name: &str,
    options: &PyColumn,
) -> PyColumn {
    PyColumn::new(spark_funcs::to_protobuf_with_options(
        data.column.clone(),
        message_name,
        options.column.clone(),
    ))
}

/// Wrapper for to_protobuf_with_descriptor_and_options(data, messageName, binaryDescriptorSet, options).
#[pyfunction]
fn pyfunc_to_protobuf_with_descriptor_and_options(
    data: &PyColumn,
    message_name: &str,
    binary_descriptor_set: Vec<u8>,
    options: &PyColumn,
) -> PyColumn {
    PyColumn::new(spark_funcs::to_protobuf_with_descriptor_and_options(
        data.column.clone(),
        message_name,
        binary_descriptor_set,
        options.column.clone(),
    ))
}

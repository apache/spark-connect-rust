//! PyO3 wrappers for Spark SQL functions.

use pyo3::prelude::*;
use spark_connect::column::Column;
use spark_connect::expression::{Expression, ExpressionNode, LiteralExpression, NamedArgument};
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
        ExpressionNode::CommonInlineUserDefinedFunction(Box::new(udf_expr)).into(),
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
            ExpressionNode::NamedArgument(Box::new(NamedArgument::new(key, value))).into(),
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

/// Convert a Python object to a Rust Column (handles Column or scalars that become literals).
pub fn to_column(obj: &Bound<'_, PyAny>) -> PyResult<Column> {
    // Try to get as PyColumn first
    if let Ok(col) = obj.extract::<Bound<'_, PyColumn>>() {
        return Ok(col.borrow().column.clone());
    }

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

    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Column::new(Expression::literal(
            if i32::try_from(i).is_ok() {
                LiteralExpression::int(i as i32)
            } else {
                LiteralExpression::long(i)
            },
        )));
    }

    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Column::new(Expression::literal(LiteralExpression::double(
            f,
        ))));
    }

    if let Ok(s) = obj.extract::<String>() {
        return Ok(Column::new(Expression::literal(LiteralExpression::string(
            s,
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
            // Shared with session::py_to_value so lit() and createDataFrame() agree (a pandas
            // Timestamp's .timestamp() treats a naive value as UTC; normalize it first).
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
    if obj.is_instance_of::<pyo3::types::PyBytes>()
        || obj.is_instance_of::<pyo3::types::PyByteArray>()
    {
        let bytes: Vec<u8> = obj.extract()?;
        return Ok(Some(LiteralExpression::Binary(bytes)));
    }

    Ok(None)
}

#[pyfunction]
fn pyfunc_col(name: &str) -> PyColumn {
    // As PySpark's `col`, of `_with_origin`.
    let _origin = crate::origin::with_origin("col");
    PyColumn::new(spark_funcs::col(name))
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

#[pyfunction]
fn pyfunc_count(col: &PyColumn) -> PyColumn {
    PyColumn::new(spark_funcs::count(col.column.clone()))
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
        ExpressionNode::UnresolvedNamedLambdaVariable(UnresolvedNamedLambdaVariable::new(name))
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
        ExpressionNode::LambdaFunction(Box::new(lambda)).into(),
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
        ExpressionNode::CallFunction(Box::new(CallFunctionWrapper::new(name, arg_exprs))).into(),
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

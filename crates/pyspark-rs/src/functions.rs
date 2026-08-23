//! PyO3 wrappers for Spark SQL functions.

use pyo3::prelude::*;
use spark_connect::column::Column;
use spark_connect::expression::{Expression, LiteralExpression};
use spark_connect::functions as spark_funcs;
use spark_connect::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

use crate::column::PyColumn;
use crate::types::PyDataType;

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
#[pyfunction]
#[pyo3(signature = (name, return_type, eval_type, command_bytes, python_ver, *args))]
fn pyfunc_make_udf(
    name: String,
    return_type: &PyDataType,
    eval_type: i32,
    command_bytes: Vec<u8>,
    python_ver: String,
    args: Vec<Bound<'_, PyAny>>,
) -> PyResult<PyColumn> {
    // Convert all arguments to Columns
    let mut col_args = Vec::new();
    for arg in args {
        col_args.push(to_column(&arg)?);
    }

    // Convert argument columns to expressions
    let arg_exprs: Vec<Expression> = col_args
        .into_iter()
        .map(|c| c.expression().clone())
        .collect();

    // Create the Python UDF payload
    let python_udf = PythonUDFPayload::new(
        return_type.inner.clone(),
        eval_type,
        command_bytes,
        python_ver,
    );

    // Create the UDF expression
    let udf_expr = CommonInlineUserDefinedFunctionExpression::new(
        name, true, // deterministic by default
        arg_exprs, python_udf,
    );

    Ok(PyColumn::new(Column::new(
        Expression::CommonInlineUserDefinedFunction(Box::new(udf_expr)),
    )))
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
    m.add_function(wrap_pyfunction!(pyfunc_transform, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_filter, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_exists, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_forall, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_zip_with, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_transform_keys, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_transform_values, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_map_filter, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunc_map_zip_with, m)?)?;
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
        return Ok(Column::new(Expression::Literal(LiteralExpression::null(
            spark_connect::types::DataType::Null,
        ))));
    }

    // Try to convert scalars to literals
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Column::new(Expression::Literal(
            LiteralExpression::boolean(b),
        )));
    }

    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Column::new(Expression::Literal(
            if i32::try_from(i).is_ok() {
                LiteralExpression::int(i as i32)
            } else {
                LiteralExpression::long(i)
            },
        )));
    }

    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Column::new(Expression::Literal(LiteralExpression::double(
            f,
        ))));
    }

    if let Ok(s) = obj.extract::<String>() {
        return Ok(Column::new(Expression::Literal(LiteralExpression::string(
            s,
        ))));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Expected Column or scalar (int, float, str, bool, None)",
    ))
}

#[pyfunction]
fn pyfunc_col(name: &str) -> PyColumn {
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
// Higher-Order Functions (HOF)
// ============================================================================

use spark_connect::expression::{LambdaFunction, UnresolvedNamedLambdaVariable};

/// Wrapper for transform(col, lambda) - accepts a body Column as-is (already built by Python wrapper).
#[pyfunction]
fn pyfunc_transform(col: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![UnresolvedNamedLambdaVariable::new("x")],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "transform",
            vec![
                col.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for filter(col, lambda).
#[pyfunction]
fn pyfunc_filter(col: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![UnresolvedNamedLambdaVariable::new("x")],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "filter",
            vec![
                col.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for exists(col, lambda).
#[pyfunction]
fn pyfunc_exists(col: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![UnresolvedNamedLambdaVariable::new("x")],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "exists",
            vec![
                col.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for forall(col, lambda).
#[pyfunction]
fn pyfunc_forall(col: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![UnresolvedNamedLambdaVariable::new("x")],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "forall",
            vec![
                col.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for aggregate(col, init, lambda).
#[pyfunction]
fn pyfunc_aggregate(col: &PyColumn, init: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![
            UnresolvedNamedLambdaVariable::new("acc"),
            UnresolvedNamedLambdaVariable::new("x"),
        ],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "aggregate",
            vec![
                col.column.expression().clone(),
                init.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for zip_with(col1, col2, lambda).
#[pyfunction]
fn pyfunc_zip_with(col1: &PyColumn, col2: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![
            UnresolvedNamedLambdaVariable::new("x"),
            UnresolvedNamedLambdaVariable::new("y"),
        ],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "zip_with",
            vec![
                col1.column.expression().clone(),
                col2.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for transform_keys(col, lambda).
#[pyfunction]
fn pyfunc_transform_keys(col: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![
            UnresolvedNamedLambdaVariable::new("k"),
            UnresolvedNamedLambdaVariable::new("v"),
        ],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "transform_keys",
            vec![
                col.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for transform_values(col, lambda).
#[pyfunction]
fn pyfunc_transform_values(col: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![
            UnresolvedNamedLambdaVariable::new("k"),
            UnresolvedNamedLambdaVariable::new("v"),
        ],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "transform_values",
            vec![
                col.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for map_filter(col, lambda).
#[pyfunction]
fn pyfunc_map_filter(col: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![
            UnresolvedNamedLambdaVariable::new("k"),
            UnresolvedNamedLambdaVariable::new("v"),
        ],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "map_filter",
            vec![
                col.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

/// Wrapper for map_zip_with(col1, col2, lambda).
#[pyfunction]
fn pyfunc_map_zip_with(col1: &PyColumn, col2: &PyColumn, body_col: &PyColumn) -> PyColumn {
    let lambda = LambdaFunction::new(
        body_col.column.expression().clone(),
        vec![
            UnresolvedNamedLambdaVariable::new("k"),
            UnresolvedNamedLambdaVariable::new("v1"),
            UnresolvedNamedLambdaVariable::new("v2"),
        ],
    );
    let result = Column::new(Expression::UnresolvedFunction(
        spark_connect::expression::UnresolvedFunction::new(
            "map_zip_with",
            vec![
                col1.column.expression().clone(),
                col2.column.expression().clone(),
                Expression::LambdaFunction(Box::new(lambda)),
            ],
        ),
    ));
    PyColumn::new(result)
}

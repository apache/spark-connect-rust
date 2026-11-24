//! The arguments that PySpark's client keeps as the Python objects that it was given, and sets in
//! the protobuf of a plan only when a request builds the plan: the names, SQL strings and literal
//! strings of its Columns, and the names, counts and other values of its DataFrames. protobuf
//! converts them then, so a value that it does not take, such as a str that UTF-8 cannot encode,
//! an int beyond its field or an object of another type, fails each request of the plan, such as
//! `df.schema` or `collect()`, and not the call that took it.
//!
//! [`PlanArgs`] converts such values as protobuf does, in the order in which PySpark sets them,
//! and keeps the error of the first that does not convert, with a value of this client in its
//! place. The plan or the expression that it builds of them carries that error
//! ([`PlanArgs::plan`], [`PlanArgs::expression`]), and the core raises it when a request builds
//! the plan, after the errors of the plans and expressions that PySpark builds before it (see
//! `LogicalPlan::Unbuildable` and `Expression::unbuildable`).

use pyo3::prelude::*;
use pyo3::types::PyString;
use spark_connect::column::Column;
use spark_connect::dataframe::DataFrame;
use spark_connect::expression::{Expression, ExpressionNode, LiteralExpression};
use spark_connect_core::error::SparkError;

/// The values of a plan or of an expression that PySpark sets when a request builds the plan,
/// converted as protobuf converts them, and the error of the first that protobuf refuses.
#[derive(Default)]
pub(crate) struct PlanArgs {
    error: Option<SparkError>,
}

impl PlanArgs {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record `error`, which PySpark raises when it builds the plan, unless an earlier value
    /// failed: the request raises the first.
    pub(crate) fn record(&mut self, error: SparkError) {
        self.error.get_or_insert(error);
    }

    /// The error of the first value that does not convert, if one does not.
    pub(crate) fn error(&self) -> Option<&SparkError> {
        self.error.as_ref()
    }

    /// `value` of a protobuf string field, as upb converts it: a str, or bytes of UTF-8. A str
    /// that UTF-8 cannot encode, bytes that are not UTF-8, or another object, is recorded with
    /// the error that protobuf raises for it ([`crate::dataframe::unbuildable_str_error`]), and
    /// is [`rendered`] in its place.
    pub(crate) fn str(&mut self, value: &Bound<'_, PyAny>) -> PyResult<String> {
        match crate::dataframe::unbuildable_str_error(value)? {
            None => crate::str_argument(value),
            Some(error) => {
                self.record(error);
                rendered(value)
            }
        }
    }

    /// The items of `values`, of a repeated protobuf string field, each as [`PlanArgs::str`]
    /// converts it.
    pub(crate) fn strs<'py>(
        &mut self,
        values: impl IntoIterator<Item = Bound<'py, PyAny>>,
    ) -> PyResult<Vec<String>> {
        values.into_iter().map(|value| self.str(&value)).collect()
    }

    /// `value` of the protobuf int field `field`, of the type `T`, as
    /// [`crate::stat_arguments::proto_int`] converts it; one that does not convert is recorded,
    /// with 0 in its place.
    pub(crate) fn int<T: TryFrom<i64> + Default>(
        &mut self,
        value: &Bound<'_, PyAny>,
        field: &str,
    ) -> PyResult<T> {
        let converted = crate::stat_arguments::proto_int(value, field);
        self.converted(value.py(), converted)
    }

    /// `value` of a protobuf double field, as upb converts it, by `PyFloat_AsDouble`: so a str
    /// raises the `TypeError` "must be real number, not str", which is recorded, with 0 in its
    /// place.
    pub(crate) fn double(&mut self, value: &Bound<'_, PyAny>) -> PyResult<f64> {
        let converted = value.extract::<f64>();
        self.converted(value.py(), converted)
    }

    /// The value of a conversion of protobuf: its error, a `TypeError`, an `OverflowError` or a
    /// `ValueError`, is recorded as [`crate::dataframe::unbuildable_error`] gives it, with the
    /// default of `T` in the place of the value. Another error is raised at the call.
    pub(crate) fn converted<T: Default>(
        &mut self,
        py: Python<'_>,
        converted: PyResult<T>,
    ) -> PyResult<T> {
        match converted {
            Ok(value) => Ok(value),
            Err(err) => {
                self.record(crate::dataframe::unbuildable_error(py, err)?);
                Ok(T::default())
            }
        }
    }

    /// Record `err`, an exception that PySpark raises when it builds the plan: an
    /// `AttributeError`, a `TypeError`, a `ValueError` or an `OverflowError` of Python itself, the
    /// last three as [`PlanArgs::converted`] takes them, a `RecursionError`, such as that of a
    /// type that holds itself, or an exception of one of PySpark's error classes, as
    /// [`crate::errors::pyspark_class_error`] gives it. Another, such as a `KeyError`, which the
    /// core does not raise again, is given back, for the call to raise.
    pub(crate) fn raised(&mut self, py: Python<'_>, err: PyErr) -> PyResult<()> {
        use pyo3::exceptions::{
            PyAttributeError, PyOverflowError, PyRecursionError, PyTypeError, PyValueError,
        };
        let kind = err.get_type(py);
        if kind.is(py.get_type::<PyAttributeError>()) {
            let message = err.value(py).str()?.to_string();
            self.record(SparkError::builtin_attribute(message));
            return Ok(());
        }
        if kind.is(py.get_type::<PyRecursionError>()) {
            self.record(recursion_error(py, &err)?);
            return Ok(());
        }
        let builtins = [
            py.get_type::<PyTypeError>(),
            py.get_type::<PyValueError>(),
            py.get_type::<PyOverflowError>(),
        ];
        if builtins.iter().any(|builtin| kind.is(builtin)) {
            return self.converted::<()>(py, Err(err));
        }
        match crate::errors::pyspark_class_error(py, &err)? {
            Some(error) => {
                self.record(error);
                Ok(())
            }
            None => Err(err),
        }
    }

    /// The values of a plan that PySpark sets before it builds `expressions`, which follow its
    /// input: the first of `expressions` carries the error, which a request then raises after
    /// the errors of the input and before those of the expressions, and the values are given
    /// back without it. Without expressions, they are given back as they are, for
    /// [`PlanArgs::plan`].
    pub(crate) fn before(mut self, expressions: &mut [Expression]) -> PlanArgs {
        if let Some(first) = expressions.first_mut() {
            if let Some(error) = self.error.take() {
                *first = Expression::unbuildable(first.clone(), Some(error), None, None);
            }
        }
        self
    }

    /// `df`, of the plan that these values are of: each request of it raises the error of the
    /// first value that did not convert, as PySpark's request fails to build the plan. It keeps
    /// the cached schema of `df`, which its operation copies of its input as PySpark's copies
    /// `_cached_schema`, as `alias`, a hint, `dropDuplicates` and `observe` do, so its `schema`
    /// is known without a request then, as in PySpark.
    pub(crate) fn plan(self, df: DataFrame) -> DataFrame {
        match self.error {
            Some(error) => df.unbuildable_with_cached_schema(error),
            None => df,
        }
    }

    /// `expression`, of the expression that these values are of: a request of a plan of it
    /// raises the error of the first value that did not convert, as PySpark's `to_plan` fails
    /// to build it, and a Column of it renders as `expression`, as PySpark's `__repr__` reads the
    /// values as they are.
    pub(crate) fn expression(self, expression: Expression) -> Expression {
        Expression::unbuildable(expression, self.error, None, None)
    }

    /// As [`PlanArgs::expression`], of the expression of `column`.
    pub(crate) fn column(self, column: Column) -> Column {
        match self.error {
            None => column,
            Some(_) => Column::new(self.expression(column.expression().clone())),
        }
    }

    /// As [`PlanArgs::column`], of values that PySpark sets after it builds the expressions
    /// below that of `column`, such as the type of a cast: their errors come after those of
    /// these expressions.
    pub(crate) fn column_after(self, column: Column) -> Column {
        match self.error {
            None => column,
            Some(error) => Column::new(Expression::unbuildable(
                column.expression().clone(),
                None,
                Some(error),
                None,
            )),
        }
    }
}

/// The error of `err`, a `RecursionError`, which the core raises again as it is: of the message of
/// its text.
pub(crate) fn recursion_error(py: Python<'_>, err: &PyErr) -> PyResult<SparkError> {
    Ok(SparkError::recursion_msg(err.value(py).str()?.to_string()))
}

/// The `operation` of PySpark's `UNSUPPORTED_OPERATION` of the data type `data_type`, which it
/// raises when it builds the plan: `f"data type {data_type}"`, of the str of `data_type` as it is,
/// a lone surrogate too, as [`crate::errors::str_param`] holds it.
pub(crate) fn data_type_operation(data_type: &Bound<'_, PyAny>) -> PyResult<String> {
    let operation = PyString::new(data_type.py(), "data type ").add(data_type.str()?)?;
    crate::errors::str_param(operation.cast::<PyString>()?)
}

/// The text of `value` in the place of a value that protobuf does not take: a str with its lone
/// surrogates escaped, as a rendering of UTF-8 holds them (PySpark's `__repr__` shows the str as
/// it is), and nothing for another object, whose text no request sends.
pub(crate) fn rendered(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if value.is_instance_of::<PyString>() {
        return value
            .call_method1("encode", ("utf-8", "backslashreplace"))?
            .call_method1("decode", ("utf-8",))?
            .extract();
    }
    Ok(String::new())
}

/// The str of `value` if it is an exact str of UTF-8, which every field takes as it is: the
/// common case, which needs no conversion of protobuf.
fn utf8_str<'a>(value: &'a Bound<'_, PyAny>) -> Option<std::borrow::Cow<'a, str>> {
    value.cast_exact::<PyString>().ok()?.to_cow().ok()
}

/// The Column of the column name `name`, as PySpark Connect's `F.col`: the star of `"*"`, the star
/// of the columns of a struct for a name that ends with `.*`, else the column of the name. As
/// PySpark, it compares `name` to `"*"` and calls its `endswith(".*")`, so an int or None raises
/// the `AttributeError` of `endswith`, and bytes its `TypeError`; another object is the
/// `AssertionError` of the expression, which asserts a str. A str that UTF-8 cannot encode makes
/// a Column whose requests fail with its `UnicodeEncodeError`, as PySpark sets the name when it
/// builds the plan.
pub(crate) fn col(name: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Some(text) = utf8_str(name) {
        return Ok(spark_connect::functions::col(&text));
    }
    if name.eq("*")? {
        return Ok(spark_connect::functions::col("*"));
    }
    let star = name.call_method1("endswith", (".*",))?.is_truthy()?;
    if !name.is_instance_of::<PyString>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    let mut args = PlanArgs::new();
    let text = args.str(name)?;
    let expression = if star {
        ExpressionNode::UnresolvedStar(Some(text), None).into()
    } else {
        Expression::column(text)
    };
    Ok(Column::new(args.expression(expression)))
}

/// The Column of the column name `name`, as PySpark Connect's `ColumnReference(name)` of the
/// arguments of a UDF, which asserts a str (another object is a bare `AssertionError`) and sets it
/// when it builds the plan, as [`col`], without its star.
pub(crate) fn column_reference(name: &Bound<'_, PyAny>) -> PyResult<Column> {
    if !name.is_instance_of::<PyString>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    let mut args = PlanArgs::new();
    let text = args.str(name)?;
    Ok(Column::new(args.expression(Expression::column(text))))
}

/// The Column of the SQL expression `text`, as PySpark Connect's `F.expr`, whose `SQLExpression`
/// asserts a str: another object is a bare `AssertionError`, and a str that UTF-8 cannot encode
/// fails the requests of its plans, as [`col`].
pub(crate) fn expr(text: &Bound<'_, PyAny>) -> PyResult<Column> {
    if let Some(text) = utf8_str(text) {
        return Ok(spark_connect::functions::expr(&text));
    }
    if !text.is_instance_of::<PyString>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    let mut args = PlanArgs::new();
    let text = args.str(text)?;
    Ok(args.column(spark_connect::functions::expr(&text)))
}

/// The string literal of the str `text`, as PySpark's `LiteralExpression` of a str, which sets
/// it when it builds the plan: a str that UTF-8 cannot encode fails the requests of its plans,
/// as [`col`].
pub(crate) fn string_literal(text: &Bound<'_, PyString>) -> PyResult<Expression> {
    if let Ok(text) = text.to_cow() {
        return Ok(Expression::literal(LiteralExpression::string(
            text.into_owned(),
        )));
    }
    let mut args = PlanArgs::new();
    let text = args.str(text.as_any())?;
    Ok(args.expression(Expression::literal(LiteralExpression::string(text))))
}

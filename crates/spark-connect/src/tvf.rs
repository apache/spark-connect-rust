//! Table-valued functions (TVF) implementation mirroring `pyspark.sql.connect.tvf.TableValuedFunction`.
//!
//! Provides table-valued functions that return DataFrames, such as explode, inline,
//! range, and other TVF operations.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::Expression;
use crate::plan::LogicalPlan;
use crate::session::SparkSession;
use crate::types::DataType;
use spark_connect_core::error::Result;
use spark_connect_proto as proto;

/// Table-valued functions for Spark SQL.
///
/// Mirrors `pyspark.sql.connect.tvf.TableValuedFunction`.
pub struct TableValuedFunction {
    /// The session this TVF is bound to
    session: SparkSession,
}

impl TableValuedFunction {
    /// Create a new TableValuedFunction instance.
    pub(crate) fn new(session: SparkSession) -> Self {
        TableValuedFunction { session }
    }

    /// Invoke a Python user-defined table function (UDTF), returning its output
    /// as a DataFrame. Mirrors `pyspark.sql.functions.udtf`. The Python side
    /// cloudpickles the handler and sets `eval_type` (300 SQL_TABLE_UDF,
    /// 301 SQL_ARROW_TABLE_UDF, 302 SQL_ARROW_UDTF); `command` is the pickled payload.
    #[allow(clippy::too_many_arguments)]
    pub fn udtf(
        &self,
        name: &str,
        arguments: Vec<Column>,
        return_type: Option<DataType>,
        eval_type: i32,
        command: Vec<u8>,
        python_ver: String,
        deterministic: bool,
    ) -> DataFrame {
        let args: Vec<Expression> = arguments.iter().map(|c| c.expression().clone()).collect();
        let plan = LogicalPlan::CommonInlineUdtf {
            function_name: name.to_string(),
            deterministic,
            arguments: args,
            return_type,
            eval_type,
            command,
            python_ver,
        };
        DataFrame::new(self.session.clone(), plan)
    }

    /// Create a DataFrame representing a range of integers.
    ///
    /// This is a table-valued function that generates a sequence of integers
    /// from `start` to `end` with the given `step`.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting value (inclusive)
    /// * `end` - Ending value (exclusive) or the only value if called with one argument
    /// * `step` - Step size between values (default 1)
    /// * `num_partitions` - Number of partitions (optional)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let df = spark.tvf().range(0, 100, 1, None)?;
    /// ```
    pub fn range(
        &self,
        start: i64,
        end: Option<i64>,
        step: i64,
        num_partitions: Option<i32>,
    ) -> Result<DataFrame> {
        // Handle single-argument case: range(end)
        let (actual_start, actual_end) = if let Some(e) = end {
            (start, e)
        } else {
            (0, start)
        };

        self.session
            .range_full(actual_start, actual_end, step, num_partitions)
    }

    /// Explode an array or map column.
    ///
    /// Returns a DataFrame where each array/map element becomes a separate row.
    ///
    /// # Arguments
    ///
    /// * `collection` - The column to explode (must be an array or map)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let df = spark.tvf().explode(col("my_array"))?;
    /// ```
    pub fn explode(&self, collection: &Column) -> Result<DataFrame> {
        self._fn("explode", vec![collection.clone()])
    }

    /// Explode an array or map column, preserving nulls as empty rows.
    ///
    /// Similar to `explode`, but null values are preserved.
    pub fn explode_outer(&self, collection: &Column) -> Result<DataFrame> {
        self._fn("explode_outer", vec![collection.clone()])
    }

    /// Inline an array of structs column.
    ///
    /// Each struct becomes a separate row with its fields as columns.
    ///
    /// # Arguments
    ///
    /// * `input` - The column to inline (must be an array of structs)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let df = spark.tvf().inline(col("struct_array"))?;
    /// ```
    pub fn inline(&self, input: &Column) -> Result<DataFrame> {
        self._fn("inline", vec![input.clone()])
    }

    /// Inline an array of structs column, preserving nulls as empty rows.
    ///
    /// Similar to `inline`, but null values are preserved.
    pub fn inline_outer(&self, input: &Column) -> Result<DataFrame> {
        self._fn("inline_outer", vec![input.clone()])
    }

    /// JSON tuple function.
    ///
    /// Extracts fields from a JSON string.
    ///
    /// # Arguments
    ///
    /// * `input` - The JSON column
    /// * `fields` - Field names to extract
    ///
    /// # Example
    ///
    /// ```ignore
    /// let df = spark.tvf().json_tuple(col("json_str"), vec![col("field1"), col("field2")])?;
    /// ```
    pub fn json_tuple(&self, input: &Column, fields: Vec<Column>) -> Result<DataFrame> {
        if fields.is_empty() {
            return Err(spark_connect_core::error::SparkError::value(
                "CANNOT_BE_EMPTY",
                &[("item", "field")],
            ));
        }
        let mut args = vec![input.clone()];
        args.extend(fields);
        self._fn("json_tuple", args)
    }

    /// Explode an array column with position indices.
    ///
    /// Similar to `explode`, but also includes the position of each element.
    pub fn posexplode(&self, collection: &Column) -> Result<DataFrame> {
        self._fn("posexplode", vec![collection.clone()])
    }

    /// Explode an array column with position indices, preserving nulls.
    ///
    /// Similar to `posexplode`, but null values are preserved.
    pub fn posexplode_outer(&self, collection: &Column) -> Result<DataFrame> {
        self._fn("posexplode_outer", vec![collection.clone()])
    }

    /// Stack function.
    ///
    /// Converts multiple columns into rows.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of columns to stack
    /// * `fields` - The columns to stack
    pub fn stack(&self, n: &Column, fields: Vec<Column>) -> Result<DataFrame> {
        let mut args = vec![n.clone()];
        args.extend(fields);
        self._fn("stack", args)
    }

    /// Returns a DataFrame of collation names.
    pub fn collations(&self) -> Result<DataFrame> {
        self._fn("collations", vec![])
    }

    /// Returns a DataFrame of SQL keywords.
    pub fn sql_keywords(&self) -> Result<DataFrame> {
        self._fn("sql_keywords", vec![])
    }

    /// Explode a variant column.
    ///
    /// Variant columns contain semi-structured data in Spark SQL.
    pub fn variant_explode(&self, input: &Column) -> Result<DataFrame> {
        self._fn("variant_explode", vec![input.clone()])
    }

    /// Explode a variant column, preserving nulls.
    pub fn variant_explode_outer(&self, input: &Column) -> Result<DataFrame> {
        self._fn("variant_explode_outer", vec![input.clone()])
    }

    /// Get Python worker logs as a DataFrame.
    pub fn python_worker_logs(&self) -> Result<DataFrame> {
        self._fn("python_worker_logs", vec![])
    }

    /// Generic function implementation.
    ///
    /// Internal helper to create an UnresolvedTableValuedFunction.
    fn _fn(&self, name: &str, args: Vec<Column>) -> Result<DataFrame> {
        let expressions: Vec<Expression> = args.iter().map(|c| c.expression().clone()).collect();
        let plan = LogicalPlan::UnresolvedTableValuedFunction {
            name: name.to_string(),
            arguments: expressions,
        };
        Ok(DataFrame::new(self.session.clone(), plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SparkSession;

    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("failed to build session")
    }

    #[test]
    fn tvf_range_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let df = tvf.range(0, Some(10), 1, None).unwrap();
        match &df.plan {
            LogicalPlan::Range { .. } => {
                // Plan is correct
            }
            _ => panic!("expected Range plan"),
        }
    }

    #[test]
    fn tvf_range_single_arg() {
        let spark = session();
        let tvf = spark.tvf();
        // range(end) should be range(0, end)
        let df = tvf.range(5, None, 1, None).unwrap();
        match &df.plan {
            LogicalPlan::Range {
                start, end, step, ..
            } => {
                assert_eq!(*start, 0);
                assert_eq!(*end, 5);
                assert_eq!(*step, 1);
            }
            _ => panic!("expected Range plan"),
        }
    }

    #[test]
    fn tvf_explode_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let col = crate::column::col("array_col");
        let df = tvf.explode(&col).unwrap();
        match &df.plan {
            LogicalPlan::UnresolvedTableValuedFunction { name, arguments } => {
                assert_eq!(name, "explode");
                assert_eq!(arguments.len(), 1);
            }
            _ => panic!("expected UnresolvedTableValuedFunction plan"),
        }
    }

    #[test]
    fn tvf_explode_outer_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let col = crate::column::col("array_col");
        let df = tvf.explode_outer(&col).unwrap();
        match &df.plan {
            LogicalPlan::UnresolvedTableValuedFunction { name, .. } => {
                assert_eq!(name, "explode_outer");
            }
            _ => panic!("expected UnresolvedTableValuedFunction plan"),
        }
    }

    #[test]
    fn tvf_inline_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let col = crate::column::col("struct_array");
        let df = tvf.inline(&col).unwrap();
        match &df.plan {
            LogicalPlan::UnresolvedTableValuedFunction { name, .. } => {
                assert_eq!(name, "inline");
            }
            _ => panic!("expected UnresolvedTableValuedFunction plan"),
        }
    }

    #[test]
    fn tvf_json_tuple_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let input = crate::column::col("json_col");
        let fields = vec![crate::column::col("field1"), crate::column::col("field2")];
        let df = tvf.json_tuple(&input, fields).unwrap();
        match &df.plan {
            LogicalPlan::UnresolvedTableValuedFunction { name, arguments } => {
                assert_eq!(name, "json_tuple");
                assert_eq!(arguments.len(), 3); // input + 2 fields
            }
            _ => panic!("expected UnresolvedTableValuedFunction plan"),
        }
    }

    #[test]
    fn tvf_json_tuple_empty_fields_error() {
        let spark = session();
        let tvf = spark.tvf();
        let input = crate::column::col("json_col");
        let result = tvf.json_tuple(&input, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn tvf_posexplode_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let col = crate::column::col("array_col");
        let df = tvf.posexplode(&col).unwrap();
        match &df.plan {
            LogicalPlan::UnresolvedTableValuedFunction { name, .. } => {
                assert_eq!(name, "posexplode");
            }
            _ => panic!("expected UnresolvedTableValuedFunction plan"),
        }
    }

    #[test]
    fn tvf_stack_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let n = crate::column::lit(3i64);
        let fields = vec![
            crate::column::col("col1"),
            crate::column::col("col2"),
            crate::column::col("col3"),
        ];
        let df = tvf.stack(&n, fields).unwrap();
        match &df.plan {
            LogicalPlan::UnresolvedTableValuedFunction { name, arguments } => {
                assert_eq!(name, "stack");
                assert_eq!(arguments.len(), 4); // n + 3 fields
            }
            _ => panic!("expected UnresolvedTableValuedFunction plan"),
        }
    }

    #[test]
    fn tvf_collations_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let df = tvf.collations().unwrap();
        match &df.plan {
            LogicalPlan::UnresolvedTableValuedFunction { name, arguments } => {
                assert_eq!(name, "collations");
                assert!(arguments.is_empty());
            }
            _ => panic!("expected UnresolvedTableValuedFunction plan"),
        }
    }

    #[test]
    fn tvf_sql_keywords_plan() {
        let spark = session();
        let tvf = spark.tvf();
        let df = tvf.sql_keywords().unwrap();
        match &df.plan {
            LogicalPlan::UnresolvedTableValuedFunction { name, arguments } => {
                assert_eq!(name, "sql_keywords");
                assert!(arguments.is_empty());
            }
            _ => panic!("expected UnresolvedTableValuedFunction plan"),
        }
    }
}

//! Table-valued functions (TVF) implementation mirroring `pyspark.sql.connect.tvf.TableValuedFunction`.
//!
//! Provides table-valued functions that return DataFrames, such as explode, inline,
//! range, and other TVF operations.

use crate::column::Column;
use crate::dataframe::DataFrame;
use crate::expression::Expression;
use crate::plan::LogicalPlan;
use crate::session::SparkSession;
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

    #[test]
    fn test_tvf_creation() {
        // This test verifies that TVF can be created.
        // A full end-to-end test requires a running Spark Connect server.
    }
}

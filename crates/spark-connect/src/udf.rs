//! User-defined function (UDF) support.
//!
//! Mirrors `pyspark.sql.connect.udf` and `pyspark.util.PythonEvalType`.
//! UDFs are cloudpickled on the Python client and wrapped into
//! `CommonInlineUserDefinedFunction` expressions for transmission to the server.

use crate::expression::Expression;
use crate::types::DataType;
use spark_connect_proto as proto;

/// Python evaluation type constants, matching `pyspark.util.PythonEvalType`.
/// These distinguish between different UDF execution modes (batched, pandas, arrow, etc.).
pub mod eval_type {
    /// Regular Python UDF, row-by-row (column as list).
    pub const SQL_BATCHED_UDF: i32 = 100;
    /// Arrow-optimized Python UDF (column as PyArrow table).
    pub const SQL_ARROW_BATCHED_UDF: i32 = 101;

    /// Pandas scalar UDF (Series -> Series).
    pub const SQL_SCALAR_PANDAS_UDF: i32 = 200;
    /// Pandas grouped map UDF (grouped DataFrame -> DataFrame).
    pub const SQL_GROUPED_MAP_PANDAS_UDF: i32 = 201;
    /// Pandas grouped aggregate UDF.
    pub const SQL_GROUPED_AGG_PANDAS_UDF: i32 = 202;
    /// Pandas window aggregate UDF.
    pub const SQL_WINDOW_AGG_PANDAS_UDF: i32 = 203;
    /// Pandas scalar iterator UDF.
    pub const SQL_SCALAR_PANDAS_ITER_UDF: i32 = 204;
    /// Pandas map iterator UDF.
    pub const SQL_MAP_PANDAS_ITER_UDF: i32 = 205;
    /// Pandas cogrouped map UDF.
    pub const SQL_COGROUPED_MAP_PANDAS_UDF: i32 = 206;
    /// Arrow map iterator UDF.
    pub const SQL_MAP_ARROW_ITER_UDF: i32 = 207;
    /// Pandas grouped map with state UDF.
    pub const SQL_GROUPED_MAP_PANDAS_UDF_WITH_STATE: i32 = 208;
    /// Arrow grouped map UDF.
    pub const SQL_GROUPED_MAP_ARROW_UDF: i32 = 209;
    /// Arrow cogrouped map UDF.
    pub const SQL_COGROUPED_MAP_ARROW_UDF: i32 = 210;
    /// Pandas transform with state UDF.
    pub const SQL_TRANSFORM_WITH_STATE_PANDAS_UDF: i32 = 211;
    /// Pandas transform with state init state UDF.
    pub const SQL_TRANSFORM_WITH_STATE_PANDAS_INIT_STATE_UDF: i32 = 212;
    /// Python row transform with state UDF.
    pub const SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_UDF: i32 = 213;
    /// Python row transform with state init state UDF.
    pub const SQL_TRANSFORM_WITH_STATE_PYTHON_ROW_INIT_STATE_UDF: i32 = 214;
    /// Arrow grouped map iterator UDF.
    pub const SQL_GROUPED_MAP_ARROW_ITER_UDF: i32 = 215;
    /// Pandas grouped map iterator UDF.
    pub const SQL_GROUPED_MAP_PANDAS_ITER_UDF: i32 = 216;
    /// Pandas grouped aggregate iterator UDF.
    pub const SQL_GROUPED_AGG_PANDAS_ITER_UDF: i32 = 217;

    /// Arrow scalar UDF.
    pub const SQL_SCALAR_ARROW_UDF: i32 = 250;
    /// Arrow scalar iterator UDF.
    pub const SQL_SCALAR_ARROW_ITER_UDF: i32 = 251;
    /// Arrow grouped aggregate UDF.
    pub const SQL_GROUPED_AGG_ARROW_UDF: i32 = 252;
    /// Arrow window aggregate UDF.
    pub const SQL_WINDOW_AGG_ARROW_UDF: i32 = 253;
    /// Arrow grouped aggregate iterator UDF.
    pub const SQL_GROUPED_AGG_ARROW_ITER_UDF: i32 = 254;

    /// SQL table UDF (UDTF).
    pub const SQL_TABLE_UDF: i32 = 300;
    /// Arrow SQL table UDF (UDTF).
    pub const SQL_ARROW_TABLE_UDF: i32 = 301;
    /// Arrow UDTF.
    pub const SQL_ARROW_UDTF: i32 = 302;
}

/// Represents a Python UDF with its serialized command and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct PythonUDFPayload {
    /// The output data type of the UDF.
    pub output_type: DataType,
    /// The evaluation type (e.g., SQL_BATCHED_UDF, SQL_SCALAR_PANDAS_UDF).
    pub eval_type: i32,
    /// The cloudpickled command bytes: typically cloudpickle.dumps((func, output_type)).
    pub command: Vec<u8>,
    /// Python version used for pickling (e.g., "3.9", "3.11").
    pub python_ver: String,
}

impl PythonUDFPayload {
    /// Create a new Python UDF payload.
    pub fn new(
        output_type: DataType,
        eval_type: i32,
        command: Vec<u8>,
        python_ver: String,
    ) -> Self {
        PythonUDFPayload {
            output_type,
            eval_type,
            command,
            python_ver,
        }
    }

    /// Convert to a proto PythonUDF message.
    pub fn to_proto(&self) -> proto::PythonUdf {
        use bytes::Bytes;
        let mut proto = proto::PythonUdf::default();
        proto.output_type = Some(self.output_type.to_proto());
        proto.eval_type = self.eval_type;
        proto.command = Bytes::copy_from_slice(&self.command);
        proto.python_ver = self.python_ver.clone();
        proto
    }
}

/// Represents a CommonInlineUserDefinedFunction expression.
/// Wraps a Python UDF with its name, determinism flag, and arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonInlineUserDefinedFunctionExpression {
    /// Name of the UDF (e.g., "my_func").
    pub function_name: String,
    /// Whether the UDF is deterministic.
    pub deterministic: bool,
    /// Argument expressions passed to the UDF.
    pub arguments: Vec<Expression>,
    /// The Python UDF payload (command, output type, eval type, python version).
    pub python_udf: PythonUDFPayload,
}

impl CommonInlineUserDefinedFunctionExpression {
    /// Create a new CommonInlineUserDefinedFunction expression.
    pub fn new(
        function_name: String,
        deterministic: bool,
        arguments: Vec<Expression>,
        python_udf: PythonUDFPayload,
    ) -> Self {
        CommonInlineUserDefinedFunctionExpression {
            function_name,
            deterministic,
            arguments,
            python_udf,
        }
    }

    /// Convert to a proto CommonInlineUserDefinedFunction message.
    pub fn to_proto(&self) -> proto::CommonInlineUserDefinedFunction {
        let mut proto = proto::CommonInlineUserDefinedFunction::default();
        proto.function_name = self.function_name.clone();
        proto.deterministic = self.deterministic;
        proto.arguments = self.arguments.iter().map(|expr| expr.to_proto()).collect();
        proto.is_distinct = false;
        proto.function = Some(
            proto::common_inline_user_defined_function::Function::PythonUdf(
                self.python_udf.to_proto(),
            ),
        );
        proto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_udf_payload_to_proto() {
        let payload = PythonUDFPayload::new(
            DataType::Integer,
            eval_type::SQL_BATCHED_UDF,
            vec![1, 2, 3, 4, 5],
            "3.9".to_string(),
        );

        let proto = payload.to_proto();

        assert_eq!(proto.eval_type, 100); // SQL_BATCHED_UDF
        assert_eq!(proto.python_ver, "3.9");
        assert_eq!(proto.command.len(), 5);
        assert!(proto.output_type.is_some());
    }

    #[test]
    fn test_common_inline_udf_expression_to_proto() {
        let payload = PythonUDFPayload::new(
            DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
            eval_type::SQL_SCALAR_PANDAS_UDF,
            b"pickled_command_bytes".to_vec(),
            "3.11".to_string(),
        );

        let udf_expr = CommonInlineUserDefinedFunctionExpression::new(
            "my_udf".to_string(),
            true,
            vec![],
            payload,
        );

        let proto = udf_expr.to_proto();

        assert_eq!(proto.function_name, "my_udf");
        assert_eq!(proto.deterministic, true);
        assert_eq!(proto.is_distinct, false);
        assert_eq!(proto.arguments.len(), 0);
        assert!(proto.function.is_some());

        // Verify the Python UDF is embedded
        if let Some(proto::common_inline_user_defined_function::Function::PythonUdf(py_udf)) =
            proto.function
        {
            assert_eq!(py_udf.eval_type, 200); // SQL_SCALAR_PANDAS_UDF
            assert_eq!(py_udf.python_ver, "3.11");
            assert_eq!(
                py_udf.command,
                bytes::Bytes::copy_from_slice(b"pickled_command_bytes")
            );
        } else {
            panic!("Expected Python UDF in CommonInlineUserDefinedFunction");
        }
    }

    #[test]
    fn test_eval_type_constants() {
        // Verify eval type constants match pyspark.util.PythonEvalType
        assert_eq!(eval_type::SQL_BATCHED_UDF, 100);
        assert_eq!(eval_type::SQL_ARROW_BATCHED_UDF, 101);
        assert_eq!(eval_type::SQL_SCALAR_PANDAS_UDF, 200);
        assert_eq!(eval_type::SQL_GROUPED_MAP_PANDAS_UDF, 201);
        assert_eq!(eval_type::SQL_GROUPED_AGG_PANDAS_UDF, 202);
        assert_eq!(eval_type::SQL_WINDOW_AGG_PANDAS_UDF, 203);
        assert_eq!(eval_type::SQL_SCALAR_PANDAS_ITER_UDF, 204);
        assert_eq!(eval_type::SQL_MAP_PANDAS_ITER_UDF, 205);
        assert_eq!(eval_type::SQL_COGROUPED_MAP_PANDAS_UDF, 206);
        assert_eq!(eval_type::SQL_MAP_ARROW_ITER_UDF, 207);
        assert_eq!(eval_type::SQL_GROUPED_MAP_ARROW_UDF, 209);
        assert_eq!(eval_type::SQL_COGROUPED_MAP_ARROW_UDF, 210);
        assert_eq!(eval_type::SQL_SCALAR_ARROW_UDF, 250);
        assert_eq!(eval_type::SQL_SCALAR_ARROW_ITER_UDF, 251);
        assert_eq!(eval_type::SQL_GROUPED_AGG_ARROW_UDF, 252);
        assert_eq!(eval_type::SQL_WINDOW_AGG_ARROW_UDF, 253);
        assert_eq!(eval_type::SQL_GROUPED_AGG_ARROW_ITER_UDF, 254);
        assert_eq!(eval_type::SQL_TABLE_UDF, 300);
        assert_eq!(eval_type::SQL_ARROW_TABLE_UDF, 301);
        assert_eq!(eval_type::SQL_ARROW_UDTF, 302);
    }
}

//! Data source registration support.
//!
//! Mirrors `pyspark.sql.connect.datasource` and allows registration of custom
//! Python data sources that can be used in SQL and DataFrame queries.
//! Data sources are cloudpickled on the Python client and wrapped into
//! `CommonInlineUserDefinedDataSource` commands for transmission to the server.

use spark_connect_proto as proto;

/// Represents a Python data source with its serialized command and metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct PythonDataSourcePayload {
    /// The cloudpickled command bytes containing the serialized data source.
    pub command: Vec<u8>,
    /// Python version used for pickling (e.g., "3.9", "3.11").
    pub python_ver: String,
}

impl PythonDataSourcePayload {
    /// Create a new Python data source payload.
    pub fn new(command: Vec<u8>, python_ver: String) -> Self {
        PythonDataSourcePayload {
            command,
            python_ver,
        }
    }

    /// Convert to a proto PythonDataSource message.
    pub fn to_proto(&self) -> proto::PythonDataSource {
        use bytes::Bytes;
        let mut proto = proto::PythonDataSource::default();
        proto.command = Bytes::copy_from_slice(&self.command);
        proto.python_ver = self.python_ver.clone();
        proto
    }
}

/// Represents a CommonInlineUserDefinedDataSource command.
/// Wraps a Python data source with its name.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonInlineUserDefinedDataSourceExpression {
    /// Name of the data source (e.g., "my_source").
    pub name: String,
    /// The Python data source payload (command and python version).
    pub python_data_source: PythonDataSourcePayload,
}

impl CommonInlineUserDefinedDataSourceExpression {
    /// Create a new CommonInlineUserDefinedDataSource expression.
    pub fn new(name: String, python_data_source: PythonDataSourcePayload) -> Self {
        CommonInlineUserDefinedDataSourceExpression {
            name,
            python_data_source,
        }
    }

    /// Convert to a proto CommonInlineUserDefinedDataSource message.
    pub fn to_proto(&self) -> proto::CommonInlineUserDefinedDataSource {
        let mut proto = proto::CommonInlineUserDefinedDataSource::default();
        proto.name = self.name.clone();
        proto.data_source = Some(
            proto::common_inline_user_defined_data_source::DataSource::PythonDataSource(
                self.python_data_source.to_proto(),
            ),
        );
        proto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_data_source_payload_to_proto() {
        let payload = PythonDataSourcePayload::new(vec![1, 2, 3, 4, 5], "3.9".to_string());

        let proto = payload.to_proto();

        assert_eq!(proto.python_ver, "3.9");
        assert_eq!(proto.command.len(), 5);
    }

    #[test]
    fn test_common_inline_data_source_to_proto() {
        let payload =
            PythonDataSourcePayload::new(b"pickled_datasource_bytes".to_vec(), "3.11".to_string());

        let ds_expr =
            CommonInlineUserDefinedDataSourceExpression::new("my_source".to_string(), payload);

        let proto = ds_expr.to_proto();

        assert_eq!(proto.name, "my_source");
        assert!(proto.data_source.is_some());

        // Verify the Python data source is embedded
        if let Some(proto::common_inline_user_defined_data_source::DataSource::PythonDataSource(
            py_ds,
        )) = proto.data_source
        {
            assert_eq!(py_ds.python_ver, "3.11");
            assert_eq!(
                py_ds.command,
                bytes::Bytes::copy_from_slice(b"pickled_datasource_bytes")
            );
        } else {
            panic!("Expected Python data source in CommonInlineUserDefinedDataSource");
        }
    }
}

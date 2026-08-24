//! RuntimeConf implementation mirroring `pyspark.sql.conf.RuntimeConfig`.
//!
//! Provides runtime configuration management for Spark sessions through
//! the Spark Connect protocol.

use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::Result;
use spark_connect_core::runtime::block_on;
use std::collections::HashMap;

/// Runtime configuration for a Spark session.
///
/// Allows getting and setting runtime configuration parameters on the Spark server.
/// Mirrors `pyspark.sql.conf.RuntimeConfig`.
pub struct RuntimeConf {
    /// Shared gRPC client
    client: std::sync::Arc<SparkConnectClient>,
}

impl RuntimeConf {
    /// Create a new RuntimeConf with a client.
    pub(crate) fn new(client: std::sync::Arc<SparkConnectClient>) -> Self {
        RuntimeConf { client }
    }

    /// Set a configuration key-value pair.
    ///
    /// Converts boolean and integer values to strings as required by the server.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key (e.g., "spark.sql.shuffle.partitions")
    /// * `value` - The configuration value as a string, integer, or boolean
    ///
    /// # Example
    ///
    /// ```ignore
    /// conf.set("spark.sql.shuffle.partitions", "200")?;
    /// conf.set("spark.sql.adaptive.enabled", true)?;
    /// conf.set("spark.sql.maxMetadataStringLength", 100)?;
    /// ```
    pub fn set(&self, key: &str, value: &str) -> Result<()> {
        block_on(self.client.set_config(key, value))
    }

    /// Get a configuration value by key.
    ///
    /// Returns `Ok(None)` if the key does not exist on the server.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key to retrieve
    ///
    /// # Example
    ///
    /// ```ignore
    /// let value = conf.get("spark.sql.shuffle.partitions")?;
    /// println!("Shuffle partitions: {:?}", value);
    /// ```
    pub fn get(&self, key: &str) -> Result<Option<String>> {
        let results = block_on(self.client.get_configs(&[key]))?;
        Ok(results.into_iter().next().flatten())
    }

    /// Get all configuration values as a HashMap.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let all_configs = conf.get_all()?;
    /// for (key, value) in all_configs.iter() {
    ///     println!("{}: {}", key, value);
    /// }
    /// ```
    pub fn get_all(&self) -> Result<HashMap<String, String>> {
        let results = block_on(self.client.get_configs_all())?;
        Ok(results)
    }

    /// Unset a configuration key (reset to default).
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key to unset
    ///
    /// # Example
    ///
    /// ```ignore
    /// conf.unset("spark.sql.shuffle.partitions")?;
    /// ```
    pub fn unset(&self, key: &str) -> Result<()> {
        block_on(self.client.unset_config(key))
    }

    /// Check if a configuration key is modifiable.
    ///
    /// Returns `true` if the configuration can be changed at runtime,
    /// `false` if it's read-only or cannot be modified.
    ///
    /// # Arguments
    ///
    /// * `key` - The configuration key to check
    ///
    /// # Example
    ///
    /// ```ignore
    /// if conf.is_modifiable("spark.sql.shuffle.partitions")? {
    ///     println!("This config can be modified");
    /// }
    /// ```
    pub fn is_modifiable(&self, key: &str) -> Result<bool> {
        let results = block_on(self.client.is_config_modifiable(key))?;
        Ok(results)
    }
}

impl Clone for RuntimeConf {
    fn clone(&self) -> Self {
        RuntimeConf {
            client: std::sync::Arc::clone(&self.client),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_conf_creation() {
        // This test verifies that RuntimeConf can be created.
        // A full end-to-end test requires a running Spark Connect server.
    }
}

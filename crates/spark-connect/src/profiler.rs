//! Client-side profiler collector for UDF and plan profiling results.
//!
//! Mirrors `pyspark.sql.connect.profiler.ConnectProfilerCollector`. Accumulates profile
//! results observed across multiple `ExecutePlanResponse`s during the session and provides
//! methods to show, dump, and clear profile data.
//!
//! Profile data is populated by the server only when UDF profiling is enabled via
//! `spark.python.profile*` or `spark.sql.pyspark.udf.profiler` configuration.

use std::collections::HashMap;
use std::sync::Mutex;

use spark_connect_core::error::Result;
use spark_connect_proto as proto;

/// Format a protobuf Literal value as a string.
fn format_literal(literal: &proto::expression::Literal) -> String {
    use proto::expression::literal::LiteralType;

    match &literal.literal_type {
        Some(LiteralType::Null(_)) => "null".to_string(),
        Some(LiteralType::Boolean(b)) => b.to_string(),
        Some(LiteralType::Byte(b)) => b.to_string(),
        Some(LiteralType::Short(s)) => s.to_string(),
        Some(LiteralType::Integer(i)) => i.to_string(),
        Some(LiteralType::Long(l)) => l.to_string(),
        Some(LiteralType::Float(f)) => f.to_string(),
        Some(LiteralType::Double(d)) => d.to_string(),
        Some(LiteralType::Decimal(_)) => "decimal".to_string(),
        Some(LiteralType::String(s)) => s.clone(),
        Some(LiteralType::Binary(b)) => format!("<binary: {} bytes>", b.len()),
        Some(LiteralType::CalendarInterval(_)) => "calendar_interval".to_string(),
        Some(LiteralType::YearMonthInterval(_)) => "year_month_interval".to_string(),
        Some(LiteralType::DayTimeInterval(_)) => "day_time_interval".to_string(),
        Some(LiteralType::Date(_)) => "date".to_string(),
        Some(LiteralType::Timestamp(_)) => "timestamp".to_string(),
        Some(LiteralType::TimestampNtz(_)) => "timestamp_ntz".to_string(),
        Some(LiteralType::Time(_)) => "time".to_string(),
        Some(LiteralType::TimestampNtzNanos(_)) => "timestamp_ntz_nanos".to_string(),
        Some(LiteralType::TimestampLtzNanos(_)) => "timestamp_ltz_nanos".to_string(),
        Some(LiteralType::Map(_)) => "map".to_string(),
        Some(LiteralType::Array(_)) => "array".to_string(),
        Some(LiteralType::Struct(_)) => "struct".to_string(),
        Some(LiteralType::SpecializedArray(_)) => "specialized_array".to_string(),
        None => "unknown".to_string(),
    }
}

/// Stores collected profile data for a single profiler ID.
#[derive(Debug, Clone)]
struct ProfileResult {
    /// The accumulated profile data (typically a string representation).
    pub data: String,
    /// Metadata about the profile (e.g., profile type, timestamp). Collected from the
    /// profiler responses (see `collect_profiles`) for parity with pyspark; retained for
    /// a future `show()`/`dump()` that surfaces it, hence not read yet.
    #[allow(dead_code)]
    pub metadata: HashMap<String, String>,
}

/// Client-side collector for accumulated profiler results.
///
/// Accumulates UDF and plan profile results across multiple query executions
/// and provides access via `show()`, `dump()`, and `clear()` methods.
#[derive(Debug, Clone)]
pub struct ProfilerCollector {
    /// Profile results keyed by profiler ID.
    profiles: std::sync::Arc<Mutex<HashMap<i64, ProfileResult>>>,
}

impl ProfilerCollector {
    /// Create a new profiler collector.
    pub fn new() -> Self {
        ProfilerCollector {
            profiles: std::sync::Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Accumulate a profile result. Called internally during query execution.
    pub(crate) fn accumulate_profile(
        &self,
        id: i64,
        data: String,
        metadata: HashMap<String, String>,
    ) {
        let mut profiles = self.profiles.lock().unwrap();
        profiles.insert(id, ProfileResult { data, metadata });
    }

    /// Accumulate observed metrics from a server response.
    pub(crate) fn accumulate_observed_metrics(
        &self,
        metrics: &[proto::execute_plan_response::ObservedMetrics],
    ) {
        for metric in metrics {
            // Use the metric name and plan_id as profile identifier
            let id = metric.plan_id;
            let mut data = String::new();
            let mut metadata = HashMap::new();

            // Pair up keys and values if available
            let num_values = metric.values.len();
            for i in 0..num_values {
                let key = if i < metric.keys.len() {
                    metric.keys[i].clone()
                } else {
                    format!("value_{}", i)
                };

                // Try to extract value as string from the Literal
                let value_str = format_literal(&metric.values[i]);

                metadata.insert(key.clone(), value_str.clone());
                // Accumulate all metric values into data
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(&format!("{}: {}", key, value_str));
            }

            // Include the metric name and plan_id in the data
            if !data.is_empty() {
                data.insert_str(0, &format!("name: {}, plan_id: {}\n", metric.name, id));
            }

            // Store the profile result if we have any data or metadata
            if !data.is_empty() || !metadata.is_empty() {
                self.accumulate_profile(id, data, metadata);
            }
        }
    }

    /// Show the profile results for a given profiler ID, or all profiles if ID is None.
    ///
    /// Returns a formatted string representation of the profile data.
    pub fn show(&self, id: Option<i64>) -> String {
        let profiles = self.profiles.lock().unwrap();

        match id {
            Some(profile_id) => profiles
                .get(&profile_id)
                .map(|p| p.data.clone())
                .unwrap_or_else(|| format!("No profile data for id: {}", profile_id)),
            None => {
                // Show all profiles
                if profiles.is_empty() {
                    "No profile data collected".to_string()
                } else {
                    let mut result = String::new();
                    for (id, profile) in profiles.iter() {
                        result.push_str(&format!("=== Profile {} ===\n", id));
                        result.push_str(&profile.data);
                        result.push('\n');
                    }
                    result
                }
            }
        }
    }

    /// Write profile results to a file.
    ///
    /// For a given profiler ID (or all profiles if ID is None), writes the profile data to the
    /// specified file path.
    pub fn dump(&self, id: Option<i64>, path: &str) -> Result<()> {
        let data = self.show(id);

        std::fs::write(path, data).map_err(|e| {
            spark_connect_core::error::SparkError::connect_msg(format!(
                "Failed to write profile to {}: {}",
                path, e
            ))
        })?;

        Ok(())
    }

    /// Clear collected profile results for a given ID, or all profiles if ID is None.
    pub fn clear(&self, id: Option<i64>) {
        let mut profiles = self.profiles.lock().unwrap();

        match id {
            Some(profile_id) => {
                profiles.remove(&profile_id);
            }
            None => {
                profiles.clear();
            }
        }
    }

    /// Get the number of collected profiles.
    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.profiles.lock().unwrap().len()
    }
}

impl Default for ProfilerCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_collector_accumulate_and_show() {
        let collector = ProfilerCollector::new();

        // Accumulate a profile
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "udf".to_string());
        collector.accumulate_profile(1, "profile data 1".to_string(), metadata);

        // Show profile by ID
        let result = collector.show(Some(1));
        assert_eq!(result, "profile data 1");

        // Show all profiles
        let all = collector.show(None);
        assert!(all.contains("profile data 1"));
        assert!(all.contains("=== Profile 1 ==="));
    }

    #[test]
    fn test_profiler_collector_show_empty() {
        let collector = ProfilerCollector::new();

        let result = collector.show(None);
        assert_eq!(result, "No profile data collected");

        let result = collector.show(Some(99));
        assert_eq!(result, "No profile data for id: 99");
    }

    #[test]
    fn test_profiler_collector_clear() {
        let collector = ProfilerCollector::new();

        let metadata = HashMap::new();
        collector.accumulate_profile(1, "profile 1".to_string(), metadata.clone());
        collector.accumulate_profile(2, "profile 2".to_string(), metadata);

        assert_eq!(collector.count(), 2);

        // Clear one profile
        collector.clear(Some(1));
        assert_eq!(collector.count(), 1);

        // Clear all
        collector.clear(None);
        assert_eq!(collector.count(), 0);
    }

    #[test]
    fn test_profiler_collector_dump_and_read() {
        let collector = ProfilerCollector::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_profile_dump.txt");

        let metadata = HashMap::new();
        collector.accumulate_profile(1, "test profile data".to_string(), metadata);

        // Dump to file
        let result = collector.dump(Some(1), path.to_str().unwrap());
        assert!(result.is_ok());

        // Read and verify
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "test profile data");

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_profiler_collector_accumulate_observed_metrics() {
        let collector = ProfilerCollector::new();

        // Create a mock observed metrics
        let mut metric = proto::execute_plan_response::ObservedMetrics::default();
        metric.name = "udf_profile".to_string();
        metric.plan_id = 1;
        metric.keys = vec!["udf_time_ms".to_string(), "python_calls".to_string()];

        // Create literal values
        metric.values.push(proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Long(150)),
            data_type: None,
        });
        metric.values.push(proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Long(42)),
            data_type: None,
        });

        collector.accumulate_observed_metrics(&[metric]);

        // Verify the profile was accumulated
        let result = collector.show(Some(1));
        assert!(result.contains("udf_time_ms: 150"));
        assert!(result.contains("python_calls: 42"));
    }

    #[test]
    fn format_literal_covers_all_variants() {
        use proto::expression::literal::{self, LiteralType};
        let lit = |lt: LiteralType| proto::expression::Literal {
            literal_type: Some(lt),
            data_type: None,
        };
        assert_eq!(
            format_literal(&lit(LiteralType::Null(proto::DataType::default()))),
            "null"
        );
        assert_eq!(format_literal(&lit(LiteralType::Boolean(true))), "true");
        assert_eq!(format_literal(&lit(LiteralType::Byte(1))), "1");
        assert_eq!(format_literal(&lit(LiteralType::Short(2))), "2");
        assert_eq!(format_literal(&lit(LiteralType::Integer(3))), "3");
        assert_eq!(format_literal(&lit(LiteralType::Long(4))), "4");
        assert_eq!(format_literal(&lit(LiteralType::Float(1.5))), "1.5");
        assert_eq!(format_literal(&lit(LiteralType::Double(2.5))), "2.5");
        assert_eq!(
            format_literal(&lit(LiteralType::Decimal(literal::Decimal::default()))),
            "decimal"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::String("hi".to_string()))),
            "hi"
        );
        assert!(
            format_literal(&lit(LiteralType::Binary(vec![1u8, 2, 3].into()))).contains("3 bytes")
        );
        assert_eq!(
            format_literal(&lit(LiteralType::CalendarInterval(
                literal::CalendarInterval::default()
            ))),
            "calendar_interval"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::YearMonthInterval(0))),
            "year_month_interval"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::DayTimeInterval(0))),
            "day_time_interval"
        );
        assert_eq!(format_literal(&lit(LiteralType::Date(0))), "date");
        assert_eq!(format_literal(&lit(LiteralType::Timestamp(0))), "timestamp");
        assert_eq!(
            format_literal(&lit(LiteralType::TimestampNtz(0))),
            "timestamp_ntz"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::Time(literal::Time::default()))),
            "time"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::TimestampNtzNanos(
                literal::TimestampNtzNanos::default()
            ))),
            "timestamp_ntz_nanos"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::TimestampLtzNanos(
                literal::TimestampLtzNanos::default()
            ))),
            "timestamp_ltz_nanos"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::Map(literal::Map::default()))),
            "map"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::Array(literal::Array::default()))),
            "array"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::Struct(literal::Struct::default()))),
            "struct"
        );
        assert_eq!(
            format_literal(&lit(LiteralType::SpecializedArray(
                literal::SpecializedArray::default()
            ))),
            "specialized_array"
        );
        // literal_type == None falls through to "unknown".
        assert_eq!(
            format_literal(&proto::expression::Literal {
                literal_type: None,
                data_type: None,
            }),
            "unknown"
        );
    }

    #[test]
    fn accumulate_observed_metrics_more_values_than_keys() {
        // values longer than keys hits the `value_{i}` fallback key branch.
        let collector = ProfilerCollector::new();
        let mut metric = proto::execute_plan_response::ObservedMetrics::default();
        metric.name = "m".to_string();
        metric.plan_id = 7;
        metric.keys = vec!["a".to_string()];
        metric.values.push(proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Long(1)),
            data_type: None,
        });
        metric.values.push(proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Long(2)),
            data_type: None,
        });
        collector.accumulate_observed_metrics(&[metric]);
        let shown = collector.show(Some(7));
        assert!(shown.contains("a: 1"));
        assert!(shown.contains("value_1: 2"));
    }

    #[test]
    fn dump_to_unwritable_path_errors() {
        let collector = ProfilerCollector::new();
        collector.accumulate_profile(1, "data".to_string(), HashMap::new());
        // A path under a non-existent directory cannot be written -> Err branch.
        let res = collector.dump(Some(1), "/nonexistent_dir_zzz_12345/profile.txt");
        assert!(res.is_err());
    }

    #[test]
    fn default_constructs_empty_collector() {
        let collector = ProfilerCollector::default();
        assert_eq!(collector.count(), 0);
    }
}

//! Observation implementation for collecting metrics.
//!
//! Mirroring `pyspark.sql.observation.Observation`.

use std::collections::HashMap;

/// An Observation for collecting metrics from a DataFrame.
///
/// Mirrors `pyspark.sql.observation.Observation`.
pub struct Observation {
    name: String,
    metrics: HashMap<String, String>,
}

impl Observation {
    /// Create a new Observation with a name.
    pub fn new(name: &str) -> Self {
        Observation {
            name: name.to_string(),
            metrics: HashMap::new(),
        }
    }

    /// Get the name of this Observation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get metrics from this Observation.
    pub fn get(&self) -> HashMap<String, String> {
        self.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_creation() {
        let obs = Observation::new("test_obs");
        assert_eq!(obs.name(), "test_obs");
        assert!(obs.get().is_empty());
    }
}

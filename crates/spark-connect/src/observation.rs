//! Observation implementation for collecting metrics.
//!
//! Mirrors `pyspark.sql.connect.observation.Observation`.

use std::sync::{Arc, Mutex, MutexGuard};

use spark_connect_core::error::{Result, SparkError, SparkErrorKind};
use spark_connect_proto as proto;

/// Metrics that the server observes on a DataFrame, mirroring `pyspark.sql.Observation`.
///
/// [`crate::DataFrame::observe_with`] observes them, by a `CollectMetrics` relation of the
/// name of the observation. Each action that runs the DataFrame, or a DataFrame of it, sets
/// them as the server observed them, which [`Observation::get`] returns. A clone shares the
/// state of the observation.
#[derive(Clone, Default)]
pub struct Observation {
    state: Arc<Mutex<ObservationState>>,
}

#[derive(Default)]
struct ObservationState {
    name: Option<String>,
    /// None before the observation is used; then the metrics observed so far, in order.
    result: Option<Vec<(String, proto::expression::Literal)>>,
    /// The error of collecting the metrics, which the server reported.
    error: Option<SparkError>,
}

impl std::fmt::Debug for Observation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Observation")
            .field("name", &self.lock().name)
            .finish()
    }
}

impl Observation {
    /// An observation of a name.
    pub fn new(name: &str) -> Self {
        Observation {
            state: Arc::new(Mutex::new(ObservationState {
                name: Some(name.to_string()),
                ..Default::default()
            })),
        }
    }

    /// An observation without a name, which is a random UUID once it is used, as PySpark's
    /// `Observation()`.
    pub fn unnamed() -> Self {
        Self::default()
    }

    /// The name of the observation; None for an unnamed observation before it is used.
    pub fn name(&self) -> Option<String> {
        self.lock().name.clone()
    }

    /// The observed metrics, mirroring `Observation.get`: those that the actions run so far
    /// observed, by their names, in order, and none before an action runs. The error of
    /// collecting them, if the server reported one, and `NO_OBSERVE_BEFORE_GET` before the
    /// observation is used.
    pub fn get(&self) -> Result<Vec<(String, proto::expression::Literal)>> {
        let state = self.lock();
        if let Some(error) = &state.error {
            return Err(error.clone());
        }
        state.result.clone().ok_or_else(|| {
            SparkError::classed(SparkErrorKind::AssertionError, "NO_OBSERVE_BEFORE_GET", &[])
        })
    }

    /// Whether `self` and `other` are the same observation, of which one is a clone.
    pub fn same(&self, other: &Observation) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// The name to observe the metrics by, as PySpark's `Observation._on` before it
    /// observes them: `REUSE_OBSERVATION` for an observation used already, and the name,
    /// a random UUID that it keeps for an unnamed observation.
    pub(crate) fn name_to_observe(&self) -> Result<String> {
        let mut state = self.lock();
        if state.result.is_some() {
            return Err(SparkError::classed(
                SparkErrorKind::AssertionError,
                "REUSE_OBSERVATION",
                &[],
            ));
        }
        Ok(state
            .name
            .get_or_insert_with(|| uuid::Uuid::new_v4().to_string())
            .clone())
    }

    /// Mark the observation used, with no metrics until an action observes them.
    pub(crate) fn start(&self) {
        self.lock().result = Some(Vec::new());
    }

    /// Set the metrics that an action observed, as PySpark's client updates the result of
    /// the observation with them: a metric replaces the value of a metric of its name.
    pub(crate) fn update(&self, keys: &[String], values: &[proto::expression::Literal]) {
        let mut state = self.lock();
        let Some(result) = state.result.as_mut() else {
            return;
        };
        for (key, value) in keys.iter().zip(values) {
            match result.iter_mut().find(|(name, _)| name == key) {
                Some((_, existing)) => *existing = value.clone(),
                None => result.push((key.clone(), value.clone())),
            }
        }
    }

    /// Set the error of collecting the metrics, which `get` returns.
    pub(crate) fn set_error(&self, error: SparkError) {
        self.lock().error = Some(error);
    }

    fn lock(&self) -> MutexGuard<'_, ObservationState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::expression::literal::LiteralType;

    fn long(value: i64) -> proto::expression::Literal {
        proto::expression::Literal {
            literal_type: Some(LiteralType::Long(value)),
            ..Default::default()
        }
    }

    #[test]
    fn test_observation_creation() {
        let obs = Observation::new("test_obs");
        assert_eq!(obs.name().as_deref(), Some("test_obs"));
        // As PySpark's: an observation that is not used has no metrics to get.
        let err = obs.get().unwrap_err();
        assert_eq!(
            err.get_condition().as_deref(),
            Some("NO_OBSERVE_BEFORE_GET")
        );
    }

    #[test]
    fn an_observation_is_used_once_and_updated_by_each_action() {
        let obs = Observation::unnamed();
        assert_eq!(obs.name(), None);
        let name = obs.name_to_observe().unwrap();
        assert_eq!(name.len(), 36);
        assert_eq!(obs.name(), Some(name.clone()));
        // Its name is kept until it is used.
        assert_eq!(obs.name_to_observe().unwrap(), name);
        obs.start();
        assert!(obs.get().unwrap().is_empty());
        let err = obs.name_to_observe().unwrap_err();
        assert_eq!(err.get_condition().as_deref(), Some("REUSE_OBSERVATION"));

        let clone = obs.clone();
        assert!(clone.same(&obs) && !Observation::unnamed().same(&obs));
        clone.update(&["a".to_string(), "b".to_string()], &[long(1), long(2)]);
        obs.update(&["b".to_string(), "c".to_string()], &[long(3), long(4)]);
        let names: Vec<(String, Option<LiteralType>)> = obs
            .get()
            .unwrap()
            .into_iter()
            .map(|(name, value)| (name, value.literal_type))
            .collect();
        assert_eq!(
            names,
            vec![
                ("a".to_string(), Some(LiteralType::Long(1))),
                ("b".to_string(), Some(LiteralType::Long(3))),
                ("c".to_string(), Some(LiteralType::Long(4))),
            ]
        );

        obs.set_error(SparkError::connect_msg("failed"));
        assert!(obs.get().unwrap_err().message().contains("failed"));
        assert!(format!("{obs:?}").contains(&name));
    }
}

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
#[derive(Default)]
pub struct Observation {
    state: Arc<Mutex<ObservationState>>,
    /// What a binding of another language keeps alive with this clone of the observation (see
    /// [`Observation::with_owner`]), which the other clones of its state do not hold.
    owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

/// A clone of the same state and owner, of which the check of the clones of a binding is told
/// (see [`crate::gc::check_clones_with`]).
impl Clone for Observation {
    fn clone(&self) -> Self {
        if let Some(owner) = &self.owner {
            crate::gc::cloned(Arc::as_ptr(owner) as *const () as usize);
        }
        Observation {
            state: Arc::clone(&self.state),
            owner: self.owner.clone(),
        }
    }
}

/// The metrics of an observation, by their names, in order.
type Metrics = Vec<(String, proto::expression::Literal)>;

/// What a binding of another language keeps of the state of a used observation, such as the
/// Python Observation of the Python client, whose `_result` and `_error` the observation sets as
/// PySpark's client sets them: it is asked whether the observation is used, and told of each
/// update of the metrics and of the error that the server reports. The shallow copies of the
/// observation share it with its metrics. Not a stable API.
#[doc(hidden)]
pub trait MetricsMirror: Send + Sync {
    /// Whether the observation of the binding is used, as PySpark's client asserts that the
    /// `_result` of an Observation is not None before it sets its metrics, which a program may
    /// have set to None: its error, such as the `AttributeError` of a `_result` that the program
    /// deleted, is that of the action. Used, unless the binding says otherwise.
    fn is_used(&self) -> Result<bool> {
        Ok(true)
    }

    /// The metrics of the names `keys` were set to `values` by an action, in their order: its
    /// error is that of the action, as PySpark's client raises the error of the update of the
    /// result of an Observation.
    fn update(&self, keys: &[String], values: &[proto::expression::Literal]) -> Result<()>;

    /// The server reported `error` of collecting the metrics: its error is that of the action, as
    /// PySpark's client raises the error of the `_set_error` of an Observation.
    fn error(&self, error: &SparkError) -> Result<()> {
        let _ = error;
        Ok(())
    }

    /// The name of the observation of the binding as it is now, which a plan of it is built of,
    /// as PySpark's `CollectMetrics` sets the `str` of the `_name` of its Observation each time it
    /// builds the plan, and by which the metrics of the plan are set: its error, as that of a name
    /// that the plan does not take, is that of the request. None, unless the binding gives it, for
    /// the name of the observation.
    fn name(&self) -> Option<Result<String>> {
        None
    }

    /// This mirror, by which the binding finds its own type.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The metrics of a used observation, and the mirror of a binding of them, which its shallow
/// copies share.
#[derive(Default)]
struct UsedMetrics {
    metrics: Mutex<Metrics>,
    mirror: Mutex<Option<Arc<dyn MetricsMirror>>>,
}

impl UsedMetrics {
    /// The mirror of a binding, if it set one.
    fn mirror(&self) -> Option<Arc<dyn MetricsMirror>> {
        self.mirror
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

#[derive(Default)]
struct ObservationState {
    name: Option<String>,
    /// None before the observation is used; then the metrics observed so far, which a shallow
    /// copy of the observation shares, as a copy of PySpark's shares its result dict.
    result: Option<Arc<UsedMetrics>>,
    /// The error of collecting the metrics, which the server reported.
    error: Option<SparkError>,
    /// What a binding of another language keeps for this observation (see
    /// [`Observation::set_handle`]), which no copy of it has.
    handle: Option<Arc<dyn std::any::Any + Send + Sync>>,
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
            owner: None,
        }
    }

    /// An observation without a name, which is a random UUID once it is used, as PySpark's
    /// `Observation()`.
    pub fn unnamed() -> Self {
        Self::default()
    }

    /// A used observation of the name `name`, of no metrics until an action observes them, as an
    /// observation once a DataFrame observes by it, for a binding whose own object of it says
    /// whether it is used (see [`MetricsMirror::is_used`]). Not a stable API.
    #[doc(hidden)]
    pub fn used(name: &str) -> Self {
        let observation = Self::new(name);
        observation.start();
        observation
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
        match &state.result {
            Some(result) => Ok(lock_metrics(&result.metrics).clone()),
            None => Err(SparkError::classed(
                SparkErrorKind::AssertionError,
                "NO_OBSERVE_BEFORE_GET",
                &[],
            )),
        }
    }

    /// The metrics observed so far, None before the observation is used, and the error of
    /// collecting them, if the server reported one: the `_result` and `_error` of PySpark's
    /// Observation, which a pickled one keeps. Not a stable API.
    #[doc(hidden)]
    pub fn state(
        &self,
    ) -> (
        Option<Vec<(String, proto::expression::Literal)>>,
        Option<SparkError>,
    ) {
        let state = self.lock();
        let result = state
            .result
            .as_ref()
            .map(|result| lock_metrics(&result.metrics).clone());
        (result, state.error.clone())
    }

    /// Whether the observation is used, as PySpark's Observation once its `_result` is not None:
    /// the mirror of a binding, if it set one, says it (see [`MetricsMirror::is_used`]).
    pub(crate) fn is_used(&self) -> Result<bool> {
        let Some(result) = self.lock().result.clone() else {
            return Ok(false);
        };
        match result.mirror() {
            Some(mirror) => mirror.is_used(),
            None => Ok(true),
        }
    }

    /// The name that a plan of this observation is built of now: the name that the mirror of a
    /// binding gives, if it gives one (see [`MetricsMirror::name`]), of its error, else none, for
    /// the name of the observation. Not a stable API.
    #[doc(hidden)]
    pub fn name_of_plan(&self) -> Option<Result<String>> {
        self.metrics_mirror()?.name()
    }

    /// The mirror of the state of this observation that a binding set (see
    /// [`Observation::set_metrics_mirror`]), if any. Not a stable API.
    #[doc(hidden)]
    pub fn metrics_mirror(&self) -> Option<Arc<dyn MetricsMirror>> {
        self.lock().result.clone()?.mirror()
    }

    /// Make `mirror` the mirror of the state of this used observation (see [`MetricsMirror`]),
    /// in the place of the one that a binding set before, if any: the metrics set so far, which
    /// the caller fills it with, as each update after it is told to it. None before the
    /// observation is used. No lock of the observation is held while the caller fills the
    /// mirror, or while a mirror is told of an update, so either may take a lock of its own. Not
    /// a stable API.
    #[doc(hidden)]
    pub fn set_metrics_mirror(
        &self,
        mirror: Arc<dyn MetricsMirror>,
    ) -> Option<Vec<(String, proto::expression::Literal)>> {
        let result = self.lock().result.clone()?;
        // Set under the lock of the metrics, so that an update is either in the metrics given to
        // the caller or told to the mirror.
        let (metrics, replaced) = {
            let metrics = lock_metrics(&result.metrics);
            let replaced = result
                .mirror
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .replace(mirror);
            (metrics.clone(), replaced)
        };
        // Dropped once the lock is released, as the drop of a mirror may run code of the binding.
        drop(replaced);
        Some(metrics)
    }

    /// A clone of this observation that keeps `owner` alive while it is, as the plan of an
    /// `observe` of the Python client holds the Python Observation, which PySpark's plan holds
    /// itself: the other clones of its state, and its copies, do not hold it. Not a stable API.
    #[doc(hidden)]
    pub fn with_owner(&self, owner: Arc<dyn std::any::Any + Send + Sync>) -> Observation {
        Observation {
            state: Arc::clone(&self.state),
            owner: Some(owner),
        }
    }

    /// The owner that this clone of the observation keeps alive (see
    /// [`Observation::with_owner`]). Not a stable API.
    #[doc(hidden)]
    pub fn owner(&self) -> Option<&Arc<dyn std::any::Any + Send + Sync>> {
        self.owner.as_ref()
    }

    /// Whether `self` and `other` are the same observation, of which one is a clone.
    pub fn same(&self, other: &Observation) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }

    /// A copy of this observation, of its name and of its state as it is now, which the actions
    /// of the copy set and this one does not, as PySpark's `copy.deepcopy` of an Observation.
    /// Not a stable API.
    #[doc(hidden)]
    pub fn detached(&self) -> Observation {
        let state = self.lock();
        Observation {
            state: Arc::new(Mutex::new(ObservationState {
                name: state.name.clone(),
                result: state.result.as_ref().map(|result| {
                    Arc::new(UsedMetrics {
                        metrics: Mutex::new(lock_metrics(&result.metrics).clone()),
                        mirror: Default::default(),
                    })
                }),
                error: state.error.clone(),
                handle: None,
            })),
            owner: None,
        }
    }

    /// Keep `handle` for this observation, by which a binding of another language finds what
    /// it made of it from the Observations of a plan: the Python client keeps a weak reference to
    /// its Python Observation, which a deep copy of a DataFrame copies with the other objects of
    /// the call, as PySpark's deep copy of a plan copies the Observation that the plan holds.
    /// Not a stable API.
    #[doc(hidden)]
    pub fn set_handle(&self, handle: Arc<dyn std::any::Any + Send + Sync>) {
        self.lock().handle = Some(handle);
    }

    /// The handle of [`Observation::set_handle`], if any. Not a stable API.
    #[doc(hidden)]
    pub fn handle(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.lock().handle.clone()
    }

    /// A copy of this observation, of its name and of its state as it is now, as PySpark's
    /// `copy.copy` of an Observation, which shares its result dict: the copy of a used
    /// observation shares its metrics, which the actions that set those of this observation
    /// set, while an error that one of them reports later is set on this observation only. The
    /// copy of an unused observation is used on its own. Not a stable API.
    #[doc(hidden)]
    pub fn shallow_copy(&self) -> Observation {
        let state = self.lock();
        Observation {
            state: Arc::new(Mutex::new(ObservationState {
                name: state.name.clone(),
                result: state.result.clone(),
                error: state.error.clone(),
                handle: None,
            })),
            owner: None,
        }
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
        self.lock().result = Some(Arc::default());
    }

    /// Set the metrics that an action observed, as PySpark's client updates the result of
    /// the observation with them: a metric replaces the value of a metric of its name. The error
    /// of the mirror of a binding, which is told of them, is returned.
    pub(crate) fn update(
        &self,
        keys: &[String],
        values: &[proto::expression::Literal],
    ) -> Result<()> {
        let Some(result) = self.lock().result.clone() else {
            return Ok(());
        };
        {
            let mut metrics = lock_metrics(&result.metrics);
            for (key, value) in keys.iter().zip(values) {
                match metrics.iter_mut().find(|(name, _)| name == key) {
                    Some((_, existing)) => *existing = value.clone(),
                    None => metrics.push((key.clone(), value.clone())),
                }
            }
        }
        // Told once the locks are released, as a mirror may take a lock of its own, such as the
        // GIL of Python, under which a thread may want the locks of the observation.
        match result.mirror() {
            Some(mirror) => mirror.update(keys, values),
            None => Ok(()),
        }
    }

    /// Set the error of collecting the metrics, which `get` returns. The error of the mirror of a
    /// binding, which is told of it, is returned.
    pub(crate) fn set_error(&self, error: SparkError) -> Result<()> {
        let mirror = {
            let mut state = self.lock();
            state.error = Some(error.clone());
            state.result.as_ref().and_then(|result| result.mirror())
        };
        match mirror {
            Some(mirror) => mirror.error(&error),
            None => Ok(()),
        }
    }

    fn lock(&self) -> MutexGuard<'_, ObservationState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The lock of `metrics`, which may be taken while the state of the observation is locked, never
/// the other way around.
fn lock_metrics(metrics: &Mutex<Metrics>) -> MutexGuard<'_, Metrics> {
    metrics.lock().unwrap_or_else(|e| e.into_inner())
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
        clone
            .update(&["a".to_string(), "b".to_string()], &[long(1), long(2)])
            .unwrap();
        obs.update(&["b".to_string(), "c".to_string()], &[long(3), long(4)])
            .unwrap();
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

        obs.set_error(SparkError::connect_msg("failed")).unwrap();
        assert!(obs.get().unwrap_err().message().contains("failed"));
        assert!(format!("{obs:?}").contains(&name));
    }

    #[test]
    fn a_detached_observation_keeps_the_state_and_sets_only_its_own() {
        let obs = Observation::new("d");
        // A copy of an unused observation is unused too, and is used on its own.
        let unused = obs.detached();
        assert!(!unused.same(&obs));
        assert_eq!(unused.name().as_deref(), Some("d"));
        assert!(unused.get().is_err());
        unused.name_to_observe().unwrap();
        unused.start();
        assert!(obs.get().is_err());

        obs.start();
        obs.update(&["a".to_string()], &[long(1)]).unwrap();
        let copy = obs.detached();
        copy.update(&["a".to_string(), "b".to_string()], &[long(2), long(3)])
            .unwrap();
        obs.update(&["c".to_string()], &[long(4)]).unwrap();
        let values = |o: &Observation| -> Vec<(String, Option<LiteralType>)> {
            o.get()
                .unwrap()
                .into_iter()
                .map(|(name, value)| (name, value.literal_type))
                .collect()
        };
        assert_eq!(
            values(&obs),
            vec![
                ("a".to_string(), Some(LiteralType::Long(1))),
                ("c".to_string(), Some(LiteralType::Long(4))),
            ]
        );
        assert_eq!(
            values(&copy),
            vec![
                ("a".to_string(), Some(LiteralType::Long(2))),
                ("b".to_string(), Some(LiteralType::Long(3))),
            ]
        );

        // A failed observation keeps its error in the copy.
        let failed = Observation::new("f");
        failed.start();
        failed.set_error(SparkError::connect_msg("failed")).unwrap();
        assert!(failed
            .detached()
            .get()
            .unwrap_err()
            .message()
            .contains("failed"));
    }

    #[test]
    fn a_shallow_copy_shares_the_metrics_of_a_used_observation_but_not_a_later_error() {
        let names = |o: &Observation| -> Vec<String> {
            o.get().unwrap().into_iter().map(|(name, _)| name).collect()
        };
        // A copy of an unused observation is used on its own, and keeps no name that the
        // observation gets when it is used.
        let unnamed = Observation::unnamed();
        let unused = unnamed.shallow_copy();
        assert!(!unused.same(&unnamed));
        unnamed.name_to_observe().unwrap();
        unnamed.start();
        assert_eq!(unused.name(), None);
        assert!(unused.get().is_err());
        unused.name_to_observe().unwrap();

        // A copy of a used observation has the metrics that the actions set later too, as
        // PySpark's copy of an Observation shares its result dict, and is used already.
        let obs = Observation::new("s");
        obs.start();
        obs.update(&["a".to_string()], &[long(1)]).unwrap();
        let copy = obs.shallow_copy();
        assert!(!copy.same(&obs));
        assert_eq!(copy.name().as_deref(), Some("s"));
        obs.update(&["b".to_string()], &[long(2)]).unwrap();
        assert_eq!(names(&copy), vec!["a", "b"]);
        let err = copy.name_to_observe().unwrap_err();
        assert_eq!(err.get_condition().as_deref(), Some("REUSE_OBSERVATION"));

        // An error that an action reports later is set on the observation only, and a copy of a
        // failed observation keeps its error.
        obs.set_error(SparkError::connect_msg("failed")).unwrap();
        assert!(obs.get().is_err());
        assert_eq!(names(&copy), vec!["a", "b"]);
        let failed = obs.shallow_copy().get().unwrap_err();
        assert!(failed.message().contains("failed"));
    }

    /// A mirror that records what it is told, as the Python Observation of a binding, and fails
    /// as it is told to.
    #[derive(Default)]
    struct Recorded {
        told: Mutex<Vec<String>>,
        unused: std::sync::atomic::AtomicBool,
        failing: std::sync::atomic::AtomicBool,
    }

    impl MetricsMirror for Recorded {
        fn is_used(&self) -> Result<bool> {
            Ok(!self.unused.load(std::sync::atomic::Ordering::SeqCst))
        }

        fn update(&self, keys: &[String], _values: &[proto::expression::Literal]) -> Result<()> {
            if self.failing.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(SparkError::connect_msg("the update failed"));
            }
            self.told.lock().unwrap().extend(keys.iter().cloned());
            Ok(())
        }

        fn error(&self, error: &SparkError) -> Result<()> {
            self.told
                .lock()
                .unwrap()
                .push(format!("error {}", error.message()));
            Ok(())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    fn recorded(mirror: &Arc<dyn MetricsMirror>) -> &Recorded {
        mirror.as_any().downcast_ref::<Recorded>().unwrap()
    }

    fn told(mirror: &Arc<dyn MetricsMirror>) -> Vec<String> {
        recorded(mirror).told.lock().unwrap().clone()
    }

    /// A mirror that gives the name of the observation of its binding as it is now.
    struct Named(Mutex<Result<String>>);

    impl MetricsMirror for Named {
        fn is_used(&self) -> Result<bool> {
            Ok(true)
        }

        fn update(&self, _keys: &[String], _values: &[proto::expression::Literal]) -> Result<()> {
            Ok(())
        }

        fn name(&self) -> Option<Result<String>> {
            Some(self.0.lock().unwrap().clone())
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn a_plan_of_an_observation_is_built_of_the_name_that_its_mirror_gives_then() {
        let built_name = |plan: &crate::plan::LogicalPlan| match plan.try_to_proto() {
            Ok(proto::Relation {
                rel_type: Some(proto::relation::RelType::CollectMetrics(metrics)),
                ..
            }) => Ok(metrics.name),
            Ok(other) => panic!("a CollectMetrics, not {other:?}"),
            Err(error) => Err(error),
        };
        let names = |plan: &crate::plan::LogicalPlan| -> Vec<String> {
            plan.observations()
                .into_iter()
                .map(|(name, _)| name)
                .collect()
        };
        let obs = Observation::used("first");
        let plan = crate::plan::LogicalPlan::Observe {
            input: Arc::new(crate::plan::range(0, 3, 1)).into(),
            name: "first".to_string(),
            exprs: vec![crate::expression::Expression::column("id")],
            observation: Some(obs.clone()),
        };
        // Without a mirror of a name, the plan is of the name of the observation.
        assert!(obs.name_of_plan().is_none());
        assert_eq!(built_name(&plan).unwrap(), "first");
        assert_eq!(names(&plan), vec!["first"]);

        // With one, of the name that it gives at each build, by which the plan names it.
        let mirror = Arc::new(Named(Mutex::new(Ok("now".to_string()))));
        assert!(obs
            .set_metrics_mirror(Arc::clone(&mirror) as Arc<dyn MetricsMirror>)
            .is_some());
        assert_eq!(built_name(&plan).unwrap(), "now");
        assert_eq!(names(&plan), vec!["now"]);

        // Its error is that of each build, until it gives a name again.
        *mirror.0.lock().unwrap() = Err(SparkError::connect_msg("no name"));
        for _ in 0..2 {
            let error = built_name(&plan).unwrap_err();
            assert!(error.message().contains("no name"), "{error}");
        }
        *mirror.0.lock().unwrap() = Ok("again".to_string());
        assert_eq!(built_name(&plan).unwrap(), "again");
        assert_eq!(names(&plan), vec!["again"]);
    }

    #[test]
    fn a_mirror_is_told_of_each_update_and_error_and_says_whether_it_is_used() {
        // An unused observation has no mirror, as PySpark's has no result before `_on`.
        let obs = Observation::new("m");
        assert!(!obs.is_used().unwrap());
        let first: Arc<dyn MetricsMirror> = Arc::new(Recorded::default());
        assert!(obs.set_metrics_mirror(Arc::clone(&first)).is_none());
        obs.start();
        assert!(obs.is_used().unwrap());
        obs.update(&["a".to_string()], &[long(1)]).unwrap();
        assert!(obs.metrics_mirror().is_none());

        // It is set with the metrics so far, which the caller fills it with.
        let metrics = obs.set_metrics_mirror(Arc::clone(&first)).unwrap();
        let names: Vec<String> = metrics.into_iter().map(|(name, _)| name).collect();
        assert_eq!(names, vec!["a"]);
        assert!(Arc::ptr_eq(&obs.metrics_mirror().unwrap(), &first));

        // Each update is told to it, after the metrics are set, also through a clone, and a
        // shallow copy shares it; a detached copy does not.
        obs.update(&["b".to_string(), "c".to_string()], &[long(2), long(3)])
            .unwrap();
        obs.clone().update(&["a".to_string()], &[long(4)]).unwrap();
        assert_eq!(told(&first), vec!["b", "c", "a"]);
        let shallow = obs.shallow_copy();
        assert!(Arc::ptr_eq(&first, &shallow.metrics_mirror().unwrap()));
        let detached = obs.detached();
        assert!(detached.metrics_mirror().is_none());
        detached.update(&["d".to_string()], &[long(5)]).unwrap();
        assert_eq!(told(&first), vec!["b", "c", "a"]);
        assert_eq!(detached.get().unwrap().len(), 4);

        // It says whether the observation is used, and its errors are those of the update.
        recorded(&first)
            .unused
            .store(true, std::sync::atomic::Ordering::SeqCst);
        assert!(!obs.is_used().unwrap());
        recorded(&first)
            .failing
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let failed = obs.update(&["e".to_string()], &[long(6)]).unwrap_err();
        assert!(failed.message().contains("the update failed"));

        // Another mirror replaces it, of the metrics so far, and is told of the error.
        let second: Arc<dyn MetricsMirror> = Arc::new(Recorded::default());
        assert_eq!(
            obs.set_metrics_mirror(Arc::clone(&second)).unwrap().len(),
            4
        );
        assert!(Arc::ptr_eq(&obs.metrics_mirror().unwrap(), &second));
        obs.set_error(SparkError::connect_msg("failed")).unwrap();
        assert_eq!(told(&second), vec!["error failed"]);
        assert!(obs.get().unwrap_err().message().contains("failed"));
    }

    #[test]
    fn an_owner_is_kept_by_its_clone_and_its_clones_only() {
        // The owner of a binding, as the Python Observation that the plan of an `observe` holds:
        // the clone that has it shares the state, and its own clones have it, but not the
        // observation it was made of, nor a copy.
        let obs = Observation::new("o");
        let owner: Arc<dyn std::any::Any + Send + Sync> = Arc::new("python");
        let held = obs.with_owner(Arc::clone(&owner));
        assert!(held.same(&obs));
        assert!(obs.owner().is_none());
        assert!(Arc::ptr_eq(held.owner().unwrap(), &owner));
        assert!(Arc::ptr_eq(held.clone().owner().unwrap(), &owner));
        assert!(held.detached().owner().is_none());
        assert!(held.shallow_copy().owner().is_none());
        drop(held);
        assert_eq!(Arc::strong_count(&owner), 1);
    }

    #[test]
    fn a_handle_is_kept_by_the_clones_of_an_observation_and_not_by_its_copies() {
        // The handle of a binding, which finds its object by it from a clone of the observation
        // that a plan holds; a copy is an observation of its own, which has none.
        let obs = Observation::new("h");
        assert!(obs.handle().is_none());
        obs.set_handle(Arc::new("python"));
        let kept = obs.clone().handle().unwrap();
        assert_eq!(kept.downcast_ref::<&str>(), Some(&"python"));
        assert!(obs.detached().handle().is_none());
        assert!(obs.shallow_copy().handle().is_none());
    }
}

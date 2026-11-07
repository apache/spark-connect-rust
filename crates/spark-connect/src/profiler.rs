//! The updates of the Python accumulators of the executions of a session, as PySpark Connect's
//! client receives them: the server sends each pickled `(id, update)` in the
//! `__python_accumulator__` observed metrics of an execution, of which the UDF profilers
//! (`spark.sql.pyspark.udf.profiler`) send their results. PySpark's client unpickles them with
//! `specialAccumulatorSer` into its `ConnectProfilerCollector`, of `spark.profile`.

use std::sync::{Arc, Mutex};

use spark_connect_proto as proto;

/// The name of the observed metrics of the updates of the Python accumulators.
pub const PYTHON_ACCUMULATOR_METRICS: &str = "__python_accumulator__";

/// A handler of the pickled `(id, update)` of each update of the Python accumulators.
pub type PythonAccumulatorHandler = Arc<dyn Fn(&[u8]) + Send + Sync>;

/// The handler of the updates of the Python accumulators of a session, shared by its clones.
#[derive(Clone, Default)]
pub(crate) struct PythonAccumulators {
    handler: Arc<Mutex<Option<PythonAccumulatorHandler>>>,
}

impl PythonAccumulators {
    pub(crate) fn set_handler(&self, handler: PythonAccumulatorHandler) {
        *self.handler.lock().unwrap() = Some(handler);
    }

    /// Hand each pickled update of the `__python_accumulator__` metrics of `metrics` to the
    /// handler, as PySpark's client takes each of their values.
    pub(crate) fn accumulate(&self, metrics: &[proto::execute_plan_response::ObservedMetrics]) {
        let Some(handler) = self.handler.lock().unwrap().clone() else {
            return;
        };
        for metric in metrics
            .iter()
            .filter(|metric| metric.name == PYTHON_ACCUMULATOR_METRICS)
        {
            for value in &metric.values {
                if let Some(proto::expression::literal::LiteralType::Binary(update)) =
                    &value.literal_type
                {
                    handler(update);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(
        name: &str,
        values: Vec<proto::expression::Literal>,
    ) -> proto::execute_plan_response::ObservedMetrics {
        proto::execute_plan_response::ObservedMetrics {
            name: name.to_string(),
            values,
            ..Default::default()
        }
    }

    fn binary(bytes: &[u8]) -> proto::expression::Literal {
        proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::Binary(
                bytes.to_vec().into(),
            )),
            ..Default::default()
        }
    }

    /// The handler takes each binary value of the `__python_accumulator__` metrics, and none of
    /// the metrics of observations.
    #[test]
    fn the_handler_takes_the_updates_of_the_python_accumulators() {
        let accumulators = PythonAccumulators::default();
        // Without a handler, the updates are not kept.
        accumulators.accumulate(&[metrics(PYTHON_ACCUMULATOR_METRICS, vec![binary(b"a")])]);
        let updates = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&updates);
        accumulators.set_handler(Arc::new(move |update: &[u8]| {
            seen.lock().unwrap().push(update.to_vec());
        }));
        accumulators.accumulate(&[
            metrics(PYTHON_ACCUMULATOR_METRICS, vec![binary(b"b"), binary(b"c")]),
            metrics("observation", vec![binary(b"d")]),
        ]);
        assert_eq!(*updates.lock().unwrap(), vec![b"b".to_vec(), b"c".to_vec()]);
    }
}

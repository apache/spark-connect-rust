// SPDX-License-Identifier: Apache-2.0
//! Standard Spark ML evaluators backed by the Connect Evaluate command.
use super::{evaluate_via_command, read_operator, Evaluator, MlOperator, OperatorType, Params};
use crate::{DataFrame, SparkSession};

macro_rules! evaluator {
    ($name:ident, $larger:expr) => {
        #[doc = concat!("Spark ML `", stringify!($name), "` evaluated on the Connect server.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            operator: MlOperator,
            params: Params,
        }
        impl $name {
            pub fn new() -> Self {
                Self {
                    operator: MlOperator::new(
                        concat!("org.apache.spark.ml.evaluation.", stringify!($name)),
                        OperatorType::Evaluator,
                    ),
                    params: Params::new(),
                }
            }
            /// Mutable explicit parameters; unset parameters retain Spark's defaults.
            pub fn params_mut(&mut self) -> &mut Params {
                &mut self.params
            }
            /// Whether this evaluator's current metric should be maximized.
            pub fn is_larger_better(&self) -> bool {
                ($larger)(self.metric_name())
            }

            #[doc = concat!("Load a `", stringify!($name), "` saved to `path`, as PySpark's `", stringify!($name), ".load(path)`: its uid and params.")]
            pub fn load(session: &SparkSession, path: &str) -> spark_connect_core::error::Result<Self> {
                let (operator, params) = read_operator(
                    session,
                    concat!("org.apache.spark.ml.evaluation.", stringify!($name)),
                    OperatorType::Evaluator,
                    path,
                )?;
                Ok(Self { operator, params })
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl Evaluator for $name {
            fn operator(&self) -> &MlOperator {
                &self.operator
            }
            fn params(&self) -> &Params {
                &self.params
            }
            fn params_mut(&mut self) -> Option<&mut Params> {
                Some(&mut self.params)
            }
            fn is_larger_better(&self) -> bool {
                self.is_larger_better()
            }
            fn evaluate(&self, dataset: &DataFrame) -> spark_connect_core::error::Result<f64> {
                evaluate_via_command(&self.operator, &self.params, dataset)
            }
        }
    };
}

evaluator!(BinaryClassificationEvaluator, |_: &str| true);
impl BinaryClassificationEvaluator {
    /// Set `rawPredictionCol`; Spark validates its value during evaluation.
    pub fn set_raw_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("rawPredictionCol", value);
        self
    }
    /// Read `rawPredictionCol`, using Spark's default when it is unset.
    pub fn raw_prediction_col(&self) -> &str {
        self.params.string_or("rawPredictionCol", "rawPrediction")
    }
    /// Set `labelCol`; Spark validates its value during evaluation.
    pub fn set_label_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", value);
        self
    }
    /// Read `labelCol`, using Spark's default when it is unset.
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }
    /// Set `metricName`; Spark validates its value during evaluation.
    pub fn set_metric_name(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("metricName", value);
        self
    }
    /// Read `metricName`, using Spark's default when it is unset.
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "areaUnderROC")
    }
    /// Set `weightCol`; Spark validates its value during evaluation.
    pub fn set_weight_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("weightCol", value);
        self
    }
    /// Read `weightCol`, using Spark's default when it is unset.
    pub fn weight_col(&self) -> Option<&str> {
        self.params.string_opt("weightCol")
    }
    /// Set `numBins`; Spark validates its value during evaluation.
    pub fn set_num_bins(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("numBins", value);
        self
    }
    /// Read `numBins`, using Spark's default when it is unset.
    pub fn num_bins(&self) -> i32 {
        self.params.int_or("numBins", 1000)
    }
}

evaluator!(RegressionEvaluator, |metric: &str| matches!(
    metric,
    "r2" | "var"
));
impl RegressionEvaluator {
    /// Set `predictionCol`; Spark validates its value during evaluation.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Read `predictionCol`, using Spark's default when it is unset.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
    /// Set `labelCol`; Spark validates its value during evaluation.
    pub fn set_label_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", value);
        self
    }
    /// Read `labelCol`, using Spark's default when it is unset.
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }
    /// Set `metricName`; Spark validates its value during evaluation.
    pub fn set_metric_name(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("metricName", value);
        self
    }
    /// Read `metricName`, using Spark's default when it is unset.
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "rmse")
    }
    /// Set `weightCol`; Spark validates its value during evaluation.
    pub fn set_weight_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("weightCol", value);
        self
    }
    /// Read `weightCol`, using Spark's default when it is unset.
    pub fn weight_col(&self) -> Option<&str> {
        self.params.string_opt("weightCol")
    }
    /// Set `throughOrigin`; Spark validates its value during evaluation.
    pub fn set_through_origin(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("throughOrigin", value);
        self
    }
    /// Read `throughOrigin`, using Spark's default when it is unset.
    pub fn through_origin(&self) -> bool {
        self.params.bool_or("throughOrigin", false)
    }
}

evaluator!(MulticlassClassificationEvaluator, |metric: &str| !matches!(
    metric,
    "weightedFalsePositiveRate" | "falsePositiveRateByLabel" | "logLoss" | "hammingLoss"
));
impl MulticlassClassificationEvaluator {
    /// Set `predictionCol`; Spark validates its value during evaluation.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Read `predictionCol`, using Spark's default when it is unset.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
    /// Set `labelCol`; Spark validates its value during evaluation.
    pub fn set_label_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", value);
        self
    }
    /// Read `labelCol`, using Spark's default when it is unset.
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }
    /// Set `metricName`; Spark validates its value during evaluation.
    pub fn set_metric_name(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("metricName", value);
        self
    }
    /// Read `metricName`, using Spark's default when it is unset.
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "f1")
    }
    /// Set `weightCol`; Spark validates its value during evaluation.
    pub fn set_weight_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("weightCol", value);
        self
    }
    /// Read `weightCol`, using Spark's default when it is unset.
    pub fn weight_col(&self) -> Option<&str> {
        self.params.string_opt("weightCol")
    }
    /// Set `metricLabel`; Spark validates its value during evaluation.
    pub fn set_metric_label(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("metricLabel", value);
        self
    }
    /// Read `metricLabel`, using Spark's default when it is unset.
    pub fn metric_label(&self) -> f64 {
        self.params.double_or("metricLabel", 0.0)
    }
    /// Set `beta`; Spark validates its value during evaluation.
    pub fn set_beta(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("beta", value);
        self
    }
    /// Read `beta`, using Spark's default when it is unset.
    pub fn beta(&self) -> f64 {
        self.params.double_or("beta", 1.0)
    }
    /// Set `probabilityCol`; Spark validates its value during evaluation.
    pub fn set_probability_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("probabilityCol", value);
        self
    }
    /// Read `probabilityCol`, using Spark's default when it is unset.
    pub fn probability_col(&self) -> &str {
        self.params.string_or("probabilityCol", "probability")
    }
    /// Set `eps`; Spark validates its value during evaluation.
    pub fn set_eps(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("eps", value);
        self
    }
    /// Read `eps`, using Spark's default when it is unset.
    pub fn eps(&self) -> f64 {
        self.params.double_or("eps", 1e-15)
    }
}

evaluator!(MultilabelClassificationEvaluator, |metric: &str| metric
    != "hammingLoss");
impl MultilabelClassificationEvaluator {
    /// Set `predictionCol`; Spark validates its value during evaluation.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Read `predictionCol`, using Spark's default when it is unset.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
    /// Set `labelCol`; Spark validates its value during evaluation.
    pub fn set_label_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", value);
        self
    }
    /// Read `labelCol`, using Spark's default when it is unset.
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }
    /// Set `metricName`; Spark validates its value during evaluation.
    pub fn set_metric_name(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("metricName", value);
        self
    }
    /// Read `metricName`, using Spark's default when it is unset.
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "f1Measure")
    }
    /// Set `metricLabel`; Spark validates its value during evaluation.
    pub fn set_metric_label(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("metricLabel", value);
        self
    }
    /// Read `metricLabel`, using Spark's default when it is unset.
    pub fn metric_label(&self) -> f64 {
        self.params.double_or("metricLabel", 0.0)
    }
}

evaluator!(ClusteringEvaluator, |_: &str| true);
impl ClusteringEvaluator {
    /// Set `predictionCol`; Spark validates its value during evaluation.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Read `predictionCol`, using Spark's default when it is unset.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
    /// Set `featuresCol`; Spark validates its value during evaluation.
    pub fn set_features_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("featuresCol", value);
        self
    }
    /// Read `featuresCol`, using Spark's default when it is unset.
    pub fn features_col(&self) -> &str {
        self.params.string_or("featuresCol", "features")
    }
    /// Set `metricName`; Spark validates its value during evaluation.
    pub fn set_metric_name(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("metricName", value);
        self
    }
    /// Read `metricName`, using Spark's default when it is unset.
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "silhouette")
    }
    /// Set `distanceMeasure`; Spark validates its value during evaluation.
    pub fn set_distance_measure(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("distanceMeasure", value);
        self
    }
    /// Read `distanceMeasure`, using Spark's default when it is unset.
    pub fn distance_measure(&self) -> &str {
        self.params.string_or("distanceMeasure", "squaredEuclidean")
    }
    /// Set `weightCol`; Spark validates its value during evaluation.
    pub fn set_weight_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("weightCol", value);
        self
    }
    /// Read `weightCol`, using Spark's default when it is unset.
    pub fn weight_col(&self) -> Option<&str> {
        self.params.string_opt("weightCol")
    }
}

evaluator!(RankingEvaluator, |_: &str| true);
impl RankingEvaluator {
    /// Set `predictionCol`; Spark validates its value during evaluation.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Read `predictionCol`, using Spark's default when it is unset.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
    /// Set `labelCol`; Spark validates its value during evaluation.
    pub fn set_label_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("labelCol", value);
        self
    }
    /// Read `labelCol`, using Spark's default when it is unset.
    pub fn label_col(&self) -> &str {
        self.params.string_or("labelCol", "label")
    }
    /// Set `metricName`; Spark validates its value during evaluation.
    pub fn set_metric_name(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("metricName", value);
        self
    }
    /// Read `metricName`, using Spark's default when it is unset.
    pub fn metric_name(&self) -> &str {
        self.params.string_or("metricName", "meanAveragePrecision")
    }
    /// Set `k`; Spark validates its value during evaluation.
    pub fn set_k(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("k", value);
        self
    }
    /// Read `k`, using Spark's default when it is unset.
    pub fn k(&self) -> i32 {
        self.params.int_or("k", 10)
    }
}

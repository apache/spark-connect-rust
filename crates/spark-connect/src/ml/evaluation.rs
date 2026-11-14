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

            /// Set the uid of the operator, as `_resetUid` of a load does: a loaded evaluator of a
            /// default instance keeps the uid that it was saved with.
            pub fn set_uid(&mut self, uid: &str) {
                self.operator.uid = uid.to_string();
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
ml_params! {
    BinaryClassificationEvaluator {
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_metric_name(value: &str) => "metricName", set_param_string;
        metric_name -> &str = |this| this.params.string_or("metricName", "areaUnderROC");
        set_num_bins(value: i32) => "numBins", set_param_int32;
        num_bins -> i32 = |this| this.params.int_or("numBins", 1000);
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

evaluator!(RegressionEvaluator, |metric: &str| matches!(
    metric,
    "r2" | "var"
));
ml_params! {
    RegressionEvaluator {
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_metric_name(value: &str) => "metricName", set_param_string;
        metric_name -> &str = |this| this.params.string_or("metricName", "rmse");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_through_origin(value: bool) => "throughOrigin", set_param_bool;
        through_origin -> bool = |this| this.params.bool_or("throughOrigin", false);
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

evaluator!(MulticlassClassificationEvaluator, |metric: &str| !matches!(
    metric,
    "weightedFalsePositiveRate" | "falsePositiveRateByLabel" | "logLoss" | "hammingLoss"
));
ml_params! {
    MulticlassClassificationEvaluator {
        set_beta(value: f64) => "beta", set_param_double;
        beta -> f64 = |this| this.params.double_or("beta", 1.0);
        set_eps(value: f64) => "eps", set_param_double;
        eps -> f64 = |this| this.params.double_or("eps", 1e-15);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_metric_label(value: f64) => "metricLabel", set_param_double;
        metric_label -> f64 = |this| this.params.double_or("metricLabel", 0.0);
        set_metric_name(value: &str) => "metricName", set_param_string;
        metric_name -> &str = |this| this.params.string_or("metricName", "f1");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_probability_col(value: &str) => "probabilityCol", set_param_string;
        probability_col -> &str = |this| this.params.string_or("probabilityCol", "probability");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

evaluator!(MultilabelClassificationEvaluator, |metric: &str| metric
    != "hammingLoss");
ml_params! {
    MultilabelClassificationEvaluator {
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_metric_label(value: f64) => "metricLabel", set_param_double;
        metric_label -> f64 = |this| this.params.double_or("metricLabel", 0.0);
        set_metric_name(value: &str) => "metricName", set_param_string;
        metric_name -> &str = |this| this.params.string_or("metricName", "f1Measure");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
    }
}

evaluator!(ClusteringEvaluator, |_: &str| true);
ml_params! {
    ClusteringEvaluator {
        set_distance_measure(value: &str) => "distanceMeasure", set_param_string;
        distance_measure -> &str =
            |this| this.params.string_or("distanceMeasure", "squaredEuclidean");
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_metric_name(value: &str) => "metricName", set_param_string;
        metric_name -> &str = |this| this.params.string_or("metricName", "silhouette");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

evaluator!(RankingEvaluator, |_: &str| true);
ml_params! {
    RankingEvaluator {
        set_k(value: i32) => "k", set_param_int32;
        k -> i32 = |this| this.params.int_or("k", 10);
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_metric_name(value: &str) => "metricName", set_param_string;
        metric_name -> &str = |this| this.params.string_or("metricName", "meanAveragePrecision");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Standard Spark ML feature stages. Fitting and transformation execute on Spark Connect.
use super::*;

macro_rules! feature_stage {
    ($name:ident, $kind:ident) => {
        #[doc = concat!("Spark ML `", stringify!($name), "` feature stage.")]
        #[derive(Debug, Clone)]
        pub struct $name {
            operator: MlOperator,
            params: Params,
        }
        impl $name {
            pub fn new() -> Self {
                Self {
                    operator: MlOperator::new(
                        concat!("org.apache.spark.ml.feature.", stringify!($name)),
                        OperatorType::$kind,
                    ),
                    params: Params::new(),
                }
            }
        }
        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
        impl $name {
            #[doc = concat!("Load a `", stringify!($name), "` saved to `path`, as PySpark's `", stringify!($name), ".load(path)`.")]
            pub fn load(session: &SparkSession, path: &str) -> spark_connect_core::error::Result<Self> {
                let jvm_class = concat!("org.apache.spark.ml.feature.", stringify!($name));
                let (operator, params) = read_operator(session, jvm_class, OperatorType::$kind, path)?;
                Ok(Self { operator, params })
            }
        }
    };
}

macro_rules! feature_estimator {
    ($name:ident, $model:ident, $jvm_model:literal, [$($model_param:literal),* $(,)?], no_model_load) => {
        feature_estimator!(@stage $name, $model, $jvm_model, [$($model_param),*]);
    };
    ($name:ident, $model:ident, $jvm_model:literal, [$($model_param:literal),* $(,)?]) => {
        feature_estimator!(@stage $name, $model, $jvm_model, [$($model_param),*]);
        impl $model {
            #[doc = concat!("Load a `", stringify!($model), "` saved to `path`, as PySpark's `", stringify!($model), ".load(path)`: the server caches it, as a fitted model.")]
            pub fn load(session: &SparkSession, path: &str) -> spark_connect_core::error::Result<Self> {
                let jvm_class = concat!("org.apache.spark.ml.feature.", $jvm_model);
                let (cached, operator, params) = read_model(session, jvm_class, path)?;
                Ok(Self { operator, params, cached })
            }
        }
    };
    (@stage $name:ident, $model:ident, $jvm_model:literal, [$($model_param:literal),* $(,)?]) => {
        feature_stage!($name, Estimator);
        impl $name {
            #[doc = concat!("Fit a typed `", stringify!($model), "`, as `Estimator::fit` fits it, which boxes it.")]
            pub fn fit(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<$model> {
                self.ensure_operator();
                let info = fit_via_command(&self.operator, &self.params, df)?;
                let cached = CachedModel::from_info(&info, &df.session, stringify!($name))?;
                // A model need not accept every training parameter (LSH seed, discretizer buckets).
                let mut params = self.params.clone();
                params.params.retain(|name, _| [$($model_param),*].contains(&name.as_str()));
                Ok($model {
                    operator: MlOperator::with_uid(concat!("org.apache.spark.ml.feature.", $jvm_model), info.uid.as_deref().unwrap_or(&self.operator.uid), OperatorType::Model),
                    params, cached,
                })
            }
        }
        impl Estimator for $name {
            fn operator(&self) -> &MlOperator { &self.operator }
            fn operator_mut(&mut self) -> &mut MlOperator { &mut self.operator }
            fn params(&self) -> &Params { &self.params }
            fn params_mut(&mut self) -> &mut Params { &mut self.params }
            fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
                Ok(Box::new(Self::fit(self, df)?))
            }
        }
        #[doc = concat!("A fitted `", stringify!($name), "`, cached in its Spark session.")]
        #[derive(Debug, Clone)]
        pub struct $model { operator: MlOperator, params: Params, cached: CachedModel }
        impl Transformer for $model {
            fn operator(&self) -> &MlOperator { &self.operator }
            fn operator_mut(&mut self) -> &mut MlOperator { &mut self.operator }
            fn params(&self) -> &Params { &self.params }
            fn params_mut(&mut self) -> &mut Params { &mut self.params }
            fn build_ml_relation(&self, df: &DataFrame) -> spark_connect_core::error::Result<proto::MlRelation> {
                self.cached.build_ml_relation(df, &self.params)
            }
            fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> { self.cached.transform(df, &self.params) }
        }
        impl Model for $model {
            fn clone_box(&self) -> Box<dyn Model> { Box::new(self.clone()) }
            fn ml_ref(&self) -> Option<&MlRef> { Some(&self.cached) }
        }
    };
}

macro_rules! feature_transformer {
    ($name:ident) => {
        feature_stage!($name, Transformer);
        impl $name {
            #[doc = concat!("The writer of this `", stringify!($name), "`, as PySpark's `write()`, which saves it of the session that `MlWriter::session` gives it.")]
            pub fn write(&self) -> MlWriter {
                MlWriter::operator(&self.operator, &self.params)
            }
        }
        impl Transformer for $name {
            fn operator(&self) -> &MlOperator {
                &self.operator
            }
            fn operator_mut(&mut self) -> &mut MlOperator {
                &mut self.operator
            }
            fn params(&self) -> &Params {
                &self.params
            }
            fn params_mut(&mut self) -> &mut Params {
                &mut self.params
            }
            fn transform_impl(
                &mut self,
                df: &DataFrame,
            ) -> spark_connect_core::error::Result<DataFrame> {
                let ml_relation = proto::Relation {
                    rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                        self.build_ml_relation(df)?,
                    ))),
                    ..Default::default()
                };
                Ok(DataFrame::new(
                    df.session.clone(),
                    LogicalPlan::MlTransform {
                        ml_relation,
                        input: df.plan.clone(),
                        owner: None,
                    },
                ))
            }
        }
    };
}

// The feature area's rows use the shared `ml_params!` macro of `ml.rs`.
macro_rules! feature_params {
    ($($t:tt)*) => {
        ml_params! { $($t)* }
    };
}

feature_transformer!(Binarizer);
feature_params! { Binarizer {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) => "outputCols", set_param_string_array; output_cols -> Vec<String> = |this| this.params.string_array("outputCols");
    set_threshold(value: f64) => "threshold", set_param_double; threshold -> f64 = |this| this.params.double_or("threshold", 0.0);
    set_thresholds(value: impl IntoIterator<Item = f64>) => "thresholds", set_param_double_array; thresholds -> Option<Vec<f64>> = |this| this.params.double_array_opt("thresholds");
} }

feature_estimator!(
    BucketedRandomProjectionLSH,
    BucketedRandomProjectionLSHModel,
    "BucketedRandomProjectionLSHModel",
    ["bucketLength", "inputCol", "numHashTables", "outputCol"]
);
feature_params! { BucketedRandomProjectionLSH {
    set_bucket_length(value: f64) => "bucketLength", set_param_double; bucket_length -> Option<f64> = |this| this.params.get_param("bucketLength").map(|_| this.params.double_or("bucketLength", 0.0));
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_num_hash_tables(value: i32) => "numHashTables", set_param_int32; num_hash_tables -> i32 = |this| this.params.int_or("numHashTables", 1);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_seed(value: i64) => "seed", set_param_int_narrow; seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash("org.apache.spark.ml.feature.BucketedRandomProjectionLSH"));
} }
feature_params! { BucketedRandomProjectionLSHModel {
    bucket_length -> Option<f64> =
        |this| this.params.double_opt("bucketLength") => "bucketLength";
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    num_hash_tables -> i32 = |this| this.params.int_or("numHashTables", 1) => "numHashTables";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_transformer!(Bucketizer);
feature_params! { Bucketizer {
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string; handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) => "outputCols", set_param_string_array; output_cols -> Vec<String> = |this| this.params.string_array("outputCols");
    set_splits(value: impl IntoIterator<Item = f64>) => "splits", set_param_double_array; splits -> Option<Vec<f64>> = |this| this.params.double_array_opt("splits");
    set_splits_array(value: impl IntoIterator<Item = impl IntoIterator<Item = f64>>) => "splitsArray", set_param_double_arrays; splits_array -> Option<Vec<Vec<f64>>> = |this| this.params.array_opt("splitsArray").map(|values| values.iter().map(|v| match &v.literal_type { Some(proto::expression::literal::LiteralType::Array(a)) => a.elements.iter().filter_map(|v| match v.literal_type { Some(proto::expression::literal::LiteralType::Double(v)) => Some(v), _ => None }).collect(), _ => Vec::new() }).collect());
} }

feature_estimator!(
    ChiSqSelector,
    ChiSqSelectorModel,
    "ChiSqSelectorModel",
    [
        "fdr",
        "featuresCol",
        "fpr",
        "fwe",
        "labelCol",
        "numTopFeatures",
        "outputCol",
        "percentile",
        "selectorType"
    ]
);
feature_params! { ChiSqSelector {
    set_fdr(value: f64) => "fdr", set_param_double; fdr -> f64 = |this| this.params.double_or("fdr", 0.05);
    set_features_col(value: &str) => "featuresCol", set_param_string; features_col -> &str = |this| this.params.string_or("featuresCol", "features");
    set_fpr(value: f64) => "fpr", set_param_double; fpr -> f64 = |this| this.params.double_or("fpr", 0.05);
    set_fwe(value: f64) => "fwe", set_param_double; fwe -> f64 = |this| this.params.double_or("fwe", 0.05);
    set_label_col(value: &str) => "labelCol", set_param_string; label_col -> &str = |this| this.params.string_or("labelCol", "label");
    set_num_top_features(value: i32) => "numTopFeatures", set_param_int32; num_top_features -> i32 = |this| this.params.int_or("numTopFeatures", 50);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_percentile(value: f64) => "percentile", set_param_double; percentile -> f64 = |this| this.params.double_or("percentile", 0.1);
    set_selector_type(value: &str) => "selectorType", set_param_string; selector_type -> &str = |this| this.params.string_or("selectorType", "numTopFeatures");
} }
feature_params! { ChiSqSelectorModel {
    fdr -> f64 = |this| this.params.double_or("fdr", 0.05) => "fdr";
    set_features_col(value: &str) => "featuresCol", set_param_string;
    features_col -> &str = |this| this.params.string_or("featuresCol", "features");
    fpr -> f64 = |this| this.params.double_or("fpr", 0.05) => "fpr";
    fwe -> f64 = |this| this.params.double_or("fwe", 0.05) => "fwe";
    label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
    num_top_features -> i32 =
        |this| this.params.int_or("numTopFeatures", 50) => "numTopFeatures";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    percentile -> f64 = |this| this.params.double_or("percentile", 0.1) => "percentile";
    selector_type -> &str =
        |this| this.params.string_or("selectorType", "numTopFeatures") => "selectorType";
} }

feature_estimator!(
    CountVectorizer,
    CountVectorizerModel,
    "CountVectorizerModel",
    [
        "binary",
        "inputCol",
        "maxDF",
        "minDF",
        "minTF",
        "outputCol",
        "vocabSize"
    ]
);
feature_params! { CountVectorizer {
    set_binary(value: bool) => "binary", set_param_bool; binary -> bool = |this| this.params.bool_or("binary", false);
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_max_df(value: f64) => "maxDF", set_param_double; max_df -> f64 = |this| this.params.double_or("maxDF", 9.223372036854776e+18);
    set_min_df(value: f64) => "minDF", set_param_double; min_df -> f64 = |this| this.params.double_or("minDF", 1.0);
    set_min_tf(value: f64) => "minTF", set_param_double; min_tf -> f64 = |this| this.params.double_or("minTF", 1.0);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_vocab_size(value: i32) => "vocabSize", set_param_int32; vocab_size -> i32 = |this| this.params.int_or("vocabSize", 262144);
} }
feature_params! { CountVectorizerModel {
    set_binary(value: bool) => "binary", set_param_bool;
    binary -> bool = |this| this.params.bool_or("binary", false);
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    max_df -> f64 = |this| this.params.double_or("maxDF", 9.223372036854776e+18) => "maxDF";
    min_df -> f64 = |this| this.params.double_or("minDF", 1.0) => "minDF";
    set_min_tf(value: f64) => "minTF", set_param_double;
    min_tf -> f64 = |this| this.params.double_or("minTF", 1.0);
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    vocab_size -> i32 = |this| this.params.int_or("vocabSize", 262144) => "vocabSize";
} }

feature_transformer!(DCT);
feature_params! { DCT {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_inverse(value: bool) => "inverse", set_param_bool; inverse -> bool = |this| this.params.bool_or("inverse", false);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_transformer!(ElementwiseProduct);
feature_params! { ElementwiseProduct {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_scaling_vec(value: impl Into<MlVector>) => "scalingVec", set_param_vector; scaling_vec -> Option<MlVector> = |this| MlVector::from_literal(this.params.literal("scalingVec")?);
} }

feature_transformer!(FeatureHasher);
feature_params! { FeatureHasher {
    set_categorical_cols(value: impl IntoIterator<Item = impl Into<String>>) => "categoricalCols", set_param_string_array; categorical_cols -> Vec<String> = |this| this.params.string_array("categoricalCols");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_num_features(value: i32) => "numFeatures", set_param_int32; num_features -> i32 = |this| this.params.int_or("numFeatures", 262144);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_transformer!(HashingTF);
feature_params! { HashingTF {
    set_binary(value: bool) => "binary", set_param_bool; binary -> bool = |this| this.params.bool_or("binary", false);
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_num_features(value: i32) => "numFeatures", set_param_int32; num_features -> i32 = |this| this.params.int_or("numFeatures", 262144);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_estimator!(
    IDF,
    IDFModel,
    "IDFModel",
    ["inputCol", "minDocFreq", "outputCol"]
);
feature_params! { IDF {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_min_doc_freq(value: i32) => "minDocFreq", set_param_int32; min_doc_freq -> i32 = |this| this.params.int_or("minDocFreq", 0);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }
feature_params! { IDFModel {
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    min_doc_freq -> i32 = |this| this.params.int_or("minDocFreq", 0) => "minDocFreq";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_estimator!(
    Imputer,
    ImputerModel,
    "ImputerModel",
    [
        "inputCol",
        "inputCols",
        "missingValue",
        "outputCol",
        "outputCols",
        "relativeError",
        "strategy"
    ]
);
feature_params! { Imputer {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_missing_value(value: f64) => "missingValue", set_param_double; missing_value -> f64 = |this| this.params.double_or("missingValue", f64::NAN);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) => "outputCols", set_param_string_array; output_cols -> Vec<String> = |this| this.params.string_array("outputCols");
    set_relative_error(value: f64) => "relativeError", set_param_double; relative_error -> f64 = |this| this.params.double_or("relativeError", 0.001);
    set_strategy(value: &str) => "strategy", set_param_string; strategy -> &str = |this| this.params.string_or("strategy", "mean");
} }
feature_params! { ImputerModel {
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
        "inputCols", set_param_string_array;
    input_cols -> Option<Vec<String>> = |this| this.params.string_array_opt("inputCols");
    missing_value -> f64 =
        |this| this.params.double_or("missingValue", f64::NAN) => "missingValue";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
        "outputCols", set_param_string_array;
    output_cols -> Option<Vec<String>> = |this| this.params.string_array_opt("outputCols");
    relative_error -> f64 =
        |this| this.params.double_or("relativeError", 0.001) => "relativeError";
    strategy -> &str = |this| this.params.string_or("strategy", "mean") => "strategy";
} }

feature_transformer!(IndexToString);
feature_params! { IndexToString {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_labels(value: impl IntoIterator<Item = impl Into<String>>) => "labels", set_param_string_array; labels -> Vec<String> = |this| this.params.string_array("labels");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_transformer!(Interaction);
feature_params! { Interaction {
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_estimator!(
    MinHashLSH,
    MinHashLSHModel,
    "MinHashLSHModel",
    ["inputCol", "numHashTables", "outputCol"]
);
feature_params! { MinHashLSH {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_num_hash_tables(value: i32) => "numHashTables", set_param_int32; num_hash_tables -> i32 = |this| this.params.int_or("numHashTables", 1);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_seed(value: i64) => "seed", set_param_int_narrow; seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash("org.apache.spark.ml.feature.MinHashLSH"));
} }
feature_params! { MinHashLSHModel {
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    num_hash_tables -> i32 = |this| this.params.int_or("numHashTables", 1) => "numHashTables";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_estimator!(
    MinMaxScaler,
    MinMaxScalerModel,
    "MinMaxScalerModel",
    ["inputCol", "max", "min", "outputCol"]
);
feature_params! { MinMaxScaler {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_max(value: f64) => "max", set_param_double; max -> f64 = |this| this.params.double_or("max", 1.0);
    set_min(value: f64) => "min", set_param_double; min -> f64 = |this| this.params.double_or("min", 0.0);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }
feature_params! { MinMaxScalerModel {
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_max(value: f64) => "max", set_param_double;
    max -> f64 = |this| this.params.double_or("max", 1.0);
    set_min(value: f64) => "min", set_param_double;
    min -> f64 = |this| this.params.double_or("min", 0.0);
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_transformer!(NGram);
feature_params! { NGram {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_n(value: i32) => "n", set_param_int32; n -> i32 = |this| this.params.int_or("n", 2);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_transformer!(Normalizer);
feature_params! { Normalizer {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_p(value: f64) => "p", set_param_double; p -> f64 = |this| this.params.double_or("p", 2.0);
} }

feature_estimator!(
    OneHotEncoder,
    OneHotEncoderModel,
    "OneHotEncoderModel",
    [
        "dropLast",
        "handleInvalid",
        "inputCol",
        "inputCols",
        "outputCol",
        "outputCols"
    ]
);
feature_params! { OneHotEncoder {
    set_drop_last(value: bool) => "dropLast", set_param_bool; drop_last -> bool = |this| this.params.bool_or("dropLast", true);
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string; handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) => "outputCols", set_param_string_array; output_cols -> Vec<String> = |this| this.params.string_array("outputCols");
} }
feature_params! { OneHotEncoderModel {
    set_drop_last(value: bool) => "dropLast", set_param_bool;
    drop_last -> bool = |this| this.params.bool_or("dropLast", true);
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string;
    handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
        "inputCols", set_param_string_array;
    input_cols -> Option<Vec<String>> = |this| this.params.string_array_opt("inputCols");
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
        "outputCols", set_param_string_array;
    output_cols -> Option<Vec<String>> = |this| this.params.string_array_opt("outputCols");
} }

feature_estimator!(PCA, PCAModel, "PCAModel", ["inputCol", "k", "outputCol"]);
feature_params! { PCA {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_k(value: i32) => "k", set_param_int32; k -> Option<i32> = |this| this.params.long_opt("k").map(|v| v as i32);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }
feature_params! { PCAModel {
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    k -> Option<i32> = |this| this.params.int_opt("k") => "k";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_transformer!(PolynomialExpansion);
feature_params! { PolynomialExpansion {
    set_degree(value: i32) => "degree", set_param_int32; degree -> i32 = |this| this.params.int_or("degree", 2);
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_estimator!(
    QuantileDiscretizer,
    QuantileDiscretizerModel,
    "Bucketizer",
    [
        "handleInvalid",
        "inputCol",
        "inputCols",
        "outputCol",
        "outputCols",
        "splits",
        "splitsArray"
    ],
    // A saved one is a Bucketizer, which `Bucketizer::load` reads, as PySpark's fit gives one.
    no_model_load
);
feature_params! { QuantileDiscretizer {
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string; handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_num_buckets(value: i32) => "numBuckets", set_param_int32; num_buckets -> i32 = |this| this.params.int_or("numBuckets", 2);
    set_num_buckets_array(value: impl IntoIterator<Item = i64>) => "numBucketsArray", set_param_int_array; num_buckets_array -> Option<Vec<i64>> = |this| this.params.array_opt("numBucketsArray").map(|values| values.iter().filter_map(|v| match v.literal_type { Some(proto::expression::literal::LiteralType::Integer(v)) => Some(i64::from(v)), Some(proto::expression::literal::LiteralType::Long(v)) => Some(v), _ => None }).collect());
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) => "outputCols", set_param_string_array; output_cols -> Vec<String> = |this| this.params.string_array("outputCols");
    set_relative_error(value: f64) => "relativeError", set_param_double; relative_error -> f64 = |this| this.params.double_or("relativeError", 0.001);
} }

feature_estimator!(
    RobustScaler,
    RobustScalerModel,
    "RobustScalerModel",
    [
        "inputCol",
        "lower",
        "outputCol",
        "relativeError",
        "upper",
        "withCentering",
        "withScaling"
    ]
);
feature_params! { RobustScaler {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_lower(value: f64) => "lower", set_param_double; lower -> f64 = |this| this.params.double_or("lower", 0.25);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_relative_error(value: f64) => "relativeError", set_param_double; relative_error -> f64 = |this| this.params.double_or("relativeError", 0.001);
    set_upper(value: f64) => "upper", set_param_double; upper -> f64 = |this| this.params.double_or("upper", 0.75);
    set_with_centering(value: bool) => "withCentering", set_param_bool; with_centering -> bool = |this| this.params.bool_or("withCentering", false);
    set_with_scaling(value: bool) => "withScaling", set_param_bool; with_scaling -> bool = |this| this.params.bool_or("withScaling", true);
} }
feature_params! { RobustScalerModel {
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    lower -> f64 = |this| this.params.double_or("lower", 0.25) => "lower";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    relative_error -> f64 =
        |this| this.params.double_or("relativeError", 0.001) => "relativeError";
    upper -> f64 = |this| this.params.double_or("upper", 0.75) => "upper";
    with_centering -> bool =
        |this| this.params.bool_or("withCentering", false) => "withCentering";
    with_scaling -> bool = |this| this.params.bool_or("withScaling", true) => "withScaling";
} }

feature_transformer!(RegexTokenizer);
feature_params! { RegexTokenizer {
    set_gaps(value: bool) => "gaps", set_param_bool; gaps -> bool = |this| this.params.bool_or("gaps", true);
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_min_token_length(value: i32) => "minTokenLength", set_param_int32; min_token_length -> i32 = |this| this.params.int_or("minTokenLength", 1);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_pattern(value: &str) => "pattern", set_param_string; pattern -> &str = |this| this.params.string_or("pattern", "\\s+");
    set_to_lowercase(value: bool) => "toLowercase", set_param_bool; to_lowercase -> bool = |this| this.params.bool_or("toLowercase", true);
} }

feature_estimator!(
    RFormula,
    RFormulaModel,
    "RFormulaModel",
    [
        "featuresCol",
        "forceIndexLabel",
        "formula",
        "handleInvalid",
        "labelCol",
        "stringIndexerOrderType"
    ]
);
feature_params! { RFormula {
    set_features_col(value: &str) => "featuresCol", set_param_string; features_col -> &str = |this| this.params.string_or("featuresCol", "features");
    set_force_index_label(value: bool) => "forceIndexLabel", set_param_bool; force_index_label -> bool = |this| this.params.bool_or("forceIndexLabel", false);
    set_formula(value: &str) => "formula", set_param_string; formula -> Option<&str> = |this| this.params.string_opt("formula");
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string; handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_label_col(value: &str) => "labelCol", set_param_string; label_col -> &str = |this| this.params.string_or("labelCol", "label");
    set_string_indexer_order_type(value: &str) => "stringIndexerOrderType", set_param_string; string_indexer_order_type -> &str = |this| this.params.string_or("stringIndexerOrderType", "frequencyDesc");
} }
feature_params! { RFormulaModel {
    features_col -> &str =
        |this| this.params.string_or("featuresCol", "features") => "featuresCol";
    force_index_label -> bool =
        |this| this.params.bool_or("forceIndexLabel", false) => "forceIndexLabel";
    formula -> Option<&str> = |this| this.params.string_opt("formula") => "formula";
    handle_invalid -> &str =
        |this| this.params.string_or("handleInvalid", "error") => "handleInvalid";
    label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
    string_indexer_order_type -> &str =
        |this| this.params.string_or(
            "stringIndexerOrderType", "frequencyDesc") => "stringIndexerOrderType";
} }

feature_transformer!(SQLTransformer);
feature_params! { SQLTransformer {
    set_statement(value: &str) => "statement", set_param_string; statement -> Option<&str> = |this| this.params.string_opt("statement");
} }
feature_transformer!(StopWordsRemover);
feature_params! { StopWordsRemover {
    set_case_sensitive(value: bool) => "caseSensitive", set_param_bool; case_sensitive -> bool = |this| this.params.bool_or("caseSensitive", false);
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_locale(value: &str) => "locale", set_param_string; locale -> Option<&str> = |this| this.params.string_opt("locale");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) => "outputCols", set_param_string_array; output_cols -> Vec<String> = |this| this.params.string_array("outputCols");
    set_stop_words(value: impl IntoIterator<Item = impl Into<String>>) => "stopWords", set_param_string_array; stop_words -> Vec<String> = |this| this.params.string_array("stopWords");
} }
impl StopWordsRemover {
    /// The default stop words of `language` that the server keeps, as PySpark's
    /// `StopWordsRemover.loadDefaultStopWords` reads them of Spark Connect's ML helper. The server
    /// rejects a language that it has none of.
    pub fn load_default_stop_words(
        session: &SparkSession,
        language: &str,
    ) -> spark_connect_core::error::Result<Vec<String>> {
        let language = proto::expression::Literal {
            literal_type: Some(proto::expression::literal::LiteralType::String(
                language.into(),
            )),
            ..Default::default()
        };
        ml_string_array(ml_helper_attribute(
            session,
            "stopWordsRemoverLoadDefaultStopWords",
            vec![language],
        )?)
    }
}
feature_estimator!(
    TargetEncoder,
    TargetEncoderModel,
    "TargetEncoderModel",
    [
        "handleInvalid",
        "inputCol",
        "inputCols",
        "labelCol",
        "outputCol",
        "outputCols",
        "smoothing",
        "targetType"
    ]
);
feature_params! { TargetEncoder {
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string; handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) => "inputCols", set_param_string_array; input_cols -> Vec<String> = |this| this.params.string_array("inputCols");
    set_label_col(value: &str) => "labelCol", set_param_string; label_col -> &str = |this| this.params.string_or("labelCol", "label");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) => "outputCols", set_param_string_array; output_cols -> Vec<String> = |this| this.params.string_array("outputCols");
    set_smoothing(value: f64) => "smoothing", set_param_double; smoothing -> f64 = |this| this.params.double_or("smoothing", 0.0);
    set_target_type(value: &str) => "targetType", set_param_string; target_type -> &str = |this| this.params.string_or("targetType", "binary");
} }
feature_params! { TargetEncoderModel {
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string;
    handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_input_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
        "inputCols", set_param_string_array;
    input_cols -> Option<Vec<String>> = |this| this.params.string_array_opt("inputCols");
    label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_output_cols(value: impl IntoIterator<Item = impl Into<String>>) =>
        "outputCols", set_param_string_array;
    output_cols -> Option<Vec<String>> = |this| this.params.string_array_opt("outputCols");
    set_smoothing(value: f64) => "smoothing", set_param_double;
    smoothing -> f64 = |this| this.params.double_or("smoothing", 0.0);
    target_type -> &str = |this| this.params.string_or("targetType", "binary") => "targetType";
} }

feature_transformer!(Tokenizer);
feature_params! { Tokenizer {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_estimator!(
    UnivariateFeatureSelector,
    UnivariateFeatureSelectorModel,
    "UnivariateFeatureSelectorModel",
    [
        "featureType",
        "featuresCol",
        "labelCol",
        "labelType",
        "outputCol",
        "selectionMode",
        "selectionThreshold"
    ]
);
feature_params! { UnivariateFeatureSelector {
    set_feature_type(value: &str) => "featureType", set_param_string; feature_type -> Option<&str> = |this| this.params.string_opt("featureType");
    set_features_col(value: &str) => "featuresCol", set_param_string; features_col -> &str = |this| this.params.string_or("featuresCol", "features");
    set_label_col(value: &str) => "labelCol", set_param_string; label_col -> &str = |this| this.params.string_or("labelCol", "label");
    set_label_type(value: &str) => "labelType", set_param_string; label_type -> Option<&str> = |this| this.params.string_opt("labelType");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_selection_mode(value: &str) => "selectionMode", set_param_string; selection_mode -> &str = |this| this.params.string_or("selectionMode", "numTopFeatures");
    set_selection_threshold(value: f64) => "selectionThreshold", set_param_double; selection_threshold -> Option<f64> = |this| this.params.get_param("selectionThreshold").map(|_| this.params.double_or("selectionThreshold", 0.0));
} }
feature_params! { UnivariateFeatureSelectorModel {
    feature_type -> Option<&str> =
        |this| this.params.string_opt("featureType") => "featureType";
    set_features_col(value: &str) => "featuresCol", set_param_string;
    features_col -> &str = |this| this.params.string_or("featuresCol", "features");
    label_col -> &str = |this| this.params.string_or("labelCol", "label") => "labelCol";
    label_type -> Option<&str> = |this| this.params.string_opt("labelType") => "labelType";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    selection_mode -> &str =
        |this| this.params.string_or("selectionMode", "numTopFeatures") => "selectionMode";
    selection_threshold -> Option<f64> =
        |this| this.params.double_opt("selectionThreshold") => "selectionThreshold";
} }

feature_estimator!(
    VarianceThresholdSelector,
    VarianceThresholdSelectorModel,
    "VarianceThresholdSelectorModel",
    ["featuresCol", "outputCol", "varianceThreshold"]
);
feature_params! { VarianceThresholdSelector {
    set_features_col(value: &str) => "featuresCol", set_param_string; features_col -> &str = |this| this.params.string_or("featuresCol", "features");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_variance_threshold(value: f64) => "varianceThreshold", set_param_double; variance_threshold -> f64 = |this| this.params.double_or("varianceThreshold", 0.0);
} }
feature_params! { VarianceThresholdSelectorModel {
    set_features_col(value: &str) => "featuresCol", set_param_string;
    features_col -> &str = |this| this.params.string_or("featuresCol", "features");
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    variance_threshold -> Option<f64> =
        |this| this.params.double_opt("varianceThreshold") => "varianceThreshold";
} }
feature_estimator!(
    VectorIndexer,
    VectorIndexerModel,
    "VectorIndexerModel",
    ["handleInvalid", "inputCol", "maxCategories", "outputCol"]
);
feature_params! { VectorIndexer {
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string; handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_max_categories(value: i32) => "maxCategories", set_param_int32; max_categories -> i32 = |this| this.params.int_or("maxCategories", 20);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }
feature_params! { VectorIndexerModel {
    handle_invalid -> &str =
        |this| this.params.string_or("handleInvalid", "error") => "handleInvalid";
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    max_categories -> i32 = |this| this.params.int_or("maxCategories", 20) => "maxCategories";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_transformer!(VectorSizeHint);
feature_params! { VectorSizeHint {
    set_handle_invalid(value: &str) => "handleInvalid", set_param_string; handle_invalid -> &str = |this| this.params.string_or("handleInvalid", "error");
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_size(value: i32) => "size", set_param_int32; size -> Option<i32> = |this| this.params.long_opt("size").map(|v| v as i32);
} }

feature_transformer!(VectorSlicer);
feature_params! { VectorSlicer {
    set_indices(value: impl IntoIterator<Item = i64>) => "indices", set_param_int_array; indices -> Option<Vec<i64>> = |this| this.params.array_opt("indices").map(|values| values.iter().filter_map(|v| match v.literal_type { Some(proto::expression::literal::LiteralType::Integer(v)) => Some(i64::from(v)), Some(proto::expression::literal::LiteralType::Long(v)) => Some(v), _ => None }).collect());
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_names(value: impl IntoIterator<Item = impl Into<String>>) => "names", set_param_string_array; names -> Vec<String> = |this| this.params.string_array("names");
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
} }

feature_estimator!(
    Word2Vec,
    Word2VecModel,
    "Word2VecModel",
    [
        "inputCol",
        "maxIter",
        "maxSentenceLength",
        "minCount",
        "numPartitions",
        "outputCol",
        "seed",
        "stepSize",
        "vectorSize",
        "windowSize"
    ]
);
feature_params! { Word2Vec {
    set_input_col(value: &str) => "inputCol", set_param_string; input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    set_max_iter(value: i32) => "maxIter", set_param_int32; max_iter -> i32 = |this| this.params.int_or("maxIter", 1);
    set_max_sentence_length(value: i32) => "maxSentenceLength", set_param_int32; max_sentence_length -> i32 = |this| this.params.int_or("maxSentenceLength", 1000);
    set_min_count(value: i32) => "minCount", set_param_int32; min_count -> i32 = |this| this.params.int_or("minCount", 5);
    set_num_partitions(value: i32) => "numPartitions", set_param_int32; num_partitions -> i32 = |this| this.params.int_or("numPartitions", 1);
    set_output_col(value: &str) => "outputCol", set_param_string; output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    set_seed(value: i64) => "seed", set_param_int_narrow; seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash("org.apache.spark.ml.feature.Word2Vec"));
    set_step_size(value: f64) => "stepSize", set_param_double; step_size -> f64 = |this| this.params.double_or("stepSize", 0.025);
    set_vector_size(value: i32) => "vectorSize", set_param_int32; vector_size -> i32 = |this| this.params.int_or("vectorSize", 100);
    set_window_size(value: i32) => "windowSize", set_param_int32; window_size -> i32 = |this| this.params.int_or("windowSize", 5);
} }
feature_params! { Word2VecModel {
    set_input_col(value: &str) => "inputCol", set_param_string;
    input_col -> Option<&str> = |this| this.params.string_opt("inputCol");
    max_iter -> i32 = |this| this.params.int_or("maxIter", 1) => "maxIter";
    max_sentence_length -> i32 =
        |this| this.params.int_or("maxSentenceLength", 1000) => "maxSentenceLength";
    min_count -> i32 = |this| this.params.int_or("minCount", 5) => "minCount";
    num_partitions -> i32 = |this| this.params.int_or("numPartitions", 1) => "numPartitions";
    set_output_col(value: &str) => "outputCol", set_param_string;
    output_col -> String = |this| this.params.output_col_or_default(&this.operator.uid);
    // A fitted model has the defaults of its estimator (`copyValues`): the seed of the JVM class
    // name of Word2Vec.
    seed -> i64 =
        |this| this.params.long_opt(
            "seed").unwrap_or(java_string_hash("org.apache.spark.ml.feature.Word2Vec")) => "seed";
    step_size -> f64 = |this| this.params.double_or("stepSize", 0.025) => "stepSize";
    vector_size -> i32 = |this| this.params.int_or("vectorSize", 100) => "vectorSize";
    window_size -> i32 = |this| this.params.int_or("windowSize", 5) => "windowSize";
} }

// The attributes and methods of the fitted feature models, as pyspark.ml 4.3.0 reads them of the
// server-side model over Spark Connect: `_call_java(name)` is a fetch of the method of the model's
// object, and `try_remote_attribute_relation` a relation of that fetch. The class methods that
// build a model call the ML helper, as `invoke_helper_attr` does.

type MlResult<T> = spark_connect_core::error::Result<T>;

/// The value of the method `name`, of no arguments, of the server-side model `object`, as
/// PySpark's `_call_java(name)` fetches it.
fn model_value(object: &MlRef, name: &str) -> MlResult<MlValue> {
    fetch(&MlTarget::object(object), name, Vec::new())?.into_value()
}

/// The DataFrame of the method `name` of the server-side model `object`, of `args`, as PySpark's
/// `try_remote_attribute_relation` reads it.
fn model_relation(object: &MlRef, name: &str, args: Vec<FetchArg>) -> MlResult<DataFrame> {
    attribute_relation(&MlTarget::object(object), name, args)
}

fn unexpected<T>(name: &str, expected: &str, value: &MlValue) -> MlResult<T> {
    Err(SparkError::connect_msg(format!(
        "{name}: the server returned {value:?}, not {expected}"
    )))
}

fn vector_of(name: &str, value: MlValue) -> MlResult<MlVector> {
    match value {
        MlValue::Vector(vector) => Ok(vector),
        other => unexpected(name, "a vector", &other),
    }
}

fn matrix_of(name: &str, value: MlValue) -> MlResult<MlMatrix> {
    match value {
        MlValue::Matrix(matrix) => Ok(matrix),
        other => unexpected(name, "a matrix", &other),
    }
}

fn array_of(name: &str, value: MlValue) -> MlResult<Vec<MlValue>> {
    match value {
        MlValue::Array(values) => Ok(values),
        other => unexpected(name, "an array", &other),
    }
}

fn long_of(name: &str, value: &MlValue) -> MlResult<i64> {
    value
        .as_i64()
        .map_or_else(|| unexpected(name, "an integer", value), Ok)
}

fn int_of(name: &str, value: &MlValue) -> MlResult<i32> {
    i32::try_from(long_of(name, value)?)
        .map_or_else(|_| unexpected(name, "a 32-bit integer", value), Ok)
}

fn strings_of(name: &str, value: MlValue) -> MlResult<Vec<String>> {
    array_of(name, value)?
        .into_iter()
        .map(|value| match value {
            MlValue::String(string) => Ok(string),
            other => unexpected(name, "an array of strings", &other),
        })
        .collect()
}

fn ints_of(name: &str, value: MlValue) -> MlResult<Vec<i32>> {
    array_of(name, value)?
        .iter()
        .map(|value| int_of(name, value))
        .collect()
}

fn longs_of(name: &str, value: MlValue) -> MlResult<Vec<i64>> {
    array_of(name, value)?
        .iter()
        .map(|value| long_of(name, value))
        .collect()
}

fn doubles_of(name: &str, value: MlValue) -> MlResult<Vec<f64>> {
    array_of(name, value)?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .map_or_else(|| unexpected(name, "an array of doubles", value), Ok)
        })
        .collect()
}

/// The value of an array argument of strings of a helper call.
fn string_array(values: &[String]) -> MlValue {
    MlValue::Array(values.iter().cloned().map(MlValue::String).collect())
}

/// `StringType()`, of PySpark's default collation.
fn string_type() -> DataType {
    DataType::String {
        collation: "UTF8_BINARY".to_string(),
    }
}

/// The `ArrayType` of `element_type`, the type that PySpark gives the labels of a helper call, so
/// that an empty array has a type.
fn array_type(element_type: DataType) -> DataType {
    DataType::Array {
        element_type: Box::new(element_type),
        contains_null: true,
    }
}

impl CountVectorizerModel {
    /// The vocabulary, as `CountVectorizerModel.vocabulary`.
    pub fn vocabulary(&self) -> MlResult<Vec<String>> {
        strings_of("vocabulary", model_value(&self.cached, "vocabulary")?)
    }

    /// A model of the vocabulary `vocabulary`, which transforms `input_col`, as
    /// `CountVectorizerModel.from_vocabulary`: the server builds it (the ML helper's
    /// `countVectorizerModelFromVocabulary`, of the uid of the new model), and the model sets
    /// `outputCol`, `minTF` and `binary` when they are given, and `vocabSize`, the size of the
    /// vocabulary. An empty vocabulary raises PySpark's builtin `ValueError`.
    pub fn from_vocabulary(
        session: &SparkSession,
        vocabulary: impl IntoIterator<Item = impl Into<String>>,
        input_col: &str,
        output_col: Option<&str>,
        min_tf: Option<f64>,
        binary: Option<bool>,
    ) -> MlResult<Self> {
        let vocabulary: Vec<String> = vocabulary.into_iter().map(Into::into).collect();
        if vocabulary.is_empty() {
            return Err(SparkError::builtin_value_msg(
                "Vocabulary list cannot be empty",
            ));
        }
        let operator = MlOperator::new(
            "org.apache.spark.ml.feature.CountVectorizerModel",
            OperatorType::Model,
        );
        let cached = fetch(
            &MlTarget::helper(session),
            "countVectorizerModelFromVocabulary",
            vec![
                FetchArg::Value(MlValue::String(operator.uid.clone())),
                FetchArg::Value(string_array(&vocabulary)),
            ],
        )?
        .into_object()?;
        let mut params = Params::new().set_param_string("inputCol", input_col);
        if let Some(output_col) = output_col {
            params = params.set_param_string("outputCol", output_col);
        }
        if let Some(min_tf) = min_tf {
            params = params.set_param_double("minTF", min_tf);
        }
        if let Some(binary) = binary {
            params = params.set_param_bool("binary", binary);
        }
        params = params.set_param_int_narrow("vocabSize", vocabulary.len() as i64);
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

impl StringIndexerModel {
    /// The labels of a model of one column, as `StringIndexerModel.labels`. The server rejects it
    /// of a model of several columns, whose labels are [`Self::labels_array`].
    pub fn labels(&self) -> MlResult<Vec<String>> {
        strings_of("labels", model_value(&self.cached, "labels")?)
    }

    /// The labels of each column, as `StringIndexerModel.labelsArray`.
    pub fn labels_array(&self) -> MlResult<Vec<Vec<String>>> {
        array_of("labelsArray", model_value(&self.cached, "labelsArray")?)?
            .into_iter()
            .map(|labels| strings_of("labelsArray", labels))
            .collect()
    }

    /// A model of the labels `labels`, which indexes `input_col`, as
    /// `StringIndexerModel.from_labels`: the server builds it (the ML helper's
    /// `stringIndexerModelFromLabels`, of the uid of the new model and the labels typed as an array
    /// of strings), and the model sets `outputCol` and `handleInvalid` when they are given.
    pub fn from_labels(
        session: &SparkSession,
        labels: impl IntoIterator<Item = impl Into<String>>,
        input_col: &str,
        output_col: Option<&str>,
        handle_invalid: Option<&str>,
    ) -> MlResult<Self> {
        let labels: Vec<String> = labels.into_iter().map(Into::into).collect();
        let operator = MlOperator::new(
            "org.apache.spark.ml.feature.StringIndexerModel",
            OperatorType::Model,
        );
        let cached = fetch(
            &MlTarget::helper(session),
            "stringIndexerModelFromLabels",
            vec![
                FetchArg::Value(MlValue::String(operator.uid.clone())),
                FetchArg::Typed(string_array(&labels), array_type(string_type())),
            ],
        )?
        .into_object()?;
        let mut params = Params::new().set_param_string("inputCol", input_col);
        if let Some(output_col) = output_col {
            params = params.set_param_string("outputCol", output_col);
        }
        if let Some(handle_invalid) = handle_invalid {
            params = params.set_param_string("handleInvalid", handle_invalid);
        }
        Ok(Self {
            operator,
            params,
            cached,
        })
    }

    /// A model of the labels of each column of `array_of_labels`, which indexes `input_cols`, as
    /// `StringIndexerModel.from_arrays_of_labels`: the server builds it (the ML helper's
    /// `stringIndexerModelFromLabelsArray`, of the arrays typed as an array of arrays of strings),
    /// and the model sets `outputCols` and `handleInvalid` when they are given.
    pub fn from_arrays_of_labels(
        session: &SparkSession,
        array_of_labels: impl IntoIterator<Item = impl IntoIterator<Item = impl Into<String>>>,
        input_cols: impl IntoIterator<Item = impl Into<String>>,
        output_cols: Option<Vec<String>>,
        handle_invalid: Option<&str>,
    ) -> MlResult<Self> {
        let arrays = MlValue::Array(
            array_of_labels
                .into_iter()
                .map(|labels| {
                    let labels: Vec<String> = labels.into_iter().map(Into::into).collect();
                    string_array(&labels)
                })
                .collect(),
        );
        let operator = MlOperator::new(
            "org.apache.spark.ml.feature.StringIndexerModel",
            OperatorType::Model,
        );
        let cached = fetch(
            &MlTarget::helper(session),
            "stringIndexerModelFromLabelsArray",
            vec![
                FetchArg::Value(MlValue::String(operator.uid.clone())),
                FetchArg::Typed(arrays, array_type(array_type(string_type()))),
            ],
        )?
        .into_object()?;
        let mut params = Params::new().set_param_string_array("inputCols", input_cols);
        if let Some(output_cols) = output_cols {
            params = params.set_param_string_array("outputCols", output_cols);
        }
        if let Some(handle_invalid) = handle_invalid {
            params = params.set_param_string("handleInvalid", handle_invalid);
        }
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

impl IDFModel {
    /// The inverse document frequencies, as `IDFModel.idf`.
    pub fn idf(&self) -> MlResult<MlVector> {
        vector_of("idf", model_value(&self.cached, "idf")?)
    }

    /// The document frequency of each term, as `IDFModel.docFreq`.
    pub fn doc_freq(&self) -> MlResult<Vec<i64>> {
        longs_of("docFreq", model_value(&self.cached, "docFreq")?)
    }

    /// The number of documents of the fit, as `IDFModel.numDocs`.
    pub fn num_docs(&self) -> MlResult<i64> {
        long_of("numDocs", &model_value(&self.cached, "numDocs")?)
    }
}

impl ImputerModel {
    /// The surrogate of each input column, as `ImputerModel.surrogateDF`.
    pub fn surrogate_df(&self) -> MlResult<DataFrame> {
        model_relation(&self.cached, "surrogateDF", Vec::new())
    }
}

impl MaxAbsScalerModel {
    /// The maximum absolute value of each feature, as `MaxAbsScalerModel.maxAbs`.
    pub fn max_abs(&self) -> MlResult<MlVector> {
        vector_of("maxAbs", model_value(&self.cached, "maxAbs")?)
    }
}

impl MinMaxScalerModel {
    /// The minimum of each feature, as `MinMaxScalerModel.originalMin`.
    pub fn original_min(&self) -> MlResult<MlVector> {
        vector_of("originalMin", model_value(&self.cached, "originalMin")?)
    }

    /// The maximum of each feature, as `MinMaxScalerModel.originalMax`.
    pub fn original_max(&self) -> MlResult<MlVector> {
        vector_of("originalMax", model_value(&self.cached, "originalMax")?)
    }
}

impl OneHotEncoderModel {
    /// The number of categories of each input column, as `OneHotEncoderModel.categorySizes`.
    pub fn category_sizes(&self) -> MlResult<Vec<i32>> {
        ints_of("categorySizes", model_value(&self.cached, "categorySizes")?)
    }
}

impl PCAModel {
    /// The principal components, one per column, as `PCAModel.pc`.
    pub fn pc(&self) -> MlResult<MlMatrix> {
        matrix_of("pc", model_value(&self.cached, "pc")?)
    }

    /// The proportion of variance that each principal component explains, as
    /// `PCAModel.explainedVariance`.
    pub fn explained_variance(&self) -> MlResult<MlVector> {
        vector_of(
            "explainedVariance",
            model_value(&self.cached, "explainedVariance")?,
        )
    }
}

impl RobustScalerModel {
    /// The median of each feature, as `RobustScalerModel.median`.
    pub fn median(&self) -> MlResult<MlVector> {
        vector_of("median", model_value(&self.cached, "median")?)
    }

    /// The quantile range of each feature, as `RobustScalerModel.range`.
    pub fn range(&self) -> MlResult<MlVector> {
        vector_of("range", model_value(&self.cached, "range")?)
    }
}

impl StandardScalerModel {
    /// The mean of each feature, as `StandardScalerModel.mean`.
    pub fn mean(&self) -> MlResult<MlVector> {
        vector_of("mean", model_value(&self.cached, "mean")?)
    }

    /// The standard deviation of each feature, as `StandardScalerModel.std`.
    pub fn std(&self) -> MlResult<MlVector> {
        vector_of("std", model_value(&self.cached, "std")?)
    }
}

impl ChiSqSelectorModel {
    /// The indices of the selected features, as `ChiSqSelectorModel.selectedFeatures`.
    pub fn selected_features(&self) -> MlResult<Vec<i32>> {
        ints_of(
            "selectedFeatures",
            model_value(&self.cached, "selectedFeatures")?,
        )
    }
}

impl UnivariateFeatureSelectorModel {
    /// The indices of the selected features, as
    /// `UnivariateFeatureSelectorModel.selectedFeatures`.
    pub fn selected_features(&self) -> MlResult<Vec<i32>> {
        ints_of(
            "selectedFeatures",
            model_value(&self.cached, "selectedFeatures")?,
        )
    }
}

impl VarianceThresholdSelectorModel {
    /// The indices of the selected features, as
    /// `VarianceThresholdSelectorModel.selectedFeatures`.
    pub fn selected_features(&self) -> MlResult<Vec<i32>> {
        ints_of(
            "selectedFeatures",
            model_value(&self.cached, "selectedFeatures")?,
        )
    }
}

impl VectorIndexerModel {
    /// The length of the vectors that the model transforms, as `VectorIndexerModel.numFeatures`.
    pub fn num_features(&self) -> MlResult<i32> {
        int_of("numFeatures", &model_value(&self.cached, "numFeatures")?)
    }

    /// The category index of each value of each categorical feature, as
    /// `VectorIndexerModel.categoryMaps`, which collects the rows of the model's `categoryMapsDF`
    /// (featureIndex, originalValue, categoryIndex): of each feature index, its original values
    /// and their category indices, ordered by the value. A feature that it has none of is
    /// continuous.
    pub fn category_maps(&self) -> MlResult<std::collections::BTreeMap<i32, Vec<(f64, i32)>>> {
        let rows = model_relation(&self.cached, "categoryMapsDF", Vec::new())?.collect()?;
        let mut maps: std::collections::BTreeMap<i32, Vec<(f64, i32)>> = Default::default();
        for row in rows {
            let field = |index: usize| row.get(index);
            let (Some(feature), Some(value), Some(category)) = (
                field(0).and_then(|v| v.as_i64()),
                field(1).and_then(|v| v.as_f64()),
                field(2).and_then(|v| v.as_i64()),
            ) else {
                return Err(SparkError::connect_msg(
                    "categoryMapsDF: the server returned a row that is not (int, double, int)",
                ));
            };
            maps.entry(feature as i32)
                .or_default()
                .push((value, category as i32));
        }
        for values in maps.values_mut() {
            values.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        Ok(maps)
    }
}

impl Word2VecModel {
    /// The vector of each word, as `Word2VecModel.getVectors`: a DataFrame of `word` and `vector`.
    pub fn get_vectors(&self) -> MlResult<DataFrame> {
        model_relation(&self.cached, "getVectors", Vec::new())
    }

    /// The `num` words closest to `word`, as `Word2VecModel.findSynonyms` of a word: a DataFrame
    /// of `word` and `similarity`. The server rejects a word out of the vocabulary.
    pub fn find_synonyms(&self, word: &str, num: i32) -> MlResult<DataFrame> {
        model_relation(
            &self.cached,
            "findSynonyms",
            vec![
                FetchArg::Value(MlValue::String(word.to_string())),
                FetchArg::Value(MlValue::Int(num)),
            ],
        )
    }

    /// The `num` words closest to `vector`, as `Word2VecModel.findSynonyms` of a vector.
    pub fn find_synonyms_of_vector(&self, vector: &MlVector, num: i32) -> MlResult<DataFrame> {
        model_relation(
            &self.cached,
            "findSynonyms",
            vec![
                FetchArg::Value(MlValue::Vector(vector.clone())),
                FetchArg::Value(MlValue::Int(num)),
            ],
        )
    }

    /// The words and similarities of [`Self::find_synonyms`], collected, as
    /// `Word2VecModel.findSynonymsArray` of a word.
    pub fn find_synonyms_array(&self, word: &str, num: i32) -> MlResult<Vec<(String, f64)>> {
        synonyms_of(self.find_synonyms(word, num)?)
    }

    /// The words and similarities of [`Self::find_synonyms_of_vector`], collected, as
    /// `Word2VecModel.findSynonymsArray` of a vector.
    pub fn find_synonyms_array_of_vector(
        &self,
        vector: &MlVector,
        num: i32,
    ) -> MlResult<Vec<(String, f64)>> {
        synonyms_of(self.find_synonyms_of_vector(vector, num)?)
    }
}

/// The rows of a DataFrame of `word` and `similarity`, as `findSynonymsArray` collects them.
fn synonyms_of(synonyms: DataFrame) -> MlResult<Vec<(String, f64)>> {
    synonyms
        .collect()?
        .into_iter()
        .map(|row| {
            match (
                row.get(0).and_then(|v| v.as_str()),
                row.get(1).and_then(|v| v.as_f64()),
            ) {
                (Some(word), Some(similarity)) => Ok((word.to_string(), similarity)),
                _ => Err(SparkError::connect_msg(
                    "findSynonyms: the server returned a row that is not (string, double)",
                )),
            }
        })
        .collect()
}

/// The LSH methods of a model of `$model`, as `_LSHModel` of pyspark.ml has them.
macro_rules! lsh_model_methods {
    ($model:ident) => {
        impl $model {
            /// The `num_nearest_neighbors` rows of `dataset` nearest to `key`, with their distance
            /// in `dist_col`, as `approxNearestNeighbors` (PySpark's `distCol` is "distCol").
            pub fn approx_nearest_neighbors(
                &self,
                dataset: &DataFrame,
                key: &MlVector,
                num_nearest_neighbors: i32,
                dist_col: &str,
            ) -> MlResult<DataFrame> {
                model_relation(
                    &self.cached,
                    "approxNearestNeighbors",
                    vec![
                        FetchArg::Input(dataset.clone()),
                        FetchArg::Value(MlValue::Vector(key.clone())),
                        FetchArg::Value(MlValue::Int(num_nearest_neighbors)),
                        FetchArg::Value(MlValue::String(dist_col.to_string())),
                    ],
                )
            }

            /// The pairs of rows of `dataset_a` and `dataset_b` within `threshold`, as
            /// `datasetA` and `datasetB` with their distance in `dist_col`, as
            /// `approxSimilarityJoin` (PySpark's `distCol` is "distCol").
            pub fn approx_similarity_join(
                &self,
                dataset_a: &DataFrame,
                dataset_b: &DataFrame,
                threshold: f64,
                dist_col: &str,
            ) -> MlResult<DataFrame> {
                model_relation(
                    &self.cached,
                    "approxSimilarityJoin",
                    vec![
                        FetchArg::Input(dataset_a.clone()),
                        FetchArg::Input(dataset_b.clone()),
                        FetchArg::Value(MlValue::Double(threshold)),
                        FetchArg::Value(MlValue::String(dist_col.to_string())),
                    ],
                )
            }
        }
    };
}

lsh_model_methods!(BucketedRandomProjectionLSHModel);
lsh_model_methods!(MinHashLSHModel);

impl QuantileDiscretizerModel {
    /// The splits of a discretizer of one column, as the `getSplits` of its server-side
    /// `Bucketizer`.
    pub fn splits(&self) -> MlResult<Vec<f64>> {
        doubles_of("getSplits", model_value(&self.cached, "getSplits")?)
    }

    /// The splits of each column of a discretizer of several columns, as the `getSplitsArray` of
    /// its server-side `Bucketizer`.
    pub fn splits_array(&self) -> MlResult<Vec<Vec<f64>>> {
        array_of(
            "getSplitsArray",
            model_value(&self.cached, "getSplitsArray")?,
        )?
        .into_iter()
        .map(|splits| doubles_of("getSplitsArray", splits))
        .collect()
    }
}

impl QuantileDiscretizer {
    /// Fit the discretizer and give the `Bucketizer` of its splits, as `QuantileDiscretizer.fit`
    /// of pyspark.ml gives it: of `getSplits`, with the `inputCol`, `outputCol` and
    /// `handleInvalid` of this discretizer, when its `inputCol` is set, and otherwise of
    /// `getSplitsArray`, with its `inputCols` and `outputCols`. The bucketizer is a new stage,
    /// of its own uid, which holds no server-side object.
    pub fn fit_bucketizer(&mut self, df: &DataFrame) -> MlResult<Bucketizer> {
        let model = Estimator::fit(self, df)?;
        let object = model
            .ml_ref()
            .ok_or_else(|| SparkError::connect_msg("QuantileDiscretizer: no fitted model"))?
            .clone();
        let model = QuantileDiscretizerModel {
            operator: MlOperator::new(
                "org.apache.spark.ml.feature.Bucketizer",
                OperatorType::Model,
            ),
            params: Params::new(),
            cached: object,
        };
        let bucketizer = Bucketizer::new().set_handle_invalid(self.handle_invalid());
        Ok(match self.input_col() {
            Some(input_col) => bucketizer
                .set_splits(model.splits()?)
                .set_input_col(input_col)
                .set_output_col(&self.output_col()),
            None => bucketizer
                .set_splits_array(model.splits_array()?)
                .set_input_cols(self.input_cols())
                .set_output_cols(self.output_cols()),
        })
    }
}

/// The typed fit and the loads of a feature estimator of `ml.rs`, which predates the
/// `feature_estimator!` stages, as theirs: the fit is the estimator's own, of the model type.
macro_rules! typed_feature_estimator {
    ($name:ident, $model:ident) => {
        impl $name {
            #[doc = concat!("Fit a typed `", stringify!($model), "`, as `Estimator::fit` fits it, which boxes it.")]
            pub fn fit(&mut self, df: &DataFrame) -> MlResult<$model> {
                let model = Estimator::fit(self, df)?;
                model.downcast_ref::<$model>().cloned().ok_or_else(|| {
                    SparkError::connect_msg(concat!("the fit gave no ", stringify!($model)))
                })
            }

            #[doc = concat!("Load a `", stringify!($name), "` saved to `path`, as PySpark's `", stringify!($name), ".load(path)`.")]
            pub fn load(session: &SparkSession, path: &str) -> MlResult<Self> {
                let jvm_class = concat!("org.apache.spark.ml.feature.", stringify!($name));
                let (operator, params) = read_operator(session, jvm_class, OperatorType::Estimator, path)?;
                Ok(Self { operator, params })
            }
        }

        impl $model {
            #[doc = concat!("Load a `", stringify!($model), "` saved to `path`, as PySpark's `", stringify!($model), ".load(path)`: the server caches it, as a fitted model.")]
            pub fn load(session: &SparkSession, path: &str) -> MlResult<Self> {
                let jvm_class = concat!("org.apache.spark.ml.feature.", stringify!($model));
                let (cached, operator, params) = read_model(session, jvm_class, path)?;
                Ok(Self { operator, params, cached })
            }
        }
    };
}

typed_feature_estimator!(MaxAbsScaler, MaxAbsScalerModel);
typed_feature_estimator!(StandardScaler, StandardScalerModel);
typed_feature_estimator!(StringIndexer, StringIndexerModel);

impl VectorAssembler {
    /// The writer of this `VectorAssembler`, as PySpark's `write()`, which saves it of the
    /// session that `MlWriter::session` gives it.
    pub fn write(&self) -> MlWriter {
        MlWriter::operator(&self.operator, &self.params)
    }

    /// Load a `VectorAssembler` saved to `path`, as PySpark's `VectorAssembler.load(path)`.
    pub fn load(session: &SparkSession, path: &str) -> MlResult<Self> {
        let jvm_class = "org.apache.spark.ml.feature.VectorAssembler";
        let (operator, params) =
            read_operator(session, jvm_class, OperatorType::Transformer, path)?;
        Ok(Self { operator, params })
    }
}

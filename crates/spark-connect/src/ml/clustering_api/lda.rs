// SPDX-License-Identifier: Apache-2.0
//! The LDA models, as PySpark's `LDAModel`, `LocalLDAModel` and `DistributedLDAModel`.
//!
//! As PySpark's `LDA._create_model`, the class of a fitted model is chosen client-side by the
//! optimizer: the em optimizer fits a distributed model, and the online one a local model.
//! [`LDA::fit`] returns the [`LDAModel`] of either, which [`LDAModel::into_local`] or
//! [`LDAModel::into_distributed`] turns into its class. [`LDAModelApi`] holds the members that
//! every LDA model has.

use super::{as_bool, as_strings};
use crate::dataframe::DataFrame;
use crate::ml::object::{attribute_relation, fetch, FetchArg, MlValue};
use crate::ml::regression_api::as_matrix;
use crate::ml::tree_api::{as_f64, as_i32, as_vector, fetch_value};
use crate::ml::{
    fit_via_command, java_string_hash, read_model, read_operator, CachedModel, Estimator, LDAModel,
    MlMatrix, MlOperator, MlRef, MlVector, Model, OperatorType, Params, Transformer, LDA,
};
use crate::session::SparkSession;
use spark_connect_core::error::Result;
use spark_connect_proto as proto;

const LDA_ESTIMATOR: &str = "org.apache.spark.ml.clustering.LDA";
const LOCAL_LDA_MODEL: &str = "org.apache.spark.ml.clustering.LocalLDAModel";
const DISTRIBUTED_LDA_MODEL: &str = "org.apache.spark.ml.clustering.DistributedLDAModel";

impl LDA {
    /// Fit the model of `df`, as PySpark's `fit` returns a `DistributedLDAModel` of the em
    /// optimizer, and a `LocalLDAModel` of the online one (see [`LDAModel::into_local`] and
    /// [`LDAModel::into_distributed`]).
    pub fn fit(&mut self, df: &DataFrame) -> Result<LDAModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator saved at `path`, as `LDA.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, LDA_ESTIMATOR, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

/// The members of an LDA model (`LDAModel`), which the local and the distributed models share:
/// each a fetch of the model's object.
pub trait LDAModelApi: Model {
    /// Whether the model is distributed, as the server tells (`isDistributed`).
    fn is_distributed(&self) -> Result<bool> {
        let value = fetch_value(&self.model_target()?, "isDistributed", vec![])?;
        as_bool("isDistributed", value)
    }

    /// The number of terms of the vocabulary (`vocabSize`).
    fn vocab_size(&self) -> Result<i32> {
        let value = fetch_value(&self.model_target()?, "vocabSize", vec![])?;
        as_i32("vocabSize", value)
    }

    /// The inferred topics, a matrix of a row per term and a column per topic (`topicsMatrix`).
    fn topics_matrix(&self) -> Result<MlMatrix> {
        let value = fetch_value(&self.model_target()?, "topicsMatrix", vec![])?;
        as_matrix("topicsMatrix", value)
    }

    /// The lower bound of the log likelihood of the documents of `dataset` (`logLikelihood`).
    fn log_likelihood(&self, dataset: &DataFrame) -> Result<f64> {
        let args = vec![FetchArg::Input(dataset.clone())];
        let value = fetch_value(&self.model_target()?, "logLikelihood", args)?;
        as_f64("logLikelihood", value)
    }

    /// The upper bound of the perplexity of the documents of `dataset` (`logPerplexity`).
    fn log_perplexity(&self, dataset: &DataFrame) -> Result<f64> {
        let args = vec![FetchArg::Input(dataset.clone())];
        let value = fetch_value(&self.model_target()?, "logPerplexity", args)?;
        as_f64("logPerplexity", value)
    }

    /// The DataFrame of the topics, each of its `max_terms_per_topic` highest weighted terms
    /// (`describeTopics`, whose default of PySpark is 10).
    fn describe_topics(&self, max_terms_per_topic: i32) -> Result<DataFrame> {
        let args = vec![FetchArg::Value(MlValue::Int(max_terms_per_topic))];
        attribute_relation(&self.model_target()?, "describeTopics", args)
    }

    /// The concentration of the documents over the topics, estimated or as set
    /// (`estimatedDocConcentration`).
    fn estimated_doc_concentration(&self) -> Result<MlVector> {
        let value = fetch_value(&self.model_target()?, "estimatedDocConcentration", vec![])?;
        as_vector("estimatedDocConcentration", value)
    }
}

impl LDAModelApi for LDAModel {}

impl LDAModel {
    /// The local model, of a model of the online optimizer, as PySpark's `fit` returns a
    /// `LocalLDAModel` of it. None of a distributed model.
    pub fn into_local(self) -> Option<LocalLDAModel> {
        (self.operator.name == LOCAL_LDA_MODEL).then(|| LocalLDAModel {
            operator: self.operator,
            params: self.params,
            cached: self.cached,
        })
    }

    /// The distributed model, of a model of the em optimizer, as PySpark's `fit` returns a
    /// `DistributedLDAModel` of it. None of a local model.
    pub fn into_distributed(self) -> Option<DistributedLDAModel> {
        (self.operator.name == DISTRIBUTED_LDA_MODEL).then(|| DistributedLDAModel {
            operator: self.operator,
            params: self.params,
            cached: self.cached,
        })
    }
}

/// A local LDA model (`LocalLDAModel`): the members of [`LDAModelApi`].
#[derive(Debug, Clone)]
pub struct LocalLDAModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    LocalLDAModel {
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        doc_concentration -> Option<Vec<f64>> = |this| this.params.double_array_opt(
            "docConcentration") => "docConcentration";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        k -> i32 = |this| this.params.int_or("k", 10) => "k";
        keep_last_checkpoint -> bool = |this| this.params.bool_or("keepLastCheckpoint", true) =>
            "keepLastCheckpoint";
        learning_decay -> f64 = |this| this.params.double_or("learningDecay", 0.51) =>
            "learningDecay";
        learning_offset -> f64 = |this| this.params.double_or("learningOffset", 1024.0) =>
            "learningOffset";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20) => "maxIter";
        optimize_doc_concentration -> bool = |this| this.params.bool_or("optimizeDocConcentration",
            true) => "optimizeDocConcentration";
        optimizer -> &str = |this| this.params.string_or("optimizer", "online") => "optimizer";
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.LocalLDAModel"));
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 0.05) =>
            "subsamplingRate";
        topic_concentration -> Option<f64> = |this| this.params.double_opt("topicConcentration") =>
            "topicConcentration";
        set_topic_distribution_col(value: &str) => "topicDistributionCol", set_param_string;
        topic_distribution_col -> &str = |this| this.params.string_or("topicDistributionCol",
            "topicDistribution");
    }
}

impl LocalLDAModel {
    /// The model saved at `path`, as `LocalLDAModel.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, LOCAL_LDA_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

/// A distributed LDA model (`DistributedLDAModel`): the members of [`LDAModelApi`], and those of
/// the training of the model.
#[derive(Debug, Clone)]
pub struct DistributedLDAModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}

ml_params! {
    DistributedLDAModel {
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10) =>
            "checkpointInterval";
        doc_concentration -> Option<Vec<f64>> = |this| this.params.double_array_opt(
            "docConcentration") => "docConcentration";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        k -> i32 = |this| this.params.int_or("k", 10) => "k";
        keep_last_checkpoint -> bool = |this| this.params.bool_or("keepLastCheckpoint", true) =>
            "keepLastCheckpoint";
        learning_decay -> f64 = |this| this.params.double_or("learningDecay", 0.51) =>
            "learningDecay";
        learning_offset -> f64 = |this| this.params.double_or("learningOffset", 1024.0) =>
            "learningOffset";
        max_iter -> i32 = |this| this.params.int_or("maxIter", 20) => "maxIter";
        optimize_doc_concentration -> bool = |this| this.params.bool_or("optimizeDocConcentration",
            true) => "optimizeDocConcentration";
        optimizer -> &str = |this| this.params.string_or("optimizer", "online") => "optimizer";
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 = |this| this.params.long_opt("seed").unwrap_or(java_string_hash(
            "org.apache.spark.ml.clustering.DistributedLDAModel"));
        subsampling_rate -> f64 = |this| this.params.double_or("subsamplingRate", 0.05) =>
            "subsamplingRate";
        topic_concentration -> Option<f64> = |this| this.params.double_opt("topicConcentration") =>
            "topicConcentration";
        set_topic_distribution_col(value: &str) => "topicDistributionCol", set_param_string;
        topic_distribution_col -> &str = |this| this.params.string_or("topicDistributionCol",
            "topicDistribution");
    }
}

impl DistributedLDAModel {
    /// The local model of the model, which discards the training dataset (`toLocal`). It is its
    /// own server-side object, of a new uid and of no params, as PySpark builds it of the id.
    pub fn to_local(&self) -> Result<LocalLDAModel> {
        let object = fetch(&self.model_target()?, "toLocal", vec![])?.into_object()?;
        Ok(LocalLDAModel {
            operator: MlOperator::new(LOCAL_LDA_MODEL, OperatorType::Model),
            params: Params::new(),
            cached: object,
        })
    }

    /// The log likelihood of the training documents and the inferred topics
    /// (`trainingLogLikelihood`).
    pub fn training_log_likelihood(&self) -> Result<f64> {
        let value = fetch_value(&self.model_target()?, "trainingLogLikelihood", vec![])?;
        as_f64("trainingLogLikelihood", value)
    }

    /// The log probability of the topics and the document distributions under the priors
    /// (`logPrior`).
    pub fn log_prior(&self) -> Result<f64> {
        as_f64(
            "logPrior",
            fetch_value(&self.model_target()?, "logPrior", vec![])?,
        )
    }

    /// The paths of the checkpoint files of the training that the model keeps
    /// (`getCheckpointFiles`).
    pub fn get_checkpoint_files(&self) -> Result<Vec<String>> {
        let value = fetch_value(&self.model_target()?, "getCheckpointFiles", vec![])?;
        as_strings("getCheckpointFiles", value)
    }

    /// The model saved at `path`, as `DistributedLDAModel.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, DISTRIBUTED_LDA_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

/// The transformer and model of a local or a distributed LDA model, as of [`LDAModel`].
macro_rules! lda_model {
    ($model:ty) => {
        impl Transformer for $model {
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

            fn build_ml_relation(&self, input: &DataFrame) -> Result<proto::MlRelation> {
                self.cached.build_ml_relation(input, &self.params)
            }

            fn transform_impl(&mut self, df: &DataFrame) -> Result<DataFrame> {
                self.cached.transform(df, &self.params)
            }
        }

        impl Model for $model {
            fn ml_ref(&self) -> Option<&MlRef> {
                Some(&self.cached)
            }

            fn clone_box(&self) -> Box<dyn Model> {
                Box::new(self.clone())
            }
        }

        impl LDAModelApi for $model {}
    };
}

lda_model!(LocalLDAModel);
lda_model!(DistributedLDAModel);

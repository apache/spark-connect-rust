// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of OneVsRestModel. Its fit and transform run in the reduction of
//! [`crate::ml::OneVsRest`] and [`crate::ml::OneVsRestModel`]; its persistence is the base's.

use crate::ml::{
    Classifier, DecisionTreeClassificationModel, FMClassificationModel, GBTClassificationModel,
    LinearSVCModel, LogisticRegressionModel, Model, MultilayerPerceptronClassificationModel,
    NaiveBayesModel, OneVsRest, OneVsRestModel, RandomForestClassificationModel,
};

/// A binary submodel of a OneVsRestModel (`models`): its typed model, as the classifier that
/// fitted it, or the boxed model of a classifier without a typed model.
pub enum SubModel {
    LogisticRegression(LogisticRegressionModel),
    LinearSVC(LinearSVCModel),
    DecisionTreeClassification(DecisionTreeClassificationModel),
    RandomForestClassification(RandomForestClassificationModel),
    GBTClassification(GBTClassificationModel),
    NaiveBayes(NaiveBayesModel),
    FMClassification(FMClassificationModel),
    MultilayerPerceptronClassification(MultilayerPerceptronClassificationModel),
    Other(Box<dyn Model>),
}

impl SubModel {
    /// The typed submodel of `model`, which the classifier of a OneVsRest fit returned.
    fn of(model: Box<dyn Model>) -> Self {
        if let Some(m) = model.downcast_ref::<LogisticRegressionModel>() {
            return Self::LogisticRegression(m.clone());
        }
        if let Some(m) = model.downcast_ref::<LinearSVCModel>() {
            return Self::LinearSVC(m.clone());
        }
        if let Some(m) = model.downcast_ref::<DecisionTreeClassificationModel>() {
            return Self::DecisionTreeClassification(m.clone());
        }
        if let Some(m) = model.downcast_ref::<RandomForestClassificationModel>() {
            return Self::RandomForestClassification(m.clone());
        }
        if let Some(m) = model.downcast_ref::<GBTClassificationModel>() {
            return Self::GBTClassification(m.clone());
        }
        if let Some(m) = model.downcast_ref::<NaiveBayesModel>() {
            return Self::NaiveBayes(m.clone());
        }
        if let Some(m) = model.downcast_ref::<FMClassificationModel>() {
            return Self::FMClassification(m.clone());
        }
        if let Some(m) = model.downcast_ref::<MultilayerPerceptronClassificationModel>() {
            return Self::MultilayerPerceptronClassification(m.clone());
        }
        Self::Other(model)
    }
}

impl OneVsRestModel {
    /// The binary submodels, one per class, in the order of the class index (`models`).
    pub fn models(&self) -> Vec<SubModel> {
        self.models
            .iter()
            .map(|model| SubModel::of(model.clone_box()))
            .collect()
    }
}

ml_params! {
    OneVsRest {
        classifier -> Option<&Classifier> = |this| this.classifier.as_ref() => "classifier";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> &str = |this| this.params.string_or("featuresCol", "features");
        set_label_col(value: &str) => "labelCol", set_param_string;
        label_col -> &str = |this| this.params.string_or("labelCol", "label");
        set_parallelism(value: i32) => "parallelism", set_param_int32;
        parallelism -> i32 = |this| this.params.int_or("parallelism", 1);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> &str =
            |this| this.params.string_or("rawPredictionCol", "rawPrediction");
        set_weight_col(value: &str) => "weightCol", set_param_string;
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol");
    }
}

ml_params! {
    OneVsRestModel {
        classifier -> Option<&Classifier> = |this| this.classifier.as_ref() => "classifier";
        set_features_col(value: &str) => "featuresCol", set_param_string;
        features_col -> Option<&str> = |this| this.params.string_opt("featuresCol");
        label_col -> Option<&str> = |this| this.params.string_opt("labelCol") => "labelCol";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> Option<&str> = |this| this.params.string_opt("predictionCol");
        set_raw_prediction_col(value: &str) => "rawPredictionCol", set_param_string;
        raw_prediction_col -> Option<&str> = |this| this.params.string_opt("rawPredictionCol");
        weight_col -> Option<&str> = |this| this.params.string_opt("weightCol") => "weightCol";
    }
}

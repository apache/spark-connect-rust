// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of OneVsRestModel. Its fit and transform run in the reduction of
//! [`crate::ml::OneVsRest`] and [`crate::ml::OneVsRestModel`]; its persistence is the base's.

use crate::ml::{
    DecisionTreeClassificationModel, FMClassificationModel, GBTClassificationModel, LinearSVCModel,
    LogisticRegressionModel, Model, MultilayerPerceptronClassificationModel, NaiveBayesModel,
    OneVsRestModel, RandomForestClassificationModel,
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

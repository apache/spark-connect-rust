// SPDX-License-Identifier: Apache-2.0
//! The members that every classification model has, as `_JavaClassificationModel` and
//! `_JavaProbabilisticClassificationModel` of PySpark: each a fetch of the model's object.

use crate::ml::object::{FetchArg, MlValue};
use crate::ml::tree_api::{as_i32, as_vector, fetch_value};
use crate::ml::{MlVector, PredictionModel};
use spark_connect_core::error::Result;

/// The members of a classification model (`_JavaClassificationModel`), with those of every
/// prediction model ([`PredictionModel`]: `numFeatures` and `predict`).
pub trait ClassificationModel: PredictionModel {
    /// The number of classes (`numClasses`).
    fn num_classes(&self) -> Result<i32> {
        as_i32(
            "numClasses",
            fetch_value(&self.model_target()?, "numClasses", vec![])?,
        )
    }

    /// The raw prediction of `features`, a vector of a confidence per class (`predictRaw`).
    fn predict_raw(&self, features: &MlVector) -> Result<MlVector> {
        let args = vec![FetchArg::Value(MlValue::Vector(features.clone()))];
        as_vector(
            "predictRaw",
            fetch_value(&self.model_target()?, "predictRaw", args)?,
        )
    }
}

/// The members of a probabilistic classification model (`_JavaProbabilisticClassificationModel`).
pub trait ProbabilisticClassificationModel: ClassificationModel {
    /// The probability of each class of `features` (`predictProbability`).
    fn predict_probability(&self, features: &MlVector) -> Result<MlVector> {
        let args = vec![FetchArg::Value(MlValue::Vector(features.clone()))];
        let value = fetch_value(&self.model_target()?, "predictProbability", args)?;
        as_vector("predictProbability", value)
    }
}

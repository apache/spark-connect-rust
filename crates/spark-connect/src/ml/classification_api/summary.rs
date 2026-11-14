// SPDX-License-Identifier: Apache-2.0
//! The members of the classification summaries, as `_ClassificationSummary`,
//! `_BinaryClassificationSummary` and `_TrainingSummary` of PySpark.
//!
//! A summary type holds the target of its server-side summary: `"<model>.summary"` of a training
//! summary, which carries the training dataset for the server to recreate a lost summary, or the
//! summary of an `evaluate`. Each member is a fetch of the target, and a member that PySpark
//! returns as a DataFrame is an attribute relation of the target.

use crate::dataframe::DataFrame;
use crate::ml::object::{attribute_relation, FetchArg, MlTarget, MlValue};
use crate::ml::tree_api::{as_f64, as_f64s, as_i32, as_string, fetch_value};
use spark_connect_core::error::Result;

/// A summary of a model: the target of its server-side summary, which every summary trait reads.
/// The summary traits of the other families build on it too.
pub trait MlSummary {
    /// The target of the summary's fetches.
    fn summary_target(&self) -> &MlTarget;
}

/// The members of a classification summary (`_ClassificationSummary`).
pub trait ClassificationSummary: MlSummary {
    /// The DataFrame of the predictions of the model (`predictions`).
    fn predictions(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "predictions", vec![])
    }

    /// The name of the prediction column (`predictionCol`).
    fn prediction_col(&self) -> Result<String> {
        as_string(
            "predictionCol",
            fetch_value(self.summary_target(), "predictionCol", vec![])?,
        )
    }

    /// The name of the label column (`labelCol`).
    fn label_col(&self) -> Result<String> {
        as_string(
            "labelCol",
            fetch_value(self.summary_target(), "labelCol", vec![])?,
        )
    }

    /// The name of the weight column (`weightCol`).
    fn weight_col(&self) -> Result<String> {
        as_string(
            "weightCol",
            fetch_value(self.summary_target(), "weightCol", vec![])?,
        )
    }

    /// The labels, in ascending order (`labels`).
    fn labels(&self) -> Result<Vec<f64>> {
        as_f64s(
            "labels",
            fetch_value(self.summary_target(), "labels", vec![])?,
        )
    }

    /// The true positive rate of each label (`truePositiveRateByLabel`).
    fn true_positive_rate_by_label(&self) -> Result<Vec<f64>> {
        by_label(self.summary_target(), "truePositiveRateByLabel")
    }

    /// The false positive rate of each label (`falsePositiveRateByLabel`).
    fn false_positive_rate_by_label(&self) -> Result<Vec<f64>> {
        by_label(self.summary_target(), "falsePositiveRateByLabel")
    }

    /// The precision of each label (`precisionByLabel`).
    fn precision_by_label(&self) -> Result<Vec<f64>> {
        by_label(self.summary_target(), "precisionByLabel")
    }

    /// The recall of each label (`recallByLabel`).
    fn recall_by_label(&self) -> Result<Vec<f64>> {
        by_label(self.summary_target(), "recallByLabel")
    }

    /// The F-measure of each label, of `beta` (`fMeasureByLabel(beta)`; PySpark's default beta is
    /// 1.0).
    fn f_measure_by_label(&self, beta: f64) -> Result<Vec<f64>> {
        let args = vec![FetchArg::Value(MlValue::Double(beta))];
        as_f64s(
            "fMeasureByLabel",
            fetch_value(self.summary_target(), "fMeasureByLabel", args)?,
        )
    }

    /// The accuracy (`accuracy`).
    fn accuracy(&self) -> Result<f64> {
        scalar(self.summary_target(), "accuracy")
    }

    /// The true positive rate, weighted by the label frequency (`weightedTruePositiveRate`).
    fn weighted_true_positive_rate(&self) -> Result<f64> {
        scalar(self.summary_target(), "weightedTruePositiveRate")
    }

    /// The false positive rate, weighted by the label frequency (`weightedFalsePositiveRate`).
    fn weighted_false_positive_rate(&self) -> Result<f64> {
        scalar(self.summary_target(), "weightedFalsePositiveRate")
    }

    /// The recall, weighted by the label frequency (`weightedRecall`).
    fn weighted_recall(&self) -> Result<f64> {
        scalar(self.summary_target(), "weightedRecall")
    }

    /// The precision, weighted by the label frequency (`weightedPrecision`).
    fn weighted_precision(&self) -> Result<f64> {
        scalar(self.summary_target(), "weightedPrecision")
    }

    /// The F-measure of `beta`, weighted by the label frequency (`weightedFMeasure(beta)`;
    /// PySpark's default beta is 1.0).
    fn weighted_f_measure(&self, beta: f64) -> Result<f64> {
        let args = vec![FetchArg::Value(MlValue::Double(beta))];
        as_f64(
            "weightedFMeasure",
            fetch_value(self.summary_target(), "weightedFMeasure", args)?,
        )
    }
}

/// The members of a binary classification summary (`_BinaryClassificationSummary`).
pub trait BinaryClassificationSummary: ClassificationSummary {
    /// The name of the score column (`scoreCol`).
    fn score_col(&self) -> Result<String> {
        as_string(
            "scoreCol",
            fetch_value(self.summary_target(), "scoreCol", vec![])?,
        )
    }

    /// The DataFrame of the receiver operating characteristic curve (`roc`): `FPR` and `TPR`.
    fn roc(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "roc", vec![])
    }

    /// The area under the ROC curve (`areaUnderROC`).
    fn area_under_roc(&self) -> Result<f64> {
        scalar(self.summary_target(), "areaUnderROC")
    }

    /// The DataFrame of the precision-recall curve (`pr`): `recall` and `precision`.
    fn pr(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "pr", vec![])
    }

    /// The DataFrame of the F-measure (beta 1.0) of each threshold (`fMeasureByThreshold`).
    fn f_measure_by_threshold(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "fMeasureByThreshold", vec![])
    }

    /// The DataFrame of the precision of each threshold (`precisionByThreshold`).
    fn precision_by_threshold(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "precisionByThreshold", vec![])
    }

    /// The DataFrame of the recall of each threshold (`recallByThreshold`).
    fn recall_by_threshold(&self) -> Result<DataFrame> {
        attribute_relation(self.summary_target(), "recallByThreshold", vec![])
    }
}

/// The members of a training summary (`_TrainingSummary`), which the regression training summaries
/// have too.
pub trait TrainingSummary: MlSummary {
    /// The objective function's value at each iteration (`objectiveHistory`).
    fn objective_history(&self) -> Result<Vec<f64>> {
        let value = fetch_value(self.summary_target(), "objectiveHistory", vec![])?;
        as_f64s("objectiveHistory", value)
    }

    /// The number of training iterations (`totalIterations`).
    fn total_iterations(&self) -> Result<i32> {
        let value = fetch_value(self.summary_target(), "totalIterations", vec![])?;
        as_i32("totalIterations", value)
    }
}

fn scalar(target: &MlTarget, method: &str) -> Result<f64> {
    as_f64(method, fetch_value(target, method, vec![])?)
}

fn by_label(target: &MlTarget, method: &str) -> Result<Vec<f64>> {
    as_f64s(method, fetch_value(target, method, vec![])?)
}

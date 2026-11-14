// SPDX-License-Identifier: Apache-2.0
//! The members of the tree models (`_DecisionTreeModel` and `_TreeEnsembleModel` of PySpark),
//! which the classification and regression trees share, over the server-side objects of
//! [`super::object`].
//!
//! A model implements [`DecisionTreeModel`] or [`TreeEnsembleModel`], and each member is a
//! fetch of the model's object, as PySpark's `_call_java` of the member. An ensemble names the type
//! of its trees, which PySpark builds of each id of the comma-joined `trees` result.

use super::object::{fetch, FetchArg, MlTarget, MlValue};
use super::{MlRef, MlVector, Model};
use spark_connect_core::error::{Result, SparkError};

/// The value of the member `method` of `target`, of the arguments `args`.
pub(crate) fn fetch_value(target: &MlTarget, method: &str, args: Vec<FetchArg>) -> Result<MlValue> {
    fetch(target, method, args)?.into_value()
}

fn unexpected(method: &str, expected: &str, value: &MlValue) -> SparkError {
    SparkError::value_msg(format!("expected {expected} of `{method}`, got {value:?}"))
}

/// An int value of `method`, as PySpark reads an `Int` literal.
pub(crate) fn as_i32(method: &str, value: MlValue) -> Result<i32> {
    match value {
        MlValue::Int(v) | MlValue::Short(v) | MlValue::Byte(v) => Ok(v),
        other => Err(unexpected(method, "an int", &other)),
    }
}

/// A double value of `method`, of a `Double` or a `Float` literal.
pub(crate) fn as_f64(method: &str, value: MlValue) -> Result<f64> {
    match value {
        MlValue::Double(v) => Ok(v),
        MlValue::Float(v) => Ok(f64::from(v)),
        other => Err(unexpected(method, "a double", &other)),
    }
}

/// A string value of `method`.
pub(crate) fn as_string(method: &str, value: MlValue) -> Result<String> {
    match value {
        MlValue::String(v) => Ok(v),
        other => Err(unexpected(method, "a string", &other)),
    }
}

/// An ML vector value of `method`.
pub(crate) fn as_vector(method: &str, value: MlValue) -> Result<MlVector> {
    match value {
        MlValue::Vector(v) => Ok(v),
        other => Err(unexpected(method, "a vector", &other)),
    }
}

/// An array of doubles of `method`, as PySpark reads a list of floats.
pub(crate) fn as_f64s(method: &str, value: MlValue) -> Result<Vec<f64>> {
    match value {
        MlValue::Array(items) => items.into_iter().map(|item| as_f64(method, item)).collect(),
        other => Err(unexpected(method, "an array of doubles", &other)),
    }
}

/// The members of a decision tree model, as `_DecisionTreeModel` of PySpark: each a fetch of the
/// model's object. The default methods read the target of [`Model::ml_ref`].
pub trait DecisionTreeModel: Model {
    /// The depth of the tree (`depth`).
    fn depth(&self) -> Result<i32> {
        as_i32(
            "depth",
            fetch_value(&self.model_target()?, "depth", vec![])?,
        )
    }

    /// The number of nodes of the tree (`numNodes`).
    fn num_nodes(&self) -> Result<i32> {
        as_i32(
            "numNodes",
            fetch_value(&self.model_target()?, "numNodes", vec![])?,
        )
    }

    /// The full description of the tree (`toDebugString`).
    fn to_debug_string(&self) -> Result<String> {
        fetch(&self.model_target()?, "toDebugString", vec![])?.into_string()
    }

    /// The importance of each feature (`featureImportances`), a sparse vector.
    fn feature_importances(&self) -> Result<MlVector> {
        let value = fetch_value(&self.model_target()?, "featureImportances", vec![])?;
        as_vector("featureImportances", value)
    }

    /// The index of the leaf that `features` falls in (`predictLeaf`).
    fn predict_leaf(&self, features: &MlVector) -> Result<f64> {
        let args = vec![FetchArg::Value(MlValue::Vector(features.clone()))];
        as_f64(
            "predictLeaf",
            fetch_value(&self.model_target()?, "predictLeaf", args)?,
        )
    }
}

/// The members of a tree ensemble model, as `_TreeEnsembleModel` of PySpark: each a fetch of the
/// model's object. An ensemble names the model type of its trees, which [`Self::tree_of`] builds of
/// the object of each tree.
pub trait TreeEnsembleModel: Model {
    /// The model type of the trees of the ensemble.
    type Tree;

    /// The tree of the server-side object `tree`, one of the ids of the `trees` result.
    fn tree_of(&self, tree: MlRef) -> Self::Tree;

    /// The trees of the ensemble (`trees`), each its own server-side object, as PySpark builds a
    /// model of each id of the comma-joined result.
    fn trees(&self) -> Result<Vec<Self::Tree>> {
        let trees = fetch(&self.model_target()?, "trees", vec![])?.into_objects()?;
        Ok(trees.into_iter().map(|tree| self.tree_of(tree)).collect())
    }

    /// The weight of each tree (`treeWeights`).
    fn tree_weights(&self) -> Result<Vec<f64>> {
        as_f64s(
            "treeWeights",
            fetch_value(&self.model_target()?, "treeWeights", vec![])?,
        )
    }

    /// The number of trees (`getNumTrees`).
    fn get_num_trees(&self) -> Result<i32> {
        as_i32(
            "getNumTrees",
            fetch_value(&self.model_target()?, "getNumTrees", vec![])?,
        )
    }

    /// The total number of nodes of the trees (`totalNumNodes`).
    fn total_num_nodes(&self) -> Result<i32> {
        let value = fetch_value(&self.model_target()?, "totalNumNodes", vec![])?;
        as_i32("totalNumNodes", value)
    }

    /// The full description of the ensemble (`toDebugString`).
    fn to_debug_string(&self) -> Result<String> {
        fetch(&self.model_target()?, "toDebugString", vec![])?.into_string()
    }

    /// The importance of each feature (`featureImportances`), a sparse vector.
    fn feature_importances(&self) -> Result<MlVector> {
        let value = fetch_value(&self.model_target()?, "featureImportances", vec![])?;
        as_vector("featureImportances", value)
    }

    /// The index of the leaf that `features` falls in, of each tree (`predictLeaf`): a vector.
    fn predict_leaf(&self, features: &MlVector) -> Result<MlVector> {
        let args = vec![FetchArg::Value(MlValue::Vector(features.clone()))];
        as_vector(
            "predictLeaf",
            fetch_value(&self.model_target()?, "predictLeaf", args)?,
        )
    }
}

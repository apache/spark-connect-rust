// SPDX-License-Identifier: Apache-2.0
//! The decision tree, random forest and gradient-boosted tree regression models, as PySpark's
//! `DecisionTreeRegressionModel`, `RandomForestRegressionModel` and `GBTRegressionModel`: the
//! members of the trees ([`DecisionTreeModel`] and [`TreeEnsembleModel`]), of every prediction
//! model ([`PredictionModel`]), and the error of a gradient-boosted model after each iteration.

use crate::dataframe::DataFrame;
use crate::ml::object::{FetchArg, MlValue};
use crate::ml::tree_api::{as_f64s, fetch_value};
use crate::ml::{
    fit_via_command, read_model, read_operator, DecisionTreeModel, DecisionTreeRegressionModel,
    DecisionTreeRegressor, Estimator, GBTRegressionModel, GBTRegressor, MlOperator, MlRef, Model,
    OperatorType, Params, PredictionModel, RandomForestRegressionModel, RandomForestRegressor,
    TreeEnsembleModel,
};
use crate::session::SparkSession;
use spark_connect_core::error::Result;

const DECISION_TREE_REGRESSOR: &str = "org.apache.spark.ml.regression.DecisionTreeRegressor";
const DECISION_TREE_REGRESSION_MODEL: &str =
    "org.apache.spark.ml.regression.DecisionTreeRegressionModel";
const RANDOM_FOREST_REGRESSOR: &str = "org.apache.spark.ml.regression.RandomForestRegressor";
const RANDOM_FOREST_REGRESSION_MODEL: &str =
    "org.apache.spark.ml.regression.RandomForestRegressionModel";
const GBT_REGRESSOR: &str = "org.apache.spark.ml.regression.GBTRegressor";
const GBT_REGRESSION_MODEL: &str = "org.apache.spark.ml.regression.GBTRegressionModel";

impl DecisionTreeRegressor {
    /// Fit the decision tree of `df`, as PySpark's `fit` returns a `DecisionTreeRegressionModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<DecisionTreeRegressionModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator saved at `path`, as `DecisionTreeRegressor.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            DECISION_TREE_REGRESSOR,
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }
}

impl DecisionTreeRegressionModel {
    /// The tree of the server-side object `ml_ref`, of the params `params` and of a new uid, as
    /// PySpark's `DecisionTreeRegressionModel(RemoteModelRef(id))` of a tree of an ensemble (whose
    /// params are its defaults).
    pub(crate) fn from_ml_ref(ml_ref: MlRef, params: Params) -> Self {
        Self {
            operator: MlOperator::new(DECISION_TREE_REGRESSION_MODEL, OperatorType::Model),
            params,
            cached: ml_ref,
        }
    }

    /// The model saved at `path`, as `DecisionTreeRegressionModel.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, DECISION_TREE_REGRESSION_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

impl PredictionModel for DecisionTreeRegressionModel {}

impl DecisionTreeModel for DecisionTreeRegressionModel {}

impl RandomForestRegressor {
    /// Fit the random forest of `df`, as PySpark's `fit` returns a `RandomForestRegressionModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<RandomForestRegressionModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator saved at `path`, as `RandomForestRegressor.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            RANDOM_FOREST_REGRESSOR,
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }
}

impl RandomForestRegressionModel {
    /// The model saved at `path`, as `RandomForestRegressionModel.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, RANDOM_FOREST_REGRESSION_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

impl PredictionModel for RandomForestRegressionModel {}

impl TreeEnsembleModel for RandomForestRegressionModel {
    type Tree = DecisionTreeRegressionModel;

    fn tree_of(&self, tree: MlRef) -> DecisionTreeRegressionModel {
        DecisionTreeRegressionModel::from_ml_ref(tree, Params::new())
    }
}

impl GBTRegressor {
    /// Fit the gradient-boosted trees of `df`, as PySpark's `fit` returns a `GBTRegressionModel`.
    pub fn fit(&mut self, df: &DataFrame) -> Result<GBTRegressionModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        self.model_from_info(info, &df.session)
    }

    /// The estimator saved at `path`, as `GBTRegressor.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) =
            read_operator(session, GBT_REGRESSOR, OperatorType::Estimator, path)?;
        Ok(Self { operator, params })
    }
}

impl GBTRegressionModel {
    /// The error of the model of `dataset` after each boosting iteration, of the loss `loss`
    /// ("squared" or "absolute"), as `GBTRegressionModel.evaluateEachIteration` of PySpark.
    pub fn evaluate_each_iteration(&self, dataset: &DataFrame, loss: &str) -> Result<Vec<f64>> {
        let args = vec![
            FetchArg::Input(dataset.clone()),
            FetchArg::Value(MlValue::String(loss.to_string())),
        ];
        let value = fetch_value(&self.model_target()?, "evaluateEachIteration", args)?;
        as_f64s("evaluateEachIteration", value)
    }

    /// The model saved at `path`, as `GBTRegressionModel.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (cached, operator, params) = read_model(session, GBT_REGRESSION_MODEL, path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
    }
}

impl PredictionModel for GBTRegressionModel {}

impl TreeEnsembleModel for GBTRegressionModel {
    type Tree = DecisionTreeRegressionModel;

    fn tree_of(&self, tree: MlRef) -> DecisionTreeRegressionModel {
        DecisionTreeRegressionModel::from_ml_ref(tree, Params::new())
    }
}

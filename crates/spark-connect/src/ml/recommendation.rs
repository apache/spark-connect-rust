// SPDX-License-Identifier: Apache-2.0
//! Alternating least squares recommendation through Spark Connect.
use super::*;

#[derive(Debug, Clone)]
pub struct ALS {
    operator: MlOperator,
    params: Params,
}
impl ALS {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new(
                "org.apache.spark.ml.recommendation.ALS",
                OperatorType::Estimator,
            ),
            params: Params::new(),
        }
    }
    /// Fit the typed ALS model, including factor tables and recommendation methods.
    pub fn fit(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<ALSModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        let mut params = self.params.clone();
        params.params.retain(|name, _| {
            [
                "blockSize",
                "coldStartStrategy",
                "itemCol",
                "predictionCol",
                "userCol",
            ]
            .contains(&name.as_str())
        });
        Ok(ALSModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.recommendation.ALSModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            rank: self.rank(),
            params,
            cached: CachedModel::from_info(&info, &df.session, "ALS")?,
        })
    }
    /// Set both user and item partition counts.
    pub fn set_num_blocks(self, value: i32) -> Self {
        self.set_num_user_blocks(value).set_num_item_blocks(value)
    }

    /// Load an `ALS` saved to `path`, as PySpark's `ALS.load(path)`: its uid and params.
    pub fn load(session: &SparkSession, path: &str) -> spark_connect_core::error::Result<Self> {
        let (operator, params) = read_operator(
            session,
            "org.apache.spark.ml.recommendation.ALS",
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
    }
}
impl Default for ALS {
    fn default() -> Self {
        Self::new()
    }
}
impl Estimator for ALS {
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
    fn fit_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<Box<dyn Model>> {
        Ok(Box::new(self.fit(df)?))
    }
}

/// A fitted, session-owned ALS model. Lazy result tables keep its cache entry alive.
#[derive(Debug, Clone)]
pub struct ALSModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
    rank: i32,
}
impl ALSModel {
    /// The rank of the model's factors: of its estimator, as fitted, or as the server reads it of a
    /// loaded model (`rank` of PySpark's `ALSModel`).
    pub fn rank(&self) -> i32 {
        self.rank
    }

    /// Load an `ALSModel` saved to `path`, as PySpark's `ALSModel.load(path)`: the server caches
    /// it, as a fitted model, and gives its rank.
    pub fn load(session: &SparkSession, path: &str) -> spark_connect_core::error::Result<Self> {
        let (cached, operator, params) =
            read_model(session, "org.apache.spark.ml.recommendation.ALSModel", path)?;
        let rank = fetch(&MlTarget::object(&cached), "rank", vec![])?.into_value()?;
        let rank = rank.as_i64().ok_or_else(|| {
            SparkError::value_msg(format!("rank of ALSModel is not an int: {rank:?}"))
        })? as i32;
        Ok(Self {
            operator,
            params,
            cached,
            rank,
        })
    }
    fn fetch(
        &self,
        method: &str,
        args: Vec<proto::fetch::method::Args>,
        input: Option<&DataFrame>,
    ) -> spark_connect_core::error::Result<DataFrame> {
        let relation = proto::Relation {
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                proto::MlRelation {
                    ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                        obj_ref: Some(self.cached.0.model_ref.clone()),
                        methods: vec![proto::fetch::Method {
                            method: method.into(),
                            args,
                        }],
                    })),
                    ..Default::default()
                },
            ))),
            ..Default::default()
        };
        if let Some(df) = input {
            Ok(DataFrame::new(
                self.cached.0.session.clone(),
                LogicalPlan::MlTransform {
                    ml_relation: relation,
                    input: df.plan.clone(),
                    owner: Some(RelationOwner(self.cached.0.clone())),
                },
            ))
        } else {
            DataFrame::from_relation(
                self.cached.0.session.clone(),
                relation,
                Some(self.cached.0.clone()),
            )
        }
    }
    pub fn user_factors(&self) -> spark_connect_core::error::Result<DataFrame> {
        self.fetch("userFactors", Vec::new(), None)
    }
    pub fn item_factors(&self) -> spark_connect_core::error::Result<DataFrame> {
        self.fetch("itemFactors", Vec::new(), None)
    }
    /// Binding seam that preserves the literal type of Python recommendation-count arguments.
    #[doc(hidden)]
    pub fn recommend_with_literal(
        &self,
        users: bool,
        subset: Option<&DataFrame>,
        count: proto::expression::Literal,
    ) -> spark_connect_core::error::Result<DataFrame> {
        use proto::fetch::method::args::ArgsType;
        let mut args = Vec::new();
        if let Some(df) = subset {
            if df.session.session_id() != self.cached.0.session.session_id() {
                return Err(spark_connect_core::error::SparkError::value_msg(
                    "A fitted ML model must recommend data from its fitting session",
                ));
            }
            args.push(proto::fetch::method::Args {
                args_type: Some(ArgsType::Input(df.relation()?)),
            });
        }
        args.push(proto::fetch::method::Args {
            args_type: Some(ArgsType::Param(count)),
        });
        let method = match (users, subset.is_some()) {
            (true, false) => "recommendForAllUsers",
            (false, false) => "recommendForAllItems",
            (true, true) => "recommendForUserSubset",
            (false, true) => "recommendForItemSubset",
        };
        self.fetch(method, args, subset)
    }
    pub fn recommend_for_all_users(
        &self,
        num_items: i32,
    ) -> spark_connect_core::error::Result<DataFrame> {
        self.recommend_with_literal(
            true,
            None,
            LiteralExpression::Integer(num_items).to_literal_proto(),
        )
    }
    pub fn recommend_for_all_items(
        &self,
        num_users: i32,
    ) -> spark_connect_core::error::Result<DataFrame> {
        self.recommend_with_literal(
            false,
            None,
            LiteralExpression::Integer(num_users).to_literal_proto(),
        )
    }
    pub fn recommend_for_user_subset(
        &self,
        dataset: &DataFrame,
        num_items: i32,
    ) -> spark_connect_core::error::Result<DataFrame> {
        self.recommend_with_literal(
            true,
            Some(dataset),
            LiteralExpression::Integer(num_items).to_literal_proto(),
        )
    }
    pub fn recommend_for_item_subset(
        &self,
        dataset: &DataFrame,
        num_users: i32,
    ) -> spark_connect_core::error::Result<DataFrame> {
        self.recommend_with_literal(
            false,
            Some(dataset),
            LiteralExpression::Integer(num_users).to_literal_proto(),
        )
    }
}
impl Transformer for ALSModel {
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
    fn build_ml_relation(
        &self,
        df: &DataFrame,
    ) -> spark_connect_core::error::Result<proto::MlRelation> {
        self.cached.build_ml_relation(df, &self.params)
    }
    fn transform_impl(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        self.cached.transform(df, &self.params)
    }
}
impl Model for ALSModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

impl ALS {
    /// Set `alpha`; Spark validates its value.
    pub fn set_alpha(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("alpha", value);
        self
    }
    /// Read `alpha`, with Spark's default when defined.
    pub fn alpha(&self) -> f64 {
        self.params.double_or("alpha", 1.0)
    }
    /// Set `blockSize`; Spark validates its value.
    pub fn set_block_size(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("blockSize", value);
        self
    }
    /// Read `blockSize`, with Spark's default when defined.
    pub fn block_size(&self) -> i32 {
        self.params.int_or("blockSize", 4096)
    }
    /// Set `checkpointInterval`; Spark validates its value.
    pub fn set_checkpoint_interval(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("checkpointInterval", value);
        self
    }
    /// Read `checkpointInterval`, with Spark's default when defined.
    pub fn checkpoint_interval(&self) -> i32 {
        self.params.int_or("checkpointInterval", 10)
    }
    /// Set `coldStartStrategy`; Spark validates its value.
    pub fn set_cold_start_strategy(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("coldStartStrategy", value);
        self
    }
    /// Read `coldStartStrategy`, with Spark's default when defined.
    pub fn cold_start_strategy(&self) -> &str {
        self.params.string_or("coldStartStrategy", "nan")
    }
    /// Set `finalStorageLevel`; Spark validates its value.
    pub fn set_final_storage_level(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("finalStorageLevel", value);
        self
    }
    /// Read `finalStorageLevel`, with Spark's default when defined.
    pub fn final_storage_level(&self) -> &str {
        self.params
            .string_or("finalStorageLevel", "MEMORY_AND_DISK")
    }
    /// Set `implicitPrefs`; Spark validates its value.
    pub fn set_implicit_prefs(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("implicitPrefs", value);
        self
    }
    /// Read `implicitPrefs`, with Spark's default when defined.
    pub fn implicit_prefs(&self) -> bool {
        self.params.bool_or("implicitPrefs", false)
    }
    /// Set `intermediateStorageLevel`; Spark validates its value.
    pub fn set_intermediate_storage_level(mut self, value: &str) -> Self {
        self.params = self
            .params
            .set_param_string("intermediateStorageLevel", value);
        self
    }
    /// Read `intermediateStorageLevel`, with Spark's default when defined.
    pub fn intermediate_storage_level(&self) -> &str {
        self.params
            .string_or("intermediateStorageLevel", "MEMORY_AND_DISK")
    }
    /// Set `itemCol`; Spark validates its value.
    pub fn set_item_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("itemCol", value);
        self
    }
    /// Read `itemCol`, with Spark's default when defined.
    pub fn item_col(&self) -> &str {
        self.params.string_or("itemCol", "item")
    }
    /// Set `maxIter`; Spark validates its value.
    pub fn set_max_iter(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("maxIter", value);
        self
    }
    /// Read `maxIter`, with Spark's default when defined.
    pub fn max_iter(&self) -> i32 {
        self.params.int_or("maxIter", 10)
    }
    /// Set `nonnegative`; Spark validates its value.
    pub fn set_nonnegative(mut self, value: bool) -> Self {
        self.params = self.params.set_param_bool("nonnegative", value);
        self
    }
    /// Read `nonnegative`, with Spark's default when defined.
    pub fn nonnegative(&self) -> bool {
        self.params.bool_or("nonnegative", false)
    }
    /// Set `numItemBlocks`; Spark validates its value.
    pub fn set_num_item_blocks(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("numItemBlocks", value);
        self
    }
    /// Read `numItemBlocks`, with Spark's default when defined.
    pub fn num_item_blocks(&self) -> i32 {
        self.params.int_or("numItemBlocks", 10)
    }
    /// Set `numUserBlocks`; Spark validates its value.
    pub fn set_num_user_blocks(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("numUserBlocks", value);
        self
    }
    /// Read `numUserBlocks`, with Spark's default when defined.
    pub fn num_user_blocks(&self) -> i32 {
        self.params.int_or("numUserBlocks", 10)
    }
    /// Set `predictionCol`; Spark validates its value.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Read `predictionCol`, with Spark's default when defined.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
    /// Set `rank`; Spark validates its value.
    pub fn set_rank(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("rank", value);
        self
    }
    /// Read `rank`, with Spark's default when defined.
    pub fn rank(&self) -> i32 {
        self.params.int_or("rank", 10)
    }
    /// Set `ratingCol`; Spark validates its value.
    pub fn set_rating_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("ratingCol", value);
        self
    }
    /// Read `ratingCol`, with Spark's default when defined.
    pub fn rating_col(&self) -> &str {
        self.params.string_or("ratingCol", "rating")
    }
    /// Set `regParam`; Spark validates its value.
    pub fn set_reg_param(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("regParam", value);
        self
    }
    /// Read `regParam`, with Spark's default when defined.
    pub fn reg_param(&self) -> f64 {
        self.params.double_or("regParam", 0.1)
    }
    /// Set `seed`; Spark validates its value.
    pub fn set_seed(mut self, value: i64) -> Self {
        self.params = self.params.set_param_int_narrow("seed", value);
        self
    }
    /// Read `seed`, with Spark's default when defined.
    pub fn seed(&self) -> Option<i64> {
        self.params.long_opt("seed")
    }
    /// Set `userCol`; Spark validates its value.
    pub fn set_user_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("userCol", value);
        self
    }
    /// Read `userCol`, with Spark's default when defined.
    pub fn user_col(&self) -> &str {
        self.params.string_or("userCol", "user")
    }
}

impl ALSModel {
    /// Set `blockSize`; Spark validates its value.
    pub fn set_block_size(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("blockSize", value);
        self
    }
    /// Read `blockSize`, with Spark's default when defined.
    pub fn block_size(&self) -> i32 {
        self.params.int_or("blockSize", 4096)
    }
    /// Set `coldStartStrategy`; Spark validates its value.
    pub fn set_cold_start_strategy(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("coldStartStrategy", value);
        self
    }
    /// Read `coldStartStrategy`, with Spark's default when defined.
    pub fn cold_start_strategy(&self) -> &str {
        self.params.string_or("coldStartStrategy", "nan")
    }
    /// Set `itemCol`; Spark validates its value.
    pub fn set_item_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("itemCol", value);
        self
    }
    /// Read `itemCol`, with Spark's default when defined.
    pub fn item_col(&self) -> &str {
        self.params.string_or("itemCol", "item")
    }
    /// Set `predictionCol`; Spark validates its value.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Read `predictionCol`, with Spark's default when defined.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
    /// Set `userCol`; Spark validates its value.
    pub fn set_user_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("userCol", value);
        self
    }
    /// Read `userCol`, with Spark's default when defined.
    pub fn user_col(&self) -> &str {
        self.params.string_or("userCol", "user")
    }
}

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
    /// The rank used by the successful fit.
    pub fn rank(&self) -> i32 {
        self.rank
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

ml_params! {
    ALS {
        set_alpha(value: f64) => "alpha", set_param_double;
        alpha -> f64 = |this| this.params.double_or("alpha", 1.0);
        set_block_size(value: i32) => "blockSize", set_param_int32;
        block_size -> i32 = |this| this.params.int_or("blockSize", 4096);
        set_checkpoint_interval(value: i32) => "checkpointInterval", set_param_int32;
        checkpoint_interval -> i32 = |this| this.params.int_or("checkpointInterval", 10);
        set_cold_start_strategy(value: &str) => "coldStartStrategy", set_param_string;
        cold_start_strategy -> &str = |this| this.params.string_or("coldStartStrategy", "nan");
        set_final_storage_level(value: &str) => "finalStorageLevel", set_param_string;
        final_storage_level -> &str =
            |this| this.params.string_or("finalStorageLevel", "MEMORY_AND_DISK");
        set_implicit_prefs(value: bool) => "implicitPrefs", set_param_bool;
        implicit_prefs -> bool = |this| this.params.bool_or("implicitPrefs", false);
        set_intermediate_storage_level(value: &str) => "intermediateStorageLevel", set_param_string;
        intermediate_storage_level -> &str =
            |this| this.params.string_or("intermediateStorageLevel", "MEMORY_AND_DISK");
        set_item_col(value: &str) => "itemCol", set_param_string;
        item_col -> &str = |this| this.params.string_or("itemCol", "item");
        set_max_iter(value: i32) => "maxIter", set_param_int32;
        max_iter -> i32 = |this| this.params.int_or("maxIter", 10);
        set_nonnegative(value: bool) => "nonnegative", set_param_bool;
        nonnegative -> bool = |this| this.params.bool_or("nonnegative", false);
        set_num_item_blocks(value: i32) => "numItemBlocks", set_param_int32;
        num_item_blocks -> i32 = |this| this.params.int_or("numItemBlocks", 10);
        set_num_user_blocks(value: i32) => "numUserBlocks", set_param_int32;
        num_user_blocks -> i32 = |this| this.params.int_or("numUserBlocks", 10);
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_rank(value: i32) => "rank", set_param_int32;
        rank -> i32 = |this| this.params.int_or("rank", 10);
        set_rating_col(value: &str) => "ratingCol", set_param_string;
        rating_col -> &str = |this| this.params.string_or("ratingCol", "rating");
        set_reg_param(value: f64) => "regParam", set_param_double;
        reg_param -> f64 = |this| this.params.double_or("regParam", 0.1);
        set_seed(value: i64) => "seed", set_param_int_narrow;
        seed -> i64 =
            |this| this.params.long_opt(
                "seed").unwrap_or(java_string_hash("org.apache.spark.ml.recommendation.ALS"));
        set_user_col(value: &str) => "userCol", set_param_string;
        user_col -> &str = |this| this.params.string_or("userCol", "user");
    }
}

ml_params! {
    ALSModel {
        set_block_size(value: i32) => "blockSize", set_param_int32;
        block_size -> i32 = |this| this.params.int_or("blockSize", 4096);
        set_cold_start_strategy(value: &str) => "coldStartStrategy", set_param_string;
        cold_start_strategy -> &str = |this| this.params.string_or("coldStartStrategy", "nan");
        set_item_col(value: &str) => "itemCol", set_param_string;
        item_col -> &str = |this| this.params.string_or("itemCol", "item");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
        set_user_col(value: &str) => "userCol", set_param_string;
        user_col -> &str = |this| this.params.string_or("userCol", "user");
    }
}

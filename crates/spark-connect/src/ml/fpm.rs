// SPDX-License-Identifier: Apache-2.0
//! Frequent itemsets, association rules and sequential patterns, executed by Spark Connect.
use super::*;

#[derive(Debug, Clone)]
pub struct FPGrowth {
    operator: MlOperator,
    params: Params,
}
impl FPGrowth {
    pub fn new() -> Self {
        Self {
            operator: MlOperator::new("org.apache.spark.ml.fpm.FPGrowth", OperatorType::Estimator),
            params: Params::new(),
        }
    }
    /// Fit and retain the typed model, including its mined itemsets and association rules.
    pub fn fit(&mut self, df: &DataFrame) -> spark_connect_core::error::Result<FPGrowthModel> {
        self.ensure_operator();
        let info = fit_via_command(&self.operator, &self.params, df)?;
        Ok(FPGrowthModel {
            operator: MlOperator::with_uid(
                "org.apache.spark.ml.fpm.FPGrowthModel",
                info.uid.as_deref().unwrap_or(&self.operator.uid),
                OperatorType::Model,
            ),
            params: self.params.clone(),
            cached: CachedModel::from_info(&info, &df.session, "FPGrowth")?,
        })
    }
}
impl Default for FPGrowth {
    fn default() -> Self {
        Self::new()
    }
}
impl Estimator for FPGrowth {
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

/// A session-owned FPGrowth model. Lazy result tables retain its server cache entry.
#[derive(Debug, Clone)]
pub struct FPGrowthModel {
    operator: MlOperator,
    params: Params,
    cached: CachedModel,
}
impl FPGrowthModel {
    fn attribute(&self, method: &str) -> spark_connect_core::error::Result<DataFrame> {
        let relation = proto::Relation {
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                proto::MlRelation {
                    ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                        obj_ref: Some(self.cached.0.model_ref.clone()),
                        methods: vec![proto::fetch::Method {
                            method: method.into(),
                            args: Vec::new(),
                        }],
                    })),
                    ..Default::default()
                },
            ))),
            ..Default::default()
        };
        DataFrame::from_relation(
            self.cached.0.session.clone(),
            relation,
            Some(self.cached.0.clone()),
        )
    }
    /// Frequent itemsets and their counts, as a lazy Spark DataFrame.
    pub fn freq_itemsets(&self) -> spark_connect_core::error::Result<DataFrame> {
        self.attribute("freqItemsets")
    }
    /// Association rules computed by the fitted server model.
    pub fn association_rules(&self) -> spark_connect_core::error::Result<DataFrame> {
        self.attribute("associationRules")
    }
}
impl Transformer for FPGrowthModel {
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
impl Model for FPGrowthModel {
    fn ml_ref(&self) -> Option<&MlRef> {
        Some(&self.cached)
    }

    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// PrefixSpan mines sequential patterns directly; it has no fit/transform model.
#[derive(Debug, Clone, Default)]
pub struct PrefixSpan {
    params: Params,
}
impl PrefixSpan {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn params(&self) -> &Params {
        &self.params
    }
    pub fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }
    /// Build Spark's lazy PrefixSpan helper relation and retain the input's resources.
    pub fn find_frequent_sequential_patterns(
        &self,
        dataset: &DataFrame,
    ) -> spark_connect_core::error::Result<DataFrame> {
        use proto::expression::literal::LiteralType as L;
        use proto::fetch::method::args::ArgsType;
        let mut params = self.params.to_proto().params;
        let defaults = [
            ("minSupport", L::Double(0.1)),
            ("maxPatternLength", L::Integer(10)),
            ("maxLocalProjDBSize", L::Integer(32000000)),
            ("sequenceCol", L::String("sequence".into())),
        ];
        let mut args = vec![proto::fetch::method::Args {
            args_type: Some(ArgsType::Input(dataset.relation()?)),
        }];
        for (name, default) in defaults {
            let value = params
                .remove(name)
                .unwrap_or_else(|| proto::expression::Literal {
                    literal_type: Some(default),
                    ..Default::default()
                });
            args.push(proto::fetch::method::Args {
                args_type: Some(ArgsType::Param(value)),
            });
        }
        let relation = proto::Relation {
            rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
                proto::MlRelation {
                    ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                        obj_ref: Some(proto::ObjectRef {
                            id: "______ML_CONNECT_HELPER______".into(),
                        }),
                        methods: vec![proto::fetch::Method {
                            method: "prefixSpanFindFrequentSequentialPatterns".into(),
                            args,
                        }],
                    })),
                    ..Default::default()
                },
            ))),
            ..Default::default()
        };
        Ok(DataFrame::new(
            dataset.session.clone(),
            LogicalPlan::MlTransform {
                ml_relation: relation,
                input: dataset.plan.clone(),
                owner: None,
            },
        ))
    }
}

ml_params! {
    FPGrowth {
        set_items_col(value: &str) => "itemsCol", set_param_string;
        items_col -> &str = |this| this.params.string_or("itemsCol", "items");
        set_min_confidence(value: f64) => "minConfidence", set_param_double;
        min_confidence -> f64 = |this| this.params.double_or("minConfidence", 0.8);
        set_min_support(value: f64) => "minSupport", set_param_double;
        min_support -> f64 = |this| this.params.double_or("minSupport", 0.3);
        set_num_partitions(value: i32) => "numPartitions", set_param_int32;
        num_partitions -> Option<i32> = |this| this.params.int_opt("numPartitions");
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
    }
}

ml_params! {
    FPGrowthModel {
        set_items_col(value: &str) => "itemsCol", set_param_string;
        items_col -> &str = |this| this.params.string_or("itemsCol", "items");
        set_min_confidence(value: f64) => "minConfidence", set_param_double;
        min_confidence -> f64 = |this| this.params.double_or("minConfidence", 0.8);
        min_support -> f64 = |this| this.params.double_or("minSupport", 0.3) => "minSupport";
        num_partitions -> Option<i32> =
            |this| this.params.int_opt("numPartitions") => "numPartitions";
        set_prediction_col(value: &str) => "predictionCol", set_param_string;
        prediction_col -> &str = |this| this.params.string_or("predictionCol", "prediction");
    }
}

ml_params! {
    PrefixSpan {
        set_max_local_proj_db_size(value: i64) => "maxLocalProjDBSize", set_param_int_narrow;
        max_local_proj_db_size -> i64 =
            |this| this.params.long_opt("maxLocalProjDBSize").unwrap_or(32000000);
        set_max_pattern_length(value: i32) => "maxPatternLength", set_param_int32;
        max_pattern_length -> i32 = |this| this.params.int_or("maxPatternLength", 10);
        set_min_support(value: f64) => "minSupport", set_param_double;
        min_support -> f64 = |this| this.params.double_or("minSupport", 0.1);
        set_sequence_col(value: &str) => "sequenceCol", set_param_string;
        sequence_col -> &str = |this| this.params.string_or("sequenceCol", "sequence");
    }
}

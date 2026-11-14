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
impl FPGrowth {
    /// Load an `FPGrowth` saved to `path`, as PySpark's `FPGrowth.load(path)`: its uid and params.
    pub fn load(session: &SparkSession, path: &str) -> spark_connect_core::error::Result<Self> {
        let (operator, params) = read_operator(
            session,
            "org.apache.spark.ml.fpm.FPGrowth",
            OperatorType::Estimator,
            path,
        )?;
        Ok(Self { operator, params })
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

    /// Load an `FPGrowthModel` saved to `path`, as PySpark's `FPGrowthModel.load(path)`: the
    /// server caches it, as a fitted model.
    pub fn load(session: &SparkSession, path: &str) -> spark_connect_core::error::Result<Self> {
        let (cached, operator, params) =
            read_model(session, "org.apache.spark.ml.fpm.FPGrowthModel", path)?;
        Ok(Self {
            operator,
            params,
            cached,
        })
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

impl FPGrowth {
    /// Set `itemsCol`. Spark validates the value when the operation runs.
    pub fn set_items_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("itemsCol", value);
        self
    }
    /// Get `itemsCol`, using Spark's default when defined.
    pub fn items_col(&self) -> &str {
        self.params.string_or("itemsCol", "items")
    }
    /// Set `minConfidence`. Spark validates the value when the operation runs.
    pub fn set_min_confidence(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("minConfidence", value);
        self
    }
    /// Get `minConfidence`, using Spark's default when defined.
    pub fn min_confidence(&self) -> f64 {
        self.params.double_or("minConfidence", 0.8)
    }
    /// Set `minSupport`. Spark validates the value when the operation runs.
    pub fn set_min_support(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("minSupport", value);
        self
    }
    /// Get `minSupport`, using Spark's default when defined.
    pub fn min_support(&self) -> f64 {
        self.params.double_or("minSupport", 0.3)
    }
    /// Set `numPartitions`. Spark validates the value when the operation runs.
    pub fn set_num_partitions(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("numPartitions", value);
        self
    }
    /// Get `numPartitions`, using Spark's default when defined.
    pub fn num_partitions(&self) -> Option<i32> {
        self.params.long_opt("numPartitions").map(|v| v as i32)
    }
    /// Set `predictionCol`. Spark validates the value when the operation runs.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Get `predictionCol`, using Spark's default when defined.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
}

impl FPGrowthModel {
    /// Set `itemsCol`. Spark validates the value when the operation runs.
    pub fn set_items_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("itemsCol", value);
        self
    }
    /// Get `itemsCol`, using Spark's default when defined.
    pub fn items_col(&self) -> &str {
        self.params.string_or("itemsCol", "items")
    }
    /// Set `minConfidence`. Spark validates the value when the operation runs.
    pub fn set_min_confidence(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("minConfidence", value);
        self
    }
    /// Get `minConfidence`, using Spark's default when defined.
    pub fn min_confidence(&self) -> f64 {
        self.params.double_or("minConfidence", 0.8)
    }
    /// Get `minSupport`, using Spark's default when defined.
    pub fn min_support(&self) -> f64 {
        self.params.double_or("minSupport", 0.3)
    }
    /// Get `numPartitions`, using Spark's default when defined.
    pub fn num_partitions(&self) -> Option<i32> {
        self.params.long_opt("numPartitions").map(|v| v as i32)
    }
    /// Set `predictionCol`. Spark validates the value when the operation runs.
    pub fn set_prediction_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("predictionCol", value);
        self
    }
    /// Get `predictionCol`, using Spark's default when defined.
    pub fn prediction_col(&self) -> &str {
        self.params.string_or("predictionCol", "prediction")
    }
}

impl PrefixSpan {
    /// Set `maxLocalProjDBSize`. Spark validates the value when the operation runs.
    pub fn set_max_local_proj_db_size(mut self, value: i64) -> Self {
        self.params = self
            .params
            .set_param_int_narrow("maxLocalProjDBSize", value);
        self
    }
    /// Get `maxLocalProjDBSize`, using Spark's default when defined.
    pub fn max_local_proj_db_size(&self) -> i64 {
        self.params
            .long_opt("maxLocalProjDBSize")
            .unwrap_or(32000000)
    }
    /// Set `maxPatternLength`. Spark validates the value when the operation runs.
    pub fn set_max_pattern_length(mut self, value: i32) -> Self {
        self.params = self.params.set_param_int32("maxPatternLength", value);
        self
    }
    /// Get `maxPatternLength`, using Spark's default when defined.
    pub fn max_pattern_length(&self) -> i32 {
        self.params.int_or("maxPatternLength", 10)
    }
    /// Set `minSupport`. Spark validates the value when the operation runs.
    pub fn set_min_support(mut self, value: f64) -> Self {
        self.params = self.params.set_param_double("minSupport", value);
        self
    }
    /// Get `minSupport`, using Spark's default when defined.
    pub fn min_support(&self) -> f64 {
        self.params.double_or("minSupport", 0.1)
    }
    /// Set `sequenceCol`. Spark validates the value when the operation runs.
    pub fn set_sequence_col(mut self, value: &str) -> Self {
        self.params = self.params.set_param_string("sequenceCol", value);
        self
    }
    /// Get `sequenceCol`, using Spark's default when defined.
    pub fn sequence_col(&self) -> &str {
        self.params.string_or("sequenceCol", "sequence")
    }
}

// SPDX-License-Identifier: Apache-2.0
//! Standard Spark ML statistics, evaluated by the Connect server.
use super::*;
use crate::column::Column;
use crate::expression::Expression;
use crate::functions::array;
use spark_connect_core::error::Result;

fn helper(
    dataset: &DataFrame,
    method: &str,
    params: Vec<proto::expression::Literal>,
) -> Result<DataFrame> {
    use proto::fetch::method::{args::ArgsType, Args};
    let mut args = vec![Args {
        args_type: Some(ArgsType::Input(dataset.to_relation()?)),
    }];
    args.extend(params.into_iter().map(|value| Args {
        args_type: Some(ArgsType::Param(value)),
    }));
    let relation = proto::Relation {
        rel_type: Some(proto::relation::RelType::MlRelation(Box::new(
            proto::MlRelation {
                ml_type: Some(proto::ml_relation::MlType::Fetch(proto::Fetch {
                    obj_ref: Some(proto::ObjectRef {
                        id: "______ML_CONNECT_HELPER______".into(),
                    }),
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
    // Retain the input plan: it may own cached models or uploaded relations.
    Ok(DataFrame::new(
        dataset.session.clone(),
        LogicalPlan::MlTransform {
            ml_relation: relation,
            input: dataset.plan.clone(),
            owner: None,
        },
    ))
}

fn string(value: &str) -> proto::expression::Literal {
    proto::expression::Literal {
        literal_type: Some(proto::expression::literal::LiteralType::String(
            value.into(),
        )),
        ..Default::default()
    }
}

/// Pearson's independence test between categorical labels and vector features.
pub struct ChiSquareTest;
impl ChiSquareTest {
    /// Return p-values, degrees of freedom and statistics; `flatten` emits one row per feature.
    pub fn test(
        dataset: &DataFrame,
        features_col: &str,
        label_col: &str,
        flatten: bool,
    ) -> Result<DataFrame> {
        helper(
            dataset,
            "chiSquareTest",
            vec![
                string(features_col),
                string(label_col),
                proto::expression::Literal {
                    literal_type: Some(proto::expression::literal::LiteralType::Boolean(flatten)),
                    ..Default::default()
                },
            ],
        )
    }
}

/// Correlation matrix of vector features.
pub struct Correlation;
impl Correlation {
    /// Compute Pearson or Spearman correlation. Spark validates `method` and the input schema.
    pub fn corr(dataset: &DataFrame, column: &str, method: &str) -> Result<DataFrame> {
        helper(dataset, "correlation", vec![string(column), string(method)])
    }
}

/// One-sample Kolmogorov-Smirnov goodness-of-fit test.
pub struct KolmogorovSmirnovTest;
impl KolmogorovSmirnovTest {
    /// Compare a numeric column to `norm`, optionally specifying its mean and standard deviation.
    #[allow(deprecated)] // Match PySpark's explicitly typed literal array.
    pub fn test(
        dataset: &DataFrame,
        sample_col: &str,
        dist_name: &str,
        params: impl IntoIterator<Item = f64>,
    ) -> Result<DataFrame> {
        use proto::expression::literal::{Array, LiteralType};
        // The empty parameter list must still be an array<double>, as in PySpark.
        let parameters = proto::expression::Literal {
            literal_type: Some(LiteralType::Array(Array {
                element_type: Some(DataType::Double.to_proto()),
                elements: params
                    .into_iter()
                    .map(|v| proto::expression::Literal {
                        literal_type: Some(LiteralType::Double(v)),
                        ..Default::default()
                    })
                    .collect(),
            })),
            ..Default::default()
        };
        helper(
            dataset,
            "kolmogorovSmirnovTest",
            vec![string(sample_col), string(dist_name), parameters],
        )
    }
}

/// A builder of aggregate statistics for an ML vector column.
#[derive(Debug, Clone)]
pub struct SummaryBuilder {
    metrics: Vec<Column>,
}
impl SummaryBuilder {
    /// Build from literal metric expressions, preserving Spark's validation of their values.
    pub fn from_metric_columns(metrics: impl IntoIterator<Item = Column>) -> Self {
        Self {
            metrics: metrics.into_iter().collect(),
        }
    }
    /// Aggregate the requested metrics, with unit weights when `weight_col` is absent.
    pub fn summary(&self, features_col: Column, weight_col: Option<Column>) -> Column {
        Column::new(Expression::function(
            "aggregate_metrics",
            vec![
                array(self.metrics.clone()).expression().clone(),
                features_col.expression().clone(),
                weight_col
                    .unwrap_or_else(|| {
                        Column::new(Expression::literal(LiteralExpression::Double(1.0)))
                    })
                    .expression()
                    .clone(),
            ],
        ))
    }
}

/// Weighted, column-wise vector summaries. Each method returns a composable SQL aggregate.
pub struct Summarizer;
impl Summarizer {
    /// Request any of mean, sum, variance, std, count, numNonZeros, max, min, normL1, normL2.
    pub fn metrics(metrics: impl IntoIterator<Item = impl Into<String>>) -> SummaryBuilder {
        SummaryBuilder::from_metric_columns(
            metrics
                .into_iter()
                .map(|m| Column::new(Expression::literal(LiteralExpression::String(m.into())))),
        )
    }
    fn single(metric: &str, col: Column, weight_col: Option<Column>) -> Column {
        let name = format!("{}({})", metric, col.expression().render());
        Self::metrics([metric])
            .summary(col, weight_col)
            .get_field(metric)
            .alias(&name)
    }
}
macro_rules! metrics {
    ($(($name:ident, $metric:literal)),* $(,)?) => { impl Summarizer { $(
        #[doc = concat!("Compute the `", $metric, "` summary of a vector column.")]
        pub fn $name(col: Column, weight_col: Option<Column>) -> Column { Self::single($metric, col, weight_col) }
    )* } };
}
metrics!(
    (mean, "mean"),
    (sum, "sum"),
    (variance, "variance"),
    (std, "std"),
    (count, "count"),
    (num_non_zeros, "numNonZeros"),
    (max, "max"),
    (min, "min"),
    (norm_l1, "normL1"),
    (norm_l2, "normL2")
);

/// Dense or sparse ML matrix storage, with the same layout as Spark's Matrix types.
#[derive(Debug, Clone, PartialEq)]
pub enum MlMatrix {
    /// Column-major values unless `is_transposed` is true.
    Dense {
        num_rows: i32,
        num_cols: i32,
        values: Vec<f64>,
        is_transposed: bool,
    },
    /// Compressed sparse columns (compressed rows when `is_transposed` is true).
    Sparse {
        num_rows: i32,
        num_cols: i32,
        col_ptrs: Vec<i32>,
        row_indices: Vec<i32>,
        values: Vec<f64>,
        is_transposed: bool,
    },
}

/// A mutable mean/covariance pair, matching PySpark's data holder without additional validation.
#[derive(Debug, Clone, PartialEq)]
pub struct MultivariateGaussian {
    pub mean: MlVector,
    pub cov: MlMatrix,
}
impl MultivariateGaussian {
    pub fn new(mean: impl Into<MlVector>, cov: MlMatrix) -> Self {
        Self {
            mean: mean.into(),
            cov,
        }
    }
}

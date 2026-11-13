// SPDX-License-Identifier: Apache-2.0
//! Exact reference protocols for every standard Spark ML statistic.
#[path = "common/statistics.rs"]
mod statistics;
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{MlMatrix, MlVector, MultivariateGaussian};
use spark_connect::SparkSession;
use spark_connect_proto as proto;

fn clean(expr: &mut proto::Expression) {
    use proto::expression::ExprType;
    expr.common = None;
    match expr.expr_type.as_mut().unwrap() {
        ExprType::UnresolvedFunction(f) => f.arguments.iter_mut().for_each(clean),
        ExprType::Alias(a) => clean(a.expr.as_mut().unwrap()),
        ExprType::UnresolvedExtractValue(v) => {
            clean(v.child.as_mut().unwrap());
            clean(v.extraction.as_mut().unwrap());
        }
        ExprType::Literal(_) | ExprType::UnresolvedAttribute(_) => {}
        other => panic!("unexpected expression: {other:?}"),
    }
}
#[test]
fn helpers_match_official_pyspark() {
    let spark = SparkSession::builder()
        .remote("sc://localhost:1")
        .get_or_create()
        .unwrap();
    let df = spark.range(4).unwrap();
    for case in statistics::fixture()["helpers"].as_array().unwrap() {
        let result = statistics::helper(case, &df).to_relation().unwrap();
        let proto::relation::RelType::MlRelation(ml) = result.rel_type.unwrap() else {
            panic!("not ML")
        };
        let proto::ml_relation::MlType::Fetch(fetch) = ml.ml_type.unwrap() else {
            panic!("not fetch")
        };
        assert_eq!(fetch.obj_ref.unwrap().id, "______ML_CONNECT_HELPER______");
        assert_eq!(fetch.methods.len(), 1);
        let method = &fetch.methods[0];
        assert_eq!(method.method, case["method"].as_str().unwrap());
        assert_eq!(
            method.args[0].args_type,
            Some(proto::fetch::method::args::ArgsType::Input(
                df.to_relation().unwrap()
            ))
        );
        for (arg, expected) in method.args[1..]
            .iter()
            .zip(case["args_b64"].as_array().unwrap())
        {
            let bytes = STANDARD.decode(expected.as_str().unwrap()).unwrap();
            assert_eq!(
                arg.args_type,
                Some(proto::fetch::method::args::ArgsType::Param(
                    proto::expression::Literal::decode(bytes.as_slice()).unwrap()
                ))
            );
        }
    }
}
#[test]
fn all_summary_expressions_match_official_pyspark() {
    for case in statistics::fixture()["summaries"].as_array().unwrap() {
        let mut actual = statistics::summary(case).expression().to_proto();
        let bytes = STANDARD.decode(case["expr_b64"].as_str().unwrap()).unwrap();
        let mut expected = proto::Expression::decode(bytes.as_slice()).unwrap();
        clean(&mut actual);
        clean(&mut expected);
        assert_eq!(actual, expected, "{case}");
    }
}
#[test]
fn gaussian_preserves_dense_and_sparse_storage() {
    let cov = MlMatrix::Dense {
        num_rows: 2,
        num_cols: 2,
        values: vec![1.0, 0.0, 0.0, 2.0],
        is_transposed: false,
    };
    let mut g = MultivariateGaussian::new([1.0, 2.0], cov.clone());
    assert_eq!(g.mean, MlVector::Dense(vec![1.0, 2.0]));
    assert_eq!(g.cov, cov);
    g.mean = MlVector::Sparse {
        size: 2,
        indices: vec![1],
        values: vec![2.0],
    };
    g.cov = MlMatrix::Sparse {
        num_rows: 2,
        num_cols: 2,
        col_ptrs: vec![0, 1, 2],
        row_indices: vec![0, 1],
        values: vec![1.0, 2.0],
        is_transposed: true,
    };
    assert_eq!(g.clone(), g);
}

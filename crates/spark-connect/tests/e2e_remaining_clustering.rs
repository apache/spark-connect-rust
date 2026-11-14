//! Native clustering execution. Set SPARK_REMOTE to use a live Spark Connect server.
mod common;

use common::live_session;
use serde_json::Value as Json;
use spark_connect::column::{col, lit_string};
use spark_connect::expression::Expression;
use spark_connect::ml::{Model, PowerIterationClustering, Transformer, LDA};
use spark_connect::row::Value;
use spark_connect::{Column, DataFrame, SparkSession};

fn fixture() -> Json {
    serde_json::from_str(include_str!(
        "../../../tests/golden/remaining_clustering.json"
    ))
    .unwrap()
}

fn documents(s: &SparkSession, rows: &Json) -> DataFrame {
    let rows = rows
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let values = row
                .as_array()
                .unwrap()
                .iter()
                .map(|v| format!("{}D", v.as_f64().unwrap()))
                .collect::<Vec<_>>()
                .join(",");
            format!("({i}, array({values}))")
        })
        .collect::<Vec<_>>()
        .join(",");
    s.sql(&format!("SELECT * FROM VALUES {rows} AS t(id, xs)"))
        .unwrap()
        .with_column(
            "input",
            Column::new(Expression::function(
                "array_to_vector",
                vec![col("xs").expression().clone()],
            )),
        )
        .coalesce(1)
}

fn topics(df: DataFrame) -> Vec<Vec<f64>> {
    let vector = Column::new(Expression::function(
        "vector_to_array",
        vec![
            col("topics").expression().clone(),
            lit_string("float64").expression().clone(),
        ],
    ));
    df.order_by(vec![col("id").asc().expression().clone()])
        .select([vector])
        .collect()
        .unwrap()
        .into_iter()
        .map(|r| {
            let Value::List(values) = r.get(0).unwrap() else {
                panic!("topic vector")
            };
            values.iter().map(|v| v.as_f64().unwrap()).collect()
        })
        .collect()
}

#[test]
fn lda_online_and_em_match_official_and_retain_lazy_predictions() {
    let Some(s) = live_session() else { return };
    let golden = fixture();
    let train = documents(&s, &golden["documents"]);
    let held = documents(&s, &golden["held_out"]);
    for case in golden["lda"].as_array().unwrap() {
        let values = &case["values"];
        let mut estimator = LDA::new()
            .set_features_col("input")
            .set_topic_distribution_col("topics")
            .set_k(2)
            .set_max_iter(10)
            .set_seed(42)
            .set_checkpoint_interval(-1)
            .set_optimizer(values["optimizer"].as_str().unwrap())
            .set_learning_offset(16.0)
            .set_learning_decay(0.7)
            .set_subsampling_rate(1.0)
            .set_optimize_doc_concentration(false)
            .set_doc_concentration(
                values["docConcentration"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap()),
            )
            .set_topic_concentration(values["topicConcentration"].as_f64().unwrap())
            .set_keep_last_checkpoint(false);
        let mut model = estimator.fit(&train).unwrap();
        assert!(model.transform(&s.new_session().range(1).unwrap()).is_err());
        let mut clone = model.clone_box();
        let result = clone.transform(&held).unwrap();
        drop(model);
        drop(clone);
        let actual = topics(result);
        let expected = case["topics"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.len(), 2);
            for (actual, expected) in actual.iter().zip(expected.as_array().unwrap()) {
                assert!(actual.is_finite() && *actual >= 0.0);
                assert!((actual - expected.as_f64().unwrap()).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn power_iteration_clustering_matches_official_weighted_and_unweighted_graphs() {
    let Some(s) = live_session() else { return };
    let golden = fixture();
    let rows = golden["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            format!(
                "({}L, {}L, {}D)",
                r[0].as_i64().unwrap(),
                r[1].as_i64().unwrap(),
                r[2].as_f64().unwrap()
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let graph = s
        .sql(&format!(
            "SELECT * FROM VALUES {rows} AS t(source, target, affinity)"
        ))
        .unwrap()
        .repartition(1);
    for case in golden["pic"].as_array().unwrap() {
        let values = &case["values"];
        let mut algorithm = PowerIterationClustering::new()
            .set_k(2)
            .set_max_iter(40)
            .set_init_mode(values["initMode"].as_str().unwrap())
            .set_src_col("source")
            .set_dst_col("target");
        if values.get("weightCol").is_some() {
            algorithm = algorithm.set_weight_col("affinity");
        }
        let result = algorithm.assign_clusters(&graph).unwrap();
        drop(algorithm);
        let rows = result
            .order_by(vec![col("id").asc().expression().clone()])
            .collect()
            .unwrap();
        let expected = case["assignments"].as_array().unwrap();
        assert_eq!(rows.len(), expected.len());
        for i in 0..rows.len() {
            assert_eq!(rows[i].get(0).unwrap().as_i64(), expected[i][0].as_i64());
            for j in 0..rows.len() {
                assert_eq!(
                    rows[i].get(1) == rows[j].get(1),
                    expected[i][1] == expected[j][1]
                );
            }
        }
    }
}

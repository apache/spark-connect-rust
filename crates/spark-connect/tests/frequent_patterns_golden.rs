// SPDX-License-Identifier: Apache-2.0
//! Frequent-pattern parameter and helper parity with official PySpark.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use serde_json::Value as Json;
use spark_connect::ml::{Estimator, FPGrowth, PrefixSpan};
use spark_connect::SparkSession;
use spark_connect_proto as proto;

fn fixture() -> Json {
    serde_json::from_str(include_str!("../../../tests/golden/frequent_patterns.json")).unwrap()
}

#[test]
fn typed_mining_parameters_and_prefixspan_helper_match_official() {
    let session = SparkSession::builder()
        .remote("sc://localhost:1")
        .get_or_create()
        .unwrap();
    let input = session.range(1).unwrap();
    for case in fixture()["params"].as_array().unwrap() {
        let values = case["values"].as_object().unwrap();
        if values.values().any(Json::is_null) {
            continue;
        } // Python tests cover explicit None.
        let expected = proto::MlParams::decode(
            STANDARD
                .decode(case["params_b64"].as_str().unwrap())
                .unwrap()
                .as_slice(),
        )
        .unwrap();
        if case["name"] == "FPGrowth" {
            let stage = if values.is_empty() {
                FPGrowth::new()
            } else {
                FPGrowth::new()
                    .set_items_col("basket")
                    .set_prediction_col("suggested")
                    .set_min_support(0.4)
                    .set_min_confidence(0.7)
                    .set_num_partitions(2)
            };
            assert_eq!(stage.params().to_proto(), expected);
        } else {
            let stage = if values.is_empty() {
                PrefixSpan::new()
            } else {
                PrefixSpan::new()
                    .set_min_support(0.5)
                    .set_max_pattern_length(5)
                    .set_max_local_proj_db_size(values["maxLocalProjDBSize"].as_i64().unwrap())
                    .set_sequence_col("events")
            };
            assert_eq!(stage.params().to_proto(), expected);
            let result = stage
                .find_frequent_sequential_patterns(&input)
                .unwrap()
                .to_relation()
                .unwrap();
            let Some(proto::relation::RelType::MlRelation(ml)) = result.rel_type else {
                panic!("ML relation")
            };
            let Some(proto::ml_relation::MlType::Fetch(fetch)) = ml.ml_type else {
                panic!("fetch")
            };
            assert_eq!(fetch.obj_ref.unwrap().id, "______ML_CONNECT_HELPER______");
            let method = &fetch.methods[0];
            assert_eq!(method.method, "prefixSpanFindFrequentSequentialPatterns");
            assert_eq!(method.args.len(), 5);
            for (arg, expected) in method.args[1..]
                .iter()
                .zip(case["args_b64"].as_array().unwrap())
            {
                let Some(proto::fetch::method::args::ArgsType::Param(actual)) = &arg.args_type
                else {
                    panic!("param")
                };
                assert_eq!(
                    actual.encode_to_vec(),
                    STANDARD.decode(expected.as_str().unwrap()).unwrap()
                );
            }
        }
    }
}

#[test]
fn mining_defaults_and_typed_getters() {
    let fp = FPGrowth::new();
    assert_eq!(fp.items_col(), "items");
    assert_eq!(fp.prediction_col(), "prediction");
    assert_eq!(fp.min_support(), 0.3);
    assert_eq!(fp.min_confidence(), 0.8);
    assert_eq!(fp.num_partitions(), None);
    assert_eq!(fp.set_num_partitions(2).num_partitions(), Some(2));
    let seq = PrefixSpan::new();
    assert_eq!(seq.min_support(), 0.1);
    assert_eq!(seq.max_pattern_length(), 10);
    assert_eq!(seq.max_local_proj_db_size(), 32000000);
    assert_eq!(seq.sequence_col(), "sequence");
    assert_eq!(
        seq.set_max_local_proj_db_size(1_i64 << 31)
            .max_local_proj_db_size(),
        1_i64 << 31
    );
}

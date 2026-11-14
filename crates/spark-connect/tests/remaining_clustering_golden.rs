//! Request and parameter parity with the official PySpark clustering client.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use serde_json::{json, Value};
use spark_connect::ml::{Estimator, PowerIterationClustering, LDA};
use spark_connect::SparkSession;
use spark_connect_proto as proto;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../tests/golden/remaining_clustering.json"
    ))
    .unwrap()
}

fn lda(values: &Value) -> LDA {
    if values.as_object().unwrap().is_empty() {
        return LDA::new();
    }
    LDA::new()
        .set_features_col("input")
        .set_max_iter(10)
        .set_seed(values["seed"].as_i64().unwrap())
        .set_checkpoint_interval(-1)
        .set_k(2)
        .set_optimizer("online")
        .set_learning_offset(16.0)
        .set_learning_decay(0.7)
        .set_subsampling_rate(1.0)
        .set_optimize_doc_concentration(false)
        .set_doc_concentration([0.5, 0.5])
        .set_topic_concentration(0.5)
        .set_topic_distribution_col("topics")
        .set_keep_last_checkpoint(false)
}

fn pic(values: &Value) -> PowerIterationClustering {
    let mut result = if values.get("srcCol").is_some() {
        PowerIterationClustering::new()
            .set_k(3)
            .set_max_iter(40)
            .set_init_mode("degree")
            .set_src_col("source")
            .set_dst_col("target")
            .set_weight_col("affinity")
    } else {
        PowerIterationClustering::new()
    };
    if let Some(k) = values.get("k") {
        *result.params_mut() = if k.is_null() {
            result.params().clone().set_param_null("k")
        } else {
            result
                .params()
                .clone()
                .set_param_int_narrow("k", k.as_i64().unwrap())
        };
    }
    if values.get("weightCol").is_some_and(Value::is_null) {
        *result.params_mut() = result.params().clone().set_param_null("weightCol");
    }
    result
}

fn lda_getters(estimator: &LDA) -> Value {
    json!({
        "featuresCol": estimator.features_col(), "maxIter": estimator.max_iter(),
        "seed": estimator.seed(), "checkpointInterval": estimator.checkpoint_interval(),
        "k": estimator.k(), "optimizer": estimator.optimizer(),
        "learningOffset": estimator.learning_offset(), "learningDecay": estimator.learning_decay(),
        "subsamplingRate": estimator.subsampling_rate(),
        "optimizeDocConcentration": estimator.optimize_doc_concentration(),
        "docConcentration": estimator.doc_concentration(),
        "topicConcentration": estimator.topic_concentration(),
        "topicDistributionCol": estimator.topic_distribution_col(),
        "keepLastCheckpoint": estimator.keep_last_checkpoint(),
    })
}

#[test]
fn clustering_parameters_and_defaults_match_official_pyspark() {
    let golden = fixture();
    // PySpark's constructor takes seed=None, which leaves the seed to Spark's default: the
    // String.hashCode of the JVM class name, which the getter returns.
    let mut defaults = golden["defaults"]["LDA"].clone();
    defaults["seed"] = json!(1435876747);
    assert_eq!(lda_getters(&LDA::default()), defaults);
    for case in golden["params"].as_array().unwrap() {
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = proto::MlParams::decode(bytes.as_slice()).unwrap();
        if case["class"] == "LDA" {
            let estimator = lda(&case["values"]);
            assert_eq!(estimator.params().to_proto(), expected);
            let mut getters = defaults.clone();
            getters
                .as_object_mut()
                .unwrap()
                .extend(case["values"].as_object().unwrap().clone());
            assert_eq!(lda_getters(&estimator), getters);
        } else {
            assert_eq!(pic(&case["values"]).params().to_proto(), expected);
        }
    }
    let default = PowerIterationClustering::default();
    assert_eq!(
        json!({"k":default.k(), "maxIter":default.max_iter(), "initMode":default.init_mode(),
        "srcCol":default.src_col(), "dstCol":default.dst_col(), "weightCol":default.weight_col()}),
        golden["defaults"]["PowerIterationClustering"]
    );
}

#[test]
fn power_iteration_helper_relation_matches_official_pyspark() {
    let session = SparkSession::builder()
        .remote("sc://localhost:1")
        .get_or_create()
        .unwrap();
    let input = session.range(1).unwrap();
    for case in fixture()["pic_fetch"].as_array().unwrap() {
        let output = pic(&case["values"]).assign_clusters(&input).unwrap();
        assert_eq!(output.spark_session().session_id(), session.session_id());
        let Some(proto::relation::RelType::MlRelation(ml)) = output.to_relation().unwrap().rel_type
        else {
            panic!("ML relation")
        };
        let Some(proto::ml_relation::MlType::Fetch(mut actual)) = ml.ml_type else {
            panic!("helper Fetch")
        };
        use proto::fetch::method::args::ArgsType;
        assert_eq!(
            actual.methods[0].args[0].args_type,
            Some(ArgsType::Input(input.to_relation().unwrap()))
        );
        actual.methods[0].args[0].args_type = Some(ArgsType::Input(proto::Relation::default()));
        let bytes = STANDARD
            .decode(case["fetch_b64"].as_str().unwrap())
            .unwrap();
        assert_eq!(actual, proto::Fetch::decode(bytes.as_slice()).unwrap());
    }
}

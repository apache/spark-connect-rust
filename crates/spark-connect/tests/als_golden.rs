//! Check native ALS parameter encodings against official PySpark, including seed widths.
#[path = "common/als.rs"]
mod als;

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{Estimator, ALS};

#[test]
fn als_parameters_match_official_pyspark() {
    for case in als::fixture()["params"].as_array().unwrap() {
        // Explicit Python None is covered by the Python binding tests; typed Rust setters
        // take concrete values, and unset parameters retain the server defaults.
        if case["values"]
            .as_object()
            .unwrap()
            .values()
            .any(|v| v.is_null())
        {
            continue;
        }
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        assert_eq!(
            als::configured(&case["values"]).params().to_proto(),
            expected
        );
    }
}

#[test]
fn als_defaults_and_shared_block_setter() {
    let estimator = ALS::default();
    assert!(estimator.params().to_proto().params.is_empty());
    let defaults = &als::fixture()["defaults"];
    assert_eq!(estimator.rank(), defaults["rank"].as_i64().unwrap() as i32);
    assert_eq!(
        estimator.max_iter(),
        defaults["maxIter"].as_i64().unwrap() as i32
    );
    assert_eq!(
        estimator.reg_param(),
        defaults["regParam"].as_f64().unwrap()
    );
    assert_eq!(estimator.user_col(), defaults["userCol"].as_str().unwrap());
    assert_eq!(estimator.item_col(), defaults["itemCol"].as_str().unwrap());
    assert_eq!(
        estimator.rating_col(),
        defaults["ratingCol"].as_str().unwrap()
    );
    assert_eq!(
        estimator.cold_start_strategy(),
        defaults["coldStartStrategy"].as_str().unwrap()
    );
    assert_eq!(estimator.prediction_col(), "prediction");
    // The default seed is the Java hash of the JVM class name, as PySpark's `class_name_hash`.
    assert_eq!(estimator.seed(), 1_994_790_107);
    let estimator = estimator.set_num_blocks(3);
    assert_eq!(estimator.num_user_blocks(), 3);
    assert_eq!(estimator.num_item_blocks(), 3);
    assert_eq!(estimator.params().to_proto().params.len(), 2);
}

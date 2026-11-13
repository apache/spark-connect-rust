//! Shared inputs captured from official PySpark 4.3.0, used by ALS protocol and live tests.
use serde_json::Value as Json;
use spark_connect::ml::ALS;

pub fn fixture() -> Json {
    serde_json::from_str(include_str!("../../../../tests/golden/als.json")).unwrap()
}

pub fn configured(values: &Json) -> ALS {
    let mut estimator = ALS::new();
    for (name, value) in values.as_object().unwrap() {
        estimator = match name.as_str() {
            "rank" => estimator.set_rank(value.as_i64().unwrap().try_into().unwrap()),
            "maxIter" => estimator.set_max_iter(value.as_i64().unwrap().try_into().unwrap()),
            "regParam" => estimator.set_reg_param(value.as_f64().unwrap()),
            "numUserBlocks" => {
                estimator.set_num_user_blocks(value.as_i64().unwrap().try_into().unwrap())
            }
            "numItemBlocks" => {
                estimator.set_num_item_blocks(value.as_i64().unwrap().try_into().unwrap())
            }
            "implicitPrefs" => estimator.set_implicit_prefs(value.as_bool().unwrap()),
            "alpha" => estimator.set_alpha(value.as_f64().unwrap()),
            "userCol" => estimator.set_user_col(value.as_str().unwrap()),
            "itemCol" => estimator.set_item_col(value.as_str().unwrap()),
            "seed" => estimator.set_seed(value.as_i64().unwrap()),
            "ratingCol" => estimator.set_rating_col(value.as_str().unwrap()),
            "nonnegative" => estimator.set_nonnegative(value.as_bool().unwrap()),
            "checkpointInterval" => {
                estimator.set_checkpoint_interval(value.as_i64().unwrap().try_into().unwrap())
            }
            "intermediateStorageLevel" => {
                estimator.set_intermediate_storage_level(value.as_str().unwrap())
            }
            "finalStorageLevel" => estimator.set_final_storage_level(value.as_str().unwrap()),
            "coldStartStrategy" => estimator.set_cold_start_strategy(value.as_str().unwrap()),
            "blockSize" => estimator.set_block_size(value.as_i64().unwrap().try_into().unwrap()),
            "predictionCol" => estimator.set_prediction_col(value.as_str().unwrap()),
            other => panic!("unexpected ALS parameter: {other}"),
        };
    }
    estimator
}

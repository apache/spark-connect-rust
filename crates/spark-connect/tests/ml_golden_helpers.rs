// SPDX-License-Identifier: Apache-2.0
//! The shared helpers of the live ML golden tests (tests/common/ml_golden.rs) render the values of
//! the Rust client as scripts/ml_golden_common.py renders those of official PySpark, and compare
//! them with its goldens.
#[path = "common/ml_golden.rs"]
mod ml_golden;

use ml_golden::*;
use serde_json::json;
use spark_connect::ml::{MlMatrix, MlVector, Params};
use spark_connect::row::{Row, Value};

#[test]
fn values_render_as_the_golden_helper_renders_those_of_pyspark() {
    assert_eq!(num(f64::NAN), json!("NaN"));
    assert_eq!(num(f64::INFINITY), json!("Infinity"));
    assert_eq!(num(f64::NEG_INFINITY), json!("-Infinity"));
    assert_eq!(num(0.5), json!(0.5));
    assert_eq!(
        vector(&MlVector::Dense(vec![1.0, f64::NAN])),
        json!({"type": "dense", "size": 2, "values": [1.0, "NaN"]})
    );
    assert_eq!(
        vector(&MlVector::Sparse {
            size: 4,
            indices: vec![1, 3],
            values: vec![1.0, 2.0],
        }),
        json!({"type": "sparse", "size": 4, "indices": [1, 3], "values": [1.0, 2.0]})
    );
    assert_eq!(
        matrix(&MlMatrix::Dense {
            num_rows: 1,
            num_cols: 2,
            values: vec![0.0, -1.0],
            is_transposed: false,
        }),
        json!({"type": "dense", "numRows": 1, "numCols": 2, "isTransposed": false, "values": [0.0, -1.0]})
    );

    // A collected VectorUDT value, dense (type 1) and sparse (type 0), and another struct, which
    // renders as the list of its values, as a PySpark Row.
    let udt = |kind: i8, size: Value, indices: Value, values: &[f64]| {
        let fields = ["type", "size", "indices", "values"]
            .map(String::from)
            .to_vec();
        let values = Value::List(values.iter().map(|v| Value::Double(*v)).collect());
        Value::Struct(Row::new(fields, vec![Value::Byte(kind), size, indices, values]).into())
    };
    assert_eq!(
        value(&udt(1, Value::Null, Value::Null, &[1.0, 2.0])),
        json!({"type": "dense", "size": 2, "values": [1.0, 2.0]})
    );
    let indices = Value::List(vec![Value::Integer(2)]);
    assert_eq!(
        value(&udt(0, Value::Integer(3), indices, &[5.0])),
        json!({"type": "sparse", "size": 3, "indices": [2], "values": [5.0]})
    );
    let row = Row::new(
        vec!["id".into(), "score".into()],
        vec![Value::Long(7), Value::Float(0.25)],
    );
    assert_eq!(value(&Value::Struct(row.into())), json!([7, 0.25]));
}

#[test]
fn values_compare_as_the_goldens_record_them() {
    // The rows of a DataFrame in the order of their JSON, as the golden sorts them.
    assert_eq!(
        normalized(json!({"schema": "struct<x:int>", "rows": [[2], [1]]})),
        json!({"schema": "struct<x:int>", "rows": [[1], [2]]})
    );
    assert_matches(
        &json!({"a": [1.0, "NaN"], "b": 2}),
        &json!({"b": 2.0, "a": [1.0000000000001, "NaN"]}),
        "close",
    );
    let differ = std::panic::catch_unwind(|| assert_matches(&json!([1.0]), &json!([1.1]), "far"));
    assert!(differ.is_err());

    // An unclassified JVM exception reaches PySpark as "(<class>) <message>".
    assert_error(
        "IllegalStateException: zeta not in vocabulary",
        &json!({"class": "UnknownException", "message": "(java.lang.IllegalStateException) zeta not in vocabulary"}),
        "unknown",
    );
    assert_error(
        "[X] requirement failed: no",
        &json!({"class": "IllegalArgumentException", "message": "requirement failed: no"}),
        "classified",
    );

    // Params compare by their values: an integer of any width, and a NaN.
    let saved = Params::new()
        .set_param_int32("seed", 42)
        .set_param_double("missingValue", f64::NAN);
    let loaded = Params::new()
        .set_param_int("seed", 42)
        .set_param_double("missingValue", f64::NAN)
        .set_param_string("outputCol", "out");
    assert_params(&loaded, &saved, "params");
}

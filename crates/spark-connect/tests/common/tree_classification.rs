// SPDX-License-Identifier: Apache-2.0
//! The fixtures of the tree classification tests: the golden of PySpark 4.3.0 and its datasets,
//! the typed values as JSON, and the comparison of the two. A test file takes them with
//! `#[path = "common/tree_classification.rs"] mod tree_classification;`.

#![allow(dead_code)] // every test binary compiles this module and uses only a part of it

use serde_json::{json, Value};
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{MlMatrix, MlValue, MlVector};
use spark_connect::{Column, DataFrame, SparkSession};

/// A golden file of `tests/golden/`, by its name.
pub fn golden(file: &str) -> Value {
    let path = format!("{}/../../tests/golden/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).unwrap()
}

/// The expected value of `member` of the model (or summary) `key` of the section `section`
/// (`models` or `summaries`) of the classification golden.
pub fn expected<'a>(golden: &'a Value, section: &str, key: &str, member: &str) -> &'a Value {
    let value = &golden[section][key]["members"][member]["value"];
    assert!(
        !value.is_null(),
        "no golden value of {section}.{key}.{member}"
    );
    value
}

/// The DataFrame of the golden dataset `name`: its columns `id` (the row's index), `label`,
/// `weight` and `features`, the last as an ML vector.
pub fn dataset(session: &SparkSession, golden: &Value, name: &str) -> DataFrame {
    let rows = golden["datasets"][name]["rows"].as_array().unwrap();
    let values: Vec<String> = rows
        .iter()
        .enumerate()
        .map(|(id, row)| {
            let features: Vec<String> = row[2]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| format!("CAST({} AS DOUBLE)", v.as_f64().unwrap()))
                .collect();
            format!(
                "(CAST({id} AS BIGINT), CAST({} AS DOUBLE), CAST({} AS DOUBLE), array({}))",
                row[0].as_f64().unwrap(),
                row[1].as_f64().unwrap(),
                features.join(", ")
            )
        })
        .collect();
    let sql = format!(
        "SELECT id, label, weight, features FROM VALUES {} AS t(id, label, weight, features)",
        values.join(", ")
    );
    session.sql(&sql).unwrap().coalesce(1).with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("features").expression().clone()],
        )),
    )
}

/// An ML vector as the golden writes it: `{size, type, values}` dense, with `indices` sparse.
pub fn vector_json(vector: &MlVector) -> Value {
    match vector {
        MlVector::Dense(values) => json!({
            "size": values.len(),
            "type": "dense",
            "values": values,
        }),
        MlVector::Sparse {
            size,
            indices,
            values,
        } => json!({
            "indices": indices,
            "size": size,
            "type": "sparse",
            "values": values,
        }),
    }
}

/// A typed value, as the golden writes it (no matrix, which the tree classifiers do not return).
pub fn value_json(value: &MlValue) -> Value {
    match value {
        MlValue::Vector(vector) => vector_json(vector),
        MlValue::Matrix(MlMatrix::Dense { .. }) => panic!("matrix values are not compared here"),
        MlValue::Double(v) => json!(v),
        MlValue::Float(v) => json!(f64::from(*v)),
        MlValue::Int(v) | MlValue::Short(v) | MlValue::Byte(v) => json!(v),
        MlValue::Long(v) => json!(v),
        MlValue::Bool(v) => json!(v),
        MlValue::String(v) => json!(v),
        MlValue::Array(items) => Value::Array(items.iter().map(value_json).collect()),
        other => panic!("no golden form for {other:?}"),
    }
}

/// Assert that `actual` is the golden `expected`: numbers within 1e-9 (relative, or absolute
/// below 1), and everything else exactly.
pub fn assert_close(actual: &Value, expected: &Value, what: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{what}: {a} is not {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{what}: length of {a:?} vs {e:?}");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                assert_close(a, e, &format!("{what}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(a.len(), e.len(), "{what}: keys of {a:?} vs {e:?}");
            for (key, e) in e {
                let a = a.get(key).unwrap_or_else(|| panic!("{what}: no {key}"));
                assert_close(a, e, &format!("{what}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{what}"),
    }
}

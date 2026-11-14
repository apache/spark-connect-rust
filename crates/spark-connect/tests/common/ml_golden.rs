// SPDX-License-Identifier: Apache-2.0
//! The ML goldens of scripts/ml_golden_common.py, for the live tests that compare the Rust client
//! with them: the datasets of their specs, the values of the Rust client as the helper renders those
//! of official PySpark, and their comparison. A test file includes it with
//! `#[path = "common/ml_golden.rs"] mod ml_golden;`.
#![allow(dead_code)] // every test binary compiles this module and uses only a part of it

use std::collections::BTreeSet;
use std::error::Error;

use serde_json::{json, Value as Json};
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{MlMatrix, MlValue, MlVector, Params};
use spark_connect::row::Value;
use spark_connect::{Column, DataFrame, SparkSession};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn fixture(text: &str) -> Json {
    serde_json::from_str(text).unwrap()
}

/// The DataFrame of a dataset spec of `Golden.register`: its SQL; its vector columns, each of an
/// array column (`"vectors": [column, ...]` in place, or `{"column": "source", ...}`); and its
/// partitions (`"coalesce"`).
pub fn dataset(session: &SparkSession, spec: &Json) -> DataFrame {
    let mut frame = session.sql(spec["sql"].as_str().unwrap()).unwrap();
    let vectors: Vec<(String, String)> = match &spec["vectors"] {
        Json::Array(columns) => columns
            .iter()
            .map(|column| {
                let column = column.as_str().unwrap().to_string();
                (column.clone(), column)
            })
            .collect(),
        Json::Object(columns) => columns
            .iter()
            .map(|(column, source)| (column.clone(), source.as_str().unwrap().to_string()))
            .collect(),
        Json::Null => Vec::new(),
        other => panic!("the vectors of a dataset spec: {other}"),
    };
    for (column, source) in vectors {
        let source = col(&source).expression().clone();
        let vector = Column::new(Expression::function("array_to_vector", vec![source]));
        frame = frame.with_column(&column, vector);
    }
    match spec["coalesce"].as_i64() {
        Some(partitions) => frame.coalesce(partitions as i32),
        None => frame,
    }
}

// The values as scripts/ml_golden_common.py renders those of official PySpark.

pub fn num(x: f64) -> Json {
    if x.is_nan() {
        json!("NaN")
    } else if x == f64::INFINITY {
        json!("Infinity")
    } else if x == f64::NEG_INFINITY {
        json!("-Infinity")
    } else {
        json!(x)
    }
}

pub fn nums(values: &[f64]) -> Json {
    values.iter().map(|x| num(*x)).collect()
}

pub fn vector(v: &MlVector) -> Json {
    match v {
        MlVector::Dense(values) => {
            json!({"type": "dense", "size": values.len(), "values": nums(values)})
        }
        MlVector::Sparse {
            size,
            indices,
            values,
        } => json!({"type": "sparse", "size": size, "indices": indices, "values": nums(values)}),
    }
}

pub fn matrix(m: &MlMatrix) -> Json {
    match m {
        MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        } => json!({
            "type": "dense",
            "numRows": num_rows,
            "numCols": num_cols,
            "isTransposed": is_transposed,
            "values": nums(values),
        }),
        MlMatrix::Sparse {
            num_rows,
            num_cols,
            col_ptrs,
            row_indices,
            values,
            is_transposed,
        } => json!({
            "type": "sparse",
            "numRows": num_rows,
            "numCols": num_cols,
            "isTransposed": is_transposed,
            "colPtrs": col_ptrs,
            "rowIndices": row_indices,
            "values": nums(values),
        }),
    }
}

/// A collected value: a vector as `vector` renders it, a struct as the list of its values, as
/// PySpark renders a Row.
pub fn value(v: &Value) -> Json {
    match v {
        Value::Null => Json::Null,
        Value::Bool(b) => json!(b),
        Value::Byte(x) => json!(x),
        Value::Short(x) => json!(x),
        Value::Integer(x) => json!(x),
        Value::Long(x) => json!(x),
        Value::Float(x) => num(f64::from(*x)),
        Value::Double(x) => num(*x),
        Value::String(s) => json!(s),
        Value::List(items) => items.iter().map(value).collect(),
        Value::Struct(row) if row.fields() == ["type", "size", "indices", "values"] => {
            // A VectorUDT value: of type 1 dense, of only its values; of type 0 sparse.
            let field = |name| row.get_by_name(name).unwrap();
            let values = value(field("values"));
            if matches!(field("type"), Value::Byte(1)) {
                let size = values.as_array().unwrap().len();
                json!({"type": "dense", "size": size, "values": values})
            } else {
                json!({
                    "type": "sparse",
                    "size": value(field("size")),
                    "indices": value(field("indices")),
                    "values": values,
                })
            }
        }
        Value::Struct(row) => row.values().iter().map(value).collect(),
        other => panic!("no rendering of {other:?}"),
    }
}

pub fn frame(df: &DataFrame) -> Result<Json> {
    let schema = df.schema()?.simple_string();
    let rows: Vec<Json> = df
        .collect()?
        .iter()
        .map(|row| row.values().iter().map(value).collect())
        .collect();
    Ok(json!({"schema": schema, "rows": rows}))
}

/// A value to compare: the rows of a DataFrame in the order of their JSON, as the golden sorts
/// them.
pub fn normalized(mut v: Json) -> Json {
    if let Some(rows) = v.get_mut("rows").and_then(Json::as_array_mut) {
        rows.sort_by_key(|row| row.to_string());
    }
    v
}

pub fn assert_matches(actual: &Json, expected: &Json, at: &str) {
    match (actual, expected) {
        (Json::Number(a), Json::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!((a - e).abs() <= 1e-9 + 1e-9 * e.abs(), "{at}: {a} != {e}");
        }
        (Json::Array(a), Json::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{at}: {actual} != {expected}");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                assert_matches(a, e, &format!("{at}[{i}]"));
            }
        }
        (Json::Object(a), Json::Object(e)) => {
            let keys =
                |m: &serde_json::Map<String, Json>| m.keys().cloned().collect::<BTreeSet<_>>();
            assert_eq!(keys(a), keys(e), "{at}: {actual} != {expected}");
            for (key, e) in e {
                assert_matches(&a[key], e, &format!("{at}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{at}"),
    }
}

/// The error of official PySpark is in the message of the Rust one. A JVM exception that the server
/// does not classify reaches PySpark as an `UnknownException` of "(<class>) <message>".
pub fn assert_error(actual: &str, expected: &Json, at: &str) {
    let message = expected["message"].as_str().unwrap();
    let message = match message.split_once(") ") {
        Some((class, rest)) if class.starts_with("(java.") => rest,
        _ => message,
    };
    assert!(
        actual.contains(message),
        "{at}: the error {actual:?} does not carry {message:?}"
    );
}

/// The params that `loaded` has of each param that `saved` sets, by their values, as PySpark reads
/// them of their literals: an integer of any width (the server reads a seed back as a long), and
/// another value by its Debug text, which compares a NaN too.
pub fn assert_params(loaded: &Params, saved: &Params, at: &str) {
    fn canonical(value: &MlValue) -> String {
        match value {
            MlValue::Array(items) => {
                let items: Vec<String> = items.iter().map(canonical).collect();
                format!("[{}]", items.join(", "))
            }
            value => value
                .as_i64()
                .map_or_else(|| format!("{value:?}"), |i| i.to_string()),
        }
    }
    fn value(literal: &spark_connect_proto::expression::Literal) -> String {
        canonical(&MlValue::from_literal(literal).unwrap())
    }
    let loaded = loaded.to_proto().params;
    for (name, literal) in saved.to_proto().params {
        assert_eq!(
            loaded.get(&name).map(value),
            Some(value(&literal)),
            "{at}: param {name}"
        );
    }
}

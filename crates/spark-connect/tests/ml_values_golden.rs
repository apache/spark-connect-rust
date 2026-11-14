// SPDX-License-Identifier: Apache-2.0
//! The values of Spark Connect ML, byte for byte as official PySpark 4.3.0 sends and reads them
//! (`serialize_param` and `deserialize_param` of `pyspark.ml.connect.serialize`), from
//! `tests/golden/ml_values.json` of `scripts/capture_ml_values_golden.py`.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{MlMatrix, MlValue, MlVector};
use spark_connect::types::DataType;
use spark_connect_proto as proto;

fn golden() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/golden/ml_values.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn literal(golden: &serde_json::Value, name: &str) -> (Vec<u8>, proto::expression::Literal) {
    let bytes = STANDARD
        .decode(golden[name]["literal_b64"].as_str().unwrap())
        .unwrap();
    let literal = proto::expression::Literal::decode(bytes.as_slice()).unwrap();
    (bytes, literal)
}

fn doubles(values: &[f64]) -> MlValue {
    MlValue::Array(values.iter().map(|v| MlValue::Double(*v)).collect())
}

#[test]
fn ml_values_are_sent_and_read_as_official_pyspark() {
    let golden = golden();
    let cases: Vec<(&str, MlValue)> = vec![
        ("double", MlValue::Double(1.5)),
        ("int", MlValue::Int(7)),
        ("long", MlValue::Long(1 << 40)),
        ("string", MlValue::String("s".into())),
        ("bool", MlValue::Bool(true)),
        ("null", MlValue::Null),
        ("double_array", doubles(&[1.0, 2.5])),
        (
            "string_array",
            MlValue::Array(vec![
                MlValue::String("a".into()),
                MlValue::String("b".into()),
            ]),
        ),
        (
            "int_array",
            MlValue::Array(vec![MlValue::Int(1), MlValue::Int(2)]),
        ),
        (
            "nested_double_array",
            MlValue::Array(vec![doubles(&[1.0]), doubles(&[2.0, 3.0])]),
        ),
        (
            "dense_vector",
            MlValue::Vector(MlVector::Dense(vec![1.0, 2.0])),
        ),
        (
            "sparse_vector",
            MlValue::Vector(MlVector::Sparse {
                size: 3,
                indices: vec![1],
                values: vec![2.0],
            }),
        ),
        (
            "dense_matrix",
            MlValue::Matrix(MlMatrix::Dense {
                num_rows: 1,
                num_cols: 2,
                values: vec![0.0, -1.0],
                is_transposed: false,
            }),
        ),
        (
            "dense_matrix_transposed",
            MlValue::Matrix(MlMatrix::Dense {
                num_rows: 2,
                num_cols: 1,
                values: vec![3.0, 4.0],
                is_transposed: true,
            }),
        ),
        (
            "sparse_matrix",
            MlValue::Matrix(MlMatrix::Sparse {
                num_rows: 2,
                num_cols: 2,
                col_ptrs: vec![0, 1, 2],
                row_indices: vec![0, 1],
                values: vec![1.0, 2.0],
                is_transposed: false,
            }),
        ),
    ];
    for (name, value) in cases {
        let (bytes, literal) = literal(&golden, name);
        assert_eq!(
            MlValue::from_literal(&literal).unwrap(),
            value,
            "read {name}"
        );
        assert_eq!(
            value.to_literal().unwrap().encode_to_vec(),
            bytes,
            "sent {name}"
        );
    }
    let string_array = DataType::Array {
        element_type: Box::new(DataType::String {
            collation: "UTF8_BINARY".to_string(),
        }),
        contains_null: true,
    };
    let double_array = DataType::Array {
        element_type: Box::new(DataType::Double),
        contains_null: true,
    };
    for (name, value, data_type) in [
        (
            "typed_empty_string_array",
            MlValue::Array(Vec::new()),
            &string_array,
        ),
        ("typed_double_array", doubles(&[1.0, 2.0]), &double_array),
    ] {
        let (bytes, literal) = literal(&golden, name);
        assert_eq!(
            MlValue::from_literal(&literal).unwrap(),
            value,
            "read {name}"
        );
        assert_eq!(
            value.to_typed_literal(data_type).unwrap().encode_to_vec(),
            bytes,
            "sent {name}"
        );
    }
}

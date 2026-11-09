// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Arrow Rust's default map names are accepted by Spark through both local
//! relation input paths. Live checks run only when SPARK_REMOTE is set.

use arrow::array::builder::{Int64Builder, StringBuilder};
use arrow::array::{ArrayRef, Int64Array, ListArray, MapBuilder, StructArray};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;
use spark_connect::row::Value;
use spark_connect::SparkSession;
use spark_connect_proto::relation::RelType;
use std::io::Cursor;
use std::sync::Arc;

struct Session(SparkSession);

impl Session {
    fn open() -> Option<Self> {
        let remote = std::env::var("SPARK_REMOTE").ok()?;
        Some(Self(
            SparkSession::builder()
                .remote(&remote)
                .get_or_create()
                .unwrap(),
        ))
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.0.stop();
    }
}

fn input() -> RecordBatch {
    let mut builder = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
    builder.keys().append_value("alpha");
    builder.values().append_value(1);
    builder.keys().append_value("beta");
    builder.values().append_null();
    builder.append(true).unwrap();
    builder.append(false).unwrap();
    builder.append(true).unwrap();
    builder.keys().append_value("π😀");
    builder.values().append_value(3);
    builder.append(true).unwrap();
    let map: ArrayRef = Arc::new(builder.finish());
    let list: ArrayRef = Arc::new(ListArray::new(
        Arc::new(Field::new("user_items", map.data_type().clone(), true)),
        OffsetBuffer::new(vec![0, 1, 2, 3, 4].into()),
        Arc::clone(&map),
        None,
    ));
    let nested: ArrayRef = Arc::new(StructArray::new(
        vec![Field::new("maps", list.data_type().clone(), true)].into(),
        vec![list],
        None,
    ));
    let schema = Schema::new(vec![
        Field::new("id", arrow::datatypes::DataType::Int64, false),
        Field::new("mapping", map.data_type().clone(), true),
        Field::new("nested", nested.data_type().clone(), true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int64Array::from(vec![0, 1, 2, 3])), map, nested],
    )
    .unwrap()
}

fn expected_maps() -> Vec<Value> {
    vec![
        Value::Map(vec![
            (Value::String("alpha".into()), Value::Long(1)),
            (Value::String("beta".into()), Value::Null),
        ]),
        Value::Null,
        Value::Map(vec![]),
        Value::Map(vec![(Value::String("π😀".into()), Value::Long(3))]),
    ]
}

#[test]
fn default_arrow_maps_round_trip_inline_and_chunked() {
    let Some(session) = Session::open() else {
        return;
    };
    // Older servers can expose a unit-suffixed default (such as "3GB"). Use
    // explicit bytes to keep this test focused on Arrow map interoperability.
    session
        .0
        .conf()
        .set("spark.sql.session.localRelationSizeLimit", "16777216")
        .unwrap();
    for (threshold, chunked) in [("1073741824", false), ("0", true)] {
        session
            .0
            .conf()
            .set("spark.sql.session.localRelationCacheThreshold", threshold)
            .unwrap();
        session
            .0
            .conf()
            .set("spark.sql.session.localRelationChunkSizeRows", "2")
            .unwrap();
        session
            .0
            .conf()
            .set("spark.sql.session.localRelationChunkSizeBytes", "1")
            .unwrap();
        let input = input();
        let dataframe = session
            .0
            .create_dataframe_from_arrow(vec![input.slice(0, 2), input.slice(2, 2)], None)
            .unwrap();
        let relation = dataframe.to_relation();
        match (chunked, relation.rel_type) {
            (true, Some(RelType::ChunkedCachedLocalRelation(local))) => {
                assert!(
                    local.data_hashes.len() >= 2,
                    "expected multiple IPC artifacts"
                );
            }
            (false, Some(RelType::LocalRelation(_))) => {}
            _ => panic!("cache threshold did not select the intended input path"),
        }
        assert_eq!(dataframe.schema().unwrap().simple_string(),
            "struct<id:bigint,mapping:map<string,bigint>,nested:struct<maps:array<map<string,bigint>>>>");
        let mut rows = dataframe.collect().unwrap();
        rows.sort_by_key(|row| match row.get(0) {
            Some(Value::Long(id)) => *id,
            value => panic!("unexpected id: {value:?}"),
        });
        let expected = expected_maps();
        assert_eq!(rows.len(), expected.len());
        for (index, (row, expected)) in rows.iter().zip(expected).enumerate() {
            assert_eq!(row.get(0), Some(&Value::Long(index as i64)));
            assert_eq!(row.get(1), Some(&expected));
            let Some(Value::Struct(nested)) = row.get(2) else {
                panic!("expected nested struct");
            };
            assert_eq!(nested.fields(), &["maps".to_owned()]);
            assert_eq!(nested.get(0), Some(&Value::List(vec![expected])));
        }
    }
    session.0.stop().unwrap();
}

#[test]
fn zero_row_arrow_maps_keep_a_canonical_header_and_analyze_on_spark() {
    let Some(session) = Session::open() else {
        return;
    };
    let dataframe = session
        .0
        .create_dataframe_from_arrow(vec![input().slice(0, 0)], None)
        .unwrap();
    let Some(RelType::LocalRelation(local)) = dataframe.to_relation().rel_type else {
        panic!("zero rows must use an inline relation");
    };
    let reader = StreamReader::try_new(Cursor::new(local.data.unwrap()), None).unwrap();
    let arrow::datatypes::DataType::Map(entries, _) = reader.schema().field(1).data_type().clone()
    else {
        panic!("expected map schema");
    };
    let arrow::datatypes::DataType::Struct(fields) = entries.data_type() else {
        panic!("expected map entry struct");
    };
    // Successful server analysis with no rows is only smoke evidence: Spark
    // need not access a map vector. These canonical IPC-header assertions are
    // the regression proof for the zero-row path.
    assert_eq!(entries.name(), "entries");
    assert_eq!(fields[0].name(), "key");
    assert_eq!(fields[1].name(), "value");
    assert_eq!(dataframe.schema().unwrap().simple_string(),
        "struct<id:bigint,mapping:map<string,bigint>,nested:struct<maps:array<map<string,bigint>>>>");
    assert!(dataframe.collect().unwrap().is_empty());
    session.0.stop().unwrap();
}

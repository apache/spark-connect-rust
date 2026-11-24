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

//! Result schemas remain consumable as one Arrow table across output batches.

#[path = "support/arrow.rs"]
mod arrow_support;
#[path = "support/requests.rs"]
mod requests;
#[path = "support/server.rs"]
mod server;
mod support;

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, BinaryViewArray, LargeBinaryArray, LargeListArray, LargeStringArray,
    ListArray, MapArray, StringArray, StringViewArray, StructArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow::record_batch::RecordBatch;
use prost::Message;
use spark_connect_proto as proto;
use tonic::{Code, Status};
use uuid::Uuid;

use arrow_support::{assembled_batches, chunking};
use server::Server;
use support::{Batch, FixtureEngine};

const CHUNK_PREFERENCE: usize = 3 * 1024 * 1024;
const LARGE_ROWS: usize = 4096;

struct BatchCase {
    schema: SchemaRef,
    batches: Vec<Batch>,
}

fn ipc(batch: &RecordBatch) -> Vec<u8> {
    let mut data = Vec::new();
    let mut writer = StreamWriter::try_new(&mut data, batch.schema().as_ref()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    drop(writer);
    data
}

async fn execute(engine: BatchCase) -> (Vec<proto::ExecutePlanResponse>, Option<Status>) {
    let engine = FixtureEngine::default().with_results(1, engine.schema, engine.batches);
    let mut server = Server::start_with_engine(Arc::new(engine)).await;
    let mut request = requests::reattachable_request(&Uuid::new_v4().to_string(), 1);
    request
        .request_options
        .push(chunking(true, Some(CHUNK_PREFERENCE as i64)));
    let mut stream = server
        .client
        .execute_plan(request)
        .await
        .unwrap()
        .into_inner();
    let result = tokio::time::timeout(Duration::from_secs(30), async {
        let mut responses = Vec::new();
        loop {
            match stream.message().await {
                Ok(Some(response)) => {
                    if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
                        &response.response_type
                    {
                        assert!(batch.data.len() <= CHUNK_PREFERENCE);
                        assert!(response.encoded_len() <= batch.data.len() + 512);
                    } else {
                        // Schema and command envelopes retain their independent 3 MiB cap.
                        assert!(response.encoded_len() <= 3 * 1024 * 1024);
                    }
                    responses.push(response);
                }
                Ok(None) => break (responses, None),
                Err(error) => break (responses, Some(error)),
            }
        }
    })
    .await
    .expect("schema fixture execution must terminate");
    server.stop().await;
    result
}

fn output_batches(responses: &[proto::ExecutePlanResponse]) -> Vec<RecordBatch> {
    assembled_batches(responses, true)
        .into_iter()
        .map(|batch| {
            let mut reader = StreamReader::try_new(Cursor::new(batch.data), None).unwrap();
            let decoded = reader.next().unwrap().unwrap();
            assert_eq!(decoded.num_rows() as i64, batch.row_count);
            assert!(reader.next().is_none());
            decoded
        })
        .collect()
}

fn assert_schema_error(responses: &[proto::ExecutePlanResponse], status: Option<Status>) {
    let status = status.expect("a changed output schema must fail");
    assert_eq!(status.code(), Code::Internal);
    assert_eq!(
        status.message(),
        "Arrow output schema changed within one result"
    );
    assert!(!completed(responses));
}

fn completed(responses: &[proto::ExecutePlanResponse]) -> bool {
    responses.iter().any(|response| {
        matches!(
            response.response_type,
            Some(proto::execute_plan_response::ResponseType::ResultComplete(
                _
            ))
        )
    })
}

fn strings(data_type: DataType, nullable: bool) -> RecordBatch {
    let values: ArrayRef = match data_type {
        DataType::Utf8 => Arc::new(StringArray::from(vec!["value"])),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec!["value"])),
        _ => unreachable!(),
    };
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("value", data_type, nullable)])),
        vec![values],
    )
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn changed_physical_types_and_nullability_fail_before_the_second_batch() {
    let first = strings(DataType::Utf8, false);
    for second in [
        strings(DataType::LargeUtf8, false),
        strings(DataType::Utf8, true),
    ] {
        let (responses, status) = execute(BatchCase {
            schema: first.schema(),
            batches: vec![Batch::Arrow(first.clone()), Batch::Arrow(second)],
        })
        .await;
        assert_schema_error(&responses, status);
        let batches = output_batches(&responses);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].schema(), first.schema());
    }
}

fn metadata_batch(note: &str) -> RecordBatch {
    let metadata = HashMap::from([("producer.note".to_string(), note.to_string())]);
    let item =
        Arc::new(Field::new("item", DataType::LargeUtf8, true).with_metadata(metadata.clone()));
    let values = ListArray::new(
        item.clone(),
        OffsetBuffer::new(vec![0, 1].into()),
        Arc::new(LargeStringArray::from(vec!["value"])),
        None,
    );
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(
            vec![Field::new("value", DataType::List(item), true).with_metadata(metadata.clone())],
            metadata,
        )),
        vec![Arc::new(values)],
    )
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metadata_can_change_and_batch_schema_can_differ_from_the_declaration() {
    let declared = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
        false,
    )]));
    let (responses, status) = execute(BatchCase {
        schema: declared,
        batches: vec![
            Batch::Arrow(metadata_batch("first")),
            Batch::Arrow(metadata_batch("second")),
        ],
    })
    .await;
    assert!(status.is_none(), "{status:?}");
    assert!(completed(&responses));
    let batches = output_batches(&responses);
    assert_eq!(batches.len(), 2);
    for (batch, note) in batches.iter().zip(["first", "second"]) {
        let schema = batch.schema();
        assert_eq!(schema.metadata()["producer.note"], note);
        assert_eq!(schema.field(0).metadata()["producer.note"], note);
        let DataType::List(item) = schema.field(0).data_type() else {
            panic!("expected list output");
        };
        assert_eq!(item.metadata()["producer.note"], note);
        assert_eq!(item.data_type(), &DataType::LargeUtf8);
        assert!(schema.field(0).is_nullable());
        assert!(item.is_nullable());
    }
}

fn text(row: usize) -> String {
    format!("{row:04}{}", "x".repeat(1020))
}

fn views(rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8View,
            false,
        )])),
        vec![Arc::new(StringViewArray::from_iter_values(
            (0..rows).map(text),
        ))],
    )
    .unwrap()
}

fn materialized_view() -> RecordBatch {
    RecordBatch::try_new(
        strings(DataType::LargeUtf8, false).schema(),
        vec![Arc::new(LargeStringArray::from(vec![text(0)]))],
    )
    .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn materialized_views_can_mix_with_matching_arrow_and_ipc_batches() {
    let view = views(1);
    let large = strings(DataType::LargeUtf8, false);
    let materialized = materialized_view();
    let data = ipc(&large);
    for use_ipc in [false, true] {
        for view_first in [false, true] {
            let other = if use_ipc {
                Batch::Ipc(data.clone(), 1)
            } else {
                Batch::Arrow(large.clone())
            };
            let mut batches = vec![Batch::Arrow(view.clone()), other];
            let mut expected = vec![materialized.clone(), large.clone()];
            if !view_first {
                batches.reverse();
                expected.reverse();
            }
            let (responses, status) = execute(BatchCase {
                schema: view.schema(),
                batches,
            })
            .await;
            assert!(status.is_none(), "{status:?}");
            assert!(responses.last().is_some_and(|response| matches!(
                response.response_type,
                Some(proto::execute_plan_response::ResponseType::ResultComplete(
                    _
                ))
            )));
            assert_eq!(output_batches(&responses), expected);
            if use_ipc {
                let output = assembled_batches(&responses, true);
                let ipc_index = if view_first { 1 } else { 0 };
                assert_eq!(output[ipc_index].data.as_ref(), data.as_slice());
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn materialized_views_cannot_mix_with_preserved_view_ipc() {
    let view = views(1);
    let materialized = materialized_view();
    let data = ipc(&view);
    for view_first in [false, true] {
        let mut batches = vec![Batch::Arrow(view.clone()), Batch::Ipc(data.clone(), 1)];
        if !view_first {
            batches.reverse();
        }
        let (responses, status) = execute(BatchCase {
            schema: view.schema(),
            batches,
        })
        .await;
        assert_schema_error(&responses, status);
        let expected = if view_first {
            materialized.clone()
        } else {
            view.clone()
        };
        assert_eq!(output_batches(&responses), vec![expected]);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn view_ipc_remains_unchanged_for_small_and_oversized_batches() {
    for rows in [2, LARGE_ROWS] {
        let batch = views(rows);
        let data = ipc(&batch);
        assert_eq!(data.len() > CHUNK_PREFERENCE, rows == LARGE_ROWS);
        let (responses, status) = execute(BatchCase {
            schema: batch.schema(),
            batches: vec![Batch::Ipc(data.clone(), rows as i64)],
        })
        .await;
        assert!(status.is_none(), "{status:?}");
        assert!(completed(&responses));
        let output = assembled_batches(&responses, true);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].data.as_ref(), data.as_slice());
        let batches = output_batches(&responses);
        assert_eq!(batches, vec![batch]);
        assert_eq!(
            batches[0].schema().field(0).data_type(),
            &DataType::Utf8View
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mixing_small_and_oversized_view_ipc_preserves_one_output_schema() {
    let small = views(2);
    let large = views(LARGE_ROWS);
    for large_first in [false, true] {
        let expected = if large_first {
            vec![large.clone(), small.clone()]
        } else {
            vec![small.clone(), large.clone()]
        };
        let (responses, status) = execute(BatchCase {
            schema: small.schema(),
            batches: expected
                .iter()
                .map(|batch| Batch::Ipc(ipc(batch), batch.num_rows() as i64))
                .collect(),
        })
        .await;
        assert!(status.is_none(), "{status:?}");
        assert!(completed(&responses));
        assert_eq!(output_batches(&responses), expected);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arrow_view_children_are_materialized_before_byte_chunking() {
    let values: ArrayRef = Arc::new(StringViewArray::from_iter_values((0..LARGE_ROWS).map(text)));
    let offsets = OffsetBuffer::new((0..=LARGE_ROWS as i32).collect::<Vec<_>>().into());
    let item = Arc::new(Field::new("item", DataType::Utf8View, false));
    let lists = ListArray::new(item.clone(), offsets.clone(), values.clone(), None);
    let large_lists = LargeListArray::new(
        item,
        OffsetBuffer::new((0..=LARGE_ROWS as i64).collect::<Vec<_>>().into()),
        values.clone(),
        None,
    );
    let entries = StructArray::new(
        vec![
            Field::new("key", DataType::Utf8View, false),
            Field::new("value", DataType::BinaryView, false),
        ]
        .into(),
        vec![
            values,
            Arc::new(BinaryViewArray::from_iter_values(
                (0..LARGE_ROWS).map(|row| text(row).into_bytes()),
            )),
        ],
        None,
    );
    let maps = MapArray::new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        offsets,
        entries,
        None,
        false,
    );
    let columns: Vec<ArrayRef> = vec![Arc::new(lists), Arc::new(large_lists), Arc::new(maps)];
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .enumerate()
            .map(|(index, column)| {
                Field::new(format!("value{index}"), column.data_type().clone(), false)
            })
            .collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let (responses, status) = execute(BatchCase {
        schema,
        batches: vec![Batch::Arrow(batch)],
    })
    .await;
    assert!(status.is_none(), "{status:?}");
    assert!(completed(&responses));
    let batches = output_batches(&responses);
    assert_eq!(batches.len(), 1);
    assert!(
        assembled_batches(&responses, true)[0]
            .num_chunks_in_batch
            .unwrap()
            > 1
    );
    let mut offset = 0;
    for batch in batches {
        let lists = batch
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let large_lists = batch
            .column(1)
            .as_any()
            .downcast_ref::<LargeListArray>()
            .unwrap();
        let maps = batch.column(2).as_any().downcast_ref::<MapArray>().unwrap();
        for row in 0..batch.num_rows() {
            let expected = text(offset + row);
            let list = lists.value(row);
            assert_eq!(
                list.as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap()
                    .value(0),
                expected
            );
            let large_list = large_lists.value(row);
            assert_eq!(
                large_list
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap()
                    .value(0),
                expected
            );
            let entries = maps.value(row);
            assert_eq!(
                entries
                    .column(0)
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap()
                    .value(0),
                expected
            );
            assert_eq!(
                entries
                    .column(1)
                    .as_any()
                    .downcast_ref::<LargeBinaryArray>()
                    .unwrap()
                    .value(0),
                expected.as_bytes()
            );
        }
        offset += batch.num_rows();
    }
    assert_eq!(offset, LARGE_ROWS);
}

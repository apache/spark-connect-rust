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

//! Verify observable Spark Connect wire behavior independently of DataFrame conveniences.

#[path = "support/arrow.rs"]
mod arrow_support;
#[path = "support/requests.rs"]
mod requests;
#[path = "support/server.rs"]
mod server;
mod support;

use std::collections::HashSet;
use std::io::Cursor;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, BinaryViewArray, Int64Array, LargeBinaryArray, LargeListArray,
    LargeStringArray, ListArray, MapArray, StringViewArray, StructArray,
};
use arrow::buffer::OffsetBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::CompressionType;
use arrow::record_batch::RecordBatch;
use prost::Message;
use proto::execute_plan_response::ResponseType;
use spark_connect_proto as proto;

use arrow_support::{assembled_batches, chunking, encoded, DEFAULT_ARROW_CHUNK_SIZE};
use requests::{reattachable_request, request};
use server::Server;
use support::{Batch, FixtureEngine};
use tonic::Code;
use uuid::Uuid;

async fn start_server() -> Server {
    Server::start_with_engine(Arc::new(FixtureEngine::default())).await
}

#[tokio::test]
async fn inbound_requests_use_the_spark_128_mib_limit() {
    let mut server = start_server().await;
    let session = Uuid::new_v4().to_string();
    const MAX_REQUEST_BYTES: usize = 128 * 1024 * 1024;
    // Allocate the next payload only after the previous request has completed.
    for encoded_bytes in [MAX_REQUEST_BYTES, MAX_REQUEST_BYTES + 1] {
        let mut input = request(&session, 0);
        // Client metadata is ignored by execution. Account for its length
        // delimiter to test the exact wire boundary without a large result.
        input.client_type = Some(String::new());
        let fixed_bytes = input.encoded_len() - prost::length_delimiter_len(0);
        let metadata_bytes =
            encoded_bytes - fixed_bytes - prost::length_delimiter_len(encoded_bytes);
        input.client_type = Some("x".repeat(metadata_bytes));
        assert_eq!(input.encoded_len(), encoded_bytes);
        let response = server.client.execute_plan(input).await;
        if encoded_bytes == MAX_REQUEST_BYTES {
            let mut stream = response.unwrap().into_inner();
            assert!(stream.message().await.unwrap().unwrap().schema.is_some());
            while stream.message().await.unwrap().is_some() {}
        } else {
            assert_eq!(response.unwrap_err().code(), Code::OutOfRange);
        }
    }
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arrow_stream_preserves_schema_ids_offsets_and_empty_results() {
    let mut server = start_server().await;
    let session = Uuid::new_v4().to_string();
    let mut server_session = None;
    for end in [0, 25_000] {
        let request = reattachable_request(&session, end);
        let operation = request.operation_id.clone().unwrap();
        let mut stream = server
            .client
            .execute_plan(request)
            .await
            .unwrap()
            .into_inner();
        let mut response_ids = HashSet::new();
        let (mut rows, mut batches, mut schemas, mut complete) = (0, 0, 0, false);
        while let Some(response) = stream.message().await.unwrap() {
            assert!(!complete, "ResultComplete must be the last response");
            assert_eq!(response.session_id, session);
            assert_eq!(response.operation_id, operation);
            Uuid::parse_str(&response.response_id).unwrap();
            assert!(response_ids.insert(response.response_id));
            Uuid::parse_str(&response.server_side_session_id).unwrap();
            let stable = server_session.get_or_insert(response.server_side_session_id.clone());
            assert_eq!(*stable, response.server_side_session_id);
            if let Some(schema) = response.schema {
                let Some(proto::data_type::Kind::Struct(schema)) = schema.kind else {
                    panic!("struct schema");
                };
                assert_eq!(schema.fields.len(), 1);
                assert_eq!(schema.fields[0].name, "id");
                assert!(!schema.fields[0].nullable);
                assert!(matches!(
                    schema.fields[0].data_type.as_ref().unwrap().kind,
                    Some(proto::data_type::Kind::Long(_))
                ));
                schemas += 1;
            }
            match response.response_type {
                Some(ResponseType::ArrowBatch(batch)) => {
                    assert_eq!(batch.start_offset, Some(rows));
                    let reader = StreamReader::try_new(Cursor::new(batch.data), None).unwrap();
                    let mut batch_rows = 0;
                    for record in reader {
                        let record = record.unwrap();
                        let values = record
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        for index in 0..values.len() {
                            assert_eq!(values.value(index), rows + batch_rows + index as i64);
                        }
                        batch_rows += record.num_rows() as i64;
                    }
                    assert_eq!(batch_rows, batch.row_count);
                    rows += batch_rows;
                    batches += 1;
                }
                Some(ResponseType::ResultComplete(_)) => complete = true,
                None => {}
                other => panic!("unexpected result: {other:?}"),
            }
        }
        assert_eq!(rows, end);
        assert_eq!(schemas, 1);
        assert!(complete);
        assert!(batches >= 1, "empty execution must still return Arrow data");
        if end > 0 {
            // The fixture supplies multiple physical batches; all must reach the wire.
            assert!(batches > 1, "large result should stream multiple batches");
        }
    }
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observed_session_mismatch_and_released_sessions_are_rejected() {
    let mut server = start_server().await;
    let session = Uuid::new_v4().to_string();
    let mut first = server
        .client
        .execute_plan(request(&session, 0))
        .await
        .unwrap()
        .into_inner();
    let original = first
        .message()
        .await
        .unwrap()
        .unwrap()
        .server_side_session_id;
    while first.message().await.unwrap().is_some() {}
    let mut mismatch = request(&session, 1);
    mismatch.client_observed_server_side_session_id = Some(Uuid::new_v4().to_string());
    let error = server.client.execute_plan(mismatch).await.unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    let release = proto::ReleaseSessionRequest {
        session_id: session.clone(),
        user_context: requests::user("alice"),
        ..Default::default()
    };
    let released = server
        .client
        .release_session(release.clone())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(released.server_side_session_id, original);
    server.client.release_session(release).await.unwrap();
    let error = server
        .client
        .execute_plan(request(&session, 1))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    // A separate session on the same transport remains usable.
    let mut other = server
        .client
        .execute_plan(request(&Uuid::new_v4().to_string(), 0))
        .await
        .unwrap()
        .into_inner();
    while other.message().await.unwrap().is_some() {}
    server.stop().await;
}

async fn collect_result(
    server: &mut Server,
    request: proto::ExecutePlanRequest,
) -> Vec<proto::ExecutePlanResponse> {
    tokio::time::timeout(Duration::from_secs(30), async {
        let mut stream = server
            .client
            .execute_plan(request)
            .await
            .unwrap()
            .into_inner();
        let mut responses = Vec::new();
        while let Some(response) = stream.message().await.unwrap() {
            responses.push(response);
        }
        assert!(matches!(
            responses.last().unwrap().response_type,
            Some(ResponseType::ResultComplete(_))
        ));
        responses
    })
    .await
    .expect("fixture execution must terminate")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn negotiated_chunks_preserve_plain_and_compressed_ipc_bytes_and_offsets() {
    // Incompressible values keep both compression codecs above the chunk limit.
    let mut random = 1u64;
    let values = Int64Array::from_iter_values((0..450_000).map(|_| {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        random as i64
    }));
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    for compression in [
        None,
        Some(CompressionType::ZSTD),
        Some(CompressionType::LZ4_FRAME),
    ] {
        let data = encoded(&batch, compression);
        assert!(data.len() > 3 * 1024 * 1024);
        let tail = batch.slice(0, 3);
        let tail_data = encoded(&tail, compression);
        let engine = FixtureEngine::default().with_results(
            42,
            schema.clone(),
            vec![
                Batch::Ipc(data.clone(), batch.num_rows() as i64),
                Batch::Ipc(tail_data.clone(), 3),
            ],
        );
        let mut server = Server::start_with_engine(Arc::new(engine)).await;
        server.client = server
            .client
            .clone()
            .max_decoding_message_size(8 * 1024 * 1024);
        for option in [
            None,
            Some(chunking(false, Some(1024))),
            Some(chunking(true, None)),
            Some(chunking(true, Some(3 * 1024 * 1024))),
        ] {
            let negotiated = option.as_ref().is_some_and(|option| matches!(
                &option.request_option,
                Some(proto::execute_plan_request::request_option::RequestOption::ResultChunkingOptions(options))
                    if options.allow_arrow_batch_chunking
            ));
            let mut request = reattachable_request(&Uuid::new_v4().to_string(), 42);
            let limited = option.as_ref().is_some_and(|option| matches!(
                &option.request_option,
                Some(proto::execute_plan_request::request_option::RequestOption::ResultChunkingOptions(options))
                    if options.allow_arrow_batch_chunking && options.preferred_arrow_chunk_size.is_some()
            ));
            request.request_options.extend(option);
            let responses = collect_result(&mut server, request).await;
            let output = assembled_batches(&responses, negotiated);
            assert_eq!(output.len(), 2);
            assert_eq!(output[0].data.as_ref(), data.as_slice());
            assert_eq!(output[1].data.as_ref(), tail_data.as_slice());
            assert_eq!(output[0].row_count, 450_000);
            assert_eq!(output[1].start_offset, Some(450_000));
            for (result, expected) in output.iter().zip([&batch, &tail]) {
                let mut reader = StreamReader::try_new(result.data.as_ref(), None).unwrap();
                assert_eq!(reader.next().unwrap().unwrap(), *expected);
                assert!(reader.next().is_none());
            }
            let chunks = responses
                .iter()
                .filter(|response| {
                    matches!(response.response_type, Some(ResponseType::ArrowBatch(_)))
                })
                .count();
            assert_eq!(chunks, if limited { 3 } else { 2 });
        }
        server.stop().await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preferred_chunk_size_uses_the_protocol_range_and_splits_inside_a_large_row() {
    use arrow::array::StringArray;
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec![
            "x".repeat(4 * 1024 * 1024)
        ]))],
    )
    .unwrap();
    let engine = FixtureEngine::default().with_batches(42, batch.schema(), vec![batch.clone()]);
    let mut options = spark_connect_server::ServerOptions::default();
    options.max_arrow_chunk_size = 3 * 1024 * 1024;
    let mut server = Server::start_with_options(Arc::new(engine), options).await;
    for preferred in [
        Some(1024),
        Some(65_537),
        Some(3 * 1024 * 1024),
        None,
        Some(0),
        Some(-1),
        Some(1023),
        Some(3 * 1024 * 1024 + 1),
        Some(i64::MAX),
    ] {
        let mut request = reattachable_request(&Uuid::new_v4().to_string(), 42);
        request.request_options.push(chunking(true, preferred));
        let responses = collect_result(&mut server, request).await;
        let payloads: Vec<_> = responses
            .iter()
            .filter_map(|response| match &response.response_type {
                Some(ResponseType::ArrowBatch(batch)) => Some(batch.data.len()),
                _ => None,
            })
            .collect();
        assert!(payloads.len() >= 2);
        if let Some(size) = preferred.filter(|size| (1024..=3 * 1024 * 1024).contains(size)) {
            assert!(payloads.iter().all(|len| *len <= size as usize));
            assert_eq!(payloads[0], size as usize);
        } else {
            assert_eq!(
                payloads.len(),
                2,
                "invalid preferences use the server maximum"
            );
        }
        let output = assembled_batches(&responses, true);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].row_count, 1);
        let mut reader = StreamReader::try_new(output[0].data.as_ref(), None).unwrap();
        assert_eq!(reader.next().unwrap().unwrap(), batch);
        assert!(reader.next().is_none());
    }
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_status_codes_survive_prepare_and_stream_failures() {
    let engine = FixtureEngine::default()
        .reject(10, Code::Unimplemented)
        .reject(11, Code::InvalidArgument)
        .stream_error(12, Code::ResourceExhausted);
    let mut server = Server::start_with_engine(Arc::new(engine)).await;
    let session = Uuid::new_v4().to_string();
    for (selector, expected) in [(10, Code::Unimplemented), (11, Code::InvalidArgument)] {
        let status = server
            .client
            .execute_plan(reattachable_request(&session, selector))
            .await
            .unwrap_err();
        assert_eq!(status.code(), expected);
    }
    let mut stream = server
        .client
        .execute_plan(reattachable_request(&session, 12))
        .await
        .unwrap()
        .into_inner();
    assert!(stream.message().await.unwrap().unwrap().schema.is_some());
    assert_eq!(
        stream.message().await.unwrap_err().code(),
        Code::ResourceExhausted
    );
    // Backend failures must not poison the protocol session or its transport.
    let mut recovered = server
        .client
        .execute_plan(reattachable_request(&session, 1))
        .await
        .unwrap()
        .into_inner();
    let (mut rows, mut complete) = (0, false);
    while let Some(response) = recovered.message().await.unwrap() {
        match response.response_type {
            Some(ResponseType::ArrowBatch(batch)) => rows += batch.row_count,
            Some(ResponseType::ResultComplete(_)) => complete = true,
            None => {}
            other => panic!("unexpected result: {other:?}"),
        }
    }
    assert_eq!(rows, 1);
    assert!(complete);
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deeply_nested_requests_follow_the_configured_protobuf_decoder_limit() {
    let mut server = start_server().await;
    let session = Uuid::new_v4().to_string();
    let mut request = request(&session, 1);
    let Some(proto::plan::OpType::Root(mut relation)) = request.plan.take().unwrap().op_type else {
        unreachable!()
    };
    for _ in 0..60 {
        relation = proto::Relation {
            rel_type: Some(proto::relation::RelType::Filter(Box::new(proto::Filter {
                input: Some(Box::new(relation)),
                condition: Some(proto::Expression {
                    expr_type: Some(proto::expression::ExprType::Literal(
                        proto::expression::Literal {
                            literal_type: Some(proto::expression::literal::LiteralType::Boolean(
                                true,
                            )),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                }),
            }))),
            ..Default::default()
        };
    }
    request.plan = Some(proto::Plan {
        op_type: Some(proto::plan::OpType::Root(relation)),
    });
    let decoded = proto::ExecutePlanRequest::decode(request.encode_to_vec().as_slice());
    let status = server.client.execute_plan(request).await.unwrap_err();
    match decoded {
        Err(error) => {
            // The default decoder rejects this before the backend can inspect it.
            assert!(error.to_string().contains("recursion"), "{error}");
            assert_eq!(status.code(), Code::Internal);
            assert!(status.message().contains("recursion"), "{status}");
        }
        Ok(_) => {
            // Other workspace crates enable prost/no-recursion-limit. In that
            // configuration the unsupported Filter reaches the fixture backend.
            assert_eq!(status.code(), Code::Unimplemented);
            assert_eq!(status.message(), "fixture requires a range selector");
        }
    }
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn view_arrays_are_materialized_before_byte_chunking_including_struct_children() {
    use arrow::array::{BinaryArray, BinaryViewArray, StringArray, StringViewArray, StructArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    let session = Uuid::new_v4().to_string();
    let text = |id: i64| format!("{id:04}{}", "x".repeat(1020));
    let nested_text = |id: i64| format!("n{id:04}{}", "y".repeat(1019));
    let binary = |id: i64| {
        let mut bytes = id.to_le_bytes().to_vec();
        bytes.extend(std::iter::repeat_n((id % 251) as u8, 1016));
        bytes
    };
    let strings = StringViewArray::from_iter_values((0..4096).map(text));
    let binaries = BinaryViewArray::from_iter_values((0..4096).map(binary));
    let nested_strings = StringViewArray::from_iter_values((0..4096).map(nested_text));
    // These view columns exceed the chunk preference, so their encoded batch
    // spans several byte chunks.
    for size in [
        strings
            .data_buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum::<usize>(),
        binaries
            .data_buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum::<usize>(),
        nested_strings
            .data_buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum::<usize>(),
    ] {
        assert!(size > 3 * 1024 * 1024);
    }
    let nested_fields: arrow::datatypes::Fields =
        vec![Field::new("text", DataType::Utf8View, false)].into();
    let nested = StructArray::new(nested_fields.clone(), vec![Arc::new(nested_strings)], None);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8View, false),
            Field::new("binary", DataType::BinaryView, false),
            Field::new("nested", DataType::Struct(nested_fields), false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(0..4096)),
            Arc::new(strings),
            Arc::new(binaries),
            Arc::new(nested),
        ],
    )
    .unwrap();
    let engine = FixtureEngine::default().with_batches(42, batch.schema(), vec![batch]);
    let mut server = Server::start_with_engine(Arc::new(engine)).await;
    let mut request = reattachable_request(&session, 42);
    request
        .request_options
        .push(chunking(true, Some(3 * 1024 * 1024)));
    let responses = collect_result(&mut server, request).await;
    let output = assembled_batches(&responses, true);
    assert_eq!(output.len(), 1);
    assert!(output[0].num_chunks_in_batch.unwrap() > 1);
    let batch = &output[0];
    assert_eq!(batch.start_offset, Some(0));
    let records = StreamReader::try_new(Cursor::new(&batch.data), None).unwrap();
    let mut rows = 0;
    for record in records {
        let record = record.unwrap();
        let ids = record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(record.column(1).data_type(), &DataType::LargeUtf8);
        assert_eq!(record.column(2).data_type(), &DataType::LargeBinary);
        let strings = arrow::compute::cast(record.column(1), &DataType::Utf8).unwrap();
        let strings = strings.as_any().downcast_ref::<StringArray>().unwrap();
        let binaries = arrow::compute::cast(record.column(2), &DataType::Binary).unwrap();
        let binaries = binaries.as_any().downcast_ref::<BinaryArray>().unwrap();
        let nested = record
            .column(3)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        assert_eq!(nested.column(0).data_type(), &DataType::LargeUtf8);
        let nested_strings = arrow::compute::cast(nested.column(0), &DataType::Utf8).unwrap();
        let nested_strings = nested_strings
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for index in 0..record.num_rows() {
            let id = rows + index as i64;
            assert_eq!(ids.value(index), id);
            assert_eq!(strings.value(index), text(id));
            assert_eq!(binaries.value(index), binary(id));
            assert_eq!(nested_strings.value(index), nested_text(id));
        }
        rows += record.num_rows() as i64;
    }
    assert_eq!(rows, batch.row_count);
    assert_eq!(rows, 4096);
    server.stop().await;
}

fn integers(rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from_iter_values(0..rows as i64))],
    )
    .unwrap()
}

#[tokio::test]
async fn repeated_chunk_options_enable_if_any_and_use_the_first_preference() {
    let batch = integers(3000);
    let engine = FixtureEngine::default().with_batches(42, batch.schema(), vec![batch.clone()]);
    let mut server = Server::start_with_engine(Arc::new(engine)).await;
    for (options, size) in [
        (
            vec![chunking(false, Some(1024)), chunking(true, None)],
            1024,
        ),
        (
            vec![chunking(false, None), chunking(true, Some(2048))],
            2048,
        ),
        (
            vec![chunking(true, Some(1023)), chunking(true, Some(1024))],
            DEFAULT_ARROW_CHUNK_SIZE,
        ),
    ] {
        let mut request = reattachable_request(&Uuid::new_v4().to_string(), 42);
        request.request_options.extend(options);
        let responses = collect_result(&mut server, request).await;
        let payloads: Vec<_> = responses
            .iter()
            .filter_map(|response| match &response.response_type {
                Some(ResponseType::ArrowBatch(batch)) => Some(batch.data.len()),
                _ => None,
            })
            .collect();
        if size < DEFAULT_ARROW_CHUNK_SIZE {
            assert_eq!(payloads[0], size);
        } else {
            assert_eq!(payloads.len(), 1);
        }
        assert!(payloads.iter().all(|len| *len <= size));
        let output = assembled_batches(&responses, true);
        assert_eq!(output.len(), 1);
        let mut reader = StreamReader::try_new(output[0].data.as_ref(), None).unwrap();
        assert_eq!(reader.next().unwrap().unwrap(), batch);
    }
    server.stop().await;
}

async fn collect_small_slices(backing: RecordBatch) -> Vec<RecordBatch> {
    let batches = (0..20)
        .map(|index| backing.slice(index * 1000, 1000))
        .collect();
    let engine = FixtureEngine::default().with_batches(42, backing.schema(), batches);
    let mut server = Server::start_with_engine(Arc::new(engine)).await;
    let output = collect_result(
        &mut server,
        reattachable_request(&Uuid::new_v4().to_string(), 42),
    )
    .await;
    let (mut batches, mut rows) = (Vec::new(), 0);
    for message in output {
        // A small slice must not serialize the much larger retained backing allocation.
        assert!(message.encoded_len() <= 3 * 1024 * 1024);
        if let Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) =
            message.response_type
        {
            assert_eq!(batch.start_offset, Some(rows));
            assert_eq!(batch.row_count, 1000);
            let mut reader = StreamReader::try_new(&batch.data[..], None).unwrap();
            let decoded = reader.next().unwrap().unwrap();
            assert_eq!(decoded.num_rows() as i64, batch.row_count);
            batches.push(decoded);
            assert!(reader.next().is_none());
            rows += batch.row_count;
        }
    }
    assert_eq!(batches.len(), 20);
    assert_eq!(rows, 20_000);
    server.stop().await;
    batches
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slices_of_a_large_allocation_remain_one_response_per_small_batch() {
    for (index, batch) in collect_small_slices(integers(10_000_000))
        .await
        .iter()
        .enumerate()
    {
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let start = index as i64 * 1000;
        assert!(values.values().iter().copied().eq(start..start + 1000));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nested_slices_preserve_referenced_child_values() {
    // The parent needs only 20,000 rows to retain the full 80 MB child. Exercise
    // each offset width and maps sequentially, sharing the same allocation.
    let child: ArrayRef = Arc::new(Int64Array::from_iter_values(0..10_000_000));
    for kind in 0..3 {
        let item = Arc::new(Field::new("item", DataType::Int64, false));
        let column: ArrayRef = match kind {
            0 => Arc::new(ListArray::new(
                item,
                OffsetBuffer::new((0..=20_000i32).collect::<Vec<_>>().into()),
                child.clone(),
                None,
            )),
            1 => Arc::new(LargeListArray::new(
                item,
                OffsetBuffer::new((0..=20_000i64).collect::<Vec<_>>().into()),
                child.clone(),
                None,
            )),
            _ => {
                let fields = vec![
                    Field::new("key", DataType::Int64, false),
                    Field::new("value", DataType::Int64, false),
                ];
                let entries = StructArray::new(
                    fields.clone().into(),
                    vec![child.clone(), child.clone()],
                    None,
                );
                Arc::new(MapArray::new(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(fields.into()),
                        false,
                    )),
                    OffsetBuffer::new((0..=20_000i32).collect::<Vec<_>>().into()),
                    entries,
                    None,
                    false,
                ))
            }
        };
        let schema = Arc::new(Schema::new(vec![Field::new(
            "nested",
            column.data_type().clone(),
            false,
        )]));
        let backing = RecordBatch::try_new(schema, vec![column]).unwrap();
        let batches = collect_small_slices(backing).await;
        let mut row = 0i64;
        for batch in batches {
            for index in 0..batch.num_rows() {
                let value = match kind {
                    0 => batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<ListArray>()
                        .unwrap()
                        .value(index),
                    1 => batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<LargeListArray>()
                        .unwrap()
                        .value(index),
                    _ => {
                        let entry = batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<MapArray>()
                            .unwrap()
                            .value(index);
                        assert_eq!(entry.len(), 1);
                        let key = entry
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        assert_eq!(key.value(0), row);
                        entry.column(1).clone()
                    }
                };
                let value = value.as_any().downcast_ref::<Int64Array>().unwrap();
                assert_eq!(value.len(), 1);
                assert_eq!(value.value(0), row);
                row += 1;
            }
        }
        assert_eq!(row, 20_000);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn view_slices_preserve_current_values() {
    fn text(row: usize) -> String {
        format!("{row:010}{}", "y".repeat(30))
    }

    let values = StringViewArray::from_iter_values((0..1_000_000).map(text));
    let (views, buffers, nulls) = values.into_parts();
    for binary in [false, true] {
        let column: ArrayRef = if binary {
            Arc::new(BinaryViewArray::new(
                views.clone(),
                buffers.clone(),
                nulls.clone(),
            ))
        } else {
            Arc::new(StringViewArray::new(
                views.clone(),
                buffers.clone(),
                nulls.clone(),
            ))
        };
        let schema = Arc::new(Schema::new(vec![Field::new(
            "view",
            column.data_type().clone(),
            false,
        )]));
        let backing = RecordBatch::try_new(schema, vec![column]).unwrap();
        let batches = collect_small_slices(backing).await;
        let mut row = 0;
        for batch in batches {
            for index in 0..batch.num_rows() {
                let expected = text(row);
                if binary {
                    let values = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<LargeBinaryArray>()
                        .unwrap();
                    assert_eq!(values.value(index), expected.as_bytes());
                } else {
                    let values = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .unwrap();
                    assert_eq!(values.value(index), expected);
                }
                row += 1;
            }
        }
        assert_eq!(row, 20_000);
    }
}

#[cfg(target_pointer_width = "64")]
#[tokio::test]
async fn unrepresentable_engine_batches_preserve_preceding_results_and_fail_cleanly() {
    use arrow::array::NullArray;
    use arrow::record_batch::RecordBatchOptions;

    let rows = i64::MAX as usize + 1;
    for null_column in [false, true] {
        let oversized = if null_column {
            RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new("nulls", DataType::Null, true)])),
                vec![Arc::new(NullArray::new(rows))],
            )
            .unwrap()
        } else {
            RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .unwrap()
        };
        let engine = FixtureEngine::default().with_batches(
            42,
            oversized.schema(),
            vec![oversized.slice(0, 3), oversized],
        );
        let mut server = Server::start_with_engine(Arc::new(engine)).await;
        let mut stream = server
            .client
            .execute_plan(reattachable_request(&Uuid::new_v4().to_string(), 42))
            .await
            .unwrap()
            .into_inner();
        let (status, counts) = tokio::time::timeout(Duration::from_secs(15), async {
            let mut counts = Vec::new();
            loop {
                match stream.message().await {
                    Ok(Some(message)) => match message.response_type {
                        Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) => {
                            assert_eq!(batch.start_offset, Some(0));
                            let mut reader = StreamReader::try_new(&batch.data[..], None).unwrap();
                            assert_eq!(reader.next().unwrap().unwrap().num_rows(), 3);
                            assert!(reader.next().is_none());
                            counts.push(batch.row_count);
                        }
                        Some(proto::execute_plan_response::ResponseType::ResultComplete(_)) => {
                            panic!("unrepresentable output must not complete successfully");
                        }
                        _ => {}
                    },
                    Err(status) => break (status, counts),
                    Ok(None) => panic!("expected row-count overflow status"),
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(status.code(), Code::ResourceExhausted);
        assert_eq!(status.message(), "result batch row count exceeds i64");
        assert_eq!(counts, vec![3]);
        server.stop().await;
    }
}

#[tokio::test]
async fn configured_maximum_bounds_payloads_and_zero_disables_chunking() {
    assert_eq!(
        spark_connect_server::ServerOptions::default().max_arrow_chunk_size,
        DEFAULT_ARROW_CHUNK_SIZE
    );
    let batch = integers(5000);
    let bytes = encoded(&batch, None);
    for maximum in [4096, 0] {
        let engine = FixtureEngine::default().with_results(
            42,
            batch.schema(),
            vec![Batch::Ipc(bytes.clone(), 5000)],
        );
        let mut options = spark_connect_server::ServerOptions::default();
        options.max_arrow_chunk_size = maximum;
        let mut server = Server::start_with_options(Arc::new(engine), options).await;
        for preferred in [None, Some(2048), Some(4096), Some(4097)] {
            let mut request = reattachable_request(&Uuid::new_v4().to_string(), 42);
            request.request_options.push(chunking(true, preferred));
            let responses = collect_result(&mut server, request).await;
            let output = assembled_batches(&responses, maximum != 0);
            assert_eq!(output.len(), 1);
            assert_eq!(output[0].data.as_ref(), bytes.as_slice());
            let chunks: Vec<_> = responses
                .iter()
                .filter_map(|response| match &response.response_type {
                    Some(ResponseType::ArrowBatch(batch)) => {
                        Some((response.encoded_len(), batch.data.len()))
                    }
                    _ => None,
                })
                .collect();
            if maximum == 0 {
                assert_eq!(chunks.len(), 1);
            } else {
                let limit = preferred
                    .filter(|size| (1024..=maximum as i64).contains(size))
                    .unwrap_or(maximum as i64) as usize;
                assert!(chunks.iter().all(|(_, payload)| *payload <= limit));
                assert_eq!(chunks[0].1, limit);
                assert!(
                    chunks[0].0 > limit,
                    "the chunk limit excludes the protobuf envelope"
                );
            }
        }
        server.stop().await;
    }
}

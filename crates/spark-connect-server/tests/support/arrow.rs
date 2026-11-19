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

//! Wire chunk negotiation, reassembly and IPC encoding shared by protocol tests.

#![allow(dead_code)]

use std::collections::HashSet;

use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow::ipc::CompressionType;
use arrow::record_batch::RecordBatch;
use prost::Message;
use proto::execute_plan_response::ResponseType;
use spark_connect_proto as proto;

pub const DEFAULT_ARROW_CHUNK_SIZE: usize = 128 * 1024 * 1024 * 9 / 10;

pub fn chunking(allow: bool, preferred: Option<i64>) -> proto::execute_plan_request::RequestOption {
    proto::execute_plan_request::RequestOption {
        request_option: Some(
            proto::execute_plan_request::request_option::RequestOption::ResultChunkingOptions(
                proto::ResultChunkingOptions {
                    allow_arrow_batch_chunking: allow,
                    preferred_arrow_chunk_size: preferred,
                },
            ),
        ),
    }
}

// Reassemble before Arrow decoding, just as Spark clients do. Every chunk counts
// the whole batch's rows and has its own response ID; row offsets advance once.
pub fn assembled_batches(
    responses: &[proto::ExecutePlanResponse],
    negotiated: bool,
) -> Vec<proto::execute_plan_response::ArrowBatch> {
    let mut output = Vec::new();
    let mut pending: Option<proto::execute_plan_response::ArrowBatch> = None;
    let (mut offset, mut index) = (0, 0);
    let mut data = Vec::new();
    let mut response_ids = HashSet::new();
    for response in responses {
        assert!(response_ids.insert(&response.response_id));
        let Some(ResponseType::ArrowBatch(chunk)) = &response.response_type else {
            continue;
        };
        // The negotiated limit covers IPC payload bytes, matching the JVM. The
        // protobuf envelope adds identifiers, field tags and chunk metadata.
        assert!(response.encoded_len() <= chunk.data.len() + 512);
        if negotiated {
            assert!(chunk.data.len() <= DEFAULT_ARROW_CHUNK_SIZE);
        }
        assert_eq!(chunk.start_offset, Some(offset));
        if !negotiated {
            assert_eq!(chunk.chunk_index, None);
            assert_eq!(chunk.num_chunks_in_batch, None);
            offset += chunk.row_count;
            output.push(chunk.clone());
            continue;
        }
        assert_eq!(chunk.chunk_index, Some(index));
        let count = chunk.num_chunks_in_batch.unwrap();
        assert!(count > index);
        if let Some(batch) = &mut pending {
            assert_eq!(batch.row_count, chunk.row_count);
            assert_eq!(batch.num_chunks_in_batch, Some(count));
        } else {
            pending = Some(chunk.clone());
        }
        data.extend_from_slice(&chunk.data);
        index += 1;
        if index == count {
            let mut batch = pending.take().unwrap();
            batch.data = std::mem::take(&mut data).into();
            offset += batch.row_count;
            output.push(batch);
            index = 0;
        }
    }
    assert!(pending.is_none(), "complete must not truncate a batch");
    output
}

pub fn encoded(batch: &RecordBatch, compression: Option<CompressionType>) -> Vec<u8> {
    let options = IpcWriteOptions::default()
        .try_with_compression(compression)
        .unwrap();
    let mut data = Vec::new();
    {
        let mut writer =
            StreamWriter::try_new_with_options(&mut data, batch.schema().as_ref(), options)
                .unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
    }
    data
}

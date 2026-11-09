//! The joining of the chunks of the Arrow batches that the server splits, as PySpark's
//! `ExecuteAndFetchAsIterator` joins them.
//!
//! The execution asks for chunking, so the server splits a batch too large for one message into
//! chunks, sent in order, each with its index and the number of chunks of the batch. A chunk is
//! part of the Arrow stream of its batch, so the chunks are joined before the batch is decoded.

use spark_connect_core::error::{Result, SparkError};
use spark_connect_proto as proto;

/// The chunks of the batch being assembled, received so far.
#[derive(Default)]
pub(crate) struct ArrowChunkAssembler {
    pending: Option<PendingBatch>,
}

struct PendingBatch {
    row_count: i64,
    start_offset: Option<i64>,
    num_chunks: i64,
    received: i64,
    data: Vec<u8>,
}

impl ArrowChunkAssembler {
    /// Take the Arrow batch of `response` as the chunks of a batch: the batch of the response is
    /// replaced by the whole batch once its last chunk is in, and removed while chunks are still
    /// missing. Responses without an Arrow batch are left as they are, as are their other fields.
    pub(crate) fn push(&mut self, response: &mut proto::ExecutePlanResponse) -> Result<()> {
        let batch = match response.response_type.take() {
            Some(proto::execute_plan_response::ResponseType::ArrowBatch(batch)) => batch,
            other => {
                response.response_type = other;
                return Ok(());
            }
        };
        let num_chunks = match batch.num_chunks_in_batch {
            None | Some(0) => 1,
            Some(n) if n > 0 => n,
            Some(n) => {
                return Err(SparkError::connect_msg(format!(
                    "Invalid number of chunks {n} in an arrow batch."
                )))
            }
        };
        let index = batch.chunk_index.unwrap_or(0);
        let mut pending = match self.pending.take() {
            None => {
                if index != 0 {
                    return Err(SparkError::connect_msg(format!(
                        "Expected chunk index 0 of the next arrow batch but got {index}."
                    )));
                }
                if num_chunks == 1 {
                    response.response_type = Some(
                        proto::execute_plan_response::ResponseType::ArrowBatch(batch),
                    );
                    return Ok(());
                }
                PendingBatch {
                    row_count: batch.row_count,
                    start_offset: batch.start_offset,
                    num_chunks,
                    received: 0,
                    data: Vec::new(),
                }
            }
            Some(pending) => {
                if index != pending.received {
                    return Err(SparkError::connect_msg(format!(
                        "Expected chunk index {} of the arrow batch but got {index}.",
                        pending.received
                    )));
                }
                if num_chunks != pending.num_chunks || batch.row_count != pending.row_count {
                    return Err(SparkError::connect_msg(format!(
                        "Chunk {index} of an arrow batch of {} rows in {} chunks does not match \
                         the batch of {} rows in {} chunks.",
                        batch.row_count, num_chunks, pending.row_count, pending.num_chunks
                    )));
                }
                pending
            }
        };
        pending.data.extend_from_slice(&batch.data);
        pending.received += 1;
        if pending.received < pending.num_chunks {
            self.pending = Some(pending);
            return Ok(());
        }
        response.response_type = Some(proto::execute_plan_response::ResponseType::ArrowBatch(
            proto::execute_plan_response::ArrowBatch {
                row_count: pending.row_count,
                data: pending.data.into(),
                start_offset: pending.start_offset,
                chunk_index: None,
                num_chunks_in_batch: None,
            },
        ));
        Ok(())
    }

    /// Check that no batch is left incomplete, once the stream of the execution has ended.
    pub(crate) fn finish(&self) -> Result<()> {
        match &self.pending {
            None => Ok(()),
            Some(pending) => Err(SparkError::connect_msg(format!(
                "The execution ended after {} of {} chunks of an arrow batch.",
                pending.received, pending.num_chunks
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::execute_plan_response::{ArrowBatch, ResponseType};

    fn chunk(index: i64, total: i64, rows: i64, data: &[u8]) -> proto::ExecutePlanResponse {
        proto::ExecutePlanResponse {
            response_id: format!("r{index}"),
            response_type: Some(ResponseType::ArrowBatch(ArrowBatch {
                row_count: rows,
                data: data.to_vec().into(),
                start_offset: (index == 0).then_some(5),
                chunk_index: Some(index),
                num_chunks_in_batch: Some(total),
            })),
            ..Default::default()
        }
    }

    fn batch_of(response: &proto::ExecutePlanResponse) -> Option<&ArrowBatch> {
        match &response.response_type {
            Some(ResponseType::ArrowBatch(batch)) => Some(batch),
            _ => None,
        }
    }

    #[test]
    fn chunks_are_joined_in_order_into_the_batch_they_make_up() {
        let mut assembler = ArrowChunkAssembler::default();
        let mut first = chunk(0, 3, 7, b"ab");
        assembler.push(&mut first).unwrap();
        assert!(first.response_type.is_none());
        let mut second = chunk(1, 3, 7, b"cd");
        assembler.push(&mut second).unwrap();
        assert!(second.response_type.is_none());
        let mut last = chunk(2, 3, 7, b"ef");
        assembler.push(&mut last).unwrap();
        assembler.finish().unwrap();

        let batch = batch_of(&last).expect("the whole batch");
        assert_eq!(&batch.data[..], b"abcdef");
        assert_eq!(batch.row_count, 7);
        assert_eq!(batch.start_offset, Some(5));
        assert_eq!(batch.chunk_index, None);
        assert_eq!(batch.num_chunks_in_batch, None);
    }

    #[test]
    fn a_batch_that_is_not_chunked_passes_through_and_responses_without_one_are_kept() {
        let mut assembler = ArrowChunkAssembler::default();
        let mut whole = proto::ExecutePlanResponse {
            response_type: Some(ResponseType::ArrowBatch(ArrowBatch {
                row_count: 2,
                data: b"whole".to_vec().into(),
                ..Default::default()
            })),
            ..Default::default()
        };
        assembler.push(&mut whole).unwrap();
        assert_eq!(&batch_of(&whole).unwrap().data[..], b"whole");

        let mut zero = chunk(0, 0, 1, b"zero");
        assembler.push(&mut zero).unwrap();
        assert_eq!(&batch_of(&zero).unwrap().data[..], b"zero");

        let mut metrics = proto::ExecutePlanResponse {
            response_type: Some(ResponseType::ResultComplete(
                proto::execute_plan_response::ResultComplete {},
            )),
            ..Default::default()
        };
        assembler.push(&mut metrics).unwrap();
        assert!(matches!(
            metrics.response_type,
            Some(ResponseType::ResultComplete(_))
        ));
        assembler.finish().unwrap();
    }

    #[test]
    fn a_response_without_a_batch_may_come_between_the_chunks() {
        let mut assembler = ArrowChunkAssembler::default();
        let mut first = chunk(0, 2, 1, b"x");
        assembler.push(&mut first).unwrap();
        let mut progress = proto::ExecutePlanResponse {
            schema: Some(Default::default()),
            ..Default::default()
        };
        assembler.push(&mut progress).unwrap();
        assert!(progress.schema.is_some());
        let mut last = chunk(1, 2, 1, b"y");
        assembler.push(&mut last).unwrap();
        assert_eq!(&batch_of(&last).unwrap().data[..], b"xy");
    }

    #[test]
    fn a_chunk_out_of_order_is_an_error() {
        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut chunk(0, 3, 1, b"a")).unwrap();
        let error = assembler.push(&mut chunk(2, 3, 1, b"c")).unwrap_err();
        assert_eq!(
            error.message(),
            "Expected chunk index 1 of the arrow batch but got 2."
        );
    }

    #[test]
    fn a_batch_that_starts_at_a_later_chunk_is_an_error() {
        let mut assembler = ArrowChunkAssembler::default();
        let error = assembler.push(&mut chunk(1, 2, 1, b"b")).unwrap_err();
        assert_eq!(
            error.message(),
            "Expected chunk index 0 of the next arrow batch but got 1."
        );
    }

    #[test]
    fn a_whole_batch_in_the_middle_of_chunks_is_an_error() {
        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut chunk(0, 2, 1, b"a")).unwrap();
        let mut whole = chunk(0, 0, 1, b"b");
        let error = assembler.push(&mut whole).unwrap_err();
        assert_eq!(
            error.message(),
            "Expected chunk index 1 of the arrow batch but got 0."
        );
    }

    #[test]
    fn chunks_that_disagree_on_the_batch_are_an_error() {
        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut chunk(0, 3, 4, b"a")).unwrap();
        let error = assembler.push(&mut chunk(1, 2, 4, b"b")).unwrap_err();
        assert_eq!(
            error.message(),
            "Chunk 1 of an arrow batch of 4 rows in 2 chunks does not match the batch of 4 rows \
             in 3 chunks."
        );

        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut chunk(0, 2, 4, b"a")).unwrap();
        assert!(assembler.push(&mut chunk(1, 2, 5, b"b")).is_err());
    }

    #[test]
    fn a_negative_number_of_chunks_is_an_error() {
        let mut assembler = ArrowChunkAssembler::default();
        let error = assembler.push(&mut chunk(0, -1, 1, b"a")).unwrap_err();
        assert_eq!(
            error.message(),
            "Invalid number of chunks -1 in an arrow batch."
        );
    }

    #[test]
    fn a_stream_that_ends_inside_a_batch_is_an_error() {
        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut chunk(0, 3, 1, b"a")).unwrap();
        let error = assembler.finish().unwrap_err();
        assert_eq!(
            error.message(),
            "The execution ended after 1 of 3 chunks of an arrow batch."
        );
    }
}

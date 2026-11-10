//! The joining of the chunks of the Arrow batches that the server splits, as PySpark's
//! `ExecuteAndFetchAsIterator` joins them.
//!
//! The execution asks for chunking, so the server splits a batch too large for one message into
//! chunks, sent in order, each with its index and the number of chunks of the batch. A chunk is
//! part of the Arrow stream of its batch, so the chunks are joined before the batch is decoded.
//! As PySpark's, each batch must start at the row where the rows before it end, and hold the
//! rows that it counts.

use spark_connect_core::error::{Result, SparkError};
use spark_connect_proto as proto;

/// The chunks of the batch being assembled, received so far.
#[derive(Default)]
pub(crate) struct ArrowChunkAssembler {
    pending: Option<PendingBatch>,
    /// The rows of the batches of the execution so far, at which the next batch starts.
    num_records: i64,
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
                return Err(SparkError::connect_exception_msg(format!(
                    "Invalid number of chunks {n} in an arrow batch."
                )))
            }
        };
        let index = batch.chunk_index.unwrap_or(0);
        let mut pending = match self.pending.take() {
            None => {
                // As PySpark's: the offset of the next batch, then its first chunk.
                if let Some(offset) = batch.start_offset.filter(|&o| o != self.num_records) {
                    return Err(SparkError::connect_exception_msg(format!(
                        "Expected arrow batch to start at row offset {} in results, but received \
                         arrow batch starting at offset {offset}.",
                        self.num_records
                    )));
                }
                if index != 0 {
                    return Err(SparkError::connect_exception_msg(format!(
                        "Expected chunk index 0 of the next arrow batch but got {index}."
                    )));
                }
                if num_chunks == 1 {
                    self.count_rows(&batch)?;
                    response.response_type = Some(
                        proto::execute_plan_response::ResponseType::ArrowBatch(batch),
                    );
                    return Ok(());
                }
                // The chunks but the last are as large as the first, so this holds the batch. A
                // size that cannot be reserved, of a count that the server got wrong, is left to
                // grow with the chunks instead.
                let mut data = Vec::new();
                let _ = data.try_reserve_exact(
                    usize::try_from(num_chunks)
                        .unwrap_or(usize::MAX)
                        .saturating_mul(batch.data.len()),
                );
                PendingBatch {
                    row_count: batch.row_count,
                    start_offset: batch.start_offset,
                    num_chunks,
                    received: 0,
                    data,
                }
            }
            Some(pending) => {
                if index != pending.received {
                    return Err(SparkError::connect_exception_msg(format!(
                        "Expected chunk index {} of the arrow batch but got {index}.",
                        pending.received
                    )));
                }
                if num_chunks != pending.num_chunks || batch.row_count != pending.row_count {
                    return Err(SparkError::connect_exception_msg(format!(
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
        let batch = proto::execute_plan_response::ArrowBatch {
            row_count: pending.row_count,
            data: pending.data.into(),
            start_offset: pending.start_offset,
            chunk_index: None,
            num_chunks_in_batch: None,
        };
        self.count_rows(&batch)?;
        response.response_type = Some(proto::execute_plan_response::ResponseType::ArrowBatch(
            batch,
        ));
        Ok(())
    }

    /// Check the rows of the whole batch `batch` against its row count, with PySpark's error, and
    /// count them for the offset of the next batch. The rows are those of all the record batches
    /// of its Arrow stream, which the row count counts; PySpark compares the count after each
    /// record batch, the same for the one record batch of a batch that the server sends. They are
    /// read from the headers of the messages of the stream; a stream whose messages cannot be
    /// read is left to the decoder of the batch, which reports it.
    fn count_rows(&mut self, batch: &proto::execute_plan_response::ArrowBatch) -> Result<()> {
        // As PySpark's, which checks as it reads the record batches, a stream of none is not.
        let Some(record_batches) = record_batch_rows(&batch.data).filter(|b| !b.is_empty()) else {
            return Ok(());
        };
        let rows = record_batches
            .into_iter()
            .fold(0i64, |rows, record_batch| rows.saturating_add(record_batch));
        if rows != batch.row_count {
            return Err(SparkError::connect_exception_msg(format!(
                "Expected {} rows in arrow batch but got {rows}.",
                batch.row_count
            )));
        }
        self.num_records = self.num_records.saturating_add(rows);
        Ok(())
    }

    /// The next response of an execution as it is returned: `push` of a response, and `finish` at
    /// the end of the stream, where `response` is `None`.
    pub(crate) fn join(
        &mut self,
        response: Option<proto::ExecutePlanResponse>,
    ) -> Result<Option<proto::ExecutePlanResponse>> {
        let Some(mut response) = response else {
            self.finish()?;
            return Ok(None);
        };
        self.push(&mut response)?;
        Ok(Some(response))
    }

    /// Check that no batch is left incomplete, once the stream of the execution has ended.
    pub(crate) fn finish(&self) -> Result<()> {
        match &self.pending {
            None => Ok(()),
            Some(pending) => Err(SparkError::connect_exception_msg(format!(
                "The execution ended after {} of {} chunks of an arrow batch.",
                pending.received, pending.num_chunks
            ))),
        }
    }
}

/// The rows of each record batch of the Arrow IPC stream `data`, in order, from the headers of
/// its messages, without reading their buffers; `None` if `data` is not such a stream. An empty
/// `data` has no record batch, as the decoders of a batch read it.
fn record_batch_rows(data: &[u8]) -> Option<Vec<i64>> {
    let word = |at: usize| -> Option<i32> {
        Some(i32::from_le_bytes(
            data.get(at..at.checked_add(4)?)?.try_into().ok()?,
        ))
    };
    let mut rows = Vec::new();
    let mut at = 0;
    // Each message: a continuation marker (absent in the legacy format), the length of its
    // metadata, the metadata, then its body; a length of 0, or the end of the data, ends it.
    while at < data.len() {
        let mut length = word(at)?;
        at += 4;
        if length == -1 {
            length = word(at)?;
            at += 4;
        }
        if length == 0 {
            break;
        }
        let end = at.checked_add(usize::try_from(length).ok()?)?;
        let message = arrow::ipc::root_as_message(data.get(at..end)?).ok()?;
        if let Some(record_batch) = message.header_as_record_batch() {
            rows.push(record_batch.length());
        }
        at = end.checked_add(usize::try_from(message.bodyLength()).ok()?)?;
        if at > data.len() {
            return None;
        }
    }
    Some(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::execute_plan_response::{ArrowBatch, ResponseType};
    use spark_connect_core::error::SparkErrorKind;

    fn chunk(index: i64, total: i64, rows: i64, data: &[u8]) -> proto::ExecutePlanResponse {
        proto::ExecutePlanResponse {
            response_id: format!("r{index}"),
            response_type: Some(ResponseType::ArrowBatch(ArrowBatch {
                row_count: rows,
                data: data.to_vec().into(),
                // The first batch of an execution starts at its first row.
                start_offset: (index == 0).then_some(0),
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
        assert_eq!(batch.start_offset, Some(0));
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
        // A SparkConnectException of a plain message, as PySpark raises it.
        assert_eq!(error.kind, SparkErrorKind::ConnectMessage);
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

    /// The Arrow IPC stream of a record batch of each of `rows` rows, as the server sends a batch.
    fn ipc(rows: &[i64]) -> Vec<u8> {
        use arrow::array::{ArrayRef, Int64Array};
        use arrow::record_batch::RecordBatch;
        let batches: Vec<RecordBatch> = rows
            .iter()
            .map(|&n| {
                let ids: ArrayRef = std::sync::Arc::new(Int64Array::from_iter_values(0..n));
                RecordBatch::try_from_iter([("id", ids)]).unwrap()
            })
            .collect();
        let mut out = Vec::new();
        let mut writer =
            arrow::ipc::writer::StreamWriter::try_new(&mut out, &batches[0].schema()).unwrap();
        for batch in &batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
        drop(writer);
        out
    }

    /// A whole batch of `rows` rows in `data`, starting at the row `offset` of the result.
    fn whole(rows: i64, offset: Option<i64>, data: Vec<u8>) -> proto::ExecutePlanResponse {
        proto::ExecutePlanResponse {
            response_type: Some(ResponseType::ArrowBatch(ArrowBatch {
                row_count: rows,
                data: data.into(),
                start_offset: offset,
                chunk_index: None,
                num_chunks_in_batch: None,
            })),
            ..Default::default()
        }
    }

    /// As PySpark's `handle_response`: a batch starts at the row where the rows of the batches
    /// before it end, if it says where it starts.
    #[test]
    fn a_batch_that_starts_elsewhere_than_after_the_rows_before_it_is_an_error_as_pyspark() {
        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut whole(3, Some(0), ipc(&[3]))).unwrap();
        let error = assembler
            .push(&mut whole(2, Some(2), ipc(&[2])))
            .unwrap_err();
        assert_eq!(
            error.message(),
            "Expected arrow batch to start at row offset 3 in results, but received arrow batch \
             starting at offset 2."
        );
        // A batch that does not say where it starts is not checked, and its rows count.
        assembler.push(&mut whole(2, None, ipc(&[2]))).unwrap();
        assembler.push(&mut whole(1, Some(5), ipc(&[1]))).unwrap();
        // The offset of a chunked batch is that of its first chunk, checked before its index.
        let mut first = chunk(1, 2, 1, b"x");
        if let Some(ResponseType::ArrowBatch(batch)) = first.response_type.as_mut() {
            batch.start_offset = Some(0);
        }
        let error = assembler.push(&mut first).unwrap_err();
        assert!(
            error
                .message()
                .starts_with("Expected arrow batch to start at row offset 6"),
            "{error}"
        );
    }

    /// As PySpark's: the rows of the record batches of a batch, counted as each is read, are the
    /// row count of the batch, for a whole batch and for the chunks of one.
    #[test]
    fn a_batch_of_other_rows_than_it_counts_is_an_error_as_pyspark() {
        let mut assembler = ArrowChunkAssembler::default();
        let error = assembler
            .push(&mut whole(4, Some(0), ipc(&[3])))
            .unwrap_err();
        assert_eq!(error.message(), "Expected 4 rows in arrow batch but got 3.");

        let data = ipc(&[5]);
        let (head, tail) = data.split_at(data.len() / 2);
        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut chunk(0, 2, 6, head)).unwrap();
        let error = assembler.push(&mut chunk(1, 2, 6, tail)).unwrap_err();
        assert_eq!(error.message(), "Expected 6 rows in arrow batch but got 5.");

        // The count is of all the record batches of the stream, of which the server sends one.
        let mut assembler = ArrowChunkAssembler::default();
        assembler
            .push(&mut whole(3, Some(0), ipc(&[0, 2, 1])))
            .unwrap();
        let error = assembler
            .push(&mut whole(4, Some(3), ipc(&[2, 1])))
            .unwrap_err();
        assert_eq!(error.message(), "Expected 4 rows in arrow batch but got 3.");

        // Joined, the chunks of a batch of the rows it counts make it up.
        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut chunk(0, 2, 5, head)).unwrap();
        let mut last = chunk(1, 2, 5, tail);
        assembler.push(&mut last).unwrap();
        assert_eq!(&batch_of(&last).unwrap().data[..], &data[..]);
        assembler.push(&mut whole(0, Some(5), ipc(&[0]))).unwrap();
    }

    #[test]
    fn the_data_of_the_chunks_of_a_batch_is_reserved_with_the_first_chunk() {
        let mut assembler = ArrowChunkAssembler::default();
        assembler.push(&mut chunk(0, 3, 1, &[7; 100])).unwrap();
        let pending = assembler.pending.as_ref().expect("a pending batch");
        assert!(
            pending.data.capacity() >= 300,
            "{}",
            pending.data.capacity()
        );
        // A number of chunks that cannot be reserved leaves the data to grow with them.
        let mut assembler = ArrowChunkAssembler::default();
        assembler
            .push(&mut chunk(0, i64::MAX, 1, &[7; 100]))
            .unwrap();
        assert_eq!(assembler.pending.as_ref().unwrap().data.len(), 100);
    }
}

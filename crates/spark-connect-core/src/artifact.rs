//! Artifact management for Spark Connect.
//!
//! Mirrors `pyspark.sql.connect.client.artifact.ArtifactManager`. Handles building
//! `AddArtifactsRequest` streams (batched for small artifacts, chunked for large ones),
//! computes CRC32 per artifact, and supports the artifact_status flow for checking
//! cached artifacts.

#![allow(clippy::result_large_err)] // SparkError is large; changing the type would break the API

use bytes::Bytes;
use crc32fast::Hasher;
use std::io::{Cursor, Read};

use spark_connect_proto::{add_artifacts_request, AddArtifactsRequest};

use crate::error::Result;
use crate::error::SparkError;

/// Using the midpoint recommendation of 32KiB for chunk size as specified in
/// https://github.com/grpc/grpc.github.io/issues/371.
pub const CHUNK_SIZE: usize = 32 * 1024;

/// Cache uploads amortize message overhead while staying below the default
/// 4 MiB gRPC receive limit. Other artifact uploads retain their existing size.
pub(crate) const CACHE_CHUNK_SIZE: usize = 1024 * 1024;

/// Local artifact data source (file or in-memory).
pub trait ArtifactData: Send + Sync {
    /// Share an in-memory payload without reading or copying it.
    fn bytes(&self) -> Option<Bytes> {
        None
    }

    /// Get the size of this artifact in bytes.
    fn size(&self) -> Result<u64>;

    /// Open a read stream for this artifact.
    fn stream(&self) -> Result<Box<dyn Read + Send>>;
}

/// A local file artifact.
pub struct FileArtifact {
    path: std::path::PathBuf,
}

impl FileArtifact {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl ArtifactData for FileArtifact {
    fn size(&self) -> Result<u64> {
        std::fs::metadata(&self.path)
            .map(|m| m.len())
            .map_err(|e| SparkError::connect_msg(format!("Failed to get artifact size: {}", e)))
    }

    fn stream(&self) -> Result<Box<dyn Read + Send>> {
        std::fs::File::open(&self.path)
            .map(|f| Box::new(f) as Box<dyn Read + Send>)
            .map_err(|e| SparkError::connect_msg(format!("Failed to open artifact: {}", e)))
    }
}

/// An in-memory artifact.
pub struct InMemoryArtifact {
    data: Bytes,
}

impl InMemoryArtifact {
    pub fn new(data: impl Into<Vec<u8>>) -> Self {
        Self {
            data: Bytes::from(data.into()),
        }
    }
}

impl ArtifactData for InMemoryArtifact {
    fn bytes(&self) -> Option<Bytes> {
        Some(self.data.clone())
    }

    fn size(&self) -> Result<u64> {
        Ok(self.data.len() as u64)
    }

    fn stream(&self) -> Result<Box<dyn Read + Send>> {
        Ok(Box::new(Cursor::new(self.data.clone())))
    }
}

/// Compute CRC32 checksum of a byte slice.
pub fn compute_crc32(data: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize()
}

/// An artifact with name and data source.
pub struct NamedArtifact {
    pub name: String,
    pub data: Box<dyn ArtifactData>,
}

impl NamedArtifact {
    pub fn new(name: String, data: Box<dyn ArtifactData>) -> Self {
        Self { name, data }
    }
}

/// Build a stream of AddArtifactsRequest messages for artifacts.
/// Small artifacts (<= CHUNK_SIZE) are batched together.
/// Large artifacts (> CHUNK_SIZE) are sent as chunked requests.
pub fn build_artifact_request_stream(
    session_id: String,
    artifacts: Vec<NamedArtifact>,
) -> Result<Vec<AddArtifactsRequest>> {
    build_artifact_request_stream_with_chunk_size(session_id, artifacts, CHUNK_SIZE)
}

/// Cache payloads can use larger messages to amortize framing overhead while
/// keeping each message well below the server's default 4 MiB receive limit.
pub(crate) fn build_artifact_request_stream_with_chunk_size(
    session_id: String,
    artifacts: Vec<NamedArtifact>,
    chunk_size: usize,
) -> Result<Vec<AddArtifactsRequest>> {
    let mut requests = Vec::new();
    let mut batch_artifacts = Vec::new();
    let mut batch_size = 0usize;

    for artifact in artifacts {
        let size = artifact.data.size()? as usize;

        if size > chunk_size {
            // Flush batch if non-empty
            if !batch_artifacts.is_empty() {
                let mut request = AddArtifactsRequest::default();
                request.session_id = session_id.clone();
                request.payload = Some(add_artifacts_request::Payload::Batch(
                    add_artifacts_request::Batch {
                        artifacts: batch_artifacts,
                    },
                ));
                requests.push(request);
                batch_artifacts = Vec::new();
                batch_size = 0;
            }

            // Send chunked artifact
            requests.extend(build_chunked_artifact_requests(
                session_id.clone(),
                artifact,
                chunk_size,
            )?);
        } else {
            // Add to batch
            // Include names and protobuf framing so many tiny cache blobs cannot
            // turn a small payload batch into an oversized gRPC message.
            let framed_size = size + artifact.name.len() + 32;
            if batch_size + framed_size > chunk_size && !batch_artifacts.is_empty() {
                // Flush current batch
                let mut request = AddArtifactsRequest::default();
                request.session_id = session_id.clone();
                request.payload = Some(add_artifacts_request::Payload::Batch(
                    add_artifacts_request::Batch {
                        artifacts: batch_artifacts,
                    },
                ));
                requests.push(request);
                batch_artifacts = Vec::new();
                batch_size = 0;
            }

            // Read artifact data
            let data = match artifact.data.bytes() {
                Some(data) => data,
                None => {
                    let mut stream = artifact.data.stream()?;
                    let mut data = Vec::new();
                    stream.read_to_end(&mut data).map_err(|e| {
                        SparkError::connect_msg(format!("Failed to read artifact: {}", e))
                    })?;
                    Bytes::from(data)
                }
            };
            let crc = compute_crc32(&data) as i64;

            batch_artifacts.push(add_artifacts_request::SingleChunkArtifact {
                name: artifact.name,
                data: Some(add_artifacts_request::ArtifactChunk { data, crc }),
            });
            batch_size += framed_size;
        }
    }

    // Flush remaining batch
    if !batch_artifacts.is_empty() {
        let mut request = AddArtifactsRequest::default();
        request.session_id = session_id.clone();
        request.payload = Some(add_artifacts_request::Payload::Batch(
            add_artifacts_request::Batch {
                artifacts: batch_artifacts,
            },
        ));
        requests.push(request);
    }

    Ok(requests)
}

/// Build requests for a large chunked artifact.
fn build_chunked_artifact_requests(
    session_id: String,
    artifact: NamedArtifact,
    chunk_size: usize,
) -> Result<Vec<AddArtifactsRequest>> {
    let mut requests = Vec::new();
    let total_size = artifact.data.size()? as i64;
    let num_chunks = (total_size as usize).div_ceil(chunk_size);

    let shared = artifact.data.bytes();
    let mut stream = if shared.is_none() {
        Some(artifact.data.stream()?)
    } else {
        None
    };
    let mut first_chunk = true;

    let mut buf = if shared.is_none() {
        vec![0u8; chunk_size]
    } else {
        Vec::new()
    };
    let mut offset = 0;
    loop {
        let chunk_data = if let Some(data) = &shared {
            let end = (offset + chunk_size).min(data.len());
            let chunk = data.slice(offset..end);
            offset = end;
            chunk
        } else {
            let n = stream.as_mut().unwrap().read(&mut buf).map_err(|e| {
                SparkError::connect_msg(format!("Failed to read artifact chunk: {}", e))
            })?;
            Bytes::copy_from_slice(&buf[..n])
        };
        if chunk_data.is_empty() {
            break;
        }
        let crc = compute_crc32(&chunk_data) as i64;

        if first_chunk {
            // Begin chunked artifact with metadata and initial chunk
            let mut request = AddArtifactsRequest::default();
            request.session_id = session_id.clone();
            request.payload = Some(add_artifacts_request::Payload::BeginChunk(
                add_artifacts_request::BeginChunkedArtifact {
                    name: artifact.name.clone(),
                    total_bytes: total_size,
                    num_chunks: num_chunks as i64,
                    initial_chunk: Some(add_artifacts_request::ArtifactChunk {
                        data: chunk_data,
                        crc,
                    }),
                },
            ));
            requests.push(request);
            first_chunk = false;
        } else {
            // Subsequent chunk
            let mut request = AddArtifactsRequest::default();
            request.session_id = session_id.clone();
            request.payload = Some(add_artifacts_request::Payload::Chunk(
                add_artifacts_request::ArtifactChunk {
                    data: chunk_data,
                    crc,
                },
            ));
            requests.push(request);
        }
    }

    Ok(requests)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_crc32() {
        // Test with known values
        let data = b"hello world";
        let crc = compute_crc32(data);
        // CRC32 of "hello world" should be deterministic
        assert_ne!(crc, 0);
    }

    #[test]
    fn test_crc32_consistency() {
        let data = b"test data";
        let crc1 = compute_crc32(data);
        let crc2 = compute_crc32(data);
        assert_eq!(crc1, crc2);
    }

    #[test]
    fn test_in_memory_artifact() {
        let artifact = InMemoryArtifact::new(b"hello".to_vec());
        assert_eq!(artifact.size().unwrap(), 5);

        let mut stream = artifact.stream().unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"hello");
    }

    #[test]
    fn test_file_artifact() {
        // Create a temporary file
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, b"file content").unwrap();

        let artifact = FileArtifact::new(&file_path);
        assert_eq!(artifact.size().unwrap(), 12);

        let mut stream = artifact.stream().unwrap();
        let mut buf = Vec::new();
        stream.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, b"file content");
    }

    #[test]
    fn test_chunk_batching_decision() {
        // Small artifact should be batched, large should be chunked
        let small_size = CHUNK_SIZE / 2;
        let large_size = CHUNK_SIZE * 2;

        assert!(small_size <= CHUNK_SIZE);
        assert!(large_size > CHUNK_SIZE);
    }

    #[test]
    fn test_build_small_artifact_requests() {
        let session_id = "test-session-123".to_string();
        let artifact = NamedArtifact::new(
            "pyfiles/test.py".to_string(),
            Box::new(InMemoryArtifact::new(b"print('hello')".to_vec())),
        );

        let requests = build_artifact_request_stream(session_id.clone(), vec![artifact]).unwrap();

        // Small artifact should produce one batch request
        assert_eq!(requests.len(), 1);
        assert!(requests[0].payload.is_some());
    }

    #[test]
    fn test_build_large_artifact_requests() {
        let session_id = "test-session-456".to_string();
        let large_data = vec![0u8; CHUNK_SIZE * 2 + 1000];
        let artifact = NamedArtifact::new(
            "jars/large.jar".to_string(),
            Box::new(InMemoryArtifact::new(large_data)),
        );

        let requests = build_artifact_request_stream(session_id.clone(), vec![artifact]).unwrap();

        // Large artifact (> CHUNK_SIZE) should produce multiple requests:
        // 1 begin_chunk + N chunk requests
        assert!(requests.len() > 1);
    }

    #[test]
    fn test_build_multiple_small_artifacts() {
        let session_id = "test-session-789".to_string();
        let artifacts = vec![
            NamedArtifact::new(
                "pyfiles/a.py".to_string(),
                Box::new(InMemoryArtifact::new(b"# file a".to_vec())),
            ),
            NamedArtifact::new(
                "pyfiles/b.py".to_string(),
                Box::new(InMemoryArtifact::new(b"# file b".to_vec())),
            ),
        ];

        let requests = build_artifact_request_stream(session_id.clone(), artifacts).unwrap();

        // Multiple small artifacts should be batched into one request
        assert_eq!(requests.len(), 1);
    }

    #[test]
    fn shared_payload_matches_file_framing_and_survives_source_drop() {
        for chunk_size in [CHUNK_SIZE, CACHE_CHUNK_SIZE] {
            for size in [0, chunk_size, chunk_size + 1, chunk_size * 3 + 17] {
                let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("payload");
                std::fs::write(&path, &data).unwrap();
                let memory = InMemoryArtifact::new(data);
                let shared = memory.bytes().unwrap();
                let requests = build_artifact_request_stream_with_chunk_size(
                    "session".into(),
                    vec![NamedArtifact::new("cache/hash".into(), Box::new(memory))],
                    chunk_size,
                )
                .unwrap();
                let file_requests = build_artifact_request_stream_with_chunk_size(
                    "session".into(),
                    vec![NamedArtifact::new(
                        "cache/hash".into(),
                        Box::new(FileArtifact::new(path)),
                    )],
                    chunk_size,
                )
                .unwrap();
                assert_eq!(requests, file_requests);
                if size > chunk_size {
                    let Some(add_artifacts_request::Payload::BeginChunk(first)) =
                        &requests[0].payload
                    else {
                        panic!("expected chunked artifact");
                    };
                    assert_eq!(
                        first.initial_chunk.as_ref().unwrap().data.as_ptr(),
                        shared.as_ptr()
                    );
                }
            }
        }
    }

    #[test]
    fn small_artifact_batches_include_name_and_framing_overhead() {
        use prost::Message;

        let artifacts = (0..100)
            .map(|i| {
                NamedArtifact::new(
                    format!("cache/{i:064x}"),
                    Box::new(InMemoryArtifact::new(vec![i as u8; 2])),
                )
            })
            .collect();
        let requests =
            build_artifact_request_stream_with_chunk_size("session".into(), artifacts, 1024)
                .unwrap();
        assert!(requests.len() > 1);
        for request in requests {
            // Allow the session/enclosing-message fields above the batch budget.
            assert!(request.encoded_len() <= 1024 + 32);
        }
    }
}

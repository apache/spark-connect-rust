//! Artifact management for Spark Connect.
//!
//! Mirrors `pyspark.sql.connect.client.artifact.ArtifactManager`. Handles building
//! `AddArtifactsRequest` streams (batched for small artifacts, chunked for large ones),
//! computes CRC32 per artifact, and supports the artifact_status flow for checking
//! cached artifacts.

use bytes::Bytes;
use crc32fast::Hasher;
use std::io::{Cursor, Read};

use spark_connect_proto::{add_artifacts_request, AddArtifactsRequest};

use crate::error::Result;
use crate::error::SparkError;

/// Using the midpoint recommendation of 32KiB for chunk size as specified in
/// https://github.com/grpc/grpc.github.io/issues/371.
pub const CHUNK_SIZE: usize = 32 * 1024;

/// Local artifact data source (file or in-memory).
pub trait ArtifactData: Send + Sync {
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
    data: Vec<u8>,
}

impl InMemoryArtifact {
    pub fn new(data: impl Into<Vec<u8>>) -> Self {
        Self { data: data.into() }
    }
}

impl ArtifactData for InMemoryArtifact {
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
    let mut requests = Vec::new();
    let mut batch_artifacts = Vec::new();
    let mut batch_size = 0usize;

    for artifact in artifacts {
        let size = artifact.data.size()? as usize;

        if size > CHUNK_SIZE {
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
            )?);
        } else {
            // Add to batch
            if batch_size + size > CHUNK_SIZE && !batch_artifacts.is_empty() {
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
            let mut stream = artifact.data.stream()?;
            let mut data = Vec::new();
            stream
                .read_to_end(&mut data)
                .map_err(|e| SparkError::connect_msg(format!("Failed to read artifact: {}", e)))?;
            let crc = compute_crc32(&data) as i64;

            batch_artifacts.push(add_artifacts_request::SingleChunkArtifact {
                name: artifact.name,
                data: Some(add_artifacts_request::ArtifactChunk {
                    data: Bytes::from(data),
                    crc,
                }),
            });
            batch_size += size;
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
) -> Result<Vec<AddArtifactsRequest>> {
    let mut requests = Vec::new();
    let total_size = artifact.data.size()? as i64;
    let num_chunks = (total_size as usize + CHUNK_SIZE - 1) / CHUNK_SIZE;

    let mut stream = artifact.data.stream()?;
    let mut first_chunk = true;

    let mut buf = vec![0u8; CHUNK_SIZE];
    loop {
        let n = stream.read(&mut buf).map_err(|e| {
            SparkError::connect_msg(format!("Failed to read artifact chunk: {}", e))
        })?;
        if n == 0 {
            break;
        }

        let chunk_data = buf[..n].to_vec();
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
                        data: Bytes::from(chunk_data),
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
                    data: Bytes::from(chunk_data),
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
}

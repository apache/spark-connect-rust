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

/// Frame cached in-memory artifacts as they are consumed by the upload stream.
/// Payload slices share the original allocation instead of copying each chunk.
pub(crate) fn cache_artifact_requests(
    session_id: String,
    artifacts: Vec<(String, Bytes)>,
) -> impl Iterator<Item = AddArtifactsRequest> + Send + 'static {
    let mut artifacts = artifacts.into_iter().peekable();
    let mut remaining_chunks = None::<Bytes>;

    std::iter::from_fn(move || {
        let payload = if let Some(mut data) = remaining_chunks.take() {
            let chunk_data = data.split_to(CHUNK_SIZE.min(data.len()));
            if !data.is_empty() {
                remaining_chunks = Some(data);
            }
            add_artifacts_request::Payload::Chunk(add_artifacts_request::ArtifactChunk {
                crc: compute_crc32(&chunk_data) as i64,
                data: chunk_data,
            })
        } else {
            let (_, next_data) = artifacts.peek()?;
            if next_data.len() > CHUNK_SIZE {
                let (name, mut data) = artifacts.next().unwrap();
                let total_bytes = data.len() as i64;
                let num_chunks = data.len().div_ceil(CHUNK_SIZE) as i64;
                let chunk_data = data.split_to(CHUNK_SIZE);
                remaining_chunks = Some(data);
                add_artifacts_request::Payload::BeginChunk(
                    add_artifacts_request::BeginChunkedArtifact {
                        name,
                        total_bytes,
                        num_chunks,
                        initial_chunk: Some(add_artifacts_request::ArtifactChunk {
                            crc: compute_crc32(&chunk_data) as i64,
                            data: chunk_data,
                        }),
                    },
                )
            } else {
                let mut batch_artifacts = Vec::new();
                let mut batch_size = 0;
                while let Some((_, data)) = artifacts.peek() {
                    if data.len() > CHUNK_SIZE || batch_size + data.len() > CHUNK_SIZE {
                        break;
                    }
                    let (name, data) = artifacts.next().unwrap();
                    batch_size += data.len();
                    batch_artifacts.push(add_artifacts_request::SingleChunkArtifact {
                        name,
                        data: Some(add_artifacts_request::ArtifactChunk {
                            crc: compute_crc32(&data) as i64,
                            data,
                        }),
                    });
                }
                add_artifacts_request::Payload::Batch(add_artifacts_request::Batch {
                    artifacts: batch_artifacts,
                })
            }
        };
        Some(AddArtifactsRequest {
            session_id: session_id.clone(),
            payload: Some(payload),
            ..Default::default()
        })
    })
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
    let num_chunks = (total_size as usize).div_ceil(CHUNK_SIZE);

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

    #[test]
    fn test_cache_requests_match_existing_framing() {
        let sizes = [
            0,
            1,
            CHUNK_SIZE - 1,
            CHUNK_SIZE,
            CHUNK_SIZE + 1,
            CHUNK_SIZE * 2,
            CHUNK_SIZE * 2 + 1,
            CHUNK_SIZE / 2,
            CHUNK_SIZE / 2 + 1,
            0,
        ];
        let artifacts: Vec<_> = sizes
            .into_iter()
            .enumerate()
            .map(|(index, size)| {
                (
                    format!("cache/artifact-{index}"),
                    Bytes::from(
                        (0..size)
                            .map(|offset| (offset + index) as u8)
                            .collect::<Vec<_>>(),
                    ),
                )
            })
            .collect();
        let eager_artifacts = artifacts
            .iter()
            .map(|(name, data)| {
                NamedArtifact::new(name.clone(), Box::new(InMemoryArtifact::new(data.to_vec())))
            })
            .collect();
        let expected =
            build_artifact_request_stream("session".to_string(), eager_artifacts).unwrap();
        let actual: Vec<_> = cache_artifact_requests("session".to_string(), artifacts).collect();

        assert_eq!(actual, expected);
        assert!(cache_artifact_requests("session".to_string(), vec![])
            .next()
            .is_none());
    }

    #[test]
    fn test_cache_requests_batch_boundaries() {
        let artifacts = [0, CHUNK_SIZE, 0, 1, CHUNK_SIZE - 1, 1]
            .into_iter()
            .enumerate()
            .map(|(index, size)| (index.to_string(), Bytes::from(vec![1; size])))
            .collect();
        let requests: Vec<_> = cache_artifact_requests("session".to_string(), artifacts).collect();
        let batches: Vec<_> = requests
            .into_iter()
            .map(|request| {
                let Some(add_artifacts_request::Payload::Batch(batch)) = request.payload else {
                    panic!("small artifacts must use batch requests");
                };
                batch
            })
            .collect();

        assert_eq!(batches.len(), 3);
        assert_eq!(
            batches[0]
                .artifacts
                .iter()
                .map(|artifact| artifact.name.as_str())
                .collect::<Vec<_>>(),
            ["0", "1", "2"]
        );
        assert_eq!(batches[0].artifacts[0].data.as_ref().unwrap().crc, 0);
        assert_eq!(batches[0].artifacts[2].data.as_ref().unwrap().crc, 0);
        assert_eq!(batches[1].artifacts.len(), 2);
        assert_eq!(batches[2].artifacts.len(), 1);
    }

    #[test]
    fn test_cache_requests_share_payload_storage() {
        let small = Bytes::from(vec![1; 7]);
        let large = Bytes::from(vec![2; CHUNK_SIZE * 2 + 5]);
        let next_large = Bytes::from(vec![3; CHUNK_SIZE + 1]);
        let artifacts = vec![
            ("small".to_string(), small.clone()),
            ("large".to_string(), large.clone()),
            ("next-large".to_string(), next_large.clone()),
        ];
        let requests: Vec<_> = cache_artifact_requests("session".to_string(), artifacts).collect();
        let chunks: Vec<_> = requests
            .iter()
            .flat_map(|request| match request.payload.as_ref().unwrap() {
                add_artifacts_request::Payload::Batch(batch) => batch
                    .artifacts
                    .iter()
                    .map(|artifact| artifact.data.as_ref().unwrap())
                    .collect::<Vec<_>>(),
                add_artifacts_request::Payload::BeginChunk(begin) => {
                    vec![begin.initial_chunk.as_ref().unwrap()]
                }
                add_artifacts_request::Payload::Chunk(chunk) => vec![chunk],
            })
            .collect();
        let expected = [
            (small.as_ptr(), small.len()),
            (large.as_ptr(), CHUNK_SIZE),
            (large.as_ptr().wrapping_add(CHUNK_SIZE), CHUNK_SIZE),
            (large.as_ptr().wrapping_add(CHUNK_SIZE * 2), 5),
            (next_large.as_ptr(), CHUNK_SIZE),
            (next_large.as_ptr().wrapping_add(CHUNK_SIZE), 1),
        ];

        assert_eq!(chunks.len(), expected.len());
        for (chunk, (pointer, len)) in chunks.into_iter().zip(expected) {
            assert_eq!(chunk.data.as_ptr(), pointer);
            assert_eq!(chunk.data.len(), len);
        }
    }

    #[test]
    fn test_cache_requests_release_storage_on_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        struct TrackedData {
            data: Vec<u8>,
            drops: Arc<AtomicUsize>,
        }

        impl AsRef<[u8]> for TrackedData {
            fn as_ref(&self) -> &[u8] {
                &self.data
            }
        }

        impl Drop for TrackedData {
            fn drop(&mut self) {
                self.drops.fetch_add(1, Ordering::SeqCst);
            }
        }

        let first_drops = Arc::new(AtomicUsize::new(0));
        let second_drops = Arc::new(AtomicUsize::new(0));
        let first = Bytes::from_owner(TrackedData {
            data: vec![1; CHUNK_SIZE + 1],
            drops: first_drops.clone(),
        });
        let second = Bytes::from_owner(TrackedData {
            data: vec![2; CHUNK_SIZE + 1],
            drops: second_drops.clone(),
        });
        let mut requests = cache_artifact_requests(
            "session".to_string(),
            vec![("first".to_string(), first), ("second".to_string(), second)],
        );
        let first_request = requests.next().unwrap();

        assert_eq!(first_drops.load(Ordering::SeqCst), 0);
        assert_eq!(second_drops.load(Ordering::SeqCst), 0);
        drop(requests);
        assert_eq!(first_drops.load(Ordering::SeqCst), 0);
        assert_eq!(second_drops.load(Ordering::SeqCst), 1);
        drop(first_request);
        assert_eq!(first_drops.load(Ordering::SeqCst), 1);
    }
}

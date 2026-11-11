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
use std::sync::{Arc, Mutex, PoisonError};

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

/// Checks the response of an artifact upload: every artifact passed its CRC check and, for
/// a cache upload, every artifact of `expected_cache_names` was acknowledged.
pub(crate) fn check_artifact_upload(
    response: spark_connect_proto::AddArtifactsResponse,
    expected_cache_names: Option<&std::collections::HashSet<String>>,
) -> Result<()> {
    let mut received = std::collections::HashSet::new();
    for summary in response.artifacts {
        if !summary.is_crc_successful {
            return Err(SparkError::connect_msg(format!(
                "CRC check failed for artifact: {}",
                summary.name
            )));
        }
        if expected_cache_names.is_some() {
            received.insert(summary.name);
        }
    }
    if let Some(expected) = expected_cache_names {
        if !expected.is_subset(&received) {
            return Err(SparkError::connect_msg(
                "Missing cache artifact upload acknowledgement",
            ));
        }
    }
    Ok(())
}

/// Frame cached in-memory artifacts as they are consumed by the upload stream.
/// Payload slices share the original allocation instead of copying each chunk.
/// The allocation is freed when its last slice is dropped, not chunk by chunk.
pub(crate) fn cache_artifact_requests(
    session_id: String,
    artifacts: Vec<(String, Bytes)>,
) -> impl Iterator<Item = AddArtifactsRequest> + Send + 'static {
    // Data in memory is always of its size and never fails to be read, so these frames end
    // only at the last artifact.
    Frames::over(
        session_id,
        artifacts.into_iter().map(|(name, data)| {
            Ok(FrameSource {
                name,
                size: data.len() as u64,
                data,
            })
        }),
    )
}

/// Build a stream of AddArtifactsRequest messages for artifacts.
/// Small artifacts (<= CHUNK_SIZE) are batched together.
/// Large artifacts (> CHUNK_SIZE) are sent as chunked requests.
///
/// All the requests are built before any is returned, so every artifact is held in memory at
/// once. The requests are those that an upload produces as its RPC consumes them.
pub fn build_artifact_request_stream(
    session_id: String,
    artifacts: Vec<NamedArtifact>,
) -> Result<Vec<AddArtifactsRequest>> {
    let mut frames = Frames::over(
        session_id,
        artifacts
            .into_iter()
            .map(|artifact| FrameSource::open(&artifact)),
    );
    let requests: Vec<AddArtifactsRequest> = frames.by_ref().collect();
    take_frame_failure(&frames.failure())?;
    Ok(requests)
}

/// The error that ended the frames of an upload early, if one did. The frames stop at a read
/// that fails; the upload reports the error once its RPC has returned.
pub(crate) type FrameFailure = Arc<Mutex<Option<SparkError>>>;

/// Take the error that ended the frames of an upload early, if one did.
pub(crate) fn take_frame_failure(failure: &FrameFailure) -> Result<()> {
    match failure
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
    {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// The data of one artifact, as its frames read it.
pub(crate) trait FrameData {
    /// All of the data, for an artifact that is sent whole in a batch.
    fn read_all(&mut self) -> Result<Bytes>;

    /// The next [`CHUNK_SIZE`] bytes of the data, or the fewer (or none) that remain at its
    /// end.
    fn read_chunk(&mut self) -> Result<Bytes>;
}

/// Data in memory: its frames are slices that share its allocation.
impl FrameData for Bytes {
    fn read_all(&mut self) -> Result<Bytes> {
        Ok(std::mem::take(self))
    }

    fn read_chunk(&mut self) -> Result<Bytes> {
        Ok(self.split_to(CHUNK_SIZE.min(self.len())))
    }
}

/// Data that is read as its frames are produced, one chunk held at a time.
pub(crate) struct ReadData(Box<dyn Read + Send>);

impl FrameData for ReadData {
    fn read_all(&mut self) -> Result<Bytes> {
        let mut data = Vec::new();
        self.0
            .read_to_end(&mut data)
            .map_err(|e| SparkError::connect_msg(format!("Failed to read artifact: {}", e)))?;
        Ok(Bytes::from(data))
    }

    fn read_chunk(&mut self) -> Result<Bytes> {
        let mut buf = vec![0u8; CHUNK_SIZE];
        let filled = read_full(self.0.as_mut(), &mut buf).map_err(|e| {
            SparkError::connect_msg(format!("Failed to read artifact chunk: {}", e))
        })?;
        buf.truncate(filled);
        Ok(Bytes::from(buf))
    }
}

/// One artifact to frame: its name, the size it declares, and its data.
pub(crate) struct FrameSource<D> {
    name: String,
    size: u64,
    data: D,
}

impl FrameSource<ReadData> {
    /// Take the size of `artifact` and open its data.
    fn open(artifact: &NamedArtifact) -> Result<Self> {
        Ok(Self {
            name: artifact.name.clone(),
            size: artifact.data.size()?,
            data: ReadData(artifact.data.stream()?),
        })
    }
}

impl<D: FrameData> FrameSource<D> {
    /// Read the next chunk of an artifact of which `sent` bytes are framed, and return it with
    /// whether it is the last one. The chunks of an artifact are announced by the size it
    /// declares (`total_bytes` and `num_chunks` of its `BeginChunk`), so data that ends
    /// before that size or goes on past it is an error; the server would reject the upload,
    /// or, when no chunk at all could be read, would be sent nothing and answer with success.
    fn next_chunk(&mut self, sent: &mut u64) -> Result<(Bytes, bool)> {
        let chunk = self.data.read_chunk()?;
        *sent += chunk.len() as u64;
        let last = chunk.len() < CHUNK_SIZE;
        if *sent > self.size || (last && *sent < self.size) {
            return Err(SparkError::connect_msg(format!(
                "Artifact {} changed size during its upload: expected {} bytes, read {} bytes{}",
                self.name,
                self.size,
                *sent,
                if last { "" } else { " so far" }
            )));
        }
        Ok((chunk, last))
    }
}

/// The frames of an upload of artifacts, produced as the RPC consumes them, so that a large
/// artifact that is read is held one chunk at a time rather than as a whole.
///
/// This is the one place that decides how artifacts are framed, as PySpark's
/// `ArtifactManager._add_artifacts` frames them. In the order of the artifacts:
///
/// - an artifact of more than [`CHUNK_SIZE`] bytes is a `BeginChunk` with its first chunk and
///   then a `Chunk` for each further one, each a full chunk but the last;
/// - consecutive artifacts of at most [`CHUNK_SIZE`] bytes each share a `Batch` for as long
///   as their sizes sum to at most [`CHUNK_SIZE`]; the artifact that would exceed it starts
///   the next batch. A batch has at least one artifact.
///
/// The frames end early at an artifact that cannot be opened or read, and at a chunked
/// artifact whose data is not of the size it declared; [`Frames::failure`] has the error.
pub(crate) struct Frames<D> {
    session_id: String,
    /// The artifacts that are not framed yet, until the last is taken or the frames fail.
    sources: Option<Box<dyn Iterator<Item = Result<FrameSource<D>>> + Send>>,
    /// An artifact taken from `sources` that did not fit the batch being built.
    lookahead: Option<FrameSource<D>>,
    /// The chunked artifact whose `BeginChunk` is produced, with the count of its bytes that
    /// are framed, until its last chunk is.
    chunked: Option<(FrameSource<D>, u64)>,
    failure: FrameFailure,
}

/// The frames of an upload of one artifact, read as the RPC consumes them. An attempt that is
/// retried builds its frames again, from the start of the artifact.
pub(crate) type ArtifactFrames = Frames<ReadData>;

impl ArtifactFrames {
    /// Open the data of `artifact`, so that a data source that cannot be read is an error before
    /// any frame is sent.
    pub(crate) fn new(session_id: String, artifact: &NamedArtifact) -> Result<Self> {
        let source = FrameSource::open(artifact)?;
        Ok(Self::over(session_id, std::iter::once(Ok(source))))
    }
}

impl<D: FrameData> Frames<D> {
    fn over(
        session_id: String,
        sources: impl Iterator<Item = Result<FrameSource<D>>> + Send + 'static,
    ) -> Self {
        Self {
            session_id,
            sources: Some(Box::new(sources)),
            lookahead: None,
            chunked: None,
            failure: Arc::default(),
        }
    }

    /// The failure that ends these frames early, which the upload checks once its RPC returns.
    pub(crate) fn failure(&self) -> FrameFailure {
        self.failure.clone()
    }

    /// The next artifact to frame.
    fn next_source(&mut self) -> Result<Option<FrameSource<D>>> {
        if let Some(source) = self.lookahead.take() {
            return Ok(Some(source));
        }
        let next = self.sources.as_mut().and_then(Iterator::next);
        if next.is_none() {
            self.sources = None;
        }
        next.transpose()
    }

    fn next_payload(&mut self) -> Result<Option<add_artifacts_request::Payload>> {
        // The next chunk of the chunked artifact that is begun. Data that ends on a chunk
        // boundary is seen to end only by a read of nothing, which is no frame.
        if let Some((mut source, mut sent)) = self.chunked.take() {
            let (data, last) = source.next_chunk(&mut sent)?;
            if !last {
                self.chunked = Some((source, sent));
            }
            if !data.is_empty() {
                return Ok(Some(add_artifacts_request::Payload::Chunk(chunk(data))));
            }
        }

        let Some(mut source) = self.next_source()? else {
            return Ok(None);
        };
        if source.size > CHUNK_SIZE as u64 {
            let mut sent = 0;
            let (data, last) = source.next_chunk(&mut sent)?;
            let begin = add_artifacts_request::BeginChunkedArtifact {
                name: source.name.clone(),
                total_bytes: source.size as i64,
                num_chunks: source.size.div_ceil(CHUNK_SIZE as u64) as i64,
                initial_chunk: Some(chunk(data)),
            };
            if !last {
                self.chunked = Some((source, sent));
            }
            return Ok(Some(add_artifacts_request::Payload::BeginChunk(begin)));
        }

        // A batch: this artifact, and those after it for as long as they fit.
        let mut artifacts = Vec::new();
        let mut batch_size = 0;
        loop {
            batch_size += source.size;
            artifacts.push(add_artifacts_request::SingleChunkArtifact {
                name: source.name,
                data: Some(chunk(source.data.read_all()?)),
            });
            match self.next_source()? {
                Some(next) if next.size <= CHUNK_SIZE as u64 - batch_size => source = next,
                next => {
                    self.lookahead = next;
                    break;
                }
            }
        }
        Ok(Some(add_artifacts_request::Payload::Batch(
            add_artifacts_request::Batch { artifacts },
        )))
    }
}

/// A chunk of an artifact with its CRC.
fn chunk(data: Bytes) -> add_artifacts_request::ArtifactChunk {
    add_artifacts_request::ArtifactChunk {
        crc: compute_crc32(&data) as i64,
        data,
    }
}

impl<D: FrameData> Iterator for Frames<D> {
    type Item = AddArtifactsRequest;

    fn next(&mut self) -> Option<AddArtifactsRequest> {
        match self.next_payload() {
            Ok(Some(payload)) => Some(AddArtifactsRequest {
                session_id: self.session_id.clone(),
                payload: Some(payload),
                ..Default::default()
            }),
            Ok(None) => None,
            Err(error) => {
                self.sources = None;
                self.lookahead = None;
                self.chunked = None;
                *self.failure.lock().unwrap_or_else(PoisonError::into_inner) = Some(error);
                None
            }
        }
    }
}

/// Fill `buf` from `reader` until it is full or the data ends, and return the count read.
fn read_full(reader: &mut dyn Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// A path of `addArtifacts` that PySpark's `ArtifactManager` does not upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsupportedArtifact {
    /// A file of a format that is not uploaded with the flags given, by its local path.
    Format(String),
    /// A URI of a scheme other than `file`.
    Scheme(String),
    /// The local path of an archive with a fragment has a `#`, which the fragment would
    /// be taken from.
    HashInArchivePath,
}

impl std::fmt::Display for UnsupportedArtifact {
    /// The message of PySpark's error for the path.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnsupportedArtifact::Format(path) => {
                write!(
                    f,
                    "[UNSUPPORTED_OPERATION] {path} file format is not supported."
                )
            }
            UnsupportedArtifact::Scheme(scheme) => {
                write!(
                    f,
                    "[UNSUPPORTED_OPERATION] {scheme} scheme is not supported."
                )
            }
            UnsupportedArtifact::HashInArchivePath => write!(
                f,
                "[VALUE_ALLOWED] Value for `artifact path` does not allow #."
            ),
        }
    }
}

/// `%XX` escapes decoded, as `url2pathname` does on POSIX.
fn unquote(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(high), Some(low)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((high * 16 + low) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The name that a path of `addArtifacts` is uploaded as, and the local file to upload, as
/// PySpark's `ArtifactManager._parse_artifacts` parses a local path or a `file:` URI:
///
/// - With `pyfile`, a `.py`, `.egg`, `.zip` or `.jar` is `pyfiles/<name>`.
/// - With `archive`, a `.zip`, `.jar`, `.tar.gz`, `.tgz` or `.tar` is `archives/<name>`, or
///   `archives/<name>#<fragment>` for a path with a fragment, which names the directory
///   that the archive is extracted to.
/// - With `file`, any file is `files/<name>`.
/// - Otherwise a `.class` is `classes/<name>`, a `.jar` is `jars/<name>`, and any other file
///   is not supported.
///
/// The `classes/` of a class file is an extension of this crate: PySpark's `ArtifactManager`
/// does not upload class files, so the PySpark drop-in refuses them.
///
/// The local path is normalized as `pathlib` normalizes it (no trailing or repeated separators,
/// and no `.` components), so that it reads the same in an error message.
pub fn parse_artifact(
    path_or_uri: &str,
    pyfile: bool,
    archive: bool,
    file: bool,
) -> std::result::Result<(String, std::path::PathBuf), UnsupportedArtifact> {
    // As `urlparse`: a scheme of a letter and then letters, digits, `+`, `-` or `.` before
    // `:`, a netloc after `//`, and the fragment after `#` and the query after `?`.
    let (scheme, rest) = match path_or_uri.find(':') {
        Some(i)
            if path_or_uri[..i].starts_with(|c: char| c.is_ascii_alphabetic())
                && path_or_uri[..i]
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)) =>
        {
            (path_or_uri[..i].to_ascii_lowercase(), &path_or_uri[i + 1..])
        }
        _ => (String::new(), path_or_uri),
    };
    let rest = match rest.strip_prefix("//") {
        Some(after) => &after[after.find(['/', '?', '#']).unwrap_or(after.len())..],
        None => rest,
    };
    let (rest, fragment) = rest.split_once('#').unwrap_or((rest, ""));
    let path = rest.split_once('?').map_or(rest, |(path, _)| path);
    let local = match scheme.as_str() {
        "" => {
            let path = std::path::PathBuf::from(unquote(path));
            if path.is_absolute() {
                path
            } else {
                std::env::current_dir().unwrap_or_default().join(path)
            }
        }
        "file" => std::path::PathBuf::from(unquote(path)),
        _ => return Err(UnsupportedArtifact::Scheme(scheme)),
    };
    let local: std::path::PathBuf = local.components().collect();
    let local_path = local.to_string_lossy().into_owned();
    let name = local
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ends = |suffixes: &[&str]| suffixes.iter().any(|suffix| name.ends_with(suffix));
    let artifact = if pyfile && ends(&[".py", ".egg", ".zip", ".jar"]) {
        format!("pyfiles/{name}")
    } else if archive && ends(&[".zip", ".jar", ".tar.gz", ".tgz", ".tar"]) {
        if fragment.is_empty() {
            format!("archives/{name}")
        } else if local_path.contains('#') {
            return Err(UnsupportedArtifact::HashInArchivePath);
        } else {
            format!("archives/{name}#{fragment}")
        }
    } else if file {
        format!("files/{name}")
    } else if ends(&[".class"]) {
        format!("classes/{name}")
    } else if ends(&[".jar"]) {
        format!("jars/{name}")
    } else {
        return Err(UnsupportedArtifact::Format(local_path));
    };
    Ok((artifact, local))
}

#[cfg(test)]
mod tests {
    use super::{parse_artifact, UnsupportedArtifact};

    #[test]
    fn artifacts_are_named_as_pyspark_parses_their_paths() {
        use UnsupportedArtifact::*;
        let named = |name: &str| Ok(name.to_string());
        let cases = [
            ("/d/a.jar", (false, false, false), named("jars/a.jar")),
            (
                "/d/A.class",
                (false, false, false),
                named("classes/A.class"),
            ),
            ("/d/m.py", (true, false, false), named("pyfiles/m.py")),
            ("/d/m.zip", (true, false, false), named("pyfiles/m.zip")),
            (
                "/d/b.tar.gz",
                (false, true, false),
                named("archives/b.tar.gz"),
            ),
            // The fragment of an archive names the directory that it is extracted to.
            (
                "/d/b.zip#alias",
                (false, true, false),
                named("archives/b.zip#alias"),
            ),
            ("/d/n.txt", (false, false, true), named("files/n.txt")),
            // Another path drops its fragment, and its query, as `urlparse` takes them.
            ("/d/a.jar#x", (false, false, false), named("jars/a.jar")),
            (
                "file:/d/a.jar?q=1",
                (false, false, false),
                named("jars/a.jar"),
            ),
            (
                "file:///d/a%20b.jar",
                (false, false, false),
                named("jars/a b.jar"),
            ),
            (
                "/d/m.py",
                (false, false, false),
                Err(Format("/d/m.py".to_string())),
            ),
            (
                "/d/m.py",
                (false, true, false),
                Err(Format("/d/m.py".to_string())),
            ),
            (
                "s3://bucket/a.jar",
                (false, false, false),
                Err(Scheme("s3".to_string())),
            ),
            (
                "/d/b%23c.zip#alias",
                (false, true, false),
                Err(HashInArchivePath),
            ),
            // The local path is normalized as pathlib normalizes it, in the error too.
            (
                "/d//x/./n.txt/",
                (false, false, false),
                Err(Format("/d/x/n.txt".to_string())),
            ),
        ];
        for (path, (pyfile, archive, file), expected) in cases {
            let parsed = parse_artifact(path, pyfile, archive, file).map(|(name, _)| name);
            assert_eq!(parsed, expected, "{path}");
        }
    }

    #[test]
    fn a_relative_artifact_path_is_in_the_current_directory() {
        let (name, local) = parse_artifact("a.jar", false, false, false).unwrap();
        assert_eq!(name, "jars/a.jar");
        assert_eq!(local, std::env::current_dir().unwrap().join("a.jar"));
    }

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
        assert_cache_requests_release_storage(false);
    }

    #[test]
    fn test_cache_requests_release_storage_on_exhaustion() {
        assert_cache_requests_release_storage(true);
    }

    fn assert_cache_requests_release_storage(exhaust_stream: bool) {
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
        let third_drops = Arc::new(AtomicUsize::new(0));
        let first = Bytes::from_owner(TrackedData {
            data: vec![1; CHUNK_SIZE + 1],
            drops: first_drops.clone(),
        });
        let second = Bytes::from_owner(TrackedData {
            data: vec![2; CHUNK_SIZE + 1],
            drops: second_drops.clone(),
        });
        let third = Bytes::from_owner(TrackedData {
            data: vec![3; CHUNK_SIZE + 1],
            drops: third_drops.clone(),
        });
        let mut requests = cache_artifact_requests(
            "session".to_string(),
            vec![
                ("first".to_string(), first),
                ("second".to_string(), second),
                ("third".to_string(), third),
            ],
        );
        // Each CHUNK_SIZE + 1 blob emits BeginChunk followed by one Chunk.
        drop(requests.next().unwrap());
        let first_final_request = requests.next().unwrap();
        assert_eq!(first_drops.load(Ordering::SeqCst), 0);
        drop(first_final_request);
        // A consumed blob is released while later blobs remain queued.
        assert_eq!(first_drops.load(Ordering::SeqCst), 1);
        assert_eq!(second_drops.load(Ordering::SeqCst), 0);

        let second_initial_request = requests.next().unwrap();
        if exhaust_stream {
            // Consume the second blob's Chunk and both frames of the third blob.
            drop(requests.next().unwrap());
            drop(requests.next().unwrap());
            drop(requests.next().unwrap());
            assert!(requests.next().is_none());
        }
        drop(requests);
        // An earlier frame keeps its allocation alive after either early drop
        // or complete exhaustion of the iterator.
        assert_eq!(second_drops.load(Ordering::SeqCst), 0);
        assert_eq!(third_drops.load(Ordering::SeqCst), 1);
        drop(second_initial_request);
        assert_eq!(second_drops.load(Ordering::SeqCst), 1);
        assert_eq!(first_drops.load(Ordering::SeqCst), 1);
    }

    /// Test data that is produced as it is read, and counts the bytes read of it.
    struct Counted {
        size: u64,
        read: Arc<std::sync::atomic::AtomicUsize>,
        /// The read that fails, when the data has one.
        fails_at_read: Option<usize>,
    }

    impl ArtifactData for Counted {
        fn size(&self) -> Result<u64> {
            Ok(self.size)
        }

        fn stream(&self) -> Result<Box<dyn Read + Send>> {
            Ok(Box::new(CountedReader {
                remaining: self.size as usize,
                read: self.read.clone(),
                reads: 0,
                fails_at_read: self.fails_at_read,
            }))
        }
    }

    struct CountedReader {
        remaining: usize,
        read: Arc<std::sync::atomic::AtomicUsize>,
        reads: usize,
        fails_at_read: Option<usize>,
    }

    impl Read for CountedReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.reads += 1;
            if self.fails_at_read == Some(self.reads) {
                return Err(std::io::Error::other("disk gone"));
            }
            let n = buf.len().min(self.remaining);
            buf[..n].fill(7);
            self.remaining -= n;
            self.read.fetch_add(n, std::sync::atomic::Ordering::SeqCst);
            Ok(n)
        }
    }

    fn counted(size: u64) -> (NamedArtifact, Arc<std::sync::atomic::AtomicUsize>) {
        let read = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let data = Counted {
            size,
            read: read.clone(),
            fails_at_read: None,
        };
        (
            NamedArtifact::new("files/n.bin".to_string(), Box::new(data)),
            read,
        )
    }

    #[test]
    fn artifact_frames_are_read_as_the_rpc_consumes_them() {
        let size = (CHUNK_SIZE * 10 + 3) as u64;
        let (artifact, read) = counted(size);
        let mut frames = ArtifactFrames::new("s".to_string(), &artifact).unwrap();
        // Opening the data reads none of it.
        assert_eq!(read.load(std::sync::atomic::Ordering::SeqCst), 0);
        frames.next().unwrap();
        assert_eq!(read.load(std::sync::atomic::Ordering::SeqCst), CHUNK_SIZE);
        frames.next().unwrap();
        assert_eq!(
            read.load(std::sync::atomic::Ordering::SeqCst),
            2 * CHUNK_SIZE
        );
        assert_eq!(frames.by_ref().count(), 9);
        assert_eq!(
            read.load(std::sync::atomic::Ordering::SeqCst),
            size as usize
        );
        assert!(take_frame_failure(&frames.failure()).is_ok());
    }

    #[test]
    fn chunked_artifact_frames_begin_and_then_carry_each_chunk_with_its_crc() {
        let size = CHUNK_SIZE * 2 + 5;
        let data: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let artifact = NamedArtifact::new(
            "jars/c.jar".to_string(),
            Box::new(InMemoryArtifact::new(data.clone())),
        );
        let frames: Vec<AddArtifactsRequest> = ArtifactFrames::new("s".to_string(), &artifact)
            .unwrap()
            .collect();
        assert_eq!(frames.len(), 3);
        let mut expected = data.chunks(CHUNK_SIZE);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(frame.session_id, "s");
            let chunk = expected.next().unwrap();
            let (payload, crc) = match frame.payload.as_ref().unwrap() {
                add_artifacts_request::Payload::BeginChunk(begin) => {
                    assert_eq!(index, 0);
                    assert_eq!(begin.name, "jars/c.jar");
                    assert_eq!(begin.total_bytes, size as i64);
                    assert_eq!(begin.num_chunks, 3);
                    let initial = begin.initial_chunk.as_ref().unwrap();
                    (initial.data.to_vec(), initial.crc)
                }
                add_artifacts_request::Payload::Chunk(chunk) => {
                    assert_ne!(index, 0);
                    (chunk.data.to_vec(), chunk.crc)
                }
                other => panic!("unexpected frame {other:?}"),
            };
            assert_eq!(payload, chunk);
            assert_eq!(crc, compute_crc32(chunk) as i64);
        }
    }

    #[test]
    fn a_small_artifact_frame_is_one_batch() {
        let artifact = NamedArtifact::new(
            "classes/A.class".to_string(),
            Box::new(InMemoryArtifact::new(b"cafebabe".to_vec())),
        );
        let frames: Vec<AddArtifactsRequest> = ArtifactFrames::new("s".to_string(), &artifact)
            .unwrap()
            .collect();
        let [frame] = frames.as_slice() else {
            panic!("expected one frame, got {}", frames.len());
        };
        let Some(add_artifacts_request::Payload::Batch(batch)) = &frame.payload else {
            panic!("expected a batch");
        };
        let [only] = batch.artifacts.as_slice() else {
            panic!("expected one artifact");
        };
        assert_eq!(only.name, "classes/A.class");
        let chunk = only.data.as_ref().unwrap();
        assert_eq!(chunk.data.as_ref(), b"cafebabe");
        assert_eq!(chunk.crc, compute_crc32(b"cafebabe") as i64);
    }

    #[test]
    fn a_failed_read_ends_the_frames_and_is_reported_once() {
        let read = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let artifact = NamedArtifact::new(
            "files/n.bin".to_string(),
            Box::new(Counted {
                size: (CHUNK_SIZE * 3) as u64,
                read,
                // The first chunk is read; the read of the second chunk fails.
                fails_at_read: Some(2),
            }),
        );
        let mut frames = ArtifactFrames::new("s".to_string(), &artifact).unwrap();
        let failure = frames.failure();
        assert!(frames.next().is_some());
        assert!(frames.next().is_none());
        assert!(frames.next().is_none());
        let error = take_frame_failure(&failure).unwrap_err();
        assert!(
            error.to_string().contains("Failed to read artifact chunk"),
            "{error}"
        );
        assert!(take_frame_failure(&failure).is_ok());
    }

    #[test]
    fn artifact_frames_are_the_same_on_every_attempt() {
        let size = CHUNK_SIZE * 2 + 1;
        let data: Vec<u8> = (0..size).map(|i| (i % 13) as u8).collect();
        let make = || {
            NamedArtifact::new(
                "jars/c.jar".to_string(),
                Box::new(InMemoryArtifact::new(data.clone())),
            )
        };
        let first = make();
        let second = make();
        let a: Vec<AddArtifactsRequest> = ArtifactFrames::new("s".to_string(), &first)
            .unwrap()
            .collect();
        let b: Vec<AddArtifactsRequest> = ArtifactFrames::new("s".to_string(), &second)
            .unwrap()
            .collect();
        assert_eq!(a, b);
        let batch = build_artifact_request_stream("s".to_string(), vec![make()]).unwrap();
        assert_eq!(a, batch);
    }

    /// The artifacts of the pinned framing, by name and size: an empty one, one of exactly a
    /// chunk, one a byte over a chunk, small ones that cross a batch boundary, small ones
    /// between chunked ones, and ones of several chunks, the last a whole number of chunks.
    const PINNED_ARTIFACTS: [(&str, usize); 13] = [
        ("files/empty", 0),
        ("files/exact", CHUNK_SIZE),
        ("jars/over.jar", CHUNK_SIZE + 1),
        ("pyfiles/a.py", 10_000),
        ("pyfiles/b.py", 10_000),
        ("pyfiles/c.py", 10_000),
        ("pyfiles/d.py", 10_000),
        ("files/tiny", 5),
        ("jars/multi.jar", CHUNK_SIZE * 3 + 7),
        ("files/one", 1),
        ("files/rest", CHUNK_SIZE - 1),
        ("files/spill", 1),
        ("jars/whole.jar", CHUNK_SIZE * 2),
    ];

    fn pinned_data(index: usize, size: usize) -> Vec<u8> {
        (0..size).map(|i| (i * 31 + index * 7) as u8).collect()
    }

    fn pinned_named(index: usize) -> NamedArtifact {
        let (name, size) = PINNED_ARTIFACTS[index];
        NamedArtifact::new(
            name.to_string(),
            Box::new(InMemoryArtifact::new(pinned_data(index, size))),
        )
    }

    /// One line for each request: its payload, with the length and the CRC of each chunk.
    fn framing(requests: &[AddArtifactsRequest]) -> Vec<String> {
        let chunk = |chunk: &Option<add_artifacts_request::ArtifactChunk>| {
            let chunk = chunk.as_ref().unwrap();
            assert_eq!(chunk.crc, compute_crc32(&chunk.data) as i64);
            format!("{}:{:08x}", chunk.data.len(), chunk.crc)
        };
        requests
            .iter()
            .map(|request| {
                assert_eq!(request.session_id, "session");
                match request.payload.as_ref().unwrap() {
                    add_artifacts_request::Payload::Batch(batch) => {
                        let artifacts: Vec<String> = batch
                            .artifacts
                            .iter()
                            .map(|artifact| format!("{}={}", artifact.name, chunk(&artifact.data)))
                            .collect();
                        format!("batch [{}]", artifacts.join(", "))
                    }
                    add_artifacts_request::Payload::BeginChunk(begin) => format!(
                        "begin {} total={} chunks={} {}",
                        begin.name,
                        begin.total_bytes,
                        begin.num_chunks,
                        chunk(&begin.initial_chunk)
                    ),
                    add_artifacts_request::Payload::Chunk(data) => {
                        format!("chunk {}", chunk(&Some(data.clone())))
                    }
                }
            })
            .collect()
    }

    /// The CRC of the requests as they are encoded for the wire, one after another.
    fn wire_crc(requests: &[AddArtifactsRequest]) -> u32 {
        use prost::Message;
        let mut hasher = Hasher::new();
        for request in requests {
            hasher.update(&request.encode_length_delimited_to_vec());
        }
        hasher.finalize()
    }

    /// Pins the requests that every way of framing artifacts puts on the wire.
    #[test]
    fn the_framing_of_artifacts_is_pinned() {
        let all = 0..PINNED_ARTIFACTS.len();
        let eager = build_artifact_request_stream(
            "session".to_string(),
            all.clone().map(pinned_named).collect(),
        )
        .unwrap();
        let cached: Vec<AddArtifactsRequest> = cache_artifact_requests(
            "session".to_string(),
            all.clone()
                .map(|index| {
                    let (name, size) = PINNED_ARTIFACTS[index];
                    (name.to_string(), Bytes::from(pinned_data(index, size)))
                })
                .collect(),
        )
        .collect();
        assert_eq!(
            framing(&eager),
            [
                "batch [files/empty=0:00000000, files/exact=32768:7bc368d8]",
                "begin jars/over.jar total=32769 chunks=2 32768:5f6349da",
                "chunk 1:35bac28a",
                "batch [pyfiles/a.py=10000:d625043c, pyfiles/b.py=10000:39b2157a, \
                 pyfiles/c.py=10000:f0508888]",
                "batch [pyfiles/d.py=10000:ed3d6035, files/tiny=5:df828f04]",
                "begin jars/multi.jar total=98311 chunks=4 32768:6036fb23",
                "chunk 32768:6036fb23",
                "chunk 32768:6036fb23",
                "chunk 7:d0965bdd",
                "batch [files/one=1:6464c2b0, files/rest=32767:a89fc864]",
                "batch [files/spill=1:da6fd2a0]",
                "begin jars/whole.jar total=65536 chunks=2 32768:a1b78124",
                "chunk 32768:a1b78124",
            ]
        );
        assert_eq!(wire_crc(&eager), 0x29c4_d88a);
        assert_eq!(framing(&cached), framing(&eager));
        assert_eq!(cached, eager);

        // One artifact at a time, as an upload of a file frames it.
        let mut single = Vec::new();
        for index in all {
            let artifact = pinned_named(index);
            let mut frames = ArtifactFrames::new("session".to_string(), &artifact).unwrap();
            let lazy: Vec<AddArtifactsRequest> = frames.by_ref().collect();
            take_frame_failure(&frames.failure()).unwrap();
            let alone =
                build_artifact_request_stream("session".to_string(), vec![pinned_named(index)])
                    .unwrap();
            assert_eq!(lazy, alone, "{}", artifact.name);
            let (name, size) = PINNED_ARTIFACTS[index];
            let alone_cached: Vec<AddArtifactsRequest> = cache_artifact_requests(
                "session".to_string(),
                vec![(name.to_string(), Bytes::from(pinned_data(index, size)))],
            )
            .collect();
            assert_eq!(lazy, alone_cached, "{}", artifact.name);
            single.extend(lazy);
        }
        assert_eq!(
            framing(&single),
            [
                "batch [files/empty=0:00000000]",
                "batch [files/exact=32768:7bc368d8]",
                "begin jars/over.jar total=32769 chunks=2 32768:5f6349da",
                "chunk 1:35bac28a",
                "batch [pyfiles/a.py=10000:d625043c]",
                "batch [pyfiles/b.py=10000:39b2157a]",
                "batch [pyfiles/c.py=10000:f0508888]",
                "batch [pyfiles/d.py=10000:ed3d6035]",
                "batch [files/tiny=5:df828f04]",
                "begin jars/multi.jar total=98311 chunks=4 32768:6036fb23",
                "chunk 32768:6036fb23",
                "chunk 32768:6036fb23",
                "chunk 7:d0965bdd",
                "batch [files/one=1:6464c2b0]",
                "batch [files/rest=32767:a89fc864]",
                "batch [files/spill=1:da6fd2a0]",
                "begin jars/whole.jar total=65536 chunks=2 32768:a1b78124",
                "chunk 32768:a1b78124",
            ]
        );
        assert_eq!(wire_crc(&single), 0xd422_9fba);
    }

    /// Data that is shorter or longer when it is read than the size it declared, as a file
    /// that is written to between the two.
    struct Resized {
        declared: usize,
        actual: usize,
    }

    impl ArtifactData for Resized {
        fn size(&self) -> Result<u64> {
            Ok(self.declared as u64)
        }

        fn stream(&self) -> Result<Box<dyn Read + Send>> {
            Ok(Box::new(Cursor::new(vec![9u8; self.actual])))
        }
    }

    fn resized(declared: usize, actual: usize) -> NamedArtifact {
        NamedArtifact::new(
            "jars/resized.jar".to_string(),
            Box::new(Resized { declared, actual }),
        )
    }

    #[test]
    fn a_chunked_artifact_that_is_not_of_its_declared_size_is_an_error() {
        let declared = CHUNK_SIZE * 2 + 1;
        // Shorter, down to nothing at all, which used to end the frames before a `BeginChunk`
        // and so upload nothing without an error; and longer.
        let cases = [
            (0, "read 0 bytes"),
            (100, "read 100 bytes"),
            (CHUNK_SIZE, "read 32768 bytes"),
            (CHUNK_SIZE * 2, "read 65536 bytes"),
            (CHUNK_SIZE * 2 + 2, "read 65538 bytes"),
            (CHUNK_SIZE * 4, "read 98304 bytes so far"),
        ];
        for (actual, read) in cases {
            let artifact = resized(declared, actual);
            let mut frames = ArtifactFrames::new("session".to_string(), &artifact).unwrap();
            let failure = frames.failure();
            let sent = frames.by_ref().count();
            assert!(frames.next().is_none());
            // No frame is sent past the size that the `BeginChunk` declared.
            assert!(sent <= 3, "{actual}: {sent} frames");
            let error = take_frame_failure(&failure).unwrap_err().to_string();
            let expected = format!(
                "Artifact jars/resized.jar changed size during its upload: \
                 expected 65537 bytes, {read}"
            );
            assert!(error.ends_with(&expected), "{actual}: {error}");

            let error = build_artifact_request_stream(
                "session".to_string(),
                vec![resized(declared, actual)],
            )
            .unwrap_err()
            .to_string();
            assert!(error.ends_with(&expected), "{actual}: {error}");
        }
        // With nothing read, nothing is sent.
        let mut frames = ArtifactFrames::new("session".to_string(), &resized(declared, 0)).unwrap();
        assert!(frames.next().is_none());
        assert!(take_frame_failure(&frames.failure()).is_err());
    }

    #[test]
    fn a_batched_artifact_is_sent_as_it_is_read() {
        // A batch declares no size, so the data read is what is uploaded, as PySpark does.
        for actual in [0, 3, CHUNK_SIZE + 1] {
            let artifact = resized(10, actual);
            let mut frames = ArtifactFrames::new("session".to_string(), &artifact).unwrap();
            let requests: Vec<AddArtifactsRequest> = frames.by_ref().collect();
            take_frame_failure(&frames.failure()).unwrap();
            let crc = compute_crc32(&vec![9u8; actual]);
            assert_eq!(
                framing(&requests),
                [format!("batch [jars/resized.jar={actual}:{crc:08x}]")]
            );
        }
    }

    #[test]
    fn an_artifact_that_cannot_be_opened_ends_the_frames_of_those_after_it() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts = vec![
            pinned_named(3),
            NamedArtifact::new(
                "files/missing".to_string(),
                Box::new(FileArtifact::new(dir.path().join("missing"))),
            ),
            pinned_named(4),
        ];
        let error = build_artifact_request_stream("session".to_string(), artifacts).unwrap_err();
        assert!(
            error.to_string().contains("Failed to get artifact size"),
            "{error}"
        );
    }
}

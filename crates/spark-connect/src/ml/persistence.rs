// SPDX-License-Identifier: Apache-2.0
//! The persistence of PySpark's Python-format ML objects over Spark Connect, of the meta-algorithms
//! (`Pipeline`, `PipelineModel`, the validators), as `pyspark.ml.util` saves and loads them:
//! - [`Metadata`], [`save_metadata`] and [`load_metadata`]: the metadata of an object, which
//!   `DefaultParamsWriter.saveMetadata` writes as a text file of one line at `<path>/metadata` (a
//!   DataFrame of one row) and `DefaultParamsReader.loadMetadata` reads;
//! - [`stage_path`]: the path of a stage of a pipeline (`PipelineSharedReadWrite.getStagePath`);
//! - [`handle_overwrite`]: the deletion of a path by the ML helper before an overwrite, as
//!   `RemoteMLWriter.handleOverwrite` asks it.
//!
//! What these write is what PySpark writes, so that each loads what the other saves.
use super::*;
use crate::column::lit_string;
use spark_connect_core::error::Result;

/// The metadata of an object of PySpark's Python format, as `DefaultParamsWriter` makes it: the
/// fully qualified name of its Python class (PySpark's own, `pyspark.ml.pipeline.Pipeline`), the
/// time of the save, the Spark version of the server, the uid, and the params and default params,
/// as the JSON texts of their objects (of [`json_object`], in PySpark's order of their keys), with
/// any `extra` fields after them (each a key and the JSON text of its value).
#[derive(Debug, Clone, PartialEq)]
pub struct Metadata {
    pub class: String,
    pub timestamp: i64,
    pub spark_version: String,
    pub uid: String,
    pub param_map: String,
    pub default_param_map: String,
    pub extra: Vec<(String, String)>,
}

/// The compact JSON text of an object of the fields `fields` (each a key and the JSON text of its
/// value), in their order, as `json.dumps` writes a Python dict, of the order of its insertion.
pub fn json_object(fields: &[(&str, String)]) -> String {
    let body = fields
        .iter()
        .map(|(key, value)| format!("{}:{value}", serde_json::json!(key)))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{{body}}}")
}

impl Metadata {
    /// The metadata of an object of the class `class` and the uid `uid`, of `param_map` and
    /// `default_param_map` (the JSON texts of objects), of now and of the Spark version
    /// `spark_version`.
    pub fn new(
        class: &str,
        uid: &str,
        spark_version: &str,
        param_map: String,
        default_param_map: String,
    ) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or_default();
        Self {
            class: class.to_string(),
            timestamp,
            spark_version: spark_version.to_string(),
            uid: uid.to_string(),
            param_map,
            default_param_map,
            extra: Vec::new(),
        }
    }

    /// The JSON of the metadata, of its fields in the order of `DefaultParamsWriter`, compact as
    /// its `json.dumps(..., separators=(",", ":"))`.
    pub fn to_json(&self) -> String {
        let mut fields = vec![
            ("class", serde_json::json!(self.class).to_string()),
            ("timestamp", self.timestamp.to_string()),
            (
                "sparkVersion",
                serde_json::json!(self.spark_version).to_string(),
            ),
            ("uid", serde_json::json!(self.uid).to_string()),
            ("paramMap", self.param_map.clone()),
            ("defaultParamMap", self.default_param_map.clone()),
        ];
        fields.extend(
            self.extra
                .iter()
                .map(|(key, value)| (key.as_str(), value.clone())),
        );
        json_object(&fields)
    }
}

/// `os.path.join(path, name)` of a path of the server's file system, with `/`.
pub fn join_path(path: &str, name: &str) -> String {
    if path.ends_with('/') {
        format!("{path}{name}")
    } else {
        format!("{path}/{name}")
    }
}

/// Write the metadata `metadata` (its JSON) to `<path>/metadata`, as `saveMetadata` does: a text
/// file of one line, of a DataFrame of one row in one partition, of the default save mode, which
/// fails if the path exists.
pub fn save_metadata(session: &SparkSession, path: &str, metadata: &str) -> Result<()> {
    session
        .range(1)?
        .select([lit_string(metadata).alias("value")])
        .coalesce(1)
        .write()
        .text(&join_path(path, "metadata"))
}

/// The text of the metadata at `<path>/metadata`, as `loadMetadata` reads it: the first line of
/// the text files there, the JSON of the metadata as Python writes it (of `NaN` and `Infinity`
/// for those floats, which [`load_metadata`] does not parse).
pub fn load_metadata_text(session: &SparkSession, path: &str) -> Result<String> {
    let metadata_path = join_path(path, "metadata");
    let row = session
        .read()
        .text(&metadata_path)
        .first()?
        .ok_or_else(|| {
            SparkError::value_msg(format!("no metadata was found at {metadata_path}"))
        })?;
    row.get(0)
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            SparkError::value_msg(format!("the metadata at {metadata_path} is not a string"))
        })
}

/// The metadata at `<path>/metadata`, as `loadMetadata` reads it: the JSON of the first line of
/// the text files there (see [`load_metadata_text`]). When `expected_class` is given, a metadata
/// of another class is an error, as `_parseMetaData` raises it.
pub fn load_metadata(
    session: &SparkSession,
    path: &str,
    expected_class: Option<&str>,
) -> Result<serde_json::Value> {
    let metadata_path = join_path(path, "metadata");
    let text = load_metadata_text(session, path)?;
    let metadata: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| SparkError::value_msg(format!("the metadata at {metadata_path}: {e}")))?;
    if let Some(expected) = expected_class {
        let class = metadata["class"].as_str().unwrap_or_default();
        if class != expected {
            return Err(SparkError::value_msg(format!(
                "Error loading metadata: Expected class name {expected} but found class name \
                 {class}"
            )));
        }
    }
    Ok(metadata)
}

/// The path of the stage of the uid `stage_uid` of a pipeline of `num_stages` stages, at the
/// index `stage_idx`, under `stages_dir`, as `getStagePath`: the index, zero-padded to the digits
/// of the number of stages, and the uid.
pub fn stage_path(
    stage_uid: &str,
    stage_idx: usize,
    num_stages: usize,
    stages_dir: &str,
) -> String {
    let digits = num_stages.to_string().len();
    join_path(stages_dir, &format!("{stage_idx:0digits$}_{stage_uid}"))
}

/// Delete `path` on the server, as `RemoteMLWriter.handleOverwrite` asks the ML helper to before
/// it saves a meta-algorithm of overwrite (`handleOverwrite(path, true)`).
pub fn handle_overwrite(session: &SparkSession, path: &str) -> Result<()> {
    fetch(
        &MlTarget::helper(session),
        "handleOverwrite",
        vec![
            FetchArg::Value(MlValue::String(path.to_string())),
            FetchArg::Value(MlValue::Bool(true)),
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_is_the_json_of_default_params_writer_in_its_order() {
        let param_map = json_object(&[
            ("stageUids", serde_json::json!(["a", "b"]).to_string()),
            ("language", serde_json::json!("Python").to_string()),
        ]);
        let mut metadata = Metadata::new(
            "pyspark.ml.pipeline.Pipeline",
            "Pipeline_4866ebac8670",
            "4.3.0",
            param_map,
            "{}".to_string(),
        );
        metadata.timestamp = 1794647224405;
        assert_eq!(
            metadata.to_json(),
            "{\"class\":\"pyspark.ml.pipeline.Pipeline\",\"timestamp\":1794647224405,\
             \"sparkVersion\":\"4.3.0\",\"uid\":\"Pipeline_4866ebac8670\",\
             \"paramMap\":{\"stageUids\":[\"a\",\"b\"],\"language\":\"Python\"},\
             \"defaultParamMap\":{}}"
        );
        metadata.extra.push(("avgMetrics".into(), "[0.5]".into()));
        assert!(metadata.to_json().ends_with(",\"avgMetrics\":[0.5]}"));
    }

    #[test]
    fn stage_paths_pad_the_index_to_the_digits_of_the_number_of_stages() {
        assert_eq!(stage_path("A_1", 0, 3, "p/stages"), "p/stages/0_A_1");
        assert_eq!(stage_path("A_1", 7, 12, "p/stages/"), "p/stages/07_A_1");
        assert_eq!(stage_path("A_1", 10, 100, "s"), "s/010_A_1");
        assert_eq!(join_path("hdfs://h/p", "metadata"), "hdfs://h/p/metadata");
    }
}

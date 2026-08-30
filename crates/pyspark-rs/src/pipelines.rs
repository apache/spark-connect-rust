//! PyO3 bindings for Spark Declarative Pipelines (SDP) command execution. Exposed as
//! module-level `_pyspark` functions (not `SparkSession` methods, since the
//! `multiple-pymethods` feature is unavailable) which the thin
//! `pyspark.pipelines.spark_connect_*` glue calls.

use std::collections::HashMap;

use pyo3::prelude::*;

use spark_connect::pipelines::{
    self, OutputSpec, PipelineRunStream, SinkDetailsSpec, TableDetailsSpec,
};
use spark_connect_proto as proto;

use crate::column::PyColumn;
use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;
use crate::session::PySparkSession;
use crate::types::py_to_data_type;

/// Build a `proto::SourceCodeLocation` from the (filename, line) pair the Python
/// `SourceCodeLocation` carries.
fn scl(file: Option<String>, line: Option<i32>) -> Option<proto::SourceCodeLocation> {
    if file.is_none() && line.is_none() {
        return None;
    }
    Some(proto::SourceCodeLocation {
        file_name: file,
        line_number: line,
        definition_path: None,
        extension: vec![],
    })
}

/// `PipelineCommand.CreateDataflowGraph` → the new graph id.
#[pyfunction]
#[pyo3(signature = (session, default_catalog=None, default_database=None, sql_conf=None))]
pub fn pipeline_create_dataflow_graph(
    session: PyRef<'_, PySparkSession>,
    default_catalog: Option<String>,
    default_database: Option<String>,
    sql_conf: Option<HashMap<String, String>>,
) -> PyResult<String> {
    let sess = session.session.clone();
    session
        .py()
        .detach(|| {
            pipelines::create_dataflow_graph(
                &sess,
                default_catalog,
                default_database,
                sql_conf.unwrap_or_default(),
            )
        })
        .to_pyerr()
}

/// `PipelineCommand.DefineOutput`. `output_type` is the `spark.connect.OutputType` enum
/// value (1 MATERIALIZED_VIEW, 2 TABLE, 3 TEMPORARY_VIEW, 4 SINK).
#[pyfunction]
#[pyo3(signature = (
    session, dataflow_graph_id, output_name, output_type, comment=None,
    table_properties=None, partition_cols=None, clustering_columns=None, format=None,
    schema_string=None, schema_data_type=None, sink_options=None, sink_format=None,
    scl_file=None, scl_line=None
))]
#[allow(clippy::too_many_arguments)]
pub fn pipeline_define_output(
    session: PyRef<'_, PySparkSession>,
    dataflow_graph_id: &str,
    output_name: String,
    output_type: i32,
    comment: Option<String>,
    table_properties: Option<HashMap<String, String>>,
    partition_cols: Option<Vec<String>>,
    clustering_columns: Option<Vec<String>>,
    format: Option<String>,
    schema_string: Option<String>,
    schema_data_type: Option<Bound<'_, PyAny>>,
    sink_options: Option<HashMap<String, String>>,
    sink_format: Option<String>,
    scl_file: Option<String>,
    scl_line: Option<i32>,
) -> PyResult<()> {
    // OutputType: 1 MATERIALIZED_VIEW, 2 TABLE carry table details; 4 SINK carries sink details.
    let table_details = if output_type == 1 || output_type == 2 {
        let schema_dt = match schema_data_type {
            Some(obj) => Some(py_to_data_type(&obj)?),
            None => None,
        };
        Some(TableDetailsSpec {
            table_properties: table_properties.unwrap_or_default(),
            partition_cols: partition_cols.unwrap_or_default(),
            clustering_columns: clustering_columns.unwrap_or_default(),
            format,
            schema_string,
            schema_data_type: schema_dt,
        })
    } else {
        None
    };
    let sink_details = if output_type == 4 {
        Some(SinkDetailsSpec {
            options: sink_options.unwrap_or_default(),
            format: sink_format,
        })
    } else {
        None
    };
    let spec = OutputSpec {
        name: output_name,
        output_type,
        comment,
        source_code_location: scl(scl_file, scl_line),
        table_details,
        sink_details,
    };
    let sess = session.session.clone();
    let gid = dataflow_graph_id.to_string();
    session
        .py()
        .detach(|| pipelines::define_output(&sess, &gid, spec))
        .to_pyerr()
}

/// `PipelineCommand.DefineFlow` with a relation body (query-defined flow).
#[pyfunction]
#[pyo3(signature = (session, dataflow_graph_id, flow_name, target, relation, sql_conf=None, scl_file=None, scl_line=None))]
#[allow(clippy::too_many_arguments)]
pub fn pipeline_define_flow(
    session: PyRef<'_, PySparkSession>,
    dataflow_graph_id: &str,
    flow_name: &str,
    target: &str,
    relation: PyRef<'_, PyDataFrame>,
    sql_conf: Option<HashMap<String, String>>,
    scl_file: Option<String>,
    scl_line: Option<i32>,
) -> PyResult<()> {
    let sess = session.session.clone();
    let df = relation.dataframe.clone();
    let (gid, fname, tgt) = (
        dataflow_graph_id.to_string(),
        flow_name.to_string(),
        target.to_string(),
    );
    let scl = scl(scl_file, scl_line);
    session
        .py()
        .detach(|| {
            pipelines::define_flow(
                &sess,
                &gid,
                &fname,
                &tgt,
                &df,
                sql_conf.unwrap_or_default(),
                scl,
            )
        })
        .to_pyerr()
}

/// `PipelineCommand.DefineFlow` carrying an Auto-CDC flow body. `stored_as_scd_type` is
/// the `SCDType` enum value (0 unspecified, 1 SCD_TYPE_1, 2 SCD_TYPE_2).
#[pyfunction]
#[pyo3(signature = (
    session, dataflow_graph_id, flow_name, target, source, keys, sequence_by,
    apply_as_deletes=None, column_list=None, except_column_list=None,
    stored_as_scd_type=0, scl_file=None, scl_line=None
))]
#[allow(clippy::too_many_arguments)]
pub fn pipeline_define_auto_cdc_flow(
    session: PyRef<'_, PySparkSession>,
    dataflow_graph_id: &str,
    flow_name: &str,
    target: &str,
    source: String,
    keys: Vec<PyRef<'_, PyColumn>>,
    sequence_by: PyRef<'_, PyColumn>,
    apply_as_deletes: Option<PyRef<'_, PyColumn>>,
    column_list: Option<Vec<PyRef<'_, PyColumn>>>,
    except_column_list: Option<Vec<PyRef<'_, PyColumn>>>,
    stored_as_scd_type: i32,
    scl_file: Option<String>,
    scl_line: Option<i32>,
) -> PyResult<()> {
    let to_expr = |c: &PyColumn| c.column.expression().to_proto();
    let details = proto::pipeline_command::define_flow::AutoCdcFlowDetails {
        source: Some(source),
        keys: keys.iter().map(|c| to_expr(c)).collect(),
        sequence_by: Some(to_expr(&sequence_by)),
        apply_as_deletes: apply_as_deletes.as_ref().map(|c| to_expr(c)),
        apply_as_truncates: None,
        column_list: column_list
            .unwrap_or_default()
            .iter()
            .map(|c| to_expr(c))
            .collect(),
        except_column_list: except_column_list
            .unwrap_or_default()
            .iter()
            .map(|c| to_expr(c))
            .collect(),
        stored_as_scd_type,
        track_history_column_list: vec![],
        track_history_except_column_list: vec![],
        ignore_null_updates_column_list: vec![],
        ignore_null_updates_except_column_list: vec![],
    };
    let sess = session.session.clone();
    let (gid, fname, tgt) = (
        dataflow_graph_id.to_string(),
        flow_name.to_string(),
        target.to_string(),
    );
    let scl = scl(scl_file, scl_line);
    session
        .py()
        .detach(|| pipelines::define_auto_cdc_flow(&sess, &gid, &fname, &tgt, details, scl))
        .to_pyerr()
}

/// `PipelineCommand.DefineSqlGraphElements`.
#[pyfunction]
pub fn pipeline_define_sql_graph_elements(
    session: PyRef<'_, PySparkSession>,
    dataflow_graph_id: &str,
    sql_text: &str,
    sql_file_path: &str,
) -> PyResult<()> {
    let sess = session.session.clone();
    let (gid, text, path) = (
        dataflow_graph_id.to_string(),
        sql_text.to_string(),
        sql_file_path.to_string(),
    );
    session
        .py()
        .detach(|| pipelines::define_sql_graph_elements(&sess, &gid, &text, &path))
        .to_pyerr()
}

/// A live, lazily-consumed stream of a pipeline run's events. Iterating yields
/// `(message: str, timestamp_micros: int)` per pipeline event; command-result responses
/// are skipped; a server-side failure is raised (mapped to the pyspark exception type,
/// e.g. `AnalysisException`) during iteration.
#[pyclass(
    name = "PipelineRunStream",
    module = "pyspark.pipelines.spark_connect_pipeline"
)]
pub struct PyPipelineRunStream {
    inner: PipelineRunStream,
}

#[pymethods]
impl PyPipelineRunStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(String, i64)>> {
        loop {
            let resp = py.detach(|| self.inner.next_response()).to_pyerr()?;
            let Some(resp) = resp else {
                return Ok(None); // stream complete -> StopIteration
            };
            match resp.response_type {
                Some(proto::execute_plan_response::ResponseType::PipelineEventResult(ev)) => {
                    let event = ev.event.unwrap_or_default();
                    let message = event.message.unwrap_or_default();
                    let micros = event
                        .timestamp
                        .map(|t| t.seconds * 1_000_000 + (t.nanos as i64) / 1_000)
                        .unwrap_or(0);
                    return Ok(Some((message, micros)));
                }
                // Command-result (e.g. the StartRun ack) and anything else: skip.
                _ => continue,
            }
        }
    }
}

/// `PipelineCommand.StartRun` → a live event stream.
#[pyfunction]
#[pyo3(signature = (session, dataflow_graph_id, full_refresh=None, full_refresh_all=false, refresh=None, dry=false, storage=None))]
#[allow(clippy::too_many_arguments)]
pub fn pipeline_start_run(
    session: PyRef<'_, PySparkSession>,
    dataflow_graph_id: &str,
    full_refresh: Option<Vec<String>>,
    full_refresh_all: bool,
    refresh: Option<Vec<String>>,
    dry: bool,
    storage: Option<String>,
) -> PyResult<PyPipelineRunStream> {
    let sess = session.session.clone();
    let gid = dataflow_graph_id.to_string();
    let inner = session
        .py()
        .detach(|| {
            pipelines::start_run(
                &sess,
                &gid,
                full_refresh.unwrap_or_default(),
                full_refresh_all,
                refresh.unwrap_or_default(),
                dry,
                storage,
            )
        })
        .to_pyerr()?;
    Ok(PyPipelineRunStream { inner })
}

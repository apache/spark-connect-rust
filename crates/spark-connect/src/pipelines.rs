//! Spark Declarative Pipelines (SDP) command execution.
//!
//! Mirrors the Connect client's `pyspark.pipelines.spark_connect_pipeline` +
//! `spark_connect_graph_element_registry`, which build `PipelineCommand` protos and
//! execute them via the low-level client. Here the proto construction and execution
//! live in Rust; the thin Python `pyspark.pipelines` connect glue calls these instead
//! of assembling protos itself.

use std::collections::HashMap;

use spark_connect_core::client::ReattachableResponseStream;
use spark_connect_core::error::{Result, SparkError};
use spark_connect_core::runtime::block_on;
use spark_connect_proto as proto;

use crate::dataframe::{assign_plan_ids, execute_command_collect, DataFrame};
use crate::session::SparkSession;
use crate::types::DataType;

use proto::pipeline_command;
use proto::pipeline_command::{
    define_flow, define_output, CreateDataflowGraph, DefineFlow, DefineOutput,
    DefineSqlGraphElements, StartRun,
};

/// The details needed to define one pipeline output (table/materialized-view/view/sink).
pub struct OutputSpec {
    pub name: String,
    /// `spark.connect.OutputType` enum value (MATERIALIZED_VIEW=1, TABLE=2, TEMPORARY_VIEW=3, SINK=4).
    pub output_type: i32,
    pub comment: Option<String>,
    pub source_code_location: Option<proto::SourceCodeLocation>,
    /// Present for TABLE/MATERIALIZED_VIEW.
    pub table_details: Option<TableDetailsSpec>,
    /// Present for SINK.
    pub sink_details: Option<SinkDetailsSpec>,
}

/// Table/materialized-view details.
pub struct TableDetailsSpec {
    pub table_properties: HashMap<String, String>,
    pub partition_cols: Vec<String>,
    pub clustering_columns: Vec<String>,
    pub format: Option<String>,
    /// Schema as a DDL string, or a structured type, or neither.
    pub schema_string: Option<String>,
    pub schema_data_type: Option<DataType>,
}

/// External-sink details.
pub struct SinkDetailsSpec {
    pub options: HashMap<String, String>,
    pub format: Option<String>,
}

fn wrap(cmd: pipeline_command::CommandType) -> proto::command::CommandType {
    proto::command::CommandType::PipelineCommand(proto::PipelineCommand {
        command_type: Some(cmd),
    })
}

/// `PipelineCommand.CreateDataflowGraph` — returns the server-assigned graph id.
pub fn create_dataflow_graph(
    session: &SparkSession,
    default_catalog: Option<String>,
    default_database: Option<String>,
    sql_conf: HashMap<String, String>,
) -> Result<String> {
    let inner = CreateDataflowGraph {
        default_catalog,
        default_database,
        sql_conf,
    };
    let responses = execute_command_collect(
        session,
        wrap(pipeline_command::CommandType::CreateDataflowGraph(inner)),
    )?;
    for resp in responses {
        if let Some(proto::execute_plan_response::ResponseType::PipelineCommandResult(r)) =
            resp.response_type
        {
            if let Some(proto::pipeline_command_result::ResultType::CreateDataflowGraphResult(g)) =
                r.result_type
            {
                if let Some(id) = g.dataflow_graph_id {
                    return Ok(id);
                }
            }
        }
    }
    Err(SparkError::connect_msg(
        "CreateDataflowGraph did not return a dataflow_graph_id",
    ))
}

/// `PipelineCommand.DefineOutput` — register a table/view/sink in the graph.
pub fn define_output(
    session: &SparkSession,
    dataflow_graph_id: &str,
    spec: OutputSpec,
) -> Result<()> {
    let details = if let Some(t) = spec.table_details {
        let schema = match (t.schema_data_type, t.schema_string) {
            (Some(dt), _) => Some(define_output::table_details::Schema::SchemaDataType(
                dt.to_proto(),
            )),
            (None, Some(s)) => Some(define_output::table_details::Schema::SchemaString(s)),
            (None, None) => None,
        };
        Some(define_output::Details::TableDetails(
            define_output::TableDetails {
                table_properties: t.table_properties,
                partition_cols: t.partition_cols,
                clustering_columns: t.clustering_columns,
                format: t.format,
                schema,
            },
        ))
    } else {
        spec.sink_details.map(|s| {
            define_output::Details::SinkDetails(define_output::SinkDetails {
                options: s.options,
                format: s.format,
            })
        })
    };
    let inner = DefineOutput {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        output_name: Some(spec.name),
        output_type: Some(spec.output_type),
        comment: spec.comment,
        source_code_location: spec.source_code_location,
        details,
    };
    execute_command_collect(
        session,
        wrap(pipeline_command::CommandType::DefineOutput(inner)),
    )?;
    Ok(())
}

/// `PipelineCommand.DefineFlow` with a relation body — the common query-defined flow.
pub fn define_flow(
    session: &SparkSession,
    dataflow_graph_id: &str,
    flow_name: &str,
    target_dataset_name: &str,
    relation_df: &DataFrame,
    sql_conf: HashMap<String, String>,
    source_code_location: Option<proto::SourceCodeLocation>,
) -> Result<()> {
    let mut relation = relation_df.plan.to_proto();
    assign_plan_ids(&mut relation, &relation_df.session)?;
    let details =
        define_flow::Details::RelationFlowDetails(define_flow::WriteRelationFlowDetails {
            relation: Some(relation),
        });
    let inner = DefineFlow {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        flow_name: Some(flow_name.to_string()),
        target_dataset_name: Some(target_dataset_name.to_string()),
        sql_conf,
        client_id: None,
        source_code_location,
        once: None,
        details: Some(details),
    };
    execute_command_collect(
        session,
        wrap(pipeline_command::CommandType::DefineFlow(inner)),
    )?;
    Ok(())
}

/// `PipelineCommand.DefineFlow` carrying an Auto-CDC flow body.
#[allow(clippy::too_many_arguments)]
pub fn define_auto_cdc_flow(
    session: &SparkSession,
    dataflow_graph_id: &str,
    flow_name: &str,
    target_dataset_name: &str,
    details: define_flow::AutoCdcFlowDetails,
    source_code_location: Option<proto::SourceCodeLocation>,
) -> Result<()> {
    let inner = DefineFlow {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        flow_name: Some(flow_name.to_string()),
        target_dataset_name: Some(target_dataset_name.to_string()),
        sql_conf: HashMap::new(),
        client_id: None,
        source_code_location,
        once: None,
        details: Some(define_flow::Details::AutoCdcFlowDetails(details)),
    };
    execute_command_collect(
        session,
        wrap(pipeline_command::CommandType::DefineFlow(inner)),
    )?;
    Ok(())
}

/// `PipelineCommand.DefineSqlGraphElements` — register raw SQL definitions.
pub fn define_sql_graph_elements(
    session: &SparkSession,
    dataflow_graph_id: &str,
    sql_text: &str,
    sql_file_path: &str,
) -> Result<()> {
    let inner = DefineSqlGraphElements {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        sql_file_path: Some(sql_file_path.to_string()),
        sql_text: Some(sql_text.to_string()),
    };
    execute_command_collect(
        session,
        wrap(pipeline_command::CommandType::DefineSqlGraphElements(inner)),
    )?;
    Ok(())
}

/// A lazily-consumed stream of a pipeline run's responses (`PipelineCommandResult` and
/// `PipelineEventResult`), so callers observe events — and any server error — as they
/// arrive, matching `handle_pipeline_events` iterating `execute_command_as_iterator`.
///
/// The gRPC stream is opened on the FIRST `next_response`, not at construction, so that a
/// fail-fast validation error (e.g. a dry run rejecting an invalid graph) surfaces during
/// iteration rather than from `start_run` itself — mirroring the reference client, whose
/// `execute_command_as_iterator` is a generator that starts the RPC lazily.
pub struct PipelineRunStream {
    session: SparkSession,
    request: Option<proto::ExecutePlanRequest>,
    stream: Option<ReattachableResponseStream>,
}

impl PipelineRunStream {
    /// The next response, or `None` when the run's stream completes. A server-side
    /// failure surfaces here as the mapped `SparkError`.
    pub fn next_response(&mut self) -> Result<Option<proto::ExecutePlanResponse>> {
        if self.stream.is_none() {
            let request = self
                .request
                .take()
                .expect("pipeline run stream already exhausted its request");
            self.stream = Some(block_on(
                self.session.client().execute_plan_reattachable(request),
            )?);
        }
        block_on(self.stream.as_mut().unwrap().message())
    }
}

/// `PipelineCommand.StartRun` — begin (or dry-run/validate) a pipeline run and return the
/// (lazily-opened) live response stream.
#[allow(clippy::too_many_arguments)]
pub fn start_run(
    session: &SparkSession,
    dataflow_graph_id: &str,
    full_refresh_selection: Vec<String>,
    full_refresh_all: bool,
    refresh_selection: Vec<String>,
    dry: bool,
    storage: Option<String>,
) -> Result<PipelineRunStream> {
    let inner = StartRun {
        dataflow_graph_id: Some(dataflow_graph_id.to_string()),
        full_refresh_selection,
        full_refresh_all: Some(full_refresh_all),
        refresh_selection,
        dry: Some(dry),
        storage,
    };
    let mut command = proto::Command::default();
    command.command_type = Some(wrap(pipeline_command::CommandType::StartRun(inner)));
    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Command(command));
    let mut request = proto::ExecutePlanRequest::default();
    request.session_id = session.client().session_id().to_string();
    request.user_context = Some(proto::UserContext::default());
    request.tags = session.tags();
    request.plan = Some(plan);
    Ok(PipelineRunStream {
        session: session.clone(),
        request: Some(request),
        stream: None,
    })
}

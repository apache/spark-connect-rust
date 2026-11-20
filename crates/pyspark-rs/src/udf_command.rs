//! The commands of the Python UDFs and UDTFs, which a request pickles when it builds the plan.
//!
//! PySpark's `PythonUDF.to_plan` and `PythonUDTF.to_plan` pickle the function of a UDF, of a
//! pandas UDF, of a UDTF and of the map and group-map functions each time they build a plan that
//! holds it: so a global that the function reads is the value it has when the request is built,
//! and a function that does not pickle raises the error of its pickling from the first request,
//! such as `df.schema`, rather than from the call that applies it. The core calls the
//! [`CommandPickler`] of such a command when it builds the plan of a request, on the stack of the
//! caller of the request, outside the recursion of the build (see `spark_connect::expression`),
//! and the pickling attaches to the interpreter there, as a hook of a session does: a request
//! runs without the GIL, and the locks that the core holds while it builds a plan, such as that of
//! the schema of a DataFrame, are taken by the binding without the GIL too. The error of a
//! pickling is the exception that Python raised, as [`crate::errors::callback_error`] carries it,
//! which the request raises as it is.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use spark_connect::udf::{CommandPickler, PicklerCopies};

/// The command of a UDTF as the shims give it: the bytes of its pickle, which are sent as they
/// are, or the pickler of a function of no arguments that returns the `PythonUDTF` message of its
/// return type and its command, which a request calls each time it builds the plan.
pub(crate) enum UdfCommand {
    Pickled(Vec<u8>),
    AtBuild(CommandPickler),
}

impl UdfCommand {
    /// The command of `command`: bytes, or a function that converts the return type and pickles
    /// the command (see [`udtf_pickler`]).
    pub(crate) fn of(command: &Bound<'_, PyAny>) -> UdfCommand {
        match command.cast::<PyBytes>() {
            Ok(bytes) => UdfCommand::Pickled(bytes.as_bytes().to_vec()),
            Err(_) => UdfCommand::AtBuild(udtf_pickler(command.clone().unbind())),
        }
    }
}

/// The object of the pickler of a command of the binding, which the garbage collector sees
/// through the DataFrame or the Column that holds it, and which a deep copy of them copies.
struct PythonCommand {
    object: Py<PyAny>,
    kind: CommandKind,
}

/// What the object of a [`PythonCommand`] is.
#[derive(Clone, Copy)]
enum CommandKind {
    /// A Python function of no arguments that returns the `PythonUDTF` message of the return
    /// type and the command of a UDTF, as the shim's `_udtf_command` builds it.
    Udtf,
    /// The `(func, output_type)` of a UDF, which `CloudPickleSerializer().dumps` pickles, after
    /// the conversion of the `output_type` that the request sends with it, as PySpark's
    /// `PythonUDF.to_plan` does.
    Udf,
    /// A data type, which is converted to its `spark.connect.DataType`.
    DataType,
}

impl PythonCommand {
    /// The pickler of this command.
    fn pickler(self) -> CommandPickler {
        match self.kind {
            CommandKind::DataType => CommandPickler::new(self, PythonCommand::pickle),
            CommandKind::Udtf | CommandKind::Udf => {
                CommandPickler::with_output_type(self, PythonCommand::pickle)
            }
        }
    }

    /// Of a UDTF, the `PythonUdtf` message that the function returns; of a UDF, the `PythonUdf`
    /// message of its output type, converted first, and of the pickle of the object, as PySpark's
    /// `PythonUDF.to_plan` converts `pyspark_types_to_proto_types(output_type)` before it pickles
    /// `(func, output_type)`, of the type as it is then; of a data type, its message. Its error is
    /// the exception itself ([`crate::errors::callback_error`]), which the request raises. A
    /// pickle of a DataFrame, a Column or a table argument calls no such pickler: it carries the
    /// object instead ([`PickleHoles`]).
    fn pickle(&self) -> spark_connect_core::error::Result<Vec<u8>> {
        Python::attach(|py| self.pickled(py).map_err(crate::errors::callback_error))
    }

    fn pickled(&self, py: Python<'_>) -> PyResult<Vec<u8>> {
        let object = self.object.bind(py);
        match self.kind {
            CommandKind::Udtf => command_of(&object.call0()?),
            CommandKind::Udf => {
                use prost::Message;
                let output_type =
                    crate::types::py_to_proto_data_type(&object.get_item(1)?)?.to_proto();
                let command = crate::dataframe::py_cloudpickle(py, object)?;
                Ok(spark_connect_proto::PythonUdf {
                    output_type: Some(output_type),
                    command: command.into(),
                    ..Default::default()
                }
                .encode_to_vec())
            }
            CommandKind::DataType => {
                use prost::Message;
                Ok(crate::types::py_to_proto_data_type(object)?
                    .to_proto()
                    .encode_to_vec())
            }
        }
    }
}

/// The pickler of `udtf`, a Python function of no arguments that returns the `PythonUDTF`
/// message of the return type and the command of a UDTF, as PySpark's `PythonUDTF.to_plan`
/// builds them each time it builds the plan, of the type object that the call kept: the return
/// type that the request sends is that of the object then, so a `StructType` changed in place
/// after the call is sent as it is changed, and a type that `pyspark_types_to_proto_types` does
/// not convert fails the request with its error, before the handler class is pickled, whose
/// error `UDTF_SERIALIZATION_ERROR` comes next.
fn udtf_pickler(udtf: Py<PyAny>) -> CommandPickler {
    PythonCommand {
        object: udtf,
        kind: CommandKind::Udtf,
    }
    .pickler()
}

/// The pickler of `udf`, the `(func, output_type)` of a UDF, of a pandas or Arrow UDF or of a map
/// or group-map function, which a request converts and pickles each time it builds the plan, as
/// PySpark's `PythonUDF.to_plan` does of the `output_type` and the function that it keeps: the
/// output type that the request sends is that of `output_type` then, which the command carries
/// too, so a `StructType` changed in place after the call is sent as it is changed, and a type
/// that `pyspark_types_to_proto_types` has no proto of fails the request with its
/// `UNSUPPORTED_OPERATION`, before the function is pickled.
pub(crate) fn udf_pickler(udf: Py<PyAny>) -> CommandPickler {
    PythonCommand {
        object: udf,
        kind: CommandKind::Udf,
    }
    .pickler()
}

/// The reader of `data_type`, a type object, which a request converts to its
/// `spark.connect.DataType` each time it builds the plan, as PySpark's `ToSchema` converts the
/// schema of `DataFrame.to` then: a `StructType` changed in place after the call is sent as it is
/// then, and a type that `pyspark_types_to_proto_types` has no proto of fails the request with its
/// `UNSUPPORTED_OPERATION`.
pub(crate) fn data_type_pickler(data_type: Py<PyAny>) -> CommandPickler {
    PythonCommand {
        object: data_type,
        kind: CommandKind::DataType,
    }
    .pickler()
}

/// The copies of `picklers` of a deep copy of a Column, a table argument or a DataFrame: the
/// pickler of `copy.deepcopy` of the object of each, with `memo`, in turn, as PySpark's deep copy
/// of their expressions and plans deep-copies the function of each UDF, of its `PythonUDF`, with
/// the memo of the call. So a function, of which `copy.deepcopy` gives the function itself, is
/// shared with the original, and a callable object, a bound method and a `functools.partial` are
/// copies, which the copies of the call share.
pub(crate) fn deep_copied_picklers(
    py: Python<'_>,
    picklers: Vec<CommandPickler>,
    memo: &Bound<'_, PyAny>,
) -> PyResult<PicklerCopies> {
    let mut copies = PicklerCopies::new();
    if picklers.is_empty() {
        return Ok(copies);
    }
    let deepcopy = py.import("copy")?.getattr("deepcopy")?;
    for pickler in picklers {
        let Some(command) = pickler.object().downcast_ref::<PythonCommand>() else {
            continue;
        };
        let object = deepcopy.call1((command.object.bind(py), memo))?;
        let copy = PythonCommand {
            object: object.unbind(),
            kind: command.kind,
        };
        copies.insert(&pickler, copy.pickler());
    }
    Ok(copies)
}

/// The Python objects of the picklers of `objects`, of the UDFs of a DataFrame or a Column that
/// only it holds, which its `__traverse__` reports.
pub(crate) fn visit_pickler_objects(
    objects: Vec<&(dyn std::any::Any + Send + Sync)>,
    visit: &pyo3::PyVisit<'_>,
) -> Result<(), pyo3::PyTraverseError> {
    for object in objects {
        if let Some(command) = object.downcast_ref::<PythonCommand>() {
            visit.call(&command.object)?;
        }
    }
    Ok(())
}

/// Visit `owners`, owners of the relations of plans of the Python client and of the errors of the
/// expressions that a request fails to build, which a Column, a table argument, or another object
/// that holds Columns alone holds (`Expression::exclusive_owners`, `TableArg::exclusive_owners`):
/// the Python objects among them, the plan of a relation, or the tuple of the plan and the
/// exception of a kept error.
pub(crate) fn visit_owners(
    owners: Vec<&(dyn std::any::Any + Send + Sync)>,
    visit: &pyo3::PyVisit<'_>,
) -> Result<(), pyo3::PyTraverseError> {
    for owner in owners {
        if let Some(object) = owner.downcast_ref::<Py<PyAny>>() {
            visit.call(object)?;
        }
    }
    Ok(())
}

/// The bytes of `command`, the pickle of a command.
fn command_of(command: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    Ok(command.cast::<PyBytes>()?.as_bytes().to_vec())
}

/// The holes of the pickle of a DataFrame, a Column or a table argument, which carries the
/// objects of the picklers of its UDFs rather than their commands, as PySpark's pickle carries the
/// plan objects and so the functions of their UDFs, of which its requests pickle the commands: a
/// plain pickle pickles a function by reference, fails at a lambda, and carries no command, so the
/// unpickled object pickles the function as it is at each of its requests (see
/// `spark_connect::udf::SentHole`). The relation or the expression is built with copies of the
/// picklers that give markers in place of the commands ([`PickleHoles::of`]), in the order of the
/// build, which [`PickleHoles::holes_of`] gives with the objects.
pub(crate) struct PickleHoles {
    /// The markers that the copies gave, in the order of the build, each of the index of its
    /// pickler in `of`, and the maker of the markers.
    log: std::sync::Arc<std::sync::Mutex<HoleLog>>,
    /// The kind and the object of each pickler of a copy.
    of: Vec<(CommandKind, Py<PyAny>)>,
}

#[derive(Default)]
struct HoleLog {
    markers: spark_connect::udf::SentHoleMarkers,
    given: Vec<(Vec<u8>, usize)>,
}

/// The object of a copy of a pickler of [`PickleHoles`].
struct MarkerOf {
    log: std::sync::Arc<std::sync::Mutex<HoleLog>>,
    index: usize,
    kind: CommandKind,
    object: Py<PyAny>,
}

impl MarkerOf {
    /// The message of a new marker in place of the command, which the log keeps: of a UDF, a
    /// `PythonUdf` of the output type of the type object as it converts now, if it does, as the
    /// plan of PySpark's pickle carries the type then, which the plan sends where it sends the
    /// output type of the function, as a transformWithState sends it as its output schema; of a
    /// UDTF, a `PythonUdtf`; of a data type, a `DataType` of the marker as its unparsed string.
    fn give(&self) -> spark_connect_core::error::Result<Vec<u8>> {
        use prost::Message;
        let marker = {
            let mut log = self.log.lock().unwrap_or_else(|e| e.into_inner());
            let marker = log.markers.marker();
            log.given.push((marker.clone(), self.index));
            marker
        };
        Ok(match self.kind {
            CommandKind::Udf => {
                let output_type = Python::attach(|py| {
                    let object = self.object.bind(py);
                    object
                        .get_item(1)
                        .and_then(|output_type| crate::types::py_to_proto_data_type(&output_type))
                        .ok()
                        .map(|output_type| output_type.to_proto())
                });
                spark_connect_proto::PythonUdf {
                    output_type,
                    command: marker.into(),
                    ..Default::default()
                }
                .encode_to_vec()
            }
            CommandKind::Udtf => spark_connect_proto::PythonUdtf {
                command: marker.into(),
                ..Default::default()
            }
            .encode_to_vec(),
            CommandKind::DataType => spark_connect_proto::DataType {
                kind: Some(spark_connect_proto::data_type::Kind::Unparsed(
                    spark_connect_proto::data_type::Unparsed {
                        data_type_string: String::from_utf8_lossy(&marker).into_owned(),
                    },
                )),
            }
            .encode_to_vec(),
        })
    }
}

impl PickleHoles {
    /// The copies of the picklers of the binding among `picklers`, which give markers, for the
    /// build of a pickle.
    pub(crate) fn of(
        py: Python<'_>,
        picklers: Vec<CommandPickler>,
    ) -> (PickleHoles, PicklerCopies) {
        let log = std::sync::Arc::new(std::sync::Mutex::new(HoleLog::default()));
        let mut of = Vec::new();
        let mut copies = PicklerCopies::new();
        for pickler in picklers {
            let Some(command) = pickler.object().downcast_ref::<PythonCommand>() else {
                continue;
            };
            let marker = MarkerOf {
                log: std::sync::Arc::clone(&log),
                index: of.len(),
                kind: command.kind,
                object: command.object.clone_ref(py),
            };
            of.push((command.kind, command.object.clone_ref(py)));
            let copy = match command.kind {
                CommandKind::DataType => CommandPickler::new(marker, MarkerOf::give),
                CommandKind::Udf | CommandKind::Udtf => {
                    CommandPickler::with_output_type(marker, MarkerOf::give)
                }
            };
            copies.insert(&pickler, copy);
        }
        (PickleHoles { log, of }, copies)
    }

    /// The holes of `encoded`, a relation or an expression that the build of the pickle gave, in
    /// the order of the build: of each marker that it holds as the whole value of a field, as the
    /// fill of a request finds it (`spark_connect::udf::held_markers`), the tuple of the marker,
    /// the kind of the hole and the object that fills it, which the pickle carries (see
    /// [`sent_holes`]).
    pub(crate) fn holes_of<'py>(
        &self,
        py: Python<'py>,
        encoded: &[u8],
    ) -> PyResult<Bound<'py, pyo3::types::PyList>> {
        let holes = pyo3::types::PyList::empty(py);
        let given = self
            .log
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .given
            .clone();
        let markers: Vec<&[u8]> = given.iter().map(|(marker, _)| marker.as_slice()).collect();
        let held = spark_connect::udf::held_markers(encoded, &markers);
        for ((marker, index), held) in given.iter().zip(held) {
            if !held {
                continue;
            }
            let (kind, object) = &self.of[*index];
            let (name, payload) = hole_payload(py, *kind, object.bind(py))?;
            holes.append((PyBytes::new(py, marker), name, payload))?;
        }
        Ok(holes)
    }
}

/// The name of the kind of a hole of `kind` and the object that a pickle carries for it: of a UDF,
/// its `(func, returnType)`; of a UDTF of the shim's `_udtf_command` (of
/// `pyspark.sql.connect.udtf`), its arguments, the handler class, its name and the return type
/// object, as PySpark's `PythonUDTF` holds them, so that it loads where PySpark is, without the
/// shim; of another UDTF, its function; of a data type, the type object.
fn hole_payload<'py>(
    py: Python<'py>,
    kind: CommandKind,
    object: &Bound<'py, PyAny>,
) -> PyResult<(&'static str, Bound<'py, PyAny>)> {
    Ok(match kind {
        CommandKind::Udf => ("udf", object.clone()),
        CommandKind::DataType => ("datatype", object.clone()),
        CommandKind::Udtf => {
            let command = py
                .import("pyspark.sql.connect.udtf")?
                .getattr("_udtf_command")?;
            let partial = py.import("functools")?.getattr("partial")?;
            if object.is_instance(&partial)? && object.getattr("func")?.is(&command) {
                ("udtf", object.getattr("args")?)
            } else {
                ("udtf-call", object.clone())
            }
        }
    })
}

/// The holes of `holes`, the tuples of the marker, the kind and the object of each hole of an
/// unpickled DataFrame, Column or table argument ([`PickleHoles::holes_of`]), of the picklers of
/// their objects, which the requests of the unpickled object call, as those of the object that
/// was pickled.
pub(crate) fn sent_holes(
    holes: Option<&Bound<'_, PyAny>>,
) -> PyResult<spark_connect::udf::SentHoles> {
    use spark_connect::udf::{SentHole, SentHoleKind, SentHoles};
    let Some(holes) = holes.filter(|holes| !holes.is_none()) else {
        return Ok(SentHoles::default());
    };
    let py = holes.py();
    let mut sent = Vec::new();
    for hole in holes.try_iter()? {
        let (marker, kind, payload): (Vec<u8>, String, Bound<'_, PyAny>) = hole?.extract()?;
        let (kind, pickler) = match kind.as_str() {
            "udf" => (SentHoleKind::PythonUdf, udf_pickler(payload.unbind())),
            "datatype" => (SentHoleKind::DataType, data_type_pickler(payload.unbind())),
            "udtf" => {
                let command = py
                    .import("pyspark.sql.connect.udtf")?
                    .getattr("_udtf_command")?;
                let args = payload.cast::<pyo3::types::PyTuple>()?;
                let partial = py
                    .import("functools")?
                    .getattr("partial")?
                    .call1(PyTuple::new(
                        py,
                        std::iter::once(command)
                            .chain(args.iter())
                            .collect::<Vec<_>>(),
                    )?)?;
                (SentHoleKind::PythonUdtf, udtf_pickler(partial.unbind()))
            }
            "udtf-call" => (SentHoleKind::PythonUdtf, udtf_pickler(payload.unbind())),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "a hole of an unknown kind {other}"
                )))
            }
        };
        sent.push(SentHole {
            marker,
            kind,
            pickler: Some(pickler),
        });
    }
    Ok(SentHoles(sent))
}

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
use pyo3::types::PyBytes;
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
    /// that of [`pickle_error`].
    fn pickle(&self) -> spark_connect_core::error::Result<Vec<u8>> {
        Python::attach(|py| self.pickled(py).map_err(|error| pickle_error(py, error)))
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

/// The error of a request of `exception`, which the pickling of a command raised: the exception
/// itself ([`crate::errors::callback_error`]). But while a DataFrame, a Column or a table argument
/// is pickled ([`crate::origin::pickling`]), the pickle keeps an error of one of PySpark's error
/// classes other than a `PySparkPicklingError`, which the conversion of a type raises before the
/// function is pickled, such as the `UNSUPPORTED_OPERATION` of a type that has no proto, as the
/// error of its plan (see `crate::column::described_error`): PySpark's pickle keeps the type
/// object as it is, and the conversion fails at the requests of the unpickled object, or at the
/// unpickle of a DataFrame, which builds its plan. The pickle raises the error of a pickling.
fn pickle_error(py: Python<'_>, exception: PyErr) -> spark_connect_core::error::SparkError {
    if crate::origin::is_pickling() {
        if let Ok(Some(error)) = crate::errors::pyspark_class_error(py, &exception) {
            if error.kind != spark_connect_core::error::SparkErrorKind::PicklingError {
                return error;
            }
        }
    }
    crate::errors::callback_error(exception)
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

/// The bytes of `command`, the pickle of a command.
fn command_of(command: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    Ok(command.cast::<PyBytes>()?.as_bytes().to_vec())
}

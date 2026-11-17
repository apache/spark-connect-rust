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
use spark_connect::udf::CommandPickler;

/// The command of a UDF or UDTF as the shims give it: the bytes of its pickle, which are sent as
/// they are, or the pickler of a function of no arguments that returns them, which a request calls
/// each time it builds the plan.
pub(crate) enum UdfCommand {
    Pickled(Vec<u8>),
    AtBuild(CommandPickler),
}

impl UdfCommand {
    /// The command of `command`: bytes, or a function that pickles the command.
    pub(crate) fn of(command: &Bound<'_, PyAny>) -> UdfCommand {
        match command.cast::<PyBytes>() {
            Ok(bytes) => UdfCommand::Pickled(bytes.as_bytes().to_vec()),
            Err(_) => UdfCommand::AtBuild(function_pickler(command.clone().unbind())),
        }
    }
}

/// The pickler of `pickle`, a Python function of no arguments that returns the command, as the
/// shims pickle the function of a UDF with `CloudPickleSerializer().dumps`, and that of a UDTF as
/// `PythonUDTF.to_plan` does, of its error `UDTF_SERIALIZATION_ERROR`. Its object is `pickle`,
/// which the garbage collector sees through the DataFrame or the Column that holds it.
fn function_pickler(pickle: Py<PyAny>) -> CommandPickler {
    CommandPickler::new(pickle, |pickle: &Py<PyAny>| {
        Python::attach(|py| command_of(&pickle.bind(py).call0()?))
            .map_err(crate::errors::callback_error)
    })
}

/// The pickler of `object`, which a request pickles with `CloudPickleSerializer().dumps`, as
/// PySpark's `PythonUDF.to_plan` pickles the `(func, output_type)` of a map or group-map function.
pub(crate) fn object_pickler(object: Py<PyAny>) -> CommandPickler {
    CommandPickler::new(object, |object: &Py<PyAny>| {
        Python::attach(|py| crate::dataframe::py_cloudpickle(py, object.bind(py)))
            .map_err(crate::errors::callback_error)
    })
}

/// The Python objects of the picklers of `objects`, of the UDFs of a DataFrame or a Column that
/// only it holds, which its `__traverse__` reports.
pub(crate) fn visit_pickler_objects(
    objects: Vec<&(dyn std::any::Any + Send + Sync)>,
    visit: &pyo3::PyVisit<'_>,
) -> Result<(), pyo3::PyTraverseError> {
    for object in objects {
        if let Some(object) = object.downcast_ref::<Py<PyAny>>() {
            visit.call(object)?;
        }
    }
    Ok(())
}

/// The bytes of `command`, the pickle of a command.
fn command_of(command: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    Ok(command.cast::<PyBytes>()?.as_bytes().to_vec())
}

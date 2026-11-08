//! As PySpark's `_with_origin`: the origin in user code of the expressions that an API function
//! builds, which the server reports in the `DataFrameQueryContext` of an error in them.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use spark_connect::origin::{set_current, Origin, OriginGuard};

static UTILS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
static SYS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
/// The directory of the `pyspark` package, as `os.path.dirname(pyspark.__file__)`.
static PYSPARK_ROOT: PyOnceLock<String> = PyOnceLock::new();
/// Whether the imports of IPython and ipykernel were tried, once per process. A failed import
/// is not retried, and a module imported later is still seen in `sys.modules`.
static IPYTHON_IMPORTS_TRIED: AtomicBool = AtomicBool::new(false);

/// While the returned guard lives, the expressions that this thread builds have the origin of
/// `fragment`, the name of the API function, at its call site in user code, when DataFrame
/// debugging is enabled. As PySpark's `_with_origin`, with the functions of
/// `pyspark.errors.utils` that it calls: `is_debugging_enabled`, which reads
/// `spark.python.sql.dataFrameDebugging.enabled` once, and `_capture_call_site`, which
/// `call_site` gives without importing a module per call. An origin that cannot be read is
/// not set, as an expression is the same without one.
pub(crate) fn with_origin(fragment: &str) -> Option<OriginGuard> {
    Python::attach(|py| -> PyResult<Option<OriginGuard>> {
        let utils = UTILS
            .get_or_try_init(py, || py.import("pyspark.errors.utils").map(Bound::unbind))?
            .bind(py);
        if !utils.call_method0("is_debugging_enabled")?.is_truthy()? {
            return Ok(None);
        }
        let call_site = call_site(py, 1)?;
        Ok(Some(set_current(Origin {
            fragment: fragment.to_string(),
            call_site,
        })))
    })
    .unwrap_or(None)
}

/// The call site of the first `depth` frames of user code, as `_capture_call_site(depth)` of
/// `pyspark.errors.utils` gives it: the frames outside the pyspark package, and within an
/// IPython shell (when IPython and ipykernel are loaded) without the frames of IPython and
/// ipykernel, each "line N in cell [K]", else "file:line", joined by newlines.
fn call_site(py: Python<'_>, depth: usize) -> PyResult<String> {
    let sys = SYS
        .get_or_try_init(py, || py.import("sys").map(Bound::unbind))?
        .bind(py);
    let pyspark_root = PYSPARK_ROOT.get_or_try_init(py, || {
        let file = py.import("pyspark")?.getattr("__file__")?;
        py.import("os")?
            .getattr("path")?
            .call_method1("dirname", (file,))?
            .extract::<String>()
    })?;

    // The frames as `inspect` walks them from the innermost: `sys._getframe(0)` is the Python
    // frame that called the API function, as the native code of it adds no frame.
    let mut selected = Vec::with_capacity(depth);
    let mut frame = sys.call_method1("_getframe", (0,))?;
    while selected.len() < depth && !frame.is_none() {
        let file: String = frame.getattr("f_code")?.getattr("co_filename")?.extract()?;
        if !file.contains(pyspark_root.as_str()) {
            selected.push((file, frame.getattr("f_lineno")?));
        }
        frame = frame.getattr("f_back")?;
    }

    let Some((ipython, ipykernel)) = loaded_ipython(py, sys)? else {
        return join_frames(&selected);
    };
    // The frames of IPython and ipykernel are dropped after the first `depth` frames, as
    // `_capture_call_site` does.
    let os_path = py.import("os")?.getattr("path")?;
    let ipy_root: String = os_path
        .call_method1("dirname", (ipython.getattr("__file__")?,))?
        .extract()?;
    let kernel_root: String = os_path
        .call_method1("dirname", (ipykernel.getattr("__file__")?,))?
        .extract()?;
    let frames: Vec<_> = selected
        .into_iter()
        .filter(|(file, _)| {
            !file.contains(ipy_root.as_str()) && !file.contains(kernel_root.as_str())
        })
        .collect();
    let shell = ipython.call_method0("get_ipython")?;
    if !shell.is_truthy()? {
        return join_frames(&frames);
    }
    // Identifying the cell is useful when the error is generated from IPython Notebook.
    let count = shell.getattr("execution_count")?;
    let cells: Vec<String> = frames
        .iter()
        .map(|(_, lineno)| Ok(format!("line {lineno} in cell [{count}]")))
        .collect::<PyResult<_>>()?;
    Ok(cells.join("\n"))
}

/// The "file:line" of each frame, joined by newlines.
fn join_frames(frames: &[(String, Bound<'_, PyAny>)]) -> PyResult<String> {
    let lines: Vec<String> = frames
        .iter()
        .map(|(file, lineno)| Ok(format!("{file}:{lineno}")))
        .collect::<PyResult<_>>()?;
    Ok(lines.join("\n"))
}

/// The modules IPython and ipykernel, when both are loaded. They are read from `sys.modules`,
/// and imported once per process when one is missing, as `_capture_call_site` imports them.
fn loaded_ipython<'py>(
    py: Python<'py>,
    sys: &Bound<'py, PyModule>,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let modules = sys.getattr("modules")?;
    let modules = modules.cast::<PyDict>()?;
    let find = |name: &str| modules.get_item(name);
    let (mut ipython, mut ipykernel) = (find("IPython")?, find("ipykernel")?);
    if (ipython.is_none() || ipykernel.is_none())
        && !IPYTHON_IMPORTS_TRIED.swap(true, Ordering::Relaxed)
    {
        // `import IPython`, then `import ipykernel`, as `_capture_call_site` does.
        let _ = py.import("IPython").and_then(|_| py.import("ipykernel"));
        (ipython, ipykernel) = (find("IPython")?, find("ipykernel")?);
    }
    Ok(ipython.zip(ipykernel))
}

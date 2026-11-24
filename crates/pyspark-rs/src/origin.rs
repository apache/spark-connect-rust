//! As PySpark's `_with_origin`: the origin in user code of the expressions that an API function
//! builds, which the server reports in the `DataFrameQueryContext` of an error in them.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use spark_connect::origin::{set_current, Origin, OriginGuard};

static UTILS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
/// `pyspark.errors.utils.is_debugging_enabled` as the module first gave it.
static IS_DEBUGGING_ENABLED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static SYS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
/// The directory of the `pyspark` package, as `os.path.dirname(pyspark.__file__)`.
static PYSPARK_ROOT: PyOnceLock<Py<pyo3::types::PyString>> = PyOnceLock::new();
/// Whether the imports of IPython and ipykernel were tried, once per process. A failed import
/// is not retried, and a module imported later is still seen in `sys.modules`.
static IPYTHON_IMPORTS_TRIED: AtomicBool = AtomicBool::new(false);

/// While the returned guard lives, the expressions that this thread builds have the origin of
/// `fragment`, the name of the API function, at its call site in user code, when DataFrame
/// debugging is enabled. As PySpark's `_with_origin`, with the functions of
/// `pyspark.errors.utils` that it calls: `is_debugging_enabled`, which reads
/// `spark.python.sql.dataFrameDebugging.enabled` once, and `_capture_call_site`, which
/// `call_site` gives without importing a module per call.
///
/// As `_with_origin`, the error of `is_debugging_enabled` is raised by the API function: the
/// KeyboardInterrupt that ends its wait for the server, or the error of a server that cannot be
/// reached. A call site that cannot be read leaves the expression without an origin, as it is
/// the same without one, unless its error is not an `Exception`, as a KeyboardInterrupt.
pub(crate) fn with_origin(fragment: &str) -> PyResult<Option<OriginGuard>> {
    Python::attach(|py| Ok(origin_of(py, fragment)?.map(set_current)))
}

/// The call site in user code of one call of a native function that does what several API
/// functions of PySpark do, such as `WindowSpec.orderBy`, which calls `col` and `asc` of each
/// column name: [`CallSiteOrigin::with`] of each of their names is [`with_origin`] of it, of the
/// call site that the first one reads, as no Python frame is entered between them.
pub(crate) struct CallSiteOrigin(std::cell::OnceCell<Option<Origin>>);

impl CallSiteOrigin {
    pub(crate) fn new() -> Self {
        CallSiteOrigin(std::cell::OnceCell::new())
    }

    /// As [`with_origin`] of `fragment`, of the call site of the first call.
    pub(crate) fn with(&self, fragment: &str) -> PyResult<Option<OriginGuard>> {
        let origin = match self.0.get() {
            Some(origin) => origin,
            None => {
                let read = Python::attach(|py| origin_of(py, fragment))?;
                self.0.get_or_init(|| read)
            }
        };
        Ok(origin.as_ref().map(|origin| {
            set_current(Origin {
                fragment: fragment.to_string(),
                call_site: origin.call_site.clone(),
            })
        }))
    }
}

/// The origin that [`with_origin`] gives the expressions of `fragment` now, as PySpark's
/// `_with_origin` makes it: None when DataFrame debugging is disabled, or when the call site
/// cannot be read.
fn origin_of(py: Python<'_>, fragment: &str) -> PyResult<Option<Origin>> {
    let utils = UTILS
        .get_or_try_init(py, || py.import("pyspark.errors.utils").map(Bound::unbind))?
        .bind(py);
    // `is_debugging_enabled()` gives its cache once it is set, which it never unsets: read it
    // at once then, and call the function before, which sets it, and of a patch of the function.
    let function = utils.getattr(pyo3::intern!(py, "is_debugging_enabled"))?;
    let original = IS_DEBUGGING_ENABLED.get_or_init(py, || function.clone().unbind());
    let cached = utils.getattr(pyo3::intern!(py, "_enable_debugging_cache"))?;
    let enabled = if cached.is_none() || !function.is(original.bind(py)) {
        function.call0()?
    } else {
        cached
    };
    if !enabled.is_truthy()? {
        return Ok(None);
    }
    let call_site = match call_site(py, 1) {
        Ok(call_site) => call_site,
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyException>(py) => return Ok(None),
        Err(err) => return Err(err),
    };
    Ok(Some(Origin {
        fragment: fragment.to_string(),
        call_site,
    }))
}

/// What the client does as a request builds a plan, where PySpark's client runs Python code in
/// the user's process: it gives the origin of an expression that the plan makes then, of
/// `_with_origin` at the call site of the request, such as the `__and__` of the conditions of a
/// join of a list of Columns, and warns of a window of no partitions, as the `warnings.warn` of
/// PySpark's `WindowExpression.to_plan` warns, of its line of `pyspark.sql.connect.expressions`
/// (see `warn_as_pyspark`). Their exceptions are those of the request, as PySpark raises them out
/// of its plan build, such as the `UserWarning` of a filter of `"error"`.
struct PlanBuild;

impl spark_connect::plan_build::PlanBuildListener for PlanBuild {
    fn origin(&self, fragment: &str) -> spark_connect::Result<Option<Origin>> {
        Python::attach(|py| origin_of(py, fragment)).map_err(crate::errors::callback_error)
    }

    fn window_without_partitions(&self) -> spark_connect::Result<()> {
        if is_pickling() {
            return Ok(());
        }
        Python::attach(|py| {
            crate::errors::warn_as_pyspark(
                py,
                "pyspark.sql.connect.expressions",
                1126,
                "WARN WindowExpression: No Partition Defined for Window operation! Moving all data \
                 to a single partition, this can cause serious performance degradation.",
                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
            )
        })
        .map_err(crate::errors::callback_error)
    }
}

/// Make the client the listener of the plans that the requests of the process build.
pub(crate) fn listen_to_plan_builds() {
    spark_connect::plan_build::set_listener(Some(std::sync::Arc::new(PlanBuild)));
}

thread_local! {
    /// Whether this thread pickles a DataFrame or a Column, whose plan the client builds to keep
    /// it in the pickle.
    static PICKLING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether this thread builds the plan of a DataFrame, a Column or a table argument to pickle it
/// ([`pickling`]).
pub(crate) fn is_pickling() -> bool {
    PICKLING.with(std::cell::Cell::get)
}

/// Run `pickle`, which builds the plan of a DataFrame or a Column to pickle it, without the
/// warning of a window of no partitions: PySpark's pickle keeps its plan objects, and builds no
/// plan.
pub(crate) fn pickling<T>(pickle: impl FnOnce() -> T) -> T {
    struct Pickling(bool);
    impl Drop for Pickling {
        fn drop(&mut self) {
            PICKLING.with(|pickling| pickling.set(self.0));
        }
    }
    let _outer = Pickling(PICKLING.with(|pickling| pickling.replace(true)));
    pickle()
}

/// The call site of the first `depth` frames of user code, as `_capture_call_site(depth)` of
/// `pyspark.errors.utils` gives it: the frames outside the pyspark package, and within an
/// IPython shell (when IPython and ipykernel are loaded) without the frames of IPython and
/// ipykernel, each "line N in cell [K]", else "file:line", joined by newlines.
fn call_site(py: Python<'_>, depth: usize) -> PyResult<String> {
    let sys = SYS
        .get_or_try_init(py, || py.import("sys").map(Bound::unbind))?
        .bind(py);
    let pyspark_root = PYSPARK_ROOT
        .get_or_try_init(py, || {
            let file = py.import("pyspark")?.getattr("__file__")?;
            py.import("os")?
                .getattr("path")?
                .call_method1("dirname", (file,))?
                .cast_into::<pyo3::types::PyString>()
                .map(Bound::unbind)
                .map_err(PyErr::from)
        })?
        .bind(py);

    // The frames as `inspect` walks them from the innermost: `sys._getframe(0)` is the Python
    // frame that called the API function, as the native code of it adds no frame. The file of a
    // frame is checked as a Python str, whose substring the root is or not, and made a Rust one
    // only for a frame that the call site names.
    let mut selected = Vec::with_capacity(depth);
    // SAFETY: the GIL is held; the frame is borrowed of the thread state, or null without one,
    // which is then None, as `sys._getframe(0)` raises of no frame, which this call never has.
    let mut frame = match unsafe { pyo3::ffi::PyEval_GetFrame() } {
        frame if frame.is_null() => sys.call_method1(pyo3::intern!(py, "_getframe"), (0,))?,
        frame => unsafe { Bound::from_borrowed_ptr(py, frame.cast()) },
    };
    while selected.len() < depth && !frame.is_none() {
        let file = frame
            .getattr(pyo3::intern!(py, "f_code"))?
            .getattr(pyo3::intern!(py, "co_filename"))?;
        if !file.contains(pyspark_root)? {
            let file: String = file.extract()?;
            selected.push((file, frame.getattr(pyo3::intern!(py, "f_lineno"))?));
        }
        frame = frame.getattr(pyo3::intern!(py, "f_back"))?;
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

/// The "file:line" of each frame, joined by newlines: of the int of a line number at once, and of
/// the `str` of another value, as None.
fn join_frames(frames: &[(String, Bound<'_, PyAny>)]) -> PyResult<String> {
    let lines: Vec<String> = frames
        .iter()
        .map(|(file, lineno)| match lineno.extract::<i64>() {
            Ok(line) if lineno.is_exact_instance_of::<pyo3::types::PyInt>() => {
                Ok(format!("{file}:{line}"))
            }
            _ => Ok(format!("{file}:{lineno}")),
        })
        .collect::<PyResult<_>>()?;
    Ok(lines.join("\n"))
}

/// The modules IPython and ipykernel, when both are loaded. They are read from `sys.modules`,
/// and imported once per process when one is missing, as `_capture_call_site` imports them.
fn loaded_ipython<'py>(
    py: Python<'py>,
    sys: &Bound<'py, PyModule>,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let modules = sys.getattr(pyo3::intern!(py, "modules"))?;
    let modules = modules.cast::<PyDict>()?;
    let ipython_name = pyo3::intern!(py, "IPython");
    let ipykernel_name = pyo3::intern!(py, "ipykernel");
    let find = |name: &Bound<'py, pyo3::types::PyString>| modules.get_item(name);
    let (mut ipython, mut ipykernel) = (find(ipython_name)?, find(ipykernel_name)?);
    if (ipython.is_none() || ipykernel.is_none())
        && !IPYTHON_IMPORTS_TRIED.swap(true, Ordering::Relaxed)
    {
        // `import IPython`, then `import ipykernel`, as `_capture_call_site` does.
        let _ = py.import("IPython").and_then(|_| py.import("ipykernel"));
        (ipython, ipykernel) = (find(ipython_name)?, find(ipykernel_name)?);
    }
    Ok(ipython.zip(ipykernel))
}

// SPDX-License-Identifier: Apache-2.0
//! The server-side objects of Spark Connect ML in Python, as official pyspark-client 4.3.0 has
//! them: the values of a fetch, as `deserialize_param` returns them; the arguments of a fetch, as
//! `serialize` sends them; and the writers and readers of the native classes (`MLWriter` and
//! `MLReader`, as `RemoteMLWriter` and `RemoteMLReader`).

use super::*;
use pyo3::types::PyString;
use spark_connect::ml::{
    FetchArg, FetchOutput, MlOperator, MlPickleBlock, MlRef, MlTarget, MlValue, OperatorType,
    PickleReach,
};
use spark_connect::SparkSession;
use std::collections::HashMap;

/// The Python value of `value`, as `deserialize_param` of `pyspark.ml.connect.serialize` returns
/// it: an ML vector or matrix of `pyspark.ml.linalg`, and another value as
/// `LiteralExpression._to_value` returns it (a struct as a `Row`).
pub(super) fn ml_value_to_py(py: Python<'_>, value: &MlValue) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    use spark_connect::ml::{MlMatrix, MlVector};
    let linalg = || py.import("pyspark.ml.linalg");
    let types = || py.import("pyspark.sql.types");
    Ok(match value {
        MlValue::Null => py.None(),
        MlValue::Bool(v) => v.into_py_any(py)?,
        MlValue::Byte(v) | MlValue::Short(v) | MlValue::Int(v) => v.into_py_any(py)?,
        MlValue::Long(v) => v.into_py_any(py)?,
        MlValue::Float(v) => f64::from(*v).into_py_any(py)?,
        MlValue::Double(v) => v.into_py_any(py)?,
        MlValue::Decimal(v) => py
            .import("decimal")?
            .getattr("Decimal")?
            .call1((v,))?
            .unbind(),
        MlValue::String(v) => v.into_py_any(py)?,
        MlValue::Binary(v) => pyo3::types::PyBytes::new(py, v).into_any().unbind(),
        MlValue::Date(v) => types()?
            .getattr("DateType")?
            .call0()?
            .call_method1("fromInternal", (*v,))?
            .unbind(),
        MlValue::Timestamp(v) => types()?
            .getattr("TimestampType")?
            .call0()?
            .call_method1("fromInternal", (*v,))?
            .unbind(),
        MlValue::TimestampNtz(v) => types()?
            .getattr("TimestampNTZType")?
            .call0()?
            .call_method1("fromInternal", (*v,))?
            .unbind(),
        MlValue::Vector(MlVector::Dense(values)) => linalg()?
            .getattr("DenseVector")?
            .call1((values.clone(),))?
            .unbind(),
        MlValue::Vector(MlVector::Sparse {
            size,
            indices,
            values,
        }) => linalg()?
            .getattr("SparseVector")?
            .call1((*size, indices.clone(), values.clone()))?
            .unbind(),
        MlValue::Matrix(MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        }) => linalg()?
            .getattr("DenseMatrix")?
            .call1((*num_rows, *num_cols, values.clone(), *is_transposed))?
            .unbind(),
        MlValue::Matrix(MlMatrix::Sparse {
            num_rows,
            num_cols,
            col_ptrs,
            row_indices,
            values,
            is_transposed,
        }) => linalg()?
            .getattr("SparseMatrix")?
            .call1((
                *num_rows,
                *num_cols,
                col_ptrs.clone(),
                row_indices.clone(),
                values.clone(),
                *is_transposed,
            ))?
            .unbind(),
        MlValue::Array(items) => PyList::new(
            py,
            items
                .iter()
                .map(|v| ml_value_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any()
        .unbind(),
        MlValue::Map(entries) => {
            let dict = PyDict::new(py);
            for (k, v) in entries {
                dict.set_item(ml_value_to_py(py, k)?, ml_value_to_py(py, v)?)?;
            }
            dict.into_any().unbind()
        }
        MlValue::Struct(fields) => {
            let names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
            let values = fields
                .iter()
                .map(|(_, v)| ml_value_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            types()?
                .getattr("_create_row")?
                .call1((names, values))?
                .unbind()
        }
    })
}

/// The public classes of the native ML classes, by the type object of the native class: the
/// subclass of `pyspark.ml.connect.<module>` of the native class and of the bases of PySpark's class
/// (see `pyspark/ml/connect/_classes.py`), of which [`new_object`] makes the objects.
fn public_classes() -> &'static std::sync::Mutex<HashMap<usize, Py<pyo3::types::PyType>>> {
    static CLASSES: std::sync::OnceLock<std::sync::Mutex<HashMap<usize, Py<pyo3::types::PyType>>>> =
        std::sync::OnceLock::new();
    CLASSES.get_or_init(Default::default)
}

/// Make `public`, a subclass of the native class `native`, the class of the objects of `native`
/// that this client makes (see [`new_object`]).
#[pyfunction]
#[pyo3(name = "_register_public_class")]
pub(crate) fn register_public_class(
    native: &Bound<'_, pyo3::types::PyType>,
    public: &Bound<'_, pyo3::types::PyType>,
) -> PyResult<()> {
    if !public.is_subclass(native)? {
        return Err(PyTypeError::new_err(format!(
            "{} is not a subclass of {}",
            public.name()?,
            native.name()?
        )));
    }
    public_classes()
        .lock()
        .expect("the public classes are not poisoned")
        .insert(native.as_type_ptr() as usize, public.clone().unbind());
    Ok(())
}

/// The public class of the native class `T` (see [`register_public_class`]), if it has one.
pub(crate) fn public_class<T: pyo3::PyTypeInfo>(
    py: Python<'_>,
) -> Option<Bound<'_, pyo3::types::PyType>> {
    let native = T::type_object_raw(py) as usize;
    let classes = public_classes()
        .lock()
        .expect("the public classes are not poisoned");
    classes.get(&native).map(|class| class.bind(py).clone())
}

/// The `__del__` of `class`, of the public class of a native model (whose `__del__` has the names
/// of PySpark's `JavaWrapper.__del__`), if it has one.
pub(crate) fn deleter_of(class: Option<&Bound<'_, pyo3::types::PyType>>) -> Option<Py<PyAny>> {
    let class = class?;
    class
        .getattr(pyo3::intern!(class.py(), "__del__"))
        .ok()
        .map(Bound::unbind)
}

/// The object of `init`, of the public class of `T` if it has one (as PySpark's objects are of the
/// class of `pyspark.ml`, of its bases), else of `T`: every object of a native ML class that this
/// client makes. It is made as PyO3's `tp_new` of the public class makes one, of `init`.
pub(crate) fn new_object<T: PyClass>(
    py: Python<'_>,
    init: impl Into<PyClassInitializer<T>>,
) -> PyResult<Bound<'_, T>> {
    let init = init.into();
    // The lock is released before the allocation, which may run Python code.
    let Some(public) = public_class::<T>(py) else {
        return Bound::new(py, init);
    };
    // SAFETY: `public` is a subclass of `T` (see `register_public_class`), of which PyO3's
    // `tp_new_impl`, the `tp_new` of a `#[new]`, makes an object of an initializer of `T`.
    let object: Bound<'_, T> = unsafe {
        let ptr = pyo3::impl_::pymethods::tp_new_impl::<PyClassInitializer<T>, T>(
            py,
            init,
            public.as_type_ptr(),
        )?;
        Bound::from_owned_ptr(py, ptr).cast_into_unchecked()
    };
    // The `_java_obj` attribute of a model, as PySpark's fit and load set it.
    if let Some(make) = helpers_of_class(&public)?.and_then(|helpers| helpers.make_java_obj) {
        make(object.as_any())?;
    }
    Ok(object)
}

/// The set values and the defaults of the params of an object by name (see [`NativeHelpers`]).
pub(crate) type ParamsOf =
    for<'py> fn(&Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)>;
/// Make the params of an object those of its param maps, of their values by name.
pub(crate) type SyncOf =
    for<'py> fn(&Bound<'py, PyAny>, &Bound<'py, PyDict>, &Bound<'py, PyDict>) -> PyResult<()>;
/// The native `_java_obj` of an object.
pub(crate) type JavaObjOf = for<'py> fn(&Bound<'py, PyAny>) -> PyResult<Option<Py<PyAny>>>;
/// What the `__del__` of PySpark's class of an object does.
pub(crate) type FinalizeOf = for<'py> fn(&Bound<'py, PyAny>) -> PyResult<()>;
/// The shallow copy of an object of a server-side object.
pub(crate) type CopyOf = for<'py> fn(&Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>;
/// The uid of the native object of an object.
pub(crate) type UidOf = for<'py> fn(&Bound<'py, PyAny>) -> PyResult<String>;

/// What a native ML class gives the functions of the native module over its objects (`_ml_params`,
/// `_ml_sync`, `_ml_uid`, `_ml_java_obj`, `_ml_finalize`, `_ml_copy`), by which the public class of
/// it reaches the native state of an object: of its params, of PySpark's `_java_obj` and of the
/// release of its `__del__`. These are no members of the class, which PySpark's class has not, and
/// the class of an object of them is a class of [`register_class`], or a subclass of one.
#[derive(Clone, Copy)]
pub(crate) struct NativeHelpers {
    pub(crate) params: Option<ParamsOf>,
    pub(crate) sync: Option<SyncOf>,
    pub(crate) uid: Option<UidOf>,
    /// The JVM class of a Java wrapper of PySpark's of no server-side object (an estimator, a
    /// transformer, an evaluator), its `_java_obj` over Spark Connect.
    pub(crate) jvm_class: Option<&'static str>,
    pub(crate) java_obj: Option<JavaObjOf>,
    pub(crate) finalize: Option<FinalizeOf>,
    pub(crate) copy: Option<CopyOf>,
    /// Make the `_java_obj` attribute of an object that this client made (see [`new_object`]).
    pub(crate) make_java_obj: Option<FinalizeOf>,
}

impl NativeHelpers {
    /// The helpers of a class of none.
    pub(crate) const NONE: NativeHelpers = NativeHelpers {
        params: None,
        sync: None,
        uid: None,
        jvm_class: None,
        java_obj: None,
        finalize: None,
        copy: None,
        make_java_obj: None,
    };

    fn is_none(&self) -> bool {
        self.params.is_none() && self.java_obj.is_none()
    }
}

/// A native ML class, of the helpers that it gives (see [`NativeHelpers`]): those of the Params
/// API of a class of params (`native_pymethods!`), and of `_java_obj` of a model (`native_model!`)
/// and of a summary (`native_summary!`).
pub(crate) trait NativeClass: PyClass {
    const HELPERS: NativeHelpers = NativeHelpers::NONE;
}

/// The helpers of the native ML classes, by the type object of each (see [`register_class`]).
fn native_classes() -> &'static std::sync::Mutex<HashMap<usize, NativeHelpers>> {
    static CLASSES: std::sync::OnceLock<std::sync::Mutex<HashMap<usize, NativeHelpers>>> =
        std::sync::OnceLock::new();
    CLASSES.get_or_init(Default::default)
}

/// Add the native ML class `T` to the module `m`, of its helpers (see [`NativeHelpers`]).
pub(crate) fn register_class<T: NativeClass>(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<T>()?;
    if !T::HELPERS.is_none() {
        let class = T::type_object_raw(m.py()) as usize;
        native_classes()
            .lock()
            .expect("the native classes are not poisoned")
            .insert(class, T::HELPERS);
    }
    Ok(())
}

/// The helpers of the class `class`: of the first of it and its bases (`__base__`, a public class
/// of its native class, a Python subclass of a public class of it) that is a class of
/// [`register_class`] of helpers.
fn helpers_of_class(class: &Bound<'_, pyo3::types::PyType>) -> PyResult<Option<NativeHelpers>> {
    let py = class.py();
    let mut class = class.clone();
    loop {
        let found = native_classes()
            .lock()
            .expect("the native classes are not poisoned")
            .get(&(class.as_type_ptr() as usize))
            .copied();
        if found.is_some() {
            return Ok(found);
        }
        let base = class.getattr(pyo3::intern!(py, "__base__"))?;
        match base.cast_into::<pyo3::types::PyType>() {
            Ok(base) => class = base,
            Err(_) => return Ok(None),
        }
    }
}

/// The helper `pick` of the class of `obj` (see [`helpers_of_class`]), or the TypeError of an
/// object of no native class of it.
fn helper_of<F>(obj: &Bound<'_, PyAny>, pick: impl Fn(&NativeHelpers) -> Option<F>) -> PyResult<F> {
    helpers_of_class(&obj.get_type())?
        .as_ref()
        .and_then(pick)
        .ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{} is not an object of a native ML class",
                obj.get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            ))
        })
}

/// What the class `class` is of the native ML classes: `"params"` of a class of params of no
/// object of the server (an estimator, a transformer, an evaluator, a Pipeline), `"model"` of a
/// model, `"summary"` of a summary, None of another class. Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_kind")]
pub(crate) fn ml_kind(class: &Bound<'_, pyo3::types::PyType>) -> PyResult<Option<&'static str>> {
    Ok(helpers_of_class(class)?.map(|helpers| {
        match (helpers.params.is_some(), helpers.java_obj.is_some()) {
            (true, true) => "model",
            (true, false) => "params",
            _ => "summary",
        }
    }))
}

/// The name of the JVM class of the native class `class` of params of no server-side object (an
/// estimator, a transformer, an evaluator), PySpark's `_java_obj` of its objects over Spark Connect
/// (`_new_java_obj` of `try_remote_return_java_class`); None of another class. Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_jvm_class")]
pub(crate) fn ml_jvm_class(
    class: &Bound<'_, pyo3::types::PyType>,
) -> PyResult<Option<&'static str>> {
    Ok(helpers_of_class(class)?.and_then(|helpers| helpers.jvm_class))
}

/// The uid of the native object of `obj`, of which the public class makes PySpark's `uid` of the
/// object at its first read (see `_native_classes._materialize`). Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_uid")]
pub(crate) fn ml_uid(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    helper_of(obj, |helpers| helpers.uid)?(obj)
}

/// The set values and the defaults of the params of `obj` by name, of which the public class makes
/// PySpark's param maps of the object (see `params_api::native_params`). Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_params")]
pub(crate) fn ml_params<'py>(
    obj: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    helper_of(obj, |helpers| helpers.params)?(obj)
}

/// Make the params of `obj` those of PySpark's param maps of it, of their values by name (see
/// `params_api::sync_params`). Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_sync")]
pub(crate) fn ml_sync(
    obj: &Bound<'_, PyAny>,
    set: &Bound<'_, PyDict>,
    defaults: &Bound<'_, PyDict>,
) -> PyResult<()> {
    helper_of(obj, |helpers| helpers.sync)?(obj, set, defaults)
}

/// PySpark's `_java_obj` of `obj`, a model or a summary that PySpark's constructor did not make
/// (see `ModelCore::java_obj`). Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_java_obj")]
pub(crate) fn ml_java_obj(obj: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
    helper_of(obj, |helpers| helpers.java_obj)?(obj)
}

/// Make the `_java_obj` attribute of `obj`, a model that this client made, of its native object, if
/// its attributes have none (but of a deleted one). Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_make_java_obj")]
pub(crate) fn ml_make_java_obj(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    match helpers_of_class(&obj.get_type())?.and_then(|helpers| helpers.make_java_obj) {
        Some(make) => make(obj),
        None => Ok(()),
    }
}

/// The shallow copy of `obj`, a model of a server-side object, of the same object, which
/// `copy.copy` of its reduce makes (see `params_api::reduce_ex`). Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_copy")]
pub(crate) fn ml_copy<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    helper_of(obj, |helpers| helpers.copy)?(obj)
}

/// The objects that the traverse of `obj` reports to Python's garbage collector, as
/// `gc.get_referents` reads them (of the `tp_traverse` of its class), by address, of no Python
/// object made: so a read runs no collection and no Python code, which would let another thread
/// take the GIL.
fn traversed(obj: &Bound<'_, PyAny>) -> Vec<usize> {
    unsafe extern "C" fn keep(
        object: *mut pyo3::ffi::PyObject,
        referents: *mut std::ffi::c_void,
    ) -> std::ffi::c_int {
        // SAFETY: `referents` is the vector of `traversed`, which outlives the traverse.
        unsafe { (*referents.cast::<Vec<usize>>()).push(object as usize) };
        0
    }
    let mut referents: Vec<usize> = Vec::new();
    // SAFETY: of the GIL, which `obj` proves, and of a live `obj`; the slot of a class of a static
    // type (of no traverse here) reads as null, of an error that is cleared.
    unsafe {
        let slot =
            pyo3::ffi::PyType_GetSlot(pyo3::ffi::Py_TYPE(obj.as_ptr()), pyo3::ffi::Py_tp_traverse);
        if slot.is_null() {
            pyo3::ffi::PyErr_Clear();
            return referents;
        }
        let traverse: pyo3::ffi::traverseproc = std::mem::transmute(slot);
        traverse(
            obj.as_ptr(),
            keep,
            std::ptr::from_mut(&mut referents).cast::<std::ffi::c_void>(),
        );
    }
    referents
}

/// Watch the traverses of `objects` (models), for the tests, of nothing that the models tell: with
/// the GIL held throughout, so that no collection runs and no code that holds the GIL changes
/// them meanwhile, read what the traverse of each reports, again and again, for `spin_us`
/// microseconds. A collection reads the traverses in each of its passes, and a reference that one
/// pass reports and a later one does not is freed while the model holds it (see
/// [`ModelCore::exclusive_reference`]): so a report that loses a reference that it had at the start
/// of the watch is a change that was made without the GIL, of a call that clones what the traverse
/// reads (an `Arc` of the model) or locks a mutex that it tries. A report that gains one (of a drop
/// without the GIL) only keeps it alive. Returns the reads, the objects whose report lost a
/// reference, and the logs of the first ones. Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_watch_traverses")]
pub(crate) fn ml_watch_traverses(
    objects: Vec<Bound<'_, PyAny>>,
    spin_us: u64,
) -> (u64, u64, Vec<String>) {
    let first: Vec<Vec<usize>> = objects.iter().map(traversed).collect();
    let start = std::time::Instant::now();
    let (mut reads, mut changes) = (0u64, 0u64);
    let mut logs = Vec::new();
    let mut seen = vec![false; objects.len()];
    loop {
        for (index, obj) in objects.iter().enumerate() {
            if seen[index] {
                continue;
            }
            reads += 1;
            let now = traversed(obj);
            let lost = first[index].iter().filter(|r| !now.contains(r)).count();
            if lost == 0 {
                continue;
            }
            seen[index] = true;
            changes += 1;
            if logs.len() < 20 {
                let name = obj
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default();
                logs.push(format!(
                    "the traverse of a {name} reported {} objects and then {} ({lost} lost), with \
                     the GIL held, after {} us",
                    first[index].len(),
                    now.len(),
                    start.elapsed().as_micros()
                ));
            }
        }
        if start.elapsed().as_micros() >= u128::from(spin_us) {
            break;
        }
    }
    (reads, changes, logs)
}

/// The release of the `__del__` of PySpark's class of `obj`, which the `__del__` of its public
/// class runs (see `ModelCore::finalize`); of an object of no release, nothing. Not a stable API.
#[pyfunction]
#[pyo3(name = "_ml_finalize")]
pub(crate) fn ml_finalize(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    match helpers_of_class(&obj.get_type())?.and_then(|helpers| helpers.finalize) {
        Some(finalize) => finalize(obj),
        None => Ok(()),
    }
}

/// A model of a fit, which is made an object of its class: the public class of its native class,
/// or a subclass of a base class (LDA's), as `Fitted` returns it.
pub(crate) trait ModelObject: PyClass + Sized {
    fn into_object(self, py: Python<'_>) -> PyResult<Py<Self>>;
}

/// The object of `init` as [`new_object`] makes it, unbound.
pub(crate) fn new_py<T: PyClass>(
    py: Python<'_>,
    init: impl Into<PyClassInitializer<T>>,
) -> PyResult<Py<T>> {
    new_object(py, init).map(Bound::unbind)
}

/// The argument `value` of a fetch, as `serialize` of `pyspark.ml.connect.serialize` sends it: a
/// DataFrame as its relation, a `(value, DataType)` pair as the literal of that type
/// (`LiteralExpression(value, dataType)`, whose asserts raise PySpark's bare AssertionError of a
/// value or type that it does not take), an ML vector or matrix as itself, and another value as
/// the literal of `LiteralExpression._from_value`, which raises PySpark's error of a value that
/// cannot be one.
pub(super) fn fetch_arg_of(value: &Bound<'_, PyAny>) -> PyResult<FetchArg> {
    if let Ok(df) = value.extract::<PyRef<'_, PyDataFrame>>() {
        return Ok(FetchArg::Input(df.dataframe.clone()));
    }
    if let Some(literal) = crate::functions::typed_pair_literal(value)? {
        return Ok(FetchArg::Literal(literal.to_literal_proto()));
    }
    let py = value.py();
    let linalg = py.import("pyspark.ml.linalg")?;
    for class in ["SparseVector", "DenseVector"] {
        if value.is_instance(&linalg.getattr(class)?)? {
            if let Converted::Vector(vector) = convert("value", Kind::Vector, value)? {
                return Ok(FetchArg::Value(MlValue::Vector(vector)));
            }
        }
    }
    for class in ["SparseMatrix", "DenseMatrix"] {
        if value.is_instance(&linalg.getattr(class)?)? {
            if let Converted::Matrix(matrix) = convert("value", Kind::Matrix, value)? {
                return Ok(FetchArg::Value(MlValue::Matrix(matrix)));
            }
        }
    }
    // A Column is no literal of PySpark's `LiteralExpression._from_value`, even of a literal
    // (`F.lit(3)`).
    if value.is_instance_of::<crate::column::PyColumn>() {
        return Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE",
            &[("data_type", &value.get_type().name()?.to_string())],
        ));
    }
    let column = crate::functions::to_value_column(value)?;
    match column.expression().node() {
        spark_connect::expression::ExpressionNode::Literal(literal) => {
            Ok(FetchArg::Literal(literal.to_literal_proto()))
        }
        _ => Err(crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE",
            &[("data_type", &value.get_type().name()?.to_string())],
        )),
    }
}

/// The Python value of the result of a fetch: a value, or the string of a `summary` result.
pub(super) fn fetch_output_to_py(py: Python<'_>, output: FetchOutput) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    match output {
        FetchOutput::Value(value) => ml_value_to_py(py, &value),
        FetchOutput::Summary(text) => text.into_py_any(py),
        FetchOutput::Objects(objects) => Err(PyValueError::new_err(format!(
            "an ML method returned {} objects where a value was expected",
            objects.len()
        ))),
    }
}

/// Fetch the method `method` of `target`, of the Python arguments `args`, interruptibly.
pub(super) fn fetch_py(
    py: Python<'_>,
    target: &MlTarget,
    method: &str,
    args: &[Bound<'_, PyAny>],
) -> PyResult<FetchOutput> {
    let args = args
        .iter()
        .map(fetch_arg_of)
        .collect::<PyResult<Vec<_>>>()?;
    // Of the plans of the summary dataset of the target and of the DataFrames of the arguments,
    // which the fetch, and the summary that it makes again, are built over.
    let _in_use = crate::gc::in_use_of_values(py, fetch_parts(target, &args));
    interruptible(py, target.session(), || {
        spark_connect::ml::fetch(target, method, args).map_err(Box::new)
    })
}

/// The references to the parts of plans of a fetch of `target`, of the arguments `args`.
fn fetch_parts<'a>(
    target: &'a MlTarget,
    args: &'a [FetchArg],
) -> impl Iterator<Item = spark_connect::gc::PartRef<'a>> {
    let arguments = args.iter().flat_map(|arg| match arg {
        FetchArg::Input(df) => df.held_parts(),
        _ => Vec::new(),
    });
    target.held_parts().into_iter().chain(arguments)
}

/// The DataFrame of the method `method` of `target`, of the Python arguments `args`.
pub(super) fn attribute_relation_py(
    target: &MlTarget,
    method: &str,
    args: &[Bound<'_, PyAny>],
) -> PyResult<PyDataFrame> {
    let args = args
        .iter()
        .map(fetch_arg_of)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyDataFrame::new(
        spark_connect::ml::attribute_relation(target, method, args).to_pyerr()?,
    ))
}

/// The path of a `Write` or a `Read`, as PySpark's protobuf string field of
/// `MlCommand.Write(path=...)` takes the Python value: None as the empty path (which the server
/// refuses), and anything else as `proto_string` takes it.
fn proto_path(path: &Bound<'_, PyAny>) -> PyResult<String> {
    if path.is_none() {
        return Ok(String::new());
    }
    proto_string(path)
}

/// The Python value `value` of a protobuf string field (of a message or of the entries of a map),
/// as upb takes it: a str as it is (of the UnicodeEncodeError of a lone surrogate), bytes of their
/// UTF-8 (of the UnicodeDecodeError of other bytes), and anything else upb's TypeError.
fn proto_string(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if value.is_instance_of::<PyString>() {
        return value.extract();
    }
    if value.is_instance_of::<pyo3::types::PyBytes>() {
        return value.call_method1("decode", ("utf-8",))?.extract();
    }
    Err(PyTypeError::new_err(
        "bad argument type for built-in operation",
    ))
}

/// The path of a save or a load of PySpark's Python format (a Pipeline, a validator), as its
/// `os.path.join(path, "metadata")` takes it: a str or an `os.PathLike` (of its `os.fspath`), and
/// the TypeError of `os.path.join` of anything else (of bytes, that it mixes them with a str).
fn python_path(path: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = path.py();
    py.import("os.path")?
        .getattr("join")?
        .call1((path, "metadata"))?;
    py.import("os")?
        .getattr("fspath")?
        .call1((path,))?
        .extract()
}

/// The active session, as the Python object that `SparkSession.getActiveSession()` returns: of
/// the saves and loads of PySpark's Python format, which `RemoteMLWriter` and `RemoteMLReader`
/// make of it, and of the objects that PySpark's own `saveInstance` and `loadInstance` save and
/// load.
fn active_spark(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let spark = py
        .import("pyspark.sql")?
        .getattr("SparkSession")?
        .call_method0("getActiveSession")?;
    if spark.is_none() {
        // As PySpark's `assert session is not None` of no active session.
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    Ok(spark)
}

/// The session of the active Spark session of Python, which PySpark's `RemoteMLWriter.save` and
/// `RemoteMLReader.load` use (`SparkSession.getActiveSession()`), or of `session` if given.
pub(super) fn session_of(py: Python<'_>, session: Option<&Py<PyAny>>) -> PyResult<SparkSession> {
    if let Some(session) = session {
        return Ok(session
            .bind(py)
            .extract::<PyRef<'_, crate::session::PySparkSession>>()?
            .session
            .clone());
    }
    // As PySpark's `assert session is not None` of no active session.
    let active = crate::session::active_session(py)
        .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err(()))?;
    let session = active.borrow(py).session.clone();
    Ok(session)
}

/// The server-side object of a summary: the target of this client's fit or evaluation; or, of a
/// summary of PySpark's constructor (`JavaWrapper.__init__(java_obj=None)`), the `java_obj` that
/// its `__init__` keeps in the object's attributes (`_java_obj`), which the members read at each
/// call, as PySpark's read `self._java_obj` (see [`SummaryObject::target`]).
pub(crate) enum SummaryObject {
    // Boxed: a target is much larger than a Python object.
    Target(Box<spark_connect::ml::MlTarget>),
    Given,
}

impl SummaryObject {
    /// A copy of the object, of the same server-side object.
    pub(crate) fn clone_ref(&self) -> Self {
        match self {
            Self::Target(target) => Self::Target(target.clone()),
            Self::Given => Self::Given,
        }
    }

    /// The target of the members of the summary `slf` that fetch, as PySpark's `try_remote_call`
    /// reads it. Of a summary of PySpark's constructor, it asserts an active session first, then
    /// `_extract_id_methods` splits the `_java_obj` of the object, the id of an object of the ML
    /// cache and the methods of its chain (`"<id>.summary"`), over the active session; another
    /// value than a string (None) raises the AttributeError of its `split`.
    pub(crate) fn target(&self, slf: &Bound<'_, PyAny>) -> PyResult<spark_connect::ml::MlTarget> {
        if let Self::Target(target) = self {
            return Ok((**target).clone());
        }
        let py = slf.py();
        let session = session_of(py, None)?;
        let object = slf.getattr(pyo3::intern!(py, "_java_obj"))?;
        let Ok(text) = object.cast::<PyString>() else {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'{}' object has no attribute 'split'",
                object.get_type().name()?
            )));
        };
        let text: String = text.extract()?;
        let mut chain = text.split('.');
        let id = chain.next().unwrap_or_default();
        let mut target = spark_connect::ml::MlTarget::unowned(&session, id);
        for method in chain {
            target = target.child(method);
        }
        Ok(target)
    }

    /// The target of the DataFrame members of the summary `slf`, as PySpark's
    /// `invoke_remote_attribute_relation` reads it: the target of its other members (see
    /// [`SummaryObject::target`]), which of the training summary of a fit has the summary dataset
    /// of the fit. Of a summary of PySpark's constructor whose methods start with `summary`
    /// (`"<id>.summary"`, as PySpark names the training summary of a fit), the summary dataset is
    /// the plan of its `_summary_dataset`, which PySpark's fit sets: a summary of none raises the
    /// AttributeError of the attribute, and one of no `_plan` the AttributeError of the value, as
    /// PySpark's do.
    pub(crate) fn relation_target(
        &self,
        slf: &Bound<'_, PyAny>,
    ) -> PyResult<spark_connect::ml::MlTarget> {
        let target = self.target(slf)?;
        if matches!(self, Self::Target(_)) || target.id().split('.').nth(1) != Some("summary") {
            return Ok(target);
        }
        let py = slf.py();
        let dataset = slf.getattr(pyo3::intern!(py, "_summary_dataset"))?;
        let plan = dataset.getattr(pyo3::intern!(py, "_plan"))?;
        let dataset = match dataset.cast::<PyDataFrame>() {
            Ok(dataset) => dataset.borrow().dataframe.clone(),
            // Another object of a plan: the DataFrame of the plan, as PySpark's `AttributeRelation`
            // takes the plan as its child.
            Err(_) => {
                let session = crate::session::active_session(py)
                    .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err(()))?;
                py.get_type::<PyDataFrame>()
                    .call1((plan, session))?
                    .cast::<PyDataFrame>()?
                    .borrow()
                    .dataframe
                    .clone()
            }
        };
        Ok(target.with_summary_dataset(&dataset))
    }
}

/// The member `name` of the class of `obj`, as the class of its MRO that has it holds it (as
/// `inspect.getattr_static` finds it, without the attributes of the metaclass), if any.
pub(crate) fn class_member<'py>(
    obj: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = obj.py();
    for class in obj
        .get_type()
        .getattr(pyo3::intern!(py, "__mro__"))?
        .try_iter()?
    {
        let members = class?.getattr(pyo3::intern!(py, "__dict__"))?;
        if members.contains(name)? {
            return Ok(Some(members.get_item(name)?));
        }
    }
    Ok(None)
}

/// The error of a pickle at `protocol` of an object that keeps the lock of a `RemoteModelRef`, as
/// PySpark's pickle of a fitted model, of the summary of an evaluation (which keeps its model) or
/// of a DataFrame of a model raises it: the error of the pickle of a lock at the protocol, the
/// TypeError of the protocols 2 and up, and the PicklingError of the lookup of its class below.
pub(crate) fn lock_pickle_error(py: Python<'_>, protocol: &Bound<'_, PyAny>) -> PyErr {
    let pickled = py
        .import("threading")
        .and_then(|threading| threading.call_method0("Lock"))
        .and_then(|lock| py.import("pickle")?.call_method1("dumps", (lock, protocol)));
    match pickled {
        Err(error) => error,
        Ok(_) => params_api::lock_error(),
    }
}

/// What PySpark's pickle of the DataFrame `df` reaches first that does not pickle, of the objects
/// of `pyspark.ml` that its plan keeps (see `spark_connect::ml::pickle_reach_with`), of the Python
/// plans of `pyspark.ml` that its relations keep too (`DataFrame(plan, session)`, the plans of
/// PySpark's own classes of `pyspark.ml`), as PySpark's pickle reaches them in its plan (see
/// [`python_plan_reach`]), and the functions of its UDFs that it reaches before, of the DataFrames
/// below those Python plans too. The first error of the walk is its error.
pub(crate) fn pickle_reach(py: Python<'_>, df: &DataFrame) -> PyResult<PickleReach> {
    reach_in(py, &mut std::collections::HashSet::new(), |owned| {
        spark_connect::ml::pickle_reach_with(df, owned)
    })
}

/// What PySpark's pickle of the expression `expression` reaches (see [`pickle_reach`]), of a
/// Column (`Column._expr`) or of the subquery of a table argument: the plans of its subqueries
/// where they are, as PySpark's pickle of a `SubqueryExpression` pickles its plan.
pub(crate) fn expression_pickle_reach(
    py: Python<'_>,
    expression: &spark_connect::expression::Expression,
) -> PyResult<PickleReach> {
    reach_in(py, &mut std::collections::HashSet::new(), |owned| {
        spark_connect::ml::expression_pickle_reach_with(expression, owned)
    })
}

/// What the walk of a reach gives of the owner of a relation built outside the core (see
/// `spark_connect::ml::pickle_reach_with`).
type OwnerReach<'a> = &'a mut dyn FnMut(&(dyn std::any::Any + Send + Sync)) -> PickleReach;

/// The reach of `walk` of the reach of the Python plan that owns a relation built outside the core
/// (see [`python_plan_reach`]), of the Python objects of `visited` visited: the first error of the
/// walk is its error.
fn reach_in(
    py: Python<'_>,
    visited: &mut std::collections::HashSet<usize>,
    walk: impl FnOnce(OwnerReach<'_>) -> PickleReach,
) -> PyResult<PickleReach> {
    let mut error = None;
    let reach = walk(&mut |owner| {
        let Some(plan) = owner.downcast_ref::<Py<PyAny>>() else {
            return PickleReach::default();
        };
        match python_plan_reach(plan.bind(py), visited) {
            Ok(reach) => reach,
            Err(failed) => {
                error = Some(failed);
                // Stop the walk, of the error.
                PickleReach {
                    picklers: Vec::new(),
                    block: Some(MlPickleBlock::ModelLock),
                }
            }
        }
    });
    match error {
        Some(error) => Err(error),
        None => Ok(reach),
    }
}

/// The objects that PySpark's pickle (or `copy.deepcopy`) of an object of the reach `reach`
/// reaches first that does not pickle, of a reach of a block, in its order: the functions of the
/// UDFs that it reaches before the block (the objects of their picklers, see
/// `udf_command::pickler_object`), and then those of the block, the summary dataset of a training
/// summary (a DataFrame, whose session, of a local function, a plain pickle refuses) and a lock,
/// as the `RemoteModelRef` of the model holds one, or the lock alone. None of a reach of no block.
fn reached_objects<'py>(
    py: Python<'py>,
    reach: &PickleReach,
) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
    let Some(block) = &reach.block else {
        return Ok(None);
    };
    let mut objects: Vec<Bound<'py, PyAny>> = reach
        .picklers
        .iter()
        .filter_map(|pickler| crate::udf_command::pickler_object(py, pickler))
        .collect();
    if let MlPickleBlock::SummarySession(dataset) = block {
        let dataset = Py::new(py, crate::dataframe::PyDataFrame::new(dataset.clone()))?;
        objects.push(dataset.into_bound(py).into_any());
    }
    objects.push(py.import("threading")?.call_method0("Lock")?);
    Ok(Some(objects))
}

/// The reduce of a pickle of an object of the reach `reach` whose own reduce is `reduce` (its
/// `(function, args, ...)`): of a reach of a block, the reduce of
/// `_column_reduce.unpickled_after` of the objects that PySpark's pickle reaches first (see
/// [`reached_objects`]), and of the function and the arguments of `reduce`, of its state, so that
/// `pickle` and cloudpickle fail at the first of those objects that they cannot pickle, at their
/// protocol, as PySpark's do; else `reduce`.
pub(crate) fn reach_reduce<'py>(
    py: Python<'py>,
    reach: &PickleReach,
    reduce: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match reached_objects(py, reach)? {
        Some(objects) => reduce_after(py, objects, reduce),
        None => Ok(reduce),
    }
}

/// The reduce of `_column_reduce.unpickled_after` of `objects` and of the function and the
/// arguments of `reduce` (its `(function, args, ...)`), of its state: a pickle takes `objects`
/// first, in their order, as their unpickle comes before the call of the function.
fn reduce_after<'py>(
    py: Python<'py>,
    objects: Vec<Bound<'py, PyAny>>,
    reduce: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let reduce = reduce.cast_into::<PyTuple>()?;
    let after = py
        .import("pyspark.sql.connect._column_reduce")?
        .getattr("unpickled_after")?;
    let args = PyTuple::new(
        py,
        [
            PyList::new(py, objects)?.into_any(),
            reduce.get_item(0)?,
            reduce.get_item(1)?,
        ],
    )?;
    let mut items = vec![after, args.into_any()];
    items.extend(reduce.iter().skip(2));
    Ok(PyTuple::new(py, items)?.into_any())
}

/// The error of `copy.deepcopy` of an object of the reach `reach` with `memo`, of a reach of a
/// block: the TypeError of the lock of the model, after the deep copies, with `memo`, of the
/// objects that PySpark's deep copy reaches before it (see [`reached_objects`]), the functions of
/// the UDFs and the summary dataset of a training summary, whose schema the copy of a DataFrame
/// requests. None of a reach of no block.
pub(crate) fn reach_deepcopy_error(
    py: Python<'_>,
    reach: &PickleReach,
    memo: &Bound<'_, PyAny>,
) -> PyResult<Option<PyErr>> {
    let Some(mut objects) = reached_objects(py, reach)? else {
        return Ok(None);
    };
    // The lock, whose copy fails.
    objects.pop();
    let deepcopy = py.import("copy")?.getattr("deepcopy")?;
    for object in objects {
        deepcopy.call1((object, memo))?;
    }
    let protocol = pyo3::types::PyInt::new(py, 4).into_any();
    Ok(Some(lock_pickle_error(py, &protocol)))
}

/// What PySpark's pickle of the Python plan `plan` (a `LogicalPlan` of `pyspark.sql.connect.plan`)
/// reaches (see [`PickleReach`]), of its attributes in their order, as the pickle takes them: its
/// child (`_child`, of a DataFrame of this client the plan of the DataFrame, of the functions of
/// its UDFs, see [`pickle_reach`]) and its references (`_references`, the plans of its subqueries)
/// first, and then the objects of `pyspark.ml` that it keeps (`__source_transformer__` of a
/// transform, `__source_instance__` of a DataFrame of a method of a model or a summary, see
/// [`object_block`]): the functions of the UDFs that it reaches before the first that does not
/// pickle, and that one. A plan of `visited` is pickled once, as of the memo of the pickle.
fn python_plan_reach(
    plan: &Bound<'_, PyAny>,
    visited: &mut std::collections::HashSet<usize>,
) -> PyResult<PickleReach> {
    let py = plan.py();
    if !visited.insert(plan.as_ptr() as usize) {
        return Ok(PickleReach::default());
    }
    let plans = py.import("pyspark.sql.connect.plan")?;
    if plan.is_instance(&plans.getattr("_DataFramePlan")?)? {
        let df = plan.getattr(pyo3::intern!(py, "_df"))?;
        let df = df.cast::<PyDataFrame>()?.borrow().dataframe.clone();
        return reach_in(py, visited, |owned| {
            spark_connect::ml::pickle_reach_with(&df, owned)
        });
    }
    let logical = plans.getattr("LogicalPlan")?;
    let mut reach = PickleReach::default();
    let Ok(attributes) = plan.getattr(pyo3::intern!(py, "__dict__")) else {
        return Ok(reach);
    };
    // The picklers of `part` of the reach, after those before it (which the walk that takes the
    // reach takes once each), and its block.
    let take = |reach: &mut PickleReach, part: PickleReach| {
        reach.picklers.extend(part.picklers);
        reach.block = part.block;
        reach.block.is_some()
    };
    for item in attributes.call_method0("items")?.try_iter()? {
        let (name, value) = item?.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
        if name.eq("__source_transformer__")? || name.eq("__source_instance__")? {
            reach.block = object_block(&value, visited)?;
            if reach.block.is_some() {
                return Ok(reach);
            }
        } else if value.is_instance(&logical)? {
            let part = python_plan_reach(&value, visited)?;
            if take(&mut reach, part) {
                return Ok(reach);
            }
        } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
            for item in value.try_iter()? {
                let item = item?;
                if item.is_instance(&logical)? {
                    let part = python_plan_reach(&item, visited)?;
                    if take(&mut reach, part) {
                        return Ok(reach);
                    }
                }
            }
        }
    }
    Ok(reach)
}

/// What PySpark's pickle of the object `obj` of `pyspark.ml` that a plan keeps reaches first that
/// does not pickle: the lock of a `RemoteModelRef` (a model's `_java_obj`, the `_remote_model_obj`
/// of a training summary), of its attributes in their order, and of those of the model that it
/// keeps (`__source_transformer__` of the summary of an evaluation); a DataFrame of its attributes
/// (the `_summary_dataset` of a training summary) pickles its plan (see [`pickle_block`]) and then
/// its session, which a plain pickle refuses before a lock after it, as
/// `MlPickleBlock::SummarySession` tells. A native model of a server-side object is its lock. An
/// object of `visited` is pickled once, as of the memo of the pickle.
fn object_block(
    obj: &Bound<'_, PyAny>,
    visited: &mut std::collections::HashSet<usize>,
) -> PyResult<Option<MlPickleBlock>> {
    let py = obj.py();
    if !visited.insert(obj.as_ptr() as usize) {
        return Ok(None);
    }
    let reference = py.import("pyspark.ml.util")?.getattr("RemoteModelRef")?;
    if obj.is_instance(&reference)? {
        return Ok(Some(MlPickleBlock::ModelLock));
    }
    if ml_kind(&obj.get_type())? == Some("model") {
        // Of a deleted `_java_obj`, which PySpark's pickle does not reach, none.
        let object = match ml_java_obj(obj) {
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => None,
            object => object?,
        };
        return Ok(object.is_some().then_some(MlPickleBlock::ModelLock));
    }
    let Ok(attributes) = obj.getattr(pyo3::intern!(py, "__dict__")) else {
        return Ok(None);
    };
    // The DataFrame whose session the pickle reaches first, before a lock.
    let mut session: Option<DataFrame> = None;
    for item in attributes.call_method0("items")?.try_iter()? {
        let (name, value) = item?.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
        let block = if value.is_instance(&reference)? {
            Some(MlPickleBlock::ModelLock)
        } else if let Ok(df) = value.cast::<PyDataFrame>() {
            let df = df.borrow().dataframe.clone();
            let block = reach_in(py, visited, |owned| {
                spark_connect::ml::pickle_reach_with(&df, owned)
            })?
            .block;
            if block.is_none() && session.is_none() {
                session = Some(df);
            }
            block
        } else if name.eq("__source_transformer__")? {
            object_block(&value, visited)?
        } else {
            None
        };
        match block {
            Some(MlPickleBlock::ModelLock) => {
                return Ok(Some(match session {
                    Some(dataset) => MlPickleBlock::SummarySession(dataset),
                    None => MlPickleBlock::ModelLock,
                }));
            }
            Some(MlPickleBlock::SummarySession(dataset)) => {
                return Ok(Some(MlPickleBlock::SummarySession(
                    session.unwrap_or(dataset),
                )));
            }
            None => {}
        }
    }
    Ok(None)
}

/// PySpark's `RemoteModelRef` of the server-side object of the id `id`, as PySpark's fit makes the
/// `_java_obj` of its model, which the `_remote_model_obj` of its training summary is too (see
/// [`ModelCore::java_obj`]): a holder of the id, which this client never releases through it, as
/// the native object releases itself.
pub(crate) fn remote_model_ref<'py>(py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("pyspark.ml.util")?
        .getattr("RemoteModelRef")?
        .call1((id,))
}

/// Release a reference of the `RemoteModelRef` `reference` that a model was made of, as PySpark's
/// `JavaWrapper.__del__` of the model releases it (`release_ref`, which deletes the object of the
/// server through the active session at the last reference, of `del_remote_cache`): with the GIL,
/// at once on a thread of Python, else on a thread of its own, as the release may wait for the
/// server (see `deliver::run_apart`). Its error is unraisable, as an error of PySpark's `__del__`
/// is (an `AssertionError` of a reference released too often), in `deleter`, the `__del__` of the
/// class of the model, as PySpark's names its `JavaWrapper.__del__`, which leaves no exception set.
/// The release of a drop while an exception propagates (of a temporary of a frame that CPython
/// unwinds, which drops the values of the frame with the exception set) neither sees nor consumes
/// that exception, which `deliver::run_apart` sets aside on a thread of Python and sets again after
/// the release (see `deliver::preserving_pending_exception`).
fn release_reference(reference: Py<PyAny>, deleter: Option<Py<PyAny>>) {
    crate::deliver::run_apart(move |py| {
        if let Err(error) = reference.call_method0(py, pyo3::intern!(py, "release_ref")) {
            let object = deleter.as_ref().unwrap_or(&reference);
            error.write_unraisable(py, Some(object.bind(py)));
        }
    });
}

/// The object `summary` of the summary of an evaluation of the model `model`, of the attributes
/// that PySpark's `evaluate` sets of it, in its order: `_java_obj`, the id of the object that the
/// server returns, and `__source_transformer__`, the model, which it keeps, so that its pickle and
/// `copy.deepcopy` raise the error of the model's lock first, as PySpark's do (an object of no
/// public class has none).
pub(crate) fn evaluated<T>(model: &Bound<'_, PyAny>, summary: Py<T>) -> PyResult<Py<T>> {
    let py = model.py();
    let object = summary.bind(py).as_any();
    if let Ok(attributes) = object.getattr(pyo3::intern!(py, "__dict__")) {
        attributes.set_item("_java_obj", ml_java_obj(object)?)?;
        attributes.set_item("__source_transformer__", model)?;
    }
    Ok(summary)
}

/// The training summary object of `make` of the model `core` and the target `target`, of the
/// attributes that PySpark's `try_remote_fit` sets of it, in its order: `_java_obj`
/// (`"<id>.summary"`), `_summary_dataset` (the summary dataset of the fit) and
/// `_remote_model_obj`, the `RemoteModelRef` of the model, its `_java_obj`, of one more reference
/// (`summary._remote_model_obj = model._java_obj; summary._remote_model_obj.add_ref()`), which
/// PySpark's `_call_java` reads and its pickle and `copy.deepcopy` take, as its attributes (an
/// object of no public class has none).
fn training_summary_object(
    py: Python<'_>,
    make: SummaryMaker,
    core: &ModelCore,
    target: MlTarget,
) -> PyResult<Py<PyAny>> {
    let object = make(py, core, target.clone())?;
    if let Ok(attributes) = object.bind(py).getattr(pyo3::intern!(py, "__dict__")) {
        attributes.set_item("_java_obj", target.id())?;
        if let Some(dataset) = target.summary_dataset() {
            attributes.set_item(
                "_summary_dataset",
                Py::new(py, PyDataFrame::new(dataset.clone()))?,
            )?;
        }
        if target.root().is_some() {
            if let Some(reference) = core.java_obj(py)? {
                reference.call_method0(py, pyo3::intern!(py, "add_ref"))?;
                attributes.set_item("_remote_model_obj", reference)?;
            }
        }
    }
    Ok(object)
}

/// What a writer saves: an operator that is not fitted (an estimator, a transformer or an
/// evaluator), of its params, a fitted model's object, of its params, or an object of PySpark's
/// Python format, the `_instance` of the writer.
pub(super) enum WriteTarget {
    Operator(MlOperator, Params),
    Object(MlRef, Params),
    /// An object of PySpark's Python format (a Pipeline, a validator), which its `PythonSave`
    /// saves.
    Python(PythonSave),
}

/// The save of an object `instance` of PySpark's Python format to `path`, of the Python session
/// `spark` and its client `session`, and of the writer's `optionMap`, as its `MLWriter.saveImpl`
/// (the writer of a validator's model reads its `persistSubModels` of it).
pub(super) type PythonSave = for<'py> fn(
    &Bound<'py, PyAny>,
    &Bound<'py, PyAny>,
    &SparkSession,
    &str,
    &Bound<'py, PyAny>,
) -> PyResult<()>;

/// The load of an object of PySpark's Python format from `path`, of the Python session `spark` and
/// its client `session`, as its `MLReader.load`.
pub(super) type PythonLoad =
    for<'py> fn(Python<'py>, &Bound<'py, PyAny>, &SparkSession, &str) -> PyResult<Py<PyAny>>;

// PySpark's `RemoteMLWriter` over Spark Connect, a plain Python class, whose state is the
// attributes of its `__init__(instance)`, in the object's attributes: `_sparkSession` (of
// `BaseReadWrite.__init__`), `shouldOverwrite` and `optionMap` (of `MLWriter.__init__`), and
// `_instance`. PySpark's own `session`, `sparkSession`, `overwrite` and `option`, which its public
// class takes, read and set them, and `save` reads them at the save, as PySpark's does; so a writer
// constructs, copies and pickles as PySpark's plain object. A writer that `write()` makes holds the
// save of its object too, of the object's params as of then, of which the `save` of a writer saves
// its `_instance` (see `save`). Commented, not documented: PyO3 makes a doc comment the class's
// docstring, where PySpark's class has none.
#[pyclass(
    subclass,
    weakref,
    dict,
    name = "RemoteMLWriter",
    module = "pyspark.ml.connect.readwrite"
)]
pub struct PyMLWriter {
    // What the save of the writer's object saves, of its params as of the writer's making; None of
    // a writer of the constructor, a copy or a pickle.
    target: Option<WriteTarget>,
    // The error of PySpark's serialization of the params, which the save raises.
    error: Option<PyErr>,
}

impl PyMLWriter {
    pub(super) fn new(target: WriteTarget) -> Self {
        Self {
            target: Some(target),
            error: None,
        }
    }

    pub(super) fn with_error(mut self, error: Option<PyErr>) -> Self {
        self.error = error;
        self
    }

    /// The writer of the object `instance`, as PySpark's `try_remote_write` makes it
    /// (`RemoteMLWriter(instance)`): of the public class, of the attributes of PySpark's
    /// `__init__`, in its order.
    pub(super) fn object_of(self, instance: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let py = instance.py();
        let writer = new_object(py, self)?;
        writer.setattr(pyo3::intern!(py, "_sparkSession"), py.None())?;
        writer.setattr(pyo3::intern!(py, "shouldOverwrite"), false)?;
        writer.setattr(pyo3::intern!(py, "optionMap"), PyDict::new(py))?;
        writer.setattr(pyo3::intern!(py, "_instance"), instance)?;
        Ok(writer.unbind())
    }
}

#[pymethods]
impl PyMLWriter {
    // As `object.__new__` of PySpark's class: the `__init__` of the public class, PySpark's
    // `RemoteMLWriter.__init__(instance)`, takes the arguments (see `reduce::check_plain_new`).
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn new_plain(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        crate::reduce::check_plain_new(cls, args, kwargs)?;
        Ok(Self {
            target: None,
            error: None,
        })
    }

    // The reduce of a plain object, of its attributes (see `readwrite_reduce`).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        readwrite_reduce::<Self>(slf.as_any(), "_new_remote_ml_writer")
    }

    // Save to `path`, as `RemoteMLWriter.save` over Spark Connect: of the active session, of its
    // `assert session is not None` first (not the one that `session` sets, which it does not
    // read), and of the writer's `_instance`, `shouldOverwrite` and `optionMap` as of the save, in
    // that order, as it reads them. A native object is saved of its params as of now, of the
    // writer that `write()` of its native class gives (`_native_writer`), as `saveInstance` reads
    // those of the object; another object as PySpark's own `RemoteMLWriter.saveInstance` saves it
    // (`NotImplementedError: Unsupported write for <class 'NoneType'>`). The `Write` of an
    // estimator, a transformer, an evaluator or a model (of the session that holds the model) takes
    // the overwrite and the options as the protobuf fields of PySpark's `Write` take them
    // (`proto_bool`, `proto_options`); a save of PySpark's Python format overwrites of the truth of
    // `shouldOverwrite`, of the ML helper's `handleOverwrite` of the value as it is, as
    // `RemoteMLWriter.handleOverwrite` does, and its writer reads the options as PySpark's does.
    fn save(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let session = session_of(py, None)?;
        let instance = slf.getattr(pyo3::intern!(py, "_instance"))?;
        let should_overwrite = slf.getattr(pyo3::intern!(py, "shouldOverwrite"))?;
        let option_map = slf.getattr(pyo3::intern!(py, "optionMap"))?;
        let fresh = py
            .import("pyspark.ml._native_classes")?
            .call_method1("_native_writer", (&instance,))?;
        if fresh.is_none() {
            let spark = active_spark(py)?;
            py.import("pyspark.ml.connect.readwrite")?
                .getattr("RemoteMLWriter")?
                .call_method1(
                    "saveInstance",
                    (&instance, path, spark, &should_overwrite, &option_map),
                )?;
            return Ok(());
        }
        let fresh = fresh.cast_into::<Self>()?;
        let (target, error) = {
            let mut fresh = fresh.try_borrow_mut()?;
            (fresh.target.take(), fresh.error.take())
        };
        if let Some(error) = error {
            return Err(error);
        }
        let target = target.ok_or_else(|| {
            PyTypeError::new_err(format!(
                "{} has no native writer",
                instance
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_default()
            ))
        })?;
        match target {
            WriteTarget::Object(object, params) => {
                let path = proto_path(path)?;
                let should_overwrite = proto_bool(&should_overwrite)?;
                let options = proto_options(&option_map)?;
                let session = object.session().clone();
                // Of the object and its params.
                crate::interrupt::interruptible_without_parts(py, &session, || {
                    spark_connect::ml::write_object_opt(
                        &object,
                        &params,
                        &path,
                        should_overwrite,
                        &options,
                    )
                    .map_err(Box::new)
                })
            }
            WriteTarget::Operator(operator, params) => {
                let path = proto_path(path)?;
                let should_overwrite = proto_bool(&should_overwrite)?;
                let options = proto_options(&option_map)?;
                crate::interrupt::interruptible_without_parts(py, &session, || {
                    spark_connect::ml::write_operator_opt(
                        &session,
                        &operator,
                        &params,
                        &path,
                        should_overwrite,
                        &options,
                    )
                    .map_err(Box::new)
                })
            }
            WriteTarget::Python(save) => {
                // As `RemoteMLWriter.saveInstance` of a meta-algorithm: the ML helper deletes the
                // path of a true `shouldOverwrite` (`handleOverwrite`, of the path and the value as
                // PySpark's `serialize` sends them), then the Python writer saves of the active
                // session, of the params that `_native_writer` synced.
                let spark = active_spark(py)?;
                if should_overwrite.is_truthy()? {
                    let args = vec![fetch_arg_of(path)?, fetch_arg_of(&should_overwrite)?];
                    // A fetch of the ML helper, of a path and a bool.
                    crate::interrupt::interruptible_without_parts(py, &session, || {
                        spark_connect::ml::fetch(
                            &spark_connect::ml::MlTarget::helper(&session),
                            "handleOverwrite",
                            args,
                        )
                        .map(|_| ())
                        .map_err(Box::new)
                    })?;
                }
                let path = python_path(path)?;
                save(&instance, &spark, &session, &path, &option_map)
            }
        }
    }
}

/// The `should_overwrite` of a `Write` of the Python value `value` (a writer's `shouldOverwrite`),
/// as PySpark's `pb2.MlCommand.Write(should_overwrite=...)` takes it: None as unset (which the
/// server reads as false), and anything else as protobuf's bool field takes it (a bool, or an int,
/// true but of 0), with upb's TypeError of another value (`'str' object cannot be interpreted as
/// an integer`).
fn proto_bool(value: &Bound<'_, PyAny>) -> PyResult<Option<bool>> {
    if value.is_none() {
        return Ok(None);
    }
    crate::pbmsg::bool_value(value).map(Some)
}

/// The `options` of a `Write` of the Python value `options` (a writer's `optionMap`), as PySpark's
/// `pb2.MlCommand.Write(options=...)` takes it: None as none, and else the entries that upb's map
/// takes of its `update` (`MutableMapping.update`, of a mapping, an object of `keys` or pairs; of
/// the items of a tuple as the arguments, as `PyObject_CallMethod` passes a tuple), of keys and
/// values of str or UTF-8 bytes (`proto_string`).
fn proto_options(options: &Bound<'_, PyAny>) -> PyResult<HashMap<String, String>> {
    let py = options.py();
    let mut entries = HashMap::new();
    if options.is_none() {
        return Ok(entries);
    }
    let map = PyDict::new(py);
    let mut args = vec![map.clone().into_any()];
    match options.cast::<PyTuple>() {
        Ok(tuple) => args.extend(tuple.iter()),
        Err(_) => args.push(options.clone()),
    }
    py.import("collections.abc")?
        .getattr("MutableMapping")?
        .getattr("update")?
        .call1(PyTuple::new(py, args)?)?;
    for (key, value) in map.iter() {
        entries.insert(proto_string(&key)?, proto_string(&value)?);
    }
    Ok(entries)
}

/// The reduce of `object`, a writer or a reader of the native class `T`, as `object.__reduce_ex__`
/// gives it of PySpark's plain object, of the state of its attributes (see
/// `reduce::plain_object_state`): `copyreg.__newobj__` of the class of an object of a subclass,
/// which a pickle names by reference; and of an object of the public class of `T` (or of `T`),
/// which no module has by its name (`pyspark.ml.connect.readwrite.RemoteMLWriter` is PySpark's
/// class), the function `new` of the native module, which makes an object of it. So a copy and a
/// pickle of a writer or a reader are of its class and its attributes, at every protocol.
fn readwrite_reduce<'py, T: pyo3::PyTypeInfo>(
    object: &Bound<'py, PyAny>,
    new: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    let py = object.py();
    let cls = object.get_type();
    let own = cls.is(py.get_type::<T>()) || public_class::<T>(py).is_some_and(|p| cls.is(&p));
    if !own {
        return crate::reduce::plain_reduce(object);
    }
    let new = py.import("pyspark._pyspark")?.getattr(new)?;
    let args = PyTuple::empty(py).into_any();
    match crate::reduce::plain_object_state(object)? {
        Some(state) => PyTuple::new(py, [new, args, state]),
        None => PyTuple::new(py, [new, args]),
    }
}

/// A writer of no attributes, of the public class (see [`new_object`]), which `copy` and `pickle`
/// make of the reduce of a writer (see `readwrite_reduce`), as `copyreg.__newobj__` makes one of
/// PySpark's class.
#[pyfunction]
#[pyo3(name = "_new_remote_ml_writer")]
pub(crate) fn new_remote_ml_writer(py: Python<'_>) -> PyResult<Py<PyMLWriter>> {
    new_py(
        py,
        PyMLWriter {
            target: None,
            error: None,
        },
    )
}

/// A reader of no attributes, of the public class, which `copy` and `pickle` make of the reduce of
/// a reader (see [`new_remote_ml_writer`]).
#[pyfunction]
#[pyo3(name = "_new_remote_ml_reader")]
pub(crate) fn new_remote_ml_reader(py: Python<'_>) -> PyResult<Py<PyMLReader>> {
    new_py(py, PyMLReader { kind: None })
}

/// The constructor of a native class of what a `Read` returns: its object (of a model), uid and
/// params.
pub(super) type FromRead =
    for<'py> fn(Python<'py>, spark_connect::ml::ReadResult) -> PyResult<Py<PyAny>>;

// PySpark's `RemoteMLReader` over Spark Connect, a plain Python class, whose state is the
// attributes of its `__init__(clazz)`, in the object's attributes: `_sparkSession` (of
// `BaseReadWrite.__init__`) and `_clazz`. PySpark's own `session` and `sparkSession`, which its
// public class takes, read and set them, and `load` reads `_clazz` at the load, as PySpark's
// does; so a reader constructs, copies and pickles as PySpark's plain object. A reader that
// `read()` makes holds the load of its class too, of which the `load` of a reader loads (see
// `load`). Commented, not documented (see `PyMLWriter`).
#[pyclass(
    subclass,
    weakref,
    dict,
    name = "RemoteMLReader",
    module = "pyspark.ml.connect.readwrite"
)]
pub struct PyMLReader {
    // What the load of the reader's class loads; None of a reader of the constructor, a copy or a
    // pickle.
    kind: Option<ReaderKind>,
}

// The writer and the reader of no helpers, of no Python object of their own: their state is in
// their instance dicts, but the writer's error, which a save raises.
impl NativeClass for PyMLWriter {}
impl NativeClass for PyMLReader {}

/// What a reader loads: a `Read` of a JVM class, or an object of PySpark's Python format.
#[derive(Clone)]
enum ReaderKind {
    Jvm(std::borrow::Cow<'static, str>, OperatorType, FromRead),
    Python(PythonLoad),
}

impl PyMLReader {
    /// The reader of `read()` of the class `cls`: of the JVM class `jvm_class` of the native class
    /// `T` and its public class, and of another class (a Python subclass of the public class), of
    /// the JVM class that PySpark's `RemoteMLReader` names of the Python class
    /// (`cls.__module__` of `org.apache.spark` for `pyspark`, and `cls.__name__`), which the server
    /// refuses (`CONNECT_ML.UNSUPPORTED_EXCEPTION`), as of PySpark's subclass.
    pub(super) fn of_class<T: pyo3::PyTypeInfo>(
        cls: &Bound<'_, pyo3::types::PyType>,
        jvm_class: &'static str,
        op_type: OperatorType,
        from_read: FromRead,
    ) -> PyResult<Self> {
        let py = cls.py();
        let own = cls.is(py.get_type::<T>()) || public_class::<T>(py).is_some_and(|p| cls.is(&p));
        let name: std::borrow::Cow<'static, str> = if own {
            jvm_class.into()
        } else {
            let module: String = cls.getattr("__module__")?.extract()?;
            let name: String = cls.getattr("__name__")?.extract()?;
            format!("{}.{name}", module.replace("pyspark", "org.apache.spark")).into()
        };
        Ok(Self {
            kind: Some(ReaderKind::Jvm(name, op_type, from_read)),
        })
    }

    /// The reader of an object of PySpark's Python format, which `load` loads.
    pub(super) fn python(load: PythonLoad) -> Self {
        Self {
            kind: Some(ReaderKind::Python(load)),
        }
    }

    /// The reader of the class `clazz`, as PySpark's `try_remote_read` makes it
    /// (`RemoteMLReader(clazz)`): of the public class, of the attributes of PySpark's `__init__`,
    /// in its order.
    pub(super) fn object_of(self, clazz: &Bound<'_, PyAny>) -> PyResult<Py<Self>> {
        let py = clazz.py();
        let reader = new_object(py, self)?;
        reader.setattr(pyo3::intern!(py, "_sparkSession"), py.None())?;
        reader.setattr(pyo3::intern!(py, "_clazz"), clazz)?;
        Ok(reader.unbind())
    }
}

#[pymethods]
impl PyMLReader {
    // As `object.__new__` of PySpark's class: the `__init__` of the public class, PySpark's
    // `RemoteMLReader.__init__(clazz)`, takes the arguments (see `reduce::check_plain_new`).
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs), text_signature = None)]
    fn new_plain(
        cls: &Bound<'_, pyo3::types::PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        crate::reduce::check_plain_new(cls, args, kwargs)?;
        Ok(Self { kind: None })
    }

    // The reduce of a plain object, of its attributes (see `readwrite_reduce`).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        readwrite_reduce::<Self>(slf.as_any(), "_new_remote_ml_reader")
    }

    // Load from `path`, as `RemoteMLReader.load` over Spark Connect: of the active session, of its
    // `assert session is not None` first (not the one that `session` sets, which it does not
    // read), and of the reader's `_clazz` as of the load. A native class (its public class, or a
    // Python subclass of them) loads of the reader that `read()` of it gives (`_native_reader`),
    // and another class or object as PySpark's own `RemoteMLReader.loadInstance` loads it
    // (`RuntimeError: Unsupported read for <class 'int'>`).
    fn load(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let session = session_of(py, None)?;
        let clazz = slf.getattr(pyo3::intern!(py, "_clazz"))?;
        let fresh = py
            .import("pyspark.ml._native_classes")?
            .call_method1("_native_reader", (&clazz,))?;
        if fresh.is_none() {
            let spark = active_spark(py)?;
            return Ok(py
                .import("pyspark.ml.connect.readwrite")?
                .getattr("RemoteMLReader")?
                .call_method1("loadInstance", (&clazz, path, spark))?
                .unbind());
        }
        let kind = fresh.cast_into::<Self>()?.try_borrow()?.kind.clone();
        match kind {
            Some(ReaderKind::Jvm(jvm_class, op_type, from_read)) => {
                let path = proto_path(path)?;
                let read = crate::interrupt::interruptible_without_parts(py, &session, || {
                    spark_connect::ml::read(&session, jvm_class.as_ref(), op_type, &path)
                        .map_err(Box::new)
                })?;
                from_read(py, read)
            }
            Some(ReaderKind::Python(load)) => {
                let spark = active_spark(py)?;
                let path = python_path(path)?;
                load(py, &spark, &session, &path)
            }
            None => Err(PyTypeError::new_err(format!(
                "{} has no native reader",
                clazz.str()?
            ))),
        }
    }
}

/// The Python string of a param name, for the errors.
#[allow(dead_code)]
fn py_str<'py>(py: Python<'py>, text: &str) -> Bound<'py, PyString> {
    PyString::new(py, text)
}

/// The summary dataset of the fit of a model, as its PySpark class's `_summary_dataset` gives it.
#[derive(Clone, Copy)]
#[allow(dead_code)] // The areas' models use each.
pub(crate) enum SummaryDataset {
    /// The model has no training summary.
    None,
    /// The training dataset itself (the classification models, LinearRegression and
    /// GeneralizedLinearRegression).
    Train,
    /// The transform of the training dataset by the model, of no params (the clustering models).
    Transformed,
}

/// The state of a native model: its core model, the session of its fit, the training summary of
/// its fit (PySpark's `_summary`, which a remote fit sets and a load does not), and the defaults
/// that it holds itself (see [`params_api::OwnDefaults`]: of its estimator, or of the uid that a
/// loaded model was made of).
pub(super) struct ModelCore {
    pub(super) model: Box<dyn Model>,
    pub(super) session_id: String,
    pub(super) summary: Option<MlTarget>,
    /// The object of the training summary, PySpark's `_summary`, which the fit makes (see
    /// [`ModelCore::summary_object`]).
    summary_object: std::sync::OnceLock<Py<PyAny>>,
    pub(super) own_defaults: params_api::OwnDefaults,
    /// PySpark's `_java_obj` of the model, the `RemoteModelRef` of its server-side object, made at
    /// its first need (see [`ModelCore::java_obj`]): a copy holds a reference of its own of the
    /// same object, as PySpark's copies hold the same object in their attributes, which each
    /// reports to the collector (see [`ModelCore::traverse`]).
    java_obj: std::sync::OnceLock<Py<PyAny>>,
    /// The values of the `functools.cached_property` members, by name (see
    /// [`ModelCore::cached_of`]).
    cache: std::sync::Mutex<Vec<(&'static str, Py<PyAny>)>>,
    /// The values of the methods of `functools.cache`, by name, of this object only (see
    /// [`ModelCore::cached_of`]).
    method_cache: std::sync::Mutex<Vec<(&'static str, Py<PyAny>)>>,
    /// The `RemoteModelRef` that the model was made of, PySpark's `_java_obj` of it, whose object
    /// this client does not own, of the model's obligation to release a reference of it (see
    /// [`ModelCore::of_reference`]); None of the others.
    reference: Option<std::sync::Arc<HeldReference>>,
    /// The name of the class of the model whose `_java_obj` was deleted (`del model._java_obj`),
    /// whose read raises PySpark's AttributeError (see [`ModelCore::assign_java_obj`]).
    deleted_java_obj: Option<String>,
    /// Whether the object's attributes had the `_java_obj` attribute (see
    /// [`ModelCore::java_obj_change`]): an attribute of none then was deleted.
    java_obj_made: std::sync::atomic::AtomicBool,
    /// Whether the value is written, which `__traverse__` checks first (see `crate::gc::GcReady`).
    pub(super) gc_ready: crate::gc::GcReady,
}

/// A transform of a model (see [`ModelCore::transform_job`]): its DataFrame, or the copy of the
/// model that transforms the dataset without the GIL, of no borrow of the model, with the target of
/// the model's summary, which a model may read.
pub(super) enum TransformJob {
    Done(DataFrame),
    Run(Box<dyn Model>, Option<MlTarget>),
}

impl TransformJob {
    /// The DataFrame of the transform of `df`.
    pub(super) fn run(self, py: Python<'_>, df: &DataFrame) -> PyResult<DataFrame> {
        match self {
            TransformJob::Done(transformed) => Ok(transformed),
            TransformJob::Run(mut model, summary) => {
                // Of the plans of `df`, which the transform is built over, and of the summary of
                // the model, which a model may read.
                let parts = df
                    .held_parts()
                    .into_iter()
                    .chain(summary.iter().flat_map(MlTarget::held_parts));
                crate::gc::detach_of_values(py, parts, || model.transform(df).map_err(Box::new))
                    .to_pyerr()
            }
        }
    }
}

/// A change of the `_java_obj` of a model of its attributes (see [`ModelCore::java_obj_change`]).
pub(super) enum JavaObjChange<'py> {
    /// None.
    None,
    /// An assignment of the value.
    Assign(Bound<'py, PyAny>),
    /// A deletion.
    Delete,
}

/// The `RemoteModelRef` of a model that was made of it (see [`ModelCore::of_reference`]), of the
/// obligation of the model object to release a reference of it, as PySpark's `__del__` of the
/// object releases one (`release_ref`): its drop releases it (see [`release_reference`]), once the
/// model and the objects of the server of it that its transforms, its writers and its calls keep
/// are all gone, which hold it.
pub(super) struct HeldReference {
    /// The `RemoteModelRef`, until the drop, or until the model gives it up, of an assignment of
    /// its `_java_obj` (see [`ModelCore::assign_java_obj`]). Locked only while no Python code
    /// runs.
    reference: std::sync::Mutex<Option<Py<PyAny>>>,
    /// Whether a `__del__` of the model released a reference (see [`ModelCore::finalize`]), as
    /// PySpark's `__del__` releases one at each call: the drop then releases none.
    released: std::sync::atomic::AtomicBool,
    /// The `__del__` of the class of the model, in whose call PySpark's model releases the
    /// reference: the object of the unraisable error of a release that fails.
    deleter: Option<Py<PyAny>>,
}

impl HeldReference {
    fn new(reference: &Bound<'_, PyAny>, deleter: Option<Py<PyAny>>) -> Self {
        Self {
            reference: std::sync::Mutex::new(Some(reference.clone().unbind())),
            released: std::sync::atomic::AtomicBool::new(false),
            deleter,
        }
    }

    /// The `RemoteModelRef`, None once it is released or given up.
    fn reference<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        self.reference
            .lock()
            .expect("the held reference is not poisoned")
            .as_ref()
            .map(|reference| reference.bind(py).clone())
    }

    /// The `RemoteModelRef`, of no more obligation to release it: the drop releases nothing.
    fn take(&self) -> Option<Py<PyAny>> {
        self.reference
            .lock()
            .expect("the held reference is not poisoned")
            .take()
    }

    /// Visit the `RemoteModelRef` and the `__del__`, of Python's garbage collector.
    fn visit(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if let Ok(reference) = self.reference.try_lock() {
            if let Some(reference) = reference.as_ref() {
                visit.call(reference)?;
            }
        }
        if let Some(deleter) = &self.deleter {
            visit.call(deleter)?;
        }
        Ok(())
    }
}

impl Drop for HeldReference {
    fn drop(&mut self) {
        let reference = self
            .reference
            .get_mut()
            .map_or_else(|poisoned| poisoned.into_inner().take(), Option::take);
        if let Some(reference) = reference {
            if !*self.released.get_mut() {
                release_reference(reference, self.deleter.take());
            }
        }
    }
}

/// The maker of the training summary object of a model, of the model and of the target of its
/// summary (`model._summaryCls(...)` of `try_remote_fit`, which may fetch, as of `numClasses`).
pub(super) type SummaryMaker = fn(Python<'_>, &ModelCore, MlTarget) -> PyResult<Py<PyAny>>;

/// The maker of the value of a `functools.cached_property` member of a model (`trees`) or of a
/// method of `functools.cache` (`toLocal`), of the target of the model, which it fetches (see
/// [`ModelCore::cached_of`]).
#[allow(dead_code)] // The tree ensembles' `trees` and DistributedLDAModel's `toLocal` use it.
pub(super) type CachedMaker = fn(Python<'_>, MlTarget) -> PyResult<Py<PyAny>>;

/// The Python value of the Python literal `text`, the default of an argument of a method of a
/// model or a summary (see the macros).
pub(crate) fn py_default(text: &str) -> Py<PyAny> {
    Python::attach(|py| {
        let code = std::ffi::CString::new(text).expect("a Python literal has no NUL");
        py.eval(&code, None, None)
            .expect("the default of an argument is a Python literal")
            .unbind()
    })
}

/// The model of no server-side object, of PySpark's `JavaModel()` of no `java_model`: its operator
/// (of a fresh uid) and its params, which the Params API reads and sets, and no object, of which
/// the members of the server, a transform and a save raise the AssertionError of PySpark's
/// (see [`ModelCore::unbound`]).
#[derive(Clone)]
struct UnboundModel {
    operator: MlOperator,
    params: Params,
}

impl Transformer for UnboundModel {
    fn operator(&self) -> &MlOperator {
        &self.operator
    }

    fn operator_mut(&mut self) -> &mut MlOperator {
        &mut self.operator
    }

    fn params(&self) -> &Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Params {
        &mut self.params
    }

    fn transform_impl(&mut self, _df: &DataFrame) -> spark_connect_core::error::Result<DataFrame> {
        Err(spark_connect_core::error::SparkError::value_msg(
            "the model has no server-side object",
        ))
    }
}

impl Model for UnboundModel {
    fn clone_box(&self) -> Box<dyn Model> {
        Box::new(self.clone())
    }
}

/// The AssertionError of PySpark's `_call_java`, transform and save of a model of no server-side
/// object (`assert self._java_obj` ...), bare.
fn unbound_error() -> PyErr {
    pyo3::exceptions::PyAssertionError::new_err(())
}

/// The AttributeError of PySpark's read of the deleted `_java_obj` of a model of the class `class`.
fn deleted_java_obj_error(class: &str) -> PyErr {
    pyo3::exceptions::PyAttributeError::new_err(format!(
        "'{class}' object has no attribute '_java_obj'"
    ))
}

/// Whether `value` is a `RemoteModelRef` of `pyspark.ml.util`.
fn is_remote_model_ref(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let references = value
        .py()
        .import("pyspark.ml.util")?
        .getattr("RemoteModelRef")?;
    value.is_instance(&references)
}

impl ModelCore {
    /// The model of a fit of `df`, whose training summary, if the class has one, is of the summary
    /// dataset of `summary`, as PySpark's `try_remote_fit` sets `_summary` (see
    /// `spark_connect::ml::training_summary`).
    pub(super) fn fitted(
        py: Python<'_>,
        model: Box<dyn Model>,
        df: &DataFrame,
        summary: SummaryDataset,
        make_summary: Option<SummaryMaker>,
    ) -> PyResult<Self> {
        let session_id = df.spark_session().session_id().to_string();
        let summary = match (summary, model.ml_ref()) {
            (SummaryDataset::None, _) | (_, None) => None,
            (SummaryDataset::Train, Some(object)) => {
                Some(spark_connect::ml::training_summary(object, df))
            }
            (SummaryDataset::Transformed, Some(object)) => {
                // Of the plan of the training dataset, which the transform is built over.
                let dataset = crate::gc::detach_of_values(py, df.held_parts(), || {
                    spark_connect::ml::summary_transform(object, df).map_err(Box::new)
                })
                .to_pyerr()?;
                Some(spark_connect::ml::training_summary(object, &dataset))
            }
        };
        let core = Self {
            model,
            session_id,
            summary,
            summary_object: std::sync::OnceLock::new(),
            own_defaults: Vec::new(),
            java_obj: Default::default(),
            cache: Default::default(),
            method_cache: Default::default(),
            reference: None,
            deleted_java_obj: None,
            java_obj_made: Default::default(),
            gc_ready: crate::gc::GcReady::new(),
        };
        // The summary object, of the fit, as `try_remote_fit` sets `_summary` right after it.
        if let (Some(make), Some(target)) = (make_summary, core.summary.clone()) {
            let object = training_summary_object(py, make, &core, target)?;
            let _ = core.summary_object.set(object);
        }
        Ok(core)
    }

    /// Make the model of no server-side object (see [`ModelCore::unbound`]) the model of the
    /// server-side object `object`, of its uid and params, as PySpark gives a model that its
    /// constructor made the `_java_obj` of a `RemoteModelRef` (see `native_model!`'s
    /// `of_constructor`).
    pub(super) fn bind(&mut self, object: spark_connect::ml::MlRef) {
        let operator = self.model.operator().clone();
        let params = self.model.params().clone();
        self.session_id = object.session().session_id().to_string();
        self.model = Box::new(spark_connect::ml::ObjectModel::new(
            object, operator, params,
        ));
        self.java_obj = Default::default();
    }

    /// The model that a `Read` loads: of its object, the model's JVM class `jvm_class`, its uid and
    /// params, without a training summary, as PySpark's loaded models have none, and of the
    /// defaults that a model of the class `class` derives from the uid that it is made of (see
    /// [`params_api::uid_defaults`]).
    pub(super) fn loaded(
        py: Python<'_>,
        jvm_class: &str,
        class: &str,
        read: spark_connect::ml::ReadResult,
    ) -> PyResult<Self> {
        let object = read.object.ok_or_else(|| {
            PyValueError::new_err(format!("the server loaded no object of {jvm_class}"))
        })?;
        let session_id = object.session().session_id().to_string();
        let operator = MlOperator::with_uid(jvm_class, &read.uid, OperatorType::Model);
        Ok(Self {
            model: Box::new(spark_connect::ml::ObjectModel::new(
                object,
                operator,
                read.params,
            )),
            session_id,
            summary: None,
            summary_object: std::sync::OnceLock::new(),
            own_defaults: params_api::uid_defaults(py, class)?,
            java_obj: Default::default(),
            cache: Default::default(),
            method_cache: Default::default(),
            reference: None,
            deleted_java_obj: None,
            java_obj_made: Default::default(),
            gc_ready: crate::gc::GcReady::new(),
        })
    }

    /// The model of no server-side object, as PySpark's `JavaModel(java_model=None)` makes one:
    /// of the model's JVM class `jvm_class` and a new uid of the class `class`, of no set params,
    /// and of the defaults that it derives from that uid (`outputCol`, see
    /// [`params_api::uid_defaults_of`]). Its Params API, copy and write work, and the members of
    /// the server, a transform and a save raise PySpark's bare AssertionError.
    pub(super) fn unbound(py: Python<'_>, jvm_class: &str, class: &str) -> PyResult<Self> {
        Self::unbound_named(py, jvm_class, class, class)
    }

    /// The model of [`ModelCore::unbound`] of the class `class`, of a uid of the class `uid_class`,
    /// as `Identifiable._randomUID` names the uid of the class of the object (a Python
    /// subclass's).
    pub(super) fn unbound_named(
        py: Python<'_>,
        jvm_class: &str,
        class: &str,
        uid_class: &str,
    ) -> PyResult<Self> {
        let uid = params_api::random_uid(py, uid_class)?;
        let operator = MlOperator::with_uid(jvm_class, &uid, OperatorType::Model);
        Ok(Self {
            model: Box::new(UnboundModel {
                operator,
                params: Params::new(),
            }),
            session_id: String::new(),
            summary: None,
            summary_object: std::sync::OnceLock::new(),
            own_defaults: params_api::uid_defaults_of(py, class, &uid)?,
            java_obj: Default::default(),
            cache: Default::default(),
            method_cache: Default::default(),
            reference: None,
            deleted_java_obj: None,
            java_obj_made: Default::default(),
            gc_ready: crate::gc::GcReady::new(),
        })
    }

    /// The model of the server-side object of the `RemoteModelRef` `reference`, as PySpark's
    /// `JavaModel(java_model)` over Spark Connect makes one of it: a model of
    /// [`ModelCore::unbound`] (of a new uid of the class `class`, of no set params), of `reference`
    /// as its `_java_obj`, and of the object of the id of `reference` in the ML cache of the active
    /// session, whose members it reads; of no active session at its making, of the object of the
    /// session of each call, as PySpark's: of the active session for a member, which raises
    /// PySpark's bare AssertionError of none, and of the dataset's for a transform. The model, its
    /// copies and the objects of the server of it that its transforms and writers keep hold the
    /// object; as PySpark's model releases a reference of its `_java_obj` when it is collected
    /// (`release_ref`), which deletes the object at the last, the model and each of its copies
    /// release one, when the last of them drops (see [`HeldReference`]): so `Params.copy`, which
    /// adds one (`add_ref`), keeps the object while either lives, and `copy.copy`, which adds none,
    /// releases one more, as PySpark's. Another `reference` is PySpark's bare AssertionError, of
    /// its `assert isinstance(java_model, RemoteModelRef)`.
    pub(super) fn of_reference(
        py: Python<'_>,
        jvm_class: &str,
        class: &str,
        reference: &Bound<'_, PyAny>,
        deleter: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        if !is_remote_model_ref(reference)? {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        let mut core = Self::unbound(py, jvm_class, class)?;
        let held = HeldReference::new(reference, deleter);
        core.reference = Some(std::sync::Arc::new(held));
        core.bind_reference(py)?;
        Ok(core)
    }

    /// Make the model the model of `value`, its assigned `_java_obj`, or of a deleted one (of no
    /// `value`, of the class named `class`), as PySpark's model is of its `_java_obj`, which its
    /// `_call_java`, its transform, its save and its `__del__` read (an assignment of PySpark's
    /// `JavaWrapper.__init__` too, of a Python subclass's `super().__init__(java_model)`, whose
    /// object the native constructor of no arguments made of none):
    /// - the model gives its object up: it no longer releases a reference of the `RemoteModelRef`
    ///   that it was made of, nor deletes the object of its fit or load, which stays on the server
    ///   until the session ends, as PySpark's `__del__` releases a reference of its `_java_obj`
    ///   only (see [`HeldReference::take`] and `MlRef::keep`); the transforms and the summaries
    ///   that it made keep their objects;
    /// - of a `RemoteModelRef`, it is the model of its object, of an obligation to release a
    ///   reference of it, of `deleter`, the `__del__` of its class, as of
    ///   [`ModelCore::of_reference`];
    /// - of another value (None), it is of no object, whose members raise PySpark's
    ///   AssertionError, and of a deletion its AttributeError of the attribute, as PySpark's read
    ///   of the `_java_obj` raises it.
    ///
    /// An assignment of the `RemoteModelRef` that the model's `_java_obj` is already (of PySpark's
    /// `JavaModel.__init__` of its constructor's, of `JavaParams.copy` of the copy's), and one of
    /// no `RemoteModelRef` to a model of no object, change nothing but a deletion.
    pub(super) fn assign_java_obj(
        &mut self,
        py: Python<'_>,
        value: Option<&Bound<'_, PyAny>>,
        class: &str,
        deleter: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.made_java_obj();
        let reference = match value {
            Some(value) if is_remote_model_ref(value)? => Some(value),
            _ => None,
        };
        let deleted = value.is_none().then(|| class.to_string());
        let same = match (reference, self.current_java_obj(py)) {
            (Some(reference), Some(current)) => current.is(reference),
            _ => false,
        };
        if same || (reference.is_none() && self.is_of_no_object()) {
            self.deleted_java_obj = deleted;
            return Ok(());
        }
        self.give_up_object();
        self.deleted_java_obj = deleted;
        if let Some(reference) = reference {
            let held = HeldReference::new(reference, deleter);
            self.reference = Some(std::sync::Arc::new(held));
            self.bind_reference(py)?;
        }
        Ok(())
    }

    /// Whether the object's attributes had the `_java_obj` attribute (see
    /// [`ModelCore::java_obj_change`]).
    pub(super) fn java_obj_made(&self) -> bool {
        self.java_obj_made
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record that the object's attributes have the `_java_obj` attribute, which this client made
    /// (see `_native_classes._lazy_attribute`) or PySpark's `JavaWrapper.__init__` set.
    pub(super) fn made_java_obj(&self) {
        self.java_obj_made
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// What `attributes`, the attributes of the object of the model, give of its `_java_obj`, as
    /// PySpark's model is of its `_java_obj` attribute, which an assignment or a deletion sets
    /// (see [`ModelCore::assign_java_obj`]): another value than the model's `_java_obj` now, or a
    /// deletion of an attribute that the attributes had, makes a change; the value of the model now
    /// and PySpark's None of a model of no object make none. Of a shared borrow: the model changes
    /// of a mutable one only for a change, so that calls of the model on other threads, of their
    /// shared borrows, go on while nothing changes.
    pub(super) fn java_obj_change<'py>(
        &self,
        py: Python<'py>,
        attributes: &Bound<'py, PyDict>,
    ) -> PyResult<JavaObjChange<'py>> {
        match attributes.get_item(pyo3::intern!(py, "_java_obj"))? {
            Some(value) => {
                self.made_java_obj();
                let same = match self.current_java_obj(py) {
                    Some(current) => current.is(&value) && self.deleted_java_obj.is_none(),
                    None => value.is_none() && self.deleted_java_obj.is_none(),
                };
                Ok(if same {
                    JavaObjChange::None
                } else {
                    JavaObjChange::Assign(value)
                })
            }
            None if self.java_obj_made() && self.deleted_java_obj.is_none() => {
                Ok(JavaObjChange::Delete)
            }
            None => Ok(JavaObjChange::None),
        }
    }

    /// The `RemoteModelRef` that the model's `_java_obj` is now: the one that it was made of, else
    /// the one of its object, of a fit or a load, which its copies share (an assignment of
    /// PySpark's `JavaParams.copy` of the copy's), once made (see [`ModelCore::java_obj`]).
    fn current_java_obj<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyAny>> {
        match &self.reference {
            Some(held) => held.reference(py),
            None => self.java_obj.get().map(|current| current.bind(py).clone()),
        }
    }

    /// Give the object of the model up (see [`ModelCore::assign_java_obj`]): the model is of no
    /// object, of its operator and params.
    fn give_up_object(&mut self) {
        if let Some(held) = self.reference.take() {
            drop(held.take());
        }
        if let Some(object) = self.model.ml_ref() {
            object.keep();
        }
        if !self.is_unbound() {
            let operator = self.model.operator().clone();
            let params = self.model.params().clone();
            self.model = Box::new(UnboundModel { operator, params });
        }
        self.java_obj = Default::default();
    }

    /// The error of a member of the server, a transform or a save of the model of no server-side
    /// object: PySpark's AttributeError of a deleted `_java_obj`, else its bare AssertionError.
    fn no_object_error(&self) -> PyErr {
        match &self.deleted_java_obj {
            Some(class) => deleted_java_obj_error(class),
            None => unbound_error(),
        }
    }

    /// The obligation of the model to release a reference of its `RemoteModelRef`, if the model
    /// alone holds it: of no holder but the model and its object, which the model alone holds,
    /// and so of no transform, writer, call or copy that holds the object (see
    /// [`ModelCore::reference_object`]). Python's garbage collector is told of the `RemoteModelRef`
    /// then, and the `__del__` of the model releases its reference (see [`ModelCore::finalize`]);
    /// else the last of its holders to drop does.
    ///
    /// A collection reads the traverse of the model in each of its passes, all with the GIL held,
    /// and a reference that one pass reports and a later one does not is freed while the model
    /// holds it. So what the traverse reads grows only with the GIL held: the model's own `MlRef`,
    /// the `Arc` of its `HeldReference` and its `_java_obj` are cloned only with the GIL held, as
    /// every call that runs without it takes its clones first (a transform, see
    /// [`ModelCore::transform_job`]; a fetch, of the target of [`ModelCore::target`]; a writer; a
    /// cached member, see [`ModelCore::cached_of`]), and the mutexes that the traverse tries (the
    /// caches, the `HeldReference`'s) are locked only with the GIL held, of no Python code and no
    /// release of the GIL under the lock. A clone or a lock without the GIL would break this; a
    /// drop without it (of the last holder of a plan, on a thread of the runtime) only makes a
    /// later pass report more, which keeps the reference alive. The instance's `__dict__`, which
    /// mirrors `_java_obj`, reports it too, and so would hide a break of it: a test watches the
    /// traverses with the GIL held while the calls run (`_ml_watch_traverses`).
    fn exclusive_reference(&self) -> Option<&HeldReference> {
        let held = self.reference.as_ref()?;
        let bound = self.model.ml_ref().is_some_and(MlRef::is_unshared);
        (std::sync::Arc::strong_count(held) == 1 + usize::from(bound)).then_some(held.as_ref())
    }

    /// What PySpark's `__del__` of the model does (`try_remote_del`), which the `__del__` of the
    /// public class runs: the AttributeError of a deleted `_java_obj`, which it reads, and else the
    /// `RemoteModelRef` of the model of which it releases a reference now, of the model alone (see
    /// [`ModelCore::exclusive_reference`]), at each call, as PySpark's; the error of the release is
    /// the error of the `__del__`, which Python reports as unraisable in it, as of PySpark's. The
    /// caller releases it of no borrow of the model, as the release may wait for the server. A
    /// model that others hold the object of leaves the release to the last of them, which releases
    /// one if no `__del__` did (see [`HeldReference`]). A model of a fit or a load releases its
    /// object at its drop, of its last holder.
    pub(super) fn finalize<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(class) = &self.deleted_java_obj {
            return Err(deleted_java_obj_error(class));
        }
        let Some(held) = self.exclusive_reference() else {
            return Ok(None);
        };
        let reference = held.reference(py);
        if reference.is_some() {
            held.released
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(reference)
    }

    /// Visit the Python objects that the model holds alone, of Python's garbage collector, as
    /// PySpark's model holds them as its attributes (see `__traverse__` of the native models): the
    /// object of its training summary (`_summary`), which may keep the model (`summary.model =
    /// model`), its own defaults, the values of its cached members, its `RemoteModelRef` (PySpark's
    /// `_java_obj`), of a reference of its own (a copy has its own of the same object), and the
    /// `RemoteModelRef` that it was made of while it alone holds its obligation (see
    /// [`ModelCore::exclusive_reference`], whose invariant keeps each decision of this traverse
    /// from losing a reference between the passes of a collection). The Python objects below the
    /// plan of the summary dataset of its training summary (`summary`, an `MlTarget`) are not
    /// reported: a cycle through them is not collected.
    pub(super) fn traverse(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if !self.gc_ready.get() {
            return Ok(());
        }
        if let Some(summary) = self.summary_object.get() {
            visit.call(summary)?;
        }
        for (_, value) in &self.own_defaults {
            visit.call(value)?;
        }
        for cache in [&self.cache, &self.method_cache] {
            if let Ok(values) = cache.try_lock() {
                for (_, value) in values.iter() {
                    visit.call(value)?;
                }
            }
        }
        if let Some(reference) = self.java_obj.get() {
            visit.call(reference)?;
        }
        if let Some(held) = self.exclusive_reference() {
            held.visit(visit)?;
        }
        Ok(())
    }

    /// Break a reference cycle that the garbage collector found unreachable: the model drops the
    /// Python objects that [`ModelCore::traverse`] reported. Its `__del__` ran before, which
    /// released the reference of its `RemoteModelRef`.
    pub(super) fn clear(&mut self) {
        if !self.gc_ready.get() {
            return;
        }
        self.summary_object = std::sync::OnceLock::new();
        self.own_defaults.clear();
        for cache in [&mut self.cache, &mut self.method_cache] {
            if let Ok(values) = cache.get_mut() {
                values.clear();
            }
        }
        self.java_obj = std::sync::OnceLock::new();
        if self.exclusive_reference().is_some() {
            self.reference = None;
        }
    }

    /// Make the model of [`ModelCore::of_reference`] the model of the object of the id of its
    /// `RemoteModelRef` in the active session, if there is one (see
    /// [`ModelCore::reference_object`]).
    fn bind_reference(&mut self, py: Python<'_>) -> PyResult<()> {
        let Some(session) = crate::session::active_session(py) else {
            return Ok(());
        };
        let session = session.borrow(py).session.clone();
        if let Some(object) = self.reference_object(py, &session)? {
            self.bind(object);
        }
        Ok(())
    }

    /// The object of the id of the `RemoteModelRef` of a model of [`ModelCore::of_reference`] in
    /// `session`, which keeps the model's obligation to release a reference of it (see
    /// [`HeldReference`]); None of another model.
    fn reference_object(&self, py: Python<'_>, session: &SparkSession) -> PyResult<Option<MlRef>> {
        let Some(held) = &self.reference else {
            return Ok(None);
        };
        let Some(reference) = held.reference(py) else {
            return Ok(None);
        };
        let id: String = reference.getattr("ref_id")?.extract()?;
        let held = std::sync::Arc::clone(held);
        Ok(Some(MlRef::of_reference(session, &id, move || drop(held))))
    }

    /// The model of a model of [`ModelCore::of_reference`] of no active session at its making, of
    /// the object of its `RemoteModelRef` in `session`, the session of a call, of its operator and
    /// params, as PySpark's `_call_java` reads the object in the active session and its transform
    /// in the dataset's; None of another model.
    fn reference_model(
        &self,
        py: Python<'_>,
        session: &SparkSession,
    ) -> PyResult<Option<Box<dyn Model>>> {
        if !self.is_unbound() {
            return Ok(None);
        }
        let Some(object) = self.reference_object(py, session)? else {
            return Ok(None);
        };
        let operator = self.model.operator().clone();
        let params = self.model.params().clone();
        Ok(Some(Box::new(spark_connect::ml::ObjectModel::new(
            object, operator, params,
        ))))
    }

    /// Whether the model is of no server-side object (see [`ModelCore::unbound`]).
    pub(super) fn is_unbound(&self) -> bool {
        // The model, of its concrete type: `as_any` of the box would be of the box's own.
        self.model.as_ref().as_any().is::<UnboundModel>()
    }

    pub(super) fn duplicate(&self, py: Python<'_>) -> Self {
        Self {
            model: self.model.clone_box(),
            session_id: self.session_id.clone(),
            summary: self.summary.clone(),
            // A copy shares the summary object, as `copy.copy` of PySpark's `Params.copy` shares
            // its `_summary`.
            summary_object: match self.summary_object.get() {
                Some(object) => std::sync::OnceLock::from(object.clone_ref(py)),
                None => std::sync::OnceLock::new(),
            },
            own_defaults: params_api::clone_own_defaults(py, &self.own_defaults),
            // A copy has the `RemoteModelRef` of the model, of a reference of its own, as
            // `copy.copy` and PySpark's `JavaParams.copy` share its `_java_obj`.
            java_obj: match self.java_obj.get() {
                Some(reference) => std::sync::OnceLock::from(reference.clone_ref(py)),
                None => std::sync::OnceLock::new(),
            },
            // A copy shares the values cached before it, as of the `__dict__` that `copy.copy`
            // copies.
            cache: std::sync::Mutex::new(params_api::clone_own_defaults(
                py,
                &self.cache.lock().expect("the cache is not poisoned"),
            )),
            // A copy is another object of `functools.cache`.
            method_cache: Default::default(),
            // A copy has an obligation of its own, as PySpark's `__del__` of a copy releases one.
            reference: self.reference.as_ref().and_then(|held| {
                let reference = held.reference(py)?;
                let deleter = held.deleter.as_ref().map(|deleter| deleter.clone_ref(py));
                Some(std::sync::Arc::new(HeldReference::new(&reference, deleter)))
            }),
            deleted_java_obj: self.deleted_java_obj.clone(),
            // The attributes of a copy are of none until it is made of them.
            java_obj_made: Default::default(),
            gc_ready: crate::gc::GcReady::new(),
        }
    }

    /// The copy of the model that a copy of it makes (see [`ModelCore::duplicate`]): of a model of
    /// a `RemoteModelRef`, bound to its object of an obligation of its own to release a reference
    /// of it, as the `__del__` of PySpark's copy releases one (see [`ModelCore::of_reference`]).
    pub(super) fn duplicate_of(&self, py: Python<'_>) -> PyResult<Self> {
        let mut copy = self.duplicate(py);
        // Of the object of the session of this model's.
        let session = match (&copy.reference, self.model.ml_ref()) {
            (Some(_), Some(object)) => object.session().clone(),
            _ => return Ok(copy),
        };
        if let Some(object) = copy.reference_object(py, &session)? {
            copy.bind(object);
        }
        Ok(copy)
    }

    /// Whether the model is of no server-side object and of no `RemoteModelRef`, PySpark's model
    /// of a `_java_obj` of None, which `copy.deepcopy` and `pickle` copy (a `RemoteModelRef` holds
    /// a lock, which they refuse).
    pub(super) fn is_of_no_object(&self) -> bool {
        self.is_unbound() && self.reference.is_none()
    }

    /// The value of the cached member `key` of the model `slf`, of its core `core`: of a
    /// `functools.cached_property` (`trees`), or of `method`, of a method of no arguments of
    /// `functools.cache` (`toLocal`), whose cache a copy does not share. It is made by `make` of
    /// the target of the model once, and is the same object after. `make` fetches: the model is
    /// borrowed to read the cache and the target and to keep the value, but not while `make` runs,
    /// so that the calls of the model on other threads go on (see [`ModelCore::transform_job`]).
    /// Of two threads that make it at once, both return the value that was kept first.
    #[allow(dead_code)] // The tree ensembles' `trees` and DistributedLDAModel's `toLocal` use it.
    pub(super) fn cached_of<T: PyClass<Frozen = False>>(
        slf: &Bound<'_, T>,
        core: for<'a> fn(&'a T) -> &'a ModelCore,
        method: bool,
        key: &'static str,
        make: CachedMaker,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let target = {
            let this = slf.try_borrow()?;
            let core = core(&this);
            let cached = core
                .cache_of(method)
                .lock()
                .expect("the cache is not poisoned");
            if let Some(value) = params_api::own_default_of(py, &cached, key) {
                return Ok(value);
            }
            drop(cached);
            core.target()?
        };
        let value = make(py, target)?;
        let this = slf.try_borrow()?;
        Self::cached_in(py, core(&this).cache_of(method), key, || Ok(value))
    }

    /// The cache of the methods of `functools.cache` of `method`, else of the cached members.
    fn cache_of(&self, method: bool) -> &std::sync::Mutex<Vec<(&'static str, Py<PyAny>)>> {
        if method {
            &self.method_cache
        } else {
            &self.cache
        }
    }

    /// The value of the cached member `key`, as a `functools.cached_property`: made by `make`
    /// once, and the same object after (see [`ModelCore::cached_of`]).
    pub(super) fn cached(
        &self,
        py: Python<'_>,
        key: &'static str,
        make: impl FnOnce() -> PyResult<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        Self::cached_in(py, &self.cache, key, make)
    }

    /// The value of `key` of `cache`, made by `make` once.
    fn cached_in(
        py: Python<'_>,
        cache: &std::sync::Mutex<Vec<(&'static str, Py<PyAny>)>>,
        key: &'static str,
        make: impl FnOnce() -> PyResult<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let lock = || cache.lock().expect("the cache is not poisoned");
        if let Some(value) = params_api::own_default_of(py, &lock(), key) {
            return Ok(value);
        }
        // Made without the lock, as `make` runs Python.
        let value = make()?;
        let mut cache = lock();
        if let Some(existing) = params_api::own_default_of(py, &cache, key) {
            return Ok(existing);
        }
        cache.push((key, value.clone_ref(py)));
        Ok(value)
    }

    /// The target of the training summary of the model `slf`, or the RuntimeError of
    /// `HasTrainingSummary.summary` over Spark Connect of a model without one (a loaded one), of
    /// the name of the class of `slf`.
    pub(super) fn training_summary_target(&self, slf: &Bound<'_, PyAny>) -> PyResult<MlTarget> {
        match &self.summary {
            Some(target) => Ok(target.clone()),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "No training summary available for this {}",
                slf.get_type().name()?
            ))),
        }
    }

    /// The training summary of the model `slf`, as `HasTrainingSummary.summary` over Spark Connect
    /// returns `_summary`: the object that the fit made, the same on every call (of `make`, if the
    /// class makes none at the fit), or PySpark's RuntimeError of a model without one (see
    /// [`ModelCore::training_summary_target`]). Every class of a training summary makes it at the
    /// fit (`summary` of `native_model!`), so a read of it fetches nothing (`numClasses`) under the
    /// borrow of the model that its caller holds (see [`ModelCore::transform_job`]).
    pub(super) fn summary_object(
        &self,
        slf: &Bound<'_, PyAny>,
        make: SummaryMaker,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if let Some(object) = self.summary_object.get() {
            return Ok(object.clone_ref(py));
        }
        let object = training_summary_object(py, make, self, self.training_summary_target(slf)?)?;
        Ok(self.summary_object.get_or_init(|| object).clone_ref(py))
    }

    /// PySpark's `_java_obj` of the model: the `RemoteModelRef` of its server-side object, None of
    /// a model of none. It is made once, at its first need (the object of the model, of its
    /// attributes, or the training summary of a fit, whose `_remote_model_obj` it is), and is the
    /// same object of the copies of the model, as PySpark's fit gives the model and its summary
    /// one and its copies share it.
    pub(super) fn java_obj(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if let Some(class) = &self.deleted_java_obj {
            return Err(deleted_java_obj_error(class));
        }
        // The `RemoteModelRef` that the model was made of, of PySpark's `JavaModel.__init__`.
        if let Some(held) = &self.reference {
            return Ok(held.reference(py).map(Bound::unbind));
        }
        let Some(object) = self.model.ml_ref() else {
            return Ok(None);
        };
        if let Some(reference) = self.java_obj.get() {
            return Ok(Some(reference.clone_ref(py)));
        }
        let reference = remote_model_ref(py, object.id())?.unbind();
        Ok(Some(self.java_obj.get_or_init(|| reference).clone_ref(py)))
    }

    /// The target of the methods of the model's object.
    pub(super) fn target(&self) -> PyResult<MlTarget> {
        if self.is_unbound() {
            // Of a model of a `RemoteModelRef` of no active session at its making, the object in
            // the active session of the call (see `ModelCore::of_reference`).
            return Python::attach(|py| {
                let session =
                    crate::session::active_session(py).ok_or_else(|| self.no_object_error())?;
                let session = session.borrow(py).session.clone();
                match self.reference_model(py, &session)? {
                    Some(model) => model.model_target().to_pyerr(),
                    None => Err(self.no_object_error()),
                }
            });
        }
        self.model.model_target().to_pyerr()
    }

    pub(super) fn uid(&self) -> String {
        self.model.operator().uid.clone()
    }

    /// The error of PySpark's serialization of the model's params: of a value that it cannot send
    /// (see [`params_api::unsendable_error`]), the first of them by name.
    pub(super) fn serialization_error(&self, py: Python<'_>) -> Option<PyErr> {
        let params = self.model.params();
        let jvm = params
            .to_proto()
            .params
            .into_keys()
            .filter_map(|key| key.strip_prefix("\u{0}unsendable:").map(str::to_string))
            .min()?;
        Some(params_api::unsendable_error(py, params, &jvm))
    }

    /// The transform of `df` by the model, of `df` as it is, as `try_remote_transform_relation` of
    /// `pyspark.ml` sends it (see `PyMLModel::transform_dataframe`), of what it needs of the model
    /// taken with the GIL (see [`TransformJob`]): the caller borrows the model only for that, and
    /// not while the transform runs without the GIL, so that the calls of the model on other
    /// threads go on (PySpark's models are of several threads at once).
    pub(super) fn transform_job(&self, py: Python<'_>, df: &DataFrame) -> PyResult<TransformJob> {
        // As `try_remote_transform_relation`, which asserts the model's object before it
        // serializes the params: of a model of a `RemoteModelRef` of no active session at its
        // making, of the object in the dataset's session (see `ModelCore::of_reference`).
        let session = df.spark_session();
        let (model, model_session) = match self.reference_model(py, &session)? {
            Some(model) => (model, session.session_id()),
            None if self.is_unbound() => return Err(self.no_object_error()),
            None => (self.model.clone_box(), self.session_id.as_str()),
        };
        if let Some(error) = self.serialization_error(py) {
            return Err(error);
        }
        if session.session_id() != model_session {
            let placeholder =
                DataFrame::from_sent_relation(session.clone(), df.placeholder_relation());
            let relation = spark_connect_proto::Relation {
                rel_type: Some(spark_connect_proto::relation::RelType::MlRelation(
                    Box::new(model.build_ml_relation(&placeholder).to_pyerr()?),
                )),
                ..Default::default()
            };
            Ok(TransformJob::Done(
                DataFrame::from_relation(session, relation, None).to_pyerr()?,
            ))
        } else {
            Ok(TransformJob::Run(model, self.summary.clone()))
        }
    }

    /// The writer of the model's object, of its params, as `RemoteMLWriter` writes a model.
    pub(super) fn writer(&self, py: Python<'_>) -> PyResult<PyMLWriter> {
        // Of a model of a `RemoteModelRef` of no active session at its making, the object in the
        // active session, which PySpark's `RemoteMLWriter.save` saves of (see
        // `ModelCore::of_reference`).
        let active = crate::session::active_session(py).map(|s| s.borrow(py).session.clone());
        let model = match &active {
            Some(session) => self.reference_model(py, session)?,
            None => None,
        };
        // The writer of a model of no object, as PySpark's `RemoteMLWriter` of one, whose save
        // raises the AssertionError.
        if self.is_unbound() && model.is_none() {
            let operator = self.model.operator().clone();
            let target = WriteTarget::Operator(operator, self.model.params().clone());
            return Ok(PyMLWriter::new(target).with_error(Some(self.no_object_error())));
        }
        let model = model.as_deref().unwrap_or(self.model.as_ref());
        let object = model
            .ml_ref()
            .ok_or_else(|| PyValueError::new_err("the model has no server-side object to save"))?;
        Ok(PyMLWriter::new(WriteTarget::Object(
            object.clone(),
            self.model.params().clone(),
        ))
        .with_error(self.serialization_error(py)))
    }
}

/// Set the param `jvm` of the model params `params` to `value`, as the setters of a model set it:
/// a value of the converter `identity` as the literal of `LiteralExpression._from_value`, and a
/// value that PySpark cannot send (an int beyond a long, or a list of ints whose literal fails)
/// under the reserved key that keeps it (see [`params_api::unsendable_key`]).
pub(super) fn put_model_param(
    py: Python<'_>,
    params: Params,
    jvm: &str,
    value: Converted,
) -> PyResult<Params> {
    let params = params.clear_param(&params_api::unsendable_key(jvm));
    Ok(match value {
        Converted::Null => params.set_param_null(jvm),
        Converted::Unsendable(text) => params
            .clear_param(jvm)
            .set_param_string(&params_api::unsendable_key(jvm), &text),
        Converted::Int(value) => params.set_param_int_narrow(jvm, value),
        Converted::Float(value) => params.set_param_double(jvm, value),
        Converted::Str(value) => params.set_param_string(jvm, &value),
        Converted::Bool(value) => params.set_param_bool(jvm, value),
        Converted::StrList(values) => params.set_param_string_array(jvm, values),
        Converted::FloatList(values) => params.set_param_double_array(jvm, values),
        Converted::IntList(literal) => params.set_param_literal(jvm, literal),
        Converted::Vector(value) => params.set_param_vector(jvm, value),
        Converted::FloatListList(value) => params.set_param_double_arrays(jvm, value),
        Converted::Matrix(value) => params.set_param_matrix(jvm, value),
        Converted::Raw(value) => {
            let column = crate::functions::to_value_column(value.bind(py))?;
            match column.expression().node() {
                spark_connect::expression::ExpressionNode::Literal(literal) => {
                    params.set_param_literal(jvm, literal.to_literal_proto())
                }
                _ => {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "PySparkTypeError",
                        "UNSUPPORTED_DATA_TYPE",
                        &[("data_type", &value.bind(py).get_type().name()?.to_string())],
                    ))
                }
            }
        }
    })
}

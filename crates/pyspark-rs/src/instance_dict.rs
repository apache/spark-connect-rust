//! The release of the instance `__dict__` of the `#[pyclass(dict)]` classes, which PyO3 0.28
//! does not do when an instance is freed.
//!
//! PySpark's classes are plain Python classes, whose instance `__dict__` CPython releases with
//! the instance. PyO3 0.28's dealloc of an instance of a `dict` class empties the dict
//! (`PyDict_Clear` in `PyClassObjectContents::dealloc`) but keeps its reference to it, so every
//! instance whose dict was made, by an attribute set, `vars()`, `dir()`, a monkeypatched method
//! or a `__reduce__` that reads it, leaves its dict behind. PyO3 0.29 releases it
//! (`release_dict`, a `Py_CLEAR`), but it cannot be used here, as arrow-pyarrow pins PyO3 0.28.
//!
//! So [`release_instance_dicts`] wraps, at module init, the `tp_dealloc` of each class whose
//! instances have the dict. The wrapper takes the dict out of its slot, runs PyO3's dealloc,
//! which then finds no dict, and releases the dict once the instance is gone, as CPython's
//! `subtype_dealloc` releases it after the weak references are cleared. Its items are then freed
//! by `dict_dealloc`, under CPython's trashcan: PyO3's `PyDict_Clear` freed them within the
//! dealloc, so that freeing a long chain of instances, each an attribute of the next, overflowed
//! the C stack. Only that changes: the dict is still CPython's instance dict, which its attribute
//! lookup reads and writes as before, and freeing an instance, with a dict or without, costs a
//! call and a few loads more.
//!
//! A Python subclass keeps the dict slot of its base, and CPython's `subtype_dealloc` calls the
//! `tp_dealloc` of the base, so it is covered. A class that `extends` another in Rust has a
//! `tp_dealloc` of its own, which runs the Rust dealloc of its bases directly, so it is wrapped
//! too: a new `dict` class, or a class that extends one, belongs in the list below.
//!
//! On CPython 3.11 and 3.12, PyO3 0.28 also loses the dict that `object.__new__` makes for each
//! instance of these classes, as it writes the fields of the instance over it. No code of the
//! extension runs in between, so only PyO3 0.29 fixes that one (`create_class_object_of_type`).
//!
//! Remove this module after the upgrade to PyO3 0.29 or later. Left in place, it is harmless
//! there: PyO3's own `Py_CLEAR` then finds the slot empty.

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

use pyo3::ffi;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::catalog::PyCatalog;
use crate::column::PyColumn;
use crate::conf::PyRuntimeConf;
use crate::dataframe::{
    PyDataFrame, PyDataFrameNaFunctions, PyDataFrameWriter, PyDataFrameWriterV2, PyMergeIntoWriter,
};
use crate::datasource::PyDataSourceRegistration;
use crate::group::{PyCoGroupedData, PyGroupedData};
use crate::observation::PyObservation;
use crate::readwriter::PyDataFrameReader;
use crate::resource::{
    PyExecutorResourceRequests, PyResourceProfile, PyResourceProfileBuilder, PyTaskResourceRequests,
};
use crate::session::PySparkSession;
use crate::stat::PyStatFunctions;
use crate::streaming::{
    PyDataStreamReader, PyDataStreamWriter, PyStreamingQuery, PyStreamingQueryManager,
};
use crate::tablearg::PyTableArg;
use crate::tvf::PyTableValuedFunction;
use crate::types::{
    PyAnsiIntervalType, PyAnyTimeType, PyArrayType, PyAtomicType, PyBinaryType, PyBooleanType,
    PyByteType, PyCalendarIntervalType, PyCharType, PyDataType, PyDateType, PyDatetimeType,
    PyDayTimeIntervalType, PyDecimalType, PyDoubleType, PyFloatType, PyFractionalType,
    PyGeographyType, PyGeometryType, PyIntegerType, PyIntegralType, PyLongType, PyMapType,
    PyNullType, PyNumericType, PyShortType, PySpatialType, PyStringType, PyStructField,
    PyStructType, PyTimeType, PyTimestampLTZNanosType, PyTimestampNTZNanosType, PyTimestampNTZType,
    PyTimestampType, PyVarcharType, PyVariantType, PyYearMonthIntervalType,
};
use crate::window::PyWindowSpec;

/// Where `tp_dealloc` is in CPython's `PyTypeObject`: after `PyObject_VAR_HEAD`, `tp_name`,
/// `tp_basicsize` and `tp_itemsize`, as in every CPython 3. The limited API keeps the struct
/// opaque, so the field is written only where it holds what `PyType_GetSlot` reads for it.
const TP_DEALLOC_OFFSET: usize =
    size_of::<ffi::PyVarObject>() + size_of::<*const c_char>() + 2 * size_of::<ffi::Py_ssize_t>();

/// What the wrapper of a class needs, as `tp_dealloc` gets only the instance.
struct Hook {
    /// The `tp_dictoffset` of the class, which its Python subclasses keep.
    dict_offset: ffi::Py_ssize_t,
    /// PyO3's `tp_dealloc` of the class.
    dealloc: ffi::destructor,
}

/// Wraps the `tp_dealloc` of each class with a wrapper and a hook of its own.
macro_rules! release_dicts_of {
    ($py:expr; $($class:ty),+ $(,)?) => {$({
        static HOOK: OnceLock<Hook> = OnceLock::new();
        unsafe extern "C" fn dealloc(obj: *mut ffi::PyObject) {
            // SAFETY: CPython calls it for an instance of the class, or of a Python subclass.
            unsafe { release_dict_and_dealloc(obj, &HOOK) }
        }
        wrap_dealloc(&$py.get_type::<$class>(), &HOOK, dealloc)?;
    })+};
}

/// Makes the instances of the classes with a `__dict__`, the `dict` classes and those that
/// extend one, release their dict when they are freed, as instances of Python classes do.
pub(crate) fn release_instance_dicts(py: Python<'_>) -> PyResult<()> {
    release_dicts_of!(py;
        PyCatalog,
        PyColumn,
        PyRuntimeConf,
        PyDataFrame,
        PyDataFrameNaFunctions,
        PyMergeIntoWriter,
        PyDataFrameWriter,
        PyDataFrameWriterV2,
        PyDataSourceRegistration,
        PyGroupedData,
        PyCoGroupedData,
        PyObservation,
        PyDataFrameReader,
        PyExecutorResourceRequests,
        PyTaskResourceRequests,
        PyResourceProfileBuilder,
        PyResourceProfile,
        PySparkSession,
        PyStatFunctions,
        PyDataStreamReader,
        PyDataStreamWriter,
        PyStreamingQuery,
        PyStreamingQueryManager,
        PyTableArg,
        PyTableValuedFunction,
        PyStructField,
        PyWindowSpec,
        // `DataType` and the classes that extend it, which keep its dict.
        PyDataType,
        PyAtomicType,
        PyNumericType,
        PyIntegralType,
        PyFractionalType,
        PyDatetimeType,
        PyAnyTimeType,
        PyAnsiIntervalType,
        PySpatialType,
        PyNullType,
        PyBooleanType,
        PyByteType,
        PyShortType,
        PyIntegerType,
        PyLongType,
        PyFloatType,
        PyDoubleType,
        PyDecimalType,
        PyStringType,
        PyBinaryType,
        PyDateType,
        PyTimestampType,
        PyTimestampNTZType,
        PyTimestampNTZNanosType,
        PyTimestampLTZNanosType,
        PyArrayType,
        PyMapType,
        PyStructType,
        PyCharType,
        PyVarcharType,
        PyTimeType,
        PyCalendarIntervalType,
        PyYearMonthIntervalType,
        PyDayTimeIntervalType,
        PyVariantType,
        PyGeometryType,
        PyGeographyType,
    );
    Ok(())
}

/// Installs `wrapper` as the `tp_dealloc` of `class`, once its hook is set. A class without an
/// instance dict at a fixed offset, or whose type object is not laid out as CPython's, is left
/// as it is.
fn wrap_dealloc(
    class: &Bound<'_, PyType>,
    hook: &'static OnceLock<Hook>,
    wrapper: ffi::destructor,
) -> PyResult<()> {
    let py = class.py();
    let dict_offset: ffi::Py_ssize_t = class.getattr(intern!(py, "__dictoffset__"))?.extract()?;
    if dict_offset <= 0 || hook.get().is_some() {
        return Ok(());
    }
    let ty = class.as_type_ptr();
    // SAFETY: `ty` is a type object, pointer aligned and larger than `TP_DEALLOC_OFFSET` plus a
    // pointer; the field is written only if it holds the `tp_dealloc` that CPython reports.
    unsafe {
        let dealloc = ffi::PyType_GetSlot(ty, ffi::Py_tp_dealloc);
        if dealloc.is_null() {
            drop(PyErr::take(py));
            return Ok(());
        }
        let field = ty.cast::<u8>().add(TP_DEALLOC_OFFSET).cast::<*mut c_void>();
        if field.read() != dealloc {
            return Ok(());
        }
        let wrapped = Hook {
            dict_offset,
            dealloc: std::mem::transmute::<*mut c_void, ffi::destructor>(dealloc),
        };
        if hook.set(wrapped).is_ok() {
            field.write(wrapper as *mut c_void);
        }
    }
    Ok(())
}

/// Runs PyO3's dealloc of `obj` without its dict, which it would empty and keep, and then
/// releases the dict, once the instance and its weak references are gone.
///
/// # Safety
/// `obj` is an instance being freed, of the class of `hook` or of a Python subclass of it.
#[inline(always)]
unsafe fn release_dict_and_dealloc(obj: *mut ffi::PyObject, hook: &OnceLock<Hook>) {
    let Some(hook) = hook.get() else {
        // The wrapper is installed only after its hook is set.
        std::process::abort()
    };
    unsafe {
        let slot: *mut *mut ffi::PyObject = obj.byte_offset(hook.dict_offset).cast();
        let dict = slot.replace(std::ptr::null_mut());
        (hook.dealloc)(obj);
        if !dict.is_null() {
            ffi::Py_DECREF(dict);
        }
    }
}

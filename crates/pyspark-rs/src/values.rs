//! The value classes of the newer types, `pyspark.sql.types.{VariantVal, Geometry, Geography}`:
//! PySpark's plain Python classes, embedded from `values.py`.
//!
//! PySpark's `__init__` stores its arguments, as they are given, as the instance attributes
//! `value` and `metadata`, or `wkb` and `srid`, which are writable, deletable and in the
//! instance `__dict__`, and a subclass's `__init__` may set them itself. A value pickles as any
//! plain class does, which the server's Python workers, of PySpark, unpickle. The classes are
//! not `#[pyclass(dict)]` types, as PyO3 0.28 clears but does not release the `__dict__` of
//! such an instance when it is freed, so that every value would keep a dict alive.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyType};

static CLASSES: PyOnceLock<ValueClasses> = PyOnceLock::new();
const VALUES_SOURCE: &std::ffi::CStr = pyo3::ffi::c_str!(include_str!("values.py"));

struct ValueClasses {
    variant_val: Py<PyType>,
    geography: Py<PyType>,
    geometry: Py<PyType>,
}

/// The classes of the embedded module, which is loaded once. Its source imports PySpark modules
/// only inside methods, avoiding import cycles while `pyspark.sql.types` is importing this
/// extension. The synthetic filename avoids embedding a build-machine path in wheels.
fn classes(py: Python<'_>) -> PyResult<&ValueClasses> {
    CLASSES.get_or_try_init(py, || {
        let module =
            PyModule::from_code(py, VALUES_SOURCE, c"pyspark/_values.py", c"pyspark._values")?;
        let class = |name: &str| -> PyResult<Py<PyType>> {
            Ok(module.getattr(name)?.cast_into::<PyType>()?.unbind())
        };
        Ok(ValueClasses {
            variant_val: class("VariantVal")?,
            geography: class("Geography")?,
            geometry: class("Geometry")?,
        })
    })
}

/// Expose the classes as `pyspark._pyspark.{VariantVal, Geography, Geometry}`, which
/// `pyspark.sql.types` re-exports.
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let classes = classes(py)?;
    module.add("VariantVal", classes.variant_val.bind(py))?;
    module.add("Geography", classes.geography.bind(py))?;
    module.add("Geometry", classes.geometry.bind(py))
}

/// The `VariantVal` class, to check the values of PySpark's checks of it.
pub(crate) fn variant_val_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    Ok(classes(py)?.variant_val.bind(py))
}

/// The `Geography` class, to check the values of PySpark's checks of it.
pub(crate) fn geography_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    Ok(classes(py)?.geography.bind(py))
}

/// The `Geometry` class, to check the values of PySpark's checks of it.
pub(crate) fn geometry_class(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    Ok(classes(py)?.geometry.bind(py))
}

/// The VariantVal of the bytes `value` and `metadata`, as PySpark's `VariantVal(value,
/// metadata)` of a collected VARIANT: an instance of the class, made by its `__new__`, with
/// its two attributes, but without running the Python `__init__` for every value.
pub(crate) fn variant_val<'py>(
    py: Python<'py>,
    value: &[u8],
    metadata: &[u8],
) -> PyResult<Bound<'py, PyAny>> {
    let class = variant_val_class(py)?;
    let variant = class.call_method1(intern!(py, "__new__"), (class,))?;
    variant.setattr(intern!(py, "value"), PyBytes::new(py, value))?;
    variant.setattr(intern!(py, "metadata"), PyBytes::new(py, metadata))?;
    Ok(variant)
}

//! PyO3-backed value holders returned/consumed by the client for the newer types,
//! mirroring `pyspark.sql.types.{VariantVal, Geometry, Geography}`.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

/// A Variant value (the `value`/`metadata` binary components). Mirrors
/// `pyspark.sql.types.VariantVal`.
#[pyclass(
    name = "VariantVal",
    module = "pyspark.sql.types",
    weakref,
    dict,
    subclass
)]
pub struct PyVariantVal {
    pub(crate) value: Vec<u8>,
    pub(crate) metadata: Vec<u8>,
}

impl PyVariantVal {
    /// Build from raw (value, metadata) bytes (used when materializing a VARIANT column).
    pub(crate) fn from_parts(value: Vec<u8>, metadata: Vec<u8>) -> Self {
        PyVariantVal { value, metadata }
    }
}

#[pymethods]
impl PyVariantVal {
    /// Accepts any arguments, so that a subclass's `__init__` may take others; `__init__` sets
    /// the value and metadata.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyVariantVal {
            value: Vec::new(),
            metadata: Vec::new(),
        }
    }
    #[pyo3(signature = (value, metadata))]
    fn __init__(&mut self, value: Vec<u8>, metadata: Vec<u8>) {
        self.value = value;
        self.metadata = metadata;
    }
    /// Pickled as a plain class instance is: its class is rebuilt without `__init__`, and its
    /// value, metadata and instance attributes are restored by `__setstate__`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        let newobj = py.import("copyreg")?.getattr("__newobj__")?;
        let args = PyTuple::new(py, [slf.get_type().into_any()])?;
        let state = PyTuple::new(
            py,
            [
                PyBytes::new(py, &this.value).into_any(),
                PyBytes::new(py, &this.metadata).into_any(),
                slf.getattr("__dict__")?,
            ],
        )?;
        PyTuple::new(py, [newobj, args.into_any(), state.into_any()])
    }
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyTuple>) -> PyResult<()> {
        let (value, metadata, attributes): (Vec<u8>, Vec<u8>, Bound<'_, PyAny>) =
            state.extract()?;
        {
            let mut this = slf.borrow_mut();
            this.value = value;
            this.metadata = metadata;
        }
        slf.getattr("__dict__")?
            .call_method1("update", (attributes,))?;
        Ok(())
    }
    #[getter]
    fn value<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.value)
    }
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.metadata)
    }
    /// Decode to a Python value. Mirrors `VariantVal.toPython` — delegates to the shared
    /// `VariantUtils.to_python` (kept in Python since the full binary codec lives there).
    fn toPython<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("pyspark.sql.variant_utils")?
            .getattr("VariantUtils")?
            .getattr("to_python")?
            .call1((self.value(py), self.metadata(py)))
    }
    /// Encode to a JSON string. Mirrors `VariantVal.toJson`.
    #[pyo3(signature = (zone_id="UTC"))]
    fn toJson(&self, py: Python<'_>, zone_id: &str) -> PyResult<String> {
        py.import("pyspark.sql.variant_utils")?
            .getattr("VariantUtils")?
            .getattr("to_json")?
            .call1((self.value(py), self.metadata(py), zone_id))?
            .extract()
    }
    #[classmethod]
    fn parseJson(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'_>,
        json_str: &str,
    ) -> PyResult<PyVariantVal> {
        let t = py
            .import("pyspark.sql.variant_utils")?
            .getattr("VariantUtils")?
            .getattr("parse_json")?
            .call1((json_str,))?;
        let (value, metadata): (Vec<u8>, Vec<u8>) = t.extract()?;
        Ok(PyVariantVal { value, metadata })
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        format!("VariantVal({:?}, {:?})", self.value(py), self.metadata(py))
    }
    /// The JSON of the value, as PySpark's `VariantVal.__str__`: `VariantUtils.to_json` of it,
    /// in UTC, as `toJson()`.
    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        self.toJson(py, "UTC")
    }
}

/// A Geography value (WKB bytes + SRID). Mirrors `pyspark.sql.types.Geography`.
#[pyclass(name = "Geography", module = "pyspark.sql.types", weakref, dict)]
pub struct PyGeography {
    pub(crate) wkb: Vec<u8>,
    pub(crate) srid: i32,
}
#[pymethods]
impl PyGeography {
    #[new]
    fn new(wkb: Vec<u8>, srid: i32) -> Self {
        PyGeography { wkb, srid }
    }
    fn getSrid(&self) -> i32 {
        self.srid
    }
    fn getBytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.wkb)
    }
    /// The spatial reference identifier, as PySpark's attribute.
    #[getter]
    fn srid(&self) -> i32 {
        self.srid
    }
    /// The well-known binary, as PySpark's attribute.
    #[getter]
    fn wkb<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.wkb)
    }
    #[classmethod]
    fn fromWKB(_cls: &Bound<'_, pyo3::types::PyType>, wkb: Vec<u8>, srid: i32) -> PyGeography {
        PyGeography { wkb, srid }
    }
    fn __eq__(&self, other: &PyGeography) -> bool {
        self.wkb == other.wkb && self.srid == other.srid
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "Geography({:?}, {})",
            PyBytes::new(py, &self.wkb),
            self.srid
        )
    }
}

/// A Geometry value (WKB bytes + SRID). Mirrors `pyspark.sql.types.Geometry`.
#[pyclass(name = "Geometry", module = "pyspark.sql.types", weakref, dict)]
pub struct PyGeometry {
    pub(crate) wkb: Vec<u8>,
    pub(crate) srid: i32,
}
#[pymethods]
impl PyGeometry {
    #[new]
    fn new(wkb: Vec<u8>, srid: i32) -> Self {
        PyGeometry { wkb, srid }
    }
    fn getSrid(&self) -> i32 {
        self.srid
    }
    fn getBytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.wkb)
    }
    /// The spatial reference identifier, as PySpark's attribute.
    #[getter]
    fn srid(&self) -> i32 {
        self.srid
    }
    /// The well-known binary, as PySpark's attribute.
    #[getter]
    fn wkb<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.wkb)
    }
    #[classmethod]
    fn fromWKB(_cls: &Bound<'_, pyo3::types::PyType>, wkb: Vec<u8>, srid: i32) -> PyGeometry {
        PyGeometry { wkb, srid }
    }
    fn __eq__(&self, other: &PyGeometry) -> bool {
        self.wkb == other.wkb && self.srid == other.srid
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Geometry({:?}, {})", PyBytes::new(py, &self.wkb), self.srid)
    }
}

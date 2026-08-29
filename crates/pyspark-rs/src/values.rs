//! PyO3-backed value holders returned/consumed by the client for the newer types,
//! mirroring `pyspark.sql.types.{VariantVal, Geometry, Geography}`.

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A Variant value (the `value`/`metadata` binary components). Mirrors
/// `pyspark.sql.types.VariantVal`.
#[pyclass(name = "VariantVal")]
pub struct PyVariantVal {
    value: Vec<u8>,
    metadata: Vec<u8>,
}

impl PyVariantVal {
    /// Build from raw (value, metadata) bytes (used when materializing a VARIANT column).
    pub(crate) fn from_parts(value: Vec<u8>, metadata: Vec<u8>) -> Self {
        PyVariantVal { value, metadata }
    }
}

#[pymethods]
impl PyVariantVal {
    #[new]
    fn new(value: Vec<u8>, metadata: Vec<u8>) -> Self {
        PyVariantVal { value, metadata }
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
}

/// A Geography value (WKB bytes + SRID). Mirrors `pyspark.sql.types.Geography`.
#[pyclass(name = "Geography")]
pub struct PyGeography {
    wkb: Vec<u8>,
    srid: i32,
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
#[pyclass(name = "Geometry")]
pub struct PyGeometry {
    wkb: Vec<u8>,
    srid: i32,
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

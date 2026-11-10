//! PyO3-backed value holders returned/consumed by the client for the newer types,
//! mirroring `pyspark.sql.types.{VariantVal, Geometry, Geography}`.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyString, PyTuple};

/// Pickle `slf` as PySpark pickles its plain class: rebuilt by `copyreg.__newobj__`, without
/// `__init__`, with the state `object.__reduce_ex__` gives, the dict of its instance attributes
/// and `fields`, or that dict and the values of its slots, `(dict, slots)`, for a subclass of
/// slots. PySpark's class, which has no `__setstate__`, takes this state as its attributes, so
/// a value pickled here loads in PySpark, as in the server's Python workers.
fn reduce_as_a_plain_class<'py>(
    slf: &Bound<'py, PyAny>,
    fields: [(&str, Bound<'py, PyAny>); 2],
) -> PyResult<Bound<'py, PyTuple>> {
    let py = slf.py();
    let copyreg = py.import("copyreg")?;
    let attributes = slf.getattr("__dict__")?.cast::<PyDict>()?.copy()?;
    for (name, value) in fields {
        attributes.set_item(name, value)?;
    }
    let slots = PyDict::new(py);
    for name in copyreg
        .call_method1("_slotnames", (slf.get_type(),))?
        .try_iter()?
    {
        let name = name?.cast_into::<PyString>()?;
        // As `object.__getstate__`: the slots that have a value.
        if let Ok(value) = slf.getattr(&name) {
            slots.set_item(name, value)?;
        }
    }
    let state = if slots.is_empty() {
        attributes.into_any()
    } else {
        PyTuple::new(py, [attributes.into_any(), slots.into_any()])?.into_any()
    };
    let args = PyTuple::new(py, [slf.get_type().into_any()])?;
    PyTuple::new(py, [copyreg.getattr("__newobj__")?, args.into_any(), state])
}

/// Restore on `slf` a state of [`reduce_as_a_plain_class`], or of PySpark's pickle of its plain
/// class, as pickle's BUILD restores a plain class: its instance attributes, and its slots of a
/// `(dict, slots)`. The values of `fields`, which are not instance attributes here, are returned.
fn restore_plain_class_state<'py>(
    slf: &Bound<'py, PyAny>,
    state: &Bound<'py, PyAny>,
    fields: [&str; 2],
) -> PyResult<[Option<Bound<'py, PyAny>>; 2]> {
    let (attributes, slots) = match state.cast::<PyTuple>() {
        Ok(pair) if pair.len() == 2 => (pair.get_item(0)?, Some(pair.get_item(1)?)),
        _ => (state.clone(), None),
    };
    let mut values = [None, None];
    if !attributes.is_none() {
        let attributes = attributes.cast::<PyDict>()?.copy()?;
        for (value, name) in values.iter_mut().zip(fields) {
            if let Some(field) = attributes.get_item(name)? {
                attributes.del_item(name)?;
                *value = Some(field);
            }
        }
        slf.getattr("__dict__")?
            .call_method1("update", (attributes,))?;
    }
    if let Some(slots) = slots.filter(|slots| !slots.is_none()) {
        for (name, value) in slots.cast::<PyDict>()?.iter() {
            slf.setattr(name.cast_into::<PyString>()?, value)?;
        }
    }
    Ok(values)
}

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
    /// Pickled as PySpark's plain class is, with its value and metadata as attributes, so that
    /// PySpark loads it, as PySpark's pickles of it load here.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let fields = {
            let this = slf.borrow();
            [
                ("value", PyBytes::new(py, &this.value).into_any()),
                ("metadata", PyBytes::new(py, &this.metadata).into_any()),
            ]
        };
        reduce_as_a_plain_class(slf.as_any(), fields)
    }
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let [value, metadata] =
            restore_plain_class_state(slf.as_any(), state, ["value", "metadata"])?;
        let value: Option<Vec<u8>> = value.map(|v| v.extract()).transpose()?;
        let metadata: Option<Vec<u8>> = metadata.map(|m| m.extract()).transpose()?;
        let mut this = slf.borrow_mut();
        if let Some(value) = value {
            this.value = value;
        }
        if let Some(metadata) = metadata {
            this.metadata = metadata;
        }
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
#[pyclass(
    name = "Geography",
    module = "pyspark.sql.types",
    weakref,
    dict,
    subclass
)]
pub struct PyGeography {
    pub(crate) wkb: Vec<u8>,
    pub(crate) srid: i32,
}
#[pymethods]
impl PyGeography {
    /// Accepts any arguments, so that a subclass's `__init__` may take others, as of PySpark's
    /// plain class; `__init__` sets the WKB and the SRID.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyGeography {
            wkb: Vec::new(),
            srid: 0,
        }
    }
    #[pyo3(signature = (wkb, srid))]
    fn __init__(&mut self, wkb: Vec<u8>, srid: i32) {
        self.wkb = wkb;
        self.srid = srid;
    }
    /// Pickled as PySpark's plain class is, with its WKB and SRID as attributes, so that PySpark
    /// loads it, as PySpark's pickles of it load here.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let fields = {
            let this = slf.borrow();
            [
                ("wkb", PyBytes::new(py, &this.wkb).into_any()),
                ("srid", this.srid.into_pyobject(py)?.into_any()),
            ]
        };
        reduce_as_a_plain_class(slf.as_any(), fields)
    }
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let [wkb, srid] = restore_plain_class_state(slf.as_any(), state, ["wkb", "srid"])?;
        let wkb: Option<Vec<u8>> = wkb.map(|w| w.extract()).transpose()?;
        let srid: Option<i32> = srid.map(|s| s.extract()).transpose()?;
        let mut this = slf.borrow_mut();
        if let Some(wkb) = wkb {
            this.wkb = wkb;
        }
        if let Some(srid) = srid {
            this.srid = srid;
        }
        Ok(())
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
#[pyclass(
    name = "Geometry",
    module = "pyspark.sql.types",
    weakref,
    dict,
    subclass
)]
pub struct PyGeometry {
    pub(crate) wkb: Vec<u8>,
    pub(crate) srid: i32,
}
#[pymethods]
impl PyGeometry {
    /// Accepts any arguments, so that a subclass's `__init__` may take others, as of PySpark's
    /// plain class; `__init__` sets the WKB and the SRID.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyGeometry {
            wkb: Vec::new(),
            srid: 0,
        }
    }
    #[pyo3(signature = (wkb, srid))]
    fn __init__(&mut self, wkb: Vec<u8>, srid: i32) {
        self.wkb = wkb;
        self.srid = srid;
    }
    /// Pickled as PySpark's plain class is, with its WKB and SRID as attributes, so that PySpark
    /// loads it, as PySpark's pickles of it load here.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let fields = {
            let this = slf.borrow();
            [
                ("wkb", PyBytes::new(py, &this.wkb).into_any()),
                ("srid", this.srid.into_pyobject(py)?.into_any()),
            ]
        };
        reduce_as_a_plain_class(slf.as_any(), fields)
    }
    fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let [wkb, srid] = restore_plain_class_state(slf.as_any(), state, ["wkb", "srid"])?;
        let wkb: Option<Vec<u8>> = wkb.map(|w| w.extract()).transpose()?;
        let srid: Option<i32> = srid.map(|s| s.extract()).transpose()?;
        let mut this = slf.borrow_mut();
        if let Some(wkb) = wkb {
            this.wkb = wkb;
        }
        if let Some(srid) = srid {
            this.srid = srid;
        }
        Ok(())
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

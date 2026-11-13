//! `pyspark.storagelevel.StorageLevel` — Rust-backed flags for DataFrame/RDD storage.

use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyString, PyTuple};

/// The int32 field of the replication of the proto, which PySpark sets at the conversion.
const REPLICATION_FIELD: &str = "spark.connect.StorageLevel.replication";

/// Flags controlling storage of a persisted DataFrame (disk/memory/off-heap,
/// (de)serialized, replication). Mirrors `pyspark.storagelevel.StorageLevel`, a plain Python
/// class: its constructor keeps the five values as they are, unchecked, as plain attributes of
/// the instance, which may be set and deleted, a subclass may override `__init__`, and an
/// instance copies, and pickles of protocol 2 or more (those of `copy`, `pickle` and
/// cloudpickle), as an instance of that class does, by its attributes. As PySpark's, it is
/// unhashable, as it defines `__eq__`.
#[pyclass(
    name = "StorageLevel",
    module = "pyspark.storagelevel",
    dict,
    subclass,
    weakref
)]
pub struct PyStorageLevel {}

/// The `replication` argument of the constructor, where PySpark tells an omitted replication,
/// which is 1, apart from a given one, which it keeps as it is, None included.
pub enum Replication {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for Replication {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Replication::Given(obj.to_owned().unbind()))
    }
}

/// Set the attributes of a StorageLevel to `values`, in the order of the constructor, as
/// PySpark's `__init__` assigns them.
fn set_attributes<'py>(level: &Bound<'py, PyAny>, values: [Bound<'py, PyAny>; 5]) -> PyResult<()> {
    let names = [
        "useDisk",
        "useMemory",
        "useOffHeap",
        "deserialized",
        "replication",
    ];
    for (name, value) in names.into_iter().zip(values) {
        level.setattr(name, value)?;
    }
    Ok(())
}

/// A class constant of PySpark's StorageLevel, of these flags and replication.
fn constant(
    py: Python<'_>,
    use_disk: bool,
    use_memory: bool,
    use_off_heap: bool,
    deserialized: bool,
    replication: i32,
) -> PyResult<Py<PyStorageLevel>> {
    let level = Bound::new(py, PyStorageLevel {})?;
    let flag = |value: bool| PyBool::new(py, value).to_owned().into_any();
    set_attributes(
        level.as_any(),
        [
            flag(use_disk),
            flag(use_memory),
            flag(use_off_heap),
            flag(deserialized),
            replication.into_pyobject(py)?.into_any(),
        ],
    )?;
    Ok(level.unbind())
}

/// The value of a bool field of `pb2.StorageLevel(...)`, as protobuf's default (upb) backend
/// converts a keyword argument: None leaves the field unset, and any other value is converted
/// as [`crate::streaming::proto_bool`] does.
fn flag_field(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    if value.is_none() {
        Ok(false)
    } else {
        crate::streaming::proto_bool(value)
    }
}

/// The proto of a pyspark `StorageLevel`, as PySpark's `storage_level_to_proto` builds it, of
/// `DataFrame.persist`, `localCheckpoint` and `Catalog.cacheTable`: it asserts a StorageLevel,
/// a bare `AssertionError` of anything else, reads the five attributes, and passes them to
/// `pb2.StorageLevel(use_disk=..., use_memory=..., use_off_heap=..., deserialized=...,
/// replication=...)`, which converts them in that order. A replication of None is unset, and
/// another is the int32 that [`crate::stat_arguments::proto_int`] converts.
pub(crate) fn storage_level_proto(
    level: &Bound<'_, PyAny>,
) -> PyResult<spark_connect_proto::StorageLevel> {
    if !level.is_instance_of::<PyStorageLevel>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    let use_disk = level.getattr("useDisk")?;
    let use_memory = level.getattr("useMemory")?;
    let use_off_heap = level.getattr("useOffHeap")?;
    let deserialized = level.getattr("deserialized")?;
    let replication = level.getattr("replication")?;
    Ok(spark_connect_proto::StorageLevel {
        use_disk: flag_field(&use_disk)?,
        use_memory: flag_field(&use_memory)?,
        use_off_heap: flag_field(&use_off_heap)?,
        deserialized: flag_field(&deserialized)?,
        replication: if replication.is_none() {
            0
        } else {
            crate::stat_arguments::proto_int(&replication, REPLICATION_FIELD)?
        },
    })
}

/// The `storageLevel` argument of `DataFrame.persist`, where PySpark tells an omitted one, its
/// default `MEMORY_AND_DISK_DESER`, apart from a given None, for which its client sends no storage
/// level (`if kwargs.get("storage_level", None) is not None`), so that the server caches at its
/// `spark.sql.defaultCacheStorageLevel`.
pub(crate) enum PersistLevel {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for PersistLevel {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(PersistLevel::Given(obj.to_owned().unbind()))
    }
}

impl PersistLevel {
    /// The storage level that `persist` sends: `MEMORY_AND_DISK_DESER` when it is omitted, none
    /// for None, and else the proto of [`storage_level_proto`].
    pub(crate) fn proto(
        &self,
        py: Python<'_>,
    ) -> PyResult<Option<spark_connect_proto::StorageLevel>> {
        use spark_connect::storage::StorageLevelExt;
        match self {
            PersistLevel::Omitted => Ok(Some(
                spark_connect_proto::StorageLevel::memory_and_disk_deser(),
            )),
            PersistLevel::Given(level) if level.is_none(py) => Ok(None),
            PersistLevel::Given(level) => storage_level_proto(level.bind(py)).map(Some),
        }
    }
}

#[pymethods]
impl PyStorageLevel {
    /// As the `object.__new__` of PySpark's plain class: of any arguments, which `__init__`
    /// takes, so that a subclass of another `__init__` is built as it is in PySpark. The class
    /// shows the signature of `__init__`, as `inspect.signature` of PySpark's class does.
    #[new]
    #[pyo3(
        signature = (*_args, **_kwargs),
        text_signature = "(useDisk, useMemory, useOffHeap, deserialized, replication=1)"
    )]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        PyStorageLevel {}
    }

    /// As PySpark's `__init__`, which keeps the values as they are, of any type.
    #[pyo3(signature = (useDisk, useMemory, useOffHeap, deserialized, replication=Replication::Omitted))]
    #[allow(non_snake_case)]
    fn __init__<'py>(
        slf: &Bound<'py, Self>,
        useDisk: Bound<'py, PyAny>,
        useMemory: Bound<'py, PyAny>,
        useOffHeap: Bound<'py, PyAny>,
        deserialized: Bound<'py, PyAny>,
        replication: Replication,
    ) -> PyResult<()> {
        let py = slf.py();
        let replication = match replication {
            Replication::Omitted => 1i32.into_pyobject(py)?.into_any(),
            Replication::Given(replication) => replication.into_bound(py),
        };
        set_attributes(
            slf.as_any(),
            [useDisk, useMemory, useOffHeap, deserialized, replication],
        )
    }

    /// As PySpark's `"StorageLevel(%s, %s, %s, %s, %s)" % (...)` of the five attributes.
    fn __repr__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let values = PyTuple::new(
            py,
            [
                slf.getattr("useDisk")?,
                slf.getattr("useMemory")?,
                slf.getattr("useOffHeap")?,
                slf.getattr("deserialized")?,
                slf.getattr("replication")?,
            ],
        )?;
        PyString::new(py, "StorageLevel(%s, %s, %s, %s, %s)").rem(values)
    }

    /// As PySpark's: the names of the flags that are true, and `"%sx Replicated" %` the
    /// replication.
    fn __str__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let mut result = String::new();
        for (name, shown) in [
            ("useDisk", "Disk "),
            ("useMemory", "Memory "),
            ("useOffHeap", "OffHeap "),
        ] {
            if slf.getattr(name)?.is_truthy()? {
                result.push_str(shown);
            }
        }
        result.push_str(if slf.getattr("deserialized")?.is_truthy()? {
            "Deserialized "
        } else {
            "Serialized "
        });
        let replicated = PyString::new(py, "%sx Replicated").rem(slf.getattr("replication")?)?;
        PyString::new(py, &result).add(replicated)
    }

    /// As PySpark's: False of another object, else `self.useMemory == other.useMemory and
    /// self.useDisk == other.useDisk and ...`, whose `and` gives the first comparison that is
    /// not true, or else the last one, as they are.
    fn __eq__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !other.is_instance_of::<PyStorageLevel>() {
            return Ok(PyBool::new(slf.py(), false).to_owned().into_any());
        }
        let compare = |name: &str| {
            slf.getattr(name)?
                .rich_compare(other.getattr(name)?, CompareOp::Eq)
        };
        for name in ["useMemory", "useDisk", "useOffHeap", "deserialized"] {
            let equal = compare(name)?;
            if !equal.is_truthy()? {
                return Ok(equal);
            }
        }
        compare("replication")
    }

    /// The arguments of `copyreg.__newobj__`, none, so that an instance pickles and copies as
    /// one of PySpark's plain class does: the class, and its attributes as its state.
    fn __getnewargs__<'py>(&self, py: Python<'py>) -> Bound<'py, PyTuple> {
        PyTuple::empty(py)
    }

    // Preset levels (class attributes), matching pyspark.
    #[classattr]
    #[allow(non_snake_case)]
    fn NONE(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, false, false, false, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn DISK_ONLY(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, true, false, false, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn DISK_ONLY_2(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, true, false, false, false, 2)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn DISK_ONLY_3(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, true, false, false, false, 3)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_ONLY(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, false, true, false, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_ONLY_2(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, false, true, false, false, 2)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_AND_DISK(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, true, true, false, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_AND_DISK_2(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, true, true, false, false, 2)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn OFF_HEAP(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, true, true, true, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_AND_DISK_DESER(py: Python<'_>) -> PyResult<Py<PyStorageLevel>> {
        constant(py, true, true, false, true, 1)
    }
}

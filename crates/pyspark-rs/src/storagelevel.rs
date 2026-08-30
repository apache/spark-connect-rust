//! `pyspark.storagelevel.StorageLevel` — Rust-backed flags for DataFrame/RDD storage.

use pyo3::prelude::*;

/// Flags controlling storage of a persisted DataFrame (disk/memory/off-heap,
/// (de)serialized, replication). Mirrors `pyspark.storagelevel.StorageLevel`.
#[pyclass(
    name = "StorageLevel",
    module = "pyspark.storagelevel",
    get_all,
    from_py_object
)]
#[derive(Clone)]
pub struct PyStorageLevel {
    pub useDisk: bool,
    pub useMemory: bool,
    pub useOffHeap: bool,
    pub deserialized: bool,
    pub replication: i32,
}

impl PyStorageLevel {
    fn mk(
        use_disk: bool,
        use_memory: bool,
        use_off_heap: bool,
        deserialized: bool,
        replication: i32,
    ) -> Self {
        PyStorageLevel {
            useDisk: use_disk,
            useMemory: use_memory,
            useOffHeap: use_off_heap,
            deserialized,
            replication,
        }
    }
}

#[pymethods]
impl PyStorageLevel {
    #[new]
    #[pyo3(signature = (useDisk, useMemory, useOffHeap, deserialized, replication=1))]
    #[allow(non_snake_case)]
    fn new(
        useDisk: bool,
        useMemory: bool,
        useOffHeap: bool,
        deserialized: bool,
        replication: i32,
    ) -> Self {
        PyStorageLevel::mk(useDisk, useMemory, useOffHeap, deserialized, replication)
    }

    fn __repr__(&self) -> String {
        format!(
            "StorageLevel({}, {}, {}, {}, {})",
            py_bool(self.useDisk),
            py_bool(self.useMemory),
            py_bool(self.useOffHeap),
            py_bool(self.deserialized),
            self.replication
        )
    }

    fn __str__(&self) -> String {
        let mut r = String::new();
        if self.useDisk {
            r.push_str("Disk ");
        }
        if self.useMemory {
            r.push_str("Memory ");
        }
        if self.useOffHeap {
            r.push_str("OffHeap ");
        }
        r.push_str(if self.deserialized {
            "Deserialized "
        } else {
            "Serialized "
        });
        r.push_str(&format!("{}x Replicated", self.replication));
        r
    }

    fn __eq__(&self, other: &PyStorageLevel) -> bool {
        self.useDisk == other.useDisk
            && self.useMemory == other.useMemory
            && self.useOffHeap == other.useOffHeap
            && self.deserialized == other.deserialized
            && self.replication == other.replication
    }

    // Preset levels (class attributes), matching pyspark.
    #[classattr]
    #[allow(non_snake_case)]
    fn NONE() -> PyStorageLevel {
        PyStorageLevel::mk(false, false, false, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn DISK_ONLY() -> PyStorageLevel {
        PyStorageLevel::mk(true, false, false, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn DISK_ONLY_2() -> PyStorageLevel {
        PyStorageLevel::mk(true, false, false, false, 2)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn DISK_ONLY_3() -> PyStorageLevel {
        PyStorageLevel::mk(true, false, false, false, 3)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_ONLY() -> PyStorageLevel {
        PyStorageLevel::mk(false, true, false, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_ONLY_2() -> PyStorageLevel {
        PyStorageLevel::mk(false, true, false, false, 2)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_AND_DISK() -> PyStorageLevel {
        PyStorageLevel::mk(true, true, false, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_AND_DISK_2() -> PyStorageLevel {
        PyStorageLevel::mk(true, true, false, false, 2)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn OFF_HEAP() -> PyStorageLevel {
        PyStorageLevel::mk(true, true, true, false, 1)
    }
    #[classattr]
    #[allow(non_snake_case)]
    fn MEMORY_AND_DISK_DESER() -> PyStorageLevel {
        PyStorageLevel::mk(true, true, false, true, 1)
    }
}

fn py_bool(b: bool) -> &'static str {
    if b {
        "True"
    } else {
        "False"
    }
}

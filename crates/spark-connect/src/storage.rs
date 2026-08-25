//! Storage-level presets, mirroring `pyspark.StorageLevel`.
//!
//! `df.persist(...)` requires a valid storage level. The proto `StorageLevel`
//! default is all-false (equivalent to `NONE`), which the server rejects with
//! "StorageLevel is null or invalid" — so callers should use one of the named
//! presets below rather than `StorageLevel::default()`.

use spark_connect_proto::StorageLevel;

/// Named storage-level presets on [`StorageLevel`], matching the constants on
/// `pyspark.StorageLevel` (`MEMORY_AND_DISK`, `DISK_ONLY`, `OFF_HEAP`, ...).
///
/// Bring this trait into scope to call e.g. `StorageLevel::memory_and_disk()`.
pub trait StorageLevelExt {
    /// No storage (`NONE`). Not a valid level to `persist(...)` with.
    fn none() -> StorageLevel;
    /// Disk only, 1 replica (`DISK_ONLY`).
    fn disk_only() -> StorageLevel;
    /// Disk only, 2 replicas (`DISK_ONLY_2`).
    fn disk_only_2() -> StorageLevel;
    /// Disk only, 3 replicas (`DISK_ONLY_3`).
    fn disk_only_3() -> StorageLevel;
    /// Memory only, 1 replica (`MEMORY_ONLY`).
    fn memory_only() -> StorageLevel;
    /// Memory only, 2 replicas (`MEMORY_ONLY_2`).
    fn memory_only_2() -> StorageLevel;
    /// Memory and disk, 1 replica (`MEMORY_AND_DISK`).
    fn memory_and_disk() -> StorageLevel;
    /// Memory and disk, 2 replicas (`MEMORY_AND_DISK_2`).
    fn memory_and_disk_2() -> StorageLevel;
    /// Memory and disk, deserialized, 1 replica (`MEMORY_AND_DISK_DESER`) — the
    /// default level used by `cache()`.
    fn memory_and_disk_deser() -> StorageLevel;
    /// Off-heap (`OFF_HEAP`).
    fn off_heap() -> StorageLevel;
}

fn level(
    use_disk: bool,
    use_memory: bool,
    use_off_heap: bool,
    deserialized: bool,
    replication: i32,
) -> StorageLevel {
    StorageLevel {
        use_disk,
        use_memory,
        use_off_heap,
        deserialized,
        replication,
    }
}

impl StorageLevelExt for StorageLevel {
    fn none() -> StorageLevel {
        level(false, false, false, false, 1)
    }
    fn disk_only() -> StorageLevel {
        level(true, false, false, false, 1)
    }
    fn disk_only_2() -> StorageLevel {
        level(true, false, false, false, 2)
    }
    fn disk_only_3() -> StorageLevel {
        level(true, false, false, false, 3)
    }
    fn memory_only() -> StorageLevel {
        level(false, true, false, false, 1)
    }
    fn memory_only_2() -> StorageLevel {
        level(false, true, false, false, 2)
    }
    fn memory_and_disk() -> StorageLevel {
        level(true, true, false, false, 1)
    }
    fn memory_and_disk_2() -> StorageLevel {
        level(true, true, false, false, 2)
    }
    fn memory_and_disk_deser() -> StorageLevel {
        level(true, true, false, true, 1)
    }
    fn off_heap() -> StorageLevel {
        level(true, true, true, false, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_match_reference_values() {
        let s = StorageLevel::memory_and_disk_deser();
        assert!(s.use_disk && s.use_memory && s.deserialized && !s.use_off_heap);
        assert_eq!(s.replication, 1);

        let s = StorageLevel::memory_and_disk();
        assert!(s.use_disk && s.use_memory && !s.deserialized && !s.use_off_heap);
        assert_eq!(s.replication, 1);

        let s = StorageLevel::disk_only_3();
        assert!(s.use_disk && !s.use_memory);
        assert_eq!(s.replication, 3);

        let s = StorageLevel::off_heap();
        assert!(s.use_off_heap && s.use_disk && s.use_memory);

        // NONE is all-false (replication 1) — invalid to persist with.
        let s = StorageLevel::none();
        assert!(!s.use_disk && !s.use_memory && !s.use_off_heap && !s.deserialized);
    }

    #[test]
    fn every_preset_matches_its_flags_and_replication() {
        // Exercise the remaining presets so all StorageLevelExt methods are covered.
        let s = StorageLevel::disk_only();
        assert!(s.use_disk && !s.use_memory && !s.use_off_heap && !s.deserialized);
        assert_eq!(s.replication, 1);

        let s = StorageLevel::disk_only_2();
        assert!(s.use_disk && !s.use_memory);
        assert_eq!(s.replication, 2);

        let s = StorageLevel::memory_only();
        assert!(s.use_memory && !s.use_disk && !s.use_off_heap);
        assert_eq!(s.replication, 1);

        let s = StorageLevel::memory_only_2();
        assert!(s.use_memory && !s.use_disk);
        assert_eq!(s.replication, 2);

        let s = StorageLevel::memory_and_disk_2();
        assert!(s.use_disk && s.use_memory);
        assert_eq!(s.replication, 2);
    }
}

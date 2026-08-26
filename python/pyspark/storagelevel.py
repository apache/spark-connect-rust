"""StorageLevel for the Spark Connect client (mirrors pyspark.storagelevel)."""

__all__ = ["StorageLevel"]


class StorageLevel:
    """Flags for controlling the storage of an RDD/DataFrame.

    Mirrors ``pyspark.storagelevel.StorageLevel``: which storage tiers are used
    (disk, memory, off-heap), whether values are stored deserialized, and how
    many replicas to keep.
    """

    def __init__(self, useDisk, useMemory, useOffHeap, deserialized, replication=1):
        self.useDisk = useDisk
        self.useMemory = useMemory
        self.useOffHeap = useOffHeap
        self.deserialized = deserialized
        self.replication = replication

    def __repr__(self):
        return "StorageLevel(%s, %s, %s, %s, %s)" % (
            self.useDisk,
            self.useMemory,
            self.useOffHeap,
            self.deserialized,
            self.replication,
        )

    def __str__(self):
        result = ""
        result += "Disk " if self.useDisk else ""
        result += "Memory " if self.useMemory else ""
        result += "OffHeap " if self.useOffHeap else ""
        result += "Deserialized " if self.deserialized else "Serialized "
        result += "%sx Replicated" % self.replication
        return result

    def __eq__(self, other):
        return isinstance(other, StorageLevel) and (
            self.useDisk,
            self.useMemory,
            self.useOffHeap,
            self.deserialized,
            self.replication,
        ) == (
            other.useDisk,
            other.useMemory,
            other.useOffHeap,
            other.deserialized,
            other.replication,
        )


StorageLevel.NONE = StorageLevel(False, False, False, False, 1)
StorageLevel.DISK_ONLY = StorageLevel(True, False, False, False, 1)
StorageLevel.DISK_ONLY_2 = StorageLevel(True, False, False, False, 2)
StorageLevel.DISK_ONLY_3 = StorageLevel(True, False, False, False, 3)
StorageLevel.MEMORY_ONLY = StorageLevel(False, True, False, False, 1)
StorageLevel.MEMORY_ONLY_2 = StorageLevel(False, True, False, False, 2)
StorageLevel.MEMORY_AND_DISK = StorageLevel(True, True, False, False, 1)
StorageLevel.MEMORY_AND_DISK_2 = StorageLevel(True, True, False, False, 2)
StorageLevel.MEMORY_AND_DISK_DESER = StorageLevel(True, True, False, True, 1)
StorageLevel.OFF_HEAP = StorageLevel(True, True, True, False, 1)

//! `pyspark.sql.udtf.SkipRestOfInputTableException`. The data classes of the `analyze` of a
//! polymorphic UDTF (`AnalyzeArgument`, `PartitioningColumn`, `OrderingColumn`,
//! `SelectedColumn` and `AnalyzeResult`) are PySpark's dataclasses, of the shim
//! `pyspark.sql.udtf`: a handler that captures them pickles with them, and the Python worker
//! of the server unpickles them as PySpark's.

// The module argument sets `__module__`; use the reference path (upstream defines this
// in pyspark/sql/udtf.py) rather than the extension's own name.
pyo3::create_exception!(
    pyspark.sql.udtf,
    SkipRestOfInputTableException,
    pyo3::exceptions::PyException
);

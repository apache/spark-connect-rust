# Known behavior differences from PySpark Connect

Behaviors of the Rust-backed Python client that differ from PySpark Connect 4.3.0, kept
here so that a version upgrade does not rediscover them. Each entry says whether the
difference is intentional or tracked for a fix.

## `createDataFrame`

- **A pandas DataFrame with a StructType of another width** (intentional, SPARK-60902).
  PySpark's pandas conversion zips the DataFrame's columns with the schema's fields. With
  fewer fields than columns it silently drops the extra columns, and with more fields it
  raises PyArrow's `ArrowInvalid`. This client raises `AXIS_LENGTH_MISMATCH` for both
  rather than drop data.
- **NumPy ndarray input** (tracked by SPARK-60772). Not accepted yet.

## `DataFrame`

- **A subclass of `DataFrame` created with `object.__new__`** (intentional, a PyO3 limit).
  `DataFrame` is a class of the Rust extension, so `object.__new__(cls)` of a subclass raises
  `TypeError: object.__new__(cls) is not safe`. `MockDataset` of `pyspark.testing.mlutils`
  creates itself so, and two of Apache Spark's ML parity tests that use it
  (`test_parity_pipeline.py`'s `test_pipeline` and `test_identity_pipeline`) fail for that
  reason only. The rest of `pyspark/ml/tests/connect/test_parity_*.py` pass with this client,
  except for a test that reads `data/mllib` of the Spark source tree from the working directory
  of the server.

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
- **Type inference across rows** (tracked by SPARK-60908). PySpark infers each row's type
  and merges them: other atomic types merge into a string type, whose values they become;
  nested values null in every row raise `CANNOT_DETERMINE_TYPE`; other conflicts raise
  `CANNOT_MERGE_TYPE` or `CANNOT_INFER_TYPE_FOR_FIELD`; and a `Decimal` value infers
  `decimal(38,18)`. This client takes each top-level field's type from its first non-null
  value, and a nested type from the first non-null element or map value.
- **NumPy ndarray input** (tracked by SPARK-60772). Not accepted yet.

## `SparkSession.sql`

- **Statements run at the first action** (tracked by SPARK-60909). PySpark sends the
  statement when `sql()` is called, so a command such as `CREATE TEMP VIEW` or `SET` takes
  effect at once. This client runs it only when an action is taken on the returned
  DataFrame.

# Known behavior differences from PySpark Connect

Behaviors of the Rust-backed Python client that differ from PySpark Connect 4.3.0, kept
here so that a version upgrade does not rediscover them. Each entry says whether the
difference is intentional or tracked for a fix.

## `createDataFrame`

- **pandas DataFrames without PyArrow** (intentional optional-dependency fallback; PySpark
  Connect makes no session without PyArrow, whose `check_dependencies` raises
  `PACKAGE_NOT_INSTALLED`, so this path and the entries below that are of it extend the drop-in
  and do not differ from a setup that PySpark supports). As in
  PySpark, every pandas DataFrame, of any class (a subclass such as GeoPandas' `GeoDataFrame`)
  and of any arrays (a third-party extension array such as BigQuery's `db_dtypes`), converts
  with PySpark's conversion of each column to Arrow, in the session time zone, which PySpark
  requires PyArrow for. When PyArrow is unavailable, as `pyarrow_available` detects it, the
  native row path converts the frame instead, of its rows of Python values (the frame's
  `astype(object)`, `where` and `notna`). As PySpark, it takes a naive `datetime64` column of a
  timestamp field, or of no schema, in the session time zone, and as wall clocks of a
  `timestamp_ntz` field, whatever the zone of the host. Otherwise it types Python values as
  local rows are typed, where PySpark's conversion takes Arrow's types: a named Row is a struct,
  not an array, a dict a map, not a struct, nested Decimals take precision 38 (see below), an
  object column of naive datetimes is a timestamp, not a timestamp_ntz, nanoseconds are
  dropped as those of local rows are, to the microsecond below, without the error of a safe cast
  (PySpark's unsafe cast of a `datetime64[ns]` column rounds a time before 1970 toward 1970:
  `1969-12-31 23:59:59.999999999` is `1970-01-01 00:00:00` there, and `1969-12-31 23:59:59.999999`
  here and in local rows), and its errors are its own. It zips the columns with the fields, as
  PySpark's conversion does: the columns past the fields are dropped, and of a struct of no fields
  the rows of a frame of columns too. Of a frame of fewer columns than the fields of its struct,
  it raises `AXIS_LENGTH_MISMATCH`, where PySpark's conversion raises the error of the conversion
  of a column that it zips with a field, if any, and then PyArrow's `ArrowInvalid` of its rename.
  Of a frame of no columns whose index is not a RangeIndex, PySpark's table is of a column of each
  level of the index (`pa.RecordBatch.from_pandas`), which the rename takes: `pd.DataFrame(index=
  [0])` is `Row(a=0)` under `a long`, and raises `AXIS_LENGTH_MISMATCH` under `StructType([])`,
  where this path raises `AXIS_LENGTH_MISMATCH` under `a long` and makes a `Row()` under
  `StructType([])`. This fallback covers
  unavailable modules detected by `pyarrow_available`, not an installed PyArrow module that
  fails during import.
- **pandas DataFrames under customized or patched types, or rebound type names** (tracked for a
  fix: SPARK-62231, SPARK-62236, SPARK-62251). With PyArrow, a frame converts with PySpark's
  conversion, in its order, of the conversions and the types that `pyspark.sql.connect.session`
  binds when it is imported, with the first session, as PySpark Connect's module of the session
  binds them (`prepare_pandas_table`, `prepare_inferred_pandas_table`), whose conversion of each
  column imports `to_arrow_type` of the converter module of `sys.modules` when it is called, as
  PySpark's does: so a type name rebound in `pyspark.sql.types` or `pyspark.sql.pandas.types`
  changes the conversion as it changes PySpark's, of its errors
  (`[UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION] LongType() is not supported in conversion to
  Arrow.`, `ArrowNotImplementedError: Unsupported cast from timestamp[ns] to bool using function
  cast_boolean`), also of a name rebound after the session: a public `BooleanType` rebound to
  `LongType`, with the converter module removed, makes `[1, 0, 2]` under `a long` `[1, 0, 1]`. As
  PySpark's createDataFrame, of any data, the session checks the schema, of a DDL string as the
  server parses it, against the `AtomicType` and the `StructType` of that module, which it reads
  when it is called (`_check_schema`), before it reads any setting: an `AtomicType` rebound to None
  before the session raises `TypeError: isinstance() arg 2 must be a type, a tuple of types, or a
  union`, and a struct that the module's `StructType` does not take raises `[NOT_EXPECTED_TYPE]
  Argument `schema` should be list, None or StructType, got StructType.`. A struct whose classes,
  or a field's, are patched, or of a Python subclass that overrides how the conversion reads it
  (its `fields`, `names` or iteration, a field's `name`, `dataType`, `nullable` or `metadata`, or
  `__getattribute__` or `__class__`, which the fast paths decline), converts with PySpark's
  conversion too, which reads the patch or the override as PySpark's does, of its table, its
  errors and its columns. What differs:
  - Of a struct of a class that defines `names` or `fields` itself, whose `names` and `fields`, as
    Python reads them, are not as many, the frame takes the native row path (see "pandas
    DataFrames without PyArrow" above), of its rows and of a relation of its native fields
    (SPARK-62231): PySpark's `__init__` assigns them through the class's setters, and the native
    `__init__` does not, so a `names` or `fields` property of a setter reads none. Of a `fields`
    property, the struct's `json()` and the relation of a frame of no rows are of the fields that
    the native `__init__` was given too, also of one whose read raises, as the `AttributeError` of
    the attribute that its setter sets, as PySpark's are of the fields that its `__init__` assigned
    (see below). Where PySpark's conversion takes such a struct, the row path
    names the columns by the native names (`a` where PySpark's are the `A` of a setter that
    upper-cases them), raises `AXIS_LENGTH_MISMATCH` of a frame of fewer columns, where PySpark
    raises the `ArrowInvalid` of its rename, and `PySparkRuntimeError: Type mismatch in row data`
    of a column that does not convert, where PySpark raises the error of its conversion. Of
    `names` that are not as many in PySpark either, it converts the rows where PySpark raises
    `IndexError` (of fewer names) or the server's `ASSIGNMENT_ARITY_MISMATCH` (of more). A `fields`
    or `names` list changed in place, of a class that defines neither, takes PySpark's conversion.
  - A property of no setter patched over an attribute that PySpark's `__init__` assigns, a
    field's `name`, `dataType`, `nullable` or `metadata`, or a struct's `fields` or `names`,
    makes PySpark's `_deduplicate_field_names` raise `AttributeError: property of 'StructField'
    object has no setter` (of `'StructType'`, of a struct's), as the `__init__` of the field or the
    struct that it makes assigns through it. The native `__init__` does not (SPARK-62231), nor has
    a native type a `__dict__` that a setter could write (SPARK-62251), so the conversion goes on
    and reads the patch where PySpark's would, and so does the relation's schema, the struct's
    `json()` (see below): a `dataType` of `BooleanType()` converts a column of ints to bools of a
    boolean field (`struct<renamed:boolean>`), a `name` raises the `ValueError` of the cast of the
    table (`Target schema's field names are not matching the table's field names: ['renamed'],
    ['changed']`), a `nullable` of False raises `Casting field 'x' with null values to
    non-nullable` of a column of nulls, and makes the field of another column not nullable, a
    `metadata` is the Arrow metadata of the column and the metadata of the field, a struct's
    `names` names the column of the table, of a relation of the native names, and its `fields` of
    `[]` makes a table and a relation of no columns, which the server raises
    `ASSIGNMENT_ARITY_MISMATCH` of, of the names of `toDF`.
  - The relation's schema is the `json()` of the struct, as PySpark's, which reads the attributes
    of the types as Python reads them, the overrides and the patches too, as the conversion does:
    of a field of a subclass whose `dataType` reads `StringType()` over a column of strs under
    `a long`, the DataFrame is `struct<a:string>`, and of `StringType.isUTF8BinaryCollation`
    patched to return False, the schema holds the collation of the field (`"string collate
    UTF8_BINARY"`, and its `__COLLATIONS` metadata), as PySpark's, also of a frame of no rows and of
    a pyarrow Table, which is cast to the struct. What stays different is what the attributes of the
    drop-in's types read (until SPARK-62231 and SPARK-62251): a property of a field whose getter
    reads the state that its setter writes, which the native `__init__` does not call, reads it
    unwritten, so a `dataType` that returns the `self._dt` of its setter makes the conversion and
    `json()` raise `AttributeError: 'Field' object has no attribute '_dt'`, where PySpark's read
    the type that `__init__` gave it; and a class attribute over a field's `name`, `dataType`,
    `nullable` or `metadata`, which PySpark's `__init__` shadows with the attribute of the instance
    and the native one does not, is what they read: of `nullable = False` in the class, the field
    of the relation and of `json()` is not nullable, where PySpark's is.

  A Python subclass of `StructType` or `StructField` that overrides none of these takes PySpark's
  conversion, as in PySpark, with the JSON of its core types, where PySpark sends its `json()`,
  which names a struct of a subclass by its `typeName()` and which the server then fails to parse
  (see the types of `docs/compatibility.md`); so does one that overrides only a struct's `names`,
  `__getattribute__` or `__class__`, which `json()` does not read. So does a name that holds a
  NUL, at any depth, whose table the importer reads of its IPC stream, as the C data interface
  holds names as C strings.
- **Local data under types that override what PySpark's conversion reads** (tracked for a fix:
  SPARK-62258). PySpark's conversion of local data (`LocalDataToArrowConversion`), of tuples,
  lists, dicts, Rows and NumPy arrays, reads the fields of the struct and their attributes as
  Python reads them, of a Python subclass of a property or an iteration of its own and of a patched
  native class, and its relation sends the `json()` of the struct. The drop-in converts the rows by
  the native types, of a relation of those types, as its rows are: under a field whose `dataType`
  reads `StringType()` over `a long`, `createDataFrame([(1,)], schema)` is `struct<a:bigint>` of
  `Row(a=1)`, where PySpark's is `struct<a:string>` of `Row(a='1')`, and `createDataFrame([("x",)],
  schema)` raises `ArrowInvalid: Could not convert 'x' with type str: tried to convert to int64`,
  where PySpark's is `Row(a='x')`. Data of no rows, a pandas DataFrame and a pyarrow Table are sent
  with the `json()` of the struct, as PySpark's (see above).
- **Type names that PySpark Connect's other modules and PySpark's types read** (intentional). The
  other modules of PySpark Connect that its session imports bind the type names of
  `pyspark.sql.types` too, and PySpark's types read them when they are called, so a rebound name
  changes PySpark's client and types, where the drop-in's, which are native, keep their behavior:
  - its client converts the results of executions by the names that it bound with the session
    (`pyspark.sql.connect.client.core`, `pyspark.sql.connect.conversion`): of a `StructType`
    rebound to `object` before the session, it takes every response for a schema
    (`_execute_and_fetch`), so `toArrow` and `collect` raise `AssertionError`, as `collect` does of
    a `TimestampType` rebound to `LongType`; and of `from_arrow_type` rebound to None in the
    converter module, `collect` raises `TypeError: 'NoneType' object is not callable`;
  - it parses the types of the server by those names (`pyspark.sql.connect.types`): of a
    `LongType` rebound to `StringType`, the schema of a DataFrame of longs is `struct<i:string>`,
    and of a `StructType` rebound to `object`, a DDL schema and the schema of every DataFrame
    raise `TypeError: object() takes no arguments`; its conversion of local rows
    (`pyspark.sql.conversion`) reads them too (`AttributeError: 'LongType' object has no
    attribute 'fieldNames'`);
  - its types read the names of `pyspark.sql.types` in their methods: `DayTimeIntervalType()`
    reads `DayTimeIntervalType.DAY`, so of a `DayTimeIntervalType` rebound to `StringType`, the
    conversion of a column of timedeltas raises `AttributeError: type object 'StringType' has no
    attribute 'DAY'`; and its functions of schemas, as `_infer_schema`, make the rebound classes,
    of a subclass of whose struct the server fails to parse the JSON;
  - of an `AtomicType` or a `StructType` rebound to a class that takes other objects than types,
    such as `object`, its createDataFrame takes them as its schema, where the drop-in's takes a
    list as names, None as no schema, and raises `NOT_EXPECTED_TYPE` of another object: of an
    `AtomicType` of `object`, the names `["a"]` raise `AssertionError: dataType ['a'] should be an
    instance of <class 'pyspark.sql.types.DataType'>`; and of a `StructType` rebound to a
    subclass, it makes the struct of the `value` of an atomic schema of local rows of that
    subclass, whose `json()` names it by its `typeName()`, which the server fails to parse (see the
    types of `docs/compatibility.md`), where the drop-in's struct is native.

  Matching these would make the native client and each use of a native type look up the names of
  `pyspark.sql.types` in Python on its hot paths, at a cost to every user, only to reproduce the
  failures that such a rebinding causes in PySpark's own client.
- **Nested pandas Decimal inference and failures** (intentional; related change:
  SPARK-60696; of the row path without PyArrow, which PySpark Connect does not have, see above).

  - **Inferred precision and scale.** On the native row path, without PyArrow, inferred nested
    Decimal fields keep precision 38 and the greatest observed scale; PySpark infers the
    smallest precision that covers the values, as this client does with PyArrow. Increasing
    scale on the native path can make a large value exceed precision 38 even when the first
    value's scale could encode every value by discarding only trailing zeros. For example,
    with `[Decimal("9" * 37 + ".0")]` followed by `[Decimal("1.0000000")]`, the first value
    would need 44 digits at scale 7, beyond precision 38; the previous native fallback
    accepted both at scale 1. PySpark Connect 4.3.0 constructs a DataFrame for this
    example, but requesting its schema or collecting results raises `ArithmeticException`
    (`DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION`, precision 44, maximum 38). This client's
    native fallback instead raises `PySparkRuntimeError` during construction, before
    sending a relation. The rejection stage and error class differ.

  - **Scale beyond 38.** A later value that increases the scale beyond 38 raises
    `ValueError` before upload. This can reject later zero or trailing-zero values that
    the previous native fallback accepted at the first value's scale. This scale handling
    applies to the row path without PyArrow described above.

  - **Explicit schema.** With an explicit schema, a nested Decimal that does not fit
    raises `PySparkRuntimeError` only on the native row path, without PyArrow. With it,
    frames use PySpark's Arrow conversion and raise its errors. Both paths reject an
    explicit-schema value that does not fit before a relation is sent.
- **Multiple invalid pandas columns with an explicit Decimal field** (intentional,
  SPARK-60696; of the row path without PyArrow, which PySpark Connect does not have, see above).
  On the native row path, without PyArrow, Decimal exactness is checked before
  the encoder converts other fields. If an earlier non-Decimal column and a later Decimal
  column are both invalid, this client reports the Decimal column first; PySpark reports the
  first invalid column in schema order. Both reject the input before a relation is sent.
- **Decimal context hooks during local-row conversion** (intentional, SPARK-60704).
  Quantum and context objects are shared by Decimal type within one conversion and built
  when its first non-null, non-NaN value is converted. PySpark constructs a context for
  each field's converter before converting values. A custom hook that changes
  `decimal.DefaultContext` or mutates its supplied context during a conversion can
  observe different timing or sharing. Changes between `createDataFrame` calls apply to
  the next conversion.
- **Decimal class rebinding** (intentional, SPARK-60704). Native Decimal conversion and
  inference for local rows and pandas input recognize the cached C `_decimal.Decimal`
  class, falling back to the cached public `decimal.Decimal` when the accelerator is
  unavailable; accelerator absence is cached too. If `_decimal` is importable at first
  resolution, its class takes precedence even when the public `decimal.Decimal` is the
  pure-Python class. Values of that pure-Python class are then not recognized as
  Decimals: local-row Decimal conversion raises `AssertionError` and inference raises
  `CANNOT_INFER_TYPE_FOR_FIELD`, while the native row path of pandas input, without PyArrow
  (which PySpark Connect does not have, see above),
  without a schema or with names only reads them as doubles through `__float__`, potentially
  losing precision. Pandas frames on the Arrow path convert them as PySpark does.
  Quantization also uses a quantum of the selected C class, even when the public alias is
  rebound. The context factory and rounding mode still come from the public `decimal`
  module at first use; mixing its pure-Python context with explicit C Decimal values is
  unsupported. The text shortcut assumes
  `_decimal.Decimal` is immutable, as on CPython; mutable implementations of that class
  on other interpreters are not covered. This can differ from PySpark when an
  application rebinds the public class alias during the process.

## `DataFrame`

- **A session time zone that the server takes and Python's zone databases do not, and a timestamp
  out of Python's range in the session time zone** (intentional; the values are those of any other
  zone and timestamp). The zones are such as `Z`, `PST`, `GMT+08:00` or `UTC+01:00`. PySpark's
  `collect`, `toLocalIterator` and `first` of a timestamp localize the values in the session time
  zone, looked up by its name in Python, so they raise pytz's `UnknownTimeZoneError`, or, without
  pytz, PyArrow's `ArrowInvalid` ("The zoneinfo module or pytz package must be installed."). This
  client's conversion of the rows reads no zone of the session: it returns the instant in the local
  time zone of the client, as both clients do of a zone that Python knows. `toArrow` gives the same
  table in both, and `toPandas` raises the same error in both, pytz's `UnknownTimeZoneError` or,
  without pytz, PyArrow's `ArrowInvalid`. So a timestamp whose wall clock in the session time zone
  is outside the years 1 to 9999 of Python's `datetime`, and in the local time zone is not, differs
  too: PySpark's localization in the session time zone raises Python's `OverflowError` ("date value
  out of range"), where this client returns the instant in the local time zone. For example, of
  `timestamp_seconds(-62135596800)`, 0001-01-01 00:00:00 UTC, with the session time zone
  `America/Los_Angeles` on a host in UTC, this client returns `datetime(1, 1, 1, 0, 0)`. Both raise
  the `OverflowError` of a wall clock out of the range in the local time zone.
- **A schema that PySpark does not convert is requested once** (intentional; the errors are the
  same). PySpark's `_schema` keeps the schema that it requested only once its conversion of it
  succeeded, so of a schema that it does not convert, such as of a user-defined type whose
  `__init__` takes arguments or of no Python class, of a nanosecond-precision timestamp, or of a
  spatial type of an SRID that it does not take, each read of the schema (`schema`, `columns`,
  `dtypes`, `printSchema` and the lookups that read the columns) requests it again, and raises.
  This client keeps the schema of the server once it requested it, as of any other schema, and
  raises the same error of its conversion at each read, of no other request.
- **The `__repr__` of a Column of a str that UTF-8 cannot encode** (intentional; the expressions
  of the core hold UTF-8). Such a str, of a name, a SQL string or a literal, fails the plans of the
  Column when they are built, as in PySpark, and its lone surrogates show escaped in the `__repr__`
  (`\ud800`), where PySpark's shows the str as it is (see `docs/compatibility.md`).
- **A `StructField` of a name or a metadata str that UTF-8 cannot encode** (intentional; the types
  of the core hold their names and metadata as UTF-8 and JSON). Its constructor raises the
  `UnicodeEncodeError` of the str, of a lone surrogate, where PySpark's builds the field, whose
  `json()`, `simpleString()`, `==` and pickle work. Of such a name, PySpark's uses of the field:
  - `createDataFrame` of local rows, or of a pandas DataFrame or a pyarrow Table of rows, and a
    UDF of such a return type, raise the `UnicodeEncodeError` at the call;
  - a cast fails with protobuf's `UnicodeEncodeError` at the first request, and so does `to`,
    whose cached schema needs no request, at the first request that builds its plan;
  - the schemas of `createDataFrame` and `emptyDataFrame` of no rows, of a NumPy array, of a
    reader's or a stream reader's schema, and of `from_json`, are the server's, which reads the
    JSON in which `json.dumps` escapes the surrogate, and names the field with a `?` in its
    place (`a?b`); collecting their rows fails on the server, which cannot write the name in the
    Arrow schema of the results (`Unpaired surrogate at index 1 of 3`, of the driver or of a
    task).

  Of such a metadata str, each of these DataFrames is made and has its schema, and collecting
  the rows of a batch one fails on the server so.
- **The methods of a `UserDefinedType` that its types call when they are made** (inherent; the
  native types hold the core type of each type that they hold). An `ArrayType`, a `MapType`, a
  `StructField` or a `StructType` of a `UserDefinedType`, also of `add`, lowers it into the core's
  type when it is made: of its `jsonValue()`, which calls its `scalaUDT()` and `module()` and, of a
  Python-only type, cloudpickles its class, and of its `sqlType()`, which is called twice. PySpark's
  makes them of the object alone and calls none of these, which its `json()` and `jsonValue()`
  call once. Their `==` and tree string lower the types again, and so do `json()` and
  `jsonValue()` before they call the type's own `jsonValue()`, so these methods run more often than
  in PySpark, and with them their side effects. An error of one is kept for the uses that read the
  JSON of the type, which raise it as PySpark's do (see `docs/compatibility.md`).
- **The removal of the relation of a dropped checkpoint, and the deletion of a dropped model**
  (intentional; a drop does not wait). As PySpark's `CachedRemoteRelation.__del__`, the drop of
  the last DataFrame of a checkpoint sends one ExecutePlan of the removal, which is not
  reattachable and goes to no hook, with the tags and the user context extensions of the thread
  that drops it. As PySpark's `RemoteModelRef`, whose `_delete_ml_cache` runs an
  `execute_command`, the drop of the last reference to a model of the native ML classes (and of
  the Rust API) sends the deletion of the model from the ML cache of the server as an execution of
  the thread that drops it: of its tags and its user context extensions, reattachable, and through
  the hooks of the executions. PySpark's finalizers wait for the server; this client sends both
  without waiting, as a drop may run on a thread of the transport's runtime, or in a finalizer
  that holds the GIL, which another thread may need to read the responses of the same
  connection. The relation or the model may still be on the server for a moment after the drop,
  and the deletion reports nothing to the progress handlers of the session, which PySpark's
  `execute_command` calls, of its done notification too, on the thread of the finalizer.
  This client deletes a model in the session that holds it (the session of its fit or load),
  where PySpark's `del_remote_cache` deletes it through the active session
  (`SparkSession.getActiveSession()`): a model of a session that is not the active one (of
  `SparkSession.builder.remote(...).create()`) is deleted from the ML cache of the active
  session, which does not hold it, and stays in its own session's until that session ends, and a
  model dropped on a thread of no active session is not deleted. A model of a `RemoteModelRef`
  that a program gives its constructor (`LinearSVCModel(RemoteModelRef(id))`) is not this client's
  to delete: its `__del__` releases a reference of the `RemoteModelRef` (`release_ref`, PySpark's
  own), which deletes the object at the last through the active session and waits for the server,
  as PySpark's does; a model whose transforms still hold its object leaves the release to the last
  of them, on the thread that drops it, and a drop on a thread of the transport's runtime, which has
  no active session (the active session of Spark Connect is of its thread), deletes nothing then.
- **An exception of the `&` of a join's list of conditions that is not of a class that the plan
  carries** (intentional; a plan of the core carries the errors of these classes, not any Python
  exception: it could hold a Python callback that raises it, as the plans of UDFs and of `to` hold
  functions of Python, which the error of an operand's `&` is not worth). Python's
  `AttributeError`, `TypeError`, `ValueError` and `OverflowError`, and PySpark's error classes,
  such as the `PySparkAttributeError` of a Row, fail the plan when it is built, as in PySpark.
  Another, such as a `KeyError` that the `__getattr__` of an operand raises, is raised by `join`
  or `_joinAsOf`, where PySpark raises it when it builds the plan.
- **`toArrow` and `toPandas` of a string that is not valid UTF-8** (intentional). PySpark's
  `toArrow` returns a table whose string column holds the invalid bytes, which pyarrow's
  full validation rejects, and its `toPandas` raises pyarrow's `ArrowException`. This
  client does not return invalid Arrow data: both raise the `UnicodeDecodeError` of the bytes
  of the first invalid string, as its `collect` does, which is PySpark's error of `collect` (the
  Rust API's `collect` returns `SparkErrorKind::UnicodeDecode`).
- **A plan of 747 to about 990 levels** (intentional; the stack guard of the client). A plan
  deeper than `MAX_PLAN_DEPTH`, 746 levels with the plans of its subqueries, raises the client's
  `RecursionError` ("a plan may be at most 746 levels") before any request. PySpark's
  serializer, with upb, sends such a plan, of up to about 990 levels (a few more or fewer by the
  kind of its relations and the call that sends it), and raises its own `RecursionError` beyond;
  the server fails the plans that PySpark sends with gRPC `UNKNOWN` ("Application error processing
  RPC"), as it fails any plan of about 510 levels or more. Both clients fail; only the error
  differs.
- **What a plan build does for a pickle** (intentional; DESIGN). A pickled DataFrame, Column or
  table argument carries the relation or the expression that the drop-in builds for it, where
  PySpark's keeps its plan objects and builds them at each request. Of the objects of the plan, it
  carries those of its UDFs, as PySpark's does, which the requests of the unpickled object pickle
  (see "UDFs"). So the requests of an unpickled DataFrame, Column or table argument do not warn of
  a window of no partitions (`WARN WindowExpression: No Partition Defined ...`), which PySpark's
  do at each build, and the `and` of the conditions of a join of a list of Columns keeps the
  origin of its call site at the pickle, where PySpark's has the call site of each request. The
  pickle itself does not warn, as PySpark's does not, but PySpark's unpickle of a DataFrame warns
  once, as it builds the plan to read the columns of the new DataFrame, where the drop-in's does
  not. The pickle of a `mapInPandas` or `mapInArrow` of a resource profile reads the `id` of the
  profile and pickles, where PySpark's raises the `TypeError` of the lock of the profile, which its
  plan keeps. (A copy of a DataFrame, in its session or another, is a copy of its plan, of copies
  of the functions of its UDFs and of its Observations, which its requests build, as PySpark's.)
  The `plan` of `DataFrame._plan` builds the relation, and warns, as PySpark's does, and a
  `DataFrame(df._plan, session)` is a DataFrame of the plan of `df`, which its requests build, as
  PySpark's is of the plan object of `df`. A `DataFrame(plan, session)` of another plan, such as a
  plan of `pyspark.ml` or `SubqueryAlias(df._plan, "s")`, is of the relation of
  `plan.to_proto(session.client)` that it builds when it is made, as the core keeps the relation
  that it built then: it warns then, once, where PySpark's warns at each request, which builds the
  plan, and of a plan whose build raises an `Exception`, each request raises a new exception of the
  one of that build, where each of PySpark's builds the plan again, and so runs its code again and
  prints again the traceback of a function of a UDF that does not pickle, as
  `CloudPickleSerializer.dumps` prints it, and raises an exception of its own. Another exception of
  the build, such as the `KeyboardInterrupt` of a Ctrl-C while it runs, is raised by
  `DataFrame(plan, session)` itself, where PySpark's DataFrame, which does not build the plan then,
  raises a `KeyboardInterrupt` or a `SystemExit` that its plan raises at its requests. A pickle of
  such a DataFrame carries its plan, as PySpark's does, which the unpickle builds again, of its
  error then; an unpickled Column of its subquery builds it at each request, of the session of the
  request, and a build of a plan over it of the client that the build is given, as
  `df._plan.plan(client)` of a DataFrame over it, as PySpark's Column does; its `to_plan(client)`
  raises the error of the build, where PySpark's gives the expression, as PySpark builds the plans
  of the subqueries of a Column with the relation of a plan (see `docs/compatibility.md`). The
  `UserWarning` is of `pyspark.sql.connect.expressions` and of its line 1126, as PySpark's, in the
  file of the drop-in's module, which ends before the line, so it shows no source line.

- **The size of a pickle of cloudpickle of a DataFrame, a Column or a table argument** (intentional;
  DESIGN). cloudpickle pickles by value the functions that unpickle such an object, those of
  `pyspark.sql.connect._column_reduce`, which `pyspark.sql.connect.column` registers, so that the
  object unpickles on a Python worker of the server, whose PySpark has no module of the drop-in, as
  of a `foreachBatch` function or a UDF that captures it: the pickle carries their code, where
  PySpark's carries the plan objects, of classes of its package that a pickle names. `dataframe_of`
  pickles of about 13.0 KB, with the parser of deep messages (`parse_deep`, 5.3 KB) and
  `raise_error`, `column_of` of about 16.2 KB and `table_arg_of` of about 12.7 KB, of Python 3.11,
  and the pickle of an object of UDFs carries the fill of their holes too (`fill_holes`, of about
  11.2 KB), which the other pickles do not. So `cloudpickle.dumps` of `spark.range(2)` is of 13.6
  KB, where PySpark's is of 1.2 KB, of a DataFrame of a UDF 24.3 KB (2.0 KB), of a Column 16.5 KB
  (0.7 KB), of a Column of a UDF 27.0 KB (1.0 KB) and of a table argument of a UDF 23.6 KB (1.4 KB).
  A plain pickle names the functions by reference, of the drop-in's module, so that of a Column or a
  table argument is of about the size of PySpark's, of 0.1 to 0.5 KB, and loads only where the
  drop-in is.

- **Public base and native MRO** (intentional; a PyO3 class cannot have a Python class as its
  base, so the public base can only be a virtual one). `pyspark.sql.DataFrame` is implemented as a
  Python base/factory, allowing subclasses such as
  `pyspark.testing.mlutils.MockDataset` to allocate their own Python state with
  `object.__new__`. The concrete class remains
  `pyspark.sql.connect.dataframe.DataFrame`, implemented by the Rust extension. It is
  registered as a virtual subclass of the public base: `isinstance` and `issubclass`
  recognize it, while its actual MRO stays native. Consequently, `super` and MRO
  introspection do not treat a native frame as inheriting from the public base. So it is of the
  Connect `Column`, `DataFrameNaFunctions`, `DataFrameStatFunctions`, `WindowSpec` and `TableArg`,
  native classes too, and of PySpark's classic classes of their names (`pyspark.sql.column`,
  `pyspark.sql.table_arg` of Apache Spark as they are, and PySpark's classes of
  `pyspark.sql.window` and `pyspark.sql.dataframe`), and of `TableValuedFunctionArgument`, whose
  metaclass, of `pyspark.sql._classic_bases`, registers each native class with its classic class
  and forwards a patch of them to it where PySpark's Connect objects inherit the patch (see
  "Public class mutation and dispatch" and `docs/compatibility.md`). Of a data descriptor without
  a setter patched over an attribute that PySpark's Connect `__init__` sets (`_expr`), PySpark's
  objects do not construct, where the drop-in's, which no `__init__` makes, read it; the `df` of
  `DataFrameNaFunctions` and `DataFrameStatFunctions`, which their `__init__` sets, is PySpark's.

- **Metaclass and subclass serialization.** The public base uses an `ABCMeta`
  subclass for virtual registration and attribute forwarding, while official
  PySpark uses `type`. It exposes `register()`, and
  multiple inheritance with an unrelated metaclass requires a combined metaclass.
  Ordinary Python subclasses use the standard `abc.ABCMeta` so by-value
  cloudpickling names an importable metaclass on official Python workers. Explicit
  user metaclass subclasses are preserved and retain their own serialization
  requirements. A local subclass and a closure capturing that subclass were
  cloudpickled by the native client and reconstructed and exercised in an
  official-only Python process without importing the native extension. This does
  not establish portability of instances, explicit user metaclasses or server UDFs. The classic
  `Column`, `DataFrameNaFunctions`, `DataFrameStatFunctions`, `WindowSpec`, `TableArg` and
  `TableValuedFunctionArgument` are of such an `ABCMeta` subclass too, of an `_abc_impl` and a
  `register()`, and their user subclasses of `abc.ABCMeta`, where PySpark's are of `type`.

- **Public class mutation and dispatch.** The native class stands for PySpark's Connect
  DataFrame, a subclass of the public base that defines every public name of the base itself
  but `__arrow_c_stream__`, so a patch of the public base reaches native frames where it reaches
  PySpark's:
  - a patch of a name that PySpark's Connect DataFrame defines, such as `DataFrame.count = f`,
    `DataFrame.show = f`, a property of `columns` or `DataFrame.__getattr__ = f`, does not reach
    native frames, whose `df.count()` stays the native one; an explicit `DataFrame.count(df)`
    calls `f`. The public base's own `DataFrame.count` calls, as PySpark's
    `dispatch_df_method`, the concrete class's method at the call, also a patch of it
    (`pyspark.sql.connect.dataframe.DataFrame.count = g`);
  - a patch of another name, such as a new helper, `__arrow_c_stream__` or a name of `object`
    (`__eq__`, `__reduce_ex__`), reaches native frames, unless the concrete class has a patch of
    its own of the name, which hides it, as an attribute of PySpark's Connect class does. Of
    `_plan`, `_session` and the other attributes that PySpark's frames set on themselves, only a
    data descriptor, such as a property, reaches them;
  - a chain of wrappers of the public attribute and of the concrete class's ends at the
    implementation, and not at a wrapper again: the base's own of `__arrow_c_stream__`, which the
    native class's original stands for, and of the other names the concrete class's;
  - deleting or undoing a patch of the base restores the native attribute that it replaced, and
    keeps a patch that the concrete class made since; deleting the base's own
    `__arrow_c_stream__` removes it from native frames, as PySpark's lose it.

  What differs: a patch that reaches native frames is in the `__dict__` of the native class,
  where PySpark's Connect class inherits it, so deleting a patch of the concrete class leaves
  native frames without the name, where PySpark's see the base's attribute again. Class
  layout, constructor, import/ABC metadata and the instance attribute protocols
  `__getattribute__`, `__setattr__` and `__delattr__` stay on the public base, where PySpark's
  Connect frames, whose class defines none of the three, see a patch of them. Those protocols
  follow ordinary Python inheritance on independent subclasses, and assignments on Python
  subclasses remain local to those subclasses.

- **Attribute resolution.** Explicit public `__getattr__` calls use the native column resolver,
  as PySpark's dispatch to the `__getattr__` of its Connect class, including existing attribute
  names, unsupported JVM names, and dunder/legacy column validation. A patch of the public
  `__getattr__` reaches independent Python subclasses only, as in PySpark, and importing the
  public base does not alter the native lookup entries. The native class has no `__getattr__`
  in its `__dict__` (intentional; PyO3 resolves a missing attribute in the lookup slot of the
  type): `pyspark.sql.connect.dataframe.DataFrame.__getattr__` raises `AttributeError`, where
  PySpark's is its function, and a patch of it on the concrete class runs only for the names
  whose resolution raises `AttributeError`, a dunder name that is not a column or a JVM
  attribute such as `_jdf`, where PySpark's runs for every missing attribute. The native-only
  private method `_get_column_by_attribute` bridges explicit public lookup to the Rust resolver;
  it is an intentional extra name in concrete-class introspection and completion.

- **Independent Python subclasses and construction.** Python subclasses override
  operations for their own data model. As in PySpark, the public properties, of empty bodies,
  read `None`, also of a native frame read through the base (`DataFrame.columns.fget(df)`),
  while native-dispatched methods,
  including `repr`, `dir`, and missing-attribute `__getattr__`, raise
  `NOT_IMPLEMENTED` on independent Python data. Consequently, `hasattr` for an
  unknown name also raises that exception, including on `MockDataset`.
  The official undecorated empty methods `checkpoint`, `localCheckpoint`,
  `lateralJoin`, `nearestByJoin`, `toPandas`, `asTable`, `scalar` and `exists` are PySpark's,
  of its parameters, which CPython binds with its own errors: called through the public base
  they return `None`, of a native frame too (`DataFrame.toPandas(df)`), as in PySpark. The
  inherited `__arrow_c_stream__` uses the
  generic `SparkArrowCStreamer` helper with the independent frame and requested
  schema, as on the official public base.
  Internal `_session` and `_plan` remain ordinary instance state on those subclasses.
  The public base has PySpark's annotations, of `is_cached` and of the private attributes of
  its frames, which make no attributes: `is_cached` is assignable instance state, as on the
  official public base. Concrete getters for these names are not installed as public
  properties.
  A public-base subclass that inherits the factory's `__new__` receives a plain
  concrete frame, and its own `__init__` is not called. Define `__new__` using
  `object.__new__` for independent Python data, or subclass
  `pyspark.sql.connect.dataframe.DataFrame` for native-backed subclass behavior.
  A direct native subclass must use the native constructor to initialize its Rust
  payload; `object.__new__` cannot safely allocate that payload.
  The factory takes the Connect constructor's `(plan, session)` (intentional; PySpark's makes a
  classic DataFrame of `(jdf, sql_ctx)`, which pyspark-client cannot make): so
  `pyspark.sql.DataFrame(plan, session)` and a subclass that inherits its `__new__` give a native
  frame, where PySpark's raise `ModuleNotFoundError: No module named 'pyspark.sql.classic'`, and
  of no arguments the `TypeError` names `plan` and `session`, where PySpark's names `jdf` and
  `sql_ctx`.

- **Validation status.** The differences of the public base that remain are those above: the
  native MRO, the metaclass, the patches that the native class holds, the native lookup of
  attributes and the factory. The methods of the base that dispatch to the native class have
  its signatures, of no annotations and of a positional-only `self`
  (`inspect.signature(DataFrame.select)` is `(self, /, *cols)`), and the docstrings of PySpark's
  classic DataFrame: those of the native class, or, where PySpark's classic and Connect DataFrames
  differ in them, of `python/pyspark/sql/_classic_docs_generated.py`.
  The unchanged official pipeline, tuning and one-vs-rest selection passed on both
  the official and native high-level clients with `PYTHONHASHSEED=0`: 16 passed and
  four upstream-disabled GC tests skipped on each client. This includes
  `test_pipeline` and `test_identity_pipeline`. An earlier run with independent
  default Python hash seeds failed native Random Forest cross-validation on a
  one-class training fold. Its cause is the default seed of a validator,
  `hash(type(self).__name__)` of PySpark's `HasSeed`, which Python's hash
  randomization makes differ per process: a run of another seed draws other folds. Of
  the same seeds, both clients split alike and fail alike (12 seeds of a CrossValidator
  and a TrainValidationSplit of a RandomForestClassifier gave the same metrics, and the
  same error of a one-class fold, on both).

## UDFs

As PySpark's `PythonUDF.to_plan` and `PythonUDTF.to_plan`, each request pickles the function of
each UDF, UDTF and map or group-map function of its plan when it builds the plan (see
`docs/compatibility.md`), also of a copy, in its session or another, and of an unpickled Column,
DataFrame or table argument, which hold the functions as PySpark's do: a plain pickle of one of a
lambda fails at the lambda, and one of a function that pickles by reference pickles, which the
requests of the unpickled object pickle as it is then. This case still differs:

- **Garbage collection** (tracked). Python's garbage collector sees the plans and the expressions
  that DataFrames, Columns and the objects that hold them share, as it sees PySpark's plan objects:
  each part of a plan or of an expression that holds a function of a UDF, a plan of the Python
  client, the plan of an error that it keeps or a Python Observation, directly or below it, has a
  Python object (`SharedPart`), to which each holder of a reference to the part holds a reference
  and reports it, so a reference cycle through such an object and any of its holders, one or
  several, is collected, as PySpark collects it (see `docs/compatibility.md`). These cases still
  differ:
  - The objects of the native classes of `pyspark.ml.connect` report the Python objects that they
    hold in Rust, as PySpark's objects hold them as attributes: a model its training summary
    (PySpark's `_summary`), its cached members (`trees`), its own defaults, the `RemoteModelRef` of
    its object (PySpark's `_java_obj`, of which a copy holds a reference of its own) and the
    `RemoteModelRef` that it was made of while no transform, writer or call holds its object; an
    estimator, a transformer, a Pipeline, a PipelineModel, a OneVsRest and its model, a validator
    and its model the Python values of their params and their attributes (stages, classifier,
    models, best model, metrics); a `fitMultiple` iterator its estimator, dataset and param maps.
    So a cycle through a model and its summary (`model.summary.model = model`) is collected, and the
    model's object is deleted from the server's ML cache, as PySpark's. The DataFrames that a model,
    its training summary or a `OneVsRestModel` keeps in Rust of its fit (the summary dataset of the
    training summary, which PySpark's summary keeps as its `_summary_dataset`, and the training
    dataset and the datasets of the classes of a `OneVsRestModel`) hold references to the parts of
    their plans that no holder reports, and a `SharedPart` reports its references only while every
    reference to its part is of a `SharedPart` or an object of the binding that reports it: so the
    parts of those plans report nothing, and the collector never frees an object that they reach. A
    cycle through such a dataset alone is collected once the model is freed; one through the dataset
    and the model, such as a function of a UDF of the training DataFrame that keeps the model
    (`job.model = model` of a UDF of `job.f`), is never collected: the model is never freed, nor its
    object in the server's ML cache, where PySpark collects the cycle.
  - While a call of the binding runs without the GIL, a collection frees no cycle through the parts
    that the call may clone or drop, which the call keeps alive anyway; it frees the others, as
    PySpark's collector does. The collector must get the same references of an object in each of
    its passes over the objects: one that it got in the first pass and not in the second would be of
    an object that it takes for unreachable while a live holder reaches it, whose weak references it
    would clear and whose finalizers it would run; and a call without the GIL may clone and drop
    parts meanwhile, which changes whether every reference to a part is seen, as a request does as
    it builds its plan. So, from before a call releases the GIL until after it takes it back, no
    `SharedPart` of a part that it may clone or drop reports a reference:
    - a call that builds a request of the plans of the objects that it is given, a DataFrame and
      what it holds, grouped data, a writer and the copy of its state, an Observation, a model, a
      dataset or the arguments of a call of an ML object, such as the actions, the analyses, the
      views, `cache` and `checkpoint`, the writes of `write`, `writeTo`, `mergeInto` and
      `writeStream`, `DataFrame.stat`, `spark.sql` of arguments, `observe`, the flows of a
      declarative pipeline, and the fits, transforms, evaluations and fetches of
      `pyspark.ml.connect`, may clone and drop only the parts of those plans: theirs are the busy
      ones;
    - any other call clones and drops no part, and makes none busy, such as the waits and the calls
      of streaming queries, of `spark.streams` and of the listener bus, the calls of the
      configuration, the catalog and the session, the events of a declarative pipeline, and the
      commands, plans and analyses that the Python client builds (`SparkSession.client`), whose
      placeholders of DataFrames the binding replaces with the relations of the DataFrames before
      the call, with the GIL held, as PySpark builds their relations before its requests;
    - the events that a listener of the Rust API of the session gets, whose code the binding does
      not know, may clone or drop any part, and no `SharedPart` reports a reference while one runs;
      the Python API registers no such listener.
    In the child of a fork, of the fork handler of the binding (`os.register_at_fork`), which
    `os.fork()` runs, as `multiprocessing` forks, the calls of the forking thread alone are counted,
    the only ones there, and the parts that the calls of the other threads of the parent made busy
    are not; the child of a fork that runs no fork handlers of Python, as one that a C extension
    makes itself, keeps the counts and the busy parts of all the threads of the parent, and collects
    none of the cycles that they keep from being collected. A call of a daemon thread that the
    interpreter ends at its exit, before the call takes the GIL back, may keep the collections of
    the exit from freeing them.
  - On a free-threaded build of CPython (`Py_GIL_DISABLED`), no part has a `SharedPart`, so no cycle
    through a function of a UDF, a plan of the Python client, an error that a plan keeps or an
    Observation that a plan holds is collected: the binding finds the `SharedPart` of a part again
    by the address of the part, which only the GIL keeps from racing with the free of the
    `SharedPart`, and the counts and the marks of the calls without the GIL need it too.
  - `gc.get_referents` of a DataFrame, a Column or another object that holds a plan gives the
    `SharedPart`s of its parts that are seen (whose every reference is of a holder that reports it,
    and that no call that runs without the GIL may clone or drop), and that of a writer the
    `WriterParts` of its state, which its shallow copies share and which gives those of the state,
    where PySpark's objects give their plan objects through their `__dict__`; the `SharedPart` of
    the pickler of a UDF gives the tuple of its function and its return type.
  - The attributes of the clients of a session, their `thread_local` and their lists, are kept by
    the session as long as it lives, which no Python object reports, so a reference cycle through
    them, such as an attribute of the client or of its `thread_local` that keeps a DataFrame of the
    session, is not collected, where PySpark's client, a Python object, is. The core client of the
    session holds them, which reads them at each request of any thread, and which every DataFrame,
    reader, catalog and query of the session holds, with the threads of the core, so that no
    Python object can answer for all of their references to them.

## Session and client (Connect)

- **A response of 1025 to 65535 levels, of `SetAllowOversizeProtos(True)`** (intentional; the stack
  guard of the client). Once upb's `SetAllowOversizeProtos(True)` is called, PySpark's client and
  the drop-in read the responses of up to 65535 levels of nested messages, where both stop at 100
  levels otherwise (see "The drop-in parses the protobufs ..." in `docs/compatibility.md`). A
  response whose messages of the fields of the protos nest more than 1024 levels deep raises the
  drop-in's `RecursionError` ("the messages of a response may be nested at most 1024 levels deep"),
  as its conversions, copies and drops of the messages that it keeps recurse once per level. PySpark
  reads such a response and converts its schemas and literals in Python, up to Python's recursion
  limit, and upb's parser ends its process of a schema or a literal of 100,000 levels. So the
  schemas of `DataFrame.schema` (of its copy), `collect()` and `toPandas()`, and the literals of the
  observed metrics, of a few hundred levels (about 650 of arrays, 370 of structs) to 1024 are read
  by the drop-in only, and those of more fail in both, where PySpark reads the schemas that
  `SparkSession.client`'s `schema` and `to_table_as_iterator` return up to about 1,000 nested types
  (2,000 levels of arrays, 3,000 of structs), and the messages that it keeps as they are, such as
  the relation of the `SqlCommandResult` of `spark.sql`, of 10,000 levels too, which the drop-in
  refuses past 1024 levels. Groups of fields that the types do not know, which a decoded response
  does not keep, nest up to 65535 levels in both. The drop-in reads a response at its bound on a
  thread of 256 KiB (`threading.stack_size`) too, and makes and operates on its types, as on the
  types of as many levels that the user makes, on stacks that grow with their levels (see
  `docs/compatibility.md`). On the main thread of Python 3.11, where PySpark raises its
  `RecursionError` from about 100 nested structs (`copy.deepcopy`), 125 (pickling), 145 (`==`), 165
  (`hash()` and `repr()`), 250 (`json()` and `simpleString()`) and 330 (`treeString()`), the drop-in
  raises its own past Python's recursion limit of the levels of its walks of the type objects, a
  level for each struct and each field (SPARK-62224, see below): from about 500 nested structs of
  `==`, pickling, `treeString()`, `hash()` and `repr()`, and 1,000 of `simpleString()` and
  `toNullable()`; of `json()` (of Python's `json` module) from about 330, and of `copy.deepcopy`
  and unpickling, which convert the JSON of the type in Python, from about 250. Of nested arrays,
  PySpark's from about 250 (`copy.deepcopy`) to 1,000 (`json()`, `simpleString()`), and the
  drop-in's from about 500 (`copy.deepcopy`, unpickling) to 1,000. A higher limit
  (`sys.setrecursionlimit`) takes more levels, in both.

- **The message of the `RecursionError` of a type that holds itself** (inherent of `json()`,
  `jsonValue()`, `repr()`, `hash()` and `treeString()`, whose text PySpark varies with the depth of
  its caller). A struct whose fields hold a field of it (an append to its `fields`, or `add`), an
  array or a map that is its own element or value, or a field of the type of the struct that holds
  it, has levels that no walk ends. As PySpark's methods, which walk them in Python, each use of the
  drop-in that reads them raises `RecursionError` (SPARK-62224), and the uses that do not read them
  work as PySpark's do (see the types of `docs/compatibility.md`). Of `json()`, `jsonValue()`,
  `repr()`, `hash()` and `treeString()`, PySpark's text is that of the operation of its Python code
  that crosses the limit, which depends on the depth of the stack of its caller: none, or `
  while calling a Python object`, or ` in comparison`. CPython's check (`_Py_CheckRecursiveCall`)
  adds the `where` of the C call that enters the frame past the limit, of which the native walks
  enter none, so the drop-in's text is of the bound of its walk, `maximum recursion depth
  exceeded`, or of Python's own check when a call of Python code in the walk crosses the limit
  first: ` while calling a Python object`, ` while getting the str of an object`, ` while getting
  the repr of an object`, ` in __instancecheck__` (of the type checks of the UDFs and the pandas
  conversions), and ` while decoding a JSON object from a unicode string` (of Python's `json`
  module, of the JSON of a pandas conversion). Of the uses of a fixed text in PySpark, the drop-in
  gives it: none of `simpleString()` of a struct, an array, a map or a field, and ` while calling a
  Python object` of `json()` of an array. The levels at which a type of many levels, which does not
  hold itself, raises it differ too (see above).

- **The groups of a response, of protobuf's pure-Python implementation** (intentional; one count of
  the levels). With `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, the drop-in reads the
  responses within the limit of the decoder of protobuf (`decoder.SetRecursionLimit`, 100 of none),
  as PySpark does, and counts the groups of the fields that a type does not know as upb counts
  them, a level each below the message that holds them. Protobuf's Python decoder counts them from
  0 in each message, and stops at the limit itself: so of a response of groups exactly 100 levels
  deep, PySpark raises "Exception deserializing response!" and the drop-in reads it, and of a
  message at level 92 to 100 with 10 levels of groups below it, PySpark reads it and the drop-in
  raises that error. A server of Spark, of proto3, sends no groups.

- **An Arrow batch of 61 to 63 nested structs** (tracked; the limit of arrow-rs). The client reads
  the Arrow batches of a result with arrow-rs, whose IPC stream reader verifies the flatbuffer of
  each message within the default depth of flatbuffers' verifier, 64, and takes no other. So of a
  result of a struct nested 61 to 63 levels, such as of `named_struct('a', ...)` nested so, PySpark
  (pyarrow 26.0.1) reads the batches of `collect()`, `toPandas()` and `toArrow()`, where the drop-in
  raises `PySparkRuntimeError` ("Failed to create Arrow stream reader: Parser error: Unable to get
  root as message: DepthLimitReached"). From 64 levels both fail, PySpark with pyarrow's
  `ArrowInvalid` ("Max recursion depth reached"), and up to 60 both read. Its protobuf schema nests
  past the 100 levels of a response from 32 nested structs, so a server's result reaches the
  difference only after `SetAllowOversizeProtos(True)`.

- **The details of a failure, of protobuf's pure-Python implementation** (intentional; one parser
  of them). With `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, the drop-in reads the
  `grpc-status-details-bin` of a failure, its `google.rpc.Status` and the first `ErrorInfo` or
  `RetryInfo` of its details, as upb's parser reads them, within the limit of the decoder's
  `SetRecursionLimit` (100 of none), counted as upb counts it, where PySpark parses them with
  protobuf's Python decoder, of its own rules and errors. So PySpark raises the `DecodeError` of a
  message of groups nested as deep as the limit (`Error parsing message: too many levels of
  nesting.`), which the drop-in reads, raising upb's of one level more; of other details that do
  not decode, its own texts (`unpack requires a buffer of 8 bytes` of `abc`, `Truncated string.`),
  where the drop-in's are upb's (`Error parsing message with type 'google.rpc.Status': Wire format
  was corrupt`); of a string that is not UTF-8, a `UnicodeDecodeError` (`... in field:
  google.rpc.Status.message`), where the drop-in raises upb's `DecodeError` (`String field had bad
  UTF-8`); and it reads a tag or a length of more than five bytes, of which the drop-in raises upb's
  `DecodeError`.

- **The check of the dependencies at `import pyspark.sql.connect` and `import pyspark.ml.connect`**
  (intentional; the transport is native). PySpark's packages run `check_dependencies()` when they
  are imported. That raises `PACKAGE_NOT_INSTALLED` without pandas 2.2.0, PyArrow 18.0.0, grpcio
  1.48.1, grpcio-status 1.48.1, googleapis-common-protos 1.56.4 or zstandard 0.25.0, and
  `UNSUPPORTED_PACKAGE_VERSION` of an older one. Of a main module without a file, as of `python -c`,
  it takes the import for a run of the doctests and imports `pyspark.testing.connectutils`, which
  exits the process: with 255 when no `SPARK_HOME` is found, or with 0 ("Skipping doctests") without
  the dependencies of the Connect tests. The drop-in's packages import `check_dependencies`, as
  PySpark's, so they have the same names (`check_dependencies`, and `utils` of
  `pyspark.sql.connect`), but they do not run it. Its transport needs none of the gRPC packages, and
  it runs without PyArrow and pandas (see "pandas DataFrames without PyArrow" above), so the import
  of `pyspark.sql.connect` works without them, also of `python -c`, and that of `pyspark.ml.connect`
  without the gRPC packages: `pyspark.ml.connect` imports its module `feature`, PySpark's as it is,
  which imports pandas, PyArrow and NumPy, so without one of them the import raises its
  `ModuleNotFoundError`, where PySpark's raises `PACKAGE_NOT_INSTALLED` of pandas and PyArrow. A
  program's call of `check_dependencies()` checks pandas and PyArrow, and the doctest run, as
  PySpark's, but not the gRPC packages.
- **Custom gRPC channels** (intentional; the transport is native). `ChannelBuilder` and
  `DefaultChannelBuilder` parse the connection string and hold its parameters, `host`, `endpoint`,
  `get`, `set`, `userId`, `token`, `session_id`, `use_ssl`, `secure`, `userAgent`, the keepalive
  properties and `metadata()`, and the drop-in's `DefaultChannelBuilder` has a `port` too, which
  PySpark's has not (it keeps the port in `_port`). When a session is made of a channel builder, the
  transport reads it as PySpark's client reads it as it is made, also when a subclass overrides what
  it reads: its `session_id` and its `userId`, and each again when it is set; of a
  `DefaultChannelBuilder`, what its `toChannel` reads, in its order, its `secure`, `use_ssl`,
  `_host` and `token`, of which the credentials of the channel are, its `endpoint`, and its
  keepalive properties, when it is enabled, of which the keepalive of the channel is; of a builder
  of its own, its `token`, `host` and `port`; and its `metadata()`. Of a builder that the program
  gave the session, or of a `metadata()` of its own, of a subclass or of the builder, it calls it
  again for each call and each of its retries, as PySpark's client does, also for the `ExecutePlan`
  of a finalizer, whose first attempt reads it on the thread of the finalizer, whose requests carry
  the metadata that it gives as grpcio encodes and checks it, of every value of a key, in its order,
  and then the `authorization` of the token, of its errors, which fail the call, but of the keys
  that gRPC's core sets itself (`te`, `content-type`, `content-length`, `grpc-timeout`,
  `grpc-status`, `grpc-message`), which are not sent; the requests of the artifacts carry what it
  gave as the session was made, as PySpark's ArtifactManager, and those of a reattachable execution
  what it gave as the execution started, as PySpark's iterator; a request that a client sends as the
  interpreter exits, whose `metadata()` cannot be read then, is not sent. What stays different: the
  `userAgent` of the builder is not read as each request is built, where PySpark's client reads it
  then, but of the `user_agent` parameter as the session is made (see `SPARK_CONNECT_USER_AGENT`
  below); a session of a connection string takes the metadata of its calls of the parameters of the
  connection string, as the session is made, and does not call the `metadata()` of the builder that
  its client makes of it, where PySpark's client calls it for each call, which a change of
  `client._builder` or of `ChannelBuilder.metadata` alone shows, and a key of them that gRPC does
  not take fails the creation of the session of `ValueError` (`metadata was invalid: 'A' is not a
  gRPC metadata key, ...`), where grpcio fails each call of `INTERNAL` (`Invalid metadata`); a retry
  of the `ExecutePlan` of a finalizer reads `metadata()` on a thread of the transport (`Dummy-1`),
  where PySpark's finalizer retries on its own thread; the metadata of different keys is sent of the
  order of their first values (`x-seq`, `authorization` twice, `x-dup` twice), as the headers of
  HTTP of hyper hold it, where grpcio sends it of its order (`x-seq`, `authorization`, `x-dup`
  twice, `authorization`), which HTTP does not tell apart; a `grpc-status` of `metadata()` is not
  sent, where grpcio's call of it waits forever; and an exception of `metadata()` fails the call
  without the `can_retry` of a retry policy of its own, which PySpark's `Retrying` asks of it. It
  connects as the channel of PySpark's `toChannel` of a `DefaultChannelBuilder` does: to its
  `endpoint`, of `_host` and `_port` (of `DefaultChannelBuilder.default_port()` without one), a
  subclass's too, and not to its `host`, which the client shows, nor to a `port` of its own; of no
  security, and no token, unless it is `secure`, else of gRPC's local credentials and the token when
  it is not `use_ssl` and its `_host` is `localhost`, over a connection whose local address is of
  the loopback only, as their handshake takes it, which a Unix socket is not, else of TLS and the
  token. The endpoint is read as gRPC reads a target (see `docs/compatibility.md`): of its bytes up
  to a NUL, of gRPC's names of it, its default authority, of the `:authority` of the calls and of
  the name that TLS verifies, and the server of the tunnel of a proxy and the host of its
  exclusions, and the name of `dns://` and an authority of an IP address and a port is asked of that
  DNS server, as gRPC's resolver asks it; the host of the connection string stands for the target
  only, and decides neither its TLS, nor its token, nor its proxy, nor its calls. What stays
  different: the channels of a target that share the connection of the process resolve its name
  again once between them as that connection fails, where each channel of grpcio resolves it again
  of its own; a new channel takes the connection of the process whatever addresses its own
  resolution gives, where grpcio's connects to those addresses (both until SPARK-62257); the search
  domains and the options of a lookup through a DNS server are those of `/etc/resolv.conf`,
  `LOCALDOMAIN` and `RES_OPTIONS` on any system, where c-ares reads them on macOS of the system's
  configuration of DNS, of each of its resolvers of no domain (their search lists joined), and not
  of `/etc/resolv.conf` (`ares_sysconfig_mac.c` of c-ares 1.34.5), and on Windows of the registry
  (the `SearchList`, `Domain`, `DhcpDomain` and `PrimaryDnsSuffix` of TCP/IP, of its policies and of
  each interface; `ares_sysconfig_win.c`), which the transport does not read (until SPARK-62254);
  the `:authority` of the calls of a target whose authority HTTP's URIs cannot hold, of a `%` but in
  brackets or of more than one `:`, as of a Unix socket (`tmp%2Fa.sock`) or of a list of addresses,
  is the host and the port of the connection string (`localhost:443`, the first address), as hyper
  and h2, the HTTP/2 of the transport, take the authority of a URI only, where gRPC sends its own,
  so the server of a long list does not refuse the headers of its calls, as grpc-java refuses those
  of grpcio past their limit; the TLS sends the name that it verifies as SNI only of a name of DNS,
  up to a `%` (`localhost` of `localhost%00`), as rustls takes no other, where BoringSSL sends the
  host of gRPC's authority as it is (`x*y.localhost`, `a%2Fb.localhost`); a target of `xds:` of a
  bootstrap of xDS in the environment fails each call of `UNAVAILABLE`, as the transport has no
  client of xDS, where grpcio's would connect through it; and gRPC's core writes to stderr the
  errors of a target that it refuses (`Error parsing URI(s)`, `channel stack builder failed`, `no
  port given for ipv4 scheme`), which the transport does not write.

  The members of a gRPC channel are not there: `toChannel`, which the transport calls of a builder
  of its own for its errors only, so a subclass whose `toChannel` connects elsewhere is of its
  endpoint, and a builder of its own of its `host` and `port` (see "A channel builder without a
  host"), where PySpark's connects through the channel, `add_interceptor` and `setChannelOption`,
  and a `channelOptions` argument raises `NOT_IMPLEMENTED` instead of being ignored. The transport
  sends its requests over the HTTP/2 connections of tonic, from Rust and from the threads of its
  runtime, and not over a channel of grpcio: it has no `grpc.Channel` to give, cannot run an
  interceptor of grpcio, whose `continuation` calls a grpcio channel and returns grpcio's `Call`,
  and has none of the channel arguments of gRPC's core that the options set. It takes the keepalive
  of PySpark's channel options of the keepalive properties of a `DefaultChannelBuilder`, as its
  `_effective_channel_options` does, and of the parameters of a builder of its own.
  `SPARK_CONNECT_USER_AGENT` is read as each call starts, where PySpark reads it as each request is
  built, so a request that the transport builds during a call, such as a reattachment or a release
  in the background, has the user agent of the last call that started.
- **The modules and names of `pyspark.sql.connect.client`** (intentional; the transport is
  native). The package is PySpark's, and its `core` defines PySpark's classes and functions, of
  PySpark's `__all__` (see `docs/compatibility.md`). PySpark's `core` imports the modules
  `artifact`, `reattach` and `retries` of the package, so the package has those names as soon as it
  is imported. The drop-in has no `artifact` and `reattach`, which the transport replaces (see
  "Retry and reattach"), and the package has `retries`, which imports grpc, once it is imported, as
  by `set_retry_policies`: before, `dir()` and `from pyspark.sql.connect.client import *` do not
  give it, and `pyspark.sql.connect.client.retries` raises `AttributeError`, where
  `import pyspark.sql.connect.client.retries` and `from pyspark.sql.connect.client import retries`
  import it. Of the names that PySpark's `core` imports, the drop-in's has those that it imports
  too, such as `os` and `logger`, and those of `retries` (`RetryPolicy`, `Retrying`,
  `DefaultPolicy`, `DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME`) as they are read, which its `dir()`
  does not list; the others, such as `grpc`, `pd`, `pa`, `ArtifactManager` and `List`, it has not.
  It has the names that its helpers import, `contextvars`, `functools`, `inspect`, `math`,
  `warnings` and `Path`, which PySpark's has not. The package is imported at the first read of a
  session's `client` of the process, with grpcio, which its `pyspark.errors.exceptions.connect`
  imports, where PySpark's `pyspark.sql.connect.session` imports it: that first read takes about
  0.1 to 0.2 s, as of the first `spark.client.close()` of a program, where `close()` itself returns
  at once when no release of an execution is in flight.
- **Retry and reattach** (intentional; DESIGN). Retries and reattachment live in the Rust
  transport. `client.retries` is PySpark's, vendored. `SparkConnectClient.set_retry_policies`
  makes the transport retry with the parameters of PySpark's own `DefaultPolicy` and `RetryPolicy`,
  without Python, read when they are set; the policies of a subclass, or of a backoff, a jitter or
  a delay that is not a whole number of milliseconds, are run by PySpark's `Retrying`, as PySpark's
  client runs them, of grpcio's `RpcError` of each failed RPC (see `docs/compatibility.md`). What
  stays different: PySpark's `Retrying` offers its policies every exception of an attempt, such as
  a `KeyboardInterrupt` or an error of its Python code that reads the responses, where the
  transport, which reads them in Rust, offers them the failures of the RPCs; and the states of the
  policies (`to_state`), which a `Retrying` makes of each call and of each response of an
  execution, as PySpark's makes them, are as many as the requests of the drop-in, which sends more
  releases than PySpark's. `RpcDeadlines` sets the deadlines of the transport, `get_status` that of
  `_get_operation_statuses`. `client.reattach` (`ExecutePlanResponseReattachableIterator`) is not
  provided: it reads the responses of a stub of the generated `base_pb2_grpc`, over a channel of
  grpcio, which the drop-in has neither of. `disable_reattachable_execute` and
  `enable_reattachable_execute` set the executions of the transport not reattachable or
  reattachable.
- **Errors that PySpark's client leaves to grpc, without grpcio** (intentional; the transport is
  native). PySpark's `SparkConnectClient.clone`, its `ArtifactManager`'s copy to the file system
  (`copy_from_local_to_fs`, and so `SparkSession.cloneSession` and `copyFromLocalToFs`) and its
  caching of artifacts raise the failure of their RPC as grpc's `RpcError` (`_InactiveRpcError`),
  unconverted and unlogged, and so does the drop-in where grpcio is installed. Without grpcio, whose
  classes the error is of, the drop-in converts the failure as those of its other RPCs:
  `clone("not-a-uuid")` raises the `UnknownException` of
  `INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_FORMAT`, and a copy to a local path that the
  server refuses the `UnsupportedOperationException` of `_LEGACY_ERROR_TEMP_3161`. The clone asks
  the server for nothing more of its failure, as PySpark's, neither the details of the error nor
  the configs of its stack trace; the copy and the caching ask for them, and with a user agent that
  is too long, they raise the `SparkConnectException` of the request of the error's details.
- **The warnings of the requests sent in the background** (intentional; the transport is native).
  A request that no caller waits for, which the transport sends from its own threads (the releases
  of the results of an execution, the removal of the relation of a dropped checkpoint, the release
  of the cached objects of the native ML classes, the interrupt of the execution of a signal),
  warns `[RETRIES_EXCEEDED]` when its retries are spent, and logs its failure where PySpark's
  sender logs it (`ReleaseExecute failed with exception: ...`), from the delivery thread of the
  drop-in, a thread of its own, where PySpark warns from the thread that sends it: the threads of
  the transport are not threads of Python, and do not wait for the GIL. So the warnings come in the
  order of the delivery thread, and the exception of a filter of `"error"` is dropped there, as
  PySpark's senders in the background catch it. A wait before a retry of such a request is slept
  as PySpark's `Retrying` sleeps it, of Python's `time.sleep`, when it is negative or of decades,
  on a thread of Python of the drop-in's own: a wait that `time.sleep` refuses fails the request at
  once, of its error, logged as PySpark's sender logs it (`ReleaseExecute failed with exception:
  timestamp out of range for platform time_t.`). That thread is one that the exit of the
  interpreter does not wait for, where PySpark's release sleeps on a thread of its
  `ThreadPoolExecutor`, which its exit joins (`concurrent.futures`): of a release that waits 2^31 s
  (68 years), PySpark's exit waits for the end of the sleep, and the drop-in exits. At the exit of
  the interpreter, the drop-in closes its clients at once, of the releases in flight cancelled,
  where PySpark's exit waits for its releases in flight (of a release that the server holds 5 s,
  PySpark's process ends after about 6.6 s, and the drop-in's after about 1 s).
- **The details of a failure that the client detects** (intentional; the transport is native). The
  message of a gRPC failure of no error of the server is the text of grpcio's error of the call, as
  PySpark's, of the details and the `debug_error_string` of gRPC's C core for a failure that the
  client detects itself: a channel that fails to connect, of the last error of its addresses
  (`failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:1: Failed to connect
  to remote host: Connection refused`, `connection attempt timed out before receiving SETTINGS
  frame`, the BoringSSL error of a TLS handshake, `HTTP proxy returned response code 403`), a name
  that does not resolve (`errors resolving ...: [field:hostname lookup error:address lookup failed
  for ...: Domain name not found]`), a call that the end of its connection fails (`Stream
  removed (Socket closed)`, `Stream removed (recvmsg:Connection reset by peer (104))`, `Stream
  removed (ping timeout)`), and a call whose stream the server resets, of the code of its reason as
  gRPC's core gives it (`INTERNAL`, `Stream removed (RST_STREAM (Received RST_STREAM with error code
  1))`). What stays different: the TLS handshake is rustls's, so its failure has
  BoringSSL's text of the same cause, a certificate that fails its verification, an alert of the
  server or a record that is not of TLS, and of any other cause a text of BoringSSL's form of
  rustls's error; a peer that resets or closes a connection as it opens has the text of the read or
  of the write that sees it first, which depends on the timing, as in gRPC's core, whose text of it
  varies the same way; a call that another failure of HTTP/2 ends, such as a GOAWAY, has tonic's
  text, and so has a stream that the server resets of `NO_ERROR` after the headers of its response,
  which hyper takes for the end of the response, of tonic's `UNKNOWN` of a missing status (`protocol
  error: missing grpc-status trailer, ...`); and the failures of a name are of the system's resolver,
  named as
  c-ares names them (`Domain name not found`, `Could not contact DNS servers`), whose codes do not
  tell every failure of c-ares apart.
- **The connections of the transport** (intentional; the transport is native). A channel connects as
  grpcio's (see `docs/compatibility.md`): to the addresses of the host, or of the proxy, each 250 ms
  after the previous one, each with a connect deadline of 20 s of its own, its calls failing at once
  once it failed to connect, while it connects again with gRPC's backoff, and the sessions of the
  same target and of no token share one connection, as gRPC's subchannel pool shares one, where a
  session of a token has a connection of its own. What stays different: a name of `/etc/hosts` other
  than `localhost` and the names under it, which are the loopback of both families, as c-ares gives
  them, is resolved by the system's resolver, which gives the addresses of both families of the
  file, where c-ares, grpcio's, gives those of the family of its first line; the keepalive pings are
  hyper's, at the time and the timeout of the parameters, where gRPC's core paces its pings itself,
  so a connection that stops answering does not end at the same time; a TLS channel trusts the
  system's store too, besides the bundle of grpcio, where grpcio trusts its bundle alone, so that a
  CA that the administrator of the system installs is trusted (see `docs/configuration.md`); of a
  roots file without a certificate, the client writes to stderr two lines of each attempt,
  `Handshaker factory creation failed with TSI_INVALID_ARGUMENT` and `Update handshaker factory
  failed.`, where gRPC's core writes those and the errors of BoringSSL's load of the roots, of each
  channel that it makes; and of a `GRPC_VERBOSITY` of `INFO` or `DEBUG`, the client writes to stderr
  the lines that it writes by default, without the other lines of gRPC's core, whose code it has
  not.
- **Messages over the size limit of a call** (intentional; the transport is native). A response or a
  request over 128 MiB fails its call with grpcio's `RESOURCE_EXHAUSTED` and the details of gRPC's
  core (see `docs/compatibility.md`). What stays different: tonic checks the length of a response
  before any codec reads it, so its status of it is told by its code, its pinned text and the limit
  of the client (`OUT_OF_RANGE`, `Error, decoded message length too large: found <length> bytes, the
  limit is: 134217728 bytes`), and a status of the same code, text and limit that a server sends, of
  no details, as a server of tonic whose own limit is 128 MiB too does of a request over it, is
  taken for the client's, where grpcio raises the server's status as it is; a request is measured by
  its encoded length before the call, where gRPC's core measures it as it sends it, of the same
  length.
- **Metadata past gRPC's limits** (intentional; the transport is native). The metadata of a response
  past gRPC's limits fails its call as grpcio's does (see `docs/compatibility.md`), of gRPC's
  default limits, 8 KiB and 16 KiB, of gRPC core's texts, which list the entries as its encoder
  gives them, and the connection advertises the hard limit as the size of a header list that it
  takes, as grpcio's channel does, so that Spark's server, of grpc-java, which honours it, resets the
  stream of a call of more, as it does of grpcio's (`Stream removed (RST_STREAM (Received RST_STREAM
  with error code 1))`); past the soft limit, the call has the details of the server's error, of
  which PySpark's `rpc_status.from_call` raises its `ValueError` (`Code in Status proto
  (StatusCode.INTERNAL) doesn't match status code (StatusCode.RESOURCE_EXHAUSTED)`), as the drop-in
  does. Real errors of Spark reach these limits: grpc-message is percent-encoded, of 9 bytes for
  each Hangul character, so an error whose message has a few hundred CJK characters, as of a
  `raise_error` or of CJK names in the error, passes the soft limit (about 300 characters) and fails
  at random, of the `ValueError`, and one of about 900 the hard limit, which Spark's server resets,
  in both clients alike, where errors of ASCII stay under 7 KB. What stays different: the drop-in
  takes no gRPC channel options (`channelOptions` raises `NOT_IMPLEMENTED`), so
  `grpc.max_metadata_size` and `grpc.absolute_max_metadata_size` do not change the limits; the
  length of a binary value (of a key ending in `-bin`, such as the `grpc-status-details-bin` of an
  error) in the text of an entry too large alone is that of the value as the server sends it to the
  drop-in, in base64, where a server of gRPC's core sends it to grpcio's channel, which asks for it,
  as its raw bytes and one more, so that the length in the text is larger, by a third, of such a
  server (of grpc-java's, as Spark's server, the two are the same); and a header list past 16 MiB
  fails as hyper fails it.
- **The context of the exception of a failed call** (intentional; the transport is native). The
  exception that the drop-in raises of a failed call that it converts, of an error of the server or
  of gRPC, has no `__context__`, where PySpark's, raised as it handles grpcio's error of the call,
  has that error as its context, which it suppresses (`raise ... from None`), so that a traceback
  shows neither; the `ValueError` of a `google.rpc.Status` that does not match its call, protobuf's
  `DecodeError` of details that upb does not read, what the `can_retry` of a retry policy raises of
  the failure, also of a policy of the user's, and Python's `JSONDecodeError` of the metadata of an
  `ErrorInfo` that is not JSON (or the `StopIteration` of `json` that it suppresses, its context)
  have grpcio's error as their context, as PySpark's, which does not suppress it, but where the
  `DefaultPolicy` reads the status before a retry, of no context in both. So the failure of the
  interrupt that a `KeyboardInterrupt` sends, as to a server that is gone, has the
  `KeyboardInterrupt` as its context, where PySpark's has grpcio's error of the interrupt, which it
  suppresses, whose context is the `KeyboardInterrupt`. The error of a call that fails while the
  client converts another failure on the thread, which both raise as grpc gives it, as of a call of
  a handler of a record of the conversion, has no context in the handler, where PySpark's has the
  failure that is converted, as it raises it while it handles that failure; once it goes out of
  the handler, and so out of the conversion, it has that failure as its context in both.
- **The children of a fork** (intentional; the transport is native). A child of `os.fork()` makes
  the threads, the runtime and the channels of the client again, as grpcio's fork support does,
  leaves the objects of its parent to the parent, and closes the sockets of its parent's connections
  and the epoll instances of its parent's runtime as the fork returns (see `docs/compatibility.md`).
  What stays different: the eventfd of the runtime of each of its parents stays open in a child, an
  eventfd of the child's own, as an object of the parent that the child holds, such as a stream of
  an execution of the parent that it reads, writes to it when it wakes a task of the parent's
  runtime, and a descriptor of the child that took its number would get those writes: a child holds
  one descriptor more than its parent, and a child of a child one more again, where grpcio's
  children hold as many as their parent; the objects of the parent that the child frees are left in
  its memory; a read in the child of an execution that the parent started waits forever, as the
  parent's runtime, which would answer it, has no thread there; and the `stop()` or `close()` of a
  session in a child does not wait for the releases of its parent, where PySpark's child waits the
  10 s of its `close()` for release futures that never complete there.
- **The logs of the client** (intentional; the transport is native). With
  `SPARK_CONNECT_LOG_LEVEL`, or a level of the logger of `pyspark.sql.connect.logging`, the client
  logs the records of PySpark's client, of their logger, levels, messages and order, at the points
  of each call where PySpark's logs them, those of the plans, the requests and the responses that
  the transport builds and reads too (see `docs/compatibility.md`); the exception of `GRPC Error
  received` has no frames of PySpark's client. The exception of a handler or a filter of the logger
  at a record fails the call there, as it goes out of PySpark's log: a `KeyboardInterrupt` at a
  response of an execution, also of `toLocalIterator`, interrupts it, as the `except
  KeyboardInterrupt` of PySpark's does, and one at a record of the conversion of its failure
  (`GRPC Error received`, `Received ErrorInfo: ...`), which PySpark's `except Exception` writes,
  interrupts nothing, of the failure as its context. What stays different:
  - the levels are read as each call of the client starts (`logger.isEnabledFor`), for the records
    of the transport until the next call of the process starts, where PySpark reads them at each
    record: a change during a call, as by a progress handler or another thread, that enables a level
    applies from the next call on, and one that disables it, of a higher level or of
    `logger.disabled`, at once, as the records are written through the methods of the logger;
  - the settings of the plan compression of a plan that `to_proto` builds are read, and logged
    (`Plan compression threshold: ...`), by the first call of the client that sends a plan, before
    the records of that call, where PySpark's `to_proto` reads them (see "Plan logging of a deep
    DataFrame");
  - the records hold the requests that the drop-in sends: its own plan IDs of the client, and so the
    bytes and the sizes of a compressed plan, which hold them; the pickles of its UDFs, which pickle
    the return type of its JSON (`_parse_datatype_json_string('"integer"')`) where PySpark's pickle
    the type (`IntegerType()`); the Arrow IPC streams of arrow-rs of its local relations, and so the
    hashes of their cached artifacts; and of a server that sends an empty server-side session ID,
    no observed one, where PySpark's requests observe it empty
    (`client_observed_server_side_session_id: ""`; a Spark server sends one, but in its answer to
    the release of a session that it does not have). The time of a compression (`duration_s`) is
    its own;
  - the records of the deletion of a model from the ML cache by its finalizer, which the transport
    sends in the background, are written on the delivery thread as the deletion goes, where
    PySpark's finalizer sends it, and logs it, on its own thread before it goes on;
  - the records of the responses of the listener bus of streaming queries are written on the thread
    of the bus, which runs the listeners, as on PySpark's thread of its `_query_event_handler`, but
    a thread of the drop-in, which Python names `Dummy-N`, where PySpark's is `Thread-N
    (_query_event_handler)`;
  - a handler or a filter of the logger that raises at a record of the retries of a request that
    the transport sends in the background, which no caller waits for (the release of an execution,
    the removal of the relation of a dropped checkpoint): PySpark's sender catches the exception
    and logs it as the failure of the request (`ReleaseExecute failed with exception: ...`); the
    drop-in drops it, but for the `KeyboardInterrupt` of a signal, which the next wait of a call
    of the main thread raises;
  - the record of the interrupt of an execution by a `KeyboardInterrupt` of Python code of the
    client at a response, of a handler or a filter of the logger at a record or of a retry policy
    (`Interrupt request received for operation=...`), is written after the done call of the
    progress handlers, where PySpark's writes it before that call;
  - at the exit of the interpreter, the cleanup of the ML cache of a client whose server is gone or
    unreachable logs the failure of its execution (`GRPC Error received`), where PySpark's does
    not: there its release of the failed execution raises `RuntimeError: cannot schedule new
    futures after interpreter shutdown` in place of the failure, before the failure is converted;
  - after a `KeyboardInterrupt` of a signal that the wait of a call took interrupted an execution,
    the drop-in reads it, of no records, until the server ends it, and then releases it, where
    PySpark's reads nothing more of it and releases it as it interrupts it: the server may refuse
    that release, in a race with the interrupt
    (`[STATE_CONSISTENCY_EXECUTION_STATE_TRANSITION_INVALID_OPERATION_STATUS_MISMATCH]`), which
    PySpark logs from the thread of its releases (`ReleaseExecute failed with exception: ...`, a
    warning), where the release of the drop-in, of an execution that ended, is not refused. A
    `KeyboardInterrupt` raised while a record is written, by its handler or by the signal handler,
    releases the execution at once, as PySpark's does, and that release may be refused as well;
  - a read of the file of an artifact that fails during its upload, once the client has opened it
    (a file removed or made unreadable meanwhile), fails without the record that PySpark's generator
    of the requests writes (`Failed to submit addArtifacts request: ...`); the interceptors of a
    channel builder, which the transport does not apply, are not logged (`Applying interceptors
    (...)`); and the schema of a DataFrame is not pickled, so nothing is logged of a schema of a
    user-defined type that pickle refuses, where PySpark's `_schema` logs that it falls back to a
    deep copy (`DataFrame schema pickle dumps failed with exception: ...`).
- **The call stack of the requests** (intentional; the methods of the extension have no frames).
  With `SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK` (`true` or `1`), the requests of the executions, the
  analyses and the configurations carry the call stack of the user's code last of their user context
  extensions, as PySpark's `_build_call_stack_trace` builds it: a `FetchErrorDetailsResponse.Error`
  of the frames from the outermost to the first frame of the pyspark package, each of the function
  of the frame after it, which the reattaches and the releases of an execution copy; the request of
  a finalizer, of the stack as it runs, of its `__del__`; an attribute that is a
  `functools.cached_property` of PySpark's (`DataFrame.isStreaming`, `SparkSession.version`), of
  the frame of the `__get__` of `functools`; and a call that runs during another call of its
  thread, as of a progress handler, a hook or a finalizer, of the stack of that call, as PySpark's
  ends at the first frame of the pyspark package of that call. What stays different: the innermost
  frame of the user's code that calls a method of the extension, which has no frame, has the name of
  the method or the attribute that the source of its call names (`collect` of `df.collect()`), where
  PySpark's has the name of the function of PySpark that it calls, the same but of a call whose
  source names no attribute, as of an alias or of `getattr`, or of no source, as of `exec`, of which
  it has its own name; the stack is read once, as the call starts, for all the requests of the call
  and of the calls during it, of the same frames of the user's code as those of PySpark's requests;
  and a call of a listener of streaming queries, which the drop-in runs on the thread of its
  listener bus, a thread of its own, carries the frames of the listener, where PySpark's carries
  those of the `threading` module of its thread, up to its handler of the pyspark package.
- **`SparkConnectClient.thread_local`** (intentional; the transport is native). The
  `thread_local` of the client, its set of tags and its lists of user context extensions are
  PySpark's plain `threading.local`, set and lists, as are the `global_user_context_extensions` of
  the client and its `_session_id` and `_user_id`, plain attributes, which the transport reads on
  the calling thread when a call starts its requests, when a finalizer sends its request, once a
  call or a finalizer of the session that ran during a call of the session ended, and when it
  builds the requests of the details of the failure of a call, of the configs of its stack trace
  and of the interrupt of a signal, of the client of the call as they are then, the copy of a
  client whose method made it or the session's, also of the `next()` of the iterator of an
  execution, as PySpark's requests read them when they are built; the requests that continue an
  execution, its reattaches and releases, copy its request, as PySpark's. The other requests that
  a call sends after its first carry them as they were at the start of the call, or at the end of
  a call or a finalizer of its session that ran during it, where PySpark's read them when they are
  built, so a change of them by a progress handler or by another thread meanwhile is not in them.
  Read for the requests of the failure of a call, a value of them that protobuf refuses, or an
  `Exception` of their read, is the failure in place of the error of the server, which becomes its
  context, of no request of the failure, as PySpark raises the error of the build of the request of
  its details while it handles the failure, and goes with a failure that the call handles itself,
  as the failed lookup of the plan compression settings; read for the interrupt of a signal, it
  replaces the `KeyboardInterrupt`, which becomes its context, and no interrupt is sent, as
  PySpark's `interrupt_operation` raises it in its `except KeyboardInterrupt`. Read at the end of a
  call or a finalizer that ran during a call, a value that cannot be read stays as it was. An
  exception of a read that is not an `Exception`, as the `KeyboardInterrupt` or the `SystemExit`
  of a signal handler that runs then: at the end of a call that ran during a call, it is raised by
  that call, in place of its result, or of the error of its refresh, which becomes its context, as
  PySpark raises a signal in the call that it arrives in, also when an attribute of the client
  raises it, which PySpark does not read then; for the failure of a call or after a finalizer that
  ran during it, it is raised by the wait of that call at its next poll, as the exception of a
  signal handler, a `KeyboardInterrupt` of which interrupts the execution that it waits for, or by
  the call when it returns. A `KeyboardInterrupt` of a progress handler interrupts the execution
  too, as the `except KeyboardInterrupt` around PySpark's handlers does, of the attributes of the
  client of the execution (an error of its build replaces the `KeyboardInterrupt`, and no
  interrupt is sent), before the done call of the handlers, where PySpark interrupts it after that
  call. A finalizer whose last reference a thread of the transport drops, not a thread of Python,
  sends the context of the last call of that thread, else of the session, where PySpark's
  finalizers run on a thread of Python. A request of more than one value that protobuf refuses
  raises the error of the session ID, of the user and of the extensions before that of a tag,
  where PySpark's execution raises that of the tag before those of the user and of the
  extensions. An extension that is not an `Any` raises upb's `TypeError`,
  which names the runtime's class of `Any` (`pyspark.sql.connect.proto._rust.any_pb2.Any`) where
  PySpark's names `google.protobuf.any_pb2.Any`. A failed request leaves `inside_error_handling`
  False in the `thread_local`, as PySpark's `_handle_error` does, which the drop-in neither reads
  nor sets True while it converts a failure: it raises the error of a request as grpc gives it
  while the client converts another failure on the thread, as PySpark's does while the flag is
  True, but not of a flag that a caller sets True, and a handler of a record of the conversion
  reads the flag as the last call left it, where PySpark's reads True. A `copy.copy` of a
  client sends its own attributes, as PySpark's, also at once with its client on another thread,
  and a call of the session that runs during a call of the copy, as of a progress handler or a
  finalizer, sends the session's, after which the call of the copy sends its own again, as they are
  then; but it shares with its session whether it is closed (`_closed`), the server-side session ID
  that the responses carried, whether its executions are reattachable and the retry policies, which
  PySpark's copy has of its own. So the `close()` of a copy closes its client too, whose calls then
  raise `NO_ACTIVE_SESSION`, where PySpark's copy shares only the channel with its client, which
  stays open (`is_closed` False) and whose calls then raise grpc's `ValueError` of a closed channel
  (`Cannot invoke RPC on closed channel!`, and `Cannot invoke RPC: Channel closed!` of an
  execution).
- **Plans of a DataFrame's relation** (intentional; DESIGN). `df._plan.plan(client)` is a
  placeholder that holds the DataFrame's relation, which the core builds at the call, as
  PySpark's `plan` does, with its pickles, warnings and errors: the core builds the relation, and
  Python has no messages of it, so it is one message, an extension whose value holds the relation
  encoded, which Python's protobuf does not parse, rather than PySpark's messages of each level.
  The core replaces it by that relation, also after the DataFrame is gone, wherever
  `plan::resolve_relation`, `resolve_command` and `resolve_analysis` find it: as the root relation
  of a plan or of an analysis, as the input of every relation of Spark Connect under it
  (`relation_inputs_of`: both sides of joins, set operations and zips, the other inputs of
  `GroupMap` and `CoGroupMap`, the root and references of a `WithRelations`, so the views of `SQL`),
  as the relation of each command that holds one (writes, views, streams, `SqlCommand`, checkpoints,
  `MergeIntoTableCommand`, the flows of pipelines), as the dataset and the fetched arguments of the
  ML commands, anywhere in the message of the extension of a relation or of a command (a
  plugin's, as Delta Lake's `merge`), and in the extensions of the expressions of the relations and
  the commands built in Python, which the core finds by the type URL of the placeholder in the
  encoding of their fields (`plan::resolve_own_fields`), when the bytes of the message that the
  client gives hold a type URL that is not of a placeholder at its relations (`plan::BuiltOutside`).
  A placeholder whose `common` a plan changes (`plan.common.plan_id = ...`) is sent with that
  `common` merged, as PySpark's changed relation.
- **A channel builder without a host** (intentional; the transport is native). The builders of
  `pyspark.sql.SparkSession` and `pyspark.sql.connect.session.SparkSession` are PySpark's two
  builders, with its errors: `CONNECT_URL_NOT_SET`, a `PySparkRuntimeError`, and
  `SESSION_NEED_CONN_STR_OR_BUILDER`, a `PySparkValueError`, also for both a URL and a channel
  builder. `channelBuilder(...)` takes any object, and the session fails when it is made, of the
  errors of PySpark's client: the transport connects to the `sc://` URL of the `endpoint` of a
  `DefaultChannelBuilder`, or of the `host` and the `port` of a builder of its own, and of what
  PySpark's client reads of the builder (see "Custom gRPC channels"), and calls the `toChannel` of a
  `ChannelBuilder` that is not a `DefaultChannelBuilder` for its errors only, so a builder of its
  own `toChannel` and no `host` raises the error of the `host` of `ChannelBuilder`, where PySpark's
  connects through its channel.
- **Plan logging of a deep DataFrame** (intentional; DESIGN). `DataFrame._plan` is a
  placeholder over the relation of the Rust core, whose `to_proto(client)` is the `Plan` of the
  placeholder, `plan(client)`, which the client's `_set_relation_in_plan` sets as the root, or
  compresses, as PySpark's, of the relation that the placeholder holds, into a
  `compressed_operation` of over `spark.connect.session.planCompression.threshold` bytes (10 MiB,
  which the client asks the server for at its first plan). Python's protobuf refuses to decode a
  message nested more than 100 levels deep: its default implementation, upb, raises a `DecodeError`
  (`Error parsing message with type 'spark.connect.Plan': Exceeded upb_DecodeOptions_MaxDepth` of
  protobuf 7, of no reason after the type of protobuf 6, where the drop-in's runtime gives
  protobuf 7's), its pure-Python one `Error parsing message: too many levels of nesting.`; the
  drop-in never decodes a deep plan in Python. `SparkConnectClient._proto_to_string` renders a
  message of the drop-in's protobuf runtime in Rust at any depth, as do the logs of the plans, the
  requests and the responses of the transport: past about 120 nested relations (of Python's
  recursion limit, of a caller of few frames), PySpark's `text_format` raises its `RecursionError`,
  and it returns `<Truncated message due to recursion error>`, where the drop-in returns the full
  text. With `truncate=True`, both cut the text at 8 levels, and their strings and bytes short, of
  protobuf 6; `_truncate` reads `FieldDescriptor.label`, which protobuf 7 and the drop-in's runtime
  without protobuf do not have, so that both return `<Truncated message due to truncation error>`
  of a message of a field of a message, a string or bytes, as the truncated plans of the logs are.
  Plan logging of a deep DataFrame therefore does not reach the recursion handling that the test
  `test_recursion_handling_for_plan_logging` exercises.
- **The `close()` of a client whose session has streaming query listeners** (tracked). PySpark's
  `client.close()` closes the channel, which ends the stream of the events of the listener bus: its
  handler thread logs `GRPC Error received` of the `CANCELLED` of the stream (`Stream removed
  (Channel closed!)`), warns `StreamingQueryListenerBus Handler thread received exception, all
  client side listeners are removed and handler thread is terminated. The error is: ...`, and
  removes the listeners. The drop-in's `close()` removes the listeners and ends the bus before it
  closes the channel, of no record and no warning. `stop()` is the same in both, as the release of
  the session ends the stream of the bus first.
- **Client session ID** (intentional). `SparkConnectClient._session_id` is the session ID that
  the requests carry, an attribute that a set changes for them, as PySpark's, so the next response
  of another server-side session marks the session closed, of an open channel, as PySpark's
  `_closed = True` of then; `_server_session_id` is exposed and settable, as PySpark's, and a
  changed value marks the session closed so. A `_session_id` of None raises
  the `TypeError` of the string field of the session ID for every request, where PySpark's
  executions and `CloneSession` send an empty session ID, which the server refuses; one of the
  bytes of a session ID is sent as its text, where PySpark's check of the responses compares the
  str of the server with the bytes and fails with `PySparkAssertionError`.
- **The exit of a dropped session in a reference cycle** (intentional). PySpark's
  `SparkSession.__del__` closes the client of a session object that is freed, and the exit hook of a
  closed client sends nothing; the drop-in closes a session once the program dropped its objects and
  the objects that hold it, also when a `conf` or a `client` of it is kept, so it sends nothing at
  exit either. Once its `catalog` or its `streams` is read, a session object of either client is in
  a reference cycle with its `Catalog`, the `catalog` of its `__dict__`, which holds the session as
  its `_sparkSession`, or with its `StreamingQueryManager`, its `_sqm`, which holds it as its
  `_session`, so that a drop does not free it: it stays open until a collection of the garbage
  collector frees it, which the allocations of the program decide, and an exit before that sends its
  `clean_cache` (and its release, with `SPARK_CONNECT_RELEASE_SESSION_ON_EXIT`). The drop-in's
  session has none of the other cycles that keep a PySpark session until a collection, so it is
  freed when the program drops it, as PySpark's is after a collection (`gc.collect()`), and its exit
  sends nothing, whatever the timing of the collections: such as that of an error that the
  `_handle_error` of its client raises again (`raise error`), whose traceback keeps the frame that
  holds the error and, with it, the frames of the calls that led to it and their variables. PySpark
  4.3.0 then does not exit at all when no collection frees the session before the teardown of the
  interpreter, as of the cycle of its `streams`, or of any of these cycles under `gc.disable()`, and
  it does not while a session other than its default one is left open to the end (of `newSession()`,
  or `SparkSession(connection=...)`): its `SparkSession.__del__` closes the client in the teardown
  of the interpreter, whose grpcio `Channel.close()` waits forever. That is a bug of PySpark, which
  the drop-in does not reproduce: it exits.
- **A `client` kept apart from its session** (intentional; the state is native). As PySpark's
  `RuntimeConf` and `SparkConnectClient`, which hold the client of a session and not the session
  object, the drop-in's do not keep the session open: once the program dropped the session and the
  objects that hold it, as its DataFrames, a kept `conf` or `client` raises `NO_ACTIVE_SESSION`, as
  PySpark's raise once the `__del__` of the session closed the client, and the exit sends nothing
  for the session. The client of such a session keeps its attributes and its retry policies, but
  reads its server-side session ID and whether its executions are reattachable of a closed session
  of its connection, which has the defaults, where PySpark's client keeps its own.
- **The `__dict__` of the objects of a session** (intentional; the state is native). The
  readers, writers and the other objects of a session or a DataFrame hold their
  state in the Rust core: their `__dict__` holds the attributes that a user sets on them,
  which their copies and pickles keep as PySpark's do, but not their state, which PySpark's
  holds there (`vars(df.write)` is empty, where PySpark's has `_df`, `_spark`, `_write` and
  `_callback`). These are not of them, and hold their state in their `__dict__`, as PySpark's: the
  `DataFrameReader` and the `DataStreamReader` of `spark.read` and `spark.readStream`, PySpark
  Connect's classes, of their `_client`, `_format`, `_schema` and `_options` (and `_source_name`),
  whose code is PySpark's, adapted to the plans of the Rust core (see `docs/compatibility.md`); an
  `Observation`, of its `_name`, `_result` and `_error`; the `Catalog`, the `TableValuedFunction`,
  the `DataSourceRegistration` and the `RuntimeConf` of a session, of the session of their
  `_sparkSession` (`sparkSession` of a registration) and of the client of their `_client`; the
  `StreamingQueryManager` of `spark.streams`, of its `_session` and its `_sqlb`, its
  `StreamingQueryListenerBus`, of its `_sqm`, `_listener_bus`, `_execution_thread` and `_lock`,
  and a `StreamingQuery`, of its `_session`, `_query_id`, `_run_id` and `_name`; the `GroupedData`
  of a DataFrame, of its `_df`, `_group_type`, `_grouping_cols`, `_pivot_col`, `_pivot_values` and
  `_grouping_sets`, its `PandasCogroupedOps`, of the grouped data of its `_gd1` and `_gd2`, and its
  `DataFrameNaFunctions` and `DataFrameStatFunctions`, of their `df`, whose methods theirs call; the
  `SparkSession.Builder` of Spark Connect, of its `_options` (the `to_str` of each value),
  `_channel_builder` and `_hook_factories`, which its setters set and its `create`, `getOrCreate`
  and `_apply_options` read; and the `SparkSession` of Spark Connect, of its `_client`,
  `_session_id` and `release_session_on_close`, and then the `_sqm` of its `streams` and its
  `catalog` and `version`, PySpark's `functools.cached_property`s. Their methods read their state
  when they make their requests or their plans, as PySpark's, and the events of the listeners of
  streaming queries are posted to the listeners of the `_listener_bus` as they are then (see
  `docs/compatibility.md`). Of the listener bus, the subscription of the session to the events is
  the native listener bus's, of which Rust programs get the events too, whose thread posts them and
  which keeps a registration of each listener that `append` adds: `_execution_thread` stays None,
  where PySpark's holds the `threading.Thread` of its handler while the bus has listeners, and the
  `_lock` is held while the list is read or changed, but not while the listeners are called, where
  PySpark's holds it around the calls, so that a listener may add or remove listeners of its
  callbacks, where PySpark's waits for ever. The end of the events, of the server, of a stop of the
  session or of a close of its client, takes the listeners off the list, as PySpark's handler
  clears it when its events fail, and `close()` removes every listener, where PySpark's removes
  each other one, as it removes them from the list that it iterates. The plan that a method of
  grouped data makes is a value of the core, of the state that the grouped data hold at the call: a
  later change of the lists of that state, such as `gd._grouping_cols.clear()` after
  `agg = gd.count()`, is not of `agg`, whose grouping is kept, where PySpark's `Aggregate` and
  `GroupMap` hold the lists and build their messages of them at each request, as the plans are the
  core's (see "Plans of a DataFrame's relation"). What remains of these is inherent, as the
  transport is native: the requests of the session, of a catalog, of the table-valued functions, of
  a streaming query and of a manager are made by the native transport of the session of the
  drop-in's client of the `_client` or the `client` of the session that they are or hold (at once
  while that is the own client of the session), or, of another object, as a proxy of a session, of
  its `client`, where PySpark's call the methods of that client, such as its `to_table` and its
  `execute_command`. So the methods that a subclass of the drop-in's client overrides are not called
  by them, a client of another class raises a `TypeError` of it, and the DataFrames that the
  session, a reader, a catalog and the table-valued functions make are of that session at the call,
  which a DataFrame holds natively (above), where PySpark's are of the object that they hold, whose
  client they read at each request: so the session of a reader of another object than a session is
  resolved when its read makes its DataFrame, where PySpark's DataFrame resolves it at a request.
  The plan of a read is the core's, which takes the options of the reader when it is made, where
  PySpark's `Read` plan of `table` keeps the dict of the reader's options, of which a later `option`
  changes its requests. A
  `RuntimeConf` calls the `config` of a client of another class, or of a subclass that overrides
  it, and a registration the `register_data_source` of the `_client` of its `sparkSession`, as
  PySpark's do. Calling the methods of the client for each request would be a second
  implementation of the requests, of PySpark's Python client over the results that those methods
  return, where the native transport makes them: with its retries, its reattachment, its progress,
  its interrupts and its decoding of the results, in Rust.
- **Subclasses** (intentional; the state is native). PySpark's classes are plain classes that a user
  subclasses. The drop-in's DataFrame, `Observation`, the `SparkSession`, `Catalog`,
  `TableValuedFunction`, `DataSourceRegistration`, `RuntimeConf`, `StreamingQuery`,
  `StreamingQueryManager`, `StreamingQueryListenerBus`, `DataFrameReader`, `DataStreamReader`,
  `GroupedData`, `PandasCogroupedOps`, `DataFrameNaFunctions` and `DataFrameStatFunctions` of
  `pyspark.sql.connect`, `SparkSession.Builder`, `SparkSession.Hook`, `StorageLevel`,
  `PythonEvalType` and the data types are subclassed as PySpark's: a subclass constructs of the
  arguments of its `__init__`, and its copies and pickles are of the subclass (but those of a
  session, which are the active session of its ID, as PySpark's), and PySpark's pickle of a reader,
  a catalog or a registration, which makes it of its class without arguments, loads in the drop-in.
  The `Window` of `pyspark.sql.window` and of `pyspark.sql.connect.window`, and the classic
  `Column`, `DataFrameNaFunctions`, `DataFrameStatFunctions`, `WindowSpec` and `TableArg`, are
  PySpark's classes, which construct and subclass as PySpark's. The catalog, the table-valued
  functions, the registration, the runtime configuration, the streaming queries, their manager and
  its listener bus, the grouped data, the cogrouped data, the null and statistics functions and the
  session keep their state in their `__dict__`, as PySpark's (see the row above): their `__init__`
  keeps any object, which their first use reads, but for the assertions of the `__init__` of
  `GroupedData` (a DataFrame, a grouping and a list of Columns) and the `_check_same_session` of the
  `_df` of the grouped data of `PandasCogroupedOps`, which raise as PySpark's, and the `connection`
  of a session, of which its `__init__` makes its client as PySpark's does, and one whose `__init__`
  never ran raises the `AttributeError` of the first attribute that PySpark's method reads
  (`_sparkSession`, `sparkSession`, `_client`, `_query_id`, `_session`, `_lock`, `_df`, `_gd1`,
  `df`), and so do the readers, PySpark's classes. The other native
  classes are not subclassable (`TypeError: type '...' is not an acceptable base type`), and do not
  construct (`cannot create '...' instances`): their objects are of the native state of a plan, an
  expression or a query, which the arguments of PySpark's constructors, the Python objects of its
  client, do not give, as the drop-in's native state replaces those objects, so a subclass would
  have no `__init__` to call (see `docs/compatibility.md`):
  - `Column`: an `Expression` of `pyspark.sql.connect.expressions`;
  - `DataFrameWriter`, `DataFrameWriterV2`, `DataStreamWriter`, `MergeIntoWriter`: the
    `LogicalPlan` of a DataFrame;
  - `WindowSpec`, `TableArg`: expressions and a `WindowFrame`, a `SubqueryExpression`.

  A Python subclass of a data type (`StringType`, `DecimalType`, `ArrayType`, `StructField`,
  `StructType` and the others) is of the core type of its class, and runs its own methods where
  PySpark's methods of the class call them, also in the types that hold it and on the Python
  workers of the server, to which it pickles of its class and its `__dict__`, as PySpark's. What
  stays different (see `docs/compatibility.md`):
  - the parameters are not entries of its `__dict__`, and a subclass whose `__init__` does not
    call that of its class has the defaults of the class, where PySpark's has no parameters;
  - the `__init__` of a subclass of a singleton type runs at each call of the class, as the
    native classes are not of the `DataTypeSingleton` metaclass (which `pyspark.sql.types` has, of
    PySpark's, for the classes made of it, such as PySpark's cloudpickle of such a subclass of
    `__main__`), and copies and pickles of its one instance are that instance; a subclass of
    another metaclass is made, where PySpark's raises a metaclass conflict;
  - a class of types of two branches of their hierarchy (`class X(StringType, IntegerType)`)
    raises CPython's `multiple bases have instance lay-out conflict`;
  - a subclass of `__slots__` pickles at the protocols 0 and 1 too, where PySpark's raises the
    `TypeError` of `__slots__` without `__getstate__`;
  - the repr of an array, a map, a field or a struct that holds a type whose repr has a lone
    surrogate raises its `UnicodeEncodeError`;
  - `createDataFrame` of local rows sends the JSON of the core types of its schema, as its rows
    are of them (until SPARK-62258), and so does one of a pandas DataFrame or a pyarrow Table of
    types that override none of the reads of `json()`, where PySpark sends the `json()` of the
    object, which the server may fail to parse; of a struct that holds a type that overrides one,
    of a pandas DataFrame, a pyarrow Table or data of no rows, it sends the `json()`, as PySpark
    (see `createDataFrame` above);
  - `simpleString()`, `repr()` and `treeString()` read the native values of the attributes that a
    subclass overrides with a property, where PySpark's read the property, as the drop-in's
    `json()` and `jsonValue()` now do (`struct<a:bigint>` and `StructField('a', LongType(), True)`
    of a field whose `dataType` reads `StringType()` over `LongType()`, where PySpark's are
    `struct<a:string>` and `StructField('a', StringType(), True)`), until SPARK-62251.

  The class's own types compare without the attributes that a user set on them, where PySpark's
  comparison of the `__dict__`s holds them (see `docs/compatibility.md`); their copies and pickles
  keep them, as PySpark's.
- **The functions of `pyspark.sql.connect.functions`** (intentional; the drop-in's, of the Rust
  core). `pyspark.sql.functions` is PySpark's package, as it is, whose functions run those of
  `pyspark.sql.connect.functions` of their names, as with `pyspark-client`. Those are the drop-in's,
  of the docstrings, the annotations and the module of PySpark Connect's, which call the `lit`,
  `column` and `_to_col` of their module as PySpark Connect's do (see `docs/compatibility.md`), but:
  a traceback of a generated one has the frames of `pyspark/sql/connect/functions/_generated.py`,
  where PySpark Connect's are of its `builtin.py`; `col` has no `__wrapped__` of PySpark's
  `_with_origin` (the Rust core captures the call site); `pyspark.sql.connect.functions.builtin` has
  none of the expression classes of PySpark Connect that its module imports (`CaseWhen`,
  `ColumnReference`, ...), nor `np`, and its private helpers are its own; and the annotations of
  `schema_of_csv`, `schema_of_json` and `schema_of_xml` are those of PySpark's classic functions,
  `Union[Column, str]`, where PySpark Connect's write `Union[str, Column]`.

## `pyspark.sql.connect.proto` and plugins

- **Generated modules of a plugin's protos** (intentional; DESIGN). A module that protoc
  generates of a file that imports the protos of Spark Connect, directly or through the files it
  imports (Delta Lake's `delta/connect/relations.proto` and `commands.proto`), of proto2, proto3 or
  editions, the module of each file of its package (of a package) that it imports (Delta Lake's
  `delta/connect/base.proto`, also once built), and the module of each file that imports such a
  file (also once built), get the classes of the drop-in's Rust protobuf
  runtime when `protobuf`'s metaclass builds them, as the modules of protoc 3.20 and later (through
  its builder) and those of an older protoc do, which `protobuf`'s lookups of their types give too,
  so their message fields take the messages of `pyspark.sql.connect.proto` as they are, as
  PySpark's plugins take PySpark's, and the server receives the plans of Delta Lake 4.0's
  `delete`, `update` and `merge` as PySpark sends them: the core replaces the placeholder that
  holds the relation of a DataFrame, which `DataFrame._plan.plan(session)` gives, in the message of
  an extension too. The files of other packages that such a file imports (a company's shared
  proto), of no package, and Google's (`google/protobuf/duration.proto`, `google/rpc/status.proto`),
  keep `protobuf`'s classes, in their modules and of `protobuf`'s lookups, so that the modules of
  other libraries that import them take their messages as they do without the drop-in, as does a
  file whose module the runtime does not find. What stays
  different (see `docs/compatibility.md`):
  - the core replaces that placeholder where it replaces the plan of a DataFrame (see "Plans of a
    DataFrame's relation" above), also one whose `common` the plugin changes, with the change;
  - a message field of such a module of a type of a file that keeps `protobuf`'s classes (of
    another package, or of Google's, such as a well-known type other than `Any` and `Timestamp`)
    gives a message of the runtime's own class of the type, of the module and qualified names of
    `protobuf`'s, where upb has one class of each type: `isinstance` with `protobuf`'s class tells
    them apart, and `protobuf`'s messages of other files take it only through its bytes or a dict
    (their `CopyFrom`, constructors and containers raise upb's `TypeError` of a message of another
    class, as `struct_pb2.Struct(fields=message.options.fields)` does), though the two are equal
    of the same contents, the runtime's fields take `protobuf`'s messages of the type as they are,
    and it pickles as upb's, of `protobuf`'s class, where `copy.copy` keeps the runtime's;
  - the runtime's class of such a type pickles as `protobuf`'s class of the type, as its messages
    do; `cloudpickle` pickles it by value, which fails, so a function that refers to one does not
    pickle;
  - the `_concrete_class` of `protobuf`'s descriptor of a type of the runtime's classes is, of
    upb, upb's `SystemError` (as upb's of a type that it made no class of, since upb gives only the
    classes that it made), and, of `protobuf`'s pure-Python implementation, the runtime's class,
    which the runtime sets of a descriptor of no class, as `protobuf`'s metaclass does, never in
    place of `protobuf`'s class of a descriptor (of a well-known type, of a module built before);
    `message_factory.GetMessageClass` gives the runtime's class of either;
  - a file that declares extensions or extension ranges, or a message set, keeps `protobuf`'s
    classes, as the runtime has no extensions, and so do the files that import it (a file of custom
    options, which imports `google/protobuf/descriptor.proto`): their classes take the package's
    messages, and those of a plugin's module of the runtime's classes, only through bytes or dicts,
    where upb's take them as they are. A file that the runtime took over that such a file imports
    gets `protobuf`'s classes back (but a proto of Spark Connect and a file that imports one),
    whether the file builds before the plugin's or after, in its module, the names of the modules
    of the plugin's top-level package and `protobuf`'s lookups, so upb's classes of the file take
    its messages. What the give-back does not reach keeps the runtime's class, which upb's classes
    of the file refuse (`TypeError`; a dict or the bytes are taken): a name bound outside that
    top-level package before (`from gb.common_pb2 import Common` in `__main__`), a message of the
    runtime's class made before the give-back, and the values of the plugin's message fields of
    the type, which are the runtime's messages;
  - a name that the runtime binds to its class in place of `protobuf`'s is a global of a module of
    the top-level package of the module that it takes over (as Delta Lake's
    `delta/connect/proto/__init__.py` star-imports `base_pb2`), bound to `protobuf`'s class of the
    type, so `protobuf`'s class that a module of another package or another object holds (an
    attribute, a closure) stays, as do the messages that it made, which the runtime's fields take
    as they are, and equal the runtime's of the same contents.

## Type checking (the stubs of `scripts/gen_type_stubs.py`)

mypy and pyright type an application as with `pyspark-client` (see `docs/compatibility.md`), but:

- **Imports of PySpark's modules** (intentional). A name that a module of PySpark imports, such as
  `DataFrame` of `pyspark.sql.connect.catalog`, is an attribute of the module, which mypy takes by
  default with `pyspark-client`. The stubs declare it only when the drop-in's module has it at run
  time, so that it is an error, as `from pyspark.sql.connect.catalog import DataFrame` fails with
  the drop-in.
  pyright reports it as a private import with `pyspark-client` too.
- **Names that only PySpark's annotations need** (intentional). `TYPE_ONLY` of
  `python/tests/test_type_stubs_offline.py` lists the names that the stubs declare and the drop-in
  does not have, with their reasons: the classic-only `DataFrame.rdd`, as type checkers of
  `pyspark-client` see it, and the type variables of the annotations of `pyspark.sql.types`.
- **The internals of PySpark Connect** (intentional). The private attributes of the Connect
  classes that PySpark annotates with its plans and expressions (`DataFrameWriter._write`, ...)
  are `Any`, as the drop-in builds them in Rust (`UNRESOLVED_IMPORTS` of the generator).
- **The modules of `pyspark.ml.connect`** (intentional; the drop-in's ML API, see
  `docs/compatibility.md`). Their classes are typed as the drop-in's, subclasses of PySpark's
  classes of `pyspark.ml`, as at run time, where those of `pyspark-client` are its ML API on
  PyTorch.
- **Docstrings of PySpark Connect** (intentional). The stubs keep the docstrings that PySpark's
  objects have at run time, also those that the modules of PySpark Connect copy of the classic
  ones, which pyright does not see in `pyspark-client`'s source: a hover over
  `pyspark.sql.connect.functions.col` or Spark Connect's `SparkSession.builder` shows PySpark's
  documentation with the drop-in, and none with `pyspark-client`.
- **The docstrings of the `__init__` of the native classes** (inherent; the classes are native).
  The `__init__` of a native class, as of the types of `pyspark.sql.types` (`StringType`,
  `ArrayType`, `StructField`, ...) and of the classes of `pyspark.sql.connect` (`SparkSession`,
  `DataFrame`, `Catalog`, `RuntimeConf`, `Observation`, ...), is a slot of CPython, whose wrapper
  has CPython's docstring (`Initialize self.  See help(type(self)) for accurate signature.`), which
  the class cannot set. PySpark's is that of its Python function: None, its own (that of the
  `SparkSession` of Spark Connect, and `Creates a new data frame` of its DataFrame), or of a
  docstring that PySpark Connect's modules copy
  (`__init__.__doc__ = PySparkRuntimeConfig.__init__.__doc__`). The other docstrings of every
  module of `pyspark.sql` and `pyspark.errors` are PySpark's, which a golden of them guards (see
  `docs/compatibility.md`).
- **The docstrings of the native DataFrame, Column, WindowSpec, TableArg, `DataFrameNaFunctions` and
  `DataFrameStatFunctions` of Spark Connect** (inherent; a virtual subclass inherits no docstring).
  PySpark Connect's classes are subclasses of the classic ones, and neither they nor their members
  have docstrings of their own but the `__init__` of the DataFrame (above), so `inspect.getdoc` and
  `help()` show those of the classic class. The native classes are virtual subclasses of the classic
  ones, which are not in their MRO: they and their members carry the classic docstrings, so that
  `inspect.getdoc` and `help()` show PySpark's text, as `pyspark-client`'s inherited ones do. Their
  `__doc__` is that text, where PySpark Connect's own is None. So is that of `Read.print` of
  `pyspark.sql.connect.plan`, of PySpark Connect's `LogicalPlan.print`, which the drop-in's
  `LogicalPlan` does not have.

# Known behavior differences from PySpark Connect

Behaviors of the Rust-backed Python client that differ from PySpark Connect 4.3.0, kept
here so that a version upgrade does not rediscover them. Each entry says whether the
difference is intentional or tracked for a fix.

## `createDataFrame`

- **A pandas DataFrame with a StructType of another width** (intentional, SPARK-60902).
  PySpark's pandas conversion zips the DataFrame's columns with the schema's fields. With
  fewer fields than columns it silently drops the extra columns, and with more fields it
  raises PyArrow's `ArrowInvalid`. This client raises `AXIS_LENGTH_MISMATCH` for both
  rather than drop data. Of more fields both clients fail, and only the class of the error
  differs: `AXIS_LENGTH_MISMATCH` is PySpark's own error for local rows and for a list of names
  of another width than the data.
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
  here and in local rows), and its errors are its own. This fallback covers
  unavailable modules detected by `pyarrow_available`, not an installed PyArrow module that
  fails during import. The row path also converts a frame, with PyArrow, when the type names
  that the conversion and the importer resolve are rebound, or when the class of the struct of
  the schema, or of one of its fields, overrides how the conversion reads it: its `fields`,
  `names` or iteration, a field's `name`, `dataType`, `nullable` or `metadata`, or
  `__getattribute__` or `__class__` (`accepts_pandas_schema`). The conversion would read the
  override, and the importer the native type. A Python subclass of `StructType` or
  `StructField` that overrides none of these takes PySpark's conversion, as in PySpark, with the
  JSON of its core types, where PySpark sends its `json()`, which names a struct of a subclass by
  its `typeName()` and which the server then fails to parse (see the types of
  `docs/compatibility.md`). So does a name that holds a NUL, at any depth, whose table the
  importer reads of its IPC stream, as the C data interface holds names as C strings.
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
  out of Python's range in the session time zone** (intentional; the values are those of any
  other zone and timestamp). The zones are such as `Z`, `PST`, `GMT+08:00` or `UTC+01:00`.
  PySpark's `collect`, `toLocalIterator` and `first` of a timestamp localize the values in the
  session time zone, looked up by its name in Python, so they raise pytz's
  `UnknownTimeZoneError`, or, without pytz, PyArrow's `ArrowInvalid` ("The zoneinfo module or
  pytz package must be installed."). This client's conversion of the rows reads no zone of the
  session: it returns the instant in the local time zone of the client, as both clients do of a
  zone that Python knows. `toArrow` gives the same table in both, and `toPandas` raises the same
  `UnknownTimeZoneError` in both.
  So a timestamp whose wall clock in the session time zone is outside the years 1 to 9999 of
  Python's `datetime`, and in the local time zone is not, differs too: PySpark's localization in
  the session time zone raises Python's `OverflowError` ("date value out of range"), where this
  client returns the instant in the local time zone. For example, of
  `timestamp_seconds(-62135596800)`, 0001-01-01 00:00:00 UTC, with the session time zone
  `America/Los_Angeles` on a host in UTC, this client returns `datetime(1, 1, 1, 0, 0)`. Both
  raise the `OverflowError` of a wall clock out of the range in the local time zone.
- **A `StructType` whose `fields` hold an element that is not a `StructField`** (inherent; the
  uses of the native type lower its fields into the core's struct type, which holds fields only).
  PySpark's `StructType` methods read the attributes of each element of `fields` as they go, so of
  a list that a str or another object was appended to in place, each use raises the
  `AttributeError` of the first attribute that it reads (`simpleString`, `jsonValue`, `name`,
  `dataType`, `needConversion`), or uses an object that has them, and its pickles and copies keep
  the element; `createDataFrame` of local rows takes them apart by the cached names first, so a
  row of another number of values raises `AXIS_LENGTH_MISMATCH` before that. This client lowers the
  list into the core's struct type at each use that needs the type (its JSON, simple string and
  tree string, a lookup by name, `add`, its pickles and copies, the plans of `to`, `cast`,
  `from_json` and a reader's schema, and `createDataFrame`), and raises
  `TypeError: StructType fields must be StructField objects` there, and an assignment of `fields`
  of such an element raises it at once. Of a list of `StructField`s changed in place, the cached
  `names` and their uses are PySpark's (see `docs/compatibility.md`).
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
  model dropped on a thread of no active session is not deleted.
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
- **A batch reader's option of None** (intentional; PySpark's plan of a table sets the None that
  its plan of a format drops, which makes its DataFrame fail where the same option works with
  `load`). An option of None removes the option, as PySpark's `load()` and readers of a format
  leave it out of their plan. So the DataFrames of `read.option("k", None).table(name)` and
  `changes(name)` run without it, where PySpark's plan of them sets the None, which fails their
  requests with protobuf's `TypeError: bad argument type for built-in operation`. `_options`
  holds the options of the reader of str keys and values: it shows no option of None (`{}`,
  where PySpark's shows `{'k': None}`), nor a key that is not a str (`{1: 'v'}` in PySpark), which
  fails at the same call as in PySpark (see `docs/compatibility.md`).

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
  pickles of about 12.1 KB, with the parser of deep messages (`parse_deep`, 5.2 KB) and
  `raise_error`, `column_of` of about 15.4 KB and `table_arg_of` of about 11.9 KB, of Python 3.11,
  and the pickle of an object of UDFs carries the fill of their holes too (`fill_holes`, of about
  11.1 KB), which the other pickles do not. So `cloudpickle.dumps` of `spark.range(2)` is of
  12.8 KB, where PySpark's is of 1.2 KB, of a DataFrame of a UDF 23.5 KB (2.0 KB), of a Column
  15.7 KB (0.7 KB), of a Column of a UDF 26.2 KB (1.0 KB) and of a table argument of a UDF 22.8 KB
  (1.4 KB). A plain pickle names the functions by reference, of the drop-in's module, so that of a
  Column or a table argument is of about the size of PySpark's, of 0.1 to 0.5 KB, and loads only
  where the drop-in is.

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
  a setter patched over an attribute that PySpark's Connect `__init__` sets (`_expr`, `df`),
  PySpark's objects do not construct, where the drop-in's, which no `__init__` makes, read it.

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
  (`inspect.signature(DataFrame.select)` is `(self, /, *cols)`), and its docstrings.
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

- **Garbage collection** (tracked). A DataFrame or a Column reports the functions of its UDFs that
  it alone holds to Python's garbage collector, and the plans of the Python client that it alone
  holds: the plan of a `DataFrame(plan, session)`, which its relation keeps, as PySpark's DataFrame
  keeps the plan object, also through the subqueries of the expressions of a plan, and the plan that
  an unpickled Column of the subquery of a DataFrame of a plan that did not build keeps, to build it
  at its requests. So do grouped data and their cogroups, a table argument, a window specification,
  the writers, `na`, `stat`, a merge builder and its clauses, the rows of `toLocalIterator`, and the
  expression of a Column and of a table argument (`_expr`, `_subquery_expr`) and its copies, of the
  Columns and the plans that they hold, and an `ArrayType`, a `MapType`, a `StructField` and a
  `StructType`, of the types, flags, metadata and fields that they hold, as PySpark's `__dict__`
  holds them, and the readers, the writers and the session builder, of the objects of their options
  and of the arguments that the writers keep, through one object of them that their shallow copies
  share. Those that hold their DataFrame report it, which reports the functions of the plan that
  they share (`crate::gc::Holders`), and so do the DataFrames of the plan of another,
  `DataFrame(df._plan, session)`, `copy.copy(df)` and a deep copy of nothing to copy, which keep
  `df`: the whole of `df`, its attributes, its cached schema and its `executionInfo`, where
  PySpark's keep the plan object alone. A function or a plan of the Python client that two of these
  objects share, through one Rust value of a plan or of an expression that both hold, is reported by
  none: a Column and a DataFrame of it; a Column and its expression (`_expr`), its `copy.copy` or
  its `copy.deepcopy`, which share its expression, but for the UDFs that a deep copy copies and the
  nodes above them; two DataFrames of plans of their own, such as a DataFrame and one derived from
  it; and a DataFrame and a Column of its subquery, or a table argument of it, which holds the plan
  and not the DataFrame. The holders share one reference to it, which the Rust value holds, and
  which the collector would count as that of the holder that reports it, so it could clear the
  object with that holder while another holder, still alive, reaches it. So a reference cycle that
  keeps two of its holders, such as through a bound method of an object that keeps both the Column
  and the DataFrame, or through a plan of the Python client that keeps both its DataFrame and a
  Column of its subquery, is not collected, where PySpark collects it, as each of its holders, a
  Python object, holds a reference of its own. A stream writer holds its DataFrame, where PySpark's
  holds the plan, so a DataFrame that keeps its stream writer is freed by the collector rather than
  at its last reference. The attributes of the clients of a session, their `thread_local` and their
  lists, are kept by the session as long as it lives, which no Python object reports, so a reference
  cycle through them, such as an attribute of the client or of its `thread_local` that keeps a
  DataFrame of the session, is not collected, where PySpark's client, a Python object, is.

## Session and client (Connect)

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
  it runs without PyArrow and pandas (see "pandas DataFrames without PyArrow" above), so the imports
  work without them, also of `python -c`. A program's call of `check_dependencies()` checks pandas
  and PyArrow, and the doctest run, as PySpark's, but not the gRPC packages.
- **Custom gRPC channels** (intentional; the transport is native). `ChannelBuilder` and
  `DefaultChannelBuilder` parse the connection string and hold its parameters, `host`,
  `endpoint`, `get`, `set`, `userId`, `token`, `session_id`, `use_ssl`, `secure` and
  `metadata()`, and the drop-in's `DefaultChannelBuilder` has a `port` too, which PySpark's has
  not (it keeps the port in `_port`). The transport reads the `host`, the `port` (15002 without
  one) and the `_params` of a channel builder as a `sc://` URL, so they apply. The gRPC channel
  members do not: `toChannel`, which the transport calls of a builder of its own for its errors
  only (see "A channel builder without a host"), `add_interceptor`, `setChannelOption`. The keepalive
  properties and `userAgent` read the parameters as PySpark does, but `userAgent` reports
  PySpark's `python/<version>` where the transport sends `python/rust` as the client type. The
  transport reads `SPARK_CONNECT_USER_AGENT` when a session is made of a connection string, and
  the sessions of `newSession` and `cloneSession` take the user agent of the session they are
  made of, where PySpark's requests read the variable each time, those of these sessions too:
  the core builds the requests without Python, on the threads of its runtime too, which do not
  wait for the GIL, so it reads the variable of the environment of the process when a session is
  made, rather than `os.environ` for each request.
  The Rust channel applies the keepalive from the same parameters. A `channelOptions` argument raises
  `NOT_IMPLEMENTED` instead of being ignored. A subclass that overrides `userId` or
  `token` is not read by the transport, which uses the parameters.
- **Retry and reattach** (intentional; DESIGN). Retries and reattachment live in the Rust
  transport. `client.retries` is PySpark's, vendored, and `SparkConnectClient.set_retry_policies`
  makes the transport retry with the parameters of the first `DefaultPolicy` of its policies,
  read when they are set. The transport classifies the failures as `DefaultPolicy.can_retry`
  does, so a policy of its own `can_retry` or `to_state`, and a backoff, jitter or delay that
  is not a whole number of milliseconds, raise `NOT_IMPLEMENTED`. `RpcDeadlines` sets the
  deadlines of the transport, `get_status` that of `_get_operation_statuses`. `client.reattach`
  (`ExecutePlanResponseReattachableIterator`) is not provided: `disable_reattachable_execute`
  and `enable_reattachable_execute` set the executions of the transport not reattachable or
  reattachable.
- **Errors that PySpark's client leaves to grpc** (intentional; the transport is native).
  PySpark's `SparkConnectClient.clone` and its `ArtifactManager`'s copy to the file system
  (`copy_from_local_to_fs`, and so `SparkSession.cloneSession` and `copyFromLocalToFs`) raise
  the error of the server as grpc's `RpcError` (`_InactiveRpcError`), which they do not convert.
  The drop-in converts it as the errors of its other RPCs: `clone("not-a-uuid")` raises the
  `UnknownException` of `INVALID_CLONE_SESSION_REQUEST.TARGET_SESSION_ID_FORMAT`, and a copy to
  a local path that the server refuses the `UnsupportedOperationException` of
  `_LEGACY_ERROR_TEMP_3161`. The clone asks the server for nothing more of its failure, as
  PySpark's, neither the details of the error nor the configs of its stack trace; the copy asks
  for them, and with a user agent that is too long, it raises the `SparkConnectException` of the
  request of the error's details.
- **The retry policy of the clones of a core client** (intentional; Rust API). The retry policy
  of `spark_connect_core::SparkConnectClient` is shared by its clones (`Clone`), as PySpark's
  `set_retry_policies` replaces the policies of a client that is in use: `set_retry_policy` of a
  clone changes the policy of the client it was cloned of too. A client of a new session
  (`with_new_session_id`, `clone_session`, and so `SparkSession::new_session` and
  `clone_session` of the Rust API) has the default policy, as PySpark's `newSession` and `clone`
  do, and not the policy of the client it was made of.
- **The warning of spent retries of the requests sent in the background** (intentional; the
  transport is native). A request that no caller waits for, which the transport sends from its
  own threads, does not warn `[RETRIES_EXCEEDED]` when its retries are spent, where PySpark's
  `Retrying` warns from the thread that sends it: the releases of the results of an execution
  (`ReleaseExecute`), the removal of the relation of a dropped checkpoint, the release of the
  cached objects of the native ML classes, and the interrupt of the execution of a signal. The
  threads of the transport are not threads of Python, and do not wait for the GIL.
- **The details of a failure that the client detects** (intentional; the transport is native).
  The message of a gRPC failure of no error of the server is the text of grpcio's error of the
  call, as PySpark's, whose details and `debug_error_string` are those of gRPC's C core in
  PySpark. Of a status of the server, they are the same; of a failure that the client detects
  itself, as a refused connection or a failed DNS lookup, they are tonic's message, such as
  `tcp connect error` and `UNAVAILABLE:tcp connect error`, where grpcio's are such as
  `failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:1: Failed to connect
  to remote host: Connection refused`. Of a connection whose server sends no settings by the
  connect deadline, they are grpcio's last error, `connection attempt timed out before receiving
  SETTINGS frame`, without the `failed to connect to all addresses; last error: UNKNOWN:
  <address>: ` before it.
- **The connections and the logs of the transport** (intentional; the transport is native). The
  `CONNECT` request of a proxy has only the `Host` and `Proxy-Authorization` headers, where
  grpcio's has `User-Agent: grpc-httpcli/0.0` too. The addresses of a proxy are tried in turn
  within the one connect deadline of 20 s, so that one that does not answer holds the next ones,
  where grpcio starts the next one 250 ms after the previous, each with a deadline of its own. A
  TLS channel trusts the system's store, where grpcio's trusts its own bundle, so that a system
  without roots fails every TLS session when it is made (`Failed to configure TLS`), where
  PySpark's connects (see `docs/configuration.md`). With `SPARK_CONNECT_LOG_LEVEL=debug`,
  PySpark's client also logs its `RpcDeadlines` when it is made (`Spark Connect RPC deadlines:
  ...`), the failure that ends its retries (`Given up on retrying. error: ...`) and each gRPC
  failure that it converts (`GRPC Error received`), which the transport, whose retries and errors
  are its own, does not.
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
  A `KeyboardInterrupt` that a signal handler raises while the transport reads them at the end of
  a call that ran during another is the error of that call, whose result is dropped. A finalizer
  whose last reference a thread of the transport drops, not a thread of Python, sends the context
  of the last call of that thread, else of the session, where PySpark's finalizers run on a thread
  of Python. A request of more than one value that protobuf refuses raises the error of the
  session ID, of the user and of the extensions before that of a tag, where PySpark's execution
  raises that of the tag before those of the user and of the extensions; and the request of the
  details of the failure of a call whose requests carry no extensions (of `ReleaseSession`,
  `GetStatus` and the artifacts) carries the extensions of the last call when one of them is not
  an `Any`, where PySpark's `_fetch_enriched_error` raises the `TypeError` of protobuf in place of
  the error of the server. An extension that is not an `Any` raises upb's `TypeError`, which
  names the runtime's class of `Any` (`pyspark.sql.connect.proto._rust.any_pb2.Any`) where
  PySpark's names `google.protobuf.any_pb2.Any`. A failed request leaves `inside_error_handling`
  False in the `thread_local`, as PySpark's `_handle_error` does, which the drop-in does not read:
  PySpark's raises the error of a request as grpc gives it while it is True. A `copy.copy` of a
  client sends its own attributes, as PySpark's, also at once with its client on another thread,
  and a call of the session that runs during a call of the copy, as of a progress handler or a
  finalizer, sends the session's, after which the call of the copy sends its own again, as they
  are then; but it shares with its session whether it is closed (`_closed`), the server-side
  session ID that the responses carried, whether its executions are reattachable and the retry
  policies, which PySpark's copy has of its own.
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
  encoding of their fields (`plan::resolve_own_fields`). A placeholder whose `common` a plan changes
  (`plan.common.plan_id = ...`) is sent with that `common` merged, as PySpark's changed relation.
- **A channel builder without a host** (intentional; the transport is native). The builders of
  `pyspark.sql.SparkSession` and `pyspark.sql.connect.session.SparkSession` are PySpark's two
  builders, with its errors: `CONNECT_URL_NOT_SET`, a `PySparkRuntimeError`, and
  `SESSION_NEED_CONN_STR_OR_BUILDER`, a `PySparkValueError`, also for both a URL and a channel
  builder. `channelBuilder(...)` takes any object, and the session fails when it is made, of the
  errors of PySpark's client: the transport connects to the `sc://` URL of the `host`, the `port`
  and the `_params` of the builder, and calls the `toChannel` of a `ChannelBuilder` that is not a
  `DefaultChannelBuilder` for its errors only, so a builder of its own `toChannel` and no `host`
  raises the error of the `host` of `ChannelBuilder`, where PySpark's connects through its channel.
- **Plan logging of a deep DataFrame** (intentional; DESIGN). `DataFrame._plan` is a
  placeholder over the relation of the Rust core, whose `to_proto(client)` is the `Plan` of the
  placeholder, `plan(client)`, as the root. The `to_proto` of a `LogicalPlan` is the `Plan` of
  its relation as it is, as the transport compresses the plans that it sends, where PySpark's
  compresses one of over `spark.connect.session.planCompression.threshold` bytes (10 MiB, which
  its client asks the server for at its first plan) into a `compressed_operation`. Python
  protobuf refuses to decode a relation nested beyond its limit (`too many levels of nesting`),
  and the drop-in never decodes a deep plan. `SparkConnectClient._proto_to_string` renders a
  message of the drop-in's protobuf runtime in Rust at any depth: past about 125 levels, PySpark's
  `text_format` raises its `RecursionError`, and it returns `<Truncated message due to recursion
  error>`, where the drop-in returns the full text; with `truncate=True`, both cut it at 8
  levels. Plan logging of a deep DataFrame therefore does not reach the recursion handling that
  the test `test_recursion_handling_for_plan_logging` exercises.
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
  `SparkSession.__del__` closes the client of a session object that is freed, and the exit hook
  of a closed client sends nothing; the drop-in closes a session once the program dropped its
  objects and the objects that hold it, also when a `conf` or a `client` of it is kept, so it
  sends nothing at exit either. But once its `catalog` or `streams` is read, PySpark's session
  object is in a reference cycle with the `Catalog` of its `functools.cached_property` or the
  `StreamingQueryManager` of its `_sqm`, each of which holds the session, so that a drop does not
  free it: it stays open until a collection of the garbage collector frees it, which the
  allocations of the program decide, and an exit before that sends its `clean_cache` (and its
  release, with `SPARK_CONNECT_RELEASE_SESSION_ON_EXIT`). The drop-in's session keeps its catalog
  and its streams without a cycle, so it is freed when the program drops it, as PySpark's is after
  a collection (`gc.collect()`), and its exit sends nothing, whatever the timing of the
  collections. The same goes for the other cycles that keep a PySpark session until a collection,
  such as that of an error that the `_handle_error` of its client raises again (`raise error`),
  whose traceback keeps the frame that holds the error and, with it, the frames of the calls that
  led to it and their variables. PySpark 4.3.0 then does not exit at all, as it does not while a
  session other than its default one is left open to the end (of `newSession()`, or
  `SparkSession(connection=...)`): its `SparkSession.__del__` closes the client in the teardown of
  the interpreter, whose grpcio `Channel.close()` waits forever. That is a bug of PySpark, which
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
  readers, writers, builders and the other objects of a session or a DataFrame hold their
  state in the Rust core: their `__dict__` holds the attributes that a user sets on them,
  which their copies and pickles keep as PySpark's do, but not their state, which PySpark's
  holds there (`vars(spark.streams)` is empty, where PySpark's has `_session` and `_sqlb`). An
  `Observation` is not one of them: its `_name`, `_result` and `_error` are of its `__dict__`, as
  PySpark's.
- **Subclasses** (intentional; the state is native). PySpark's classes are plain classes that a
  user subclasses. The drop-in's DataFrame, `Observation`, the `SparkSession`, `Catalog`,
  `DataSourceRegistration`, `DataFrameReader` and `DataStreamReader` of `pyspark.sql.connect`,
  `SparkSession.Hook`, `StorageLevel`, `PythonEvalType` and the data types are subclassed as
  PySpark's: a subclass constructs of the arguments of its `__init__`, and its copies and pickles
  are of the subclass
  (but those of a session, which are the active session of its ID, as PySpark's), and PySpark's
  pickle of a reader, a catalog or a registration, which makes it of its class without
  arguments, loads in the drop-in. The `Window` of `pyspark.sql.window` and of
  `pyspark.sql.connect.window`, and the classic `Column`, `DataFrameNaFunctions`,
  `DataFrameStatFunctions`, `WindowSpec` and `TableArg`, are PySpark's classes, which construct
  and subclass as PySpark's. The session, the readers and the catalog are of the native
  state of a session, which the `__init__` of the class, of PySpark's arguments, gives them: one
  whose `__init__` never ran, as of a subclass whose `__init__` does not call it, is of the
  session of no server, whose requests raise `NO_ACTIVE_SESSION` and whose `is_stopped` is True,
  where PySpark's raise the `AttributeError` of their `_client` or `_sparkSession` (a reader or a
  catalog is of the first session among the arguments of the subclass, if any, which then
  works). The `__init__` of a reader, a catalog or a registration takes a session, and raises the
  `TypeError` of another object, where PySpark's keeps any object until it is used.
  The other native classes are not subclassable (`TypeError: type '...' is not an acceptable
  base type`), and do not construct (`cannot create '...' instances`): their objects are of the
  native state of a plan, an expression or a query, which the arguments of PySpark's
  constructors, the Python objects of its client, do not give, as the drop-in's native state
  replaces those objects, so a subclass would have no `__init__` to call (see
  `docs/compatibility.md`):
  - `Column`: an `Expression` of `pyspark.sql.connect.expressions`;
  - `DataFrameWriter`, `DataFrameWriterV2`, `DataStreamWriter`, `MergeIntoWriter`: the
    `LogicalPlan` of a DataFrame;
  - `GroupedData`, `PandasCogroupedOps`: a DataFrame and the expressions of its grouping;
  - `DataFrameNaFunctions`, `DataFrameStatFunctions`: a DataFrame of a native plan;
  - `WindowSpec`, `TableArg`: expressions and a `WindowFrame`, a `SubqueryExpression`;
  - `RuntimeConf`: a `SparkConnectClient`;
  - `StreamingQueryManager`, `StreamingQuery`: a session and the IDs of a query, of the
    listener bus and the queries of the Rust core;
  - `TableValuedFunction`: a session, of the native core.

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
    surrogate raises its `UnicodeEncodeError`, and so does `DataFrame.to` of such a struct of a
    type of no proto, at the call;
  - `createDataFrame` of data sends the JSON of the core types of its schema, where PySpark sends
    the `json()` of the object, which the server may fail to parse.

  The class's own types compare without the attributes that a user set on them, where PySpark's
  comparison of the `__dict__`s holds them (see `docs/compatibility.md`); their copies and pickles
  keep them, as PySpark's.
- **The functions of `pyspark.sql.connect.functions`** (intentional; the drop-in's, of the Rust
  core). `pyspark.sql.functions` is PySpark's package, as it is, whose functions run those of
  `pyspark.sql.connect.functions` of their names, as with `pyspark-client`. Those are the
  drop-in's, of the docstrings, the annotations and the module of PySpark Connect's, which call
  the `lit`, `column` and `_to_col` of their module as PySpark Connect's do (see
  `docs/compatibility.md`), but: a traceback of a generated one has the frames of
  `pyspark/sql/connect/functions/_generated.py`, where PySpark Connect's are of its `builtin.py`;
  `col` has no `__wrapped__` of PySpark's `_with_origin` (the Rust core captures the call site);
  `pyspark.sql.connect.functions.builtin` has none of the expression classes of PySpark Connect
  that its module imports (`CaseWhen`, `ColumnReference`, ...), nor `np`, and its private helpers
  are its own; and the annotations of `schema_of_csv`, `schema_of_json` and `schema_of_xml` are
  those of PySpark's classic functions, `Union[Column, str]`, where PySpark Connect's write
  `Union[str, Column]`. The annotations, of `pyspark.sql.functions` too, name the class of
  `pyspark.sql.column.Column`, which is PySpark Connect's class in the drop-in (see
  `docs/compatibility.md`), where `pyspark-client`'s name its classic class.

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
  `pyspark-client` see it, the type variables of the annotations of `pyspark.sql.types`, and the
  base and the internal class of PySpark Connect's classes that the drop-in implements natively,
  the `OptionUtils` of its readers and writers and its `StreamingQueryListenerBus`.
- **`SparkSession.client`** (intentional; the drop-in's own class). The client has PySpark
  Connect's public API, of the class `pyspark.sql.connect.client.SparkConnectClient` rather than
  `pyspark.sql.connect.client.core`'s.
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
  `ArrayType`, `StructField`, ...) and of the classes of `pyspark.sql.connect` (`Catalog`,
  `RuntimeConf`, `Observation`, ...), is a slot of CPython, whose wrapper has CPython's docstring
  (`Initialize self.  See help(type(self)) for accurate signature.`), which the class cannot set.
  PySpark's is that of its Python function: None, or of a docstring that PySpark Connect's modules
  copy (`__init__.__doc__ = PySparkRuntimeConfig.__init__.__doc__`). The other docstrings of the
  modules of `pyspark.sql` and `pyspark.errors` are PySpark's (see `docs/compatibility.md`).
- **The docstrings of the native Column, WindowSpec, TableArg, `DataFrameNaFunctions` and
  `DataFrameStatFunctions` of Spark Connect** (inherent; a virtual subclass inherits no docstring).
  PySpark Connect's classes are subclasses of the classic ones, and neither they nor their members
  have docstrings of their own, so `inspect.getdoc` and `help()` show those of the classic class.
  The native classes are virtual subclasses of the classic ones, which are not in their MRO: they
  and their members carry the classic docstrings, so that `inspect.getdoc` and `help()` show
  PySpark's text, as `pyspark-client`'s inherited ones do. Their `__doc__` is that text, where
  PySpark Connect's own is None. So is that of `Read.print` of `pyspark.sql.connect.plan`, of
  PySpark Connect's `LogicalPlan.print`, which the drop-in's `LogicalPlan` does not have.

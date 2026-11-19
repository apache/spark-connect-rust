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

- **What a plan build does for a pickle or a deep copy** (intentional; DESIGN). A pickled
  DataFrame, Column or table argument, and a deep copy of a DataFrame that is copied into another
  session, carry the relation that the drop-in builds for them, where PySpark's keep their plan
  objects and build them at each request. (A deep copy of a DataFrame in its session is a copy of
  its plan, of copies of the functions of its UDFs and of its Observations, which its requests
  build, as PySpark's; see "A deep copy of a DataFrame into another session".)
  So the requests of an unpickled DataFrame, Column or table argument, or of such a deep copy, do
  not warn of a window of no partitions (`WARN WindowExpression: No Partition Defined ...`),
  which PySpark's do at each build, and the `and` of the conditions of a join of a list of
  Columns keeps the origin of its call site at the pickle or the copy, where PySpark's has
  the call site of each request. The pickle itself does not warn, as PySpark's does not, but
  PySpark's unpickle of a DataFrame warns once, as it builds the plan to read the columns of the
  new DataFrame, where the drop-in's does not. The `plan` of `DataFrame._plan` builds the relation,
  and warns, as PySpark's does; a `DataFrame(plan, session)` of a plan over it warns once, when it
  is made, as the core keeps the relation that it built then, where PySpark's warns at each
  request, which builds the plan. The `UserWarning` is of `pyspark.sql.connect.expressions` and of
  its line 1126, as PySpark's, in the file of the drop-in's module, which ends before the line, so
  it shows no source line.

- **A DataFrame of a plan that fails to build** (tracked). `DataFrame(plan, session)`, of the
  class or of a subclass, builds the relation of `plan.plan(session.client)` when it is made, so a
  plan that fails, such as the `_plan` of a DataFrame of a UDF that does not pickle or of a read of
  an option that UTF-8 cannot encode, raises its error there, where PySpark's DataFrame keeps the
  plan and raises the error at its first request.

- **Public base and native MRO** (intentional; a PyO3 class cannot have a Python class as its
  base, so the public base can only be a virtual one). `pyspark.sql.DataFrame` is implemented as a
  Python base/factory, allowing subclasses such as
  `pyspark.testing.mlutils.MockDataset` to allocate their own Python state with
  `object.__new__`. The concrete class remains
  `pyspark.sql.connect.dataframe.DataFrame`, implemented by the Rust extension. It is
  registered as a virtual subclass of the public base: `isinstance` and `issubclass`
  recognize it, while its actual MRO stays native. Consequently, `super` and MRO
  introspection do not treat a native frame as inheriting from the public base.

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
  not establish portability of instances, explicit user metaclasses or server UDFs.

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
`docs/compatibility.md`). These cases still differ, all of copies and pickles:

- **A pickled Column or DataFrame** (intentional). It carries the commands that its functions
  pickled to when it was pickled, in the protobuf of its expression or relation. PySpark's carries
  the functions, which its requests pickle again: a value that a function reads, changed after the
  pickle, is seen by PySpark's unpickled Column of a function that `pickle` pickles by reference,
  such as a module-level function of `__main__`, but not by this client's. So `pickle.dumps` of a
  Column of a UDF of a lambda pickles, where PySpark's raises the `PicklingError` of the lambda,
  and that of a function that does not pickle by value raises its `PicklingError`, where
  PySpark's pickles a module-level function by reference. Nothing requires the commands: the
  server's Python worker could rebuild PySpark's UDFs of the functions, as it rebuilds PySpark's
  Column of the expression. Carrying the functions, which would close these cases, is the
  trade-off: a plain pickle of a Column of a UDF of a lambda, which pickles here, would then fail
  as PySpark's does, and the pickles of every holder of a plan would change.
- **A deep copy of a DataFrame into another session** (tracked). A deep copy whose session copies
  to another one, of a `memo` that maps it or of another active session of its ID, is a DataFrame
  of the relation of the copied one, so that its Observations are copies, as PySpark's, and so it
  carries the commands that the functions pickled to when it was copied, where PySpark's holds the
  copies of the functions, which its requests pickle again. In the session of the DataFrame, the
  copies of a DataFrame, a Column, grouped data and a table argument hold the functions as
  PySpark's do: `copy.copy` shares them, and `copy.deepcopy` deep-copies each with the `memo` of
  the call, which gives a function itself, and a copy of a callable object, a bound method or a
  `functools.partial`.
- **Garbage collection** (tracked). A DataFrame or a Column reports the functions of its UDFs that
  it alone holds to Python's garbage collector, and so do grouped data and their cogroups, a table
  argument, a window specification, the writers, `na`, `stat`, a merge builder and its clauses, the
  rows of `toLocalIterator`, and the expression of a Column and of a table argument (`_expr`,
  `_subquery_expr`) and its copies, of the Columns and the plans that they hold, and an `ArrayType`,
  a `MapType`, a `StructField` and a `StructType`, of the types, flags, metadata and fields that
  they hold, as PySpark's `__dict__` holds them. Those that hold their DataFrame report it, which
  reports the functions of the plan that they share (`crate::gc::Holders`). A function that a Column
  and a DataFrame of it share, or two DataFrames, or a DataFrame and a table argument of it, which
  holds the plan and not the DataFrame, is reported by none: the holders share one reference to it,
  which the Rust value of the plan or of the expression holds, and which the collector would count
  as that of the holder that reports it, so it could clear the function with that holder while
  another holder, still alive, reaches it. So a reference cycle through it, such as through a bound
  method of an object that keeps both the Column and the DataFrame, is not collected, where PySpark
  collects it, as each of its holders, a Python object, holds a reference of its own. A stream
  writer holds its DataFrame, where PySpark's holds the plan, so a DataFrame that keeps its stream
  writer is freed by the collector rather than at its last reference.

## Session and client (Connect)

- **Custom gRPC channels** (intentional; the transport is native). `ChannelBuilder` and
  `DefaultChannelBuilder` parse the connection string and hold its parameters, `host`,
  `endpoint`, `get`, `set`, `userId`, `token`, `session_id`, `use_ssl`, `secure` and
  `metadata()`, and the drop-in's `DefaultChannelBuilder` has a `port` too, which PySpark's has
  not (it keeps the port in `_port`). The transport reads the `host`, the `port` (15002 without
  one) and the `_params` of a channel builder as a `sc://` URL, so they apply. The gRPC channel
  members do not: `toChannel`, `add_interceptor`, `setChannelOption`. The keepalive
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
  `_LEGACY_ERROR_TEMP_3161`. With a user agent that is too long, they raise the
  `SparkConnectException` of the request of the error's details.
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
- **`SparkConnectClient.thread_local`** (intentional). A subclass of `threading.local`, of
  subclasses of `set` and `list` for its tags and its user context extensions, which keep the Rust
  transport's lists of them in step, where PySpark's is a `threading.local` of a plain set and
  list. The global list of the extensions is a subclass of `list` too. A set of tags refuses an
  object that is neither a str nor bytes when it is added, where PySpark's raises the `TypeError`
  of the field of the tags when it builds a request. A list that the attribute of a `copy.copy` of
  a client is set to is not one that the requests of the session carry.
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
  ML commands, and anywhere in the message of the extension of a relation or of a command (a
  plugin's, as Delta Lake's `merge`) that holds it as `plan(session)` gives it. The core does not
  decode the expressions of such a plan, which hold no relation, so as not to decode every
  expression of each plan for what only a plugin's expression extension could hold, and it takes a
  placeholder in an extension's message by its encoding, which a change alters: so a placeholder in
  an expression's extension is sent as it is, and so is one that the message of an extension changes
  (`plan.common.plan_id = ...`), whose change the core does not apply to the relation; the server
  rejects them with `CONNECT_INVALID_PLAN.NO_HANDLER_FOR_EXTENSION`.
- **A channel builder without a host** (intentional; the transport is native). The builders of
  `pyspark.sql.SparkSession` and `pyspark.sql.connect.session.SparkSession` are PySpark's two
  builders, with its errors: `CONNECT_URL_NOT_SET`, a `PySparkRuntimeError`, and
  `SESSION_NEED_CONN_STR_OR_BUILDER`, a `PySparkValueError`, also for both a URL and a channel
  builder. `channelBuilder(...)` rejects a channel builder without a `host` with a `ValueError`,
  as the transport connects to the `sc://` URL of its `host`, `port` and `_params`; PySpark's
  takes any object, and its session fails when it is made, of the error of the object (the
  `TypeError` of a str's slicing, or the error of its `toChannel`).
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
  the requests carry, which a set changes for them, as PySpark's, so the next response of
  another server-side session closes the session; `_server_session_id` is exposed and settable,
  as PySpark's, and a changed value closes the session. A value that is not a str raises the
  `TypeError` of the string field of the session ID of the requests when it is set, where
  PySpark raises it when it builds the next request.
- **`_closed`** (intentional). Reading `_closed` gives the session's stopped state. Setting it
  to True closes the session, and setting it to False opens it again, as PySpark's. After
  `close()` or `stop()`, which close PySpark's channel too, PySpark's requests still fail once
  `_closed` is False, with grpc's `ValueError: Cannot invoke RPC: Channel closed!`, where the
  drop-in's are sent again.
- **`_user_id`** (intentional). It is read-only, where PySpark's is an attribute that a set
  changes for the requests.
- **The exit of a dropped session in a reference cycle** (intentional). PySpark's
  `SparkSession.__del__` closes the client of a session object that is freed, and the exit hook
  of a closed client sends nothing; the drop-in's exit hook holds its session weakly, so a
  session that the program dropped, of which no object is left, sends nothing at exit either.
  But once its `catalog` or `streams` is read, PySpark's session object is in a reference cycle
  with the `Catalog` of its `functools.cached_property` or the `StreamingQueryManager` of its
  `_sqm`, each of which holds the session, so that a drop does not free it: it stays open until
  a collection of the garbage collector frees it, and an exit before that sends its
  `clean_cache` (and its release, with `SPARK_CONNECT_RELEASE_SESSION_ON_EXIT`), which waits on
  the retries when the server is gone. The drop-in's session keeps its catalog and its streams
  without a cycle, so it is freed when the program drops it, and its exit sends nothing,
  whatever the timing of the collections. The same goes for the other cycles that keep a
  PySpark session until a collection, such as that of an error that the `_handle_error` of its
  client raises again (`raise error`), whose traceback keeps the frame that holds the error and,
  with it, the frames of the calls that led to it and their variables. And PySpark 4.3.0 does not
  exit at all while a session other than its default one is left open to the end (of
  `newSession()`, or `SparkSession(connection=...)`), as its `SparkSession.__del__` closes the
  client in the teardown of the interpreter, whose grpcio `Channel.close()` waits forever; the
  drop-in exits.
- **A `conf` or a `client` kept apart from its session** (tracked). PySpark's `RuntimeConf` and
  the `SparkConnectClient` of a session hold the client, not the session object: once the
  program drops the session, its `__del__` closes the client, so the `conf` or `client` that the
  program kept raises `NO_ACTIVE_SESSION`, and the exit sends nothing for the session. The
  drop-in's hold the session, which stays open while either is alive: their requests work, and
  the exit cleans its ML cache. A client that no session object owns, of
  `SparkConnectClient(...)` or of the `newSession()` or `clone()` of a client, lives until the
  exit in both, as the `atexit` registration of PySpark's keeps it.
- **The `__dict__` of the objects of a session** (intentional; the state is native). The
  readers, writers, builders and the other objects of a session or a DataFrame hold their
  state in the Rust core: their `__dict__` holds the attributes that a user sets on them,
  which their copies and pickles keep as PySpark's do, but not their state, which PySpark's
  holds there (`vars(spark.streams)` is empty, where PySpark's has `_session` and `_sqlb`).
- **Subclasses** (intentional; the state is native). PySpark's classes are plain classes that a
  user subclasses. The drop-in's DataFrame, `Observation`, `SparkSession`, `Catalog`,
  `DataSourceRegistration`, `DataFrameReader`, `DataStreamReader`, `SparkSession.Hook`,
  `StorageLevel`, `PythonEvalType` and the data types are subclassed as PySpark's: a subclass
  constructs of the arguments of its `__init__`, and its copies and pickles are of the subclass
  (but those of a session, which are the active session of its ID, as PySpark's), and PySpark's
  pickle of a reader, a catalog or a registration, which makes it of its class without
  arguments, loads in the drop-in. The `Window` of `pyspark.sql.window` and of
  `pyspark.sql.connect.window` is subclassed, of its static methods, and it and a subclass make
  objects of no state, as PySpark's. The session, the readers and the catalog are of the native
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
- **Metadata of the fields of a type** (tracked; the core holds JSON values). A field compares its
  metadata dict as PySpark's, of the values as they were given, writes it with `json.dumps`, and
  copies and pickles it as it is, but a plan and `createDataFrame` of data send the core's JSON
  values of it, of an int of more than 64 bits as a float. That float is the nearer value, as the
  server keeps only the low 64 bits of PySpark's exact int (`10**30` becomes
  `5076944270305263616`), but an Arrow-optimized UDF of a return type of such metadata fails with
  `ARROW_TYPE_MISMATCH`, as the plan's type of the float is not that of the worker, of the exact
  int, where PySpark's runs (and its `pandas_udf` fails too). A float that is not finite (`inf`,
  `nan`) raises serde_json's `ValueError` when the field is made, where PySpark makes the field,
  whose every request the server refuses, so only its local uses differ (intentional).
- **The `createDataFrame` of a type of a `UserDefinedType` of no JSON** (tracked; `createDataFrame`
  lowers its schema first). An array, a map, a field or a struct of a `UserDefinedType` whose
  `module()` or `sqlType()` is not implemented is made and read as PySpark's, and the uses that read
  its JSON raise its error, but `createDataFrame` of local rows or of a pandas DataFrame of one
  raises the `NOT_IMPLEMENTED` of its `jsonValue()`, where PySpark's converters raise that of its
  `sqlType()` or `serialize` (see `docs/compatibility.md`).
- **Direct construction** (intentional). `SparkSession(connection=..., userId=...,
  hook_factories=...)` makes a session that is neither default nor active, as PySpark's
  constructor does. `userId` is added to the URL unless the URL has a `user_id`, which
  takes precedence, as PySpark's does.

## `pyspark.sql.connect.proto` and plugins

- **Generated modules of a plugin's protos** (intentional; DESIGN). A module that protoc
  generates of a proto3 file that imports the protos of Spark Connect, directly or through the
  files it imports (Delta Lake's `delta/connect/relations.proto` and `commands.proto`), gets the
  classes of the drop-in's Rust protobuf runtime when `protobuf`'s builder builds it, as it builds
  the modules of protoc 3.20 and later, so its message fields take the messages of
  `pyspark.sql.connect.proto` as they are, as PySpark's plugins take PySpark's, and the server
  receives the plans of Delta Lake 4.0's `delete`, `update` and `merge` as PySpark sends them: the
  core replaces the placeholder that holds the relation of a DataFrame, which
  `DataFrame._plan.plan(session)` gives, in the message of an extension too. What stays different
  (see `docs/compatibility.md`):
  - the core replaces that placeholder where it replaces the plan of a DataFrame (see "Plans of a
    DataFrame's relation" above). One that a plugin changes in the message of an extension reaches
    the server as a placeholder, which it refuses, where PySpark's plugins send the relation: the
    core takes a placeholder there by its encoding, which the change alters;
  - its classes are the runtime's, not subclasses of `google.protobuf.message.Message`, while its
    `DESCRIPTOR`, its `_*` descriptors, its extensions and services stay `protobuf`'s, and
    `protobuf`'s `message_factory` and `symbol_database` give `protobuf`'s classes of its types,
    whose messages equal (`==`) the runtime's of the same contents, as upb's of its one class of a
    type do;
  - a file that it imports that does not import the protos of Spark Connect (Delta Lake's
    `delta/connect/base.proto`) keeps `protobuf`'s classes, as the drop-in cannot know, when
    `protobuf`'s builder builds it, that a plugin's file will import it later, and giving the
    runtime's classes to every proto3 module would change unrelated libraries. A field of the
    runtime of one of its types takes `protobuf`'s messages of the type, and gives a message of the
    runtime's class of the type, not of the module's class, which upb gives, so `isinstance` with
    the module's class is false, the message pickles only within the message that holds it, and the
    module's class takes it only through bytes or dicts (`CopyFrom` raises `TypeError`), though the
    two are equal of the same contents;
  - a file that is not proto3, or that imports one or a well-known type other than `Any` and
    `Timestamp`, keeps `protobuf`'s classes, which take the package's messages, and those of a
    module of a file that it imports that has the runtime's classes (a proto3 file of the plugin
    that imports the protos of Spark Connect), only through bytes or dicts, where upb's take them
    as they are. So does a module of protoc before 3.20, which makes its classes with
    `reflection.GeneratedProtocolMessageType` rather than with `protobuf`'s builder, and a module
    that builds before `pyspark.sql.connect.proto` is imported, as only one that imports another
    copy of Spark's generated modules can.

## Type checking (the stubs of `scripts/gen_type_stubs.py`)

mypy and pyright type an application as with `pyspark-client` (see `docs/compatibility.md`), but:

- **Imports of PySpark's modules** (intentional). A name that a module of PySpark imports, such as
  `DataFrame` of `pyspark.sql.catalog`, is an attribute of the module, which mypy takes by default
  with `pyspark-client`. The stubs declare it only when the drop-in's module has it at run time, so
  that it is an error, as `from pyspark.sql.catalog import DataFrame` fails with the drop-in.
  pyright reports it as a private import with `pyspark-client` too.
- **Names that only PySpark's annotations need** (intentional). `TYPE_ONLY` of
  `python/tests/test_type_stubs_offline.py` lists the names that the stubs declare and the drop-in
  does not have, with their reasons: the classic-only `DataFrame.rdd`, as type checkers of
  `pyspark-client` see it, and the bases, metaclass, decorators and internal classes of PySpark's
  classes that the drop-in implements natively.
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
- **Modules of messages** (intentional). pyright warns that `pyspark.sql.connect.proto.base_pb2`
  and the other modules of messages, which the drop-in's protobuf runtime makes at run time, have
  a stub and no source (`reportMissingModuleSource`), as it does of
  `pyspark.sql.functions.builtin`.

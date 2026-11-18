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
- **pandas DataFrames without PyArrow** (intentional optional-dependency fallback). As in
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
  truncated without the error of a safe cast, and its errors are its own. This fallback covers
  unavailable modules detected by `pyarrow_available`, not an installed PyArrow module that
  fails during import. The row path also converts a frame, with PyArrow, when the type names
  that the conversion and the importer resolve are rebound, or the classes of the struct of
  the schema customized (`accepts_pandas_schema`).
- **Nested pandas Decimal inference and failures** (intentional; related change:
  SPARK-60696).

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
  SPARK-60696). On the native row path, without PyArrow, Decimal exactness is checked before
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
  `CANNOT_INFER_TYPE_FOR_FIELD`, while the native row path of pandas input, without PyArrow,
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

- **A session time zone that the server takes and Python's zone databases do not** (intentional;
  the values are those of any other zone), such as `Z`, `PST`, `GMT+08:00` or `UTC+01:00`.
  PySpark's `collect`, `toLocalIterator` and `first` of a timestamp localize the values in the
  session time zone, looked up by its name in Python, so they raise pytz's
  `UnknownTimeZoneError`, or, without pytz, PyArrow's `ArrowInvalid` ("The zoneinfo module or
  pytz package must be installed."). This client's conversion of the rows reads no zone of the
  session: it returns the instant in the local time zone of the client, as both clients do of a
  zone that Python knows. `toArrow` and `toPandas` give the same values in both.
- **The `__repr__` of a Column of a str that UTF-8 cannot encode** (intentional; the expressions
  of the core hold UTF-8). Such a str, of a name, a SQL string or a literal, fails the plans of the
  Column when they are built, as in PySpark, and its lone surrogates show escaped in the `__repr__`
  (`\ud800`), where PySpark's shows the str as it is (see `docs/compatibility.md`).
- **A `StructField` of a name or a metadata str that UTF-8 cannot encode** (intentional; the types
  of the core hold their names and metadata as UTF-8 and JSON). Its constructor raises the
  `UnicodeEncodeError` of the str, of a lone surrogate, where PySpark's builds the field, whose
  `json()`, `simpleString()`, `==` and pickle work, and whose uses fail later: the DataFrames of
  `to`, `createDataFrame`, `from_json`, a cast or a UDF of such a type are made, and fail at the
  first request that sends the type, with protobuf's `UnicodeEncodeError` of a name, or with the
  server's error of the string of a lone surrogate (`Unpaired surrogate`) of the JSON in which
  `json.dumps` escapes it.
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
- **`schema` and `columns` of a join on a tuple** (intentional). As in PySpark, the plan fails with
  the `AttributeError` of `to_plan` of the tuple when it is built, and `schema`, a property, then
  gives a Column of the `__getattr__` of the DataFrame: of `schema`, where PySpark's is of
  `_schema`; `columns` gives the Column of `columns`, where PySpark's raises `NOT_ITERABLE`.
- **An exception of the `&` of a join's list of conditions that is not of a class that the plan
  carries** (intentional; a plan of the core carries an error, not a Python object). Python's
  `AttributeError`, `TypeError`, `ValueError` and `OverflowError`, and PySpark's error classes,
  such as the `PySparkAttributeError` of a Row, fail the plan when it is built, as in PySpark.
  Another, such as a `KeyError` that the `__getattr__` of an operand raises, is raised by `join`
  or `_joinAsOf`, where PySpark raises it when it builds the plan.
- **`toArrow` and `toPandas` of a string that is not valid UTF-8** (intentional). PySpark's
  `toArrow` returns a table whose string column holds the invalid bytes, which pyarrow's
  full validation rejects, and its `toPandas` raises pyarrow's `ArrowException`. This
  client does not return invalid Arrow data: both raise a `PySparkRuntimeError` of the
  Arrow decode. `collect` raises PySpark's `UnicodeDecodeError` for the bytes of the first
  invalid string (the Rust API's `collect` returns `SparkErrorKind::UnicodeDecode`).

- **What a plan build does for a pickle or a deep copy** (intentional; DESIGN). A pickled
  DataFrame or Column, and a deep copy of a DataFrame whose plan holds no function of a UDF or
  that is copied into another session, carry the relation that the drop-in builds for them,
  where PySpark's keep their plan objects and build them at each request. (A deep copy of a
  DataFrame of such functions in its session is a copy of its plan, of copies of them, which
  its requests build, as PySpark's; see "A deep copy of a DataFrame into another session".)
  So the requests of an unpickled DataFrame or Column, or of such a deep copy, do not warn of a
  window of no partitions (`WARN WindowExpression: No Partition Defined ...`), which
  PySpark's do at each build, and the `and` of the conditions of a join of a list of
  Columns keeps the origin of its call site at the pickle or the copy, where PySpark's has
  the call site of each request. The pickle itself does not warn, as PySpark's does not.
  The `plan` of `DataFrame._plan` is a placeholder of the relation (see "Plan logging of a
  deep DataFrame" below), so it does not warn, and a request of a plan over it warns once,
  where PySpark's warns at both. The `UserWarning` is of `pyspark.sql.connect.expressions`,
  as PySpark's, but of a line of the drop-in's module, where PySpark's is of line 1126 of its
  own.

- **Public base and native MRO.** `pyspark.sql.DataFrame` is implemented as a
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

- **Public class mutation and dispatch.** Public-base method and property calls
  look up the concrete implementation at call time, including concrete-class
  patches made after import. Assigning or deleting user extensions directly on the
  public class also updates the native class, so added helpers and overridden
  methods reach existing native frames;
  assignments on Python subclasses remain local to those subclasses.
  Deleting an initial public method or property removes its public-base exposure
  and restores the concrete implementation that preceded a public override, while
  preserving any later independent concrete patch. Deleting an added
  helper removes the forwarded helper; deleting a temporary extension that shadows
  a native-only attribute restores that native attribute.
  Class layout, constructor, import/ABC metadata and instance attribute protocols
  `__getattribute__`, `__setattr__` and `__delattr__` stay on the public base. Those
  protocols follow ordinary Python inheritance on independent subclasses; virtual
  native membership does not inherit them.

- **Attribute resolution.** Explicit public `__getattr__` calls use the native
  column resolver, including existing attribute names, unsupported JVM names, and
  dunder/legacy column validation. While a user override of public `__getattr__`
  is active, native frames use generic Python attribute lookup before calling that
  hook. Undo restores the saved native lookup entry and prior hook state, including
  absence of a native `__getattr__` entry. Importing the public base does not alter
  the native lookup entries. The native-only private method
  `_get_column_by_attribute` bridges explicit public lookup to the Rust resolver;
  it is an intentional extra name in concrete-class introspection and completion.

- **Independent Python subclasses and construction.** Python subclasses override
  operations for their own data model. As in PySpark, inherited public properties
  read `None`, while native-dispatched methods,
  including `repr`, `dir`, and missing-attribute `__getattr__`, raise
  `NOT_IMPLEMENTED` on independent Python data. Consequently, `hasattr` for an
  unknown name also raises that exception, including on `MockDataset`.
  The official undecorated empty methods `checkpoint`, `localCheckpoint`,
  `lateralJoin`, `nearestByJoin`, `toPandas`, `asTable`, `scalar` and `exists` return
  `None` after argument binding. The inherited `__arrow_c_stream__` uses the
  generic `SparkArrowCStreamer` helper with the independent frame and requested
  schema, as on the official public base.
  Internal `_session` and `_plan` remain ordinary instance state on those subclasses.
  `is_cached` is an annotation and assignable instance state, as on the official
  public base. Concrete getters for these names are not installed as public
  properties.
  A public-base subclass that inherits the factory's `__new__` receives a plain
  concrete frame, and its own `__init__` is not called. Define `__new__` using
  `object.__new__` for independent Python data, or subclass
  `pyspark.sql.connect.dataframe.DataFrame` for native-backed subclass behavior.
  A direct native subclass must use the native constructor to initialize its Rust
  payload; `object.__new__` cannot safely allocate that payload.

- **Validation status.** All 22 subclass cases passed in the native offline suite.
  The unchanged official pipeline, tuning and one-vs-rest selection passed on both
  the official and native high-level clients with `PYTHONHASHSEED=0`: 16 passed and
  four upstream-disabled GC tests skipped on each client. This includes
  `test_pipeline` and `test_identity_pipeline`. An earlier run with independent
  default Python hash seeds failed native Random Forest cross-validation on a
  one-class training fold; that failure is retained, and its cause is not proven
  by the fixed-hash passes. API inspection records the public-base and
  concrete-class differences; inherited signature and annotation differences remain.
  These results do not establish complete DataFrame or ML parity.

## UDFs

As PySpark's `PythonUDF.to_plan` and `PythonUDTF.to_plan`, each request pickles the function of
each UDF, UDTF and map or group-map function of its plan when it builds the plan (see
`docs/compatibility.md`). These cases still differ, all of copies and pickles:

- **A pickled Column or DataFrame** (intentional). It carries the commands that its functions
  pickled to when it was pickled, in the protobuf of its expression or relation, which the
  server's Python worker unpickles without the native client. PySpark's carries the functions,
  which its requests pickle again: a value that a function reads, changed after the pickle, is
  seen by PySpark's unpickled Column of a function that `pickle` pickles by reference, such as a
  module-level function of `__main__`, but not by this client's. So `pickle.dumps` of a Column of
  a UDF of a lambda pickles, where PySpark's raises the `PicklingError` of the lambda, and that of
  a function that does not pickle by value raises its `PicklingError`, where PySpark's pickles a
  module-level function by reference.
- **A deep copy of a DataFrame into another session** (tracked). A deep copy whose session copies
  to another one, of a `memo` that maps it or of another active session of its ID, is a DataFrame
  of the relation of the copied one, so that its Observations are copies, as PySpark's, and so it
  carries the commands that the functions pickled to when it was copied, where PySpark's holds the
  copies of the functions, which its requests pickle again. In the session of the DataFrame, the
  copies of a DataFrame, a Column, grouped data and a table argument hold the functions as
  PySpark's do: `copy.copy` shares them, and `copy.deepcopy` deep-copies each with the `memo` of
  the call, which gives a function itself, and a copy of a callable object, a bound method or a
  `functools.partial`.
- **The UDFs of a `mergeInto` clause** (tracked). Its condition and assignments are pickled when
  the clause is added, as PySpark's `_build_merge_action` builds them, and again when `merge` runs,
  whose command carries the second pickle; PySpark's carries the first.
- **Garbage collection** (tracked). A DataFrame or a Column reports the functions of its UDFs that
  it alone holds to Python's garbage collector. A function that a Column and a DataFrame of it
  share, or two DataFrames, is reported by neither, so that no reference is counted twice: a
  reference cycle through it, such as through a bound method of an object that keeps both the
  Column and the DataFrame, is not collected, where PySpark collects it.

## Session and client (Connect)

- **Custom gRPC channels** (intentional; the transport is native). `ChannelBuilder` and
  `DefaultChannelBuilder` parse the connection string and hold its parameters, `host`,
  `port`, `endpoint`, `get`, `set`, `userId`, `token`, `session_id`, `use_ssl`, `secure`
  and `metadata()`. The transport reads them as a `sc://` URL, so they apply. The gRPC
  channel members do not: `toChannel`, `add_interceptor`, `setChannelOption`. The keepalive
  properties and `userAgent` read the parameters as PySpark does, but `userAgent` reports
  PySpark's `python/<version>` where the transport sends `python/rust` as the client type. The
  transport reads `SPARK_CONNECT_USER_AGENT` once, when the session is made, where PySpark's
  requests read it each time: it does not read the environment of the process while Python
  threads may change it.
  The Rust channel applies the keepalive from the same parameters. A `channelOptions` argument raises
  `NOT_IMPLEMENTED` instead of being ignored. A subclass that overrides `userId` or
  `token` is not read by the transport, which uses the parameters.
- **Retry and reattach** (intentional; DESIGN). Retries and reattachment live in the Rust
  transport. `client.retries` is PySpark's, vendored, and `SparkConnectClient.set_retry_policies`
  makes the transport retry with the parameters of the first `DefaultPolicy` of its policies,
  read when they are set. The transport classifies the failures as `DefaultPolicy.can_retry`
  does, so a policy of its own `can_retry` or `to_state`, and a backoff, jitter or delay that
  is not a whole number of milliseconds, raise `NOT_IMPLEMENTED`. `RpcDeadlines` sets the
  deadlines of the transport, which has no GetStatus RPC (`get_status`). `client.reattach`
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
  `_LEGACY_ERROR_TEMP_3161`.
- **The retry policy of the clones of a core client** (intentional; Rust API). The retry policy
  of `spark_connect_core::SparkConnectClient` is shared by its clones (`Clone`), as PySpark's
  `set_retry_policies` replaces the policies of a client that is in use: `set_retry_policy` of a
  clone changes the policy of the client it was cloned of too. A client of a new session
  (`with_new_session_id`, `clone_session`, and so `SparkSession::new_session` and
  `clone_session` of the Rust API) has the default policy, as PySpark's `newSession` and `clone`
  do, and not the policy of the client it was made of.
- **`SparkConnectClient.get_tags`** (intentional). A copy of the tags of the thread, where
  PySpark returns the set that its client keeps, which its later changes change too, as
  `SparkSession.getTags` does.
- **Plans of a DataFrame's relation** (intentional; DESIGN). `df._plan.plan(client)` is a
  placeholder of the DataFrame's relation, which the core replaces by the relation while the
  DataFrame is alive, wherever `plan::resolve_relation`, `resolve_command` and `resolve_analysis`
  find it: as the root relation of a plan or of an analysis, as the input of every relation of
  Spark Connect under it (`relation_inputs_of`: both sides of joins, set operations and zips, the
  other inputs of `GroupMap` and `CoGroupMap`, the root and references of a `WithRelations`, so
  the views of `SQL`), as the relation of each command that holds one (writes, views, streams,
  `SqlCommand`, checkpoints, `MergeIntoTableCommand`, the flows of pipelines), as the dataset and
  the fetched arguments of the ML commands, and anywhere in the message of the extension of a
  relation or of a command (a plugin's, as Delta Lake's `merge`) that holds it as `plan(session)`
  gives it. The core does not look into expressions, which hold no relation, so a placeholder in
  an expression's extension is sent as it is, and so is one that the message of an extension
  changes (`plan.common.plan_id = ...`); the server rejects them with
  `CONNECT_INVALID_PLAN.NO_HANDLER_FOR_EXTENSION`. A plan of it that `client.to_table`,
  `client.schema` or another method of the client sends after the DataFrame is freed fails with
  "a placeholder of a DataFrame that no longer exists".
- **Default port** (intentional). `DefaultChannelBuilder.default_port()` is 15002. PySpark
  takes the port of a local JVM server in its testing mode; the drop-in has no JVM.
- **Missing URL error** (intentional). The one builder of the drop-in serves
  `pyspark.sql.SparkSession.builder` and `pyspark.sql.connect.session.SparkSession.builder`.
  Without a URL or a channel builder it raises `CONNECT_URL_NOT_SET`, which is both a
  `PySparkRuntimeError` (the unified builder) and a `PySparkValueError` (the Connect
  builder's `SESSION_NEED_CONN_STR_OR_BUILDER` family). Giving both a channel builder and
  a URL raises `SESSION_NEED_CONN_STR_OR_BUILDER`. A channel builder without a host is
  rejected at `channelBuilder(...)`, not when the session is created.
- **Plan logging of a deep DataFrame** (intentional; DESIGN). `DataFrame._plan` is a
  placeholder over the relation of the Rust core, so `_plan.to_proto` is not provided.
  Python protobuf refuses to decode a relation nested beyond its limit (`too many levels
  of nesting`), and the drop-in never decodes a deep plan. `SparkConnectClient._proto_to_string`
  renders any message, including a deep one built in Python, as PySpark's does. Plan
  logging of a deep DataFrame therefore does not reach the recursion handling that the
  test `test_recursion_handling_for_plan_logging` exercises.
- **Client session ID** (not provided). `SparkConnectClient._session_id` is not exposed and
  is not settable; the session ID belongs to the Rust core. `_server_session_id` is
  exposed and settable, as PySpark's, and a changed value closes the session.
- **`_closed`** (tracked). Reading `_closed` gives the session's stopped state. Setting it
  to True closes the session, as a test's stand-in for the client's close. Setting it to
  False does not reopen a closed session.
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
  with it, the frames of the calls that led to it and their variables.
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
  user subclasses. The drop-in's DataFrame, `Observation`, `DataFrameReader`, `DataStreamReader`,
  `SparkSession.Hook`, `StorageLevel`, `PythonEvalType` and the data types are subclassed as
  PySpark's: a subclass constructs of the arguments of its `__init__`, and its copies and pickles
  are of the subclass.
  The `Window` of `pyspark.sql.window` and of `pyspark.sql.connect.window` is subclassed, of its
  static methods, but neither it nor a subclass has instances (`Window()` raises
  `cannot create '...' instances`), as it has no state. A reader is made of a session, one of
  those arguments, so PySpark's pickle of a reader, which makes it of its class without arguments,
  does not load in the drop-in (`TypeError`). The other native classes are not subclassable
  (`TypeError: type '...' is not an acceptable base type`), and of them only `SparkSession`,
  `Catalog` and `DataSourceRegistration` construct, as PySpark's do; the others
  raise `cannot create '...' instances`, as their constructors take the Python objects of
  PySpark's client, which the drop-in's native state replaces (see `docs/compatibility.md`):
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
    drop-in has no `DataTypeSingleton` metaclass, and copies and pickles of its one instance are
    that instance;
  - a class of types of two branches of their hierarchy (`class X(StringType, IntegerType)`)
    raises CPython's `multiple bases have instance lay-out conflict`;
  - `createDataFrame` of data sends the JSON of the core types of its schema, where PySpark sends
    the `json()` of the object, which the server may fail to parse, and its conversion of local
    rows names the class's type in its errors and does not call the `toInternal` of a subclass
    of a spatial type;
  - the `__repr__` of a cast to a type of a subclass shows the `simpleString` of the class's core
    type.
- **Metadata of the fields of a type** (tracked; the core holds JSON values). A field compares its
  metadata dict as PySpark's, of the values as they were given, and writes it with `json.dumps`,
  but a plan and `createDataFrame` of data send the core's JSON values of it, of an int of more
  than 64 bits as a float, and a float that is not finite (`inf`, `nan`) raises a `ValueError` when
  the field is made, where PySpark makes the field.
- **Shallow copy of a session builder** (tracked). `copy.copy(SparkSession.builder)` has
  options of its own, where PySpark's shares its `_options` dict and its hook factories
  with the builder that it copies, so that a `config` of either one is seen by both.
- **Session registry lock** (intentional). `SparkSession._lock` is a `threading.RLock`
  for the Python code that takes it. The drop-in serializes its own session creation with
  a Rust lock, not with this one.
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
  core replaces the shallow placeholder of the relation of a DataFrame, which
  `DataFrame._plan.plan(session)` gives, in the message of an extension too. What stays different
  (see `docs/compatibility.md`):
  - the core replaces that placeholder where it replaces the plan of a DataFrame (see "Plans of a
    DataFrame's relation" above). One that a plugin changes in the message of an extension reaches
    the server as a placeholder, which it refuses, where PySpark's plugins send the relation;
  - its classes are the runtime's, not subclasses of `google.protobuf.message.Message`, while its
    `DESCRIPTOR`, its `_*` descriptors, its extensions and services stay `protobuf`'s, and
    `protobuf`'s `message_factory` and `symbol_database` give `protobuf`'s classes of its types,
    whose messages equal (`==`) the runtime's of the same contents, as upb's of its one class of a
    type do;
  - a file that it imports that does not import the protos of Spark Connect (Delta Lake's
    `delta/connect/base.proto`) keeps `protobuf`'s classes. A field of the runtime of one of its
    types takes `protobuf`'s messages of the type, and gives a message of the runtime's class of
    the type, not of the module's class, which upb gives, so `isinstance` with the module's class
    is false, the message pickles only within the message that holds it, and the module's class
    takes it only through bytes or dicts (`CopyFrom` raises `TypeError`), though the two are equal
    of the same contents;
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
- **`SparkSession.client` and `SQLContext`** (intentional; the drop-in's own classes). The client
  has PySpark Connect's public API, of the class `pyspark.sql.connect.client.SparkConnectClient`,
  whose parameters of protobuf messages are `Any`, as the drop-in's protobuf runtime is native;
  `SQLContext` takes and returns `Any` in more places than PySpark's.
- **The internals of PySpark Connect** (intentional). The private attributes of the Connect
  classes that PySpark annotates with its plans, expressions and protobuf messages
  (`MergeIntoWriter._matched_actions`, `DataFrameWriter._write`, ...) are `Any`, as the drop-in
  builds them in Rust (`UNRESOLVED_IMPORTS` of the generator).

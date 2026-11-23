# Compatibility

This client aims for **full API parity** with the reference PySpark Spark Connect
client (`pyspark.sql.connect.*`) and the shared modules it depends on. The goal is
that existing Spark Connect code runs unchanged and gets byte-for-byte identical
results. For how this client relates to the `pyspark` and `pyspark-client` packages
on PyPI - and how to tell which one you have installed - see
[Which client am I using?](which-client.md).

## Supported Spark

Apache Spark **4.3.0 and later**. A release of the crates and the wheel has the version
of the Spark release it targets (starting at `4.3.0`), so the version number of a release
tells you which Spark it speaks. The `master` branch carries the next development version
(`4.4.0` now), and speaks the Spark release that it pins (4.3.0) until it moves to the next
one.

## Rust migration for the 4.3 line

These are the changes for Rust users from v4.2.0 to v4.3.0.

`Value` and `LiteralExpression` add a `DayTimeInterval(i64)` variant for signed
microseconds. Downstream exhaustive matches on these public enums must handle
the new variant. A collected map keeps its keys typed: `Value::Map` is a `Vec<(Value, Value)>` of
its entries in order, where it was a `BTreeMap<String, Value>` of the text of its keys, so a
day-time interval key is a `Value::DayTimeInterval` (a `timedelta` in Python, as in PySpark), and
`create_dataframe` takes such keys. `Value::Struct` holds a `StructValue`
(`Row::new(names, values).into()`), where it held a `Vec<(String, Value)>`. `Value` adds
`TimestampNtz(i64)`, the microseconds of a `TIMESTAMP_NTZ` value, which collected as a
`Value::Timestamp` before, and `Time(i64)`, the nanoseconds since midnight of a `TIME` value, which
collected as its ISO text in a `Value::String`. `DataType` adds `TimestampNtzNanos` and
`TimestampLtzNanos`; `SparkErrorKind` adds `BuiltinOverflowError`, `BuiltinRuntimeError`,
`BuiltinTypeError`, `BuiltinValueError`, `ConnectMessage`, `RecursionError`, `Spark`,
`UnicodeDecode` and `UnicodeEncode`; and `LogicalPlan` adds `ApplyInPandasWithState`, `AsOfJoin`,
`CachedRelation`, `ChunkedCachedLocalRelation`, `HtmlString`, `Parse`, `Raw`, `ShowString`,
`Unbuildable` and `WithRelations`. None of these enums is `#[non_exhaustive]`, so an exhaustive
match of one of them must handle its new variants.

`DataFrame::sort`, `order_by`, `sort_within_partitions`, `agg`, `observe`, `observe_with`,
`hint_expressions`, `repartition_by_range`, `repartition_by_range_opt`,
`repartition_by_expressions`, `GroupedData::agg`, the `partition_by` and `order_by` of `Window`
and `WindowSpec`, and `TableValuedFunction::udtf` take
`impl IntoIterator<Item = impl Into<Column>>`, as `select` and `group_by` do: Columns, column
names or expressions, such as `df.sort([col("x").desc()])` and
`.agg([sum(col("id")).alias("total")])`, without `.expression().clone()`. So does the first list
of `drop_columns`, of column references: a name in it is `col(name)`, as in PySpark's
`drop(F.col(name))`, so a name of several columns is `AMBIGUOUS_REFERENCE`, and a dotted name such
as `a.b` refers to the field `b` of a column `a`, which drops nothing (``"`a.b`"`` refers to a
column named `a.b`). Its second list is of plain names, as that of `drop`: a name drops every
column of that name, and `a.b` the column named `a.b`. `grouping_sets` takes any iterable of
grouping sets, each any iterable of Columns, such as `[[col("a")], [col("a"), col("b")]]`, and
`grouping_sets_with_columns` takes such sets and any iterable of the grouping Columns. The names
of `drop`, `describe`, `to_df`, `to_schema`, `melt`, the numeric aggregates of `GroupedData`
(`sum`, `avg`, `min`, `max`, `mean`, `numeric_agg`), the `freq_items` and `approx_quantile` of
`StatFunctions`, `Column::drop_fields` and the `partition_by` and `cluster_by` of
`DataStreamWriter` take any iterable of names, as do the SQL expressions of `select_expr`, the
statistics of `summary` and the paths of `DataFrameReader::load_paths`. The `(name, Column)`
pairs of `with_columns` and the `(old, new)` pairs of `with_columns_renamed` take any iterable of
pairs, and so do the `(String, Value)` pairs of `fillna_map`, such as a `BTreeMap<String, Value>`
(its names stay `String`s, so that `vec![]` and `("id".into(), value)` still infer their type).
`functions::count_distinct_columns` and `UserDefinedFunction::to_expression` (wasm) take any
Columns. A `Vec` or an array of them compiles as before. The `Vec<Expression>` that the sort and
aggregate methods took still compiles, by the new `From<Expression> for Column`, and so does the
`Vec<SortOrder>` of the `order_by` of a window, by `From<SortOrder> for Column`; there a Column
that is not a sort order is sorted ascending, nulls first, as PySpark's `WindowSpec.orderBy` sorts
it. An empty `vec![]` or `[]` has no item type to infer: pass `Vec::<Column>::new()` or
`Vec::<&str>::new()` instead. This breaks a caller of v4.2.0 that passes an empty literal to
`select_expr` or `summary`, which took a `Vec<&str>` there: `df.summary(vec![])`, for the default
statistics, becomes `df.summary(Vec::<&str>::new())`. `load_paths` is new in this release: v4.2.0
had only `load`, so no v4.2.0 caller breaks on it, and a new caller passes an empty literal as
`load_paths(Vec::<&str>::new())`. The optional
`subset` arguments of the null functions, the optional `value_vars` of `melt` and the optional
`column_names` of `drop_duplicates` and `drop_duplicates_within_watermark` stay
`Option<Vec<&str>>`, as in v4.2.0, so that `None` still infers its type. The optional `values` of
`unpivot` takes any iterable, as in v4.2.0, so its `None` has no item type to infer, unlike that
of `melt`: pass `None::<Vec<Column>>` or `None::<[&str; 0]>` to unpivot every column that is not
an id.

`BinaryClassificationEvaluator::set_score_col` and `score_col`, and the Python
`scoreCol=` argument and `setScoreCol` of v4.2.0 are removed, without an alias:
Spark's evaluator has no `scoreCol` param, so no value of it worked on the server.
Use `set_raw_prediction_col` and `raw_prediction_col` in Rust, and
`rawPredictionCol=` / `setRawPredictionCol` in Python. They follow standard
`pyspark.ml.evaluation.BinaryClassificationEvaluator`, whose default input column
is `rawPrediction` and whose scores may be doubles or vectors.

The native ML classes take a JVM `Int` param as an `i32`: `LogisticRegression::set_max_iter`,
`LinearSVC::set_max_iter` and `LinearSVC::set_aggregation_depth` take an `i32`, and
`LogisticRegression::max_iter` returns one. Every native estimator has a getter of each
of its params (`KMeans::k`, `RandomForestClassifier::num_trees`, ...), which reads its
`Params`: the value that is set, else Spark's default, which for `seed` is the Java hash code of
the JVM class name, the seed that the server takes when none is sent (PySpark's `getSeed` gives
Python's `hash` of the class name, which it does not send), and `None` for `weight_col` and
`thresholds`, which have none. The fields of `LogisticRegressionModel`, `StringIndexerModel`,
`MaxAbsScalerModel` and `PipelineModel` are private, as those of the other fitted models: use
`Transformer::operator` and `Transformer::params`, `LogisticRegressionModel::features_col`,
`label_col` and `prediction_col`, and `PipelineModel::stages`. `VectorAssembler::input_cols`
returns a `Vec<String>`, and `VectorAssembler::set_input_cols` takes
`impl IntoIterator<Item = impl Into<String>>`. `Pipeline::set_stages` takes the stages
themselves, `impl IntoIterator<Item = PipelineStage>`, where it took their names, and
`Pipeline::stages` returns them, a `&[PipelineStage]`. The `input_col` of `StandardScaler`,
`MaxAbsScaler` and `StringIndexer` returns an `Option<&str>`, which is `None` unless set: as in
Spark, it has no default. Their `output_col`, and that of `VectorAssembler`, return a `String`,
Spark's `<uid>__output` unless set, the column that the server writes. Before, they gave names
that the server never wrote, such as `scaled_features`.

`Params::set_param_int32`, `set_param_int_narrow`, `set_param_string_array` and
`set_param_double_array` are public, with the new `Params::clear_param`, and the evaluators
have a `params_mut`, as estimators and transformers do. Seeds, `CrossValidator::set_seed`'s
included, are sent as the narrowest integer literal that holds them, as PySpark sends a Python
int. `MlOperator::new` makes a uid of the simple class name and 12 hex digits
(`KMeans_1a2b3c4d5e6f`), as PySpark and Spark do, instead of a bare UUID.

`DataFrame::fillna_value` and `DataFrame::fillna_map` return a `Result`, as
`dropna` does. As PySpark's `fillna`, they take a bool, an integer, a float or a
string (an integer is sent as a long, a float as a double), and reject other values
with `NOT_EXPECTED_TYPE` and an empty map with `CANNOT_BE_EMPTY`. Before, such a
value filled string columns with its debug text. A `Value::Map` given to
`fillna_value` fills the columns of its keys, as PySpark's `fillna` of a dict, instead
of being rejected. `fillna`, `fillna_double`, `fillna_string` and `fillna_bool` are
unchanged.

A transport failure has the gRPC code that grpc-python gives it, which PySpark's errors and
retries see. `SparkError::grpc_code` is `UNAVAILABLE` (14) for a connection that closes or is reset
under a call, where it was `UNKNOWN` (2); for a call that its connection closes before sending it,
where it was `CANCELLED` (1); and for a connection that the server ends with a GOAWAY, where it was
the code of the reason of the GOAWAY, such as `INTERNAL` (13) for `NO_ERROR` and
`RESOURCE_EXHAUSTED` (8) for `ENHANCE_YOUR_CALM`. It is `INTERNAL` (13) for a stream that the
server resets with a reason that tonic leaves unmapped, such as `STREAM_CLOSED`, which is no
longer retried.

The 4.3 line changes the public Rust representation of expressions and logical
plans to share immutable nodes. This is a breaking change for code that directly
constructs or matches these representations. The `Column` and `DataFrame` APIs,
Python APIs, and Spark Connect protocol are unchanged by this representation change.

`Expression` is now a shared handle; its previous enum variants live in
`ExpressionNode`. Construct an `ExpressionNode` and convert it with `.into()` (or
pass it to `Expression::new`). Match on `expression.node()`, which borrows an
`&ExpressionNode`. Cloning an `Expression` shares its immutable node instead of
recursively copying its children. To change an expression, construct a new node.
Common nodes also have `Expression::literal`, `Expression::column`, and
`Expression::function` constructors.

For example, replace direct `Expression::Literal(...)` construction and matching
with:

```rust
use spark_connect::expression::{Expression, ExpressionNode, LiteralExpression};

let expression: Expression = ExpressionNode::Literal(LiteralExpression::Long(1)).into();
let shared = expression.clone();
match shared.node() {
    ExpressionNode::Literal(LiteralExpression::Long(value)) => assert_eq!(*value, 1),
    _ => panic!("expected a long literal"),
}
```

Child expressions are handles too: `ExpressionNode::DirectShufflePartitionId`
takes an `Expression`, and `CaseWhen::else_expr` is `Option<Expression>`, without
the previous extra `Box`.

Logical-plan child fields now use `Arc<LogicalPlan>` instead of `Box<LogicalPlan>`.
When constructing variants directly, replace `Box::new(child)` with
`Arc::new(child)`, or clone an existing `Arc` to share that child. Plan-building
helpers accept `impl Into<Arc<LogicalPlan>>`, so they accept either an owned
`LogicalPlan` or an existing `Arc<LogicalPlan>`. Serialization still produces the
same protobuf tree; sharing does not deduplicate repeated relations on the wire.

`LogicalPlan` implements `Drop`, so that a plan of many levels is dropped without
recursing. A field can therefore not be moved out of an owned `LogicalPlan`:
destructuring one by value is a compile error (E0509). Match on a reference instead
(`match &plan`, or `match plan.as_ref()` of an `Arc<LogicalPlan>`) and clone the
fields you need; the inputs are `Arc`s, so cloning them shares them.

The new `LogicalPlan::Raw { relation, owner, depth, observations }` is a relation built outside
the core, as `pyspark.ml` builds one over its input. Its `depth` is the levels of its relation as a
plan, with those of the DataFrames that it holds, which count for `MAX_PLAN_DEPTH`, as PySpark's
serializer counts the plan of the input within its own. Code that constructs the variant directly
sets it to 1 for a relation without nested plans. `MAX_PLAN_DEPTH` (746) guards the stack of this
client, and is not PySpark's limit: PySpark's serializer, with upb, sends a plan of up to about
990 levels, which the server fails from about 510 levels.
`LogicalPlan::try_to_proto` serializes a plan as a
request does: a plan deeper than `MAX_PLAN_DEPTH`, or with an expression deeper than
`MAX_EXPRESSION_DEPTH`, is a `RecursionError`. `to_proto` still serializes a plan of
any depth, but prost encodes and drops a protobuf by recursing, so the proto of a plan
of many thousands of levels can overflow a small stack.

`Value` and `LiteralExpression` add a `YearMonthInterval(i32)` variant for signed
months. Collect gives it for a year-month interval, where it gave the text of the
interval (`Value::String("1-0")`), so that a pivot on collected values names its
columns as the pivot without values does (`INTERVAL '1-0' YEAR TO MONTH`), and
`create_dataframe` takes it for an `INTERVAL YEAR TO MONTH` column (not for `INTERVAL YEAR`
or `INTERVAL MONTH`, which the server cannot narrow the Arrow interval to). Downstream exhaustive
matches on these enums must handle the new variant. The Python client still raises
`NOT_IMPLEMENTED` when it collects a year-month interval, as PySpark does.

`DataFrame::repartition_by_range` and `repartition_by_range_opt` return a `Result`.
As PySpark's `repartitionByRange`, a partition count that is not positive is
`VALUE_NOT_POSITIVE` and an empty column list `CANNOT_BE_EMPTY`. Before, the server
split the rows round-robin for an empty column list. `StatFunctions::sample_by`
returns a `Result` too: a stratum that is not a literal of a bool, an integer, a float
or a string (a null, a decimal, binary, a date, a timestamp, an interval or an array)
is `DISALLOWED_TYPE_FOR_CONTAINER`, as PySpark rejects a key that is not a float, an
int or a str. Before, it was sent as an empty literal, which failed only when the
DataFrame ran, or as its own literal, which sampled no rows. Without a seed,
`sample_by` sends one that the client draws, as `sample` and `random_split` do.

A panic of a native `StreamingQueryListener` on an event of the listener bus is
isolated, as on the started event and as PySpark isolates the exception of a listener:
it is reported, the other listeners still get the event, and the subscription goes on.
Before, it ended the subscription and removed every listener. An error that
`try_on_event` returns still ends it.

The random functions `rand`, `randn`, `uuid`, `shuffle`, `uniform`, `randstr` and
`count_min_sketch` draw a new seed for each call, as PySpark and the Scala client do.
Before, each sent the same fixed seed, so two `uuid()` columns, or the `uuid()` of
two runs, were equal. For reproducible values, use the `_with_seed` variants
(`rand_with_seed(42)`, `uuid_with_seed(lit(42))`, ...). `random` and
`random_with_seed` are added as the aliases of `rand` that PySpark has.

`DataFrameWriter::clear_bucket_by` is added: it unsets the bucketing of a writer, so that
its write sends none, as PySpark's `bucketBy` of a count that is not positive, the last
`bucketBy` winning. `bucket_by` is unchanged and sends the count it is given.

`DataFrame::persist` takes `impl Into<Option<StorageLevel>>`, so a `StorageLevel` is passed
as before. `None` sends no storage level, as PySpark's `persist(None)`, and the server then
caches at its `spark.sql.defaultCacheStorageLevel`. `cache` is unchanged and persists at
`MEMORY_AND_DISK_DESER`.

`DataType::from_json` reads the spatial types as the JSON reader of Spark does: a bare
`geometry` or `geography` is of SRID 4326, `geography(<algorithm>)` is of the CRS `OGC:CRS84`, a
CRS is of the form `<authority>:<code>`, and an algorithm other than `SPHERICAL` is
`ST_INVALID_ALGORITHM_VALUE`. It no longer takes `geometry(<srid>)`, `geometry(any)` and their
`geography` forms, which are of the DDL, not of the JSON, of Spark. `DataType::from_ddl` takes a
number or `any` as an SRID, as the DDL parser of Spark does, and no longer takes such forms as
`geometry(+5)`, `geometry(SRID:ANY)` or `geography(4326, SPHERICAL)`.

## Rust migration for the 4.4 line (unreleased)

These are the changes for Rust users after v4.3.0, on `master`, which ship in 4.4.0.

`SparkErrorKind` adds `BuiltinAttributeError` and `BuiltinAssertionError`, Python's `AttributeError`
and `AssertionError` as PySpark's client raises them, which `SparkError::builtin_attribute` and
`SparkError::builtin_assertion` make. `SparkErrorKind` is not `#[non_exhaustive]`, so an exhaustive
match of it must handle them. `ServerErrorInfo` adds a hidden `rpc_error`, grpcio's error of the
call that failed (an `error::GrpcioError`), `metadata`, the metadata of its `ErrorInfo`, and
`fetched_parameters`, whether `FetchErrorDetails` gave its message parameters, which are not a
stable API: a struct expression of `ServerErrorInfo` that names every field adds them, or takes the
others of `..ServerErrorInfo::default()`.

`LiteralExpression` adds a hidden `ByteArray { value, rendered }` variant, the literal of a Python
`bytearray`, which the Python client makes: it is sent as a `Binary` literal of `value`, and its
`render` is `rendered`, `bytearray(b'...')`, as PySpark renders a bytearray. It is not a stable API;
downstream exhaustive matches on `LiteralExpression` must handle it, or have a wildcard arm.

`ExpressionNode` adds a hidden `PlanBuildOrigin` variant, an expression that the Python client makes
when it builds a plan, such as the `&` of the conditions of a join of a list of Columns, so that its
origin is of that time, as PySpark makes it then. `CastTarget` adds a hidden
`Shown(DataType, String)` variant, a cast that sends its type as `CastTarget::Type` does and renders
as the given string, the `simpleString()` of the Python type object, as PySpark's
`CastExpression.__repr__` renders it, and a hidden `Read(Box<CastRead>)` variant, of
`Column::cast_read`, a cast of a type object of the Python client that reads it when it renders
and when a request builds the plan, as PySpark's `CastExpression` does: the readers of the
`CastRead` give the text that the render shows and the `spark.connect.DataType` that the request
sends, of the errors of each then. None is a stable API; downstream exhaustive matches on
`ExpressionNode` and `CastTarget` must handle them, or have a wildcard arm.

`GroupedData::transform_with_state` takes the output schema of its `TransformWithStateInfo`, as
`transform_with_state_in_pandas` does, where it sent the return type of its function: PySpark's
`transformWithState` sends `outputStructType` as it is given, a DDL string as an unparsed type,
which the server parses. The `update` and `insert` of the clauses of `MergeIntoWriter` take
`impl IntoIterator<Item = (K, Column)>` of `K: Into<String>`, so a `HashMap<String, Column>` is
passed as before, and the merge sends the assignments in their order, as PySpark sends those of
`assignments.items()`: a `Vec` or a `BTreeMap` keeps an order, and a `HashMap` sends its own.

`Params::sendable` serializes the params as a fit, a transform, an evaluation and a save send them,
as PySpark's `serialize_ml_params`: of the error of a value that PySpark cannot send, before any
request. `Params::extend_from` sets the params of another `Params` over them, as a copy of PySpark
of a param map sets its values. `Params::set_param_int_array` sends a list of ints as PySpark's
literal of it, of the type of its first element, where it sent a `long` array when an element did
not fit 32 bits. `PredictionModel::num_features` and `VectorIndexerModel::category_maps` of an int
out of the range of an `i32` are an error, where they truncated it, and a grid int beyond a long,
such as a whole float `1e20` of an int param, is PySpark's `VALUE_NOT_BETWEEN`, where it saturated
to `i64::MAX`. `OneVsRest::max_label` returns the `Value` of the largest label, of which the fit
takes Python's `int()`, where it returned an `Option<f64>`; `OneVsRest::fit_classes` takes the
weight column of the fit, and returns the binary training set of each class with the model;
`OneVsRest::check_required_params` is removed, as the fit raises PySpark's errors in PySpark's
order; and `OneVsRestModel` adds `new` and `set_classifier`, and its `apply_param_map` applies the
map to each of its models too, as PySpark's `OneVsRestModel.copy` copies them (see
[Native OneVsRest](#native-onevsrest)). `Model::fit_warning` and `MlRef::fit_warning` give the
warning that the server sends with the model of a fit, and
`SparkSession::set_ml_fit_warning_handler` sets the handler that each fit calls with it (see
[Native ML objects](#native-ml-objects-fetch-summaries-save-and-load-rust-api)).
`MlRef::of_reference` (not a stable API) is an object of the ML cache that the client does not
own, whose last owner runs a release of its owner in place of its deletion, `MlRef::keep` keeps an
object on the server at the drop of its last owner, and `MlRef::is_unshared` tells an owner of no
clone (not stable APIs either), and `PickleReach`, `pickle_reach_with` and
`expression_pickle_reach_with` of `spark_connect::ml` (not a stable API) give what PySpark's pickle
of a DataFrame or of an expression reaches of the objects of `pyspark.ml` that its plans keep, and
the functions of its UDFs before them, in PySpark's order, of the reach that a callback gives of
the owner of a relation built outside the core.

`RetryPolicy` adds a `retries_failures` field, true by default: a policy of false retries no
failure, as PySpark's own `RetryPolicy`, whose `can_retry` takes no failure, retries none, and
`RetryPolicy::no_retries` is such a policy. The jitter of each retry is a draw of its own, uniform
over its range, as PySpark's `random.uniform(0, jitter)` draws it, where the retries of calls that
failed together waited the same jitter. A struct expression of `RetryPolicy` that names every field
adds it, or takes the others of `..RetryPolicy::default()`. The core tells the listener of
`retries::set_listener` (a `RetriesListener`, not a stable API) when the retries of a failure that a
policy retries are spent, where PySpark's `Retrying` warns `[RETRIES_EXCEEDED]`, and its error is
the error of the call; when a request that it sends in the background, which no caller waits for,
fails (`background_failure`, of a `retries::BackgroundRequest` and a `retries::BackgroundFailure`),
where PySpark's sender warns and logs; and of what PySpark's client logs of its retries and of its
errors (`retrying`, `retrying_at_once`, `given_up`, `server_retry_delay`, `rpc_error` and
`error_info`), and, in the dialect of grpcio, of a wait before a retry that is negative or of
decades, which the Python binding sleeps of Python's `time.sleep` (`sleep_long`, of a
`retries::LongSleep`, which the call awaits). These methods have default bodies that do nothing.
The listener is one of the
process, for the calls of every client: the Python binding sets it, to warn from the thread of the
call and to log to the logger of Spark Connect. `SparkError` adds a hidden `pyspark_message`, the
message of PySpark's exception of a gRPC failure whose status carries no error of the server, which
the Python binding raises: the text of grpcio's error of the call, as PySpark's `_handle_rpc_error`
raises it; and a hidden `grpcio`, grpcio's error of a failed call that PySpark's client leaves to
grpc, which the Python binding raises as grpcio's `RpcError` (an `error::GrpcioError` of
`error::grpcio_error`, not a stable API), set only in the dialect of grpcio (see below). The message
of the error of the Rust API is still the description of the code in brackets and the message of the
status. A struct expression of `SparkError` that names every field adds them, or takes the others of
an error of its constructors, such as `..SparkError::connect_msg(message)`.
`SparkError::from_grpc_status_of` converts the status of a call of a stream of responses
(`GrpcCall::ServerStreaming`, of grpcio's `_MultiThreadedRendezvous`). The retries of the calls of a
client are decided by its `RetryPolicy`, or by a `retries::RetryDecider` that
`SparkConnectClient::set_retry_decider` and `SparkSession::set_retry_decider` set in its place, of a
`retries::CallRetries` for each call that gives a `retries::Decision` of each failed attempt: as the
Python binding runs PySpark's `Retrying` of policies that the core does not retry with. These are
not a stable API. The reads of the configs of the server's stack trace, as the error of the server
of a call is converted, are retried under the retry policy of the client, as PySpark's `config()`
retries them, where each was read once.

The module `client_log` of the core (not a stable API) has what PySpark's client logs of its calls:
the listener of the process (`client_log::set_listener`, a `client_log::Listener`, which the Python
binding sets) is asked whether a level is logged and given each record of one, a
`client_log::Message` of a text, or of an encoded message of protobuf with the text before it, which
the binding writes of PySpark's `_proto_to_string`, of a `retries::Delivery` whose error, as of a
handler of the logger that raised, fails the call at the record (`client_log::log` is
`#[must_use]`: a record is written as its delivery is awaited), and which writes the records of a
thread that `client_log::write_records_on_this_thread` marks on that thread, as those of the
listener bus of streaming queries. `SparkConnectClient::execute_plan_reattachable` logs as
PySpark's `to_table` a plan of a relation, and as its `execute_command` a command;
`SparkConnectClient::execute_plan_reattachable_logged` logs as the method of a
`client_log::ExecutionLog`, of the plan compressed, and
`SparkConnectClient::execute_plan_reattachable_as_is` of the plan as it is, as the plan that
`SparkConnectClient::prepare_plan` prepared, of one try of its compression, as PySpark's `to_proto`
tries it; `SparkConnectClient::build_execution` and `SparkConnectClient::execute_built` make the
two parts of such an execution, before PySpark's `Progress` of it starts and in it (a
`client::BuiltExecution`); `SparkConnectClient::analyze_plan_logged` logs as the method of a
`client_log::AnalysisLog`, of the plans compressed, and `SparkConnectClient::analyze_plan_as_is` of
the plans as they are; `client_log::explain_plan` is the record of PySpark's `explain_string`, of
the mode as it is given; and `SparkSession::set_thread_call_stack` and
`SparkConnectClient::set_thread_call_stack` give the requests of a thread the call stack of the
user's code of their call. `SparkSession::execute_plan_proto_logged`,
`SparkSession::resolved_plan_proto`, `SparkSession::compressed_plan_proto`,
`SparkSession::known_plan_compression` (of `SparkConnectClient::known_plan_compression`, the
threshold of the plan compression once its settings are read) and `SparkSession::result_chunking`
serve the plans and the requests of the Python client, and `client::converting_failure` gives
the failure that the calling thread converts, of grpcio's error and of the operation ID of its
execution (a `client::Conversion`), during which a signal interrupts nothing, as during PySpark's
`_handle_error`, and `SparkConnectClient::inside_error_handling` (and
`SparkSession::inside_error_handling`) whether the calling thread converts a failure of the client,
during which the failure of another of its calls is not converted, as PySpark's
`inside_error_handling` tells it: of a call that runs within a poll of the conversion, as of a
handler of one of its records or of a `RetriesListener`, also through the blocking API, or on the
thread of a call of the Python binding whose conversion runs, as of a signal handler, but not of
another task of an async caller of the client, which the thread polls between two polls of the
conversion. The methods of a `RetriesListener` of what
PySpark's client logs of its retries (`retrying`, `retrying_at_once`, `given_up`,
`server_retry_delay`) return the error of the record, which fails the call, and
`RetriesListener::raised_in_handling` is given the error of a record that the client writes as it
handles the failure of a call, of that failure (a `retries::Handled`: the failed RPC of a
conversion, or the error of the failed lookup of the plan compression), and gives the error of the
call. All of these are hidden. Without a listener, nothing is logged. A closed client, as of a
stopped session, writes the records of a call before it fails it with NO_ACTIVE_SESSION, where it
failed it first, and the first plan of a closed client looks the plan compression up, which fails
of the closed client and disables the compression, as PySpark's. A plan
is compressed in a frame of zstd of the size of its content, as python-zstandard's `compress`,
where it was a frame of no size, so the server's `PLAN_SIZE_LARGER_THAN_MAX` of a plan over its
limit reports the size that the frame declares, as of PySpark's plans, and the threshold of the
compression is read as Python's `int()` reads it (`+1_000`).

`SparkConnectClient::read_plan_compression` reads the plan compression settings of the server
before a plan is sent, as PySpark's client reads them when it builds its first plan: before the
request of a relation (`to_proto`), which then observes the server-side session ID of the response
of the read, and after the request of a command (`execute_command`), so the first command of a
session observes none. `SparkConnectClient::execute_plan_reattachable_uncompressed` sends a plan
as it is, neither compressed nor after that read, as PySpark's registrations of functions, table
functions, data sources and Java functions set their command in the plan of their request, which
the registrations of the core do.

`SparkSession` adds the settings of PySpark's client: `retry_policy` and `set_retry_policy`, as
PySpark's `get_retry_policies` and `set_retry_policies`; `reattachable_execute` and
`set_reattachable_execute`, as its `enable_reattachable_execute` and `disable_reattachable_execute`;
`token_parameter`, the `token` parameter of the connection string; `rpc_deadlines`, the
`RpcDeadlines` of its calls; and `set_global_user_context_extensions`, which replaces the global
user context extensions, as a change of PySpark's `global_user_context_extensions`.
`SparkConnectClient` adds them too (it had `set_retry_policy`, `rpc_deadlines` and
`set_rpc_deadlines`), with `replace_retry_policy`, which sets the policy of a client in use, and
`result_chunking` and `set_result_chunking`, the result chunking of its executions, a
`ResultChunking` of PySpark's `allow_arrow_batch_chunking` and `preferred_arrow_chunk_size`, which
by default allows chunks of no preferred size, as before. The clones of a client share its retry
policy, where each clone had a policy of its own, and the client of a new session, of
`SparkConnectClient::with_new_session_id` and `clone_session` (and so of `SparkSession::new_session`
and `clone_session`), has the default policy and chunking and no decider of its retries, as
PySpark's `newSession` and `clone`, where it kept the policy of the client. An execution that is not
reattachable, as `ExecutePlanResponseReattachableIterator::not_reattachable` makes its iterator, is
one `ExecutePlan`, sent again from the start when its stream fails in a way that the retry policy
retries, and never released, as PySpark's. `SparkSession::cache_artifacts` caches blobs as artifacts
of their SHA-256 hashes, as PySpark's `SparkConnectClient.cache_artifacts`, and
`SparkSession::add_artifacts_of_flags` adds files as PySpark's `SparkConnectClient.add_artifacts`
names them by its flags.

`RpcDeadlines` adds a `get_status` field, the deadline of the `GetStatus` RPC, of 10 minutes by
default and none in `RpcDeadlines::disabled()`, as PySpark's `RpcDeadlines.get_status`: a struct
expression of `RpcDeadlines` that names every field adds it, or takes the others of
`..RpcDeadlines::default()`. `SparkConnectClient::get_status_proto` sends the `GetStatus` RPC of a
request, of the session ID, the user, the client type and the observed server-side session ID of the
client, under its retry policy and the `get_status` deadline, as PySpark's
`_get_operation_statuses`. `ChannelBuilder::params` gives the parameters of the connection string,
in the order of their keys, and `ChannelBuilder::remove` removes one, as PySpark's `newSession` pops
the session ID of the channel builder of its client.

A channel connects to the addresses of its host, or of its proxy, as grpcio's: the next one 250 ms
after the previous one or once the previous one failed, each with a connect deadline of 20 s of its
own, which its TCP connection, the tunnel of a proxy, its TLS handshake and the HTTP/2 settings that
the server sends first share, of a TLS connection too, and which bounds the lookup of the name too.
A connection is the process's, of one address, as a subchannel of gRPC's global subchannel pool:
the clients whose channels resolve the target to that address, of the same proxy and its
credentials, TLS and keepalive, and of no token, share it, as gRPC's pool shares the subchannels of
the channels of a process. Each client has a channel of its own, which resolves the name of the
target as each call that needs a connection connects, and takes a connection of its addresses that
another client made, the first of them in their order, else connects to them in turn: the clients
of `SparkConnectClient::with_new_session_id` and `clone_session` share the connection of their
client, as before, and so does a client of another `connect` of the same target, which had a
connection of its own, while their names resolve to its address; a client whose name resolves to
other addresses connects to them, where it took the connection of the target that the process had.
A client of a connection string of a token has a connection of its own, also of a new session or a
clone of it, and of another `connect` of the same token, as grpcio's channels of the credentials of
a token share no connection, where its new sessions shared the connection of the client; the clones
of a client (`Clone`) share its channel. A connection ends once no open client holds it:
`SparkConnectClient::close` lets go of it, as grpcio's `close` of a channel, where it stayed while
the closed client lived. The window of a connection is the
largest of HTTP/2, so that the results that a client does not read on, as of executions that it
holds open, do not hold the calls of the other clients of the connection, nor its other calls, where
a connection of five such results of 2 MiB waited, as gRPC's connections, whose windows grow with
those of their streams, do not wait; the window of each stream, of 2 MiB, bounds what it holds
unread. The channels run on the shared runtime of the core (`runtime::get_runtime`), not on the
runtime of the caller, as they are the process's: a caller's runtime, or its paused clock of a test,
does not drive their timers. A child of a `fork()` gets a shared runtime and channels of its own, as
a parent's runtime has no thread there: the clients of the parent take a channel of the child's in
the child, as grpcio's fork support resets the channels of a parent, and `runtime::fork_generation`
(not a stable API) tells the generation of the process, of which a handler of `pthread_atfork` makes
another in each child. The child leaves the objects of its parent's runtime as they are, never
dropped there: a client of the parent that the child drops leaves the connection of its channel to
the parent, where the drop removed the socket of a connection that the channel held from the epoll
instance that the child shares with its parent, so that the parent's next call on it waited forever,
and a stream of an execution of the parent that the child drops is not released, as the parent reads
it. As the fork returns, the child closes the sockets of the connections of its parent and the epoll
instances of its parent's runtime, without a call on the epoll instance, so that a connection of the
parent ends on the server with the parent, as grpcio's fork support closes the descriptors of the
event engine of a parent, and makes the eventfd of the runtime of each of its parents an eventfd of
its own, rather than closing it: an object of the parent that wakes a task of that runtime writes to
it, which a descriptor of the child could take the number of. A child then holds one descriptor of
each generation before it, where it held the three of the runtime of each of its parents, and the
sockets of their connections. The state that the threads of the process keep under a lock is of each
generation, or read without one, so that a lock that a thread of the parent held at the fork does
not hold the child: the runtime, the channels, the limit of the logs of the failed tunnels of a
proxy, the jitter of the retries, which a child draws of a seed of its own, as Python seeds `random`
again in a child, and, of a client, its session IDs and its retry policy, which a child keeps, and
its executions in flight, of which a child has none of its parent's. A fork while another thread
makes a call can leave the child waiting, as grpcio's fork support asks that no other thread makes a
call then, and so can a fork while the client makes a call in the background (a release of an
execution, the removal of a cached relation or of a model, the reads of `to_local_iterator` of
`prefetch_partitions` and of the events of the streaming query listeners), which holds the lock of
the hook, the user context extensions, the plan compression or the last execution of its session for
the moment that it reads them. The per-thread state of the other threads of the parent, of their
tags and user context extensions, stays in the child, which never forgets it, as those threads end
in the parent only: a leak bounded by the threads of the parent. The first call of a connection
waits for the settings of the server before its request is sent, a round trip, as grpcio's does. One
that does not connect fails its calls with `UNAVAILABLE`, which the retry policy retries, and the
next call that needs a connection connects again (see [What "drop-in" means](#what-drop-in-means)
for what stays different from grpcio). Before, a direct connection bounded its TCP connection only,
by 10 s, the addresses of a proxy were tried in turn within one deadline, so a TLS handshake that
the server never answered held a call until its deadline, and a plaintext server that never answered
until the keepalive of the connection. The status of a channel that cannot connect has the client's
message, of the last error of its addresses, such as `failed to connect to all addresses; last
error: 127.0.0.1:1: tcp connect error: Connection refused (os error 111)`, or of a name that does
not resolve, `dns error: failed to look up <host>:<port>: ...`, of one whose lookup timed out, `dns
error: failed to look up <host>:<port>: timed out`, where it was tonic's `tcp connect error`, and a
call that the end of its connection fails has tonic's message of it. The Python binding sets the
dialect of grpcio instead, with `set_grpcio_dialect` (not a stable API): the texts of grpcio, the
`User-Agent: grpc-httpcli/0.0` of grpcio's `CONNECT` request of a proxy, `localhost` and the names
under it as the loopback of both families, as c-ares resolves them, and the state of gRPC's channel,
whose calls fail at once once it failed to connect, while its subchannels connect again in the
background with gRPC's backoff, until a call takes the connection that one made there, until 30
minutes without a call, or until its client is closed (`SparkConnectClient::close`), and whose
resolver resolves the name of the target before the first call, and again as its connection ends
or fails, no sooner than 30 s after its last resolution, as gRPC's, which a Rust client does not
have, as its calls recover from an outage at once. The binding sets the end of the user
agent too, `python/<version>` where the Rust API's is `python/rust`, with
`channel::set_user_agent_language` (not a stable API either). The user agent is sent as the client
type of the requests only, as PySpark sends it, and the HTTP `user-agent` header is tonic's own, so
a user agent of a control character connects, where its connection failed. A user agent longer than
2048 characters after URL escaping, and a `SPARK_CONNECT_USER_AGENT` that is not UTF-8, are the
error of each request that carries the user agent, as PySpark's `userAgent` raises them when a
request is built, where the first failed the connection of the session and the second sent
`_SPARK_CONNECT_PYTHON` in its place. The user agent of a client is read when the client is made, of
the connection string, else of `SPARK_CONNECT_USER_AGENT`, a client of a new session keeps it, and
`SparkConnectClient::refresh_user_agent` and `SparkSession::refresh_user_agent` (not a stable API)
read it again, as the Python binding does as each of its calls starts. A tab, CR or LF in the
connection string is removed, as `urllib.parse.urlsplit` removes them, where it failed. The
`session_id` of the connection string is checked as PySpark's `uuid.UUID(session_id, version=4)`
checks it, of Python's messages, and is sent as it is given, where it was checked as a UUID of the
uuid crate, of the origin `invalid UUID`. A retry policy of no `max_backoff_ms` keeps its initial
backoff, as PySpark's does, where its backoff grew by its multiplier.

A `ChannelBuilder` of `ChannelBuilder::with_credentials` (of `channel::Credentials`, with
`ChannelBuilder::credentials` and `SparkSessionBuilder::channel_credentials`, not a stable API),
which the Python binding gives the channel of a `DefaultChannelBuilder` as PySpark's `toChannel`
decides them, has those credentials in place of the rule of `ChannelBuilder::secure`, of `use_ssl`
and of a token on a host that is not `localhost`: `Insecure` is plaintext and sends no token, of the
connection string or of `SPARK_CONNECT_AUTHENTICATE_TOKEN`; `Local`, gRPC's local credentials, is
plaintext, of the token of the connection string, over a connection whose local address is
`127.0.0.1` or `::1` only, as gRPC's handshake takes it, else the call fails of its `Endpoint is
neither UDS or TCP loopback address.`; and `Ssl` is TLS, of the token of the connection string. A
connection string never sets them: one of a parameter of any name is as before, so the connection
strings of `remote()`, `SPARK_REMOTE` and the Rust API are unchanged.

A `ChannelBuilder` of `ChannelBuilder::with_target` (of `channel::ChannelTarget`, with
`ChannelBuilder::target`, `SparkSessionBuilder::channel_target` and `grpc_target::parse`, not a
stable API), which the Python binding gives the channel of a channel builder of the target of gRPC
of its `endpoint`, connects to the name, the addresses or the Unix sockets of the target, in their
order, in place of its host and its port, or fails each call of the status of a target that gRPC
refuses, and of `ChannelBuilder::with_metadata` (with `SparkSessionBuilder::channel_metadata`), the
binding's of the `metadata()` of the builder, sends that metadata in place of its parameters; and
`SparkConnectClient::set_metadata_source` (with `SparkSession::set_metadata_source` and
`channel::MetadataSource`, not a stable API) gives the calls of a client the metadata of a source,
which the binding reads of a `metadata()` of its own for each call. A connection string sets none of
them. Of the connections of every client, a connection that fails as it starts, of the error of
`connect`, as one of a network that cannot be reached, fails of gRPC's text of it in the dialect of
grpcio (`FAILED_PRECONDITION: ... connect failed: addr: ... error: Network is unreachable`), and in
that dialect a name is resolved as c-ares takes it: an address of four decimal numbers of leading
zeros and an IPv6 address of a zone are addresses, a name of a character other than
`[A-Za-z0-9-._/*]` or of an empty label is not resolved (`Misformatted domain name`), and a name
that the system's resolver reads as an address, as `127.1`, is a name that it looks up. The name of
a `channel::ChannelTarget::Name` of a `server`, the DNS server of the authority of a target of
`dns://`, is resolved through that server, of both dialects, as gRPC's resolver resolves it (see the
`endpoint` of a channel builder below), but through the tunnel of a proxy, whose own name the
system's resolver resolves.

`GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` and its file are read once in a process, when its first TLS
session is made, as grpcio reads them, where each client read them when it connected, and an empty
file is as no file, without a warning. A TLS channel trusts the roots of that file alone, as grpcio,
else those of the system's trust store, as before, and the roots of the file of
`client::set_bundled_ssl_roots_file` (not a stable API) besides, which the Python binding sets to
grpcio's bundle where grpcio is installed. A roots file without a certificate does not fail the
session when it is made: each call fails `UNAVAILABLE`, as grpcio fails to make the handshaker of
its sessions, where the session failed with `Failed to configure TLS`, and so does a system of no
root. The core no longer enables tonic's `tls-ring` and `tls-native-roots` features, which it does
not use, as it makes the TLS sessions itself: a program that uses tonic's own TLS enables them. The
crates enable tonic's `channel` and `codegen` features only, of its client, where they enabled its
default features, of its server (`server` and `router`) too: a program that uses tonic's server
enables them. An invalid proxy address of the environment fails only the sessions of the hosts that
the proxy applies to, as grpcio connects the others directly, where it failed the sessions of every
host. What grpcio logs of a channel, of a proxy variable that is not a proxy, of a roots file that
cannot be read, and, in the dialect of grpcio, of a TLS handshake that fails and of a proxy that
refuses the tunnel, also goes to the listener of `grpc_log::set_listener` (a `grpc_log::Listener`,
not a stable API), as it goes to `tracing`, of the `grpc_log::Severity` of grpcio's line, which
`grpc_log::Logged` adds as its `severity`, `Info` or `Error`; the Python client writes it to stderr,
as grpcio does. The client of a new session logs the variables of the proxy again, as it reads them
again for its channel. `RetriesListener::caught` returns the error that the call fails with, of one
that PySpark's `except Exception` does not catch, as a `KeyboardInterrupt` of a retry policy of the
Python binding, and so `SparkConnectClient::read_plan_compression` returns a `Result`.

A message over the size limit of a call, of 128 MiB, fails the call with `RESOURCE_EXHAUSTED` (8),
as grpc-python's: a response, which tonic does not read, where `SparkError::grpc_code` was tonic's
`OUT_OF_RANGE` (11), and a request, which the client does not send, as it checks its length before
the call, where tonic's check, as it encodes the request in the body of the call, reset the stream
of the call, of the error `INTERNAL` (13), `h2 protocol error: http2 error`, of no cause. The texts
are tonic's (`Error, decoded message length too large: found <length> bytes, the limit is: <limit>
bytes`, and `encoded` of a request), and in the dialect of grpcio those of gRPC's core (`Stream
removed (CLIENT: Received message larger than max (<length> vs. <limit>))`, and `CLIENT: Sent
message larger than max (<length> vs. <limit>)` of a request). As tonic checks the length of a
response before any codec reads it, its status of it is told by its code, its text and the limit of
the client, 134217728: a status of a server of the same code, text and limit, of no details, as a
server of tonic whose own limit is 128 MiB too sends of a request over it, is read as the client's,
where the status of a server of another limit, as of tonic's default of 4 MiB, is the server's; the
client's own status of a request that it does not send is tagged as the client's. A TLS session is
never resumed, as grpcio does not resume one by default, where rustls kept the sessions of its
connections in a cache, whose lock a child of a `fork()` shared with its parent. grpcio's error of a
response that the client fails to decode has no `debug_error_string`
(`GrpcioError::debug_error_string` is an `Option`, not a stable API), as grpcio's, where it had the
name of the code and prost's message, in the error that a retry policy of the Python binding is
asked of, in the log of the errors of the calls and in the warning of a release that fails in the
background. The metadata of a response, its headers, or its trailers, that reaches gRPC's hard
limit of 16 KiB, of HPACK's sizes of its entries (of the name, the value and 32 bytes each), fails
the call with `RESOURCE_EXHAUSTED` of gRPC core's text (`received metadata size exceeds hard limit
(value length <length> vs. 16384)`, of an entry whose value, or whose name, reaches it alone with 32
bytes, or `(<size> vs. 16384);  <entries>`, of the entries as gRPC's core lists them, the headers
that it knows first), where hyper's limit of a header list of 16 KiB reset the stream of the call,
of `INTERNAL`, or ended the connection of a far larger one, of `UNAVAILABLE`, which the retry policy
retried: hyper now takes a header list of up to 16 MiB, which the channel checks, while the
connection still advertises 16 KiB as the size of a header list that it takes, as grpcio's channel
does, so that a server that honours it, as grpc-java's, resets the stream of a call of more rather
than sending it. A binary value (of a name ending in `-bin`) that gRPC's core does not decode as
base64 fails the call `INTERNAL`, `Error parsing '<name>' metadata: illegal base64 encoding`,
whatever the code of the server, as gRPC's core fails it, the `grpc-status-details-bin` of an error
too, where tonic panicked of it, of which the call has the details, empty, as gRPC's core gives them
the call. gRPC's core takes base64 of no padding and of any `=` at its end, such as `YWJj=`, and
refuses a last group whose unused bits are not zero, such as `YW==`. A stream that the server
resets fails its call of the code that gRPC's
core gives the reason, `CANCELLED` of `CANCEL`, `UNAVAILABLE` of `REFUSED_STREAM`,
`RESOURCE_EXHAUSTED` of `ENHANCE_YOUR_CALM`, `PERMISSION_DENIED` of `INADEQUATE_SECURITY` and
`INTERNAL` of any other, where tonic's channel gave them all `UNKNOWN`, which
`retries::grpc_python_code` made `INTERNAL`. In the dialect of grpcio, the metadata past gRPC's soft
limit of 8 KiB fails its call at random too, as gRPC's core fails it, the more often the larger it
is, of `received metadata size exceeds soft limit (<size> vs. 8192);  <entries>`, of the details of
the server's error, which gRPC's core still gives the call; a reset stream has gRPC core's details,
`Stream removed (RST_STREAM (Received RST_STREAM with error code <reason>))`; and a failure whose
`google.rpc.Status` (its `grpc-status-details-bin`) is not of its code or its details, as of
metadata past the soft limit or of empty details, a status of the code `OK`, is the `ValueError` of
grpc_status's `rpc_status.from_call` where PySpark's client reads the status, and one whose details
upb's parser does not read as a `google.rpc.Status`, as of one nested past its limit, is protobuf's
`DecodeError` (`Error parsing message with type 'google.rpc.Status': Wire format was corrupt`)
there, and so is one whose first `ErrorInfo` or `RetryInfo` upb does not read, of its type, where
PySpark unpacks it (see `SparkError::grpcio`, not a stable API). In the dialect of grpcio, the delay
of a `RetryInfo` is PySpark's, signed (the `ToMilliseconds()` of its `retry_delay`, 0 of one of
none), and so is the wait before a retry, of Python's numbers; `retry_delay_ms` and
`RetryPolicyState::next_attempt` are unchanged, of a delay of a `u64` of milliseconds, of the
seconds and the nanoseconds each clamped at 0 and of a sum that saturates, which the Rust client
waits as it is. The error of a closed client or channel of an execution comes once the lookup of
the plan compression and the hook of the executions ran (`set_execute_plan_hook`), as PySpark reads
its `_stub` once its plan is built and its hooks ran, and carries the operation ID of its request,
when it has one (`SparkError::operation_id`), as of its other failures.

`PythonUDFPayload` and `LogicalPlan::CommonInlineUdtf` add a `pickler` field, which is not a
stable API: the pickling of the command of a UDF or UDTF of the Python client, which a request
calls each time it builds the plan, as PySpark pickles the function then. Code that constructs
the struct or the variant directly sets it to `None`. `PythonUDFPayload::new` and
`TableValuedFunction::udtf` are unchanged: a command of bytes is sent as it is.
`LogicalPlan::MapPartitions` adds a `profile_id_reader` field, which is not a stable API either:
the reader of the id of the resource profile of the Python client's `mapInPandas` and
`mapInArrow`, which a request calls when it builds the plan, as PySpark reads `profile.id` then.
Code that constructs the variant directly sets it to `None`; `profile_id` is sent as before.
`LogicalPlan::ToSchema` adds a `schema_reader` field, which is not a stable API either: the reader
of the schema of the Python client's `DataFrame.to`, which a request calls when it builds the plan,
as PySpark converts the schema object then. Code that constructs the variant directly sets it to
`None`; `DataFrame::to` sends its schema as before.
`SentExpression`, the expression of a pickled Column (`ExpressionNode::Sent`), adds a `subqueries`
field: the plans that its subqueries refer to, of `Expression::sent_with_subqueries`, which the
relation that holds the expression sends in its `WithRelations`, as it sends the plan of a
`SubqueryExpression`. Code that constructs the struct directly sets it to
`SentSubqueries::default()`, of no plans, as `Expression::sent` gives it.
`PreparedExecution::collect_record_batches_checked`, which is not a stable API, is
`collect_record_batches_and_info`: it gives the `ExecutionInfo` of the execution too, which the
DataFrame keeps once `DataFrame::keep_execution_info` is given it, as PySpark's `toPandas` sets
the information of the execution once it has converted the result.

`CoGroupedData::input_columns` is removed. It gave the columns of both sides for the arguments of
the function of a cogroup, but that function binds no columns, as PySpark's `CoGroupMap` builds it:
it takes the whole cogroups, so the inputs need no analysis.
`GroupedData::apply_in_pandas_with_state_with_schemas` sends the output and the state schemas as the
strings that it is given, DDL or JSON, which the server parses, as PySpark sends a schema given as a
string; `apply_in_pandas_with_state` sends the JSON of its types, as before.
`LogicalPlan::ApplyInPandasWithState` holds the two schemas as these strings: its `output_schema`
and `state_schema` are `String`s, where they were `DataType`s. `Value::calendar_interval_parts`
reads the months, days and nanoseconds of a collected calendar interval back from its text, and
`Value::calendar_interval` makes that value of the parts. `LocalRowIterator::next_decoded` takes the
next row of the batch that the iterator holds without fetching another, and
`LocalRowIterator::has_received_batch` tells whether a record batch of the result has arrived, also
one without rows.

These raise PySpark's errors where the request was sent before: the requests of the DataFrame of a
`lateral_join` of another join type than an inner, a left outer or a cross join raise
`UNSUPPORTED_JOIN_TYPE`, those of a `DataStreamReader` given a name that PySpark's `name` refuses
`INVALID_STREAMING_SOURCE_NAME`, and those of the `changes` of a reader of a schema
`_LEGACY_ERROR_TEMP_1189`; `Catalog::create_table_with` and `create_external_table_with` of a
schema that is not a struct raise `NOT_EXPECTED_TYPE`; and the `await_termination` and
`await_any_termination` of a timeout that is not positive raise `VALUE_NOT_POSITIVE`, where they
raised `INVALID_TIMEOUT`, which PySpark does not have, and of a NaN, an infinite or a too large
timeout the errors of PySpark's `int(timeout * 1000)`. PySpark raises the errors of
`lateralJoin`, `name` and `changes` at the call; those builders return a DataFrame, not a
`Result`, so the plan of the DataFrame carries the error, which fails each request of it, and the
request sends nothing.

The same way, these raise PySpark's errors where the request was sent before: the requests of the
DataFrames of `repartition`, `coalesce` and `repartition_by_id` of a count that is not positive,
and of `repartition_by_expressions` of a negative count, raise `VALUE_NOT_POSITIVE`
(`repartition_by_expressions` of a count of 0 sends no count, as PySpark's `repartition(*cols)`);
those of `approx_quantile` of a probability below 0 or above 1 raise `NOT_EXPECTED_TYPE`, and of a
negative relative error `NEGATIVE_VALUE`; and those of `DataStreamReader::load` and of its format
methods of a path that is empty once Python's `strip()` takes its whitespace off raise
`VALUE_NOT_NON_EMPTY_STR`.

`DataFrame::replace`, `NaFunctions::replace` and `plan::na_replace` send both strings of each pair
as string literals, as PySpark's `replace` of strings sends them, so that a string that reads as a
number, such as `"1"`, `"nan"` or `"inf"`, is replaced in the string columns. Before, such a
string was sent as a double, which replaced the number in the numeric columns and left the strings
as they were, or failed the request with `CAST_INVALID_INPUT` when the replacement was not a
number. Replace numbers with `replace_literals`, such as
`replace_literals(vec![(LiteralExpression::Long(0), LiteralExpression::Long(9))], None)`: as
PySpark's `replace`, it sends an integer or a float as a double, where it sent it as it was, and
the requests of its DataFrame raise `MIXED_TYPE_REPLACEMENT` unless the values to replace and the
replacements that are not null are all booleans, all strings or all numbers, of which a boolean is
one too, as a Python bool is an int. `NaFunctions::replace_literals` does the same.

`DataFrameWriter::bucket_by` of a count that is not positive does as `clear_bucket_by` does, as
PySpark's `WriteOperation` sends the bucketing only of a count above 0; it sent the count, which the
server refused.

`StatFunctions::corr` takes the `method` of PySpark's `corr(col1, col2, method=None)`: `None` and
`""` are `"pearson"`, which the request sends, and another method is `VALUE_NOT_ALLOWED`;
`LogicalPlan::StatCorr` adds its `method` field. `DataFrame::col("*")` is the star of the
DataFrame, as PySpark's `df["*"]`. `functions::reduce` sends the function `reduce`, as PySpark's,
where it sent `aggregate`, and `reduce_with_finish` is added. `DataFrame::unpersist_opt` takes an
optional `blocking`, of which `None` sends none, as PySpark's `unpersist(None)`.
`SparkSession::create_dataframe_from_arrow` of batches of no rows and a schema sends the schema
alone, as PySpark's empty data, where it sent the batches, which the server failed to cast to a
schema of other columns. Without a schema, it sends batches of no rows with their Arrow schema,
where PySpark's `createDataFrame` of empty data, of a table too, raises `CANNOT_INFER_EMPTY_SCHEMA`
before it reads the schema of the table, as the Python client does: the batches carry their
schema, so there is none to infer.
`PythonUDFPayload::function` makes the payload of the function of a
streaming `foreach` or `foreachBatch` sink, which, as PySpark's, sends the command and the Python
version of a payload only, and not its output type and eval type. A window frame sends a bound of
0 as the current row, and a start of `Preceding(i64::MAX)` or before and an end of
`Following(i64::MAX)` or after as unbounded, as PySpark's `rowsBetween` and `rangeBetween` of
`-sys.maxsize` and `sys.maxsize`; an unbounded bound on the other side of the frame is the literal
of its int, and an offset of a row frame beyond an int fails the request with `VALUE_NOT_BETWEEN`,
where the server refused it.

`spark_connect::col` (`column::col`) of `"*"` or `"t.*"` is a star, as `functions::col` was, as
PySpark's `col`, and so are the names `"*"` and `"t.*"` that the methods take as Columns, by
`From<&str>` and `From<String>`, such as `select(["*"])`, where they were attributes of those names,
which the server failed with `UNRESOLVED_COLUMN`. `WindowSpec::partition_by` and
`WindowSpec::order_by` replace the partitioning or the ordering of the spec with their columns, as
PySpark's and Spark's do, where they appended them to those of the spec. The three-argument
`variant_set` and `try_variant_set` send `create_if_missing` of `true`, and the `tuple_union_*`
functions `lgNomEntries` of 12 and `mode` of `"sum"`, the defaults that PySpark sends, where they
sent no such argument.

`DataFrameReader::text` and `DataStreamReader::text` set the option `wholetext` to false, also over
an earlier one, as PySpark's `text` of its `wholetext=False`, so that they read lines, and a read
without a schema sends an empty one, as PySpark's readers do, which the server takes as none.
`DataFrameWriter::mode("errorifexists")` fails the write with `UNSUPPORTED_OPERATION` before it
sends anything, as PySpark Connect's `WriteOperation` does, where it wrote with
`SAVE_MODE_ERROR_IF_EXISTS`: `SaveMode::from_str` no longer takes `errorifexists`.
`Catalog::create_external_table` and `create_external_table_with` send a `CreateTable`, of a
`description` option as the description of the table, as PySpark's `createExternalTable` calls
`createTable`, where they sent a `CreateExternalTable`. The scalar operations of `Catalog`, such as
`current_database`, `table_exists`, `cache_table` and `drop_temp_view`, run as the execution of the
DataFrame of their plan, of its plan ID and of the tags of the thread, as PySpark's
`_execute_and_fetch` runs them, where they sent their relation without them.
`DataFrameReader::parse` of a JSON schema of a type that PySpark's `StructType.fromJson` does not
read, such as `null`, `calendarinterval` or `unparsed(...)`, gives a DataFrame whose requests raise
PySpark's `CANNOT_PARSE_DATATYPE`, or `UNSUPPORTED_DATA_TYPE` of an object of another type than an
array, a map, a struct and a UDT, and send nothing, as PySpark's `Parse` plan raises them when it is
built.

`SparkSession::create_dataframe` and `create_dataframe_from_columns` read the settings of the
session in one request, once they have converted the rows, as `create_dataframe_from_arrow` does, so
that a row that does not convert fails without a request, and upload their data when its size
reaches the server's `spark.sql.session.localRelationCacheThreshold`, as PySpark does, where data
below 64 KiB stayed inline. When `spark.sql.execution.arrow.useLargeVarTypes` is `true`, they send
the strings and binaries of the schema as `large_string` and `large_binary`, as PySpark does, and
the size that each path of local data compares with the threshold is that of pyarrow's `nbytes`, so
that a relation is inline or uploaded as PySpark's is. `create_dataframe_from_columns` of no rows
sends the schema alone, as PySpark's empty data, where it sent a Config request and Arrow data.

The deletion of a native ML model that is dropped is built on the thread that drops it, of its tags
and user context extensions, and sent reattachable through the hooks, as PySpark's
`_delete_ml_cache`, where a worker of the runtime built it without them. The removal of the cached
relation of a checkpoint that is dropped is one execution that is not reattachable, with no hooks
and no `ReleaseExecute`, as PySpark's `__del__` sends it, where it was reattachable and released.
The registration of the listener bus carries the tags of the session, as PySpark's
`execute_command_as_iterator`. The subqueries and table arguments of a DataFrame of another session
resolve, as their references are of the plan IDs of their own session, as PySpark's, where they
failed with `Missing relation in WithRelations`.

`DataFrameWriter::jdbc` and `DataStreamReader::xml` are added, as PySpark's `jdbc` of a writer and
`xml` of a stream reader, and so are the `approx_quantile`, `corr`, `cov`, `crosstab`,
`freq_items` and `sample_by` of `DataFrame`, which PySpark's DataFrame has besides `stat`, the
`fill_value` and `fill_map` of `NaFunctions`, of the values of `fillna_value` and `fillna_map`,
and `Column::outer` and `Column::transform`.

`TableArg::is_partitioned` tells whether `partition_by` or `with_single_partition` has partitioned a
table argument, as PySpark's `TableArg._is_partitioned`. `DataType::class_name` is the name of
PySpark's class of a type, `type(dt).__name__`, such as `LongType`, by which PySpark's errors name
it, and `DataType::simple_string` of the `VectorUDT` and `MatrixUDT` of `pyspark.mllib.linalg` is
`vector` and `matrix`, as of those of `pyspark.ml.linalg`, where it was `udt`.

`StructField::metadata` is a `types::Metadata`, where it was a `BTreeMap<String,
serde_json::Value>`: a JSON object, of its keys in their order and of its values as they are
written, an int beyond 64 bits and the `NaN` and `Infinity` of Python's JSON too, which a plan sends
as PySpark's `json.dumps(metadata)` writes it, where an int beyond 64 bits was sent as a float.
`Metadata::new()`, or `Default::default()`, is no metadata, and `Metadata::from` a `BTreeMap` or a
`serde_json::Map` of `serde_json` values, or `collect` of key and value pairs, makes metadata of
them, so a struct expression of `metadata: BTreeMap::new()` takes `Metadata::new()`, and one of a
map takes its `.into()`; `Metadata::from_json` reads the text of a JSON object, and `Metadata::json`
writes it. The hidden `Metadata::from_json_value`, which is not a stable API, reads the text of any
JSON value, of which another value than an object, such as a list, has no keys and is not empty,
and is written and sent as it is, as PySpark's field keeps and sends such metadata. `get` returns the value of a key as an owned `serde_json::Value`, where it returned a
reference: an int beyond 64 bits as its nearest float and a float that is not finite as null, of
which `get_json` gives the exact text. `insert` sets the value of a key at its place, or after the
others, and returns the value it had, `remove`, `contains_key`, `len` and `is_empty` are as those
of the map, and `keys` gives each key once in its order, where the map's were sorted. It has no
`iter`, `values`, `entry`, `get_mut`, indexing (`metadata["comment"]`) or `IntoIterator`: `to_map`
gives the map of its keys and `serde_json` values, to iterate in the order of the keys, or `keys`
and `get` in their order, and its `Debug` shows its JSON. It is `Clone`, `Default` and `PartialEq`,
as Python compares the dicts of its JSON, but not `Eq`, `Hash`, `Serialize` or `Deserialize`, which
the map was. `DataType::add` takes a map of `serde_json` values, as before. The Arrow schema of
`SparkSession::create_dataframe` and `create_dataframe_from_columns` of a `DataType::Udt` holds the
metadata of each field of its `sql_type`, at any depth (`json.dumps` of it, under
`SPARK::metadata::json`), as PySpark's `to_arrow_type` writes it, and that of no other field, as
before, and the names of the fields of its `sql_type` as they are, also of duplicates, which the
server then fails to read, as PySpark's, where they were deduplicated. The name of a float literal, and
the JSON of a float of an ML param, are Python's `repr` of it, of an even last digit of a tie of
its shortest digits, where a tie was rounded up (`74334700308107.625` is `74334700308107.62`).
`DataType::json`, and the new `StructField::json`, write the metadata as it is, of its keys sorted,
and each character of a string that is not printable ASCII as a `\u` escape, as PySpark's `json()`
writes them, where they wrote UTF-8; `json_value` holds the values of the metadata as `serde_json`
values. `DataType::from_json_str` and `StructField::from_json_str` read the JSON of Python's
`json.dumps`, of `NaN` and `Infinity` too, of the metadata as it is written, and the detail of their
`INVALID_JSON` is the message of their own parser.

`functions` adds a form of each optional argument of a PySpark function that had none, which sends
PySpark's request, such as `to_date_with_format`, `first_with_ignore_nulls`,
`from_json_with_options` (of name and value pairs) and `make_interval_with_parts` (of an `Option`
of each part).

The client decodes the responses of the server within the limit of prost's decoder, of 100 levels of
nested messages, and a deeper response is prost's error of its limit (`[Internal error] failed to
decode Protobuf message: ...: recursion limit reached`), also in a build of prost's
`no-recursion-limit` feature (which another crate of the build may enable, as the Python binding
does), where a response decoded at any depth and one of some thousands of levels ended the process:
the codec of the generated client, `apache-spark-connect-proto`'s `ClientCodec`, scans the bytes of
each response for the levels of their messages before prost decodes them, and so do the raw RPCs of
`SparkConnectClient` (`execute_plan_raw`, `reattach_execute_raw` and `analyze_plan_raw`) and the
reads of the `grpc-status-details-bin` of a failure. `apache-spark-connect-proto` adds a `server`
feature, off by default, which generates tonic's server bindings (`spark_connect_service_server`),
of the codec of the client: the generated server decodes the requests that it receives as prost
does, of no limit of the responses. `ClientCodec`, of its `ClientDecoder`, and its
`ClientDecodeError`, the source of the status of a response that the client fails to decode, which
`is_client_decode_error` tells, are new, as is `bytes_codec::ClientBytesCodec<U>`, of its
`ClientBytesDecoder<U>`, the `BytesCodec` of a client whose responses nest within the limit of the
responses of the client: a deeper one fails with the status of prost's error of the limit, tagged as
the client's (`ClientDecodeError`). So is `prost::Name` of each message of the proto crate, of its
full name in the package `spark.connect` and of the type URL
`type.googleapis.com/spark.connect.<name>`, so that `prost_types::Any::from_msg` packs it.

The core also adds doc-hidden items that the Python client uses, which are not a stable API, such as
the `plan_build` module of a `PlanBuildListener` of the process, `WeakSparkSession`,
`TableArg::relation_with_levels`, `SparkSession::add_unencodable_tag`, of a tag that UTF-8 cannot
encode, and `StreamingQuery::await_termination_ms` and
`StreamingQueryManager::await_any_termination_ms`; doc-hidden methods of v4.3.0 change their
arguments; and the doc-hidden `DataFrame::exclusive_relation_owners` of v4.3.0 is replaced by the
doc-hidden `gc` module, of the references to the parts of plans and expressions that values share
(`gc::PartRef`), which the doc-hidden `held_parts` of DataFrames, grouped data and cogroups,
statistics, table arguments, window specifications and writers give, as a binding shows them to its
garbage collector.

`LogicalPlan::Raw` and `SentExpression` add a doc-hidden `holes` field (`udf::SentHoles`), which is
not a stable API: the holes of the commands of the UDFs of a relation or an expression of a pickle
of the Python client, which a request fills with the messages that their picklers give when it
builds the plan, as PySpark's unpickled plan pickles the functions of its UDFs then. Code that
constructs the variant or the struct directly sets it to `Default::default()`, of no holes, of which
the relation or the expression is sent as before, and a pattern that names every field of either
adds `holes`, or `..`, as a pattern without `..` names every field. The doc-hidden
`expression::UnbuildableExpression` adds an `owner` field so, which keeps alive what the error of
the expression needs (`Expression::unbuildable_owned`); `Expression::unbuildable` gives none. The
doc-hidden `DataFrame::deep_copy_with_pickler_copies` of another session gives a copy of the plan,
of the copies of the picklers and of the Observations, whose nodes have in that session the plan IDs
that they have in the session of the DataFrame, and the root plan IDs within the `WithRelations` of
those of subqueries too, where it gave a DataFrame of the relation of the plan. The plan IDs of the
core are of the one counter of the process from which the doc-hidden `plan::fresh_plan_id` gives the
Python client the IDs of its own plans.

`SparkConnectClient::close` (and so `SparkSession::close` and `stop`) closes the channel of the
client, as PySpark's `close` closes its gRPC channel: a client that is open again after it (the
doc-hidden `SparkConnectClient::set_closed(false)` and `reopen`, and `SparkSession::reopen`) fails
each RPC with the `BuiltinValueError` of grpc's closed channel, `Cannot invoke RPC: Channel closed!`
of an execution or a reattach and `Cannot invoke RPC on closed channel!` of the others, where it
sent them, and the RPCs of the artifacts of a closed client fail with that `BuiltinValueError`
rather than `NO_ACTIVE_SESSION`, and so do the reattaches and the retries of a reattachable
execution that runs as its session is stopped, as by another thread, and its release, which is
logged (`ReleaseExecute failed with exception: Cannot invoke RPC on closed channel!.`), as
PySpark's iterator sends them through the stub that it was made with. As grpcio's `close` cancels
the calls of the channel, a call of the client, or of its clones, that is in flight when it is
closed, as by another thread, ends then, of `CANCELLED`: `Channel closed!` before its response
begins, and `Stream removed (Channel closed!)` while its response is read, as of an execution whose
next response the server has not sent. The stream of an execution of the client that is read after
the close, as of an iterator left open across `stop()`, fails at its next read so too: of
`CANCELLED` (`Stream removed (Channel closed!)`), and of no more of its responses, or of the
`ValueError` of grpc_status of a failure that the server ended it with before, whose
`google.rpc.Status` does not match the cancel, as PySpark's. `SparkSession::close` and `stop` (and
so `client.close()` and `stop()` of the Python client) first wait for the releases in the background
of the executions whose streams ended, at most 10 s, as PySpark's `close` waits for the release
futures that its iterators left it, so that the close of the channel does not cancel them: a release
that the server refuses, as of the session that `stop()` released, is logged once it is answered,
and the releases of a stream that is still read are not waited for. A signal of the process that
ends that wait, in the Python client, raises its exception and leaves the session open, as the
`KeyboardInterrupt` of PySpark's wait leaves its `close`; the doc-hidden
`SparkSession::close_at_once` closes at once, `close_after_releases` gives None when the wait was
ended so, of `SparkConnectClient::wait_for_releases`, and `has_releases_to_wait_for` tells whether
the wait has releases to wait for. A client that is marked closed
(`set_closed(true)`, `SparkSession::mark_closed`), as a response of another server-side session and
`INVALID_HANDLE.SESSION_CHANGED` mark it, keeps its channel open: it uploads artifacts, as PySpark's
ArtifactManager does, reattaches an execution that it reads, as PySpark's iterator, and sends its
RPCs again once it is open.
`SparkConnectClient::release_session` of a closed client raises `NO_ACTIVE_SESSION`. The client of
`with_new_session_id` and `clone_session`, and so the session of `SparkSession::new_session` and
`clone_session`, is of the `user_id` of the connection, else of the user of the requests of the
client that makes it, which the doc-hidden `set_request_user_id` sets (the user of the artifacts
stays that of the client). The doc-hidden `set_thread_request_context` gives the requests of the
current thread a session ID, a user and global user context extensions of their own, which
`client_session_id`, `request_user_id` and `global_user_context_extensions` give on that thread. The
doc-hidden `SparkConnectClient::with_new_session_id_of_user` and `clone_session_of_user`, and
`SparkSession::new_session_of_user` and `clone_session_with_id_of_user`, make the client or the
session of a new session or of a clone of the user that they are given, as it is, also when the
connection has a `user_id`, which `with_new_session_id`, `clone_session`, `new_session` and
`clone_session_with_id` take in its place. The doc-hidden `SparkSession::has_thread_tags` tells
whether the current thread has a set of tags, and `SparkSession::keep_thread_request_context` keeps
the context of the requests of the calling thread (the session ID, the user and the global
extensions of `set_thread_request_context`, the extensions of the thread and its tags) until the
guard that it returns drops, of `SparkConnectClient::thread_request_state` and
`restore_thread_request_state`. The doc-hidden `spark_connect::session::set_finalizer_request_hook`
sets, once for the process, the `FinalizerRequestHook` of the functions that a session calls on the
thread of a finalizer around the release of the relation of a checkpoint or of a cached ML object
whose last handle dropped: `before`, before the request is built, which is not sent when it returns
false, and `after`, once the thread has its context of before `before` again. The doc-hidden
`spark_connect_core::client::set_failure_requests_hook` sets, once for the process, a function that
a client calls on the thread that converts the failure of a call, an error of the server, which it
is given, before it builds the requests of the details of the error and of the configs of its stack
trace, and which returns whether to send them: when it does not, the error of the server is the
error of the call as it is, and an error of the function is the error of the call in place of it.
The per-thread state of a session and of its client (its tags, its user context extensions and its
request context) is that of `spark_connect_core::runtime::calling_thread_id`: a future of a call
that `block_on` polls on another thread, inside a `current_thread` runtime, reads and sets the state
of the thread that calls, which ends with that thread, and the future of a `Watch::Interrupt` runs
as a call of the thread that waits. A failed `SparkConnectClient::clone_session` (and so
`SparkSession::clone_session`) raises the error of the status of its RPC without the details and the
stack trace of the server, which it no longer requests, as PySpark's `clone` leaves its failure to
grpc, and its response is not checked against the session of the client (a session ID or a
server-side session ID of the response that differs is not an error, and does not close the client),
as PySpark's `clone` does not check it; the details of a failure are not requested once the wait
watch of the call returned a `Watch::Interrupt`. The doc-hidden `spark_connect::session::Tag` is a
tag of a thread, which the doc-hidden `SharedTags::new`, `replace` and `tags` take and give: `Text`,
`Unencodable`, of a str that UTF-8 cannot encode, and `NotText` and `NotUtf8`, tags that protobuf's
string field does not take, with which each execution of the thread fails.

The doc-hidden `plan::BuiltOutside` is a message that a client built outside the core, with the type
URLs of the placeholders of DataFrames in the encoding that `BuiltOutside::decoded` decodes it of:
the core reads the fields of its relations and commands and the messages of its extensions for
placeholders only when that encoding holds a type URL that is not of a placeholder at its
relations. The doc-hidden `DataFrame::from_relation`, `SparkSession::execute_plan_proto`,
`execute_command_proto` and `analyze_plan_proto` take `impl Into<BuiltOutside<_>>`, of which a
message passed as before is read everywhere, as it was; and the doc-hidden `plan::resolve_request`
takes one and returns the request, where it changed a request in place.

## Native ML objects: fetch, summaries, save and load (Rust API)

`spark_connect::ml` reads and writes the server-side objects of Spark Connect ML as PySpark 4.3.0
does over Spark Connect, for the typed model accessors, summaries and persistence of the native
classes to build on:

- `MlRef` is a server-side object: a fitted model (`Model::ml_ref`), or another object that a
  fetch returns, such as a tree of an ensemble. Its clones and the DataFrames whose plans read it
  share it, and the last owner to drop deletes it on the server, as PySpark's `RemoteModelRef`.
  `MlRef::of_reference(&session, id, release)` (not a stable API) is an object that the client
  does not own, which the Python client makes of the `RemoteModelRef` that a program gives a
  model's constructor: its last owner runs `release` in place of the deletion, which releases a
  reference of the `RemoteModelRef`. `MlRef::keep` (not a stable API) keeps the object on the
  server at the drop of its last owner, as the Python client keeps the object of a model whose
  `_java_obj` is assigned another value.
- `fetch(&MlTarget, method, args)` runs a method of an object, as `_call_java` of `pyspark.ml`
  does over Spark Connect (a `Fetch` command). `MlTarget::object(&model_ref).child("summary")`
  reaches the object of a method chain (`"<id>.summary"`), and `MlTarget::helper(&session)` the
  ML helper (`invoke_helper_attr`). Its result is a value (`MlValue`), new objects (an id list
  that names the trees of an ensemble is one object per tree), or a string. When the server has
  lost a summary with its offloaded model, the fetch recreates the summary of the target's summary
  dataset (`CreateSummary`) and runs once more, as PySpark does.
- `MlValue` is a value as `deserialize_param` reads it and `serialize_param` sends it: ML vectors
  and matrices, scalars, strings and arrays, byte for byte as PySpark's literals
  (`tests/golden/ml_values.json`). An empty array, or one whose first element is null, has no type
  to infer: send it as `FetchArg::Typed`, as PySpark sends a `(value, DataType)` pair.
- `attribute_relation(&MlTarget, method, args)` is the DataFrame of a method, as
  `invoke_remote_attribute_relation` reads it, of the summary dataset of a summary target. It keeps
  the object and the plans of its DataFrame arguments alive.
- `write_operator` and `write_object` save an estimator, transformer or evaluator, or a fitted
  model (`Write`), and `read` loads one of its JVM class (`Read`), as `RemoteMLWriter` and
  `RemoteMLReader` of `pyspark.ml.connect.readwrite`. A loaded model is a new `MlRef`.
- `Model::write()`, `Estimator::write()` and `Evaluator::write()` return an `MlWriter`, as
  PySpark's `write()` returns an `MLWriter`: `.overwrite()`, `.option(key, value)` and
  `.save(path)`. A model saves of the session that holds it. An estimator or an evaluator, which
  holds no session, saves of the one that `.session(&spark)` sets (as `MLWriter.session`), since
  the Rust client has no active session.
- The fit of the `Estimator` trait returns a `Box<dyn Model>`; `model.downcast_ref::<M>()` gives
  its concrete model type. `Model::describe()` is the server's description of a model (PySpark's
  `__repr__` of a model, its `toString`). `PredictionModel` gives `num_features()` and
  `predict(&vector)`, the members of every prediction model of `pyspark.ml`. PySpark reads a
  fetched int as a Python int of any size: `num_features()` (and the other `i32` members) of one
  out of the range of an `i32` is an error, not the truncated int.
- The server sends a warning with the model of a fit when the memory control of its ML cache
  stopped the training of a tree model early, as the model outgrew 2/5 of the session's
  `spark.connect.session.connectML.mlCache.memoryControl.maxModelSize` (`The random forest training
  stops early because the model size (35016 bytes) exceeds threshold (24000 bytes).`; the warnings
  of a fit a line each, of a boosting tree also the forest's of its tree). PySpark's
  `try_remote_fit` logs it, to the logger `pyspark.ml.util` at WARNING. The Rust API, which has no
  Python logging, keeps it on the model: `Model::fit_warning()` (and `MlRef::fit_warning()`), None
  of a fit without one, a load and a fetch, and of a pipeline model, a OneVsRest model and a
  validator's model, whose stages, class models and best model have theirs.
  `SparkSession::set_ml_fit_warning_handler` sets the handler that each fit of the session calls
  with its warning, on the thread of the fit (of the pool of a OneVsRest), once the result of the
  fit has arrived and before the model is made of it, as PySpark logs it; an error of the handler
  is the error of the fit. The Python client logs the warnings by it.
- A fit, transform, evaluate and save sends the params as `Params::sendable` serializes them, as
  PySpark's `serialize_ml_params` does: of the error of a value that PySpark cannot send, before
  any request. `Params::set_param_int_array` (of `VectorSlicer::set_indices` and
  `QuantileDiscretizer::set_num_buckets_array`) sends a list of ints as PySpark's literal of it,
  of the type of its first element (`Integer` when it fits 32 bits, else `Long`) of every element.
  Of a first element that fits 32 bits and another that does not, PySpark's `LiteralExpression`
  fails its `assert` of the element's range: the list is kept, for its getter, and is that
  `AssertionError`.
- `ObjectModel` is a model of a server-side object, of its operator and params, which transforms
  as every cached model does: the model of an object that `read_model` loads, or of one that a
  fetch returns. `training_summary(&model_ref, &dataset)` is the target of the training summary of
  a fitted model, of its summary dataset (the training dataset itself for the models whose
  `_summary_dataset` returns it, the classification models and the linear regressions;
  `summary_transform` of it, the transform of the dataset by the model of no params, for the
  others, as PySpark's default `_summary_dataset`). `evaluated_summary` is the target of the
  summary of an `evaluate`, of the id that the server returns, which the client does not release,
  as PySpark does not.

## Native ML classes of the full pyspark.ml API (Python)

The native Python classes of `pyspark.ml.connect` follow the API of `pyspark.ml` 4.3.0 over Spark
Connect in full, one for each class of the `__all__` of its classification, clustering,
evaluation, feature, fpm, recommendation, regression and tuning modules, and for `Pipeline` and
`PipelineModel` (`Evaluator` and `ParamGridBuilder` are PySpark's own).

They live in the package of PySpark's deprecated `pyspark.ml.connect` (deprecated since Spark 4.0,
of pandas and PyTorch training), as `pyspark.ml` itself stays PySpark's vendored classes over Spark
Connect, and replace its classes of the same names: `LogisticRegression`, `StandardScaler`,
`MaxAbsScaler`, `CrossValidator` and `Pipeline` (and their models), and its `RegressionEvaluator`,
`BinaryClassificationEvaluator` and `MulticlassClassificationEvaluator`. So the package and its
modules have the native classes and an `__all__` of them, but for `pyspark.ml.connect.pipeline`, the
module of `Pipeline` and `PipelineModel`, which has none (PySpark's `pyspark.ml.connect` exports
`Estimator`, `Evaluator`, `Model`, `Pipeline`, `PipelineModel`, `Transformer` and its modules, which
have no `__all__`); the constructors take the params of `pyspark.ml`, so the PyTorch params of the
deprecated `LogisticRegression` (`numTrainWorkers`, `batchSize`, `learningRate`, `momentum` and
`seed`) raise the `TypeError` of an unexpected keyword argument; and they fit, transform and
evaluate Spark DataFrames only, so a pandas DataFrame raises the `AttributeError` of its
`sparkSession`, where the deprecated classes fit it. The base classes `Estimator`, `Transformer`,
`Model` and `Evaluator` of the package, its `ArrayAssembler`, its `summarizer` and its other helper
modules stay PySpark's deprecated ones, unchanged. Each class has, where its PySpark class has them:

- **The Params API of `pyspark.ml.param.Params`, as PySpark's own code**: `params`, `getParam`,
  `hasParam`, `isSet`, `isDefined`, `hasDefault`, `getOrDefault`, `explainParam(s)`,
  `extractParamMap`, `set` and `copy`, the getters and setters, `setParams` and the `keyword_only`
  constructors are PySpark's own functions, over PySpark's param maps (`_paramMap` and
  `_defaultParamMap`) in the object's attributes. So their errors, messages and keywords are
  PySpark's, `_set`, `_setDefault`, `_copyValues`, `_resetUid` and `_input_kwargs` behave as
  PySpark's, values are converted by PySpark's `TypeConverters` (a NumPy string stays as given, an
  `np.float32` raises PySpark's TypeError of `toFloat`), `extractParamMap()` is in PySpark's order
  (the order that the params were set; the defaults of a new object in the order that its
  constructors set them), and an assignment of `uid` keeps the Params of the old uid, as PySpark's
  does. The maps of the objects that this client makes are in PySpark's order too: of the
  constructor of the class (the defaults in the order of its MRO, which is not of their names) of
  a model of no server-side object, of the model of an object that a fetch returns (a tree of an
  ensemble, `toLocal`), of a model of a class method (`StringIndexerModel.from_labels`), of the
  `Bucketizer` of a QuantileDiscretizer's fit (of its keywords), and of the model of a validator's
  or a OneVsRest's fit; of the model of a JVM estimator's fit as PySpark's `try_remote_fit` makes
  them, whose `_resetUid` rebuilds the defaults of the model's constructor in the order of the
  names of the params and whose `_copyValues` then adds the estimator's defaults that the model's
  class does not set and sets the estimator's params, each of their names (an ALSModel's
  `blockSize` and `predictionCol`, and then ALS's `coldStartStrategy`, `itemCol` and `userCol`);
  and of a copy of `Params.copy`, of the names. The defaults of a load are those of its class's
  constructor, of the names, as `_resetUid` rebuilds them. The set params of a load are of their
  names too, a deterministic choice: PySpark's `_set` of a load follows the server's map of them,
  which upb's iteration of a map gives in an order that differs from one process to another (the
  set params of a load of PySpark's Python format, a OneVsRest's or a validator's, are in the
  order of its metadata, as PySpark's). `clear` does nothing over Spark Connect, as PySpark's
  `JavaParams.clear`, of any positional arguments. The native object's params are made those of
  the maps before each native member reads them (a fit, a transform, an evaluation, a writer, a
  model's members), and so are
  those of the objects that it
  holds (an OneVsRest's classifier, a Pipeline's stages, a validator's estimator, evaluator and the
  values of its grids, a PipelineModel's stages, an OneVsRestModel's models, a validator model's
  best and sub-models). A writer reads them at its `save`, as PySpark's `RemoteMLWriter` reads
  those of its `_instance`.
  The maps of an object that this client makes (of a fit, a load, a native copy) are made of the
  native object's values at their first read. The native members are functions of PySpark's
  names, qualified names, modules, docstrings, signatures and annotations, of the class of
  PySpark's that defines the member (`JavaMLWritable.write`, `JavaPredictionModel.predict`), and so
  are the getters of the native properties (`LinearSVCModel.coefficients`,
  `JavaPredictionModel.numFeatures`, `_ClassificationSummary.accuracy` of a summary), the
  `__repr__` and `__str__` of a class (`Identifiable.__repr__`, `JavaModel.__repr__`) and the
  `__del__` of a Java wrapper (`JavaWrapper.__del__`, which releases a reference of the
  `RemoteModelRef` of a model of one, see "A model of a `RemoteModelRef`" below, and raises the
  `AttributeError` of a deleted `_java_obj`, which Python reports as unraisable in it, as
  PySpark's; the native object of a fit or a load releases its server-side objects itself), of the
  globals of PySpark's function, of which
  `typing.get_type_hints` resolves their annotations (`-> "LinearSVCModel"`). They are not
  PySpark's code, of which `inspect.getsource` gives none, nor wrappers of it: where PySpark's
  function is the wrapper of a decorator (`try_remote_attribute_relation`, `try_remote_write`),
  they have no `__wrapped__`. Python binds a call of one as it
  binds PySpark's function, of the parameters of the wrapper of its decorator (`JavaParams.clear`
  of `try_remote_intercept` takes any positional arguments, a DataFrame method of
  `try_remote_attribute_relation` any arguments): a call that does not bind raises PySpark's
  message (`JavaMLWritable.write() takes 1 positional argument but 2 were given`) before the
  native member runs. So do the class methods and the static methods, of PySpark's kind
  (`read`, of `JavaMLReadable.read` and the `try_remote_read` of a Pipeline's or a validator's,
  `StringIndexerModel.from_labels`, `StopWordsRemover.loadDefaultStopWords`), whose message of a
  class method counts its `cls` (`JavaMLReadable.read() takes 1 positional argument but 2 were
  given`). Such a function, which runs the native member of a reference to it of its class's and
  its own names, pickles of cloudpickle by value (as a property of one, a bound method of one, and
  a closure or a `functools.partial` of one, the function of a UDF), where PySpark's pickle by
  reference; a plain `pickle` of one of the class (`pickle.dumps(LinearSVCModel.summary)`), which
  pickles a function by its qualified name, finds PySpark's function there, and raises
  `PicklingError` (`... it's not the same object as
  pyspark.ml.classification.LinearSVCModel.summary`), where PySpark's pickles. The objects have one
  member of the native classes that PySpark's do not, which `dir()` lists: `__setstate__` of an
  object of params, which takes the state of a copy and of a pickle, of the objects that the native
  object holds of a PipelineModel, a OneVsRestModel and a validator's model (`stages`, `models`,
  `bestModel`, ...), and the state of the pickles of this client's earlier versions (the uid, params
  and defaults of the native object); and on Python 3.9 and 3.10, whose `object` has no
  `__getstate__`, `__getstate__` too, which gives PySpark's state of the object. The public class
  reaches the native state of an object by functions of the native module of its native class
  (`_ml_params`, `_ml_sync`, `_ml_java_obj`, ...), of no member. The attributes that PySpark's
  constructor, fit or load sets of each object (`uid`, `_paramMap`, `_defaultParamMap`, `_params`,
  `_java_obj`, `_summary`) are of the object, as PySpark's, not of the class: of an object that this
  client makes (of a fit, a load, a native copy), they are made at their first read (but the
  `_java_obj` of a model, there from its making), by the `__getattribute__` of the native class,
  which looks an attribute up as `object.__getattribute__` does (of the error of a getter that
  raises `AttributeError`, raised once) and makes one of them that it does not find (see "The
  attributes of the objects that this client makes" below); the native class has no
  `__setattr__` nor `__delattr__`, so `object.__setattr__` of an object of a subclass sets its
  attribute, as of PySpark's. The client makes such an object of its native params, which the
  native code that makes it may set after it makes it, and of which a copy of the native state
  sends no request; `dir()` of the object lists them, but `_summary` before its first read and a
  deleted `_java_obj` (see there). The class has PySpark's signature of its own `__new__`, and no
  `__signature__`, as PySpark's has none; a class of no Java wrapper (a Pipeline, a OneVsRest, a
  validator and their models) has no `__del__`, as PySpark's. The Param
  objects are PySpark's own `pyspark.ml.param.Param`, of the docs,
  converters and defaults of the official class (`crates/pyspark-rs/src/ml/ml_params.json`,
  captured from pyspark-client 4.3.0 by `scripts/capture_ml_api_golden.py`), and a getter per
  param (`getMaxIter`).
- **PySpark's types and class Params**: the class of `pyspark.ml.connect.<module>` is a subclass
  of its native class and of PySpark's class of the same name of `pyspark.ml.<module>`
  (`python/pyspark/ml/_native_classes.py`), so its objects are instances of every class of the MRO
  of PySpark's class: `MLWritable`, `MLReadable` and `Identifiable`, `HasTrainingSummary` and the
  summary bases, the param mixins (`HasMaxIter`), `PredictionModel` and `ClassificationModel`,
  PySpark's Java wrappers, and PySpark's class itself. Every object that this client makes (of a
  fit, a load, a copy, a summary) is of that class, which is a subclass of the class of the same
  module of each base of PySpark's class of the module (`LinearSVCSummary` of
  `LinearSVCTrainingSummary`, and both `BinaryLogisticRegressionSummary` and
  `LogisticRegressionTrainingSummary` of `BinaryLogisticRegressionTrainingSummary`), of PySpark's
  metaclass (`type` of the summaries). So `isinstance(LinearSVCModel(), MLWritable)` holds, and a
  Pipeline of `pyspark.ml` fits and saves native stages. A summary constructs as PySpark's
  `JavaWrapper.__init__(java_obj=None)`, PySpark's own function, with its errors of a call that
  does not bind: it keeps the `java_obj` in the summary's attributes (`_java_obj`, also of the
  `super().__init__` of a subclass), which its members read at each call as PySpark's
  `_extract_id_methods`: the id of an object of the ML cache and its methods (`"<id>.summary"`) of
  a string, over the active session, and PySpark's
  `AttributeError: 'NoneType' object has no attribute 'split'` of a summary of no object. A
  DataFrame member (`predictions`, `residuals()`) of a summary of the `summary` method first, as
  PySpark names the training summary of a fit, reads the plan of its `_summary_dataset`, which
  PySpark's fit sets, as the summary dataset of the relation, and raises the AttributeError of the
  attribute of a summary of none, as PySpark's `invoke_remote_attribute_relation` does. The Param of a class (`LinearSVC.maxIter`) is PySpark's, of
  the parent `undefined`, so `ParamGridBuilder().addGrid(LinearSVC.maxIter, ...)` builds.
  The AttributeError of a missing attribute names the class as PySpark's does, of an object as of
  the class itself (`type object 'DecisionTreeRegressor' has no attribute 'maxIter'`).
  `MLWriter`s and `MLReader`s are `RemoteMLWriter` and `RemoteMLReader`, as over Spark Connect.
  - They are plain objects of the attributes of PySpark's `__init__`: `_sparkSession`,
    `shouldOverwrite`, `optionMap` and `_instance` of a writer (`RemoteMLWriter(instance)`), and
    `_sparkSession` and `_clazz` of a reader (`RemoteMLReader(clazz)`), which PySpark's own
    `session`, `sparkSession`, `overwrite` and `option` set and read. So they construct, take any
    value of an attribute, and copy, copy deep and pickle at every protocol as PySpark's, of their
    class and attributes; as no module has the drop-in's class by its name
    (`pyspark.ml.connect.readwrite.RemoteMLWriter` is PySpark's), a pickle of one names the native
    module's `_new_remote_ml_writer` (`_new_remote_ml_reader`), where PySpark's names
    `copyreg.__newobj__` of the class.
  - Their classes are subclasses of PySpark's (`isinstance(w, RemoteMLWriter)` holds), not
    PySpark's classes themselves (`type(w) is RemoteMLWriter` is false), of the native classes.
  - Their `save` and `load` are functions of PySpark's names, docstrings and signatures
    (`RemoteMLWriter.save`, `RemoteMLReader.load`), which bind a call as PySpark's do, assert an
    active session first (also of a model, whose `Write` the session that holds it sends), and read
    the attributes at the call, as PySpark's do: a writer saves its `_instance`, of its params as of
    the save, and a reader loads its `_clazz`; another object or class than a native one is saved
    and loaded by PySpark's own `saveInstance` and `loadInstance`, with their errors
    (`NotImplementedError: Unsupported write for <class 'NoneType'>`). The `Write` of an estimator,
    a transformer, an evaluator or a model takes `shouldOverwrite` and `optionMap` as its protobuf
    fields take them: an int as a bool, None as an unset overwrite (false) and as no options, a
    dict (or pairs) of strings, and upb's error of anything else (`TypeError: bad argument type
    for built-in operation`). A save of PySpark's Python format overwrites of the truth of
    `shouldOverwrite`, of the ML helper's `handleOverwrite` of the value as it is, and the writer of
    a validator's model reads its `persistSubModels` of `optionMap`, as PySpark's do.
  - Their `save` and `load` (and the object's) take a path as PySpark's do. A path of an
    estimator, a model or an evaluator is the protobuf field of its `Write` or `Read`: bytes of
    their UTF-8, None the empty path (which the server refuses), and anything else (a
    `pathlib.Path` too) upb's `TypeError: bad argument type for built-in operation`. A path of the
    Python format of a Pipeline or a validator is that of its `os.path.join(path, "metadata")`: an
    `os.PathLike` (a `pathlib.Path`), and the TypeError of anything else.
  - A Python subclass's `load` reads the JVM class of its own name, as PySpark's `RemoteMLReader`
    names it (the module, of `org.apache.spark` for `pyspark`, and the name), which the server
    refuses (`CONNECT_ML.UNSUPPORTED_EXCEPTION`); a Pipeline's or a validator's loads its class.
- **PySpark's defaults of each instance**: the default `seed` is the `hash` of the name of the
  class, and the default `outputCol` the uid of the instance and `__output`; a fitted model has the
  defaults of its estimator, which PySpark's `_copyValues` copies to it (a seed of the name of the
  estimator's class), and a loaded object keeps the defaults of the uid that it was made of, as
  PySpark's `_resetUid` keeps them.
- **The members of the model**: its properties and methods (`coefficients`, `numFeatures`,
  `predict(vector)`, `clusterCenters()`, ...), which read the server's model object as PySpark's
  `_call_java` does, its setters, and `__repr__`, the server's description of the model. The
  `__str__` of an RFormulaModel names its resolved formula, of the server, as PySpark's does
  (`RFormulaModel(ResolvedRFormula(label=y, terms=[x], hasIntercept=true)) (uid=...)`). Its
  DataFrame methods (`LDAModel.describeTopics`, `Word2VecModel.getVectors` and `findSynonyms`,
  the LSH models' `approxNearestNeighbors` and `approxSimilarityJoin`, ALSModel's
  `recommendFor*`, and the `residuals` of a GeneralizedLinearRegressionSummary) send the
  positional arguments of a call only, as PySpark's `try_remote_attribute_relation` does over
  Spark Connect: they drop keyword arguments (`describeTopics(maxTermsPerTopic=1)` describes the
  server's 10 terms, `residuals(residualsType="pearson")` gives the deviance residuals), and check
  no arity, so a call of other arguments than the server's method takes fails on the server when
  the DataFrame is analyzed. The model of an object that a fetch returns (a tree of an ensemble,
  `toLocal`) is of the constructor of its class, as PySpark's
  `DecisionTreeClassificationModel(RemoteModelRef(...))`: of a new uid, of its defaults and no set
  params.
- **`_java_obj` and `_call_java`**: a model of a server-side object (of a fit, a load, a copy, a
  fetch or a class method such as `StringIndexerModel.from_labels`) has PySpark's `_java_obj`, the
  `RemoteModelRef` of its object (the one that it was made of, of
  `LinearSVCModel(RemoteModelRef(id))`, see "A model of a `RemoteModelRef`" below), and a summary
  the id of its object and methods
  (`"<id>.summary"` of a training summary, the server's id of the summary of an `evaluate`), so
  PySpark's own `_call_java(name, *args)` of either sends PySpark's Fetch (`try_remote_call`). A
  model with a training summary has PySpark's `_summary`, the summary. A training summary has the
  attributes that PySpark's `try_remote_fit` sets, in its order: `_java_obj`, `_summary_dataset` (a
  DataFrame of the summary dataset) and `_remote_model_obj`, the model's `_java_obj` itself, of a
  reference of its own (`add_ref()`, so `_ref_count` is 2 after a fit, as PySpark's), and the
  summary of an `evaluate` those of PySpark's `evaluate`: `_java_obj` and
  `__source_transformer__`, the model. The copies of a model share its `RemoteModelRef`, as
  PySpark's `copy.copy` and `Params.copy` share the `_java_obj` of its attributes, also of a copy
  made before either reads it. PySpark's Pipeline, OneVsRest, validators and their models are no
  Java wrappers, of no `_java_obj`, and neither are the native ones.
- **The attributes of the objects that this client makes**: the attributes of an object that
  PySpark's constructor makes (its param maps, Params, `_params` and uid, of `Params.__init__`,
  and `_input_kwargs` and `_java_obj`) are in `vars()` from its `__init__`, as PySpark's, and so
  are those of a PipelineModel, a OneVsRestModel and a validator's model of their constructors (but
  the objects that the constructors take, native properties, see "Native validators").
  Those of an object that this client makes in Rust (of a fit, a load, a native copy, a fetch) are
  made at their first read: its param maps, Params, `_params` and uid at the first read of one of
  them (by any member of the Params API, a native member, a copy or a pickle; a read of the uid
  alone reads the native object's), and its `_summary` at its first read, which makes the training
  summary object (of a request of `numClasses` of a logistic regression, as PySpark's fit makes
  it), so `vars()` of a fitted model, and `inspect.getattr_static` of it, lack them until then,
  where PySpark's fit and load set them, and `dir()` of it lists `_summary` from then on
  (`Params.params` reads every attribute that `dir()` lists, and so makes no summary). The state of
  a copy or a pickle has the param maps and lacks `_summary` until its first read, so a pickle of a
  model of a training summary and of no server-side object (of a deleted `_java_obj`) pickles,
  where PySpark's fails at the session of its summary dataset. The `_java_obj` of a model is in
  `vars()` from its making, as PySpark's. Their values are those of PySpark's whenever they are
  read (of the order of PySpark's maps, see above), but the Params of an object whose uid was
  assigned before they were made, which are of the uid then, where PySpark's are of the uid of the
  fit or the load, and do not belong to the object (`ValueError: Param ... does not belong to
  ...` of its getters). They are not made at the fit or the load, which makes the objects of many
  classes in Rust of their native params, so that a fit or a fetch (of the trees of an ensemble)
  reads no params of an object for `vars()` alone. The attributes are the state of the object, as
  PySpark's: an assignment or a deletion of the uid or of `_java_obj` (also of
  `object.__setattr__`, `object.__delattr__` or a write of `__dict__`) is of the native object at
  its next use (see "A model of a `RemoteModelRef`"), and a deleted `_java_obj` raises PySpark's
  AttributeError, is not listed by `dir()`, and is deleted in the copies and the pickles of the
  object, as PySpark's; one of the others deleted after it was made is made again at its next
  read, where PySpark's raises the AttributeError.
- **Summaries**: the training summary of a fit (`summary`, `hasSummary`), the same object on every
  call as PySpark's `_summary`, and the summary of `evaluate`; a loaded model has none, and raises
  PySpark's `RuntimeError`. Their DataFrames (`predictions`, `roc`, `cluster`, ...) are of the
  summary dataset of the fit. The `__repr__` of a GeneralizedLinearRegressionTrainingSummary is the
  server's description of it (its table of coefficients and its deviances), as PySpark's is. Their
  members have PySpark's docstrings, and their methods PySpark's signatures: a member is a property
  of a property of PySpark's (`r2`), and a method of a method (`fMeasureByLabel(beta=1.0)`).
- **Fits of param maps**: `fit(dataset, paramMap)` fits a copy of the estimator with the map, a
  list or tuple of maps fits once per map, and `fitMultiple(dataset, paramMaps)` returns PySpark's
  `_FitMultipleIterator`. `transform(dataset, paramMap)` transforms of a copy of the model.
- **The warning of a fit**: the warning that the server sends with the model of a fit, of a tree
  training that the memory control of its ML cache stopped early (see "Native ML objects: fetch,
  summaries, save and load (Rust API)"), is logged as PySpark's `try_remote_fit` logs it:
  `_logger.warning(warning_msg)` of `pyspark.ml.util`, to the logger `pyspark.ml.util` at WARNING,
  of the server's message as it is, of a record of that line of `pyspark/ml/util.py` (its
  `pathname`, `lineno` and `funcName` `wrapped`), once the result of the Fit has arrived and before
  the model is made of it (before the trees of a forest are read). So each fit logs its own: of a
  param map, of each stage of a Pipeline, of each class of a OneVsRest, on the thread that fits it,
  and of each fit of a validator. An exception of the logging (of a filter that raises) is the
  exception of the fit, as PySpark's.
- **Persistence**: `write()` (an `MLWriter` of `pyspark.ml.util`, with `overwrite()`, `option`,
  `optionMap`, `session` and `save`), `save(path)`, `read()` (an `MLReader`) and `load(path)`, as
  `RemoteMLWriter` and `RemoteMLReader` save and load them, so that this client and PySpark load
  what each other saves.
- **Subclasses**: a Python subclass of a class behaves as a subclass of PySpark's class.
  - Its `__init__` passes its keywords to `super().__init__(...)`, which sets them as PySpark's
    `keyword_only` `__init__` does (by `setParams`, recording `_input_kwargs`), of any arguments of
    its own. A call that does not bind raises PySpark's message (`LogisticRegression.__init__() got
    an unexpected keyword argument 'foo'`, `JavaModel.__init__() takes from 1 to 2 positional
    arguments but 3 were given`).
  - An object of a subclass of an estimator, transformer or evaluator, or of a model class of
    PySpark's `JavaModel`, has a uid of the subclass's name (`Identifiable._randomUID`), and the
    defaults of that uid and name (`outputCol`, `seed`), and so does an object of the constructor
    of a subclass of PipelineModel, OneVsRestModel or a validator's model
    (`MyPipelineModel(stages)`). The models of a fit are of their base class, of its name, as
    PySpark's.
  - `fit`, `fitMultiple`, `transform`, `evaluate`, `save` and `load` are PySpark's own (of
    `Estimator`, `Transformer`, `Evaluator`, `MLWritable` and `MLReadable`), with PySpark's
    signatures and docstrings. They run the object's `_fit`, `_transform`, `_evaluate`, `copy`,
    `write` and `read`, so a subclass's `_fit` runs, also in the copies that a Pipeline or a
    CrossValidator fits.
  - A `copy` is of the class of the object, of its attributes, as PySpark's `Params.copy` makes it
    of `copy.copy`.
  - `uid` is an attribute of the object (`vars()` has it), which an assignment sets, as PySpark's:
    it keeps the Params of the old uid, which then do not belong to the object (until its
    `_resetUid`), so the getters, `isSet` and `explainParam` of a Param raise PySpark's
    `ValueError` (`... does not belong to ...`), and the setters and the calls of a name set and
    read the maps. It is a native property over the attribute, whose value the native members
    send: `del obj.uid` raises `AttributeError: property has no deleter`, where PySpark's deletes
    the attribute (and its next read raises the `AttributeError` of a missing attribute).
  - A fit, transform, evaluate and save send every param of `_paramMap`, as PySpark's
    `serialize_ml_params`: a Param of the subclass that is set too, of the literal of its value, which
    the server refuses (`Param myParam does not exist.`), as PySpark's; a Param of only a default is
    not sent. The error of a value of no literal, of the first param of `_paramMap` that cannot be
    sent, is raised before any request.
  - A save of a subclass of a OneVsRest, a Pipeline, a validator or a model of one writes the class
    of the object in its metadata (`__main__.MyPipeline`), as PySpark's `DefaultParamsWriter` does,
    and the Params of the subclass: the params of its `extractParamMap()` (of a OneVsRest and a
    validator, but the classifier, the estimator, the evaluator and the grid, which are saved of
    their own; a Pipeline's are its `stageUids` and `language`), and the defaults of its
    `_defaultParamMap`, as PySpark's writers write them. So the loads of such a file are PySpark's:
    the readers of a OneVsRest and a Pipeline load it of their own class, those of a validator fail
    of PySpark's `TypeError` (they read the grid of a class of no `pyspark.ml.` as of the JVM's
    writer), and a stage of a pipeline loads of its class (an `AttributeError` of a class that its
    module does not have).
- **`copy.copy`, `copy.deepcopy` and `pickle`** take an object as PySpark's do:
  - An estimator, a transformer, an evaluator, a Pipeline or a validator, and a model of no
    server-side object, copy, copy deep and pickle, at every protocol, of their uid, params and
    attributes. A shallow copy shares the values (a Pipeline's stages), and a deep copy and a
    pickle copy them. Their state (`__getstate__`) is PySpark's, the attributes of the object
    (`__dict__`, of its param maps and its uid), as `object.__getstate__` gives it (of a subclass
    of slots, the pair of the attributes and the slots), which `__setstate__` takes as
    `copy._reconstruct` and the unpickler take PySpark's, of `__dict__.update`. As PySpark's class
    has no `__copy__`, `copy.copy` (and `Params.copy`, of `copy.copy`), `copy.deepcopy` and
    `pickle` of an object of a Python subclass of its own `__reduce__`, `__reduce_ex__`,
    `__getnewargs__`, `__getstate__`, `__setstate__` or slots, or of a class of a reducer of
    `copyreg.dispatch_table`, run them as of PySpark's subclass: a subclass whose `__getstate__`
    drops a lock of `__dict__` (and whose `__setstate__` makes it again) keeps its uid, params and
    attributes, alone and in a Pipeline. `__setstate__` also takes the state of the pickles of this
    client's earlier versions (the uid, the set params and the own defaults of the object by name,
    and its attributes).
  - `copy.copy` copies of the reduce, as of PySpark's objects, which have no `__copy__`: an object
    of the class's `__new__`, of no `__init__`, of the attributes of the object (its param maps,
    and the `stages`, `models`, `bestModel`, ... of a PipelineModel, a OneVsRestModel and a
    validator's model, which their state carries after the others, as PySpark's attributes);
    a model of a server-side object is copied of the same object, of the native copy that its
    reduce names after a lock (`_column_reduce.unpickled_after`), at which its pickle and its
    deep copy fail, as PySpark's fail at the lock of its `RemoteModelRef`. So a failed deep copy of
    such a model makes no object, where PySpark's makes one of the class's `__new__` before it
    fails at the lock of the state, whose `__del__` prints the unraisable `AttributeError` of its
    read of `_java_obj`. A model of a server-side object of a Python subclass of a reduce, a state
    or slots of its own (as above) reduces as PySpark's, of `object.__reduce_ex__` and the
    attributes of the object, which run the subclass's members: its copy is of its class and
    slots, a state of its own (one that drops `_java_obj`) copies and pickles of that state, and
    a failed deep copy prints the unraisable error of PySpark's. A summary is copied of the class's
    `__new__` and its attributes, whose members read its `_java_obj`, as PySpark's.
  - A model of a server-side object, and its summary, copy, sharing the object, and are not
    copied deep nor pickled, nor is a DataFrame that keeps such a model: PySpark's raise the error
    of the first object that does not pickle, of the pickle of the lock of the model's
    `RemoteModelRef` at the protocol (a `PicklingError` below 2, the TypeError
    `cannot pickle '_thread.lock' object` from 2), or of the session of a DataFrame, which PySpark
    pickles after its plan (the AttributeError of its local function). A transform of a model, a
    DataFrame that reads one (also only through a subquery of its expressions, `exists()`,
    `scalar()` or `isin(df)`, as the references of PySpark's plan keep the plans of its subqueries,
    after its child), and a DataFrame of a method of a model or of the summary of an `evaluate` keep
    the model in their plans, before their sessions; the summary of an `evaluate` keeps its model
    (`__source_transformer__`). A training summary pickles its summary dataset
    first, which keeps the model when it is a transform of a model (the transform of the training
    dataset by the model, of the KMeans, BisectingKMeans and GaussianMixture summaries, and a
    training dataset of a stage after a model of a Pipeline), and else raises the AttributeError
    of the session of the training DataFrame, and so does a DataFrame of a method of the summary
    (`summary.predictions`). Cloudpickle pickles the session, of its function by value, and so
    raises the TypeError of the lock of the model of the summary (`_remote_model_obj`) then.
    `copy.deepcopy` raises the TypeError of the lock, after PySpark's requests of the schemas of
    the DataFrames that it copies on the way (the lookup of `__deepcopy__` of PySpark's DataFrame
    reads its columns).
  - So are the objects that keep the plan of such a DataFrame, as PySpark's pickle reaches the
    plan in them: a Column of a subquery of it (`exists()`, `scalar()`, `isin(df)`, also within
    other expressions) and its expression (`Column._expr`), a table argument of it (`asTable()`,
    also partitioned, ordered or of a single partition) and its `_subquery_expr`, the plan of the
    DataFrame (`DataFrame._plan`), and its grouped data, cogrouped data and writers (`groupBy`,
    `cogroup`, `write`, `writeTo`), of which PySpark pickles the plan before the session. Their
    pickle raises PySpark's error of the lock of the model, or that of the session of a training
    summary's `_summary_dataset` (of `pickle`) and of the lock (of cloudpickle), and
    `copy.deepcopy` the TypeError of the lock; `copy.copy` copies them, sharing the model. Cogrouped
    data pickle their two grouped data in their order, each of its DataFrame, of its plan and then
    of its session: a plain DataFrame first fails at its session (the AttributeError of its local
    function), at every protocol, before the model of the second, and a deep copy requests the
    schemas of their DataFrames before the TypeError of the lock, as PySpark's.
  - The pickle of a plan or an expression reaches the functions of its UDFs and the objects of
    `pyspark.ml` that it keeps in PySpark's order of their plan objects: of a plan, its first input
    (`_child`), then the plans of the subqueries of its expressions (`_references`), its other
    inputs (the right side of a join), its expressions, of the functions of their UDFs, and last the
    objects that it keeps; of a grouped map (`applyInPandas`, `applyInArrow`,
    `applyInPandasWithState`), its grouping columns before its function, of a cogroup its other
    side between them, and of a `transformWithState` its initial state and its grouping columns
    after its function; of a Column, its expression in its order, the plan of a subquery where it
    is. So a plain pickle fails at the lambda of a UDF that comes before the model
    (`df.select(udf_of_a_lambda("x")).where(transformed.exists())`,
    `(udf_of_a_lambda("x") > 0) & transformed.exists()`), and at the lock of the model where the
    model comes first (`transformed.select(udf_of_a_lambda("x"))`), as PySpark's, and cloudpickle,
    which pickles the lambda, at the lock.
  - The DataFrames of the models of PySpark's own `pyspark.ml` classes, which build PySpark's
    plans over the drop-in's DataFrames (`pyspark.ml.regression.LinearRegression`, where the
    drop-in's own classes are those of `pyspark.ml.connect`), are refused as PySpark's: a DataFrame
    of such a plan (`DataFrame(plan, session)`) reaches what PySpark's pickle of the plan reaches,
    of the models and the summaries that the plan keeps (`__source_transformer__`,
    `__source_instance__`), its child first (the plan of a DataFrame of the drop-in, of the
    functions of its UDFs, at which a plain pickle fails first). So a transform
    of such a model, and a DataFrame of a method of it or of the summary of its `evaluate`, raise
    the error of the lock of the model, and a DataFrame of a method of its training summary the
    AttributeError of the session of the summary's `_summary_dataset` (of `pickle`) and the
    TypeError of the lock of its `_remote_model_obj` (of cloudpickle and `copy.deepcopy`), or the
    lock of a summary dataset that is a transform of the model (of a clustering model).
  - A summary of PySpark's constructor (`LinearRegressionSummary("<id>.summary")`), of no
    server-side object, copies, copies deep and pickles, at every protocol, as PySpark's plain
    object: of its class and of its attributes (`_java_obj`).
  - A shallow copy of a model keeps the server-side object while either object lives (the native
    objects hold it, and its last owner deletes it). PySpark's `copy.copy` shares its
    `RemoteModelRef` without a reference of its own, and the `__del__` of each object releases one,
    which deletes the server-side object at a count of 0: the collection of one copy of a model of
    no training summary deletes the object of the original (`CONNECT_ML.CACHE_INVALID` at its next
    use), and that of a model with one, whose summary holds a reference too, survives one copy's
    collection and not two (of copies of the model or of its summary), where this client's
    original keeps working. A copy of a Python subclass of a reduce of its own, which copies as
    PySpark's of the attributes of the object (see above), holds the `RemoteModelRef` of its
    `_java_obj` as a model of it, and so releases it as PySpark's copy does.
- **A model of no server-side object**: a model class constructs as PySpark's
  `JavaModel(java_model=None)` (`KMeansModel()`, `LDAModel()`), of PySpark's `JavaModel.__init__`,
  a model of a new uid of its class and of its params, none set, whose Params API, `copy` and
  `write()` work, and whose members of the server, `__repr__`, a transform and a save raise
  PySpark's bare `AssertionError`, as does the
  `params` walk of a model of `functools.cached_property` members (`trees`), and so its
  `explainParams` and `copy`.
- **A model of a `RemoteModelRef`**: a model class constructs of the `RemoteModelRef` of a
  server-side object as PySpark's `JavaModel(java_model)` over Spark Connect
  (`LinearSVCModel(RemoteModelRef(id))`): a model of a new uid of its class and of its params,
  none set, whose `_java_obj` is the reference, of no reference more, and whose members, `__repr__`,
  transforms and saves read the object of its id in the ML cache, of no training summary: of the
  session that is active when it is made, as the members of this client's models are of their
  model's session (see the native ML section below), where PySpark's `_call_java` reads the active
  session of each call. Of a model made on a thread of no active session, they read the session of
  each call, as PySpark's: a member and a save of the active session, which raise PySpark's bare
  `AssertionError` of none, and a transform of the dataset's.
  So does a model whose `_java_obj` is assigned a `RemoteModelRef`, as of the
  `super().__init__(java_model)` of a Python subclass of an `__init__` of its own, whose
  `JavaWrapper.__init__` sets it, and also a model of another server-side object (of a fit, of
  another reference), which gives its object up, as PySpark's model is of its `_java_obj`: it no
  longer releases the reference before, nor deletes the object of its fit, which stays in the ML
  cache until the session ends, as PySpark's `__del__` releases a reference of its `_java_obj`
  only. A model whose `_java_obj` is assigned None (or another value) is of no object (see "A model
  of no server-side object" below), and one whose `_java_obj` is deleted raises PySpark's
  `AttributeError` of the attribute at a read of it, of its members and in its `__del__`, of no
  release. The model keeps its object in the ML cache as PySpark's: the `__del__` of the model
  releases a reference of the `RemoteModelRef` (`release_ref`) at each call, when it is collected
  and at an explicit call, as PySpark's, of the model alone, and else the last of its transforms,
  which keep the object as PySpark's transforms keep their model (`__source_transformer__`),
  releases one when it is gone, if no `__del__` did (an explicit `__del__` of a model whose
  transform lives releases none, where PySpark's releases one at each call); the release deletes
  the object at the last reference, through the active session. An error of the release
  (an `AssertionError` of a reference released too often) is unraisable in the `__del__` of the
  model's class (`JavaWrapper.__del__`), as PySpark's. A model of a reference in a reference cycle
  (`ref.owner = model`) is collected, and its `__del__` releases the reference, as PySpark's.
  `Params.copy` adds a reference (`add_ref`), which the copy releases, and `copy.copy` adds none and
  releases one too, as PySpark's copies do, so the collection of a shallow copy of a model of a
  reference of one deletes the object of the original. Another `java_model` raises PySpark's bare
  `AssertionError` (of its `assert isinstance(java_model, RemoteModelRef)`).

`tests/golden/ml_api.json` holds the API of every public class of `pyspark.ml`, and
`python/tests/test_ml_api_offline.py` compares each native class with it: the members, the kinds of
the members, and the parameters of each method and keyword-only constructor.
`tests/golden/misc_api_cases.json` holds the same for NaiveBayes, MultilayerPerceptronClassifier,
FMClassifier and OneVsRest, from `python/tests/_misc_api_cases.py` and
`scripts/capture_misc_api_cases_golden.py`, and `test_misc_api_offline.py` and
`test_misc_api_e2e.py` compare them.
`tests/golden/ml_api_cases.json` holds the results of PySpark 4.3.0 of the cases of
`python/tests/_ml_api_cases.py` (the Params API, the models, summaries, transforms and persistence),
which `test_ml_api_offline.py` and `test_ml_api_e2e.py` compare with.
`tests/golden/ml_unbound_models.json` holds the models of no server-side object of every model class
of PySpark 4.3.0 of `JavaModel(java_model=None)` (`python/tests/_ml_unbound_cases.py`,
`scripts/capture_ml_unbound_models_golden.py`), which `test_ml_api_e2e.py` compares with.

The tree classes have their own cases, `python/tests/_tree_api_cases.py` and
`tests/golden/tree_api_cases.json`, with `test_tree_api_offline.py` and
`test_tree_api_e2e.py`.

## Native regression models (Rust API)

Each regression estimator of `spark_connect::ml` fits its typed model, as PySpark's `fit` returns
one: `LinearRegression::fit` returns a `LinearRegressionModel` (the fit of the `Estimator` trait
boxes it). A member of a model is a fetch of its server-side object, as `_call_java` of PySpark
4.3.0 reads the member, and returns the type that PySpark returns:

- `LinearRegressionModel` and `GeneralizedLinearRegressionModel`: `coefficients`, `intercept` (and
  `scale` of a linear regression), `evaluate(&dataset)`, and the training summary of a fit,
  `has_summary()` and `summary()`. As PySpark's fit over Spark Connect, the fit sets up the
  training summary client-side, of the training dataset, so that the server can recreate a lost
  summary. A loaded model has no training summary.
- Their summaries implement `LinearRegressionSummaryApi` (and `TrainingSummary` of the training
  summary) and `GeneralizedLinearRegressionSummaryApi`, whose training summary also has its solver,
  its iterations, the statistics of the coefficients and `describe()`. A DataFrame member, such as
  `predictions` or `residuals(residuals_type)`, is an attribute relation of the summary. The
  summary of `evaluate` is its own server-side object, which the client does not release, as in
  PySpark.
- `AFTSurvivalRegressionModel` (`coefficients`, `intercept`, `scale`, `predict_quantiles`),
  `IsotonicRegressionModel` (`boundaries`, `predictions`, `num_features`, and `predict` of a single
  feature value) and `FMRegressionModel` (`intercept`, `linear`, `factors`).
- `DecisionTreeRegressionModel` implements `DecisionTreeModel`, and `RandomForestRegressionModel`
  and `GBTRegressionModel` implement `TreeEnsembleModel`, of trees that are each a
  `DecisionTreeRegressionModel` of its own server-side object. `GBTRegressionModel` has
  `evaluate_each_iteration(&dataset, loss)`.
- Every model but the isotonic one implements `PredictionModel` (`num_features`, `predict`), and
  every estimator and model has `load(&session, path)`, as PySpark's `load`, besides `write()`.

The values are the server's, and the members are those that the server allows of each model
(`MLUtils` of Spark 4.3.0). The native Python classes of `pyspark.ml.connect.regression` have them
too, with PySpark's names: their fit returns the typed model.

## Native clustering models (Rust API)

Each clustering estimator of `spark_connect::ml` fits its typed model, as the regression ones do
(see "Native regression models (Rust API)"): `KMeans::fit` returns a `KMeansModel`.

- `KMeansModel` and `BisectingKMeansModel`: `cluster_centers()` (the rows of the server's
  `clusterCenterMatrix`, as PySpark reads them), `num_features`, `predict` of a cluster index,
  `compute_cost(&dataset)` of a bisecting model (which PySpark deprecates since 3.0.0, in favor
  of `ClusteringEvaluator` and of the training summary's cost), and the training summary of a
  fit.
- `GaussianMixtureModel`: `weights`, `gaussians_df()`, `gaussians()` (a `MultivariateGaussian` of
  each collected row of `gaussiansDF`, as PySpark builds them client-side), `predict`,
  `predict_probability` and the training summary of a fit.
- The training summaries implement `ClusteringSummary`: `KMeansSummary` and
  `BisectingKMeansSummary` (`training_cost`), and `GaussianMixtureSummary` (`probability_col`,
  `probability`, `log_likelihood`).
- `LDAModel`, `LocalLDAModel` and `DistributedLDAModel` implement `LDAModelApi`
  (`describe_topics`, `topics_matrix`, `log_likelihood`, `log_perplexity`, ...). As PySpark's
  `LDA._create_model`, a fit is local or distributed by its optimizer, and
  `LDAModel::into_local()` or `into_distributed()` gives the model of its class. A distributed
  model has `to_local()`, `training_log_likelihood`, `log_prior` and `get_checkpoint_files`.
- Every estimator and model has `load(&session, path)`. `PowerIterationClustering` has the uid of
  the algorithm, and `write()` and `load` save and load it as PySpark does: the params of the
  server's `PowerIterationClusteringWrapper`.

The native Python classes of `pyspark.ml.connect.clustering` have them too, with PySpark's names:
their fit returns the typed model, and `LDAModel` is the base class of the two LDA models, as
PySpark's.

## Native LogisticRegression

The native `spark_connect::ml::LogisticRegression` fits Spark's JVM
`org.apache.spark.ml.classification.LogisticRegression` through the Connect ML
protocol. Its training features must be ML vectors, in the Rust API as in Python; convert an
array column with `array_to_vector` before fitting. The Python class,
`pyspark.ml.connect.classification.LogisticRegression`, follows
`pyspark.ml.classification.LogisticRegression`, PySpark's standard API over Spark Connect, not the
deprecated PyTorch class of `pyspark.ml.connect`. Its keyword-only constructor takes the 21 params
of `pyspark.ml`'s, with its defaults and `TypeConverters`: `featuresCol`, `labelCol`,
`predictionCol`, `maxIter`, `regParam`, `elasticNetParam`, `tol`, `fitIntercept`, `threshold`,
`thresholds`, `probabilityCol`, `rawPredictionCol`, `standardization`, `weightCol`,
`aggregationDepth`, `family`, the matrices `lowerBoundsOnCoefficients` and
`upperBoundsOnCoefficients`, the vectors `lowerBoundsOnIntercepts` and `upperBoundsOnIntercepts`,
and `maxBlockSizeInMB`. It has a setter of each, and `setParams`. A fit sends a matrix or a vector as PySpark's ML
serializer sends a `DenseMatrix`, a `SparseMatrix` or a vector; in the Rust API,
`Params::set_param_matrix` sends an `MlMatrix` so. The params of the PyTorch training of the
deprecated class (`numTrainWorkers`, `batchSize`, `learningRate`, `momentum` and `seed`) raise the
`TypeError` of an unexpected keyword argument. As `pyspark.ml`'s, the constructor and
`setParams` raise PySpark's `ValueError` when the `threshold` and the `thresholds` that are set
disagree (`setParams` leaves the params that it is given set), and
`setThreshold` and `setThresholds` leave the other one set, as they do over Spark Connect, where
the `clear` of `pyspark.ml` does nothing: a fit then sends both, and the JVM raises its error when
they disagree. Its model outputs the columns of the JVM model, as `pyspark.ml`'s does: a
`rawPrediction` and a `probability` vector, and a `double` prediction. The JVM checks the labels
(an integer in `[0, 100)`, not null), the types of the columns, and the output columns that the
DataFrame has already, and raises its errors, as with `pyspark.ml`: the client makes no checks of
its own, and sends no query of the labels.

Fitting returns a server-cached model. Prediction uses that model's object
reference and must use the same Spark session. Models are obtained through
`Estimator::fit`; direct struct construction of `LogisticRegressionModel` is no
longer supported. The cached model stays available while the model, a clone of
it, or a DataFrame that it transformed is alive, including when such a DataFrame is
executed later. When the last of them is dropped, the cached model is deleted on the
server, as `pyspark.ml` does, without waiting for the server; a fit that is dropped
at once leaves nothing cached. A Python model of a `RemoteModelRef` that a program gives its
constructor releases a reference of the `RemoteModelRef` instead, as PySpark's model does, which
deletes the object at the last (see "A model of a `RemoteModelRef`" above).

The Rust `transform` of a fitted model rejects a DataFrame of another session. In
Python, as with PySpark, it returns a DataFrame whose analysis or execution fails
on the server of that session with `CONNECT_ML.CACHE_INVALID`.

In Python, a fit, its training summary and the members of its model (`predict`, `__repr__`, the
summary's DataFrames) are sent over the session of the dataset and of the model, so ML works on a
session that is not the active one (a second session of
`SparkSession.builder.remote(...).create()`) and on a thread of no active session. This is a
deliberate difference from PySpark, which sends the members of a model and of a summary
(`_call_java`, the DataFrame methods of `try_remote_attribute_relation`) over
`SparkSession.getActiveSession()`, and its fits of the dataset's session (`dataset.sparkSession`):

- On a session that is not the active one, PySpark's members of a model and of its summary, the
  summary's DataFrames and the model's `__repr__` fail with `CONNECT_ML.CACHE_INVALID` (`Cannot
  retrieve Summary object ... from the ML cache`), and so do the fits that read a member of their
  model: of LogisticRegression and RandomForestClassifier, whose `_summaryCls` reads `numClasses`,
  and of GBTClassifier, whose params walk of `_resetUid` reads its `trees`. The fits of the others
  (LinearRegression, DecisionTreeClassifier, KMeans, StringIndexer, ...) and the transforms succeed.
- On a thread of no active session (a thread that the program starts, as of `fitMultiple` shared by
  threads), the same members and fits raise PySpark's bare `AssertionError` (its `assert session is
  not None`), where this client's run.
- PySpark deletes a dropped model through the active session (`del_remote_cache`): a model of a
  session that is not the active one is not deleted from the ML cache of its session, and one
  dropped on a thread of no active session neither. This client deletes it in its own session (see
  "The removal of the relation of a dropped checkpoint, and the deletion of a dropped model" in
  `dev/parity/known-differences.md`).

A model used from a session other than its own fails on both.

The native estimators, transformers and evaluators of `pyspark.ml.connect` take
keyword-only constructor parameters. The evaluators follow standard
`pyspark.ml.evaluation`: multiclass classification defaults to `metricName="f1"`,
and binary classification uses `rawPredictionCol="rawPrediction"`.
Each value is converted by the `TypeConverters` of its param in
PySpark (an integral float for an int param, for example), and a value that the
converter rejects raises its `TypeError`. A fit, transform or evaluate sends only the
params that the constructor or a setter set, so the server applies its defaults to
the others, as with PySpark. An int beyond a long raises PySpark's
`VALUE_NOT_BETWEEN` at the next fit, transform or save. A list of ints (`layers`, `indices`,
`numBucketsArray`) is sent as PySpark's literal of it, of the type of its first element (an
integer, else a long) of every element: a list of an element out of the range of that type raises
PySpark's bare `AssertionError` at the next fit, transform or save, and one whose first element is
beyond a long `VALUE_NOT_BETWEEN`. Of several such params, the error is of the first of PySpark's
`_paramMap`. `None` is sent as a null literal, as PySpark sends
it: the server reads a null `bootstrap` as false and rejects the other nulls. A param that is
not given is not sent, so the server's default applies. A list param without elements is
accepted by its constructor and setters, and a fit, transform or evaluate raises PySpark's
`CANNOT_INFER_ARRAY_ELEMENT_TYPE` for it, as PySpark's serialization does.

The `getThreshold`, `getThresholds`, `setThreshold` and `setThresholds` methods of the estimator
and of its model, and their `threshold` and `thresholds` Param objects, are PySpark's
`_LogisticRegressionParams` methods, in Python, so they match its errors: the `ValueError` of
the equivalent threshold when `thresholds` has a length other than two, and of an inconsistent
pair, the `ZeroDivisionError` and `TypeError` of their arithmetic, and the `KeyError` of an unset
`thresholds`. The `%g` values of the inconsistency message are formatted by Python.

A native class has the `setParams` of its `pyspark.ml` class. It is keyword-only, takes the params
of the constructor, sets the params that it is given, in order, and returns the object, as
PySpark's `_set` does. As PySpark's, the `setParams` of `LogisticRegression` then checks its
`threshold` and `thresholds`, whichever call set them, as its constructor does. Its signature is
PySpark's for every class, the feature stages and the validators included. The `params` argument
of a fit, transform or
evaluate, which `pyspark.ml` takes as a param map, is accepted: `None` and an empty dict leave the
params as they are. A non-empty map is copied with the map, as `Estimator.fit` and
`Transformer.transform` copy it, so a key that is not a Param raises PySpark's `TypeError` of that
copy. A list or tuple of maps to
a fit is fitted as `pyspark.ml` fits it: one model per map, in order, returned as a list,
so that `[]` fits nothing and does not read the dataset. Any other value raises the `TypeError` that
`pyspark.ml` raises for it. The `params` argument is checked first, before the dataset, as
`pyspark.ml` checks it.

Native `fit`, `transform` and `evaluate` require a Spark DataFrame, matching the dataset
contract of standard `pyspark.ml` over Spark Connect. Passing a pandas DataFrame raises the
`AttributeError` of its own `sparkSession` (`'DataFrame' object has no attribute 'sparkSession'`),
as `pyspark.ml` does, before the params of the class are validated, the schema is read or any
request is sent (the helper calls below raise PySpark's error of their serialization). Only a
`params` argument that is not a param map is checked before it (see above). This also applies to
empty pandas DataFrames. The `evaluate` of a model's summary (`LogisticRegressionModel.evaluate`
and its peers) raises the `TypeError` that `pyspark.ml` raises for a dataset that is not a
DataFrame. The native Pipeline and validators raise the errors of PySpark's: a Pipeline that of
its first stage, and a validator the `AttributeError` of the `select` of its split. A
OneVsRest fits a dataset that is not a Spark DataFrame by PySpark's own `OneVsRest._fit`, so a
pandas DataFrame raises the `KeyError` that it raises (see Native OneVsRest).
Convert local data explicitly with `df = spark.createDataFrame(pdf)` before calling
`estimator.fit(df)`, `model.transform(df)` or `evaluator.evaluate(df)`. Transforms return Spark
DataFrames; call `toPandas()` explicitly when a local result is needed. The deprecated
`pyspark.ml.connect` pandas contract is not the target for these native methods. The feature stages described below use the same Spark DataFrame contract.

Another object than a DataFrame raises the `AttributeError` of `sparkSession`, the attribute
that `pyspark.ml` reads of the dataset first (`'int' object has no attribute 'sparkSession'` of
`fit(5)`), before any request. Of an object that has a `sparkSession`, a fit, a transform and an
evaluate read the `client` of the session (a transform asserts a session that is not None first,
of PySpark's bare `AssertionError`) and then the `_plan` of the object, of their `AttributeError`s
(`'HasSession' object has no attribute '_plan'`), as PySpark's `try_remote_fit`,
`try_remote_transform_relation` and `try_remote_evaluate` do, and an object of both is the
DataFrame of its plan and session (`DataFrame(plan, session)`), whose relation they send, as
PySpark sends that of the plan. A transform reads them before it asserts the server-side object of
a model and serializes its params, which PySpark does before it reads the `_plan`. The methods that
`pyspark.ml` runs as a helper or attribute call read nothing of their dataset.
`PowerIterationClustering.assignClusters`, `PrefixSpan.findFrequentSequentialPatterns`, and
`ALSModel.recommendForUserSubset` and `recommendForItemSubset`, send a dataset that is not a
DataFrame as PySpark does, and the server rejects it when the result is analyzed or collected.
PySpark's serialization (`pyspark.ml.connect.serialize`)
converts any other value to a literal, as `LiteralExpression._from_value` does:
- One that cannot be a literal raises PySpark's error of the conversion. That includes a pandas
  DataFrame, which raises `UNSUPPORTED_DATA_TYPE` here rather than the `AttributeError` above, as
  with `pyspark.ml`, and an empty list, which raises `CANNOT_INFER_ARRAY_ELEMENT_TYPE`.
- A `(value, DataType)` pair is the literal of that type, as `LiteralExpression(value, dataType)`
  makes it, whose asserts raise a bare `AssertionError` of a value that is not of the type
  (`("x", IntegerType())`) and of a type that no literal has (a `MapType`, `StructType`,
  `CharType`, `VarcharType`, `YearMonthIntervalType` or UDT), also with a None value, as with
  `pyspark.ml` (`tests/golden/ml_fetch_pairs.json` records the pairs of official PySpark).
- PySpark sends the others for the server to reject: an ML vector or matrix as they are, and a
  `(value, DataType)` pair as a literal of that type. `assignClusters`, PrefixSpan's
  `findFrequentSequentialPatterns` and ALSModel's `recommendForUserSubset` and
  `recommendForItemSubset` send them too, in a lazy DataFrame that the server rejects when it is
  analyzed or collected.

As with
`pyspark.ml`, a fit sends the DataFrame as it is, and the transform of a model is the transform of
the DataFrame as it is, which sends no request: the JVM estimators and models check the columns,
and raise their errors of a column of a wrong type, or of an output column that the DataFrame has
already, in the fit, or when the transformed DataFrame is analyzed or run.

The column params of a fit and of a transform are sent as they are given, as `pyspark.ml` sends
them, and the JVM reads a column by its exact name: `featuresCol="Features"` of a column `features`
raises the JVM's `FIELD_NOT_FOUND` in a fit, as with `pyspark.ml`, while some JVM models resolve
such a name. Evaluators send column names and types unchanged too, leaving their validation to
Spark as standard `pyspark.ml.evaluation` does. Their metric results are Spark's double-precision
values.

The native estimators take the `VectorUDT` features of `pyspark.ml`: the JVM raises its error of
an array `featuresCol` (`Column features must be of type class
org.apache.spark.ml.linalg.VectorUDT:... but was actually class
org.apache.spark.sql.types.ArrayType:array<double>.`), in the fit and when the transformed
DataFrame is analyzed. Convert an array column with `pyspark.ml.functions.array_to_vector` first,
as for `pyspark.ml`. (The JVM `KMeans`, `BisectingKMeans`, `GaussianMixture` and `LDA` also take
arrays whose elements cannot be null, which the arrays of `createDataFrame` can be.)

The feature stages, including `StandardScaler` and `MaxAbsScaler`, use Spark's JVM
implementations through Connect. Vector-valued inputs must use `VectorUDT`; convert numeric
arrays explicitly with `pyspark.ml.functions.array_to_vector`. `StandardScaler` defaults to
`withMean=False, withStd=True`, matching standard `pyspark.ml`.

The Rust `LogisticRegression` and `LogisticRegressionModel` have the params of `pyspark.ml`, with
one generated getter and setter per param (`set_max_iter` and `max_iter()`, `set_features_col` and
`features_col()`, and so on), and `PARAM_NAMES` lists them. `LogisticRegression::feature_col` and
`set_feature_col`, the names of v4.2.0, stay as deprecated aliases of `features_col` and
`set_features_col`. `threshold()`, `thresholds()`, `set_threshold` and `set_thresholds` follow
`_LogisticRegressionParams`: the getters check that the two agree to 1e-5 and `threshold()` requires
two thresholds. The setters leave the other param set, as over Spark Connect, where the `clear` of
`pyspark.ml` does nothing, so a disagreement raises the getters' error. Their arithmetic is
PySpark's, in Python: a zero divisor of the equivalent threshold of two thresholds, `1 / (1 + t0 /
t1)` (a zero `t1`, or a ratio of -1), is the error "float division by zero" of its
`ZeroDivisionError`, of `threshold()` and, with a threshold set too, of `thresholds()`. The model
has the typed members of `LinearSVCModel` (see "Native LinearSVC" below), plus
`coefficient_matrix()` and `intercept_vector()`; `coefficients()` and `intercept()` raise the
server's error of a multinomial model, as PySpark does. The model's `summary()` is binary for at
most two classes, of the `numClasses` that it fetches when called (PySpark fetches it at the fit),
and `evaluate` makes the same choice on its dataset. `has_summary()` is client side. The Rust API
differs from `pyspark.ml` as the LinearSVC section says.

## Native RandomForestRegressor

`spark_connect::ml::RandomForestRegressor` fits Spark's JVM random forest regressor
through Connect using ML vector features and numeric labels. Rust builder setters
and getters cover all 19 PySpark constructor parameters, including instance weights,
bootstrap sampling, feature sampling and optional per-tree leaf indices. Unset
parameters use Spark's defaults; a seed can select another reproducible sample.

The native Python estimator is available from
`pyspark.ml.connect.regression.RandomForestRegressor` and
`pyspark.ml.connect.RandomForestRegressor`. It uses PySpark's keyword-only constructor
names, defaults and parameter conversions, with mutating setters. Fitted models
expose DataFrame `transform` for the fitting session. Models, clones and lazy
transformed DataFrames retain the cached server model until their last owner drops.
The Rust model's members and persistence are in "Native regression models (Rust API)". In Python,
`RandomForestRegressor` and its `RandomForestRegressionModel` follow the full API of `pyspark.ml`
(see "Native ML classes of the full pyspark.ml API"): the Params API, `trees` (each a
`DecisionTreeRegressionModel`, fetched at the fit and at a load, as PySpark's), `treeWeights`,
`getNumTrees`, `totalNumNodes`, `featureImportances`, `toDebugString`, `numFeatures`, `predict` and
`predictLeaf` of a vector, and save and load.

## Native GBTRegressor

`spark_connect::ml::GBTRegressor` fits Spark's JVM gradient-boosted regressor
through Connect with ML vector features and continuous numeric labels. Rust
setters/getters cover all 22 PySpark constructor parameters. It supports squared
and absolute loss, instance weights, row/feature sampling, learning rate,
iteration limits and validation-set early stopping.

The Python binding is available as `pyspark.ml.connect.regression.GBTRegressor`
and `pyspark.ml.connect.GBTRegressor`. Its keyword-only signature and 21 named
setters match PySpark. `validationTol` has no Python setter. The reference's
signature advertises `validationTol=0.1`, but omitting it leaves the parameter
unset and uses Spark's `0.01` default. This binding preserves that behavior;
`GBTRegressor(validationTol=0.1)` explicitly sends `0.1`. The Rust getter returns
`0.01` when unset.

`maxIter` limits boosting iterations, and `stepSize` scales each tree after the
first. With `validationIndicatorCol`, false rows train and true rows validate;
early stopping can fit fewer trees. DataFrame transforms expose predictions and
an optional vector of leaf IDs, one per fitted tree. Models, clones and lazy
transformed DataFrames retain the cached model in its fitting session until the
last owner is dropped. The Rust model's members and persistence are in "Native regression models
(Rust API)". In Python, `GBTRegressor` and its `GBTRegressionModel` follow the full API of
`pyspark.ml`, as the random forest ones do: the Params API, the members of a tree ensemble,
`featureImportances`, `evaluateEachIteration(dataset, loss)`, and save and load.

## Native GBTClassifier

`spark_connect::ml::GBTClassifier` fits Spark's JVM gradient-boosted binary
classifier through Connect with ML vector features and labels 0 or 1. Rust builder
setters/getters cover all 22 PySpark constructor parameters, plus probability/raw
prediction column names and class thresholds. Unset parameters use Spark's defaults.
A seed controls split-candidate row sampling, subsampling and feature sampling.

The Python binding is available as `pyspark.ml.connect.classification.GBTClassifier`
and `pyspark.ml.connect.GBTClassifier`. Its keyword-only constructor mirrors PySpark.
Probability/raw prediction column names and thresholds are set through
`setProbabilityCol`, `setRawPredictionCol` and `setThresholds`. `validationTol` is a
constructor argument; PySpark does not provide a Python setter for it.

`maxIter` limits boosting iterations, and `stepSize` scales each tree after the
first. With `validationIndicatorCol`, false rows train the trees and true rows
validate early stopping; the resulting ensemble can contain fewer than `maxIter`
trees. DataFrame transforms expose predictions, probabilities, raw margins and an
optional vector of leaf IDs, one per fitted tree. Instance weights are supported.
Models, clones and lazy transformed DataFrames retain the cached model in its
fitting session until the last owner is dropped. The Rust model has the typed
accessors of PySpark's `GBTClassificationModel`: `trees` (regression tree models),
`tree_weights`, `get_num_trees`, `total_num_nodes`, `feature_importances`,
`to_debug_string`, `predict`, `predict_leaf`, `predict_raw`, `predict_probability`,
`evaluate_each_iteration(&df)` and the supported-value lists. It has `load`, and
`write().save(path)`. Unlike PySpark, the fit does not fetch the trees up front; they
are fetched when asked. In Python, `GBTClassifier` and its `GBTClassificationModel` follow the full API of
`pyspark.ml` (see "Native ML classes of the full pyspark.ml API"); as PySpark's, its fit reads
the trees right after the Fit.

## Native DecisionTreeRegressor

`spark_connect::ml::DecisionTreeRegressor` fits Spark's JVM decision tree
regressor through Connect with ML vector features and continuous numeric labels.
Rust builder setters/getters cover all 16 PySpark constructor parameters,
including `set_variance_col`, `set_weight_col` and `set_leaf_col`. Unset
parameters use Spark's defaults.

The native Python binding is available as
`pyspark.ml.connect.regression.DecisionTreeRegressor` and
`pyspark.ml.connect.DecisionTreeRegressor`. Its keyword-only constructor mirrors
PySpark's parameter names and defaults. Mutating `setFeaturesCol`, `setMaxDepth`,
`setVarianceCol`, etc. methods return the estimator.

Fitted models expose DataFrame `transform`, including predictions, optional
weighted label variance in the predicted leaf and optional scalar double leaf
indices. Variance is computed within the predicted leaf; it is not a confidence
interval for the prediction. Models, clones and lazy transformed DataFrames keep
the cached server model alive in its fitting session until the last owner is
dropped. The Rust model's members and persistence are in "Native regression models (Rust API)". In
Python, `DecisionTreeRegressor` and its `DecisionTreeRegressionModel` follow the full API of
`pyspark.ml` (see "Native ML classes of the full pyspark.ml API"): the Params API, `depth`,
`numNodes`, `featureImportances`, `toDebugString`, `numFeatures`, `predict` and `predictLeaf` of a
vector, and save and load.

## Native DecisionTreeClassifier

`spark_connect::ml::DecisionTreeClassifier` trains Spark's JVM decision tree
classifier through Connect with ML vector features and non-negative integer class
labels below numClasses. Rust builder setters cover all 18 PySpark constructor
parameters, including `pruneTree` (Spark 4.3.0), plus class thresholds. Unset
parameters use Spark's defaults.

The Python binding is available as
`pyspark.ml.connect.classification.DecisionTreeClassifier` and
`pyspark.ml.connect.DecisionTreeClassifier`. Its keyword-only constructor matches
PySpark's parameter names and defaults; mutating `setFeaturesCol`, `setMaxDepth`,
etc. methods and `setThresholds` return the estimator.

Fitted models expose DataFrame `transform`, including prediction, probability,
raw per-leaf weighted class counts and an optional scalar double leaf index.
Models, clones and lazy transformed DataFrames retain the cached model in its
fitting session until the last owner is dropped. The Rust model has the typed
accessors of PySpark's `DecisionTreeClassificationModel`: `depth`, `num_nodes`,
`feature_importances`, `to_debug_string`, `predict`, `predict_leaf`, `predict_raw`,
`predict_probability` and `supported_impurities`, and `load`; `write().save(path)`
saves it. In Python, `DecisionTreeClassifier` and its `DecisionTreeClassificationModel` follow the full
API of `pyspark.ml` (see "Native ML classes of the full pyspark.ml API").

## Native RandomForestClassifier

`spark_connect::ml::RandomForestClassifier` trains Spark's JVM random forest
classifier through Connect with ML vector features and non-negative integer class
labels below numClasses. Rust builder setters cover all 22 PySpark constructor
parameters, including `pruneTree` (Spark 4.3.0), plus class thresholds. Unset
parameters use Spark's defaults. Without a seed, the sampling uses Spark's default
seed (the hash of the class name), as with PySpark, so fits are reproducible either
way.

The native Python binding is available as
`pyspark.ml.connect.classification.RandomForestClassifier` and
`pyspark.ml.connect.RandomForestClassifier`. Its keyword-only constructor matches
PySpark's parameter names and defaults, with mutating `setFeaturesCol`,
`setNumTrees`, etc. methods and `setThresholds`.

Fitted models expose DataFrame `transform`, including prediction, probability,
raw prediction and optional per-tree leaf index columns. Models, clones and lazy
transformed DataFrames retain the cached model in its fitting session until the
last owner is dropped. The Rust model has the typed accessors of PySpark's
`RandomForestClassificationModel`: `trees` (decision tree models), `tree_weights`,
`get_num_trees`, `total_num_nodes`, `feature_importances`, `to_debug_string`,
`predict`, `predict_leaf`, `predict_raw`, `predict_probability`, `has_summary`,
`summary()` (the binary or multiclass training summary, chosen by `numClasses`),
`evaluate(&df)` (the binary or multiclass summary of a dataset), the supported-value
lists, and `load`; `write().save(path)` saves it. A loaded model has no training
summary. The fit does not fetch `numClasses` and the trees up front as PySpark does;
`summary()` and `trees()` fetch them when asked. In Python, `RandomForestClassifier`, its `RandomForestClassificationModel` and the four
random forest summaries follow the full API of `pyspark.ml` (see "Native ML classes of the
full pyspark.ml API"); as PySpark's, its fit reads `numClasses` and then the trees right after
the Fit.

## Native KMeans

`spark_connect::ml::KMeans` fits Spark's JVM K-means estimator with ML vector
features through the Connect ML protocol. Rust builder setters cover all 12
constructor parameters, including `set_k`, `set_seed`, `set_weight_col`,
`set_distance_measure` and `set_solver`. Unset parameters use Spark's defaults.
Without a seed, the initialization uses Spark's default seed (the hash of the class
name), as with PySpark, so fits are reproducible either way. Cluster numbers are
arbitrary: compare membership rather than relying on particular numeric labels.

The native Python binding is available as `pyspark.ml.connect.clustering.KMeans`
and `pyspark.ml.connect.KMeans`, with the upstream keyword-only constructor and
mutating parameter setters. Fitted models expose DataFrame `transform` in the
fitting session. Models, clones and transformed DataFrames keep the cached server
model alive until the last owner is dropped. The Rust model's members, summaries and
persistence are in "Native clustering models (Rust API)". In Python, `KMeans`, its `KMeansModel`
and `KMeansSummary` follow the full API of `pyspark.ml` (see "Native ML classes of the full
pyspark.ml API"): the Params API, `clusterCenters()`, `numFeatures`, `predict` of a vector, the
training summary (`k`, `clusterSizes`, `trainingCost`, `numIter`, `predictions` and `cluster`, of
the transform of the training dataset, as PySpark's), and save and load. The vendored
`pyspark.ml.clustering` API is unchanged.

## Native LinearRegression

`spark_connect::ml::LinearRegression` fits Spark's JVM linear regression estimator
through the Connect ML protocol, with ML vector features and numeric labels.
Rust builder setters cover all 15 constructor parameters, including regularization,
instance weights, solver selection and squared-error or Huber loss. Unset parameters
use Spark's defaults.

The native Python binding is available as
`pyspark.ml.connect.regression.LinearRegression` and `pyspark.ml.connect.LinearRegression`.
Its keyword-only constructor matches the JVM PySpark estimator's parameter names
and defaults, with mutating `setFeaturesCol`, `setRegParam`, etc. methods.
Fitted models expose `transform` on DataFrames from the fitting session. The model,
its clones and transformed DataFrames keep the cached server model alive until
the last owner is dropped. The Rust model's members and persistence are in "Native regression models
(Rust API)". In Python, `LinearRegression`, its `LinearRegressionModel`, `LinearRegressionSummary`
and `LinearRegressionTrainingSummary` follow the full API of `pyspark.ml` (see "Native ML classes of
the full pyspark.ml API"): the Params API, `coefficients`, `intercept`, `scale`, `numFeatures`,
`predict` of a vector, the training summary (of the training dataset, as PySpark's), `evaluate`, and
save and load.

## Native LinearSVC

`spark_connect::ml::LinearSVC` trains Spark's JVM binary linear support vector
classifier through the Connect ML protocol. Use vector features and labels 0 or 1.
Rust builder setters cover all 13 constructor parameters, including `set_reg_param`,
`set_tol`, `set_weight_col` and `set_threshold`. Unset parameters use Spark's defaults.
The Python binding is available as `pyspark.ml.connect.classification.LinearSVC`
and `pyspark.ml.connect.LinearSVC`, with the upstream keyword-only constructor
parameters and mutating `setFeaturesCol`, `setRegParam`, etc. methods.

`fit` returns a server-cached model; `transform` must use the fitting session.
Clones and lazy prediction DataFrames keep the cached model usable while they are
alive. When the last of the model, its clones and the DataFrames it transformed is
dropped, the server deletes the cached model, as `pyspark.ml` does.
In Python, `LinearSVC`, its `LinearSVCModel` and their summaries follow the full API of
`pyspark.ml` (see "Native ML classes of the full pyspark.ml API"): the Params API,
`coefficients`, `intercept`, `numClasses`, `numFeatures`, `predict` and `predictRaw` of a
vector, `summary()`, `evaluate`, and save and load.

The Rust `LinearSVCModel` has the typed members of `pyspark.ml`: `coefficients()`, `intercept()`,
`num_classes()`, `num_features()`, `predict(&vector)` and `predict_raw(&vector)`, `has_summary()`
(client side), `summary()` (the training summary, with `accuracy()`, `area_under_roc()`,
`objective_history()`, `total_iterations()` and the curve DataFrames `roc()` and `pr()`),
`evaluate(&df)` (the same summary members on a dataset), `LinearSVCModel::load(&session, path)`,
and `model.write().overwrite().save(path)`. Its setters `set_features_col`,
`set_prediction_col`, `set_raw_prediction_col` and `set_threshold` set the params of its
transform. `LinearSVC::fit_multiple(&df, &maps)` fits one model per param map, and
`LinearSVC::load` loads an estimator. Differences from `pyspark.ml`: `summary()` is a typed value
rather than a Python object, `fit_multiple` returns a `Vec` rather than an iterator, and `read()`
is `load`, with no reader object. The summary's DataFrames are built on the training DataFrame,
as PySpark builds them.

In Python, `LogisticRegression` and its model follow the full API of `pyspark.ml` in the same way,
with `coefficientMatrix`, `interceptVector`, `predictProbability`, the `summary` property and
`hasSummary`. The fit of a LogisticRegression is followed by a `numClasses` request, which its
training summary reads, as PySpark's does: the summary is the binary one of two classes or fewer,
else the multinomial one.

## Native Pipeline and PipelineModel

`pyspark.ml.connect.Pipeline` and `pyspark.ml.connect.PipelineModel` are PySpark's `Pipeline` and
`PipelineModel` of `pyspark.ml.pipeline`, natively and in full, as PySpark 4.3.0 has them over
Spark Connect:

- **Stages:** the stages are any estimators and transformers: the native classes,
  `pyspark.ml`'s, or the user's.
- **Fit:** `fit` is `Pipeline._fit`. Up to the last estimator, a transformer transforms and an
  estimator fits, and its model transforms for the stages after it. The stages after the last
  estimator are kept as they are, in a `PipelineModel` of a new uid. As `Pipeline._fit`, it
  iterates the stages three times (to check their types, to find the last estimator, and to fit
  them), so stages of an iterator, such as a generator, which the first iteration consumes, fit a
  `PipelineModel` of no stages, as PySpark's.
- **Transform:** `PipelineModel.transform` transforms by each stage in order.
- **Copy:** `Pipeline.copy` keeps the uid and copies each stage with the extra params, while
  `PipelineModel.copy` is of a new uid.
- **Persistence:** `write`, `save`, `read` and `load` are of PySpark's Python format:
  - the metadata of `pyspark.ml.pipeline.Pipeline` (or `PipelineModel`), of the `stageUids`;
  - each stage, saved by its own writer at `stages/<index>_<uid>`.

  As PySpark's writers, a save iterates the stages three times: to check that they write
  (`validateStages`), for their uids, and to save them. So the stages of a one-shot iterator (a
  generator), which the check consumes, are saved as no stages (`stageUids` of `[]`), and a fit of
  the Pipeline after the save fits none, as PySpark's.

  PySpark loads what this client saves, and this client loads what PySpark saves, and a pipeline,
  a validator or a OneVsRest (and its model) that the JVM writer saved (of the `org.apache.spark.ml`
  class names, as Scala and classic PySpark's `JavaMLWriter` write them), also as a stage. A stage
  loads as the native class of `pyspark.ml.connect` of its name where one loads, else as PySpark's
  class. An overwrite deletes the path by the ML helper first, as `RemoteMLWriter` does.

The vendored `pyspark.ml.Pipeline` and the validators of `pyspark.ml.tuning` fit, transform and
save native stages, which are `MLWritable` (see "PySpark's types and class Params"). A fitted model
is a stage of `PipelineStage::model`, which a save writes by the model's server-side object: a
`PipelineStage::transformer` of a model fails at the save, as a transformer holds no such object.
Their loads
(`pyspark.ml.PipelineModel.load`, `pyspark.ml.tuning.CrossValidatorModel.load`) give PySpark's
classes of `pyspark.ml`, as they give of PySpark's own stages: `DefaultParamsReader` resolves each
stage of its JVM class. The loads of `pyspark.ml.connect` give the native classes. The native
meta-algorithms take an object of a native class as writable, and any other object only if it is
an `MLWritable`, as PySpark does.

## Native validators

`pyspark.ml.connect.tuning` has PySpark's validators of `pyspark.ml.tuning`, natively and in full,
as PySpark 4.3.0 has them over Spark Connect: `CrossValidator`, `CrossValidatorModel`,
`TrainValidationSplit` and `TrainValidationSplitModel`. Its `ParamGridBuilder` is PySpark's own
class.

- **Estimators and evaluators:** any estimator and evaluator, of the native classes, of
  `pyspark.ml`, or of the user's. A Pipeline is an estimator too.
- **Fit:** `fit` is the `_fit` of PySpark's class, on the client:
  - The splits are those of PySpark's: of `CrossValidator`, `numFolds` folds of a column
    `<uid>_rand` of `rand(seed)`, or the folds of the numbers of `foldCol`; of
    `TrainValidationSplit`, the rows of `rand(seed)` below `trainRatio` train. A fold number out
    of `[0, numFolds)` fails the fit, and so does an empty fold, as PySpark raises them.
  - Each split is cached while its models are fitted and evaluated, as `_cache_spark_dataset`
    caches it.
  - The fits of the param maps run in a pool of `parallelism` threads, of the estimator's
    `fitMultiple`, as PySpark's `_parallelFitTasks`.
  - `collectSubModels` keeps the model of each fold and param map.
  - The metrics are PySpark's: the mean and standard deviation of numpy over the folds
    (`avgMetrics`, `stdMetrics`), or the metric of each param map (`validationMetrics`), and the
    best of them by `isLargerBetter`.
  - The best param map is fitted on the whole dataset, in a model of a new uid that gets the
    params of the validator, as `_copyValues` gives them.
- **Models:** `bestModel`, the metrics (`avgMetrics`, `stdMetrics`, `validationMetrics`) and
  `subModels`, which are attributes of PySpark's instance, are native properties, assignable as
  PySpark's attributes: `vars()` lacks them, and `del` of one raises `AttributeError: property has
  no deleter`, where PySpark's deletes the attribute (and its next read raises the `AttributeError`
  of a missing attribute). So are PipelineModel's `stages` and OneVsRestModel's `models`, which the
  native members read. `transform` transforms by the best model. `copy` copies the best model and
  the sub-models, and a model of no sub-models fails as PySpark's `assert` does.
- **Copy:** `CrossValidator.copy` and `TrainValidationSplit.copy` keep the uid and copy the
  estimator and the evaluator with the extra params. The param maps stay the same.
- **Persistence:** of PySpark's Python format (`_ValidatorSharedReadWrite`):
  - the metadata of `pyspark.ml.tuning.CrossValidator` (or the others) and of its params, in
    PySpark's order. The param maps are of the JSON of each value, or the path of a value that is
    a stage (`epm_<name><n>`).
  - the estimator and the evaluator at `estimator` and `evaluator`, and a model's best model at
    `bestModel`. The sub-models are at `subModels`, of the writer's option `persistSubModels`
    (by default of whether the model has them).

  PySpark loads what this client saves, and this client loads what PySpark saves. A load of a
  model of a cross validation repeats one list of the sub-models for each fold, as PySpark's
  reader does.
- **Checks of a save:** as PySpark's, a validator saves only if its estimator, the stages that
  the estimator nests, and its evaluator write, and each param of its param maps is a param of one
  of them. A nested validator is refused.

The native Pipeline, PipelineModel and validators are subclasses of PySpark's classes of them
(`pyspark.ml.Pipeline`, `pyspark.ml.tuning.CrossValidator`, ...), as every native class is. So
`isinstance` checks of PySpark's classes hold of them, and PySpark's `MetaAlgorithmReadWrite`
reads the nested stages of a native Pipeline.

## Native meta-algorithms (Rust API)

`spark_connect::ml` has PySpark's meta-algorithms, of the estimators, transformers, models and
evaluators of the crate (or the user's), as PySpark 4.3.0 runs them over Spark Connect, on the
client:

- **Stages**: a `PipelineStage` is an estimator, a transformer or a fitted model
  (`PipelineStage::estimator(LinearSVC::new())`), of any type that clones (`EstimatorStage`,
  `TransformerStage`, `EvaluatorStage`). A fitted model is given to `PipelineStage::model`, and
  a fitted model given to `PipelineStage::transformer` fails at the save (see "Native Pipeline
  and PipelineModel").
- **Param maps**: a `ParamMap` holds values of params of stages, by the uid of a stage and the
  name of its param, as PySpark's `Param.parent` and `Param.name`. `ParamGridBuilder` builds
  PySpark's grids (`add_grid`, `base_on`, `build`, the last param varying fastest), and
  `apply_param_map` of an estimator, a transformer or an evaluator sets the values of its params,
  as a copy of a param map of PySpark does: each value is converted by the type converter of its
  param (`Param.typeConverter`, so an integer grid value of a double param is a double), and a
  param that the stage's class does not have is not set. A pipeline applies them to its stages, a
  `OneVsRest` to its classifier and a `OneVsRestModel` to its models, as their `copy(extra)` does,
  so that `{lr.featuresCol: "f2"}` of the classifier's uid sets the features of each model.
  The grid keeps the values as they were given, and a validator converts them when it applies or
  saves them, not when the grid is built as PySpark does, so a value that PySpark refuses at build
  (a `TypeError`) is refused at the fit or the save of the validator. So is a value that PySpark
  takes but cannot send, which its fit refuses: an int beyond a long (a whole float such as
  `1e20` of an int param), PySpark's `VALUE_NOT_BETWEEN`, and a list of ints of an element out of
  the range of the type of its first element, the `AssertionError` of its `LiteralExpression`. A
  list of ints is sent as PySpark sends it, of the type of its first element.
- **`Pipeline` and `PipelineModel`**: `fit` (or `fit_pipeline`, of the typed model) is
  `Pipeline._fit`; `PipelineModel` transforms by each stage in order. `copy` keeps the pipeline's
  uid, and a pipeline model's copy has a new one.
- **`CrossValidator` and `TrainValidationSplit`**: their fit (`fit_validator`, of the typed
  model) splits the dataset with `k_fold` or `train_validation_split` and caches each split, as
  `_cache_spark_dataset` does (of `SPARKML_TEMP_DFS_PATH` too). It fits a copy of the estimator
  of each param map in `parallelism` threads and evaluates each with the evaluator. The metrics
  are numpy's (the mean and the standard deviation of the population over the folds), and the
  best is numpy's `argmax` or `argmin`, of the first NaN of any. The best param map is fitted on
  the whole dataset. Their models have `best_model`, the metrics and `sub_models`
  (`collect_sub_models`), and transform by the best model. The evaluation of a param map that is
  not empty transforms by a copy of its model of the map (`model.transform(validation,
  epm[index])`), and of an empty one by the model itself. A model's `apply_param_map` is
  PySpark's `copy(extra)` of it: it applies the map to the best model, fails as PySpark's
  `assert self.subModels is not None` of a model of no sub-models (so the evaluation of a param
  map of a validator of no sub-models nested in a pipeline fails, as PySpark's does), copies the
  sub-models of no map, and sets the model's own params of the map (`numFolds`, `foldCol`,
  `trainRatio` and `seed`). A validator's `apply_param_map` sets
  its own params (`numFolds`, `foldCol`, `trainRatio`, `seed`, `parallelism` and
  `collectSubModels`) first, then its estimator's and evaluator's, as PySpark's `copy(extra)`
  does, so a validator nested in a grid takes the values of its own params from it. A fit of no
  fold (a `numFolds` of no fold with a fold column, or a negative one) is PySpark's `TypeError`.
  A validator of no seed has the default seed of its class: PySpark's default is
  `hash(type(self).__name__)`, which is the same for each validator of a class in a process. This
  client draws one random seed per class and per process, so its value differs from PySpark's (as
  PySpark's differs between processes), and the validators of a class in a process split the same
  way. A model of a validator has the default seed of its own class, and the validator's seed
  when it is set.
- **Persistence**: `write()` and `load` are of PySpark's Python format, so that PySpark loads what
  the Rust API saves and the Rust API loads what PySpark saves:
  - the metadata of `pyspark.ml.pipeline.Pipeline` (and the others) in PySpark's order of its
    keys, of NaN and Infinity floats as Python writes them;
  - each stage, saved by its own writer;
  - the validators' param maps, estimator, evaluator, best model and sub-models
    (`persistSubModels`).

  `load_stage` loads any saved stage, of its class: a meta-algorithm (of PySpark's Python format,
  or of the JVM writer's class names, a OneVsRest and its model too, whose readers check no class,
  as PySpark's do), or an estimator, transformer or model of the server's class
  (`OperatorEstimator`, `OperatorTransformer`, `ObjectModel`). The evaluator of a loaded validator
  is the evaluator of its class. A param map
  of a value that is a saved stage, which PySpark writes of a stage-valued param, does not load.
  A fitted model of an estimator has the JVM class that PySpark's `_create_model` names (the
  classes of the decision trees, random forests, gradient-boosted trees, factorization machines
  and perceptrons are their `...ClassificationModel` and `...RegressionModel`, and
  QuantileDiscretizer's model is a `Bucketizer`), and the params of the estimator that that class
  has, as PySpark's model copies them.

## pyspark.ml

The DataFrame-based `pyspark.ml` API (`pyspark.ml.classification`, `feature`, `regression`,
`clustering`, `evaluation`, `tuning`, `stat`, `Pipeline`, ...) is Apache Spark's own, vendored from
v4.3.0: estimators fit, models transform, and pipelines save and load on the server as with PySpark
Connect, over the Rust transport. These modules build their own `spark.connect` messages with
`pyspark.sql.connect.proto`, which is the drop-in's own protobuf runtime in Rust
(`pyspark.sql.connect.proto._rust`) in place of the generated modules of Apache Spark, whether or
not `protobuf` is installed, so `protobuf` is not a dependency of `pyspark-client-rust`. The runtime
has the classes, containers, descriptors and descriptor pool of upb, the implementation of a
`protobuf` install, with their bytes, text, errors and presence (of protobuf 7, where protobuf 6's
differ), the methods of protobuf's `Any` and `Timestamp` (a Timestamp field takes a `datetime`), and
the modules of the protos with the names of the generated modules
(`pyspark.sql.connect.proto.base_pb2`, ..., and `example_plugins_pb2`, of Spark's example plugin),
whose files in the package are the source that type checkers find next to their stubs. Where
`protobuf` imports, the classes are of the shape of upb's: of a metaclass `MessageMeta` (which
refuses a subclass of a message class, gives the one class of a descriptor and gives the attributes
of the nested enums, their values and the numbers of the fields), subclasses of protobuf's
`google.protobuf.message.Message` (and of protobuf's methods of a well-known type,
`well_known_types.WKTBASES`), of protobuf's descriptors of its default pool, to which the files of
the protos are added, and of protobuf's `EnumTypeWrapper`; they pickle as protobuf's
`Message.__reduce__` pickles upb's (a message of a nested type of protobuf's
`_InternalConstructMessage`, so that such a pickle loads where `protobuf` imports), and a parse that
fails raises protobuf's `DecodeError` itself, of upb's message (`Error parsing message with type
'spark.connect.Plan': Wire format was corrupt`), as a serialization of a message that lacks a
required field raises its `EncodeError`. As upb's, a message of more than 100 levels of messages
below it fails where upb parses its bytes, of upb's `Exceeded upb_DecodeOptions_MaxDepth` and of
what the parse keeps: in a parse, the load of a pickle, `Any.Unpack`, `MergeFrom`, `copy.copy`, a
constructor given a message for a field or a repeated field (`Plan(root=relation)`), and `append`,
`extend` and `insert`, while `CopyFrom` (of a message of `protobuf`'s of the type too), `==`,
`copy.deepcopy`, the value of a map and the mappings of a constructor copy any depth, and the
drop-in's own parses of the messages that PySpark builds in Python (`Column._expr`, a type of
`pyspark_types_to_proto_types`, the requests that session hooks see) take any depth, while the
responses of the server are read within the limit, as upb reads them (see
[What "drop-in" means](#what-drop-in-means)). Once upb's `SetAllowOversizeProtos(True)`
(`api_implementation._c_module`) is called, before the package imports or after (also of a reference
to the function taken before, which a parse that reaches the limit of 100 levels asks upb of again,
as upb tells of no other way), the limit is 65535 levels, as upb's; upb's parser, which recurses on
the stack, ends the process past about 50,000 levels, where the runtime's grows its stack. A call of
a reference taken before that sets it back to False is not seen once a parse found it set. Of
protobuf's pure-Python implementation, the limit is that of its `decoder.SetRecursionLimit` (100 of
none), as of its parsers, of upb's message (where its own parser's is `Error parsing message: too
many levels of nesting.`), as of the runtime's other errors. A parse that fails keeps what upb's
decoder keeps of its input stream: a value cut short by the end of its message is read of the bytes
that follow it, and of zeros past the end of the data, and set before the parse fails, but an
undeclared value of a closed enum of a field that is not packed read past the end of its message,
which is not kept; an undeclared value of a closed enum is kept in the unknown fields of the bytes
of its field as they are (of a packed field, of a tag of its own and the 64 bits of the value), a
length of more than five bytes is corrupt, a field (of a group too) of the number 0 is an unknown
field of a message type of no fields (`google.protobuf.Empty`), and an unknown group past the limit
is corrupt (`Wire format was corrupt`), as upb's. A string or bytes member of a oneof whose bytes
fail to parse after a member that holds a message is empty, where upb reads the memory of that
message as the string, of the length of a string that the oneof held before, which no program can
rely on. `protobuf`'s `json_format` and `text_format` print and parse its messages, and
`unknown_fields.UnknownFieldSet` reads their unknown fields, which `text_format` prints with
`print_unknown_fields=True` (see below). Without `protobuf` (which PySpark does not run without),
the classes are of the runtime's own descriptors and pool, which Python code adds no file to, of the
runtime's base of protobuf's methods of `Message` (`_rust._MessageBase`), a message of a nested type
pickles of its class's qualified name, `GetOptions()` of a descriptor needs `protobuf`, and the
errors are the runtime's `DecodeError` and `EncodeError`. The differences:

- its types are of `pyspark.sql.connect.proto`, not of upb's `google._upb._message`: its metaclass
  is `pyspark.sql.connect.proto._rust.MessageMeta`, its classes subclass the runtime's `Message`
  (`pyspark.sql.connect.proto._rust.Message`, whose module is named `pyspark.sql.connect.proto`,
  which does not export it), its maps of scalars and of messages are its `ScalarMapContainer` and
  `MessageMapContainer`, of a base `MapContainer`, and its messages take weak references, which
  upb's refuse, as the runtime's unset messages and containers hold the message that they set
  weakly. The runtime's `Message` has the `__reduce__`, `__getstate__` and `__setstate__` of the
  results of protobuf's (of upb's pickles), which take less time than protobuf's Python ones, and a
  `__copy__`, which copies a message of its fields (of upb's bytes past the limit of a parse), where
  upb's `copy.copy` is protobuf's `__reduce__`, a parse of the bytes: so `dir()` of a class and of a
  message list `__copy__`, which upb's do not. The metaclass gives the attributes of a class (the
  wrappers of the nested enums, the numbers of their values and of the fields) when it makes the
  class, where upb's binds each when it is read, so that a read of an attribute of a class takes no
  more time than of a Python class: `dir()` and `vars()` of a class list them before they are read,
  and the `AttributeError` of a name that a class lacks is Python's (`type object 'Join' has no
  attribute 'nope'`), where upb's is of the name only (`nope`); as upb's, a name of the suffix
  `_FIELD_NUMBER` is only the number of a field, not a nested enum or a value of one of the name.
  The methods of its containers take their arguments as upb's do, positionally
  (`RepeatedScalarContainer.append() takes no keyword arguments`), with upb's errors, of the names
  of upb's classes of the maps too (`ScalarMapContainer.MergeFrom() takes exactly one argument (0
  given)`). A dunder method of a map called directly with the wrong arguments (`map.__eq__()`) is an
  exception: its error is that of CPython's slot wrapper of the native type (`expected 1 argument,
  got 0`), where upb's `__eq__` of a map is `Mapping.__eq__`, a Python function. So is a keyword
  argument of `setdefault`, `insert`, `pop` or `__deepcopy__` (upb's `METH_VARARGS` methods) called
  through a bound method object (`f = map.setdefault; f(key="k")`): CPython's error of upb's names
  the method only (`setdefault() takes no keyword arguments`), as of a bound builtin method, and the
  runtime's method, which cannot tell that call from one of the map, names the class too. Before
  Python 3.12, while a trace or a profile function is set (`coverage`, a debugger), CPython calls
  each such method as a bound method, and the runtime's error then names the method only too;
- where Python code that a method of a container runs (an `__index__`, an `__eq__`, an iterable)
  shortens the container, a read past its end raises `IndexError` and an insertion or a deletion
  takes the container as it is, where upb reads or writes past the end of its array, which can
  end the process;
- a copy of a message into a message that it holds copies the message first, where upb's copy
  recurses until it ends the process (see "Copies into a message that the value holds" of
  `dev/parity/known-differences.md`): a `CopyFrom` into a message below its argument (an unset
  child of it too) or into a value of a map or a repeated field of it, and the constructor and
  `MergeFrom` of a map of messages, of a value that holds the map, each of what the value was
  before; and a repeated field of messages extended (or merged) by itself takes the messages that
  it has first, as a list's `extend` of itself, where upb's reads the field as it grows. A merge
  into a message that its value holds, the constructor (a second `__init__` too), `add`, `append`
  and `extend` of a value that holds the message are upb's: upb reads the value as it is when it
  reads it (it serializes the argument of `MergeFrom` first), after the fields before it are set
  and the new message of `add` and `append` is appended;
- the iteration order of a map is the order of insertion, where upb's is that of its hash table,
  which changes from one process to another (two processes of upb serialize the same `Struct` of
  other orders), so no program can rely on upb's. The serialization of a map writes its entries in
  the order of their keys of upb's deterministic serialization
  (`SerializeToString(deterministic=True)`), with `deterministic` or without it, where upb's
  serialization without it writes them in the order of its hash table; `str()` and `repr()` write
  them in upb's order of the keys of its text, and `text_format` in its own, as of upb's messages.
  upb's orders take the keys of a signed integer type as unsigned numbers, so that a negative key
  comes after the others;
- it makes a Python object of each message that it parses or merges into, where upb makes one when
  the message is read. A copy (of a message of the runtime given to a constructor, of `CopyFrom`,
  `copy.copy` and `copy.deepcopy`) holds what it copies natively, of no Python object, and makes
  the object of a message, a repeated field or a map of it, of one level, when Python first reads
  it, as upb makes its objects; what a copy holds natively, a copy of the copy shares, so
  `proto.Expression(merge_action=action)` copies an action of any size at once. `MergeFrom` copies
  its argument so first, as upb serializes it first, and merges the copy, of which a field that the
  message does not hold takes the value as it is (so `append` and `extend`, which merge each value
  into a new message, make no object of it either). So the parse, the copy of a message that a
  parse made and the release of a message of many messages take several times as long as upb's
  (of CPU time, of a message of a million messages: 0.3 s for the parse, upb's 0.05 s, 0.25 s for
  `CopyFrom`, `copy.deepcopy` and `MergeFrom`, upb's 0.1 s (0.09 s of `MergeFrom`), and 0.33 s for
  a `MergeFrom` into a message that has the repeated field already, whose objects it makes of the
  copy, upb's 0.08 s), and a read of every message of a copy takes as long as the copy of each
  message as an object took before. The parse, the constructor (of lists and mappings too),
  the copies (`CopyFrom`, and `copy.deepcopy` of a message or of a repeated field of messages), the
  merges (`MergeFrom`), the `extend` of a repeated field of messages and the `update` and
  `MergeFrom` of a map let another thread take the GIL, and a signal interrupt them with the
  exception of its handler, every 1,024 steps of the operation (a message that it reads, makes or
  copies, an entry of a map, or a value of a repeated field that the constructor takes), where upb
  holds the GIL until they end (but in the `update` and `MergeFrom` of a map and the constructor of
  a map of scalars, which run `MutableMapping.update` in Python, as the runtime's constructor runs
  it of a value that is not a dict), so another thread can use a message that one of them reads
  meanwhile. `SerializeToString`, `==` of two messages of the runtime and the release of a message
  hold the GIL until they end, as upb's do: 0.09 s, 0.07 s and 0.08 s for a million messages, where
  upb's take 0.03 s, 0.13 s and 3 ms. The messages that a parse makes within the message that it
  parses into, those that Python reads of a copy (of `copy.deepcopy` too), and that of a read of an
  unset field are not tracked by the garbage collector (`gc.is_tracked`), as they make no reference
  cycles, so that a collection does not go through the messages of a large message; the message of
  a constructor, of `FromString` and of `copy.copy`, and that `ParseFromString` or `CopyFrom`
  fills, are tracked, as upb's;
- the classes of `Any` and `Timestamp` are the runtime's own, of its modules
  (`pyspark.sql.connect.proto._rust.any_pb2` and `timestamp_pb2`), where `protobuf`'s modules
  (`google.protobuf.any_pb2`) keep `protobuf`'s classes. Where `protobuf` imports, a message of
  one pickles as upb's, of `protobuf`'s class (`google.protobuf.any_pb2.Any`), of the same bytes,
  so the pickle loads wherever `protobuf` does, as in the Python worker of the server that runs a
  UDF that holds one, and in the drop-in as a message of `protobuf`'s class, which the runtime's
  fields take as they are, as one of PySpark's does; `copy.copy` keeps the runtime's class.
  Without `protobuf`, a pickle of protocol 4 or later names the class by the name that the module
  of the first proto of Spark Connect that imports its file binds it to, which is the class's own
  (`pyspark.sql.connect.proto.base_pb2` and `google_dot_protobuf_dot_any__pb2.Any`, whose `repr`
  the class has), of no other global: it loads in PySpark as a message of `protobuf`'s class (whose
  generated module binds `protobuf`'s module there), where `pyspark.sql.connect` imports (which
  needs `grpcio` and the other dependencies of PySpark's Spark Connect, as PySpark's pickles of the
  package's messages do), and in the drop-in of the runtime's, with or without `protobuf`. One of
  an earlier protocol, which has no such dotted names, names the runtime's function
  `pyspark.sql.connect.proto._rust._new_message`, and loads only in the drop-in, as the class
  itself does of such a protocol, which `pickle` names as the attribute of the runtime's module of
  its file, which pickles by its name.
  A message of a nested type pickles of `protobuf`'s `_InternalConstructMessage` where `protobuf`
  imports, as upb's, so a drop-in without `protobuf` does not load it (as it does not load
  PySpark's), and of its class's qualified name without `protobuf`
  (`pyspark.sql.connect.proto.expressions_pb2` and `Expression.Literal`), which loads in PySpark
  and in the drop-in, with or without `protobuf`.

Where `protobuf` is installed, the files of the protos of Spark Connect are added to its default
descriptor pool, as the generated modules add them. The modules that protoc generates of other
protos that import them (those of a plugin, such as Delta Lake's Connect client) then build, and
`json_format` finds the types of an `Any`. Such a module, of a file of proto2, proto3 or editions
that imports the protos of Spark Connect directly or through the files it imports, gets the classes
of the runtime when `protobuf`'s metaclass builds them: the drop-in wraps
`reflection.GeneratedProtocolMessageType` (and the one of `message_factory`), which the builder of
the modules of protoc 3.20 and later calls, as do the modules of an older protoc themselves, and
`message_factory.GetMessageClass`, and the files are added to the runtime's pool. So their message
fields take the messages of `pyspark.sql.connect.proto` as they are, as `protobuf`'s messages of
PySpark's plugins take PySpark's: Delta Lake 4.0's `DeltaTable.delete`, `update` and `merge` build
their plans with `CopyFrom` of the package's messages, of the bytes, text and JSON of PySpark's,
with either implementation of `protobuf`. The runtime reads the presence, defaults, packing,
required fields, closed enums, groups and the strings that a parse does not check of proto2 and of
the features of editions (of maps too), of upb's behaviors and errors of them. The module of a file
of the package of such a file that it imports (of a package: files of no package are not of one)
gets the runtime's classes too, also one that `protobuf`'s builder built first (Delta Lake's
`delta/connect/proto/__init__.py` imports `base_pb2`, of `delta/connect/base.proto`, of the package
`delta.connect`, which does not import the protos, before the modules that do): the runtime takes
over the module, found as the module of `protobuf`'s class of a message of the file, else as a
module of the top-level package of that module's name or of the plugin's module whose `DESCRIPTOR`
is the file and that has the descriptors of its builder (`_DELTATABLE`, which a package that
star-imports it has not), as a module of the name of protoc's path of the file
(`delta.connect.base_pb2`) is imported of another (`delta.connect.proto.base_pb2`), and each name
of a global of a module of its top-level package (`delta`) that is bound to one of `protobuf`'s
classes that they replace (the package's, of its star import), but a name bound to another object:
the runtime reads the names of `sys.modules` once of a top-level package, and rebinds the globals of
its modules once, of all the modules that it takes over at once. The classes are of the module that
`protobuf`'s builder named, as upb's. A file whose module the
runtime does not find keeps `protobuf`'s classes. Each file that imports a file that the runtime
holds so, but no proto of Spark Connect, gets the runtime's classes too, as Delta Lake's
`commands.proto` would of `base.proto` alone, of its module built before (of `protobuf`'s symbol
database) or after, so that its messages and those of the file take each other's as upb's of one
class of each type. The files of other packages that such a file imports (a company's shared
proto), and Google's (`google/protobuf/duration.proto`, `google/rpc/status.proto`), keep
`protobuf`'s classes, in their modules and of `protobuf`'s lookups, as do the files that no such
file imports, so that the modules of other libraries that import them (`grpcio-status`,
`google/longrunning/operations.proto`) take their messages as they do without the drop-in. A module
that protobuf's builder built before `pyspark.sql.connect.proto` is imported, as only a copy of
Apache Spark's generated modules of the same protos can, gets the classes of the package (of a
proto of Spark Connect) or the runtime's (of a file that imports them) once the package is
imported, of the modules of the classes of `protobuf`'s symbol database. Whether a file imports the
protos is known of each file once, and `message_factory.GetMessageClass` of a type of the default
pool that keeps `protobuf`'s class looks it up once, so neither reads `sys.modules` nor costs more
than a lookup of a dict afterwards; of a type of another pool, it costs the lookups of its file and
pool too. The `DESCRIPTOR` of such a class is `protobuf`'s descriptor of its type, the
module's (`_DELETEFROMTABLE`), and `protobuf`'s lookups of its type
(`message_factory.GetMessageClass`, `symbol_database`, and the load of a pickle of a nested type)
give the class. The relation of a DataFrame that such a message takes
(`df._plan.plan(session)`, as `merge` takes the plan of its source DataFrame) is a placeholder that
holds the relation of the DataFrame (see "SparkSession.client" below), as in the plans of
`pyspark.ml`: the core replaces it when the plan or the command reaches it, so the server receives
the relation, as PySpark's plugins send it: also anywhere in the message of the extension of a
relation or of a command, of a type that it does not know, as long as the message holds the
placeholder as `plan(session)` gives it, or with the `common` that the plugin sets on it.
"SparkSession.client" below lists the places where the core replaces it: under any relation, in
any command that holds one, and in the extension of an expression. What differs of such a module:

- a message field of such a module of a type of a file that keeps `protobuf`'s classes (of a
  well-known type other than `Any` and `Timestamp`, such as `Duration` or `Struct`, or of a file of
  another package) gives a message of the runtime's own class of the type, of the module and
  qualified names of `protobuf`'s (`google.protobuf.duration_pb2.Duration`), of protobuf's methods
  of a well-known type (`ToTimedelta`, the mapping of a `Struct`) and of upb's assignments,
  comparisons and membership of a `Struct` and a `ListValue`, where upb has one class of each type:
  `isinstance(message.timeout, duration_pb2.Duration)` is false, and `protobuf`'s messages of other
  files take the runtime's message only through its bytes or a dict (their `CopyFrom`,
  constructors and containers raise upb's `TypeError` of a message of another class, as
  `struct_pb2.Struct(fields=message.options.fields)` does, where
  `struct_pb2.Struct.FromString(message.options.SerializeToString())` reads it), and errors name
  the runtime's class, though the two are equal (`==`) of the same contents and the runtime's
  fields take `protobuf`'s messages as they are. A message of one pickles as upb's, of
  `protobuf`'s class, so it loads as a message of `protobuf`'s class, where `copy.copy` gives one
  of the runtime's;
- the class of the runtime of such a type pickles as `protobuf`'s class of the type (`pickle`
  gives `protobuf`'s class), as its messages do; `cloudpickle` pickles a class that is not of its
  module's name by value, which it cannot of such a class, so a function that refers to one does
  not pickle;
- the `_concrete_class` of `protobuf`'s descriptor of a type of the runtime's classes is, of upb,
  upb's `SystemError: error return without exception set`, as of a type that upb made no class of
  (upb gives only the classes that it made), and, of `protobuf`'s pure-Python implementation, the
  runtime's class, which the runtime sets of a descriptor of no class, as `protobuf`'s metaclass
  does, but never in place of `protobuf`'s class of a descriptor (of a well-known type, or of a
  module built before), so `protobuf`'s messages of other files make and take theirs;
  `message_factory.GetMessageClass` gives the runtime's class of either;
- a file that declares extensions or extension ranges, or a message set, keeps `protobuf`'s
  classes, as the runtime has no extensions, and so do the files that import it (as a file of
  custom options imports `google/protobuf/descriptor.proto`): their classes take the package's
  messages, and those of a plugin's module of the runtime's classes, only through bytes or dicts,
  where upb's take them as they are. Since upb's classes of such a file take only upb's messages
  of the files that it imports, a file that the runtime took over (of a plugin's package, or that
  imports one that it holds) that it imports gets `protobuf`'s classes back, in its module, the
  names of the modules of the plugin's top-level package and `protobuf`'s lookups, with the files
  that it imports (but a proto of Spark Connect and a file that imports one), whether the file
  builds before the plugin's or after (also of a module built after the runtime held its file of a
  lookup, as `message_factory.GetMessageClass` of a plugin's type of the default pool holds its
  files before their modules build): its messages are then of `protobuf`'s classes, as without
  the drop-in, and the runtime's classes of its types are of the shadows of a file of another
  package (above). What the give-back does not reach keeps the runtime's class, which upb's
  classes of such a file refuse (`TypeError`, where a dict or the bytes are taken): a name bound
  outside the modules of that top-level package before (`from gb.common_pb2 import Common` in
  `__main__`), a message of the runtime's class made before the give-back, and the values of the
  plugin's message fields of the types, which are the runtime's messages;
- the names that the runtime binds to its classes in place of `protobuf`'s are the globals of the
  modules of the top-level package of the module that it takes over, so `protobuf`'s class of a
  type that a module of another package or another object holds (an attribute, a closure) stays,
  as do the messages that it made before, which the runtime's fields take as they are. The
  globals of a module are read of its `__dict__`, so that a lazy module
  (`importlib.util.LazyLoader`) is not loaded.

The other way, the messages of `pyspark.sql.connect.proto` take `protobuf`'s messages of their types
as they are, the `MergeFrom` of their maps takes the items of `protobuf`'s maps, as upb's takes
those of any mapping but a dict, and an `Any` of either packs and unpacks a message of either. A
message of the runtime equals (`==`, on either side) a message of `protobuf`'s of its type of
`protobuf`'s default pool of the same contents, as upb's messages of its one class of a type do: the
runtime parses the bytes of `protobuf`'s message to compare it, as a field of the runtime takes it,
so `in`, `remove` and `==` of the containers find such messages too. Of `protobuf`'s pure-Python
implementation, whose `==` of its message reads the unknown fields of the other
(`_unknown_fields`), a message of the runtime has them, of the bytes of the tag and of the value of
each. A message of a type of another
pool, of another class for upb too, is not equal, and `CopyFrom`, `MergeFrom`, the constructors and
the containers refuse it with upb's errors of a message of another class, as they refuse an object
that only has the `DESCRIPTOR` and the `SerializeToString` of a message. As upb's, `==` of two
messages compares a float or a double field of its bits, so that a NaN equals a NaN of the same bits
and -0.0 is not 0.0 (nor the default of a field without presence), and their unknown fields of their
numbers, wire types and values, in the order of their tags, so that the order of fields of other
tags and the encoding of a varint do not count. `==` of the containers and maps compares their
values as Python's `==` does, as upb's `list` and `Mapping` comparisons do: a NaN value is not equal
to itself, and a map is equal to a map of other key or value types of equal keys and values.

`google.protobuf.unknown_fields.UnknownFieldSet` is the drop-in's wrapper of `protobuf`'s class,
once `pyspark.sql.connect.proto` is imported, which takes the messages of the package too, where
`protobuf`'s class refuses them ("Expected a message object"). The set of such a message is
`protobuf`'s set of a message of `protobuf`'s `google.protobuf.Empty` that holds the unknown
fields of the message, so it has the fields, values and order of upb's, and
`text_format.MessageToString(message, print_unknown_fields=True)`, which reads the attribute of
the module, prints them as upb's. Of other arguments, the wrapper gives `protobuf`'s set or raises
`protobuf`'s error. Its sets are of `protobuf`'s class, as `isinstance` of the wrapper tells, but
the wrapper is not that class (`type(UnknownFieldSet(message)) is UnknownFieldSet` is false), and
a name bound to `protobuf`'s class before the package is imported
(`from google.protobuf.unknown_fields import UnknownFieldSet`) refuses the package's messages.

A DataFrame of a plan of `pyspark.sql.connect.plan` (`DataFrame(plan, session)`), such as a plan
of `pyspark.ml`, keeps the plan ID of the plan (`plan._plan_id`), which its columns refer to and
its relation is sent with, as PySpark's DataFrame keeps its plan: the plans of the module take
their IDs of the counter from which the core numbers the relations of its plans, one for the
process, as PySpark numbers its plans of one counter, so a plan or a relation of any session is
never numbered of the ID of another. The plan of a DataFrame of the core (`df._plan`) has the plan IDs of the
DataFrame (`_plan_id`, `_root_plan_id` of the root of its `WithRelations`), and
`DataFrame(df._plan, session)`, also of a subclass of the DataFrame, is a DataFrame of the plan of
`df`, in its session or another, which sends it of those IDs, as PySpark's is of the plan object of
`df`: its columns and those of `df` refer to one plan and resolve against either, and a join of
the two is a self-join, whose condition of their columns the server rejects as
`AMBIGUOUS_COLUMN_REFERENCE`, as PySpark's. Its requests build the plan, as PySpark's do, so they
pickle the functions of its UDFs, of the `PicklingError` of one that does not pickle, and raise the
error of an option that UTF-8 cannot encode, at each request; the DataFrame is made. Of another
plan, the DataFrame is of the relation of `plan.to_proto(session.client)` when it is made (see
"What a plan build does for a pickle" in `dev/parity/known-differences.md`), and of a plan whose
build raises an `Exception`, the DataFrame is made, and each of its requests raises a new exception
of the one of that build, of its class, its arguments, its state, and of copies of its cause and
its context, as each of PySpark's requests builds the plan and raises an exception of its own: the
DataFrame keeps the exception and its chain of no traceback, so that it keeps no frame of the build
alive, and each exception of a request keeps the frames of that request alone. Another exception
of the build, such as the `KeyboardInterrupt` of a Ctrl-C while it runs, or a `SystemExit`, is
raised by `DataFrame(plan, session)` itself, as it is where PySpark's program is when it comes,
and a later `DataFrame(plan, session)` builds the plan again; PySpark's DataFrame, which builds
its plan at its requests, raises the exception of a plan that raises it there. A pickle of the
DataFrame, of a Column of its subquery, of its table argument or of a DataFrame over it carries
the plan, as PySpark's pickle carries its plan objects, and the unpickle builds it again, as
PySpark's builds them when the unpickled object is used, of the error of that build, of the class,
the arguments, the state and the chain that it gives then: the unpickle of the DataFrame raises
it, as PySpark's, which builds the plan to read the columns of the new DataFrame, but for an
`AttributeError`, of which it raises the `NOT_ITERABLE` of PySpark's `__getattr__`, and the
requests of an unpickled table argument raise it, also on the server's Python worker, which
builds the plan at each request, as PySpark's does. An unpickled Column of its subquery builds the
plan at each request, of the session of the request, also on a thread that has no active session,
as PySpark's Column builds it at each request of the client of the request, and a pickle of it
carries the plan again. A build of a plan over it outside a request builds it of the session of
the client that the build is given, as `df._plan.plan(client)` and `df._plan.to_proto(client)` of
a DataFrame over it do, as PySpark's builds it of that client, and of the session of the DataFrame
for an object that is not a client of the drop-in. Its `to_plan(client)`, and that of a Column
over it, raises the error of that build, where PySpark's gives the expression, as PySpark builds
the plans of the subqueries of a Column with the relation of a plan, and as the Column that was
pickled gives it. The pickle carries the plan and not that of an object over the DataFrame of the
plan, so of such an object, as a DataFrame over it or a Column of its subquery, whose plan builds
when it is unpickled or requested, the requests raise a `RuntimeError` that says so, where
PySpark's work. The plan of a DataFrame of the core (`df._plan`) that such a plan holds, as a
`SubqueryAlias(df._plan, "s")` does, pickles of the relation of the DataFrame as it was sent, of
the functions of its UDFs, of the error of a plan that does not build and of the Observations of
the plan, as a plan of `pyspark.sql.connect.plan` that has no session, as PySpark's plan object of
a DataFrame pickles its fields: so a plain pickle of a Column of the subquery of such a plan, or of
its table argument, pickles, and a UDF or a `foreachBatch` function that captures one unpickles on
the server's Python worker, as PySpark's do. The plan that a pickle or a cloudpickle gives back
pickles again, plainly too, of the functions of the drop-in's module by their names, as PySpark's
plan objects do. The unpickled plan has copies of the Observations (`observations`), as
PySpark's, which an execution of the client that is given them sets; an execution of a DataFrame
of a plan over it on the client does not set them, where PySpark's does, as the core takes the
Observations of the DataFrames whose plans a plan holds, and the unpickled plan holds the relation
alone. The `_plan` of a `DataFrame(plan, session)` is the plan of its relation, not `plan`, as
PySpark's is, and a new plan at each read.

## SparkSession.client

`spark.client` is PySpark Connect's `SparkConnectClient` (of `pyspark.sql.connect.client.core`,
which its package names too), with the public API of pyspark-client 4.3.0, over the Rust transport
of the session. It sends the messages of `pyspark.sql.connect.proto` that it is given as they are,
and reads their responses as PySpark's client does:

- the analyses of a `pb2.Plan` (`schema`, `explain_string`, `same_semantics`, `semantic_hash`), the
  configuration requests of a `ConfigRequest.Operation` (`config`, which returns a `ConfigResult`),
  and the executions of a plan (`to_table`, `to_table_as_iterator`, `to_pandas`) or of a command
  (`execute_command`, `execute_command_as_iterator`), with the `ExecutionInfo` of the execution, the
  Observations that they are given set by their names, and the progress of the progress handlers
  (`register_progress_handler`), except for `execute_command_as_iterator`, as in PySpark. The
  observed metrics of an `ExecutionInfo`, of `DataFrame.executionInfo` and of the `attrs` of
  `to_pandas` are PySpark's `PlanObservedMetrics`: their `metrics` are the `Expression.Literal`
  messages that the server sent, of their `repr`, which `pairs` and `flows` convert as
  `LiteralExpression._to_value` does, raising its `UNSUPPORTED_LITERAL` when they are read;
- the registrations of `register_udf`, `register_udtf`, `register_java` and
  `register_data_source`, which set their arguments in the fields of their commands, convert the
  return type and pickle the function in the order of PySpark's, of protobuf's errors for a value
  that a field does not take (an int is a bool, a bool is not an int), the artifacts of
  `add_artifacts`, `copy_from_local_to_fs`, `cache_artifact` and `cache_artifacts`, with the
  checks of PySpark's client rather than those of `SparkSession.addArtifacts` and
  `copyFromLocalToFs`, the tags, the interrupts, `token`, `host`, and the clients of `newSession`
  and `clone`. `spark.udf.register`, `registerJavaFunction` and `registerJavaUDAF` register
  through the client, as PySpark's do. The executions read their `observations` as PySpark's: of
  an object that is not a dict, such as None or an empty list, they set none, and of one that is
  not a container, such as None, they raise the `TypeError` of `in` when observed metrics arrive.

`SparkConnectClient(connection, user_id=None, ...)` takes the arguments of PySpark's: it makes a
client of a session of its own of the connection string or the `ChannelBuilder` `connection`, which
is neither the default nor the active session. Its `retry_policy`, `use_reattachable_execute`,
`session_hooks`, `allow_arrow_batch_chunking`, `preferred_arrow_chunk_size`, `rpc_deadlines`
(`pyspark.sql.connect.client.RpcDeadlines`) and
`max_retry_exception_elapsed_time` are those of the transport. A `channel_options` of a connection
string raises `NOT_IMPLEMENTED`, as `DefaultChannelBuilder` does: the Rust transport opens the
channel; as PySpark's, the client of a `ChannelBuilder` does not read it.

The members of PySpark's client that its tests and some libraries read are there too. `_builder`
is the channel builder of the connection, a `DefaultChannelBuilder` of the session's connection
string for `spark.client` (of its `endpoint`, `userAgent` and parameters), and the one given for a
session or a client made of one. `_session_id` and `_user_id` are attributes of the client, as
PySpark's: the session ID and the user that the requests carry, which a set changes for them, of
the error of protobuf for a value that the string field does not take when the next request is
built, while the session's `session_id` stays, and so do the session ID and the user of the
requests of the artifacts, as PySpark's ArtifactManager keeps those that the client was built with.
The next response of another server-side session than the one that the client observed, or an
`INVALID_HANDLE.SESSION_CHANGED` of the server, marks the session closed, as PySpark's sets
`_closed`, of an open channel: its artifacts are sent, and it is open again once `_closed` is set to
False. The sessions of the `newSession` and the `clone` of the client, and of those of the session,
are of its `_user_id`, unless the connection has a `user_id`, and so are their artifacts.
`_server_session_id` is the server-side session ID. `_closed` marks the session closed
when it is set to True, so that its requests raise `NO_ACTIVE_SESSION` but those of its artifacts,
which PySpark's ArtifactManager sends of a stub of its own, and opens it again when it is set to
False; `close()` and `stop()` close the channel too, after which the requests raise grpc's
`ValueError` of a closed channel once `_closed` is False (`Cannot invoke RPC: Channel closed!` of
an execution, `Cannot invoke RPC on closed channel!` of the others, and of the artifacts, closed or
not), and a `newSession` is of a channel of its own, as PySpark's. `_use_reattachable_execute` is
whether the executions are reattachable, and `_get_operation_statuses` the statuses of the
operations of the session, of a GetStatus RPC under the `get_status` deadline. The client and the
`conf` of a session hold it as PySpark's hold its client: once the program dropped the session and
the objects that hold it, such as its DataFrames, a kept `conf` or client raises
`NO_ACTIVE_SESSION`, as PySpark's do once the `__del__` of the session closed its client, and such
a client keeps its attributes, as PySpark's: its server-side session ID, whether its executions are
reattachable and its retry policies among them. The differences that remain are in
`dev/parity/known-differences.md` ("Client session ID" and "The attributes of
`SparkConnectClient`").

The retry policies of `get_retry_policies` and `set_retry_policies` are PySpark's own
(`pyspark.sql.connect.client.retries`, vendored from v4.3.0, which needs `grpcio`, `grpcio-status`
and `googleapis-common-protos`). Policies of PySpark's classes `DefaultPolicy` and `RetryPolicy`
themselves, of backoffs, a jitter and a delay of whole milliseconds, are retried by the transport,
without Python: of the parameters of the first `DefaultPolicy` of the list, its retries, backoffs,
jitter and the delay that the server asks for, as PySpark's `Retrying` retries with them, of the
failures that `DefaultPolicy.can_retry` retries; a `RetryPolicy` retries none. It reads the
parameters of the `_retry_policies` of the client as each call starts, so a change to a policy, or
to the list, applies from the next call on. Any other policies, such as those of a subclass of a `can_retry` or a `to_state` of
its own, are run by PySpark's `Retrying`, which the transport makes for each call, as PySpark's
client makes one, and asks of each failed RPC, of grpcio's `RpcError` of it (`_InactiveRpcError`, or
`_MultiThreadedRendezvous` of a stream of responses): `Retrying` calls the methods of the policies,
and logs and warns as in PySpark, with the GIL, on the thread of the call, or on the delivery thread
of the drop-in for a request that the transport sends in the background, and the transport waits the
backoff that it gives. The client of a new session has the default policy, as PySpark's `newSession`
and `clone` do. With `disable_reattachable_execute`, an execution has no reattach option and no
deadline, is never released, and is sent again when it fails in a way that its policy retries, as
PySpark's.

The state of the client is the `__dict__` of its attributes, as PySpark's: `thread_local`,
`_builder`, `_user_id`, `_retry_policies`, `_max_retry_exception_elapsed_time`, `_session_id`,
`_closed`, `_rpc_deadlines`, `_use_reattachable_execute`, `_allow_arrow_batch_chunking`,
`_preferred_arrow_chunk_size`, `_session_hooks`, `_server_session_id`, `_profiler_collector`,
`_progress_handlers`, `_plan_compression_threshold`, `_plan_compression_algorithm`,
`_release_session_on_exit`, `global_user_context_extensions` and
`global_user_context_extensions_lock`, in that order, of PySpark's values, which the clients of a
session share, as PySpark's session has one client, and of which `copy.copy` of a client makes a
dict of its own, of the same values. The transport reads them as each call starts, under the GIL,
as PySpark's requests read them when they are built: a change of an attribute, of an assignment, of
the object that it holds (`client._session_hooks.append(hook)`) or of `vars(client)`, applies from
the next call on, of the requests of the client of the call, or of the copy whose method made it.
The methods read and set them as PySpark's: `set_retry_policies` sets `_retry_policies`,
`enable_reattachable_execute` and `disable_reattachable_execute` set `_use_reattachable_execute`,
`register_progress_handler`, `remove_progress_handler` and `clear_progress_handlers` change the list
of `_progress_handlers`, which the `registerProgressHandler`, `removeProgressHandler` and
`clearProgressHandlers` of the session call, `spark.profile` is a `Profile` of the
`_profiler_collector`, which the profiles of the executions update, and `close()` sets `_closed`, as
`stop()` and the drop of the last object of the session do. The responses set `_server_session_id`,
the `_closed` of a server-side session change and the plan compression settings of their lookup,
and a None of either setting makes the next plan look them up again. The exit of the interpreter
reads `_closed` and `_release_session_on_exit`, as PySpark's `_on_exit` does. A client whose
`__init__` never ran raises the `AttributeError` of `_closed`, which PySpark's `is_closed` reads.
The `[DefaultPolicy()]` of `_retry_policies` is made at the first read of the attribute, or at a
`copy.copy` of the client, as its module imports grpcio, which the drop-in does not require. The
client has none of the attributes of PySpark's that hold the objects of grpcio and of its Python
compression of plans, `_channel`, `_internal_stub`, `_artifact_manager`, `_release_futures` and
`_zstd_module`, which the native transport replaces (see `dev/parity/known-differences.md`, "The
attributes of `SparkConnectClient`").

The relation of a DataFrame's plan (`df._plan.plan(client)`) is a placeholder that holds the
relation of the DataFrame, which the core builds at the call, as PySpark's `plan` builds the
relation then: the call pickles the functions of its UDFs, warns of a window of no partitions and
raises the errors of the build, and no request is sent. The core builds the relation, and Python
has no messages of it, so the placeholder is one message, an extension whose value holds the
relation encoded, which Python's protobuf does not parse, whatever the depth of the DataFrame,
where PySpark's `plan` gives the messages of each level of the relation: so its fields are not
those of the relation (`WhichOneof("rel_type")` is `"extension"`). The core replaces the
placeholder by the relation that it holds when it sends a plan, a command or an analysis that
holds it, also after the DataFrame is gone, and when the message is parsed again of its bytes, as
PySpark's relation is sent. It replaces it wherever PySpark's plans and commands hold the relation
of a DataFrame:

- as the root relation of a plan, of an execution, of an analysis (`schema`, `explain_string`,
  `same_semantics`, `semantic_hash`) or of `DataFrame(plan, session)`, and as the relation of the
  analyses of a relation (`persist`, `unpersist`, `get_storage_level`);
- under it, as the input of any relation of Spark Connect, both sides of a join, a set operation
  or a zip and the other inputs of a `GroupMap` and a `CoGroupMap` included: those of
  `pyspark.sql.connect.plan`, such as `SubqueryAlias(df._plan, "s")` and the views of its `SQL`
  (`SQL("SELECT * FROM v", views=[SubqueryAlias(df._plan, "v")])`, aliases among the references
  of a `WithRelations`), those of `pyspark.ml` (an `MlRelation`'s input, its model summary dataset
  and its fetched arguments), and those built with `pyspark.sql.connect.proto`;
- as the relation of a command that holds one (a `WriteOperation`, `WriteOperationV2`,
  `CreateDataFrameView`, `WriteStreamOperationStart`, `SqlCommand`, `CheckpointCommand` or
  `MergeIntoTableCommand`, and the flow of a pipeline), and as the dataset of the ML commands
  (`Fit`, `Evaluate`, `CreateSummary`) and the arguments of a `Fetch`;
- anywhere in the message of the extension of a relation or of a command, a plugin's message such
  as Delta Lake's `MergeIntoTable` (see the Delta Lake paragraph above), and of the extension of an
  expression of a relation or a command built in Python, such as a plugin's expression that holds
  the relation of a DataFrame.

A placeholder whose `common` a plan or a plugin's message changes (`plan.common.plan_id = ...`), as
PySpark's relation of `plan(session)` is changed, is replaced by the relation with that `common`
merged into its own, as protobuf's `MergeFrom` sets the fields, so its plan ID is the one set, also
in the message of an extension. The core does not walk the expressions of a plan built in Python: of
each relation of the plan of a kind that holds expressions (a projection, a filter, a join, an
aggregation, ...), it encodes the fields other than its inputs, and reads them for a placeholder
only when the encoding holds the type URL of one; the relations that the core builds, such as that
of a placeholder, are not encoded. It encodes and reads them, and the messages of extensions, only
when the bytes that the client gives of the message (of `plan(session)`, or of the plan, the
command, the analysis or the request of a hook) hold a type URL of a placeholder that is not of one
at the relations of the plan, which the core meets as it walks them: so a plan whose placeholders
are all at its relations, or that holds none, costs a search of its bytes for the URL, and no
encoding. The messages of extensions are read in a time of their length, however deep they nest.

The `observations` of a DataFrame's plan (`df._plan.observations`) are the Observations of the
DataFrame by their names, as PySpark's `LogicalPlan.observations`: those of `observe` on it and on
its inputs, both sides of a join and of a set operation, in the order of PySpark's dict, but not a
DataFrame observed by a name. Each is the object that `observe` was given, of its class and its
attributes, which the plan holds, as PySpark's plan does, also once the program dropped it. A plan
of `pyspark.sql.connect.plan` over it, such as `SubqueryAlias(df._plan, "s")`, has them too, and an
execution of the client that is given them (`client.to_table(plan, df._plan.observations)`) sets
them, as PySpark's DataFrame gives them to its client. An Observation that is not used, given for
the name of metrics that the plan observes (`client.to_table(plan, {"x": Observation("x")})`),
fails the execution with PySpark's `AssertionError` when those metrics arrive.

As PySpark's, `_name`, `_result` and `_error` are the attributes of an Observation, which `vars()`
shows and a program sets and deletes, and `DataFrame.observe` of an Observation gives what its `_on`
gives, of a subclass too. Each build of a plan of the Observation, at each request of its
DataFrames, reads its `_name`, as PySpark's `CollectMetrics` does: the plan is of `str(_name)`, by
which the metrics are set, so an Observation renamed after `observe` is observed by its new name,
and a name whose `__str__` raises, or a str that UTF-8 cannot encode, fails each request with that
error until it is renamed again; `observe` does not read it. A deep copy of a DataFrame holds a copy
of the Observation, of its name then. The `get` of an Observation raises its `_error` if it is not
None, and is its `_result` otherwise: one dict, which the executions that set its metrics update in
place, as it is then, so a dict read before an action has the metrics once the action ran, what is
set on it stays, and a `copy.copy` of the Observation shares it. `_result` is None before the
Observation is used, which `observe` checks, so an Observation whose `_result` a program sets to
None is used again, and an execution of its metrics then raises PySpark's `AssertionError`. The
executions convert the metrics when they set them, so a metric that does not convert, such as a
timestamp of the year 1 that `datetime` cannot make local, raises the error of its conversion from
the action, and none is set. The error of collecting them that the server reports is given to the
`_set_error` of the Observation, which sets `_error`, the one exception that `get` raises. A
reference cycle through the attributes, as through the result dict, is freed.

The errors of the RPCs that PySpark's client leaves to grpc, those of `clone` (and
`SparkSession.cloneSession`), of `copy_from_local_to_fs` (and `SparkSession.copyFromLocalToFs`) and
of `cache_artifacts`, are grpcio's `RpcError` (`_InactiveRpcError`), as they are there: of the code,
the details, the `debug_error_string` and the trailing metadata of the failed call, neither
converted nor logged, where grpcio is installed. Without grpcio, whose classes they are of, they are
converted as those of the other RPCs: they are the exception of their error class, such as
`UnknownException` or `UnsupportedOperationException`.

The clients of a session share their attributes, as PySpark's session has one client: an
attribute that is set on `spark.client` is there the next time, and so are its lock, its
`_session_id` and `_user_id`, and its `global_user_context_extensions`, the list attribute of
PySpark's client, whose changes, and a list that it is set to, the requests of all the threads
carry, while a list that it is no longer set to is not read. The `thread_local` of the client is
PySpark's plain `threading.local`: its `tags` and `user_context_extensions`, a plain set and a plain
list, exist once a thread uses them, and are the set and the list that the requests of the thread
carry, as they change or are set, as the tags of `SparkSession.addTag` and `getTags` are; another
attribute that a thread sets is its own. As PySpark's, an execution gives a thread of no tags an
empty set of them, as the `get_tags` of its request does, and a failed request leaves
`inside_error_handling` False, as its `_handle_error` does, but one that fails while the client
converts another failure on the thread, which leaves it as it is. A thread of
`inheritable_thread_target` shares the copies of the attributes of the thread that decorated its
function, as PySpark's does.
The transport reads them, on the calling thread, of the client that it calls, when a call starts its
requests and when a finalizer sends its request (the release of the relation of a checkpointed
DataFrame or of the cached objects of a model), as PySpark reads them when it builds each request,
so a `google.protobuf.Any` of the lists that is changed in place is sent as it is; the `next()` of
the iterator of an execution reads nothing of them, as its reattaches and releases copy the request
of the execution, as PySpark's.
The set of tags takes any object, as PySpark's: a tag that the string field of the tags of a request
does not take, an object that is neither a str nor bytes, bytes that are not UTF-8 or a str that
UTF-8 cannot encode, fails each execution, whose request carries the tags, with protobuf's error,
before it is sent, while the analyses, the configurations and the interrupts are sent. The lists
of the extensions take any object, as PySpark's: an extension that is not an `Any` (another
message, or not a message), or an item that is not a pair, fails each call whose requests carry the
extensions, those of executions, analyses, configurations and interrupts, with upb's `TypeError`
(or the error of the unpacking of the item) before the request is sent, of the global extensions
first, as PySpark's requests raise it when they are built, until it is removed, while the calls of
the artifacts, `cloneSession`, `stop` and the release of a session at exit, whose requests carry
none, are made. The error names the
runtime's class of `Any` (`pyspark.sql.connect.proto._rust.any_pb2.Any`), where upb's names
`google.protobuf.any_pb2.Any`.
A `copy.copy` of the client has a copy of the attributes of its session, as PySpark's: the
requests of its methods carry its own session ID, user and lists of extensions, also once the copy
rebinds them and at once with its client on another thread, and those of the session, as those of
a progress handler or of a finalizer that runs during a method of the copy, the attributes of the
session's own client, after which the requests of the method, as those of the details of its
failure, carry the copy's again, as they are then. The requests of the details of the failure of a
call, of the configs of its stack trace and of the interrupt of a signal read the attributes of the
client of the call when they are built, as PySpark's, also those of the iterator of an execution of
a copy after a call of the session between two of its `next()`, and the error of their build, as
of an extension that is not an `Any`, is the error of the call, of no request of the failure and of
no interrupt, which goes with a failure that the call handles itself; the other requests of a call
carry those of the start of the call (see "`SparkConnectClient.thread_local`" in
`dev/parity/known-differences.md`). The official tests of the client
(`sql/tests/connect/client/test_client*.py`) mock the gRPC stub and the reattachable iterator of
PySpark's client, which the drop-in does not have; those of its retries (`test_connect_retry.py`)
and of `pyspark.errors.exceptions.connect.convert_exception` (`test_connect_errors_conversion.py`)
pass.

## Spark Declarative Pipelines CLI

The CLI of Spark Declarative Pipelines (`python -m pyspark.pipelines.cli init`, `run` and `dry-run`)
and its modules are Apache Spark's own, vendored from v4.3.0, but for those that run a pipeline over
the client (`spark_connect_pipeline` and `spark_connect_graph_element_registry`), which build its
commands in the Rust core, and Apache Spark's tests of them pass with this client. The CLI needs
`pyyaml`, which `pyspark-client-rust` depends on, as `pyspark-client` does. The user context
extensions of a client (`spark.client.add_threadlocal_user_context_extension` and
`add_global_user_context_extension`), which the CLI adds for the analyses of a pipeline, are sent
with the requests of executions, analyses, configurations, error details and interrupts, as PySpark
sends them.

## How parity is verified

Two independent gates in CI cover the two halves of a Spark Connect client - plan
building and the transport/result path:

- **Golden-proto tests** assert that the `spark.connect` protobuf plans this
  client builds (plans, expressions, and every SQL function) match the reference
  client **byte-for-byte**. This guards plan-building correctness.
- **The official Apache Spark Connect test suite** runs the standard PySpark tests
  against this client, exercising the **transport and Arrow result paths**
  end-to-end against a real server. See the official-test-suite notes
  and the CI notes under `dev/design/`.

Together the two gates cover what the client sends and what it does with what
comes back.

## The parity ledger

Coverage is tracked mechanically. `scripts/gen_parity_ledger.py` AST-parses the
reference PySpark source and emits one row per public **class**, **function**, and
**method** into a ledger (`dev/parity/inventory.csv`), so nothing is silently
dropped. The current inventory covers roughly:

| Kind      | Count |
|-----------|-------|
| Methods   | ~1360 |
| Functions | ~620  |
| Classes   | ~300  |

Each row carries a status:

- **done** / **verified** - mirrored by the client (and, where applicable,
  confirmed against the reference).
- **n/a** - *architecture-satisfied*: the equivalent wire behavior is provided by
  the Rust core (`spark-connect` / `-core` / `-proto`), so a reference-private or
  generated symbol is not mirrored one-to-one.

The ledger is a development tracking tool (kept under `dev/parity/`); the official
test suite is the authoritative gate.

## What "drop-in" means

From Python, `pyspark-client-rust` replaces the `pyspark-client` package: same
`import pyspark`, same public API, same server - see [Installation](installation.md).
Use it exactly like [PySpark](https://spark.apache.org/docs/latest/api/python/).

The native [Rust API](dataframes.md) mirrors the same surface, but for the APIs that need Python
itself: the functions and classes of Python, which the server runs in Python, that `udf`, `udtf`
and `arrow_udtf` of `pyspark.sql.functions`, `udf.register`, `udtf.register`,
`dataSource.register`, `Catalog.registerFunction` and `GroupedData.apply` take, where the Rust API
takes their commands as Python pickled them, as `SparkSession::register_function`,
`register_data_source` and `GroupedData::apply_in_pandas` do, and has UDFs of its own in
[WebAssembly](udfs.md); pandas, of `toPandas`, `pandas_api` and `plot`, where the Rust API gives
Arrow record batches (`DataFrame::collect_record_batches`); and Python's implicit session, of
`SparkSession.active` and `getActiveSession`, where a Rust program passes its session. A
`SparkSession.Hook` is a closure in the Rust API (`SparkSession::set_execute_plan_hook`), and
`SparkSession.client`, which PySpark gives for tests, is the core crate's `SparkConnectClient`,
which a session does not expose.

Three differences come with being a different package with a native core. The distribution is
`pyspark-client-rust`, so `importlib.metadata.version("pyspark-client")` raises
`PackageNotFoundError`, where `pyspark.__version__` is PySpark's. A traceback of an error that the
server raises has the frames of the user's code only, without those of PySpark's client
(`collect`, `_to_table`, `_execute_and_fetch`, ...), which the native core does not have. At run
time, `pyspark.sql.SparkSession` and the functions of `pyspark.sql.functions`, PySpark's own, have
PySpark's signatures and docstrings, and so do the functions of `pyspark.sql.connect.functions`, of
which PySpark Connect's copy them too, while the methods of the native classes, such as
`DataFrame.select`, have PySpark's parameters and docstrings, without its annotations.
The docstrings of the modules of `pyspark.sql` and `pyspark.errors`, of their classes and functions
and of the members of the classes, are PySpark's: the types of `pyspark.sql.types`, the classes and
the members whose docstrings PySpark Connect's modules copy of PySpark's classic classes (the
`SparkSession`, the registrations of UDFs, UDTFs and data sources, the `Catalog`, the `RuntimeConf`,
the `GroupedData` and the `PandasCogroupedOps`, the `MergeIntoWriter` and its clauses, the
`Observation`, the `TableValuedFunction`, the readers and the writers of DataFrames and of streams,
and the `StreamingQuery` and its manager), and the others, of the modules of `pyspark.sql.connect`,
such as the `Builder` and the `Hook` of the `SparkSession`, and the `SparkConnectClient` and the
other classes of `pyspark.sql.connect.client.core`, as PySpark's source writes them. The native
classes are built with them, of the source of PySpark at the tracked tag
(`scripts/gen_connect_docstrings.py`), but for the `__init__` of a native class, the slot of
CPython, whose docstring is CPython's (see `dev/parity/known-differences.md`).
The DataFrame, the Column, the WindowSpec, the TableArg, the `DataFrameNaFunctions` and the
`DataFrameStatFunctions` of Spark Connect and their members have the docstrings of PySpark's classic
classes, which `inspect.getdoc` and `help()` show of PySpark Connect's, whose own are None (see
`dev/parity/known-differences.md`). The classic `pyspark.sql.DataFrame`, the public base of the
native DataFrame, takes the docstrings of the native class, and PySpark's of its classic DataFrame
where the two differ (`python/pyspark/sql/_classic_docs_generated.py`): of the aliases, such as
`groupby`, whose docstring that `help()` shows of PySpark Connect's is that of the member that it
names, and of the members that PySpark Connect's DataFrame wraps in a cache of functools, such as
`isEmpty`, of which `help()` shows none. `tests/golden/runtime_docstrings.json` holds the digests of
PySpark's docstrings of every module of `pyspark.sql` and `pyspark.errors`, which a test compares
with the drop-in's: of the `__doc__` of each object, or of the `inspect.getdoc` of one of no
docstring of its own that inherits one, also of the members of the classes of class bodies, such as
`SparkSession.Builder`.
A default of an object, which the text signature of a native method cannot hold (`inspect` takes
only literals there), shows as `Ellipsis`: the `StorageLevel(True, True, False, True, 1)` of
`DataFrame.persist`, and the `<no value>` (`pyspark._globals._NoValue`) of the `value` of
`DataFrame.replace` and `DataFrameNaFunctions.replace` and of the `default` of `RuntimeConf.get`.
Their calls take PySpark's defaults.
The `__init__` of a native type of parameters, such as `ArrayType`, is a slot of its class, whose
`inspect.signature` is CPython's `(self, /, *args, **kwargs)`; that of the class shows the
parameters, as PySpark's does.

Type checkers and editors see PySpark's API and its docstrings, as with `pyspark-client`. The
package is marked typed as PySpark's is (`py.typed`), and the modules whose classes are native or
whose names are made at import time, such as `pyspark.sql.connect.functions`,
`pyspark.sql.dataframe`, `pyspark.sql.types`, `pyspark.sql.connect.datasource`, the Avro and
Protobuf functions of `pyspark.sql.connect`, the messages of `pyspark.sql.connect.proto` and
`pyspark.ml.stat`, have stubs (`.pyi`) of PySpark 4.3.0's modules, with their annotations, their
overloads and the attributes that their classes assign in their methods, which
`scripts/gen_type_stubs.py` generates. The modules of `pyspark.ml.connect` of the native ML classes
have stubs of their public classes, each a subclass of PySpark's class of its name of `pyspark.ml`,
as at run time, of which type checkers see PySpark's API. So mypy and pyright type an application as
they do with `pyspark-client`, which `python/tests/test_type_stubs_offline.py` checks with both, of
each name of the stubbed modules and each member of their classes. The stubs keep the docstring that
each of PySpark's objects has at run time (`inspect.getdoc`), of what `pyspark.since` adds and of
what the modules of PySpark Connect copy of the classic ones (`col.__doc__ =
pysparkfuncs.col.__doc__`), which the test compares with those of `pyspark-client`
(`tests/golden/stub_docstrings.json`). The differences:

- The stubs declare only the names that the drop-in's modules have at run time. A name that a
  module of PySpark imports, which mypy takes for an attribute of the module by default, is
  re-exported when the drop-in's module has it too, so that the `PySparkTypeError` of
  `pyspark.sql.connect.functions.builtin`, which pyright reports as a private import with
  `pyspark-client`, as mypy with `--no-implicit-reexport` does, is not reported. One that the
  drop-in's module does not have, such as `pyspark.sql.connect.catalog.DataFrame`, is an error,
  where mypy takes it with `pyspark-client` by default, as it fails at run time. So is the
  plumbing of PySpark's modules that the drop-in does not have, such as the py4j converters of
  `pyspark.sql.types`.
- A few names that PySpark's annotations need are declared, which the drop-in does not have at run
  time: `DataFrame.rdd`, which PySpark defines when not `is_remote_only()`, so that type checkers
  of `pyspark-client` see it too, and neither client has it (`TYPE_ONLY` of the test lists them
  with their reasons). `pyspark.sql.types` has `DataTypeSingleton`, the metaclass
  of the singleton types of the stubs, at run time too, of which the native classes are not. The
  `SparkContext`, `RDD` and the other names of classic PySpark that PySpark's `pyspark` module
  imports of `pyspark.core` when not `is_remote_only()`, which `pyspark-client` does not ship,
  the drop-in's module imports so too, in a block that does not run either: mypy takes them as
  `Any` and pyright reports their imports as of an unknown symbol, as with `pyspark-client`.
- The classes that the drop-in implements in another way: the internal expressions and plans of
  PySpark Connect, which annotate private attributes (`WindowSpec._orderSpec`) and a few internal
  constructors, and which are `Any`.
- The classes of `pyspark.ml.connect` are those of the drop-in, of PySpark's API of `pyspark.ml`
  (see "Native ML classes of the full pyspark.ml API"), where `pyspark-client`'s modules of these
  names are its ML API on PyTorch, so a type checker types `LogisticRegression` of
  `pyspark.ml.connect.classification` as a subclass of that of `pyspark.ml.classification` with
  the drop-in, as at run time.
- An editor shows the docstrings that the objects of PySpark Connect have at run time, which
  pyright does not see in PySpark's source, as its modules copy them there at run time: a hover
  over `col` of `pyspark.sql.connect.functions` or the `builder` of its `SparkSession` shows its
  documentation with the drop-in, and none with `pyspark-client`.
- The names that only the drop-in has, such as `Trigger` of
  `pyspark.sql.connect.streaming.readwriter` and `FrameBound` of `pyspark.sql.window`, are typed
  with the drop-in and are errors with `pyspark-client`, as at run time.

`pyspark.sql.functions` is PySpark's package, of its modules `builtin` and `partitioning`, of
Apache Spark as they are. As with `pyspark-client`, each of its functions is PySpark's, which
`try_remote_functions` (or `try_partitioning_remote_functions`) of `pyspark.sql.utils` wraps: on
Spark Connect, a call runs the function of `pyspark.sql.connect.functions` (or of its
`partitioning`) of the name, as that module has it at the call, so a patch of it is what the call
runs; with `PYSPARK_NO_NAMESPACE_SHARE` set, it runs the function's own body, of Spark Classic,
which fails in either client (`col` imports the `SparkContext` of `pyspark`, `sum` the module
`pyspark.sql.classic`). So its `__file__`, `__path__`, what `pkgutil` reports, the `__module__`,
`__wrapped__`, docstrings and signatures of the functions, their pickles and its names (`dir()`,
`__all__`) are `pyspark-client`'s, and type checkers read its source as with `pyspark-client`:
mypy run on the files by path finds one module of it, and pyright resolves
`from pyspark.sql.functions import partitioning` from source, where it reports `F.partitioning` as
a private import, of PySpark's `__all__`, as with `pyspark-client`. A call costs the time of the
wrapper too, as with `pyspark-client`, which is more than that of most functions of
`pyspark.sql.connect.functions`, of the Rust core.

The functions of `pyspark.sql.connect.functions` are the drop-in's, which build their columns in the
Rust core: those that `scripts/gen_fn_dispatch.py` generates of PySpark Connect's functions
(`pyspark/sql/connect/functions/_generated.py`), which run in the namespace of
`pyspark.sql.connect.functions.builtin`, so that they call its `lit`, `column` and `_to_col` as it
has them at the call, as PySpark Connect's do, and those that the module writes. As PySpark
Connect's, they are of that module, and have the docstrings of PySpark's functions (but `date_part`,
`datepart` and `uuid`, which PySpark Connect leaves without one) and their annotations;
`pyspark.sql.connect.functions.partitioning` is PySpark Connect's, as it is. The differences: the
frames of a traceback of a generated function are of `_generated.py`, where PySpark Connect's are of
`builtin.py`; `col` is not wrapped by PySpark's `_with_origin`, so it has no `__wrapped__` (the
drop-in captures the call site of a Column in the Rust core); the module has none of the classes of
PySpark Connect's expressions that its module imports (`CaseWhen`, `ColumnReference`, ...), nor
`np`, and the private helpers of the Rust core are its own. Its annotations name PySpark's classic
`Column` of `pyspark.sql.column`, as those of `pyspark-client` do, of which the Columns of a session
are instances (see below). An import of a module of messages, such as `from
pyspark.sql.connect.proto import base_pb2`, has no warning of pyright: its stub is PySpark's
(`base_pb2.pyi`), and the module, which the drop-in's protobuf runtime makes at run time, has a file
of the package as its source.

The drop-in parses the protobufs that a pickled Column or DataFrame carries, and the requests that
session hooks see, at any depth that PySpark's serializer reaches, where the parsers of protobuf and
prost stop at 100 nested messages: its Rust client enables prost's `no-recursion-limit` feature for
them. It decodes the responses of the server as PySpark's client does, whose deserializer is upb's
`FromString`: of at most 100 levels of messages below a response, counted as upb and prost count
them (a message, a map entry and a group are each a level, also a group of a field that the type
does not know, while the other fields that the type does not know are not), whatever the features of
prost. The client scans the bytes of each response for the levels of their messages, with a stack of
its own, before prost decodes them; so does `SparkSession.client` of the messages of the responses
that it parses (of `schema`, `config`, `to_table`, the results of commands and the statuses of
operations). Prost then decodes them as before, and refuses some bytes that upb reads, which a
server of Spark does not send: a field that the type knows of another wire type (which upb keeps as
a field that the type does not know), a varint that does not end and a tag of 0. Of protobuf's
pure-Python implementation, the limit is that of its `decoder.SetRecursionLimit` (100 of none),
counted as upb counts it, where its decoder counts the groups of the fields that a type does not
know from 0 in each message, and stops at the limit itself: so a response of such groups exactly 100
levels deep fails in PySpark, and one of a message at level 92 to 100 with 10 levels of such groups
below it is read in PySpark, where the drop-in reads the first and refuses the second (a server, of
proto3, sends no groups). So a column of a struct nested 31 levels deep is read, and of one nested
32 levels deep the schema, `collect()`, `toPandas()` and the observed metrics (of `collect()`,
`toPandas()`, `executionInfo` and the `Observation`'s `get`) fail, as in PySpark, without a crash at
any depth (of a thread of a small stack too, see below). A response that is deeper, or that does not
parse, raises PySpark's error of a response that gRPC's deserializer refuses, of each kind of call
(an analysis, a configuration request, the status of operations, an execution and its reattachment):
the `SparkConnectGrpcException` of the status `INTERNAL`, of the text of grpc's error of the call
(`_InactiveRpcError`, or `_MultiThreadedRendezvous` of an execution) whose details are "Exception
deserializing response!" (and whose `debug_error_string` is `None`), where the error of the Rust API
is prost's error of its limit (`[Internal error] failed to decode Protobuf message: DataType.kind:
...: recursion limit reached`, of the fields that lead past the limit): the codec of the client tags
its own failure, so a status of the server of the same text (of a server of tonic that fails to
decode a request) is the server's error. Once upb's `SetAllowOversizeProtos(True)` is called, also
before the drop-in's protobuf runtime is imported (a response past 100 levels asks upb for its
limit), a response of up to 65535 levels is read, as upb reads it, decoded on a stack that grows for
its levels, and a deeper one raises the same error, where upb's parser, which recurses on the stack,
ends PySpark's process of a schema or a literal of 100,000 levels. As of the runtime's own parses, a
call of a reference to the function taken before the runtime is installed that sets it back to False
is not seen once a response found it set. Within the limit of 65535 levels, a response whose
messages of the fields that their types have (which the decoded response keeps) nest more than 1024
levels deep raises the client's `RecursionError` ("the messages of a response may be nested at most
1024 levels deep"), which bounds the stack that the conversions, copies and drops of them take.
PySpark converts the schemas and the literals of its responses in Python, up to Python's recursion
limit: it raises its own `RecursionError` of the schema of `DataFrame.schema` (of its copy), of
`collect()` and of `toPandas()`, and of the literals of the observed metrics, from a few hundred
levels (about 650 of arrays, 370 of structs), so that these are read by the drop-in only from there
to 1024 levels and fail in both past it; it reads the schemas that `SparkSession.client`'s `schema`
and `to_table_as_iterator` return up to about 1,000 nested types (2,000 levels of arrays, 3,000 of
structs), and the messages that it keeps as they are, such as the relation of the `SqlCommandResult`
of `spark.sql`, of 10,000 levels too, where the drop-in raises its `RecursionError` past 1024
levels. The waits of the client for the server, whose futures nest those of tonic, hyper, the
deadlines and the retries (of frames of tens of KiB), and its decodes, copies and conversions of the
responses, run on stacks with room for them, of a thread of a small stack (of
`threading.stack_size`) too: a call on a thread of 64 KiB reads its response, as PySpark's does, and
one on a thread of 256 KiB reads a response at that bound. So do the types that the calls return,
and the types of as many levels that the user makes: they are made, compared, written, copied and
lowered for the requests on stacks that grow with their levels, so that their `==`, `repr()`,
`json()`, `jsonValue()`, `fromJson`, `simpleString()`, `treeString()`, `toNullable()`,
`needConversion()` and `hash()` give on a thread of 128 KiB what they give on the main thread.
`copy.deepcopy` and pickling copy and pickle the `__dict__`s of the types as PySpark's do, of the
levels of the frames of Python's `copy` and `pickle`: they raise their `RecursionError` from about
100 and 125 nested structs on the main thread and on a thread of 256 KiB, as PySpark's do. On a
thread of 128 KiB, pickling overflows from there, as PySpark's does, and `copy.deepcopy`, whose
copier of the native classes grows its stack (SPARK-62287, see below), raises its `RecursionError`
where PySpark's overflows. On
the main thread, `json()`, `jsonValue()` and `fromJson` raise the `RecursionError` of Python's
`json` module of a JSON of more than about 1,000 levels (of about 330 nested structs), and the
walks of the type objects raise theirs past Python's recursion limit of their levels, a level for
each struct and each field: `==`, `treeString()`, `hash()` and `repr()` from about 500 nested
structs, and `simpleString()` and `toNullable()` from about 1,000, of more of a higher limit
(`sys.setrecursionlimit`), as PySpark's walks take. On Python 3.11, PySpark's raise
their `RecursionError` from about 100 nested structs (`copy.deepcopy`), 125 (pickling), 145 (`==`),
165 (`hash()` and `repr()`), 250 (`json()` and `simpleString()`) and 330 (`treeString()`), and its
`repr()` overflows a thread of 256 KiB from about 150. The
`grpc-status-details-bin` of a failure is read within the same limit, as upb reads it where
PySpark's `rpc_status.from_call` does: a `google.rpc.Status` nested past it, as of 101 nested groups
of a field that the type does not know, or of 100 in a detail, which is a level of its own, raises
protobuf's `DecodeError` (`Error parsing message with type 'google.rpc.Status': Wire format was
corrupt`) in both clients, whatever its code and message, as details that are not a
`google.rpc.Status` do (see below). The server is trusted, as SECURITY.md sets out, so the limit is
not a bound on the size of the responses.

A type that holds itself, as a struct whose `fields` hold a field of it (an append to its
`fields`, or `add`), an array or a map set to be its own element or value, or a field set to the
type of the struct that holds it, has levels that no walk ends. As PySpark's methods, which walk
the levels in Python, each use that reads them raises `RecursionError`, of the bound of the walks
by Python's recursion limit: `json()`, `jsonValue()`, `simpleString()`, `repr()`, `hash()`,
`toNullable()`, `treeString()` of more levels than the limit, and the uses that convert the type,
such as `createDataFrame`, `from_json`, `udf` and the readers' `schema`. The others work as
PySpark's: `==` of the type itself, `needConversion()` of a struct, the constructors and the
assignments that keep it (`StructField("x", s)`, `ArrayType(s)`), `treeString(maxDepth)` of the
levels that it prints, and pickles and copies, which keep the cycle (of the class and the
`__dict__` of the type, as PySpark pickles it). A cast and `DataFrame.to` return their Column and
DataFrame, as PySpark's do, whose repr and requests raise the error. Of an array or a map that
holds itself, `needConversion()` raises it, and so does a `StructType` of a field of it, whose
constructor reads it, as PySpark's does. The text of the error may differ from PySpark's (see "The message of the
`RecursionError` of a type that holds itself" in `dev/parity/known-differences.md`).

The error of grpcio of a response that the client fails to decode, of the details "Exception
deserializing response!" and no `debug_error_string`, is also the one that PySpark's client sees
elsewhere: the error that a retry policy is asked of (its `details()`, and its
`debug_error_string()`, `None`), the error that the `GRPC Error received` log gives, the warning of
a release that fails in the background (`ReleaseExecute failed with exception: ...`), and the error
of a call that PySpark leaves to grpc (`cloneSession()`, the uploads of `copyFromLocalToFs` and of
the caching of artifacts). As grpcio's, the `debug_error_string()` of such an error of a stream,
`_MultiThreadedRendezvous`, waits forever for a debug string that nothing sets; a policy that the
client asks of a stream that it reads in the background, as of the events of the streaming query
listeners or of the removal of a model, and the log of such an error, run on a thread of their own
(`pyspark-apart`), so that such a wait holds only its call, as in grpcio, and not the delivery
thread of the client, which runs the Python of every other request in the background. A call that
the client makes in the background awaits the log of its error (`GRPC Error received`, and `Received
ErrorInfo`) before it goes on with it, so that the log precedes what follows, such as the warning
that the listener bus ends with, as PySpark's handler thread logs the error before it warns. The
query context loggers log grpcio's error of the call that failed as the exception that PySpark
handles as it builds its exception, of its text: `_InactiveRpcError` of a call of one response, such
as `AnalyzePlan`, and `_MultiThreadedRendezvous` of a stream, of the `debug_error_string` of grpcio
1.85, the name of the code and the details, where it was a `_MultiThreadedRendezvous` of either, of
the `debug_error_string` of an older grpcio (`UNKNOWN:Error received from peer {grpc_message:...}`).

The drop-in also sends a plan that PySpark's client cannot encode: from about 50 nested expressions,
PySpark raises upb's `DecodeError: Exceeded upb_DecodeOptions_MaxDepth` before it sends the plan,
where the drop-in sends it, and the server runs it within its own limits. The commands of
`DataFrameWriterV2` and of `MergeIntoWriter` are PySpark's, over the drop-in's message of each
Column, which holds an expression of any depth, but PySpark's command of `DataFrameWriterV2` extends
its repeated field with the messages of the partitioning columns of `partitionedBy`, and its merge
makes the messages of its command and of its actions of those of its conditions and assignments,
which protobuf copies of their parse: one of more than 100 levels, such as a chain of about 50 `+`,
raises that `DecodeError`, as PySpark's does, where of one of about 500 `+` or more PySpark's
`to_plan`, which recurses, raises Python's `RecursionError` first (see "A deep Column in the
commands of `writeTo` and `mergeInto`" in `dev/parity/known-differences.md`). The server parses a
request of up to `spark.connect.grpc.marshallerRecursionLimit` nested messages (1024), so 508 nested
`abs` of a `select` run and 509 fail, with the `SparkConnectGrpcException` of `UNKNOWN`
(`Application error processing RPC`), and its analyzer resolves a plan in up to
`spark.sql.analyzer.maxIterations` (100) iterations, so a chain of 98 `+` fails with its
`AnalysisException` ("Max iterations (100) reached for batch Resolution"). A Column of more than 745
levels raises a `RecursionError` when it is pickled, of the drop-in's own message
(`an expression may be at most 745 levels deep`). On Python 3.11, PySpark's pickler raises one from
about 200 levels, under Python's default recursion limit, so a Column of about 200 to 745 levels
pickles in the drop-in only. `copy.copy` and `copy.deepcopy` of a Column share its expression, of
which a deep copy makes again only the levels above the UDFs whose functions it copies, so they copy
a Column of any depth, as PySpark's `copy.copy` does; PySpark's `copy.deepcopy` raises a
`RecursionError` from about 165 levels.

The frame bounds of `Window.rowsBetween` and `rangeBetween` are converted when the plan is built,
as in PySpark: the window keeps them in PySpark's `WindowFrame` as they are, so a bound of a row
frame beyond an int, such as a start of `-2**40` or of `Window.unboundedFollowing`, a bound of
`True`, which protobuf's int field refuses, and an int beyond a long make a window that shows them
as they are, as does the `repr` of a window expression of it (`F.sum("x").over(w)`), of which each
request of a plan raises PySpark's error, of the start and then of the end (`VALUE_NOT_BETWEEN`,
the `TypeError` of protobuf, `Value out of range`).
The type of `cast` and `try_cast` is converted when the plan is
built, as in PySpark: a `StructField`, the base `DataType()`, an abstract type such as `AtomicType()`
and a Python subclass of them, also one nested in the type, make a Column that renders as the
`simpleString` of the type, and whose plans raise PySpark's `UNSUPPORTED_OPERATION`.

`Column.alias` and `Column.name` raise PySpark's errors when PySpark's `ColumnAlias` raises them.
At the call, they raise the metadata of several names (`ONLY_ALLOWED_FOR_SINGLE_COLUMN`) and other
keywords. When a request builds the plan, they raise:
- a name that protobuf refuses (a `TypeError`, or the `UnicodeEncodeError` or `UnicodeDecodeError`
  of UTF-8);
- metadata of one name that `json.dumps` refuses;
- metadata of no names (`CANNOT_PROVIDE_METADATA`).

So a Column that is not used raises none of these. The `__repr__` of an alias of a name that is not
a `str` (bytes too) raises the `TypeError` of PySpark's join of the names. A Column of such an
alias copies and pickles, and its copy raises the same errors, also on a Python worker of the
server. A str of lone surrogates, of a name or of another value of a Column (see below), shows
escaped in the `__repr__` of the Column (`a\ud800b`), where PySpark's shows it as it is: in upper
case (`\UD800`) in the type of a cast, which the `__repr__` upper-cases, and escaped twice in a list
literal of `Column.otherwise` or `getItem`, which shows the `repr` of its strs.

The literal of a `bytearray` is of the bytes that it holds when the Column is made, which its
`__repr__` shows as PySpark's (`Column<'bytearray(b'xy')'>`), so a change to the bytearray after
`lit` is in neither the `__repr__` nor the plan, where PySpark's literal keeps the bytearray and
shows and sends what it holds at the time. The literals of the core are immutable bytes, so its
plans, pickles and copies never share a mutable Python object.

An `Observation` or a `SparkSession.Hook`, of the class or of a subclass (which a module defines
in both clients), that PySpark pickled at protocol 0 or 1 raises a `TypeError` when the drop-in
unpickles it, as CPython's `copyreg._reconstructor` cannot create the drop-in's, a class with a
native base; protocols 2 and up, pickle's default, load in both clients, of the class or the
subclass, and PySpark loads the drop-in's at every protocol. A hook of a subclass with `__slots__`
pickles at protocols 0 and 1 too, where PySpark's raises the `TypeError` of a class that defines
`__slots__` without `__getstate__`. A `StorageLevel` pickles at every protocol, and PySpark loads
the drop-in's; one that PySpark pickled at 0 or 1 raises `TypeError: object.__new__(StorageLevel)
is not safe, use StorageLevel.__new__()` when the drop-in unpickles it, and at the protocols 2 and
up both clients load each other's.

So it is for the data types. PySpark's are plain classes, which pickle as their class and their
`__dict__`: at the protocols 2 and up the drop-in makes its type of the class, without arguments,
and sets the attributes, so a pickled `df.schema` loads with its fields, names, metadata,
collations, nullable flags and parameters, as PySpark's types of them. At protocol 0 or 1 the
drop-in raises `TypeError: object.__new__(DecimalType) is not safe, use DecimalType.__new__()`. A
type of parameters (a decimal, a string, a char, a varchar, a time, an interval of fields or a
spatial type), an array, a map, a field and a struct pickle, copy and deep-copy as PySpark's plain
class does, as `copyreg.__newobj__(cls)` and their `__dict__`, which holds their parameters (see
below), also at the protocols 0 and 1, which the Python workers of the server load at every
protocol: the pickle of a type of parameters is PySpark's own, byte for byte. The drop-in's other
types pickle, copy and deep-copy as `_parse_datatype_json_string` of their JSON. So the copies of a
struct, and of an array or a map of one, keep the metadata dicts of its fields as they are, which
JSON does not (a tuple, a key that is not a str, the order of the keys), and are equal to it, as
PySpark's pickle of the `__dict__` is, and `toNullable()` makes each field of the dict of the
field, as PySpark's `metadata=field.metadata`. A type of no parameters of an attribute that a user
set pickles and copies as its class and PySpark's `__dict__` of it, as a type of a Python subclass
does (see below), also in the types that hold it. A shallow copy (`copy.copy`) of a struct, an
array, a map or a field is of its class and the same attribute objects, as PySpark's copy of the
`__dict__`, at any depth: a struct's copy shares its `fields`, `names` and `_needConversion` lists.
`copy.deepcopy` of an object of a native class of the types itself is made by a copier of the
extension, of no Python code of each object (SPARK-62287). `pyspark.sql.types` registers it in
`copy._deepcopy_dispatch`, the dict of the copiers of `copy.deepcopy` by exact class, for the
native classes themselves only: the 13 singletons, the types of parameters, `StructField`,
`ArrayType`, `MapType` and `StructType`. No attribute of the classes is added, so their `vars()`
and `dir()` are PySpark's, and an object of a subclass, which the dispatch does not hold, is
copied of Python's protocol. The copier makes the copies that Python's `copy.deepcopy` makes of
PySpark's plain classes, of `copyreg.__newobj__` and `_reconstruct`, and the same entries of the
memo and of its list of `_keep_alive`, in the same order, also of a memo that is not a dict and of
one that holds an object already. It falls back to Python's protocol, as `copy.deepcopy` runs it
past its dispatch, of an object of a native class patched over a method of the protocol, or of a
class of its MRO patched so (`__deepcopy__`, `__reduce_ex__`, `__reduce__`, `__getstate__`,
`__setstate__`, `__new__`, `__getattribute__`, `__getattr__`, `__slotnames__`, or the `json` and
`jsonValue` that a singleton's reduce reads), of an entry of `copyreg.dispatch_table` for its
class, of a `pyspark.sql.types._parse_datatype_json_string` rebound, of which a singleton's reduce
is, or of such a hook in its own `__dict__`. It counts the frames of Python's `copy` as the levels
of its recursion, of Python's limit up to Python 3.11, which takes the levels of C calls with
Python's frames; from Python 3.12, which counts them apart, of the limit of C calls, so a type of
more levels than Python's limit copies where Python's protocol raises its `RecursionError`.
The class's own types of no parameters compare without the attributes that a user set, where
PySpark's comparison of the `__dict__`s holds them: a `VariantType()` of an attribute equals
`VariantType()`, where a type of parameters, an array, a map, a field, a struct and a type of a
Python subclass compare them, of their `__dict__`s (see below).

`createDataFrame` of a pandas DataFrame of rows under a struct converts it with PySpark's conversion
where PyArrow is, also of a struct of a Python subclass that reads its fields of its own, and zips
its columns with the fields, as that conversion does: the columns past the fields are dropped, and not
converted, and of a struct of no fields the rows of a frame of columns too, as the table of the
zip has no column and no row; of fewer columns than the fields, it raises the error of the
conversion of a column that it zips with a field, if any, and then PyArrow's `ArrowInvalid` of the
rename of its table to the names of the fields (`tried to rename a table of 1 columns but only 2
names were provided`). A frame of no columns is the table of `pa.RecordBatch.from_pandas`, of its
rows: of a RangeIndex, of no column, so `pd.DataFrame(index=range(2))` is two `Row()` under
`StructType([])`; of another index, of a column of each level of the index, which the rename takes,
so `pd.DataFrame(index=[0])` is `Row(a=0)` under `a long`, and raises `AXIS_LENGTH_MISMATCH` under
`StructType([])`, of one more column than the fields. Without PyArrow, the frame takes the row path,
which zips the columns with the fields too, but raises `AXIS_LENGTH_MISMATCH` of a frame of fewer
columns than the fields, and does not take the columns of an index (see "pandas DataFrames without
PyArrow" in `dev/parity/known-differences.md`).

PySpark's conversion of a pandas DataFrame, of a struct or without a schema, runs of the conversions
and the types that the module of the session, `pyspark.sql.connect.session`, binds when it is
imported, with the first session, as PySpark Connect's module binds them, in PySpark's order, and
its conversion of each column imports `to_arrow_type` of the converter module of `sys.modules` when
it is called, as PySpark's does. So a type name rebound in `pyspark.sql.types` or
`pyspark.sql.pandas.types` before the first session, in that module, or in the converter module,
changes the conversion as it changes PySpark's, of its values and errors, and a public name rebound
after the first session, with the converter module removed, only the conversion of each column: a
public `BooleanType` rebound to `LongType` so makes `[1, 0, 2]` under `a long` `[1, 0, 1]`, of the
bools of the column that the cast to the schema takes back to longs. A struct whose classes are
patched converts with PySpark's conversion too, which reads the patch, and `createDataFrame`, of any
data, checks its schema, of a DDL string as the server parses it, against the `AtomicType` and the
`StructType` of that module before it reads any setting, as PySpark's does, of PySpark's
`TypeError` of an `isinstance` of a name that is not a class, and of its `NOT_EXPECTED_TYPE` of a
schema that they do not take. What differs, of a patched property that PySpark's `__init__` assigns,
and of the other modules of PySpark Connect, which bind the names with the session too, is in
"pandas DataFrames under customized or patched types, or rebound type names" and "Type names that
PySpark Connect's other modules and PySpark's types read" in `dev/parity/known-differences.md`.

A `StructType` caches its `names` as it is made and as `add` adds a field, as PySpark's, so an
in-place change of its `fields` list, such as a `StructField` appended, leaves `names` and
`fieldNames()` as they were. `createDataFrame` takes local rows, and the struct values below the
top level, apart by those names, as PySpark's conversion reads `fieldNames()`, and converts their
values by position with the fields as they are: a row of another number of values than the names
raises `AXIS_LENGTH_MISMATCH`, a dict is read by them, and a field past them has no value
(`IndexError`); below the top level, a value past the fields and fewer names than fields raise
`IndexError` too. A pandas DataFrame is named with `toDF(*schema.fieldNames())`, of which the server
raises for another number of names than columns.

`createDataFrame` under a struct whose fields hold an element that is not a `StructField`, as
`s.fields.append("x")` gives, at any level (in a struct inside it changed in place, also of a
struct, an array or a map that is itself an element of `fields`, and in the `sqlType()` of a
`UserDefinedType`), is PySpark's, of any data; the width of a pandas DataFrame is zipped with the
fields as of any struct (see above). It converts the data as PySpark's conversion does,
of its errors: of local rows or a pyarrow Table, it reads the fields, of the `AttributeError` of the
first attribute that an element lacks, and takes local rows apart by the cached names first, of
`AXIS_LENGTH_MISMATCH`; of a pandas DataFrame of rows, it deduplicates the names of the fields first
(`_deduplicate_field_names`), of the `IndexError` of an element past the cached `names`; and of an
array of rows, it reads no field and checks the width of the array. It then writes the `json()` of
the schema object as PySpark's does, of no rows too, of the `jsonValue` of each element of the
fields of its structs in their order, so it raises the `AttributeError` of an element of none, and
of an element that has one sends the converted data of that JSON, with the names of the struct of a
pandas DataFrame (`toDF`). A field of such a struct writes the collations of the strings of its
arrays and maps into its metadata (`__COLLATIONS`), and its type without them, as PySpark's
`StructField.jsonValue`, which reads the attributes of the types in its order, so a type of a Python
subclass of a property of its own, such as a `dataType` that makes a new struct at each read, is
written of what each read gives.

The other uses of such a struct read its elements as PySpark's `StructType` reads them, of each
element as it is: of one that lacks the attribute that a use reads, its `AttributeError`, and of one
that has it, the use of it. `json()` and `jsonValue()` write the `jsonValue` of each element, as
above; `simpleString()` joins the `simpleString()` of each element; `treeString()` calls the
`_build_formatted_string` of each element, as PySpark's `treeString` of its `StringConcat`; a lookup
by name compares the `name` of each element, and a slice is `StructType(...)` of the elements, of
the errors of its `__init__`; `add` appends the field and its name and then reads the
`needConversion()` of each element; and `toNullable()` reads the `name`, the `dataType` and the
`metadata` of each element, in that order. An assignment of `fields` takes any elements; pickles and
copies keep them, as PySpark's pickle of the `__dict__` of the struct (a shallow copy shares the
list); and a `StructField`, an `ArrayType` and a `MapType` of such a
struct keep it as it is. The plans of a UDF, of a map or group function and of `transformWithState`
and `transformWithStateInPandas` send the type that PySpark's `pyspark_types_to_proto_types` reads
of the elements when the request builds the plan, of each its `name`, `dataType`, `nullable` and
`metadata` (of a sibling field of a type that it has no proto of, such as `AtomicType()`, its
`UNSUPPORTED_OPERATION`), and each request raises the error of that. So do a cast and
`DataFrame.to`, as PySpark's `CastExpression` and `ToSchema` keep the object and read it when the
plan is built, of the struct as it is then, and the repr of the cast shows the `simpleString()` of
the object in upper case as it is when it renders, of the error of that then, as PySpark's
`__repr__`: the call reads nothing of the object. `from_json`, a reader's schema and
`catalog.createTable` read the `json()` or the elements of the object at the call, as PySpark's. A struct whose fields hold the struct itself has levels that no walk ends, and its uses
raise `RecursionError` as PySpark's do (see "A type that holds itself" above).

The parameters of a type of parameters, a `DecimalType`, `StringType`, `CharType`, `VarcharType`,
`TimeType`, `DayTimeIntervalType`, `YearMonthIntervalType`, `GeometryType` or `GeographyType`, are
entries of its `__dict__`, as PySpark's `__init__` sets them, of any value: `t.scale = 3`, or
`t.__dict__["scale"] = 3`, makes a `DecimalType(10, 2)` a `DecimalType(10, 3)`, as its repr,
JSON, `==` and hash show, also in the struct, array or map that holds it, `vars(t)` holds them,
and `==` compares the `__dict__`s, as PySpark's. Its methods read them where PySpark's read them,
so a value that PySpark's methods fail of fails the same: `%d` of a str (`%d format: a real number
is required, not str`), and the `KeyError` of a field that an interval does not have
(`DayTimeIntervalType().startField = 9`); the plan of the type raises the error of protobuf's int
field of a parameter that it refuses, as of a bool (`Field spark.connect.DataType.Decimal.scale:
Expected an int, got a boolean.`) or an int out of its range (`Value out of range`). The
constructors take any value, as PySpark's: `DecimalType(10.0)` is `DecimalType(10,0)`, and the
`precision` of `DecimalType(np.int64(12))` is the NumPy int. A type of a Python subclass whose
`__init__` does not call that of its class has none of its parameters, and raises the
`AttributeError` of the first that a method reads, as PySpark's. A set `srid` of a spatial type
shows in its repr and simple string, and its JSON stays that of the `_crs` that its constructor
set, of which it is equal to no type of that SRID, as PySpark's. The assertions of
PySpark's constructors (`field name 1 should be a string`, `elementType string should be an
instance of ...`) are made under `python -O` too, which skips PySpark's, whose types are then made.
The parameters of an `ArrayType`, a `MapType` and a `StructField` are entries of the `__dict__` of
the type, as PySpark's `__init__` sets them, of any value, which their methods read as PySpark's
read `self.<name>`: of a property of a subclass, of a raw write of the `__dict__`, and of the
instance's own over a class attribute (`nullable = False`). The constructors assert the types and
the name as PySpark's do, and an assignment keeps any value: `ArrayType(LongType()).elementType =
"string"` keeps the str, which its repr shows (`ArrayType(string, True)`), and its
`simpleString()`, `json()` and `needConversion()` raise the `AttributeError` of the str, as
PySpark's. A plan converts them as PySpark's `pyspark_types_to_proto_types` does: a value that is
not a type raises its `UNSUPPORTED_OPERATION`, a flag that is not an int raises protobuf's
`TypeError` of its bool field (`'NoneType' object cannot be interpreted as an integer`), and a
name that is not a str protobuf's `bad argument type for built-in operation` of its string field.
A struct's `names` hold the name as it is, and its `treeString()` raises the error of its
`replace`, as PySpark's `escape_meta_characters` does (`'int' object has no attribute
'replace'`). The `fields`, `names`, `_needConversion` and `_needSerializeAnyField` of a
`StructType` are entries of its `__dict__` too, which its `__init__` and `add` assign through
`setattr`, as PySpark's, of a subclass's setters and their errors, and which its methods read as
PySpark's: its iteration, length, items, `simpleString()`, repr, `jsonValue()` and
`treeString()` of `self.fields`, of a property of a subclass too, and `fieldNames()` of
`list(self.names)`. The type objects that an `ArrayType`, a
`MapType`, a `StructField` and a `StructType` are given, by their constructors, `add` and
assignments, are kept as they are, as PySpark's keep them, so a `UserDefinedType` whose
`module()` or `sqlType()` is not implemented, such as `UserDefinedType()` itself, makes them: their
repr,
`simpleString`, `needConversion`, `==`, hash, copies, pickles, tree string, `names`, `add` and
`toNullable` are PySpark's, of the objects. A use that reads the JSON of the type raises the
`NOT_IMPLEMENTED` of its `jsonValue()` then, as PySpark's does: `json()`, the plans of a cast,
`DataFrame.to`, `from_json`, a table, a Java UDF and a UDTF, a reader's schema and the relation of
data of no rows. `toInternal` and `fromInternal` raise the errors of the type's own methods. A cast
to one is made and shows its `simpleString()`, as PySpark's Column, and raises its error at each
request. `createDataFrame` of local rows or of a pandas DataFrame of one, also of local rows of
values of such a type (its `__UDT__`), whose schema it infers first, of the errors of PySpark's
inference, such as the `CANNOT_MERGE_TYPE` of the type and the type of a value of another row and
the `CANNOT_INFER_TYPE_FOR_FIELD` of a value of no type, then converts the data, as PySpark's
converters do, of the errors of the `sqlType()` and the `serialize` of the type, and then raises the
error of its `jsonValue()`, of the `json()` of the schema.

The abstract classes of the types, `DataType`, `AtomicType`, `NumericType`, `IntegralType`,
`FractionalType`, `DatetimeType`, `AnyTimeType`, `SpatialType` and `AnsiIntervalType`, make types
of their own, as PySpark's plain classes do. `AtomicType()` has the `typeName`, `simpleString` and
`jsonValue` of its class, `atomic`, converts no values, and is equal to a type of its class and
`__dict__` only. It pickles as its class and its `__dict__`: at the protocols 2 and up its pickle
is PySpark's own, byte for byte, and PySpark loads the drop-in's at every protocol. The struct,
array or map that holds it names it in its `simpleString`, JSON and tree string, and pickles of
the types it is made of. It compares it, and names it in its `simpleString` and tree string, by the
object that it holds, as PySpark's does, rather than by its JSON: so two types of the same JSON, of
one class and different `__dict__`s or of two classes of one `typeName`, stay apart in the types
that hold them, as `UserDefinedType`s of other parameters do, and `toNullable()` keeps them as they
are. The schema of a DataFrame and of its rows, as the server gives it, holds a new object of each
user-defined type, as PySpark's `UserDefinedType.fromJson` makes it of its class, and so does
`fromJson`: not the object that the client was given, nor of its attributes, and a class whose
`__init__` takes arguments raises its `TypeError` wherever PySpark converts that schema, as in
PySpark: the schema, `columns`, `dtypes`, `printSchema`, the rows, the lookups that read the
columns (`df[0]`, `dir(df)`, names of two underscores), the copies and the unpickles, the map
functions and the numeric aggregations of grouped data. The cached schema of a DataFrame holds the
objects that PySpark's `_cached_schema` holds, those of `to` or of a map function, else the new
objects of the schema of the server: a set operation compares them as PySpark's does, by the
`__dict__` of their user-defined types, and a pickle carries them, which the unpickled DataFrame
keeps as its cached schema. `DataType.simpleString(t)`, the method of the base class, is
`t.typeName()`, as PySpark's, which the types of a string of their own, such as `bigint` or
`struct<a:bigint>`, override. `IntegralType()` is one instance, as PySpark's
`DataTypeSingleton` makes it, and so is each Python subclass of it, of an instance of its own, and a
Python subclass of the other classes, such as a `UserDefinedType` of parameters, takes the arguments
of its `__init__`.

Neither client has a proto or an Arrow type of such a type, also of one nested in another or in the
`sqlType()` of a user-defined type. A plan of one raises the `UNSUPPORTED_OPERATION` of PySpark's
`pyspark_types_to_proto_types`, as PySpark's plans do: a cast, `DataFrame.to`, a UDF that is not
Arrow-optimized, a UDTF, and `transformWithState` and `transformWithStateInPandas` of an output type
of one at their first request, and the registration of a UDTF or of a Java UDF,
`Catalog.createTable` and the schema of a table or a materialized view of a pipeline at the call or
the decorator. As the schema of `createDataFrame`, local rows of one raise
`UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION` once their values are converted, as PySpark's
`to_arrow_schema` raises it after its converters run, so that the errors of the values, and of the
`serialize` of a user-defined type, come first; and a pandas DataFrame of one as its only type
raises `UNSUPPORTED_DATA_TYPE_FOR_ARROW`. Data of no rows and a NumPy array make a relation of the
JSON of the schema, PySpark's `json()`, in which such a type is its own JSON, also nested; the
server fails to parse it (`PARSE_SYNTAX_ERROR`), and so does a pyarrow Table of one as its only
type. A reader's `csv`, `json` or `xml` of a DataFrame reads a schema of JSON as PySpark's `Parse`
plan does when it is built, with `StructType.fromJson`, so each request of the DataFrame raises the
`CANNOT_PARSE_DATATYPE` of the name of a type of its class only, and the `UNSUPPORTED_DATA_TYPE` of
a `StructField` as a type. `DataType()` and a Python subclass of it that is not an `AtomicType` are
not expected as a schema, as PySpark's `isinstance` reads them: their `NOT_EXPECTED_TYPE` names
their class.

PySpark's singleton types, from `NullType` and `IntegerType` to `CalendarIntervalType` and
`IntegralType`, are classes of the metaclass `DataTypeSingleton`, which `type(IntegerType)` shows.
The drop-in's are native classes, whose metaclass is `type`: PyO3 has no option of a metaclass of
a class, and CPython refuses to assign the `__class__` of a class of `type` after it is made. So an
error that names the type of such a class, as the `NOT_EXPECTED_TYPE` of `udf(f, IntegerType)` (of
the class, not of an instance), names `type` where PySpark's names `DataTypeSingleton`.
`pyspark.sql.types` has PySpark's `DataTypeSingleton`, of which a class is made as PySpark's makes
it, such as a subclass of a singleton type of `__main__`, which PySpark's cloudpickle sends by
value of its metaclass: it loads, of one instance, as in PySpark. The constructors of the native
classes return the one instance and raise the `TypeError` of `DataTypeSingleton.__call__` for an
argument, as PySpark's, and `inspect.signature` of the class shows no parameters. A copy, a deep
copy and a pickle of the one instance are the one instance too, where PySpark's `copy.copy`,
`copy.deepcopy` and `pickle` make another object of the class, as the note of its
`DataTypeSingleton` says. A subclass of such a class of another metaclass, `class X(IntegerType,
metaclass=Meta)`, is made, where PySpark's raises the `TypeError` of a metaclass conflict.

PySpark's concrete types are plain classes too, which a user subclasses, and so are the drop-in's,
from `NullType` and `StringType` to `ArrayType`, `StructField` and `StructType`. A type of a Python
subclass is of the core type of its class, which the plans and the conversions read, as PySpark's
read it by `isinstance`, and the methods of its class are PySpark's, of the methods of the object:
- `typeName()` is the name of the subclass (`subinteger` of `class SubIntegerType(IntegerType)`),
  of which `simpleString()`, `jsonValue()` and `json()` are, where the class has none of its own,
  and the `type` of the JSON of an array, a map and a struct; the repr of a class whose repr names
  its class names the subclass (`SubBooleanType()`, `SubDayTimeIntervalType(0, 3)`), and so does
  its hash, of the repr;
- where a method of PySpark's calls another method of the object, or of a type that an array, a
  map or a struct holds, the drop-in's calls it too, so an override of a subclass is of their JSON,
  simple string, repr, tree string and conversions, also on the Python workers of the server, to
  which a UDF pickles its return type: a UDF that is not Arrow-optimized converts its results with
  the `toInternal` of the subclass, as PySpark's;
- `json()` and `jsonValue()` read the attributes that PySpark's `jsonValue` reads as Python reads
  them: of a struct, its fields as it iterates itself (`iter(self.fields)`, or an `__iter__` of
  its own), of a field its `name`, `dataType`, `nullable` and `metadata`, and of an array and a
  map their types and flags, so a property of a subclass that overrides one is of the JSON, at any
  depth, and so is a patch of a native class (of `StringType.isUTF8BinaryCollation` patched to
  return False, a string is `"string collate UTF8_BINARY"`, of the `__COLLATIONS` metadata of its
  field); `createDataFrame` of a pandas DataFrame, a pyarrow Table, local data or data of no rows
  sends that `json()` as the schema of its relation, as PySpark's, and converts the data with
  PySpark's conversion, which reads the attributes alike (see `dev/parity/known-differences.md`);
- the conversion of the local rows of `createDataFrame` takes the type objects of the schema, as
  PySpark's converters hold them: its errors name the object (`input for SubBooleanType() must not
  be None`), of its repr, and a geometry or a geography converts with the `toInternal` of a
  subclass, whose result is converted as PyArrow converts it;
- a type of a subclass is equal only to a type of its class or of a subclass of it, of the same
  parameters and of the same attributes of its `__dict__`, as PySpark's
  `isinstance(other, self.__class__)` and `__dict__`: `StringType() == SubStringType()` is false in
  both orders, as `==` calls the `__eq__` of the subclass first;
- its `__init__` takes arguments of its own, which the `__new__` of the class takes as
  `object.__new__` does, and passes the parameters to the `__init__` of the class, which sets them,
  as PySpark's, also of `StringType`, `DecimalType`, `TimeType`, the intervals and `StructType`,
  whose errors of arguments are those of `__init__`
  (`DecimalType.__init__() got an unexpected keyword argument 'foo'`), but for the counts of too
  many positional arguments, which are PyO3's, without `self`, as for the other methods (see
  below): `DecimalType.__init__() takes from 0 to 2 positional arguments but 3 were given`, where
  PySpark's says "from 1 to 3" and "4";
- it pickles, copies and deep-copies, at the protocols 2 and up, as `copyreg.__newobj__(cls)` and
  PySpark's `__dict__` of it, of its parameters (and of the `_crs` and `_alg` of a spatial type) and
  of the attributes that a user set, and of its `__slots__`, so each client loads the other's, and
  cloudpickle sends a class of `__main__` by value: the pickle of a type of no types in it, such as
  a `DecimalType` of a subclass, is PySpark's own, byte for byte, and the types that a type holds
  pickle as the class's own do (see above);
- a subclass of a singleton type has an instance of its own, as PySpark's `DataTypeSingleton` makes
  it, and takes no arguments;
- the Column of a `cast` or `try_cast` to it renders its `simpleString()`, as PySpark's
  (`CAST(id AS SUB)`), and so does one to a user-defined type of its own `simpleString()`
  (`CAST(id AS SCALE2)`), also in an array, a map or a struct, while the plan is of the core type.

A subclass of `__slots__` of no `__getstate__` of its own raises `TypeError: a class that defines
__slots__ without defining __getstate__ cannot be pickled` at the protocols 0 and 1, as PySpark's,
and pickles of its slots at the protocols 2 and up, as PySpark's.

What differs:
- Python calls the `__init__` of a subclass of a singleton type at each call of the class, which
  returns the one instance, where PySpark's metaclass calls it on the first call, as the
  drop-in's `DataTypeSingleton` does of a class of it, such as of PySpark's cloudpickle of such a
  subclass of `__main__` (see above). Copies and pickles of the instance are the instance, as for
  the class's own, where PySpark's `copyreg.__newobj__` makes another object, and a subclass of
  another metaclass is made, where PySpark's raises its `TypeError: metaclass conflict`. This is
  inherent: PySpark's `DataTypeSingleton.__call__` returns the one instance without calling
  `__init__`, and a native class is of the metaclass `type`, whose `__call__` calls the `__init__`
  each time the `__new__` gives an instance of the class: the extension makes its classes with
  `PyType_FromSpec`, of the stable ABI of Python 3.9, which has no `PyType_FromMetaclass` to make a
  class of another metaclass (see `dev/parity/known-differences.md`).
- A class of type classes of two branches of their hierarchy, such as
  `class X(StringType, IntegerType)` or `class Y(ArrayType, MapType)`, raises
  `TypeError: multiple bases have instance lay-out conflict` when it is made, where PySpark makes
  the class (`X()` is `StringType()`). This is inherent: an object of such a class would be of the
  native layout of one branch, which the native methods of the other read as theirs, as CPython
  raises of bases of conflicting layouts.
- `createDataFrame` of data sends the JSON of the core types of its schema, where PySpark sends the
  `json()` of the schema object, which names a type of a subclass by its `typeName()`
  (`"subinteger"`) and which the server then fails to parse (`PARSE_SYNTAX_ERROR`), so sending it
  would only bring back that failure; of data of no rows it sends the `json()` of the object, as
  PySpark, and so it does of any data under a struct that holds a type that overrides a read of
  `jsonValue` (see above), or a struct of a `names` of its own, which PySpark's conversion reads,
  whose data PySpark's conversion converts, as PySpark's. The readers,
  `from_json`, `from_xml`, `applyInPandasWithState` and `toDDL` send the `json()` of the object,
  as PySpark's.

The metadata of a field is the value that it was given, as PySpark's `metadata or {}` keeps it: a
dict, or another value, such as a list or a str (see below). `==` compares it as Python compares
dicts, as PySpark compares the `__dict__`s of its fields, also in a struct and in the types that hold it:
`(1, 2)` is not `[1, 2]`, nor `{1: "a"}` `{"1": "a"}`, nor `10**30` `10**30 + 1`. Its `jsonValue()`
holds that dict, which `json()` writes with `json.dumps`, as PySpark's: an int of more than 64 bits
as it is, a float as Python's `repr` writes it, of the shortest digits that read back as it and of
an even last digit of a tie (`1e-07`, and `74334700308107.62` of `74334700308107.625`), `NaN` and
`Infinity`, and the keys sorted as `sort_keys` sorts them, of the ints and floats by their values
(`{10: "a", 9: "b"}` is `{"9":"b","10":"a"}`). The core holds the metadata of the JSON of
`json.dumps` of the dict, of its values as they are written, so a plan sends that text, as PySpark's
`pyspark_types_to_proto_types` sends the metadata of a field, and so do the readers and
`createDataFrame` in the JSON of the schema that they send, of any depth, also of the metadata of
the fields of the `sqlType()` of a `UserDefinedType`, as PySpark's: of a schema that was given, or
of one inferred of local rows, of the `__UDT__` objects of their values, of their own `jsonValue()`.
The server keeps the low 64 bits of such an int (`10**30` becomes `5076944270305263616`), and
refuses a float that is not finite. The Arrow schema of `createDataFrame` of a pyarrow Table holds
`json.dumps` of the metadata of each field, as PySpark's `to_arrow_schema` writes it; that of local
rows, of a pandas DataFrame, of each of its conversions of a column, and of a numpy array holds it
on the fields of the `sqlType()` of a `UserDefinedType` only, as PySpark's
`_deduplicate_field_names` drops the metadata of the others, which names the fields of each struct
of the schema (`a_0`, `a_1`), of the top level too for a pandas DataFrame, but not those of the
`sqlType()`. A dict that `json.dumps` refuses, such as one of a set or of a key of a tuple, is kept,
and only the uses that write its JSON raise the error of `json.dumps`, as PySpark's: `json()`, of
its sorted keys too (`'<' not supported between instances of 'str' and 'int'`), each request of a
plan of the type, such as of a cast, also of one made in the `sqlType()` that `jsonValue()` calls,
and `createDataFrame`, after it converts the data, of the error of its Arrow schema first where that
holds the metadata (`keys must be str, int, float, bool or None, not tuple`). `mapInPandas`,
`mapInArrow`, `applyInPandas` and `applyInArrow`, after they read the columns of the DataFrame, and
those of a cogroup raise PySpark's `NOT_IMPLEMENTED` of the return type of such a schema, of their
UDF's check of its Arrow type. The JSON of a type writes each character of its names and strings
that is not printable ASCII as the `\u` escape of `json.dumps`, as PySpark's `json()`.

Metadata that is not a dict, such as `[1, 2]`, `"abc"` or `5`, is kept as it is given, as PySpark's
field keeps it, by the constructor (of false metadata, such as `[]` or `0`, a new dict), an
assignment and `StructType.add`, and so do its `jsonValue()`, `json()`, `toNullable()`, copies,
pickles and `fromJson`. A plan sends `json.dumps` of it when it has a length, as PySpark's
`len(field.metadata) > 0`, which the server refuses (`ClassCastException`: a `JArray` cannot be cast
to a `JObject`), and each request of a plan of metadata of no length, such as an int, raises the
`TypeError` of the length (`object of type 'int' has no len()`); `createDataFrame` and `from_json`
send it in the JSON of the schema, which the server fails to parse. Its `jsonValue()`
of a field of collated strings raises the `TypeError` of PySpark's `{**self.metadata}`
(`'list' object is not a mapping`).

A `TableArg`, of `DataFrame.asTable()`, pickles at every protocol, plainly and by cloudpickle, as
PySpark's, of its user-set attributes, and `copy.copy` and `copy.deepcopy` copy it as PySpark's. Its
pickle keeps the relation of the plan of its DataFrame, as a pickled DataFrame does (see "What a
plan build does for a pickle" in `dev/parity/known-differences.md`), of the functions of its UDFs,
and its partitioning, ordering and single partition; the unpickle sends no request, as PySpark's,
and the table argument that it gives sends the same plan and subquery, of the same plan ID, whose
functions its requests pickle. A plain pickle of one of a function that does not pickle by
reference, such as a lambda, fails at it, as PySpark's. A plan that does
not build pickles too, and the requests of the unpickled table argument raise its error, as
PySpark's. The pickle names the session of its DataFrame by its ID, so it unpickles only where the
session of that ID is active, as a pickled DataFrame does: it raises `NO_ACTIVE_SESSION` where none
is, such as on another thread, and `PySparkAssertionError` where another one is, where PySpark's,
whose plan holds no session, unpickles anywhere. On a
Python worker of the server it unpickles as PySpark Connect's table argument of the relation, which
needs no session. Its `_subquery_expr` is the entry of its `__dict__`, as PySpark's, an expression
of `pyspark.sql.connect.expressions`, as a Column's `_expr` is, of PySpark's `to_plan` and `repr`,
but not the `SubqueryExpression` of PySpark, whose fields of the subquery it does not have (see the
`Column`, the `WindowSpec` and the `TableArg` below). `_is_partitioned()` is PySpark's.

The `__repr__` of a Column of a subquery names its plan by its plan ID,
`SubqueryExpression(plan=<plan 2>, type=scalar)`, where PySpark's shows the repr of its plan
object, `<pyspark.sql.connect.plan.Project object at 0x...>`.

`DataFrame.inputFiles()` returns a `list` of its paths, where PySpark returns protobuf's repeated
field of the response, a `RepeatedScalarContainer`, which compares equal to the list.

The keys of `getMessageParameters()` of a server error are in the server's order. PySpark's
follow the protobuf map of the error details that it fetches: with protobuf's pure-Python
implementation that is the server's order too, and with upb, protobuf's default, the order of its
hash table, which changes from one process to the next. For its own errors, PySpark keeps the
order of each call site.

A signal ends the wait of a call for the server as it ends PySpark's. The `KeyboardInterrupt` of a
handler, as of Ctrl-C, interrupts the execution in flight on the server, as the `except
KeyboardInterrupt` of PySpark's execution does, and a second signal ends the wait without waiting
for the interrupt; the exception of another handler ends the wait at once and is raised, without an
interrupt. A signal while the failure of a call is converted, as during the request of the details
of an error of the server or the read of its stack-trace settings, ends the wait at once, of no
interrupt, and its exception has grpcio's error of the failure as its context, and the operation ID
of the execution of the failure (`_operation_id`), as PySpark's `_handle_error` converts the failure
in the `except Exception` clause of the call; a call that a handler of a log of the conversion makes
meanwhile waits as any other, and a `KeyboardInterrupt` interrupts its execution, if it has one,
never the converted one. The lookup of the plan compression settings of the first plan of a session,
and the read of the stack-trace settings of an error, catch the exception of a handler that is an
`Exception`, as the `except Exception` of PySpark's do: the lookup is then of no settings, as a
failed one, and the call goes on and waits again.

At interpreter exit, a session that is not stopped has its ML cache cleaned by one
`clean_cache` request of the `_cleanup_ml_cache` of its client, as PySpark's `_on_exit` cleans it,
so a program that replaces that method, as PySpark's tests do for a client of no server, replaces
the cleanup. Its errors are ignored, but when the server
is gone the exit waits on that request's retries, as PySpark's does, so the exit can take as
long as those retries. A signal ends that wait, as it ends PySpark's, whose cleanup catches
`Exception` only: the `KeyboardInterrupt` of Ctrl-C is raised out of the exit hook, which
`atexit` reports before it runs the next one, and the session is then neither released nor
closed. The exception of a handler that is an `Exception` ends the request that waits, which the
exit hook goes on from, unless it arrives while the session looks the plan compression settings
up for the `clean_cache`, as for its first plan: the lookup catches it, as PySpark's, and the
`clean_cache` is sent, which waits again. A stopped session sends nothing at exit. PySpark 4.3.0 does not exit at all while a
session other than its default one is left open to the end, such as one of `newSession()` or of
`SparkSession(connection=...)`: its `SparkSession.__del__` closes the client in the teardown of the
interpreter, whose grpcio `Channel.close()` waits forever, a bug of PySpark that the drop-in does
not reproduce: it exits.
Nor does a session that the program dropped send anything, of which no object is left but a `conf`
or a `client` of it: neither the session nor a DataFrame, reader, writer, catalog, stream or ML
object of it. PySpark's session object closes its client when it is freed, and the exit sends
nothing for a closed client. The drop-in drops such a session, closes its client, and closes the
connection of the process to its server once no session uses it, which the sessions of
`newSession()` and `cloneSession()` share, as PySpark's channels share their subchannels, but a
session of a token, whose connection is its own; so does the `stop()` of the last session of the
connection, as grpcio's `close` of the channel of its client. Once its `catalog` or its `streams` is
read, the session object of either client is in a reference cycle with its catalog, the `catalog` of
its `__dict__`, which holds it as its `_sparkSession`, or with its `StreamingQueryManager`, its
`_sqm`, which holds it as its `_session`, and which only the garbage collector frees, so that until
a collection its exit still cleans its ML cache, as PySpark's.
A client that no session object owns, of `SparkConnectClient(...)` or of its `newSession()` or
`clone()`, lives until the exit, as PySpark's `atexit` registration keeps it, so its ML cache
is cleaned at exit unless it is closed, even when the program dropped it.
`SPARK_CONNECT_RELEASE_SESSION_ON_EXIT` is read when a session is made, as PySpark reads it:
when it is `true` or `1`, the session is then also released and closed at exit, after the
cleanup.

PySpark's client sends the `ReleaseExecute` requests of an execution through a
`ThreadPoolExecutor`, which refuses new work once the interpreter has begun to shut down. The
drop-in refuses them from the same point. An execution after it, such as a `collect()` in an
`atexit` handler, raises `RuntimeError: cannot schedule new futures after interpreter shutdown`
after its first response and sends no `ReleaseExecute`, as in PySpark. The exit's `clean_cache`
ends the same way, and its error is ignored. Requests that are not executions, such as the
`AnalyzePlan` of a schema, still work then.

Once the exit has run the `atexit` hook of the drop-in's extension, after the threads that are
not daemons have ended, a call that fails on another thread, a daemon thread, as the calls of a
session that an earlier hook closed fail, raises nothing: the thread waits until the process
ends, as CPython 3.14 holds such a thread once the interpreter finalizes, and the hook waits for
the exceptions that such threads are building, for up to 5 seconds. Before 3.14, the finalization
ends a thread that waits for the GIL with `pthread_exit`, which aborts the process when the thread
runs the Python code of an exception, such as its first import of the exception classes, below
the frames of the extension. PySpark's daemon thread raises its exception, until the finalization
ends it.

The `TypeError` for too many positional arguments counts them without `self`: where CPython says
"takes from 2 to 4 positional arguments but 5 were given", the drop-in says "takes from 1 to 3
positional arguments but 4 were given". A method that takes no arguments gets "takes no arguments
(1 given)", where CPython says "takes 1 positional argument but 2 were given" for
`DataFrame.distinct(1)`. A method that takes one argument gets the plural "positional arguments"
even for a count of one: the drop-in says "takes 1 positional arguments but 3 were given" for
`DataFrame.limit(1, 2, 3)`, where CPython says "takes 2 positional arguments but 4 were given".
These are the messages of PyO3 0.29.3, whose `too_many_positional_arguments`
(`src/impl_/extract_argument.rs`) counts the parameters of the Rust function, of no `self`, and
writes "positional arguments" of any count: matching CPython's would need a patched PyO3.

A warning that the extension raises, such as the `FutureWarning` of `registerTempTable`,
`Catalog.registerFunction`, `createExternalTable`, a `dbName` of the catalog, and `getItem` and
`getField` of a Column, and the `UserWarning` of `replace` of a dict and a value, of
`GroupedData.apply`, of type hints that `applyInPandas` and `applyInArrow` cannot read, of the
arguments that `createDataFrame` ignores, of the configurations of the server, and of the listener
bus of the streams, is attributed as PySpark's is: to the module of `pyspark.sql.connect` of
PySpark's `warnings.warn`, of the registry of the warnings that the module showed, and to the line
of that call, in the file of the drop-in's module of that name. So
`warnings.filterwarnings(..., module="pyspark")` and the filters of a line match it, and the
default filter shows it once, as PySpark's. The drop-in's module ends before that line, so the
warning shows no source line, where PySpark's shows its call.
The warning of a window of no partitions, `WARN WindowExpression: No Partition Defined for Window
operation! ...`, is PySpark's: a `UserWarning` of `pyspark.sql.connect.expressions`, of its line
1126, which the default filter shows once, at each request that builds the plan of such a window,
as PySpark warns as it builds the plan. A pickle of a DataFrame or a Column does not warn, as
PySpark's builds no plan; but PySpark's unpickle of a DataFrame warns once, as it builds the plan to
read the columns of the new DataFrame, where the drop-in's does not; and a DataFrame, a Column or a
table argument that is unpickled sends the plan that was built for the pickle, of the functions of
its UDFs pickled at each request (see the UDFs below), so its requests do not warn again, where
PySpark's warn at each. Such a plan keeps the origin of the `and` of the conditions of a join of a
list of Columns of that time too, where PySpark gives it the call site of each request that builds
it, as it does in the drop-in otherwise. A copy of a DataFrame, in its session or another, builds
its plan at each request, and warns then, as PySpark's.
A request that PySpark builds in `functools`, as of `isStreaming`, a `functools.cached_property`,
gives that `and` the call site of `functools.py`, where the drop-in gives the user's.

The `FutureWarning` of `BisectingKMeansModel.computeCost` is attributed as PySpark's is, to the
line of its `warnings.warn` in `pyspark/ml/clustering.py` and to the module
`pyspark.ml.clustering`, of its registry of shown warnings, so the filters of that module match it.

The `TypeError` for a bad keyword argument of a method that takes no `**options` is PyO3's, which
CPython words differently in two cases. For a key that is not a str, as `df.limit(1, **{1: 2})`,
CPython says "keywords must be strings", where the drop-in says "DataFrame.limit() got an
unexpected keyword argument '1'". For a method that takes no arguments, as `df.count(foo=1)`,
CPython says "DataFrame.count() got an unexpected keyword argument 'foo'", where the drop-in says
"DataFrame.count() takes no keyword arguments". The methods that take `**options`, and the
readers', writers' and `SparkSession.builder.config`'s that take named options, word both as
CPython does.

A str that UTF-8 cannot encode, such as one with a lone surrogate, fails where PySpark builds the
request that carries it, which is not always the call that takes it. The error is PySpark's
`UnicodeEncodeError` with the same message, and no request is sent:
- a reader's format, schema, option key or value, path or table name fails the DataFrame when its
  plan is built, at its first request, such as `df.schema`;
- a `DataFrameWriter`'s or a `DataFrameWriterV2`'s format or provider, option, table property,
  partitioning, bucketing, sorting or clustering column, path or table name fails the write, such
  as `save`, `saveAsTable` or `create`;
- a `withColumns` name, or an as-of join's `how` or `direction`, fails the DataFrame when its plan
  is built, at its first request, such as `collect` or `columns`. A `direction` of bytes that are
  not UTF-8 fails there with PySpark's `UnicodeDecodeError`, as protobuf's string field gives it;
- the names of `createDataFrame` of a pandas DataFrame, of its column labels or of a list, and of
  a pyarrow Table, which PySpark applies with `toDF`, fail the DataFrame when its plan is built,
  at its first request. Those of other data, local rows or a NumPy array, which PySpark's
  conversion names its table by, fail the call, as PySpark's, which reads the settings of its
  conversions first, where the drop-in sends no request;
- the names, SQL strings and literal strs of Columns fail each DataFrame whose plan has the Column
  when its plan is built, while the Column is made, renders, copies and pickles: those of `F.col`,
  `F.expr`, `F.lit` and the other functions, `df[name]`, `df.name`, `colRegex`, `metadataColumn`,
  the field of `getField`, `withField` and `dropFields`, the function of `call_function` and
  `call_udf`, the type of `cast` and `try_cast`, and the column names and keywords of a UDF or UDTF
  call;
- so do the arguments of the methods of a DataFrame, of its `na` and `stat`, and of grouped data
  that their plans set as they are given, such as the alias of `alias`, the names of
  `withColumn(s)Renamed`, `withMetadata`, `toDF`, `drop`, `dropDuplicates`, `withWatermark`,
  `unpivot`, `describe`, `summary`, `crosstab`, `freqItems` and the NA functions, a hint, the
  name of `observe`, the names of a using join, the name of `zipWithIndex`, the functions of `agg`
  of a dict, the output mode and the timeout of `applyInPandasWithState`, and the output mode, the
  time mode and the event-time column of `transformWithStateInPandas` and `transformWithState`,
  and the name of an `Observation`, which fails the DataFrames that it observes;
- the schema of `from_avro` and `to_avro`, and the message name and the descriptor set of
  `from_protobuf` and `to_protobuf`, are literals, as PySpark's `lit` of them, which PySpark does
  not check otherwise: such a str fails each DataFrame of the Column, and a message name of
  another type, such as 5, bytes or None, is the server's error of the function;
- a tag of `addTag` fails each execution that the thread starts, as its request is built after its
  plan, until `removeTag` removes it; a request without tags, such as that of `df.schema`, is sent.
  `getTags` gives the tag as it is;
- a key or a value of `SparkSession.builder.config`, or an `appName`, fails `getOrCreate`;
- a `DataStreamWriter`'s fails the call that takes it, as PySpark sets that writer's request as it
  goes.

Such a str is not one of the modes of `explain`, of the `how` of `dropna`, of the join types of
`join` and `lateralJoin`, of the modes, directions and join types of `nearestByJoin`, nor a column
of `sum`, `avg`, `min` and `max` of grouped data: they raise PySpark's `VALUE_NOT_ALLOWED`,
`UNSUPPORTED_JOIN_TYPE`, the `NEAREST_BY_JOIN` errors and `NOT_NUMERIC_COLUMNS`, of the str as it
is. So is the `UNSUPPORTED_OPERATION` of `to` and of a cast of such a str, or of another object
whose str has a lone surrogate. `nearestByJoin` reads the `lower()` of its mode, direction and join
type, as PySpark's, so an object without one raises its `AttributeError`, bytes as a join type the
`TypeError` of `replace`, and bytes as a mode or a direction its error, of the bytes as they are.
Such a str is no name of `readStream.name`, which raises `INVALID_STREAMING_SOURCE_NAME` of it, as
PySpark's `re.match(r"^[a-zA-Z0-9_]+$", name)` does not match it; that pattern matches a name of
a final newline, which the server refuses. A Column's attribute of such a name is the field of that
name, as `col[name]`, so `hasattr` of it is True, and a session's raises the `AttributeError` of
`object.__getattribute__`, so `hasattr` of it is False, as in PySpark. `explain` builds the plan
first, so the error of a plan that fails to build comes first.

The commands that PySpark sends when they are called set their names as protobuf converts them,
at the call: `registerJavaFunction` and `registerJavaUDAF` (the name, then the class), the
`register` of `spark.udf` of a UDF and of `spark.udtf`, the `name()` of the class of
`spark.dataSource.register`, `interruptTag` and `interruptOperation` take bytes of UTF-8, and
raise protobuf's `TypeError` "bad argument type for built-in operation" of another value;
`interruptTag(None)` and `interruptOperation(None)` raise the `AssertionError` of PySpark's
request. The `register` of a plain function names the function of its UDF first, so a name that
is not a str raises the `TypeError` of `__name__`, as PySpark's. A function registered of the name
None is registered as `fun_` and a random hex, as PySpark's client does, and `register` returns
its UDF. `SparkSession.builder.master` of Spark Connect takes any value and does nothing, as
PySpark Connect's.

So it is for the other values that a plan sets as they are given, which protobuf converts when it
builds the plan: a string field takes bytes of UTF-8, such as `alias(b"q")`, and refuses another
object with the `TypeError` "bad argument type for built-in operation", such as the alias 5 or
None, the names of `withColumnRenamed(1, "z")`, `unpivot`, `describe(b"ab")` (of its ints),
`freqItems(["a", 1])`, the `subset` of `na.replace`, and a using join on `["a", 1]`; an int field
refuses an int beyond it with a `ValueError`, such as `fillna(2**70)`, `dropna(thresh=2**40)` and
the arguments of `spark.range` and `spark.tvf.range` after their `int()`, and a bool with a
`TypeError`, such as `dropna(thresh=True)`; a double field refuses a str, such as the `support` of
`freqItems`. A join on a tuple, or on a list of conditions whose `&` fails, such as of a Column and
an object that is neither a Column nor a literal, raises the `AttributeError` of `to_plan` or of
`&`, which PySpark computes when it builds the plan, such as the `PySparkAttributeError` of the
`_expr` of a Row; an exception of `&` of another class, such as a `KeyError` of an object's
`__getattr__`, is raised by the join. Of several such values, the requests raise the
error of the first in the order in which PySpark builds the plan: the input of a plan before the
plan, but for the partitioning of a repartition, which comes first, and its count of partitions,
which comes after its input, and the name of a hint or an observation before its expressions; the
order of the other values of one plan can differ. As in PySpark, a DataFrame of such a plan raises
at its requests, `schema` and `collect()` among them, and the call raises none. Of a join on a
tuple, or on a list of conditions whose `&` fails, the properties read the Column of `_schema`, as
PySpark's: their `self._schema` raises an `AttributeError`, after which Python calls the
`__getattr__` of the DataFrame, of that name. So `schema` gives a deep copy of the Column,
`columns`, `dtypes`, the representation, `dir()`, the key completions and `toArrow` raise the
`NOT_ITERABLE` of iterating it, and `printSchema` the `TypeError` of a call of a Column. The
drop-in's DataFrame has no `_schema` itself, so `df._schema` is the Column of that name, where
PySpark's is the schema.

A DataFrame whose plan fails to build pickles, also by cloudpickle, and its unpickle raises the
error, as PySpark's: its pickle keeps the objects of the plan, and its unpickle builds the plan to
read the columns of the new DataFrame. Of an `AttributeError`, such as of a join on a tuple or on a
list of conditions whose `&` fails, PySpark's read of the columns falls back from its `_schema` to
the Column of `__getattr__`, whose `fields` is a Column, which is not iterable: so the unpickle, a
`copy.copy` and a `copy.deepcopy`, which read the columns too, raise `NOT_ITERABLE`, in both.
`alias`, a hint, `dropDuplicates`, `observe`, `filter`, `sort`, `sortWithinPartitions` and the
repartitions of such a value keep the cached schema of the DataFrame that they are of, as PySpark
copies it, so their `schema` is known without a request, and their other requests raise the error.

The `barrier` of `mapInPandas` and `mapInArrow` is a protobuf bool, which their plan sets once it
pickled the function, when it is built: an int is taken, as `barrier=1`, and another value, such
as `"x"`, None or a float, fails each request with the `TypeError` of protobuf, and not the call.
The DataFrame keeps its schema, of the schema that it is given, as PySpark's `_map_partitions`
caches it: a `StructType` as the object that it is, so `schema` gives a copy of it as it is then,
and `columns` and `dtypes` its fields then, as do those of `applyInPandas`, `applyInArrow` and the
cogroup's; the type of a DDL string is parsed at the call. The plan reads `profile.id` of its
resource profile after that, so the profile is created on the server by the first request that
builds the plan, such as `collect`, and not by the call nor by `schema`, as in PySpark; a profile
without an `id`, such as a str, fails the requests with that `AttributeError`, and the id of a
request whose flag fails is not read.

The merge writer and its clauses are PySpark's (see below), which keep the condition and the
conditions of the clauses as they are given: `mergeInto(table, 5)` and `whenMatched(5)` raise
nothing, the clause's action raises the
`AttributeError` of the `to_plan` that 5 lacks, and so does the merge of a condition of `mergeInto`.
An action builds its assignments as PySpark's `_build_merge_action`, in the order of
`assignments.items()`, in which the merge sends them, and raises their errors in that order, before
those of the condition: a key that is not a str, bytes too, is the bare `AssertionError` of
PySpark's `expr`, a value that is not a Column the `AttributeError` of its `to_plan`, and an
object that is not a mapping the `AttributeError` of its `items`.

A str that a later call unsets or replaces is no longer in the request, in the same cases as in
PySpark. A stream reader's option of None leaves its option as it was, as PySpark's does. A batch
reader's keeps it as None, which its `load` leaves out of its plan, and which the plans of its
`table` and `changes` set, which fails their requests with protobuf's `TypeError`, as PySpark's
do. An option
key, a table property or a config key that is not a str fails at the same call as PySpark's, with
the same error: an `AssertionError` at a reader's load of a path, and protobuf's `TypeError` where
the request is built otherwise.

The function of a UDF, of a pandas or Arrow UDF of any eval type, of a UDTF (called, of a table
argument, in a lateral join) and of `mapInPandas`, `mapInArrow`, `applyInPandas`, `applyInArrow`,
the cogroup's `applyInPandas` and `applyInArrow`, `applyInPandasWithState`,
`transformWithStateInPandas` and `transformWithState` is pickled each time a request builds a plan
that holds it, as PySpark's `PythonUDF.to_plan` and `PythonUDTF.to_plan` pickle it. So a value that
the function reads, changed after the function was applied and before an action, is the value that
the action runs with, and so for each later action. The output type of a UDF and of a map or
group-map function is converted then too, of the type object that PySpark's `PythonUDF` keeps, as
are the return type of a UDTF, of the object that its `PythonUDTF` keeps, the output schema of
`transformWithState` and `transformWithStateInPandas` and the schema of `DataFrame.to`: a
`StructType` changed in place after the call, as by `add` or by a field renamed, is the type that
the request sends, and that the pickled command of a UDF carries, as it is then. A type
that the conversion does not take, such as `AtomicType()` or a `StructField` as a type, fails the
requests with its `UNSUPPORTED_OPERATION`, not the call, where the eval type of the UDF does not
refuse it at the call, as the Arrow and pandas ones do with `NOT_IMPLEMENTED`. A function that does
not pickle fails each request of the plan, from the first, such as `df.schema`, with PySpark's
error, the `PicklingError` of `CloudPickleSerializer.dumps`, or `UDTF_SERIALIZATION_ERROR` of a
UDTF, and no request is sent; the call that applies it does not fail. The functions are pickled in
the order in which PySpark
builds the plan, the initial state of `transformWithState` before its function and the arguments of
a UDTF before its handler, and the first error of the build, of a pickling or of a value that the
plan does not take, such as a str above, is raised. The return type of a UDTF is converted after
its arguments and before its handler is pickled, as PySpark's `PythonUDTF.to_plan` converts it with
`pyspark_types_to_proto_types`, of the `PythonUDTF` that the call made: of the return type object
and the handler class of the UDTF at the call, a str as its `UnparsedDataType`, and of the
`__name__` of the class then, which `UDTF_SERIALIZATION_ERROR` names; so a `returnType` or a `func`
set on the UDTF after the call is not that of the call. A value that is neither a DataType nor a
str, such as 5, is kept by the UDTF as it is, and each request of a call of it raises
`UNSUPPORTED_OPERATION` of `data type 5`, of the str of the value when the request is built, or the
error of that str; so does a type that has no proto, such as `AtomicType()`, also one that a type
changed after the call holds. `spark.udf.register` and `spark.udtf.register` are commands, which
pickle the function when they are called, as PySpark's do; `spark.udtf.register` and
`spark.client.register_udtf` set the name and convert the return type first, as it is then, and the
latter reads the `__name__` of the handler class before them, as PySpark's `PythonUDTF` does. A
copy of a Column, a DataFrame, grouped data or a table argument holds the functions, which the
requests of the copy pickle, as PySpark's:
`copy.copy` shares them, and `copy.deepcopy` deep-copies each with the `memo` of the call, so a
function, a lambda or a closure is the function itself, and a callable object, a bound method or a
`functools.partial` is a copy, which the copies of one call share, as the two Columns of a UDF of
`copy.deepcopy([c1, c2])` do. `copy.copy` shares the output type of a UDF and the return type of a
UDTF too, and `copy.deepcopy` copies them, as PySpark's copy of the plan does: a type changed in
place after a deep copy is not that of the copy, nor one changed after a pickle that of the
unpickled DataFrame. A request warns once of a window of no partitions, as PySpark's, in the order
of the build, though the drop-in builds the plan of a request twice when it has such functions, as
it pickles them between the two builds: a window that comes after a function in the build warns in
its turn among the picklings, so a function before it that does not pickle raises the error of its
pickling without the warning, and the error of the warning, of an `"error"` filter of the
warnings, is raised before the functions after the window are pickled, as PySpark's.

A pickled Column, DataFrame or table argument carries the functions of its UDFs, as PySpark's
pickle carries those of its plan objects: the `(func, returnType)` of a UDF, the handler class, its
name and the return type of a UDTF, and the schema of `DataFrame.to`, which the requests of the
unpickled object convert and pickle when they build its plan, of the holes in place of their
commands in the relation or the expression of the pickle, also on the server's Python worker,
where PySpark's plan pickles them. So `pickle.dumps` of one of a UDF of a lambda raises the error
of pickling the lambda, as PySpark's, and of a function that `pickle` pickles by reference, such as
a function of a module or of `__main__`, pickles, of the function by its name, which the requests
of the unpickled object pickle as it is then: a value that the function reads, changed after the
pickle, is the value that they send, and a function that does not pickle then fails them with the
`PicklingError` of `CloudPickleSerializer.dumps`. `cloudpickle.dumps` of a function that does not
pickle by value raises the error of pickling it, such as the `TypeError` of a lock that it reads,
as PySpark's, and `CloudPickleSerializer.dumps` its `PicklingError`. A plain pickle of a DataFrame
fails at the first function that it does not pickle, and else at the function that gives back its
session, as PySpark's pickles its plan and then its session. An output type of a UDF or a return
type of a UDTF that the build of the plan does not convert, such as `AtomicType()` of a UDF without
Arrow, or a `StructType` changed to hold it after the call, does not fail the pickle: as PySpark's,
which keeps the type object, the requests of the unpickled Column and table argument raise its
`UNSUPPORTED_OPERATION`, and so does the unpickle of a DataFrame, which builds its plan.

A pickled Column of subqueries, of `scalar()`, `exists()`, `isin` of a DataFrame or a table
argument, carries the relations of the plans that they refer to, of their plan IDs, as PySpark's
carries the plan objects of its subqueries, which keep their `_plan_id`: a plan of the unpickled
Column sends them in its `WithRelations`, as a plan of the Column does, also on the server's Python
worker, where the Column is PySpark Connect's. `Column._expr` pickles as the expression of the
unpickled Column, and `TableArg._subquery_expr` as that of a Column of its subquery, as PySpark's
expressions pickle their fields. A pickled, cloudpickled or deep-copied Column keeps the origins
of its expressions, of their call sites, as PySpark's unpickled expressions keep their `origin`,
whether the capture of origins is on or not (`is_debugging_enabled`): so does a sort order or a
CASE WHEN, which the unpickle makes again of its parts, and the sort orders of a window
specification or of a table argument.

A DataFrame or a Column holds the functions of its UDFs and the plans of the Python client as
Python's garbage collector sees them, as it sees PySpark's plan objects: each part of a plan or of
an expression that holds such an object, directly or below it, has a Python object, a
`SharedPart`, which holds a reference to the `SharedPart`s of the parts that it holds, one for each
of its references to them, and reports the object of its part. The objects are the function of a
UDF, a UDTF or a map or group-map function, the plan of a `DataFrame(plan, session)`, which its
relation keeps, also through the subqueries of the expressions of a plan, the plan that an unpickled
Column of the subquery of a DataFrame of a plan that did not build keeps, to build it at its
requests, the plan and the exception of a build that failed, which a plan keeps, and the Python
Observations of `observe`. Each object that holds a reference to such a part holds a reference to
its `SharedPart` and reports it, as each of PySpark's objects holds its plan objects: a DataFrame, a
generator of `toLocalIterator`, which holds its DataFrame until it ends, as PySpark's generator
holds `self`, and the expressions of the client, the `_expr` of a Column, the `_subquery_expr` of a
table argument and the entries of the lists of a window specification, which `copy.copy` and
`copy.deepcopy` copy as they copy a Column, and which the `__dict__` of the Column, the table
argument and the window holds, as PySpark's hold their expressions, so the collector sees them
through it, and a Column of an expression that it replaced holds the parts of the new one only. So
a reference cycle through such an object and any of the objects that hold it, one or several, is
collected, as PySpark collects it: through a bound method of an object that keeps a Column and a
DataFrame of it, or a DataFrame and one derived from it; through a plan of the Python client whose
attribute
keeps both its DataFrame and a Column of its subquery, or a table argument of it. A function or a
plan that a live object reaches is kept as it is, and pickled at the requests of that object. Those
that hold their DataFrame report it too, and grouped data, their cogroups, `na` and `stat` hold
their DataFrames, Columns and grouped data in their `__dict__`, and the writers of `write` and
`writeTo`, the stream writer of `writeStream` and the merge writer of `mergeInto` the plans of their
DataFrames, and the merge writer and its clauses their Columns and the messages of the actions, as
PySpark's, which the collector sees, so a cycle through the DataFrame alone, such as
`df.back = df.groupBy("id")`, is collected.
The DataFrames of the plan of another,
`DataFrame(df._plan, session)`, `copy.copy(df)` and a `copy.deepcopy(df)` that copies no function or
Observation of the plan, hold the plan of `df` and not `df`, as PySpark's hold its plan object, so
`df` is freed at its last reference, its attributes with it. The plan object of `df._plan` holds
`df`, where PySpark's does not: so a DataFrame that keeps its stream writer, of that plan object
(`df.w = df.writeStream`), is a reference cycle, which the collector frees, where PySpark frees the
DataFrame at its last reference (see `dev/parity/known-differences.md`). A `SharedPart` reports
nothing while a
reference to its part is of a holder that the collector does not see, such as the DataFrames that a
native model of `pyspark.ml.connect` keeps of its fit: the summary dataset of its training summary,
which the summary keeps too, and the training dataset and the datasets of the classes of a
OneVsRestModel. So the collector frees no object that such a holder reaches, and a cycle through one
is collected once the holder is gone. A cycle through the holder and what it keeps never is: a
function of a UDF of the training DataFrame of a model that keeps the model, or a Python plan of
that DataFrame that keeps it, keeps the model alive, and its object in the server's ML cache, where
PySpark collects them (see `dev/parity/known-differences.md`). The objects of the native ML classes
report the other Python objects that they hold, as PySpark's hold them as attributes: a model its
training summary, its cached members (`trees`), its own defaults and its `RemoteModelRef`, of which
a copy holds a reference of its own, and the one that it was made of while no transform, writer or
call holds its object; an estimator, a transformer, a Pipeline, a PipelineModel, a OneVsRest and its
model, and a validator and its model the values of their params and their attributes (stages,
classifier, models, best model, metrics); a `fitMultiple` iterator its estimator, dataset and param
maps. So a model and its summary in a cycle (`model.summary.model = model`) are collected, fit after
fit, and the model's object is deleted from the ML cache of the server, as PySpark's. Nor does a
`SharedPart` report anything while a call of the binding that may clone or drop its part runs
without the GIL, on any thread: a call that builds a request of the plans of the objects that it is
given, such as an action of a DataFrame or a fit of `pyspark.ml.connect`, clones and drops the parts
of those plans, of what the call keeps alive anyway; the other calls,
such as the waits of streaming queries, the calls of the configuration, the catalog and the session,
and the messages that the Python client builds, such as the commands of the writers, of the starts
of the stream writer and of the merges, whose DataFrames the binding replaces in them before the
call, clone none. So the collector gets the same references of each object in each of its
passes, and a cycle through parts that no call clones is collected whenever the collector runs, as
PySpark's is, except while a listener of the Rust API of the session gets an event, which may clone
any part. On a free-threaded build of CPython, the parts have no `SharedPart`, and a cycle through
them is not collected (see "Garbage collection" in `dev/parity/known-differences.md`).
An `ArrayType`, a `MapType`, a
`StructField` and a `StructType` report the type objects, the flags and the metadata dict that they
hold, and the lists of a struct, as PySpark's `__dict__` holds them, so a cycle through a type that
one holds, such as a `DecimalType` with an attribute that keeps the struct, is collected. The
stream writer, PySpark's, keeps the functions of `foreachBatch` and `foreach` as their pickles, so a
function that keeps its writer is freed with its last reference, and a copy or a pickle of the
writer carries the pickle of the call. The registration of `dataSource` reports its
session, and the builder of `pyspark.sql.connect.session.SparkSession` holds the dict of its
options, its channel builder and the list of its hook factories in its `__dict__`, which its shallow
copies share, as PySpark's plain objects hold them, so a session that keeps its `dataSource`, or an
option, a channel builder or a hook factory that keeps its builder or a copy of it, is collected.
The readers and the writers of `read`, `readStream`, `write` and `writeTo`, PySpark's classes, hold
the dict of their options, whose keys and values may be any objects, such as a key that is not a
str, and the writers the format, the save mode, the partitioning and clustering columns and the
provider as they are, in their `__dict__` (a writer in its `_write`), which their shallow copies
share, as PySpark's copies share the dict of their options and the `_write` of a writer: so a cycle
through one, such as a key of an option that keeps its reader, is collected, also with a copy of the
reader, and freed with the last of the copies.

A clause of `mergeInto`, PySpark's, builds the messages of its condition and assignments when it is
added (`_build_merge_action`), which pickles the functions of their UDFs then and raises there, and
the merge sends them as they were built then: so a value that a function reads, changed after the
clause, is not that of the merge, also of a copy of the writer, and a window of no partitions of a
clause warns once, at the clause.

A UDF and a UDTF are of PySpark Connect's classes, `UserDefinedFunction` of
`pyspark.sql.connect.udf` and `UserDefinedTableFunction` of `pyspark.sql.connect.udtf`, as in
PySpark. The classic `pyspark.sql.udf` and `pyspark.sql.udtf` are PySpark's modules, of Apache
Spark as they are, whose classes, which `pyspark.sql.functions` names too, need a SparkContext: so
`isinstance` of a UDTF that `udtf` builds against the classic `UserDefinedTableFunction` is False,
as in PySpark. `udf` gives a function in both clients, which is of neither class.

So it is for the objects of a session: each is of PySpark Connect's class, of its module of
`pyspark.sql.connect`, as `repr`, `type(x).__module__` and pickle show it. These are the
SparkSession, its `conf`, `catalog`, `read`, `readStream`, `streams`, `udf`, `udtf`,
`dataSource`, `tvf` and `client`, and a DataFrame's `write`, `writeTo`, `writeStream`,
`mergeInto` and its clauses, `na`, `stat`, `groupBy`, `cogroup`, `asTable` and the WindowSpec of
its windows, and an `Observation`. So are `SparkSession.Hook`, nested in the session class, and
the classes of the client, which `pyspark.sql.connect.client.core` defines, as PySpark's: its
channel builders, `SparkConnectClient` and `RpcDeadlines`, of its `__all__`, which its package
names too, and `PlanObservedMetrics`, `AnalyzeResult` and `ConfigResult`, which only `core` names.
The functions of these classes are of `core` too, where `inspect` finds their source, and their
pickles and those of the results are PySpark's. The package `pyspark.sql.connect.client` is
PySpark's, of Apache Spark as it is: of no `__all__`, it imports `*` of `core` and the `getLogLevel`
of `pyspark.sql.connect.logging`. So its names, of `dir()` and of
`from pyspark.sql.connect.client import *`, are PySpark's, those classes, `getLogLevel` and the
modules `core` and `retries`, but for PySpark's modules `artifact` and `reattach`, which the
transport replaces, and for `retries` until it is imported (by `get_retry_policies`,
`set_retry_policies`, a read of `_retry_policies` or a `copy.copy` of a client, a `retry_policy` of
`SparkConnectClient`, or an import of it), as it imports grpc: until then the package has no attribute `retries`, which PySpark's `core` imports. `core`
defines PySpark's names, `PYSPARK_ROOT`, `_is_pyspark_source` and `_import_zstandard_if_available`
too, and the private helpers of the drop-in. Of the names that PySpark's `core` imports, it has
those that it imports too, such as `logger` and `os`, and those of `retries` (`RetryPolicy`,
`Retrying`, `DefaultPolicy` and `DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME`) once they are read,
which its `dir()` does not list, but not the others, such as `grpc`, `pd` and `ArtifactManager`.
The package and `core` need none of pandas, PyArrow, grpc and protobuf, which PySpark's `core`
imports, and import without them (see `dev/parity/known-differences.md`).

The module that the class of each of the objects of a session above names is imported once the
object is made, as PySpark imports it with the module that defines the class: the module of the
session imports those of its DataFrames, its `conf`, its readers and its queries, and with them
those of grouped data, merges, writers, columns, observations and table arguments, as PySpark's
does, and `catalog`, `tvf` and `dataSource` import theirs when they make their objects, as
PySpark's properties do, and a Column that a plain pickle gives back, of no session, imports its
own, as PySpark's unpickle imports the module of its class. So cloudpickle, which pickles a class
by reference only when its module is in `sys.modules`, and else by value, as a dynamic class,
pickles these classes by reference, also in a new interpreter: a UDF or a `foreachBatch` function
that captures such an object pickles it as PySpark's, which the server's Python worker unpickles as
PySpark's object.

The functions of `pyspark.sql.connect.functions` are of its modules (`builtin` and
`partitioning`), and so are those of `pyspark.sql.connect.avro.functions` and
`pyspark.sql.connect.protobuf.functions`: they are not those of `pyspark.sql.functions`,
`pyspark.sql.avro.functions` and `pyspark.sql.protobuf.functions`, as in PySpark. An alias, such
as `column`, `negate`, `power` or `random`, is the function `col`, `negative`, `pow` or `rand`, of
that name, as PySpark binds it.

`pyspark.sql.utils` is PySpark's module, of Apache Spark as it is, of its names and its helpers:
`is_remote()` and `is_remote_only()` are True, and its decorators, such as `try_remote_functions`
and `dispatch_df_method`, run the functions of PySpark Connect's modules of their names, and the
methods of the classes of Spark Connect, which an object of another class does not have
(`NOT_IMPLEMENTED`). The package `pyspark.sql.connect` has the names of PySpark's,
`check_dependencies` and `utils`, which its `__init__` imports, and importing `pyspark` imports no
module of it, as in PySpark: the functions of `pyspark.sql.functions`, PySpark's, import
`pyspark.sql.connect.functions` when one of them is first called. Its import does not run
`check_dependencies`, which PySpark's does, nor does the import of `pyspark.ml.connect`, which has
the name too, as PySpark's (see `dev/parity/known-differences.md`).

`pyspark.sql.SparkSession` is PySpark's classic class, the module `pyspark.sql.session` of Apache
Spark as it is, as in PySpark. Its builder reads `spark.remote` or `SPARK_REMOTE`, raises
`CONNECT_URL_NOT_SET` without either, also while a session is active, and makes the session with
the builder of `pyspark.sql.connect.session.SparkSession`, of which the session is. So
`isinstance(spark, pyspark.sql.SparkSession)` is False, as in PySpark, and `active()` and
`getActiveSession()` of the classic class give the session of Spark Connect. The builder of
`pyspark.sql.connect.session.SparkSession`, of a `classproperty` of `pyspark.sql.session`, is
PySpark Connect's: it does not read `SPARK_REMOTE`, raises `SESSION_NEED_CONN_STR_OR_BUILDER`
without a URL or a channel builder, returns the active session from `getOrCreate`, and its
`enableHiveSupport` raises `NOT_IMPLEMENTED`, where that of the classic builder sets
`spark.sql.catalogImplementation`.

The classic classes of `pyspark.sql.conf`, `pyspark.sql.tvf`, `pyspark.sql.datasource`,
`pyspark.sql.merge`, `pyspark.sql.observation`, `pyspark.sql.catalog`, `pyspark.sql.readwriter`,
`pyspark.sql.udf`, `pyspark.sql.udtf`, `pyspark.sql.group`, `pyspark.sql.pandas.group_ops`,
`pyspark.sql.streaming.readwriter` and `pyspark.sql.streaming.query` are PySpark's, of those
modules of Apache Spark as they are, which `pyspark.sql` and `pyspark.sql.streaming` name as
PySpark's do. So `isinstance` against them is False, as in PySpark, of a session's `conf`, `tvf`,
`dataSource`, `catalog`, `read`, `readStream`, `streams`, `udf` and `udtf`, of a DataFrame's
`mergeInto`, `write`, `writeTo`, `writeStream`, `groupBy` and `cogroup`, of a StreamingQuery and of
the UDTFs above. Their constructors and methods are those of Spark Classic, as PySpark's:
`pyspark.sql.DataFrameReader(spark)` and `pyspark.sql.Catalog(spark)` raise the
`JVM_ATTRIBUTE_NOT_SUPPORTED` of `_jsparkSession`, and so does the `register` of a
`pyspark.sql.udf.UDFRegistration(spark)`, as in PySpark, where the reader of `spark.read` and
`pyspark.sql.connect.readwriter.DataFrameReader(spark)` are PySpark Connect's.
`Observation(...)` makes an Observation of `pyspark.sql.connect.observation`, which is not an
instance of `pyspark.sql.Observation`, as PySpark's `__new__` makes one.
`pyspark.sql.connect.window.Window` is a subclass of `pyspark.sql.window.Window`, as in PySpark, of
PySpark Connect's code.

The classic `Column`, `DataFrameNaFunctions`, `DataFrameStatFunctions`, `Window`, `WindowSpec`,
`TableArg` and `TableValuedFunctionArgument` are PySpark's classes, of their modules and MROs
(`pyspark.sql.column` and `pyspark.sql.table_arg` of Apache Spark as they are), which
`pyspark.sql` names as PySpark's does. Their Connect classes, which the objects of a session are
of, are native, and a native class cannot have a Python base, so they are their virtual
subclasses, as the DataFrame is of the public `pyspark.sql.DataFrame`: `isinstance` and
`issubclass` hold as in PySpark, while the MRO of such an object is its native class and
`object`, where PySpark's has the classic class and its bases after its Connect class. The
explicit calls of the classic methods dispatch as PySpark's (`Column.alias(col, "x")` runs the
Connect class's, and of another object it raises `NOT_IMPLEMENTED`), the classic `Column(...)`
and `WindowSpec(...)` raise the `ModuleNotFoundError` of `pyspark.sql.classic`, and the others
construct and subclass as PySpark's plain classes do. A patch of a classic class or of
`TableValuedFunctionArgument`, such as a method that a library sets on `pyspark.sql.Column`,
reaches the objects of a session as it reaches PySpark's Connect objects, which inherit it: not
where the Connect class has the name itself, nor where the objects set it on themselves (`_expr`
of a Column, `df` of `na`) and the patch is not a data descriptor; deleting it gives the native
class its own attribute back. A data descriptor without a setter patched over such an attribute
(`_expr` of a Column, `_partitionSpec` of a WindowSpec, `_subquery_expr` of a TableArg, `df` of
`na` and `stat`) raises its `AttributeError` when the `__init__` of a Connect object sets it, as
PySpark's. What stays different: these classic classes are of an `ABCMeta`
subclass, of an `_abc_impl` among their attributes, and a class that subclasses one of them is
of `abc.ABCMeta`, where PySpark's are of `type`. The readers and writers of PySpark Connect are
subclasses of the `OptionUtils` of `pyspark.sql.connect.readwriter`, a class of private methods
only: so are the drop-in's readers and writers, PySpark's classes.

PySpark's classes are plain Python classes, which a user subclasses. So are these of the drop-in:
the DataFrame, `Observation`, and the `SparkSession`, `Catalog`, `TableValuedFunction`,
`DataSourceRegistration`, `RuntimeConf`, `StreamingQuery`, `StreamingQueryManager`,
`StreamingQueryListenerBus`, `DataFrameReader`, `DataStreamReader`, `GroupedData`,
`PandasCogroupedOps`, `DataFrameNaFunctions`, `DataFrameStatFunctions`, `DataFrameWriter`,
`DataFrameWriterV2`, `DataStreamWriter`, `MergeIntoWriter` and its clauses, `Column`, `WindowSpec`
and `TableArg` of `pyspark.sql.connect`, which construct of their PySpark arguments
(`SparkSession(connection=...)`, `Catalog(spark)`, `RuntimeConf(spark.client)`,
`StreamingQuery(spark, id, runId)`, `DataFrameReader(spark)`,
`GroupedData(df, "groupby", [F.col("id")])`, `DataFrameNaFunctions(df)`, `Column(col._expr)`,
`WindowSpec([], [], None)`, `TableArg(table._subquery_expr)`), also a subclass of an `__init__` of
its own, and whose setters, copies and pickles are of the subclass (but those of a session, which
are the active session of its ID, as PySpark's); `SparkSession.Builder`,
`SparkSession.Hook`, `StorageLevel` and `PythonEvalType`; and the data types, from `StringType` and
`DecimalType` to `ArrayType`, `StructField` and `StructType` (see above). A session of a subclass is
the session that its DataFrames, readers and catalogs give back, as in PySpark.

The builder of `pyspark.sql.connect.session.SparkSession`, `SparkSession.Builder()`, which
`SparkSession.builder` makes at each access, keeps its state in its `__dict__`, as PySpark's: its
`_options`, a dict of the `to_str` of each value that `config` is given, of any key (`remote` and
`appName` call the `config` of the builder, of a subclass too, of `spark.remote` and
`spark.app.name`), its `_channel_builder` and its `_hook_factories`, a list that `_registerHook`
appends to. `create` and `getOrCreate` read them at the call, as PySpark's do: a channel builder and
a `spark.remote`, or neither, raise `SESSION_NEED_CONN_STR_OR_BUILDER`, a `spark.remote` of None
fails PySpark's assertion, and a str that UTF-8 cannot encode raises the `UnicodeEncodeError` of its
channel. Both call the `create` and the `_apply_options` of the builder, of a subclass too, which
sets the options but `spark.remote` and `spark.master` on the session in one request, after the
`spark.sql.` options of the last `PYSPARK_REMOTE_INIT_CONF_<i>` of the environment, of
`PYSPARK_REMOTE_INIT_CONF_LEN` of them, as PySpark's `_apply_options` does; the first option that
the request does not take, such as a key that is not a str, fails it before any option is set. So
`vars(SparkSession.builder.remote("sc://h"))` is PySpark's, an assignment of `_options` or a
subclass of the class changes what `create` reads, a builder whose `__init__` never ran raises the
`AttributeError` of `_options` at its first `config`, as PySpark's, and the copies and pickles of a
builder are those of its `__dict__`: a shallow copy shares the dict of its options and the list of
its hook factories, and a deep copy and a pickle copy them.

The `SparkSession` of `pyspark.sql.connect.session` keeps its state in its `__dict__`, as PySpark's:
its `__init__`, and the builder, `newSession()` and `cloneSession()` of the session that they make,
set its `_client`, the `SparkConnectClient` of the session that `spark.client` gives, its
`_session_id`, the `_session_id` of the client, and its `release_session_on_close`, True. `streams`
keeps its `StreamingQueryManager` as its `_sqm`, unless the session has one, and `catalog` and
`version` are `functools.cached_property`s of PySpark's functions, whose first access keeps the
value in the `__dict__`, which the next ones read: `version` asks the server once. The methods of
the session read its state at the call, as PySpark's: `client` is the `_client`, `session_id` the
`_session_id`, of which the session pickles, `stop` releases the session unless
`release_session_on_close` is false, `is_stopped` is the `is_closed` of its `client`, and the
requests of the session, of its catalog, of its streams and of its `conf` are of its `_client`, or
of its `client` where PySpark reads that: of the session itself while that is its own client, and of
the session of another `SparkConnectClient` of the drop-in that is assigned to it, such as the
client of another session. So `vars(spark)` is PySpark's, `_client`, `_session_id` and
`release_session_on_close`, and then `_sqm`, `catalog` and `version` in the order that they are
read; a session whose `__init__` never ran, as of a subclass whose `__init__` does not call it,
raises the `AttributeError` of its `_client` at its first use, as PySpark's; and a session whose
catalog or streams were read is in a reference cycle with them, as PySpark's, which a collection of
the garbage collector frees (see `dev/parity/known-differences.md`). What differs is of the native
transport: a client of another class raises a `TypeError` of the requests of the session, where
PySpark's calls the methods of that client, though its `conf` calls the `config` of the client and
`is_stopped` reads its `is_closed`; and the DataFrames of `range`, `sql` and `createDataFrame` are
of the session of the client at the call, which they hold natively, where PySpark's DataFrame reads
the client of its session at each request (until SPARK-62218).

The catalog, the table-valued functions, the data source registration and the runtime
configuration of a session keep their state in their `__dict__`, as PySpark's: their `__init__`
sets the object that it is given, whatever it is, as their `_sparkSession` (`sparkSession` of a
registration, `_client` of a configuration), and their methods read that attribute where PySpark's
read it, through a property, a `__getattr__` or an assignment of a subclass too. So
`vars(spark.catalog)` is `{'_sparkSession': spark}`, an object whose `__init__` never ran, as of a
subclass whose `__init__` does not call it, raises the `AttributeError` of that attribute at its
first use, as PySpark's, and their copies and pickles are those of their `__dict__`. Of another
object than a session, as a proxy of a session, a catalog and the table-valued functions make the
requests of the session of the client of its `client`, the `range` of the table-valued functions
is the `range` of that object and the `registerFunction` of a catalog its `udf.register`, as
PySpark's call them. A configuration calls the `config` of its `_client` with the
`ConfigRequest.Operation` of `pyspark.sql.connect.proto` of each request, and a registration the
`register_data_source` of the `_client` of its `sparkSession`, where that client is of another
class or overrides the method, as PySpark's do; the native transport of the drop-in's client makes
them otherwise. A catalog and the table-valued functions of a client of another class raise its
`TypeError`, and the DataFrames that they make are of the session of the client, where PySpark's
call the methods of the client and make DataFrames of the object (see
`dev/parity/known-differences.md`).

So do the streaming queries, `spark.streams` and its listener bus, of PySpark's state: a
`StreamingQuery` of its `_session`, `_query_id`, `_run_id` and `_name`, which `id`, `runId` and
`name` give back and its methods read at each request, as `_session.client`; the
`StreamingQueryManager` of its `_session` and its `_sqlb`, the `StreamingQueryListenerBus(self)` of
its `__init__`, to which `addListener` and `removeListener` add and remove the listeners, as
`self._sqlb.append(listener)` and `self._sqlb.remove(listener)`; and the bus, of its `_sqm`, its
`_listener_bus` list, its `_execution_thread` and its `_lock`. The listeners of the bus are the
entries of `_listener_bus`, in its order, which `append` and `remove` change as PySpark's do (of the
first entry that is the listener or equal to it), and to which the events are posted, of
`post_to_all`, as the list is then: so a listener that a user puts on the list gets the events too,
and one of two entries gets each event twice. The started event of a query that a `DataStreamWriter`
starts goes to what `spark.streams._sqlb.post_to_all` is then, as PySpark's writer posts it: of the
bus's own method, as a delivery of the native listener bus of the session, which also gets it to
its native listeners, and for which a removal of a listener from another thread waits, as for
PySpark's bus, which holds its lock while it posts the event; of a patch of the method, of the bus
or of its class, of the method of a subclass of the bus and of a bus of another class, as PySpark
calls it. The bus subscribes the session to the events of the
server with the first listener of the native listener bus of the session, and ends the
subscription with its last, whose thread posts the events: `_execution_thread` stays None, and the
`_lock` is not held while the listeners are called, so that a listener may add or remove listeners
of its callbacks, where PySpark's waits for ever. The end of the events, of a failure, of the end of
the stream of the server, or of a stop of the session or a close of its client, takes the listeners
off the list, as PySpark's handler clears it when its events fail. The copies and the pickles of
these are those of their `__dict__`, as PySpark's: a shallow copy shares the state, and a plain
pickle of `spark.streams`, its bus or a query fails with the `AttributeError` of the creator of the
session of its `_session`, and a deep copy and cloudpickle of the manager or its bus with the
`TypeError` of its lock.

The grouped data, the cogrouped data and the null and statistics functions of a DataFrame keep
their state in their `__dict__` too, as PySpark's: the `_df`, `_group_type`, `_grouping_cols`,
`_pivot_col`, `_pivot_values` and `_grouping_sets` of `GroupedData`, the grouped data `_gd1` and
`_gd2` of `PandasCogroupedOps`, and the `df` of `DataFrameNaFunctions` and
`DataFrameStatFunctions`, which their `__init__` sets of PySpark's arguments. The `__init__` of
grouped data asserts them as PySpark's (a bare `AssertionError` of an argument that is not a
DataFrame, a grouping of PySpark's names or a list of Columns), and makes the pivot values the
`F.lit` of each; that of cogrouped data checks `gd1._df._check_same_session(gd2._df)` first, of
any objects of a `_df`. Their methods read that state where PySpark's read it, of a property, a
`__getattr__` or an assignment of a subclass too: an aggregation of grouped data is of the
grouping that they hold then, with the assertions of PySpark's plans, `count` and `apply` call the
`agg` and the `applyInPandas` of the object, a `pivot` makes grouped data of the same DataFrame and
the same list of grouping columns, of `self._df[pivot_col]`, and a `_df` that is not a DataFrame,
such as a proxy of one, is grouped as the DataFrame of its `_plan` and its `_session`. The null and
statistics functions call the method of their `df` of each of theirs, as PySpark's
(`df.na.drop()` is `df.dropna(how="any", thresh=None, subset=None)`, of any object, and an omitted
`value` of `replace` is PySpark's `_NoValue`, which a DataFrame takes for an omitted value). An
object whose `__init__` never ran raises the `AttributeError` of its first attribute that
PySpark's method reads (`_df`, `_gd1`, `df`), and their copies and pickles are those of their
`__dict__`, which reach the DataFrame first, as PySpark's. What differs is the plan that a method
of grouped data makes, a value of the core, of the state at the call: a later change of the lists
of the state, such as `gd._grouping_cols.clear()` after `agg = gd.count()`, is not of `agg`, where
PySpark's plan holds the list and builds its message of it at each request.

The `DataFrameReader` and the `DataStreamReader` of `pyspark.sql.connect` are PySpark Connect's
classes, adapted from the source of PySpark 4.3.0 (`pyspark/sql/connect/readwriter.py` and
`pyspark/sql/connect/streaming/readwriter.py`), whose code is PySpark's but for the plans of their
reads, which are the Rust core's: the state of a reader is its `__dict__`, of the `_client`,
`_format`, `_schema` and `_options` (and `_source_name`) of its `__init__`, which keeps any object,
and its setters, its format methods, `_set_opts`, its errors, its copies and its pickles are
PySpark's. A plan of a read checks its arguments as PySpark's `DataSource`, `Read`,
`RelationChanges` and `Parse` do when they are made, with their bare `AssertionError`s (of a format,
a schema, a path or a predicate that is not a str, and of an option of a key or a value that is
not one, but of a value of None, which it leaves out), and fails the requests of its DataFrame with
the errors of the arguments that it cannot send, as PySpark's plan raises them when it is sent: a
str that UTF-8 cannot encode, and an option of None, or of a key or a value that is not a str or
bytes, of `table` and `changes`. The DataFrame of a read is of the session of the reader's
`_client`, which the read resolves (see `dev/parity/known-differences.md`).

So are the `DataFrameWriter` and the `DataFrameWriterV2` of `pyspark.sql.connect`, adapted from the
same source, with PySpark's `WriteOperation` and `WriteOperationV2` of
`pyspark/sql/connect/plan.py`: the state of a writer is its `__dict__`, of the `_df`, `_spark`,
`_write` and `_callback` (and `_table_name`) of its `__init__`, whose setters set the fields of its
`_write` and whose errors, copies and pickles are PySpark's. A write builds PySpark's command of the
fields of its `_write`, over the relation of the plan of its `_df`, the placeholder of the relation
that the core builds of the DataFrame, as `df._plan.plan(client)` gives it, and the
`execute_command` of the client of its `_spark` executes it, whose `ExecutionInfo` the callback of
`df.write` and `df.writeTo` makes the `executionInfo` of the DataFrame. The `_df` of a writer is the
plan object that `df._plan` gave when `write` was read, of which the drop-in's DataFrame makes a new
one at each read (see `dev/parity/known-differences.md`).

So is the `DataStreamWriter` of `pyspark.sql.connect.streaming.readwriter`, adapted from the same
source, with PySpark's `WriteStreamOperation`: its state is its `__dict__`, of the `_session`,
`_write_stream` and `_write_proto` (the `WriteStreamOperationStart` of the `_write_stream`) of its
`__init__`, whose setters set the fields of its `_write_proto` and whose errors, copies and pickles
are PySpark's. A start builds PySpark's command of its `_write_proto`, over the relation of the plan
of the `_write_stream`, which the `execute_command` of the client of its `_session` executes, makes
the `StreamingQuery` of its result, and posts the started event of the query, decoded as PySpark
decodes it, to what `self._session.streams._sqlb.post_to_all` is then, as PySpark's writer posts it
(see the listener bus below). The `_write_stream._child` of a stream writer is the plan object that
`df._plan` gave when `writeStream` was read.

So is the `MergeIntoWriter` of `pyspark.sql.connect.merge`, with its clauses `WhenMatched`,
`WhenNotMatched` and `WhenNotMatchedBySource`, adapted from `pyspark/sql/connect/merge.py`: its
state is its `__dict__`, of the `_client`, `_target_table`, `_source_plan`, `_condition`,
`_callback`, `_schema_evolution_enabled` and the lists of the `MergeAction`s of the clauses of its
`__init__`, and that of a clause of its `writer` and `_condition`, whose errors, copies and pickles
are PySpark's. A clause builds its `MergeAction` when it is added, of the messages of the drop-in's
Columns, and the merge builds PySpark's command over the relation of the plan of its `_source_plan`,
which the `execute_command` of its `_client` executes, whose `ExecutionInfo` the callback of
`df.mergeInto` makes the `executionInfo` of the DataFrame. The `_source_plan` of a merge writer is
the plan object that `df._plan` gave when `mergeInto` was called.

The `Column`, the `WindowSpec` and the `TableArg` of `pyspark.sql.connect` keep their state in their
`__dict__` too, as PySpark's: the `_expr` of a Column, the `_partitionSpec`, `_orderSpec` and
`_frame` of a window specification and the `_subquery_expr` of a table argument, which their
`__init__` sets of PySpark's arguments and their methods read at the call. `Column(expr)` takes the
expression of another Column, as `outer()` does, or an `Expression` of
`pyspark.sql.connect.expressions`, such as a `LiteralExpression` or one of a class of the user's,
whose `repr` the Column shows and whose `to_plan` its requests send, of the fields of the object
then, and raises PySpark's `NOT_EXPECTED_TYPE` of another object, of the errors of CPython's binding
of the arguments of `def __init__(self, expr)` too. An expression whose `str` fails, as one of a
class of the user's whose `__repr__` gives None, as PySpark's base `Expression` does, makes a Column
that is sent as its `to_plan` gives it, and whose `repr`, and that of a Column made of it, raises
that error, as PySpark's raises it at the `repr`. `WindowSpec(partitionSpec, orderSpec, frame)`
asserts PySpark's types of a bare `AssertionError` (a list of expressions, a list of sort orders, a
`WindowFrame` or None), and its `__new__` takes the same arguments. The sort orders of `orderSpec`
are the client's only, those of `asc()`, `desc()` and the others, as the drop-in's
`pyspark.sql.connect.expressions` has no `SortOrder`, whose objects PySpark's takes; `partitionBy`,
`orderBy`, `rowsBetween` and `rangeBetween` make a window of the base class, of the lists and the
frame that they do not replace, shared with this one; `orderBy` of a window and of a table argument
sorts a Column of a subclass by its own `asc()`, and a patch of `Column.asc`,
`Column.__getattribute__` or `Column.__getattr__` is called where PySpark's `_sort_col` and the
methods of a Column call them. `TableArg(subquery_expr)` keeps any object, whose fields its methods
read (an `int` raises the `AttributeError` of `_partition_spec`, as PySpark's). So `vars()` of them
is PySpark's, an assignment of `_expr` gives a Column of the other expression, a subclass whose
property gives the state is of it, an object whose `__init__` never ran raises the `AttributeError`
of its first attribute that PySpark's method reads (`_partitionSpec`, `_orderSpec`,
`_subquery_expr`), or of a Column the `RecursionError` of its `__getattr__`, which reads `_expr`
again. Their copies are those of their `__dict__`, of `__new__` of their class and of the arguments
of their `__getnewargs__`, as PySpark's: a shallow copy shares the state, so a copy of a window
shares its lists, and a deep copy copies it. A window pickles so too, of `copyreg.__newobj__`, at
every protocol, also to a Python worker of the server, where it unpickles as PySpark Connect's; a
Column and a table argument pickle their expression and its plans, as above, and the other entries
of their `__dict__`, and of a subclass unpickle as an object of the subclass, of those entries, as
PySpark's. What differs is of the expressions of the core: a Column that a method makes is a value
of the expressions at the call, so the window expression of `over` is of the lists and the frame of
the window then, and a later change of them is not of it, where PySpark's `WindowExpression` holds
the window and builds its message of it at each request, and a Column made of a Column of an
`Expression` of PySpark's classes holds its message as its `to_plan` gave it then. The expressions
that the drop-in makes, the `_expr` of a Column, the entries of the lists of a window and the
`_subquery_expr` of a table argument, are of the client's class `_ColumnExpression` of
`pyspark.sql.connect.expressions`, of PySpark's `to_plan` and `repr`, and not of PySpark's classes
of the module (`ColumnReference`, `SortOrder`, `SubqueryExpression` and the others), whose fields
they do not have, as the expressions are the core's (see `dev/parity/known-differences.md`).

A pickle of PySpark, which makes an object of its class without arguments and then sets its
`__dict__`, loads in the drop-in: a reader, a catalog and a registration, of their `__dict__`; and
the drop-in's pickles of them load in PySpark, of their classes, as PySpark's own do: where the
session of their ID is active. The other native classes cannot be subclassed (`TypeError: type
'...' is not an acceptable base type`), and do not construct (`TypeError: cannot create '...'
instances`): their objects are made of the state of the Rust core, a plan, an expression or a
query, which the arguments of PySpark's constructors, the Python objects of its client, do not give,
as the drop-in does not have them, so a subclass would have no `__init__` to call.

A `copy.copy` or `copy.deepcopy` of a DataFrame requests its schema as PySpark's does, one request
for the copy, and one more for the DataFrame in a deep copy when it has no schema yet, so a stopped
session fails the copy with `SparkConnectException`, as PySpark's does. One difference remains:
PySpark's `DataFrame.__getattr__` answers `getattr(df, "__deepcopy__")` and
`getattr(df, "__copy__")` by requesting the schema, to read the columns, and raises
`ATTRIBUTE_NOT_SUPPORTED`, so `hasattr(df, "__copy__")` is false; the drop-in defines both
itself, which it finds without a request. An unpickled DataFrame requests its schema once, at the
unpickle, which raises the error of the request or of its conversion, as PySpark's pickle looks the
`__setstate__` of the new DataFrame up through its `__getattr__`. Its cached schema is then that of
the pickled DataFrame, as the state of PySpark's pickle sets it, of the objects that it carries: of
none, its `schema` requests it again. So does the unpickle of an object that PySpark pickles with
its DataFrame: a writer, of its callback, grouped data, `na` and `stat`; a stream writer, which
PySpark's pickles of the plan of the DataFrame, requests nothing.

A copy, a deep copy or an unpickle of a DataFrame keeps the schema that it requests apart from the
cached schema of the state, as PySpark's keeps it as the `_cached_schema_serialized` that its
`schema` returns. So the two differ when the schema on the server changed since the DataFrame
cached it, such as that of a view that was replaced, and the copy gives, as PySpark's, the schema
that it requested for `schema`, `printSchema`, `toArrow`, the set operations of equal cached
schemas (`c.union(c).schema`) and the numeric aggregations of its grouped data (`sum()` of no
columns), and the cached schema for `columns`, `dtypes`, `repr`, the column lookups and the
methods that copy the cached schema, such as `filter`. A copy of such a copy keeps the schema that
its own reconstruction requests, and the cached schema, which a pickle of it carries too. Of a
DataFrame that had no cached schema, both give the schema of the server, which the next read of the
copy requests again.

`to` caches its schema whatever it is, as the object that it is, as PySpark's, so of a DataType that
is not a struct, such as `LongType()`, `schema` gives a copy of it without a request, while its plan
is sent, which the server rejects; and of a `StructType` changed in place after the call, `schema`,
`columns` and the plan that the requests send are of the type as it is then. The methods that copy
the cached schema of their input in PySpark copy that of `to` too: `alias`, `filter` and `where`,
`limit`, `sort`, `orderBy` and `sortWithinPartitions`, `hint`, `repartition`, `repartitionByRange`,
`repartitionById` and `coalesce`, `distinct` and `dropDuplicates`, `sample`, `randomSplit` and
`observe`, as do the set operations of equal ones.

The other objects of a session above keep the attributes that are set on them through copies and
pickles, as PySpark's copy and pickle their `__dict__`: a shallow copy shares them, and a deep copy,
a pickle and cloudpickle copy them, where they copy the object at all, and where PySpark's fail, the
drop-in's fail with the same errors. Their `__dict__` holds only those attributes, where PySpark's
holds the state of the object too.

A table name that is neither a str nor bytes, and bytes that are not UTF-8, give a DataFrame of
`read.table`, `readStream.table` and `changes`, whose plan raises protobuf's `TypeError` or the
decode error when it is built, as PySpark's do. A reader's format that is not a str is kept as the
object it is, which `_format` gives back, and a copy or a pickle of the reader keeps it.
The Connect builder's `remote` keeps a value that is not a str as PySpark's does, and `create`
rejects it as an invalid Spark Connect URL, as PySpark's `create` does for its `str()`. An explicit
`None`, given as `remote(None)`, `remote(location=None)` or `config("spark.remote", None)`, is
kept as None: `create` and `getOrCreate` fail PySpark's assertion, unless a session exists, which
`getOrCreate` returns. A later `remote` replaces it. A `remote()` without an argument is
`sc://localhost`, as in PySpark's Connect builder. The locks are PySpark's: `getOrCreate` holds
`SparkSession._lock` from its lookup to the options that it sets, `create` to make its session the
default one, and `stop()` throughout, and the builder holds `SparkSession.Builder._lock` in
`config`, `remote`, `appName`, `channelBuilder` and `_registerHook`, and while it sets the options
of a session. Each is a class attribute, an `RLock`, read at each call, so a thread of the program
that holds one holds off the calls of the other threads that take it, and a signal ends the wait
for it. As PySpark's, `getOrCreate` sets the options of a session that it creates twice, in
`create` and again.

The client honors the HTTP proxy of the environment (`grpc_proxy`, `https_proxy`, `http_proxy`,
`no_grpc_proxy`, `no_proxy`) with grpcio's rules, see [HTTP Proxies](configuration.md#http-proxies),
of a loopback host too, unless `no_proxy` lists it, with one difference: a proxy address with an
invalid host or port is the error of the creation
of a session of a host that the proxy applies to, where grpcio makes the session and fails its
requests, but for a port past 65535, which it takes modulo 65536 and connects to (a proxy of port
99999 is reached at 34463): the error names the variable at once, rather than requests that fail
later or a connection to a port that was not given. A host that `no_proxy` lists connects
directly whatever that address, as in grpcio, which resolves it only for a host that it proxies.
As grpcio, the client logs a proxy variable that is not a proxy, and a roots file that cannot be
read, as errors on stderr (see [Errors on stderr](configuration.md#errors-on-stderr)).

The `endpoint` of a channel builder is read as grpcio's channel of PySpark's `toChannel` reads its
target: bytes as they are, up to a NUL, as gRPC's core reads a C string, else the `encode("utf8")`
of the object; a URI of the scheme of a resolver of gRPC, `dns`, `ipv4`, `ipv6`, `unix`,
`unix-abstract`, `xds` or `google-c2p`, of any case, else `dns:///` and the endpoint; of its
authority and its path percent-decoded, of their bytes (`loc%61lhost:15002` is `localhost:15002`,
and `[fe80::1%25eth0]:15002` of the zone `eth0`), and of its query and its fragment dropped, of the
characters that gRPC takes there (`127.0.0.1:15002?x=1#y` is `127.0.0.1:15002`). The name of `dns:`
is the path, a host and a port as gRPC's resolver splits them, of the port 443 without one, 80 and
443 of `http` and `https`, of a sign and of spaces around it, and of 16 bits (`localhost:99999` is
the port 34463), resolved as c-ares resolves it, of the name of each error of the resolver (`errors
resolving localhost:abc: [field:hostname lookup error:Failed to parse port in name: localhost:abc]`,
`Unparsable name`, `host must not be empty in name`, `Misformatted domain name`); of an authority
that is not an IP address and a port, each call fails of gRPC's `INTERNAL` (`Illegal status code
from resolver; original status: INVALID_ARGUMENT: Cannot parse authority: ...`). Of an authority of
an IP address and a port (`dns://127.0.0.1:53/spark.example.com:15002`), the name is asked of that
DNS server alone, as gRPC's resolver has c-ares ask it: for each family at once, of IPv6 too where
the system has an IPv6 loopback, the hosts file and DNS are looked up in the order of
`/etc/nsswitch.conf`, the hosts file of the family of an entry, `localhost` and the names under it
are the loopback, which DNS is not asked for, and DNS is asked for the `A` or the `AAAA` records of
the name, over UDP, of recursion desired and no EDNS, and again over TCP of a truncated answer, of
the search domains and `ndots` of `/etc/resolv.conf` (on macOS, as c-ares reads them there, of the
resolvers of no domain, reachable and of a name server of the system's configuration of DNS, their
search lists joined, and `/etc/resolv.conf` only of no configuration to read; on Windows, of the
search list of the registry, the `SearchList` and the `Domain` of TCP/IP, the `SearchList` and the
`PrimaryDNSSuffix` of the policies of its DNS client, and the `SearchList`, the `Domain` and the
`DhcpDomain` of each interface), `LOCALDOMAIN` and `RES_OPTIONS` (the name first of `ndots` dots at
least, alone of a last `.`), until an answer has an address: of `NXDOMAIN` or of no data, the next
name is asked, of `SERVFAIL`, `REFUSED` or `NOTIMP` the query is sent again, and of its attempts (3,
of `attempts`) the timeout of the first is that of `timeout` (2 s, 5 s at most) and that of each
next one twice the previous one but a random part of a half at most. The addresses of both families
are sorted as gRPC's resolver sorts them (RFC 6724) and tried of each family in turn, as gRPC's
`pick_first` interleaves them; of none, each call fails of c-ares's error of the family that ended
last (`errors resolving spark.example.com:15002: [field:hostname lookup error:address lookup failed
for spark.example.com:15002: Domain name not found]`, `DNS server returned answer with no data`,
`DNS server returned general failure`, `DNS server refused query`, `Timeout while contacting DNS
servers`, `Could not contact DNS servers`), and a resolution that runs for 120 s, gRPC's timeout of
it, or whose TCP connection fails to connect, is cancelled (`DNS query cancelled`). The addresses of
`ipv4:` and `ipv6:`, separated by `,`, each an IP address of the family, of a zone of IPv6, and a
port of 0 to 65535, are tried in turn as those of a host are, and the Unix sockets of `unix:` and
`unix-abstract:`, of their paths or of their abstract names, of the bytes that their paths decode to
(`unix:/tmp/%FF.sock`); of none, each call fails of `UNAVAILABLE` (`empty address list`). A target
of `google-c2p` is its name of DNS, as gRPC's resolver of it on a machine that is not of Google's
cloud, and each call of a target of `xds` fails of `UNAVAILABLE`, of no bootstrap of xDS in the
environment of grpcio's text (`Failed to create XdsClient: Environment variables GRPC_XDS_BOOTSTRAP
or GRPC_XDS_BOOTSTRAP_CONFIG not defined`). Of each target, the channel takes gRPC's names, and not
the host of its connection string, which stands for the target only: gRPC's default authority of the
target, its path of no first `/`, percent-encoded as an authority (`127.0.0.1:1,127.0.0.1:2`,
`tmp%2Fa.sock`), is the `:authority` of the calls where HTTP can carry it, and of its host up to a
`%` the name that TLS verifies (`::1` of `[::1%25lo]`, `tmp` of `tmp%2Fa.sock`), as gRPC checks it
twice. Once the handshake is done, as its `HostNameCertificateVerifier`: a name of DNS of the
certificate matches it of either case and of a wildcard of its first label alone (`*.localhost` of
`xyz.localhost`, of `x*y.localhost` too), an IP address of the certificate as its text (`::1`, which
`0:0:0:0:0:0:0:1` is not), and of a certificate of neither, its common name; of none, each attempt
fails of gRPC's `Custom verification check failed with error: UNAUTHENTICATED: Hostname Verification
Check failed.`, and nothing is sent. Then of the host of each call, as its `SslCheckCallHost`, of a
wildcard of a rest of two labels at least (`*.spark.example`): of a certificate that is of the name
but not of the host of the calls (`*.localhost` of `xyz.localhost`), each call fails of
`UNAUTHENTICATED` (`call host does not match SSL server name`), which is not retried. Whether a
token goes in plaintext is of the host of the authority (`localhost`) too. The proxy of the
environment is chosen of the host of the path, of any target but one of `unix:`, which gRPC does not
proxy, and its `CONNECT` names the `host:port` of the path, of the port 443 without one
(`[127.0.0.1:1,127.0.0.1:2]:443`, `name:443` of `unix-abstract:name`), up to a NUL, as gRPC's
argument of it is a C string (`localhost` of `localhost%00:15002`). A Unix socket is not of gRPC's
local credentials of TCP, so a channel of them fails each call to it of `Endpoint is neither UDS or
TCP loopback address.`; a socket of no file fails of gRPC's text of it (`FAILED_PRECONDITION:
unix:/tmp/a.sock: connect failed: addr: unix:/tmp/a.sock error: No such file or directory`). A
target that gRPC does not take fails each call of its `UNKNOWN` (`Failed to create channel to
'127.0.0.1:15002?':the target uri is not valid: dns:///127.0.0.1:15002?`), which PySpark's retry
policies do not retry, as they retry `UNAVAILABLE`. The connection string of the channel takes the
host and the port of the target, and nothing else of the endpoint.

As grpcio's, a channel connects to the addresses of its host, or of its proxy, as the system's
resolver gives them, within the connect deadline, but those of `localhost` and of the names under
it, the loopback of both families, as c-ares, grpcio's resolver, gives them (`::1`, then
`127.0.0.1`): to the first, then to each next one 250 ms after the previous one, or at once once the
previous one failed, while none is done, each with a connect deadline of 20 s of its own, which its
TCP connection, the tunnel of the proxy, the TLS handshake and the HTTP/2 settings that the server
sends first share. The first to complete is the connection, and the others are dropped. The
connection is the process's, of one address, as a subchannel of gRPC's global subchannel pool,
which the sessions of the same proxy, TLS and keepalive, and of no token, whose channels resolve the
target to that address, share: the channel of each session resolves the name of its target of its
own and takes at once a connection of its addresses that another session made, the first of them in
their order, as grpcio's `pick_first` over the subchannels of the pool, else connects to its
addresses in turn, so a session of `newSession()`, `cloneSession()` or the `clone` of a client, and
any other session of the same connection string, uses the connection of its session while its name
resolves to its address, and connects to the addresses that its resolution gives otherwise, while
each reads the variables of the proxy again, as grpcio's channel of a new client does; a session of
a token has a connection of its own, also of `newSession()` and `cloneSession()`, as grpcio's
channels of the credentials of a token, of each session, share none. The window of a connection is
the largest of HTTP/2, so that the
results that a session holds open and does not read on, as of `toLocalIterator()`, do not hold the
calls of the other sessions of the connection, nor its other calls, as gRPC's connections, whose
windows grow with those of their streams; the window of each stream, of 2 MiB, bounds what it holds
unread. A child of `os.fork()` makes the threads, the runtime and the channels of the client again,
as grpcio's fork support does, so that its sessions, and those of its parent, run queries there, and
leaves those of its parent to the parent: a session of the parent that the child frees or stops
leaves the connection of its channel to the parent, which reads it on, and an execution of the
parent that the child drops is not released. The child closes the sockets of the connections of its
parent and the epoll instances of its parent's runtime as the fork returns, as grpcio's fork support
closes the descriptors of the event engine of a parent, so that a connection of the parent ends on
the server with the parent; the eventfd of its parent's runtime stays, an eventfd of the child's
own, which an object of the parent writes to when it wakes a task of that runtime, one descriptor of
each generation (see `dev/parity/known-differences.md`). The client writes the lines of gRPC's core
to stderr without the lock of Rust's stderr, and the state that its threads keep is of each
generation or read without a lock, so that a lock that a thread of the parent held at the fork does
not hold the child. A fork while another thread makes a call can leave the child waiting, as
grpcio's fork support asks that no other thread makes a call then, and so can a fork while the
client makes a call in the background (a release of an execution, the removal of a cached relation
or of a model, the events of the streaming query listeners), which holds the lock of the hooks, the
user context extensions, the plan compression or the last execution of its session for the moment
that it reads them, and so can a fork while a thread of the client releases a Python object without
the GIL, as of a retry policy, whose release PyO3 defers under a lock of its own, which the next
call of the child into the client waits for. The tags and the user context extensions of the other
threads of the parent stay in the child, which never forgets them, as those threads end in the
parent only: a leak bounded by the threads of the parent. A message over the size limit of a call,
128 MiB, as PySpark's channel sets `grpc.max_receive_message_length` and
`grpc.max_send_message_length`, fails the call with grpcio's `RESOURCE_EXHAUSTED`, of the details of
gRPC's core: `Stream removed (CLIENT: Received message larger than max (<length> vs. <limit>))` of a
response, and `CLIENT: Sent message larger than max (<length> vs. <limit>)` of a request, which is
not sent. So does the metadata of a response, its headers or its trailers, past gRPC's limits of the
metadata that a call receives, of HPACK's sizes of its entries: at 16 KiB
(`grpc.absolute_max_metadata_size`), `Stream removed (received metadata size exceeds hard limit
(value length <length> vs. 16384))` of an entry too large alone, or `(<size> vs. 16384);
<entries>`, at once, also of metadata far larger, where the drop-in failed with `INTERNAL`
(`transport error`) or retried an `UNAVAILABLE` of the end of its connection; and past 8 KiB
(`grpc.max_metadata_size`), at random, the more often the larger it is, of `soft limit`, of the
entries as gRPC's core lists them. The connection advertises the hard limit as the size of a header
list that it takes, as grpcio's channel does, so that Spark's server, of grpc-java, resets the
stream of a call whose metadata would reach it, as an error whose message has a few hundred CJK
characters gives, of grpcio's `INTERNAL`, `Stream removed (RST_STREAM (Received RST_STREAM with error
code 1))`, where the drop-in raised the hard limit of the metadata that it received; a stream that a
server resets fails so of the code of the reason, as gRPC's core gives it. Past the soft limit, the
call still has the details of the server's error, as grpcio's call does, and its failure is
PySpark's `ValueError` of grpc_status, `Code in Status proto (StatusCode.INTERNAL) doesn't match
status code (StatusCode.RESOURCE_EXHAUSTED)`, where it was the `SparkConnectGrpcException` of
`RESOURCE_EXHAUSTED`: as PySpark reads the `google.rpc.Status` of each failure with
`rpc_status.from_call`, the failure of a status that its details do not match, of their code or of
their message, is that `ValueError`, raised as grpcio's error of the call is handled (its
`__context__`), of no log of the error where the `DefaultPolicy` or the iterator of a reattachable
execution reads it, and after `GRPC Error received` where `_handle_rpc_error` does (see
`dev/parity/known-differences.md`), of no context where the `DefaultPolicy` reads it before a
retry, as PySpark's `Retrying` does between its attempts. So is the failure of a
`grpc-status-details-bin` that gRPC's core does not decode as base64: as gRPC's core, the client
fails the call `INTERNAL` of the details `Stream removed (Error parsing 'grpc-status-details-bin'
metadata: illegal base64 encoding)`, whatever the code of the server, and gives it the status
details empty, a status of the code `OK` (`Code in Status proto (StatusCode.OK) doesn't match
status code (StatusCode.INTERNAL)`). Of details of base64 that upb's parser, PySpark's protobuf,
does not read as a `google.rpc.Status`, as of one nested past its limit (see above), of a string
that is not UTF-8 (`String field had bad UTF-8`) or of a tag of more than five bytes, the failure is
protobuf's `DecodeError`, as PySpark's `rpc_status.from_call` raises it, while a field that the type
knows of another wire type is read as one that it does not know, as upb reads it; and so is the
failure of a status whose first `ErrorInfo`, which `_handle_rpc_error` unpacks, or first
`RetryInfo`, which the `DefaultPolicy` unpacks, upb does not read (`Error parsing message with type
'google.rpc.ErrorInfo': Wire format was corrupt`), where PySpark unpacks it. The first detail of a
type is the first that protobuf's `Any.Is` tells of it, of a type URL of a `/`, of any host, and the
metadata of an `ErrorInfo` holds the entries that upb sets in its map, not those of a field that the
entry type does not know or of a key or a value of another wire type. As PySpark's
`convert_exception`, the `classes` and the `messageParameters` of the metadata are parsed with
Python's `json.loads`, of the values of the JSON as they are, and its error is the failure, as the
`JSONDecodeError` of JSON that does not parse, of grpcio's error as its context (or as that of the
`StopIteration` of `json` that it suppresses); the `errorClass` of the metadata, also empty, is the
condition. The delay of a `RetryInfo` is PySpark's, the
`ToMilliseconds()` of its `retry_delay`, signed and unbounded and 0 of one of none, which the log
gives as it is, and so is the wait before a retry, of Python's numbers, which a negative delay
never lengthens. A negative wait, and one of decades, are slept of Python's `time.sleep`, as
PySpark's `Retrying` sleeps them, which raises at once, of no context, of one that it cannot hold:
its `ValueError` of a negative one (`sleep length must be non-negative`), its `OverflowError` of one
past its nanoseconds of 64 bits (`timestamp out of range for platform time_t` from Python 3.11 on),
or the system's `OSError` of one that the clock cannot hold. A signal that comes while it sleeps
ends the wait as one that comes during any other wait of the call: its KeyboardInterrupt interrupts
the execution in flight. A request sent in the background sleeps it on a thread of Python of its
own, and fails of its error, which is logged as PySpark's sender logs it (`ReleaseExecute failed
with exception: ...`). A policy that
PySpark's `Retrying` decides, as a subclass of `DefaultPolicy`, is asked of a failure as grpcio's
error of the call is handled, as PySpark's `AttemptManager` asks it, so that what its `can_retry`
raises has that error as its context. A channel that cannot
connect fails its calls with
`UNAVAILABLE`, which the retry policy retries, of the last error of its addresses: of a plaintext
server that accepts the connection and never answers, grpcio's `failed to connect to all addresses;
last error: UNKNOWN: ipv4:127.0.0.1:15002: connection attempt timed out before receiving SETTINGS
frame`. Once it failed, it fails the later calls at once, of that error, while each of its addresses
connects again in the background with gRPC's backoff, 1 s and then 1.6 times longer each time up to
120 s, with a jitter of 20 %, each attempt with the longer of its backoff and 20 s as its deadline,
once for each address of the process, whatever the sessions that wait for it, until a connection
succeeds, which the next call takes. The channel of each session, of a new session or a clone too,
resolves the name of its target of its own before its first call, as the resolver of each channel of
grpcio does, whose calls fail of the error of it, at once, while it tries it again in the background
a backoff after its end, as gRPC's resolver does, and connects once it resolves; and it resolves the
name again of its own, in the background, as its connection ends, or as a GOAWAY of the server ends
it for new calls, which then connect again while the calls in flight go on, and each time its
addresses failed again, no sooner than 30 s after the start of its last resolution, as gRPC's
resolver resolves a name again no sooner (`grpc.dns_min_time_between_resolutions_ms`), of whose
addresses it connects meanwhile: a resolution that fails after one that succeeded leaves it the
addresses of that one, and a resolution that comes while it is connected makes it take at once the
first connection of the new addresses that the process has, its own too, or, of none, close its own
once no other session uses it and connect to the new addresses at its next call, as gRPC's
`pick_first` drops its connection and goes idle then. Once no call came for 30 min to a channel that failed, or once its session is stopped, it stops
connecting, as grpcio's channel goes idle, or shuts down, and the next call connects. A connection
that ends is made again by the next call. The Python client logs the
failures of the connections that gRPC's core logs, on stderr (see [Errors on
stderr](configuration.md#errors-on-stderr)), and its `CONNECT` request of a proxy has grpcio's
`User-Agent: grpc-httpcli/0.0`. What stays different, such as the keepalive pings, which are
hyper's, is in `dev/parity/known-differences.md`.

The user agent, `<user_agent> spark/4.3.0 os/<os> python/<version>` of the version of Python, as
PySpark's `ChannelBuilder.userAgent` makes it, is sent as PySpark sends it: as the client type of
the requests that PySpark builds with it, and not as a header. The `user-agent` header is the client
library's own, `tonic/<version>` here and grpcio's in PySpark, such as `grpc-python/1.85.1
grpc-c/58.0.0 (linux; chttp2)`. A `user-agent` parameter of the connection string is checked as
metadata and not sent, as grpcio replaces it with its own. So a user agent may hold any character, a
control character included. As PySpark's `userAgent`, which its requests read when they are built, a
user agent longer than 2048 characters after URL escaping, or a `SPARK_CONNECT_USER_AGENT` that is
not UTF-8, is the error of each request that carries it, which is not sent: `SparkConnectException`,
or the `UnicodeEncodeError` of `urllib.parse.quote`. The session is made all the same, its `stop()`
logs the error as that of its release, and the requests that PySpark builds without the user agent,
of the artifacts and of `CloneSession`, are sent. As `urllib.parse.urlsplit` does, a tab, CR or LF
anywhere in the connection string is removed. `SPARK_CONNECT_USER_AGENT` is read as each call of the
client starts, so that a change of `os.environ` applies to the next call, as PySpark's requests read
the variable when each is built, and a request that the transport builds during a call, such as a
reattachment or a release in the background, has the user agent of the last call that started.

A gRPC failure whose status carries no error of the server, such as of an unreachable server or of a
proxy, is the `SparkConnectGrpcException` of its code whose message is the text of grpcio's error of
the call, as PySpark's `_handle_rpc_error` raises it: `<_InactiveRpcError of RPC that terminated
with:`, or `<_MultiThreadedRendezvous` for `ExecutePlan` and `ReattachExecute`, and the lines of its
code (`status = StatusCode.UNAVAILABLE`), its details and its `debug_error_string`, as grpcio 1.85
gives them. The message of a call past its deadline begins with that text, as PySpark's `f"{error}:
RPC deadline exceeded. ..."`, and that of a status of a `google.rpc.Status` without an `ErrorInfo`
is the message of the status. The details are the message of a status that the server sent, and the
`debug_error_string` is the name of the code and the details, as in grpcio. Those of a failure that
the client detects itself are those of gRPC's C core: of a channel that cannot connect, `failed to
connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:1: Failed to connect to remote host:
Connection refused`, of the last error of its addresses, such as a TLS handshake of BoringSSL's
error or a proxy that refuses the tunnel (`HTTP proxy returned response code 403`), of a name that
does not resolve, `errors resolving ...`, and of a call that the end of its connection fails,
`Stream removed (Socket closed)`, or `Stream removed (sendmsg: Broken pipe (32))` of a write of it
that fails. The errors of `copyFromLocalToFs` and `cloneSession`, which
PySpark leaves to grpc, are grpcio's `RpcError`, as there (see
[SparkSession.client](#sparksessionclient)).

As PySpark's `Retrying`, a call whose retries of a failure that a policy retries are spent warns
`[RETRIES_EXCEEDED] The maximum number of retries has been exceeded.` before it raises the failure:
a `UserWarning` of the location of PySpark's `warnings.warn` in
`pyspark.sql.connect.client.retries`, and of that module, which its filters and the actions of a
location take as PySpark's, and which a filter of `"error"` raises in place of the failure. So do
the lookup of the plan compression of the first plan of a session, whose failure is caught, the read
of the conf of the server's stack trace as an error of the server is raised, which is retried as
PySpark's, and the release of the session of `stop()`. A policy retries the failures that its
`can_retry` takes, as PySpark's `Retrying` asks it, also a subclass of `RetryPolicy` that is not a
`DefaultPolicy`, and warns when its retries are spent; policies none of which takes a failure, such
as PySpark's own `RetryPolicy`, retry nothing, and do not warn. The requests that the client sends
in the background, which no
caller waits for, warn too, and their failures are logged as PySpark's senders log them
(`ReleaseExecute failed with exception: ...`), from the delivery thread of the drop-in (see
`dev/parity/known-differences.md`). `stop()` logs the failure of the release of its session to the
logger of Spark Connect (`pyspark.sql.connect.logging`), which is disabled unless
`SPARK_CONNECT_LOG_LEVEL` is set, as PySpark's: `session.stop(): Session could not be released.
Error: $<error>`.

With `SPARK_CONNECT_LOG_LEVEL`, or a level of the logger of `pyspark.sql.connect.logging`, the
client logs to the logger of Spark Connect what PySpark's client logs, of the same levels, messages
and order, through the logger's methods, as PySpark's: the deadlines of a new client (`Spark Connect
RPC deadlines: ...`); the settings of the plan compression that the first plan of a session reads
(`Plan compression threshold: ...`, or the error of a failed read as the argument of a text of no
placeholder for it, as PySpark's) and the compression of a plan (`Plan compression: original_size=
...`, of the frame of python-zstandard's `compress`, and `Plan compression not effective. Using
original plan.`); the plan of each execution and analysis, as the method of PySpark's client that it
stands for logs it (`Executing plan ...`, `Execute command for command ...`, `Execute command as
iterator for command ...`, `Schema for plan: ...`, `Explain (mode=...) for plan ...`, `Execute` and
`ExecuteAndFetch`), of the text of `SparkConnectClient._proto_to_string` of the drop-in's protobuf
runtime; the request of each execution and each response as the execution reads it
(`ExecuteAndFetchAsIterator. Request: ...` and `Response received: ...`, `Received the schema.`,
`Received the SQL command result.`, `Received metric batch.`, `Received observed metric batch.`,
`Received arrow batch ...` and `Assembling arrow batch ...`); the deadline of a stream that the
execution reattaches (`Deadline exceeded on stream for operation ...`) and the interrupt of an
execution by a `KeyboardInterrupt` (`Interrupt request received for operation=...`), after which
nothing more of the execution is logged of its responses, as PySpark's reads nothing more of them
(PySpark's release of the execution, which it sends as it interrupts it, may fail on the server,
which it logs, `ReleaseExecute failed with exception: ...`, where the drop-in releases it once the
server ended it); each retry and the end of the retries of a call (`Given up on retrying. error:
...`); each gRPC failure that it converts (`GRPC Error received`, of grpcio's `RpcError`,
`Received ErrorInfo: ...` of an error of the server, and `Disabling plan compression for the
session due to CONNECT_INVALID_PLAN.CANNOT_PARSE error.`); and of `SparkSession.client`, its
resources and resource profiles (`Fetching the resources`, `Creating the ResourceProfile`) and a
delete or a cleanup of the ML cache that it skips while one is in flight on its thread (`Skipping
re-entrant ML cache ...`). The exception of a handler or a filter of the logger at a record fails
the call there, as PySpark's: before the request of an execution when the record precedes it, and
of a `KeyboardInterrupt` at a response, of `toLocalIterator` too, with the interrupt of the
execution, as PySpark's `except KeyboardInterrupt`, but at a record of the conversion of the failure
of the execution (`GRPC Error received`, `Received ErrorInfo: ...`), which PySpark's `except
Exception` writes, of no interrupt; the exception of a record that the client writes as it handles
a failure, of the conversion of a failure or of the failed lookup of the plan compression, has the
failure as its context, as PySpark's. A call of the client that fails while the client converts
another failure on the thread, as a call of a handler of such a record, raises its error as grpc
gives it, grpcio's `RpcError` of the call, of no records, no request of the details of the error or
of its stack-trace settings, and no `_operation_id`, as PySpark's `_handle_error` raises the error
of a request as it is while it handles another (`inside_error_handling`); the failure of a call of
the client of another session is converted. A call of a closed client, as of a stopped session,
writes the records that come before the request, and then raises NO_ACTIVE_SESSION, as PySpark's
client, which reads its `_stub`, of that error, only to send. The level of the logger is read once
per call, as the call starts: a record of a level that the logger does not write is not made, and
nothing is formatted for it, so a level that a progress handler enables during a call applies from
the next call on, while one that it disables applies at once, as the logger's methods filter the
records. With
`SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK`, the requests of the executions, the analyses and the
configurations carry the call stack of the user's code, as PySpark's (see
`dev/parity/known-differences.md` for both).

For TLS roots, including `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`, see
[TLS Root Certificates](configuration.md#tls-root-certificates).

## Native NaiveBayes

The Rust `ml::NaiveBayes` and `ml::NaiveBayesModel` train and predict through Spark
Connect's cached-model protocol. The Rust-backed Python API exports `NaiveBayes`
from `pyspark.ml.connect` and `pyspark.ml.connect.classification`; `fit` returns the
typed `NaiveBayesModel`, whose `transform` produces DataFrame predictions.

All nine PySpark constructor parameters and named setters are supported:
`featuresCol`, `labelCol`, `predictionCol`, `probabilityCol`, `rawPredictionCol`,
`smoothing`, `modelType`, `thresholds` and `weightCol`. Rust exposes corresponding
builder setters and getters. Unset parameters are omitted so the server uses its
defaults. The Python constructor is keyword-only and converts values with
PySpark's parameter converters.

All four model types are covered by live tests: `multinomial`, `bernoulli`,
`complement` and `gaussian`. Multinomial/complement inputs must be nonnegative;
Bernoulli inputs must be binary; Gaussian inputs may contain continuous negative
values. Tests also cover multiclass dense/sparse inputs, probability/raw score
columns, smoothing and instance weights, threshold decisions, suppressed vector
outputs, independent fits, session isolation and lazy model lifetime. A golden
fixture captures all nine parameters for each model type from official PySpark.

The Rust `NaiveBayesModel` has PySpark 4.3.0's model members: `pi`, `theta` and `sigma` (a
0 by 0 matrix for multinomial, complement and Bernoulli models), `numClasses`, `numFeatures`,
`predict`, `predictRaw` and `predictProbability` of one vector, and `NaiveBayesModel::load`.
`NaiveBayes::load` reads a saved estimator with one ML Read, as PySpark's `NaiveBayes.load`
does. The estimator's `fit` returns this typed model. PySpark 4.3.0 gives the model no training
summary. The Rust param accessors are generated from PySpark's param table: `features_col` and
`set_features_col` replace the former `feature_col` and `set_feature_col`.

In Python, `NaiveBayes` and its `NaiveBayesModel` follow the full API of `pyspark.ml`
(see "Native ML classes of the full pyspark.ml API"): the Params API, `pi`, `theta`, `sigma`,
`numClasses`, `numFeatures`, `predict`, `predictRaw` and `predictProbability` of a vector,
and save and load. A NaiveBayes model has no training summary, as in PySpark. The vendored
`pyspark.ml.classification` API continues to use its existing implementation.

## Native BisectingKMeans

The Rust `ml::BisectingKMeans` and `ml::BisectingKMeansModel` support training and
DataFrame cluster assignment through Spark Connect's cached-model protocol. Python
exports the native estimator from `pyspark.ml.connect` and
`pyspark.ml.connect.clustering`; `fit` returns its `BisectingKMeansModel`.

All eight constructor parameters and named setters are supported: `featuresCol`,
`predictionCol`, `maxIter`, `seed`, `k`, `minDivisibleClusterSize`, `distanceMeasure`
and `weightCol`. Rust also provides parameter getters. The keyword-only Python
constructor matches official PySpark's order/defaults and parameter conversions.
Unset parameters are omitted so the server uses its defaults, including `k=4`.

Live tests cover hierarchical splitting, absolute/fractional minimum divisible
cluster sizes (which may produce fewer than `k` leaves), instance weights,
Euclidean/cosine distance, sparse/dense inputs, seeded reproducibility, independent
fits, session isolation and lazy model lifetime. Golden fixtures compare all eight
parameters at six signed seed/integer-width boundaries with official PySpark.

The Rust model's members, summaries and persistence are in "Native clustering models (Rust API)". In
Python, `BisectingKMeans`, its `BisectingKMeansModel` and `BisectingKMeansSummary` follow the full
API of `pyspark.ml` (see "Native ML classes of the full pyspark.ml API"): the Params API,
`clusterCenters()`, `computeCost(dataset)`, `numFeatures`, `predict` of a vector, the training
summary (`k`, `clusterSizes`, `trainingCost`, `numIter`, `predictions` and `cluster`, of the
transform of the training dataset, as PySpark's), and save and load. The vendored
`pyspark.ml.clustering` API is unchanged.

## Native GeneralizedLinearRegression

`spark_connect::ml::GeneralizedLinearRegression` fits Spark's JVM estimator and
returns a cached `GeneralizedLinearRegressionModel`. Python exposes it from
`pyspark.ml.connect` and `pyspark.ml.connect.regression`; `fit` returns its
`GeneralizedLinearRegressionModel`.

All 16 PySpark constructor parameters and named setters are available:
`labelCol`, `featuresCol`, `predictionCol`, `family`, `link`, `fitIntercept`,
`maxIter`, `tol`, `regParam`, `weightCol`, `solver`, `linkPredictionCol`,
`variancePower`, `linkPower`, `offsetCol` and `aggregationDepth`. Rust provides
builder setters and getters for each parameter. Unset optional parameters stay
unset: Spark chooses the family default link, or the Tweedie link power from
`variancePower`.

Fitting accepts ML vector features and numeric labels. Transform produces
predictions and, when requested, the linear predictor in `linkPredictionCol`.
Instance weights, offsets and L2 regularization are handled by Spark. The same
numerical tests run against official PySpark and the native Python client,
covering all five families (Gaussian, binomial, Poisson, Gamma and Tweedie),
default and explicit links, dense/sparse features, weights, offsets, independent
fitted models, lazy model ownership and rejected inputs. Official PySpark golden
fixtures verify the parameter encoding, and live Rust tests exercise fitting
and prediction directly.

The Rust model's members and persistence are in "Native regression models (Rust API)". In Python,
`GeneralizedLinearRegression`, its `GeneralizedLinearRegressionModel`,
`GeneralizedLinearRegressionSummary` and `GeneralizedLinearRegressionTrainingSummary` follow the
full API of `pyspark.ml` (see "Native ML classes of the full pyspark.ml API"): the Params API,
`coefficients`, `intercept`, `numFeatures`, `predict` of a vector, the training summary (of the
training dataset, as PySpark's, with `residuals(residualsType)` and the statistics of the
coefficients), `evaluate`, and save and load. As PySpark's over Spark Connect, `residuals` sends its
positional arguments only: `residuals(residualsType="pearson")` gives the deviance residuals. As
with PySpark, the `repr` of a model whose `link` is not set fails on the server.

## Native GaussianMixture

Rust `ml::GaussianMixture` and `ml::GaussianMixtureModel` train Gaussian mixture
models and produce cluster assignments and component probabilities through Spark
Connect's cached-model protocol. Python exports the native estimator from
`pyspark.ml.connect` and `pyspark.ml.connect.clustering`; `fit` returns its `GaussianMixtureModel`.

All nine constructor parameters and named setters are supported: `featuresCol`,
`predictionCol`, `k`, `probabilityCol`, `tol`, `maxIter`, `seed`, `aggregationDepth`
and `weightCol`. Rust also provides corresponding getters. The keyword-only
Python constructor matches official PySpark's order/defaults and conversions.
Unset values are omitted so the server uses its defaults.

Live tests cover one- and two-dimensional mixtures, two/three components,
normalized probabilities and assignments, dense/sparse vectors, weights, seeded
reproducibility, independent fits, disabled probability output, server validation,
session isolation and lazy model lifetime. Golden fixtures compare all nine
parameters at six signed seed/integer-width boundaries with official PySpark.

The Rust model's members, summaries and persistence are in "Native clustering models (Rust API)". In
Python, `GaussianMixture`, its `GaussianMixtureModel` and `GaussianMixtureSummary` follow the full
API of `pyspark.ml` (see "Native ML classes of the full pyspark.ml API"): the Params API, `weights`,
`gaussiansDF`, `gaussians` (a `MultivariateGaussian` of each collected row of `gaussiansDF`, as
PySpark builds them), `numFeatures`, `predict` and `predictProbability` of a vector, the training
summary (`probability`, `probabilityCol`, `logLikelihood` and the members of a clustering summary,
of the transform of the training dataset, as PySpark's), and save and load. The vendored
`pyspark.ml.clustering` API is unchanged.

## Native FMClassifier

`spark_connect::ml::FMClassifier` and `pyspark.ml.connect.classification.FMClassifier` fit Spark's
binary factorization machine classifier. The 17 parameters of PySpark's constructor have Rust
builder setters and getters and named Python setters; `weightCol`, a param of the class that
PySpark's constructor does not take and that PySpark has no setter of, has a Rust getter only.
Fitted models produce prediction, probability and raw-prediction columns, with configurable
thresholds. Both the `gd` and `adamW` solvers run on the Spark server.

The Rust `FMClassificationModel` has `factors`, `linear`, `intercept`, `predict`, `predictRaw`
and `predictProbability` of one vector, `numClasses`, `numFeatures`, and `load`.
`FMClassifier::load` reads a saved estimator with one ML Read, as PySpark's `FMClassifier.load`
does. Its `summary()` is the binary training summary of a fitted model, and `has_summary()` says
whether one exists;
`summary()` fails for a loaded model, as PySpark does. `evaluate(&df)` returns the binary summary
of a DataFrame, with the ROC, PR and threshold DataFrames.

In Python, `FMClassifier`, its `FMClassificationModel` and their summaries follow the full
API of `pyspark.ml` (see "Native ML classes of the full pyspark.ml API"): the Params API,
`factors`, `linear`, `intercept`, `numClasses`, `numFeatures`, `predict`, `predictRaw` and
`predictProbability` of a vector, `summary()` (the training summary: `objectiveHistory`,
`totalIterations`, `roc`, `pr` and the other curves), `evaluate`, and save and load.

## Native MultilayerPerceptronClassifier

`spark_connect::ml::MultilayerPerceptronClassifier` and its native Python binding support the 14
params of PySpark's class: the 13 of its keyword-only constructor, and `thresholds`, which the
constructor does not take, as PySpark's does not, and which `setThresholds` (`set_thresholds` in
Rust) sets. Configure `layers` before fitting; input and output sizes must match the feature
dimension and number of classes. `initialWeights` accepts dense or sparse ML vectors: Rust uses
`MlVector` (a `Vec<f64>` or array also converts to a dense vector), and Python accepts the values
supported by PySpark's `TypeConverters.toVector`. Layer arrays and vector parameters are encoded in
Rust and checked against fixtures captured with official PySpark.

The fitted model produces multiclass prediction, probability and raw-prediction
columns. Both `l-bfgs` and `gd` are supported. As in PySpark, fitting an empty
DataFrame can return a model; the native client preserves that server behavior.

The Rust `MultilayerPerceptronClassificationModel` has `weights`, `predict`, `predictRaw` and
`predictProbability` of one vector, `numClasses`, `numFeatures`, and `load`.
`MultilayerPerceptronClassifier::load` reads a saved estimator with one ML Read, as PySpark's
`MultilayerPerceptronClassifier.load` does. Its `summary()` is the multiclass training summary of
a fitted model, and `has_summary()` says whether one exists.
`summary()` fails for a loaded model, as PySpark does. `evaluate(&df)` returns the summary of
a DataFrame. The param accessors are generated from PySpark's param table.

In Python, `MultilayerPerceptronClassifier`, its `MultilayerPerceptronClassificationModel`
and their summaries follow the full API of `pyspark.ml` (see "Native ML classes of the full
pyspark.ml API"): the Params API, `weights`, `numClasses`, `numFeatures`, `predict`,
`predictRaw` and `predictProbability` of a vector, `summary()` (the training summary:
`objectiveHistory` and `totalIterations`), `evaluate`, and save and load.

## Native OneVsRest

`spark_connect::ml::OneVsRest` and `pyspark.ml.connect.classification.OneVsRest`
implement the multiclass reduction in Rust. Supply a native binary classifier,
for example `OneVsRest::new().set_classifier(LinearSVC::new())` in Rust or
`OneVsRest(classifier=LinearSVC())` in Python. The `Classifier` enum accepts all
eight native binary classifiers: LogisticRegression, LinearSVC,
DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, NaiveBayes,
FMClassifier and MultilayerPerceptronClassifier. Perceptron base classifiers need
a two-unit output layer.

In Python, the classifier can be any other object, as in PySpark. The native fit is that of a
classifier of exactly a native class. Any other classifier, such as PySpark's own classes of
`pyspark.ml.classification`, or a Python subclass of them or of a native class, is fit by PySpark's
own `OneVsRest._fit` over the estimator's params, which runs the classifier's own `fit` and `_fit`
for each class (`classifier.fit(trainingDataset, paramMap)`), overrides of a subclass too. Its
model is PySpark's `OneVsRestModel` of the classifier's own models. A None classifier is fit there
too, and fails at its first use. So is a dataset that is not a Spark DataFrame: a pandas DataFrame
raises the `KeyError` of PySpark's pandas aggregate of its max label.

All seven PySpark constructor parameters have named setters. Rust exposes scalar
parameter getters and the configured classifier. Fits clone the base classifier
for each class and run at most `parallelism` fits at once. Python observes
mutations to the supplied classifier made before `fit`, while fitting leaves that
classifier unchanged. Instance weights are forwarded to each class's fit when the base
classifier is a `HasWeightCol` of PySpark, which every native classifier is but the
perceptron (FMClassifier included). For the perceptron, Python warns
`weightCol is ignored, as it is not supported by <uid> now.`, with the classifier's
uid, as PySpark does, and fits without the weights. The warning is attributed as PySpark's is, to
the line of its `warnings.warn` in `pyspark/ml/classification.py` and to the module
`pyspark.ml.classification`, whose filters match it.

A fit runs the steps of PySpark's `OneVsRest._fit`, in its order and of its requests, in Rust and
in Python, but for the reads of each class's model that PySpark's makes after its fit (`numClasses`
of a LogisticRegression model), which the Rust API makes when they are asked for:

- **Classes:** the fit reads the max label first, as
  `dataset.agg({labelCol: "max"}).head()["max(" + labelCol + ")"]`: the max of the label column as
  named, which the server resolves (a name with a dot names a field of a struct), of the column's
  own type. The classes are `int(max) + 1`, of Python's `int`, so the max of a string column is its
  greatest string (labels "10", "9" and "0" fit ten classes, and the rows of "10" are of no class),
  a fraction is truncated (a max of -0.5 fits one class), and a null (of no rows), a NaN, an
  infinity or a string that is no integer raises the `TypeError`, `ValueError` or `OverflowError`
  of `int`. Python takes PySpark's own Python values and `int`. Rust converts the max as `int` does,
  of the same messages, but of the ASCII digits of a string only, where Python's `int` takes the
  decimal digits of other scripts too.
- **Dataset:** the label, features and weight columns (`dataset.select(labelCol, featuresCol,
  weightCol)`) are cached while the classes fit when the input is not persisted, as PySpark's
  `_cache_spark_dataset(..., StorageLevel.MEMORY_AND_DISK)` caches them: persisted, and unpersisted
  after the fits and the reads of their models; or, with `SPARKML_TEMP_DFS_PATH` set, saved under
  that path, read back, and deleted after by the ML helper (`handleOverwrite`). The binary label of
  each class is `1.0` where the label column of that dataset (`multiclassLabeled[labelCol]`) is
  the class, and the training summary of each class's model is of its binary training set.
- **Pool:** then PySpark makes its list of the tasks of the classes and its pool of
  `min(parallelism, classes)` threads, so that a parallelism below one, a null one, and a max label
  below zero raise their errors after the max label is read and the dataset is cached:
  `ValueError: Number of processes must be at least 1`, or the `TypeError` of `min` of a None. This
  client fits at most 2147483647 classes: a count beyond raises a `MemoryError` in Python (and an
  error in Rust), where PySpark allocates its list of that many tasks (`[trainSingleClass] *
  numClasses`, of 8 bytes a class, 16 GiB of 2^31 classes, before any fit), and beyond 2^63 the
  `OverflowError` of that list, as PySpark's.
- **Params:** the params are read where PySpark reads them, with no check of their own: an unset
  classifier raises PySpark's `KeyError` of the param, a null `labelCol` the `AttributeError` of
  `F.col(None)`, a null `featuresCol` the `TypeError` of the select (after the max query), a null
  `predictionCol` is sent to the classifier's fits, which the server rejects, and a null
  `rawPredictionCol` fits, as PySpark's does. In Python, `parallelism` is an int of any size, which
  PySpark compares and never sends, and which saves and loads as it is.

The model aggregates the positive-class raw scores of its submodels into `rawPrediction` and
chooses the first highest score for `prediction`, ignoring NaN/null scores as PySpark does; it
produces no probability column. In Python, `transform` is PySpark's own
`OneVsRestModel._transform`, and the Rust transform is a port of it, with the same DataFrame
operations and so the same requests. Each submodel transforms the data with its own `featuresCol`
and `rawPredictionCol`, not the model's, so a model whose `featuresCol` names no column still
transforms, and a model `rawPredictionCol` that the submodels do not produce fails with
`UNRESOLVED_COLUMN` when the transform is evaluated, as PySpark's does. The scores are accumulated
in a `mbc$acc<uuid>` column. When the input is not persisted, the transform persists it at
`MEMORY_AND_DISK` while it builds the composition and unpersists it then, as PySpark's
`_cache_spark_dataset` does. With `SPARKML_TEMP_DFS_PATH` set, a copy of the input is written there
and read back, and the copy is deleted before the transform returns: as in PySpark, the returned
DataFrame then fails with `PATH_NOT_FOUND` when it is evaluated. Temporary columns and fit-created
caches are cleaned up, and transformed DataFrames retain the binary models for lazy evaluation.
Models must transform data in their fitting session.

Rust `OneVsRest::fit` returns the typed `OneVsRestModel`. Its `models()` gives each binary
submodel as its typed model, one `SubModel` variant per native classifier, and `classifier()`
gives the classifier. Single-vector prediction is on the submodels. `OneVsRestModel::new(session,
models)` constructs a model of the given models, of a new uid and no set params or classifier, as
PySpark's `OneVsRestModel(models)`, and its `set_classifier` sets the classifier.

Persistence is of PySpark's Python format, in Rust and in Python, as PySpark's `OneVsRestWriter`,
`OneVsRestModelWriter` and their readers write it: `OneVsRest::write().save(path)` and
`OneVsRest::load(session, path)`, and the same for `OneVsRestModel`. The metadata of the estimator
has its params without the classifier, which is saved at `classifier`, in the order of the maps of
the Python object that it saves, as PySpark's `extractJsonParams` writes them (of the
constructor's defaults of a OneVsRest of its constructor, of the names of a load or a copy); the
metadata of the model
also has `numClasses`, and each submodel is saved at `model_<index>` by its own writer. What each
of PySpark and this client saves, the other loads, with the same uids, params and predictions, and
the metadata files equal PySpark's apart from their timestamps and uids (of a subclass, of its class,
see "Subclasses" above). As PySpark's readers, the
loads read metadata of any class, of PySpark's Python format and of the JVM's class names, which
classic PySpark's `JavaMLWriter` (and Scala's writer) saves of a OneVsRest of a Java classifier, in
the same layout, also as a stage of a pipeline. A load of a model loads its classifier and each of
its models once, of PySpark's requests. A `CrossValidator` or a `TrainValidationSplit` over a
OneVsRest saves and loads with it, and a param map of the validator sets the params of the
classifier, as PySpark's `copy` does. In Rust, a param map of a model sets the params of its
models (`apply_param_map`), as PySpark's `OneVsRestModel.copy(extra)` copies each model with it.

In Python, the `OneVsRest` estimator has the Params API of `pyspark.ml` (its `classifier` param
included), `setClassifier`, `fitMultiple` and `clear`, and its `fit` returns the typed
`OneVsRestModel`, which has `classifier`, the other params, `transform`, and a `uid` of its own,
as PySpark's. `OneVsRestModel(models)` constructs a model of the given models, of a new uid and
the defaults of the class, with no set params and no classifier, as PySpark's constructor does,
and its `models` attribute can be assigned. Its `models` are the typed submodels, each with the
uid of the classifier, the
binary label of its class as `labelCol`, the classifier's params, and its training summary. They
are made from the fitted submodels without a request, each of the typed model class of its
classifier. A LogisticRegression submodel is a typed
`LogisticRegressionModel`, with its training summary; a DecisionTreeClassification submodel is a
typed `DecisionTreeClassificationModel`; a RandomForestClassification submodel is a typed
`RandomForestClassificationModel` with its training summary (binary, as its class's fit is); and a
GBTClassification submodel is a typed `GBTClassificationModel`, with the trees of its ensemble. The
per-class fits send the requests of PySpark's, and PySpark's client also reads the post-fit requests
of each submodel: `numClasses` of a LogisticRegression submodel, and `numClasses` and `trees` of a
RandomForestClassification one, and `trees` of a GBTClassification one. PySpark's makes the reads
of a class right after its fit, on the thread of its pool that fits it (its `try_remote_fit`), so
they come between the fits of the classes; this client's core fits the classes, in the threads of
its own pool, and then the Python client makes the reads of each class, in their order, once all
the fits are done: the requests are the same, of another order. Their order would need a Python
call of each class's fit on the threads of the core's pool, for no other difference.

As PySpark's, the `write()` of a `OneVsRest` or a `OneVsRestModel` gives its writer, whose `save`
raises the `KeyError` of the unset `classifier` param of one of no classifier (a `OneVsRestModel`
of its constructor too), and then PySpark's `ValueError` of `validateParams` of a classifier or a
submodel that is not writable (`OneVsRest write will fail because it contains ... which is not
writable.`), before any request.

## Native AFTSurvivalRegression, IsotonicRegression and FMRegressor

All eight regression estimators in PySpark 4.3.0 now have native Rust fit/transform
implementations. The three additions are available under `spark_connect::ml` and
`pyspark.ml.connect.regression`, and are re-exported from `pyspark.ml.connect`.
They fit Spark's JVM estimators and retain the returned server-cached models for
as long as a model clone or transformed DataFrame needs them.

- **AFTSurvivalRegression** fits a Weibull accelerated failure time model. It
  accepts positive survival times and a censor column (1 for observed events, 0
  for censored observations), and can produce a vector of requested quantiles.
  All 11 constructor parameters have Rust and Python setters, including
  `quantileProbabilities`, `quantilesCol`, and the aggregation/block-size settings.
- **IsotonicRegression** fits increasing or decreasing univariate regression.
  It supports scalar features or a selected vector component (`featureIndex`),
  instance weights, interpolation between fitted boundaries, and clamping beyond
  the training range. All six constructor parameters have setters.
- **FMRegressor** fits factorization machines with pairwise interactions and
  optional intercept and linear terms, using either `adamW` or `gd`. All 14
  constructor parameters have setters, including factor size, initialization,
  mini-batch fraction, regularization and seed.

Unset parameters use Spark's defaults. The Python wrappers preserve PySpark's
keyword-only constructor signatures and type conversion, and take ML vector
features, as `pyspark.ml`'s do.

The Rust models' members and persistence are in "Native regression models (Rust API)". In Python,
the three estimators and their models follow the full API of `pyspark.ml` (see "Native ML classes of
the full pyspark.ml API"): the Params API, `numFeatures` and `predict`, `coefficients`, `intercept`,
`scale` and `predictQuantiles` of a survival model, `boundaries` and `predictions` of an isotonic
model, whose `predict` takes a single feature value, `intercept`, `linear` and `factors` of an FM
model, and save and load.

Parameter goldens come from official PySpark 4.3.0. Native Rust and Python tests
exercise censoring and quantiles, weighted isotonic interpolation, both FM
solvers, validation errors and cached-model lifetimes. The Python behavior tests
also run unchanged against the official client.


## Native LDA and PowerIterationClustering

The native Rust and Python APIs now implement all five clustering algorithms in
PySpark 4.3.0: KMeans, BisectingKMeans, GaussianMixture, LDA and PowerIterationClustering.
The Rust models' members and persistence are in "Native clustering models (Rust API)".

Rust `ml::LDA` supports online and EM training and cached-model DataFrame transforms.
Python exposes `LDA` through `pyspark.ml.connect` and `pyspark.ml.connect.clustering`;
`fit` returns a `DistributedLDAModel` of the `em` optimizer and a `LocalLDAModel` of the `online`
one, as PySpark's. All 14 constructor parameters have Python setters and Rust builder
setters/getters, including document/topic concentrations, learning settings, checkpoint settings,
seed and topic-distribution output column.
Both optimizers return a vector of topic probabilities per document; empty documents
produce zero vectors. As with `pyspark.ml`, an array feature column is sent as it is, and the JVM
rejects one whose elements may be null, such as the arrays of `createDataFrame`.

`ml::PowerIterationClustering::assign_clusters` and Python `assignClusters` build
Spark Connect's `powerIterationClusteringAssignClusters` helper relation. This algorithm
has no fit/transform API or fitted model. It accepts source/destination vertex columns
and optional nonnegative edge weights, with `k`, `maxIter` and `initMode` settings, and
returns a Spark DataFrame with `id: bigint` and `cluster: int`. Parameter values are
captured when the relation is built. Omitting `weightCol` uses unweighted edges; an
explicit null remains distinct from an omitted parameter and is validated by Spark.
The lazy relation retains any cached models used by its input DataFrame.

Both APIs require Spark DataFrames and reject pandas inputs. Reference-derived golden
fixtures and live tests cover both LDA optimizers, dense/sparse inputs, graph initialization
modes, weighted/unweighted graphs, custom columns, setters, validation and model lifetime.
The Rust models' members and persistence are in "Native clustering models (Rust API)". In Python,
`LDA`, `LDAModel`, `LocalLDAModel` and `DistributedLDAModel` follow the full API of `pyspark.ml`
(see "Native ML classes of the full pyspark.ml API"): the Params API, the members of an LDA model
(`describeTopics`, `topicsMatrix`, `logLikelihood`, `logPerplexity`, `vocabSize`, `isDistributed` and
`estimatedDocConcentration`; `describeTopics` sends its positional arguments only, as PySpark's
over Spark Connect, so `describeTopics(maxTermsPerTopic=1)` describes the server's 10 terms),
`toLocal` (the same `LocalLDAModel` on every call, as PySpark's, of the constructor of its class),
`trainingLogLikelihood`, `logPrior` and `getCheckpointFiles` of a distributed model, and save and
load. `PowerIterationClustering` has its uid and the Params API, and saves and loads as PySpark's
does: as the params of the server's `PowerIterationClusteringWrapper`. `LDAModel` is the base class
of the two LDA models, as PySpark's, with the members of an LDA model but no persistence of its own:
an LDA model is an instance of it, `LDAModel.load` is an AttributeError, and a copy or a load of a
model is of the model's class. The vendored `pyspark.ml.clustering` API is unchanged.

## Native feature stages

All 38 standard `pyspark.ml.feature` estimators and transformers have native Rust execution
and Python bindings. Rust exposes them as `spark_connect::ml::<Name>`, with typed builder
setters/getters. Python exports them from `pyspark.ml.connect` and
`pyspark.ml.connect.feature`, with standard `pyspark.ml.feature` keyword-only constructor
signatures and named mutating setters. The vendored `pyspark.ml.feature` module is unchanged.

| Area | Stages |
| --- | --- |
| Text and categorical features | CountVectorizer, HashingTF, IDF, Word2Vec, Tokenizer, RegexTokenizer, NGram, StopWordsRemover, StringIndexer, IndexToString, OneHotEncoder, TargetEncoder, FeatureHasher |
| Scaling and numeric transforms | StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, Normalizer, Binarizer, Bucketizer, QuantileDiscretizer, Imputer, DCT, PolynomialExpansion, ElementwiseProduct |
| Assembly and selection | VectorAssembler, VectorIndexer, VectorSizeHint, VectorSlicer, Interaction, PCA, ChiSqSelector, UnivariateFeatureSelector, VarianceThresholdSelector, RFormula, SQLTransformer |
| Locality-sensitive hashing | BucketedRandomProjectionLSH, MinHashLSH |

The 18 transformers build lazy ML relations, and the 20 estimators fit through Spark Connect's ML
command. In Python each is a typed class with PySpark's Params API (`params`, `getParam`, the
getters and setters, `copy`, `extractParamMap`), `write`/`save`/`read`/`load`; the estimators also have `fitMultiple`.
A transformer's `transform`, and an estimator's `fit`, apply a non-empty param map to a copy as
PySpark does. `HashingTF.indexOf` fails at PySpark's JVM import, as the official one does over
Spark Connect. The `outputCol` default of a loaded stage is the one of the object that its load
makes (`<a new uid>__output`), not of the saved uid, as PySpark's load keeps the defaults of that
object when it takes the saved uid (`_resetUid`). A fit returns
one of 19 typed model classes, with PySpark's members: getters and setters, attributes such as
`mean`, `vocabulary`, `pc` and `categoryMaps`, DataFrames such as `surrogateDF`,
`approxNearestNeighbors`, `approxSimilarityJoin`, `getVectors` and `findSynonyms`, and the class
methods `CountVectorizerModel.from_vocabulary` and `StringIndexerModel.from_labels` and
`from_arrays_of_labels`. Over Spark Connect, PySpark's relation decorator
(`try_remote_attribute_relation`) drops every keyword argument of `approxNearestNeighbors`,
`approxSimilarityJoin`, `getVectors` and `findSynonyms`, and ours does too, as PySpark's behavior
over Spark Connect: a `distCol` given by keyword is ignored (the column stays `distCol`), and a call
whose positional arguments are not all given, or of more (`getVectors(1)`), fails at the server,
as official's does. The class methods take their arguments in PySpark's order: an empty
vocabulary raises the `ValueError` of `from_vocabulary` (of its `len`) first; then the model of
the class's constructor gets the object that the ML helper builds of its uid and of `list(...)` of
the labels of any iterable (a str, a set, the keys of a dict), sent as PySpark's `serialize` sends
them (`(list(labels), ArrayType(StringType()))` of `from_labels`, whose literal asserts each
string), after the assertion of an active session; and then the setters of the given params,
whose `TypeConverters` raise their `TypeError` after the request. `QuantileDiscretizer`'s fit
returns a `Bucketizer` of a new uid, with its splits and its input and output columns, of its
keyword constructor, as PySpark's `_create_model` does, and of the defaults of the discretizer
that `_copyValues` gives it (an `outputCol` of the discretizer's uid). Transformed DataFrames retain
their fitted models and upstream input plans. The Rust wrapper of the discretizer's splits is named
`QuantileDiscretizerModel`. Training-only parameters are omitted from model transforms.
Multi-column parameters, nested Bucketizer splits, vector-valued ElementwiseProduct weights,
NaN imputation values, and UnivariateFeatureSelector's setter-only parameters are supported.
RegexTokenizer's `gaps`, which has no converter in PySpark, keeps its Python value, which a
transform and a save send as its literal (of `LiteralExpression._from_value`, as PySpark's
`serialize_param`), so that a load has it; a value of no literal raises its error at either.

`StandardScaler` and `MaxAbsScaler` exports now refer to the native implementations. This
changes the previous deprecated `pyspark.ml.connect` scaler behavior: pandas frames and
array-valued feature columns are rejected; transforms return Spark DataFrames containing
vectors. Mean centering is opt-in through `withMean=True`. Convert pandas data with
`spark.createDataFrame` and array columns with `array_to_vector` before fitting.

This covers constructor parameters, named setters, fit and DataFrame transform, and
`StopWordsRemover.loadDefaultStopWords`, which reads the server's default stop words through
Spark Connect's ML helper, as PySpark does (Rust `StopWordsRemover::load_default_stop_words`).
As PySpark's constructor does, `StopWordsRemover()` fetches its default locale and then its English
stop words of the active session, once each, so without one it raises PySpark's bare
`AssertionError`. They are the defaults of PySpark's own `__init__` (`_setDefault`), which the
native stage takes, so a load sends its `Read` and then these two requests, and a copy sends none,
as PySpark's.
The Rust `StopWordsRemover::load` sends its `Read` alone, where PySpark's load also fetches these
two defaults, in the constructor of the loaded stage: only the requests differ, since a Rust stage,
new or loaded, reads an unset `locale` or `stop_words` as `None` or empty and the server applies
its defaults.

In Rust, each fitted model also has the members of its PySpark model class, which it reads of
the server-side model as PySpark does: for example `CountVectorizerModel::vocabulary`,
`PCAModel::pc`, the scaler statistics, `VectorIndexerModel::category_maps`, the LSH models'
`approx_nearest_neighbors` and `approx_similarity_join`, and `Word2VecModel::get_vectors` and
`find_synonyms`. `CountVectorizerModel::from_vocabulary` and `StringIndexerModel::from_labels`
and `from_arrays_of_labels` build a model of given values, and
`QuantileDiscretizer::fit_bucketizer` gives the `Bucketizer` of the fitted splits, as PySpark's
fit does. Every stage and model saves with `write()` and loads with `load(session, path)`, as
PySpark's `save` and `load`. Pipeline/tuning execution is separate from this native surface.
Existing vendored helper classes remain available.

Tests compare all 38 stages with official PySpark 4.3.0 parameter literals and Spark results,
including 10 multi-column/parameter variants, both constructor and setter configuration,
model lifetime, invalid inputs and pandas rejection. Schema checks compare class identifiers,
nullability and ML metadata; Connect omits the backing SQL schema of JVM UDTs on the wire.
Run `scripts/capture_ml_features_golden.py` with the official client to regenerate the fixture.
`scripts/capture_feature_models_golden.py` captures the members of every fitted model, the class
methods, and the save and load of every stage, which `e2e_feature_models.rs` compares.
Native live tests run in CI; set `FEATURE_REFERENCE=1` explicitly to run their reference mode.
They also check that every feature object that this client makes (a stage, its copy and load, a
fit and the copy and load of its model, the models of the class methods) is of the class of its
module, and that the `Pipeline` of `pyspark.ml` fits, saves and loads the stages.


## Native FPGrowth and PrefixSpan

Rust `ml::FPGrowth` fits Spark's frequent-itemset model and returns a typed
`FPGrowthModel`. The model exposes lazy `freq_itemsets()` and `association_rules()`
DataFrames and `transform()` predictions. The Python classes are exported from
`pyspark.ml.connect` and `pyspark.ml.connect.fpm`; fitted models expose
`freqItemsets` and `associationRules` as properties. All five estimator constructor
parameters have named setters; model setters cover the item column, prediction
column and minimum confidence. Parameters for prediction are captured when its lazy
plan is built. The itemset/rule properties fetch the fitted server model, matching
standard PySpark's Connect attribute relations.

The Python `FPGrowth` and `FPGrowthModel` follow `pyspark.ml.fpm` in full: the Params API
(`params`, `getParam`, `extractParamMap`, `explainParams`, `copy`, `set`, `clear`, Param objects
and getters), `setParams` and `fitMultiple` of the estimator, and `write`, `save`, `read` and
`load` of both. A fitted model copies its estimator's defaults, as PySpark's `_copyValues` does.
`PrefixSpan` has the same Params API, and no persistence, as Spark has none.

Rust `ml::PrefixSpan::find_frequent_sequential_patterns` and Python
`PrefixSpan.findFrequentSequentialPatterns` mine arrays of itemsets directly through
the Connect helper, with all four constructor parameters and named setters. They
return sequence/frequency DataFrames and have no fit/transform model.

These APIs accept Spark DataFrames and reject pandas frames before requests; a dataset that is
not a DataFrame is sent as PySpark sends it (see the argument rules above). Lazy
model tables retain their cached model; PrefixSpan retains its input plan, including
any upstream fitted models. A fitted model's transform of a DataFrame of another session is lazy
and fails with `CONNECT_ML.CACHE_INVALID` when the result is analyzed or collected, as PySpark's does.
In Rust, `FPGrowth` and `FPGrowthModel` save with `write()` and load with
`load(session, path)`, as PySpark's `save` and `load`. The vendored `pyspark.ml.fpm` module is
unchanged.

Spark 4.3.0's Connect helper currently rejects `Long` argument literals: PrefixSpan's
`maxLocalProjDBSize` works within the signed 32-bit range, including its default
32,000,000; larger values raise the same `LONG not supported` server error as official
PySpark. The client preserves PySpark's integer widths and explicit null parameters.
Tests cover this limitation, typed string/integer itemsets, association rules, held-out
predictions, model setters, sequence-length limits, invalid data, and model lifetime.

## Native ALS recommendation

`ml::ALS` / `ml::ALSModel` in Rust and `pyspark.ml.connect.recommendation.ALS` /
`ALSModel` in the Rust-backed Python package run Spark's standard ALS over Connect.
The estimator exposes all 18 parameters, including explicit/implicit feedback,
nonnegative factors, regularization, block counts, storage levels and seed.
Python mirrors the 17 official constructor keywords; use `setPredictionCol` for
an alternate prediction column, and `setNumBlocks` to set both block counts.

The fitted model supports transform, rank, user/item factor tables, recommendations
for all users/items, and recommendations for a subset of users/items. Rust uses
`user_factors`, `item_factors`, and `recommend_for_*`; Python exposes `userFactors`,
`itemFactors`, and `recommendFor*`. Model setters control transform parameters,
including `coldStartStrategy="drop"` to remove predictions for unknown IDs.
Factor and recommendation relations access the fitted server model, matching
PySpark Connect. Lazy results retain their cached model and any models referenced
by the subset DataFrame until evaluation completes.

The Python `ALS` and `ALSModel` follow `pyspark.ml.recommendation` in full: the Params API,
getters and setters, `setParams`, `setNumBlocks`, `fitMultiple`, `copy`, and `write`, `save`,
`read` and `load`. A model's `coldStartStrategy`, `userCol`, `itemCol` and `blockSize` come from
its estimator, as `_copyValues` gives them. The `rank` of a model is read from the server on each
access, as PySpark's `_call_java("rank")` does, so a load sends no request for it. PySpark's fit
also fetches it once, or twice when `rank` is set, as its `_copyValues` probes `hasParam("rank")`
of the new model and so reads that property; the Python and Rust fits do not, so only the
requests differ.

Fit, transform and subset recommendation inputs must be Spark DataFrames; pandas
DataFrames are rejected before conversion or RPC, and a subset's other values are sent for the
server to reject, as PySpark sends them. Model operations must use the
fit session for transforms: another session's DataFrame fails with `CONNECT_ML.CACHE_INVALID` when
the transform is analyzed or collected, as PySpark's does. The subset recommendations accept another
session's DataFrame and run, as PySpark's do.
Official PySpark 4.3.0 fixtures cover explicit and implicit ratings, predictions, cold starts,
factors, and all four recommendation methods, with live
Rust and native Python tests. In Rust, `ALS` and `ALSModel` save with `write()` and load
with `load(session, path)`, and a loaded model reads its rank of the server, as PySpark's does.
Live tests also check that the `CrossValidator` and `TrainValidationSplit` of `pyspark.ml` fit,
save and load an `ALS` with a native `RegressionEvaluator`.


## Native ML evaluators

Rust and the Rust-backed `pyspark.ml.connect.evaluation` exports provide all six
standard Spark ML evaluators: regression, binary classification, multiclass
classification, multilabel classification, clustering and ranking. Their
constructor parameters and named setters follow `pyspark.ml.evaluation` in
PySpark 4.3.0. Rust also provides typed parameter getters. Metric direction is
available as `is_larger_better()` / `isLargerBetter()` without contacting Spark.
`pyspark.ml.connect.evaluation` also exports PySpark's own `pyspark.ml.evaluation.Evaluator`, the
base class of the six, so that it has the names of `pyspark.ml.evaluation` (as
`pyspark.ml.connect.tuning` exports PySpark's `ParamGridBuilder`), while the `Evaluator`,
`Estimator`, `Transformer` and `Model` of the `pyspark.ml.connect` package stay the deprecated
classes of `pyspark.ml.connect.base`, as in PySpark's deprecated package.

The three evaluators of upstream's deprecated `pyspark.ml.connect.evaluation` (regression, binary
and multiclass classification), which the native classes of this module replace, use standard
Spark ML defaults:
`BinaryClassificationEvaluator` takes `rawPredictionCol="rawPrediction"`, and
`MulticlassClassificationEvaluator` defaults to `metricName="f1"` and requires
keyword arguments. Use `rawPredictionCol` / `setRawPredictionCol` instead of the
deprecated `probabilityCol` / `setProbabilityCol` binary-evaluator argument.

Evaluation sends Spark DataFrames and their column names/types directly to Spark.
A pandas DataFrame raises the `AttributeError` of its own `sparkSession`, as `pyspark.ml`
does, before serialization or RPC; a model's summary `evaluate` raises the `TypeError` of
`pyspark.ml` instead. Prepare double
prediction columns, vector scores, and arrays of doubles where required
by the corresponding standard evaluator. Clustering features may be vectors or
arrays of doubles. Evaluators no longer silently cast
columns, convert array scores to vectors, or normalize field-name casing.

Official PySpark fixtures and live Rust/Python tests cover every supported metric,
weights, regression through the origin, binary score bins, multiclass log-loss
clipping and beta, per-label metrics, both silhouette distances, ranking cutoffs,
invalid inputs, and metric direction. In Rust, each evaluator saves with `write()` and
loads with `load(session, path)`, as PySpark's, and `set_uid` sets its uid, as PySpark's
`_resetUid` does. In Python, each evaluator has PySpark's Params
API, its getters and setters, `setParams`, and `write`/`save`/`read`/`load`, which keep its uid
and params. `evaluate(dataset, params)` with a non-empty param map evaluates a copy of the
evaluator, as PySpark's `Evaluator.evaluate` does, and leaves the evaluator as it was.
`scripts/capture_evaluation_api_cases_golden.py` captures these cases from official PySpark.

`scripts/capture_fpm_recommendation_evaluation_golden.py` captures the members of the FPGrowth
and ALS models and of the evaluators, and their save and load, of the configurations of these
fixtures, which `e2e_fpm_recommendation_evaluation.rs` compares.


## Native ML statistics

Rust's `ml::{ChiSquareTest, Correlation, KolmogorovSmirnovTest, Summarizer,
SummaryBuilder, MultivariateGaussian}` and the native Python classes in
`pyspark.ml.stat` cover the standard ML statistics APIs from PySpark 4.3.0.
The Python module is a Rust-backed shim, replacing its vendored implementation,
except `MultivariateGaussian`, which stays PySpark's plain Python class.
`pyspark.ml.connect` adds no statistics module; its `summarizer` is the one of PySpark's
deprecated package, unchanged.

The module has PySpark's names and no `__all__`, as PySpark's, so `from pyspark.ml.stat import *`
imports the same names (the names of its imports too, such as `Column` and `JavaWrapper`).
`ChiSquareTest`, `Correlation`, `KolmogorovSmirnovTest` and `Summarizer` construct as PySpark's
plain classes: of no arguments (`ChiSquareTest() takes no arguments`) unless a subclass has an
`__init__` of its own, of attributes and weak references, and pickled and copied as plain objects.
`SummaryBuilder` is a `JavaWrapper` of PySpark's `__init__(self, jSummaryBuilder)`, which keeps no
Java object over Spark Connect: its `summary` reads the `_metrics` of its attributes, which
`Summarizer.metrics` sets with a `_java_obj` of None, so `SummaryBuilder(None).summary(col)` raises
PySpark's AttributeError, and a subclass that sets `_metrics` summarizes of them. The classes have
PySpark's docstrings, and their static methods (and `SummaryBuilder.summary`) are functions of the
module of PySpark's signatures, annotations and docstrings (`Summarizer.mean.__module__` is
`pyspark.ml.stat`, and its docstring has the `.. versionadded:: 2.4.0` of PySpark's `since`), over
the native methods, so `inspect` and `help` give PySpark's and a call binds as PySpark's does
(`SummaryBuilder.summary() takes from 2 to 3 positional arguments but 4 were given`);
`Summarizer._check_param` and `_get_single_metric` are PySpark's code over Spark Connect. The
annotations name PySpark's classic `Column` of `pyspark.sql.column`, as PySpark's do, of which the
Columns of a session are instances, and the classes are subclasses of the native classes, which
their MROs name after them (`ChiSquareTest`, `ChiSquareTest`, `object`), where PySpark's are plain
classes.

The statistical tests return lazy Spark DataFrames. Chi-square supports both the
vector result and one row per feature (`flatten=True`); correlation supports
Pearson and Spearman; the KS test supports Spark's normal distribution, with
optional mean and standard deviation. Spark validates column types and
statistical arguments. As with `pyspark.ml.stat`, the dataset and the other arguments are serialized
as the arguments of a helper call, in that order, before any RPC: a pandas DataFrame raises
PySpark's `UNSUPPORTED_DATA_TYPE`, and another argument that cannot be a literal raises the error of
its conversion. A literal of another type than the helper takes, such as a dataset that is not a
DataFrame or a column name that is not a str, is sent as PySpark sends it, and the server rejects it
when the result is analyzed or collected (`tests/golden/statistics_arguments.json` records these
cases of official PySpark). A `(value, DataType)` pair is sent as the literal of that type, as
PySpark sends it, and raises the `AssertionError` of PySpark's `LiteralExpression(value, dataType)`
of a value or type that a literal does not take. The KS test converts its parameters with
`float()` first, as PySpark does. Results retain any fitted models used by their input plans.

`Summarizer.metrics(...).summary(features, weight)` builds a SQL aggregate that
works in `select` and grouped aggregations. All ten metrics are supported:
`mean`, `sum`, `variance`, `std`, `count`, `numNonZeros`, `max`, `min`, `normL1`,
and `normL2`. Each also has a static convenience method; Rust uses snake_case
(`num_non_zeros`, `norm_l1`, `norm_l2`). Omitted weights default to 1.0. Inputs
are ML vector Columns, including sparse vectors. Metric names are case sensitive;
`numNonZeros` uses a capital Z. An empty metric selection returns an empty struct.

`MultivariateGaussian` is a mutable mean/covariance holder, with no computation
or extra validation. Rust uses `MlVector` and `MlMatrix` (dense or sparse storage,
the matrix type of the native ML params). Python keeps PySpark's plain class, whose
instances hold the original vector and matrix objects and pickle as PySpark's. Python
results materialize vectors and matrices through the existing ML UDT conversion.

Reference protobuf fixtures and live Rust/Python tests cover every metric with
and without weights, grouped summaries, chi-square flattening, both correlation
methods, KS parameters, constant-feature NaNs, invalid inputs, and lazy model
ownership. Python tests additionally compare sparse and dense inputs. Native
classes are asserted explicitly; the tests also run separately with official
PySpark as the reference.

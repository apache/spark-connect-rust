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

`Value` and `LiteralExpression` add a `DayTimeInterval(i64)` variant for signed
microseconds. Downstream exhaustive matches on these public enums must handle
the new variant. Rust interval map keys are stringified as raw microseconds;
Python `timedelta` map keys are not yet supported.

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
`Params`: the value that is set, else Spark's default (`None` for `seed`, `weight_col` and
`thresholds`). The fields of `LogisticRegressionModel`, `StringIndexerModel` and
`MaxAbsScalerModel` are private, as those of the other fitted models but `PipelineModel`: use
`Transformer::operator` and `Transformer::params`, and `LogisticRegressionModel::features_col`,
`label_col` and `prediction_col`. `VectorAssembler::input_cols` and `Pipeline::stages`
return a `Vec<String>`, and `VectorAssembler::set_input_cols` and `Pipeline::set_stages` take
`impl IntoIterator<Item = impl Into<String>>`. The `input_col` of `StandardScaler`,
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

`LogicalPlan::Raw` adds a `depth` field: the levels of its relation as a plan, with
those of the DataFrames that it holds, as `pyspark.ml` builds a relation over its
input. They count for `MAX_PLAN_DEPTH`, as PySpark's serializer counts the plan of the
input within its own. Code that constructs the variant directly sets it to 1 for a
relation without nested plans. `LogicalPlan::try_to_proto` serializes a plan as a
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

`CoGroupedData::input_columns` is removed. It gave the columns of both sides for the arguments
of the function of a cogroup, but that function binds no columns, as PySpark's `CoGroupMap`
builds it: it takes the whole cogroups, so the inputs need no analysis.
`GroupedData::apply_in_pandas_with_state_with_schemas` sends the output and the state schemas
as the strings that it is given, DDL or JSON, which the server parses, as PySpark sends a
schema given as a string; `apply_in_pandas_with_state` sends the JSON of its types, as before.
`Value::calendar_interval_parts` reads the months, days and nanoseconds of a collected calendar
interval back from its text. `LocalRowIterator::next_decoded` takes the next row of the batch
that the iterator holds without fetching another, and `LocalRowIterator::has_received_batch`
tells whether a record batch of the result has arrived, also one without rows.
`SparkConnectClient::read_plan_compression` reads the plan compression settings of the server
before a plan is sent, as PySpark's client reads them when it builds its first plan.

`PythonUDFPayload` and `LogicalPlan::CommonInlineUdtf` add a `pickler` field, which is not a
stable API: the pickling of the command of a UDF or UDTF of the Python client, which a request
calls each time it builds the plan, as PySpark pickles the function then. Code that constructs
the struct or the variant directly sets it to `None`. `PythonUDFPayload::new` and
`TableValuedFunction::udtf` are unchanged: a command of bytes is sent as it is.

## Native ML objects: fetch, summaries, save and load (Rust API)

`spark_connect::ml` reads and writes the server-side objects of Spark Connect ML as PySpark 4.3.0
does over Spark Connect, for the typed model accessors, summaries and persistence of the native
classes to build on:

- `MlRef` is a server-side object: a fitted model (`Model::ml_ref`), or another object that a
  fetch returns, such as a tree of an ensemble. Its clones and the DataFrames whose plans read it
  share it, and the last owner to drop deletes it on the server, as PySpark's `RemoteModelRef`.
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
`PipelineModel` (`Evaluator` and `ParamGridBuilder` are PySpark's own). Each class has, where its
PySpark class has them:

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
  or a OneVsRest's fit; of the names of the params of the model of a JVM estimator's fit and of a
  load, which PySpark's `_resetUid` rebuilds (the set params of a load of PySpark's Python format,
  a OneVsRest's or a validator's, are then in the order of its metadata), and of a copy of
  `Params.copy`. `clear` does nothing over Spark Connect, as PySpark's `JavaParams.clear`, of any
  positional arguments. The native object's params are made those of the maps before each native
  member reads them (a fit, a transform, an evaluation, a writer, a model's members), and so are
  those of the objects that it
  holds (an OneVsRest's classifier, a Pipeline's stages, a validator's estimator, evaluator and the
  values of its grids, a PipelineModel's stages, an OneVsRestModel's models, a validator model's
  best and sub-models). A writer reads them at its `save`, as PySpark's `RemoteMLWriter` reads
  those of its `_instance`.
  The maps of an object that this client makes (of a fit, a load, a native copy) are made of the
  native object's values at their first read. The native members have PySpark's docstrings,
  signatures, modules and qualified names, of the class of PySpark's that defines the member
  (`JavaMLWritable.write`, `JavaPredictionModel.predict`), and Python binds a call of one as it
  binds PySpark's function, of the parameters of the wrapper of its decorator (`JavaParams.clear`
  of `try_remote_intercept` takes any positional arguments, a DataFrame method of
  `try_remote_attribute_relation` any arguments): a call that does not bind raises PySpark's
  message (`JavaMLWritable.write() takes 1 positional argument but 2 were given`) before the
  native member runs. The Param objects are PySpark's own `pyspark.ml.param.Param`, of the docs,
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
  `MLWriter`s and `MLReader`s are `RemoteMLWriter` and `RemoteMLReader`, as over Spark Connect, of
  the attributes `shouldOverwrite` and `optionMap`, and the AttributeError of a missing attribute
  names the class as PySpark's does, of an object as of the class itself
  (`type object 'DecisionTreeRegressor' has no attribute 'maxIter'`).
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
    defaults of that uid and name (`outputCol`, `seed`). A subclass of PipelineModel,
    OneVsRestModel or a validator's model keeps the uid of its class's name.
  - `fit`, `fitMultiple`, `transform`, `evaluate`, `save` and `load` are PySpark's own (of
    `Estimator`, `Transformer`, `Evaluator`, `MLWritable` and `MLReadable`), with PySpark's
    signatures and docstrings. They run the object's `_fit`, `_transform`, `_evaluate`, `copy`,
    `write` and `read`, so a subclass's `_fit` runs, also in the copies that a Pipeline or a
    CrossValidator fits.
  - A `copy` is of the class of the object, of its attributes, as PySpark's `Params.copy` makes it
    of `copy.copy`.
  - `uid` is assignable. An assignment re-parents the object's Params, where PySpark's keeps the
    Params of the old uid, which then do not belong to the object (until its `_resetUid`).
- **`copy.copy`, `copy.deepcopy` and `pickle`** take an object as PySpark's do:
  - An estimator, a transformer, an evaluator, a Pipeline or a validator, and a model of no
    server-side object, copy, copy deep and pickle of their uid, params and attributes. A shallow
    copy shares the values (a Pipeline's stages), and a deep copy and a pickle copy them.
  - A PipelineModel, a OneVsRestModel and a validator's model are made again of their
    constructor's objects (`stages`, `models`, `bestModel`, ...).
  - A model of a server-side object, and its summary, copy, sharing the object, and are not
    copied deep nor pickled: PySpark's raise the TypeError of the lock of the model's
    `RemoteModelRef`. A pickle of a summary raises the error of the first of its attributes that
    does not pickle, as PySpark's: of the summary of an `evaluate`, the lock of its model
    (`__source_transformer__`); of a training summary, the lock of the model in the plan of its
    summary dataset when that reads a transform of a model (the transform of the training dataset
    by the model, of the KMeans, BisectingKMeans and GaussianMixture summaries, and a training
    dataset of a stage after a model of a Pipeline), else the AttributeError of the session of the
    training DataFrame, which PySpark pickles after its plan. The error of a lock is that of its
    pickle at the protocol (a `PicklingError` below 2). Of cloudpickle, which pickles the local
    function of the session by value, PySpark's training summary of a training DataFrame raises
    the TypeError of the lock of its model, which comes next, where the drop-in's raises the
    AttributeError of the session, as of `pickle`. PySpark's `copy.deepcopy` of a training summary
    requests the schema of its summary dataset before its error (its DataFrame's `__getattr__`
    reads the columns), where the drop-in's raises at once.
  - A summary of PySpark's constructor (`LinearRegressionSummary("<id>.summary")`), of no
    server-side object, copies, copies deep and pickles, at every protocol, as PySpark's plain
    object: of its class and of its attributes (`_java_obj`).
  - A shallow copy of a model keeps the server-side object while either object lives. PySpark's
    shares its `RemoteModelRef` without a reference of its own, so the garbage collection of the
    copy releases the object of the original (`CONNECT_ML.CACHE_INVALID` at its next use).
- **A model of no server-side object**: a model class constructs as PySpark's
  `JavaModel(java_model=None)` (`KMeansModel()`, `LDAModel()`), of PySpark's `JavaModel.__init__`,
  a model of a new uid of its class and of its params, none set, whose Params API, `copy` and
  `write()` work, and whose members of the server, `__repr__`, a transform and a save raise
  PySpark's bare `AssertionError`, as does the
  `params` walk of a model of `functools.cached_property` members (`trees`), and so its
  `explainParams` and `copy`. Another `java_model` is the `AssertionError` of PySpark's over Spark
  Connect, which takes its `RemoteModelRef` only.

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
at once leaves nothing cached.

The Rust `transform` of a fitted model rejects a DataFrame of another session. In
Python, as with PySpark, it returns a DataFrame whose analysis or execution fails
on the server of that session with `CONNECT_ML.CACHE_INVALID`.

In Python, a fit, its training summary and the members of its model (`predict`, the summary's
DataFrames) are sent over the session of the dataset and of the model, so ML works on a session
that is not the active one (a second session of `SparkSession.builder.remote(...).create()`).
This is a deliberate difference from PySpark: it sends the summary of a fit and the members of a
model (`_call_java`) over `SparkSession.getActiveSession()`, so on such a session its fits of a
model with a summary and its models' members fail with `CONNECT_ML.CACHE_INVALID` (`Cannot
retrieve Summary object ... from the ML cache`). A model used from a session other than its own
fails on both.

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
`threshold` and `thresholds`, whichever call set them, as its constructor does. Its signature is PySpark's for every class, the feature stages included, except
`CrossValidator`, which takes `numFolds`, `parallelism` and `seed`, with a default of `None` for
the first two. The `params` argument of a fit, transform or
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
`fit(5)`), before any request; an object that has a `sparkSession` raises a `TypeError`. The
methods that `pyspark.ml` runs as a helper or attribute call read nothing of their dataset.
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

The Rust `LogisticRegression` and `LogisticRegressionModel` have the params of `pyspark.ml`, with one
generated getter and setter per param (`set_max_iter` and `max_iter()`, `set_features_col` and
`features_col()`, and so on), and `PARAM_NAMES` lists them. `LogisticRegression::feature_col`
and `set_feature_col`, the names of v4.2.0, stay as deprecated aliases of `features_col` and
`set_features_col`. `threshold()`, `thresholds()`,
`set_threshold` and `set_thresholds` follow `_LogisticRegressionParams`: the getters check that
the two agree to 1e-5 and `threshold()` requires two thresholds. The setters leave the other
param set, as over Spark Connect, where the `clear` of `pyspark.ml` does nothing, so a
disagreement raises the getters' error. Their arithmetic is PySpark's, in Python: a zero divisor
of the equivalent threshold of two thresholds, `1 / (1 + t0 / t1)` (a zero `t1`, or a ratio of
-1), is the error "float division by zero" of its `ZeroDivisionError`, of `threshold()` and,
with a threshold set too, of `thresholds()`.
The model has the typed members of `LinearSVCModel` above, plus `coefficient_matrix()` and
`intercept_vector()`; `coefficients()` and `intercept()` raise the server's error of a multinomial
model, as PySpark does. The model's `summary()` is binary for at most two classes, of the
`numClasses` that it fetches when called (PySpark fetches it at the fit), and `evaluate` makes
the same choice on its dataset. `has_summary()` is client side. The
Rust API differs from `pyspark.ml` as the LinearSVC section says.

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
  estimator are kept as they are, in a `PipelineModel` of a new uid.
- **Transform:** `PipelineModel.transform` transforms by each stage in order.
- **Copy:** `Pipeline.copy` keeps the uid and copies each stage with the extra params, while
  `PipelineModel.copy` is of a new uid.
- **Persistence:** `write`, `save`, `read` and `load` are of PySpark's Python format:
  - the metadata of `pyspark.ml.pipeline.Pipeline` (or `PipelineModel`), of the `stageUids`;
  - each stage, saved by its own writer at `stages/<index>_<uid>`.

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
- **Models:** `bestModel`, the metrics and `subModels` are attributes of the instance, as
  PySpark's. `transform` transforms by the best model. `copy` copies the best model and the
  sub-models, and a model of no sub-models fails as PySpark's `assert` does.
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

The DataFrame-based `pyspark.ml` API (`pyspark.ml.classification`, `feature`,
`regression`, `clustering`, `evaluation`, `tuning`, `stat`, `Pipeline`, ...) is
Apache Spark's own, vendored from v4.3.0: estimators fit, models transform, and
pipelines save and load on the server as with PySpark Connect, over the Rust
transport. These modules build their own `spark.connect` messages with
`pyspark.sql.connect.proto`, which is the drop-in's own protobuf runtime in Rust
(`pyspark.sql.connect.proto._rust`) in place of the generated modules of Apache Spark, whether or
not `protobuf` is installed, so `protobuf` is not a dependency of `pyspark-client-rust`. The
runtime has the classes, containers, descriptors and descriptor pool of upb, the implementation
of a `protobuf` install, with their bytes, text, errors and presence (of protobuf 7, where
protobuf 6's differ), the methods of protobuf's `Any` and `Timestamp` (a Timestamp field takes a
`datetime`), and the modules of the protos with the names of the generated modules
(`pyspark.sql.connect.proto.base_pb2`, ..., and `example_plugins_pb2`, of Spark's example plugin).
`protobuf`'s `json_format` and `text_format` print and parse its messages. The differences:

- its types and its `DecodeError` are of `pyspark.sql.connect.proto`, not `google.protobuf`, and
  it has one map container type where upb has a scalar one and a message one;
- it parses, prints and builds (of nested dicts) messages of any depth, where upb refuses to
  parse more than 100 nested messages, its constructor overflows the stack of a thread from
  thousands of nested dicts, and `text_format` exceeds Python's recursion limit;
- where Python code that a method of a container runs (an `__index__`, an `__eq__`, an iterable)
  shortens the container, a read past its end raises `IndexError` and an insertion or a deletion
  takes the container as it is, where upb reads or writes past the end of its array, which can
  end the process;
- the iteration order of a map is the order of insertion, where upb's is that of its hash table.
  Serialization and the text format write the entries in upb's order of the keys;
- `GetOptions()` of a descriptor gives `protobuf`'s `descriptor_pb2` options, as upb's does, so
  it needs `protobuf`.

Where `protobuf` is installed, the files of the protos of Spark Connect are added to its default
descriptor pool, as the generated modules add them. The modules that protoc generates of other
protos that import them (those of a plugin, such as Delta Lake's Connect client) then build, and
`json_format` finds the types of an `Any`. The messages of such a module are `protobuf`'s, which
take messages of their own classes only: their `CopyFrom`, `MergeFrom` and constructors raise
`TypeError` for a message of `pyspark.sql.connect.proto`, which a field of theirs takes through its
bytes (`plugin.input.ParseFromString(relation.SerializeToString())`) or a dict instead. PySpark's
messages are of `protobuf`'s classes, so its plugins pass them as they are: Delta Lake 4.0's
`DeltaTable.delete`, `update` and `merge` build their plans with `CopyFrom` of PySpark's
messages, and raise that `TypeError` here. The other way, the messages of
`pyspark.sql.connect.proto` take `protobuf`'s messages of their types as they are, and an `Any` of
either packs and unpacks a message of either.

A DataFrame of a plan of `pyspark.ml` (`DataFrame(plan, session)`) takes the plan ID
of this client rather than the one of the plan, which its columns refer to.

## Spark Declarative Pipelines CLI

The CLI of Spark Declarative Pipelines (`python -m pyspark.pipelines.cli init`, `run` and
`dry-run`) and its modules are Apache Spark's own, vendored from v4.3.0, and Apache Spark's
tests of them pass with this client. The CLI needs `pyyaml`, which `pyspark-client-rust`
depends on, as `pyspark-client` does. The user context extensions of a
client (`spark.client.add_threadlocal_user_context_extension` and
`add_global_user_context_extension`), which the CLI adds for the analyses of a pipeline, are
sent with the requests of executions, analyses, configurations, error details and interrupts,
as PySpark sends them.

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
| Methods   | ~1340 |
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
The native [Rust API](dataframes.md) mirrors the same surface.

Three differences come with being a different package with a native core. The distribution is
`pyspark-client-rust`, so `importlib.metadata.version("pyspark-client")` raises
`PackageNotFoundError`, where `pyspark.__version__` is PySpark's. A traceback of an error that the
server raises has the frames of the user's code only, without those of PySpark's client
(`collect`, `_to_table`, `_execute_and_fetch`, ...), which the native core does not have. The
package is marked typed as PySpark's is (`py.typed` and PySpark's stubs), and
`pyspark.sql.SparkSession`, PySpark's own class, has PySpark's signatures and docstrings. The
functions of `pyspark.sql.functions` and the methods of the native classes, such as
`DataFrame.select`, have PySpark's parameters, without its annotations and its full docstrings.
The `__init__` of a native type of parameters, such as `ArrayType`, is a slot of its class, whose
`inspect.signature` is CPython's `(self, /, *args, **kwargs)`; that of the class shows the
parameters, as PySpark's does.

The drop-in parses the protobufs that a pickled Column or DataFrame carries, and the requests
that session hooks see, at any depth that PySpark's serializer reaches, where the parsers of
protobuf and prost stop at 100 nested messages. Its Rust client enables prost's
`no-recursion-limit` feature, which applies to every message the drop-in decodes. So it also
decodes server responses deeper than 100 nested messages, such as the schema of a struct nested
34 levels deep, which PySpark's client rejects with "Exception deserializing response!". The
server is trusted, as SECURITY.md sets out, so this is not a bound on the responses.

The drop-in also sends a plan that PySpark's client cannot encode: from about 50 nested
expressions, PySpark raises upb's `DecodeError: Exceeded upb_DecodeOptions_MaxDepth` before it
sends the plan, where the drop-in sends it and the server fails it at its own limit of 100 levels.
A Column of 745 levels or more raises a `RecursionError` when it is pickled, of the drop-in's own
message. PySpark's pickler raises one from about 600 levels, under Python's default recursion limit,
so a Column of 600 to 744 levels pickles in the drop-in only. `copy.copy` and `copy.deepcopy` of a
Column share its expression, of which a deep copy makes again only the levels above the UDFs whose
functions it copies, so they copy a Column of any depth, as PySpark's `copy.copy` does; PySpark's
`copy.deepcopy` raises a `RecursionError` from about 165 levels.

A few values that PySpark's client converts when it builds the plan, the drop-in converts when it
builds the Column or the window: the frame bounds of `Window.rowsBetween` and `rangeBetween`, and a
`StructField` or an abstract type such as `AtomicType()`, also one nested in the type, given as the
return type of a UDF that is not Arrow-optimized. Their errors are PySpark's, raised at the call
rather than when the plan is built. The type of `cast` and `try_cast` is converted when the plan is
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

An `Observation` or a `SparkSession.Hook`, of the class or of a subclass, that PySpark pickled at
protocol 0 or 1 raises a `TypeError` when the drop-in unpickles it, as CPython's
`copyreg._reconstructor` cannot create the drop-in's, a class with a native base; protocols 2 and
up, pickle's default, load in both clients, and PySpark loads the drop-in's at every protocol. A
hook of a subclass with `__slots__` pickles at protocols 0 and 1 too, where PySpark's raises the
`TypeError` of a class that defines `__slots__` without `__getstate__`.

So it is for the data types. PySpark's are plain classes, which pickle as their class and their
`__dict__`: at the protocols 2 and up the drop-in makes its type of the class, without arguments,
and sets the attributes, so a pickled `df.schema` loads with its fields, names, metadata,
collations, nullable flags and parameters, as PySpark's types of them. The private `_crs` and
`_alg` of PySpark's spatial types, which its constructors set of the SRID, are not attributes of
the drop-in's. At protocol 0 or 1 the drop-in raises `TypeError: object.__new__(DecimalType) is not
safe, use DecimalType.__new__()`. The drop-in's own types pickle, copy and deep-copy as
`_parse_datatype_json_string` of their JSON, or, with a user-defined type or a nullable flag that is
not a bool, of the types and flags they are made of, which the Python workers of the server load at
every protocol. So such a pickle does not keep what PySpark's pickle of the `__dict__` keeps: an
attribute that a user set, a `hasPrecisionInfo` that is not true, and the `names` and
`_needConversion` of a `StructType` that an edit of its fields made differ.

The parameters of a type are its attributes, as PySpark's: `t.scale = 3` makes a
`DecimalType(10, 2)` a `DecimalType(10, 3)`, as its repr, JSON, `==` and hash show, also in the
struct, array or map that holds it. A drop-in type holds only the values that its constructor
makes, so the assignment of another, such as a str of `scale` or a field that an interval does not
have, raises the constructor's error, where PySpark keeps any value, of which its repr or JSON may
fail later. A set `srid` of a spatial type is of its JSON too, where PySpark's JSON stays that of
the CRS of its constructor. The parameters are not entries of the `__dict__` of a type, so
`vars(t)` is empty, where PySpark's has them.

The abstract classes of the types, `DataType`, `AtomicType`, `NumericType`, `IntegralType`,
`FractionalType`, `DatetimeType`, `AnyTimeType`, `SpatialType` and `AnsiIntervalType`, make types
of their own, as PySpark's plain classes do. `AtomicType()` has the `typeName`, `simpleString` and
`jsonValue` of its class, `atomic`, converts no values, and is equal to a type of its class and
`__dict__` only. It pickles as its class and its `__dict__`: at the protocols 2 and up its pickle
is PySpark's own, byte for byte, and PySpark loads the drop-in's at every protocol. The struct,
array or map that holds it names it in its `simpleString`, JSON and tree string, and pickles of
the types it is made of. `IntegralType()` is one instance, as PySpark's `DataTypeSingleton` makes
it, and a Python subclass of these classes, such as a `UserDefinedType` of parameters, takes the
arguments of its `__init__`. A Python subclass of `IntegralType` cannot be instantiated, where
PySpark's `DataTypeSingleton` makes one instance of each subclass: the constructor of a native
class either returns an object that exists or makes one of the class it is called for. Neither
client has a proto or an Arrow type of such a type: a plan of one raises `UNSUPPORTED_OPERATION`,
and local rows of one `UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION`.

PySpark's singleton types, from `NullType` and `IntegerType` to `CalendarIntervalType` and
`IntegralType`, are classes of the metaclass `DataTypeSingleton`, which `type(IntegerType)` shows.
The drop-in's are native classes, whose metaclass is `type`, and `pyspark.sql.types` has no
`DataTypeSingleton`: PyO3 has no option of a metaclass of a class, and CPython refuses to assign
the `__class__` of a class of `type` after it is made. Their constructors return the one instance
and raise the `TypeError` of `DataTypeSingleton.__call__` for an argument, as PySpark's, and
`inspect.signature` of the class shows no parameters.

The pickle of a `TableArg`, of `DataFrame.asTable()`, raises a `TypeError`, where PySpark's
pickles; `copy.copy` and `copy.deepcopy` copy it as PySpark's.

The keys of `getMessageParameters()` of a server error are in the server's order. PySpark's
follow the protobuf map of the error details that it fetches: with protobuf's pure-Python
implementation that is the server's order too, and with upb, protobuf's default, the order of its
hash table, which changes from one process to the next. For its own errors, PySpark keeps the
order of each call site.

At interpreter exit, a session that is not stopped has its ML cache cleaned by one
`clean_cache` request, as PySpark's client does. Its errors are ignored, but when the server
is gone the exit waits on that request's retries, as PySpark's does, so the exit can take as
long as those retries. A stopped session sends nothing at exit.
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

The `TypeError` for too many positional arguments counts them without `self`: where CPython says
"takes from 2 to 4 positional arguments but 5 were given", the drop-in says "takes from 1 to 3
positional arguments but 4 were given". A method that takes no arguments gets "takes no arguments
(1 given)", where CPython says "takes 1 positional argument but 2 were given" for
`DataFrame.distinct(1)`. A method that takes one argument gets the plural "positional arguments"
even for a count of one: the drop-in says "takes 1 positional arguments but 3 were given" for
`DataFrame.limit(1, 2, 3)`, where CPython says "takes 2 positional arguments but 4 were given".
A warning that the Rust core raises, such as the
`FutureWarning` of `registerTempTable`, is attributed to the caller's frame and module, so a
`warnings.filterwarnings(..., module="pyspark")` does not match it, as it matches PySpark's.
The warning of a window of no partitions, `WARN WindowExpression: No Partition Defined for Window
operation! ...`, is PySpark's: a `UserWarning` of `pyspark.sql.connect.expressions`, of one line of
that module, which the default filter shows once, at each request that builds the plan of such a
window, as PySpark warns as it builds the plan. Its line is that of the drop-in's module. A pickle
of a DataFrame or a Column does not warn, as PySpark's builds no plan; but a DataFrame or a Column
that is unpickled, and a deep copy of a DataFrame whose plan holds no function of a UDF or that is
copied into another session (see the UDFs below), send the plan that was built for the pickle or
the copy, so their requests do not warn again, where PySpark's warn at each. Such a plan keeps the
origin of the `and` of the conditions of a join of a list of Columns of that time too, where
PySpark gives it the call site of each request that builds it, as it does in the drop-in otherwise.

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
  name of `observe`, the names of a using join, and the functions of `agg` of a dict, and the
  name of an `Observation`, which fails the DataFrames that it observes;
- a tag of `addTag` fails each execution that the thread starts, as its request is built after its
  plan, until `removeTag` removes it; a request without tags, such as that of `df.schema`, is sent.
  `getTags` gives the tag as it is;
- a key or a value of `SparkSession.builder.config`, or an `appName`, fails `getOrCreate`;
- a `DataStreamWriter`'s fails the call that takes it, as PySpark sets that writer's request as it
  goes.

Such a str is not one of the modes of `explain`, of the `how` of `dropna`, of the join types of
`join` and `lateralJoin`, nor a column of `sum`, `avg`, `min` and `max` of grouped data: they raise
PySpark's `VALUE_NOT_ALLOWED`, `UNSUPPORTED_JOIN_TYPE` and `NOT_NUMERIC_COLUMNS`, of the str as it
is. `explain` builds the plan first, so the error of a plan that fails to build comes first.

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
plan, but for the partitioning of a repartition, which comes first, and the name of a hint or an
observation before its expressions; the order of the other values of one plan can differ. As in
PySpark, a DataFrame of such a plan raises at its requests, `schema` and `collect()` among them,
and the call raises none. Of a join on a tuple, or on a list of conditions whose `&` fails,
`schema` gives a Column, as in PySpark: the property raises an `AttributeError`, after which Python
calls the `__getattr__` of the DataFrame. It is the Column of `schema`, where PySpark's is of
`_schema`, and `columns` gives the Column of `columns`, where PySpark's raises `NOT_ITERABLE`.

A DataFrame whose plan fails to build raises that error when it is pickled, also by cloudpickle,
where PySpark's cloudpickle pickles it, of a copy that raises the error at its requests. `alias`, a
hint, `dropDuplicates`, `observe`, `filter`, `sort`, `sortWithinPartitions` and the repartitions of
such a value keep the cached schema of the DataFrame that they are of, as PySpark copies it, so
their `schema` is known without a request, and their other requests raise the error.

A str that a later call unsets or replaces is no longer in the request, in the same cases as in
PySpark. A stream reader's option of None leaves its option as it was, as PySpark's does. An option
key, a table property or a config key that is not a str fails at the same call as PySpark's, with
the same error: an `AssertionError` at a reader's load of a path, and protobuf's `TypeError` where
the request is built otherwise.

The function of a UDF, of a pandas or Arrow UDF of any eval type, of a UDTF (called, of a table
argument, in a lateral join) and of `mapInPandas`, `mapInArrow`, `applyInPandas`, `applyInArrow`,
the cogroup's `applyInPandas` and `applyInArrow`, `applyInPandasWithState`,
`transformWithStateInPandas` and `transformWithState` is pickled each time a request builds a plan
that holds it, as PySpark's `PythonUDF.to_plan` and `PythonUDTF.to_plan` pickle it. So a value that
the function reads, changed after the function was applied and before an action, is the value that
the action runs with, and so for each later action. A function that does not pickle fails each
request of the plan, from the first, such as `df.schema`, with PySpark's error, the `PicklingError`
of `CloudPickleSerializer.dumps`, or `UDTF_SERIALIZATION_ERROR` of a UDTF, and no request is sent;
the call that applies it does not fail. The functions are pickled in the order in which PySpark
builds the plan, the initial state of `transformWithState` before its function and the arguments of
a UDTF before its handler, and the first error of the build, of a pickling or of a value that the
plan does not take, such as a str above, is raised. `spark.udf.register` and `spark.udtf.register`
are commands, which pickle the function when they are called, as PySpark's do. A copy of a Column, a
DataFrame, grouped data or a table argument holds the functions, which the requests of the copy
pickle, as PySpark's: `copy.copy` shares them, and `copy.deepcopy` deep-copies each with the `memo`
of the call, so a function, a lambda or a closure is the function itself, and a callable object, a
bound method or a `functools.partial` is a copy, which the copies of one call share, as the two
Columns of a UDF of `copy.deepcopy([c1, c2])` do. A request warns once of a window of no partitions,
as PySpark's, though the drop-in builds the plan of a request twice when it has such functions, as
it pickles them between the two builds: so where a function that does not pickle comes before such a
window in the build, the drop-in warns of the window before it raises the error of the pickling,
where PySpark raises it without the warning.

A pickled Column or DataFrame, and a deep copy of a DataFrame into another session, whose session
copies to another one (of a `memo` that maps it, or another active session of its ID), carry the
commands that their functions pickled to when they were pickled or copied, as protobufs that the
server's Python worker unpickles too, where PySpark's carry the functions, which their requests
pickle again. So a value that a function reads, changed after that, is seen by PySpark's such deep
copy, and by its unpickled Column of a function that `pickle` pickles by reference, such as a
function of `__main__`, but not by the drop-in's. `pickle.dumps` of a Column of a UDF of a function
that does not pickle by value raises the `PicklingError` of its command, where PySpark's pickles a
module-level function by reference, and `pickle.dumps` of the Column of a UDF of a lambda pickles,
where PySpark's raises the `PicklingError` of the lambda. `CloudPickleSerializer.dumps` of such a
Column raises the same error in both.

A DataFrame or a Column reports the functions of its UDFs to Python's garbage collector, as
PySpark's plan of Python objects is seen by it, so a reference cycle through a function and the
DataFrame or the Column that holds it, such as through a bound method of an object that keeps the
DataFrame, is collected. A function that a Column and a DataFrame of it share, or two DataFrames,
is reported by neither, so that no reference is counted twice: a cycle through it is freed only
when it is broken, where PySpark collects it.

The condition and the assignments of a clause of `mergeInto` are pickled when the clause is added,
which raises there, as PySpark's, and again when the merge runs, whose request carries the later
pickle, where PySpark's carries the one of the clause.

The drop-in has one class for each UDF and UDTF, the Connect class of `pyspark.sql.connect.udf`
and `pyspark.sql.connect.udtf`, which the classic names `pyspark.sql.udf.UserDefinedFunction` and
`pyspark.sql.udtf.UserDefinedTableFunction` also name. So `isinstance` of a UDTF that `udtf` builds
is True against either name, where PySpark 4.3.0 gives False for the classic name, whose class
needs a SparkContext. `udf` gives a function in both clients, which is of neither class.

So it is for the objects of a session: each is of PySpark Connect's class, of its module of
`pyspark.sql.connect`, as `repr`, `type(x).__module__` and pickle show it. These are the
SparkSession, its `conf`, `catalog`, `read`, `readStream`, `streams`, `udf`, `udtf`,
`dataSource`, `tvf` and `client`, and a DataFrame's `write`, `writeTo`, `writeStream`,
`mergeInto` and its clauses, `na`, `stat`, `groupBy`, `cogroup`, `asTable` and the WindowSpec of
its windows, and an `Observation`. So are `SparkSession.Hook`, nested in the session class, and
the classes of `pyspark.sql.connect.client` (its channel builders, `PlanObservedMetrics` and
`SparkConnectClient`), which are of `pyspark.sql.connect.client.core`.
The functions of `pyspark.sql.connect.functions` are of its modules (`builtin` and
`partitioning`), and so are those of `pyspark.sql.connect.avro.functions` and
`pyspark.sql.connect.protobuf.functions`: they are not those of `pyspark.sql.functions`,
`pyspark.sql.avro.functions` and `pyspark.sql.protobuf.functions`, as in PySpark. An alias, such
as `column`, `negate`, `power` or `random`, is the function `col`, `negative`, `pow` or `rand`, of
that name, as PySpark binds it.

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
`pyspark.sql.merge` and `pyspark.sql.observation` are PySpark's, of those modules of Apache Spark
as they are, so `isinstance` of a session's `conf`, `tvf`, `dataSource` and `mergeInto` against
them is False, as in PySpark. `Observation(...)` makes an Observation of
`pyspark.sql.connect.observation`, which is not an instance of `pyspark.sql.Observation`, as
PySpark's `__new__` makes one. `pyspark.sql.connect.window.Window` is a subclass of
`pyspark.sql.window.Window`, as in PySpark. For the other objects of a session, the classic name
names PySpark Connect's class, so `isinstance` against it is True, where PySpark gives False:
`Catalog`, `DataFrameReader`, `DataFrameWriter`, `DataFrameWriterV2`, `DataStreamReader`,
`DataStreamWriter`, `StreamingQueryManager`, `StreamingQuery`, `UDFRegistration`,
`UDTFRegistration`, `GroupedData` and `PandasCogroupedOps`, and the UDTFs above. The classic
`DataFrameNaFunctions`, `DataFrameStatFunctions`, `WindowSpec` and `TableArg` name PySpark
Connect's classes too, which in PySpark are their subclasses: `isinstance` is True in both, and
the MRO of such an object is its native class and `object`. The concrete DataFrame and Column are
not subclasses of the public `pyspark.sql.DataFrame` and `pyspark.sql.Column`: `isinstance` holds
by registration, so the MRO of a DataFrame is its native class and `object`, and
`pyspark.sql.Column` names the concrete class.

A `copy.copy` or `copy.deepcopy` of a DataFrame requests its schema as PySpark's does, one request
for the copy, and one more for the DataFrame in a deep copy when it has no schema yet, so a stopped
session fails the copy with `SparkConnectException`, as PySpark's does. One difference remains:
PySpark's `DataFrame.__getattr__` answers `getattr(df, "__deepcopy__")` by reading the columns, which
the drop-in does not, since it defines `__deepcopy__` itself.

The other objects of a session above, a streaming query and `SparkSession.builder` keep the
attributes that are set on them through copies and pickles, as PySpark's copy and pickle their
`__dict__`: a shallow copy shares them, and a deep copy, a pickle and cloudpickle copy them, where
they copy the object at all. Where PySpark's fail, the drop-in's fail with the same errors, such as
the `AttributeError` of the creator of the session of a plain pickle of `spark.streams`, and the
`TypeError` of its lock of a deep copy and of cloudpickle. Their `__dict__` holds only those
attributes, where PySpark's holds the state of the object too, as the `_options` of a builder: so a
shallow copy of a builder has options of its own, where PySpark's shares its `_options` dict.

A table name that is neither a str nor bytes, and bytes that are not UTF-8, give a DataFrame of
`read.table`, `readStream.table` and `changes`, whose plan raises protobuf's `TypeError` or the
decode error when it is built, as PySpark's do. A reader's format that is not a str is kept as the
object it is, which `_format` gives back, and a copy or a pickle of the reader keeps it.
The Connect builder's `remote` keeps a value that is not a str as PySpark's does, and `create`
rejects it as an invalid Spark Connect URL, as PySpark's `create` does for its `str()`. An explicit
`None`, given as `remote(None)`, `remote(location=None)` or `config("spark.remote", None)`, is
kept as None: `create` and `getOrCreate` fail PySpark's assertion, unless a session exists, which
`getOrCreate` returns. A later `remote` replaces it. A `remote()` without an argument is
`sc://localhost`, as in PySpark's Connect builder.

The client honors the HTTP proxy of the environment (`grpc_proxy`, `https_proxy`, `http_proxy`,
`no_grpc_proxy`, `no_proxy`) with grpcio's rules, see [HTTP Proxies](configuration.md#http-proxies),
with two differences: a loopback host is never proxied, where grpcio proxies `localhost` unless
`no_proxy` lists it, and a proxy address with an invalid host or port is an error, where grpcio
cannot connect (and takes a port past 65535 modulo 65536).

As in grpcio, a connection has 20 s, its connect deadline, to complete: the TCP connection and the
TLS handshake, and the tunnel of a proxy. One that does not fails with `UNAVAILABLE`, which the
retry policy retries. One difference remains: grpcio also waits at most 20 s for the server's
HTTP/2 settings, while here a plaintext connection to a server that accepts it and never answers
ends at the keepalive of the connection, 80 s by default (`grpc_keepalive_time_ms` and
`grpc_keepalive_timeout_ms`), or at the deadline of the call without a keepalive.

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

`spark_connect::ml::FMClassifier` and
`pyspark.ml.connect.classification.FMClassifier` fit Spark's binary factorization
machine classifier. All 18 PySpark constructor parameters have Rust builder
setters/getters and named Python setters. Fitted models produce prediction,
probability and raw-prediction columns, with configurable thresholds. Both the
`gd` and `adamW` solvers run on the Spark server.

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

`spark_connect::ml::MultilayerPerceptronClassifier` and its native Python binding
support all 14 PySpark constructor parameters, including `thresholds`. Configure
`layers` before fitting; input and output sizes must match the feature dimension
and number of classes. `initialWeights` accepts dense or sparse ML vectors:
Rust uses `MlVector` (a `Vec<f64>` or array also converts to a dense vector), and
Python accepts the values supported by PySpark's `TypeConverters.toVector`.
Layer arrays and vector parameters are encoded in Rust and checked against
fixtures captured with official PySpark.

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
  error in Rust), where PySpark allocates its list of that many tasks, and beyond 2^63 the
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
the metadata files equal PySpark's apart from their timestamps and uids. As PySpark's readers, the
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
RandomForestClassification one, and `trees` of a GBTClassification one. This client makes the same
reads, but after all the class fits rather than between them, so their order differs. The remaining
differences from PySpark are these:

- A `OneVsRest` or a `OneVsRestModel` whose classifier or submodel is not writable raises
  `ValueError` on `write`, as PySpark's `validateParams` does. A `OneVsRest` of no classifier
  raises the `KeyError` of its unset param on `write`, as PySpark's does.

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
PySpark's AttributeError, and a subclass that sets `_metrics` summarizes of them. The methods have
PySpark's parameters, without its annotations and docstrings, and only `SummaryBuilder` has
PySpark's class docstring.

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

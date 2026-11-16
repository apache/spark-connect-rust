# Compatibility

This client aims for **full API parity** with the reference PySpark Spark Connect
client (`pyspark.sql.connect.*`) and the shared modules it depends on. The goal is
that existing Spark Connect code runs unchanged and gets byte-for-byte identical
results. For how this client relates to the `pyspark` and `pyspark-client` packages
on PyPI - and how to tell which one you have installed - see
[Which client am I using?](which-client.md).

## Supported Spark

Apache Spark **4.3.0 and later**. The crate and wheel version tracks the Spark
release it targets (starting at `4.3.0`), so the version number tells you which
Spark it speaks.

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
  `predict(&vector)`, the members of every prediction model of `pyspark.ml`.
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
  does. `clear` does nothing over Spark Connect, as PySpark's `JavaParams.clear`. The native
  object's params are made those of the maps before each native member reads them (a fit, a
  transform, an evaluation, a writer, a model's members), and so are those of the objects that it
  holds (an OneVsRest's classifier, a Pipeline's stages, a validator's estimator, evaluator and the
  values of its grids, a PipelineModel's stages, an OneVsRestModel's models, a validator model's
  best and sub-models). A writer takes them at `write()`, where PySpark's reads them at its `save`.
  The maps of an object that this client makes (of a fit, a load, a native copy) are made of the
  native object's values at their first read. The native members have PySpark's docstrings and
  signatures. The Param objects are PySpark's own `pyspark.ml.param.Param`, of the docs,
  converters and defaults of the official class (`crates/pyspark-rs/src/ml/ml_params.json`,
  captured from pyspark-client 4.3.0 by `scripts/capture_ml_api_golden.py`), and a getter per
  param (`getMaxIter`).
- **PySpark's types and class Params**: the class of `pyspark.ml.connect.<module>` is a subclass
  of its native class and of PySpark's class of the same name of `pyspark.ml.<module>`
  (`python/pyspark/ml/_native_classes.py`), so its objects are instances of every class of the MRO
  of PySpark's class: `MLWritable`, `MLReadable` and `Identifiable`, `HasTrainingSummary` and the
  summary bases, the param mixins (`HasMaxIter`), `PredictionModel` and `ClassificationModel`,
  PySpark's Java wrappers, and PySpark's class itself. Every object that this client makes (of a
  fit, a load, a copy, a summary) is of that class, and a class of the same module that is a base
  of PySpark's class (`LinearSVCSummary` of `LinearSVCTrainingSummary`) has it as a virtual
  subclass. So `isinstance(LinearSVCModel(), MLWritable)` holds, and a Pipeline of `pyspark.ml`
  fits and saves native stages. The Param of a class (`LinearSVC.maxIter`) is PySpark's, of
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
  (`RFormulaModel(ResolvedRFormula(label=y, terms=[x], hasIntercept=true)) (uid=...)`).
- **Summaries**: the training summary of a fit (`summary`, `hasSummary`), the same object on every
  call as PySpark's `_summary`, and the summary of `evaluate`; a loaded model has none, and raises
  PySpark's `RuntimeError`. Their DataFrames (`predictions`, `roc`, `cluster`, ...) are of the
  summary dataset of the fit. The `__repr__` of a GeneralizedLinearRegressionTrainingSummary is the
  server's description of it (its table of coefficients and its deviances), as PySpark's is.
- **Fits of param maps**: `fit(dataset, paramMap)` fits a copy of the estimator with the map, a
  list or tuple of maps fits once per map, and `fitMultiple(dataset, paramMaps)` returns PySpark's
  `_FitMultipleIterator`. `transform(dataset, paramMap)` transforms of a copy of the model.
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
  - An object of a subclass of an estimator, transformer or evaluator has a uid of the subclass's
    name (`Identifiable._randomUID`), and the defaults of that uid and name (`outputCol`, `seed`).
    A subclass of a model class keeps the uid of the model class's name.
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
    copied deep nor pickled: PySpark's raise the TypeError of the lock of its `RemoteModelRef`, and
    the pickle of a summary the AttributeError of its DataFrames' session.
  - A shallow copy of a model keeps the server-side object while either object lives. PySpark's
    shares its `RemoteModelRef` without a reference of its own, so the garbage collection of the
    copy releases the object of the original (`CONNECT_ML.CACHE_INVALID` at its next use).
- **A model of no server-side object**: a model class constructs as PySpark's
  `JavaModel(java_model=None)` (`KMeansModel()`, `LDAModel()`), a model of a new uid of its class
  and of its params, none set, whose Params API, `copy` and `write()` work, and whose members of
  the server, `__repr__`, a transform and a save raise PySpark's bare `AssertionError`, as does the
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
  `compute_cost(&dataset)` of a bisecting model, and the training summary of a fit.
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

The native estimators, transformers and evaluators of `pyspark.ml.connect` take
keyword-only constructor parameters. The evaluators follow standard
`pyspark.ml.evaluation`: multiclass classification defaults to `metricName="f1"`,
and binary classification uses `rawPredictionCol="rawPrediction"`.
Each value is converted by the `TypeConverters` of its param in
PySpark (an integral float for an int param, for example), and a value that the
converter rejects raises its `TypeError`. A fit, transform or evaluate sends only the
params that the constructor or a setter set, so the server applies its defaults to
the others, as with PySpark. An int beyond a long raises PySpark's
`VALUE_NOT_BETWEEN` at the next fit. `None` is sent as a null literal, as PySpark sends
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
contract of standard `pyspark.ml` over Spark Connect. Passing a pandas DataFrame raises
`TypeError` before the params of the class are validated, the schema is read or any request is
sent (the helper calls below raise PySpark's error of its serialization); only a `params` argument that is not a param map is checked before it (see above). This
also applies to empty pandas DataFrames. The native Pipeline and validators raise the errors of
PySpark's: a Pipeline that of its first stage, and a validator the `AttributeError` of the
`select` of its split.
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
  DataFrame, which raises `UNSUPPORTED_DATA_TYPE` here rather than the `TypeError` above, as with
  `pyspark.ml`, and an empty list, which raises `CANNOT_INFER_ARRAY_ELEMENT_TYPE`.
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
disagreement raises the getters' error.
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

  PySpark loads what this client saves, and this client loads what PySpark saves. A stage loads
  as the native class of `pyspark.ml.connect` of its name where one loads, else as PySpark's
  class. An overwrite deletes the path by the ML helper first, as `RemoteMLWriter` does.

The vendored `pyspark.ml.Pipeline` and the validators of `pyspark.ml.tuning` fit, transform and
save native stages, which are `MLWritable` (see "PySpark's types and class Params"). Their loads
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
  `TransformerStage`, `EvaluatorStage`).
- **Param maps**: a `ParamMap` holds values of params of stages, by the uid of a stage and the
  name of its param, as PySpark's `Param.parent` and `Param.name`. `ParamGridBuilder` builds
  PySpark's grids (`add_grid`, `base_on`, `build`, the last param varying fastest), and
  `apply_param_map` of an estimator, a transformer or an evaluator sets the values of its params,
  as a copy of a param map of PySpark does: each value is converted by the type converter of its
  param (`Param.typeConverter`, so an integer grid value of a double param is a double), and a
  param that the stage's class does not have is not set. A pipeline applies them to its stages.
  The grid keeps the values as they were given, and a validator converts them when it applies or
  saves them, not when the grid is built as PySpark does, so a value that PySpark refuses at build
  (a `TypeError`) is refused at the fit or the save of the validator.
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
  (`collect_sub_models`), and transform by the best model. A validator's `apply_param_map` sets
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

  `load_stage` loads any saved stage, of its class: a meta-algorithm, or an estimator,
  transformer or model of the server's class (`OperatorEstimator`, `OperatorTransformer`,
  `ObjectModel`). The evaluator of a loaded validator is the evaluator of its class. A param map
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
transport. These modules build their own `spark.connect` messages with the
generated `pyspark.sql.connect.proto`, which needs the `protobuf` package
(`pip install "protobuf>=6.33.5"`). The package is not a dependency of
`pyspark-client-rust`: without it, `pyspark.ml` imports, but a fit or a transform
raises an `AttributeError` that names `protobuf`.

A DataFrame of a plan of `pyspark.ml` (`DataFrame(plan, session)`) takes the plan ID
of this client rather than the one of the plan, which its columns refer to.

## Spark Declarative Pipelines CLI

The CLI of Spark Declarative Pipelines (`python -m pyspark.pipelines.cli init`, `run` and
`dry-run`) and its modules are Apache Spark's own, vendored from v4.3.0, and Apache Spark's
tests of them pass with this client. The CLI needs `pyyaml` and `protobuf`, which, as for
`pyspark.ml`, are not dependencies of `pyspark-client-rust`. The user context extensions of a
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

The drop-in parses the protobufs that a pickled Column or DataFrame carries, and the requests
that session hooks see, at any depth that PySpark's serializer reaches, where the parsers of
protobuf and prost stop at 100 nested messages. Its Rust client enables prost's
`no-recursion-limit` feature, which applies to every message the drop-in decodes. So it also
decodes server responses deeper than 100 nested messages, such as the schema of a struct nested
34 levels deep, which PySpark's client rejects with "Exception deserializing response!". The
server is trusted, as SECURITY.md sets out, so this is not a bound on the responses.

An `Observation` that PySpark pickled at protocol 0 or 1 raises a `TypeError` when the drop-in
unpickles it, as CPython's `copyreg._reconstructor` cannot create the drop-in's `Observation`, a
class with a native base; protocols 2 and up, pickle's default, load in both clients, and PySpark
loads the drop-in's at every protocol.

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
- a key or a value of `SparkSession.builder.config`, or an `appName`, fails `getOrCreate`;
- a `DataStreamWriter`'s fails the call that takes it, as PySpark sets that writer's request as it
  goes.

A str that a later call unsets or replaces is no longer in the request, in the same cases as in
PySpark. A stream reader's option of None leaves its option as it was, as PySpark's does. An option
key, a table property or a config key that is not a str fails at the same call as PySpark's, with
the same error: an `AssertionError` at a reader's load of a path, and protobuf's `TypeError` where
the request is built otherwise.

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
coefficients), `evaluate`, and save and load. As with PySpark, the `repr` of a model whose `link` is
not set fails on the server.

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

In Python, the classifier can be any other object, as in PySpark. A classifier that is not a
native one, such as PySpark's own classes of `pyspark.ml.classification` or a Python subclass of
them, is fit by PySpark's own `OneVsRest._fit` over the estimator's params. Its model is PySpark's
`OneVsRestModel` of the classifier's own models. A None classifier is fit there too, and fails at
its first use. So is a dataset that is not a Spark DataFrame: a pandas DataFrame raises the
`KeyError` of PySpark's pandas aggregate of its max label.

All seven PySpark constructor parameters have named setters. Rust exposes scalar
parameter getters and the configured classifier. Fits clone the base classifier
for each class and run at most `parallelism` fits at once. Python observes
mutations to the supplied classifier made before `fit`, while fitting leaves that
classifier unchanged. Class indices run from zero through the maximum label, as
in PySpark. Instance weights are forwarded to each class's fit when the base
classifier is a `HasWeightCol` of PySpark, which every native classifier is but the
perceptron (FMClassifier included). For the perceptron, Python warns
`weightCol is ignored, as it is not supported by <uid> now.`, with the classifier's
uid, as PySpark does, and fits without the weights.

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
gives the classifier. Single-vector prediction is on the submodels.

Persistence is of PySpark's Python format, in Rust and in Python, as PySpark's `OneVsRestWriter`,
`OneVsRestModelWriter` and their readers write it: `OneVsRest::write().save(path)` and
`OneVsRest::load(session, path)`, and the same for `OneVsRestModel`. The metadata of the estimator
has its params without the classifier, which is saved at `classifier`; the metadata of the model
also has `numClasses`, and each submodel is saved at `model_<index>` by its own writer. What each
of PySpark and this client saves, the other loads, with the same uids, params and predictions, and
the metadata files equal PySpark's apart from their timestamps and uids. A `CrossValidator` or a
`TrainValidationSplit` over a OneVsRest saves and loads with it, and a param map of the validator
sets the params of the classifier, as PySpark's `copy` does.

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
`estimatedDocConcentration`), `toLocal` (the same `LocalLDAModel` on every call, as PySpark's),
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
`approxSimilarityJoin` and `findSynonyms`, and ours does too, as PySpark's behavior over Spark
Connect: a `distCol` given by keyword is ignored (the column stays `distCol`), and a call whose
positional arguments are not all given fails at the server, as official's does. `QuantileDiscretizer`'s fit returns a `Bucketizer` of a new uid, with its splits and
its input and output columns, as PySpark's `_create_model` does. Transformed DataFrames retain
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
As PySpark's constructor does, `StopWordsRemover()` fetches its default locale and English stop
words of the active session first, so without one it raises PySpark's bare `AssertionError`.
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
Pandas DataFrames are rejected before serialization or RPC. Prepare double
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

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
`Transformer::operator` and `Transformer::params`, and `LogisticRegressionModel::feature_col`,
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
(`MLUtils` of Spark 4.3.0). The native Python classes do not have these members yet: their fit
still returns the common `MLModel`.

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

A native class has the `setParams` of its `pyspark.ml` class. It is keyword-only, takes the params
of the constructor, sets the params that it is given, in order, and returns the object, as
PySpark's `_set` does. As PySpark's, the `setParams` of `LogisticRegression` then checks its
`threshold` and `thresholds`, whichever call set them, as its constructor does. Its signature is PySpark's for every class, the feature stages included, except
`CrossValidator`, which takes `numFolds`, `parallelism` and `seed`, with a default of `None` for
the first two. The `params` argument of a fit, transform or
evaluate, which `pyspark.ml` takes as a param map, is accepted: `None` and an empty dict leave the
params as they are. The native classes have no Param objects to key a map by, so a non-empty
map raises a `TypeError` that says the native classes take no param map. A list or tuple of maps to
a fit is fitted as `pyspark.ml` fits it: one model per map, in order, returned as a list,
so that `[]` fits nothing and does not read the dataset. Any other value raises the `TypeError` that
`pyspark.ml` raises for it. The `params` argument is checked first, before the dataset, as
`pyspark.ml` checks it.

Native `fit`, `transform` and `evaluate` require a Spark DataFrame, matching the dataset
contract of standard `pyspark.ml` over Spark Connect. Passing a pandas DataFrame raises
`TypeError` before the params of the class are validated, the schema is read or any request is
sent (the helper calls below raise PySpark's error of its serialization); only a `params` argument that is not a param map is checked before it (see above). This
also applies to empty pandas DataFrames and to the native Pipeline and CrossValidator
placeholders.
Convert local data explicitly with `df = spark.createDataFrame(pdf)` before calling
`estimator.fit(df)`, `model.transform(df)` or `evaluator.evaluate(df)`. Transforms return Spark
DataFrames; call `toPandas()` explicitly when a local result is needed. The deprecated
`pyspark.ml.connect` pandas contract is not the target for these native methods. The feature stages described below use the same Spark DataFrame contract.

Another object than a DataFrame raises the `AttributeError` of `sparkSession`, the attribute
that `pyspark.ml` reads of the dataset first (`'int' object has no attribute 'sparkSession'` of
`fit(5)`), before any request; an object that has a `sparkSession` raises a `TypeError`. The
methods that `pyspark.ml` runs as a helper or attribute call (`PowerIterationClustering.assignClusters`,
`PrefixSpan.findFrequentSequentialPatterns`, and `ALSModel.recommendForUserSubset` and
`recommendForItemSubset`) read nothing of their dataset. PySpark's serialization
(`pyspark.ml.connect.serialize`) converts any other object to a literal, as
`LiteralExpression._from_value` does:
- One that cannot be a literal raises PySpark's error of the conversion. That includes a pandas
  DataFrame, which raises `UNSUPPORTED_DATA_TYPE` here rather than the `TypeError` above, as with
  `pyspark.ml`, and an empty list, which raises `CANNOT_INFER_ARRAY_ELEMENT_TYPE`.
- PySpark sends the others for the server to reject. These include an ML vector or matrix and a
  `(value, DataType)` pair, which it sends as they are. Here they raise a `TypeError` before any
  request.

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
The Rust model's members and persistence are in "Native regression models (Rust
API)"; the native Python model does not have them yet.

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
last owner is dropped. The Rust model's members and persistence are in "Native
regression models (Rust API)"; the native Python model does not have them yet.

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
fitting session until the last owner is dropped. Multiclass training, model
inspection, per-iteration evaluation, single-vector prediction and persistence
are outside this native addition.

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
dropped. The Rust model's members and persistence are in "Native
regression models (Rust API)"; the native Python model does not have them yet.

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
fitting session until the last owner is dropped. Tree inspection, feature
importances, single-vector prediction and persistence are outside this addition.

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
last owner is dropped. Tree inspection, feature importances, training summaries,
single-vector prediction and persistence are not exposed by this native addition.

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
model alive until the last owner is dropped. Cluster-center access, training
summaries, single-vector prediction and persistence are not exposed by this native
addition. The vendored `pyspark.ml.clustering` API is unchanged.

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
the last owner is dropped. The Rust model's members and persistence are in "Native
regression models (Rust API)"; the native Python model does not have them yet.

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
The Python model uses the existing `MLModel.transform` interface. Model inspection
(coefficients, intercept and summaries), single-vector prediction and model
persistence are not exposed by this addition.

## Native Pipeline and CrossValidator: fit is not implemented

`spark_connect::ml::Pipeline` and `spark_connect::ml::CrossValidator`, and their Python
classes `pyspark.ml.connect.Pipeline` and `pyspark.ml.connect.CrossValidator`, do not fit.
Their `fit` raises `NOT_IMPLEMENTED` (`PySparkNotImplementedError` in Python), naming
`Pipeline.fit` or `CrossValidator.fit`, before any request is sent, and returns no model.
The Connect ML protocol cannot run either one: the native Pipeline holds stage names, which
the server cannot fit, and it has no pipeline model to transform; the native CrossValidator
holds its numeric params only, without the estimator, evaluator or param grid. Their
constructors and setters still set the params, as before.

To fit a pipeline or to tune a model from Python, use the vendored `pyspark.ml.Pipeline` and
`pyspark.ml.tuning.CrossValidator`, which take the estimator and transformer objects, as
PySpark does. The Rust API has no way to fit a pipeline or to tune a model.

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

## Native NaiveBayes

The Rust `ml::NaiveBayes` and `ml::NaiveBayesModel` train and predict through Spark
Connect's cached-model protocol. The Rust-backed Python API exports `NaiveBayes`
from `pyspark.ml.connect` and `pyspark.ml.connect.classification`; `fit` returns the
shared native `MLModel`, whose `transform` produces DataFrame predictions.

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

This native surface covers construction, parameter setters, fit and DataFrame
transform. Model coefficient/statistic inspection, single-vector prediction and
persistence are outside this surface. The vendored `pyspark.ml.classification`
API continues to use its existing implementation.

## Native BisectingKMeans

The Rust `ml::BisectingKMeans` and `ml::BisectingKMeansModel` support training and
DataFrame cluster assignment through Spark Connect's cached-model protocol. Python
exports the native estimator from `pyspark.ml.connect` and
`pyspark.ml.connect.clustering`; `fit` returns the shared native `MLModel`.

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

This native API covers construction, setters, fit and DataFrame transform.
Cluster-center/summary/cost inspection, single-vector prediction and persistence
are outside this surface. The vendored `pyspark.ml.clustering` API retains its
existing implementation.

## Native GeneralizedLinearRegression

`spark_connect::ml::GeneralizedLinearRegression` fits Spark's JVM estimator and
returns a cached `GeneralizedLinearRegressionModel`. Python exposes it from
`pyspark.ml.connect` and `pyspark.ml.connect.regression`; `fit` returns the shared
native `MLModel`.

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

This native API covers construction, setters, fitting and DataFrame transforms.
The Rust model's members and persistence are in "Native regression models (Rust
API)"; the native Python model does not have them yet.

## Native GaussianMixture

Rust `ml::GaussianMixture` and `ml::GaussianMixtureModel` train Gaussian mixture
models and produce cluster assignments and component probabilities through Spark
Connect's cached-model protocol. Python exports the native estimator from
`pyspark.ml.connect` and `pyspark.ml.connect.clustering`; `fit` returns the shared
native `MLModel`.

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

The native surface covers construction, parameter setters, fit and DataFrame
transform. Component/weight/covariance inspection, training summaries,
single-vector prediction and persistence are outside this surface. The vendored
`pyspark.ml.clustering` API retains its existing implementation.

## Native FMClassifier

`spark_connect::ml::FMClassifier` and
`pyspark.ml.connect.classification.FMClassifier` fit Spark's binary factorization
machine classifier. All 17 PySpark constructor parameters have Rust builder
setters/getters and named Python setters. Fitted models produce prediction,
probability and raw-prediction columns, with configurable thresholds. Both the
`gd` and `adamW` solvers run on the Spark server.

## Native MultilayerPerceptronClassifier

`spark_connect::ml::MultilayerPerceptronClassifier` and its native Python binding
support all 13 PySpark constructor parameters plus `setThresholds`. Configure
`layers` before fitting; input and output sizes must match the feature dimension
and number of classes. `initialWeights` accepts dense or sparse ML vectors:
Rust uses `MlVector` (a `Vec<f64>` or array also converts to a dense vector), and
Python accepts the values supported by PySpark's `TypeConverters.toVector`.
Layer arrays and vector parameters are encoded in Rust and checked against
fixtures captured with official PySpark.

The fitted model produces multiclass prediction, probability and raw-prediction
columns. Both `l-bfgs` and `gd` are supported. As in PySpark, fitting an empty
DataFrame can return a model; the native client preserves that server behavior.

## Native OneVsRest

`spark_connect::ml::OneVsRest` and `pyspark.ml.connect.classification.OneVsRest`
implement the multiclass reduction in Rust. Supply a native binary classifier,
for example `OneVsRest::new().set_classifier(LinearSVC::new())` in Rust or
`OneVsRest(classifier=LinearSVC())` in Python. The `Classifier` enum accepts all
eight native binary classifiers: LogisticRegression, LinearSVC,
DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, NaiveBayes,
FMClassifier and MultilayerPerceptronClassifier. Perceptron base classifiers need
a two-unit output layer.

All seven PySpark constructor parameters have named setters. Rust exposes scalar
parameter getters and the configured classifier. Fits clone the base classifier
for each class and run at most `parallelism` fits at once. Python observes
mutations to the supplied classifier made before `fit`, while fitting leaves that
classifier unchanged. Class indices run from zero through the maximum label, as
in PySpark. Instance weights are forwarded when the base classifier supports
them; Python warns when they are ignored for FMClassifier or the perceptron.

The model aggregates positive-class raw scores into `rawPrediction` and chooses
the first highest score for `prediction`, ignoring NaN/null scores as PySpark
does. It produces no probability column. Temporary columns and fit-created
caches are cleaned up, and transformed DataFrames retain the binary models for
lazy evaluation. Models must transform data in their fitting session.

These additions cover native estimator construction, parameters, fitting and
DataFrame transforms. Model coefficient/weight inspection, training summaries,
persistence and single-vector prediction remain separate API work.


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

This support covers construction, configuration, fitting and DataFrame
transforms. The Rust model's members and persistence are in "Native regression models (Rust
API)"; the native Python model does not have them yet.

Parameter goldens come from official PySpark 4.3.0. Native Rust and Python tests
exercise censoring and quantiles, weighted isotonic interpolation, both FM
solvers, validation errors and cached-model lifetimes. The Python behavior tests
also run unchanged against the official client.


## Native LDA and PowerIterationClustering

The native Rust and Python APIs now implement all five clustering algorithms in
PySpark 4.3.0: KMeans, BisectingKMeans, GaussianMixture, LDA and PowerIterationClustering.
This describes algorithm execution, not complete model-inspection or persistence parity.

Rust `ml::LDA` supports online and EM training and cached-model DataFrame transforms.
Python exposes `LDA` through `pyspark.ml.connect` and `pyspark.ml.connect.clustering`;
`fit` returns the shared native `MLModel`. All 14 constructor parameters have Python
setters and Rust builder setters/getters, including document/topic concentrations,
learning settings, checkpoint settings, seed and topic-distribution output column.
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
LDA topic matrices, likelihood/perplexity methods, topic descriptions, distributed-model
inspection/conversion and persistence are outside this native surface. The vendored
`pyspark.ml.clustering` API retains its existing implementation.

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

The 20 estimators fit through Spark Connect's ML command and return session-scoped cached
models; Python returns the shared native `MLModel`. The 18 transformers build lazy ML
relations. Transformed DataFrames retain their fitted models and upstream input plans.
`QuantileDiscretizer` returns a cached JVM `Bucketizer`; its Rust model wrapper is named
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

In Rust, each fitted model also has the members of its PySpark model class, which it reads of
the server-side model as PySpark does: for example `CountVectorizerModel::vocabulary`,
`PCAModel::pc`, the scaler statistics, `VectorIndexerModel::category_maps`, the LSH models'
`approx_nearest_neighbors` and `approx_similarity_join`, and `Word2VecModel::get_vectors` and
`find_synonyms`. `CountVectorizerModel::from_vocabulary` and `StringIndexerModel::from_labels`
and `from_arrays_of_labels` build a model of given values, and
`QuantileDiscretizer::fit_bucketizer` gives the `Bucketizer` of the fitted splits, as PySpark's
fit does. Every stage and model saves with `write()` and loads with `load(session, path)`, as
PySpark's `save` and `load`. The Python classes gain these members with the native model
classes. Pipeline/tuning execution is separate from this native surface. Existing vendored
helper classes remain available.

Tests compare all 38 stages with official PySpark 4.3.0 parameter literals and Spark results,
including 10 multi-column/parameter variants, both constructor and setter configuration,
model lifetime, invalid inputs and pandas rejection. Schema checks compare class identifiers,
nullability and ML metadata; Connect omits the backing SQL schema of JVM UDTs on the wire.
Run `scripts/capture_ml_features_golden.py` with the official client to regenerate the fixture.
`scripts/capture_feature_models_golden.py` captures the members of every fitted model, the class
methods, and the save and load of every stage, which `e2e_feature_models.rs` compares.
Native live tests run in CI; set `FEATURE_REFERENCE=1` explicitly to run their reference mode.


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

Rust `ml::PrefixSpan::find_frequent_sequential_patterns` and Python
`PrefixSpan.findFrequentSequentialPatterns` mine arrays of itemsets directly through
the Connect helper, with all four constructor parameters and named setters. They
return sequence/frequency DataFrames and have no fit/transform model.

These APIs accept Spark DataFrames and reject pandas frames before requests. Lazy
model tables retain their cached model; PrefixSpan retains its input plan, including
any upstream fitted models. Fitted-model transforms require the fitting session.
Persistence is outside this native addition. The vendored `pyspark.ml.fpm` module is
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

Fit, transform and subset recommendation inputs must be Spark DataFrames; pandas
DataFrames are rejected before conversion or RPC. Model operations must use the
fit session. Official PySpark 4.3.0 fixtures cover explicit and implicit ratings,
predictions, cold starts, factors, and all four recommendation methods, with live
Rust and native Python tests. Model persistence is separate from these APIs.


## Native ML evaluators

Rust and the Rust-backed `pyspark.ml.connect.evaluation` exports provide all six
standard Spark ML evaluators: regression, binary classification, multiclass
classification, multilabel classification, clustering and ranking. Their
constructor parameters and named setters follow `pyspark.ml.evaluation` in
PySpark 4.3.0. Rust also provides typed parameter getters. Metric direction is
available as `is_larger_better()` / `isLargerBetter()` without contacting Spark.

The three existing Python evaluator wrappers now use standard Spark ML defaults:
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
invalid inputs, and metric direction. Generic parameter-map APIs and evaluator
persistence remain separate work.


## Native ML statistics

Rust's `ml::{ChiSquareTest, Correlation, KolmogorovSmirnovTest, Summarizer,
SummaryBuilder, MultivariateGaussian}` and the native Python classes in
`pyspark.ml.stat` cover the standard ML statistics APIs from PySpark 4.3.0.
The Python module is a Rust-backed shim, replacing its vendored implementation,
except `MultivariateGaussian`, which stays PySpark's plain Python class.
The deprecated `pyspark.ml.connect` public API is excluded from statistics parity
work; statistics have no new export under that namespace.

The statistical tests return lazy Spark DataFrames. Chi-square supports both the
vector result and one row per feature (`flatten=True`); correlation supports
Pearson and Spearman; the KS test supports Spark's normal distribution, with
optional mean and standard deviation. Spark validates column types and
statistical arguments. As with `pyspark.ml.stat`, the dataset and the other
arguments are serialized as the arguments of a helper call, in that order, before
any RPC (see the native ML helper calls above): a pandas DataFrame raises PySpark's
`UNSUPPORTED_DATA_TYPE`, an argument that cannot be a literal raises the error of its
conversion, and a literal of another type than PySpark sends for the server to reject
raises a `TypeError`. The KS test converts its parameters with `float()` first, as
PySpark does. Results retain any fitted models used by their input plans.

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

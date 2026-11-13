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
statistics, becomes `df.summary(Vec::<&str>::new())`. The same goes for `load_paths`. The optional
`subset` arguments of the null functions, the optional `value_vars` of `melt` and the optional
`column_names` of `drop_duplicates` and `drop_duplicates_within_watermark` stay
`Option<Vec<&str>>`, as in v4.2.0, so that `None` still infers its type. The optional `values` of
`unpivot` takes any iterable, as in v4.2.0, so its `None` has no item type to infer, unlike that
of `melt`: pass `None::<Vec<Column>>` or `None::<[&str; 0]>` to unpivot every column that is not
an id.

`BinaryClassificationEvaluator::set_score_col` and `score_col`, and the Python
`scoreCol=` argument and `setScoreCol` of v4.2.0 are removed, without an alias:
Spark's evaluator has no `scoreCol` param, so no value of it worked on the server.
Use `set_raw_prediction_col` and `raw_prediction_col` in Rust, the column of raw prediction
vectors that the evaluator scores. In Python it is `probabilityCol=`, as in PySpark, and
`setProbabilityCol`, a setter of this client (PySpark's evaluators have none): the column of
probabilities, which the server reads as its raw prediction column. A model without a
probability column, such as the native `LinearSVC`, is scored with
`probabilityCol="rawPrediction"`.

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

## Native LogisticRegression

The native `spark_connect::ml::LogisticRegression` fits Spark's JVM
`org.apache.spark.ml.classification.LogisticRegression` through the Connect ML
protocol. In the Rust API, training features must be ML vectors; convert an array column
with `array_to_vector` before fitting. The Python class takes an array column of numbers too,
as the deprecated `pyspark.ml.connect.classification.LogisticRegression` of PySpark does (see
below), but fits the JVM estimator, where PySpark's trains with PyTorch. Its constructor takes
the params of PySpark's that the JVM estimator honors, `featuresCol`, `labelCol`,
`predictionCol`, `probabilityCol`, `maxIter` and `tol`, whose defaults are PySpark's, and it has
their setters. The params of PySpark's PyTorch training, `numTrainWorkers`, `batchSize`,
`learningRate`, `momentum` and `seed`, have no counterpart in the L-BFGS of the JVM estimator,
and raise the `TypeError` of an unexpected keyword argument. Its model outputs the columns of
PySpark's: a `bigint` prediction, and then the `array<double>` of the probabilities unless
`probabilityCol` is empty, and no raw prediction: the constructor sends an empty
`rawPredictionCol`, a param of the JVM estimator that PySpark's does not have. The
probabilities are the JVM's doubles, where PySpark's are of float32 tensors. As PySpark's, its
fit reads only the features and the labels, so that a DataFrame with a `prediction`,
`probability` or `rawPrediction` column is fitted.

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

The Python classes of the native estimators, transformers and evaluators of
`pyspark.ml.connect` take their params as PySpark's do. The constructors are
keyword-only, raising PySpark's `TypeError`, except `MulticlassClassificationEvaluator`,
which takes `metricName`, `labelCol` and `predictionCol` by position as well, as
PySpark's does; its `metricName` is `accuracy` by default, as in PySpark, where the JVM
evaluator and the Rust API default to `f1`. The `probabilityCol` of
`BinaryClassificationEvaluator` is `probability` by default, as in PySpark, and is sent as the
`rawPredictionCol` that the JVM evaluator scores.
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

Native `fit`, `transform` and `evaluate` require a Spark DataFrame, matching the dataset
contract of standard `pyspark.ml` over Spark Connect. Passing a pandas DataFrame raises
`TypeError` before parameter validation, schema access or any server request. This also applies
to empty pandas DataFrames and to the native Pipeline and CrossValidator placeholders.
Convert local data explicitly with `df = spark.createDataFrame(pdf)` before calling
`estimator.fit(df)`, `model.transform(df)` or `evaluator.evaluate(df)`. Transforms return Spark
DataFrames; call `toPandas()` explicitly when a local result is needed. The deprecated
`pyspark.ml.connect` pandas contract is not the target for these native methods. The vendored
Python scalers described below retain their upstream behavior.

Other non-DataFrame objects retain their existing errors: a missing `select` in a fit or
an evaluate, or `columns` in a transform, raises `AttributeError`; an object with that attribute
raises `TypeError`. LogisticRegression still selects only its features and labels for fitting.
An output column that overlaps either input raises the JVM's error during fitting. Its model
checks for existing output column names before constructing the transform; other native models
leave output validation to their JVM models.

The fit of `LogisticRegression` checks its labels before it fits, as PySpark's `_fit` does, on the
distinct labels of the server's `collect_set` of the label column (a null label is not one of them):
there must be at least two, each an integer in `[0, numClasses)`, else PySpark's `ValueError` is
raised. The other checks of PySpark's fit (the row count, the size of the features) are not made, as
they do not raise. The evaluators cast a prediction or a probability of another numeric type to a
double, and send an array of probabilities as a vector (`array_to_vector`), whose element 1 is
scored, as PySpark's evaluator reads the column 1 of an array. The values are the JVM evaluators'
doubles, where PySpark's are of float32 tensors (`0.6` where PySpark returns `0.6000000238418579`).
The native estimators accept Spark DataFrames with vector or numeric array feature columns.
A fit and the transform of its model read the DataFrame schema and send an array `featuresCol`
as the vector (`array_to_vector`) that the JVM reads; the transform preserves the input array
column in its output. Standard `pyspark.ml` estimators require a `VectorUDT` feature column.

The scalers of `pyspark.ml.connect`, `StandardScaler` and `MaxAbsScaler`, are PySpark's own classes
(`pyspark.ml.connect.feature`), vendored unchanged: they fit with PySpark's `summarize_dataframe`
and transform with PySpark's closures, on the server for a Spark DataFrame and locally for a pandas
DataFrame, so their values, types and errors are PySpark's, and they send no ML request. The Rust API keeps its `StandardScaler` and `MaxAbsScaler`,
the JVM's scalers, whose `StandardScaler::set_with_mean` centers the features (`withMean`, false by
default).

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
Tree inspection, feature importances, single-vector prediction and persistence are
not exposed by this native addition.

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
last owner is dropped. Tree inspection, per-iteration evaluation, single-vector
prediction and persistence are outside this native addition.

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
dropped. Tree inspection, feature importances, single-vector prediction and
persistence are outside this addition.

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
the last owner is dropped. Model inspection (coefficients, intercept and summaries),
single-vector prediction and persistence are not exposed by this native addition.

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
Coefficient/intercept access, model summaries, evaluation, persistence and
single-vector prediction are not exposed by this addition.

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
keyword-only constructor signatures and type conversion; numeric arrays are
converted to ML vectors by the existing native fit/transform adapter.

This support covers construction, configuration, fitting and DataFrame
transforms. Python returns the common `MLModel`. Specialized model inspection
(e.g. AFT coefficients/scale, isotonic boundaries/predictions, FM factors),
single-instance prediction and model persistence are not exposed by these
native wrappers.

Parameter goldens come from official PySpark 4.3.0. Native Rust and Python tests
exercise censoring and quantiles, weighted isotonic interpolation, both FM
solvers, validation errors and cached-model lifetimes. The Python behavior tests
also run unchanged against the official client.

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

`BinaryClassificationEvaluator::set_score_col` and `score_col`, and the Python
`scoreCol=` argument and `setScoreCol` of v4.2.0 are removed, without an alias:
Spark's evaluator has no `scoreCol` param, so no value of it worked on the server.
Use `set_raw_prediction_col` and `raw_prediction_col` (Python `rawPredictionCol=` and
`setRawPredictionCol`), the column of raw prediction vectors that the evaluator scores.

The native ML classes take a JVM `Int` param as an `i32`: `LogisticRegression::set_max_iter`,
`LinearSVC::set_max_iter` and `LinearSVC::set_aggregation_depth` take an `i32`, and
`LogisticRegression::max_iter` returns one. Every native estimator has a getter of each
of its params (`KMeans::k`, `RandomForestClassifier::num_trees`, ...), which reads its
`Params`: the value that is set, else Spark's default (`None` for `seed`, `weight_col` and
`thresholds`). The fields of `LogisticRegressionModel`, `StringIndexerModel` and
`MaxAbsScalerModel` are private, as those of the other fitted models: use
`Transformer::operator` and `Transformer::params`, and `LogisticRegressionModel::feature_col`,
`label_col` and `prediction_col`. `VectorAssembler::input_cols` and `Pipeline::stages`
return a `Vec<String>`.

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
value filled string columns with its debug text. `fillna`, `fillna_double`,
`fillna_string` and `fillna_bool` are unchanged.

A transport failure has the gRPC code that grpc-python gives it, which PySpark's errors and
retries see. `SparkError::grpc_code` is `UNAVAILABLE` (14) for a connection that closes or is reset
under a call, that the server ends with a GOAWAY, or that closes before sending a call, where it
was `UNKNOWN` (2) or `CANCELLED` (1). It is `INTERNAL` (13) for a stream that the server resets with
a reason that tonic leaves unmapped, such as `STREAM_CLOSED`, which is no longer retried.

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
`create_dataframe` takes it for a year-month interval column. Downstream exhaustive
matches on these enums must handle the new variant. The Python client still raises
`NOT_IMPLEMENTED` when it collects a year-month interval, as PySpark does.

`DataFrame::repartition_by_range` and `repartition_by_range_opt` return a `Result`.
As PySpark's `repartitionByRange`, a partition count that is not positive is
`VALUE_NOT_POSITIVE` and an empty column list `CANNOT_BE_EMPTY`. Before, the server
split the rows round-robin for an empty column list. `StatFunctions::sample_by`
returns a `Result` too: a stratum that is not a literal is
`DISALLOWED_TYPE_FOR_CONTAINER`, as PySpark rejects a key that is not a float, an int
or a str. Before, it was sent as an empty literal, which failed only when the
DataFrame ran.

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

## Native LogisticRegression

The native `spark_connect::ml::LogisticRegression` fits Spark's JVM
`org.apache.spark.ml.classification.LogisticRegression` through the Connect ML
protocol. Training features must be ML vectors; convert an array column with
`array_to_vector` before fitting. This differs from the deprecated Python
`pyspark.ml.connect.classification.LogisticRegression`, which uses array features
and PyTorch training.

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
PySpark's does. Each value is converted by the `TypeConverters` of its param in
PySpark (an integral float for an int param, for example), and a value that the
converter rejects raises its `TypeError`. A fit, transform or evaluate sends only the
params that the constructor or a setter set, so the server applies its defaults to
the others, as with PySpark. An int beyond a long raises PySpark's
`VALUE_NOT_BETWEEN` at the next fit. `None` leaves a param unset, so the server's
default applies; PySpark sends a null, which the server rejects.

## Native RandomForestClassifier

`spark_connect::ml::RandomForestClassifier` trains Spark's JVM random forest
classifier through Connect with ML vector features and contiguous numeric class
labels starting at zero. Rust builder setters cover all 22 PySpark constructor
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

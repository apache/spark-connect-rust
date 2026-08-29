# DataFrames

DataFrames are the core abstraction in Spark Connect-immutable, distributed tables. Create them from ranges, SQL, external data, or in-memory collections, then transform and aggregate them using a fluent API.

## Creating DataFrames

```rust
use spark_connect::SparkSession;

let spark = SparkSession::builder()
    .remote("sc://localhost:15002")
    .get_or_create()?;

// From a range
let df = spark.range(100)?;

// From SQL
let df = spark.sql("SELECT 1 as id")?;

// From external source (planned)
// let df = spark.read().parquet("path/to/file.parquet")?;

// From in-memory data (planned)
// let df = spark.create_data_frame(...)?;
```

!!! note
    Remote session requires a Spark Connect server running at `sc://localhost:15002`.

## Transformations

Transform DataFrames with chainable operations like `select`, `filter`, and `with_column`.

```rust
use spark_connect::{functions as f, lit};

// Select columns
let df = df.select([f::col("id"), f::col("name")]);

// Filter rows
let df = df.filter(f::col("id").gt(lit(10)));

// Add/modify column
let df = df.with_column("double_id", f::col("id") * lit(2));

// Rename column
let df = df.with_column_renamed("id", "row_id");

// Drop columns
let df = df.drop(vec!["temp_col"]);

// Distinct rows
let df = df.distinct();

// Sort
let df = df.order_by(vec![f::col("id").expression().clone()]);

// Limit
let df = df.limit(5);
```

## Grouping and Aggregation

Group rows and compute aggregates like count, sum, and average.

```rust
use spark_connect::{functions as f, lit};

// Group by one column
let agg_df = df.group_by([f::col("category")])
    .agg(vec![f::count(lit(1)).expression().clone()]);

// Group by multiple columns with multiple aggregates
let agg_df = df.group_by([f::col("category"), f::col("year")])
    .agg(vec![
        f::sum(f::col("amount")).expression().clone(),
        f::avg(f::col("value")).expression().clone(),
        f::min(f::col("price")).alias("min_price").expression().clone(),
    ]);
```

## Joins

Combine DataFrames on shared keys.

```rust
use spark_connect::{functions as f, plan::JoinType};

// Inner join (default)
let joined = df1.join(
    &df2,
    Some(f::col("df1.id").eq(f::col("df2.id"))),
    JoinType::Inner,
);

// Left outer join
let joined = df1.join(&df2, Some(f::col("id").eq(f::col("id"))), JoinType::LeftOuter);

// Cross join
let crossed = df1.join(&df2, None, JoinType::Cross);
```

## Set Operations

Combine or deduplicate across DataFrames.

```rust
// Union (stacks rows, allows duplicates)
let combined = df1.union(&df2);

// Union by name (aligns columns)
let combined = df1.union_by_name(&df2);

// Except (rows in df1 not in df2)
let diff = df1.except_all(&df2);
```

## Actions

Execute and retrieve results.

```rust
// Display first n rows
df.show(10)?;

// Collect all rows to driver
let rows = df.collect()?;

// Count rows
let count = df.count()?;

// Get first n rows
let first_rows = df.take(5)?;

// Get first row
let first = df.first()?;
```

## Example Pipeline

```rust
use spark_connect::{functions as f, lit};

let result = spark.range(101)?
    .with_column("squared", f::col("id") * f::col("id"))
    .filter(f::col("squared").gt(lit(100)))
    .select([f::col("id"), f::col("squared")])
    .order_by(vec![f::col("id").expression().clone()])
    .limit(10);
result.show(10)?;
```

!!! tip
    See [Columns and Functions](columns-and-functions.md) for expression building, [SQL](sql.md) for SQL queries, and [Reading and Writing](data-sources.md) for I/O.

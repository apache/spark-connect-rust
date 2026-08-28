# Examples

Curated recipes for common Spark Connect tasks. See the repository's `examples/` directory for more runnable Rust examples.

## Read and Aggregate CSV

Read a CSV file and compute summary statistics:

```rust
use spark_connect::functions as f;

let df = spark.read()
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(Some("data.csv"));

df
    .group_by([f::col("category")])
    .agg(vec![
        f::sum(f::col("amount")).expression().clone(),
        f::avg(f::col("price")).expression().clone(),
    ])
    .show(20)?;
```

## Word Count

Classic MapReduce-style word count over a text column:

```rust
use spark_connect::{functions as f, lit_string};

let df = spark.sql(
    "SELECT * FROM VALUES ('hello world'), ('hello spark') AS t(line)"
)?;

df
    .select([f::explode(
        f::split(f::col("line"), lit_string(" "))
    ).alias("word")])
    .group_by([f::col("word")])
    .agg(vec![f::count(f::col("word")).expression().clone()])
    .show(20)?;
```

## Join Two DataFrames

Join two DataFrames on a common key:

```rust
use spark_connect::{functions as f, lit, plan::JoinType};

let users = spark.range(2)?
    .with_column("name", f::col("id"));
let orders = spark.range(3)?
    .with_column("user_id", f::col("id"))
    .with_column("amount", lit(100));

users
    .join(&orders, Some(f::col("id").eq(f::col("user_id"))), JoinType::Inner)
    .select([f::col("name"), f::col("amount")])
    .show(20)?;
```

## Write Parquet

Save a DataFrame to Parquet format:

```rust
let df = spark.range(1000)?;
df.write()
    .mode("overwrite")
    .format("parquet")
    .save(Some("/tmp/output"))?;
```

## SQL Query

Run a SQL query directly:

```rust
let df = spark.range(100)?;
df.create_or_replace_temp_view("numbers")?;

spark
    .sql("SELECT * FROM numbers WHERE id > 50")?
    .show(20)?;
```

## More Examples

Explore the repository's `examples/` directory for additional runnable examples, including streaming, complex transformations, and integration with external data sources.

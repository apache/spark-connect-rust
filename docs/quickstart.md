# Quickstart

Get up and running with Spark Connect in minutes. This guide walks you through your first Rust query.

## Prerequisites

Before you start, ensure you have a running Spark Connect server. See [Configuration and Connection](configuration.md) for how to start one locally:

```bash
$SPARK_HOME/sbin/start-connect-server.sh
```

The server listens on `sc://localhost:15002` by default.

## Your First Query

Connect to the server and run a simple query:

```rust
use spark_connect::SparkSession;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark = SparkSession::builder()
        .remote("sc://localhost:15002")
        .get_or_create()?;

    let df = spark.range(10)?;
    df.show(20)?;
    println!("Count: {}", df.count()?);

    Ok(())
}
```

## Filtering and Selection

Add a filter and select specific columns:

```rust
use spark_connect::{functions as f, lit};

let df = spark.range(100)?;
df
    .filter(f::col("id").gt(lit(50)))
    .select([f::col("id")])
    .show(20)?;
```

## Aggregation and Grouping

Compute aggregations over grouped data:

```rust
use spark_connect::{functions as f, lit};

let df = spark.range(20)?;
df
    .with_column("category", f::col("id") % lit(3))
    .group_by([f::col("category")])
    .agg([
        f::sum(f::col("id")).alias("total"),
        f::avg(f::col("id")).alias("average"),
    ])
    .show(20)?;
```

## Using it from an async runtime

The API is synchronous and can be called from any thread, including inside a Tokio runtime of
either flavor. The call blocks the thread it runs on until the server answers, so keep it out of
tasks that must keep the rest of their runtime running. A `current_thread` runtime, such as the
one of `#[tokio::test]`, works too: the query runs on a thread of its own while the calling
thread waits, and it keeps the tags and user context extensions of the calling thread.

## Next Steps

- [DataFrames](dataframes.md) - column operations, joins, window functions
- [SQL](sql.md) - run SQL queries directly
- [Reading and Writing](data-sources.md) - work with CSV, Parquet, Delta, and other formats
- [Configuration and Connection](configuration.md) - connect to remote servers, set session config

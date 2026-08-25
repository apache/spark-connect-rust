# Quickstart

Get up and running with Spark Connect in minutes. This guide walks you through your first Rust query.

## Prerequisites

Before you start, ensure you have a running Spark Connect server. See [Configuration and Connection](configuration.md) for how to start one locally:

```bash
$SPARK_HOME/sbin/start-connect-server.sh --packages "org.apache.spark:spark-connect_2.13:4.2.0"
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
    .select(vec![f::col("id")])
    .show(20)?;
```

## Aggregation and Grouping

Compute aggregations over grouped data:

```rust
use spark_connect::{functions as f, lit};

let df = spark.range(20)?;
df
    .with_column("category", f::col("id") % lit(3))
    .group_by(vec![f::col("category")])
    .agg(vec![
        f::sum(f::col("id")).alias("total").expression().clone(),
        f::avg(f::col("id")).alias("average").expression().clone(),
    ])
    .show(20)?;
```

## Next Steps

- [DataFrames](dataframes.md) - column operations, joins, window functions
- [SQL](sql.md) - run SQL queries directly
- [Reading and Writing](data-sources.md) - work with CSV, Parquet, Delta, and other formats
- [Configuration and Connection](configuration.md) - connect to remote servers, set session config

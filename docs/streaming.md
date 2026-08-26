# Structured Streaming

The `spark_connect` crate supports Structured Streaming for continuous data processing from streaming sources and sinks.

## Reading Streams

Use `spark.read_stream()` to create a streaming DataFrame:

```rust
// Rate source (generates rows at specified rate; useful for testing)
let df = spark.read_stream()
    .format("rate")
    .option("rowsPerSecond", "10")
    .load(None);

// File source (monitors directory for new files)
let df = spark.read_stream()
    .format("csv")
    .option("header", "true")
    .schema("value string")
    .load(Some("/path/to/stream/input"));
```

## Stream Transformations

Apply SQL operations on streaming DataFrames as you would on static ones:

```rust
use spark_connect::{col, lit};

let result = df
    .filter(col("value").gt(lit(10)))
    .select(vec![col("timestamp"), col("value")])
    .write_stream()
    .format("console")
    .start("")?;
```

## Writing Streams

Use `df.write_stream()` to write a streaming DataFrame to a sink:

```rust
use spark_connect::streaming::Trigger;

// Console sink
let query = df.write_stream()
    .format("console")
    .output_mode("append")
    .trigger(Trigger::ProcessingTime("5 seconds".to_string()))
    .option("checkpointLocation", "/tmp/checkpoint")
    .start("")?;

// Memory sink (debug only; stores in table)
let query = df.write_stream()
    .format("memory")
    .query_name("my_stream")
    .output_mode("append")
    .start("")?;

// File sink (Parquet)
let query = df.write_stream()
    .format("parquet")
    .output_mode("append")
    .option("checkpointLocation", "/tmp/checkpoint")
    .start("/path/to/output")?;
```

### Output Modes

| Mode | Behavior |
|------|----------|
| `append` | Add new rows to sink only |
| `complete` | Rewrite entire result set (aggregations only) |
| `update` | Update changed rows only (aggregations only) |

### Triggers

Control how often results are written:

- `trigger("processingTime=5 seconds")` - Write every 5 seconds.
- `trigger("once=true")` - Process one micro-batch then stop.
- `trigger("continuous=1 second")` - Continuous mode (lower latency).

## Managing Queries

### Awaiting Termination

Block until a query stops (either by error or `stop()`):

```rust
let query = df.write_stream().format("console").start("")?;
query.await_termination(None)?;
```

### Stopping a Query

Gracefully stop a streaming query:

```rust,ignore
query.stop()?;
```

### Active Queries

List all active streaming queries in the session:

```rust
for q in spark.streams().active()? {
    println!("Query {}: {:?}", q.id(), q.is_active()?);
}
```

!!! warning
    Checkpoints are mandatory for fault tolerance. Always set `checkpointLocation` for production queries.

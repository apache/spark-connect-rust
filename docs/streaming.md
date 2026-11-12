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
    .select([col("timestamp"), col("value")])
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

### Listener failures

For native Rust listeners, a terminal stream end or error clears the subscription's
registrations. The default `StreamingQueryListener::on_stream_error` hook is silent;
override it to report failures, and re-add listeners to open a new subscription. A
normal stream end clears registrations without calling the error hook. Adapters can
return errors through `try_on_event` or provide shared asynchronous dispatch through
`try_post_to_all`. Use `DataStreamWriter::start_with_event_handler` or
`to_table_with_event_handler` for shared synchronous start-event decoding.

`SparkSession::stop()` and `close()` cancel listener dispatch and clear registrations
locally, without sending a listener-removal RPC or joining callbacks already running.
Reference PySpark leaves listener-bus cleanup to its handler thread during session stop.

The PySpark drop-in listener bus deserializes each event once before invoking callbacks and
shares that event object across all listener registrations. A malformed start event
raises from `start()` or `toTable()`, even when no listeners are registered. A
transport failure, malformed asynchronous event or unknown event type warns and
clears the subscription's listeners. Add the listeners again to open a subscription.
A server-side end of the event stream also clears registrations without a warning;
reference PySpark retains them after a normal stream end.

`streams.close()` removes every listener. Reference PySpark's `close()` removes the listeners
while it iterates over its list, which each removal shortens, so with three listeners it leaves
the second one registered, with the server's listener. The drop-in does not mirror that.

Callback exceptions derived from `Exception` produce per-listener warnings and
dispatch continues. An uncaught `BaseException`, or a callback warning promoted to
an error by Python's warning filters, terminates the native subscription and clears
its registrations. This cleanup is deliberate: reference PySpark can retain stale
registrations when its handler exits through `BaseException`. The terminal handler
warning is best effort when warning filters themselves raise errors.

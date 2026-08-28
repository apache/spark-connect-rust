# Reading and Writing

The `spark_connect` crate lets you read data from and write data to multiple file formats and data sources, with the server resolving all paths.

## Reading Data

### Basic Read

Use `spark.read()` to access the DataFrameReader, specifying a format and path.

```rust
// Generic format API
let df = spark.read()
    .format("parquet")
    .load(Some("/path/to/file.parquet"));

// Shortcuts
let df = spark.read().parquet("/path/to/file.parquet");
let df = spark.read().csv("/path/to/file.csv");
let df = spark.read().json("/path/to/file.json");
```

### Common Read Options

Set read behavior with `.option(key, value)`:

```rust
let df = spark.read()
    .format("csv")
    .option("header", "true")
    .option("delimiter", ",")
    .option("inferSchema", "true")
    .load(Some("/path/to/data.csv"));
```

| Option | Format | Description |
|--------|--------|-------------|
| `header` | CSV | First row contains column names |
| `delimiter` | CSV | Field separator (default: comma) |
| `inferSchema` | CSV, JSON | Auto-detect column types |
| `recursiveFileLookup` | Parquet, CSV, JSON | Recursively search directories |
| `mode` | All | `PERMISSIVE`, `DROPMALFORMED`, `FAILFAST` for error handling |

## Supported Formats

- **Parquet**: Columnar format; default for DataFrames.
- **CSV**: Delimiter-separated values; requires `header` option.
- **JSON**: Line-delimited JSON objects.
- **ORC**: Optimized columnar format.
- **Delta**: Unified format with ACID transactions (Delta Lake).

## Writing Data

### Basic Write

Use `df.write()` to access the DataFrameWriter, specifying a format and save location:

```rust
df.write()
    .format("parquet")
    .mode("overwrite")
    .save(Some("/path/to/output"))?;

// Shortcuts
df.write().parquet("/path/to/output.parquet")?;
df.write().csv("/path/to/output.csv")?;
df.write().json("/path/to/output.json")?;
```

### Save Modes

Control behavior when the target path already exists:

| Mode | Behavior |
|------|----------|
| `overwrite` | Replace existing data |
| `append` | Add data to existing directory |
| `ignore` | Do nothing if path exists |
| `error` | Raise an error (default) |

### Partitioning

Organize output by one or more columns:

```rust
df.write()
    .format("parquet")
    .partition_by(["year".to_string(), "month".to_string()])
    .mode("overwrite")
    .save(Some("/path/to/output"))?;
```

### Catalog Tables

Save as a managed or external table:

```rust
df.write().mode("overwrite").save_as_table("my_table")?;
df.write()
    .mode("overwrite")
    .option("path", "/custom/path")
    .save_as_table("my_table")?;
```

!!! note
    All file paths are resolved by the Spark Connect server, not the client. Ensure the server has access to the target location.

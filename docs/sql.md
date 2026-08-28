# SQL

Execute SQL directly against DataFrames and data sources. Mix SQL queries with the DataFrame API for maximum flexibility.

## Running SQL Queries

Use `spark.sql()` to execute SQL and retrieve results as a DataFrame.

```rust
use spark_connect::SparkSession;

let spark = SparkSession::builder()
    .remote("sc://localhost:15002")
    .get_or_create()?;

// Simple query
let df = spark.sql("SELECT 1 as id, 'hello' as msg")?;
df.show(10)?;

// Aggregate query
let df = spark.sql(
    r#"SELECT category, COUNT(*) as cnt, AVG(price) as avg_price
       FROM products
       GROUP BY category
       ORDER BY cnt DESC"#
)?;
df.show(10)?;
```

## Registering Temporary Views

Make DataFrames queryable via SQL by creating temporary views.

```rust
use spark_connect::SparkSession;

let spark = SparkSession::builder()
    .remote("sc://localhost:15002")
    .get_or_create()?;

// Create from range
let df = spark.range(4)?;

// Register as temp view
df.create_or_replace_temp_view("users")?;

// Query it
let result = spark.sql("SELECT * FROM users WHERE id > 1")?;
result.show(10)?;

// Replace view
let df_updated = spark.sql("SELECT id FROM users")?;
df_updated.create_or_replace_temp_view("users")?;
```

!!! note
    Temporary views are scoped to the session and are dropped when the session ends.

## Dynamic values

`SparkSession::sql` takes a single query string; there is no server-side
parameter-binding API. For **caller-controlled values, prefer the typed
DataFrame API** rather than splicing them into SQL text -- values are passed as
literals, so untrusted input cannot break the query or inject SQL:

```rust
use spark_connect::{SparkSession, functions as f, lit, lit_string};

let spark = SparkSession::builder()
    .remote("sc://localhost:15002")
    .get_or_create()?;

let threshold = 50;
let category = "electronics";

// Injection-safe: the values are literals, not part of the SQL string.
let df = spark.sql("SELECT * FROM products")?
    .filter(f::col("price").gt(lit(threshold)))
    .filter(f::col("category").eq(lit_string(category)));
```

!!! warning
    Building a query by interpolation (`format!`) is **not** injection-safe and
    must only be used with trusted, validated input:

    ```rust
    // Trusted input only - NOT safe for user-controlled values.
    let threshold = 50;
    let df = spark.sql(&format!("SELECT * FROM products WHERE price > {}", threshold))?;
    ```

## Mixing SQL and DataFrames

Alternate between SQL queries and DataFrame API transformations.

```rust
use spark_connect::{SparkSession, functions as f, lit};

// Start with SQL
let raw = spark.sql("SELECT * FROM raw_data")?;

// Transform with DataFrame API
let cleaned = raw
    .filter(f::col("value").gt(lit(0)))
    .select([f::col("id"), f::col("value")])
    .with_column("scaled", f::col("value") * lit(2));

// Register result for SQL
cleaned.create_or_replace_temp_view("cleaned_data")?;

// Query it with SQL
let aggregated = spark.sql(
    r#"SELECT id, COUNT(*) as cnt, AVG(scaled) as avg_scaled
       FROM cleaned_data
       GROUP BY id"#
)?;
aggregated.show(10)?;
```

## SQL and DataFrame Interop Example

```rust
use spark_connect::{SparkSession, functions as f, lit, lit_string};
use spark_connect::column::when; // CASE/WHEN builder (supports `.otherwise`)

// Create test data (from range with derived columns)
let df = spark.range(5)?
    .with_column(
        "category",
        when(f::col("id").le(lit(2)), lit_string("Electronics"))
            .otherwise(lit_string("Books"))
    )
    .with_column(
        "price",
        when(f::col("id").eq(lit(1)), lit(299))
            .when(f::col("id").eq(lit(2)), lit(150))
            .when(f::col("id").eq(lit(3)), lit(25))
            .otherwise(lit(30))
    );
df.create_or_replace_temp_view("products")?;

// SQL: aggregate by category
let by_cat = spark.sql(
    r#"SELECT category, COUNT(*) as count, AVG(price) as avg_price
       FROM products
       GROUP BY category"#
)?;
by_cat.create_or_replace_temp_view("category_stats")?;

// DataFrame API: filter and order
let result = spark.sql("SELECT * FROM category_stats")?
    .filter(f::col("avg_price").gt(lit(50)))
    .order_by(vec![f::col("count").desc().expression().clone()]);
result.show(10)?;
```

!!! tip
    Temporary views make it easy to break complex transformations into readable steps. See [DataFrames](dataframes.md) for more on the API, and [Catalog](catalog.md) for managing tables and schemas.

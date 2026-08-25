# Catalog

The Spark catalog provides metadata and management operations for databases, tables, columns, and functions.

## Accessing the Catalog

```rust
let catalog = spark.catalog();
```

## Databases

List, switch, and inspect databases:

```rust
// List all databases (returns a DataFrame)
let dbs = spark.catalog().list_databases()?;
dbs.show(20)?;

// Current database
let current = spark.catalog().current_database()?;

// Set current database
spark.catalog().set_current_database("my_db")?;

// Check existence
let exists = spark.catalog().database_exists("my_db")?;
```

## Tables

List, inspect, and check tables:

```rust
// List tables in current database (returns a DataFrame)
let tables = spark.catalog().list_tables()?;
tables.show(20)?;

// List tables in specific database
let tables = spark.catalog().list_tables_in_database("my_db")?;

// Check existence
let exists = spark.catalog().table_exists("my_table")?;
let exists = spark.catalog().table_exists_with_database("my_table", Some("my_db"))?;
```

## Columns

List columns in a table:

```rust
// List columns in current database (returns a DataFrame)
let cols = spark.catalog().list_columns("my_table")?;
cols.show(20)?;

// List columns in specific database
let cols = spark.catalog().list_columns_with_database("my_table", Some("my_db"))?;
```

## Functions

List and check functions:

```rust
// List all functions (returns a DataFrame)
let funcs = spark.catalog().list_functions()?;
funcs.show(20)?;

// List functions in specific database
let funcs = spark.catalog().list_functions_in_database("my_db")?;

// Check existence
let exists = spark.catalog().function_exists("my_func")?;
let exists = spark.catalog().function_exists_with_database("my_func", Some("my_db"))?;
```

## Temporary Views

Register and drop temporary views (local to session):

```rust
// Register temp view
df.create_temp_view("my_view")?;

// Register and replace
df.create_or_replace_temp_view("my_view")?;

// Register global temp view (cross-session)
df.create_global_temp_view("my_global_view")?;

// Query temp view
let result = spark.sql("SELECT * FROM my_view")?;

// Drop temp view
spark.catalog().drop_temp_view("my_view")?;

// Drop global temp view
spark.catalog().drop_global_temp_view("my_global_view")?;
```

## Cache Management

Cache and uncache tables for performance:

```rust
// Cache table
spark.catalog().cache_table("my_table")?;

// Uncache table
spark.catalog().uncache_table("my_table")?;

// Clear all caches
spark.catalog().clear_cache()?;
```

!!! note
    Temporary views are session-local and persist until the session ends. Global temporary views are prefixed with `global_temp.` by default.

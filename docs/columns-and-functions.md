# Columns and Functions

Column expressions form the building blocks of transformations. Reference columns, build expressions with operators, and apply built-in functions from the `spark_connect::functions` module.

## Column References and Literals

Access DataFrame columns and create constant values.

```rust
use spark_connect::{functions as f, lit, lit_string};

// Column reference
let col_expr = f::col("column_name");

// Literal value
let lit_expr = lit(42);
let lit_expr = lit_string("hello");

// Use in expression
let df = df.filter(f::col("age").gt(lit(18)));
```

## Column Operations

### Arithmetic & Comparison

```rust
use spark_connect::{functions as f, lit, lit_string};

// Arithmetic
let expr = f::col("x") + f::col("y");
let expr = f::col("x") - f::col("y");
let expr = f::col("x") * lit(2);
let expr = f::col("x") / lit(10);

// Comparison
let expr = f::col("age").gt(lit(18));
let expr = f::col("age").ge(lit(18));
let expr = f::col("salary").eq(lit(50000));
let expr = f::col("status").ne(lit_string("inactive"));
```

### Boolean Logic

```rust
use spark_connect::{functions as f, lit, lit_string, lit_boolean};

// AND
let expr = (f::col("age").gt(lit(18)))
    .and(f::col("city").eq(lit_string("NYC")));

// OR
let expr = (f::col("status").eq(lit_string("active")))
    .or(f::col("vip").eq(lit_boolean(true)));

// NOT
let expr = f::col("archived").eq(lit_boolean(true)).not();
```

### Aliasing, Casting, Null Handling

```rust
use spark_connect::{functions as f, lit_string};
use spark_connect::column::when; // CASE/WHEN builder (supports `.otherwise`)

// Alias
let expr = f::col("amount").alias("total");

// Cast to type
let expr = f::col("id").cast_str("string");
let expr = f::col("price").cast_str("decimal(10, 2)");

// Null checks
let expr = f::col("email").is_null();
let expr = f::col("email").is_not_null();

// Null fallback (coalesce-style) via when/otherwise
let expr = when(f::col("email").is_not_null(), f::col("email"))
    .otherwise(f::col("backup_email"));
```

## Built-in Functions

The `functions` module provides hundreds of operations across string, math, date, aggregate, and conditional categories.

### String Functions

| Function | Usage | Purpose |
|----------|-------|---------|
| `upper` | `f::upper(f::col("col"))` | Convert to uppercase |
| `lower` | `f::lower(f::col("col"))` | Convert to lowercase |
| `concat` | `f::concat(vec![...])` | Concatenate strings |
| `substring` | `f::substring(f::col("col"), 1, 3)` | Extract substring |
| `length` | `f::length(f::col("col"))` | String length |
| `trim` | `f::trim(f::col("col"))` | Remove leading/trailing spaces |
| `reverse` | `f::reverse(f::col("col"))` | Reverse string |

### Math Functions

| Function | Usage | Purpose |
|----------|-------|---------|
| `abs` | `f::abs(f::col("x"))` | Absolute value |
| `sqrt` | `f::sqrt(f::col("x"))` | Square root |
| `round` | `f::round(f::col("x"), 2)` | Round to d decimals |
| `ceil` | `f::ceil(f::col("x"))` | Ceiling |
| `floor` | `f::floor(f::col("x"))` | Floor |
| `sin/cos/tan` | `f::sin(...)` | Trigonometric |
| `log/log10/exp` | `f::log(...)` | Logarithmic/exponential |

### Date and Time Functions

| Function | Usage | Purpose |
|----------|-------|---------|
| `current_date` | `f::current_date()` | Current date |
| `current_timestamp` | `f::current_timestamp()` | Current timestamp |
| `to_date` | `f::to_date(...)` | Parse date string |
| `date_add` | `f::date_add(...)` | Add days |
| `date_sub` | `f::date_sub(...)` | Subtract days |
| `datediff` | `f::datediff(...)` | Days between dates |
| `year/month/day` | `f::year(f::col("d"))` | Extract date part |

### Aggregate Functions

| Function | Usage | Purpose |
|----------|-------|---------|
| `count` | `f::count(f::col("id"))` | Count non-null rows |
| `sum` | `f::sum(f::col("amt"))` | Sum values |
| `avg` | `f::avg(f::col("price"))` | Average |
| `min` | `f::min(f::col("val"))` | Minimum |
| `max` | `f::max(f::col("val"))` | Maximum |
| `stddev` | `f::stddev(f::col("x"))` | Standard deviation |
| `collect_list` | `f::collect_list(...)` | Collect into array |

### Conditional Functions

```rust
use spark_connect::{functions as f, lit, lit_string};
use spark_connect::column::when; // CASE/WHEN builder (supports `.otherwise`)

// CASE / WHEN
let expr = when(f::col("age").lt(lit(18)), lit_string("minor"))
    .when(f::col("age").lt(lit(65)), lit_string("adult"))
    .otherwise(lit_string("senior"));

// IF NULL fallback
let expr = when(f::col("phone").is_not_null(), f::col("phone"))
    .otherwise(lit_string("N/A"));
```

### Array and Collection Functions

| Function | Usage | Purpose |
|----------|-------|---------|
| `array` | `f::array(vec![...])` | Create array |
| `explode` | `f::explode(f::col("arr"))` | Expand array to rows |
| `size` | `f::size(f::col("arr"))` | Array/map size |
| `element_at` | `f::element_at(...)` | Get array element |
| `array_contains` | `f::array_contains(...)` | Check membership |

!!! tip
    See [DataFrames](dataframes.md) for transformation examples and [SQL](sql.md) for SQL-based expressions.

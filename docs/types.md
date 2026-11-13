# Types and Schemas

Spark SQL uses a rich type system. Learn how to work with schemas in the `spark_connect` crate and cast columns between types.

## Spark SQL Types

Each Spark SQL type is a variant of the `spark_connect::types::DataType` enum, which is also
exported as `spark_connect::DataType`. The tables below give, for each type:

- **DDL**: how a DDL string writes the type, which is also its `simple_string()`.
  `DataType::from_ddl` and the server parse this and the aliases in parentheses.
- **`DataType`**: the Rust value of the type.
- **Collected as**: the `spark_connect::row::Value` that `collect()` gives for a value of the
  type in a `Row`.

A NULL of any type is collected as `Value::Null`.

### Numbers and booleans

| Spark type | DDL | `DataType` | Collected as |
|---|---|---|---|
| BooleanType | `boolean` | `DataType::Boolean` | `Value::Bool(bool)` |
| ByteType | `tinyint` (`byte`) | `DataType::Byte` | `Value::Byte(i8)` |
| ShortType | `smallint` (`short`) | `DataType::Short` | `Value::Short(i16)` |
| IntegerType | `int` (`integer`) | `DataType::Integer` | `Value::Integer(i32)` |
| LongType | `bigint` (`long`) | `DataType::Long` | `Value::Long(i64)` |
| FloatType | `float` (`real`) | `DataType::Float` | `Value::Float(f32)` |
| DoubleType | `double` | `DataType::Double` | `Value::Double(f64)` |
| DecimalType | `decimal(10,2)` (`dec`, `numeric`) | `DataType::Decimal { precision: 10, scale: 2 }` | `Value::Decimal { value, precision, scale }` |

A decimal is collected exactly, as text: `CAST(1.5 AS DECIMAL(10,2))` is
`Value::Decimal { value: "1.50", precision: Some(10), scale: Some(2) }`. In DDL, a bare
`decimal` is `decimal(10,0)` and `decimal(p)` is `decimal(p,0)`.

### Strings and binary

| Spark type | DDL | `DataType` | Collected as |
|---|---|---|---|
| StringType | `string` | `DataType::String { collation: "UTF8_BINARY".to_string() }` | `Value::String(String)` |
| StringType with a collation | `string collate UTF8_LCASE` | `DataType::String { collation: "UTF8_LCASE".to_string() }` | `Value::String(String)` |
| CharType | `char(5)` (`character(5)`) | `DataType::Char { length: 5 }` | `Value::String(String)` |
| VarcharType | `varchar(10)` | `DataType::Varchar { length: 10 }` | `Value::String(String)` |
| BinaryType | `binary` | `DataType::Binary` | `Value::Binary(Vec<u8>)` |

`collation` names the collation of a string, which decides how strings compare. `UTF8_BINARY`
is the default, and `simple_string()` leaves it out; any other, such as `UTF8_LCASE`, `UNICODE`
or `UNICODE_CI`, is written `string collate <name>`. `from_ddl` gives the name in the case
that the server uses, so `string collate unicode_ci` has the collation `UNICODE_CI`. In the
JSON of a struct field (`json()`), the collations of its strings are in the `__COLLATIONS`
entry of the field's metadata, where Spark writes them.

A cast to `char(5)` or `varchar(5)` gives a `string` column, with no padding and no length
check, unless the session sets `spark.sql.preserveCharVarcharTypeInfo` to `true`: then the
column keeps `DataType::Char` or `DataType::Varchar`, and a CHAR value is padded to its
length. The server rejects CHAR and VARCHAR in the schema of `create_dataframe` and of
`DataFrameReader::schema` with `UNSUPPORTED_CHAR_OR_VARCHAR_AS_STRING`, even with that
setting; use `string` there.

### Dates, times and timestamps

| Spark type | DDL | `DataType` | Collected as |
|---|---|---|---|
| DateType | `date` | `DataType::Date` | `Value::Date(i32)`: days since 1970-01-01 |
| TimestampType | `timestamp` (`timestamp_ltz`) | `DataType::Timestamp` | `Value::Timestamp(i64)`: microseconds since 1970-01-01 00:00:00 UTC |
| TimestampNTZType | `timestamp_ntz` | `DataType::TimestampNtz` | `Value::TimestampNtz(i64)`: microseconds of the date-time since 1970-01-01 00:00:00 |
| TimeType | `time(6)` (`time`) | `DataType::Time { precision: 6 }` | `Value::Time(i64)`: nanoseconds since midnight |
| TimestampNTZNanosType | `timestamp_ntz(9)` | `DataType::TimestampNtzNanos { precision: 9 }` | `Value::TimestampNtz(i64)`, in microseconds |
| TimestampLTZNanosType | `timestamp_ltz(9)` (`timestamp(9)`) | `DataType::TimestampLtzNanos { precision: 9 }` | `Value::Timestamp(i64)`, in microseconds |

A `Value::Timestamp` is an instant: `TIMESTAMP'2024-01-02 03:04:05.123456'` is
`Value::Timestamp(1704164645123456)` when the session time zone
(`spark.sql.session.timeZone`) is UTC, and the time zone only decides which instant a literal
or a string means. A `Value::TimestampNtz` is a date-time that no time zone shifts.
`from_ddl` always reads `timestamp` as `DataType::Timestamp`, while the server reads it as
`DataType::TimestampNtz` when `spark.sql.timestampType` is `TIMESTAMP_NTZ`;
`SparkSession::parse_ddl` parses a DDL string on the server, as the server reads it.

A TIME has a precision of 0 to 9 fractional digits, and `time` is `time(6)`. A value is
collected in nanoseconds: `TIME'12:34:56.123456'` is `Value::Time(45296123456000)`. Spark 4.3
supports TIME only when `spark.sql.timeType.enabled` is `true`, which is not its default;
otherwise a query of a TIME fails with `UNSUPPORTED_TIME_TYPE`.

`timestamp_ntz(p)` and `timestamp_ltz(p)` with a precision `p` of 7 to 9 are the nanosecond
timestamps, a preview in Spark 4.3: `timestamp(p)` is `timestamp_ltz(p)`, while `timestamp(6)`
and `timestamp_ntz(6)` are the microsecond types. The server takes them only when
`spark.sql.timestampNanosTypes.enabled` is `true`, which is not its default and which a
session can set with `spark.conf().set(...)`; otherwise it fails with `FEATURE_NOT_ENABLED`.
`collect` gives them in microseconds, in the `Value` variants of the microsecond types,
rounding the sub-microsecond digits down; `DataFrame::to_arrow` keeps the nanoseconds.

### Intervals

| Spark type | DDL | `DataType` | Collected as |
|---|---|---|---|
| YearMonthIntervalType | `interval year to month` (`interval year`, `interval month`) | `DataType::YearMonthInterval { start_field: 0, end_field: 1 }` | `Value::YearMonthInterval(i32)`: months |
| DayTimeIntervalType | `interval day to second` (`interval day`, `interval hour to minute`, ...) | `DataType::DayTimeInterval { start_field: 0, end_field: 3 }` | `Value::DayTimeInterval(i64)`: microseconds |
| CalendarIntervalType | `interval` | `DataType::CalendarInterval` | `Value::String(String)`, as `"1 months 2 days 11045500000000 ns"` |

`start_field` and `end_field` are the first and the last field of an interval:
`YearMonthIntervalType::YEAR` (0) and `MONTH` (1), or `DayTimeIntervalType::DAY` (0), `HOUR`
(1), `MINUTE` (2) and `SECOND` (3). `INTERVAL '1-2' YEAR TO MONTH` is collected as
`Value::YearMonthInterval(14)`, and `INTERVAL '1 02:03:04.5' DAY TO SECOND` as
`Value::DayTimeInterval(93784500000)`. `create_dataframe` takes these values for
`interval year to month` and `interval day to second` columns; the server rejects them for
the narrower interval types, such as `interval year` or `interval hour to minute`, with
`INVALID_COLUMN_OR_FIELD_DATA_TYPE`.

`interval` alone is the calendar interval, of months, days and microseconds, that functions
such as `make_interval` return. It has no `Value` variant: it is collected as the text of its
months, days and nanoseconds, and `create_dataframe` cannot take it.

### Arrays, maps and structs

| Spark type | DDL | `DataType` | Collected as |
|---|---|---|---|
| ArrayType | `array<int>` | `DataType::Array { element_type: Box::new(DataType::Integer), contains_null: true }` | `Value::List(Vec<Value>)` |
| MapType | `map<string,int>` | `DataType::Map { key_type, value_type, value_contains_null }` | `Value::Map(Vec<(Value, Value)>)`, its entries in order |
| StructType | `struct<a:int,b:string>` | `DataType::Struct { fields }`, of `StructField`s | `Value::Struct(StructValue)`, its names and values in order |

DDL does not say whether the elements of an array or the values of a map may be null:
`from_ddl` gives `contains_null: true` and `value_contains_null: true`. A struct field is
nullable unless it is `NOT NULL`. See [Nested Types](#nested-types).

### Variant and spatial types

| Spark type | DDL | `DataType` | Collected as |
|---|---|---|---|
| VariantType | `variant` | `DataType::Variant` | `Value::Variant { value, metadata }`, both `Vec<u8>` |
| GeometryType | `geometry(4326)` (`geometry(any)`) | `DataType::Geometry { srid: 4326 }` | `Value::Struct` of `srid` (`Value::Integer`) and `wkb` (`Value::Binary`) |
| GeographyType | `geography(4326)` (`geography(any)`) | `DataType::Geography { srid: 4326 }` | `Value::Struct` of `srid` and `wkb` |

A variant is collected as the two binary parts of Spark's variant encoding, which this crate
does not decode. To read one as JSON, select `to_json` of it on the server.

A spatial type has the SRID of the spatial reference system of its values, or allows any
SRID, which is `srid: -1` (`SpatialType::MIXED_SRID`) and `any` in DDL. A value is collected as
a struct of its SRID and its WKB (well-known binary) geometry, and `create_dataframe` takes the
same struct. `from_ddl` checks the SRID as Spark's parser does: an SRID that the type does not
take, such as 99999, or a geography of 3857, is `ST_INVALID_SRID_VALUE`.

The SRIDs are those of Spark's spatial reference system registry (about 10,700 systems, embedded
from Spark's `srs_registry.csv`). A geometry takes any of them, and a geography only a geographic
one. The JSON of a spatial type (`json()`) names its reference system by the string ID that Spark
writes: `geometry(OGC:CRS84)` is SRID 4326, `geometry(EPSG:2154)` is SRID 2154 and
`geometry(SRID:0)` is SRID 0; a geography also names its algorithm, as
`geography(OGC:CRS83, SPHERICAL)` for SRID 4269. `DataType::from_json` and the server read these
back, and the aliases of the OGC systems, such as `EPSG:4269`, too. A CRS that Spark does not
know, or that the type does not take, is `ST_INVALID_CRS_VALUE`. DDL, and the protobuf form in
which the server sends a schema, carry any SRID.

The drop-in is stricter, as PySpark 4.3.0 is: its `GeometryType` takes the SRIDs 0, 3857 and 4326,
and its `GeographyType` 4326, and it reads only the CRS names that those take, so a schema of the
server with another SRID raises `ST_INVALID_SRID_VALUE` when its type is read.

### Other types

| Spark type | DDL | `DataType` | Collected as |
|---|---|---|---|
| NullType | `void` | `DataType::Null` | `Value::Null` |
| UserDefinedType | (`simple_string()` is `udt`) | `DataType::Udt { .. }` | the `Value` of its SQL type |
| A type for the server to parse | (holds a DDL type string) | `DataType::Unparsed { data_type_string }` | (the server returns the parsed type) |

A user-defined type, such as the vector type of Spark ML, is in a schema as `DataType::Udt`,
with its JVM and Python classes (`jvm_class: Some("org.apache.spark.ml.linalg.VectorUDT")`), and
its values are collected as those of its SQL type: a vector is a `Value::Struct` of `type`,
`size`, `indices` and `values`.

`DataType::Unparsed` sends a DDL type string as it is, for the server to parse, wherever a
`DataType` is sent: `col("x").cast(DataType::Unparsed { data_type_string: "decimal(5,2)".to_string() })`
casts to `decimal(5,2)`. `create_dataframe` cannot take it.

!!! note "Python drop-in"
    In the Python drop-in, `pyspark.sql.types` has PySpark's type classes (`IntegerType()`,
    `StructType`, ...), backed by this same `DataType`; see the
    [PySpark documentation](https://spark.apache.org/docs/latest/api/python/).

## Inspecting Schemas

`DataFrame::schema` returns the schema of a DataFrame as a `DataType::Struct`, whose
`StructField`s have the name, type and nullability of each column:

```rust
use spark_connect::types::DataType;

let df = spark.sql(
    "SELECT 1L AS id, CAST(2.5 AS DECIMAL(10,2)) AS price, \
     INTERVAL '1-2' YEAR TO MONTH AS term, array(named_struct('k', 'a', 'v', 1)) AS kv",
)?;

// Get full schema
if let DataType::Struct { fields } = df.schema()? {
    for field in &fields {
        // A DataType displays as its simple string: "bigint", "decimal(10,2)", ...
        println!("{}: {} (nullable = {})", field.name, field.data_type, field.nullable);
    }
}

// Column names: ["id", "price", "term", "kv"]
let names = df.columns()?;

// (name, type) pairs: [("id", "bigint"), ("price", "decimal(10,2)"), ...]
let dtypes = df.dtypes()?;
for (name, dtype) in dtypes {
    println!("{}: {}", name, dtype);
}

// Print schema
df.print_schema()?;
```

`print_schema` prints the schema as a tree, as `DataType::tree_string` gives it:

```text
root
 |-- id: long (nullable = false)
 |-- price: decimal(10,2) (nullable = false)
 |-- term: interval year to month (nullable = false)
 |-- kv: array (nullable = false)
 |    |-- element: struct (containsNull = false)
 |    |    |-- k: string (nullable = false)
 |    |    |-- v: integer (nullable = false)
```

A `DataType` has the string forms of PySpark's types:

| Method | Gives | Example |
|---|---|---|
| `simple_string()`, also `to_string()` | the DDL form, which `dtypes` shows | `bigint`, `array<struct<k:string,v:int>>` |
| `type_name()` | PySpark's `typeName()`, which `print_schema` shows for most types | `long`, `array` |
| `json()` | Spark's JSON form, which `DataType::from_json_str` parses | `"long"` |
| `to_ddl()` | the DDL of the fields of a struct | `id BIGINT NOT NULL,price DECIMAL(10,2) NOT NULL,...` |

## Building a Schema

Define a schema explicitly when reading or creating DataFrames. A reader takes a DDL string,
or the JSON of a `DataType` from `json()`:

```rust
let df = spark
    .read()
    .schema("name string, age int, salary bigint".to_string())
    .csv("/path/to/data.csv");
```

`DataType::from_ddl` parses a DDL string into a `DataType` without the server, and a list of
fields becomes a `DataType::Struct`. `NOT NULL` makes a field non-nullable, and
`COMMENT 'text'` puts the text in the field's metadata under `comment`. `create_dataframe`
takes the rows of a DataFrame and its schema: each `Row` has one `Value` for each field, in the
order of the fields.

```rust
use spark_connect::row::{Row, Value};
use spark_connect::types::DataType;

let schema = DataType::from_ddl(
    "id bigint NOT NULL, amount decimal(10,2), wait interval day to second, \
     note string COMMENT 'free text'",
)?;

let rows = vec![Row::new(
    vec!["id".into(), "amount".into(), "wait".into(), "note".into()],
    vec![
        Value::Long(1),
        Value::Decimal {
            value: "12.50".into(),
            precision: Some(10),
            scale: Some(2),
        },
        // 1 hour 30 minutes, in microseconds
        Value::DayTimeInterval(90 * 60 * 1_000_000),
        Value::String("first".into()),
    ],
)];
let df = spark.create_dataframe(rows, schema)?;
```

A schema can also be built in Rust, with `DataType::Struct` and `StructField` (see
[Nested Types](#nested-types)), or with `add`, which returns the struct with one more field:

```rust
use spark_connect::types::{DataType, DayTimeIntervalType};

let schema = DataType::Struct { fields: vec![] }
    .add("id", DataType::Long, false, None)?
    .add(
        "code",
        DataType::String {
            collation: "UTF8_LCASE".to_string(),
        },
        true,
        None,
    )?
    .add(
        "wait",
        DataType::DayTimeInterval {
            start_field: DayTimeIntervalType::DAY,
            end_field: DayTimeIntervalType::SECOND,
        },
        true,
        None,
    )?;

assert_eq!(
    schema,
    DataType::from_ddl(
        "id bigint NOT NULL, code string collate UTF8_LCASE, wait interval day to second"
    )?
);
```

## Casting Columns

Convert a column to a different type with `.cast()`:

```rust
use spark_connect::{col, types::DataType};

// Cast to LongType
let df = df.with_column("id", col("id").cast(DataType::Long));

// Cast to DoubleType
let df = df.with_column("price", col("price").cast(DataType::Double));

// Cast with a type name string
let df = df.with_column("created", col("created").cast_str("timestamp"));
```

`cast_str` sends the type string for the server to parse, so `"timestamp"` is
TIMESTAMP_NTZ there when `spark.sql.timestampType` is `TIMESTAMP_NTZ`. `try_cast` and
`try_cast_str` give NULL for a value that cannot be cast, where `cast` fails the query in ANSI
mode, Spark's default:

```rust
use spark_connect::{
    col,
    types::{DataType, DayTimeIntervalType},
};

let df = spark.sql("SELECT '1 02:03:04.5' AS wait, 'abc' AS n")?;

// To an interval day to second: Value::DayTimeInterval(93784500000)
let df = df.with_column(
    "wait",
    col("wait").cast(DataType::DayTimeInterval {
        start_field: DayTimeIntervalType::DAY,
        end_field: DayTimeIntervalType::SECOND,
    }),
);

// "abc" is not a number, so this is NULL, while cast(DataType::Integer) fails
let df = df.with_column("n", col("n").try_cast(DataType::Integer));
```

## Nested Types

Work with complex nested structures:

```rust
use spark_connect::types::{DataType, StructField};
use std::collections::BTreeMap;

// Array of strings
let array_of_strings = DataType::Array {
    element_type: Box::new(DataType::String {
        collation: "UTF8_BINARY".to_string(),
    }),
    contains_null: true,
};

// Map with string keys and integer values
let map_type = DataType::Map {
    key_type: Box::new(DataType::String {
        collation: "UTF8_BINARY".to_string(),
    }),
    value_type: Box::new(DataType::Integer),
    value_contains_null: true,
};

// Nested struct
let nested = DataType::Struct {
    fields: vec![
        StructField {
            name: "name".to_string(),
            data_type: DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
            nullable: true,
            metadata: BTreeMap::new(),
        },
        StructField {
            name: "tags".to_string(),
            data_type: DataType::Array {
                element_type: Box::new(DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                }),
                contains_null: true,
            },
            nullable: true,
            metadata: BTreeMap::new(),
        },
    ],
};

// The same types in DDL
assert_eq!(array_of_strings, DataType::from_ddl("array<string>")?);
assert_eq!(map_type, DataType::from_ddl("map<string,int>")?);
assert_eq!(nested, DataType::from_ddl("struct<name:string,tags:array<string>>")?);
```

!!! note
    Nullable fields allow `NULL` values. Set `nullable` to `false` when a field must always
    have a value. `contains_null` and `value_contains_null` do the same for the elements of an
    array and the values of a map.

Nested values are collected as `Value::List`, `Value::Map` and `Value::Struct`; the fields of
a struct can be read by name or by position:

```rust
use spark_connect::row::Value;

let rows = spark
    .sql("SELECT named_struct('name', 'a', 'tags', array('x', 'y')) AS s, map('k', 1) AS m")?
    .collect()?;

if let Some(Value::Struct(s)) = rows[0].get_by_name("s") {
    assert_eq!(s.get_by_name("name"), Some(&Value::String("a".into())));
    if let Some(Value::List(tags)) = s.get(1) {
        assert_eq!(tags, &vec![Value::String("x".into()), Value::String("y".into())]);
    }
}
if let Some(Value::Map(entries)) = rows[0].get_by_name("m") {
    for (key, value) in entries {
        println!("{key} = {value}"); // k = 1
    }
}
```

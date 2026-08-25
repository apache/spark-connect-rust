# Types and Schemas

Spark SQL uses a rich type system. Learn how to work with schemas in the `spark_connect` crate and cast columns between types.

## Spark SQL Types

Map Spark SQL types to their Rust `spark_connect::types::DataType` equivalents:

| Spark Type | Rust DataType |
|-----------|---------|
| StringType | `DataType::String { collation: "UTF8_BINARY".to_string() }` |
| IntegerType | `DataType::Integer` |
| LongType | `DataType::Long` |
| FloatType | `DataType::Float` |
| DoubleType | `DataType::Double` |
| BooleanType | `DataType::Boolean` |
| BinaryType | `DataType::Binary` |
| DateType | `DataType::Date` |
| TimestampType | `DataType::Timestamp` |
| DecimalType | `DataType::Decimal { precision: 10, scale: 2 }` |
| ArrayType | `DataType::Array { element_type: Box::new(DataType::String { ... }), contains_null: true }` |
| MapType | `DataType::Map { key_type: Box::new(...), value_type: Box::new(...), value_contains_null: true }` |
| StructType | `DataType::Struct { fields: Vec<StructField> }` |

## Inspecting Schemas

Access schema information from a DataFrame:

```rust
// Get full schema
let schema = df.schema()?;

// Print schema
df.print_schema()?;

// Get list of (name, type) tuples
let dtypes = df.dtypes()?;
for (name, dtype) in dtypes {
    println!("{}: {}", name, dtype);
}
```

## Building a Schema

Define a schema explicitly when reading or creating DataFrames:

```rust
use spark_connect::types::DataType;

// Define schema as DDL string
let schema_ddl = "name string, age int, salary long";

let df = spark.read()
    .schema(schema_ddl.to_string())
    .csv("/path/to/data.csv");
```

## Casting Columns

Convert a column to a different type with `.cast()`:

```rust
use spark_connect::types::DataType;

// Cast to LongType
let df = df.with_column("id", col("id").cast(DataType::Long));

// Cast to DoubleType
let df = df.with_column("price", col("price").cast(DataType::Double));

// Cast with a type name string
let df = df.with_column("created", col("created").cast_str("timestamp"));
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
```

!!! note
    Nullable fields allow `NULL` values. Set to `False` when a field must always have a value.

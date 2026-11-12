//! DataType type system mirroring PySpark's `pyspark.sql.types`.
//!
//! Defines the type hierarchy and conversion functions between Python DataTypes
//! and the Spark Connect protobuf representation.

use std::collections::BTreeMap;
use std::fmt::{self, Write};
use std::hash::{Hash, Hasher};

use spark_connect_core::error::{Result, SparkError, SparkErrorKind};

use crate::srs;

/// The base DataType representation, mirroring `pyspark.sql.types.DataType`.
///
/// All concrete types are variants of this enum. Each variant carries the data
/// needed to fully specify that type (e.g., DecimalType carries precision and scale).
// Not `Eq` (only `PartialEq`): a Struct field's metadata holds arbitrary JSON values.
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// `pyspark.sql.types.NullType`
    Null,
    /// `pyspark.sql.types.BooleanType`
    Boolean,
    /// `pyspark.sql.types.ByteType` (tinyint)
    Byte,
    /// `pyspark.sql.types.ShortType` (smallint)
    Short,
    /// `pyspark.sql.types.IntegerType` (int)
    Integer,
    /// `pyspark.sql.types.LongType` (bigint)
    Long,
    /// `pyspark.sql.types.FloatType`
    Float,
    /// `pyspark.sql.types.DoubleType`
    Double,
    /// `pyspark.sql.types.DecimalType`
    Decimal { precision: i32, scale: i32 },
    /// `pyspark.sql.types.StringType`
    String { collation: String },
    /// `pyspark.sql.types.CharType`
    Char { length: i32 },
    /// `pyspark.sql.types.VarcharType`
    Varchar { length: i32 },
    /// `pyspark.sql.types.BinaryType`
    Binary,
    /// `pyspark.sql.types.DateType`
    Date,
    /// `pyspark.sql.types.TimestampType`
    Timestamp,
    /// `pyspark.sql.types.TimestampNTZType`
    TimestampNtz,
    /// `pyspark.sql.types.TimeType`
    Time { precision: i32 },
    /// Spark's `TimestampNTZNanosType(precision)` (precision 7..=9): a timestamp
    /// without time zone with nanosecond-capable fractional seconds. PySpark 4.3.0
    /// has no Python class for it; this mirrors the server/proto type.
    TimestampNtzNanos { precision: i32 },
    /// Spark's `TimestampLTZNanosType(precision)` (precision 7..=9): a timestamp
    /// with local time zone with nanosecond-capable fractional seconds.
    TimestampLtzNanos { precision: i32 },
    /// `pyspark.sql.types.CalendarIntervalType`
    CalendarInterval,
    /// `pyspark.sql.types.YearMonthIntervalType`
    YearMonthInterval { start_field: i32, end_field: i32 },
    /// `pyspark.sql.types.DayTimeIntervalType`
    DayTimeInterval { start_field: i32, end_field: i32 },
    /// `pyspark.sql.types.ArrayType`
    Array {
        element_type: Box<DataType>,
        contains_null: bool,
    },
    /// `pyspark.sql.types.MapType`
    Map {
        key_type: Box<DataType>,
        value_type: Box<DataType>,
        value_contains_null: bool,
    },
    /// `pyspark.sql.types.StructType`
    Struct { fields: Vec<StructField> },
    /// `pyspark.sql.types.VariantType`
    Variant,
    /// `pyspark.sql.types.GeometryType`
    Geometry { srid: i32 },
    /// `pyspark.sql.types.GeographyType`
    Geography { srid: i32 },
    /// `pyspark.sql.types.UserDefinedType` (stub)
    Udt {
        type_str: String,
        jvm_class: Option<String>,
        python_class: Option<String>,
        serialized_python_class: Option<String>,
        sql_type: Option<Box<DataType>>,
    },
    /// `pyspark.sql.connect.types.UnparsedDataType` - a DDL type string left for
    /// the server to parse (round-trips through the `unparsed` proto).
    Unparsed { data_type_string: String },
}

/// Field-index constants for [`DataType::YearMonthInterval`], mirroring
/// `pyspark.sql.types.YearMonthIntervalType`.
pub struct YearMonthIntervalType;
impl YearMonthIntervalType {
    pub const YEAR: i32 = 0;
    pub const MONTH: i32 = 1;
}

/// Field-index constants for [`DataType::DayTimeInterval`], mirroring
/// `pyspark.sql.types.DayTimeIntervalType`.
pub struct DayTimeIntervalType;
impl DayTimeIntervalType {
    pub const DAY: i32 = 0;
    pub const HOUR: i32 = 1;
    pub const MINUTE: i32 = 2;
    pub const SECOND: i32 = 3;
}

/// Constants for [`DataType::Date`], mirroring `pyspark.sql.types.DateType`.
pub struct DateType;
impl DateType {
    /// Number of days from 0001-01-01 to 1970-01-01.
    pub const EPOCH_ORDINAL: i32 = 719163;
}

/// CRS/SRID constants shared by the spatial types, mirroring
/// `pyspark.sql.types.SpatialType`.
pub struct SpatialType;
impl SpatialType {
    pub const MIXED_CRS: &'static str = "SRID:ANY";
    pub const MIXED_SRID: i32 = -1;
}

/// Constants for [`DataType::Geometry`], mirroring `pyspark.sql.types.GeometryType`.
pub struct GeometryType;
impl GeometryType {
    pub const DEFAULT_CRS: &'static str = "OGC:CRS84";
    pub const DEFAULT_SRID: i32 = 4326;
    pub const MIXED_CRS: &'static str = "SRID:ANY";
    pub const MIXED_SRID: i32 = -1;
}

/// Constants for [`DataType::Geography`], mirroring `pyspark.sql.types.GeographyType`.
pub struct GeographyType;
impl GeographyType {
    pub const DEFAULT_ALG: &'static str = "SPHERICAL";
    pub const DEFAULT_CRS: &'static str = "OGC:CRS84";
    pub const DEFAULT_SRID: i32 = 4326;
    pub const MIXED_CRS: &'static str = "SRID:ANY";
    pub const MIXED_SRID: i32 = -1;
}

/// A field in a StructType, mirroring `pyspark.sql.types.StructField`.
// Not `Eq`: metadata values are arbitrary JSON (`serde_json::Value`, which is only `PartialEq`
// because of floats), matching pyspark's `Dict[str, Any]` field metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl DataType {
    /// Parses a DDL-formatted string into a DataType, mirroring `DataType.fromDDL()`.
    ///
    /// This supports:
    /// - Primitive types: int, bigint, string, double, boolean, date, timestamp, binary,
    ///   tinyint, smallint, float, decimal(p,s), char(n), varchar(n), interval
    /// - Complex types: array<...>, map<...,...>, struct<name:type,...>, with any whitespace
    ///   between the tokens, as `ARRAY < INT >`, which Spark's parser takes too
    /// - Top-level struct can omit the "struct<>" wrapper for backward compatibility
    /// - DDL like "a INT, b STRING" is parsed as a struct
    /// - A field may end with `NOT NULL` and `COMMENT 'text'`, as `StructType.toDDL` writes them
    ///
    /// The parse is local, so `timestamp` is always [`DataType::Timestamp`] (TIMESTAMP_LTZ),
    /// while the server's parser makes it TIMESTAMP_NTZ when `spark.sql.timestampType` is
    /// `TIMESTAMP_NTZ`; [`crate::SparkSession::parse_ddl`] parses a DDL string as the server
    /// does.
    ///
    /// Examples:
    /// ```ignore
    /// DataType::from_ddl("int") // IntegerType
    /// DataType::from_ddl("array<string>") // ArrayType(StringType, true)
    /// DataType::from_ddl("struct<name:string,age:int>") // StructType
    /// DataType::from_ddl("a INT, b STRING") // Top-level struct
    /// ```
    pub fn from_ddl(ddl_str: &str) -> Result<DataType> {
        parse_datatype_string(ddl_str)
    }

    /// Returns whether this type needs conversion between Python objects and internal SQL objects.
    /// This is used to avoid unnecessary conversions for ArrayType/MapType/StructType.
    ///
    /// Types that need conversion include:
    /// - DateType: needs conversion to/from datetime.date
    /// - TimestampType: needs conversion to/from datetime.datetime
    /// - TimestampNTZType: needs conversion to/from datetime.datetime (no timezone)
    /// - TimeType: needs conversion to/from datetime.time
    /// - DayTimeIntervalType: needs conversion to/from datetime.timedelta
    /// - CalendarIntervalType: needs conversion
    /// - YearMonthIntervalType: needs conversion (complex)
    /// - VariantType: needs conversion to/from `VariantVal`
    /// - GeometryType and GeographyType: need conversion to/from `Geometry` and `Geography`
    /// - UserDefinedType: needs conversion to/from its Python class
    /// - ArrayType: if element type needs conversion
    /// - MapType: if key or value type needs conversion
    /// - StructType: always needs conversion
    pub fn need_conversion(&self) -> bool {
        match self {
            DataType::Date
            | DataType::Timestamp
            | DataType::TimestampNtz
            | DataType::TimestampNtzNanos { .. }
            | DataType::TimestampLtzNanos { .. }
            | DataType::Time { .. }
            | DataType::DayTimeInterval { .. }
            | DataType::YearMonthInterval { .. }
            | DataType::CalendarInterval
            | DataType::Variant
            | DataType::Geometry { .. }
            | DataType::Geography { .. }
            | DataType::Udt { .. } => true,
            DataType::Array { element_type, .. } => element_type.need_conversion(),
            DataType::Map {
                key_type,
                value_type,
                ..
            } => key_type.need_conversion() || value_type.need_conversion(),
            DataType::Struct { .. } => true,
            _ => false,
        }
    }

    /// Returns the type name, mirroring `DataType.typeName()`.
    ///
    /// For most types, this is the class name with the "Type" suffix removed and lowercased.
    /// E.g., "ByteType" -> "byte", but NullType -> "void", and special handling for others.
    pub fn type_name(&self) -> String {
        match self {
            DataType::Null => "void".to_string(),
            DataType::Boolean => "boolean".to_string(),
            DataType::Byte => "byte".to_string(),
            DataType::Short => "short".to_string(),
            DataType::Integer => "integer".to_string(),
            DataType::Long => "long".to_string(),
            DataType::Float => "float".to_string(),
            DataType::Double => "double".to_string(),
            DataType::Decimal { .. } => "decimal".to_string(),
            DataType::String { .. } => "string".to_string(),
            DataType::Char { .. } => "char".to_string(),
            DataType::Varchar { .. } => "varchar".to_string(),
            DataType::Binary => "binary".to_string(),
            DataType::Date => "date".to_string(),
            DataType::Timestamp => "timestamp".to_string(),
            DataType::TimestampNtz => "timestamp_ntz".to_string(),
            DataType::Time { .. } => "time".to_string(),
            DataType::TimestampNtzNanos { precision } => format!("timestamp_ntz({})", precision),
            DataType::TimestampLtzNanos { precision } => format!("timestamp_ltz({})", precision),
            DataType::CalendarInterval => "interval".to_string(),
            DataType::YearMonthInterval { .. } => "yearmonthinterval".to_string(),
            DataType::DayTimeInterval { .. } => "daytimeinterval".to_string(),
            DataType::Array { .. } => "array".to_string(),
            DataType::Map { .. } => "map".to_string(),
            DataType::Struct { .. } => "struct".to_string(),
            DataType::Variant => "variant".to_string(),
            DataType::Geometry { .. } => "geometry".to_string(),
            DataType::Geography { .. } => "geography".to_string(),
            // UserDefinedType.typeName() is the lower-cased class name, the whole of it.
            DataType::Udt { python_class, .. } => python_class
                .as_deref()
                .and_then(|c| c.rsplit('.').next())
                .filter(|class_name| !class_name.is_empty())
                .map(str::to_lowercase)
                .unwrap_or_else(|| "udt".to_string()),
            DataType::Unparsed { .. } => "unparsed".to_string(),
        }
    }

    /// Returns the simple string representation, mirroring `DataType.simpleString()`.
    ///
    /// For example:
    /// - "int", "string", "boolean"
    /// - "decimal(10,0)", "char(50)", "varchar(100)"
    /// - "array<int>", "map<string,int>", "struct<name:string,age:int>"
    /// - "interval day to second"
    pub fn simple_string(&self) -> String {
        match self {
            DataType::Null => "void".to_string(),
            DataType::Boolean => "boolean".to_string(),
            DataType::Byte => "tinyint".to_string(),
            DataType::Short => "smallint".to_string(),
            DataType::Integer => "int".to_string(),
            DataType::Long => "bigint".to_string(),
            DataType::Float => "float".to_string(),
            DataType::Double => "double".to_string(),
            DataType::Decimal { precision, scale } => {
                format!("decimal({},{})", precision, scale)
            }
            DataType::String { collation } => {
                // The default collation (empty, or the explicit UTF8_BINARY) renders as
                // plain "string"; only a non-default collation adds " collate <name>".
                // Previously an empty collation produced the malformed "string collate ".
                if collation.is_empty() || collation == "UTF8_BINARY" {
                    "string".to_string()
                } else {
                    format!("string collate {}", collation)
                }
            }
            DataType::Char { length } => format!("char({})", length),
            DataType::Varchar { length } => format!("varchar({})", length),
            DataType::Binary => "binary".to_string(),
            DataType::Date => "date".to_string(),
            DataType::Timestamp => "timestamp".to_string(),
            DataType::TimestampNtz => "timestamp_ntz".to_string(),
            DataType::Time { precision } => format!("time({})", precision),
            DataType::TimestampNtzNanos { precision } => format!("timestamp_ntz({})", precision),
            DataType::TimestampLtzNanos { precision } => format!("timestamp_ltz({})", precision),
            DataType::CalendarInterval => "interval".to_string(),
            DataType::YearMonthInterval {
                start_field,
                end_field,
            } => interval_string(*start_field, *end_field, YEAR_MONTH_FIELDS),
            DataType::DayTimeInterval {
                start_field,
                end_field,
            } => interval_string(*start_field, *end_field, DAY_TIME_FIELDS),
            DataType::Array {
                element_type,
                contains_null: _,
            } => {
                format!("array<{}>", element_type.simple_string())
            }
            DataType::Map {
                key_type,
                value_type,
                value_contains_null: _,
            } => {
                format!(
                    "map<{},{}>",
                    key_type.simple_string(),
                    value_type.simple_string()
                )
            }
            DataType::Struct { fields } => {
                let field_strs: Vec<String> = fields.iter().map(|f| f.simple_string()).collect();
                format!("struct<{}>", field_strs.join(","))
            }
            DataType::Variant => "variant".to_string(),
            DataType::Geometry { srid } => {
                if *srid == -1 {
                    "geometry(any)".to_string()
                } else {
                    format!("geometry({})", srid)
                }
            }
            DataType::Geography { srid } => {
                if *srid == -1 {
                    "geography(any)".to_string()
                } else {
                    format!("geography({})", srid)
                }
            }
            DataType::Udt { .. } => "udt".to_string(),
            DataType::Unparsed { data_type_string } => {
                // Round-trips with the `unparsed(...)` branch in `from_json`.
                format!("unparsed({})", data_type_string)
            }
        }
    }

    /// Returns the JSON value representation, mirroring `DataType.jsonValue()`.
    ///
    /// Most simple types return their type name as a string. Complex types
    /// (Array, Map, Struct) return a dictionary with type and component info.
    pub fn json_value(&self) -> serde_json::Value {
        match self {
            DataType::Null => serde_json::json!("void"),
            DataType::Boolean => serde_json::json!("boolean"),
            DataType::Byte => serde_json::json!("byte"),
            DataType::Short => serde_json::json!("short"),
            DataType::Integer => serde_json::json!("integer"),
            DataType::Long => serde_json::json!("long"),
            DataType::Float => serde_json::json!("float"),
            DataType::Double => serde_json::json!("double"),
            DataType::Decimal { precision, scale } => {
                serde_json::json!(format!("decimal({},{})", precision, scale))
            }
            DataType::String { collation } => {
                if collation == "UTF8_BINARY" {
                    serde_json::json!("string")
                } else {
                    serde_json::json!(format!("string collate {}", collation))
                }
            }
            DataType::Char { length } => {
                serde_json::json!(format!("char({})", length))
            }
            DataType::Varchar { length } => {
                serde_json::json!(format!("varchar({})", length))
            }
            DataType::Binary => serde_json::json!("binary"),
            DataType::Date => serde_json::json!("date"),
            DataType::Timestamp => serde_json::json!("timestamp"),
            DataType::TimestampNtz => serde_json::json!("timestamp_ntz"),
            DataType::Time { precision } => {
                serde_json::json!(format!("time({})", precision))
            }
            DataType::TimestampNtzNanos { precision } => {
                serde_json::json!(format!("timestamp_ntz({})", precision))
            }
            DataType::TimestampLtzNanos { precision } => {
                serde_json::json!(format!("timestamp_ltz({})", precision))
            }
            DataType::CalendarInterval => serde_json::json!("interval"),
            DataType::YearMonthInterval {
                start_field,
                end_field,
            } => {
                serde_json::json!(interval_string(*start_field, *end_field, YEAR_MONTH_FIELDS))
            }
            DataType::DayTimeInterval {
                start_field,
                end_field,
            } => {
                serde_json::json!(interval_string(*start_field, *end_field, DAY_TIME_FIELDS))
            }
            DataType::Array {
                element_type,
                contains_null,
            } => {
                serde_json::json!({
                    "type": "array",
                    "elementType": element_type.json_value(),
                    "containsNull": contains_null,
                })
            }
            DataType::Map {
                key_type,
                value_type,
                value_contains_null,
            } => {
                serde_json::json!({
                    "type": "map",
                    "keyType": key_type.json_value(),
                    "valueType": value_type.json_value(),
                    "valueContainsNull": value_contains_null,
                })
            }
            DataType::Struct { fields } => {
                let field_values: Vec<serde_json::Value> =
                    fields.iter().map(|f| f.json_value()).collect();
                serde_json::json!({
                    "type": "struct",
                    "fields": field_values,
                })
            }
            DataType::Variant => serde_json::json!("variant"),
            DataType::Geometry { srid } => {
                serde_json::json!(format!("geometry({})", spatial_crs(*srid, false)))
            }
            DataType::Geography { srid } => serde_json::json!(format!(
                "geography({}, {})",
                spatial_crs(*srid, true),
                GeographyType::DEFAULT_ALG
            )),
            DataType::Udt {
                type_str: _,
                jvm_class,
                python_class,
                serialized_python_class,
                sql_type,
            } => {
                let mut obj = serde_json::Map::new();
                obj.insert("type".to_string(), serde_json::json!("udt"));
                if let Some(jvm_cls) = jvm_class {
                    obj.insert("class".to_string(), serde_json::json!(jvm_cls));
                }
                if let Some(py_cls) = python_class {
                    obj.insert("pyClass".to_string(), serde_json::json!(py_cls));
                }
                if let Some(serialized) = serialized_python_class {
                    obj.insert("serializedClass".to_string(), serde_json::json!(serialized));
                }
                if let Some(sql_ty) = sql_type {
                    obj.insert("sqlType".to_string(), sql_ty.json_value());
                }
                serde_json::Value::Object(obj)
            }
            DataType::Unparsed { data_type_string } => {
                serde_json::json!(format!("unparsed({})", data_type_string))
            }
        }
    }

    /// Converts to a JSON string, mirroring `DataType.json()`.
    pub fn json(&self) -> String {
        self.json_value().to_string()
    }

    /// Parses a JSON value into a DataType, mirroring the reverse of `json()` / `jsonValue()`.
    pub fn from_json(value: &serde_json::Value) -> Result<DataType> {
        parse_json_value(value, "", &CollationsMap::new())
    }

    /// Parses a JSON string into a DataType (convenience over [`from_json`]).
    pub fn from_json_str(s: &str) -> Result<DataType> {
        let value: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| SparkError::value("INVALID_JSON", &[("detail", &e.to_string())]))?;
        DataType::from_json(&value)
    }

    /// Converts to a protobuf DataType, mirroring
    /// `pyspark.sql.connect.types.pyspark_types_to_proto_types`.
    pub fn to_proto(&self) -> spark_connect_proto::DataType {
        let mut proto = spark_connect_proto::DataType::default();

        match self {
            DataType::Null => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Null(
                    spark_connect_proto::data_type::Null::default(),
                ));
            }
            DataType::Boolean => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Boolean(
                    spark_connect_proto::data_type::Boolean::default(),
                ));
            }
            DataType::Byte => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Byte(
                    spark_connect_proto::data_type::Byte::default(),
                ));
            }
            DataType::Short => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Short(
                    spark_connect_proto::data_type::Short::default(),
                ));
            }
            DataType::Integer => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Integer(
                    spark_connect_proto::data_type::Integer::default(),
                ));
            }
            DataType::Long => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Long(
                    spark_connect_proto::data_type::Long::default(),
                ));
            }
            DataType::Float => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Float(
                    spark_connect_proto::data_type::Float::default(),
                ));
            }
            DataType::Double => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Double(
                    spark_connect_proto::data_type::Double::default(),
                ));
            }
            DataType::Decimal { precision, scale } => {
                let mut decimal = spark_connect_proto::data_type::Decimal::default();
                decimal.precision = Some(*precision);
                decimal.scale = Some(*scale);
                proto.kind = Some(spark_connect_proto::data_type::Kind::Decimal(decimal));
            }
            DataType::String { collation } => {
                let mut string = spark_connect_proto::data_type::String::default();
                string.collation = collation.clone();
                proto.kind = Some(spark_connect_proto::data_type::Kind::String(string));
            }
            DataType::Char { length } => {
                let mut char_type = spark_connect_proto::data_type::Char::default();
                char_type.length = *length;
                proto.kind = Some(spark_connect_proto::data_type::Kind::Char(char_type));
            }
            DataType::Varchar { length } => {
                let mut varchar_type = spark_connect_proto::data_type::VarChar::default();
                varchar_type.length = *length;
                proto.kind = Some(spark_connect_proto::data_type::Kind::VarChar(varchar_type));
            }
            DataType::Binary => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Binary(
                    spark_connect_proto::data_type::Binary::default(),
                ));
            }
            DataType::Date => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Date(
                    spark_connect_proto::data_type::Date::default(),
                ));
            }
            DataType::Timestamp => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Timestamp(
                    spark_connect_proto::data_type::Timestamp::default(),
                ));
            }
            DataType::TimestampNtz => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::TimestampNtz(
                    spark_connect_proto::data_type::TimestampNtz::default(),
                ));
            }
            DataType::Time { precision } => {
                let mut time_type = spark_connect_proto::data_type::Time::default();
                time_type.precision = Some(*precision);
                proto.kind = Some(spark_connect_proto::data_type::Kind::Time(time_type));
            }
            DataType::TimestampNtzNanos { precision } => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::TimestampNtzNanos(
                    spark_connect_proto::data_type::TimestampNtzNanos {
                        precision: Some(*precision),
                        ..Default::default()
                    },
                ));
            }
            DataType::TimestampLtzNanos { precision } => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::TimestampLtzNanos(
                    spark_connect_proto::data_type::TimestampLtzNanos {
                        precision: Some(*precision),
                        ..Default::default()
                    },
                ));
            }
            DataType::CalendarInterval => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::CalendarInterval(
                    spark_connect_proto::data_type::CalendarInterval::default(),
                ));
            }
            DataType::YearMonthInterval {
                start_field,
                end_field,
            } => {
                let mut ymi = spark_connect_proto::data_type::YearMonthInterval::default();
                ymi.start_field = Some(*start_field);
                ymi.end_field = Some(*end_field);
                proto.kind = Some(spark_connect_proto::data_type::Kind::YearMonthInterval(ymi));
            }
            DataType::DayTimeInterval {
                start_field,
                end_field,
            } => {
                let mut dti = spark_connect_proto::data_type::DayTimeInterval::default();
                dti.start_field = Some(*start_field);
                dti.end_field = Some(*end_field);
                proto.kind = Some(spark_connect_proto::data_type::Kind::DayTimeInterval(dti));
            }
            DataType::Array {
                element_type,
                contains_null,
            } => {
                let mut array = spark_connect_proto::data_type::Array::default();
                array.element_type = Some(Box::new(element_type.to_proto()));
                array.contains_null = *contains_null;
                proto.kind = Some(spark_connect_proto::data_type::Kind::Array(Box::new(array)));
            }
            DataType::Map {
                key_type,
                value_type,
                value_contains_null,
            } => {
                let mut map = spark_connect_proto::data_type::Map::default();
                map.key_type = Some(Box::new(key_type.to_proto()));
                map.value_type = Some(Box::new(value_type.to_proto()));
                map.value_contains_null = *value_contains_null;
                proto.kind = Some(spark_connect_proto::data_type::Kind::Map(Box::new(map)));
            }
            DataType::Struct { fields } => {
                let mut struct_type = spark_connect_proto::data_type::Struct::default();
                for field in fields {
                    let mut proto_field = spark_connect_proto::data_type::StructField::default();
                    proto_field.name = field.name.clone();
                    proto_field.data_type = Some(field.data_type.to_proto());
                    proto_field.nullable = field.nullable;
                    if !field.metadata.is_empty() {
                        proto_field.metadata =
                            Some(serde_json::to_string(&field.metadata).unwrap_or_default());
                    }
                    struct_type.fields.push(proto_field);
                }
                proto.kind = Some(spark_connect_proto::data_type::Kind::Struct(struct_type));
            }
            DataType::Variant => {
                proto.kind = Some(spark_connect_proto::data_type::Kind::Variant(
                    spark_connect_proto::data_type::Variant::default(),
                ));
            }
            DataType::Geometry { srid } => {
                let mut geometry = spark_connect_proto::data_type::Geometry::default();
                geometry.srid = *srid;
                proto.kind = Some(spark_connect_proto::data_type::Kind::Geometry(geometry));
            }
            DataType::Geography { srid } => {
                let mut geography = spark_connect_proto::data_type::Geography::default();
                geography.srid = *srid;
                proto.kind = Some(spark_connect_proto::data_type::Kind::Geography(geography));
            }
            DataType::Udt {
                type_str: _,
                jvm_class,
                python_class,
                serialized_python_class,
                sql_type,
            } => {
                let mut udt = spark_connect_proto::data_type::Udt::default();
                udt.r#type = "udt".to_string();
                if let Some(jvm_cls) = jvm_class {
                    udt.jvm_class = Some(jvm_cls.clone());
                }
                if let Some(py_cls) = python_class {
                    udt.python_class = Some(py_cls.clone());
                }
                if let Some(serialized) = serialized_python_class {
                    udt.serialized_python_class = Some(serialized.clone());
                }
                if let Some(sql_ty) = sql_type {
                    udt.sql_type = Some(Box::new(sql_ty.to_proto()));
                }
                proto.kind = Some(spark_connect_proto::data_type::Kind::Udt(Box::new(udt)));
            }
            DataType::Unparsed { data_type_string } => {
                let mut unparsed = spark_connect_proto::data_type::Unparsed::default();
                unparsed.data_type_string = data_type_string.clone();
                proto.kind = Some(spark_connect_proto::data_type::Kind::Unparsed(unparsed));
            }
        }

        proto
    }

    /// Converts from a protobuf DataType, mirroring
    /// `pyspark.sql.connect.types.proto_schema_to_pyspark_data_type`.
    pub fn from_proto(proto: &spark_connect_proto::DataType) -> Result<DataType> {
        match &proto.kind {
            Some(spark_connect_proto::data_type::Kind::Null(_)) => Ok(DataType::Null),
            Some(spark_connect_proto::data_type::Kind::Boolean(_)) => Ok(DataType::Boolean),
            Some(spark_connect_proto::data_type::Kind::Byte(_)) => Ok(DataType::Byte),
            Some(spark_connect_proto::data_type::Kind::Short(_)) => Ok(DataType::Short),
            Some(spark_connect_proto::data_type::Kind::Integer(_)) => Ok(DataType::Integer),
            Some(spark_connect_proto::data_type::Kind::Long(_)) => Ok(DataType::Long),
            Some(spark_connect_proto::data_type::Kind::Float(_)) => Ok(DataType::Float),
            Some(spark_connect_proto::data_type::Kind::Double(_)) => Ok(DataType::Double),
            Some(spark_connect_proto::data_type::Kind::Decimal(d)) => {
                let precision = d.precision.unwrap_or(10);
                let scale = d.scale.unwrap_or(0);
                Ok(DataType::Decimal { precision, scale })
            }
            Some(spark_connect_proto::data_type::Kind::String(s)) => {
                let collation = if s.collation.is_empty() {
                    "UTF8_BINARY".to_string()
                } else {
                    s.collation.clone()
                };
                Ok(DataType::String { collation })
            }
            Some(spark_connect_proto::data_type::Kind::Char(c)) => {
                Ok(DataType::Char { length: c.length })
            }
            Some(spark_connect_proto::data_type::Kind::VarChar(vc)) => {
                Ok(DataType::Varchar { length: vc.length })
            }
            Some(spark_connect_proto::data_type::Kind::Binary(_)) => Ok(DataType::Binary),
            Some(spark_connect_proto::data_type::Kind::Date(_)) => Ok(DataType::Date),
            Some(spark_connect_proto::data_type::Kind::Timestamp(_)) => Ok(DataType::Timestamp),
            Some(spark_connect_proto::data_type::Kind::TimestampNtz(_)) => {
                Ok(DataType::TimestampNtz)
            }
            Some(spark_connect_proto::data_type::Kind::Time(t)) => {
                let precision = t.precision.unwrap_or(6);
                Ok(DataType::Time { precision })
            }
            Some(spark_connect_proto::data_type::Kind::CalendarInterval(_)) => {
                Ok(DataType::CalendarInterval)
            }
            Some(spark_connect_proto::data_type::Kind::YearMonthInterval(ymi)) => {
                let start_field = ymi.start_field.unwrap_or(0);
                let end_field = ymi.end_field.unwrap_or(1);
                Ok(DataType::YearMonthInterval {
                    start_field,
                    end_field,
                })
            }
            Some(spark_connect_proto::data_type::Kind::DayTimeInterval(dti)) => {
                let start_field = dti.start_field.unwrap_or(0);
                let end_field = dti.end_field.unwrap_or(3);
                Ok(DataType::DayTimeInterval {
                    start_field,
                    end_field,
                })
            }
            Some(spark_connect_proto::data_type::Kind::Array(a)) => {
                let element_type =
                    Box::new(DataType::from_proto(a.element_type.as_ref().ok_or_else(
                        || SparkError::connect_msg("Array element_type is missing"),
                    )?)?);
                Ok(DataType::Array {
                    element_type,
                    contains_null: a.contains_null,
                })
            }
            Some(spark_connect_proto::data_type::Kind::Struct(s)) => {
                let mut fields = Vec::new();
                for field_proto in &s.fields {
                    let data_type =
                        DataType::from_proto(field_proto.data_type.as_ref().ok_or_else(|| {
                            SparkError::connect_msg("StructField data_type is missing")
                        })?)?;
                    let metadata = if let Some(meta_str) = &field_proto.metadata {
                        serde_json::from_str(meta_str).unwrap_or_default()
                    } else {
                        BTreeMap::new()
                    };
                    fields.push(StructField {
                        name: field_proto.name.clone(),
                        data_type,
                        nullable: field_proto.nullable,
                        metadata,
                    });
                }
                Ok(DataType::Struct { fields })
            }
            Some(spark_connect_proto::data_type::Kind::Map(m)) => {
                let key_type =
                    Box::new(DataType::from_proto(m.key_type.as_ref().ok_or_else(
                        || SparkError::connect_msg("Map key_type is missing"),
                    )?)?);
                let value_type =
                    Box::new(DataType::from_proto(m.value_type.as_ref().ok_or_else(
                        || SparkError::connect_msg("Map value_type is missing"),
                    )?)?);
                Ok(DataType::Map {
                    key_type,
                    value_type,
                    value_contains_null: m.value_contains_null,
                })
            }
            Some(spark_connect_proto::data_type::Kind::Variant(_)) => Ok(DataType::Variant),
            Some(spark_connect_proto::data_type::Kind::Geometry(g)) => {
                Ok(DataType::Geometry { srid: g.srid })
            }
            Some(spark_connect_proto::data_type::Kind::Geography(g)) => {
                Ok(DataType::Geography { srid: g.srid })
            }
            Some(spark_connect_proto::data_type::Kind::Udt(u)) => {
                let sql_type = if let Some(ref st) = u.sql_type {
                    Some(Box::new(DataType::from_proto(st)?))
                } else {
                    None
                };
                Ok(DataType::Udt {
                    type_str: u.r#type.clone(),
                    jvm_class: u.jvm_class.clone(),
                    python_class: u.python_class.clone(),
                    serialized_python_class: u.serialized_python_class.clone(),
                    sql_type,
                })
            }
            Some(spark_connect_proto::data_type::Kind::Unparsed(u)) => Ok(DataType::Unparsed {
                data_type_string: u.data_type_string.clone(),
            }),
            // Precision defaults to 9 when omitted (types.proto).
            Some(spark_connect_proto::data_type::Kind::TimestampNtzNanos(tn)) => {
                Ok(DataType::TimestampNtzNanos {
                    precision: tn.precision.unwrap_or(9),
                })
            }
            Some(spark_connect_proto::data_type::Kind::TimestampLtzNanos(tl)) => {
                Ok(DataType::TimestampLtzNanos {
                    precision: tl.precision.unwrap_or(9),
                })
            }
            None => Err(SparkError::connect_msg("DataType kind not set")),
        }
    }
}

impl StructField {
    /// Returns the simple string representation, mirroring `StructField.simpleString()`.
    ///
    /// Format: "name:type"
    pub fn simple_string(&self) -> String {
        format!("{}:{}", self.name, self.data_type.simple_string())
    }

    /// Returns the JSON value representation, mirroring `StructField.jsonValue()`: the
    /// collations of the strings of the field, also in its arrays and maps, are in the
    /// `__COLLATIONS` of its metadata, by their paths, and not in its type, as Spark writes them.
    pub fn json_value(&self) -> serde_json::Value {
        let mut collations = serde_json::Map::new();
        visit_collations(&self.data_type, &self.name, &mut collations);
        let (data_type, metadata) = if collations.is_empty() {
            (self.data_type.json_value(), self.metadata.clone())
        } else {
            let mut metadata = self.metadata.clone();
            metadata.insert(
                COLLATIONS_METADATA_KEY.to_string(),
                serde_json::Value::Object(collations),
            );
            (without_collations(&self.data_type).json_value(), metadata)
        };
        serde_json::json!({
            "name": &self.name,
            "type": data_type,
            "nullable": self.nullable,
            "metadata": metadata,
        })
    }

    /// Parses a StructField from its JSON string, mirroring `StructField.fromJson`.
    pub fn from_json_str(s: &str) -> Result<StructField> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| SparkError::value("INVALID_JSON", &[("detail", &e.to_string())]))?;
        let field = v.as_object().ok_or_else(|| {
            SparkError::value("INVALID_JSON", &[("detail", "a field is not an object")])
        })?;
        parse_json_field(field)
    }
}

/// Helper methods for StructType operations, mirroring `pyspark.sql.types.StructType`.
/// Since StructType is represented as `DataType::Struct { fields }`, these methods provide
/// convenience operations for struct types.
impl DataType {
    /// Returns all field names in a StructType, mirroring `StructType.fieldNames()`.
    ///
    /// Returns an error if called on a non-Struct type.
    pub fn field_names(&self) -> Result<Vec<String>> {
        match self {
            DataType::Struct { fields } => Ok(fields.iter().map(|f| f.name.clone()).collect()),
            _ => Err(SparkError::value(
                "INVALID_TYPE",
                &[("detail", "fieldNames() can only be called on StructType")],
            )),
        }
    }

    /// Alias for `field_names()`, also mirroring pyspark's `names` attribute.
    pub fn names(&self) -> Result<Vec<String>> {
        self.field_names()
    }

    /// DDL string for a StructType, mirroring `StructType.toDDL()`:
    /// comma-separated `name type[ NOT NULL][ COMMENT '...']` per field. The names are quoted
    /// and the types written as `DataType.sql`, so the server parses the DDL back.
    pub fn to_ddl(&self) -> Result<String> {
        match self {
            DataType::Struct { fields } => Ok(fields
                .iter()
                .map(|f| {
                    format!(
                        "{} {}{}",
                        quote_if_needed(&f.name),
                        f.data_type.sql_type(),
                        field_ddl_suffix(f)
                    )
                })
                .collect::<Vec<_>>()
                .join(",")),
            _ => Err(SparkError::value(
                "INVALID_TYPE",
                &[("detail", "toDDL() can only be called on StructType")],
            )),
        }
    }

    /// The SQL form of a type, mirroring `DataType.sql`: `INT`, `STRING COLLATE UTF8_LCASE`,
    /// `ARRAY<...>`, `MAP<K, V>` and `STRUCT<name: TYPE, ...>`, the form `toDDL` writes.
    fn sql_type(&self) -> String {
        match self {
            DataType::Struct { fields } => {
                let fields: Vec<String> = fields
                    .iter()
                    .map(|f| {
                        format!(
                            "{}: {}{}",
                            quote_if_needed(&f.name),
                            f.data_type.sql_type(),
                            field_ddl_suffix(f)
                        )
                    })
                    .collect();
                format!("STRUCT<{}>", fields.join(", "))
            }
            DataType::Array { element_type, .. } => format!("ARRAY<{}>", element_type.sql_type()),
            DataType::Map {
                key_type,
                value_type,
                ..
            } => format!("MAP<{}, {}>", key_type.sql_type(), value_type.sql_type()),
            DataType::String { collation }
                if !(collation.is_empty() || collation == "UTF8_BINARY") =>
            {
                format!("STRING COLLATE {}", collation.to_uppercase())
            }
            _ => self.simple_string().to_uppercase(),
        }
    }

    /// Tree-string for a StructType, mirroring `StructType.treeString()`.
    pub fn tree_string(&self) -> Result<String> {
        self.tree_string_with_depth(i32::MAX)
    }

    /// Like [`DataType::tree_string`], but limits nesting through structs,
    /// arrays, and maps to `max_depth` levels. Top-level fields are depth 1;
    /// zero or negative values print the full tree.
    /// Mirrors `StructType.treeString(maxDepth)`.
    pub fn tree_string_with_depth(&self, max_depth: i32) -> Result<String> {
        match self {
            DataType::Struct { .. } => {
                let mut out = String::from("root\n");
                self.append_tree(
                    &mut out,
                    " |",
                    if max_depth > 0 { max_depth } else { i32::MAX },
                );
                Ok(out)
            }
            _ => Err(SparkError::value(
                "INVALID_TYPE",
                &[("detail", "treeString() can only be called on StructType")],
            )),
        }
    }

    fn tree_type_name(&self) -> String {
        match self {
            DataType::Decimal { .. }
            | DataType::Char { .. }
            | DataType::Varchar { .. }
            | DataType::Time { .. }
            | DataType::YearMonthInterval { .. }
            | DataType::DayTimeInterval { .. } => self.simple_string(),
            _ => self.type_name(),
        }
    }

    fn append_tree(&self, out: &mut String, prefix: &str, remaining: i32) {
        if remaining <= 0 {
            return;
        }
        match self {
            DataType::Struct { fields } => {
                let child_prefix = format!("{prefix}    |");
                for f in fields {
                    out.push_str(prefix);
                    out.push_str("-- ");
                    push_escaped_name(out, &f.name);
                    writeln!(
                        out,
                        ": {} (nullable = {})",
                        f.data_type.tree_type_name(),
                        f.nullable
                    )
                    .expect("writing to a String cannot fail");
                    f.data_type.append_tree(out, &child_prefix, remaining - 1);
                }
            }
            DataType::Array {
                element_type,
                contains_null,
            } => {
                let child_prefix = format!("{prefix}    |");
                writeln!(
                    out,
                    "{}-- element: {} (containsNull = {})",
                    prefix,
                    element_type.tree_type_name(),
                    contains_null
                )
                .expect("writing to a String cannot fail");
                element_type.append_tree(out, &child_prefix, remaining - 1);
            }
            DataType::Map {
                key_type,
                value_type,
                value_contains_null,
            } => {
                let child_prefix = format!("{prefix}    |");
                writeln!(out, "{}-- key: {}", prefix, key_type.tree_type_name())
                    .expect("writing to a String cannot fail");
                key_type.append_tree(out, &child_prefix, remaining - 1);
                writeln!(
                    out,
                    "{}-- value: {} (valueContainsNull = {})",
                    prefix,
                    value_type.tree_type_name(),
                    value_contains_null
                )
                .expect("writing to a String cannot fail");
                value_type.append_tree(out, &child_prefix, remaining - 1);
            }
            _ => {}
        }
    }

    /// Return a copy with every field made nullable (recursively), mirroring
    /// `StructType.toNullable()`.
    pub fn to_nullable(&self) -> DataType {
        match self {
            DataType::Struct { fields } => DataType::Struct {
                fields: fields
                    .iter()
                    .map(|f| StructField {
                        name: f.name.clone(),
                        data_type: f.data_type.to_nullable(),
                        nullable: true,
                        metadata: f.metadata.clone(),
                    })
                    .collect(),
            },
            DataType::Array { element_type, .. } => DataType::Array {
                element_type: Box::new(element_type.to_nullable()),
                contains_null: true,
            },
            DataType::Map {
                key_type,
                value_type,
                ..
            } => DataType::Map {
                key_type: Box::new(key_type.to_nullable()),
                value_type: Box::new(value_type.to_nullable()),
                value_contains_null: true,
            },
            other => other.clone(),
        }
    }

    /// Adds a field to a StructType, mirroring `StructType.add()`.
    ///
    /// This is a builder method that returns a new StructType with the field added.
    /// Returns an error if called on a non-Struct type.
    ///
    /// Example:
    /// ```ignore
    /// let struct_type = DataType::Struct { fields: vec![] };
    /// let with_field = struct_type.add(
    ///     "name",
    ///     DataType::String { collation: "UTF8_BINARY".to_string() },
    ///     true,
    ///     None,
    /// )?;
    /// ```
    pub fn add(
        &self,
        field_name: &str,
        field_type: DataType,
        nullable: bool,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    ) -> Result<DataType> {
        match self {
            DataType::Struct { fields } => {
                let mut new_fields = fields.clone();
                new_fields.push(StructField {
                    name: field_name.to_string(),
                    data_type: field_type,
                    nullable,
                    metadata: metadata.unwrap_or_default(),
                });
                Ok(DataType::Struct { fields: new_fields })
            }
            _ => Err(SparkError::value(
                "INVALID_TYPE",
                &[("detail", "add() can only be called on StructType")],
            )),
        }
    }
}

fn push_escaped_name(out: &mut String, name: &str) {
    for ch in name.chars() {
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x0c' => out.push_str("\\f"),
            '\x08' => out.push_str("\\b"),
            '\x0b' => out.push_str("\\v"),
            '\x07' => out.push_str("\\a"),
            _ => out.push(ch),
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.simple_string())
    }
}

impl Hash for DataType {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.simple_string().hash(state);
    }
}

// Helper constants for interval field mappings
const DAY_TIME_FIELDS: &[(&str, i32)] = &[("day", 0), ("hour", 1), ("minute", 2), ("second", 3)];

const YEAR_MONTH_FIELDS: &[(&str, i32)] = &[("year", 0), ("month", 1)];

/// Helper to format interval strings
fn interval_string(start_field: i32, end_field: i32, fields: &[(&str, i32)]) -> String {
    let field_map: std::collections::HashMap<i32, &str> =
        fields.iter().map(|(name, code)| (*code, *name)).collect();

    if let (Some(&start_name), Some(&end_name)) =
        (field_map.get(&start_field), field_map.get(&end_field))
    {
        if start_name == end_name {
            format!("interval {}", start_name)
        } else {
            format!("interval {} to {}", start_name, end_name)
        }
    } else {
        "interval".to_string()
    }
}

/// Quotes a field name with backticks, as Spark's `QuotingUtils.quoteIfNeeded` does: a name
/// that is not `[a-zA-Z_][a-zA-Z0-9_]*` is quoted, and each backtick in it is doubled.
fn quote_if_needed(name: &str) -> String {
    let mut chars = name.chars();
    let plain = matches!(chars.next(), Some(c) if c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric());
    if plain {
        name.to_string()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

/// The ` NOT NULL` and ` COMMENT '...'` suffix of a field, as `StructField.toDDL` writes it. The
/// backslashes and single quotes of the comment are escaped as Spark's `escapeSingleQuotedString`
/// does: the backslashes first, so that the escape of a quote is not escaped again.
fn field_ddl_suffix(field: &StructField) -> String {
    let mut s = String::new();
    if !field.nullable {
        s.push_str(" NOT NULL");
    }
    if let Some(comment) = field.metadata.get("comment").and_then(|c| c.as_str()) {
        let escaped = comment.replace('\\', "\\\\").replace('\'', "\\'");
        s.push_str(&format!(" COMMENT '{escaped}'"));
    }
    s
}

/// The key of the metadata of a struct field of the collations of its strings, as Spark's.
const COLLATIONS_METADATA_KEY: &str = "__COLLATIONS";

/// The collations of the strings of a field of `data_type` at `path`, also in its arrays and
/// maps but not in its structs, whose fields have their own, as PySpark's
/// `StructField.getCollationMetadata`: `<provider>.<collation>` of each path.
fn visit_collations(
    data_type: &DataType,
    path: &str,
    collations: &mut serde_json::Map<String, serde_json::Value>,
) {
    match data_type {
        DataType::Array { element_type, .. } => {
            visit_collations(element_type, &format!("{path}.element"), collations)
        }
        DataType::Map {
            key_type,
            value_type,
            ..
        } => {
            visit_collations(key_type, &format!("{path}.key"), collations);
            visit_collations(value_type, &format!("{path}.value"), collations);
        }
        DataType::String { collation } if collation != "UTF8_BINARY" => {
            // As PySpark's `StringType.collationProvider`.
            let provider = if collation.starts_with("UTF8") {
                "spark"
            } else {
                "icu"
            };
            collations.insert(path.to_string(), format!("{provider}.{collation}").into());
        }
        _ => {}
    }
}

/// `data_type` without the collations of its strings, also in its arrays and maps, which the
/// JSON of a struct field keeps in its metadata.
fn without_collations(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Array {
            element_type,
            contains_null,
        } => DataType::Array {
            element_type: Box::new(without_collations(element_type)),
            contains_null: *contains_null,
        },
        DataType::Map {
            key_type,
            value_type,
            value_contains_null,
        } => DataType::Map {
            key_type: Box::new(without_collations(key_type)),
            value_type: Box::new(without_collations(value_type)),
            value_contains_null: *value_contains_null,
        },
        DataType::String { .. } => DataType::String {
            collation: "UTF8_BINARY".to_string(),
        },
        other => other.clone(),
    }
}

/// The collations of the strings of a struct field by their paths, `__COLLATIONS` of its
/// metadata without their providers, as the `collationsMap` of PySpark's `fromJson`.
type CollationsMap = std::collections::HashMap<String, String>;

/// The JSON type names to which a collation of a struct field applies, as PySpark's
/// `_all_mappable_types`: it makes a `string` a collated string, and raises
/// `INVALID_JSON_DATA_TYPE_FOR_COLLATIONS` for the others.
const MAPPABLE_TYPE_NAMES: [&str; 16] = [
    "string",
    "binary",
    "boolean",
    "decimal",
    "float",
    "double",
    "byte",
    "short",
    "integer",
    "long",
    "date",
    "timestamp",
    "timestamp_ntz",
    "void",
    "variant",
    "interval",
];

/// The path of `child` (`element`, `key` or `value`) of a type at `path`, as PySpark's
/// `fromJson` of arrays and maps names it.
fn child_path(path: &str, child: &str) -> String {
    if path.is_empty() {
        child.to_string()
    } else {
        format!("{path}.{child}")
    }
}

/// The collations of the `__COLLATIONS` of `metadata`, which this removes, as PySpark's
/// `StructField.fromJson`: each is `<provider>.<collation>`, of the `spark` or `icu` provider.
fn take_field_collations(
    metadata: &mut BTreeMap<String, serde_json::Value>,
) -> Result<CollationsMap> {
    let Some(collations) = metadata.remove(COLLATIONS_METADATA_KEY) else {
        return Ok(CollationsMap::new());
    };
    let mut by_path = CollationsMap::new();
    for (path, value) in collations.as_object().into_iter().flatten() {
        let parts: Vec<&str> = value.as_str().unwrap_or_default().split('.').collect();
        let [provider, collation] = parts[..] else {
            return Err(SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "a collation is not of a provider and a name")],
            ));
        };
        if !["spark", "icu"].contains(&provider.to_lowercase().as_str()) {
            return Err(SparkError::value(
                "COLLATION_INVALID_PROVIDER",
                &[("provider", provider), ("supportedProviders", "spark, icu")],
            ));
        }
        by_path.insert(path.clone(), collation.to_string());
    }
    Ok(by_path)
}

/// Parses the JSON object of a struct field, mirroring `StructField.fromJson`: the strings at
/// the paths of the `__COLLATIONS` of its metadata are of their collations.
fn parse_json_field(field: &serde_json::Map<String, serde_json::Value>) -> Result<StructField> {
    let name = field
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "StructField missing name")],
            )
        })?
        .to_string();
    let field_type = field.get("type").ok_or_else(|| {
        SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "StructField missing type")],
        )
    })?;
    let nullable = field
        .get("nullable")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let mut metadata: BTreeMap<String, serde_json::Value> = field
        .get("metadata")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    let collations = take_field_collations(&mut metadata)?;
    Ok(StructField {
        data_type: parse_json_value(field_type, &name, &collations)?,
        name,
        nullable,
        metadata,
    })
}

/// Raises, as PySpark's `_assert_valid_type_for_collation`, for a collation of a field path of
/// a type that is not a string.
fn check_collated_type(path: &str, json_type: &str, collations: &CollationsMap) -> Result<()> {
    if collations.contains_key(path) && json_type != "string" {
        return Err(SparkError::classed(
            SparkErrorKind::TypeError,
            "INVALID_JSON_DATA_TYPE_FOR_COLLATIONS",
            &[("jsonType", json_type)],
        ));
    }
    Ok(())
}

/// Parses a JSON value into a DataType at `field_path` of a struct field, mirroring PySpark's
/// `_parse_datatype_json_value`: a string of a path of `collations` is of its collation.
fn parse_json_value(
    value: &serde_json::Value,
    field_path: &str,
    collations: &CollationsMap,
) -> Result<DataType> {
    match value {
        serde_json::Value::String(s) => match collations.get(field_path) {
            Some(collation) if MAPPABLE_TYPE_NAMES.contains(&s.as_str()) => {
                check_collated_type(field_path, s, collations)?;
                Ok(DataType::String {
                    collation: collation.clone(),
                })
            }
            _ => parse_json_type_string(s),
        },
        serde_json::Value::Object(obj) => parse_json_object(obj, field_path, collations),
        _ => Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Expected string or object for DataType")],
        )),
    }
}

/// Parses the string of a type of JSON, as PySpark's `_parse_datatype_json_value` reads one: a
/// name of [`common_atomic_type`], or a form of [`parse_parameterized_type`] of
/// [`TypeStringGrammar::Json`].
fn parse_json_type_string(s: &str) -> Result<DataType> {
    if let Some(data_type) = common_atomic_type(s) {
        return Ok(data_type);
    }
    match s {
        // `null` is read as the server's JSON parser reads it, for the schemas written before
        // NullType was named `void`, and PySpark rejects it. `calendarinterval` is a lenient alias
        // that the drop-in's Python parser accepted before it delegated here.
        "null" => Ok(DataType::Null),
        "calendarinterval" => Ok(DataType::CalendarInterval),
        // A bare spatial name is of the default CRS, `OGC:CRS84`, as Spark's JSON reader reads
        // it; DDL names a spatial type with its SRID only.
        "geometry" => Ok(DataType::Geometry {
            srid: GeometryType::DEFAULT_SRID,
        }),
        "geography" => Ok(DataType::Geography {
            srid: GeographyType::DEFAULT_SRID,
        }),
        // The inverse of `json_value`, which writes `unparsed(<data_type_string>)`.
        s if s.starts_with("unparsed(") && s.ends_with(')') => Ok(DataType::Unparsed {
            data_type_string: s["unparsed(".len()..s.len() - 1].to_string(),
        }),
        _ => {
            // PySpark's patterns match a prefix of the string, so whitespace after a form with
            // arguments or fields is not a part of it and is accepted. A bare name is matched
            // whole, so whitespace after it, as in `interval `, is not a type.
            let trimmed = s.trim_end();
            let form = if common_atomic_type(trimmed).is_some() {
                s
            } else {
                trimmed
            };
            parse_parameterized_type(form, TypeStringGrammar::Json).unwrap_or_else(|| {
                Err(SparkError::value(
                    "INVALID_DATATYPE_STRING",
                    &[("detail", &format!("Unknown type string: {}", s))],
                ))
            })
        }
    }
}

fn parse_json_object(
    obj: &serde_json::Map<String, serde_json::Value>,
    field_path: &str,
    collations: &CollationsMap,
) -> Result<DataType> {
    let type_field = obj.get("type").and_then(|v| v.as_str());
    if let Some(json_type @ ("array" | "map" | "struct")) = type_field {
        check_collated_type(field_path, json_type, collations)?;
    }

    match type_field {
        Some("array") => {
            let element_type = obj.get("elementType").ok_or_else(|| {
                SparkError::value(
                    "INVALID_DATATYPE_FORMAT",
                    &[("detail", "Array missing elementType")],
                )
            })?;
            let contains_null = obj
                .get("containsNull")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            Ok(DataType::Array {
                element_type: Box::new(parse_json_value(
                    element_type,
                    &child_path(field_path, "element"),
                    collations,
                )?),
                contains_null,
            })
        }
        Some("map") => {
            let key_type = obj.get("keyType").ok_or_else(|| {
                SparkError::value(
                    "INVALID_DATATYPE_FORMAT",
                    &[("detail", "Map missing keyType")],
                )
            })?;
            let value_type = obj.get("valueType").ok_or_else(|| {
                SparkError::value(
                    "INVALID_DATATYPE_FORMAT",
                    &[("detail", "Map missing valueType")],
                )
            })?;
            let value_contains_null = obj
                .get("valueContainsNull")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            Ok(DataType::Map {
                key_type: Box::new(parse_json_value(
                    key_type,
                    &child_path(field_path, "key"),
                    collations,
                )?),
                value_type: Box::new(parse_json_value(
                    value_type,
                    &child_path(field_path, "value"),
                    collations,
                )?),
                value_contains_null,
            })
        }
        Some("struct") => {
            let fields_arr = obj
                .get("fields")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    SparkError::value(
                        "INVALID_DATATYPE_FORMAT",
                        &[("detail", "Struct missing fields")],
                    )
                })?;

            let mut fields = Vec::new();
            for field in fields_arr.iter().filter_map(|field| field.as_object()) {
                fields.push(parse_json_field(field)?);
            }
            Ok(DataType::Struct { fields })
        }
        // The inverse of `json_value` of a UDT: its classes and the JSON of its SQL type.
        Some("udt") => {
            let text = |key: &str| obj.get(key).and_then(|v| v.as_str()).map(str::to_string);
            let sql_type = match obj.get("sqlType") {
                Some(sql_type) => Some(Box::new(DataType::from_json(sql_type)?)),
                None => None,
            };
            Ok(DataType::Udt {
                type_str: "udt".to_string(),
                jvm_class: text("class"),
                python_class: text("pyClass"),
                serialized_python_class: text("serializedClass"),
                sql_type,
            })
        }
        _ => Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Unknown type in object")],
        )),
    }
}

/// The grammar that a type string is read by. The JSON and the DDL parsers share one parser of
/// each atomic and parameterized form, [`common_atomic_type`] and [`parse_parameterized_type`],
/// and this names the few places where the two grammars really differ, as PySpark's do:
///
/// | form | [`Json`](Self::Json) (PySpark's `_parse_datatype_json_value`) | [`Ddl`](Self::Ddl) (the server's parser) |
/// |---|---|---|
/// | letter case and spacing | as written: lower case, single spaces | any case, any whitespace between keywords |
/// | `decimal(p)` | not a type: the scale is required | scale 0 |
/// | `decimal(p,-s)` | a negative scale | not a type: an argument is an unsigned integer |
/// | `interval <a> to <b>` | any two fields of one family, as `DayTimeIntervalType(a, b)` | `b` later than `a` |
/// | `time` | not a type: the precision is required | `time(6)` |
/// | `string collate <name>` | the name as written | the name in its canonical form, `system.builtin.` dropped |
/// | `geometry(...)`, `geography(...)` | a CRS, and of a geography an algorithm too (`geography(OGC:CRS84, SPHERICAL)`), or either alone, as Spark's JSON reader; a bare `geometry` and `geography` are of `OGC:CRS84` | an SRID or `any` (`geometry(4326)`) |
///
/// Neither checks the range of an argument, such as the precision of a decimal: PySpark's types
/// take any, and the server refuses the type. A decimal that Arrow cannot hold is refused when it
/// is converted (see `decimal128_precision_and_scale` of the session).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TypeStringGrammar {
    Json,
    Ddl,
}

/// The type of a name that is one in both grammars, as it is written in JSON, in lower case.
fn common_atomic_type(s: &str) -> Option<DataType> {
    Some(match s {
        "void" => DataType::Null,
        "boolean" => DataType::Boolean,
        "byte" => DataType::Byte,
        "short" => DataType::Short,
        "integer" => DataType::Integer,
        "long" => DataType::Long,
        "float" => DataType::Float,
        "double" => DataType::Double,
        "string" => DataType::String {
            collation: "UTF8_BINARY".to_string(),
        },
        "binary" => DataType::Binary,
        "date" => DataType::Date,
        "timestamp" => DataType::Timestamp,
        "timestamp_ntz" => DataType::TimestampNtz,
        "interval" => DataType::CalendarInterval,
        "variant" => DataType::Variant,
        // A bare `decimal`, as `DecimalType().typeName()` and Spark's `DECIMAL`, is the default
        // decimal(10,0).
        "decimal" => DataType::Decimal {
            precision: 10,
            scale: 0,
        },
        _ => return None,
    })
}

/// Parses `s` if it is of a form that takes arguments or is of several words: `decimal(p,s)`,
/// `char(n)`, `varchar(n)`, `time(p)`, `timestamp_ntz(p)`, `timestamp_ltz(p)`, `interval ...`,
/// `string collate <name>`, `geometry(...)` or `geography(...)`. `None` if it is of none of them,
/// and an error if it is a malformed one.
///
/// This is the one parser of these forms, of the JSON and of the DDL parser: a DDL string comes
/// in lower case and with its whitespace collapsed (see [`canonical_type_spelling`]), so that it
/// is spelled as the JSON form is. `grammar` selects what [`TypeStringGrammar`] lists.
fn parse_parameterized_type(s: &str, grammar: TypeStringGrammar) -> Option<Result<DataType>> {
    if let Some(args) = type_arguments(s, "decimal") {
        Some(args.and_then(|args| parse_decimal(s, args, grammar)))
    } else if let Some(args) = type_arguments(s, "char") {
        Some(args.and_then(|args| {
            Ok(DataType::Char {
                length: type_argument(s, args, false)?,
            })
        }))
    } else if let Some(args) = type_arguments(s, "varchar") {
        Some(args.and_then(|args| {
            Ok(DataType::Varchar {
                length: type_argument(s, args, false)?,
            })
        }))
    } else if let Some(args) = type_arguments(s, "time") {
        Some(args.and_then(|args| {
            Ok(DataType::Time {
                precision: type_argument(s, args, false)?,
            })
        }))
    } else if let Some(args) = type_arguments(s, "timestamp_ntz") {
        Some(args.and_then(|args| {
            Ok(DataType::TimestampNtzNanos {
                precision: timestamp_nanos_precision(s, args)?,
            })
        }))
    } else if let Some(args) = type_arguments(s, "timestamp_ltz") {
        Some(args.and_then(|args| {
            Ok(DataType::TimestampLtzNanos {
                precision: timestamp_nanos_precision(s, args)?,
            })
        }))
    } else if let Some(args) = type_arguments(s, "geometry") {
        Some(args.and_then(|args| parse_geometry(s, args, grammar)))
    } else if let Some(args) = type_arguments(s, "geography") {
        Some(args.and_then(|args| parse_geography(s, args, grammar)))
    } else if s.starts_with("interval") {
        Some(parse_interval_string(s, grammar))
    } else if s
        .strip_prefix("string")
        .is_some_and(|rest| rest.starts_with(char::is_whitespace))
    {
        Some(parse_collated_string(s, grammar))
    } else {
        None
    }
}

/// The `INVALID_DATATYPE_FORMAT` error of the type string `s` that is not a valid `what`.
fn invalid_type_string(s: &str, what: &str) -> SparkError {
    SparkError::value(
        "INVALID_DATATYPE_FORMAT",
        &[("detail", &format!("Invalid {what}: {s}"))],
    )
}

/// The text between the parentheses of `s` if it is `name(...)`: `None` if `s` does not start
/// with `name(`, and an error if it does not end with the `)`.
fn type_arguments<'a>(s: &'a str, name: &str) -> Option<Result<&'a str>> {
    let rest = s.strip_prefix(name)?.strip_prefix('(')?;
    Some(
        rest.strip_suffix(')')
            .ok_or_else(|| invalid_type_string(s, &format!("{name} type"))),
    )
}

/// The integer argument `arg` of the type string `s`, which may have whitespace around it: only
/// digits, as the `\d+` of PySpark's patterns of JSON and the `INTEGER_VALUE` of Spark's grammar,
/// so not `+5`, and `-5` only where the argument is `signed`, the scale of a decimal of JSON.
fn type_argument(s: &str, arg: &str, signed: bool) -> Result<i32> {
    let arg = arg.trim();
    let digits = match arg.strip_prefix('-') {
        Some(digits) if signed => digits,
        _ => arg,
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid_type_string(s, "type argument"));
    }
    arg.parse()
        .map_err(|_| invalid_type_string(s, "type argument"))
}

/// Parses the arguments of `decimal(p,s)`. In DDL, `decimal(p)` has scale 0, as Spark's does; in
/// JSON, which names a decimal with both, the scale may be negative, as PySpark's `_FIXED_DECIMAL`.
/// The range of neither is checked, as [`TypeStringGrammar`] says.
fn parse_decimal(s: &str, args: &str, grammar: TypeStringGrammar) -> Result<DataType> {
    let (precision, scale) = match (args.split_once(','), grammar) {
        (Some((precision, scale)), _) => (precision, Some(scale)),
        (None, TypeStringGrammar::Ddl) => (args, None),
        (None, TypeStringGrammar::Json) => return Err(invalid_type_string(s, "decimal type")),
    };
    Ok(DataType::Decimal {
        precision: type_argument(s, precision, false)?,
        scale: match scale {
            Some(scale) => type_argument(s, scale, grammar == TypeStringGrammar::Json)?,
            None => 0,
        },
    })
}

/// The precision of `timestamp_ntz(p)` / `timestamp_ltz(p)` (p in 7..=9), mirroring the
/// `TIMESTAMP_{NTZ,LTZ}_NANOS_TYPE` patterns in Scala `DataType.nameToType`.
fn timestamp_nanos_precision(s: &str, args: &str) -> Result<i32> {
    type_argument(s, args, false)
        .ok()
        .filter(|p| (7..=9).contains(p))
        .ok_or_else(|| invalid_type_string(s, "timestamp precision (expected 7..9)"))
}

/// Parses `interval`, `interval <field>` or `interval <field> to <field>`, its words separated
/// by single spaces. A range is in one family (`year` and `month`, or `day` to `second`).
///
/// In DDL its end field is later than its start field, as the grammar of Spark's parser has it,
/// and anything else is an error, as the server's parser rejects it. In JSON it is of any two
/// fields of its family, as PySpark's `_INTERVAL_DAYTIME` and `_INTERVAL_YEARMONTH` read
/// `interval second to day` as `DayTimeIntervalType(3, 0)`, whose JSON it is.
fn parse_interval_string(s: &str, grammar: TypeStringGrammar) -> Result<DataType> {
    let invalid = || invalid_type_string(s, "interval type");
    let words: Vec<&str> = s.split(' ').collect();
    let (start, end, range) = match words[..] {
        ["interval"] => return Ok(DataType::CalendarInterval),
        ["interval", field] => (field, field, false),
        ["interval", start, "to", end] => (start, end, true),
        _ => return Err(invalid()),
    };
    let fields_of = |fields: &[(&str, i32)]| {
        Some((interval_field(start, fields)?, interval_field(end, fields)?))
    };
    let in_order = |start_field: i32, end_field: i32| {
        !range || grammar == TypeStringGrammar::Json || start_field < end_field
    };
    match (fields_of(YEAR_MONTH_FIELDS), fields_of(DAY_TIME_FIELDS)) {
        (Some((start_field, end_field)), _) if in_order(start_field, end_field) => {
            Ok(DataType::YearMonthInterval {
                start_field,
                end_field,
            })
        }
        (_, Some((start_field, end_field))) if in_order(start_field, end_field) => {
            Ok(DataType::DayTimeInterval {
                start_field,
                end_field,
            })
        }
        _ => Err(invalid()),
    }
}

/// The code of an interval field name in `fields`, such as `second` in the day-time fields.
fn interval_field(name: &str, fields: &[(&str, i32)]) -> Option<i32> {
    fields
        .iter()
        .find(|(field, _)| *field == name)
        .map(|&(_, code)| code)
}

/// Parses `string collate <name>`, of any whitespace between its words, whose name is of letters,
/// digits and underscores.
///
/// In JSON the name is of the letters of any script, as the `\w+` of PySpark's
/// `_STRING_WITH_COLLATION` reads them, and the collation is the name as it is written, as
/// PySpark's `StringType(name)` keeps it. In DDL the name is a multipart identifier of ASCII
/// letters, digits and underscores, as Spark's lexer reads an unquoted one, each of its parts may
/// be in backquotes, and its dots may have whitespace around them (see [`multipart_identifier`]).
/// Its keywords and unquoted parts are read in ASCII lowercase, so a letter outside ASCII, such
/// as the Kelvin sign, is not taken for an ASCII one. The collation is the name in the canonical
/// form that the server gives it (see [`canonical_collation_name`]), which may be qualified as
/// `system.builtin.<name>`, as Spark's `CollationFactory.resolveFullyQualifiedName` reads it.
fn parse_collated_string(s: &str, grammar: TypeStringGrammar) -> Result<DataType> {
    let invalid = || invalid_type_string(s, "collated string type");
    let name = s
        .strip_prefix("string")
        .and_then(|rest| rest.trim_start().strip_prefix("collate"))
        .filter(|rest| rest.starts_with(char::is_whitespace))
        .map(str::trim)
        .ok_or_else(invalid)?;
    let parts = match grammar {
        TypeStringGrammar::Json if !name.contains(char::is_whitespace) => vec![name.to_string()],
        TypeStringGrammar::Json => return Err(invalid()),
        TypeStringGrammar::Ddl => multipart_identifier(name).ok_or_else(invalid)?,
    };
    let name = match parts.as_slice() {
        [name] => name,
        [catalog, schema, name]
            if catalog.eq_ignore_ascii_case("system") && schema.eq_ignore_ascii_case("builtin") =>
        {
            name
        }
        _ => return Err(invalid()),
    };
    if !is_word(name, grammar) {
        return Err(invalid());
    }
    Ok(DataType::String {
        collation: match grammar {
            TypeStringGrammar::Json => name.clone(),
            TypeStringGrammar::Ddl => canonical_collation_name(name),
        },
    })
}

/// Whether `text` is a word: one or more letters, digits and underscores. Of JSON they are of any
/// script, as the `\w` of PySpark's patterns reads them, and of DDL of ASCII, as Spark's lexer
/// reads an unquoted identifier.
fn is_word(text: &str, grammar: TypeStringGrammar) -> bool {
    let is_word_char = |c: char| match grammar {
        TypeStringGrammar::Json => c == '_' || c.is_alphanumeric(),
        TypeStringGrammar::Ddl => c == '_' || c.is_ascii_alphanumeric(),
    };
    !text.is_empty() && text.chars().all(is_word_char)
}

/// The parts of the multipart identifier `name` of DDL, as Spark's `multipartIdentifier` reads
/// it: parts separated by dots, with any whitespace around them, each of them either as it is
/// written or in backquotes, where a doubled backquote is one backquote. `None` if `name` is
/// malformed, such as a part with whitespace in it that is not in backquotes.
fn multipart_identifier(name: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut chars = name.chars().peekable();
    loop {
        while chars.next_if(|c| c.is_whitespace()).is_some() {}
        let mut part = String::new();
        if chars.next_if_eq(&'`').is_some() {
            loop {
                match chars.next()? {
                    '`' if chars.next_if_eq(&'`').is_some() => part.push('`'),
                    '`' => break,
                    c => part.push(c),
                }
            }
        } else {
            while let Some(c) = chars.next_if(|c| *c != '.' && !c.is_whitespace()) {
                part.push(c);
            }
        }
        while chars.next_if(|c| c.is_whitespace()).is_some() {}
        parts.push(part);
        match chars.next() {
            Some('.') => {}
            None => return Some(parts),
            Some(_) => return None,
        }
    }
}

/// Parses the arguments `args` of `geometry(...)`, the type string `s`: in DDL its SRID (see
/// [`spatial_ddl_srid`]), and in JSON its CRS, as the `geometry(<crs>)` of Spark's JSON reader
/// (see [`spatial_json_crs`] and [`spatial_crs_srid`]).
fn parse_geometry(s: &str, args: &str, grammar: TypeStringGrammar) -> Result<DataType> {
    let srid = match grammar {
        TypeStringGrammar::Ddl => spatial_ddl_srid(s, args, false)?,
        TypeStringGrammar::Json => {
            let crs =
                spatial_json_crs(args).ok_or_else(|| invalid_type_string(s, "geometry type"))?;
            spatial_crs_srid(crs, false)?
        }
    };
    Ok(DataType::Geometry { srid })
}

/// Parses the arguments `args` of `geography(...)`, the type string `s`: in DDL its SRID (see
/// [`spatial_ddl_srid`]), and in JSON its CRS, its algorithm or both, as the patterns of Spark's
/// JSON reader, `geography(<crs>)`, `geography(<alg>)` and `geography(<crs>, <alg>)`, of the
/// default CRS `OGC:CRS84` and the default algorithm `SPHERICAL`. As Spark's
/// `GeographyType(crs, algorithm)`, the algorithm is checked before the CRS: `SPHERICAL`, the one
/// that Spark knows, is of any case, and another one is `ST_INVALID_ALGORITHM_VALUE`.
fn parse_geography(s: &str, args: &str, grammar: TypeStringGrammar) -> Result<DataType> {
    if grammar == TypeStringGrammar::Ddl {
        return Ok(DataType::Geography {
            srid: spatial_ddl_srid(s, args, true)?,
        });
    }
    let invalid = || invalid_type_string(s, "geography type");
    let (crs, algorithm) = match args.split_once(',') {
        Some((crs, algorithm)) => (spatial_json_crs(crs).ok_or_else(invalid)?, Some(algorithm)),
        None => match spatial_json_crs(args) {
            Some(crs) => (crs, None),
            None => (GeographyType::DEFAULT_CRS, Some(args)),
        },
    };
    if let Some(algorithm) = algorithm.map(str::trim) {
        if !is_word(algorithm, grammar) {
            return Err(invalid());
        }
        if !algorithm.eq_ignore_ascii_case(GeographyType::DEFAULT_ALG) {
            return Err(SparkError::value(
                "ST_INVALID_ALGORITHM_VALUE",
                &[("alg", algorithm)],
            ));
        }
    }
    Ok(DataType::Geography {
        srid: spatial_crs_srid(crs, true)?,
    })
}

/// The coordinate reference system that a spatial type's SRID is named by in its JSON, as
/// Spark's `GeometryType.jsonValue` and `GeographyType.jsonValue` name it: the string ID that
/// Spark writes for the SRID (`OGC:CRS84` for 4326, `EPSG:2154`, `SRID:0`), and `SRID:ANY` for a
/// mixed SRID. An SRID that Spark does not know has no string ID, so it is written `SRID:<srid>`,
/// which Spark rejects too.
fn spatial_crs(srid: i32, geographic: bool) -> String {
    match srid {
        SpatialType::MIXED_SRID => SpatialType::MIXED_CRS.to_string(),
        _ => {
            srs::string_id(srid, geographic).map_or_else(|| format!("SRID:{srid}"), str::to_string)
        }
    }
}

/// The CRS `text` of the JSON of a spatial type, without the whitespace around it, if it is of
/// the form of the patterns of Spark's JSON reader, `\w+:-?\w+`: a word, a colon, and a word that
/// may follow a minus sign, as `EPSG:2154`, `OGC:CRS84` and `SRID:ANY`. `None` if it is not, as
/// `4326`, `any` and `SPHERICAL` are not.
fn spatial_json_crs(text: &str) -> Option<&str> {
    let crs = text.trim();
    let (authority, code) = crs.split_once(':')?;
    let code = code.strip_prefix('-').unwrap_or(code);
    (is_word(authority, TypeStringGrammar::Json) && is_word(code, TypeStringGrammar::Json))
        .then_some(crs)
}

/// The SRID of the CRS `crs` of the JSON of a spatial type of the given kind, as Spark's
/// `GeometryType.toSrid` and `GeographyType.toSrid` read it: `SRID:ANY`, of any case, is the mixed
/// SRID, and another CRS is the SRID of the string ID of Spark's registry that it is (the aliases
/// included, and the match is exact) if the type takes it, else `ST_INVALID_CRS_VALUE`.
fn spatial_crs_srid(crs: &str, geographic: bool) -> Result<i32> {
    if crs.eq_ignore_ascii_case(SpatialType::MIXED_CRS) {
        return Ok(SpatialType::MIXED_SRID);
    }
    srs::srid(crs, geographic)
        .ok_or_else(|| SparkError::value("ST_INVALID_CRS_VALUE", &[("crs", crs)]))
}

/// The SRID of the argument `args` of the DDL `s` of a spatial type of the given kind, as Spark's
/// parser reads `GEOMETRY(<srid>)` and `GEOGRAPHY(<srid>)`, whose argument is the keyword `ANY` or
/// an unsigned integer: `any`, of any case, is the mixed SRID, and an integer is an SRID that the
/// type must take, else `ST_INVALID_SRID_VALUE`, as an integer out of the 32-bit range is too.
/// Any other text, such as `-1`, `+5` or a CRS, is not of Spark's grammar, whose
/// `PARSE_SYNTAX_ERROR` it is: the `INVALID_DATATYPE_FORMAT` of a malformed type of this parser.
fn spatial_ddl_srid(s: &str, args: &str, geographic: bool) -> Result<i32> {
    let text = args.trim();
    if text.eq_ignore_ascii_case("any") {
        return Ok(SpatialType::MIXED_SRID);
    }
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid_type_string(s, "type argument"));
    }
    let invalid_srid = |srid: &str| SparkError::value("ST_INVALID_SRID_VALUE", &[("srid", srid)]);
    match text.parse::<i32>() {
        Ok(srid) if srs::is_supported(srid, geographic) => Ok(srid),
        // The SRID that the type is made of, as Spark's `GeometryType(srid)` names it.
        Ok(srid) => Err(invalid_srid(&srid.to_string())),
        Err(_) => Err(invalid_srid(text)),
    }
}

/// Main DDL parser: parses DDL strings into DataType.
/// Handles primitive types, complex types (array, map, struct), and top-level schemas.
fn parse_datatype_string(input: &str) -> Result<DataType> {
    let trimmed = input.trim();
    // An empty schema is an empty struct, as the server's parser gives it.
    if trimmed.is_empty() {
        return Ok(DataType::Struct { fields: vec![] });
    }

    // A top-level schema is "a INT, b STRING" (multiple fields) OR a single "a INT"
    // (one field, no comma) -- both parse to a StructType, matching PySpark's
    // _parse_datatype_string. Detect either a top-level comma or a top-level space
    // separating a field name from its type; fall back to single-type parsing if the
    // schema interpretation fails (e.g. a bare type that itself contains spaces such as
    // "interval day to second"). A top-level colon names a field too, as in "a:int".
    if after_complex_keyword(trimmed, "struct").is_none()
        && (contains_top_level_comma(trimmed)
            || contains_top_level_whitespace(trimmed)
            || contains_top_level_colon(trimmed))
    {
        if let Ok(dt) = parse_top_level_schema(trimmed) {
            return Ok(dt);
        }
    }

    parse_single_type(trimmed)
}

/// Whether `s` has a top-level whitespace char (outside `<>`/`()` and quotes), i.e. it looks like a
/// "name type" field rather than a bare type such as `int` or `array<int>`.
fn contains_top_level_whitespace(s: &str) -> bool {
    top_level_positions(s)
        .into_iter()
        .any(|i| s[i..].chars().next().is_some_and(char::is_whitespace))
}

/// Whether `s` has a top-level colon, which names a field as in "a:int" (outside `<>`/`()` and
/// quotes, so that "x:array<int>" is a field and "array<a:int>" is not).
fn contains_top_level_colon(s: &str) -> bool {
    top_level_positions(s)
        .into_iter()
        .any(|i| s.as_bytes()[i] == b':')
}

/// Check if a string contains a comma at the top level (not inside brackets/parens/quotes)
fn contains_top_level_comma(s: &str) -> bool {
    top_level_positions(s)
        .into_iter()
        .any(|i| s.as_bytes()[i] == b',')
}

/// The byte positions of the chars of `s` outside `<>`/`()` and outside quoted strings and
/// backquoted names, so that a comment such as `'x, y'` does not split a field.
fn top_level_positions(s: &str) -> Vec<usize> {
    let bytes = s.as_bytes();
    let mut positions = Vec::new();
    let mut depth = 0;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' | b'"' | b'`' => match quoted_end(s, i) {
                Some(end) => {
                    i = end;
                    continue;
                }
                // An unterminated quote: the rest is not a field, so the parse fails later.
                None => break,
            },
            b'<' | b'(' => depth += 1,
            b'>' | b')' => depth -= 1,
            _ if depth == 0 && s.is_char_boundary(i) => positions.push(i),
            _ => {}
        }
        i += 1;
    }
    positions
}

/// The byte index just after the quoted string or backquoted name that opens at `start`, or
/// `None` if it is not closed. A doubled quote is one quote. A backslash escapes the next char in
/// a string, and a raw string (`R'...'`) has no escapes, as Spark's lexer reads them.
fn quoted_end(s: &str, start: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let quote = *bytes.get(start)?;
    let raw = quote != b'`'
        && start > 0
        && matches!(bytes[start - 1], b'R' | b'r')
        && (start < 2 || !is_ident_byte(bytes[start - 2]));
    let mut i = start + 1;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' && quote != b'`' && !raw {
            i += 2;
        } else if c == quote {
            if !raw && bytes.get(i + 1) == Some(&quote) {
                i += 2;
            } else {
                return Some(i + 1);
            }
        } else {
            i += 1;
        }
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Parse a top-level schema like "a INT, b STRING" or "a: int, b: string"
fn parse_top_level_schema(input: &str) -> Result<DataType> {
    let mut fields = Vec::new();
    for part in split_top_level_comma(input) {
        fields.push(parse_struct_field(part.trim())?);
    }

    Ok(DataType::Struct { fields })
}

/// Parses one field, `name [:] type [NOT NULL] [COMMENT 'string']`, as Spark's `colType` and
/// `complexColType` grammar: the name is a backquoted name or an identifier, the keywords are in
/// any case, `!NULL` is `NOT NULL`, and the comment is one or more adjacent string literals.
fn parse_struct_field(part: &str) -> Result<StructField> {
    let (name, rest) = parse_field_name(part)?;
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':').unwrap_or(rest).trim_start();
    let (type_str, not_null, comment) = split_type_and_clause(rest, part)?;
    let data_type = parse_single_type(type_str)?;
    let metadata = comment
        .map(|c| BTreeMap::from([("comment".to_string(), serde_json::Value::String(c))]))
        .unwrap_or_default();
    Ok(StructField {
        name,
        data_type,
        nullable: !not_null,
        metadata,
    })
}

/// The name at the start of a field, and the text after it.
fn parse_field_name(part: &str) -> Result<(String, &str)> {
    let invalid = || {
        SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", &format!("Cannot parse field: {}", part))],
        )
    };
    if part.starts_with('`') {
        let end = quoted_end(part, 0).ok_or_else(invalid)?;
        let name = part[1..end - 1].replace("``", "`");
        return Ok((name, &part[end..]));
    }
    // An unquoted identifier is ASCII letters, digits and underscores, as Spark's parser accepts
    // them: it rejects others, such as `이름`, with INVALID_IDENTIFIER (back quote them).
    let end = part
        .find(|c: char| c.is_whitespace() || c == ':')
        .unwrap_or(part.len());
    if end == 0 || !part.as_bytes()[..end].iter().all(|&b| is_ident_byte(b)) {
        return Err(invalid());
    }
    Ok((part[..end].to_string(), &part[end..]))
}

/// Splits the text after a field name into its type and its `NOT NULL` and `COMMENT` clauses, in
/// that order. Returns whether the field is `NOT NULL` and its comment, unescaped.
fn split_type_and_clause<'a>(rest: &'a str, part: &str) -> Result<(&'a str, bool, Option<String>)> {
    let invalid = || {
        SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", &format!("Cannot parse field: {}", part))],
        )
    };
    let clause_start = top_level_positions(rest)
        .into_iter()
        .find(|&i| is_clause_start(rest, i));
    let (type_str, clause) = match clause_start {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let mut clause = clause.trim_start();
    let mut not_null = false;
    let mut comment = None;
    if let Some(after) = strip_keyword(clause, "NOT").or_else(|| clause.strip_prefix('!')) {
        let after = strip_keyword(after.trim_start(), "NULL").ok_or_else(invalid)?;
        not_null = true;
        clause = after.trim_start();
    }
    if let Some(after) = strip_keyword(clause, "COMMENT") {
        let mut text = String::new();
        let mut s = after.trim_start();
        loop {
            let (literal, next) = string_literal(s).ok_or_else(invalid)?;
            text.push_str(&literal);
            s = next.trim_start();
            if !starts_string_literal(s) {
                break;
            }
        }
        comment = Some(text);
        clause = s;
    }
    if !clause.is_empty() || type_str.trim().is_empty() {
        return Err(invalid());
    }
    Ok((type_str.trim_end(), not_null, comment))
}

/// Whether the text at `i` starts the `NOT NULL`, `!NULL` or `COMMENT` clause of a field type.
fn is_clause_start(rest: &str, i: usize) -> bool {
    let tail = &rest[i..];
    if tail.starts_with('!') {
        return true;
    }
    let after_ident = rest[..i]
        .chars()
        .last()
        .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
    !after_ident
        && (strip_keyword(tail, "NOT").is_some() || strip_keyword(tail, "COMMENT").is_some())
}

/// The text after a keyword at the start of `s`, if `s` starts with it (in any case) at a word
/// boundary.
fn strip_keyword<'a>(s: &'a str, keyword: &str) -> Option<&'a str> {
    let head = s.get(..keyword.len())?;
    let after = &s[keyword.len()..];
    let boundary = after
        .chars()
        .next()
        .is_none_or(|c| !(c.is_ascii_alphanumeric() || c == '_'));
    (head.eq_ignore_ascii_case(keyword) && boundary).then_some(after)
}

fn starts_string_literal(s: &str) -> bool {
    let raw = s.len() > 1
        && matches!(s.as_bytes()[0], b'R' | b'r')
        && matches!(s.as_bytes()[1], b'\'' | b'"');
    raw || matches!(s.as_bytes().first(), Some(b'\'' | b'"'))
}

/// The value of the string literal at the start of `s`, and the text after it: a single-quoted
/// or double-quoted string with Spark's escapes, or a raw string `R'...'`.
fn string_literal(s: &str) -> Option<(String, &str)> {
    if !starts_string_literal(s) {
        return None;
    }
    let start = if s.as_bytes()[0].is_ascii_alphabetic() {
        1
    } else {
        0
    };
    let end = quoted_end(s, start)?;
    let token = &s[start..end];
    let text = if start == 1 {
        token[1..token.len() - 1].to_string()
    } else {
        unescape_sql_string(token)
    };
    Some((text, &s[end..]))
}

/// The value of a quoted string token, as Spark's `unescapeSQLString` reads it: the quotes are
/// removed, a doubled quote is one quote, and the escapes `\0 \b \n \r \t \Z`, `\uXXXX`,
/// `\UXXXXXXXX` and three-digit octal `\NNN` are decoded. Any other escaped char is itself, except
/// `\%` and `\_`, which keep their backslash.
fn unescape_sql_string(token: &str) -> String {
    let quote = token.as_bytes()[0] as char;
    let chars: Vec<char> = token[1..token.len() - 1].chars().collect();
    let n = chars.len();
    let hex = |range: std::ops::Range<usize>| {
        !range.is_empty() && chars[range].iter().all(|c| c.is_ascii_hexdigit())
    };
    let code_point = |range: std::ops::Range<usize>, radix: u32| {
        let digits: String = chars[range].iter().collect();
        u32::from_str_radix(&digits, radix)
            .ok()
            .and_then(char::from_u32)
            .unwrap_or('\u{FFFD}')
    };
    let mut out = String::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c == quote && chars.get(i + 1) == Some(&quote) {
            out.push(quote);
            i += 2;
        } else if c != '\\' || i + 1 == n {
            out.push(c);
            i += 1;
        } else {
            i += 1;
            let escaped = chars[i];
            if escaped == 'u' && i + 5 <= n && hex(i + 1..i + 5) {
                out.push(code_point(i + 1..i + 5, 16));
                i += 5;
            } else if escaped == 'U' && i + 9 <= n && hex(i + 1..i + 9) {
                out.push(code_point(i + 1..i + 9, 16));
                i += 9;
            } else if i + 3 <= n
                && matches!(chars[i], '0' | '1')
                && chars[i + 1..i + 3].iter().all(|c| ('0'..='7').contains(c))
            {
                out.push(code_point(i..i + 3, 8));
                i += 3;
            } else {
                match escaped {
                    '0' => out.push('\0'),
                    'b' => out.push('\u{8}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'Z' => out.push('\u{1A}'),
                    // As Spark does, `\%` and `\_` keep their backslash.
                    '%' | '_' => out.push_str(&format!("\\{escaped}")),
                    other => out.push(other),
                }
                i += 1;
            }
        }
    }
    out
}

/// Split by top-level commas (respecting angle brackets, parentheses and quotes)
fn split_top_level_comma(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    for i in top_level_positions(s) {
        if s.as_bytes()[i] == b',' {
            result.push(&s[start..i]);
            start = i + 1;
        }
    }
    result.push(&s[start..]);
    result
}

/// The canonical form of a collation name, as Spark's server gives it. A `UTF8_*` or `UNICODE*`
/// name is in upper case. An ICU locale name, `language[_Script][_COUNTRY][_SPECIFIERS]`, has a
/// lower-case language, a title-case script (a four-letter part) and upper-case other parts, such
/// as `sr_Cyrl_SRB_CI_AI`. Collation names are case-insensitive, so any case is accepted.
fn canonical_collation_name(name: &str) -> String {
    let provider = name.split('_').next().unwrap_or_default();
    if provider.eq_ignore_ascii_case("UTF8") || provider.eq_ignore_ascii_case("UNICODE") {
        return name.to_uppercase();
    }
    name.split('_')
        .enumerate()
        .map(|(i, part)| {
            if i == 0 {
                part.to_lowercase()
            } else if part.len() == 4 && part.chars().all(|c| c.is_ascii_alphabetic()) {
                let mut chars = part.chars();
                let first = chars.next().map(|c| c.to_ascii_uppercase());
                first
                    .into_iter()
                    .chain(chars.map(|c| c.to_ascii_lowercase()))
                    .collect()
            } else {
                part.to_uppercase()
            }
        })
        .collect::<Vec<_>>()
        .join("_")
}

/// The spelling of a type under the name that [`parse_single_type`] takes, for the aliases of
/// Spark's grammar: `dec` and `numeric` are `decimal`, `real` is `float` and `character` is `char`.
/// A time zone clause names the timestamp type (`timestamp without time zone` is `timestamp_ntz`,
/// `timestamp with local time zone` is `timestamp_ltz`), `timestamp(6)` is the microsecond
/// timestamp and `timestamp(p)` the nanosecond one. Keywords may be separated by any whitespace.
/// Any other spelling is returned as it is.
fn canonical_type_spelling(s: &str) -> String {
    let words = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let (base, zone) = if let Some(base) = words.strip_suffix(" without time zone") {
        (base, Some("ntz"))
    } else if let Some(base) = words.strip_suffix(" with local time zone") {
        (base, Some("ltz"))
    } else {
        (words.as_str(), None)
    };
    let (head, args) = match base.find('(') {
        Some(i) => base.split_at(i),
        None => (base, ""),
    };
    let head = head.trim_end();
    let args = if head.starts_with("timestamp") && args == "(6)" {
        ""
    } else {
        args
    };
    let head = match (head, zone) {
        ("timestamp", Some("ntz")) => "timestamp_ntz",
        ("timestamp", Some("ltz")) => "timestamp_ltz",
        ("time", Some("ntz")) => "time",
        ("timestamp", None) if !args.is_empty() => "timestamp_ltz",
        ("dec" | "numeric", None) => "decimal",
        ("character", None) => "char",
        ("real", None) => "float",
        (_, Some(_)) => return s.to_string(),
        (head, None) => head,
    };
    format!("{head}{args}")
}

/// The text after the `<` of the complex type `keyword` that `input` starts with, in any case and
/// with any whitespace before the `<`, as Spark's lexer reads `ARRAY <INT>`; `None` if `input`
/// does not start with it.
fn after_complex_keyword<'a>(input: &'a str, keyword: &str) -> Option<&'a str> {
    let input = input.trim_start();
    let head = input.get(..keyword.len())?;
    if !head.eq_ignore_ascii_case(keyword) {
        return None;
    }
    input[keyword.len()..].trim_start().strip_prefix('<')
}

/// The type parameters of the complex type `keyword` of `input`, the text between its angle
/// brackets, or the `INVALID_DATATYPE_FORMAT` error of `detail`.
fn complex_type_inner<'a>(input: &'a str, keyword: &str, detail: &str) -> Result<&'a str> {
    after_complex_keyword(input, keyword)
        .and_then(|rest| rest.trim_end().strip_suffix('>'))
        .ok_or_else(|| SparkError::value("INVALID_DATATYPE_FORMAT", &[("detail", detail)]))
}

/// Parse a single type (primitive or complex)
fn parse_single_type(input: &str) -> Result<DataType> {
    // ASCII only, as Spark's lexer, whose keywords and unquoted identifiers are of ASCII letters:
    // a lowercase of the letters of any script would read the Kelvin sign as a `k`.
    let trimmed = canonical_type_spelling(&input.trim().to_ascii_lowercase());

    // Complex types
    if after_complex_keyword(input, "array").is_some() {
        return parse_array_type(input);
    }
    if after_complex_keyword(input, "map").is_some() {
        return parse_map_type(input);
    }
    if after_complex_keyword(input, "struct").is_some() {
        return parse_struct_type(input);
    }

    // The names of both grammars, then those of DDL alone, then the forms with arguments.
    if let Some(data_type) = common_atomic_type(&trimmed) {
        return Ok(data_type);
    }
    match trimmed.as_str() {
        // A bare `time` is the default time(6) of Spark's grammar; JSON names a time with its
        // precision only, as `time(p)`.
        "time" => Ok(DataType::Time { precision: 6 }),
        "tinyint" => Ok(DataType::Byte),
        "smallint" => Ok(DataType::Short),
        "int" => Ok(DataType::Integer),
        "bigint" => Ok(DataType::Long),
        "timestamp_ltz" => Ok(DataType::Timestamp),
        _ => parse_parameterized_type(&trimmed, TypeStringGrammar::Ddl).unwrap_or_else(|| {
            Err(SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", &format!("Unknown type: {}", input))],
            ))
        }),
    }
}

/// Parse array<elementType>
fn parse_array_type(input: &str) -> Result<DataType> {
    let inner = complex_type_inner(input, "array", "Invalid array type format")?;
    let element_type = parse_single_type(inner)?;

    Ok(DataType::Array {
        element_type: Box::new(element_type),
        contains_null: true,
    })
}

/// Parse map<keyType,valueType>
fn parse_map_type(input: &str) -> Result<DataType> {
    let inner = complex_type_inner(input, "map", "Invalid map type format")?;
    let parts = split_map_parts(inner)?;
    if parts.len() != 2 {
        return Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[(
                "detail",
                "Map must have exactly 2 type parameters (key and value)",
            )],
        ));
    }

    let key_type = parse_single_type(parts[0])?;
    let value_type = parse_single_type(parts[1])?;

    Ok(DataType::Map {
        key_type: Box::new(key_type),
        value_type: Box::new(value_type),
        value_contains_null: true,
    })
}

/// Split map key and value types by comma at the top level
fn split_map_parts(s: &str) -> Result<Vec<&str>> {
    Ok(split_top_level_comma(s))
}

/// Parse struct<field1:type1,field2:type2,...>
fn parse_struct_type(input: &str) -> Result<DataType> {
    let inner = complex_type_inner(input, "struct", "Invalid struct type format")?;
    if inner.trim().is_empty() {
        return Ok(DataType::Struct { fields: vec![] });
    }

    let mut fields = Vec::new();
    for part in split_top_level_comma(inner) {
        fields.push(parse_struct_field(part.trim())?);
    }

    Ok(DataType::Struct { fields })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_type() {
        let dt = DataType::Null;
        assert_eq!(dt.type_name(), "void");
        assert_eq!(dt.simple_string(), "void");
        assert_eq!(dt.json(), "\"void\"");
    }

    #[test]
    fn test_simple_types() {
        let tests = vec![
            (DataType::Boolean, "boolean", "boolean", "\"boolean\""),
            (DataType::Byte, "byte", "tinyint", "\"byte\""),
            (DataType::Short, "short", "smallint", "\"short\""),
            (DataType::Integer, "integer", "int", "\"integer\""),
            (DataType::Long, "long", "bigint", "\"long\""),
            (DataType::Float, "float", "float", "\"float\""),
            (DataType::Double, "double", "double", "\"double\""),
            (DataType::Binary, "binary", "binary", "\"binary\""),
            (DataType::Date, "date", "date", "\"date\""),
            (
                DataType::Timestamp,
                "timestamp",
                "timestamp",
                "\"timestamp\"",
            ),
            (
                DataType::TimestampNtz,
                "timestamp_ntz",
                "timestamp_ntz",
                "\"timestamp_ntz\"",
            ),
            (
                DataType::CalendarInterval,
                "interval",
                "interval",
                "\"interval\"",
            ),
            (DataType::Variant, "variant", "variant", "\"variant\""),
        ];

        for (dt, expected_typename, expected_simple, expected_json) in tests {
            assert_eq!(dt.type_name(), expected_typename, "type_name for {:?}", dt);
            assert_eq!(
                dt.simple_string(),
                expected_simple,
                "simple_string for {:?}",
                dt
            );
            assert_eq!(dt.json(), expected_json, "json for {:?}", dt);
        }
    }

    #[test]
    fn test_decimal_type() {
        let dt = DataType::Decimal {
            precision: 10,
            scale: 0,
        };
        assert_eq!(dt.simple_string(), "decimal(10,0)");
        assert_eq!(dt.json(), "\"decimal(10,0)\"");

        let dt2 = DataType::Decimal {
            precision: 38,
            scale: 18,
        };
        assert_eq!(dt2.simple_string(), "decimal(38,18)");
    }

    #[test]
    fn test_char_varchar_types() {
        let char_dt = DataType::Char { length: 50 };
        assert_eq!(char_dt.simple_string(), "char(50)");
        assert_eq!(char_dt.json(), "\"char(50)\"");

        let varchar_dt = DataType::Varchar { length: 100 };
        assert_eq!(varchar_dt.simple_string(), "varchar(100)");
        assert_eq!(varchar_dt.json(), "\"varchar(100)\"");
    }

    #[test]
    fn test_array_type() {
        let dt = DataType::Array {
            element_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            contains_null: true,
        };
        assert_eq!(dt.simple_string(), "array<string>");

        let json_val = dt.json_value();
        assert_eq!(json_val["type"], "array");
        assert_eq!(json_val["containsNull"], true);

        let dt2 = DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: false,
        };
        assert_eq!(dt2.simple_string(), "array<int>");
    }

    #[test]
    fn test_map_type() {
        let dt = DataType::Map {
            key_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            value_type: Box::new(DataType::Integer),
            value_contains_null: true,
        };
        assert_eq!(dt.simple_string(), "map<string,int>");

        let json_val = dt.json_value();
        assert_eq!(json_val["type"], "map");
        assert_eq!(json_val["valueContainsNull"], true);
    }

    #[test]
    fn test_struct_type() {
        let dt = DataType::Struct {
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
                    name: "age".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        assert_eq!(dt.simple_string(), "struct<name:string,age:int>");

        let json_val = dt.json_value();
        assert_eq!(json_val["type"], "struct");
        assert_eq!(json_val["fields"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_interval_types() {
        let dti = DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        };
        assert_eq!(dti.simple_string(), "interval day to second");

        let ymi = DataType::YearMonthInterval {
            start_field: 0,
            end_field: 1,
        };
        assert_eq!(ymi.simple_string(), "interval year to month");
    }

    #[test]
    fn test_string_collation() {
        let dt = DataType::String {
            collation: "UTF8_BINARY".to_string(),
        };
        assert_eq!(dt.simple_string(), "string");

        let dt2 = DataType::String {
            collation: "UNICODE".to_string(),
        };
        assert_eq!(dt2.simple_string(), "string collate UNICODE");
    }

    #[test]
    fn test_time_type() {
        let dt = DataType::Time { precision: 6 };
        assert_eq!(dt.simple_string(), "time(6)");
        assert_eq!(dt.json(), "\"time(6)\"");
    }

    #[test]
    fn test_proto_roundtrip_simple() {
        let types = vec![
            DataType::Null,
            DataType::Boolean,
            DataType::Byte,
            DataType::Integer,
            DataType::Long,
            DataType::Float,
            DataType::Double,
            DataType::Binary,
            DataType::Date,
            DataType::Timestamp,
            DataType::TimestampNtz,
            DataType::CalendarInterval,
            DataType::Variant,
        ];

        for dt in types {
            let proto = dt.to_proto();
            let roundtrip = DataType::from_proto(&proto).unwrap();
            assert_eq!(dt, roundtrip, "proto roundtrip failed for {:?}", dt);
        }
    }

    #[test]
    fn test_proto_roundtrip_decimal() {
        let dt = DataType::Decimal {
            precision: 38,
            scale: 18,
        };
        let proto = dt.to_proto();
        let roundtrip = DataType::from_proto(&proto).unwrap();
        assert_eq!(dt, roundtrip);
    }

    #[test]
    fn test_proto_roundtrip_array() {
        let dt = DataType::Array {
            element_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            contains_null: false,
        };
        let proto = dt.to_proto();
        let roundtrip = DataType::from_proto(&proto).unwrap();
        assert_eq!(dt, roundtrip);
    }

    #[test]
    fn test_proto_roundtrip_struct() {
        let dt = DataType::Struct {
            fields: vec![
                StructField {
                    name: "a".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "b".to_string(),
                    data_type: DataType::String {
                        collation: "UTF8_BINARY".to_string(),
                    },
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        let proto = dt.to_proto();
        let roundtrip = DataType::from_proto(&proto).unwrap();
        assert_eq!(dt, roundtrip);
    }

    #[test]
    fn test_json_parse_decimal() {
        let dt = DataType::from_json(&serde_json::json!("decimal(10,0)")).unwrap();
        assert_eq!(dt.simple_string(), "decimal(10,0)");
    }

    #[test]
    fn test_json_parse_array() {
        let json = serde_json::json!({
            "type": "array",
            "elementType": "string",
            "containsNull": true,
        });
        let dt = DataType::from_json(&json).unwrap();
        assert_eq!(dt.simple_string(), "array<string>");
    }

    #[test]
    fn test_json_parse_struct() {
        let json = serde_json::json!({
            "type": "struct",
            "fields": [
                {
                    "name": "name",
                    "type": "string",
                    "nullable": true,
                    "metadata": {}
                },
                {
                    "name": "age",
                    "type": "integer",
                    "nullable": true,
                    "metadata": {}
                }
            ]
        });
        let dt = DataType::from_json(&json).unwrap();
        assert_eq!(dt.simple_string(), "struct<name:string,age:int>");
    }

    #[test]
    fn test_hash_consistency() {
        let dt1 = DataType::Integer;
        let dt2 = DataType::Integer;
        let mut hasher1 = std::collections::hash_map::DefaultHasher::new();
        let mut hasher2 = std::collections::hash_map::DefaultHasher::new();
        dt1.hash(&mut hasher1);
        dt2.hash(&mut hasher2);
        assert_eq!(
            std::hash::Hasher::finish(&hasher1),
            std::hash::Hasher::finish(&hasher2)
        );
    }

    #[test]
    fn test_equality() {
        let dt1 = DataType::Decimal {
            precision: 10,
            scale: 0,
        };
        let dt2 = DataType::Decimal {
            precision: 10,
            scale: 0,
        };
        assert_eq!(dt1, dt2);

        let dt3 = DataType::Decimal {
            precision: 10,
            scale: 1,
        };
        assert_ne!(dt1, dt3);
    }

    #[test]
    fn test_struct_field_simple_string() {
        let field = StructField {
            name: "field_name".to_string(),
            data_type: DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
            nullable: true,
            metadata: BTreeMap::new(),
        };
        assert_eq!(field.simple_string(), "field_name:string");
    }

    #[test]
    fn test_geometry_type() {
        let dt = DataType::Geometry { srid: 4326 };
        assert_eq!(dt.simple_string(), "geometry(4326)");

        let dt_any = DataType::Geometry { srid: -1 };
        assert_eq!(dt_any.simple_string(), "geometry(any)");
    }

    #[test]
    fn test_geography_type() {
        let dt = DataType::Geography { srid: 4326 };
        assert_eq!(dt.simple_string(), "geography(4326)");

        let dt_any = DataType::Geography { srid: -1 };
        assert_eq!(dt_any.simple_string(), "geography(any)");
    }

    #[test]
    fn test_golden_values_simple_strings() {
        // Golden values from Python pyspark.sql.types
        struct TestCase {
            dt: DataType,
            expected_simple: &'static str,
            expected_type_name: &'static str,
        }

        let tests = vec![
            TestCase {
                dt: DataType::Null,
                expected_simple: "void",
                expected_type_name: "void",
            },
            TestCase {
                dt: DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
                expected_simple: "string",
                expected_type_name: "string",
            },
            TestCase {
                dt: DataType::Boolean,
                expected_simple: "boolean",
                expected_type_name: "boolean",
            },
            TestCase {
                dt: DataType::Byte,
                expected_simple: "tinyint",
                expected_type_name: "byte",
            },
            TestCase {
                dt: DataType::Short,
                expected_simple: "smallint",
                expected_type_name: "short",
            },
            TestCase {
                dt: DataType::Integer,
                expected_simple: "int",
                expected_type_name: "integer",
            },
            TestCase {
                dt: DataType::Long,
                expected_simple: "bigint",
                expected_type_name: "long",
            },
            TestCase {
                dt: DataType::Float,
                expected_simple: "float",
                expected_type_name: "float",
            },
            TestCase {
                dt: DataType::Double,
                expected_simple: "double",
                expected_type_name: "double",
            },
            TestCase {
                dt: DataType::Date,
                expected_simple: "date",
                expected_type_name: "date",
            },
            TestCase {
                dt: DataType::Timestamp,
                expected_simple: "timestamp",
                expected_type_name: "timestamp",
            },
            TestCase {
                dt: DataType::TimestampNtz,
                expected_simple: "timestamp_ntz",
                expected_type_name: "timestamp_ntz",
            },
            TestCase {
                dt: DataType::Binary,
                expected_simple: "binary",
                expected_type_name: "binary",
            },
            TestCase {
                dt: DataType::Decimal {
                    precision: 10,
                    scale: 0,
                },
                expected_simple: "decimal(10,0)",
                expected_type_name: "decimal",
            },
            TestCase {
                dt: DataType::Char { length: 50 },
                expected_simple: "char(50)",
                expected_type_name: "char",
            },
            TestCase {
                dt: DataType::Varchar { length: 100 },
                expected_simple: "varchar(100)",
                expected_type_name: "varchar",
            },
            TestCase {
                dt: DataType::CalendarInterval,
                expected_simple: "interval",
                expected_type_name: "interval",
            },
            TestCase {
                dt: DataType::DayTimeInterval {
                    start_field: 0,
                    end_field: 3,
                },
                expected_simple: "interval day to second",
                expected_type_name: "daytimeinterval",
            },
            TestCase {
                dt: DataType::YearMonthInterval {
                    start_field: 0,
                    end_field: 1,
                },
                expected_simple: "interval year to month",
                expected_type_name: "yearmonthinterval",
            },
            TestCase {
                dt: DataType::Variant,
                expected_simple: "variant",
                expected_type_name: "variant",
            },
        ];

        for test in tests {
            assert_eq!(
                test.dt.simple_string(),
                test.expected_simple,
                "simpleString mismatch for {:?}",
                test.dt
            );
            assert_eq!(
                test.dt.type_name(),
                test.expected_type_name,
                "typeName mismatch for {:?}",
                test.dt
            );
        }
    }

    #[test]
    fn test_golden_values_complex_types() {
        // ArrayType with StringType
        let array_string = DataType::Array {
            element_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            contains_null: true,
        };
        assert_eq!(array_string.simple_string(), "array<string>");

        // ArrayType with IntegerType, contains_null=false
        let array_int = DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: false,
        };
        assert_eq!(array_int.simple_string(), "array<int>");

        // MapType<StringType, IntegerType>
        let map_type = DataType::Map {
            key_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            value_type: Box::new(DataType::Integer),
            value_contains_null: true,
        };
        assert_eq!(map_type.simple_string(), "map<string,int>");

        // StructType([StructField('name', StringType), StructField('age', IntegerType)])
        let struct_type = DataType::Struct {
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
                    name: "age".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        assert_eq!(struct_type.simple_string(), "struct<name:string,age:int>");
    }

    #[test]
    fn test_golden_json_values() {
        // Test JSON for simple types
        let null_json = DataType::Null.json();
        assert_eq!(null_json, "\"void\"");

        let string_json = DataType::String {
            collation: "UTF8_BINARY".to_string(),
        }
        .json();
        assert_eq!(string_json, "\"string\"");

        let decimal_json = DataType::Decimal {
            precision: 10,
            scale: 0,
        }
        .json();
        assert_eq!(decimal_json, "\"decimal(10,0)\"");

        // Test JSON for array type
        let array_json = DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: false,
        };
        let json_val = array_json.json_value();
        assert_eq!(json_val["type"], "array");
        assert_eq!(json_val["containsNull"], false);

        // Test JSON for map type
        let map_json = DataType::Map {
            key_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            value_type: Box::new(DataType::Integer),
            value_contains_null: true,
        };
        let json_val = map_json.json_value();
        assert_eq!(json_val["type"], "map");
        assert_eq!(json_val["valueContainsNull"], true);
    }

    #[test]
    fn test_from_ddl_primitive_types() {
        // Test primitive types parsing
        assert_eq!(DataType::from_ddl("int").unwrap(), DataType::Integer);
        assert_eq!(DataType::from_ddl("INT").unwrap(), DataType::Integer);
        assert_eq!(DataType::from_ddl("bigint").unwrap(), DataType::Long);
        assert_eq!(
            DataType::from_ddl("string").unwrap(),
            DataType::String {
                collation: "UTF8_BINARY".to_string()
            }
        );
        assert_eq!(DataType::from_ddl("double").unwrap(), DataType::Double);
        assert_eq!(DataType::from_ddl("boolean").unwrap(), DataType::Boolean);
        assert_eq!(DataType::from_ddl("date").unwrap(), DataType::Date);
        assert_eq!(
            DataType::from_ddl("timestamp").unwrap(),
            DataType::Timestamp
        );
        assert_eq!(DataType::from_ddl("binary").unwrap(), DataType::Binary);
        assert_eq!(DataType::from_ddl("tinyint").unwrap(), DataType::Byte);
        assert_eq!(DataType::from_ddl("smallint").unwrap(), DataType::Short);
        assert_eq!(DataType::from_ddl("float").unwrap(), DataType::Float);
    }

    #[test]
    fn test_from_ddl_decimal_and_fixed_length() {
        // Test decimal
        assert_eq!(
            DataType::from_ddl("decimal(10,2)").unwrap(),
            DataType::Decimal {
                precision: 10,
                scale: 2
            }
        );

        // Test char
        assert_eq!(
            DataType::from_ddl("char(50)").unwrap(),
            DataType::Char { length: 50 }
        );

        // Test varchar
        assert_eq!(
            DataType::from_ddl("varchar(100)").unwrap(),
            DataType::Varchar { length: 100 }
        );

        // Test time; a bare time is of the default precision, as Spark's TIME.
        assert_eq!(
            DataType::from_ddl("time(6)").unwrap(),
            DataType::Time { precision: 6 }
        );
        assert_eq!(
            DataType::from_ddl("time").unwrap(),
            DataType::Time { precision: 6 }
        );
        // JSON names a time with its precision only, as `TimeType.jsonValue` does.
        assert!(DataType::from_json_str("\"time\"").is_err());
        assert_eq!(
            DataType::from_json_str("\"time(6)\"").unwrap(),
            DataType::Time { precision: 6 }
        );
    }

    #[test]
    fn test_from_ddl_array_type() {
        let dt = DataType::from_ddl("array<int>").unwrap();
        assert_eq!(dt.simple_string(), "array<int>");

        let dt2 = DataType::from_ddl("array<string>").unwrap();
        assert_eq!(dt2.simple_string(), "array<string>");

        let dt3 = DataType::from_ddl("array<array<int>>").unwrap();
        assert_eq!(dt3.simple_string(), "array<array<int>>");
    }

    #[test]
    fn test_from_ddl_map_type() {
        let dt = DataType::from_ddl("map<string,int>").unwrap();
        assert_eq!(dt.simple_string(), "map<string,int>");

        let dt2 = DataType::from_ddl("map<string,array<int>>").unwrap();
        assert_eq!(dt2.simple_string(), "map<string,array<int>>");
    }

    #[test]
    fn test_from_ddl_struct_type() {
        let dt = DataType::from_ddl("struct<name:string,age:int>").unwrap();
        assert_eq!(dt.simple_string(), "struct<name:string,age:int>");

        let dt2 = DataType::from_ddl("struct<a:int,b:array<string>>").unwrap();
        assert_eq!(dt2.simple_string(), "struct<a:int,b:array<string>>");
    }

    /// Whitespace before and inside the angle brackets of a complex type, which the server's
    /// parser takes (each type as the server's `DDLParse` gives it), and the malformed spellings
    /// that it rejects.
    #[test]
    fn from_ddl_takes_whitespace_around_angle_brackets_as_the_server() {
        for (ddl, server) in [
            ("array <int>", "array<int>"),
            ("STRUCT <a:int>", "struct<a:int>"),
            ("map <string, int>", "map<string,int>"),
            ("a array <int>", "struct<a:array<int>>"),
            ("a: array <int>", "struct<a:array<int>>"),
            (
                "a ARRAY < INT >, b map < string , int >",
                "struct<a:array<int>,b:map<string,int>>",
            ),
            ("array<struct <x: int>>", "array<struct<x:int>>"),
            (
                "struct < a : int , b : array < string > >",
                "struct<a:int,b:array<string>>",
            ),
            ("struct <>", "struct<>"),
            ("a struct <>", "struct<a:struct<>>"),
        ] {
            let parsed = DataType::from_ddl(ddl).unwrap_or_else(|e| panic!("{ddl}: {e}"));
            assert_eq!(parsed.simple_string(), server, "{ddl}");
        }
        let DataType::Struct { fields } = DataType::from_ddl("a array <int> NOT NULL").unwrap()
        else {
            panic!("a struct");
        };
        assert!(!fields[0].nullable);
        for rejected in [
            "array <int> NOT NULL",
            "array <>",
            "array <int",
            "map <int>",
        ] {
            assert!(DataType::from_ddl(rejected).is_err(), "{rejected}");
        }
    }

    #[test]
    fn test_from_ddl_top_level_schema() {
        // Top-level schema without struct<>
        let dt = DataType::from_ddl("a INT, b STRING").unwrap();
        assert_eq!(dt.simple_string(), "struct<a:int,b:string>");

        let dt2 = DataType::from_ddl("a DOUBLE, b CHAR(50)").unwrap();
        assert_eq!(dt2.simple_string(), "struct<a:double,b:char(50)>");

        let dt3 = DataType::from_ddl("name string, age int").unwrap();
        assert_eq!(dt3.simple_string(), "struct<name:string,age:int>");
    }

    #[test]
    fn test_from_ddl_roundtrip() {
        // Test that from_ddl -> simple_string roundtrips correctly
        let test_cases = vec![
            "int",
            "string",
            "array<int>",
            "map<string,int>",
            "struct<name:string,age:int>",
            "decimal(10,2)",
            "char(50)",
            "varchar(100)",
            "array<map<string,int>>",
            "struct<a:int,b:array<string>>",
        ];

        for case in test_cases {
            let dt =
                DataType::from_ddl(case).unwrap_or_else(|_| panic!("Failed to parse: {}", case));
            let simple = dt.simple_string();
            let dt2 = DataType::from_ddl(&simple)
                .unwrap_or_else(|_| panic!("Failed to roundtrip: {}", simple));
            assert_eq!(dt, dt2, "Roundtrip failed for: {}", case);
        }
    }

    #[test]
    fn test_need_conversion() {
        // Types that need conversion
        assert!(DataType::Date.need_conversion());
        assert!(DataType::Timestamp.need_conversion());
        assert!(DataType::TimestampNtz.need_conversion());
        assert!(DataType::CalendarInterval.need_conversion());
        assert!(DataType::Time { precision: 6 }.need_conversion());
        // As PySpark's: variants, spatial values and the values of UDTs, also in arrays.
        assert!(DataType::Variant.need_conversion());
        assert!(DataType::Geometry { srid: 4326 }.need_conversion());
        assert!(DataType::Geography { srid: 4326 }.need_conversion());
        let udt = DataType::Udt {
            type_str: "udt".to_string(),
            jvm_class: None,
            python_class: Some("mymod.MyUDT".to_string()),
            serialized_python_class: None,
            sql_type: Some(Box::new(DataType::Integer)),
        };
        assert!(DataType::Array {
            element_type: Box::new(udt),
            contains_null: true,
        }
        .need_conversion());

        // Types that don't need conversion
        assert!(!DataType::Integer.need_conversion());
        assert!(!DataType::String {
            collation: "UTF8_BINARY".to_string()
        }
        .need_conversion());
        assert!(!DataType::Boolean.need_conversion());
        assert!(!DataType::Double.need_conversion());

        // Array/Map propagate conversion needs
        let array_with_date = DataType::Array {
            element_type: Box::new(DataType::Date),
            contains_null: true,
        };
        assert!(array_with_date.need_conversion());

        let array_without = DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: true,
        };
        assert!(!array_without.need_conversion());

        // StructType always needs conversion
        let struct_type = DataType::Struct { fields: vec![] };
        assert!(struct_type.need_conversion());

        let struct_with_primitives = DataType::Struct {
            fields: vec![
                StructField {
                    name: "a".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "b".to_string(),
                    data_type: DataType::String {
                        collation: "UTF8_BINARY".to_string(),
                    },
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        assert!(struct_with_primitives.need_conversion());
    }

    #[test]
    fn test_field_names() {
        let struct_type = DataType::Struct {
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
                    name: "age".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
            ],
        };

        let names = struct_type.field_names().unwrap();
        assert_eq!(names, vec!["name", "age"]);

        // Test names() alias
        let names2 = struct_type.names().unwrap();
        assert_eq!(names2, vec!["name", "age"]);

        // Test error on non-struct
        let int_type = DataType::Integer;
        assert!(int_type.field_names().is_err());
    }

    #[test]
    fn test_struct_add_builder() {
        let empty_struct = DataType::Struct { fields: vec![] };

        // Add first field
        let with_name = empty_struct
            .add(
                "name",
                DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
                true,
                None,
            )
            .unwrap();

        assert_eq!(with_name.field_names().unwrap(), vec!["name"]);

        // Add second field
        let with_both = with_name.add("age", DataType::Integer, true, None).unwrap();

        assert_eq!(with_both.field_names().unwrap(), vec!["name", "age"]);
        assert_eq!(with_both.simple_string(), "struct<name:string,age:int>");

        // Test add with metadata
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "key".to_string(),
            serde_json::Value::String("value".to_string()),
        );
        let with_metadata = with_both
            .add("score", DataType::Double, false, Some(metadata))
            .unwrap();

        assert_eq!(
            with_metadata.field_names().unwrap(),
            vec!["name", "age", "score"]
        );

        // Test error on non-struct
        let int_type = DataType::Integer;
        assert!(int_type
            .add(
                "field",
                DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
                true,
                None
            )
            .is_err());
    }

    #[test]
    fn test_from_ddl_intervals() {
        // Test interval types
        assert_eq!(
            DataType::from_ddl("interval").unwrap(),
            DataType::CalendarInterval
        );

        assert_eq!(
            DataType::from_ddl("interval day").unwrap(),
            DataType::DayTimeInterval {
                start_field: 0,
                end_field: 0
            }
        );

        assert_eq!(
            DataType::from_ddl("interval year to month").unwrap(),
            DataType::YearMonthInterval {
                start_field: 0,
                end_field: 1
            }
        );

        assert_eq!(
            DataType::from_ddl("interval day to second").unwrap(),
            DataType::DayTimeInterval {
                start_field: 0,
                end_field: 3
            }
        );
    }

    #[test]
    fn test_from_ddl_case_insensitive() {
        // Test case insensitivity
        assert_eq!(
            DataType::from_ddl("INT").unwrap(),
            DataType::from_ddl("int").unwrap()
        );

        assert_eq!(
            DataType::from_ddl("STRUCT<Name:STRING,Age:INT>")
                .unwrap()
                .simple_string(),
            "struct<Name:string,Age:int>"
        );

        assert_eq!(
            DataType::from_ddl("ARRAY<INT>").unwrap(),
            DataType::from_ddl("array<int>").unwrap()
        );
    }

    #[test]
    fn test_to_nullable_primitive_types() {
        // Primitive types should remain unchanged
        let int_type = DataType::Integer;
        assert_eq!(int_type.to_nullable(), DataType::Integer);

        let string_type = DataType::String {
            collation: "UTF8_BINARY".to_string(),
        };
        assert_eq!(string_type.to_nullable(), string_type);

        let bool_type = DataType::Boolean;
        assert_eq!(bool_type.to_nullable(), bool_type);

        let double_type = DataType::Double;
        assert_eq!(double_type.to_nullable(), double_type);

        let date_type = DataType::Date;
        assert_eq!(date_type.to_nullable(), date_type);

        let timestamp_type = DataType::Timestamp;
        assert_eq!(timestamp_type.to_nullable(), timestamp_type);
    }

    #[test]
    fn test_to_nullable_array_type() {
        let array_type = DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: false,
        };
        let nullable = array_type.to_nullable();

        if let DataType::Array {
            element_type: _,
            contains_null,
        } = nullable
        {
            assert!(contains_null);
        } else {
            panic!("Expected Array type");
        }
    }

    #[test]
    fn test_to_nullable_array_nested() {
        let nested_array = DataType::Array {
            element_type: Box::new(DataType::Array {
                element_type: Box::new(DataType::Integer),
                contains_null: false,
            }),
            contains_null: false,
        };
        let nullable = nested_array.to_nullable();

        if let DataType::Array {
            element_type,
            contains_null,
        } = nullable
        {
            assert!(contains_null);
            if let DataType::Array {
                contains_null: inner_null,
                ..
            } = element_type.as_ref()
            {
                assert!(inner_null);
            } else {
                panic!("Expected nested Array type");
            }
        } else {
            panic!("Expected Array type");
        }
    }

    #[test]
    fn test_to_nullable_map_type() {
        let map_type = DataType::Map {
            key_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            value_type: Box::new(DataType::Integer),
            value_contains_null: false,
        };
        let nullable = map_type.to_nullable();

        if let DataType::Map {
            key_type: _,
            value_type: _,
            value_contains_null,
        } = nullable
        {
            assert!(value_contains_null);
        } else {
            panic!("Expected Map type");
        }
    }

    #[test]
    fn test_to_nullable_struct_type() {
        let struct_type = DataType::Struct {
            fields: vec![
                StructField {
                    name: "a".to_string(),
                    data_type: DataType::Integer,
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "b".to_string(),
                    data_type: DataType::String {
                        collation: "UTF8_BINARY".to_string(),
                    },
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        let nullable = struct_type.to_nullable();

        if let DataType::Struct { fields } = nullable {
            assert_eq!(fields.len(), 2);
            assert!(fields[0].nullable);
            assert!(fields[1].nullable);
        } else {
            panic!("Expected Struct type");
        }
    }

    #[test]
    fn test_simple_string_all_types() {
        // Test simple_string for all basic types
        assert_eq!(DataType::Null.simple_string(), "void");
        assert_eq!(DataType::Boolean.simple_string(), "boolean");
        assert_eq!(DataType::Byte.simple_string(), "tinyint");
        assert_eq!(DataType::Short.simple_string(), "smallint");
        assert_eq!(DataType::Integer.simple_string(), "int");
        assert_eq!(DataType::Long.simple_string(), "bigint");
        assert_eq!(DataType::Float.simple_string(), "float");
        assert_eq!(DataType::Double.simple_string(), "double");
        assert_eq!(DataType::Binary.simple_string(), "binary");
        assert_eq!(DataType::Date.simple_string(), "date");
        assert_eq!(DataType::Timestamp.simple_string(), "timestamp");
        assert_eq!(DataType::TimestampNtz.simple_string(), "timestamp_ntz");
        assert_eq!(DataType::CalendarInterval.simple_string(), "interval");
        assert_eq!(DataType::Variant.simple_string(), "variant");
    }

    #[test]
    fn test_simple_string_with_collation() {
        let string_default = DataType::String {
            collation: "".to_string(),
        };
        assert_eq!(string_default.simple_string(), "string");

        let string_utf8 = DataType::String {
            collation: "UTF8_BINARY".to_string(),
        };
        assert_eq!(string_utf8.simple_string(), "string");

        let string_custom = DataType::String {
            collation: "UNICODE".to_string(),
        };
        assert_eq!(string_custom.simple_string(), "string collate UNICODE");
    }

    #[test]
    fn test_type_name_all_types() {
        assert_eq!(DataType::Null.type_name(), "void");
        assert_eq!(DataType::Boolean.type_name(), "boolean");
        assert_eq!(DataType::Byte.type_name(), "byte");
        assert_eq!(DataType::Integer.type_name(), "integer");
        assert_eq!(
            DataType::String {
                collation: "UTF8_BINARY".to_string()
            }
            .type_name(),
            "string"
        );
    }

    #[test]
    fn test_from_json_string_types() {
        let json_int = serde_json::json!("integer");
        let dt = DataType::from_json(&json_int).unwrap();
        assert_eq!(dt, DataType::Integer);

        let json_string = serde_json::json!("string");
        let dt = DataType::from_json(&json_string).unwrap();
        assert_eq!(
            dt,
            DataType::String {
                collation: "UTF8_BINARY".to_string()
            }
        );

        let json_timestamp = serde_json::json!("timestamp");
        let dt = DataType::from_json(&json_timestamp).unwrap();
        assert_eq!(dt, DataType::Timestamp);
    }

    #[test]
    fn test_from_json_str_simple() {
        let dt = DataType::from_json_str("\"integer\"").unwrap();
        assert_eq!(dt, DataType::Integer);

        let dt = DataType::from_json_str("\"double\"").unwrap();
        assert_eq!(dt, DataType::Double);

        let dt = DataType::from_json_str("\"boolean\"").unwrap();
        assert_eq!(dt, DataType::Boolean);
    }

    #[test]
    fn test_from_json_complex_types() {
        // Array type
        let json_array = serde_json::json!({
            "type": "array",
            "elementType": "integer",
            "containsNull": true
        });
        let dt = DataType::from_json(&json_array).unwrap();
        match dt {
            DataType::Array {
                element_type,
                contains_null,
            } => {
                assert_eq!(*element_type, DataType::Integer);
                assert!(contains_null);
            }
            _ => panic!("Expected Array type"),
        }

        // Map type
        let json_map = serde_json::json!({
            "type": "map",
            "keyType": "string",
            "valueType": "integer",
            "valueContainsNull": false
        });
        let dt = DataType::from_json(&json_map).unwrap();
        match dt {
            DataType::Map {
                key_type,
                value_type,
                value_contains_null,
            } => {
                assert_eq!(
                    *key_type,
                    DataType::String {
                        collation: "UTF8_BINARY".to_string()
                    }
                );
                assert_eq!(*value_type, DataType::Integer);
                assert!(!value_contains_null);
            }
            _ => panic!("Expected Map type"),
        }

        // Struct type
        let json_struct = serde_json::json!({
            "type": "struct",
            "fields": [
                {
                    "name": "a",
                    "type": "integer",
                    "nullable": true,
                    "metadata": {}
                },
                {
                    "name": "b",
                    "type": "string",
                    "nullable": false,
                    "metadata": {}
                }
            ]
        });
        let dt = DataType::from_json(&json_struct).unwrap();
        match dt {
            DataType::Struct { fields } => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "a");
                assert_eq!(fields[1].name, "b");
            }
            _ => panic!("Expected Struct type"),
        }
    }

    #[test]
    fn test_from_json_decimal() {
        let json_decimal = serde_json::json!("decimal(10,2)");
        let dt = DataType::from_json(&json_decimal).unwrap();
        match dt {
            DataType::Decimal { precision, scale } => {
                assert_eq!(precision, 10);
                assert_eq!(scale, 2);
            }
            _ => panic!("Expected Decimal type"),
        }
    }

    #[test]
    fn test_decimal_out_of_range_is_not_rejected_at_parse_time() {
        // As PySpark, whose `DecimalType(39, 0)`, `_parse_datatype_json_string` and
        // `pyspark_types_to_proto_types` do not validate the precision or the scale: the server
        // rejects the type on analysis. The client only refuses it where it has to build an
        // Arrow type (`decimal128_precision_and_scale` in session.rs).
        for (precision, scale) in [(39, 0), (5, 6), (10, -2), (300, 0), (0, 0)] {
            let dt = DataType::Decimal { precision, scale };
            let text = format!("decimal({precision},{scale})");
            assert_eq!(DataType::from_json(&serde_json::json!(text)).unwrap(), dt);
            assert_eq!(dt.json_value(), serde_json::json!(text));
            assert_eq!(DataType::from_proto(&dt.to_proto()).unwrap(), dt);
        }
    }

    #[test]
    fn test_from_json_unparsed_roundtrip() {
        let dt = DataType::Unparsed {
            data_type_string: "int".to_string(),
        };
        assert_eq!(DataType::from_json(&dt.json_value()).unwrap(), dt);
    }

    #[test]
    fn test_from_json_lenient_aliases() {
        let parse = |s: &str| DataType::from_json(&serde_json::json!(s)).unwrap();
        assert_eq!(
            parse("decimal"),
            DataType::Decimal {
                precision: 10,
                scale: 0
            }
        );
        assert_eq!(parse("null"), DataType::Null);
        assert_eq!(parse("calendarinterval"), DataType::CalendarInterval);
    }

    #[test]
    fn test_from_json_geometry() {
        // JSON names the mixed SRID by its CRS, `SRID:ANY`; `geometry(any)` is its DDL.
        let json_geom = serde_json::json!("geometry(SRID:ANY)");
        let dt = DataType::from_json(&json_geom).unwrap();
        match dt {
            DataType::Geometry { srid } => {
                assert_eq!(srid, -1);
            }
            _ => panic!("Expected Geometry type"),
        }
    }

    #[test]
    fn test_json_value_simple_types() {
        let int_val = DataType::Integer.json_value();
        assert_eq!(int_val, serde_json::json!("integer"));

        let string_val = DataType::String {
            collation: "UTF8_BINARY".to_string(),
        }
        .json_value();
        assert_eq!(string_val, serde_json::json!("string"));

        let bool_val = DataType::Boolean.json_value();
        assert_eq!(bool_val, serde_json::json!("boolean"));
    }

    #[test]
    fn test_json_value_complex_types() {
        let array_val = DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: true,
        }
        .json_value();
        assert!(array_val.is_object());
        assert_eq!(
            array_val.get("type").and_then(|v| v.as_str()),
            Some("array")
        );
    }

    #[test]
    fn test_to_ddl() {
        let struct_type = DataType::Struct {
            fields: vec![
                StructField {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "name".to_string(),
                    data_type: DataType::String {
                        collation: "UTF8_BINARY".to_string(),
                    },
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        let ddl = struct_type.to_ddl().unwrap();
        assert!(ddl.contains("id"));
        assert!(ddl.contains("name"));
        assert!(ddl.to_uppercase().contains("INT") || ddl.contains("int"));
        assert!(ddl.to_uppercase().contains("STRING") || ddl.contains("string"));
    }

    #[test]
    fn test_tree_string() {
        let struct_type = DataType::Struct {
            fields: vec![
                StructField {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "nested".to_string(),
                    data_type: DataType::Struct {
                        fields: vec![StructField {
                            name: "x".to_string(),
                            data_type: DataType::Double,
                            nullable: true,
                            metadata: BTreeMap::new(),
                        }],
                    },
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        let tree = struct_type.tree_string().unwrap();
        assert!(tree.contains("id"));
        assert!(tree.contains("nested"));
        assert!(tree.contains("x"));
    }

    #[test]
    fn test_to_proto_simple_types() {
        let int_proto = DataType::Integer.to_proto();
        assert!(int_proto.kind.is_some());

        let string_proto = DataType::String {
            collation: "UTF8_BINARY".to_string(),
        }
        .to_proto();
        assert!(string_proto.kind.is_some());

        let bool_proto = DataType::Boolean.to_proto();
        assert!(bool_proto.kind.is_some());
    }

    #[test]
    fn test_to_proto_array_type() {
        let array_type = DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: false,
        };
        let proto = array_type.to_proto();
        assert!(proto.kind.is_some());
    }

    #[test]
    fn test_to_proto_map_type() {
        let map_type = DataType::Map {
            key_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            value_type: Box::new(DataType::Integer),
            value_contains_null: true,
        };
        let proto = map_type.to_proto();
        assert!(proto.kind.is_some());
    }

    #[test]
    fn test_to_proto_struct_type() {
        let struct_type = DataType::Struct {
            fields: vec![
                StructField {
                    name: "a".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "b".to_string(),
                    data_type: DataType::String {
                        collation: "UTF8_BINARY".to_string(),
                    },
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
            ],
        };
        let proto = struct_type.to_proto();
        assert!(proto.kind.is_some());
    }

    #[test]
    fn test_display_trait() {
        let int_type = DataType::Integer;
        let display_str = format!("{}", int_type);
        assert_eq!(display_str, "int");

        let array_type = DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: true,
        };
        let display_str = format!("{}", array_type);
        assert_eq!(display_str, "array<int>");
    }

    #[test]
    fn test_interval_year_month() {
        let ym = DataType::YearMonthInterval {
            start_field: 0,
            end_field: 1,
        };
        assert_eq!(ym.simple_string(), "interval year to month");
    }

    #[test]
    fn test_interval_day_time() {
        let dt = DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        };
        assert_eq!(dt.simple_string(), "interval day to second");
    }

    #[test]
    fn test_udt_type() {
        let udt = DataType::Udt {
            type_str: "com.example.MyUDT".to_string(),
            jvm_class: Some("com.example.MyUDT".to_string()),
            python_class: None,
            serialized_python_class: None,
            sql_type: None,
        };
        assert_eq!(udt.simple_string(), "udt");
        assert_eq!(udt.type_name(), "udt");
    }

    #[test]
    fn test_char_varchar() {
        let char_type = DataType::Char { length: 50 };
        assert_eq!(char_type.simple_string(), "char(50)");

        let varchar_type = DataType::Varchar { length: 100 };
        assert_eq!(varchar_type.simple_string(), "varchar(100)");
    }
}

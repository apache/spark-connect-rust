//! DataType type system mirroring PySpark's `pyspark.sql.types`.
//!
//! Defines the type hierarchy and conversion functions between Python DataTypes
//! and the Spark Connect protobuf representation.

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use spark_connect_core::error::{Result, SparkError};

/// The base DataType representation, mirroring `pyspark.sql.types.DataType`.
///
/// All concrete types are variants of this enum. Each variant carries the data
/// needed to fully specify that type (e.g., DecimalType carries precision and scale).
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// A field in a StructType, mirroring `pyspark.sql.types.StructField`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub metadata: BTreeMap<String, String>,
}

impl DataType {
    /// Parses a DDL-formatted string into a DataType, mirroring `DataType.fromDDL()`.
    ///
    /// This supports:
    /// - Primitive types: int, bigint, string, double, boolean, date, timestamp, binary,
    ///   tinyint, smallint, float, decimal(p,s), char(n), varchar(n), interval
    /// - Complex types: array<...>, map<...,...>, struct<name:type,...>
    /// - Top-level struct can omit the "struct<>" wrapper for backward compatibility
    /// - DDL like "a INT, b STRING" is parsed as a struct
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
    /// - ArrayType: if element type needs conversion
    /// - MapType: if key or value type needs conversion
    /// - StructType: always needs conversion
    pub fn need_conversion(&self) -> bool {
        match self {
            DataType::Date
            | DataType::Timestamp
            | DataType::TimestampNtz
            | DataType::Time { .. }
            | DataType::DayTimeInterval { .. }
            | DataType::YearMonthInterval { .. }
            | DataType::CalendarInterval => true,
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
            DataType::CalendarInterval => "interval".to_string(),
            DataType::YearMonthInterval { .. } => "interval".to_string(),
            DataType::DayTimeInterval { .. } => "interval".to_string(),
            DataType::Array { .. } => "array".to_string(),
            DataType::Map { .. } => "map".to_string(),
            DataType::Struct { .. } => "struct".to_string(),
            DataType::Variant => "variant".to_string(),
            DataType::Geometry { .. } => "geometry".to_string(),
            DataType::Geography { .. } => "geography".to_string(),
            DataType::Udt { .. } => "udt".to_string(),
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
                if collation == "UTF8_BINARY" {
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
                if *srid == -1 {
                    serde_json::json!("geometry(SRID:ANY)")
                } else {
                    serde_json::json!(format!("geometry(OGC:CRS84)"))
                }
            }
            DataType::Geography { srid } => {
                if *srid == -1 {
                    serde_json::json!("geography(SRID:ANY, SPHERICAL)")
                } else {
                    serde_json::json!(format!("geography(OGC:CRS84, SPHERICAL)"))
                }
            }
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
        parse_json_value(value, None)
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
            Some(spark_connect_proto::data_type::Kind::TimestampNtzNanos(tn)) => {
                let precision = tn.precision.unwrap_or(9);
                Ok(DataType::Time { precision })
            }
            Some(spark_connect_proto::data_type::Kind::TimestampLtzNanos(_)) => {
                Ok(DataType::Timestamp)
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

    /// Returns the JSON value representation, mirroring `StructField.jsonValue()`.
    pub fn json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "name": &self.name,
            "type": self.data_type.json_value(),
            "nullable": self.nullable,
            "metadata": self.metadata,
        })
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
        metadata: Option<BTreeMap<String, String>>,
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

/// Parses a JSON value into a DataType
fn parse_json_value(value: &serde_json::Value, _field_name: Option<&str>) -> Result<DataType> {
    match value {
        serde_json::Value::String(s) => parse_json_type_string(s),
        serde_json::Value::Object(obj) => parse_json_object(obj),
        _ => Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Expected string or object for DataType")],
        )),
    }
}

fn parse_json_type_string(s: &str) -> Result<DataType> {
    match s {
        "void" => Ok(DataType::Null),
        "boolean" => Ok(DataType::Boolean),
        "byte" => Ok(DataType::Byte),
        "short" => Ok(DataType::Short),
        "integer" => Ok(DataType::Integer),
        "long" => Ok(DataType::Long),
        "float" => Ok(DataType::Float),
        "double" => Ok(DataType::Double),
        "binary" => Ok(DataType::Binary),
        "date" => Ok(DataType::Date),
        "timestamp" => Ok(DataType::Timestamp),
        "timestamp_ntz" => Ok(DataType::TimestampNtz),
        "interval" => Ok(DataType::CalendarInterval),
        "variant" => Ok(DataType::Variant),
        s if s.starts_with("decimal(") => parse_decimal(s),
        s if s.starts_with("char(") => parse_char(s),
        s if s.starts_with("varchar(") => parse_varchar(s),
        s if s.starts_with("time(") => parse_time(s),
        s if s.starts_with("interval") => parse_interval_string(s),
        s if s.starts_with("string collate") => {
            let collation = s.trim_start_matches("string collate").trim().to_string();
            Ok(DataType::String { collation })
        }
        "string" => Ok(DataType::String {
            collation: "UTF8_BINARY".to_string(),
        }),
        s if s.starts_with("geometry(") => parse_geometry(s),
        s if s.starts_with("geography(") => parse_geography(s),
        s if s.starts_with("unparsed(") => Ok(DataType::Unparsed {
            data_type_string: s.to_string(),
        }),
        _ => Err(SparkError::value(
            "INVALID_DATATYPE_STRING",
            &[("detail", &format!("Unknown type string: {}", s))],
        )),
    }
}

fn parse_json_object(obj: &serde_json::Map<String, serde_json::Value>) -> Result<DataType> {
    let type_field = obj.get("type").and_then(|v| v.as_str());

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
                element_type: Box::new(parse_json_value(element_type, None)?),
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
                key_type: Box::new(parse_json_value(key_type, None)?),
                value_type: Box::new(parse_json_value(value_type, None)?),
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
            for field_obj in fields_arr {
                if let Some(field_map) = field_obj.as_object() {
                    let name = field_map
                        .get("name")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            SparkError::value(
                                "INVALID_DATATYPE_FORMAT",
                                &[("detail", "StructField missing name")],
                            )
                        })?
                        .to_string();
                    let field_type = field_map.get("type").ok_or_else(|| {
                        SparkError::value(
                            "INVALID_DATATYPE_FORMAT",
                            &[("detail", "StructField missing type")],
                        )
                    })?;
                    let data_type = parse_json_value(field_type, Some(&name))?;
                    let nullable = field_map
                        .get("nullable")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    let metadata = field_map
                        .get("metadata")
                        .and_then(|v| v.as_object())
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.to_string())).collect())
                        .unwrap_or_default();

                    fields.push(StructField {
                        name,
                        data_type,
                        nullable,
                        metadata,
                    });
                }
            }
            Ok(DataType::Struct { fields })
        }
        _ => Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Unknown type in object")],
        )),
    }
}

fn parse_decimal(s: &str) -> Result<DataType> {
    let inner = s
        .strip_prefix("decimal(")
        .and_then(|s| s.strip_suffix(")"))
        .ok_or_else(|| {
            SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "Invalid decimal format")],
            )
        })?;

    let parts: Vec<&str> = inner.split(',').collect();
    if parts.len() != 2 {
        return Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Decimal requires precision and scale")],
        ));
    }

    let precision = parts[0].trim().parse::<i32>().map_err(|_| {
        SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Invalid precision")],
        )
    })?;
    let scale = parts[1].trim().parse::<i32>().map_err(|_| {
        SparkError::value("INVALID_DATATYPE_FORMAT", &[("detail", "Invalid scale")])
    })?;

    Ok(DataType::Decimal { precision, scale })
}

fn parse_char(s: &str) -> Result<DataType> {
    let inner = s
        .strip_prefix("char(")
        .and_then(|s| s.strip_suffix(")"))
        .ok_or_else(|| {
            SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "Invalid char format")],
            )
        })?;

    let length = inner.trim().parse::<i32>().map_err(|_| {
        SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Invalid char length")],
        )
    })?;

    Ok(DataType::Char { length })
}

fn parse_varchar(s: &str) -> Result<DataType> {
    let inner = s
        .strip_prefix("varchar(")
        .and_then(|s| s.strip_suffix(")"))
        .ok_or_else(|| {
            SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "Invalid varchar format")],
            )
        })?;

    let length = inner.trim().parse::<i32>().map_err(|_| {
        SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Invalid varchar length")],
        )
    })?;

    Ok(DataType::Varchar { length })
}

fn parse_time(s: &str) -> Result<DataType> {
    let inner = s
        .strip_prefix("time(")
        .and_then(|s| s.strip_suffix(")"))
        .ok_or_else(|| {
            SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "Invalid time format")],
            )
        })?;

    let precision = inner.trim().parse::<i32>().map_err(|_| {
        SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Invalid time precision")],
        )
    })?;

    Ok(DataType::Time { precision })
}

fn parse_interval_string(s: &str) -> Result<DataType> {
    if s == "interval" {
        return Ok(DataType::CalendarInterval);
    }

    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(DataType::CalendarInterval);
    }

    if parts.len() == 2 {
        // Single field like "interval day"
        let field_name = parts[1];
        for &(name, code) in YEAR_MONTH_FIELDS {
            if name == field_name {
                return Ok(DataType::YearMonthInterval {
                    start_field: code,
                    end_field: code,
                });
            }
        }
        for &(name, code) in DAY_TIME_FIELDS {
            if name == field_name {
                return Ok(DataType::DayTimeInterval {
                    start_field: code,
                    end_field: code,
                });
            }
        }
    } else if parts.len() == 4 && parts[2] == "to" {
        // Range like "interval day to second"
        let start = parts[1];
        let end = parts[3];

        // Try year-month fields
        if let (Some(&(_, start_code)), Some(&(_, end_code))) = (
            YEAR_MONTH_FIELDS.iter().find(|(name, _)| *name == start),
            YEAR_MONTH_FIELDS.iter().find(|(name, _)| *name == end),
        ) {
            return Ok(DataType::YearMonthInterval {
                start_field: start_code,
                end_field: end_code,
            });
        }

        // Try day-time fields
        if let (Some(&(_, start_code)), Some(&(_, end_code))) = (
            DAY_TIME_FIELDS.iter().find(|(name, _)| *name == start),
            DAY_TIME_FIELDS.iter().find(|(name, _)| *name == end),
        ) {
            return Ok(DataType::DayTimeInterval {
                start_field: start_code,
                end_field: end_code,
            });
        }
    }

    Ok(DataType::CalendarInterval)
}

fn parse_geometry(s: &str) -> Result<DataType> {
    let inner = s
        .strip_prefix("geometry(")
        .and_then(|s| s.strip_suffix(")"))
        .ok_or_else(|| {
            SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "Invalid geometry format")],
            )
        })?;

    let srid = if inner.to_lowercase() == "any" {
        -1
    } else {
        inner.parse::<i32>().map_err(|_| {
            SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "Invalid geometry srid")],
            )
        })?
    };

    Ok(DataType::Geometry { srid })
}

fn parse_geography(s: &str) -> Result<DataType> {
    let inner = s
        .strip_prefix("geography(")
        .and_then(|s| s.strip_suffix(")"))
        .ok_or_else(|| {
            SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", "Invalid geography format")],
            )
        })?;

    let srid = if inner.to_lowercase() == "any" {
        -1
    } else {
        inner
            .split(',')
            .next()
            .unwrap_or("4326")
            .parse::<i32>()
            .map_err(|_| {
                SparkError::value(
                    "INVALID_DATATYPE_FORMAT",
                    &[("detail", "Invalid geography srid")],
                )
            })?
    };

    Ok(DataType::Geography { srid })
}

/// Main DDL parser: parses DDL strings into DataType.
/// Handles primitive types, complex types (array, map, struct), and top-level schemas.
fn parse_datatype_string(input: &str) -> Result<DataType> {
    let trimmed = input.trim();

    // Check if it's a top-level schema (multiple fields like "a INT, b STRING")
    // Heuristic: contains comma at top level (not inside angle brackets/parens)
    // and doesn't start with "struct<"
    if !trimmed.to_lowercase().starts_with("struct<") && contains_top_level_comma(trimmed) {
        return parse_top_level_schema(trimmed);
    }

    parse_single_type(trimmed)
}

/// Check if a string contains a comma at the top level (not inside brackets/parens)
fn contains_top_level_comma(s: &str) -> bool {
    let mut depth = 0;
    for c in s.chars() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

/// Parse a top-level schema like "a INT, b STRING" or "a: int, b: string"
fn parse_top_level_schema(input: &str) -> Result<DataType> {
    let mut fields = Vec::new();
    let parts = split_top_level_comma(input);

    for part in parts {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Parse "name type" or "name: type" format
        let (name, type_str) = if let Some(colon_idx) = trimmed.find(':') {
            let name = trimmed[..colon_idx].trim();
            let type_str = trimmed[colon_idx + 1..].trim();
            (name, type_str)
        } else {
            // Assume first word is name, rest is type
            let parts: Vec<&str> = trimmed.splitn(2, char::is_whitespace).collect();
            if parts.len() != 2 {
                return Err(SparkError::value(
                    "INVALID_DATATYPE_FORMAT",
                    &[("detail", &format!("Cannot parse field: {}", trimmed))],
                ));
            }
            (parts[0], parts[1])
        };

        let data_type = parse_single_type(type_str)?;
        fields.push(StructField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: BTreeMap::new(),
        });
    }

    Ok(DataType::Struct { fields })
}

/// Split by top-level commas (respecting angle brackets and parentheses)
fn split_top_level_comma(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut start = 0;
    let mut depth = 0;
    let bytes = s.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'<' | b'(' => depth += 1,
            b'>' | b')' => depth -= 1,
            b',' if depth == 0 => {
                result.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    result.push(&s[start..]);
    result
}

/// Parse a single type (primitive or complex)
fn parse_single_type(input: &str) -> Result<DataType> {
    let trimmed = input.trim().to_lowercase();

    // Complex types
    if trimmed.starts_with("array<") {
        return parse_array_type(input);
    }
    if trimmed.starts_with("map<") {
        return parse_map_type(input);
    }
    if trimmed.starts_with("struct<") {
        return parse_struct_type(input);
    }

    // Primitive types
    match trimmed.as_str() {
        "null" | "void" => Ok(DataType::Null),
        "boolean" => Ok(DataType::Boolean),
        "byte" | "tinyint" => Ok(DataType::Byte),
        "short" | "smallint" => Ok(DataType::Short),
        "int" | "integer" => Ok(DataType::Integer),
        "long" | "bigint" => Ok(DataType::Long),
        "float" => Ok(DataType::Float),
        "double" => Ok(DataType::Double),
        "string" => Ok(DataType::String {
            collation: "UTF8_BINARY".to_string(),
        }),
        "binary" => Ok(DataType::Binary),
        "date" => Ok(DataType::Date),
        "timestamp" => Ok(DataType::Timestamp),
        "timestamp_ntz" => Ok(DataType::TimestampNtz),
        "interval" => Ok(DataType::CalendarInterval),
        "variant" => Ok(DataType::Variant),
        _ => {
            // Try complex parsing with already-lowercased trimmed version
            if trimmed.starts_with("decimal(") {
                parse_decimal(&trimmed)
            } else if trimmed.starts_with("char(") {
                parse_char(&trimmed)
            } else if trimmed.starts_with("varchar(") {
                parse_varchar(&trimmed)
            } else if trimmed.starts_with("time(") {
                parse_time(&trimmed)
            } else if trimmed.starts_with("interval") {
                parse_interval_string(&trimmed)
            } else if trimmed.starts_with("string collate") {
                let collation = trimmed
                    .trim_start_matches("string")
                    .trim_start_matches("collate")
                    .trim()
                    .to_string();
                Ok(DataType::String { collation })
            } else if trimmed.starts_with("geometry(") {
                parse_geometry(&trimmed)
            } else if trimmed.starts_with("geography(") {
                parse_geography(&trimmed)
            } else {
                Err(SparkError::value(
                    "INVALID_DATATYPE_FORMAT",
                    &[("detail", &format!("Unknown type: {}", input))],
                ))
            }
        }
    }
}

/// Parse array<elementType>
fn parse_array_type(input: &str) -> Result<DataType> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("array<") || !trimmed.ends_with('>') {
        return Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Invalid array type format")],
        ));
    }

    let inner = &trimmed[6..trimmed.len() - 1];
    let element_type = parse_single_type(inner)?;

    Ok(DataType::Array {
        element_type: Box::new(element_type),
        contains_null: true,
    })
}

/// Parse map<keyType,valueType>
fn parse_map_type(input: &str) -> Result<DataType> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("map<") || !trimmed.ends_with('>') {
        return Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Invalid map type format")],
        ));
    }

    let inner = &trimmed[4..trimmed.len() - 1];
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
    let mut depth = 0;
    let mut parts = Vec::new();
    let mut start = 0;

    for (i, c) in s.chars().enumerate() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);

    Ok(parts)
}

/// Parse struct<field1:type1,field2:type2,...>
fn parse_struct_type(input: &str) -> Result<DataType> {
    let trimmed = input.trim();
    if !trimmed.to_lowercase().starts_with("struct<") || !trimmed.ends_with('>') {
        return Err(SparkError::value(
            "INVALID_DATATYPE_FORMAT",
            &[("detail", "Invalid struct type format")],
        ));
    }

    let inner = &trimmed[7..trimmed.len() - 1];
    if inner.trim().is_empty() {
        return Ok(DataType::Struct { fields: vec![] });
    }

    let mut fields = Vec::new();
    let parts = split_struct_fields(inner);

    for part in parts {
        let part_trimmed = part.trim();
        if part_trimmed.is_empty() {
            continue;
        }

        // Find the colon separating field name from type
        if let Some(colon_idx) = find_unbracketed_colon(part_trimmed) {
            let name = part_trimmed[..colon_idx].trim();
            let type_str = part_trimmed[colon_idx + 1..].trim();

            let data_type = parse_single_type(type_str)?;
            fields.push(StructField {
                name: name.to_string(),
                data_type,
                nullable: true,
                metadata: BTreeMap::new(),
            });
        } else {
            return Err(SparkError::value(
                "INVALID_DATATYPE_FORMAT",
                &[("detail", &format!("Invalid struct field: {}", part_trimmed))],
            ));
        }
    }

    Ok(DataType::Struct { fields })
}

/// Split struct fields by comma at the top level
fn split_struct_fields(s: &str) -> Vec<&str> {
    let mut depth = 0;
    let mut parts = Vec::new();
    let mut start = 0;

    for (i, c) in s.chars().enumerate() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);

    parts
}

/// Find the position of an unbracketed colon
fn find_unbracketed_colon(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.chars().enumerate() {
        match c {
            '<' | '(' => depth += 1,
            '>' | ')' => depth -= 1,
            ':' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
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
                expected_type_name: "interval",
            },
            TestCase {
                dt: DataType::YearMonthInterval {
                    start_field: 0,
                    end_field: 1,
                },
                expected_simple: "interval year to month",
                expected_type_name: "interval",
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

        // Test time
        assert_eq!(
            DataType::from_ddl("time(6)").unwrap(),
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
            let dt = DataType::from_ddl(case).expect(&format!("Failed to parse: {}", case));
            let simple = dt.simple_string();
            let dt2 =
                DataType::from_ddl(&simple).expect(&format!("Failed to roundtrip: {}", simple));
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
        metadata.insert("key".to_string(), "value".to_string());
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
}

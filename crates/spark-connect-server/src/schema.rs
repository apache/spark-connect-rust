// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The supported Spark schema boundary. Unsupported Arrow types fail explicitly.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, TimeUnit};
use prost::Message;
use spark_connect_proto as proto;
use tonic::Status;

pub(crate) const MAX_METADATA_RESPONSE_BYTES: usize = 3 * 1024 * 1024;

pub(crate) fn result_schema(schema: &Schema) -> Result<proto::DataType, Status> {
    let schema = spark_schema(schema)?;
    validate_schema_size(&schema)?;
    Ok(schema)
}

pub(crate) fn validate_schema_size(schema: &proto::DataType) -> Result<(), Status> {
    if schema.encoded_len() > MAX_METADATA_RESPONSE_BYTES {
        return Err(Status::resource_exhausted(
            "Spark schema exceeds the 3 MiB schema/command response limit",
        ));
    }
    Ok(())
}

/// Transform fields after recursively transforming their supported child fields.
pub(crate) fn map_fields(
    fields: &Fields,
    transform: &impl Fn(&FieldRef, DataType) -> FieldRef,
) -> Fields {
    fn field(input: &FieldRef, transform: &impl Fn(&FieldRef, DataType) -> FieldRef) -> FieldRef {
        let data_type = match input.data_type() {
            DataType::List(child) => DataType::List(field(child, transform)),
            DataType::LargeList(child) => DataType::LargeList(field(child, transform)),
            DataType::Struct(children) => DataType::Struct(map_fields(children, transform)),
            DataType::Map(entries, sorted) => DataType::Map(field(entries, transform), *sorted),
            other => other.clone(),
        };
        transform(input, data_type)
    }

    fields.iter().map(|input| field(input, transform)).collect()
}

/// Compare Spark-visible fields and types, allowing equivalent Arrow physical
/// representations and differences in nullability or field/schema metadata.
pub(crate) fn comparable_schema(schema: &Schema) -> Result<proto::DataType, Status> {
    let fields = map_fields(schema.fields(), &|field, data_type| {
        Arc::new(Field::new(field.name(), data_type, true))
    });
    spark_schema(&Schema::new(fields))
}

pub fn spark_schema(schema: &Schema) -> Result<proto::DataType, Status> {
    spark_type(&DataType::Struct(schema.fields().clone()))
}

fn validate_field_metadata(field: &Field) -> Result<(), Status> {
    if !field.metadata().is_empty() {
        return Err(Status::unimplemented(
            "Arrow field metadata in Spark schemas",
        ));
    }
    Ok(())
}

fn spark_field(field: &Field) -> Result<proto::data_type::StructField, Status> {
    validate_field_metadata(field)?;
    Ok(proto::data_type::StructField {
        name: field.name().clone(),
        data_type: Some(spark_type(field.data_type())?),
        nullable: field.is_nullable(),
        metadata: None,
    })
}

fn spark_type(data_type: &DataType) -> Result<proto::DataType, Status> {
    use proto::data_type::{self as pb, Kind};
    let kind = match data_type {
        DataType::Null => Kind::Null(pb::Null::default()),
        DataType::Boolean => Kind::Boolean(pb::Boolean::default()),
        DataType::Int8 => Kind::Byte(pb::Byte::default()),
        DataType::Int16 => Kind::Short(pb::Short::default()),
        DataType::Int32 => Kind::Integer(pb::Integer::default()),
        DataType::Int64 => Kind::Long(pb::Long::default()),
        DataType::Float32 => Kind::Float(pb::Float::default()),
        DataType::Float64 => Kind::Double(pb::Double::default()),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Kind::String(pb::String {
            collation: "UTF8_BINARY".into(),
            ..Default::default()
        }),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => {
            Kind::Binary(pb::Binary::default())
        }
        DataType::Date32 => Kind::Date(pb::Date::default()),
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            Kind::TimestampNtz(pb::TimestampNtz::default())
        }
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => {
            Kind::Timestamp(pb::Timestamp::default())
        }
        DataType::Decimal128(precision, scale) => Kind::Decimal(pb::Decimal {
            precision: Some(i32::from(*precision)),
            scale: Some(i32::from(*scale)),
            ..Default::default()
        }),
        DataType::List(field) | DataType::LargeList(field) => {
            validate_field_metadata(field)?;
            Kind::Array(Box::new(pb::Array {
                element_type: Some(Box::new(spark_type(field.data_type())?)),
                contains_null: field.is_nullable(),
                ..Default::default()
            }))
        }
        DataType::Struct(fields) => Kind::Struct(pb::Struct {
            fields: fields
                .iter()
                .map(|field| spark_field(field))
                .collect::<Result<_, _>>()?,
            ..Default::default()
        }),
        DataType::Map(entries, _) => {
            validate_field_metadata(entries)?;
            let DataType::Struct(fields) = entries.data_type() else {
                return Err(Status::invalid_argument(
                    "Arrow map entries must be a struct",
                ));
            };
            if fields.len() != 2 {
                return Err(Status::invalid_argument(
                    "Arrow map entries need key and value fields",
                ));
            }
            validate_field_metadata(&fields[0])?;
            validate_field_metadata(&fields[1])?;
            Kind::Map(Box::new(pb::Map {
                key_type: Some(Box::new(spark_type(fields[0].data_type())?)),
                value_type: Some(Box::new(spark_type(fields[1].data_type())?)),
                value_contains_null: fields[1].is_nullable(),
                ..Default::default()
            }))
        }
        other => {
            return Err(Status::unimplemented(format!(
                "Arrow result type {other:?}"
            )))
        }
    };
    Ok(proto::DataType { kind: Some(kind) })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn annotated(name: &str, data_type: DataType, nullable: bool) -> Field {
        Field::new(name, data_type, nullable).with_metadata(HashMap::from([(
            "producer.note".into(),
            "cannot be represented in Spark schemas".into(),
        )]))
    }

    #[test]
    fn declared_metadata_is_rejected_on_list_and_every_map_child() {
        let entries = |key: Field, value: Field| {
            Field::new("entries", DataType::Struct(vec![key, value].into()), false)
        };
        let types = [
            DataType::List(Arc::new(annotated("item", DataType::Int64, true))),
            DataType::LargeList(Arc::new(annotated("item", DataType::Int64, true))),
            DataType::Map(
                Arc::new(annotated(
                    "entries",
                    DataType::Struct(
                        vec![
                            Field::new("key", DataType::Utf8, false),
                            Field::new("value", DataType::Int64, true),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            ),
            DataType::Map(
                Arc::new(entries(
                    annotated("key", DataType::Utf8, false),
                    Field::new("value", DataType::Int64, true),
                )),
                false,
            ),
            DataType::Map(
                Arc::new(entries(
                    Field::new("key", DataType::Utf8, false),
                    annotated("value", DataType::Int64, true),
                )),
                false,
            ),
        ];
        for data_type in types {
            let schema = Schema::new(vec![Field::new("nested", data_type, true)]);
            assert_eq!(
                result_schema(&schema).unwrap_err().code(),
                tonic::Code::Unimplemented
            );
            // Batch annotations remain acceptable when comparing the declared
            // Spark-visible shape, including all nested child fields.
            comparable_schema(&schema).unwrap();
        }
    }
}

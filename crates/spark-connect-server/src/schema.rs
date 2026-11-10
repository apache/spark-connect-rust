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

use arrow::datatypes::{DataType, Field, FieldRef, Schema, TimeUnit};
use prost::Message;
use spark_connect_proto as proto;
use tonic::Status;

pub(crate) const MAX_RESULT_BYTES: usize = 3 * 1024 * 1024;

pub(crate) fn result_schema(schema: &Schema) -> Result<proto::DataType, Status> {
    let schema = spark_schema(schema)?;
    validate_schema_size(&schema)?;
    Ok(schema)
}

pub(crate) fn validate_schema_size(schema: &proto::DataType) -> Result<(), Status> {
    if schema.encoded_len() > MAX_RESULT_BYTES {
        return Err(Status::resource_exhausted(
            "Spark schema exceeds the 3 MiB result limit",
        ));
    }
    Ok(())
}

/// Compare Spark-visible fields and types, allowing equivalent Arrow physical
/// representations and differences in nullability or field/schema metadata.
pub(crate) fn comparable_schema(schema: &Schema) -> Result<proto::DataType, Status> {
    fn field(input: &FieldRef) -> FieldRef {
        Arc::new(Field::new(input.name(), data_type(input.data_type()), true))
    }

    fn data_type(input: &DataType) -> DataType {
        match input {
            DataType::List(child) => DataType::List(field(child)),
            DataType::LargeList(child) => DataType::LargeList(field(child)),
            DataType::Struct(fields) => DataType::Struct(fields.iter().map(field).collect()),
            DataType::Map(entries, sorted) => DataType::Map(field(entries), *sorted),
            other => other.clone(),
        }
    }

    let fields: Vec<_> = schema.fields().iter().map(field).collect();
    spark_schema(&Schema::new(fields))
}

pub fn spark_schema(schema: &Schema) -> Result<proto::DataType, Status> {
    spark_type(&DataType::Struct(schema.fields().clone()))
}

fn spark_field(field: &Field) -> Result<proto::data_type::StructField, Status> {
    if !field.metadata().is_empty() {
        return Err(Status::unimplemented(
            "Arrow field metadata in Spark schemas",
        ));
    }
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
        DataType::List(field) | DataType::LargeList(field) => Kind::Array(Box::new(pb::Array {
            element_type: Some(Box::new(spark_type(field.data_type())?)),
            contains_null: field.is_nullable(),
            ..Default::default()
        })),
        DataType::Struct(fields) => Kind::Struct(pb::Struct {
            fields: fields
                .iter()
                .map(|field| spark_field(field))
                .collect::<Result<_, _>>()?,
            ..Default::default()
        }),
        DataType::Map(entries, _) => {
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

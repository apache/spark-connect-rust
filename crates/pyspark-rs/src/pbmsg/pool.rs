//! The descriptors of the messages and enums of Spark Connect, of the descriptor set that the
//! proto crate embeds ([`spark_connect_proto::FILE_DESCRIPTOR_SET`]), as the runtime of
//! [`super`] reads them: the fields of each message, in the order of their declaration and by
//! number, its oneofs, and its map fields.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use prost::Message as _;
use prost_types::{DescriptorProto, EnumDescriptorProto, FileDescriptorSet};

/// The type of a field, as `google.protobuf.FieldDescriptorProto.Type` numbers it. Groups, which
/// no Spark Connect proto has, are not supported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Double,
    Float,
    Int64,
    UInt64,
    Int32,
    Fixed64,
    Fixed32,
    Bool,
    String,
    Message,
    Bytes,
    UInt32,
    Enum,
    SFixed32,
    SFixed64,
    SInt32,
    SInt64,
}

impl Kind {
    fn of_proto(value: i32) -> Option<Kind> {
        Some(match value {
            1 => Kind::Double,
            2 => Kind::Float,
            3 => Kind::Int64,
            4 => Kind::UInt64,
            5 => Kind::Int32,
            6 => Kind::Fixed64,
            7 => Kind::Fixed32,
            8 => Kind::Bool,
            9 => Kind::String,
            11 => Kind::Message,
            12 => Kind::Bytes,
            13 => Kind::UInt32,
            14 => Kind::Enum,
            15 => Kind::SFixed32,
            16 => Kind::SFixed64,
            17 => Kind::SInt32,
            18 => Kind::SInt64,
            _ => return None,
        })
    }

    /// The number of the type, as `FieldDescriptor.type` gives it.
    pub(crate) fn proto_type(self) -> i32 {
        match self {
            Kind::Double => 1,
            Kind::Float => 2,
            Kind::Int64 => 3,
            Kind::UInt64 => 4,
            Kind::Int32 => 5,
            Kind::Fixed64 => 6,
            Kind::Fixed32 => 7,
            Kind::Bool => 8,
            Kind::String => 9,
            Kind::Message => 11,
            Kind::Bytes => 12,
            Kind::UInt32 => 13,
            Kind::Enum => 14,
            Kind::SFixed32 => 15,
            Kind::SFixed64 => 16,
            Kind::SInt32 => 17,
            Kind::SInt64 => 18,
        }
    }

    /// The C++ type of the type, as `FieldDescriptor.cpp_type` gives it.
    pub(crate) fn cpp_type(self) -> i32 {
        match self {
            Kind::Int32 | Kind::SInt32 | Kind::SFixed32 => 1,
            Kind::Int64 | Kind::SInt64 | Kind::SFixed64 => 2,
            Kind::UInt32 | Kind::Fixed32 => 3,
            Kind::UInt64 | Kind::Fixed64 => 4,
            Kind::Double => 5,
            Kind::Float => 6,
            Kind::Bool => 7,
            Kind::Enum => 8,
            Kind::String | Kind::Bytes => 9,
            Kind::Message => 10,
        }
    }

    /// The wire type of one value of the type.
    pub(crate) fn wire_type(self) -> u8 {
        match self {
            Kind::Int64
            | Kind::UInt64
            | Kind::Int32
            | Kind::Bool
            | Kind::UInt32
            | Kind::Enum
            | Kind::SInt32
            | Kind::SInt64 => 0,
            Kind::Double | Kind::Fixed64 | Kind::SFixed64 => 1,
            Kind::String | Kind::Bytes | Kind::Message => 2,
            Kind::Float | Kind::Fixed32 | Kind::SFixed32 => 5,
        }
    }

    /// Whether a repeated field of the type may be packed: every scalar type but strings and
    /// bytes.
    pub(crate) fn packable(self) -> bool {
        !matches!(self, Kind::String | Kind::Bytes | Kind::Message)
    }
}

/// The key and value of a map field, of its map entry message.
#[derive(Clone, Debug)]
pub(crate) struct MapDesc {
    /// The full name of the map entry message, whose fields `key` and `value` the errors of a
    /// value of the map name, as protobuf's do.
    pub(crate) entry: String,
    pub(crate) key: Kind,
    pub(crate) value: Kind,
    /// The full name of the message or enum of the value, if it is one.
    pub(crate) value_type: String,
}

/// A field of a message.
#[derive(Clone, Debug)]
pub(crate) struct FieldDesc {
    pub(crate) name: String,
    pub(crate) full_name: String,
    pub(crate) json_name: String,
    pub(crate) number: u32,
    pub(crate) index: usize,
    pub(crate) kind: Kind,
    pub(crate) repeated: bool,
    /// The full name, without the leading dot, of the message or enum of the field.
    pub(crate) type_name: String,
    /// The oneof of the field, real or synthetic (that of a proto3 `optional`).
    pub(crate) oneof: Option<usize>,
    pub(crate) proto3_optional: bool,
    pub(crate) packed: bool,
    /// The key and value of a map field.
    pub(crate) map: Option<MapDesc>,
    /// Whether the field has options, such as `deprecated`, as protobuf's `has_options` tells.
    pub(crate) has_options: bool,
}

impl FieldDesc {
    /// Whether the field tells being set apart from holding the default: a singular message, a
    /// member of a oneof, or a proto3 `optional` (a member of a synthetic oneof).
    pub(crate) fn has_presence(&self) -> bool {
        !self.repeated && (self.kind == Kind::Message || self.oneof.is_some())
    }
}

/// A oneof of a message.
#[derive(Clone, Debug)]
pub(crate) struct OneofDesc {
    pub(crate) name: String,
    pub(crate) full_name: String,
    /// The indices of its fields.
    pub(crate) fields: Vec<usize>,
    /// Whether it is the synthetic oneof of a proto3 `optional` field.
    pub(crate) synthetic: bool,
}

/// A message type.
#[derive(Debug)]
pub(crate) struct MessageDesc {
    pub(crate) name: String,
    pub(crate) full_name: String,
    pub(crate) file: String,
    pub(crate) package: String,
    /// The full name of the message that declares this one, if it is nested.
    pub(crate) containing: Option<String>,
    /// The fields, in the order of their declaration.
    pub(crate) fields: Vec<FieldDesc>,
    pub(crate) by_name: HashMap<String, usize>,
    pub(crate) by_number: HashMap<u32, usize>,
    /// The indices of the fields in the order of their numbers, the order of serialization.
    pub(crate) order: Vec<usize>,
    pub(crate) oneofs: Vec<OneofDesc>,
    pub(crate) oneof_by_name: HashMap<String, usize>,
    pub(crate) map_entry: bool,
    pub(crate) has_options: bool,
    /// The full names of the messages and enums that this message declares.
    pub(crate) nested_messages: Vec<String>,
    pub(crate) nested_enums: Vec<String>,
}

/// An enum type.
#[derive(Debug)]
pub(crate) struct EnumDesc {
    pub(crate) name: String,
    pub(crate) full_name: String,
    pub(crate) file: String,
    pub(crate) containing: Option<String>,
    /// The values, in the order of their declaration.
    pub(crate) values: Vec<(String, i32)>,
    pub(crate) has_options: bool,
}

impl EnumDesc {
    /// The number of the value `name`, which an enum field takes for its value as protobuf's
    /// takes the name of a value.
    pub(crate) fn number_of(&self, name: &str) -> Option<i32> {
        self.values
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, number)| *number)
    }

    pub(crate) fn name_of(&self, number: i32) -> Option<&str> {
        self.values
            .iter()
            .find(|(_, n)| *n == number)
            .map(|(name, _)| name.as_str())
    }
}

/// A file of the descriptor set: its top-level messages and enums.
#[derive(Debug)]
pub(crate) struct FileDesc {
    pub(crate) name: String,
    pub(crate) package: String,
    pub(crate) messages: Vec<String>,
    pub(crate) enums: Vec<String>,
    /// The names of the files that it imports, and of those that it imports publicly.
    pub(crate) dependencies: Vec<String>,
    pub(crate) public_dependencies: Vec<String>,
    /// Its `FileDescriptorProto`, serialized, as protobuf's `FileDescriptor.serialized_pb`.
    pub(crate) serialized: Vec<u8>,
    pub(crate) has_options: bool,
}

/// Every message and enum of the descriptor set, by full name.
#[derive(Debug)]
pub(crate) struct Pool {
    pub(crate) messages: HashMap<String, Arc<MessageDesc>>,
    pub(crate) enums: HashMap<String, Arc<EnumDesc>>,
    pub(crate) files: Vec<FileDesc>,
}

/// The pool of [`spark_connect_proto::FILE_DESCRIPTOR_SET`].
pub(crate) fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        Pool::build(spark_connect_proto::FILE_DESCRIPTOR_SET)
            .expect("the embedded descriptor set of the Spark Connect protos is valid")
    })
}

impl Pool {
    fn build(bytes: &[u8]) -> Result<Pool, String> {
        let set = FileDescriptorSet::decode(bytes).map_err(|e| e.to_string())?;
        let mut messages: Vec<MessageDesc> = Vec::new();
        let mut enums: Vec<EnumDesc> = Vec::new();
        let mut files = Vec::new();
        for file in &set.file {
            let name = file.name().to_string();
            let package = file.package().to_string();
            let prefix = if package.is_empty() {
                String::new()
            } else {
                format!("{package}.")
            };
            let mut top_messages = Vec::new();
            let mut top_enums = Vec::new();
            for message in &file.message_type {
                top_messages.push(format!("{prefix}{}", message.name()));
                add_message(
                    &mut messages,
                    &mut enums,
                    &name,
                    &package,
                    &prefix,
                    None,
                    message,
                )?;
            }
            for enumeration in &file.enum_type {
                top_enums.push(format!("{prefix}{}", enumeration.name()));
                add_enum(&mut enums, &name, &prefix, None, enumeration);
            }
            let public_dependencies = file
                .public_dependency
                .iter()
                .filter_map(|&i| {
                    usize::try_from(i)
                        .ok()
                        .and_then(|i| file.dependency.get(i))
                        .cloned()
                })
                .collect();
            files.push(FileDesc {
                name,
                package,
                messages: top_messages,
                enums: top_enums,
                dependencies: file.dependency.clone(),
                public_dependencies,
                serialized: file.encode_to_vec(),
                has_options: file.options.is_some(),
            });
        }

        // The map fields: repeated fields of a map entry message.
        let entries: HashMap<String, MapDesc> = messages
            .iter()
            .filter(|m| m.map_entry)
            .map(|m| {
                let field = |number: u32| m.by_number.get(&number).map(|&i| &m.fields[i]);
                let (key, value) = (field(1), field(2));
                let map = MapDesc {
                    entry: m.full_name.clone(),
                    key: key.map_or(Kind::String, |f| f.kind),
                    value: value.map_or(Kind::String, |f| f.kind),
                    value_type: value.map_or(String::new(), |f| f.type_name.clone()),
                };
                (m.full_name.clone(), map)
            })
            .collect();
        for message in &mut messages {
            for field in &mut message.fields {
                if field.repeated && field.kind == Kind::Message {
                    field.map = entries.get(&field.type_name).cloned();
                }
            }
        }

        Ok(Pool {
            messages: messages
                .into_iter()
                .map(|m| (m.full_name.clone(), Arc::new(m)))
                .collect(),
            enums: enums
                .into_iter()
                .map(|e| (e.full_name.clone(), Arc::new(e)))
                .collect(),
            files,
        })
    }

    pub(crate) fn message(&self, full_name: &str) -> Option<&Arc<MessageDesc>> {
        self.messages.get(full_name)
    }

    pub(crate) fn enumeration(&self, full_name: &str) -> Option<&Arc<EnumDesc>> {
        self.enums.get(full_name)
    }

    /// The file `name`, such as `spark/connect/base.proto`.
    pub(crate) fn file(&self, name: &str) -> Option<&FileDesc> {
        self.files.iter().find(|f| f.name == name)
    }
}

fn add_message(
    messages: &mut Vec<MessageDesc>,
    enums: &mut Vec<EnumDesc>,
    file: &str,
    package: &str,
    prefix: &str,
    containing: Option<&str>,
    proto: &DescriptorProto,
) -> Result<(), String> {
    let full_name = format!("{prefix}{}", proto.name());
    let nested_prefix = format!("{full_name}.");

    let mut oneofs: Vec<OneofDesc> = proto
        .oneof_decl
        .iter()
        .map(|o| OneofDesc {
            name: o.name().to_string(),
            full_name: format!("{full_name}.{}", o.name()),
            fields: Vec::new(),
            synthetic: false,
        })
        .collect();

    let mut fields = Vec::with_capacity(proto.field.len());
    for (index, field) in proto.field.iter().enumerate() {
        let kind = Kind::of_proto(field.r#type.unwrap_or_default()).ok_or_else(|| {
            format!(
                "the field {full_name}.{} has the unsupported type {}",
                field.name(),
                field.r#type.unwrap_or_default()
            )
        })?;
        let repeated = field.label == Some(3);
        let oneof = field.oneof_index.map(|i| i as usize);
        if let Some(o) = oneof {
            oneofs
                .get_mut(o)
                .ok_or_else(|| format!("the field {full_name}.{} has no oneof", field.name()))?
                .fields
                .push(index);
        }
        let proto3_optional = field.proto3_optional.unwrap_or(false);
        let packed = repeated
            && kind.packable()
            && field
                .options
                .as_ref()
                .and_then(|o| o.packed)
                .unwrap_or(true);
        fields.push(FieldDesc {
            name: field.name().to_string(),
            full_name: format!("{full_name}.{}", field.name()),
            json_name: field
                .json_name
                .clone()
                .unwrap_or_else(|| field.name().to_string()),
            number: field.number() as u32,
            index,
            kind,
            repeated,
            type_name: field.type_name().trim_start_matches('.').to_string(),
            oneof,
            proto3_optional,
            packed,
            map: None,
            has_options: field.options.is_some(),
        });
    }
    for oneof in &mut oneofs {
        oneof.synthetic = oneof.fields.len() == 1 && fields[oneof.fields[0]].proto3_optional;
    }

    let mut order: Vec<usize> = (0..fields.len()).collect();
    order.sort_by_key(|&i| fields[i].number);

    let nested_messages = proto
        .nested_type
        .iter()
        .map(|n| format!("{nested_prefix}{}", n.name()))
        .collect();
    let nested_enums = proto
        .enum_type
        .iter()
        .map(|e| format!("{nested_prefix}{}", e.name()))
        .collect();

    messages.push(MessageDesc {
        name: proto.name().to_string(),
        full_name: full_name.clone(),
        file: file.to_string(),
        package: package.to_string(),
        containing: containing.map(str::to_string),
        by_name: fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name.clone(), i))
            .collect(),
        by_number: fields
            .iter()
            .enumerate()
            .map(|(i, f)| (f.number, i))
            .collect(),
        order,
        oneof_by_name: oneofs
            .iter()
            .enumerate()
            .map(|(i, o)| (o.name.clone(), i))
            .collect(),
        oneofs,
        fields,
        has_options: proto.options.is_some(),
        map_entry: proto
            .options
            .as_ref()
            .and_then(|o| o.map_entry)
            .unwrap_or(false),
        nested_messages,
        nested_enums,
    });

    for nested in &proto.nested_type {
        add_message(
            messages,
            enums,
            file,
            package,
            &nested_prefix,
            Some(&full_name),
            nested,
        )?;
    }
    for enumeration in &proto.enum_type {
        add_enum(enums, file, &nested_prefix, Some(&full_name), enumeration);
    }
    Ok(())
}

fn add_enum(
    enums: &mut Vec<EnumDesc>,
    file: &str,
    prefix: &str,
    containing: Option<&str>,
    proto: &EnumDescriptorProto,
) {
    enums.push(EnumDesc {
        name: proto.name().to_string(),
        full_name: format!("{prefix}{}", proto.name()),
        file: file.to_string(),
        containing: containing.map(str::to_string),
        values: proto
            .value
            .iter()
            .map(|v| (v.name().to_string(), v.number()))
            .collect(),
        has_options: proto.options.is_some(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_embedded_set() {
        let pool = pool();
        let relation = pool.message("spark.connect.Relation").expect("Relation");
        assert!(relation.oneof_by_name.contains_key("rel_type"));
        assert!(pool.message("google.protobuf.Any").is_some());
        let options = pool
            .message("spark.connect.Read.DataSource")
            .expect("Read.DataSource");
        let field = &options.fields[options.by_name["options"]];
        let map = field.map.as_ref().expect("options is a map");
        assert_eq!((map.key, map.value), (Kind::String, Kind::String));
    }

    #[test]
    fn marks_proto3_optional_oneofs_synthetic() {
        let pool = pool();
        let found = pool.messages.values().any(|m| {
            m.oneofs
                .iter()
                .any(|o| o.synthetic && m.fields[o.fields[0]].proto3_optional)
        });
        assert!(
            found,
            "the Spark Connect protos have proto3 optional fields"
        );
    }
}

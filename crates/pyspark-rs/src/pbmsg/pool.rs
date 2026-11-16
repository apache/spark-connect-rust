//! The descriptors of the messages, enums and services of Spark Connect, of the descriptor set
//! that the proto crate embeds ([`spark_connect_proto::FILE_DESCRIPTOR_SET`]), as the runtime of
//! [`super`] reads them: the fields of each message, in the order of their declaration and by
//! number, its oneofs, and its map fields.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use prost::Message as _;
use prost_types::{
    DescriptorProto, EnumDescriptorProto, FileDescriptorProto, FileDescriptorSet,
    ServiceDescriptorProto,
};

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
    pub(crate) has_options: bool,
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
    /// Whether each value has options, in the order of the values.
    pub(crate) value_has_options: Vec<bool>,
    pub(crate) has_options: bool,
    /// Whether the enum is closed, as an enum of a proto2 file is: a parsed value that it does not
    /// declare goes to the unknown fields.
    pub(crate) closed: bool,
}

/// A method of a service.
#[derive(Debug)]
pub(crate) struct MethodDesc {
    pub(crate) name: String,
    pub(crate) full_name: String,
    /// The full names, without the leading dot, of its request and response messages.
    pub(crate) input_type: String,
    pub(crate) output_type: String,
    pub(crate) client_streaming: bool,
    pub(crate) server_streaming: bool,
    pub(crate) has_options: bool,
}

/// A service, such as `spark.connect.SparkConnectService`.
#[derive(Debug)]
pub(crate) struct ServiceDesc {
    pub(crate) name: String,
    pub(crate) full_name: String,
    pub(crate) file: String,
    /// Its index among the services of its file.
    pub(crate) index: usize,
    /// The methods, in the order of their declaration.
    pub(crate) methods: Vec<MethodDesc>,
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

/// A file of the descriptor set: its top-level messages, enums and services.
#[derive(Debug)]
pub(crate) struct FileDesc {
    pub(crate) name: String,
    pub(crate) package: String,
    pub(crate) messages: Vec<String>,
    pub(crate) enums: Vec<String>,
    pub(crate) services: Vec<String>,
    /// The names of the files that it imports, and of those that it imports publicly.
    pub(crate) dependencies: Vec<String>,
    pub(crate) public_dependencies: Vec<String>,
    /// Its `FileDescriptorProto`, serialized, as protobuf's `FileDescriptor.serialized_pb`.
    pub(crate) serialized: Vec<u8>,
    pub(crate) has_options: bool,
}

/// Every message, enum and service of the descriptor set, by full name.
#[derive(Debug)]
pub(crate) struct Pool {
    pub(crate) messages: HashMap<String, Arc<MessageDesc>>,
    pub(crate) enums: HashMap<String, Arc<EnumDesc>>,
    pub(crate) services: HashMap<String, Arc<ServiceDesc>>,
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
        let mut services: Vec<ServiceDesc> = Vec::new();
        let mut files = Vec::new();
        for file in &set.file {
            let name = file.name().to_string();
            let package = file.package().to_string();
            let prefix = if package.is_empty() {
                String::new()
            } else {
                format!("{package}.")
            };
            // The enums of a proto2 file (whose syntax protoc leaves unset) are closed; those of
            // proto3 and of editions (of their default features) are open.
            let scope = FileScope {
                name: &name,
                package: &package,
                closed_enums: matches!(file.syntax(), "" | "proto2"),
            };
            let mut top_messages = Vec::new();
            let mut top_enums = Vec::new();
            for message in &file.message_type {
                top_messages.push(format!("{prefix}{}", message.name()));
                add_message(&mut messages, &mut enums, &scope, &prefix, None, message)?;
            }
            for enumeration in &file.enum_type {
                top_enums.push(format!("{prefix}{}", enumeration.name()));
                add_enum(&mut enums, &scope, &prefix, None, enumeration);
            }
            let mut top_services = Vec::new();
            for (index, service) in file.service.iter().enumerate() {
                let full_name = format!("{prefix}{}", service.name());
                top_services.push(full_name.clone());
                services.push(service_desc(&name, &full_name, index, service));
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
                services: top_services,
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
            services: services
                .into_iter()
                .map(|s| (s.full_name.clone(), Arc::new(s)))
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

    pub(crate) fn service(&self, full_name: &str) -> Option<&Arc<ServiceDesc>> {
        self.services.get(full_name)
    }

    /// The file `name`, such as `spark/connect/base.proto`.
    pub(crate) fn file(&self, name: &str) -> Option<&FileDesc> {
        self.files.iter().find(|f| f.name == name)
    }

    /// The `FileDescriptorProto` of the file `name`, as the descriptor set holds it.
    pub(crate) fn file_proto(&self, name: &str) -> Option<FileDescriptorProto> {
        FileDescriptorProto::decode(self.file(name)?.serialized.as_slice()).ok()
    }

    /// The `DescriptorProto` of the message type `desc`, as its file declares it.
    pub(crate) fn message_proto(&self, desc: &MessageDesc) -> Option<DescriptorProto> {
        let file = self.file_proto(&desc.file)?;
        let mut path = scope_path(&desc.full_name, &desc.package).into_iter();
        let first = path.next()?;
        let mut message = file.message_type.into_iter().find(|m| m.name() == first)?;
        for name in path {
            message = message.nested_type.into_iter().find(|m| m.name() == name)?;
        }
        Some(message)
    }

    /// The `EnumDescriptorProto` of the enum type `desc`, as its file declares it.
    pub(crate) fn enum_proto(&self, desc: &EnumDesc) -> Option<EnumDescriptorProto> {
        let enums = match &desc.containing {
            Some(containing) => self.message_proto(self.message(containing)?)?.enum_type,
            None => self.file_proto(&desc.file)?.enum_type,
        };
        enums.into_iter().find(|e| e.name() == desc.name)
    }

    /// The `ServiceDescriptorProto` of the service `desc`, as its file declares it.
    pub(crate) fn service_proto(&self, desc: &ServiceDesc) -> Option<ServiceDescriptorProto> {
        self.file_proto(&desc.file)?
            .service
            .into_iter()
            .nth(desc.index)
    }
}

/// The names of the scopes of `full_name` below its package: `["A", "B"]` of `spark.connect.A.B`
/// of the package `spark.connect`.
fn scope_path<'a>(full_name: &'a str, package: &str) -> Vec<&'a str> {
    let relative = if package.is_empty() {
        full_name
    } else {
        full_name
            .strip_prefix(package)
            .and_then(|rest| rest.strip_prefix('.'))
            .unwrap_or(full_name)
    };
    relative.split('.').collect()
}

/// The file of the messages and enums that [`add_message`] and [`add_enum`] add.
struct FileScope<'a> {
    name: &'a str,
    package: &'a str,
    closed_enums: bool,
}

fn service_desc(
    file: &str,
    full_name: &str,
    index: usize,
    proto: &ServiceDescriptorProto,
) -> ServiceDesc {
    let methods = proto
        .method
        .iter()
        .map(|m| MethodDesc {
            name: m.name().to_string(),
            full_name: format!("{full_name}.{}", m.name()),
            input_type: m.input_type().trim_start_matches('.').to_string(),
            output_type: m.output_type().trim_start_matches('.').to_string(),
            client_streaming: m.client_streaming(),
            server_streaming: m.server_streaming(),
            has_options: m.options.is_some(),
        })
        .collect();
    ServiceDesc {
        name: proto.name().to_string(),
        full_name: full_name.to_string(),
        file: file.to_string(),
        index,
        methods,
        has_options: proto.options.is_some(),
    }
}

fn add_message(
    messages: &mut Vec<MessageDesc>,
    enums: &mut Vec<EnumDesc>,
    scope: &FileScope<'_>,
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
            has_options: o.options.is_some(),
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
        file: scope.name.to_string(),
        package: scope.package.to_string(),
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
            scope,
            &nested_prefix,
            Some(&full_name),
            nested,
        )?;
    }
    for enumeration in &proto.enum_type {
        add_enum(enums, scope, &nested_prefix, Some(&full_name), enumeration);
    }
    Ok(())
}

fn add_enum(
    enums: &mut Vec<EnumDesc>,
    scope: &FileScope<'_>,
    prefix: &str,
    containing: Option<&str>,
    proto: &EnumDescriptorProto,
) {
    enums.push(EnumDesc {
        name: proto.name().to_string(),
        full_name: format!("{prefix}{}", proto.name()),
        file: scope.name.to_string(),
        containing: containing.map(str::to_string),
        values: proto
            .value
            .iter()
            .map(|v| (v.name().to_string(), v.number()))
            .collect(),
        value_has_options: proto.value.iter().map(|v| v.options.is_some()).collect(),
        has_options: proto.options.is_some(),
        closed: scope.closed_enums,
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

    #[test]
    fn reads_the_services_and_the_protos_of_the_declarations() {
        let pool = pool();
        let service = pool
            .service("spark.connect.SparkConnectService")
            .expect("SparkConnectService");
        let execute = &service.methods[0];
        assert_eq!(
            (execute.full_name.as_str(), execute.input_type.as_str()),
            (
                "spark.connect.SparkConnectService.ExecutePlan",
                "spark.connect.ExecutePlanRequest"
            )
        );
        assert!(execute.server_streaming && !execute.client_streaming);
        let base = pool.file("spark/connect/base.proto").expect("base.proto");
        assert_eq!(base.services, ["spark.connect.SparkConnectService"]);
        let proto = pool.service_proto(service).expect("its proto");
        assert_eq!(proto.method.len(), service.methods.len());

        let nested = pool
            .message("spark.connect.Read.DataSource")
            .expect("Read.DataSource");
        assert_eq!(
            pool.message_proto(nested).expect("its proto").name(),
            "DataSource"
        );
        let join_type = pool
            .enumeration("spark.connect.Join.JoinType")
            .expect("Join.JoinType");
        let proto = pool.enum_proto(join_type).expect("its proto");
        assert_eq!(proto.value.len(), join_type.values.len());
        // The enums of the proto3 files of Spark Connect are open.
        assert!(pool.enums.values().all(|e| !e.closed));
        assert!(pool
            .message("spark.connect.ExamplePluginRelation")
            .is_some());
    }
}

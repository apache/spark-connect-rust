//! The descriptors of the messages, enums and services of Spark Connect, of the descriptor set
//! that the proto crate embeds ([`spark_connect_proto::FILE_DESCRIPTOR_SET`]), as the runtime of
//! [`super`] reads them: the fields of each message, in the order of their declaration and by
//! number, its oneofs, and its map fields.
//!
//! The pool also holds the files that [`register`] adds at runtime: those of other protos that
//! import the protos of Spark Connect, such as a plugin's, whose modules protoc generates for
//! `protobuf` (see `pyspark.sql.connect.proto._rust`), and the files that they import. A lookup of
//! a name that the embedded set does not declare finds it in those.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock, RwLock, RwLockReadGuard};

use pyo3::types::PyType;
use pyo3::Py;

use prost::Message as _;

use super::scalar::Scalar;
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
    /// The full name of the map entry message.
    pub(crate) entry: String,
    /// The full names of the fields `key` and `value` of the entry message, which the errors of a
    /// key and of a value of the map name, as protobuf's do (made once, not for each key).
    pub(crate) key_name: String,
    pub(crate) value_name: String,
    pub(crate) key: Kind,
    pub(crate) value: Kind,
    /// The full name of the message or enum of the value, if it is one.
    pub(crate) value_type: String,
    /// Whether a parse checks the UTF-8 of a string key and of a string value, of the fields of
    /// the entry message (see [`FieldDesc::validate_utf8`]): an entry of proto2 takes any bytes.
    pub(crate) key_validate_utf8: bool,
    pub(crate) value_validate_utf8: bool,
    /// The values of a closed enum of the values (see [`FieldDesc::closed_values`]).
    pub(crate) value_closed: Option<Arc<[i32]>>,
    /// Whether the key and the value fields of the entry message have presence (of proto2, or of
    /// the `EXPLICIT` presence of editions), so that an entry encoded again writes one of the
    /// default that it read.
    pub(crate) key_presence: bool,
    pub(crate) value_presence: bool,
    /// The message type of a value of a message, once a parse looked it up (see
    /// [`MapDesc::value_message`]).
    pub(crate) value_message: OnceLock<Arc<MessageDesc>>,
}

impl MapDesc {
    /// The message type of the values of a map of messages, of the pool, looked up once it is
    /// found (see [`FieldDesc::message_type`]).
    pub(crate) fn value_message(&self) -> Option<&Arc<MessageDesc>> {
        if let Some(desc) = self.value_message.get() {
            return Some(desc);
        }
        let desc = pool().message(&self.value_type)?;
        Some(self.value_message.get_or_init(|| desc.clone()))
    }
}

/// Whether a field tells being set apart from holding its default, as protobuf's presence of a
/// field: of a proto3 scalar (implicit), of a singular field of proto2, a message field, a member
/// of a oneof or a proto3 `optional` (explicit), or of a `required` field of proto2 (or of the
/// `LEGACY_REQUIRED` presence of editions), which a message must set to be initialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Presence {
    Implicit,
    Explicit,
    Required,
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
    /// The key and value of a map field, which its maps share.
    pub(crate) map: Option<Arc<MapDesc>>,
    /// Whether the field has options, such as `deprecated`, as protobuf's `has_options` tells.
    pub(crate) has_options: bool,
    /// The presence of a singular field.
    pub(crate) presence: Presence,
    /// The value of an unset scalar field, of its `[default = ...]` (of proto2 and editions), else
    /// of the first value of the enum of an enum field of explicit presence, else none (the zero
    /// of its type).
    pub(crate) default: Option<Scalar>,
    /// Whether a message field is encoded as a group (a group of proto2, or a message field of the
    /// `DELIMITED` encoding of editions), between a start and an end tag.
    pub(crate) delimited: bool,
    /// Whether a parse checks the UTF-8 of a string field, as proto3 and editions do (of the
    /// `VERIFY` validation); one of proto2 (or of `NONE`) takes any bytes.
    pub(crate) validate_utf8: bool,
    /// The values that a closed enum of an enum field declares, in order (of an enum of proto2, or
    /// of the `CLOSED` enum type of editions), which a parse looks a value up in: another value
    /// goes to the unknown fields. None of an open enum, which takes any value, and of a field of
    /// another kind.
    pub(crate) closed_values: Option<Arc<[i32]>>,
    /// The message type of a message field, once a parse looked it up (see
    /// [`FieldDesc::message_type`]).
    pub(crate) message_type: OnceLock<Arc<MessageDesc>>,
}

impl FieldDesc {
    /// The message type of the message field, of the pool, looked up once it is found (a type of
    /// a file that `register` adds later is looked up again until then), so that a parse of many
    /// messages of the field does not hash its full name for each.
    pub(crate) fn message_type(&self) -> Option<&Arc<MessageDesc>> {
        if let Some(desc) = self.message_type.get() {
            return Some(desc);
        }
        let desc = pool().message(&self.type_name)?;
        Some(self.message_type.get_or_init(|| desc.clone()))
    }

    /// Whether the field tells being set apart from holding the default: a singular field of
    /// explicit or required presence (see [`Presence`]).
    pub(crate) fn has_presence(&self) -> bool {
        !self.repeated && self.presence != Presence::Implicit
    }

    /// Whether the field is required, of proto2's `required` (or editions' `LEGACY_REQUIRED`).
    pub(crate) fn is_required(&self) -> bool {
        !self.repeated && self.presence == Presence::Required
    }

    /// The wire type of a value of the field: 3 of a group, else that of its kind.
    pub(crate) fn wire_type(&self) -> u8 {
        if self.delimited {
            3
        } else {
            self.kind.wire_type()
        }
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
    /// The indices of the required fields, in the order of their declaration.
    pub(crate) required: Vec<usize>,
    /// Whether a message of the type can lack a required field, of the types that it can hold
    /// too, which the first check of a message of the type finds (see `may_lack_required` of
    /// `message.rs`).
    pub(crate) may_lack_required: OnceLock<bool>,
    /// The class registered of the type (none, of a type that has none) at the first read of it
    /// of the classes of the runtime, and the generation of the classes then, which the messages
    /// of the type are made of, of any generation that the class of the type is the same at (see
    /// `kept_class` of `message.rs`).
    pub(crate) class: OnceLock<(u64, Option<Py<PyType>>)>,
    /// The generation of the classes that the last registration that gave the type another class
    /// began (none, of a type of the class it had at first), which a registration sets before it
    /// begins the generation.
    pub(crate) class_changed: AtomicU64,
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

/// The values of an enum of the pool that a file that [`register`] adds may use, by its full name,
/// and whether it is closed.
type KnownEnum<'a> = dyn Fn(&str) -> Option<(Vec<(String, i32)>, bool)> + 'a;

/// The descriptors of the files of a set, as [`Pool::build_files`] reads them.
struct Built {
    messages: Vec<MessageDesc>,
    enums: Vec<EnumDesc>,
    services: Vec<ServiceDesc>,
    files: Vec<FileDesc>,
}

impl Pool {
    fn build(bytes: &[u8]) -> Result<Pool, String> {
        let set = FileDescriptorSet::decode(bytes).map_err(|e| e.to_string())?;
        let serialized: Vec<Vec<u8>> = set.file.iter().map(|f| f.encode_to_vec()).collect();
        let Built {
            messages,
            enums,
            services,
            files,
        } = Self::build_files(&set.file, &serialized, &|_| None)?;
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

    /// The descriptors of the messages, enums and services of `protos`, each of the bytes of its
    /// file in `serialized`, and of the files, with the map fields of the messages (a map entry
    /// message is nested in the message of its map field, in the same file) and the defaults of
    /// their enum fields, of the features of their files (see [`FileView`]).
    fn build_files(
        protos: &[FileDescriptorProto],
        serialized: &[Vec<u8>],
        known_enum: &KnownEnum<'_>,
    ) -> Result<Built, String> {
        let mut messages: Vec<MessageDesc> = Vec::new();
        let mut enums: Vec<EnumDesc> = Vec::new();
        let mut services: Vec<ServiceDesc> = Vec::new();
        let mut files = Vec::new();
        for (file, bytes) in protos.iter().zip(serialized) {
            let name = file.name().to_string();
            let package = file.package().to_string();
            let prefix = if package.is_empty() {
                String::new()
            } else {
                format!("{package}.")
            };
            let view = FileView::decode(bytes.as_slice())
                .map_err(|e| format!("{name} is not a FileDescriptorProto: {e}"))?;
            let syntax = Syntax::of(file.syntax(), view.edition).ok_or_else(|| {
                format!("{name} is of a syntax or edition that the runtime does not read")
            })?;
            let features = syntax
                .defaults()
                .with(view.options.as_ref().and_then(|o| o.features.as_ref()));
            let scope = FileScope {
                name: &name,
                package: &package,
                syntax,
            };
            let mut top_messages = Vec::new();
            let mut top_enums = Vec::new();
            for (index, message) in file.message_type.iter().enumerate() {
                top_messages.push(format!("{prefix}{}", message.name()));
                let view = view.message_type.get(index).cloned().unwrap_or_default();
                add_message(
                    &mut messages,
                    &mut enums,
                    &scope,
                    (&prefix, None, features),
                    message,
                    &view,
                )?;
            }
            for (index, enumeration) in file.enum_type.iter().enumerate() {
                top_enums.push(format!("{prefix}{}", enumeration.name()));
                let view = view.enum_type.get(index).cloned().unwrap_or_default();
                add_enum(
                    &mut enums,
                    &scope,
                    (&prefix, None, features),
                    enumeration,
                    &view,
                );
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
                serialized: bytes.clone(),
                has_options: file.options.is_some(),
            });
        }

        // The values of the enums of these files or of the pool, and whether each is closed.
        let declared_enums: HashMap<&str, &EnumDesc> =
            enums.iter().map(|e| (e.full_name.as_str(), e)).collect();
        let enum_of = |name: &str| -> Option<(Vec<(String, i32)>, bool)> {
            declared_enums
                .get(name)
                .map(|e| (e.values.clone(), e.closed))
                .or_else(|| known_enum(name))
        };
        // The values of the closed enum of each enum field, which a parse looks its values up in.
        for message in &mut messages {
            for field in &mut message.fields {
                if field.kind != Kind::Enum {
                    continue;
                }
                if let Some((values, true)) = enum_of(&field.type_name) {
                    let mut numbers: Vec<i32> = values.iter().map(|(_, n)| *n).collect();
                    numbers.sort_unstable();
                    numbers.dedup();
                    field.closed_values = Some(numbers.into());
                }
            }
        }

        // The map fields: repeated fields of a map entry message.
        let entries: HashMap<String, Arc<MapDesc>> = messages
            .iter()
            .filter(|m| m.map_entry)
            .map(|m| {
                let field = |number: u32| m.by_number.get(&number).map(|&i| &m.fields[i]);
                let (key, value) = (field(1), field(2));
                let map = MapDesc {
                    entry: m.full_name.clone(),
                    key_name: format!("{}.key", m.full_name),
                    value_name: format!("{}.value", m.full_name),
                    key: key.map_or(Kind::String, |f| f.kind),
                    value: value.map_or(Kind::String, |f| f.kind),
                    value_type: value.map_or(String::new(), |f| f.type_name.clone()),
                    key_validate_utf8: key.is_none_or(|f| f.validate_utf8),
                    value_validate_utf8: value.is_none_or(|f| f.validate_utf8),
                    value_closed: value.and_then(|f| f.closed_values.clone()),
                    key_presence: key.is_some_and(FieldDesc::has_presence),
                    value_presence: value.is_some_and(FieldDesc::has_presence),
                    value_message: OnceLock::new(),
                };
                (m.full_name.clone(), Arc::new(map))
            })
            .collect();
        for message in &mut messages {
            for field in &mut message.fields {
                if field.repeated && field.kind == Kind::Message {
                    field.map = entries.get(&field.type_name).cloned();
                }
            }
        }

        // The defaults of the enum fields, of the enums of these files or of the pool: of the name
        // of a value of `[default = ...]`, else, of a field of presence, the first value.
        for message in &mut messages {
            for field in &mut message.fields {
                if field.kind != Kind::Enum || field.repeated {
                    continue;
                }
                let Some((values, _)) = enum_of(&field.type_name) else {
                    continue;
                };
                field.default = match field.default.take() {
                    Some(Scalar::Str(name)) => {
                        let number = values
                            .iter()
                            .find(|(n, _)| *n == name)
                            .map(|(_, number)| *number)
                            .ok_or_else(|| {
                                format!("the default {name} of {} is no value", field.full_name)
                            })?;
                        Some(Scalar::Enum(number))
                    }
                    Some(other) => Some(other),
                    None if field.presence != Presence::Implicit => {
                        values.first().map(|(_, number)| Scalar::Enum(*number))
                    }
                    None => None,
                };
            }
        }

        Ok(Built {
            messages,
            enums,
            services,
            files,
        })
    }

    /// The message type `full_name`, of the descriptor set or of a file that [`register`] added.
    pub(crate) fn message(&'static self, full_name: &str) -> Option<&'static Arc<MessageDesc>> {
        self.messages
            .get(full_name)
            .or_else(|| registered().messages.get(full_name).copied())
    }

    /// The enum type `full_name`, of the descriptor set or of a file that [`register`] added.
    pub(crate) fn enumeration(&'static self, full_name: &str) -> Option<&'static Arc<EnumDesc>> {
        self.enums
            .get(full_name)
            .or_else(|| registered().enums.get(full_name).copied())
    }

    /// The service `full_name`, of the descriptor set or of a file that [`register`] added.
    pub(crate) fn service(&'static self, full_name: &str) -> Option<&'static Arc<ServiceDesc>> {
        self.services
            .get(full_name)
            .or_else(|| registered().services.get(full_name).copied())
    }

    /// The file `name`, such as `spark/connect/base.proto`, of the descriptor set or that
    /// [`register`] added.
    pub(crate) fn file(&'static self, name: &str) -> Option<&'static FileDesc> {
        self.files
            .iter()
            .find(|f| f.name == name)
            .or_else(|| registered().file(name))
    }

    /// Every file of the pool: those of the embedded set, then those that [`register`] added, in
    /// the order of their addition.
    pub(crate) fn all_files(&'static self) -> Vec<&'static FileDesc> {
        self.files
            .iter()
            .chain(registered().files.iter().copied())
            .collect()
    }

    /// The `FileDescriptorProto` of the file `name`, as the descriptor set holds it.
    pub(crate) fn file_proto(&'static self, name: &str) -> Option<FileDescriptorProto> {
        FileDescriptorProto::decode(self.file(name)?.serialized.as_slice()).ok()
    }

    /// The `DescriptorProto` of the message type `desc`, as its file declares it.
    pub(crate) fn message_proto(&'static self, desc: &MessageDesc) -> Option<DescriptorProto> {
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
    pub(crate) fn enum_proto(&'static self, desc: &EnumDesc) -> Option<EnumDescriptorProto> {
        let enums = match &desc.containing {
            Some(containing) => self.message_proto(self.message(containing)?)?.enum_type,
            None => self.file_proto(&desc.file)?.enum_type,
        };
        enums.into_iter().find(|e| e.name() == desc.name)
    }

    /// The `ServiceDescriptorProto` of the service `desc`, as its file declares it.
    pub(crate) fn service_proto(
        &'static self,
        desc: &ServiceDesc,
    ) -> Option<ServiceDescriptorProto> {
        self.file_proto(&desc.file)?
            .service
            .into_iter()
            .nth(desc.index)
    }
}

/// The files that [`register`] added to those of the embedded set, and their messages, enums and
/// services by full name. As protobuf's pool, the registry never removes a file, so the
/// descriptors of a file, leaked once when the file is added, live as long as the process, as
/// those of the embedded set do.
#[derive(Default)]
struct Registry {
    files: Vec<&'static FileDesc>,
    messages: HashMap<String, &'static Arc<MessageDesc>>,
    enums: HashMap<String, &'static Arc<EnumDesc>>,
    services: HashMap<String, &'static Arc<ServiceDesc>>,
}

impl Registry {
    fn file(&self, name: &str) -> Option<&'static FileDesc> {
        self.files.iter().find(|f| f.name == name).copied()
    }
}

fn registry() -> &'static RwLock<Registry> {
    static REGISTRY: OnceLock<RwLock<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Registry::default()))
}

/// The registry, to read it. A lookup holds it only while it reads a name, and never takes it
/// again meanwhile, so a reader does not wait for itself behind a writer that waits.
fn registered() -> RwLockReadGuard<'static, Registry> {
    registry().read().unwrap_or_else(|e| e.into_inner())
}

/// Add `serialized`, files as their serialized `FileDescriptorProto`s (protobuf's
/// `FileDescriptor.serialized_pb`), to the pool, as protobuf's `AddSerializedFile` adds a file to
/// its pool: the runtime then finds their messages, enums and services by name, as those of the
/// embedded set, and makes and reads their messages. The files are those of a module that protoc
/// generates of a proto that imports the protos of Spark Connect, and those that it imports, of
/// protobuf's default pool, which holds one file of a name: so a file of a name that the pool has
/// is left as it is, and adding the files of a module again adds nothing.
///
/// The files are added together or not at all: each file that the pool does not have must be of
/// proto2, proto3 or the editions 2023 and 2024 (whose presence, defaults, packing, closed enums,
/// groups and UTF-8 validation the runtime reads of their syntax and features), declare no
/// extensions, extension ranges or message sets, import files that the pool has or that
/// `serialized` adds, declare no name that another file declares, and declare fields of messages
/// and enums that those files declare. The pool keeps the bytes of each file as they are given
/// (which prost-types, of no features of editions, would not encode again). Gives each file of
/// `serialized`, as the pool has it, in their order.
pub(crate) fn register(serialized: &[Vec<u8>]) -> Result<Vec<&'static FileDesc>, String> {
    let protos = serialized
        .iter()
        .map(|bytes| {
            FileDescriptorProto::decode(bytes.as_slice())
                .map_err(|e| format!("a file that is not a FileDescriptorProto: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let pool = pool();
    let mut registry = registry().write().unwrap_or_else(|e| e.into_inner());
    let has_file = |registry: &Registry, name: &str| {
        pool.files.iter().any(|f| f.name == name) || registry.file(name).is_some()
    };

    let mut names = HashSet::new();
    let added: Vec<&FileDescriptorProto> = protos
        .iter()
        .filter(|p| !has_file(&registry, p.name()) && names.insert(p.name()))
        .collect();
    for file in &added {
        if let Some(what) = unheld_declaration(file) {
            return Err(format!(
                "{} declares {what}, which the runtime does not hold",
                file.name()
            ));
        }
        if let Some(missing) = file
            .dependency
            .iter()
            .find(|d| !has_file(&registry, d) && !names.contains(d.as_str()))
        {
            return Err(format!(
                "{} imports {missing}, which is not in the pool",
                file.name()
            ));
        }
    }
    let added_bytes: Vec<Vec<u8>> = added
        .iter()
        .map(|file| {
            let at = protos
                .iter()
                .position(|p| std::ptr::eq(p, *file))
                .expect("a file of the protos");
            serialized[at].clone()
        })
        .collect();
    let known_enum = |name: &str| {
        pool.enums
            .get(name)
            .or_else(|| registry.enums.get(name).copied())
            .map(|e| (e.values.clone(), e.closed))
    };
    let built = Pool::build_files(
        &added.into_iter().cloned().collect::<Vec<_>>(),
        &added_bytes,
        &known_enum,
    )?;

    let taken = |name: &str| {
        pool.messages.contains_key(name)
            || pool.enums.contains_key(name)
            || pool.services.contains_key(name)
            || registry.messages.contains_key(name)
            || registry.enums.contains_key(name)
            || registry.services.contains_key(name)
    };
    let mut declared = HashSet::new();
    let declarations = built
        .messages
        .iter()
        .map(|m| &m.full_name)
        .chain(built.enums.iter().map(|e| &e.full_name))
        .chain(built.services.iter().map(|s| &s.full_name));
    for name in declarations {
        if taken(name) || !declared.insert(name.as_str()) {
            return Err(format!("{name} is declared by another file"));
        }
    }
    let new_messages: HashSet<&str> = built
        .messages
        .iter()
        .map(|m| m.full_name.as_str())
        .collect();
    let new_enums: HashSet<&str> = built.enums.iter().map(|e| e.full_name.as_str()).collect();
    for field in built.messages.iter().flat_map(|m| &m.fields) {
        let name = field.type_name.as_str();
        let declared = match field.kind {
            Kind::Message => {
                new_messages.contains(name)
                    || pool.messages.contains_key(name)
                    || registry.messages.contains_key(name)
            }
            Kind::Enum => {
                new_enums.contains(name)
                    || pool.enums.contains_key(name)
                    || registry.enums.contains_key(name)
            }
            _ => true,
        };
        if !declared {
            return Err(format!(
                "the field {} is of the type {name}, which no file declares",
                field.full_name
            ));
        }
    }

    let Built {
        messages,
        enums,
        services,
        files,
    } = built;
    for message in messages {
        let message: &'static Arc<MessageDesc> = Box::leak(Box::new(Arc::new(message)));
        registry.messages.insert(message.full_name.clone(), message);
    }
    for enumeration in enums {
        let enumeration: &'static Arc<EnumDesc> = Box::leak(Box::new(Arc::new(enumeration)));
        registry
            .enums
            .insert(enumeration.full_name.clone(), enumeration);
    }
    for service in services {
        let service: &'static Arc<ServiceDesc> = Box::leak(Box::new(Arc::new(service)));
        registry.services.insert(service.full_name.clone(), service);
    }
    for file in files {
        registry.files.push(Box::leak(Box::new(file)));
    }
    Ok(protos
        .iter()
        .filter_map(|p| {
            pool.files
                .iter()
                .find(|f| f.name == p.name())
                .or_else(|| registry.file(p.name()))
        })
        .collect())
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

/// The JSON name of a field of the name `name` that declares none, as protobuf's `ToJsonName`
/// (and upb's) makes it: each letter after an underscore upper-cased, without the underscores,
/// so `delete_from_table` is `deleteFromTable`. The descriptors that protoc embeds in the modules
/// that it generates leave out the JSON names of their fields that declare none.
fn json_name_of(name: &str) -> String {
    let mut json_name = String::with_capacity(name.len());
    let mut upper_next = false;
    for c in name.chars() {
        if c == '_' {
            upper_next = true;
        } else if upper_next {
            json_name.push(c.to_ascii_uppercase());
            upper_next = false;
        } else {
            json_name.push(c);
        }
    }
    json_name
}

/// The file of the messages and enums that [`add_message`] and [`add_enum`] add.
struct FileScope<'a> {
    name: &'a str,
    package: &'a str,
    syntax: Syntax,
}

/// The syntax of a file, as its semantics differ: proto2, proto3, or an edition of the features
/// of its descriptors.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Syntax {
    Proto2,
    Proto3,
    Editions,
}

/// The editions that the runtime reads, as `google.protobuf.Edition` numbers them: 2023 and 2024.
const EDITIONS: [i32; 2] = [1000, 1001];

impl Syntax {
    /// The syntax of a file of the `syntax` and `edition` of its descriptor, as protoc writes them
    /// (no syntax of proto2), if the runtime reads it.
    fn of(syntax: &str, edition: Option<i32>) -> Option<Syntax> {
        match syntax {
            "" | "proto2" => Some(Syntax::Proto2),
            "proto3" => Some(Syntax::Proto3),
            "editions" if edition.is_some_and(|e| EDITIONS.contains(&e)) => Some(Syntax::Editions),
            _ => None,
        }
    }

    /// The features of the files of the syntax, as protobuf resolves those of proto2 and proto3,
    /// and the defaults of editions 2023 and 2024.
    fn defaults(self) -> Resolved {
        match self {
            Syntax::Proto2 => Resolved {
                presence: features::EXPLICIT,
                closed_enums: true,
                packed: false,
                validate_utf8: false,
                delimited: false,
            },
            Syntax::Proto3 => Resolved {
                presence: features::IMPLICIT,
                closed_enums: false,
                packed: true,
                validate_utf8: true,
                delimited: false,
            },
            Syntax::Editions => Resolved {
                presence: features::EXPLICIT,
                closed_enums: false,
                packed: true,
                validate_utf8: true,
                delimited: false,
            },
        }
    }
}

/// The numbers of the values of the features of `google.protobuf.FeatureSet` that the runtime
/// reads.
mod features {
    pub(super) const EXPLICIT: i32 = 1;
    pub(super) const IMPLICIT: i32 = 2;
    pub(super) const LEGACY_REQUIRED: i32 = 3;
    pub(super) const OPEN: i32 = 1;
    pub(super) const CLOSED: i32 = 2;
    pub(super) const PACKED: i32 = 1;
    pub(super) const EXPANDED: i32 = 2;
    pub(super) const VERIFY: i32 = 2;
    pub(super) const NONE: i32 = 3;
    pub(super) const LENGTH_PREFIXED: i32 = 1;
    pub(super) const DELIMITED: i32 = 2;
}

/// The features of a scope of a file (the file, a message, a field or an enum), as its own
/// features over those of the scope that holds it resolve them. Only editions declare features;
/// proto2 and proto3 have those of their syntax.
#[derive(Clone, Copy, Debug)]
struct Resolved {
    presence: i32,
    closed_enums: bool,
    packed: bool,
    validate_utf8: bool,
    delimited: bool,
}

impl Resolved {
    /// These features, of those that `declared` declares over them.
    fn with(self, declared: Option<&FeaturesView>) -> Resolved {
        let Some(declared) = declared else {
            return self;
        };
        let mut resolved = self;
        if let Some(presence) = declared.field_presence.filter(|&p| p != 0) {
            resolved.presence = presence;
        }
        match declared.enum_type {
            Some(features::CLOSED) => resolved.closed_enums = true,
            Some(features::OPEN) => resolved.closed_enums = false,
            _ => {}
        }
        match declared.repeated_field_encoding {
            Some(features::PACKED) => resolved.packed = true,
            Some(features::EXPANDED) => resolved.packed = false,
            _ => {}
        }
        match declared.utf8_validation {
            Some(features::VERIFY) => resolved.validate_utf8 = true,
            Some(features::NONE) => resolved.validate_utf8 = false,
            _ => {}
        }
        match declared.message_encoding {
            Some(features::DELIMITED) => resolved.delimited = true,
            Some(features::LENGTH_PREFIXED) => resolved.delimited = false,
            _ => {}
        }
        resolved
    }
}

/// The features of editions of a descriptor, which prost-types 0.14 does not read: those of
/// `google.protobuf.FeatureSet` that the runtime reads, of their numbers.
#[derive(Clone, PartialEq, prost::Message)]
struct FeaturesView {
    #[prost(int32, optional, tag = "1")]
    field_presence: Option<i32>,
    #[prost(int32, optional, tag = "2")]
    enum_type: Option<i32>,
    #[prost(int32, optional, tag = "3")]
    repeated_field_encoding: Option<i32>,
    #[prost(int32, optional, tag = "4")]
    utf8_validation: Option<i32>,
    #[prost(int32, optional, tag = "5")]
    message_encoding: Option<i32>,
}

/// The `features` of the options of a descriptor, which each type of options numbers: those of a
/// file 50, of a message 12, of a field 21 and of an enum 7.
#[derive(Clone, PartialEq, prost::Message)]
struct FileOptionsView {
    #[prost(message, optional, tag = "50")]
    features: Option<FeaturesView>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct MessageOptionsView {
    #[prost(message, optional, tag = "12")]
    features: Option<FeaturesView>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct FieldOptionsView {
    #[prost(message, optional, tag = "21")]
    features: Option<FeaturesView>,
}

#[derive(Clone, PartialEq, prost::Message)]
struct EnumOptionsView {
    #[prost(message, optional, tag = "7")]
    features: Option<FeaturesView>,
}

/// The options of a field, as [`FileView`] reads them.
#[derive(Clone, PartialEq, prost::Message)]
struct FieldView {
    #[prost(message, optional, tag = "8")]
    options: Option<FieldOptionsView>,
}

/// The options of an enum, as [`FileView`] reads them.
#[derive(Clone, PartialEq, prost::Message)]
struct EnumView {
    #[prost(message, optional, tag = "3")]
    options: Option<EnumOptionsView>,
}

/// The options of a message and of its members, as [`FileView`] reads them.
#[derive(Clone, PartialEq, prost::Message)]
struct MessageView {
    #[prost(message, repeated, tag = "2")]
    field: Vec<FieldView>,
    #[prost(message, repeated, tag = "3")]
    nested_type: Vec<MessageView>,
    #[prost(message, repeated, tag = "4")]
    enum_type: Vec<EnumView>,
    #[prost(message, optional, tag = "7")]
    options: Option<MessageOptionsView>,
}

/// The edition of a file and the features of the options of its descriptors, of the bytes of its
/// `FileDescriptorProto`, in the order of the descriptors of prost-types' decoding of them.
#[derive(Clone, PartialEq, prost::Message)]
struct FileView {
    #[prost(message, repeated, tag = "4")]
    message_type: Vec<MessageView>,
    #[prost(message, repeated, tag = "5")]
    enum_type: Vec<EnumView>,
    #[prost(message, optional, tag = "8")]
    options: Option<FileOptionsView>,
    #[prost(int32, optional, tag = "14")]
    edition: Option<i32>,
}

/// What `file` declares that the runtime does not hold, if anything: extensions (of the file or of
/// a message), extension ranges, or the message-set wire format of a message. Its messages then
/// keep `protobuf`'s classes.
fn unheld_declaration(file: &FileDescriptorProto) -> Option<String> {
    fn of_message(message: &DescriptorProto) -> Option<String> {
        if !message.extension.is_empty() || !message.extension_range.is_empty() {
            return Some(format!("the extensions of {}", message.name()));
        }
        if message
            .options
            .as_ref()
            .and_then(|o| o.message_set_wire_format)
            .unwrap_or(false)
        {
            return Some(format!("the message set {}", message.name()));
        }
        message.nested_type.iter().find_map(of_message)
    }
    if !file.extension.is_empty() {
        return Some("extensions".to_string());
    }
    file.message_type.iter().find_map(of_message)
}

/// The value of the default `text` of a field of `kind`, as protoc writes it in a descriptor: of
/// an enum field, the name of its value, as a string to resolve once the enums are built.
fn default_of(kind: Kind, text: &str) -> Option<Scalar> {
    let float = |text: &str| -> Option<f64> {
        match text {
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            "nan" | "-nan" => Some(f64::NAN),
            _ => text.parse().ok(),
        }
    };
    Some(match kind {
        Kind::Double => Scalar::F64(float(text)?),
        Kind::Float => Scalar::F32(float(text)? as f32),
        Kind::Int64 | Kind::SInt64 | Kind::SFixed64 => Scalar::I64(text.parse().ok()?),
        Kind::UInt64 | Kind::Fixed64 => Scalar::U64(text.parse().ok()?),
        Kind::Int32 | Kind::SInt32 | Kind::SFixed32 => Scalar::I32(text.parse().ok()?),
        Kind::UInt32 | Kind::Fixed32 => Scalar::U32(text.parse().ok()?),
        Kind::Bool => Scalar::Bool(text == "true"),
        Kind::String | Kind::Enum => Scalar::Str(text.to_string()),
        Kind::Bytes => Scalar::Bytes(c_unescape(text)),
        Kind::Message => return None,
    })
}

/// The bytes of `text`, of the C escapes that protoc writes of the default of a bytes field
/// (`\001\377x`): an octal escape of up to three digits, `\x` of up to two hexadecimal ones, and
/// the escapes of single characters.
fn c_unescape(text: &str) -> Vec<u8> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        let byte = bytes[at];
        at += 1;
        if byte != b'\\' || at >= bytes.len() {
            out.push(byte);
            continue;
        }
        let escape = bytes[at];
        at += 1;
        match escape {
            b'0'..=b'7' => {
                let mut value = u32::from(escape - b'0');
                for _ in 0..2 {
                    match bytes.get(at) {
                        Some(&d @ b'0'..=b'7') => {
                            value = value * 8 + u32::from(d - b'0');
                            at += 1;
                        }
                        _ => break,
                    }
                }
                out.push(value as u8);
            }
            b'x' | b'X' => {
                let mut value = 0u32;
                for _ in 0..2 {
                    match bytes.get(at).and_then(|&d| (d as char).to_digit(16)) {
                        Some(d) => {
                            value = value * 16 + d;
                            at += 1;
                        }
                        None => break,
                    }
                }
                out.push(value as u8);
            }
            b'a' => out.push(0x07),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'v' => out.push(0x0b),
            other => out.push(other),
        }
    }
    out
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

/// The scope that a message or an enum is declared in: the prefix of its full name, the full name
/// of the message that declares it, if any, and the features of that scope.
type Within<'a> = (&'a str, Option<&'a str>, Resolved);

fn add_message(
    messages: &mut Vec<MessageDesc>,
    enums: &mut Vec<EnumDesc>,
    scope: &FileScope<'_>,
    (prefix, containing, outer): Within<'_>,
    proto: &DescriptorProto,
    view: &MessageView,
) -> Result<(), String> {
    let full_name = format!("{prefix}{}", proto.name());
    let nested_prefix = format!("{full_name}.");
    let features = outer.with(view.options.as_ref().and_then(|o| o.features.as_ref()));

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
        let proto_type = field.r#type.unwrap_or_default();
        // A group of proto2 is a message field encoded between a start and an end tag.
        let kind = if proto_type == GROUP {
            Some(Kind::Message)
        } else {
            Kind::of_proto(proto_type)
        }
        .ok_or_else(|| {
            format!(
                "the field {full_name}.{} has the unsupported type {proto_type}",
                field.name()
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
        let field_view = view.field.get(index).cloned().unwrap_or_default();
        let resolved = features.with(
            field_view
                .options
                .as_ref()
                .and_then(|o| o.features.as_ref()),
        );
        // A field of `LEGACY_REQUIRED` of editions is required, a message field too, as upb's.
        let presence = match scope.syntax {
            Syntax::Proto2 if field.label == Some(2) => Presence::Required,
            Syntax::Proto2 => Presence::Explicit,
            Syntax::Editions if !repeated && resolved.presence == features::LEGACY_REQUIRED => {
                Presence::Required
            }
            _ if kind == Kind::Message || oneof.is_some() || proto3_optional => Presence::Explicit,
            Syntax::Proto3 => Presence::Implicit,
            Syntax::Editions => match resolved.presence {
                features::IMPLICIT => Presence::Implicit,
                _ => Presence::Explicit,
            },
        };
        let declared_packed = field.options.as_ref().and_then(|o| o.packed);
        let packed = repeated
            && kind.packable()
            && match scope.syntax {
                Syntax::Proto2 => declared_packed.unwrap_or(false),
                Syntax::Proto3 => declared_packed.unwrap_or(true),
                Syntax::Editions => resolved.packed,
            };
        let delimited = proto_type == GROUP
            || (scope.syntax == Syntax::Editions && kind == Kind::Message && resolved.delimited);
        let default = field
            .default_value
            .as_deref()
            .filter(|_| !repeated)
            .and_then(|text| default_of(kind, text));
        fields.push(FieldDesc {
            name: field.name().to_string(),
            full_name: format!("{full_name}.{}", field.name()),
            json_name: field
                .json_name
                .clone()
                .unwrap_or_else(|| json_name_of(field.name())),
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
            presence,
            default,
            delimited,
            validate_utf8: resolved.validate_utf8,
            closed_values: None,
            message_type: OnceLock::new(),
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
    let required = (0..fields.len())
        .filter(|&i| fields[i].is_required())
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
        required,
        may_lack_required: OnceLock::new(),
        class: OnceLock::new(),
        class_changed: AtomicU64::new(0),
    });

    for (index, nested) in proto.nested_type.iter().enumerate() {
        let view = view.nested_type.get(index).cloned().unwrap_or_default();
        add_message(
            messages,
            enums,
            scope,
            (&nested_prefix, Some(&full_name), features),
            nested,
            &view,
        )?;
    }
    for (index, enumeration) in proto.enum_type.iter().enumerate() {
        let view = view.enum_type.get(index).cloned().unwrap_or_default();
        add_enum(
            enums,
            scope,
            (&nested_prefix, Some(&full_name), features),
            enumeration,
            &view,
        );
    }
    Ok(())
}

fn add_enum(
    enums: &mut Vec<EnumDesc>,
    scope: &FileScope<'_>,
    (prefix, containing, outer): Within<'_>,
    proto: &EnumDescriptorProto,
    view: &EnumView,
) {
    let closed = match scope.syntax {
        Syntax::Proto2 => true,
        Syntax::Proto3 => false,
        Syntax::Editions => {
            outer
                .with(view.options.as_ref().and_then(|o| o.features.as_ref()))
                .closed_enums
        }
    };
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
        closed,
    });
}

/// The number of the type of a group, as `google.protobuf.FieldDescriptorProto.Type` numbers it.
const GROUP: i32 = 10;

#[cfg(test)]
mod tests {
    use super::*;
    use prost_types::field_descriptor_proto::{Label, Type};
    use prost_types::{EnumValueDescriptorProto, FieldDescriptorProto};

    fn field(name: &str, number: i32, kind: Type, type_name: &str) -> FieldDescriptorProto {
        FieldDescriptorProto {
            name: Some(name.to_string()),
            number: Some(number),
            label: Some(Label::Optional as i32),
            r#type: Some(kind as i32),
            type_name: (!type_name.is_empty()).then(|| type_name.to_string()),
            ..Default::default()
        }
    }

    fn message(name: &str, fields: Vec<FieldDescriptorProto>) -> DescriptorProto {
        DescriptorProto {
            name: Some(name.to_string()),
            field: fields,
            ..Default::default()
        }
    }

    /// A file of `messages`, serialized as protobuf's `serialized_pb` gives it, of protoc's
    /// fully qualified type names and of no JSON names.
    fn file(
        name: &str,
        syntax: &str,
        dependencies: &[&str],
        messages: Vec<DescriptorProto>,
    ) -> Vec<u8> {
        FileDescriptorProto {
            name: Some(name.to_string()),
            package: Some("tests.pool".to_string()),
            dependency: dependencies.iter().map(|d| d.to_string()).collect(),
            message_type: messages,
            syntax: Some(syntax.to_string()),
            ..Default::default()
        }
        .encode_to_vec()
    }

    #[test]
    fn registers_files_that_import_those_of_the_set_once() {
        // A plugin's file, which imports relations.proto and a file of its own, which the pool
        // does not have either.
        let mut base = FileDescriptorProto::decode(
            file(
                "tests/pool/base.proto",
                "proto3",
                &[],
                vec![message(
                    "Table",
                    vec![field("table_name", 1, Type::String, "")],
                )],
            )
            .as_slice(),
        )
        .unwrap();
        base.enum_type.push(EnumDescriptorProto {
            name: Some("Mode".to_string()),
            value: vec![EnumValueDescriptorProto {
                name: Some("MODE_UNSPECIFIED".to_string()),
                number: Some(0),
                ..Default::default()
            }],
            ..Default::default()
        });
        let base = base.encode_to_vec();
        let plugin = file(
            "tests/pool/plugin.proto",
            "proto3",
            &["spark/connect/relations.proto", "tests/pool/base.proto"],
            vec![message(
                "Scan",
                vec![
                    field("table", 1, Type::Message, ".tests.pool.Table"),
                    field("plan_input", 2, Type::Message, ".spark.connect.Relation"),
                    field("mode", 3, Type::Enum, ".tests.pool.Mode"),
                ],
            )],
        );
        let added = register(&[base.clone(), plugin.clone()]).expect("the files are added");
        let names: Vec<&str> = added.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["tests/pool/base.proto", "tests/pool/plugin.proto"]);
        let pool = pool();
        let scan = pool.message("tests.pool.Scan").expect("Scan");
        assert_eq!(scan.file, "tests/pool/plugin.proto");
        // The JSON names, which the serialized file leaves out, are protobuf's.
        assert_eq!(scan.fields[1].json_name, "planInput");
        assert!(pool.enumeration("tests.pool.Mode").is_some());
        assert!(pool
            .all_files()
            .iter()
            .any(|f| f.name == "tests/pool/plugin.proto"));
        assert_eq!(
            pool.file_proto("tests/pool/plugin.proto")
                .expect("its proto")
                .encode_to_vec(),
            plugin
        );

        // Adding them again, in another order, adds nothing: the pool gives the files it has.
        let again = register(&[plugin, base]).expect("the files are there");
        assert!(std::ptr::eq(again[0], added[1]) && std::ptr::eq(again[1], added[0]));
        assert!(std::ptr::eq(pool.message("tests.pool.Scan").unwrap(), scan));
        // As a file of the descriptor set, which is left as it is.
        let base = pool.file("spark/connect/base.proto").expect("base.proto");
        let set = register(std::slice::from_ref(&base.serialized)).expect("the file is there");
        assert!(std::ptr::eq(set[0], base));
    }

    #[test]
    fn refuses_files_that_the_runtime_does_not_hold_and_adds_none_of_them() {
        let good = file(
            "tests/pool/good.proto",
            "proto3",
            &[],
            vec![message("Good", vec![field("name", 1, Type::String, "")])],
        );
        let mut extendable = message("X", vec![]);
        extendable
            .extension_range
            .push(prost_types::descriptor_proto::ExtensionRange {
                start: Some(100),
                end: Some(200),
                options: None,
            });
        let cases = [
            (
                file("tests/pool/editions.proto", "editions", &[], vec![]),
                "tests/pool/editions.proto is of a syntax or edition that the runtime does not read",
            ),
            (
                file("tests/pool/extendable.proto", "proto2", &[], vec![extendable]),
                "tests/pool/extendable.proto declares the extensions of X, which the runtime does \
                 not hold",
            ),
            (
                file(
                    "tests/pool/imports.proto",
                    "proto3",
                    &["tests/pool/no.proto"],
                    vec![],
                ),
                "imports tests/pool/no.proto, which is not in the pool",
            ),
            (
                file(
                    "tests/pool/undeclared.proto",
                    "proto3",
                    &[],
                    vec![message(
                        "U",
                        vec![field("u", 1, Type::Message, ".tests.pool.No")],
                    )],
                ),
                "the field tests.pool.U.u is of the type tests.pool.No, which no file declares",
            ),
            (
                file(
                    "tests/pool/enum.proto",
                    "proto3",
                    &[],
                    vec![message(
                        "E",
                        vec![field("e", 1, Type::Enum, ".tests.pool.Good")],
                    )],
                ),
                "the field tests.pool.E.e is of the type tests.pool.Good, which no file declares",
            ),
            (
                file(
                    "tests/pool/twice.proto",
                    "proto3",
                    &[],
                    vec![message("Good", vec![])],
                ),
                "tests.pool.Good is declared by another file",
            ),
            (
                file(
                    "tests/pool/group.proto",
                    "proto3",
                    &[],
                    vec![message("G", vec![field("g", 1, Type::Group, "")])],
                ),
                "the field tests.pool.G.g is of the type , which no file declares",
            ),
            (vec![0xff], "a file that is not a FileDescriptorProto"),
        ];
        for (bad, error) in cases {
            let refused = register(&[good.clone(), bad]).expect_err(error);
            assert!(refused.contains(error), "{refused}");
            assert!(pool().file("tests/pool/good.proto").is_none());
            assert!(pool().message("tests.pool.Good").is_none());
        }
        // A file that declares a name of the descriptor set.
        let mut relation = FileDescriptorProto::decode(
            file(
                "tests/pool/relation.proto",
                "proto3",
                &[],
                vec![message("Relation", vec![])],
            )
            .as_slice(),
        )
        .unwrap();
        relation.package = Some("spark.connect".to_string());
        let refused = register(&[relation.encode_to_vec()]).unwrap_err();
        assert_eq!(
            refused,
            "spark.connect.Relation is declared by another file"
        );
    }

    #[test]
    fn reads_the_presence_defaults_packing_and_groups_of_proto2_and_editions() {
        // proto2: optional fields of presence, of their defaults (the first value of a closed
        // enum of none), a required field, packing of [packed = true] only, and groups.
        let mut defaults = message(
            "D",
            vec![
                field("count", 1, Type::Int32, ""),
                field("data", 2, Type::Bytes, ""),
                field("color", 3, Type::Enum, ".tests.pool2.Color"),
                field("first", 4, Type::Enum, ".tests.pool2.Color"),
                field("id", 5, Type::String, ""),
                field("unpacked", 6, Type::Int32, ""),
                field("packed", 7, Type::Int32, ""),
                field("item", 8, Type::Group, ".tests.pool2.D.Item"),
            ],
        );
        defaults.field[0].default_value = Some("-5".to_string());
        defaults.field[1].default_value = Some("\\001\\377x".to_string());
        defaults.field[2].default_value = Some("GREEN".to_string());
        defaults.field[4].label = Some(Label::Required as i32);
        defaults.field[5].label = Some(Label::Repeated as i32);
        defaults.field[6].label = Some(Label::Repeated as i32);
        defaults.field[6].options = Some(prost_types::FieldOptions {
            packed: Some(true),
            ..Default::default()
        });
        defaults
            .nested_type
            .push(message("Item", vec![field("x", 9, Type::Int32, "")]));
        let mut proto2 = FileDescriptorProto::decode(
            file("tests/pool2/d.proto", "proto2", &[], vec![defaults]).as_slice(),
        )
        .unwrap();
        proto2.package = Some("tests.pool2".to_string());
        proto2.enum_type.push(EnumDescriptorProto {
            name: Some("Color".to_string()),
            value: ["RED", "GREEN"]
                .iter()
                .enumerate()
                .map(|(i, name)| EnumValueDescriptorProto {
                    name: Some(name.to_string()),
                    number: Some(i as i32 + 1),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        });
        register(&[proto2.encode_to_vec()]).expect("a proto2 file is held");
        let pool = pool();
        let desc = pool.message("tests.pool2.D").expect("D");
        let field_of = |name: &str| &desc.fields[desc.by_name[name]];
        assert_eq!(field_of("count").presence, Presence::Explicit);
        assert_eq!(field_of("count").default, Some(Scalar::I32(-5)));
        assert_eq!(
            field_of("data").default,
            Some(Scalar::Bytes(vec![0o001, 0o377, b'x']))
        );
        assert_eq!(field_of("color").default, Some(Scalar::Enum(2)));
        assert_eq!(field_of("first").default, Some(Scalar::Enum(1)));
        assert!(field_of("id").is_required() && desc.required == vec![4]);
        assert!(!field_of("unpacked").packed && field_of("packed").packed);
        let item = field_of("item");
        assert!(item.delimited && item.kind == Kind::Message && item.wire_type() == 3);
        assert!(!field_of("id").validate_utf8);
        assert!(pool.enumeration("tests.pool2.Color").unwrap().closed);
        // The values of the closed enum of each enum field, which a parse looks a value up in.
        assert_eq!(
            field_of("color").closed_values.as_deref(),
            Some(&[1, 2][..])
        );
        assert_eq!(field_of("count").closed_values, None);

        // Editions: the features of the file and of the fields over the defaults of the edition
        // 2023 (explicit presence, packed, checked UTF-8, length-prefixed), of the bytes of the
        // file as protoc writes them, which prost-types does not read.
        let held = register(&[editions_file_with_fields("tests/ed/f.proto")]);
        assert!(held.is_ok(), "{held:?}");
        let desc = pool.message("tests.ed.F").expect("F");
        let field_of = |name: &str| &desc.fields[desc.by_name[name]];
        assert_eq!(field_of("implicit").presence, Presence::Implicit);
        assert_eq!(field_of("explicit").presence, Presence::Explicit);
        assert!(!field_of("expanded").packed);
        assert!(field_of("inner").delimited);
        assert!(!field_of("unchecked").validate_utf8);
    }

    /// The bytes of a file of the edition 2023 of a message `F` of fields of features, as protoc
    /// writes it: of the file's feature of implicit presence, and of the fields' features.
    fn editions_file_with_fields(name: &str) -> Vec<u8> {
        fn varint(mut value: u64, out: &mut Vec<u8>) {
            while value >= 0x80 {
                out.push((value as u8) | 0x80);
                value >>= 7;
            }
            out.push(value as u8);
        }
        fn len_field(number: u32, payload: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            varint(u64::from((number << 3) | 2), &mut out);
            varint(payload.len() as u64, &mut out);
            out.extend_from_slice(payload);
            out
        }
        fn field_proto(
            name: &str,
            number: u8,
            kind: u8,
            label: u8,
            features: &[u8],
            type_name: &str,
        ) -> Vec<u8> {
            let mut out = len_field(1, name.as_bytes());
            out.extend_from_slice(&[0x18, number, 0x20, label, 0x28, kind]);
            if !type_name.is_empty() {
                out.extend(len_field(6, type_name.as_bytes()));
            }
            if !features.is_empty() {
                // FieldOptions.features is field 21: a tag of two bytes.
                let mut options = vec![0xaa, 0x01, features.len() as u8];
                options.extend_from_slice(features);
                out.extend(len_field(8, &options));
            }
            out
        }
        let mut message = len_field(1, b"F");
        message.extend(len_field(2, &field_proto("implicit", 1, 5, 1, &[], "")));
        message.extend(len_field(
            2,
            &field_proto("explicit", 2, 5, 1, &[0x08, 0x01], ""),
        ));
        message.extend(len_field(
            2,
            &field_proto("expanded", 3, 5, 3, &[0x18, 0x02], ""),
        ));
        message.extend(len_field(
            2,
            &field_proto("inner", 4, 11, 1, &[0x28, 0x02], ".tests.ed.F"),
        ));
        message.extend(len_field(
            2,
            &field_proto("unchecked", 5, 9, 1, &[0x20, 0x03], ""),
        ));
        let mut file = len_field(1, name.as_bytes());
        file.extend(len_field(2, b"tests.ed"));
        file.extend(len_field(4, &message));
        // FileOptions.features (50) of field_presence IMPLICIT.
        file.extend(len_field(8, &[0x92, 0x03, 0x02, 0x08, 0x02]));
        file.extend(len_field(12, b"editions"));
        file.extend_from_slice(&[0x70, 0xe8, 0x07]);
        file
    }

    #[test]
    fn reads_the_defaults_that_protoc_writes() {
        assert_eq!(
            default_of(Kind::Double, "inf"),
            Some(Scalar::F64(f64::INFINITY))
        );
        assert_eq!(
            default_of(Kind::Float, "-inf"),
            Some(Scalar::F32(f32::NEG_INFINITY))
        );
        assert!(matches!(default_of(Kind::Double, "nan"), Some(Scalar::F64(v)) if v.is_nan()));
        assert_eq!(
            default_of(Kind::UInt64, "18446744073709551615"),
            Some(Scalar::U64(u64::MAX))
        );
        assert_eq!(default_of(Kind::Bool, "true"), Some(Scalar::Bool(true)));
        assert_eq!(
            default_of(Kind::String, "a\"b"),
            Some(Scalar::Str("a\"b".to_string()))
        );
        assert_eq!(
            c_unescape("\\001\\377x\\n\\\\\\x41\\'"),
            b"\x01\xffx\n\\A'".to_vec()
        );
    }

    #[test]
    fn makes_the_json_names_that_fields_do_not_declare_as_protobuf() {
        // As upb's of a field without a json_name.
        let cases = [
            ("delete_from_table", "deleteFromTable"),
            ("target", "target"),
            ("a__b", "aB"),
            ("_x", "X"),
            ("x_", "x"),
            ("x1_y", "x1Y"),
            ("with_2d", "with2d"),
            ("camelCase_x", "camelCaseX"),
        ];
        for (name, json_name) in cases {
            assert_eq!(json_name_of(name), json_name, "{name}");
        }
    }

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

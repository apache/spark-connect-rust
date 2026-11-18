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
use std::sync::{Arc, OnceLock, RwLock, RwLockReadGuard};

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
    /// The key and value of a map field, which its maps share.
    pub(crate) map: Option<Arc<MapDesc>>,
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
        let Built {
            messages,
            enums,
            services,
            files,
        } = Self::build_files(&set.file)?;
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

    /// The descriptors of the messages, enums and services of `protos`, and of the files, with the
    /// map fields of the messages: a map entry message is nested in the message of its map field, in
    /// the same file.
    fn build_files(protos: &[FileDescriptorProto]) -> Result<Built, String> {
        let mut messages: Vec<MessageDesc> = Vec::new();
        let mut enums: Vec<EnumDesc> = Vec::new();
        let mut services: Vec<ServiceDesc> = Vec::new();
        let mut files = Vec::new();
        for file in protos {
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
        let entries: HashMap<String, Arc<MapDesc>> = messages
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
/// The files are added together or not at all: each file that the pool does not have must be a
/// proto3 file (the runtime reads the fields of proto3, whose presence, defaults and packing are
/// not those of proto2 or of editions, and which have no groups), import files that the pool has
/// or that `serialized` adds, declare no name that another file declares, and declare fields of
/// messages and enums that those files declare. Gives each file of `serialized`, as the pool has
/// it, in their order.
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
        if file.syntax() != "proto3" {
            return Err(format!("{} is not a proto3 file", file.name()));
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
    let built = Pool::build_files(&added.into_iter().cloned().collect::<Vec<_>>())?;

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
        let cases = [
            (
                file("tests/pool/proto2.proto", "proto2", &[], vec![]),
                "tests/pool/proto2.proto is not a proto3 file",
            ),
            (
                file("tests/pool/editions.proto", "editions", &[], vec![]),
                "tests/pool/editions.proto is not a proto3 file",
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
                "the field tests.pool.G.g has the unsupported type 10",
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

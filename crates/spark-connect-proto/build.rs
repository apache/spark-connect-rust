use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read when the script runs, not baked in when it is compiled: a build dir shared by
    // several checkouts can run a script compiled in another one, which may be gone.
    let manifest_dir =
        std::env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is not set")?;
    let proto_root = PathBuf::from(manifest_dir).join("proto");
    let connect_dir = proto_root.join("spark").join("connect");

    // The vendored Spark Connect .proto files. Keep in sync with proto/PROTO_VERSION.txt.
    // example_plugins.proto holds the messages of Spark's example plugin, which PySpark's
    // `pyspark.sql.connect.proto.example_plugins_pb2` has, as Spark's own proto library does.
    let protos = [
        "base",
        "catalog",
        "commands",
        "common",
        "example_plugins",
        "expressions",
        "ml",
        "ml_common",
        "pipelines",
        "relations",
        "types",
    ]
    .iter()
    .map(|f| connect_dir.join(format!("{f}.proto")))
    .collect::<Vec<_>>();

    for p in &protos {
        println!("cargo:rerun-if-changed={}", p.display());
    }

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?);
    let full_set = out_dir.join("spark_connect_descriptor_set_full.bin");
    // The client's codec is prost's, of its errors of a response tagged, and of a limit of the
    // nesting of a response (see src/codec.rs), which finds the type of a response by its name
    // (`prost::Name`).
    // The type URL of a message is protobuf's, of the domain `type.googleapis.com`, as the
    // `type_url` of the `google.protobuf.Any` that `prost_types::Any::from_msg` packs it in.
    let mut config = tonic_prost_build::Config::new();
    config
        .enable_type_names()
        .type_name_domain(["."], "type.googleapis.com");
    tonic_prost_build::configure()
        .build_server(cfg!(feature = "server"))
        .build_client(true)
        // async_trait repeats Future's must_use on generated server methods.
        .trait_attribute("SparkConnectService", "#[allow(clippy::double_must_use)]")
        // Tag client response decoding errors. Generated server bindings share this codec;
        // their error sources stay local and are not sent to clients, and they decode the
        // requests as prost does: the limit of the nesting is the client's, of the responses.
        .codec_path("crate::ClientCodec")
        .bytes(".")
        .file_descriptor_set_path(&full_set)
        .compile_with_config(config, &protos, &[proto_root])?;

    // The descriptor set of the protos and of the ones they import (google/protobuf/any.proto),
    // for `FILE_DESCRIPTOR_SET`. The source info, which holds the comments of every declaration,
    // is dropped: it is most of the set, and nothing reads it.
    use prost::Message;
    let mut set = prost_types::FileDescriptorSet::decode(std::fs::read(&full_set)?.as_slice())?;
    for file in &mut set.file {
        file.source_code_info = None;
    }
    std::fs::write(
        out_dir.join("spark_connect_descriptor_set.bin"),
        set.encode_to_vec(),
    )?;
    std::fs::write(out_dir.join("message_fields.rs"), message_fields(&set))?;

    Ok(())
}

/// The table of `src/depth.rs`: each message type of `set`, sorted by its full name, with the
/// fields of it that hold messages (of a message type, a map or a group), sorted by their numbers,
/// and the identifiers of the generated code that prost's errors name them by: the struct of the
/// message, and the field, or the oneof that holds it, as prost-build names them; and the full
/// names of the responses of the services of `set`, sorted.
fn message_fields(set: &prost_types::FileDescriptorSet) -> String {
    use prost_types::field_descriptor_proto::Type;
    use prost_types::DescriptorProto;

    struct Field {
        number: i32,
        kind: &'static str,
        type_name: String,
        ident: String,
    }
    struct MessageType {
        ident: String,
        map_entry: bool,
        fields: Vec<Field>,
    }

    fn collect(
        prefix: &str,
        message: &DescriptorProto,
        types: &mut std::collections::BTreeMap<String, MessageType>,
    ) {
        let full_name = format!("{prefix}.{}", message.name());
        let fields = message
            .field
            .iter()
            .filter(|field| matches!(field.r#type(), Type::Message | Type::Group))
            .map(|field| {
                let ident = match field.oneof_index {
                    Some(index) if !field.proto3_optional() => {
                        to_snake(message.oneof_decl[index as usize].name())
                    }
                    _ => to_snake(field.name()),
                };
                Field {
                    number: field.number(),
                    kind: if field.r#type() == Type::Group {
                        "Group"
                    } else {
                        "Message"
                    },
                    type_name: field.type_name().trim_start_matches('.').to_string(),
                    ident,
                }
            })
            .collect();
        types.insert(
            full_name.trim_start_matches('.').to_string(),
            MessageType {
                ident: to_upper_camel(message.name()),
                map_entry: message.options.as_ref().is_some_and(|o| o.map_entry()),
                fields,
            },
        );
        for nested in &message.nested_type {
            collect(&full_name, nested, types);
        }
    }

    let mut types = std::collections::BTreeMap::new();
    for file in &set.file {
        let package = if file.package().is_empty() {
            String::new()
        } else {
            format!(".{}", file.package())
        };
        for message in &file.message_type {
            collect(&package, message, &mut types);
        }
    }
    let index: std::collections::HashMap<&str, usize> = types
        .keys()
        .enumerate()
        .map(|(at, name)| (name.as_str(), at))
        .collect();
    let mut out = String::from(
        "// @generated by build.rs from the descriptor set of the protos.\n\
         pub(crate) static MESSAGE_TYPES: &[MessageType] = &[\n",
    );
    for (name, message) in &types {
        let mut fields: Vec<&Field> = message.fields.iter().collect();
        fields.sort_by_key(|field| field.number);
        out.push_str(&format!(
            "    MessageType {{ name: {name:?}, ident: {:?}, fields: &[\n",
            message.ident
        ));
        for field in fields {
            let target = index[field.type_name.as_str()];
            // A repeated field of a map entry type is a map.
            let kind = if types[&field.type_name].map_entry {
                "Map"
            } else {
                field.kind
            };
            out.push_str(&format!(
                "        Field {{ number: {}, kind: FieldKind::{kind}, message: {target}, ident: {:?} }},\n",
                field.number, field.ident
            ));
        }
        out.push_str("    ] },\n");
    }
    out.push_str("];\n");
    // The responses of the service, of which the client decodes the messages within the limit:
    // the generated server, which shares the codec, decodes the requests as prost does.
    let mut responses = std::collections::BTreeSet::new();
    for file in &set.file {
        for service in &file.service {
            for method in &service.method {
                responses.insert(method.output_type().trim_start_matches('.').to_string());
            }
        }
    }
    out.push_str("pub(crate) static RESPONSE_TYPES: &[&str] = &[\n");
    for name in responses {
        out.push_str(&format!("    {name:?},\n"));
    }
    out.push_str("];\n");
    out
}

/// A Rust identifier of `ident`, as prost-build's `sanitize_identifier` makes it: a raw identifier
/// of a keyword, `_` after one that cannot be raw, and `_` before a leading digit.
fn sanitize_identifier(ident: String) -> String {
    match ident.as_str() {
        "as" | "break" | "const" | "continue" | "else" | "enum" | "false" | "fn" | "for" | "if"
        | "impl" | "in" | "let" | "loop" | "match" | "mod" | "move" | "mut" | "pub" | "ref"
        | "return" | "static" | "struct" | "trait" | "true" | "type" | "unsafe" | "use"
        | "where" | "while" | "dyn" | "abstract" | "become" | "box" | "do" | "final" | "macro"
        | "override" | "priv" | "typeof" | "unsized" | "virtual" | "yield" | "async" | "await"
        | "try" | "gen" => format!("r#{ident}"),
        "_" | "super" | "self" | "Self" | "extern" | "crate" => format!("{ident}_"),
        s if s.starts_with(|c: char| c.is_numeric()) => format!("_{ident}"),
        _ => ident,
    }
}

/// The identifier of a field (or oneof) of the name `name`, as prost-build's `to_snake` makes it.
fn to_snake(name: &str) -> String {
    sanitize_identifier(heck::ToSnakeCase::to_snake_case(name))
}

/// The identifier of a message of the name `name`, as prost-build's `to_upper_camel` makes it.
fn to_upper_camel(name: &str) -> String {
    sanitize_identifier(heck::ToUpperCamelCase::to_upper_camel_case(name))
}

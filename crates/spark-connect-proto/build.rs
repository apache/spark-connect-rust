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
    tonic_prost_build::configure()
        .build_server(cfg!(feature = "server"))
        .build_client(true)
        // async_trait repeats Future's must_use on generated server methods.
        .trait_attribute("SparkConnectService", "#[allow(clippy::double_must_use)]")
        .bytes(".")
        .file_descriptor_set_path(&full_set)
        .compile_protos(&protos, &[proto_root])?;

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

    Ok(())
}

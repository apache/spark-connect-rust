use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read when the script runs, not baked in when it is compiled: a build dir shared by
    // several checkouts can run a script compiled in another one, which may be gone.
    let manifest_dir =
        std::env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is not set")?;
    let proto_root = PathBuf::from(manifest_dir).join("proto");
    let connect_dir = proto_root.join("spark").join("connect");

    // The vendored Spark Connect .proto files. Keep in sync with proto/PROTO_VERSION.txt.
    let protos = [
        "base",
        "catalog",
        "commands",
        "common",
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

    tonic_prost_build::configure()
        .build_server(false)
        .build_client(true)
        .bytes(".")
        .compile_protos(&protos, &[proto_root])?;

    Ok(())
}

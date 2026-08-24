#!/usr/bin/env bash
# Build the pyspark package with Rust bindings
set -eo pipefail

cd "$(dirname "$0")/.."

echo "Building pyspark-rs extension with offline cargo..."
CARGO_NET_OFFLINE=true cargo build -p pyspark-rs --release

echo "Copying dylib to Python package..."
mkdir -p python/pyspark

# Copy the built dylib to the Python package directory
# The dylib will be named lib_pyspark.dylib (or .so on Linux, .pyd on Windows)
# and needs to be named _pyspark.so for Python import
if [ -f "target/release/lib_pyspark.dylib" ]; then
    cp target/release/lib_pyspark.dylib python/pyspark/_pyspark.so
    echo "Copied to python/pyspark/_pyspark.so"
elif [ -f "target/release/lib_pyspark.so" ]; then
    cp target/release/lib_pyspark.so python/pyspark/_pyspark.so
    echo "Copied to python/pyspark/_pyspark.so"
else
    echo "ERROR: Could not find built dylib/so file"
    exit 1
fi

echo "Build complete!"

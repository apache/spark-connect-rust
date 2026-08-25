# Troubleshooting

Common issues and solutions when using Spark Connect Rust.

### Cannot Connect to Server / Connection Refused

**Symptom:** `Error: Failed to connect to sc://localhost:15002` or connection timeout.

**Solution:**
1. Verify the server is running: `$SPARK_HOME/sbin/start-connect-server.sh --packages "org.apache.spark:spark-connect_2.13:4.2.0"`
2. Check the port is correct. By default it's `15002`. To verify, look for `spark.connect.grpc.binding.port` in the server logs.
3. If connecting to a remote server, verify network connectivity: `ping spark.example.com` and check firewalls.
4. Ensure Apache Spark 4.2.0 or later is installed: `$SPARK_HOME/bin/spark-shell --version`

!!! tip
    Run the server in the foreground to see startup messages: `$SPARK_HOME/sbin/start-connect-server.sh --packages "org.apache.spark:spark-connect_2.13:4.2.0" 2>&1`

### TLS / Authentication Error

**Symptom:** `Error: TLS handshake failed` or `UNAUTHENTICATED` error when connecting.

**Solution:**
1. Verify you are using the correct connection string format with TLS enabled:
   ```rust
let spark = SparkSessionBuilder::default()
    .remote("sc://host:15002/;use_ssl=true;token=YOUR_TOKEN")
    .get_or_create()?;
```
2. If the server uses self-signed certificates, consult your infrastructure team.
3. Check that your token or authentication credentials are valid.
4. Verify the server is configured to accept your authentication method (check server config or logs).

!!! note
    Authentication parameters are passed in the connection string, not as config options.

### Slow First Query / Long Session Startup

**Symptom:** The first query takes 10–30 seconds, or SparkSession creation is slow.

**Solution:**
This is normal. The JVM and Spark runtime warm up on first use. Subsequent queries are fast.

- If this is unacceptable for your use case, consider keeping a long-lived session and reusing it.
- Profile your workload: most of the overhead is in JVM startup, not the Rust client.
- On a remote server, network latency may add to startup time.

### Protobuf Compilation Error

**Symptom:** Error during `cargo build`: `protobuf compiler not found` or similar.

**Solution:**
1. Install `protobuf-compiler`:
   - macOS: `brew install protobuf`
   - Linux (Debian/Ubuntu): `apt-get install protobuf-compiler`
   - Other systems: See [Protocol Buffers documentation](https://developers.google.com/protocol-buffers/docs/downloads)
2. Verify installation: `protoc --version`
3. Retry: `cargo build --release`

### Dataset Version Mismatch

**Symptom:** Errors about incompatible schema or unexpected null values.

**Solution:**
1. Verify the Spark Connect server and client are compatible versions: both should be Spark 4.2.0 or later.
2. If you recently upgraded Apache Spark, restart the Spark Connect server to apply the new version.
3. Check that data on disk matches the expected schema (e.g., after schema changes, re-write the data).

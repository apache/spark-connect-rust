# Configuration and Connection

Configure connections to local or remote Spark Connect servers, set session parameters, and manage authentication.

## Connection String Format

Connection strings follow the format `sc://host:port/;param=value`. Parameters are optional:

| Parameter | Purpose | Example |
| --- | --- | --- |
| `token` | Authentication token | `sc://host:15002/;token=abc123` |
| `user_id` | User identity | `sc://host:15002/;user_id=alice` |
| `session_id` | Session identifier | `sc://host:15002/;session_id=sess_xyz` |
| `use_ssl` | Enable TLS | `sc://host:15002/;use_ssl=true` |

!!! note
    TLS and authentication parameters are passed in the connection string. Consult your infrastructure team for any required credentials or certificates.

## Building a Session

```rust
use spark_connect::SparkSessionBuilder;

let spark = SparkSessionBuilder::default()
    .remote("sc://localhost:15002")
    .get_or_create()?;

// Session configuration is applied at runtime via `spark.conf()`:
spark.conf().set("spark.sql.shuffle.partitions", "20")?;
```

## Runtime Configuration

Access and modify session configuration at runtime:

```rust
// Get a config value
let partitions = spark.conf().get("spark.sql.shuffle.partitions")?;

// Set a config value
spark.conf().set("spark.sql.adaptive.enabled", "true")?;
```

## Starting a Local Spark Connect Server

To run a Spark Connect server locally for development:

```bash
# Set SPARK_HOME to your Spark installation
export SPARK_HOME=/path/to/spark

# Start the server on sc://localhost:15002
$SPARK_HOME/sbin/start-connect-server.sh \
  --packages "org.apache.spark:spark-connect_2.13:4.2.0"
```

The server listens on port `15002` by default. To use a different port, add `--conf spark.connect.grpc.binding.port=<port>`.

!!! tip
    The Spark Connect server requires a JVM and Apache Spark 4.2.0 or later. Stop it with `$SPARK_HOME/sbin/stop-connect-server.sh`.

## Remote Connections

To connect to a remote Spark Connect server:

```rust
let spark = SparkSessionBuilder::default()
    .remote("sc://spark.example.com:15002/;token=YOUR_TOKEN")
    .get_or_create()?;
```

Replace `spark.example.com`, the port, and authentication parameters as needed. If TLS is required:

```rust
let spark = SparkSessionBuilder::default()
    .remote("sc://spark.example.com:15002/;use_ssl=true;token=YOUR_TOKEN")
    .get_or_create()?;
```

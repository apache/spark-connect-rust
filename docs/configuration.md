# Configuration and Connection

Configure connections to local or remote Spark Connect servers, set session parameters, and manage authentication.

## Connection String Format

Connection strings follow the format `sc://host:port/;param=value`. Parameters are optional:

| Parameter | Purpose | Example |
| --- | --- | --- |
| `token` | Authentication token, sent over TLS to any host but localhost | `sc://host:15002/;token=abc123` |
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
$SPARK_HOME/sbin/start-connect-server.sh
```

The Spark distribution bundles the Spark Connect server, so it needs no `--packages`.

The server listens on port `15002` by default. To use a different port, add `--conf spark.connect.grpc.binding.port=<port>`.

!!! tip
    The Spark Connect server requires a JVM and Apache Spark 4.3.0 or later. Stop it with `$SPARK_HOME/sbin/stop-connect-server.sh`.

## Remote Connections

To connect to a remote Spark Connect server:

```rust
use spark_connect::SparkSessionBuilder;

let spark = SparkSessionBuilder::default()
    .remote("sc://spark.example.com:15002/;token=YOUR_TOKEN")
    .get_or_create()?;
```

Replace `spark.example.com`, the port, and authentication parameters as needed. As in PySpark, a token implies TLS, so this connects over TLS. A token of the `SPARK_CONNECT_AUTHENTICATE_TOKEN` environment variable does the same. Only to `localhost` is a token sent without TLS, for local development; to `127.0.0.1` and `[::1]` it is sent over TLS, as in PySpark. Without a token, `use_ssl=true` turns TLS on:

```rust
use spark_connect::SparkSessionBuilder;

let spark = SparkSessionBuilder::default()
    .remote("sc://spark.example.com:15002/;use_ssl=true")
    .get_or_create()?;
```

## HTTP Proxies

As PySpark's client (through grpcio), the client reaches the server through the HTTP proxy of the
environment, with an HTTP/1.1 `CONNECT` tunnel. TLS is negotiated with the Spark server inside the
tunnel and checked against the server's name, so the proxy sees neither the token nor the data.

```bash
export https_proxy=http://user:password@proxy.example.com:3128
export no_proxy=.internal.example.com,10.0.0.0/8
```

The proxy is the first of `grpc_proxy`, `https_proxy` and `http_proxy` that is set, and the hosts
that are not proxied are those of `no_grpc_proxy`, or else of `no_proxy`. These are the rules of
grpcio:

- Only the lowercase names are read; `HTTPS_PROXY` and `NO_PROXY` are not.
- Only an `http://` proxy is supported. Another scheme (`https://`, `socks5://`), a value
  without a scheme or an empty one connects directly, with a warning for an invalid value.
- The user and password of the proxy URI, percent-decoded, are sent as
  `Proxy-Authorization: Basic`. A proxy without a port is on port 443.
- A `no_proxy` entry matches the host itself or its subdomains (`example.com` and
  `.example.com` both match `spark.example.com`, not `notexample.com`), and an IP host in a CIDR
  range such as `10.0.0.0/8`. There is no `*` wildcard, and an entry with a port does not match.

Unlike grpcio, a loopback host (`localhost`, `127.0.0.0/8`, `[::1]`) is never proxied, as a
token is sent to `localhost` without TLS. A proxy address with an invalid host or port is an
error when the session is created.

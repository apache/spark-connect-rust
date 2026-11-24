# Configuration and Connection

Configure connections to local or remote Spark Connect servers, set session parameters, and manage authentication.

## Connection String Format

Connection strings follow the format `sc://host:port/;param=value`. Parameters are optional:

| Parameter | Purpose | Example |
| --- | --- | --- |
| `token` | Authentication token, sent over TLS to any host but localhost (see below) | `sc://host:15002/;token=abc123` |
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
    .app_name("my-app")
    .config("spark.sql.shuffle.partitions", 20)
    .configs([("spark.sql.ansi.enabled", "true")])
    .get_or_create()?;

// Or at runtime via `spark.conf()`:
spark.conf().set("spark.sql.shuffle.partitions", "20")?;
```

As PySpark's `SparkSession.builder` of Spark Connect, the session gets the options of `config`,
`configs` and `app_name` (`spark.app.name`) once it connects, and `master` has no effect. The Rust
API keeps no active session, so `get_or_create` creates a new session at each call, as `create` does.

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

Replace `spark.example.com`, the port, and authentication parameters as needed. As in PySpark, a token implies TLS, so this connects over TLS. A token of the `SPARK_CONNECT_AUTHENTICATE_TOKEN` environment variable does the same. Only to `localhost` is a token sent without TLS, for local development; to `127.0.0.1` and `[::1]` it is sent over TLS, as in PySpark. In the Python client, a `DefaultChannelBuilder` of a `_host` of `localhost` whose `endpoint` is another address, such as one of `127.0.0.1`, sends it without TLS over gRPC's local credentials, as PySpark's `toChannel` does, of a connection of the loopback only. Without a token, `use_ssl=true` turns TLS on:

```rust
use spark_connect::SparkSessionBuilder;

let spark = SparkSessionBuilder::default()
    .remote("sc://spark.example.com:15002/;use_ssl=true")
    .get_or_create()?;
```

## HTTP Proxies

As PySpark's client (through grpcio), the client reaches the server through the HTTP proxy of the
environment, with an HTTP/1.1 `CONNECT` tunnel. TLS is negotiated with the Spark server inside the
tunnel and checked against the server's name, so the proxy sees neither the token nor the data. As
grpcio, the client connects to the addresses of the proxy as to those of a server, the next one
250 ms after the previous one, each with a connect deadline of 20 s, and its `CONNECT` request
carries the `Host` and, of a proxy URI with a user, the `Proxy-Authorization` headers, and of the
Python client grpcio's `User-Agent: grpc-httpcli/0.0` too.

```bash
export https_proxy=http://user:password@proxy.example.com:3128
export no_proxy=.internal.example.com,10.0.0.0/8
```

The proxy is the first of `grpc_proxy`, `https_proxy` and `http_proxy` that is set, and the hosts
that are not proxied are those of `no_grpc_proxy`, or else of `no_proxy`. These are the rules of
grpcio:

- Only the lowercase names are read; `HTTPS_PROXY` and `NO_PROXY` are not.
- Only an `http://` proxy is supported. Another scheme (`https://`, `socks5://`), a value
  without a scheme or an empty one connects directly. An invalid value is logged as an error of
  each session, whatever its host (see [Errors on stderr](#errors-on-stderr)).
- The user and password of the proxy URI, percent-decoded, are sent as
  `Proxy-Authorization: Basic`. A proxy without a port is on port 443.
- A `no_proxy` entry matches the host itself or its subdomains (`example.com` and
  `.example.com` both match `spark.example.com`, not `notexample.com`), and an IP host, an IPv6
  one of a zone too (`[::1%25lo]`), in a CIDR range such as `10.0.0.0/8`. There is no `*`
  wildcard, and an entry with a port does not match.
- A loopback host (`localhost`, `127.0.0.1`) is proxied as any other unless `no_proxy` lists it
  (`no_proxy=localhost,127.0.0.0/8,::1`). Of the Python client's local credentials, a token goes
  to `localhost` without TLS only over a connection whose local address is a loopback one, so
  that a proxy on another machine gets none.

Unlike grpcio, a session of the Rust API whose token goes to `localhost` without TLS, of a
connection string that is not `use_ssl` (see `ChannelBuilder::secure`), connects to a loopback
host (`localhost` and the names under `.localhost`, `127.0.0.0/8`, `[::1]`, and IPv4-mapped
loopback addresses) directly, never through a proxy, which would get the token in the clear: the
Rust API has no local credentials that check the address of the connection. The sessions of the
Python client are proxied as grpcio's are. A proxy address with an invalid host or
port is an error when a session of a host that the proxy applies to is created, where grpcio makes
the session and fails its requests, as it cannot resolve the address, or takes a port past 65535
modulo 65536 and connects to that port: the error names the variable at once, rather than requests
that fail later or a connection to a port that was not given. As in grpcio, which resolves the
address of the proxy only for a host that it proxies, a host that `no_proxy` lists, or a loopback
host of such a session of the Rust API, connects directly whatever that address. A value that is not UTF-8 is not used, and the
client connects directly, where grpcio reads its bytes.

## TLS Root Certificates

A TLS channel trusts the root certificates of the system's trust store, which `SSL_CERT_FILE` and
`SSL_CERT_DIR` can replace, and, of the Python client where grpcio is installed (PySpark's retry
policies need it), those of grpcio's own bundle too, the `roots.pem` of the installed `grpc`
package, which grpcio's channels trust by default. As in grpcio, `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH`
names a PEM file whose certificates are trusted instead, such as the CA of a corporate proxy that
inspects TLS:

```bash
export GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/etc/ssl/corporate-ca.pem
```

The file replaces the other roots rather than adding to them. As in grpcio, the variable and the
file are read once, when the first TLS session of the process is made, and an empty file is as no
file. A file that cannot be read is logged as an error (see [Errors on stderr](#errors-on-stderr)),
and the client uses the default roots. A file without a certificate does not fail the session when
it is made, as in grpcio: each call fails `UNAVAILABLE`, `Failed to create security handshaker.`,
and so does each call of a system without roots, where neither grpcio nor a file gives some.

The Python client trusts more than PySpark does, on purpose: a CA of the system's trust store, such
as the private CA of a company that its administrators install, is trusted, which PySpark, of
grpcio's bundle alone, does not trust. So a program whose server has such a CA keeps connecting when
grpcio is installed as a dependency of another package, and every server that PySpark's client
trusts is trusted. The Rust API takes the system's store rather than a bundle of its own as it holds
the CAs that the administrator of the machine installs, and is kept up to date with the system
rather than with the releases of the client.

## Errors on stderr

PySpark's own code says nothing of a proxy variable that is not a proxy, of a roots file that cannot
be read, of a TLS handshake that fails or of a proxy that refuses the tunnel: gRPC's C core, in
grpcio, logs them to stderr, the failure of a TLS handshake as information (`I`) and the others as
errors (`E`). The Python client logs them there as lines of the same form, which name the Rust
source of the call, where grpcio's name its own:

```text
E1118 16:53:41.241267 4026384 http_proxy_mapper.cc:<line>] 'https' scheme not supported in proxy URI
E1118 17:37:31.168823  299507 proxy.rs:<line>] the 'https' scheme of the proxy URI is not supported in https_proxy; not using a proxy
I1119 01:06:45.505781 3300774 ssl_transport_security.cc:<line>] Handshake failed with error SSL_ERROR_SSL: error:1000007d:SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate
I1119 02:15:41.966502  505468 connector.rs:<line>] Handshake failed with error SSL_ERROR_SSL: error:1000007d:SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate
E1119 01:03:42.670473 3254142 http_connect_client_handshaker.cc:<line>] HTTP proxy handshake with ipv4:127.0.0.1:3128 failed: UNKNOWN: HTTP proxy returned response code 403
E1119 02:15:42.060818  505955 connector.rs:<line>] HTTP proxy handshake with ipv4:127.0.0.1:3128 failed: UNKNOWN: HTTP proxy returned response code 403
```

The `<line>` is the line of the source file that logs it.

As in grpcio, a proxy variable is logged for each session that is connected, once for a session of
`newSession()` and of `cloneSession()` too, whose channel reads it again, the roots file once, when
it is read, a TLS handshake that fails at each attempt to connect, in the background too, and a
tunnel that fails once a minute at most, whatever the attempts in between, as gRPC's core limits the
line of its source. Nothing is logged when `GRPC_VERBOSITY` is `NONE` as the client is imported, and
only the errors when it is `ERROR`, as grpcio reads it when it initializes; of `INFO` or `DEBUG`,
the client logs what it logs by default, without the other lines that gRPC's core logs then. The
Rust API gives the errors of the variables and of the roots file to `tracing`, as warnings; the
failures of its connections are in the errors of its calls.

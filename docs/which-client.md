# Which client am I using?

`pyspark-client-rust` installs under the **`pyspark`** import name and mirrors the
PySpark Spark Connect API exactly, so your code looks identical to code written for
the reference client. That is the whole point of a drop-in - but it also means it is
worth being deliberate about **which** client is installed, so that behavior and
bug reports are attributed to the right project.

## The three packages on PyPI

| Package | Import | What it is |
|---|---|---|
| [`pyspark`](https://pypi.org/project/pyspark/) | `pyspark` | Full Apache Spark for Python - Spark Classic (JVM/py4j) **and** the Spark Connect client. |
| [`pyspark-client`](https://pypi.org/project/pyspark-client/) | `pyspark` | The **reference** Spark Connect client: Connect-only, pure Python, gRPC via `grpcio`. |
| **`pyspark-client-rust`** (this project) | `pyspark` | A Spark Connect client with the **same API surface**, backed by a native **Rust** engine (gRPC via `tonic`) instead of `grpcio`/py4j. |

All three expose the `pyspark` package name, and only one can be installed into an
environment at a time. `pyspark-client-rust` is a drop-in for `pyspark-client`:
uninstall any existing `pyspark` / `pyspark-client`, install `pyspark-client-rust`,
and your Spark Connect code runs unchanged (see [Installation](installation.md)).

## Same API, different engine

What is **the same**: the Python API surface, the `spark.connect` protobuf protocol
on the wire, and the results you get back. Plans built by this client are checked
**byte-for-byte** against the reference client, and the official PySpark Connect test
suite runs against it (see [Compatibility](compatibility.md)).

What is **different**: the engine. The reference client builds protobuf plans,
manages the gRPC channel, and decodes Arrow results in Python; `pyspark-client-rust`
does all of that in Rust. That is where the performance difference comes from -
biggest on client-bound work such as plan building and result decoding - and it is
also why the two are separate implementations that can have separate bugs.

## How to tell, at runtime

=== "Python attribute"

    ```python
    import pyspark
    pyspark.__rust_client__   # True only in pyspark-client-rust
    pyspark.__engine__        # "rust"
    pyspark.__version__       # the Spark version this client targets, e.g. "4.2.0"
    ```

=== "Log on connect"

    The first time a session connects, the client emits a one-line `INFO` log on the
    `pyspark` logger:

    ```
    pyspark-client-rust 4.2.0 active: Spark Connect client backed by the native Rust
    engine (tonic), a drop-in for pyspark-client -- not the reference Python client.
    ```

    Enable it with `logging.basicConfig(level=logging.INFO)`.

=== "pip"

    ```console
    $ pip show pyspark-client-rust
    ```

!!! note "Switching back"
    Because all three packages share the `pyspark` import directory, **uninstalling
    `pyspark-client-rust` alone leaves you without a working `pyspark`** - reinstall
    `pyspark-client` (or `pyspark`) to switch back. See
    [Installation](installation.md) for the exact steps.

## Reporting issues

`pyspark-client-rust` is a **separate implementation**. If you hit a problem while
using it, please file it against
**[apache/spark-connect-rust](https://github.com/apache/spark-connect-rust)** (or
the [SPARK JIRA](https://issues.apache.org/jira/browse/SPARK)) rather than reporting
it as a reference `pyspark-client` bug - confirm `pyspark.__rust_client__` is `True`
first. If a behavior differs from the reference client, that difference is itself the
bug we want to hear about, since the goal is byte-for-byte parity.

## Server compatibility

This client speaks to Apache Spark Connect servers **4.2.0 and later** (see
[Compatibility](compatibility.md)). To talk to servers older than 4.2, keep using the
reference client; the two can coexist in separate environments.

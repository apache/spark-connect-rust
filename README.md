<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Spark Connect Rust Client

A fast, native **Rust** client for **Apache Spark Connect** - and a drop-in
`pyspark` replacement. It builds `spark.connect` protobuf plans, manages the gRPC
channel, and decodes Arrow results in Rust, speaking the same protocol and
returning the same results as the reference client.

[![PyPI](https://img.shields.io/pypi/v/pyspark-client-rust?color=c2410c&label=pyspark-client-rust)](https://pypi.org/project/pyspark-client-rust/)
![Spark](https://img.shields.io/badge/Apache%20Spark-4.2.0%2B-c2410c)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)

## 📖 Documentation

**Full documentation lives at
[apache.github.io/spark-connect-rust](https://apache.github.io/spark-connect-rust/)**
- installation, quickstart, the DataFrame / Columns / SQL / Reading &
Writing / Streaming / Catalog / Types API, [Rust UDFs via
WebAssembly](https://apache.github.io/spark-connect-rust/udfs/), and the
[architecture](https://apache.github.io/spark-connect-rust/architecture/).

## Install

**Python** - a faster, drop-in replacement for the
[`pyspark-client`](https://pypi.org/project/pyspark-client/) PyPI package
(uninstall any existing `pyspark` / `pyspark-client` first):

```bash
pip install pyspark-client-rust
```

Your Spark Connect code then runs unchanged; use it exactly like
[PySpark](https://spark.apache.org/docs/latest/api/python/).

**Rust** - the native crate:

```toml
[dependencies]
apache-spark-connect = "4.2"
```

## Quickstart (Rust)

```rust
use spark_connect::{SparkSession, functions as f, lit};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark = SparkSession::builder()
        .remote("sc://localhost:15002")
        .get_or_create()?;

    let df = spark
        .range(1_000_000)?
        .select(vec![(f::col("id") * lit(2)).alias("x")])
        .filter((f::col("x") % lit(3)).eq(lit(0)));

    println!("count = {}", df.count()?);
    df.show(20)?;
    Ok(())
}
```

See the [documentation](https://apache.github.io/spark-connect-rust/) for the full
API, running a Spark Connect server, and more.

## Contributing

Issues are tracked in ASF JIRA under
[SPARK](https://issues.apache.org/jira/browse/SPARK) (GitHub Issues are disabled).
See the [contributing guide](https://apache.github.io/spark-connect-rust/contributing/).

## License

Apache License 2.0.

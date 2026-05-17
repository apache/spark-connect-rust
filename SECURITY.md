# Security policy

## Reporting a vulnerability

To report a security vulnerability in apache/spark-connect-rust,
follow the ASF security process at
<https://www.apache.org/security/>. Reports can be sent privately
to security@apache.org or to security@spark.apache.org.

## Threat model

This repository implements the Rust client for the Spark Connect
RPC protocol. Apache Spark's security guidance for operators is
documented at:

<https://spark.apache.org/docs/latest/security.html>

That document covers authentication, encryption, access
control, Kerberos integration, trust boundaries, environmental
assumptions, security properties provided and not provided,
and downstream operator responsibilities. The client-side
trust assumptions for Spark Connect (RPC channel, credential
material, TLS validation) inherit from that model.

Automated security scanning agents should consult that
document for the project's in-scope / out-of-scope
declarations before reporting issues.

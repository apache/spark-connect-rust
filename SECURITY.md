# Security policy

## Reporting a vulnerability

To report a security vulnerability in apache/spark-connect-rust,
follow the ASF security process at
<https://www.apache.org/security/>. Reports can be sent privately
to security@apache.org or to private@spark.apache.org.

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

## Experimental server frontend

The optional `spark-connect-server` library is an experimental protocol
frontend for local, trusted workloads. It does not provide authentication,
authorization or TLS. Session IDs, operation IDs and client-provided user
context identify protocol state; they do not establish a caller's identity
or access rights.

Bind its listener only to loopback. The embedding application supplies the
listener and the `Engine` implementation; the library does not enforce the
listener's network scope. Any process that can reach it can submit requests
to that engine. Protocol limits and session isolation are resource and
correctness guards, not a security boundary between mutually untrusted users.

This frontend does not include an execution backend or a standalone server.
Its injected-engine tests establish frontend protocol behavior, not safe
exposure of an execution engine to untrusted clients. Apache Spark's operator
security guidance remains relevant to any application that connects an engine.

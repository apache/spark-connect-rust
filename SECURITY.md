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

## Experimental JVM server

The optional `spark-connect-server` and `spark-connect-jni` crates are a work in
progress for trusted local clients and query inputs. The launcher defaults to
`127.0.0.1:15002`. This frontend does not implement authentication, authorization,
or TLS. Client-supplied user and session identifiers are metadata, not verified
identities. Keep its listener on loopback; changing the address does not add
network protection or make it suitable for a multi-tenant deployment.

Queries execute with the permissions and resource access of the hosting Spark
process. The native library, Java adapter, Spark assembly and dependencies must
be trusted. Arrow C Data descriptors are an internal interface between that Java
producer and Rust in the same process. Session separation, batch bounds and task
admission do not provide process isolation or complete memory and CPU quotas.

Each engine has fixed capacities of 64 initialized JVM sessions and 64 active
tasks, with a two-second deadline for session/task permit admission. There is no
idle session expiry. Clients that exit without releasing sessions can exhaust
session capacity; unread result streams can retain task capacity. Release known
sessions with `ReleaseSession` and let owned work and deferred cleanup drain to
recover capacity, or restart the server. Abandoned sessions whose identifiers
are lost remain until a restart. These admission bounds do not prevent one
admitted query from consuming the hosting process's available resources.

See the [server guide](crates/spark-connect-server/README.md) for its supported
operations, platform requirements and lifecycle limits. Apache Spark's security
configuration does not automatically apply to this experimental Rust listener.

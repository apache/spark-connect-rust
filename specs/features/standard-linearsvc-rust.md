---
service: spark-connect-rust
created: 2026-11-26
---

# Delegate standard LinearSVC operations to Rust

## Intent

Make standard Python LinearSVC use the existing Rust ML implementation, following
LogisticRegression. Reuse the public adapter and eagerly synchronize Python parameters.

## Context

LogisticRegression already delegates through PyO3 to Rust. LinearSVC has native
operations and bindings, but standard imports still expose upstream Python orchestration.
The classification module already preserves the upstream definitions in a private file.

## Requirements

- R1: Standard imports expose native LinearSVC, LinearSVCModel, LinearSVCSummary,
  and LinearSVCTrainingSummary, identical to the existing fork-native aliases.
- R2: Preserve Python signatures, defaults, type conversion, copying, pickling, and subclass hooks.
- R3: Eagerly synchronize estimator/model parameter mutations, defaults, UID, copies, and restoration;
  retain synchronization before native operations for dictionary edits and bypassing overrides.
- R4: Fit, transform, model access, summaries, evaluation, and persistence use Rust ML operations.
- R5: Retain Python references in native defaults safely, including cyclic references.
- R6: Add no runtime dependencies and preserve unchanged upstream classification source.
- R7: Preserve immediate Python model-reference counts and server-cache cleanup when the last
  native wrapper is deleted, while keeping objects owned by lazy plans and writers alive.

## Clarifications

This is standard pyspark.ml over Spark Connect. Deprecated upstream pyspark.ml.connect
algorithms are excluded; existing fork-specific native imports remain compatibility aliases.

## Acceptance Criteria

- AC1: GIVEN either import order WHEN loading standard and native modules THEN the four
  LinearSVC classes have identical identity, public module names, and native implementations.
- AC2: GIVEN estimator/model mutations WHEN a method returns or a partial conversion fails
  THEN native parameters match Python, with upstream return values and errors preserved.
- AC3: GIVEN copies, default overrides, UID changes, pickle restoration, or subclasses WHEN
  the resulting object is used THEN native state is current and subclass dispatch is preserved.
- AC4: GIVEN a live server with Python ML request builders disabled WHEN fitting, transforming,
  evaluating, or saving/loading THEN Rust ML succeeds and lazy plans retain their model references.
- AC5: GIVEN a standard Pipeline and CrossValidator with LinearSVC WHEN fitting and persisting
  THEN resulting stages use the native model and predictions remain correct.
- AC6: GIVEN blocked optional Python transport/ML dependencies WHEN using LinearSVC THEN
  operations succeed without importing them; upstream standard classification parity tests pass.
- AC7: GIVEN a default containing a reference cycle WHEN its estimator becomes unreachable
  THEN garbage collection releases it; non-migrated classifiers keep existing synchronization.

- AC8: GIVEN fitted models, copies, and training summaries WHEN their last Python owner is deleted
  THEN reference counts and the next server-cache query reflect the deletion without polling
  or forced collection, while a retained lazy transform remains executable.

## Constraints

Python parameter maps remain authoritative. Synchronization is local and needs no server.
Preserve the existing Rust model/session lifetime and the upstream source byte-for-byte.
Use a locally built matching extension for validation. Add no package dependencies.

## Out of Scope

Other classifier migrations, removing upstream Python definitions, and changing Spark's
algorithms or wire protocol.

## Plan

Publish the four LinearSVC classes through the standard classification adapter and opt the
estimator/model into existing eager hooks. Re-export those classes from the compatibility
module. Add native estimator default storage with GC support following LogisticRegression.
Share standard-import tests between the migrated classifier families and preserve each
algorithm's summary and threshold semantics. Refresh generated aliases and typing checks.

## Tasks

- [x] T1: Publish standard LinearSVC aliases and eager synchronization with native default storage.
- [x] T2: Share and extend standard-import, eager-state, subclass, typing, and live routing tests.
- [x] T3: Run offline suites, live tests, upstream parity, generated-stub checks, and Rust gates.
- [x] T4: Create the tracking JIRA and prepare the PR description.

- [x] T5: Resolve CI stub-alias coverage and native model finalization regressions.

### Validation

- Tracking issue: [SPARK-62385](https://issues.apache.org/jira/browse/SPARK-62385), with
  Affects/Fix Version connect-rust-4.4.0 and an is-part-of link to SPARK-61821.
- Built the matching extension locally with Rust 1.100.0 and Python 3.13.
- Offline ML API, subclass, and standard-import suites: 917 passed, 68 skipped.
  A regression test fails with the previous extension because native defaults remain stale.
- Live standard classification and LogisticRegression golden tests: 11 passed on Spark 4.3.0.
  These include disabled Python request builders, summaries, evaluation, persistence, lazy
  model lifetime, Python overrides, Pipeline/CrossValidator, and blocked optional dependencies.
- Official-test runner in drop-in mode: 17 classification tests and 3 OneVsRest tests passed,
  with no skips. These import our package and exercise the Rust ML implementation.
- Affected typing/runtime-import/docstring tests: 3 passed, 1 skipped (pyright unavailable).
  The generated alias stubs and preserved upstream classification source match their references.
- Fresh inspection of all ten migrated classes against official PySpark 4.3.0 found no public-name,
  member-kind, or signature differences. Shared tests retain all pre-existing LogisticRegression
  scenarios and now cover LinearSVC too.
- Ruff, whitespace, and coverage-scope checks passed. Dependency manifests are unchanged.
- Rust library tests: 864 passed. All integration-test targets compiled with default features
  and with wasm-udf. Formatting and workspace Clippy across all targets passed, with Clippy
  run using `-D warnings`.
- Full-project parity and coverage are deferred to CI.

### CI follow-up (November 27)

The first CI run exposed two failures: the stub coverage check counted class declarations but
not the four new LinearSVC aliases, and upstream cache tests exposed native model cleanup
that lagged behind Python reference counts and the next cache query. The alias check now
verifies re-export identity. Native finalizers mirror wrapper reference counts, release their
model borrow and the GIL before waiting for deletion, and retain the nonblocking Rust drop path.
Model fields drop their ownership bookkeeping before the Python summary, so its finalizer
sees the correct count. Regression cases cover model copies and retained training summaries.

Follow-up validation used Python 3.11, matching CI: 2,060 ML offline and type-stub tests passed (73 skipped), 80 live ML lifecycle tests passed, and all 24 upstream cache, classification, and OneVsRest tests passed with no skips. The live regression cases cover ordinary copies, uncounted shallow copies, retained summaries, and immediate cache cleanup. The API comparison found no public-name, member-kind, or signature differences across the 10 migrated classes. Rust validation passed: 864 library tests, all integration-test compilation with and without wasm-udf, cargo fmt, and workspace Clippy with warnings denied. Ruff and whitespace checks passed.

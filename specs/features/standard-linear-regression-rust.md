---
service: spark-connect-rust
created: 2026-11-27
---

# Delegate standard LinearRegression operations to Rust

## Intent
Make the standard pyspark.ml.regression LinearRegression API use the existing Rust ML implementation, with eager parameter synchronization and the public behavior of PySpark 4.3.0.

## Context
LogisticRegression and LinearSVC already use this boundary. LinearRegression has native estimator, model, summary, evaluation, and persistence support, but the standard Python module still uses upstream request construction.

## Requirements
- R1: Publish LinearRegression, LinearRegressionModel, LinearRegressionSummary, and LinearRegressionTrainingSummary under their standard module and preserve compatibility aliases.
- R2: Synchronize constructor values, setters, defaults, clearing, UID changes, copies, and restored state eagerly into native estimators and models.
- R3: Delegate fit, transform, model access, summaries, evaluation, and persistence to Rust without adding optional Python runtime dependencies.
- R4: Preserve Python signatures, documentation, subclass dispatch, copying, pickling, and model ownership, including lazy plans.
- R5: Preserve the unchanged upstream regression definitions and enforce their provenance with the vendor drift check.
- R6: Run upstream regression tests through the drop-in package and exercise the migrated path in CI, including typing and lifecycle tests.

## Clarifications
- Q1: Does this implement a new algorithm or the deprecated pyspark.ml.connect API?
  A1: It migrates the standard Python API to the existing Rust implementation; deprecated upstream pyspark.ml.connect algorithms remain excluded.

## Acceptance Criteria
- AC1: GIVEN either import order WHEN importing the four classes THEN standard and compatibility imports identify the same native-backed public classes.
- AC2: GIVEN an estimator or model WHEN its Python parameters change THEN the native parameter state matches immediately without an RPC.
- AC3: GIVEN Python ML request builders are disabled WHEN fitting, transforming, evaluating, or persisting THEN the native operation succeeds.
- AC4: GIVEN upstream regression tests and API signatures WHEN compared with the drop-in THEN the migrated classes preserve tested behavior and public signatures.
- AC5: GIVEN copies, summaries, and lazy plans WHEN Python model wrappers are deleted THEN native ownership keeps needed models alive and releases unused cache objects.
- AC6: GIVEN optional Python transport and ML dependencies are blocked WHEN running the standard regression API THEN the native operations succeed.

## Constraints
Use the existing native publication and parameter synchronization machinery. Keep upstream definitions byte-identical to the pinned Spark version. Add no runtime dependencies.

## Out of Scope
Migration of other regressors, changes to the server algorithm, and deprecated pyspark.ml.connect implementations.

## Plan

### Approach
Move the unchanged regression definitions into ml/_upstream and load them under the standard module before publishing the four native classes. Re-export those classes from the compatibility module. Extend shared migrated-estimator tests to regression and update typing, documentation, and drift checks.

### Affected Components
Python ML adapters, upstream source mapping, generated type aliases, shared migration tests, and CI test selection. The LinearRegression PyO3 estimator also stores overridden Python defaults and traces those references for garbage collection, as the previously migrated estimators do.

### Data Model Changes
None.

### API Changes
Public signatures are unchanged; standard LinearRegression operations use Rust and setters synchronize eagerly.

### Risks
Import ordering, Python hooks, summary lifecycle, type alias identity, and persistence must remain compatible.

## Tasks
- [x] T1: Publish the standard native regression classes and preserve upstream source provenance.
- [x] T2: Extend shared migration and type-alias tests to LinearRegression, including regression summaries and optional dependency isolation.
- [x] T3: Wire the tests into CI and regenerate affected stubs.
- [x] T-val: Run acceptance criteria verification, upstream regression parity, and required Rust checks.

### Validation

Validated with Python 3.11 and a rebuilt native extension against Spark 4.3.0. The ML offline and typing suites passed 2,079 tests with 73 skips (including five pyright checks unavailable locally). Live routing, persistence, summary, pipeline, copying, and model-lifecycle tests passed 88 tests. The official regression, classification, cache, and OneVsRest drop-in suites passed 34 tests with no skips. The four migrated regression classes have no public-name, member-kind, or signature differences against official PySpark.

Rust validation passed 864 library tests, all integration-test compilation with and without wasm-udf, formatting, and workspace Clippy with warnings denied. Tooling tests passed 23 tests and three subtests. Generated-stub checks, pinned vendor drift checks, Ruff, and whitespace checks passed. A separate 12-class SQL API inventory was unchanged from the base (its existing DataFrame.rdd difference is outside this migration).

Tracked by SPARK-62403 for connect-rust-4.4.0.

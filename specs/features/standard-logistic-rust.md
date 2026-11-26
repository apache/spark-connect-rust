---
service: spark-connect-rust
created: 2026-11-19
---

# Delegate standard LogisticRegression operations to Rust

## Intent

Make the standard Python LogisticRegression API use the existing Rust ML operations.
Scope the migration to one estimator, its model, and its summaries.
Reuse the Rust ML implementation without adding Python runtime dependencies.

## Context

Standard pyspark.ml builds ML requests in Python. Separate native classes already use
Rust for fit, transform, model access, and persistence, while retaining Python parameter maps.

## Requirements

- R1: Standard classification imports expose the native LogisticRegression family.
- R2: Python retains Params, defaults, setters, copying, and subclass dispatch.
- R3: Fit, transform, model access, and persistence use the Rust ML implementation.
- R4: Existing native imports alias the same public classes without replacing registrations.
- R5: Preserve the upstream classification definitions byte-for-byte and check their provenance.
- R6: Add no Python runtime dependencies; native operations must work without Python protobuf,
  gRPC, Py4J, zstandard, scipy, scikit-learn, or torch.
- R7: Preserve composition with standard Pipeline and CrossValidator APIs. Reused upstream
  standard ML tests must import our package so they exercise shared Rust ML operations.
- R8: Eagerly synchronize ordinary parameter mutations for LogisticRegression and
  LogisticRegressionModel only, including construction, setters, defaults, copies, UID changes,
  and state restoration. Preserve Python return values, signatures, and conversion errors.
- R9: Keep operation-time synchronization for direct parameter-dictionary mutations and other
  bypasses. Eager synchronization performs no server calls and adds no session requirement.

## Clarifications

The change covers standard ML over Spark Connect, excluding the deprecated upstream
pyspark.ml.connect public API. Existing fork-specific native imports remain aliases.

## Acceptance Criteria

- AC1: GIVEN standard imports WHEN constructing and copying an estimator THEN its signatures,
  UID, parameters, subclass behavior, and pickle round trips follow PySpark.
- AC2: GIVEN a Spark DataFrame WHEN fitting, transforming, and reading a model THEN native ML
  operations succeed even when the Python ML command and relation builders are disabled.
- AC3: GIVEN a fitted model WHEN saving and loading THEN the public model class and predictions
  are preserved, including after deleting the original model before a DataFrame action.
- AC4: GIVEN either import order WHEN loading standard and native modules THEN each migrated
  class is the same object and native factories return that class.
- AC5: GIVEN the preserved upstream source WHEN checking vendored files THEN its bytes match
  the pinned Spark tag at its original path.
- AC6: GIVEN a fresh process blocking optional Python dependencies WHEN constructing, copying,
  fitting, evaluating, transforming, saving, and loading THEN native LogisticRegression works
  without importing the blocked modules. Package runtime requirements remain unchanged.
- AC7: GIVEN a standard Pipeline and CrossValidator containing LogisticRegression WHEN fitting,
  transforming, and saving/loading the pipeline THEN it contains the native model and preserves
  predictions.
- AC8: GIVEN a LogisticRegression estimator or model WHEN its parameters change through the
  supported mutation methods THEN the native parameter snapshot immediately matches Python
  before any fit, transform, writer, or native property access.
- AC9: GIVEN defaults, parameter overrides, copying, pickling, or a subclass WHEN an updated
  object is returned THEN its native parameters and UID match Python while the original
  object remains unchanged where PySpark leaves it unchanged.
- AC10: GIVEN another native ML class WHEN a setter changes its Python parameter map THEN its
  existing operation-time synchronization behavior is unchanged.

## Constraints

Keep Python parameter maps authoritative. For LogisticRegression and LogisticRegressionModel,
synchronize after supported mutations as well as before native operations.
Keep the existing Rust model lifetime and session handling. No new wire format or runtime flag.
Upstream tests use the drop-in package, not upstream Python with only Rust transport injected.
Python test dependencies do not become package runtime requirements.

## Out of Scope

Eager synchronization of other algorithms, replacing Python parameter dictionaries, and
rewriting upstream Python setters in Rust.

## Plan

Keep the upstream classification source under a private source directory. A maintained public
module loads those definitions with their original module name and replaces only the six
LogisticRegression classes through the existing native-class adapter. Preserve class identity
across both import paths. Update the vendor mapping and exercise standard public imports.
Opt only the estimator and model into eager synchronization through the adapter's parameter
mutation methods; preserve the upstream sources unchanged. Retain the existing operation-time
sync for direct dictionary writes and subclass overrides that bypass those methods.

## Tasks

- [x] T1: Add the public adapter, preserve upstream source, and reuse native class registrations.
- [x] T2: Cover import order, Params, copying, pickling, and subclass dispatch.
- [x] T3: Exercise native routing, model access, persistence, and lazy model lifetime on a server.
- [x] T4: Add fresh-process dependency guards and standard pipeline/tuning coverage.
- [x] T5: Merge current master, preserving native parameter synchronization and model finalizers.
- [x] T6: Fix mapped-vendor lint exclusions, offline coverage selection, and OneVsRest class checks.
- [x] T7: Opt LogisticRegression and its model into eager parameter synchronization.
- [x] T8: Verify immediate native state, mutation errors, copying, subclasses, pickle restoration,
  and isolation from other ML classes.
- [x] T-val: Validate the updated revision with a matching extension, required local Rust gates,
  and affected upstream parity tests.

### Verification of eager synchronization (November 24)

- Built the matching PyO3 extension locally with Rust 1.100.0 and Python 3.13. The temporary
  toolchain and dependency cache are outside the repository; package dependencies are unchanged.
- Offline API, subclass, and standard-import tests: 896 passed, 68 skipped. Tests inspect native
  snapshots directly after mutations, including partial conversion failures, defaults, copies,
  UID updates, pickle restoration, subclass overrides, and reference-cycle collection.
- Live standard LogisticRegression and native golden tests: 6 passed against Spark 4.3.0,
  covering fitted-model eager updates, persistence, lazy model lifetime, Pipeline/CrossValidator,
  and execution with optional Python dependencies blocked.
- The official-test runner in drop-in mode passed all 17 tests in Spark 4.3.0's standard
  classification parity file, with no skips. These tests import our package and native ML core.
  A separate LogisticRegression-focused selection passed 7 tests.
- Affected typing/runtime-import/docstring checks: 3 passed, 1 skipped (pyright unavailable).
  Ruff, generated type-stub drift, and whitespace checks passed.
- The preserved upstream classification source matches Spark 4.3.0 byte-for-byte.
- Rust library tests: 864 passed. All integration tests compiled with both the default
  features and `wasm-udf`. `cargo fmt --all --check` and workspace clippy across all targets
  passed; clippy ran with warnings denied.
- Fresh comparison with official Spark 4.3.0 found no missing public names on either
  LogisticRegression class.
- All 18 CI checks passed on the preceding head `4bfeb76f`. The updated head still needs CI;
  the complete project-wide official suites and coverage are delegated to those checks.

### Historical validation of the November 24 master merge

- Merged Apache master at `d90a730e79fe216dffa9bb66e6a2eff566530e2f`.
- Offline API, subclass, and standard-import tests: 884 passed, 68 skipped, including the
  dependency-blocking subprocess test.
- Ruff, mapped-vendor Ruff exclusions, coverage-scope guards, and generated type-stub drift
  checks passed.
- Runtime dependency declarations are unchanged from master. Relocated upstream classification
  source is byte-identical to the original vendored source.
- Local tests used the CI extension from run 45128869973, commit
  `b6126ae5311ef23cb38842bb1996b60915adec64`, not a build of the updated revision.
- Live tests could not initialize SparkSession, and three typing/runtime-import checks were
  blocked by the same mismatch: this older extension lacks the `_DataSource` binding required
  by current master. The live tests,
  including the new pipeline/tuning and dependency tests, have not passed on this revision.
- Cargo/rustc are unavailable locally. Required Rust compilation, fmt, clippy, full-project
  parity, and coverage gates were unverified locally. All 18 CI checks subsequently passed on
  `4bfeb76f` before the eager-synchronization follow-up.

### Historical validation of the original November 19 draft

These results predate the master merge and do not validate the updated revision.


- Offline API, subclass, and standard-import tests: 874 passed, 68 skipped.
- Live standard LogisticRegression tests and existing golden cases: 4 passed
  against Spark 4.3.0, including fit/transform with Python ML request builders
  disabled, persistence, Python subclass delegation, and lazy model lifetime.
- Spark 4.3.0 official classification tests selected by `logistic or invalid_load`:
  7 passed, 10 deselected. These also cover multinomial logistic regression.
- Affected typing/docstring tests: 4 passed, 1 skipped (pyright unavailable).
  A mypy smoke check covered every public classification class and standard fit typing.
- Vendor drift, generated type-stub drift, Ruff, and whitespace checks passed. The relocated upstream
  classification source is byte-identical to the original vendored source.
- Broader ML offline suite: 416 passed, 3 failed. All three failures reproduce
  on unchanged master with this Python 3.13 environment and compare the pickle
  exception wording (`Can't get local object` vs `Can't pickle local object`).
- Runtime tests used the native extension from project CI run 44477221016,
  commit 47cf486b347bda9675868074f0d12248429f4c2a. Relevant Rust ML sources match
  the draft base; this is not a locally rebuilt current-base extension.
- Cargo/rustc are unavailable, so the required local Rust compilation, fmt,
  and clippy gates could not run. Full-project official parity and coverage
  gates have not been run. This remains a draft for design review.

The standard-import tests cover both reader-first and writer-first initialization.
Shared native reader/writer adapters are registered lazily through the existing
native package initializer, preserving `overwrite()` and the upstream inheritance
before any explicit `pyspark.ml.connect` import by the caller. Pickle restoration
initializes those adapters before invoking the native factory. Static typing exports
remain available for every public classification class, and the existing native
namespace stubs re-export the six migrated classes as aliases.

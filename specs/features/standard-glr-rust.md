---
service: spark-connect-rust
created: 2026-11-27
---

# Delegate standard GeneralizedLinearRegression operations to Rust

## Intent
Make the standard GeneralizedLinearRegression API use the existing Rust implementation, with eager parameter synchronization and PySpark-compatible model and summary behavior.

## Context

Tracking: [SPARK-62407](https://issues.apache.org/jira/browse/SPARK-62407).
LogisticRegression, LinearSVC, and LinearRegression already delegate standard Python operations to Rust. GeneralizedLinearRegression has native estimator, model, summary, evaluation, and persistence implementations; its standard Python import still uses upstream request construction.

## Requirements
- R1: Publish GeneralizedLinearRegression, GeneralizedLinearRegressionModel, GeneralizedLinearRegressionSummary, and GeneralizedLinearRegressionTrainingSummary under their standard module and preserve compatibility aliases.
- R2: Synchronize constructor values, setters, defaults, clearing, UID changes, copying, and restored state eagerly into the native objects, including tracing Python defaults for garbage collection.
- R3: Delegate fit, transform, prediction, summaries, evaluation, and persistence to Rust without adding Python runtime dependencies.
- R4: Preserve public signatures, documentation, Python subclass hooks, copying, and model ownership.
- R5: Validate family/link settings, offsets, weights, residuals, and regression-specific metrics through the existing native implementation and upstream regression suite.

## Clarifications
- Q1: Is this a new algorithm implementation?
  A1: No. It migrates the standard Python API to the existing Rust implementation, continuing the previous migrations.

## Acceptance Criteria
- AC1: GIVEN either import order WHEN importing the four classes THEN standard and compatibility imports identify the same native-backed classes.
- AC2: GIVEN an estimator or model WHEN its parameters, defaults, or identity change THEN native state reflects the change immediately without an RPC.
- AC3: GIVEN Python ML request builders are disabled WHEN fitting, transforming, evaluating, accessing summaries, or persisting THEN the native operations succeed.
- AC4: GIVEN the upstream API and regression tests WHEN compared with the migrated API THEN public signatures and tested regression behavior match.
- AC5: GIVEN copies, summaries, and lazy plans WHEN Python wrappers are deleted THEN native ownership keeps required models alive and releases unused objects.
- AC6: GIVEN optional Python transport and ML dependencies are blocked WHEN running the migrated estimator THEN fit, evaluation, summaries, transform, and persistence succeed.

## Constraints
Reuse the existing publication and eager synchronization machinery. Keep upstream definitions byte-identical to the pinned Spark version. Add no runtime dependencies.

## Out of Scope
Other algorithm migrations, server algorithms, and deprecated upstream pyspark.ml.connect implementations.

## Plan

### Approach
Extend the regression adapter and compatibility aliases to the four GeneralizedLinearRegression classes. Give the native estimator the same Python-default storage, copying, and GC handling as the migrated estimators. Extend shared migration tests and run existing family/link/offset tests and upstream standard regression tests.

### Affected Components
Python regression adapters, the GeneralizedLinearRegression PyO3 estimator, generated type aliases, shared migration tests, and documentation fingerprints.

### Data Model Changes
None.

### API Changes
Public signatures remain unchanged. Standard operations run through Rust; setters synchronize native state eagerly.

### Risks
Default propagation, summary method/property differences, import order, copying, Python hooks, and model lifetime require regression coverage.

## Tasks
- [x] T1: Publish the native standard classes and retain default state in the binding.
- [x] T2: Extend shared migration tests, type aliases, and summary/family coverage.
- [x] T3: Regenerate and verify affected type stubs and documentation fingerprints.
- [x] T-val: Run acceptance criteria verification, upstream ML tests, and mandatory Rust checks.

### Validation results

- Python 3.11 offline and typing suites: 2,099 passed, 73 skipped, including five
  requiring unavailable pyright; mypy used CI's pinned 2.4.1 version.
- Live shared migration and GLR tests against Spark 4.3.0 Connect: 118 passed.
- Upstream standard ML regression, classification, cache, and OneVsRest suites:
  34 passed with no skips.
- Public API comparison across the four migrated GLR classes: zero differences
  in member names, member kinds, and signatures.
- Rust library tests: 864 passed. All integration tests compile in release mode,
  both with and without `wasm-udf`. The PyO3 release build passed.
- Formatting, workspace/all-target Clippy with warnings denied, Ruff, generated
  stubs, vendored upstream drift, and whitespace checks passed.
- Tooling tests: 23 passed plus three subtests.
- SQL API inventory remains unchanged from the baseline; the existing
  `DataFrame.rdd` surface difference is outside this migration.

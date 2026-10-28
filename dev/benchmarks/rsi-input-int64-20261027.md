<!-- SPDX-License-Identifier: Apache-2.0 -->

# Local input and Int64 decoding: recorded RSI observations

Recorded October 27, 2026. These independent experiments motivated this draft PR; neither passed overall acceptance.

Local input conversion builds a second complete row matrix, then clones each column before Arrow conversion. Coerce one column at a time to remove those intermediate copies while retaining the existing coercion rules. For homogeneous Int64 result batches, downcast each column once and preserve the generic decoder for all other batches.

Each optimization was measured **independently** against `74b8e41`; that baseline has exactly the same source tree as master base `c36c846`. Each screen and fresh confirmation used 15 fresh processes per build/case in rotating order. These are observed median changes; the combined PR head has not been benchmarked.

| Workload | CPU, screen / confirmation | Wall, screen / confirmation | Held RSS delta, MiB |
| --- | ---: | ---: | ---: |
| Tuple input, 10,000 × 32 | −10.10% / −13.56% | −10.05% / −13.49% | −28.03 / −28.95 |
| String input, 10,000 × 3 | −9.36% / −8.24% | −9.29% / −8.24% | −3.84 / −3.44 |
| Int64 collect, 4,096 × 512 | −55.08% / −57.34% | −62.35% / −67.50% | +0.0625 / +0.125 |
| Int64 collect + full scan, 4,096 × 512 | −6.76% / −7.09% | −7.03% / −7.27% | +0.0781 / +0.2344 |

**Draft; overall CPU and memory acceptance remain rejected.** The retained Duration(Microsecond) case fails during warmup in both Rust builds, and several timing guards remain uncertain. Available RSS guards pass. The repeated wide-input RSS reduction does not override those failures. Full live Spark parity and CI remain outstanding.

## Provenance and acceptance

- Measured baseline: `74b8e4125ee8c80f4aad51beabfd37277e39215a`.
- Current master base: `c36c846fcb4f22c9995e5de5c69c45ae76a5ad9a`.
- Both complete source trees: `9ce8689c4323b86777b2adff4b3c741c542bea58`.
- [Frozen public harness](https://github.com/dbtsai/pyspark-connect-benchmark/tree/23bdf9046815ae615c921b2e6bab827eed00aaae), macOS arm64 release wheels, isolated Python 3.12 environments. Work was not moved outside the timer.
- Input patch SHA256: `b8ae8c26b6c78c81bc14f6998d9891d24a10d975663e13360343f457f82f62f8`.
- Int64 patch SHA256: `62761085a98f886bce217646fae9ca10ecfed475ebadd59e165200b170f44e65`.
- CPU target: at least 5% supported by the paired exploratory 95% bootstrap interval. Every case guards CPU 5%, wall 10%, and each RSS increase at most the larger of 5% or 4 MiB.
- Peak, held, post-GC and incremental peak RSS are assessed separately. All available successful-operation RSS guards pass; Duration failures have no timed metrics.
- Separate memory criterion: the same target/held-or-peak measure must save both 10% and 8 MiB, supported in screening and independent confirmation, with all raw-sample and CPU/wall/RSS guards also passing. Wide tuple input supports only the target-specific saving; overall memory acceptance remains false. Int64 has no supported saving.
- Per-metric bootstrap intervals are exploratory and are not simultaneous multiple-comparison guarantees. RSS is sampled process memory, not allocated bytes per call.

| Run group | Attempts | Successful | Duration warmup failures | Missing |
| --- | ---: | ---: | ---: | ---: |
| Input screen + confirmation, 27 cases | 2,430 | 2,370 | 60 | 0 |
| Int64 screen + confirmation, 29 cases | 2,610 | 2,550 | 60 | 0 |
| Initial same-binary target controls | 180 | 180 | 0 | 0 |
| Final same-binary full-matrix control | 1,305 | 1,275 | 30 | 0 |

Counts include official Python reference samples. Each input run records 15/15 Rust failures per build for `types-mixed-record-w256-collect-scan`; Int64/control runs retain the same failures. Error: `Unsupported Arrow type Duration(Microsecond) - cannot convert to Value`. Official Python succeeds. No case or threshold was waived.

## Timing metrics whose guards remain inconclusive

Each listed metric retains its original regression bound. Full raw samples, build identities, per-case reports, intervals and independent audits are preserved in the author's local experiment evidence; they are not included in this source patch.

### Column-at-time input

screen: core-construct-expression-10 / cpu; core-outbound-rpc-expression-10 / wall; core-outbound-rpc-expression-10 / cpu; return-long-4096x512-toArrow / cpu; rpc-tableExists / cpu; rpc-listTables / cpu; types-map-struct-w3-collect-scan / cpu.

confirm: core-outbound-rpc-expression-10 / cpu; rpc-listTables / cpu; types-mixed-d4-innerw16-w256-collect-scan / cpu.

### Int64 batch decoding

screen: core-construct-expression-10 / wall; core-construct-expression-10 / cpu; core-outbound-rpc-expression-10 / cpu; input-rows-explicit-10000x3 / cpu; return-long-4096x512-toArrow / wall; return-long-4096x512-toArrow / cpu; input-type-tuples-string-10000x3 / cpu; rpc-currentDatabase / cpu; rpc-tableExists / cpu; rpc-listTables / wall; rpc-listTables / cpu; types-struct-array-map-w3-tuples-construct / cpu; types-mixed-d16-innerw8-w256-iterator-consume / cpu.

confirm: core-construct-expression-10 / wall; core-construct-expression-10 / cpu; core-outbound-rpc-expression-10 / wall; core-outbound-rpc-expression-10 / cpu; input-rows-explicit-10000x3 / cpu; return-long-4096x512-toArrow / wall; return-long-4096x512-toArrow / cpu; rpc-currentDatabase / cpu; rpc-tableExists / cpu; rpc-listTables / cpu; types-map-struct-w3-tuples-construct / cpu; types-array-struct-w3-tuples-construct / cpu.

Int64 confirmation expression-construction CPU was +5.83% and outbound expression CPU +6.48%; their intervals cross the allowed bounds. They remain unresolved. A subsequent full-matrix identical-binary control itself showed expression CPU +14.37% and wall +11.60%, with 12 inconclusive timing metrics. This records variability and does not prove candidate regressions were noise or change either rejection.

## Correctness and interpretation

Each separate candidate passed 416 Rust library/golden/sharing checks and independent Python/API comparisons. The combined draft adds a Long/null/extrema and nested-mixed encode/decode round trip and receives fresh correctness/build/API/Python checks; see the PR for the exact result. Production logic matches the independently measured patches; differences are test comments and the new round trip. There are no combined-head timing results.

Collect CPU in Int64 confirmation was 35.030 → 14.945 ms. Collect plus a full Python value scan was 305.330 → 283.676 ms: roughly 7%, not 57%, lower CPU on the scanned path. The 100,000 × 50 collect guard observed CPU −51.27% / −50.73%, wall −56.26% / −55.88%.

Wide input held RSS was 183.91 → 155.88 MiB and 184.23 → 155.28 MiB. Supplementary allocator diagnostics repeated five rounds: 340,405 → 404 calls, 42,364,072 → 17,984,072 requested bytes. These are diagnostic counts, not process RSS.

The fixture returns canned Arrow/gRPC responses without executing Spark plans. Selected 27/29-case coverage is not the full benchmark catalog; live-server parity was unavailable. No accepted overall speedup, accepted memory saving, green CI or merge readiness is claimed.

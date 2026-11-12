<!-- SPDX-License-Identifier: Apache-2.0 -->

# Client plan representation

`LogicalPlan` is an unresolved query node. It stores the operator payload in
`LogicalPlanKind` and its ordered inputs as shared `Arc<LogicalPlan>` nodes.
Expressions keep their existing representation; name resolution, type inference
and function binding remain server operations.

DataFrames, writers and table arguments share the root node. A transformation
adds its own root and shares the input nodes. Cloning a DataFrame preserves its
root identity; cloning an owned `LogicalPlan` copies its payload while sharing
its children.

## Input contracts

`try_new(kind, children)` and `with_children(children)` accept an
`IntoIterator<Item = Arc<LogicalPlan>>`, including arrays, vectors and
iterators. Construction validates input counts. `children()` exposes inputs as
an immutable slice. A caller cannot mutate a kind or change its input count
without validation. `LogicalPlanKind::arity()` describes the accepted range, and
`InvalidPlanArity` reports the operator, expected bounds and actual count.
Its `expected_max: Option<usize>` is `None` for an unbounded input count and
`Some(max)` for a finite upper bound; `arity()` remains an inclusive range.

Unary operators take one input. Binary operators use left then right;
`CoGroupMap` uses input then other. `GroupMap` accepts its primary input
followed by an optional initial-state input. `WithRelations` is variadic: its
first input is the root and the remaining inputs are references in their wire
order.

`Raw` and `CachedRelation` contain opaque protobuf relations rather than
structural inputs. `MlTransform` keeps a structural input alive, including its
owners, while its existing opaque protobuf supplies the wire relation. Its
structural input is not reserialized into that protobuf. `with_children` accepts
the existing input `Arc` for this operator, but rejects a different allocation
with `PlanRebuildError::OpaqueInputReplacement`, even when the replacement has
equivalent contents. It returns a validated clone of the node; it does not
rewire the opaque relation. Changing that input would require rebuilding its
wire payload. Invalid input counts produce `PlanRebuildError::InvalidArity`.

`Raw` also retains the observation handles of its original DataFrame child
before placeholder resolution. Observation discovery follows PySpark's child
paths: an outer observation and a binary right input take precedence for a
duplicate name. `GroupMap` initial state, the other `CoGroupMap` input,
`WithRelations` references and expression subqueries do not contribute
observations. Shared nodes are inspected once. Cloning and rebuilding a node
keep the same observation handles.

`visit` visits and expands each allocation once, in first-encounter preorder,
with expected O(V+E) work for V nodes and E structural inputs. It deduplicates
by pointer identity, so distinct nodes with equal payloads remain separate.
`visit_all_paths` visits a shared input along every path to it; a compact DAG
that repeats its inputs at each level can require exponentially many visits.
All visitors follow structural inputs in order and leave expressions and opaque
protobuf payloads unvisited.

`with_children` validates replacement inputs and retains the operator payload,
including owners and common metadata held in opaque relations. Even rebuilding
with unchanged inputs clones the payload; the rebuilt node has its own identity
when placed in a new `Arc`, while its unchanged input nodes retain theirs. Reuse
the existing `Arc` to preserve the root's identity and avoid that copying.

On an `Arc<LogicalPlan>`, `try_transform_up` performs an iterative bottom-up
rewrite. It memoizes each original allocation, transforms its children first,
then calls the callback once with a candidate `Arc`. Unchanged child pointers
retain the original node. Changed pointers rebuild it through `with_children`.
Repeated inputs share the same transformed allocation. The callback can retain
the candidate `Arc` or return a replacement; returned replacement subtrees are
not traversed again. If an `MlTransform`'s transformed direct input is a
different `Arc` allocation, rebuilding fails with `OpaqueInputReplacement`
before calling the callback for that `MlTransform` or its ancestors. The
callback cannot recover by replacing the whole `MlTransform` at that point.
It can replace a whole `MlTransform` when its direct input remains unchanged.

The callback returns `Result<Arc<LogicalPlan>, E>`, where
`E: From<PlanRebuildError>`. A no-op callback, `|node| Ok(Arc::clone(node))`,
preserves every node's identity. Specify the error type when the callback does
not otherwise determine it. For example, this rewrite replaces a shared leaf
once and preserves the sharing of the union's inputs:

```rust
use std::sync::Arc;
use spark_connect::plan::{self, LogicalPlanKind, PlanRebuildError, SetOpType};

let shared = Arc::new(plan::range(0, 10, 1));
let root = Arc::new(plan::set_operation(
    Arc::clone(&shared),
    Arc::clone(&shared),
    SetOpType::Union,
    true,
    false,
    false,
));
let mut replaced = 0;
let rewritten = root
    .try_transform_up::<PlanRebuildError>(|node| {
        if matches!(node.kind(), LogicalPlanKind::Range { .. }) {
            replaced += 1;
            Ok(Arc::new(plan::range(0, 20, 1)))
        } else {
            Ok(Arc::clone(node))
        }
    })
    .expect("the union accepts its rewritten inputs");

assert_eq!(replaced, 1);
assert!(Arc::ptr_eq(&rewritten.children()[0], &rewritten.children()[1]));
assert!(!Arc::ptr_eq(&root, &rewritten));
```

## Serialization, metadata and identity

The Connect protobuf remains the server boundary. `to_proto()` creates a fresh
protobuf tree with the existing operator payloads and input ordering. The
fallible `try_to_proto()` and request paths continue enforcing the existing plan
and expression depth bounds. Iterative plan destruction releases structural
inputs without recursively dropping a deep chain. Opaque relation owners remain
alive for as long as the plans that hold them.

`Raw`, `CachedRelation` and `MlTransform` hold their protobufs in an
`OpaqueRelation`. `OpaqueRelation::new` accepts an owned or boxed protobuf
relation, measures its relation depth once, and keeps both the relation and
that depth private. `as_proto()` borrows the relation immutably;
`into_proto()` returns the owned protobuf without cloning it. To modify an
opaque relation, consume the wrapper, edit the protobuf, then construct a new
wrapper so its measured depth stays current.

Cloning an `OpaqueRelation`, a node or a standalone kind reuses the
wrapper's cached depth and grows the stack for its protobuf clone. These
clones do not remeasure the opaque relation. A custom `Raw` node's
caller-declared depth still controls request depth validation. The measured
physical depth sizes protobuf clones only; it does not replace that
declaration when validating requests. Automatic `WithRelations` wrappers can
add physical relation nesting without adding logical plan levels. Cloning
protobuf expressions and nested data types retains its existing recursive
behavior. Debug formatting, protobuf encoding and protobuf destruction also
retain recursive limits. Debug formats shared inputs on every occurrence, so
output can grow exponentially in the number of distinct nodes. The
structural destruction guarantee does not cover arbitrary opaque protobuf
payloads.

`kind()` borrows the operator payload. `into_parts()` consumes an owned node
and returns its `(LogicalPlanKind, Vec<Arc<LogicalPlan>>)` without cloning the
payload or inputs. This lets a caller edit the kind and pass the parts back to
`try_new`, which validates the input count. The caller must preserve the
correspondence between an `MlTransform` input and its opaque protobuf payload.
Input allocations and opaque owners are retained; putting the rebuilt node in a
new `Arc` gives its root a new request identity. The returned inputs still use
iterative plan destruction when dropped.

Common metadata retains each operator's existing serialization behavior. `Raw`
preserves present common metadata and inserts default common metadata when it
is absent. `CachedRelation` always emits fresh default common metadata.
`MlTransform` preserves its protobuf's common metadata, including its absence.

Request serialization retains the session's existing `PlanIds` registry, which
identifies shared nodes by their `Arc` allocation. Repeated serialization of a
node uses the same ID. DataFrame-bound columns, subquery references, placeholder
resolution and cached relation owners retain their existing behavior. Request
serialization assigns stable node IDs even when an opaque protobuf already
contains a plan ID. Standalone `to_proto()` uses the operator's common metadata
behavior described above. Expression subqueries can still introduce
`WithRelations` wrappers with separate root and wrapper IDs.

## Rust API migration

High-level DataFrame methods and the existing `plan::*` builder helpers retain
their signatures. Direct construction and pattern matching on the former
`LogicalPlan` enum are a Rust source API change. Construct a checked node from a
kind and its inputs, and match its kind separately:

```rust
use std::sync::Arc;
use spark_connect::column::col;
use spark_connect::plan::{self, LogicalPlan, LogicalPlanKind};

let input = Arc::new(plan::range(0, 10, 1));
let projected = LogicalPlan::try_new(
    LogicalPlanKind::Project {
        columns: vec![col("id")],
    },
    [input],
).expect("Project requires one input");

match projected.kind() {
    LogicalPlanKind::Project { columns } => assert_eq!(columns.len(), 1),
    _ => unreachable!(),
}
assert_eq!(projected.children().len(), 1);

let (kind, children) = projected.into_parts();
let rebuilt = LogicalPlan::try_new(kind, children).expect("the input count is unchanged");
assert_eq!(rebuilt.children().len(), 1);
```

The kind and error types are non-exhaustive so callers should allow future
additions.

## Validation

The existing golden and serialization suites continue checking wire payloads.
The structural suite lives in the test-only `plan::structure_tests` module. One
fixture declaration generates both the constructor inventory and an exhaustive
match on `LogicalPlanKind`, so adding an operator requires a fixture for tests
to compile. Each fixture independently states its expected arity and checks
serialized input order, including optional and variadic inputs.

The suite also checks sharing, owned extraction without payload copying,
replacement, both traversal modes, bottom-up rewrites, public error behavior,
opaque owners and exact common metadata. These metadata checks are separate
because normalized golden fixtures discard common metadata and plan IDs.
Existing tests continue covering stable IDs, subquery references, placeholder
resolution and depth limits. Deep unary chains and shared fan-out DAGs exercise
destruction on a small stack; a deep shared DAG also checks unique traversal
without expanding every occurrence.

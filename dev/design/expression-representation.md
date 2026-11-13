<!-- SPDX-License-Identifier: Apache-2.0 -->

# Client expression representation

`Expression` shares an immutable node whose operator payload and ordered inputs
are separate. The stored fields are private:

```text
Expression -> Arc<ExpressionNode>
ExpressionNode {
    kind: ExpressionKind,
    children: SmallVec<[Expression; 2]>,
    ty: ExpressionType,
    origin: Option<Arc<Origin>>,
}
```

Cloning an expression shares its complete node, including its origin.
`ptr_eq` checks that identity; value equality compares payloads and children,
ignores origins and does not shortcut shared NaNs. `kind()`, `children()` and
`ty()` borrow the stored representation. `node()` borrows the enclosing struct.
This layout gives structural consumers one input interface without moving
server analysis into the client.

## Construction and rebuilding

`ExpressionKind` holds payloads without structural expression inputs.
`Expression::try_new(kind, children)` accepts any
`IntoIterator<Item = Expression>` and validates the input count and checked
segment counts. It does not validate expression semantics. The
`InvalidExpressionInputs` error distinguishes an invalid input count from
overflow in segment counts. `ExpressionKind::arity()` returns the checked
inclusive range; an error's `expected_max: None` means an unbounded count.

`ExpressionSpec` retains the typed builders previously named `ExpressionNode`.
Pass a specification to `Expression::new`, or convert it with `.into()`; it is
normalized into the canonical payload and inputs rather than stored alongside
them. The `literal`, `column` and `function` convenience constructors remain.
`to_spec()` reconstructs an owned builder specification, sharing its expression
inputs and cloning payloads. Prefer borrowed inspection when no builder is
needed.

`with_children(children)` checks the replacement layout and preserves the
payload and original construction origin. Its new root has its own allocation,
even when every supplied child is unchanged. Clone the existing handle to
preserve its root identity. New construction captures the current origin.

`ExpressionType::Known` records an explicitly declared type: a typed null's
type, `CastTarget::Type`, or a Python UDF's output type. Other expressions use
`ExpressionType::Unresolved`, including casts expressed as type strings.
This field does not infer function return types, resolve names, or validate
coercions. Construction computes this immutable type metadata from the payload;
input replacement shares it because the payload is unchanged.

UDF payloads and declared types are shared across input replacements, so
rewriting arguments copies neither the pickled command nor the output schema.
Reconstructing an owned `ExpressionSpec` copies
that payload as required by the builder's ownership contract. Rebuilding typed
nulls and structured casts guards each recursive declared-type clone level;
arbitrary JSON field metadata retains its existing clone behavior.

## Ordered inputs

| Kind | Structural input order |
| --- | --- |
| Literal, column reference, star, regex, SQL, lambda variable, Sent | None |
| Alias, Cast, DirectShufflePartitionId, SortOrder, NamedArgument | One child |
| UnresolvedFunction, CallFunction, inline Python UDF | Arguments in call order |
| UnresolvedExtractValue | Source, then extraction key |
| UpdateFields | Struct, then value when present |
| CaseWhen | Condition/value pairs, then the else value when present |
| WindowExpression | Window function, partition expressions, then ordering base expressions |
| LambdaFunction | Body only; binder declarations remain payload metadata |
| Subquery | Partition expressions, ordering expressions, then IN values |

CASE branch counts and optional-value flags describe the flattened layout.
Window partition counts and `SortOrderProperties` identify its segments;
directions, null ordering and frame bounds stay in the payload. Window ordering
inputs must be base expressions without `SortOrder` wrappers; `try_new` and
`with_children` check their count but do not enforce this semantic invariant.
Subquery ordering inputs may be explicit `SortOrder` expressions or plain base
expressions, which serialize with ascending order and nulls first. A subquery's
segment counts, type and single-partition flag stay with its plan dependency.
Its plan `Arc` and session plan ID are retained together when expression inputs
are rebuilt. The referenced plan is not an expression child.

## Traversal and rewriting

`visit` visits each allocation once in first-encounter preorder, following
inputs in order. It uses expected O(V+E) work for V nodes and E input edges.
Equal nodes with distinct allocations are
visited separately. `visit_all_paths` visits every input occurrence; a shared
DAG can therefore produce exponentially many visits.

`try_transform_up` rewrites inputs before their parent and memoizes each
original allocation. The callback sees the candidate with its rewritten
inputs, once per original node. Unchanged input identities retain the original
node and origin; changed inputs rebuild through `with_children`. Shared inputs
share their transformed replacement. Callback-created subtrees are not visited
again. Callback and rebuild errors propagate through `E`, which must implement
`From<InvalidExpressionInputs>`.

For example, a no-op rewrite retains the root's allocation:

```rust
use spark_connect::expression::{Expression, InvalidExpressionInputs};

let shared = Expression::column("value");
let root = Expression::function("+", vec![shared.clone(), shared]);
let rewritten = root
    .try_transform_up::<InvalidExpressionInputs>(|expression| Ok(expression.clone()))
    .expect("unchanged inputs retain their layout");
assert!(root.ptr_eq(&rewritten));
assert!(rewritten.children()[0].ptr_eq(&rewritten.children()[1]));
```

Visitors and rewrites follow structural expression inputs only. Subquery-owned
plans, lambda binder declarations, nested literal/type payloads and opaque
`Sent` protobuf contents are excluded.

## Serialization and limits

The Spark Connect protobuf remains the server boundary. Builder normalization
preserves expression ordering, origin metadata, subquery references and request
depth accounting. A `Sent` expression retains its existing protobuf,
rendering and declared levels; changing structural inputs does not edit that
opaque protobuf.

The canonical `SentExpressionPayload` is immutable and shared. Normalization
measures its physical protobuf depth once, and subsequent clones use that cached
depth to size their stack guard. Its retained logical expression levels still
control request depth checks. The owned `SentExpression` builder has public
mutable fields, so cloning that builder measures its current protobuf again.

Structural traversal, rewriting and destruction are iterative. Guarded cloning
of opaque `Sent` expressions does not establish a general stack bound for
arbitrary expression/type payloads. Debug formatting, value comparison,
protobuf encoding and opaque protobuf destruction retain recursive behavior.
Debug output can repeat shared inputs on every occurrence. Existing request
depth checks still apply; the structural APIs do not make every operation on an
arbitrarily deep expression safe.

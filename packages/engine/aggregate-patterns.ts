'use strict';

// =====================================================================
// aggregate-patterns.ts — structural recognisers for aggregate bodies
// =====================================================================
//
// **Background.** Before P5 of the broadcast/aggregate/batching
// consolidation (TODO-flatppl-js.md "In-flight P1-P9"), THREE layers
// independently recognised the same IR shapes:
//
//   1. `dissolver.ts:884-1044` (`_tryDissolveAggregate`) — matmul /
//      matvec / outer matchers at IR-rewrite time; emits direct
//      `mul(A, B)` IR.
//   2. `sampler-aggregate.ts:150-577` (`AGGREGATE_PATTERNS` table) —
//      matchers at runtime; dispatches to specialised evaluators.
//   3. `ops-declarations.ts:1469-1618` (`_maybeFastBroadcasted`) —
//      broadcast-head recogniser at runtime; re-does what
//      `dissolver._tryDissolveSingleOp` did.
//
// Each carried identical-but-separate structural pattern matchers
// (`_aggregateBodyClassifyA/B/V` in dissolver; `classifyA/B/V`
// inside each AGGREGATE_PATTERNS entry). Drift surface = three places
// to keep in sync; the mean-correction logic, the matmul/matvec
// axis-swapping convention, and the `get`-arity validation all had to
// agree across files.
//
// **P5 contract.** ONE module hoists the structural recognisers; both
// dissolver and runtime consume them. Dissolver maps the classifier
// output to an IR rewrite (mul / matvec / outer); runtime maps the
// same output to its specialised evaluator. The pure-structural
// classification is bindings-free and side-effect-free.
//
// Per spec §04 §sec:aggregate the matmul-family conventions are:
//
//   - Standard A layout `A[.i, .j]` — output dim first, reduce dim
//     second. Transposed: `A[.j, .i]`.
//   - Standard B layout `B[.j, .k]` — reduce dim first, output dim
//     second. Transposed: `B[.k, .j]`. (Asymmetric "normal" convention
//     vs A — operands have different default axis orderings.)
//   - Scalar multiplication is commutative, so either factor order in
//     the `mul` body matches the same logical product. The classifier
//     tries both (f1=A, f2=B) and (f1=B, f2=A).
//
// SOTA alignment: MLIR's pattern-driven rewrite framework — one
// matcher feeds multiple rewriters (tile / fuse / vectorise / lower-
// to-loops). Our equivalent is one classifier feeding both dissolver
// IR rewrites AND runtime specialiser dispatch.

/**
 * Classify a factor of a matmul body as "A-positioned" (output dim
 * `iName` is one of its two axes). Returns `{src, trans, jName}` where
 * `trans=true` when the iName axis is in the second slot (i.e. the
 * factor is transposed relative to the standard `A[.i, .j]` layout).
 *
 * Returns null when the factor isn't a 2-axis `get(_, .x, .y)` call
 * matching the position.
 */
function classifyAxisGetA(
  fac: any, iName: string,
): { src: any; trans: boolean; jName: string } | null {
  if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
  if (!fac.args || fac.args.length !== 3) return null;
  const s0 = fac.args[1], s1 = fac.args[2];
  if (!s0 || s0.kind !== 'axis') return null;
  if (!s1 || s1.kind !== 'axis') return null;
  if (s0.name === iName) return { src: fac.args[0], trans: false, jName: s1.name };
  if (s1.name === iName) return { src: fac.args[0], trans: true,  jName: s0.name };
  return null;
}

/**
 * Classify a factor of a matmul body as "B-positioned" (output dim
 * `kName` is one of its two axes). Returns `{src, trans, jName}`
 * where `trans=true` when the kName axis is in the FIRST slot (i.e.
 * the factor is transposed relative to the standard `B[.j, .k]`
 * layout — note the convention is opposite to A's: B's standard puts
 * the output dim second).
 */
function classifyAxisGetB(
  fac: any, kName: string,
): { src: any; trans: boolean; jName: string } | null {
  if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
  if (!fac.args || fac.args.length !== 3) return null;
  const s0 = fac.args[1], s1 = fac.args[2];
  if (!s0 || s0.kind !== 'axis') return null;
  if (!s1 || s1.kind !== 'axis') return null;
  if (s1.name === kName) return { src: fac.args[0], trans: false, jName: s0.name };
  if (s0.name === kName) return { src: fac.args[0], trans: true,  jName: s1.name };
  return null;
}

/**
 * Classify a factor as a 1-axis vector indexing `v[.j]`. Returns the
 * source IR (the container) or null if the factor doesn't match.
 */
function classifyAxisGetV(fac: any, jName: string): any | null {
  if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
  if (!fac.args || fac.args.length !== 2) return null;
  const s0 = fac.args[1];
  if (!s0 || s0.kind !== 'axis') return null;
  if (s0.name !== jName) return null;
  return fac.args[0];
}

/**
 * The axis name of a 1-axis vector indexing `v[.j]`, or null if the
 * factor isn't that shape. Used by the dot-product recogniser, which
 * doesn't know the contraction axis upfront (no output axes to read
 * it from) — it reads the name off the first factor and requires the
 * second to match.
 */
function singleAxisGetName(fac: any): string | null {
  if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
  if (!fac.args || fac.args.length !== 2) return null;
  const s0 = fac.args[1];
  if (!s0 || s0.kind !== 'axis') return null;
  return s0.name;
}

/**
 * Top-level matmul-family recogniser. Given an aggregate body
 * `mul(_, _)` and the output-axis names from `aggregate`'s
 * `output_axes`, return one of:
 *
 *   { kind: 'matmul',  aIR, bIR, transA, transB, jName }
 *   { kind: 'matvec',  aIR, vIR, transA, jName }
 *   { kind: 'outer',   uIR, vIR }
 *   { kind: 'dot',     uIR, vIR, jName }
 *
 * — or null if the body isn't a recognised contraction shape.
 *
 * The classifier is PURELY STRUCTURAL — it doesn't consult bindings
 * or type information. The dissolver wraps this with a type/phase
 * check (operands must be statically-known same-phase arrays);
 * runtime callers consume the result directly.
 */
function classifyMatmulBody(bodyIR: any, outAxes: any[]): any | null {
  if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
  if (!bodyIR.args || bodyIR.args.length !== 2) return null;
  const f1 = bodyIR.args[0], f2 = bodyIR.args[1];

  // Two output axes → matmul or outer.
  if (outAxes.length === 2) {
    const iName = outAxes[0].name;
    const kName = outAxes[1].name;
    if (iName === kName) return null;

    // Matmul: each factor binds one output axis + a shared reduce axis.
    for (const [fA, fB] of [[f1, f2], [f2, f1]]) {
      const ca = classifyAxisGetA(fA, iName);
      const cb = classifyAxisGetB(fB, kName);
      if (!ca || !cb) continue;
      if (ca.jName !== cb.jName) continue;
      // Reduce axis can't shadow an output axis.
      if (ca.jName === iName || ca.jName === kName) continue;
      return {
        kind: 'matmul',
        aIR: ca.src, bIR: cb.src,
        transA: ca.trans, transB: cb.trans,
        jName: ca.jName,
      };
    }

    // Outer product: each factor binds ONE distinct output axis, no
    // reduction axis. Body shape `u[.i] * v[.j]` with .i, .j the two
    // output axes.
    for (const [fU, fV] of [[f1, f2], [f2, f1]]) {
      const uSrc = classifyAxisGetV(fU, iName);
      const vSrc = classifyAxisGetV(fV, kName);
      if (!uSrc || !vSrc) continue;
      return { kind: 'outer', uIR: uSrc, vIR: vSrc };
    }
    return null;
  }

  // One output axis → matvec.
  if (outAxes.length === 1) {
    const iName = outAxes[0].name;
    for (const [fA, fV] of [[f1, f2], [f2, f1]]) {
      const ca = classifyAxisGetA(fA, iName);
      if (!ca) continue;
      const vSrc = classifyAxisGetV(fV, ca.jName);
      if (!vSrc) continue;
      return {
        kind: 'matvec',
        aIR: ca.src, vIR: vSrc,
        transA: ca.trans,
        jName: ca.jName,
      };
    }
    return null;
  }

  // Zero output axes → dot product: `aggregate(sum, [], u[.j] * v[.j])`
  // (the spec's `s[] := u[.i] * v[.i]` full-reduction shorthand). Both
  // factors must be 1-axis gets over the SAME axis. `u[.i] * v[.j]`
  // with DISTINCT axes is a full reduction of the outer product — a
  // different contraction; it stays on the broadcast-reduce default.
  if (outAxes.length === 0) {
    const jName = singleAxisGetName(f1);
    if (!jName) return null;
    const uSrc = classifyAxisGetV(f1, jName);
    const vSrc = classifyAxisGetV(f2, jName);
    if (!uSrc || !vSrc) return null;
    return { kind: 'dot', uIR: uSrc, vIR: vSrc, jName };
  }

  return null;
}

module.exports = {
  classifyAxisGetA,
  classifyAxisGetB,
  classifyAxisGetV,
  singleAxisGetName,
  classifyMatmulBody,
};

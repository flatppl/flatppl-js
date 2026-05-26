'use strict';

// =====================================================================
// ops.ts — unified op declaration registry + dispatcher (Phase 1)
// =====================================================================
//
// One declaration per non-scalar op drives multiple engine surfaces:
//   - the static type signature (consulted by `types.ts`)
//   - the runtime impl (consulted by `sampler.evaluateCall`)
//   - the atom-batched dispatch (consulted by `evaluateExprN` /
//     materialiser per-atom paths)
//   - the builtins / EVALUABLE_OPS catalogue checks
//
// The §17.5 sampler split landed the *structural* decomposition; this
// declaration model is the *semantic* unification — one shape model
// shared between static analysis and runtime dispatch.
//
// Atom-batching is the engine's job, not the op's. Each declaration's
// `logical` takes atom-indep inputs matching its signature; the
// dispatcher recognises atom-batched inputs (shape=[N, …logical]) and
// either calls a `batched` fast-path (when provided) or runs `logical`
// per atom and stitches the per-atom results back into `[N, …]`.
//
// Scope (Phase 1):
//   - Non-scalar value-domain ops with fixed input arities + ranks
//     (cross, self_outer, inv, linsolve, lower_cholesky, ...).
//   - Scalar arith ops stay on ARITH_OPS_N / _SCALAR_PRIM_ARITY.
//   - Higher-order ops (broadcast / aggregate / reduce / scan / filter)
//     and measure-algebra ops are NOT covered here — phases 4-5 lift
//     the model to cover them.
//
// This module is ADDITIVE in Phase 1: it ships the registry +
// dispatcher + conformance harness, but does NOT replace any
// existing ARITH_OPS entries. The transition plan is to declare ops
// here in parallel, conformance-test them against ARITH_OPS, then
// route `evaluateCall` through `dispatch()` once every consumer
// agrees.

const valueLib = require('./value.ts');

// ---------------------------------------------------------------------
// Op declaration shape
// ---------------------------------------------------------------------
//
// A declaration is a plain object:
//
//   {
//     name:      string,                 // op name (matches IR `op` field)
//     signature: TypeSignature,          // {args, kwargs?, result}; logical
//                                        // shape, NO atom-batch axis
//     argRanks:  number[],               // logical rank of each positional
//                                        // arg (drives batch-axis detection)
//     logical:   (...args) => result,    // atom-indep impl; takes inputs
//                                        // matching the signature ranks
//     batched?:  (args, N) => result,    // optional fast-path for atom-
//                                        // batched inputs (shape=[N, …])
//   }
//
// The signature is the same shape `types.SIGNATURE_FACTORIES` returns
// today; the registry just exposes it by name.
//
// `argRanks` is a Phase-1 convenience: derived from the signature in
// principle, but the type AST's shape-variable handling is more
// involved than we need now, so we list ranks explicitly per op until
// the signature-driven derivation matures.

// Op kind discriminator (engine-concepts §18.7):
//
//   - 'fixed-rank'        — each arg has a fixed logical rank; the
//                           dispatcher detects atom-batching as
//                           `rank == argRanks[k] + 1`. Default.
//   - 'rank-polymorphic'  — arg ranks vary per-call (e.g. transpose
//                           accepts vector or matrix; linsolve(A, b)
//                           takes b as either). The dispatcher does
//                           NOT auto-atom-batch — explicit
//                           `broadcast` wrapping is the contract for
//                           batching such ops. `argRanks` is unused.
//   - 'variadic'          — arg count varies per-call (e.g. cat,
//                           vector). The dispatcher calls `logical`
//                           with all args spread; no atom-batch
//                           detection. `argRanks` is unused.
//   - 'higher-order'      — at least one arg is a callable; the op
//                           threads body-inference / function-call
//                           machinery through dispatch. Deferred —
//                           see §18.7. Engines must surface a clear
//                           error if the dispatcher encounters this
//                           kind before the dedicated dispatch
//                           extension lands.
type OpKind = 'fixed-rank' | 'rank-polymorphic' | 'variadic' | 'higher-order';

interface OpDecl {
  name: string;
  signature?: any;
  // Required for kind='fixed-rank'; ignored for 'rank-polymorphic',
  // 'variadic', 'higher-order'.
  argRanks?: number[];
  kind?: OpKind;                      // default 'fixed-rank'
  logical: (...args: any[]) => any;
  batched?: (args: any[], N: number) => any;
}

// ---------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------

const REGISTRY: Map<string, OpDecl> = new Map();

function register(decl: OpDecl) {
  if (REGISTRY.has(decl.name)) {
    throw new Error('ops.register: duplicate op declaration for ' + decl.name);
  }
  const kind: OpKind = decl.kind || 'fixed-rank';
  // Fixed-rank ops require argRanks; rank-polymorphic / variadic /
  // higher-order ops don't (argRanks is unused for those kinds).
  if (kind === 'fixed-rank') {
    if (!Array.isArray(decl.argRanks)) {
      throw new Error('ops.register: op \'' + decl.name +
        '\' is kind=fixed-rank but argRanks is missing');
    }
    if (decl.signature && decl.argRanks.length !== decl.signature.args.length) {
      throw new Error('ops.register: op \'' + decl.name +
        '\' argRanks.length (' + decl.argRanks.length +
        ') must match signature.args.length (' +
        decl.signature.args.length + ')');
    }
  }
  REGISTRY.set(decl.name, decl);
}

function lookup(name: string): OpDecl | null {
  return REGISTRY.get(name) || null;
}

function isDeclared(name: string): boolean {
  return REGISTRY.has(name);
}

function listDeclared(): string[] {
  return Array.from(REGISTRY.keys());
}

function signatureOf(name: string): any {
  const decl = REGISTRY.get(name);
  return decl ? decl.signature : null;
}

// ---------------------------------------------------------------------
// Runtime dispatch — atom-batched broadcasting
// ---------------------------------------------------------------------
//
// `dispatch(name, args)` is the unified entry point. It:
//   1. Looks up the declaration.
//   2. For each positional arg, determines whether it's atom-indep
//      (shape rank == logical rank) or atom-batched (rank == logical
//      rank + 1, with the leading dim N).
//   3. If every arg is atom-indep, calls `logical(...args)` directly.
//   4. If any arg is atom-batched, all atom-batched args must share
//      the same N. Then:
//        - If `batched` is provided, calls `batched(args, N)`.
//        - Else runs the per-atom fallback: for each i in 0..N,
//          extract atom-i sub-Values from the batched args (keeping
//          atom-indep args unchanged), call `logical`, collect, and
//          pack back into a Value of shape=[N, ...resultShape].
//
// Bare JS arrays / Float64Arrays without a `.shape` field are treated
// as atom-indep at this layer — they don't carry batch info. Callers
// that need atom-batched scalars/arrays must use Values.

// Detect the atom-batch size N for an arg given its declared logical
// rank. Returns:
//   { batched: false }              — atom-indep (matches logical rank)
//   { batched: true,  N: number }   — atom-batched; leading dim is N
//   throws                          — rank doesn't fit either case
function _classifyArg(v: any, logicalRank: number): { batched: boolean; N?: number } {
  if (v == null) return { batched: false };
  // Value: inspect shape vs logical rank.
  if (valueLib.isValue(v)) {
    const rank = v.shape.length;
    if (rank === logicalRank) return { batched: false };
    if (rank === logicalRank + 1) return { batched: true, N: v.shape[0] };
    throw new Error(
      'ops.dispatch: argument rank ' + rank +
      ' incompatible with logical rank ' + logicalRank +
      ' (expected ' + logicalRank + ' or ' + (logicalRank + 1) +
      ' for atom-batched)');
  }
  // Bare numeric — only valid where logicalRank === 0.
  if (typeof v === 'number' || typeof v === 'boolean') {
    if (logicalRank !== 0) {
      throw new Error('ops.dispatch: scalar passed where rank-' +
        logicalRank + ' input expected');
    }
    return { batched: false };
  }
  // Bare typed array / nested JS array — assume atom-indep, ranks not
  // tracked. Phase 1 leaves stricter validation to the op's `logical`.
  return { batched: false };
}

// Extract atom i from a possibly-batched arg. For atom-indep args
// returns the arg unchanged; for atom-batched Values returns a
// sub-Value of shape=`v.shape.slice(1)` backed by a subarray view of
// the underlying Float64Array.
function _argAtAtom(v: any, batched: boolean, i: number): any {
  if (!batched) return v;
  // v must be a Value with shape=[N, …rest]
  const tailDims = v.shape.slice(1);
  const tailLen = tailDims.reduce((a: number, b: number) => a * b, 1);
  // Real part subarray view.
  const subData = v.data.subarray(i * tailLen, (i + 1) * tailLen);
  const sub: any = { shape: tailDims, data: subData };
  // Carry complex imaginary part if present.
  if (v.dtype === 'complex' && v.im) {
    sub.dtype = 'complex';
    sub.im = v.im.subarray(i * tailLen, (i + 1) * tailLen);
  }
  return sub;
}

// Stitch per-atom logical results into a single Value shape=[N, ...].
// Every per-atom result must have the same shape; mismatch is an
// engine bug (the logical impl violated its signature).
function _stackPerAtom(perAtomResults: any[], N: number): any {
  if (N === 0) {
    // Edge case: zero-atom output. Return a Value with shape=[0, …]
    // matching the first dimension of a hypothetical result. Caller
    // shouldn't usually hit this; surface a clear failure if it does.
    throw new Error('ops.dispatch: cannot stack 0 atoms (caller must ensure N > 0)');
  }
  const first = perAtomResults[0];
  let tailShape: number[];
  let isComplex = false;
  if (valueLib.isValue(first)) {
    tailShape = first.shape;
    isComplex = first.dtype === 'complex';
  } else if (first instanceof Float64Array) {
    tailShape = [first.length];
  } else if (Array.isArray(first)) {
    // Nested JS array — flatten to detect shape. We accept this for
    // back-compat with ops that haven't migrated to Values, but the
    // recommended `logical` returns Values for non-scalar outputs.
    let probe: any = first;
    tailShape = [];
    while (Array.isArray(probe) || (probe && probe.BYTES_PER_ELEMENT !== undefined)) {
      tailShape.push(probe.length);
      probe = probe[0];
    }
  } else {
    // Scalar per-atom result (op produces a scalar from per-atom
    // inputs). Output is shape=[N].
    tailShape = [];
  }
  const tailLen = tailShape.reduce((a, b) => a * b, 1);
  const out = new Float64Array(N * tailLen);
  const outIm = isComplex ? new Float64Array(N * tailLen) : null;
  for (let i = 0; i < N; i++) {
    const r = perAtomResults[i];
    if (valueLib.isValue(r)) {
      const base = i * tailLen;
      // Flatten the per-atom Value's data into the output buffer.
      out.set(r.data, base);
      if (outIm && r.dtype === 'complex' && r.im) {
        outIm.set(r.im, base);
      }
    } else if (r instanceof Float64Array) {
      out.set(r, i * tailLen);
    } else if (Array.isArray(r)) {
      // Flatten the nested array atom-by-atom (slower path).
      _writeNested(out, i * tailLen, r);
    } else if (typeof r === 'number' || typeof r === 'boolean') {
      out[i] = +r;
    } else {
      throw new Error('ops.dispatch: unsupported per-atom result type ' + typeof r);
    }
  }
  if (outIm) {
    return valueLib.complexValue(out, outIm, [N, ...tailShape]);
  }
  return { shape: [N, ...tailShape], data: out };
}

// Recursive helper for the nested-array → flat write in _stackPerAtom.
function _writeNested(dst: Float64Array, offset: number, src: any): number {
  if (Array.isArray(src) || (src && src.BYTES_PER_ELEMENT !== undefined)) {
    for (let k = 0; k < src.length; k++) {
      offset = _writeNested(dst, offset, src[k]);
    }
    return offset;
  }
  dst[offset++] = +src;
  return offset;
}

function dispatch(name: string, args: any[]): any {
  const decl = REGISTRY.get(name);
  if (!decl) {
    throw new Error('ops.dispatch: no declaration for op \'' + name + '\'');
  }
  const kind: OpKind = decl.kind || 'fixed-rank';

  // Rank-polymorphic and variadic ops bypass atom-batch detection.
  // The §2.1 shape contract still applies (Values keep their shape),
  // but the dispatcher just hands the inputs to `logical` as-is and
  // the op handles whatever rank/arity it received. Callers that
  // want atom-batched semantics over a rank-polymorphic op wrap
  // with explicit `broadcast(fn(op(_)), atom_batched)` —
  // engine-concepts §18.7.
  if (kind === 'rank-polymorphic' || kind === 'variadic') {
    return decl.logical(...args);
  }

  // Higher-order ops haven't grown dispatch extensions yet; if one
  // reaches this point, surface a clear error rather than silently
  // mis-dispatching.
  if (kind === 'higher-order') {
    throw new Error('ops.dispatch: op \'' + name + '\' has kind=higher-order; ' +
      'higher-order dispatch extension not yet implemented (engine-concepts §18.7)');
  }

  // Fixed-rank dispatch: classify each arg vs declared logical rank,
  // detect atom-batching by leading-axis convention, run logical or
  // batched accordingly.
  const argRanks = decl.argRanks;
  if (!argRanks) {
    throw new Error('ops.dispatch: op \'' + name +
      '\' is kind=fixed-rank but argRanks is missing');
  }
  if (args.length !== argRanks.length) {
    throw new Error('ops.dispatch: op \'' + name + '\' expects ' +
      argRanks.length + ' args, got ' + args.length);
  }
  // Classify each arg; collect the batch size.
  const batchedFlags: boolean[] = new Array(args.length);
  let N: number | null = null;
  for (let k = 0; k < args.length; k++) {
    const c = _classifyArg(args[k], argRanks[k]);
    batchedFlags[k] = c.batched;
    if (c.batched) {
      if (N === null) N = c.N!;
      else if (N !== c.N) {
        throw new Error('ops.dispatch: op \'' + name +
          '\' atom-batch size mismatch (arg ' + k + ' has N=' + c.N +
          ', earlier arg had N=' + N + ')');
      }
    }
  }
  // All atom-indep: one-shot.
  if (N === null) {
    return decl.logical(...args);
  }
  // Atom-batched. Try the fast-path first.
  if (decl.batched) {
    return decl.batched(args, N);
  }
  // Per-atom fallback.
  const perAtom: any[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const sliced = new Array(args.length);
    for (let k = 0; k < args.length; k++) {
      sliced[k] = _argAtAtom(args[k], batchedFlags[k], i);
    }
    perAtom[i] = decl.logical(...sliced);
  }
  return _stackPerAtom(perAtom, N);
}

module.exports = {
  register,
  lookup,
  isDeclared,
  listDeclared,
  signatureOf,
  dispatch,
  // Exported for the conformance harness:
  _classifyArg,
  _argAtAtom,
  _stackPerAtom,
};

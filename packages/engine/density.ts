'use strict';

// =====================================================================
// density.js — batched log-density evaluation with consume/rest
// =====================================================================
//
// The single density implementation for the engine. Every production
// density caller (matLogdensityof, matBayesupdate, profileN-logdensity,
// worker.logDensityN) goes through here. Both batched (N atoms in one
// call) and single-point (N=1) evaluation share the same code path —
// batching is the natural primitive, single-point is the trivial case.
//
// Naming follows the engine-wide convention:
//   - `…N(…count…)`   — batched primitive over `count` atoms
//   - `…Consume…`     — returns a `rest` (the unconsumed value
//                       leftover) alongside the result, for callers
//                       that compose with other consumers
//
// Layered API:
//
//   logDensityConsumeN(ir, value, refArrays, count, opts)
//     → { logps: Float64Array(count), rest }
//     The foundation. Walks the IR structure once (atom-independent
//     consume/rest splitting), evaluates per-leaf logpdf N times
//     (the only per-atom work), threads consumed values into `baseEnv`
//     for downstream sub-walks (env-threading).
//
//   logDensityN(ir, value, refArrays, count, opts) → Float64Array
//     Strict batched wrapper — asserts the value was fully consumed
//     (rest === null). Same shape as the worker's logDensityN message.
//
//   logDensityConsume(ir, value, env, opts) → { logp, rest }
//     Single-point convenience: count=1 wrapper around the foundation.
//
//   logDensity(ir, value, env, opts) → number
//     Single-point strict: count=1 wrapper around logDensityN.
//
// Per-atom variation comes from `refArrays` ({ [name]: Float64Array(N) }):
// the value-position refs that change per atom (typically a prior's
// per-atom θ samples). `baseEnv` is the atom-independent portion
// (session env from fixed-phase bindings, plus values written by
// env-threading from consumed observation fields — these are shared
// across atoms since `value` itself is shared).
//
// Value-shape conventions (what `value` may be at each consumer):
//   - number              — single scalar; fully consumed by a leaf
//   - Float64Array/TypedArray — flat numeric vector; prefix-slice
//   - Array (plain JS)    — same as typed array; `.slice` for rest
//   - Object (plain)      — record by field name; shallow copy minus
//                           consumed key
//
// Refs:
//   - Value-position refs (e.g. `Normal(mu = ref a)`) resolve through
//     baseEnv ∪ refArrays. Per-atom resolution happens at each leaf.
//   - Measure-position refs (e.g. `joint(M1ref, M2ref)`) resolve via
//     opts.resolveMeasureRef(name) → ir | null. Production callers
//     (matBayesupdate / matLogdensityof / matBroadcastLogdensity /
//     materialiser.ts in general) PRE-EXPAND via `orchestrator.expand
//     Measure(input, { derivations, bindings })` (engine-concepts
//     §17.1 unification) before invoking the density walker — refs
//     in measure position then never reach walkAcc. The opt is the
//     escape hatch for callers that feed un-expanded IR directly
//     (test/density.test.ts:317 covers it; tests are today's only
//     resolveMeasureRef consumers).

import type { IRNode } from './engine-types';

const samplerLib = require('./sampler.ts');
const valueLib   = require('./value.ts');
const densityPrims = require('./density-prims.ts');

// =====================================================================
// Shape helpers — atom-independent consume/rest splitting
// =====================================================================

/** True iff `rest` is null/undefined or an empty container. */
function isEmptyRest(rest: any) {
  if (rest == null) return true;
  if (typeof rest === 'number') return false;
  if (rest && rest.BYTES_PER_ELEMENT && typeof rest.length === 'number') {
    return rest.length === 0;
  }
  if (Array.isArray(rest)) return rest.length === 0;
  if (typeof rest === 'object') {
    for (const _k in rest) return false;
    return true;
  }
  return false;
}

/**
 * Consume one scalar entry from the head of `value`. Returns
 * { head: number, rest }. Handles:
 *   - bare number → fully consumed; rest = null
 *   - typed array / plain array of length ≥ 1 → element [0] as head,
 *     remaining as rest (a Float64Array subarray view or array slice)
 */
function consumeScalar(value: any) {
  if (value == null) {
    throw new Error('density: scalar leaf has no entry to consume (value exhausted)');
  }
  if (typeof value === 'number') return { head: value, rest: null };
  // A boolean observation for a scalar discrete leaf (e.g. a non-broadcast
  // `z ~ Bernoulli(t)` scored against `true`/`false`) — coerce to 0/1, the
  // same numeric form the array/typed-array branches below already produce
  // for boolean vectors. Without this a single-coin Bernoulli likelihood
  // threw "cannot consume scalar from value of type boolean".
  if (typeof value === 'boolean') return { head: value ? 1 : 0, rest: null };
  // Shape-explicit Value: a rank-1 Value behaves like a
  // typed array; rank-0 yields head and a null rest. Higher-rank
  // Values (matrix observations) can't be consumed by a scalar leaf.
  if (valueLib.isValue(value)) {
    if (value.shape.length === 0) {
      return { head: value.data[0], rest: null };
    }
    if (value.shape.length === 1) {
      const k = value.shape[0];
      if (k === 0) {
        throw new Error('density: scalar leaf has no entry to consume (vector exhausted)');
      }
      return {
        head: value.data[0],
        rest: k === 1 ? null
          : { shape: [k - 1], data: value.data.subarray(1) },
      };
    }
    throw new Error('density: cannot consume scalar from Value of rank '
      + value.shape.length + ' (shape=' + JSON.stringify(value.shape) + ')');
  }
  if (value && value.BYTES_PER_ELEMENT && typeof value.length === 'number') {
    if (value.length === 0) {
      throw new Error('density: scalar leaf has no entry to consume (vector exhausted)');
    }
    return { head: value[0], rest: value.length === 1 ? null : value.subarray(1) };
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      throw new Error('density: scalar leaf has no entry to consume (array exhausted)');
    }
    return { head: +value[0], rest: value.length === 1 ? null : value.slice(1) };
  }
  throw new Error('density: cannot consume scalar from value of type '
    + (typeof value));
}

/**
 * Pull the leading n-vector off a value, used by multivariate leaves
 * (MvNormal today; Dirichlet/Wishart/LKJ etc. via their direct mat* paths). Returns:
 *
 *   { head: Float64Array(n) or Value shape=[n], rest }
 *
 * `value` may be:
 *   - Value shape=[n] (k===n)        → head=value.data, rest=null
 *   - Value shape=[k] (k>n)          → head=subarray, rest=Value [k-n]
 *   - Float64Array length n          → head=value, rest=null
 *   - Float64Array length k > n      → head=subarray, rest=subarray
 *   - JS array length n              → head=Float64Array.from, rest=null
 *
 * Records / shape>1 Values are not consumable as a vector — the caller
 * should structurally walk those first.
 */
function consumeVector(value: any, n: any) {
  if (value == null) {
    throw new Error('density: vector leaf has no entry to consume (value exhausted)');
  }
  if (typeof value === 'number') {
    if (n !== 1) {
      throw new Error('density: cannot consume vector of length ' + n + ' from scalar');
    }
    const head = new Float64Array(1);
    head[0] = value;
    return { head, rest: null };
  }
  if (valueLib.isValue(value)) {
    if (value.shape.length !== 1) {
      throw new Error('density: consumeVector expects a rank-1 Value, got shape=' +
        JSON.stringify(value.shape));
    }
    const k = value.shape[0];
    if (k < n) {
      throw new Error('density: vector leaf wants ' + n + ' entries, only '
        + k + ' available');
    }
    return {
      head: value.data.subarray(0, n),
      rest: k === n ? null
        : { shape: [k - n], data: value.data.subarray(n) },
    };
  }
  if (value && value.BYTES_PER_ELEMENT && typeof value.length === 'number') {
    if (value.length < n) {
      throw new Error('density: vector leaf wants ' + n + ' entries, only '
        + value.length + ' available');
    }
    return {
      head: value.subarray(0, n),
      rest: value.length === n ? null : value.subarray(n),
    };
  }
  if (Array.isArray(value)) {
    if (value.length < n) {
      throw new Error('density: vector leaf wants ' + n + ' entries, only '
        + value.length + ' available');
    }
    const head = Float64Array.from(value.slice(0, n));
    return {
      head,
      rest: value.length === n ? null : value.slice(n),
    };
  }
  throw new Error('density: cannot consume vector from value of type '
    + (typeof value));
}

/**
 * Pull a leading n×n matrix off `value`, used by Wishart/InverseWishart/
 * LKJ/LKJCholesky leaves. Returns { head: Value shape=[n, n], rest }.
 * Accepts:
 *   - Value shape=[n, n]               → head=value, rest=null
 *   - Value shape=[n*n]                → reshape head to [n, n], rest=null
 *   - Value shape=[k] with k > n*n     → head=Value [n, n], rest=Value [k-n*n]
 *   - nested array of length n         → flatten + wrap
 */
function consumeMatrix(value: any, n: any) {
  if (value == null) {
    throw new Error('density: matrix leaf has no entry to consume (value exhausted)');
  }
  const N2 = n * n;
  if (valueLib.isValue(value)) {
    if (value.shape.length === 2
        && value.shape[0] === n && value.shape[1] === n) {
      return { head: value, rest: null };
    }
    if (value.shape.length === 1) {
      const k = value.shape[0];
      if (k < N2) {
        throw new Error('density: matrix leaf wants ' + n + 'x' + n + ' = '
          + N2 + ' entries, only ' + k + ' available');
      }
      return {
        head: { shape: [n, n], data: value.data.subarray(0, N2) },
        rest: k === N2 ? null
          : { shape: [k - N2], data: value.data.subarray(N2) },
      };
    }
    throw new Error('density: consumeMatrix expects a rank-1 or rank-2 Value, got shape='
      + JSON.stringify(value.shape));
  }
  if (Array.isArray(value) && value.length === n && Array.isArray(value[0])) {
    // nested array
    const flat = new Float64Array(N2);
    for (let i = 0; i < n; i++) {
      const row = value[i];
      if (!Array.isArray(row) || row.length !== n) {
        throw new Error('density: consumeMatrix expects square ' + n + 'x' + n
          + ', got row ' + i + ' length ' + (row && row.length));
      }
      for (let j = 0; j < n; j++) flat[i * n + j] = +row[j];
    }
    return { head: { shape: [n, n], data: flat }, rest: null };
  }
  if (value && value.BYTES_PER_ELEMENT && typeof value.length === 'number') {
    if (value.length < N2) {
      throw new Error('density: matrix leaf wants ' + N2 + ' entries, only '
        + value.length + ' available');
    }
    return {
      head: { shape: [n, n], data: value.subarray(0, N2) },
      rest: value.length === N2 ? null : value.subarray(N2),
    };
  }
  throw new Error('density: cannot consume matrix from value of type '
    + (typeof value));
}

/**
 * Pull a single named field from a record `value`. Returns
 * { head: value[name], rest } where rest is a shallow copy of value
 * with `name` removed (or null when the record was a single-field one).
 * Throws when the key isn't present — the caller's shape didn't match
 * the measure's declared fields.
 */
function consumeField(value: any, name: any) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || value.BYTES_PER_ELEMENT) {
    throw new Error('density: cannot consume named field \''
      + name + '\' from non-record value');
  }
  if (!Object.prototype.hasOwnProperty.call(value, name)) {
    throw new Error('density: record missing field \'' + name + '\'');
  }
  const head = value[name];
  const rest: Record<string, any> = {};
  for (const k in value) {
    if (k !== name && Object.prototype.hasOwnProperty.call(value, k)) rest[k] = value[k];
  }
  return { head, rest: isEmptyRest(rest) ? null : rest };
}

// =====================================================================
// Foundation: batched recursive walker
// =====================================================================
//
// `logDensityConsumeN` walks the IR once (the structural walk is
// atom-independent), accumulating per-atom logpdf contributions into a
// pre-allocated Float64Array. The only per-atom work happens at leaves
// (and at weighted / logweighted operands whose weight expression may
// reference refArrays).
//
// Internal contract:
//   - `acc` is a Float64Array(count) the caller pre-allocates; we add
//     each atom's contribution in place (so composite measures naturally
//     accumulate by recursing with the same acc).
//   - `baseEnv` is the atom-independent env. Env-threading writes the
//     consumed value of an earlier joint-field into baseEnv so later
//     fields' leaf-kwarg refs resolve to the observation (not to a
//     per-atom prior sample).
//   - `refArrays` is { name → Float64Array(count) }. Used only where
//     leaf params (or weight exprs) reference these names; never
//     duplicated structurally.
//   - Returns the (atom-independent) `rest` of `value`.

// ---------------------------------------------------------------------
// Density-walker pipeline (P3b; engine-concepts §18.2 / §20.10)
// ---------------------------------------------------------------------
//
// `logDensityConsumeN(ir, value, refArrays, count, opts)` runs through
// a pipeline of stages. Each stage has signature
//   (ir, value, refArrays, count, opts, next) => { logps, rest }
// where `next(ir, value, refArrays, count, opts)` invokes the rest
// of the pipeline. Stages compose via direct function nesting
// (middleware / MLIR-pass pattern), mirroring the materialiser's
// pipeline (materialiser.ts P3b).
//
// Default pipeline ships with only the terminal `densityCoreStage`
// (which runs the existing acc + walkAcc dispatch), so behaviour is
// unchanged when no stages are registered. Useful pre-dispatch
// stages: density result cache, tracing for profiling, shape
// validation, custom-handler interception for MCMC etc.

interface DensityStage {
  (ir: IRNode, value: any, refArrays: any, count: any, opts: any,
   next: (ir: IRNode, value: any, refArrays: any, count: any, opts: any) => any): any;
  label?: string;
}

const _DENSITY_PIPELINE: DensityStage[] = [];

function registerDensityStage(stage: DensityStage): void {
  if (typeof stage !== 'function') {
    throw new Error('registerDensityStage: stage must be a function');
  }
  _DENSITY_PIPELINE.push(stage);
}

function _resetDensityPipeline(): void {
  _DENSITY_PIPELINE.length = 0;
}

// Terminal stage: existing acc + walkAcc dispatch. Receives `next`
// for signature symmetry but never calls it.
const densityCoreStage: DensityStage = function (ir, value, refArrays, count, opts, _next) {
  const N = count | 0;
  refArrays = refArrays || {};
  const baseEnv = (opts && opts.baseEnv) || {};
  const acc = new Float64Array(N);
  const rest = walkAcc(ir, value, refArrays, N, opts || {}, acc, baseEnv, null);
  return { logps: acc, rest };
};
densityCoreStage.label = 'densityCore';

function _runDensityPipeline(ir: IRNode, value: any, refArrays: any, count: any, opts: any): any {
  let chain: (ir: IRNode, v: any, r: any, c: any, o: any) => any =
    (i, v, r, c, o) => densityCoreStage(i, v, r, c, o, () => {
      throw new Error('density: terminal stage cannot call next');
    });
  for (let i = _DENSITY_PIPELINE.length - 1; i >= 0; i--) {
    const stage = _DENSITY_PIPELINE[i];
    const inner = chain;
    chain = (irX, v, r, c, o) => stage(irX, v, r, c, o, inner);
  }
  return chain(ir, value, refArrays, count, opts);
}

function logDensityConsumeN(ir: IRNode, value: any, refArrays: any, count: any, opts: any) {
  opts = opts || {};
  const N = count | 0;
  // Input-validation gates stay OUTSIDE the pipeline — they're not
  // composable concerns, just argument sanity checks.
  if (N <= 0) throw new Error('density: count must be positive');
  if (!ir) throw new Error('density: missing IR');
  return _runDensityPipeline(ir, value, refArrays, N, opts);
}

// ---------------------------------------------------------------------
// Tracing stage example (density side, P3b observability)
// ---------------------------------------------------------------------
//
// `makeDensityTracingStage(opts?)` mirrors `materialiser.makeTracingStage`
// for density evaluations. Same TraceEntry shape so a unified tracer
// can write materialise + density timings to the same buffer.

interface DensityTraceEntry {
  // Identifier for the density call. Defaults to the IR's `op`
  // field; callers may supply a richer key via opts.keyFn.
  key: string;
  durationMs: number;
  count: number;       // batch size N
  ok: boolean;
  error?: string;
}

interface DensityTracingStageOpts {
  buffer?: DensityTraceEntry[];
  onEvent?: (entry: DensityTraceEntry) => void;
  // Optional clock; same convention as materialiser's tracing stage.
  now?: () => number;
  // Optional key extractor; default uses ir.op || '<noop>'.
  keyFn?: (ir: any) => string;
}

function makeDensityTracingStage(opts?: DensityTracingStageOpts): DensityStage {
  const o = opts || {};
  const buf = o.buffer;
  const onEvent = o.onEvent;
  const now = o.now || (typeof performance !== 'undefined' && performance.now
    ? () => performance.now()
    : () => Date.now());
  const keyFn = o.keyFn || ((ir: any) => (ir && ir.op) || '<noop>');
  const stage: DensityStage = function (ir, value, refArrays, count, ostage, next) {
    const t0 = now();
    const key = keyFn(ir);
    try {
      const r = next(ir, value, refArrays, count, ostage);
      const entry: DensityTraceEntry = {
        key, durationMs: now() - t0, count: count | 0, ok: true,
      };
      if (buf) buf.push(entry);
      if (onEvent) onEvent(entry);
      return r;
    } catch (err: any) {
      const entry: DensityTraceEntry = {
        key, durationMs: now() - t0, count: count | 0, ok: false,
        error: (err && err.message) ? err.message : String(err),
      };
      if (buf) buf.push(entry);
      if (onEvent) onEvent(entry);
      throw err;
    }
  };
  stage.label = 'densityTracing';
  return stage;
}

// In-place recursive accumulator. Adds contributions to `acc` and
// returns rest.
//
// Env-precedence at each leaf (highest first):
//   1. overlay (env-threading: consumed observation field values).
//      Atom-independent. Wins over refArrays because env-threaded
//      values represent the *observed* state of a prior component —
//      not a per-atom prior sample.
//   2. refArrays per-atom values (typically prior θ_i samples).
//   3. baseEnv (session env from fixed-phase bindings).
//
// `overlay` is null when empty; walkLeaf branches on that to skip the
// extra copy when no env-threading is active above us. Composite
// walkers grow the overlay copy-on-write when adding env-threaded
// values.
function walkAcc(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any): any {
  if (ir.kind === 'ref' && ir.ns === 'self') {
    const resolver = opts.resolveMeasureRef;
    if (typeof resolver !== 'function') {
      throw new Error('density: measure ref \'' + ir.name
        + '\' without resolveMeasureRef opt');
    }
    const expanded = resolver(ir.name);
    if (!expanded) {
      throw new Error('density: cannot resolve measure ref \'' + ir.name + '\'');
    }
    return walkAcc(expanded, value, refArrays, N, opts, acc, baseEnv, overlay);
  }
  if (ir.kind !== 'call') {
    throw new Error('density: unsupported IR kind \'' + ir.kind + '\'');
  }
  const op = ir.op;
  // draw(M) / lawof(M) → peel and recurse on M. `obs ~ M` lowers to a binding
  // `obs = draw(M)`; inlined into a MEASURE position (a reified kernel body's
  // record field) it denotes the measure M, not a value-position draw — so for
  // density it is transparent. Mirrors the materialiser's sampler-side peel;
  // position-correct because walkAcc only ever walks measures. (A draw NESTED
  // in a value expression rides the per-atom value evaluator, never reaching
  // here as a top-level measure node.)
  if ((op === 'draw' || op === 'lawof')
      && Array.isArray(ir.args) && ir.args.length === 1) {
    return walkAcc(ir.args[0], value, refArrays, N, opts, acc, baseEnv, overlay);
  }
  const handler: any = op != null ? (OP_HANDLERS as any)[op] : null;
  if (handler) return handler(ir, value, refArrays, N, opts, acc, baseEnv, overlay);

  // Leaf distribution — the only kind not in OP_HANDLERS (because
  // there are many of them and they're discovered via the REGISTRY).
  if (samplerLib.isKnownDistribution(op)) {
    return walkLeaf(ir, value, refArrays, N, opts, acc, baseEnv, overlay);
  }
  throw new Error('density: unsupported measure op \'' + op + '\'');
}

// ---- Per-op handlers (the in-place accumulators) --------------------

function walkLeaf(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  // The only per-atom work in the entire walk: consume one entry from
  // the value, then loop atoms resolving params and adding logpdf to
  // acc[i]. consumeScalar is atom-independent (head + rest are derived
  // from `value`, which is shared).
  //
  // FlatPDL note (spec §07 §sec:measure-eval-prims, engine-concepts §13.6):
  // per-atom dispatch routes through `densityPrims.builtinLogdensityofPositional`
  // so the leaf path explicitly names the FlatPDL primitive — same
  // numbers as `entry.logpdfFn(head, ...params)`, but the primitive
  // is the cross-engine ABI any future codegen path lowers to.
  const entry = samplerLib.lookupDistribution(ir);
  const kernelName: string = (ir as any).op;
  const blp = densityPrims.builtinLogdensityofPositional;
  const refNames = Object.keys(refArrays);
  const overlayKeys = overlay ? Object.keys(overlay) : null;

  // Points-batched mode (broadcast(logdensityof, M, pts), §11/§12):
  // here the N-axis indexes EVALUATION POINTS, not prior atoms — atom
  // i observes `value[i]`. A scalar leaf consumes the whole length-N
  // points vector (no structural rest; M is scalar). Same logpdf
  // catalogue + param resolution as below; only the observation is
  // per-atom instead of shared, so the result is bit-identical to the
  // per-point reference route. (matBroadcastLogdensity guarantees no
  // per-atom prior refs in this mode — the points axis is the only
  // N-axis — so the constant-param path is the normal case; the
  // overlay/baseEnv-resolved param path is kept for generality.)
  if (opts.pointsBatched) {
    const pv = value;
    const ptAt = valueLib.isValue(pv) ? (i: any) => pv.data[i] : (i: any) => pv[i];
    if (refNames.length === 0 && !overlayKeys) {
      const params = samplerLib.resolveParams(ir, entry, baseEnv);
      for (let i = 0; i < N; i++) {
        acc[i] += blp(kernelName, params, ptAt(i));
      }
      return null;
    }
    const callEnv = Object.assign({}, baseEnv);
    if (overlayKeys) {
      for (let j = 0; j < overlayKeys.length; j++) {
        callEnv[overlayKeys[j]] = overlay[overlayKeys[j]];
      }
    }
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < refNames.length; j++) {
        const v = refArrays[refNames[j]];
        callEnv[refNames[j]] = valueLib.isValue(v) ? v.data[i] : v[i];
      }
      if (overlayKeys) {
        for (let j = 0; j < overlayKeys.length; j++) {
          callEnv[overlayKeys[j]] = overlay[overlayKeys[j]];
        }
      }
      const params = samplerLib.resolveParams(ir, entry, callEnv);
      acc[i] += blp(kernelName, params, ptAt(i));
    }
    return null;
  }

  const { head, rest } = consumeScalar(value);
  // Hot path: no per-atom refs AND no overlay → params constant across
  // atoms. Resolve once, broadcast.
  if (refNames.length === 0 && !overlayKeys) {
    const params = samplerLib.resolveParams(ir, entry, baseEnv);
    const logp = blp(kernelName, params, head);
    for (let i = 0; i < N; i++) acc[i] += logp;
    return rest;
  }
  // Per-atom path. callEnv layering (bottom-up): baseEnv, refArrays[i],
  // overlay. Overlay applied last so env-threaded observation values
  // win over per-atom refs.
  //
  // refArrays entries are either Float64Array (scalar-atom parent) or
  // Value (vector-atom parent). Pre-compute the access
  // pattern so the inner loop stays branch-free.
  const callEnv = Object.assign({}, baseEnv);
  const accessors = new Array(refNames.length);
  for (let j = 0; j < refNames.length; j++) {
    const k = refNames[j];
    const v = refArrays[k];
    if (valueLib.isValue(v)) {
      const shape = v.shape;
      if (shape.length === 1) {
        const data = v.data;
        accessors[j] = (i: any) => data[i];
      } else {
        const tailDims = shape.slice(1);
        const tailLen = tailDims.reduce((a: any, b: any) => a * b, 1);
        const data = v.data;
        accessors[j] = (i: any) => ({
          shape: tailDims,
          data: data.subarray(i * tailLen, (i + 1) * tailLen),
        });
      }
    } else {
      const arr = v;
      accessors[j] = (i: any) => arr[i];
    }
  }
  // Apply the (atom-independent) overlay once outside the loop —
  // refArrays writes would otherwise stomp these names per atom.
  // We re-apply inside the loop after refArrays.
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < refNames.length; j++) {
      const k = refNames[j];
      callEnv[k] = accessors[j](i);
    }
    if (overlayKeys) {
      for (let j = 0; j < overlayKeys.length; j++) {
        const k = overlayKeys[j];
        callEnv[k] = overlay[k];
      }
    }
    const params = samplerLib.resolveParams(ir, entry, callEnv);
    acc[i] += blp(kernelName, params, head);
  }
  return rest;
}

function walkWeighted(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any): any {
  // weighted(w, base): adds log(w) per atom. Recurse into base first
  // (with the same acc), then add log(w_i). Negative or zero weights
  // collapse the atom's logp to -Infinity.
  const wIR = ir.args[0];
  const rest: any = walkAcc(ir.args[1], value, refArrays, N, opts, acc, baseEnv, overlay);
  applyAtomScalar(wIR, refArrays, N, baseEnv, overlay, acc, addLogW);
  return rest;
}

function walkLogWeighted(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any): any {
  // logweighted(g, base): adds g directly per atom. -Infinity is
  // permitted; NaN is left as-is (callers detect downstream).
  const gIR = ir.args[0];
  const rest: any = walkAcc(ir.args[1], value, refArrays, N, opts, acc, baseEnv, overlay);
  applyAtomScalar(gIR, refArrays, N, baseEnv, overlay, acc, addRaw);
  return rest;
}

function walkTruncate(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  // truncate(M, S): indicator(S) × base density. Per spec §06 this does
  // NOT normalise — density inside S equals base density; outside S
  // it's -Infinity. The consumed scalar is atom-independent (lives in
  // `value`), so the indicator gate is evaluated once and applied
  // uniformly to acc when out-of-support.
  const parseSet = opts.parseSet;
  if (typeof parseSet !== 'function') {
    throw new Error('density: truncate requires parseSet opt');
  }
  const setDescr = parseSet(ir.args[1]);
  // Unreducible set IR (dynamic / per-atom bounds): scoring while IGNORING
  // the support restriction would return the un-truncated density for
  // out-of-support points — silent-wrong (audit Phase 6, inSet(null)→throw).
  // Fail loud instead; supported forms are named sets + intervals with
  // literal or session-env-resolvable bounds.
  if (setDescr == null) {
    throw new Error('density: truncate set could not be resolved to a '
      + 'structural descriptor (op \''
      + (ir.args[1] && (ir.args[1] as any).op) + '\') — dynamic / per-atom '
      + 'set bounds are not supported in density evaluation');
  }
  const rest = walkAcc(ir.args[0], value, refArrays, N, opts, acc, baseEnv, overlay);
  const consumed = inferConsumedScalar(value, rest);
  if (consumed != null && !inSet(consumed, setDescr)) {
    for (let i = 0; i < N; i++) acc[i] = -Infinity;
  }
  return rest;
}

function walkNormalize(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  // normalize(base) = base / totalmass(base); density shifts by −log Z.
  // A normalize node should not normally reach the worker: the expansion
  // lowers the closed-form-Z case to logweighted(−logZ, inner) directly,
  // and the non-closed-form case carries a massFrom spec the materialiser
  // resolves to the same logweighted form pre-scoring (audit M3,
  // mat-density.resolveNormalizeMasses). An UNRESOLVED massFrom here
  // means a scoring path skipped that resolution — scoring with a silent
  // 0 shift would be the unnormalized density, so fail loud instead.
  if ((ir as any).massFrom) {
    throw new Error('density: normalize with unresolved totalmass'
      + ' (massFrom "' + (ir as any).massFrom.ref + '") reached the worker —'
      + ' the scoring path must resolve it via resolveNormalizeMasses');
  }
  // Legacy bare normalize (no spec): Z treated as 1 — only correct when
  // the inner is already a probability measure.
  return walkAcc(ir.args[0], value, refArrays, N, opts, acc, baseEnv, overlay);
}

function walkJointFieldsOrPositional(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  // record / kwarg-joint: ir.fields = [{name, value: subIR, source?}, …].
  // Consume named fields in declared order; env-thread each consumed
  // head into the overlay so later fields' leaf-kwarg refs to f.name /
  // f.source see the OBSERVED value, even when refArrays also has an
  // entry under that name (e.g. matLogdensityof passes per-atom prior
  // samples under the same binding names that the observation pins).
  if (Array.isArray(ir.fields)) {
    const fields = ir.fields;
    let cur = value;
    let curOverlay = overlay;
    for (let i = 0; i < fields.length; i++) {
      const f = fields[i];
      const { head, rest: outerRest } = consumeField(cur, f.name);
      const innerRest = walkAcc(f.value, head, refArrays, N, opts, acc, baseEnv, curOverlay);
      if (!isEmptyRest(innerRest)) {
        throw new Error('density: field \'' + f.name
          + '\' did not fully consume its value');
      }
      cur = outerRest;
      if (i + 1 < fields.length) {
        // Copy-on-first-write: don't entangle the caller's overlay
        // across sibling joint walks.
        curOverlay = curOverlay ? Object.assign({}, curOverlay) : {};
        curOverlay[f.name] = head;
        if (f.source && f.source !== f.name) curOverlay[f.source] = head;
      }
    }
    return cur;
  }
  // Positional joint: args = [M1, M2, ...]. Independent components
  // (`joint(M1,M2)`) ignore each other's values. Dependent positional
  // jointchain (`jointchain(M, K)` → `joint([base, K-body refing s0])`)
  // needs the consumed scalar of component i threaded under the
  // synthetic name `s{i}` so a later component's kernel-body ref
  // resolves to the OBSERVED value — exactly the `fields`/`source`
  // overlay mechanism, positionally. Threading extra `s{i}` keys is
  // inert for independent joints (they never ref them).
  if (Array.isArray(ir.args)) {
    let cur = value;
    let curOverlay = overlay;
    for (let i = 0; i < ir.args.length; i++) {
      const before = cur;
      cur = walkAcc(ir.args[i], before, refArrays, N, opts, acc, baseEnv,
        curOverlay);
      if (i + 1 < ir.args.length && Array.isArray(before)) {
        const restLen = Array.isArray(cur) ? cur.length
          : (cur == null ? 0 : 1);
        if (before.length - restLen === 1) {     // consumed one scalar
          curOverlay = curOverlay ? Object.assign({}, curOverlay) : {};
          curOverlay['s' + i] = before[0];
        }
      }
    }
    return cur;
  }
  throw new Error('density: joint with neither fields nor args');
}

function walkIid(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  // iid(M, size): prod(size) copies of M's footprint. Size is
  // atom-independent — a value-position expression typically of
  // fixed phase, per spec §06 a positive integer (1-D length) or a
  // vector of positive integers (multi-axis shape). We evaluate
  // against baseEnv (no per-atom or overlay coverage); if a user
  // pattern needs per-atom sizes, ragged storage would be a
  // separate refactor.
  const args = ir.args || [];
  if (args.length !== 2) throw new Error('density: iid expected 2 args, got ' + args.length);
  const sizeVal: any = samplerLib.evaluateExpr(args[1], baseEnv);
  let total = 0;
  if (typeof sizeVal === 'number') {
    total = sizeVal | 0;
  } else if (sizeVal && typeof sizeVal === 'object') {
    let arr: any = sizeVal;
    if (arr.shape && arr.data) arr = arr.data;
    if (arr.BYTES_PER_ELEMENT && typeof arr.length === 'number') {
      total = 1;
      for (let i = 0; i < arr.length; i++) total *= (arr[i] | 0);
    } else if (Array.isArray(arr)) {
      total = 1;
      for (let i = 0; i < arr.length; i++) total *= (arr[i] | 0);
    }
  }
  if (total < 0) throw new Error('density: iid size negative: ' + total);
  let cur = value;
  for (let i = 0; i < total; i++) {
    cur = walkAcc(args[0], cur, refArrays, N, opts, acc, baseEnv, overlay);
  }
  return cur;
}

function walkJointchainStub() {
  // jointchain isn't a derivation-level shape that ever reaches density
  // — orchestrator.expandMeasureIR canonicalises jointchain into
  // `record` (or `joint`-fields) IR before density sees it. Left as a
  // loud error so a future regression here surfaces immediately.
  throw new Error('density: jointchain should have been canonicalised to '
    + 'record/joint by expandMeasureIR before reaching density');
}

// True if the IR expression `body` references any name present in
// `refArrays` (per-atom kernel inputs). ns-agnostic — `evaluateExpr`'s
// resolveRef keys the env by bare name regardless of namespace. Walks
// args/kwargs of the bijection-body expression tree (calls/binops);
// bijection bodies are flat expressions, no nested measure scopes.
function _bijBodyRefsAny(body: any, refArrays: any): boolean {
  if (body == null || typeof body !== 'object') return false;
  if (body.kind === 'ref' && body.name != null
      && Object.prototype.hasOwnProperty.call(refArrays, body.name)) return true;
  if (Array.isArray(body.args)) {
    for (const a of body.args) if (_bijBodyRefsAny(a, refArrays)) return true;
  }
  // Defensive kwargs recursion: bijection bodies are flat deterministic
  // transforms (binops / positional calls), so a kwarg-bearing sub-node does
  // not arise in practice; mirrors the covered args recursion above.
  /* c8 ignore start */
  if (body.kwargs) {
    for (const k of Object.keys(body.kwargs)) if (_bijBodyRefsAny(body.kwargs[k], refArrays)) return true;
  }
  /* c8 ignore stop */
  return false;
}

function walkPushfwd(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  // pushfwd(f, M) density per spec §06: requires f to be a
  // `bijection(f, f_inv, logvolume)` annotation so we have an inverse
  // and a Jacobian volume element. Then:
  //
  //   log p_{f*M}(y) = log p_M(f_inv(y)) − logvolume(f_inv(y))
  //
  // The bijection metadata travels on `ir.bijection`, attached by
  // `expandMeasureIR` when the f arg was a bijection-typed binding.
  // No callback / resolveBijection opt needed — the IR self-describes.
  //
  // The walker:
  //   1. consume y from the head of `value` (atom-independent scalar)
  //   2. compute x = f_inv(y) — atom-independent if f_inv's body
  //      doesn't reference refArrays (common case: standard bijections
  //      like exp/log, affine maps with literal coefficients)
  //   3. recurse on M with x as the consumed value
  //   4. subtract logvolume(x) — either a literal scalar or a function
  //      of x
  //
  // Per-atom-parameterised bijections (where f_inv's body references
  // refArrays — would arise from MvNormal-via-spec-rewrite with per-
  // atom mu/L) aren't supported here yet because they'd push a per-
  // atom value into the recursive walk, which density.js assumes is
  // atom-independent. MvNormal goes via path B (direct REGISTRY
  // entry) so this gap doesn't bite. When it does, extending the
  // recursive walker to accept per-atom values is the natural next
  // step.
  const bij = ir.bijection;
  const M_ir = ir.args[1];
  if (!bij) {
    const fRef = ir.args[0];
    const fName = fRef && fRef.name;
    throw new Error("density: pushfwd of '" + (fName || '<?>') + "' requires a "
      + "bijection annotation — use bijection(f, f_inv, logvolume) to enable "
      + "pushforward density");
  }
  // Phase 5.1 Session 5d commit 3 — registry fast path.
  //
  // When bij carries `registryName` + `paramIRs` (Sessions 5c+5d
  // additive markers), dispatch through the bijection-registry's
  // atom-batched inverse + logDetJ rather than evaluating the AST
  // f_inv against a scalar y. This handles vector-base pushforwards
  // (e.g. lowered MvNormal: `pushfwd(<affine-marked-bij>, iid(Normal,
  // D))`) that the scalar AST walker can't represent today.
  //
  // Sign convention: registry.logDetJ returns log|det J_{f^{-1}}(y)|
  // (= -log|det J_f(x)| = -log|det L| for affine). The scalar AST
  // path below SUBTRACTS forward logvolume per the formula
  //   log p_{f*M}(y) = log p_M(f_inv(y)) − log|det J_f(x)|
  // The registry path ADDS log|det J_{f^{-1}}(y)| — same number,
  // opposite-sign field.
  if (bij.registryName) {
    const fRefForErr = ir.args[0];
    const fNameForErr = (fRefForErr && fRefForErr.name) || '<?>';
    if (!bij.paramIRs) {
      throw new Error("density: pushfwd bijection '" + fNameForErr
        + "' has registryName='" + bij.registryName
        + "' without paramIRs (additive-invariant violation — producer bug)");
    }
    const bijRegistry = require('./bijection-registry.ts');
    const entry = bijRegistry.getBijection(bij.registryName);
    if (!entry) {
      throw new Error("density: pushfwd bijection '" + fNameForErr
        + "' has registryName='" + bij.registryName
        + "' which is not found in bijection-registry");
    }
    if (!entry.atomBatchedInverse || !entry.logDetJ) {
      throw new Error("density: pushfwd bijection '" + fNameForErr
        + "' uses bijection-registry entry '" + bij.registryName
        + "' which is missing inverse or logDetJ (density not "
        + "supported for this bijection yet)");
    }
    // Resolve each paramIR against baseEnv ∪ overlay. Session 5d MVP:
    // atom-independent params only (per-atom params defer to a later
    // session alongside per-atom MvNormal mu/cov). The eval-env wrap
    // mirrors the AST path's finvEnv construction.
    const paramEnv: Record<string, any> = Object.assign({}, baseEnv);
    if (overlay) Object.assign(paramEnv, overlay);
    const params: any = {};
    for (const k of Object.keys(bij.paramIRs)) {
      const v = samplerLib.evaluateExpr(bij.paramIRs[k], paramEnv);
      // Wrap arrays/matrices into Value form for the registry contract.
      // Literals come through as nested arrays for matrices, flat arrays
      // for vectors, or numbers for scalars.
      if (Array.isArray(v) && v.length > 0 && Array.isArray(v[0])) {
        // rank-2 matrix literal — pack into Value.
        const rows = v.length;
        const cols = v[0].length;
        const data = new Float64Array(rows * cols);
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) data[i * cols + j] = v[i][j];
        }
        params[k] = { shape: [rows, cols], data };
      } else if (Array.isArray(v)) {
        params[k] = { shape: [v.length], data: Float64Array.from(v) };
      } else if (typeof v === 'number') {
        params[k] = v;
      } else if (v && v.shape && v.data) {
        params[k] = v;
      } else {
        throw new Error("density: paramIRs." + k + " resolved to "
          + "unsupported type " + (typeof v) + " for registry '"
          + bij.registryName + "'");
      }
    }
    // Discover D from the bijection's shapeContract or — pragmatically
    // for 'affine' — from params.b's LAST dim. Future bijections add
    // shape-contract-driven lookup here.
    //
    // 5f-2 / R4: use the last dim, not shape[0]. For atom-INDEP b=[D]
    // they coincide; for atom-BATCHED b=[N,D] (per-atom mean) the event
    // dim is shape[1]. Reading shape[0] there would mistake N for D and
    // over-consume the variate.
    let D;
    if (bij.registryName === 'affine') {
      if (params.b && params.b.shape && params.b.shape.length >= 1) {
        D = params.b.shape[params.b.shape.length - 1];
      }
    }
    if (typeof D !== 'number') {
      throw new Error("density: pushfwd bijection '" + fNameForErr
        + "' cannot determine D for bijection-registry '"
        + bij.registryName + "'");
    }
    // 5f-2: atom-DEPENDENT params (per-atom mean b=[N,D] or per-atom
    // scale L=[N,D,D]) reaching this single-observation registry path
    // means a marginal density `logdensityof(hierarchical_X, y)` — the
    // density of a pushfwd whose bijection parameters are themselves
    // random. That's a Monte Carlo marginal, not the closed-form
    // single-obs score this path computes; the y-batch widening that
    // would handle it is deferred. Fail with an actionable diagnostic
    // rather than a cryptic registry shape-mismatch.
    const bAtomBatched = params.b && params.b.shape && params.b.shape.length === 2;
    const LAtomBatched = params.L && params.L.shape && params.L.shape.length === 3;
    if (bAtomBatched || LAtomBatched) {
      throw new Error("density: pushfwd bijection '" + fNameForErr
        + "' has atom-dependent registry params (per-atom mean/scale); "
        + "marginal density of a hierarchical pushfwd is a Monte Carlo "
        + "estimate not supported on the closed-form single-observation "
        + "path (deferred — sampling such models works via matPushfwd)");
    }
    // Consume a length-D vector head from value. wrap into [1, D]
    // atom-batched Value for the registry contract (single observation;
    // an N-atom case wired through matLogdensityof would pre-wrap).
    const { head: yHead, rest: yRest } = consumeVector(value, D);
    const yValue = { shape: [1, D], data: yHead };
    let z;
    try {
      z = entry.atomBatchedInverse(yValue, params, /*N=*/1);
    } catch (err) {
      throw new Error("density: pushfwd bijection '" + fNameForErr
        + "' registry call to '" + bij.registryName
        + "'.atomBatchedInverse failed: " + (err as any).message);
    }
    let logDetJ;
    try {
      logDetJ = entry.logDetJ(yValue, params, /*N=*/1);
    } catch (err) {
      throw new Error("density: pushfwd bijection '" + fNameForErr
        + "' registry call to '" + bij.registryName
        + "'.logDetJ failed: " + (err as any).message);
    }
    // Recurse on M scoring at z. The base measure is iid(scalar, D),
    // which walkIid loops over scalar leaves consuming from a length-D
    // value. We hand z.data (Float64Array length D) — walkIid +
    // walkLeaf + consumeScalar peel scalars off in order.
    const zForRecurse = { shape: [D], data: z.data };
    walkAcc(M_ir, zForRecurse, refArrays, N, opts, acc, baseEnv, overlay);
    // ADD logDetJ per atom (sign convention — see comment block above).
    // Float64Array per-atom logDetJ (future: per-atom params) is not yet
    // reachable: the affine-registry entry returns a scalar logDetJ for a
    // static [D,D] scale, so only the `typeof === 'number'` branch executes.
    if (typeof logDetJ === 'number') {
      if (logDetJ !== 0) for (let i = 0; i < N; i++) acc[i] += logDetJ;
    /* c8 ignore start */
    } else {
      for (let i = 0; i < N; i++) acc[i] += logDetJ[i];
    }
    /* c8 ignore stop */
    return yRest;
  }
  // Consume the head — pushfwd's variate footprint matches M's.
  const { head: y, rest } = consumeScalar(value);

  // Latent (kernel-fed) bijection params: when f_inv's or logvolume's body
  // references a per-atom refArrays name (the affine shift/scale ARE the
  // latent kernel inputs — e.g. a location-scale StudentT likelihood with
  // latent loc+scale via locscale → pushfwd), the bijection must be
  // evaluated PER ATOM, not once against baseEnv ∪ overlay. The common case
  // (constant or session-bound bijection params) keeps the atom-independent
  // fast path below, unchanged.
  const refNames = Object.keys(refArrays);
  const needsPerAtom = refNames.length > 0 && (
    _bijBodyRefsAny(bij.fInv.body, refArrays)
    || (bij.logVolume.kind !== 'scalar' && bij.logVolume.body
        && _bijBodyRefsAny(bij.logVolume.body, refArrays)));

  if (needsPerAtom) {
    const overlayKeys = overlay ? Object.keys(overlay) : null;
    for (let i = 0; i < N; i++) {
      // Per-atom env layering (mirrors walkLeaf): baseEnv, then this atom's
      // refArrays values, then overlay (overlay wins).
      const envI: Record<string, any> = Object.assign({}, baseEnv);
      for (let j = 0; j < refNames.length; j++) {
        const v = refArrays[refNames[j]];
        envI[refNames[j]] = valueLib.isValue(v) ? v.data[i] : v[i];
      }
      if (overlayKeys) {
        for (let j = 0; j < overlayKeys.length; j++) {
          envI[overlayKeys[j]] = overlay[overlayKeys[j]];
        }
      }
      envI[bij.fInv.paramName] = y;
      const xi = samplerLib.evaluateExpr(bij.fInv.body, envI);
      // Defensive: a scalar bijection's f_inv always returns a number here;
      // mirrors the identical (also-unreached) guard on the atom-independent
      // fast path below. Vector-valued bijections are rejected upstream, so
      // this throw is not reachable from a valid model.
      /* c8 ignore start */
      if (typeof xi !== 'number') {
        throw new Error('density: pushfwd f_inv returned non-scalar (got '
          + (typeof xi) + '); per-atom or vector-valued bijections not yet '
          + 'supported here');
      }
      /* c8 ignore stop */
      // Score the base measure for THIS atom only (N=1). Slice each refArray
      // to a length-1 view so M's own latent-param refs resolve to atom i.
      const refArraysI: Record<string, any> = {};
      for (let j = 0; j < refNames.length; j++) {
        const k = refNames[j];
        const v = refArrays[k];
        // Vector-atom (shape rank>1) latent refArrays stay Value-typed at the
        // worker boundary; scalar-column latents (the common and tested case)
        // are unwrapped to a Float64Array and take the else branch below. The
        // Value slice mirrors that Float64Array slice; a vector-latent
        // bijection is not currently constructible at this site.
        /* c8 ignore start */
        if (valueLib.isValue(v)) {
          const tailLen = v.data.length / v.shape[0];
          refArraysI[k] = {
            shape: [1, ...v.shape.slice(1)],
            data: v.data.subarray(i * tailLen, (i + 1) * tailLen),
          };
        } else {
          /* c8 ignore stop */
          refArraysI[k] = v.subarray(i, i + 1);
        }
      }
      const accI = new Float64Array(1);
      walkAcc(M_ir, xi, refArraysI, 1, opts, accI, baseEnv, overlay);
      acc[i] += accI[0];
      // Subtract logvolume at xi for this atom.
      let lv: number;
      if (bij.logVolume.kind === 'scalar') {
        lv = +bij.logVolume.value;
      } else {
        if (bij.logVolume.paramName) envI[bij.logVolume.paramName] = xi;
        lv = +samplerLib.evaluateExpr(bij.logVolume.body, envI);
      }
      if (lv !== 0) acc[i] -= lv;
    }
    return rest;
  }

  // Atom-independent fast path (unchanged): compute x = f_inv(y) once.
  const finvEnv = Object.assign({}, baseEnv);
  if (overlay) Object.assign(finvEnv, overlay);
  finvEnv[bij.fInv.paramName] = y;
  const x = samplerLib.evaluateExpr(bij.fInv.body, finvEnv);
  if (typeof x !== 'number') {
    throw new Error('density: pushfwd f_inv returned non-scalar (got '
      + (typeof x) + '); per-atom or vector-valued bijections not yet '
      + 'supported here');
  }
  // Recurse on M scoring at x. x is atom-independent → walks through
  // M's structure with the standard consume/rest semantics.
  walkAcc(M_ir, x, refArrays, N, opts, acc, baseEnv, overlay);
  // Subtract logvolume(x). Scalar logvolume is a uniform shift; a
  // function evaluates body at x.
  if (bij.logVolume.kind === 'scalar') {
    const lv = +bij.logVolume.value;
    if (lv !== 0) for (let i = 0; i < N; i++) acc[i] -= lv;
  } else {
    const lvEnv = Object.assign({}, baseEnv);
    if (overlay) Object.assign(lvEnv, overlay);
    // paramName null means a 0-arg function — a closed-form constant
    // like `fn(log(2.0))`. We still evaluate the body, just without
    // binding x into env. paramName non-null is the 1-arg case where
    // logvolume varies with the bijection's domain point.
    if (bij.logVolume.paramName) lvEnv[bij.logVolume.paramName] = x;
    const lv = +samplerLib.evaluateExpr(bij.logVolume.body, lvEnv);
    if (lv !== 0) for (let i = 0; i < N; i++) acc[i] -= lv;
  }
  return rest;
}

// ---- Marginalising pushforward: change-of-variables + MC-integration --
//
// `mcmarginal{…}` is the density rule for a per-event observation that is
// a pushforward BIJECTIVE in one RETAINED innovation but MARGINALISES a
// latent draw (spec §06 case-3 opt-in; engine-concepts §6). It is the
// density sibling of `walkPushfwd`: the retained-innovation inverse and
// forward LADJ are EXACT (bijection-registry.invertExpr), and only the
// latent is integrated — by Monte Carlo, so M is small (≈100).
//
//   log p(z=d | θ) ≈ logsumexp_m[ logp_ret(u*_m) − ladj_fwd(u*_m) ] − log M
//   u*_m = f⁻¹(d; x_m, θ),  x_m ~ marginal,  m = 1..M.
//
// The node is SELF-CONTAINED (the main-thread composition transform folds
// any deterministic outer maps into `inverseIR`/`ladjIR` and inlines the
// marginal's dist IR), so the worker resolves everything from its session
// env + the threaded RNG — no bindings needed here (the dumb-worker
// contract). Per-event: `walkIid` factorises the iid product and calls us
// once per event, consuming one scalar.
//
// The retained latent is a Uniform innovation on `retainedInterval`
// (logp_ret = −log(hi−lo) inside the fibre, −inf outside). Two atom modes:
//   - ATOM-INDEPENDENT marginal (profile case: θ in the session env) — one
//     MC estimate fanned out to every atom.
//   - PER-ATOM marginal (hierarchical/posterior: θ_i in refArrays, e.g. each
//     prior particle's `pars`) — a common-random-numbers loop estimates log
//     p(z | θ_i) per atom. This is what makes bayesupdate over a reified
//     generative kernel produce a posterior (audit §3 / H1 record-param).

// Numerically-stable log Σ exp over a Float64Array (all −inf ⇒ −inf).
function _logSumExp(a: Float64Array): number {
  let mx = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
  if (!Number.isFinite(mx)) return mx;          // all −inf (or +inf)
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - mx);
  return mx + Math.log(s);
}

// Collect every externally-fed ref name reachable in an IR (for the
// atom-dependence check below). BOTH `self` and `%local` are collected:
// the bayesupdate expansion can name the kernel's parametric input either
// way (`self pars` in the inverse/LADJ but `%local _pars_` in the inlined
// marginal dist), and both are looked up by bare name in env (resolveRef).
function _collectSelfRefNames(ir: any, out: Set<string>): void {
  if (!ir || typeof ir !== 'object') return;
  if (ir.kind === 'ref' && (ir.ns === 'self' || ir.ns === '%local') && ir.name) { out.add(ir.name); return; }
  if (Array.isArray(ir.args)) for (const a of ir.args) _collectSelfRefNames(a, out);
  if (ir.kwargs) for (const k in ir.kwargs) _collectSelfRefNames(ir.kwargs[k], out);
  if (Array.isArray(ir.fields)) for (const f of ir.fields) _collectSelfRefNames(f && f.value, out);
}

// One Monte-Carlo marginal estimate of log p(z | env), drawing M latents from
// `rng`. The retained innovation is inverted EXACTLY (bijection-registry) and
// the latent is integrated by Monte Carlo:
//   logp ≈ logsumexp_m[ logp_ret(u*_m) − ladj_fwd(u*_m) ] − log M
//   u*_m = f⁻¹(z; x_m),  x_m ~ marginalDistIR,  m = 1..M.
// Returns the estimate plus the advanced RNG state. Splitting this out lets
// the per-atom loop reuse the SAME starting `rng` for every atom (common
// random numbers) — see walkMcMarginal.
function _mcMarginalEstimate(node: any, z: number, M: number, rng: any, env: any) {
  // 1. Draw M marginalised latents x_m ~ marginalDistIR (sync, in-worker).
  const drawn = samplerLib.sampleLeafN(rng, node.marginalDistIR, env, [M]);
  const xcol = Float64Array.from(drawn.value);
  // 2. u*_m = inverseIR(z; x), one batched pass (the §20.10 vertical collapse).
  const zcol = new Float64Array(M);
  zcol.fill(z);
  const invRefs: any = { [node.outName]: zcol, [node.marginalRef]: xcol };
  const ustar = _asFloatColumn(samplerLib.evaluateExprN(node.inverseIR, invRefs, M, env), M);
  // 3. ladj_fwd(u*; x), one batched pass.
  const ladjRefs: any = { [node.retainedRef]: ustar, [node.marginalRef]: xcol };
  const ladj = _asFloatColumn(samplerLib.evaluateExprN(node.ladjIR, ladjRefs, M, env), M);
  // 4. Conditional log-density per latent: logp_ret(u*) − ladj_fwd(u*).
  //    Uniform innovation on [lo,hi]: logp_ret = −log(hi−lo) inside, −inf
  //    outside (the support mask bounding the fibre).
  const lo = node.retainedInterval[0], hi = node.retainedInterval[1];
  const logWidth = Math.log(hi - lo);
  const cond = new Float64Array(M);
  for (let m = 0; m < M; m++) {
    const u = ustar[m];
    cond[m] = (u >= lo && u <= hi && Number.isFinite(ladj[m]))
      ? (-logWidth - ladj[m]) : -Infinity;
  }
  // 5. logsumexp over the M innovations − log M.
  return { logp: _logSumExp(cond) - Math.log(M), state: drawn.state };
}

// Per-atom accessor for a refArray entry — mirrors walkLeaf's accessor build.
// A Value unwraps to a scalar (shape [N]) or a leading-axis slice (shape
// [N, …]); a plain JS array (e.g. per-atom record objects from
// measureToPerAtomRecords) indexes directly.
function _refAtomAccessor(v: any): (i: number) => any {
  if (valueLib.isValue(v)) {
    const shape = v.shape, data = v.data;
    if (shape.length === 1) return (i: number) => data[i];
    const tail = shape.slice(1);
    const tailLen = tail.reduce((a: number, b: number) => a * b, 1);
    return (i: number) => ({ shape: tail, data: data.subarray(i * tailLen, (i + 1) * tailLen) });
  }
  return (i: number) => v[i];
}

function walkMcMarginal(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  const node: any = ir;
  const { head: z, rest } = consumeScalar(value);
  const M = ((opts && opts.mcMarginalizationCount) | 0) || 100;
  const rng = opts && opts.mcRng;
  if (!rng) {
    throw new Error('density: mcmarginal requires the worker RNG (opts.mcRng) — '
      + 'the logDensityN handler must thread it (Monte Carlo marginalisation '
      + 'draws latents in-worker)');
  }
  // Atom-dependence: which self-refs in the marginal dist / inverse / LADJ
  // arrive PER-ATOM (in refArrays)? If any, the marginal differs per atom —
  // the hierarchical/posterior case (each prior particle carries its own θ,
  // e.g. `pars`). None → the atom-independent profile case (θ in session env).
  const refKeys = refArrays ? Object.keys(refArrays) : [];
  let perAtomKeys: string[] | null = null;
  if (refKeys.length > 0) {
    const names = new Set<string>();
    _collectSelfRefNames(node.marginalDistIR, names);
    _collectSelfRefNames(node.inverseIR, names);
    _collectSelfRefNames(node.ladjIR, names);
    const pk = refKeys.filter((k) => k !== node.outName && names.has(k));
    if (pk.length > 0) perAtomKeys = pk;
  }

  if (!perAtomKeys) {
    // Atom-independent: one estimate fanned out to every atom (iid ⇒ walkIid
    // sums events). Thread the RNG forward so successive events draw fresh.
    const env = Object.assign({}, baseEnv);
    if (overlay) Object.assign(env, overlay);
    const est = _mcMarginalEstimate(node, z, M, rng, env);
    opts.mcRng = est.state;
    for (let i = 0; i < N; i++) acc[i] += est.logp;
    return rest;
  }

  // Per-atom MC marginal (posterior / hierarchical): for each atom build an
  // env carrying that atom's refs and estimate log p(z | θ_i). COMMON RANDOM
  // NUMBERS — every atom reuses the SAME starting `rng` for this event, so the
  // M latents share standard innovations and differ across atoms ONLY through
  // θ_i. The MC noise is then common-mode and largely cancels in the relative
  // posterior weights (otherwise independent per-atom noise makes the
  // importance weights pathological). The stream is advanced once (post-draw
  // state) so the next event draws fresh latents.
  const accessors: Record<string, (i: number) => any> = {};
  for (const k of perAtomKeys) accessors[k] = _refAtomAccessor(refArrays[k]);
  let advanced = rng;
  for (let i = 0; i < N; i++) {
    const env = Object.assign({}, baseEnv);
    for (const k of perAtomKeys) env[k] = accessors[k](i);
    if (overlay) Object.assign(env, overlay);
    const est = _mcMarginalEstimate(node, z, M, rng, env);
    advanced = est.state;
    acc[i] += est.logp;
  }
  opts.mcRng = advanced;
  return rest;
}

// evaluateExprN may return a Float64Array(count) (batched), a number (atom-
// independent broadcast), or a shape-[count] Value. Normalise to a length-M
// Float64Array for the MC reduction.
function _asFloatColumn(r: any, M: number): Float64Array {
  if (r && r.BYTES_PER_ELEMENT !== undefined && r.length === M) return r;
  if (typeof r === 'number') { const a = new Float64Array(M); a.fill(r); return a; }
  if (r && Array.isArray(r.shape) && r.data && r.data.BYTES_PER_ELEMENT !== undefined
      && r.shape.length === 1 && r.shape[0] === M) return r.data;
  if (Array.isArray(r) && r.length === M) return Float64Array.from(r);
  throw new Error('density: mcmarginal batched evaluation produced an '
    + 'unexpected shape (expected length-' + M + ' column)');
}

// ---- Kernel-broadcast: array-valued independent product measure ----
//
// `broadcast(Dist, c1, c2, …)` (spec §04 sec:higher-order) — independent
// product of `Dist(params_j)` at each j ∈ [0, K). Density at a
// length-K observation y is the SUM of per-element logpdfs:
//
//   logp(y | broadcast(Dist, args)) = Σ_j logpdf_Dist(y[j]; params_j)
//
// Per-atom (outer batch axis i ∈ [0, N)): each parameter expression may
// reference per-atom prior refs (refArrays / overlay), so the effective
// params at (i, j) are evaluated from each arg's resolved shape:
//
//   atom-indep scalar  (number / Value shape=[])    → const value
//   atom-indep K-vec   (Value shape=[K] / Array)    → data[j]
//   per-atom  scalar   (Float64Array(N) / [N])      → data[i]
//   per-atom  K-vec    (Value shape=[N, K])         → data[i*K + j]
//
// This is the natural batched shape (engine-concepts §2.1: leading axis
// is the batch). `evaluateExprN` already produces these shapes; the
// walker just dispatches accessors off them. Singleton intrinsic axes
// (length 1) broadcast across K, mirroring matKernelBroadcast.
//
// Hot path: when NO parameter expression touches a per-atom ref, the
// per-element sum is constant in i — compute it once, fan out to acc[i].
// This is the same atom-indep short-circuit walkLeaf / walkMultivariate
// take when refArrays is empty for them.
//
// FlatPDL note: per-element dispatch goes through
// `densityPrims.builtinLogdensityofPositional`, the same primitive
// walkLeaf uses — same numbers, same cross-engine ABI.
function walkBroadcast(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  const args = ir.args || [];
  if (args.length < 1) {
    throw new Error('density: broadcast requires at least the kernel head arg');
  }
  // Kernel head must name a built-in sampleable distribution. Same
  // restriction matKernelBroadcast / classifyKernelBroadcast accept.
  const hd: any = args[0];
  let kernelName: string | null = null;
  if (hd && hd.kind === 'ref' && hd.ns === 'self') kernelName = hd.name;
  else if (hd && hd.kind === 'lit' && typeof hd.value === 'string') kernelName = hd.value;
  if (!kernelName || !samplerLib.isKnownDistribution(kernelName)) {
    // A user-kernel head (not a built-in distribution) reaching the density
    // path is, in the generative-bodied case (engine-concepts §21), the
    // INTRACTABLE-density situation spec §06 case 3 calls out: the kernel
    // body is `lawof(<value-expr-with-internal-draw>)` — a non-bijective
    // transform that MARGINALISES the internal draw, so its pushforward
    // density has no closed form. We refuse LOUDLY with the actionable
    // §06 guidance rather than ever returning a silent NaN. `resolve-
    // MeasureRef` (when supplied) lets us recognise the user-kernel head;
    // without it, the generic refusal still fires (never silent).
    const resolver = opts && opts.resolveMeasureRef;
    let headIsUserKernel = false;
    if (kernelName && typeof resolver === 'function') {
      try {
        const headIR = resolver(kernelName);
        headIsUserKernel = !!(headIR && headIR.kind === 'call'
          && headIR.op === 'functionof');
      } catch (_) { /* resolver may throw for non-bindings; ignore */ }
    }
    if (headIsUserKernel) {
      throw new Error('density: broadcast head \'' + kernelName + '\' is a '
        + 'user-defined kernel, not a built-in distribution. If its body is a '
        + 'generative value-expression that closes over an internal draw '
        + '(engine-concepts §21), its pushforward marginalises that draw and '
        + 'has NO tractable density (spec §06 case 3: the density of the '
        + 'pushforward of an unannotated non-bijection is a static error). '
        + 'Wrap the deterministic map with bijection(f, f_inv, logvolume) to '
        + 'enable the closed-form pushforward density, or opt into a Monte '
        + 'Carlo density estimate. (Sampling/DRAW of this kernel works via '
        + 'the generative kernel-broadcast executor.)');
    }
    throw new Error('density: broadcast head ' + (kernelName ? "'" + kernelName + "'" : '<?>')
      + ' is not a built-in distribution kernel');
  }
  const entry = samplerLib.lookupDistribution({ kind: 'call', op: kernelName });
  const paramNames: string[] = entry.params;
  const aliases: Record<string, string> = entry.aliases || {};
  const positional: any[] = args.slice(1);
  const kwargsIR: Record<string, any> = (ir as any).kwargs || {};
  const usingKwargs = Object.keys(kwargsIR).length > 0;
  // Map each declared parameter to its IR (kwargs > positional).
  const paramIRs: any[] = new Array(paramNames.length);
  for (let i = 0; i < paramNames.length; i++) {
    const p = paramNames[i];
    if (Object.prototype.hasOwnProperty.call(kwargsIR, p)) {
      paramIRs[i] = kwargsIR[p];
    } else if (aliases[p] && Object.prototype.hasOwnProperty.call(kwargsIR, aliases[p])) {
      paramIRs[i] = kwargsIR[aliases[p]];
    } else if (!usingKwargs && i < positional.length) {
      paramIRs[i] = positional[i];
    } else {
      throw new Error("density: broadcast(" + kernelName + ") missing parameter '"
        + p + "'");
    }
  }
  // Evaluate each parameter expression once over the atom batch. The
  // resulting shape (combined with `_exprUsesAny` to disambiguate
  // length-N atom-indep vectors from per-atom scalars) classifies the
  // accessor pattern.
  const refNames: string[] = refArrays ? Object.keys(refArrays) : [];
  const evalOpts = overlay ? { overlay } : undefined;
  const accessors: Array<(i: number, j: number) => number> = new Array(paramNames.length);
  const perAtomFlags: boolean[] = new Array(paramNames.length);
  let K = 1;
  let anyAtomDep = false;
  for (let pi = 0; pi < paramNames.length; pi++) {
    const pIR = paramIRs[pi];
    const usesAtom = _exprUsesAny(pIR, refNames);
    perAtomFlags[pi] = usesAtom;
    if (usesAtom) anyAtomDep = true;
    const v: any = samplerLib.evaluateExprN(pIR, refArrays, N, baseEnv, evalOpts);
    let intrinsicK = 1;
    let access: (i: number, j: number) => number;
    if (typeof v === 'number') {
      const val = +v;
      access = (_i, _j) => val;
    } else if (typeof v === 'boolean') {
      const val = v ? 1 : 0;
      access = (_i, _j) => val;
    } else if (valueLib.isValue(v)) {
      const shape = v.shape;
      const data = v.data;
      if (shape.length === 0) {
        const val = data[0];
        access = (_i, _j) => val;
      } else if (shape.length === 1) {
        // Length-N is ambiguous when N === K; `usesAtom` is the
        // disambiguator: a per-atom expression always produces leading-N.
        // The canonical `valueLib.isAtomBatched` predicate consults
        // outerRank too (P3) so a producer-tagged value is honoured.
        if (usesAtom && valueLib.isAtomBatched(v, N)) {
          access = (i, _j) => data[i];
        } else {
          intrinsicK = shape[0];
          if (intrinsicK === 1) {
            const val = data[0];
            access = (_i, _j) => val;
          } else {
            access = (_i, j) => data[j];
          }
        }
      } else if (valueLib.isAtomBatched(v, N) && shape.length === 2) {
        intrinsicK = shape[1];
        if (intrinsicK === 1) {
          access = (i, _j) => data[i];
        } else {
          const stride = intrinsicK;
          access = (i, j) => data[i * stride + j];
        }
      } else {
        throw new Error("density: broadcast(" + kernelName + ") parameter '"
          + paramNames[pi] + "' resolved to unsupported shape "
          + JSON.stringify(shape) + ' (v1 supports scalar / [K] / [N] / [N, K])');
      }
    } else if (v && v.BYTES_PER_ELEMENT !== undefined
                && typeof v.length === 'number') {
      const data: any = v;
      if (usesAtom && valueLib.isAtomBatched(v, N)) {
        access = (i, _j) => data[i];
      } else {
        intrinsicK = data.length;
        if (intrinsicK === 1) {
          const val = data[0];
          access = (_i, _j) => val;
        } else {
          access = (_i, j) => data[j];
        }
      }
    } else if (Array.isArray(v)) {
      intrinsicK = v.length;
      if (intrinsicK === 1) {
        const val = +v[0];
        access = (_i, _j) => val;
      } else {
        const arr = v;
        access = (_i, j) => +arr[j];
      }
    } else {
      throw new Error("density: broadcast(" + kernelName + ") parameter '"
        + paramNames[pi] + "' resolved to non-numeric value (type " + (typeof v) + ')');
    }
    accessors[pi] = access;
    if (intrinsicK > 1) {
      if (K === 1) K = intrinsicK;
      else if (K !== intrinsicK) {
        throw new Error("density: broadcast(" + kernelName + ") incompatible "
          + 'collection lengths (' + K + ' vs ' + intrinsicK + ") on parameter '"
          + paramNames[pi] + "'");
      }
    }
  }
  // Variate footprint is K scalars consumed from the head of `value`.
  const { head, rest } = consumeVector(value, K);
  const blp = densityPrims.builtinLogdensityofPositional;
  const P = paramNames.length;
  const params = new Array(P);
  if (!anyAtomDep) {
    // Atom-indep parameters across the entire batch: compute the
    // per-element sum once, fan out into acc[i].
    let total = 0;
    for (let j = 0; j < K; j++) {
      for (let pi = 0; pi < P; pi++) params[pi] = accessors[pi](0, j);
      total += blp(kernelName, params, head[j]);
    }
    if (total !== 0) for (let i = 0; i < N; i++) acc[i] += total;
    return rest;
  }
  // Per-atom × per-element loop. K is the inner axis (densest stride).
  for (let i = 0; i < N; i++) {
    let total = 0;
    for (let j = 0; j < K; j++) {
      for (let pi = 0; pi < P; pi++) params[pi] = accessors[pi](i, j);
      total += blp(kernelName, params, head[j]);
    }
    acc[i] += total;
  }
  return rest;
}

// ---- Discrete-selector mixture (engine-concepts §11) ---------------

// Footprint size of a `rest` (how many scalar slots remain). Used only
// to assert that every select branch consumed the SAME observation
// footprint — components of a superpose/mixture/ifelse share one
// variate space (spec §06), so their rests must agree.
function restSize(r: any) {
  if (r == null) return 0;
  if (typeof r === 'number') return 1;
  if (valueLib.isValue(r)) return r.data.length;
  if (Array.isArray(r) || (r && typeof r.length === 'number')) return r.length;
  return 1;
}

// Reference-measure class of a select branch's expanded measure IR:
// 'continuous' (density w.r.t. Lebesgue) vs 'noncontinuous' (pmf
// w.r.t. counting, OR an atomic point mass / Dirac). `null` =
// undeterminable — caller PERMITS those (zero-false-positive
// discipline: never reject a valid homogeneous mixture; an unknown is
// no worse than today). The spike-and-slab trap is mixing the two:
// logsumexp-ing a per-length density with a per-atom probability is
// dimensionally meaningless. Note `Dirac.discrete` is deliberately
// false in the REGISTRY (the engine can't tell its scalar type from
// IR), so Dirac is special-cased atomic here — exactly the slab/spike
// case. weighted/logweighted/normalize/truncate/pushfwd don't change
// the reference measure (they scale / restrict / bijection-map), so
// recurse into the base; a nested select inherits its branches'.
function selectBranchRefClass(ir: any, depth: any) {
  if (!ir || ir.kind !== 'call' || (depth | 0) > 64) return null;
  const op = ir.op;
  if (op === 'Dirac') return 'noncontinuous';   // atomic point mass
  if (samplerLib.isKnownDistribution(op)) {
    const entry = samplerLib.lookupDistribution(ir);
    return entry && entry.discrete ? 'noncontinuous' : 'continuous';
  }
  if ((op === 'weighted' || op === 'logweighted' || op === 'pushfwd')
      && Array.isArray(ir.args) && ir.args.length === 2) {
    return selectBranchRefClass(ir.args[1], (depth | 0) + 1);
  }
  if ((op === 'normalize' || op === 'truncate')
      && Array.isArray(ir.args) && ir.args.length >= 1) {
    return selectBranchRefClass(ir.args[0], (depth | 0) + 1);
  }
  if (op === 'select' && Array.isArray(ir.branches)) {
    let cont = false, nonc = false;
    for (const b of ir.branches) {
      const c = selectBranchRefClass(b, (depth | 0) + 1);
      if (c === 'continuous') cont = true;
      else if (c === 'noncontinuous') nonc = true;
    }
    if (cont && nonc) return 'mixed';            // nested heterogeneous
    if (cont) return 'continuous';
    if (nonc) return 'noncontinuous';
    return null;
  }
  // joint / record / iid / MvNormal-as-vector / unknown: not a scalar
  // leaf we classify for this guard → permit (null).
  return null;
}

function walkSelect(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  // select(branches, logweights): the discrete-selector mixture — the
  // EXACT (finite, no −logN) discrete sibling of the kchain MC
  // marginal. Density:
  //
  //   log p(x) = logsumexp_k ( logw_k + log p_{branch_k}(x) )
  //
  // Every branch is an alternative measure over ONE shared variate
  // space (spec §06: superpose components share variate space), so
  // each consumes the same observation footprint from `value`; we
  // walk each into its own scratch accumulator, combine per atom via
  // a numerically-stable logsumexp, then add to the caller's `acc`.
  // `logweights:null` ⇒ all-zero (raw superpose ν = Σ M_k; each
  // component self-carries any weighted()/normalize() factor through
  // its own expanded sub-IR). Closed-form: zero sampling — the branch
  // walks are themselves closed-form for tractable components.
  const branches = ir.branches;
  if (!Array.isArray(branches) || branches.length === 0) {
    throw new Error('density: select requires a non-empty branches array');
  }
  const K = branches.length;

  // Retain mode (engine-concepts §11 — discrete sibling of jointchain
  // retain): if the selector binding has already been consumed by an
  // enclosing joint and threaded into the env overlay, the density
  // becomes the JOINT score `logp_{branch_k}(x) + logw_k` with k
  // determined by the observed selector value — no logsumexp over
  // branches. Matches the kchain-vs-jointchain pattern: same node,
  // different policy chosen by env-threading. Absent selector in env
  // ⇒ fall through to the marginalising mixture path below.
  if (ir.selectorName) {
    let sel = (overlay && Object.prototype.hasOwnProperty.call(overlay, ir.selectorName))
      ? overlay[ir.selectorName]
      : (baseEnv && Object.prototype.hasOwnProperty.call(baseEnv, ir.selectorName))
      ? baseEnv[ir.selectorName]
      : undefined;
    if (sel !== undefined) {
      // Same branch-index calculation as matSelect's gather: K==2 ⇒
      // truthy → branch 0 (Bernoulli/ifelse); K≥2 ⇒ (sel|0) − base
      // (Categorical 1-based by default, Categorical0 ⇒ base 0),
      // clamped to [0, K−1].
      let k;
      if (K === 2 && (ir.selectorBase == null)) {
        k = sel ? 0 : 1;
      } else {
        const base = (ir.selectorBase != null) ? ir.selectorBase : 1;
        k = (sel | 0) - base;
      }
      if (k < 0) k = 0; else if (k >= K) k = K - 1;
      // No `logweight` added in retain mode: the selector's
      // probability mass is paid by the joint's own selector field
      // (e.g. the Bernoulli/Categorical field that observed `c`).
      // Adding logw_k here would double-count the selector prior.
      // In marginalize mode (the fall-through below), logw_k IS the
      // selector prior — the anonymous-selector case has no separate
      // field to bill it to.
      return walkAcc(branches[k], value, refArrays, N, opts, acc,
        baseEnv, overlay);
    }
  }

  // Reference-measure guard (engine-concepts §11/§12). logsumexp-ing
  // branch log-densities is only meaningful when every branch scores
  // w.r.t. the SAME reference measure. Mixing a continuous (Lebesgue)
  // branch with a discrete/atomic one (the spike-and-slab trap, e.g.
  // ifelse(c, Normal, Dirac) or superpose(Normal, Poisson)) combines a
  // per-length density with a per-atom probability — dimensionally
  // meaningless. Refuse with a clear error rather than return a wrong
  // number. Zero-false-positive: throw ONLY when ≥1 branch is
  // positively continuous AND ≥1 is positively non-continuous (or a
  // nested select is itself heterogeneous); undeterminable branches
  // are permitted (a homogeneous mixture is never rejected). This is a
  // DENSITY-only guard — sampling such a mixture (matSelect) is
  // well-defined and stays allowed.
  let sawCont = false, sawNonc = false, sawMixed = false;
  for (let k = 0; k < K; k++) {
    const c = selectBranchRefClass(branches[k], 0);
    if (c === 'continuous') sawCont = true;
    else if (c === 'noncontinuous') sawNonc = true;
    else if (c === 'mixed') sawMixed = true;
  }
  if (sawMixed || (sawCont && sawNonc)) {
    const cls = branches.map((b, k) =>
      `  branch ${k}: ${selectBranchRefClass(b, 0) || 'unknown'}`).join('\n');
    throw new Error(
      'density: select/mixture combines branches with incommensurable '
      + 'reference measures — cannot logsumexp a continuous (Lebesgue) '
      + 'density with a discrete/atomic (counting / point-mass) one '
      + '(the spike-and-slab case, e.g. ifelse(c, Normal, Dirac)). '
      + 'Its SAMPLING is well-defined and still allowed; only its '
      + 'logdensityof is refused.\n' + cls);
  }
  const bacc = new Array(K);
  let rest0; let haveRest = false;
  for (let k = 0; k < K; k++) {
    const a = new Float64Array(N);
    const rk = walkAcc(branches[k], value, refArrays, N, opts, a, baseEnv, overlay);
    bacc[k] = a;
    if (!haveRest) { rest0 = rk; haveRest = true; }
    else if (restSize(rk) !== restSize(rest0)) {
      throw new Error('density: select branches consumed different '
        + 'observation footprints — components must share one variate '
        + 'space (spec §06)');
    }
  }
  // Per-branch log-weights (constant in the observation point; may
  // depend on params/per-atom refs). null ⇒ raw superpose (all 0).
  let lwArr: Float64Array[] | null = null;
  if (ir.logweights != null) {
    const lw = ir.logweights;
    if (!Array.isArray(lw) || lw.length !== K) {
      throw new Error('density: select logweights length must equal branches');
    }
    lwArr = new Array(K);
    for (let k = 0; k < K; k++) {
      const w = new Float64Array(N);
      applyAtomScalar(lw[k], refArrays, N, baseEnv, overlay, w, addRaw);
      lwArr[k] = w;
    }
  }
  // Stable per-atom logsumexp over the K weighted branch log-densities.
  for (let i = 0; i < N; i++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) {
      const t = bacc[k][i] + (lwArr ? lwArr[k][i] : 0);
      if (t > m) m = t;
    }
    if (m === -Infinity) { acc[i] += -Infinity; continue; }
    if (m === Infinity)  { acc[i] += Infinity;  continue; }
    let s = 0;
    for (let k = 0; k < K; k++) {
      s += Math.exp(bacc[k][i] + (lwArr ? lwArr[k][i] : 0) - m);
    }
    acc[i] += m + Math.log(s);
  }
  return rest0;
}

const OP_HANDLERS = {
  weighted:    walkWeighted,
  logweighted: walkLogWeighted,
  truncate:    walkTruncate,
  normalize:   walkNormalize,
  joint:       walkJointFieldsOrPositional,
  record:      walkJointFieldsOrPositional,
  iid:         walkIid,
  jointchain:  walkJointchainStub,
  pushfwd:     walkPushfwd,
  mcmarginal:  walkMcMarginal,
  select:      walkSelect,
  broadcast:   walkBroadcast,
  // Multivariate leaves all route through the generic walker that
  // dispatches into `density-prims.builtinLogdensityof`. Per-kernel
  // density math lives in density-prims's MV_DENSITY_FNS — the walker
  // here only handles structural concerns (variate consume/rest,
  // per-atom vs atom-indep kwarg resolution).
  //
  // `walkMultivariate` + `MV_DENSITY_FNS` are SHARED across all 8
  // multivariate kernels below — only MvNormal has a §22 pushfwd-of-iid
  // lowering; the other 7 (Dirichlet, Multinomial, BinnedPoissonProcess,
  // Wishart, InverseWishart, LKJ, LKJCholesky) have no lowering and this
  // is their sole density path. 5h-A ADJUDICATION (the lift gate now
  // lowers every binding-level MvNormal; matMvNormal is a thin refusal):
  // the MvNormal entries here still CANNOT retire —
  //   (a) MV_DENSITY_FNS.MvNormal backs builtin_logdensityof('MvNormal',…),
  //       the spec §07 FlatPDL ABI ("an engine must implement
  //       builtin_logdensityof … for every built-in measure kernel");
  //   (b) this OP_HANDLERS walker entry scores raw MvNormal nodes inside
  //       composite kernel bodies (per-atom mu/cov — the formal-carrying
  //       shapes the lift gate deliberately skips; pinned by the
  //       walkMultivariate per-atom tests). Retiring (b) is 5h-B scope
  //       (teach the composite recognizers the lowered pushfwd form).
  MvNormal:             walkMultivariate,
  Dirichlet:            walkMultivariate,
  Multinomial:          walkMultivariate,
  BinnedPoissonProcess: walkMultivariate,
  Wishart:              walkMultivariate,
  InverseWishart:       walkMultivariate,
  LKJ:                  walkMultivariate,
  LKJCholesky:          walkMultivariate,
};

// =====================================================================
// Generic multivariate walker (FlatPDL leaf dispatch, engine-concepts §13.6)
// =====================================================================
//
// Replaces the per-kernel walkMvNormal / walkDirichlet / walkMultinomial /
// walkBinnedPoissonProcess / walkWishart / walkInverseWishart / walkLKJ /
// walkLKJCholesky handlers. The structural work is identical across
// these kernels: consume a leading vector/matrix block off `value`,
// resolve kernel kwargs, call the FlatPDL primitive
// `builtinLogdensityof(name, kwRecord, x)`, broadcast into `acc`.
// The per-kernel density math lives entirely in
// `density-prims.MV_DENSITY_FNS`; this walker is kernel-agnostic.
//
// Per-atom params are supported automatically: when a kwarg expr
// references a per-atom name from refArrays (or an env-threaded
// overlay value), kwargs resolve per atom and the primitive is called
// in a tight loop. This unlocks hierarchical priors over multivariate
// kernels (e.g. `Sigma ~ Wishart(...); X ~ MvNormal(0, Sigma)`) that
// the per-kernel walkers couldn't represent.
function walkMultivariate(ir: IRNode, value: any, refArrays: any, N: any, opts: any, acc: any, baseEnv: any, overlay: any) {
  const kernelName: string = (ir as any).op;
  const shapeDescr = densityPrims.multivariateShape(kernelName);
  if (!shapeDescr) {
    throw new Error('density: walkMultivariate called with non-multivariate kernel '
      + kernelName);
  }
  const kwargsIR: Record<string, any> = (ir as any).kwargs || {};
  const refNames = refArrays ? Object.keys(refArrays) : [];
  const overlayKeys = overlay ? Object.keys(overlay) : null;
  // Atom-independent env baseline. Per-atom names get added inside the
  // per-atom loop below; the atom-indep eval uses baseEnv + overlay only.
  const atomIndepEnv = Object.assign({}, baseEnv);
  if (overlayKeys) {
    for (let j = 0; j < overlayKeys.length; j++) {
      atomIndepEnv[overlayKeys[j]] = overlay[overlayKeys[j]];
    }
  }
  // Per-atom path triggers only when some kwarg expression actually
  // references a per-atom name; detect this with a free-ref walk.
  // refArrays-listed names that no kwarg uses don't force the per-atom
  // path (cf. walkLeaf's same optimisation).
  const perAtomSet: Record<string, true> = {};
  for (const k in kwargsIR) {
    if (Object.prototype.hasOwnProperty.call(kwargsIR, k)
        && _exprUsesAny(kwargsIR[k], refNames)) {
      perAtomSet[k] = true;
    }
  }
  const perAtomKwKeys = Object.keys(perAtomSet);
  const anyKwUsesPerAtom = perAtomKwKeys.length > 0;
  // Resolve the atom-independent kwargs once. Per-atom kwargs stay as
  // IR and get evaluated inside the per-atom loop. We need to compute
  // the variate shape `n` BEFORE consuming from `value`; the shape's
  // controlling kwarg (per shapeDescr.sizeFrom) may itself be per-atom
  // — in that case we resolve it once against atom-0's env to read n,
  // assuming variate shape is structurally fixed (the only sane
  // case — the IR-shape is static).
  const kwInit: Record<string, any> = {};
  for (const k in kwargsIR) {
    if (Object.prototype.hasOwnProperty.call(kwargsIR, k) && !perAtomSet[k]) {
      kwInit[k] = samplerLib.evaluateExpr(kwargsIR[k], atomIndepEnv);
    }
  }
  // For per-atom kwargs we still need ENOUGH to derive `n`. Evaluate
  // the per-atom kwargs against atom 0's env into a shape-probe record.
  let kwForSize: Record<string, any> = kwInit;
  if (anyKwUsesPerAtom) {
    const probeEnv = Object.assign({}, baseEnv);
    if (overlayKeys) {
      for (let j = 0; j < overlayKeys.length; j++) {
        probeEnv[overlayKeys[j]] = overlay[overlayKeys[j]];
      }
    }
    for (let j = 0; j < refNames.length; j++) {
      const v = refArrays[refNames[j]];
      probeEnv[refNames[j]] = valueLib.isValue(v) ? _atomSlice(v, 0) : v[0];
    }
    kwForSize = Object.assign({}, kwInit);
    for (let j = 0; j < perAtomKwKeys.length; j++) {
      const k = perAtomKwKeys[j];
      kwForSize[k] = samplerLib.evaluateExpr(kwargsIR[k], probeEnv);
    }
  }
  const n = shapeDescr.sizeFrom(kwForSize);
  const { head, rest } = (shapeDescr.kind === 'vector')
    ? consumeVector(value, n) : consumeMatrix(value, n);
  if (!anyKwUsesPerAtom) {
    const lp = densityPrims.builtinLogdensityof(kernelName, kwInit, head);
    for (let i = 0; i < N; i++) acc[i] += lp;
    return rest;
  }
  // Per-atom path.
  const callEnv = Object.assign({}, baseEnv);
  if (overlayKeys) {
    for (let j = 0; j < overlayKeys.length; j++) {
      callEnv[overlayKeys[j]] = overlay[overlayKeys[j]];
    }
  }
  const kwForAtom = Object.assign({}, kwInit);
  // Reuse atom-0 resolution we already computed for the size probe.
  for (let j = 0; j < perAtomKwKeys.length; j++) {
    kwForAtom[perAtomKwKeys[j]] = kwForSize[perAtomKwKeys[j]];
  }
  for (let j = 0; j < refNames.length; j++) {
    const v = refArrays[refNames[j]];
    callEnv[refNames[j]] = valueLib.isValue(v) ? _atomSlice(v, 0) : v[0];
  }
  acc[0] += densityPrims.builtinLogdensityof(kernelName, kwForAtom, head);
  for (let i = 1; i < N; i++) {
    for (let j = 0; j < refNames.length; j++) {
      const v = refArrays[refNames[j]];
      callEnv[refNames[j]] = valueLib.isValue(v) ? _atomSlice(v, i) : v[i];
    }
    if (overlayKeys) {
      for (let j = 0; j < overlayKeys.length; j++) {
        callEnv[overlayKeys[j]] = overlay[overlayKeys[j]];
      }
    }
    for (let j = 0; j < perAtomKwKeys.length; j++) {
      const k = perAtomKwKeys[j];
      kwForAtom[k] = samplerLib.evaluateExpr(kwargsIR[k], callEnv);
    }
    acc[i] += densityPrims.builtinLogdensityof(kernelName, kwForAtom, head);
  }
  return rest;
}

// Per-atom slice of a Value: rank-1 yields a scalar number, rank≥2
// yields a Value of one rank lower (data shares storage).
function _atomSlice(v: any, i: number): any {
  if (v.shape.length === 1) return v.data[i];
  const tail = v.shape.slice(1);
  const tailLen = tail.reduce((a: number, b: number) => a * b, 1);
  return { shape: tail, data: v.data.subarray(i * tailLen, (i + 1) * tailLen) };
}

// Cheap structural walk: does `ir` (a kwarg expression) reference any
// name in `nameSet`? Used to decide whether the multivariate walker
// can stay on the atom-indep hot path.
function _exprUsesAny(ir: any, names: string[]): boolean {
  if (!ir || typeof ir !== 'object') return false;
  if (ir.kind === 'ref' && names.indexOf(ir.name) >= 0) return true;
  if (Array.isArray(ir.args)) {
    for (const a of ir.args) if (_exprUsesAny(a, names)) return true;
  }
  if (ir.kwargs) {
    for (const k in ir.kwargs) if (_exprUsesAny(ir.kwargs[k], names)) return true;
  }
  if (Array.isArray(ir.fields)) {
    for (const f of ir.fields) if (_exprUsesAny(f.value, names)) return true;
  }
  return false;
}

function _anyKwUsesPerAtom(kwargsIR: Record<string, any>, refNames: string[]): boolean {
  for (const k in kwargsIR) {
    if (Object.prototype.hasOwnProperty.call(kwargsIR, k)
        && _exprUsesAny(kwargsIR[k], refNames)) return true;
  }
  return false;
}

// ---- Per-atom scalar contribution helpers ---------------------------

// Evaluate `wIR` once over the whole atom batch via sampler.evaluateExprN,
// then accumulate via `combine(acc, i, value)`. The single batched call
// replaces what used to be N scalar evaluateExpr loops — same env-
// precedence (overlay > refArrays > baseEnv) but enforced inside
// evaluateExprN.
// =====================================================================
// Per-kernel multivariate density walkers have moved to density-prims.ts;
// dispatch happens through `walkMultivariate` above, which calls into
// `density-prims.builtinLogdensityof`. See engine-concepts §13.6.
// =====================================================================

function applyAtomScalar(wIR: any, refArrays: any, N: any, baseEnv: any, overlay: any, acc: any, combine: any) {
  const result = samplerLib.evaluateExprN(
    wIR, refArrays || null, N, baseEnv,
    overlay ? { overlay } : undefined);
  if (typeof result === 'number' || typeof result === 'boolean') {
    const v = +result;
    for (let i = 0; i < N; i++) combine(acc, i, v);
    return;
  }
  // Float64Array(N) or generic Array(N).
  for (let i = 0; i < N; i++) combine(acc, i, +result[i]);
}

function addLogW(acc: any, i: any, w: any) {
  if (!(w > 0)) acc[i] = -Infinity;
  else acc[i] += Math.log(w);
}
function addRaw(acc: any, i: any, v: any) {
  acc[i] += v;
}

// =====================================================================
// Public wrappers
// =====================================================================

/**
 * Strict batched API: runs `logDensityConsumeN` and asserts every atom
 * consumed the value fully. Returns just `logps` — the form most
 * callers want, and the same shape as the worker.logDensityN message
 * reply.
 */
function logDensityN(ir: any, value: any, refArrays: any, count: any, opts: any) {
  const { logps, rest } = logDensityConsumeN(ir, value, refArrays, count, opts);
  if (!isEmptyRest(rest)) {
    throw new Error('logDensityN: value has unconsumed leftover after walking IR'
      + ' (op=' + (ir && ir.op) + ')');
  }
  return logps;
}

/**
 * Single-point consume — count=1 wrapper. Returns { logp, rest } in
 * the same shape as the original logDensityConsume API.
 */
function logDensityConsume(ir: any, value: any, env: any, opts: any) {
  const callOpts = env ? Object.assign({}, opts || {}, { baseEnv: env }) : opts;
  const { logps, rest } = logDensityConsumeN(ir, value, null, 1, callOpts);
  return { logp: logps[0], rest };
}

/**
 * Single-point strict — count=1 wrapper over logDensityN. The
 * Float64Array(1) allocation is the only single-point overhead vs the
 * batched form.
 */
function logDensity(ir: any, value: any, env: any, opts: any) {
  const callOpts = env ? Object.assign({}, opts || {}, { baseEnv: env }) : opts;
  return logDensityN(ir, value, null, 1, callOpts)[0];
}

// =====================================================================
// Local helpers — set-membership for truncate, scalar back-inference.
// =====================================================================

function inferConsumedScalar(value: any, rest: any) {
  // For truncate(M, S) where M is scalar-leaf-like: the head consumed
  // by the recursive call is value[0]. We recover it cheaply rather
  // than threading it through the return shape.
  if (typeof value === 'number') return value;
  if (value && value.BYTES_PER_ELEMENT) return +value[0];
  if (Array.isArray(value) && value.length > 0) return +value[0];
  return null;
}

function inSet(x: any, setDescr: any) {
  // No null-allow: walkTruncate throws on an unresolved set BEFORE this is
  // reached; an unknown descriptor kind is an engine drift (a new set kind
  // wired into parseSetIR but not here) — silently treating it as "inside"
  // would score the un-truncated density, so fail loud (audit Phase 6).
  if (!setDescr) {
    throw new Error('density: inSet called with no set descriptor');
  }
  switch (setDescr.kind) {
    case 'interval':    return x >= +setDescr.lo && x <= +setDescr.hi;
    case 'reals':       return Number.isFinite(x) || x === -Infinity || x === Infinity;
    case 'posreals':    return x > 0;
    case 'nonnegreals': return x >= 0;
    case 'unitinterval':return x >= 0 && x <= 1;
    // Discrete sets (previously fell to the silent-allow default — a
    // non-integer point got the un-truncated density instead of −inf).
    case 'integers':       return Number.isInteger(x);
    case 'posintegers':    return Number.isInteger(x) && x >= 1;
    case 'nonnegintegers': return Number.isInteger(x) && x >= 0;
    case 'booleans':       return x === 0 || x === 1;
    default:
      throw new Error('density: unknown set descriptor kind \''
        + setDescr.kind + '\' in truncate support check');
  }
}

module.exports = {
  // Foundation (batched, returns rest)
  logDensityConsumeN,
  // Strict batched
  logDensityN,
  // Single-point conveniences (count=1 of the foundation)
  logDensityConsume,
  logDensity,
  // P3b — density-walker pipeline (engine-concepts §18.2 /
  // §20.10). Stages compose around `densityCoreStage`
  // (the terminal acc+walkAcc dispatch). Caching, tracing,
  // shape-validation, and custom inference hooks all land as
  // additional stages without touching per-op-handler code.
  registerDensityStage,
  _resetDensityPipeline,
  densityCoreStage,
  makeDensityTracingStage,
  _runDensityPipeline,
  // Test/debug surface — exposes the shape helpers in case callers
  // outside the dispatch want to compose their own consumers.
  _internal: { consumeScalar, consumeField, consumeVector, isEmptyRest, inSet },
};

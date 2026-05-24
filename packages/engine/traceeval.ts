'use strict';

import type { IRNode } from './engine-types';

// =====================================================================
// traceeval.js — pure-sampling walker for FlatPPL measure expressions
// =====================================================================
//
// Given a measure IR and an env mapping value-position refs to numerics,
// draw a value from that measure, threading rng state through any
// stochastic ancestors that the env doesn't already carry.
//
// This is the sample-side primitive only. Scoring (logdensityof,
// bayesupdate, likelihoodof) lives in `density.js` — the single density
// implementation for the whole engine. Earlier versions of this walker
// carried a `tally` / `observed` protocol that doubled as a density
// evaluator; that's been retired in favour of density.js's batched
// `logDensityConsumeN` foundation.
//
// Public API
// ----------
//   walk(state, ir, env, opts) → { value, state }
//     state    - rng.State (consumed at every leaf site).
//     ir       - measure IR (FlatPIR call form). Self-refs to other
//                measure bindings resolve via opts.resolveMeasureRef.
//     env      - per-i env mapping ref-names to numerics, same shape
//                as sampler.evaluateExpr's env.
//     opts.resolveMeasureRef - optional `(name) → measureIR`. The
//                              orchestrator supplies a closure over
//                              its bindings/derivations map for
//                              fields of a joint / measure args of
//                              iid/weighted etc. that are written as
//                              self-refs to other measure bindings
//                              rather than inline calls.
//     opts.resolveValueRef   - optional `(name, state) → [value, newState]`.
//                              Lazy resolution of value-position
//                              self-refs that env doesn't carry —
//                              typical of `rand(state, M)` where M's
//                              params reference stochastic ancestors.
//                              The resolver threads state through any
//                              recursive sampling it does and is
//                              expected to cache its result through env
//                              so two refs to the same name share one
//                              draw.
//   → { value, state }
//
// Operations supported in the IR
// ------------------------------
//   leaf distribution  — any op in sampler.REGISTRY. Sampled via the
//                        REGISTRY entry's prng-bound factory.
//   joint / record     — { kind:'call', op:'joint'|'record',
//                          fields:[{name,value}, …] }. Field-wise
//                          recursion; value is a record keyed by name.
//   joint (positional) — { kind:'call', op:'joint', args:[M1,…] }.
//                        Value is an array of per-component samples.
//   iid                — { kind:'call', op:'iid', args:[M,n] }. n
//                        draws of M sharing env (that's what "iid"
//                        means here).
//   weighted /
//   logweighted        — sampling pass-through (the weight only
//                        affects density; density.js handles that).
//   lawof / draw       — pass-through (`lawof(draw(M)) ≡ M`).

const samplerLib = require('./sampler.ts');

function walk(state: any, ir: IRNode, env: any, opts: any) {
  opts = opts || {};
  const ctx = {
    resolveRef: opts.resolveMeasureRef || null,
    resolveValueRef: opts.resolveValueRef || null,
  };
  return walkInner(state, ir, env, ctx);
}

function walkInner(state: any, ir: IRNode, env: any, ctx: any): any {
  // Self-ref to another measure binding. The orchestrator passes a
  // resolver so we can dereference without baking the binding map
  // into this module.
  if (ir && ir.kind === 'ref' && ir.ns === 'self') {
    if (!ctx.resolveRef) {
      throw new Error(
        `traceeval: encountered measure ref '${ir.name}' but no ` +
        `resolveMeasureRef was supplied. Inline the measure or pass ` +
        `opts.resolveMeasureRef.`
      );
    }
    const inner = ctx.resolveRef(ir.name);
    if (!inner) {
      throw new Error(`traceeval: resolveMeasureRef returned no IR for '${ir.name}'`);
    }
    return walkInner(state, inner, env, ctx);
  }

  if (!ir || ir.kind !== 'call') {
    throw new Error(
      `traceeval: expected a measure call IR (or self-ref), got ` +
      `kind=${ir && ir.kind}`
    );
  }
  const op = ir.op;

  // Leaf distribution — base case.
  if (samplerLib.isKnownDistribution(op)) {
    return walkLeaf(state, ir, env, ctx);
  }

  // Dispatch through MEASURE_OP_WALKERS. Adding a new measure-algebra
  // walker (pushfwd, truncate, …) is one entry here plus the handler
  // function — no edits to walkInner itself.
  const handler: any = op != null ? (MEASURE_OP_WALKERS as any)[op] : null;
  if (handler) return handler(state, ir, env, ctx);
  throw new Error(
    `traceeval: op '${op}' is not a measure expression we can ` +
    `sample. Known: leaf distributions, ` +
    Object.keys(MEASURE_OP_WALKERS).join(', ') + '.'
  );
}

function walkLeaf(state: any, ir: IRNode, env: any, ctx: any) {
  // Pre-fill env with any value-position refs in the kwargs that
  // aren't already known. The leaf's distribution params (`mu = ref a`,
  // `sigma = ref b`, …) may reference bindings the caller hasn't
  // materialised yet — typical of rand(state, M) where M's params
  // depend on stochastic ancestors. fillEnvFromRefs threads state
  // through any recursive sampling the resolver does.
  state = fillEnvFromRefs(state, ir, env, ctx);
  const entry = samplerLib.lookupDistribution(ir);
  const params = samplerLib.resolveParams(ir, entry, env);
  const prng = samplerLib.makePhiloxPrngAdapter(state);
  const sampler = entry.randFn.factory(...params, { prng });
  const value = sampler();
  return { value, state: prng.getState() };
}

function walkJoint(state: any, ir: IRNode, env: any, ctx: any) {
  // Two surface forms:
  //   * kwarg-joint / record: ir.fields = [{ name, value }, ...].
  //     Output value is a record keyed by field name.
  //   * positional joint:    ir.args   = [M1, M2, ...]. Output value
  //     is an array of per-component samples.
  if (Array.isArray(ir.fields)) {
    const out: Record<string, any> = {};
    let st = state;
    for (let i = 0; i < ir.fields.length; i++) {
      const f = ir.fields[i];
      const r = walkInner(st, f.value, env, ctx);
      out[f.name] = r.value;
      st = r.state;
    }
    return { value: out, state: st };
  }
  if (Array.isArray(ir.args)) {
    const components = ir.args;
    const out = new Array(components.length);
    let st = state;
    for (let i = 0; i < components.length; i++) {
      const r = walkInner(st, components[i], env, ctx);
      out[i] = r.value;
      st = r.state;
    }
    return { value: out, state: st };
  }
  throw new Error('traceeval: joint with neither fields nor args');
}

function walkIid(state: any, ir: IRNode, env: any, ctx: any) {
  // iid(M, size): per spec §06, `size` is either a positive integer
  // (1-D length) or a vector of positive integers (multi-axis shape).
  // We draw prod(size) iid samples from M sharing params and reshape
  // into a nested array of the requested shape.
  //
  // Inner env is shared — params do NOT change across the inner
  // draws (that's what makes it 'iid' vs a vectorised call with
  // per-index params).
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`traceeval: iid expected 2 args (measure, size), got ${args.length}`);
  }
  const M = args[0];
  // size may reference bindings (e.g. `iid(M, n)` where n is a
  // fixed-phase value binding) — pre-fill before evaluating.
  state = fillEnvFromRefs(state, args[1], env, ctx);
  const sizeVal: any = samplerLib.evaluateExpr(args[1], env);
  const dims = _sizeAsDims(sizeVal);
  const total = dims.reduce((p: number, n: number) => p * n, 1);
  // Sequential draws: total `total` samples threading state.
  const flat = new Array(total);
  let st = state;
  for (let j = 0; j < total; j++) {
    const r = walkInner(st, M, env, ctx);
    flat[j] = r.value;
    st = r.state;
  }
  // 1-D shape returns the flat array directly (matches the prior
  // single-int behaviour). Multi-axis shapes get reshaped into
  // nested JS arrays.
  const value = (dims.length <= 1) ? flat : _reshapeNested(flat, dims);
  return { value, state: st };
}

// Normalise a size argument to a number[] of axis lengths. Accepts
// a positive integer, a JS array, a Float64Array, or a shape-rich
// Value. Mirrors sampler.ts `_sizeAsDims` (kept duplicated to avoid
// a circular import — sampler ↔ traceeval).
function _sizeAsDims(size: any): number[] {
  if (typeof size === 'number') {
    if (!Number.isInteger(size) || size < 0) {
      throw new Error('iid: size must be a non-negative integer, got ' + size);
    }
    return [size];
  }
  let arr: any = size;
  if (arr && typeof arr === 'object' && arr.shape && arr.data) arr = arr.data;
  if (arr && arr.BYTES_PER_ELEMENT && typeof arr.length === 'number') {
    arr = Array.from(arr as ArrayLike<number>);
  }
  if (!Array.isArray(arr)) {
    throw new Error('iid: size must be an integer or vector of integers, got '
      + (typeof size));
  }
  const dims: number[] = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const n = arr[i];
    if (!Number.isInteger(n) || n < 0) {
      throw new Error('iid: size axis ' + (i + 1)
        + ' is not a non-negative integer (got ' + n + ')');
    }
    dims[i] = n as number;
  }
  return dims;
}

// Reshape a flat JS array of length prod(dims) into a nested JS
// array of shape `dims` (row-major). Each leaf is the original
// scalar / value; for purely numeric inputs the result is a nested
// array of plain JS numbers (matching the existing nested-array
// representation used elsewhere in the engine).
function _reshapeNested(flat: any[], dims: number[]): any {
  if (dims.length === 0) return flat[0];
  if (dims.length === 1) return flat;
  const n = dims[0];
  const innerDims = dims.slice(1);
  const innerSize = innerDims.reduce((p, n) => p * n, 1);
  const out: any[] = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = _reshapeNested(flat.slice(i * innerSize, (i + 1) * innerSize), innerDims);
  }
  return out;
}

// weighted / logweighted: sampling is a pure pass-through. The weight
// only affects density, which lives in density.js.
function walkWeightedPassThrough(state: any, ir: IRNode, env: any, ctx: any): any {
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`traceeval: weighted/logweighted expected 2 args, got ${args.length}`);
  }
  return walkInner(state, args[1], env, ctx);
}

// lawof(M) / draw(M): pass-through wrappers per spec identity
// `lawof(draw(M)) ≡ M` (and the dual). Either may appear surfaced in
// inline forms the orchestrator hasn't yet canonicalised; we unwrap
// here so callers don't need to do it upstream.
function walkUnwrap(state: any, ir: IRNode, env: any, ctx: any): any {
  const args = ir.args || [];
  if (args.length !== 1) {
    throw new Error(`traceeval: ${ir.op} expected 1 arg, got ${args.length}`);
  }
  return walkInner(state, args[0], env, ctx);
}

const MEASURE_OP_WALKERS = {
  joint:       walkJoint,
  record:      walkJoint,
  iid:         walkIid,
  weighted:    walkWeightedPassThrough,
  logweighted: walkWeightedPassThrough,
  lawof:       walkUnwrap,
  draw:        walkUnwrap,
};

/**
 * Walk a value-position IR expression collecting every `(ref self <name>)`
 * the env doesn't already know. For each, invoke ctx.resolveValueRef
 * (if supplied) to compute the value, threading state through. Mutates
 * `env` in place so callers can subsequently call evaluateExpr against
 * a fully-populated environment.
 *
 * Skips structural recursion into measure-position children: this
 * helper handles value-position IR only (distribution kwargs, iid
 * counts, weighted weights, …). Refs to measure bindings are resolved
 * separately via ctx.resolveRef.
 *
 * Returns the (possibly advanced) state. Idempotent when env already
 * has every referenced name.
 */
function fillEnvFromRefs(state: any, ir: any, env: any, ctx: any) {
  if (!ctx || !ctx.resolveValueRef) return state;
  const refs = new Set<string>();
  collectValueRefs(ir, refs);
  for (const name of refs) {
    if (env && env[name] !== undefined) continue;
    const r = ctx.resolveValueRef(name, state);
    if (!r) continue;
    env[name] = r[0];
    state = r[1];
  }
  return state;
}

function collectValueRefs(ir: any, out: Set<string>) {
  if (ir == null || typeof ir !== 'object') return;
  if (ir.kind === 'ref' && ir.ns === 'self') { out.add(ir.name); return; }
  if (Array.isArray(ir.args)) for (const a of ir.args) collectValueRefs(a, out);
  if (ir.kwargs) for (const k in ir.kwargs) collectValueRefs(ir.kwargs[k], out);
  // Don't descend into `fields` / `body` — those are measure / scope
  // boundaries handled by the surrounding walker's recursion.
}

module.exports = { walk };

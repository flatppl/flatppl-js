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
//     opts.resolveMeasureRef(name) → ir | null. Caller typically
//     supplies a closure over `orchestrator.expandMeasureIR(name,
//     derivations)`.

const samplerLib = require('./sampler');

// =====================================================================
// Shape helpers — atom-independent consume/rest splitting
// =====================================================================

/** True iff `rest` is null/undefined or an empty container. */
function isEmptyRest(rest) {
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
function consumeScalar(value) {
  if (value == null) {
    throw new Error('density: scalar leaf has no entry to consume (value exhausted)');
  }
  if (typeof value === 'number') return { head: value, rest: null };
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
 * Pull a single named field from a record `value`. Returns
 * { head: value[name], rest } where rest is a shallow copy of value
 * with `name` removed (or null when the record was a single-field one).
 * Throws when the key isn't present — the caller's shape didn't match
 * the measure's declared fields.
 */
function consumeField(value, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || value.BYTES_PER_ELEMENT) {
    throw new Error('density: cannot consume named field \''
      + name + '\' from non-record value');
  }
  if (!Object.prototype.hasOwnProperty.call(value, name)) {
    throw new Error('density: record missing field \'' + name + '\'');
  }
  const head = value[name];
  const rest = {};
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

function logDensityConsumeN(ir, value, refArrays, count, opts) {
  opts = opts || {};
  const N = count | 0;
  if (N <= 0) throw new Error('density: count must be positive');
  if (!ir) throw new Error('density: missing IR');
  refArrays = refArrays || {};
  const baseEnv = opts.baseEnv || {};

  const acc = new Float64Array(N);
  // overlay starts empty; env-threading from joint fields adds entries
  // that win over both baseEnv and per-atom refArrays at each leaf.
  const rest = walkAcc(ir, value, refArrays, N, opts, acc, baseEnv, null);
  return { logps: acc, rest };
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
function walkAcc(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
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
  const handler = OP_HANDLERS[op];
  if (handler) return handler(ir, value, refArrays, N, opts, acc, baseEnv, overlay);

  // Leaf distribution — the only kind not in OP_HANDLERS (because
  // there are many of them and they're discovered via the REGISTRY).
  if (samplerLib.isKnownDistribution(op)) {
    return walkLeaf(ir, value, refArrays, N, opts, acc, baseEnv, overlay);
  }
  throw new Error('density: unsupported measure op \'' + op + '\'');
}

// ---- Per-op handlers (the in-place accumulators) --------------------

function walkLeaf(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
  // The only per-atom work in the entire walk: consume one entry from
  // the value, then loop atoms resolving params and adding logpdf to
  // acc[i]. consumeScalar is atom-independent (head + rest are derived
  // from `value`, which is shared).
  const { head, rest } = consumeScalar(value);
  const entry = samplerLib.lookupDistribution(ir);
  const refNames = Object.keys(refArrays);
  const overlayKeys = overlay ? Object.keys(overlay) : null;
  // Hot path: no per-atom refs AND no overlay → params constant across
  // atoms. Resolve once, broadcast.
  if (refNames.length === 0 && !overlayKeys) {
    const params = samplerLib.resolveParams(ir, entry, baseEnv);
    const logp = entry.logpdfFn(head, ...params);
    for (let i = 0; i < N; i++) acc[i] += logp;
    return rest;
  }
  // Per-atom path. callEnv layering (bottom-up): baseEnv, refArrays[i],
  // overlay. Overlay applied last so env-threaded observation values
  // win over per-atom refs.
  const callEnv = Object.assign({}, baseEnv);
  // Apply the (atom-independent) overlay once outside the loop —
  // refArrays writes would otherwise stomp these names per atom.
  // We re-apply inside the loop after refArrays.
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < refNames.length; j++) {
      const k = refNames[j];
      callEnv[k] = refArrays[k][i];
    }
    if (overlayKeys) {
      for (let j = 0; j < overlayKeys.length; j++) {
        const k = overlayKeys[j];
        callEnv[k] = overlay[k];
      }
    }
    const params = samplerLib.resolveParams(ir, entry, callEnv);
    acc[i] += entry.logpdfFn(head, ...params);
  }
  return rest;
}

function walkWeighted(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
  // weighted(w, base): adds log(w) per atom. Recurse into base first
  // (with the same acc), then add log(w_i). Negative or zero weights
  // collapse the atom's logp to -Infinity.
  const wIR = ir.args[0];
  const rest = walkAcc(ir.args[1], value, refArrays, N, opts, acc, baseEnv, overlay);
  applyAtomScalar(wIR, refArrays, N, baseEnv, overlay, acc, addLogW);
  return rest;
}

function walkLogWeighted(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
  // logweighted(g, base): adds g directly per atom. -Infinity is
  // permitted; NaN is left as-is (callers detect downstream).
  const gIR = ir.args[0];
  const rest = walkAcc(ir.args[1], value, refArrays, N, opts, acc, baseEnv, overlay);
  applyAtomScalar(gIR, refArrays, N, baseEnv, overlay, acc, addRaw);
  return rest;
}

function walkTruncate(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
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
  const rest = walkAcc(ir.args[0], value, refArrays, N, opts, acc, baseEnv, overlay);
  const consumed = inferConsumedScalar(value, rest);
  if (consumed != null && !inSet(consumed, setDescr)) {
    for (let i = 0; i < N; i++) acc[i] = -Infinity;
  }
  return rest;
}

function walkNormalize(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
  // normalize(base) = base / totalmass(base). Density shifts by
  // -log(totalmass(base)). The caller supplies the parent's
  // logTotalmass via opts.measureLogTotalmass — atom-independent today;
  // a per-atom variant can be added when needed.
  const getLTM = opts.measureLogTotalmass;
  const baseLTM = typeof getLTM === 'function' ? +getLTM(ir.args[0]) : 0;
  const rest = walkAcc(ir.args[0], value, refArrays, N, opts, acc, baseEnv, overlay);
  if (baseLTM !== 0) {
    for (let i = 0; i < N; i++) acc[i] -= baseLTM;
  }
  return rest;
}

function walkJointFieldsOrPositional(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
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
  // Positional joint: args = [M1, M2, ...]. No env-threading (each Mi
  // sees the same env); just thread `rest` through.
  if (Array.isArray(ir.args)) {
    let cur = value;
    for (let i = 0; i < ir.args.length; i++) {
      cur = walkAcc(ir.args[i], cur, refArrays, N, opts, acc, baseEnv, overlay);
    }
    return cur;
  }
  throw new Error('density: joint with neither fields nor args');
}

function walkIid(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
  // iid(M, n): n copies of M's footprint. Count `n` is atom-independent
  // — it's a value-position expression typically of fixed phase. We
  // evaluate against baseEnv (no per-atom or overlay coverage); if a
  // user pattern needs per-atom counts, we'd switch to ragged storage
  // and that's a separate refactor.
  const args = ir.args || [];
  if (args.length !== 2) throw new Error('density: iid expected 2 args, got ' + args.length);
  const n = +samplerLib.evaluateExpr(args[1], baseEnv) | 0;
  if (n < 0) throw new Error('density: iid count negative: ' + n);
  let cur = value;
  for (let i = 0; i < n; i++) {
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

const OP_HANDLERS = {
  weighted:    walkWeighted,
  logweighted: walkLogWeighted,
  truncate:    walkTruncate,
  normalize:   walkNormalize,
  joint:       walkJointFieldsOrPositional,
  record:      walkJointFieldsOrPositional,
  iid:         walkIid,
  jointchain:  walkJointchainStub,
};

// ---- Per-atom scalar contribution helpers ---------------------------

// Evaluate `wIR` once over the whole atom batch via sampler.evaluateExprN,
// then accumulate via `combine(acc, i, value)`. The single batched call
// replaces what used to be N scalar evaluateExpr loops — same env-
// precedence (overlay > refArrays > baseEnv) but enforced inside
// evaluateExprN.
function applyAtomScalar(wIR, refArrays, N, baseEnv, overlay, acc, combine) {
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

function addLogW(acc, i, w) {
  if (!(w > 0)) acc[i] = -Infinity;
  else acc[i] += Math.log(w);
}
function addRaw(acc, i, v) {
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
function logDensityN(ir, value, refArrays, count, opts) {
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
function logDensityConsume(ir, value, env, opts) {
  const callOpts = env ? Object.assign({}, opts || {}, { baseEnv: env }) : opts;
  const { logps, rest } = logDensityConsumeN(ir, value, null, 1, callOpts);
  return { logp: logps[0], rest };
}

/**
 * Single-point strict — count=1 wrapper over logDensityN. The
 * Float64Array(1) allocation is the only single-point overhead vs the
 * batched form.
 */
function logDensity(ir, value, env, opts) {
  const callOpts = env ? Object.assign({}, opts || {}, { baseEnv: env }) : opts;
  return logDensityN(ir, value, null, 1, callOpts)[0];
}

// =====================================================================
// Local helpers — set-membership for truncate, scalar back-inference.
// =====================================================================

function inferConsumedScalar(value, rest) {
  // For truncate(M, S) where M is scalar-leaf-like: the head consumed
  // by the recursive call is value[0]. We recover it cheaply rather
  // than threading it through the return shape.
  if (typeof value === 'number') return value;
  if (value && value.BYTES_PER_ELEMENT) return +value[0];
  if (Array.isArray(value) && value.length > 0) return +value[0];
  return null;
}

function inSet(x, setDescr) {
  if (!setDescr) return true;
  switch (setDescr.kind) {
    case 'interval':    return x >= +setDescr.lo && x <= +setDescr.hi;
    case 'reals':       return Number.isFinite(x) || x === -Infinity || x === Infinity;
    case 'posreals':    return x > 0;
    case 'nonnegreals': return x >= 0;
    case 'unitinterval':return x >= 0 && x <= 1;
    default: return true;
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
  // Test/debug surface — exposes the shape helpers in case callers
  // outside the dispatch want to compose their own consumers.
  _internal: { consumeScalar, consumeField, isEmptyRest, inSet },
};

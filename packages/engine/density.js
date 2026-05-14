'use strict';

// =====================================================================
// density.js — log-density evaluation with consume/rest semantics
// =====================================================================
//
// Provides a single primitive that walks a measure IR alongside a
// candidate variate value, consuming the leading part of the value
// that the measure occupies and returning what's left:
//
//   logDensityConsume(ir, value, env, opts) → { logp, rest }
//
// This is the foundation for log-density evaluation throughout the
// engine. Composite measures (joint / jointchain / iid / …) split the
// value across their components by recursing into logDensityConsume
// and threading `rest` from one component into the next. Leaf
// distributions consume one entry and produce one logp. The wrapper
// `logDensity(ir, value, env, opts)` asserts that nothing remains
// after the walk; a non-empty `rest` indicates a shape mismatch
// between the measure and the value.
//
// Why a separate primitive (rather than reusing traceeval.walk):
//
//   * traceeval.walk handles a unified sample-OR-score path, where
//     `observed` is pre-shaped to mirror the IR (records keyed by
//     name, iid blocks as arrays of length n, …). Composite measures
//     whose variates are CAT'd (positional joint, jointchain) don't
//     fit that template — the caller would have to pre-split the
//     value, which duplicates the engine's shape knowledge in every
//     call site.
//
//   * The consume/rest pattern moves all that shape-aware splitting
//     into one place. Each measure-kind handler knows its own
//     footprint; the caller hands over a flat value and gets back
//     exactly what wasn't consumed.
//
//   * The empty-rest invariant (in `logDensity`) catches shape
//     mismatches as proper errors rather than silently scoring the
//     wrong slice — useful both for users and as a regression check
//     when adding new composite measures.
//
// Value-shape conventions (what `value` may be at each consumer):
//
//   - number              — a single scalar entry. Fully consumed by
//                           a scalar leaf distribution. rest = null.
//   - Float64Array / TypedArray
//                         — a flat numeric vector. Consumers take a
//                           prefix slice via subarray; rest is the
//                           remaining suffix (or null if all consumed).
//   - Array (plain JS)    — same shape conventions as a typed array;
//                           consumers use slice for "rest".
//   - Object (plain)      — a record keyed by field name. Consumers
//                           pull named keys and return a shallow copy
//                           with those keys removed (null if empty).
//
// Refs:
//   - Value-position refs in distribution kwargs (e.g. `Normal(mu =
//     ref a)`) resolve through `env`, the same map the sampler /
//     traceeval expects.
//   - Measure-position refs (e.g. `joint(M1ref, M2ref)`) resolve via
//     opts.resolveMeasureRef(name) → ir | null. The caller typically
//     supplies a closure over `orchestrator.expandMeasureIR(name,
//     derivations)`.

const samplerLib = require('./sampler');

// =====================================================================
// Shape helpers — consume-from-front, return-rest. Centralised so each
// measure-kind dispatch stays small.
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
 *   - 1-key record (rare): not allowed — scalar leaves don't consume
 *     from records (the caller should have routed through a field).
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
 * with `name` removed. Throws when the key isn't present — the
 * caller's shape didn't match the measure's declared fields.
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
// Per-IR-kind dispatch
// =====================================================================

function logDensityConsume(ir, value, env, opts) {
  opts = opts || {};
  if (!ir) throw new Error('density: missing IR');

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
    return logDensityConsume(expanded, value, env, opts);
  }

  if (ir.kind !== 'call') {
    throw new Error('density: unsupported IR kind \'' + ir.kind + '\'');
  }

  const op = ir.op;

  // Scalar leaf distribution. Single-entry consume from `value`,
  // evaluate logpdf at that entry, no further recursion.
  if (samplerLib.isKnownDistribution(op)) {
    const { head, rest } = consumeScalar(value);
    const entry = samplerLib.lookupDistribution(ir);
    const params = samplerLib.resolveParams(ir, entry, env);
    const logp = entry.logpdfFn(head, ...params);
    return { logp, rest };
  }

  // weighted(w, base) — density adds log(w). Negative or zero weights
  // collapse to -Infinity (zero-mass atom). The orchestrator may have
  // pre-computed a uniform shift via `logShift` on the derivation, but
  // here we only have the IR — compute fresh from the weight expr.
  if (op === 'weighted') {
    const w = +samplerLib.evaluateExpr(ir.args[0], env);
    const { logp, rest } = logDensityConsume(ir.args[1], value, env, opts);
    if (!(w > 0)) return { logp: -Infinity, rest };
    return { logp: logp + Math.log(w), rest };
  }
  if (op === 'logweighted') {
    const lw = +samplerLib.evaluateExpr(ir.args[0], env);
    const { logp, rest } = logDensityConsume(ir.args[1], value, env, opts);
    return { logp: logp + lw, rest };
  }

  // truncate(M, S) — indicator over S × base density. Per spec §06
  // does NOT normalise, so density inside S equals base density;
  // outside S it's -Infinity. The set descriptor is opaque IR here;
  // the caller's opts.parseSet bridges to numeric bounds.
  if (op === 'truncate') {
    const parseSet = opts.parseSet;
    if (typeof parseSet !== 'function') {
      throw new Error('density: truncate requires parseSet opt');
    }
    const setDescr = parseSet(ir.args[1]);
    const { logp, rest } = logDensityConsume(ir.args[0], value, env, opts);
    const consumed = inferConsumedScalar(value, rest);
    if (consumed != null && !inSet(consumed, setDescr)) {
      return { logp: -Infinity, rest };
    }
    return { logp, rest };
  }

  if (op === 'normalize') {
    // normalize(base) = base / totalmass(base). Density shifts by
    // -log(totalmass(base)). The caller must supply the parent's
    // logTotalmass via opts.measureLogTotalmass — we can't compute
    // it analytically from the IR alone for arbitrary base measures.
    const getLTM = opts.measureLogTotalmass;
    const baseLTM = typeof getLTM === 'function' ? getLTM(ir.args[0]) : 0;
    const { logp, rest } = logDensityConsume(ir.args[0], value, env, opts);
    return { logp: logp - baseLTM, rest };
  }

  // record / kwarg-joint: { kind:'call', op:'record'|'joint',
  // fields: [{name, value}, ...] }. Consume named keys in declared
  // order; sum component logp's.
  if ((op === 'record' || op === 'joint') && Array.isArray(ir.fields)) {
    const fields = ir.fields;
    let cur = value;
    let total = 0;
    for (let i = 0; i < fields.length; i++) {
      const f = fields[i];
      const { head, rest: outerRest } = consumeField(cur, f.name);
      const { logp, rest: innerRest } = logDensityConsume(f.value, head, env, opts);
      if (!isEmptyRest(innerRest)) {
        throw new Error('density: field \'' + f.name
          + '\' did not fully consume its value');
      }
      total += logp;
      cur = outerRest;
    }
    return { logp: total, rest: cur };
  }

  // Positional joint: args = [M1, M2, ...]. Consume each Mi from the
  // current rest. Independent components → sum logp's, no env
  // threading (each Mi sees the same env).
  if (op === 'joint' && Array.isArray(ir.args)) {
    let cur = value;
    let total = 0;
    for (let i = 0; i < ir.args.length; i++) {
      const r = logDensityConsume(ir.args[i], cur, env, opts);
      total += r.logp;
      cur = r.rest;
    }
    return { logp: total, rest: cur };
  }

  // jointchain(M, K1, K2, ...): like positional joint for consumption,
  // but each subsequent kernel sees the prior components' consumed
  // values bound in env under the variate names declared at lift
  // time. Today's jointchain isn't yet a derivation kind — left as a
  // marker so callers get a clear error rather than silent fall-through.
  if (op === 'jointchain') {
    throw new Error('density: jointchain not yet wired through density.js');
  }

  // iid(M, n): n copies of M's footprint. Loop consume.
  if (op === 'iid' && Array.isArray(ir.args) && ir.args.length === 2) {
    const n = samplerLib.evaluateExpr(ir.args[1], env) | 0;
    if (n < 0) throw new Error('density: iid count negative: ' + n);
    let cur = value;
    let total = 0;
    for (let i = 0; i < n; i++) {
      const r = logDensityConsume(ir.args[0], cur, env, opts);
      total += r.logp;
      cur = r.rest;
    }
    return { logp: total, rest: cur };
  }

  throw new Error('density: unsupported measure op \'' + op + '\'');
}

/**
 * Strict wrapper: walks the IR alongside the value, then asserts the
 * value was fully consumed. Returns just the log-density. Any leftover
 * `rest` indicates a shape mismatch between the measure and value —
 * surfaced as an error rather than scoring an off-by-one slice.
 */
function logDensity(ir, value, env, opts) {
  const { logp, rest } = logDensityConsume(ir, value, env, opts);
  if (!isEmptyRest(rest)) {
    throw new Error('logDensity: value has unconsumed leftover after walking IR'
      + ' (op=' + (ir && ir.op) + ')');
  }
  return logp;
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
  logDensityConsume,
  logDensity,
  // Test/debug surface — exposes the shape helpers in case callers
  // outside the dispatch want to compose their own consumers.
  _internal: { consumeScalar, consumeField, isEmptyRest, inSet },
};

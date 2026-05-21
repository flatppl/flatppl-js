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

const samplerLib = require('./sampler.ts');
const valueLib   = require('./value.ts');

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
  // Shape-explicit Value (Phase 5): a rank-1 Value behaves like a
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
 * (MvNormal in Phase 6, Dirichlet in follow-ups). Returns:
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
function consumeVector(value, n) {
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
  const entry = samplerLib.lookupDistribution(ir);
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
    const ptAt = valueLib.isValue(pv) ? (i) => pv.data[i] : (i) => pv[i];
    if (refNames.length === 0 && !overlayKeys) {
      const params = samplerLib.resolveParams(ir, entry, baseEnv);
      for (let i = 0; i < N; i++) {
        acc[i] += entry.logpdfFn(ptAt(i), ...params);
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
      acc[i] += entry.logpdfFn(ptAt(i), ...params);
    }
    return null;
  }

  const { head, rest } = consumeScalar(value);
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
  //
  // refArrays entries are either Float64Array (scalar-atom parent) or
  // Value (Phase 7b vector-atom parent). Pre-compute the access
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
        accessors[j] = (i) => data[i];
      } else {
        const tailDims = shape.slice(1);
        const tailLen = tailDims.reduce((a, b) => a * b, 1);
        const data = v.data;
        accessors[j] = (i) => ({
          shape: tailDims,
          data: data.subarray(i * tailLen, (i + 1) * tailLen),
        });
      }
    } else {
      const arr = v;
      accessors[j] = (i) => arr[i];
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

function walkPushfwd(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
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
  // Consume the head — pushfwd's variate footprint matches M's.
  const { head: y, rest } = consumeScalar(value);
  // Compute x = f_inv(y). Atom-indep eval against baseEnv ∪ overlay.
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

// ---- Discrete-selector mixture (engine-concepts §11) ---------------

// Footprint size of a `rest` (how many scalar slots remain). Used only
// to assert that every select branch consumed the SAME observation
// footprint — components of a superpose/mixture/ifelse share one
// variate space (spec §06), so their rests must agree.
function restSize(r) {
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
function selectBranchRefClass(ir, depth) {
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

function walkSelect(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
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
  // A runtime-weight spec must have been resolved to literal
  // logweights by the materialiser before the IR reaches the density
  // worker (engine-concepts §11 MC-weight selector). If it survived
  // here, fall-through would silently score a raw (equal-weight)
  // superpose — a wrong number. Fail loud instead.
  if (ir.weightsFrom != null && ir.logweights == null) {
    throw new Error('density: select carries an unresolved runtime-weight '
      + 'spec (weightsFrom) — the materialiser must estimate and substitute '
      + 'literal logweights before density evaluation');
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
  select:      walkSelect,
  MvNormal:    walkMvNormal,
};

// ---- Per-atom scalar contribution helpers ---------------------------

// Evaluate `wIR` once over the whole atom batch via sampler.evaluateExprN,
// then accumulate via `combine(acc, i, value)`. The single batched call
// replaces what used to be N scalar evaluateExpr loops — same env-
// precedence (overlay > refArrays > baseEnv) but enforced inside
// evaluateExprN.
// =====================================================================
// MvNormal — closed-form multivariate Normal density (Phase 6)
// =====================================================================
//
// Per spec §08: MvNormal(mu, cov) is the n-variate normal with mean
// vector mu and covariance matrix cov. Density:
//
//   log p(x; mu, cov) = -½·n·log(2π) - ½·log|cov| - ½·(x-mu)ᵀ·cov⁻¹·(x-mu)
//
// With L = lower_cholesky(cov):
//
//   log|cov| = 2·Σ_i log(L_ii)
//   cov⁻¹·(x-mu) = L⁻ᵀ·L⁻¹·(x-mu)
//   (x-mu)ᵀ·cov⁻¹·(x-mu) = ‖L⁻¹·(x-mu)‖² = ‖y‖²
//                          where y is the forward-solve of L·y = x-mu
//
// We compute L once (atom-indep) and the Mahalanobis quadratic per
// atom. For the simplest case — atom-indep mu/cov, atom-shared
// observation x — the whole logpdf is a single scalar broadcast over
// the N-atom accumulator. Per-atom obs (e.g. via a vector-valued
// refArray) is supported by looping over atoms and re-running the
// forward solve.
//
// Atom-batched mu / cov (per-atom parameter pinning) is deferred —
// it'd require a per-atom Cholesky which doesn't arise in current
// engine usage (matMvNormal materialises against atom-indep params).
function walkMvNormal(ir, value, refArrays, N, opts, acc, baseEnv, overlay) {
  const kwargs = ir.kwargs || {};
  if (!kwargs.mu || !kwargs.cov) {
    throw new Error('density: MvNormal requires mu and cov kwargs');
  }
  // Resolve mu and cov atom-indep. evaluateExpr accepts the baseEnv +
  // overlay; the IR-level Value-aware mul/add in sampler ensures the
  // shape-rich paths dispatch correctly.
  const muEnv = Object.assign({}, baseEnv);
  if (overlay) Object.assign(muEnv, overlay);
  const muRaw = samplerLib.evaluateExpr(kwargs.mu, muEnv);
  const covRaw = samplerLib.evaluateExpr(kwargs.cov, muEnv);
  // Normalise both to Values.
  const valueLibLocal = require('./value.ts');
  const valueOpsLocal = require('./value-ops.ts');
  const muValue = valueLibLocal.asValue(muRaw);
  const covValue = Array.isArray(covRaw) && covRaw.length > 0 && Array.isArray(covRaw[0])
    ? valueOpsLocal._nestedToValue(covRaw)
    : valueLibLocal.asValue(covRaw);
  if (muValue.shape.length !== 1) {
    throw new Error('density: MvNormal mu must be a vector, got shape='
      + JSON.stringify(muValue.shape));
  }
  const n = muValue.shape[0];
  if (covValue.shape.length !== 2 || covValue.shape[0] !== n || covValue.shape[1] !== n) {
    throw new Error('density: MvNormal cov must be ' + n + 'x' + n
      + ', got shape=' + JSON.stringify(covValue.shape));
  }
  // Cholesky.
  const L = samplerLib._internal.ARITH_OPS.lower_cholesky(covValue);
  // log|cov| = 2 * sum_i log(L_ii). L is shape=[n, n] row-major.
  let logDet = 0;
  for (let i = 0; i < n; i++) logDet += Math.log(L.data[i * n + i]);
  logDet *= 2;
  const logNormConst = -0.5 * (n * Math.log(2 * Math.PI) + logDet);
  // Consume the observation vector: length-n head off the value.
  const { head: x, rest } = consumeVector(value, n);
  // Mahalanobis^2 = ‖L⁻¹ (x - mu)‖². Atom-indep observation → one
  // forward solve, scalar broadcast across acc.
  // (Per-atom obs / params: extend by looping; deferred for now.)
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = x[i] - muValue.data[i];
  // Forward substitution: y_i = (d_i - Σ_{k<i} L_ik y_k) / L_ii.
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = d[i];
    for (let k = 0; k < i; k++) sum -= L.data[i * n + k] * y[k];
    y[i] = sum / L.data[i * n + i];
  }
  let mahal2 = 0;
  for (let i = 0; i < n; i++) mahal2 += y[i] * y[i];
  const logp = logNormConst - 0.5 * mahal2;
  for (let i = 0; i < N; i++) acc[i] += logp;
  return rest;
}

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
  _internal: { consumeScalar, consumeField, consumeVector, isEmptyRest, inSet },
};

'use strict';

// =====================================================================
// traceeval.js — unified generative + scoring evaluator for FlatPPL
// measure expressions.
// =====================================================================
//
// Why this module exists
// ----------------------
// FlatPPL has two operations on a measure expression that share most
// of their structural logic:
//
//   sampling: given a measure M, produce a draw x ~ M.
//   scoring:  given a measure M and a value x, compute log p(x | M).
//
// Implementing them separately would duplicate the recursion over
// joint / iid / weighted / logweighted / … twice — once "produce a
// value", once "consume a value". Trace-based execution (Anglican,
// Gen, Pyro) collapses both into ONE walker that takes a measure IR
// *and* a (possibly-partial) "observed" template:
//
//   - At each leaf distribution site, if `observed` provides a value
//     at that site, use it (clamped) and tally `logpdf(value | params)`
//     without advancing the RNG.
//   - Otherwise sample, tally if requested, advance the RNG.
//
// The same code thereby implements:
//   sampling             (observed=undefined, tally='none')
//   scoring              (observed=full,      tally='all')
//   likelihoodof, bayesupdate
//                        (observed=obs only,  tally='clamped'  +
//                         per-i env carrying prior θ)
//   joint trace eval     (observed=undefined, tally='all')
//   MC-marginalisation   (observed=partial,   tally='clamped'  +
//                         repeat M times to average exp(logp))
//
// The orchestrator decides the mode and passes the right arguments;
// the walker doesn't care which it is.
//
// Public API
// ----------
//   walk(state, ir, env, observed, opts)
//     state    - rng.State; consumed only at unobserved leaf sites.
//     ir       - measure IR (FlatPIR call form). Must be CLOSED w.r.t.
//                kernel boundaries — the orchestrator's closure walk
//                substitutes those upstream. Refs that remain are
//                ordinary self-refs that resolve through env.
//     env      - per-i env mapping ref-names to numerics, same shape
//                as samplerLib.evaluateExpr's env.
//     observed - same value-shape as ir's variates: a number for a
//                leaf, a record (plain object keyed by field name)
//                for joint/record, an Array for iid. `undefined`/
//                `null` at a site means "not observed; sample fresh."
//     opts.tally - 'none' | 'clamped' | 'all'.
//                  'none'    — never accumulate; pure-sampling fast
//                              path. (Today's sampleN behaviour.)
//                  'clamped' — only at observed leaves; what
//                              bayesupdate / likelihoodof need: score
//                              the data, ignore the latent draws.
//                  'all'     — at every leaf (sampled or clamped);
//                              joint log-density of the whole trace.
//     opts.resolveMeasureRef - optional `(name) → measureIR`. Used
//                              when fields of a joint, the measure
//                              arg of iid/weighted/logweighted, etc.
//                              are written as self-refs to other
//                              measure bindings rather than inline
//                              calls. The orchestrator supplies a
//                              closure over its bindings map.
//     opts.resolveValueRef   - optional `(name, state) → [value, newState]`.
//                              Lazy on-demand resolution of value-
//                              position self-refs the env doesn't
//                              already carry. The walker calls this
//                              whenever a leaf-distribution kwarg /
//                              iid count / weighted weight refers to
//                              a binding by name and env is empty for
//                              that name — typical of rand(state, M)
//                              where M's distribution params depend on
//                              stochastic ancestors the env hasn't
//                              been pre-populated with. The resolver
//                              is expected to thread state through
//                              any recursive sampling it does, and to
//                              cache its result via env so two refs to
//                              the same name share one draw. The
//                              orchestrator supplies a closure that
//                              knows how to sample stochastic
//                              bindings (recursively, through this
//                              same walker) and inline deterministic
//                              ones.
//   → { value, logp, state }
//     value - the trace value (matches the IR's variate shape, with
//             clamped sites filled with the observed value, unclamped
//             sites with the sampled value).
//     logp  - log-density accumulator. -Infinity is allowed (zero-
//             mass atoms, out-of-support points). NaN is treated as a
//             hard error and signalled via thrown exception so callers
//             don't propagate silent corruption.
//     state - trailing rng state.
//
// Operations supported in the IR
// ------------------------------
//   leaf distribution  — any op in samplerLib's REGISTRY. Sampled via
//                        a one-shot prng-bound factory; scored via
//                        REGISTRY[op].logpdfFn.
//   joint / record     — { kind: 'call', op: 'joint'|'record',
//                          fields: [{name, value}, ...] }. Field-wise
//                          recursion; observed split per field name.
//                          (joint and record share IR shape; the
//                          walker treats them identically here.)
//   iid                — { kind: 'call', op: 'iid', args: [M, n] }.
//                        Recurses n times, reusing env (params const
//                        across the inner draws — that's what 'iid'
//                        means here). observed split per index.
//   weighted           — { kind: 'call', op: 'weighted',
//                          args: [w, M] }. Density:
//                          log p_w(x) = log w + log p_M(x).
//                          Weight evaluated against env. Negative or
//                          NaN weights raise; w=0 yields -Infinity.
//   logweighted        — { kind: 'call', op: 'logweighted',
//                          args: [g, M] }. Adds g to the tally,
//                          recurses on M. -Infinity is permitted.
//
// Reference measures
// ------------------
// Each leaf distribution carries an implicit reference (Lebesgue for
// continuous, counting for discrete). Joint / iid carry the *product*
// reference automatically — summing per-leaf logpdfs equals the logpdf
// w.r.t. the product reference. Other measure-algebra ops
// (`normalize`, `totalmass`, `pushfwd`) need non-local integrals or a
// Jacobian and are deliberately NOT handled here. Bindings using them
// will need either a separate primitive or pre-lowering by the
// orchestrator.
//
// Function-of-variate weights
// ---------------------------
// `weighted(fn(_), M)` — the weight is a function of M's own variate
// — is NOT handled in this first cut: we'd need to bind `_` to the
// trace value at this site before evaluating the weight expression.
// In practice the orchestrator lowers that pattern to a per-binding
// logweighted derivation that runs `evaluateN` over the weight IR
// against the cached samples of the base, so it works at the
// orchestrator layer without the trace evaluator needing this
// capability. If we later need it for nested measures, extend
// walkWeighted to evaluate the weight against env extended with a
// caller-chosen variate name.

const samplerLib = require('./sampler');

/**
 * Top-level entry. See header comment for full semantics.
 */
function walk(state, ir, env, observed, opts) {
  opts = opts || {};
  const tally = opts.tally || 'none';
  if (tally !== 'none' && tally !== 'clamped' && tally !== 'all') {
    throw new Error(`traceeval.walk: opts.tally must be 'none' | 'clamped' | 'all', got '${tally}'`);
  }
  const ctx = {
    tally,
    resolveRef: opts.resolveMeasureRef || null,
    resolveValueRef: opts.resolveValueRef || null,
  };
  const r = walkInner(state, ir, env, observed, ctx);
  if (Number.isNaN(r.logp)) {
    // A NaN in the tally is almost always an upstream bug (e.g.
    // logpdf called outside support without -Infinity protection).
    // Surface it loudly rather than silently propagating.
    throw new Error('traceeval.walk: log-density tally became NaN — check inputs');
  }
  return r;
}

function walkInner(state, ir, env, observed, ctx) {
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
    return walkInner(state, inner, env, observed, ctx);
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
    return walkLeaf(state, ir, env, observed, ctx);
  }

  // Dispatch through MEASURE_OP_WALKERS. Adding a new measure-algebra
  // walker (pushfwd, truncate, …) is one entry here plus the handler
  // function — no edits to walkInner itself.
  const handler = MEASURE_OP_WALKERS[op];
  if (handler) return handler(state, ir, env, observed, ctx);
  throw new Error(
    `traceeval: op '${op}' is not a measure expression we can ` +
    `sample or score. Known: leaf distributions, ` +
    Object.keys(MEASURE_OP_WALKERS).join(', ') + '.'
  );
}

function walkLeaf(state, ir, env, observed, ctx) {
  // Pre-fill env with any value-position refs in the kwargs that
  // aren't already known. The leaf's distribution params (`mu = ref a`,
  // `sigma = ref b`, …) may reference bindings the caller hasn't
  // materialised yet — typical of rand(state, M) where M's params
  // depend on stochastic ancestors. fillEnvFromRefs threads state
  // through any recursive sampling the resolver does.
  state = fillEnvFromRefs(state, ir, env, ctx);
  const entry = samplerLib.lookupDistribution(ir);
  const params = samplerLib.resolveParams(ir, entry, env);

  let value, nextState = state;
  if (observed != null) {
    // Clamped: use the observed numeric, no RNG advance.
    value = +observed;
  } else {
    // Sampled: build a one-shot prng-bound sampler. Per-leaf factory
    // build is fine here because `walk()` itself is called once per
    // outer atom in hot paths — the orchestrator's per-binding
    // sampleN already amortises factory cost across atoms via
    // makeParametricSampler. Reusing leaf samplers across walk() calls
    // would require a different (worker-level) primitive; out of scope
    // for the basic walker.
    const prng = samplerLib.makePhiloxPrngAdapter(state);
    const sampler = entry.randFn.factory(...params, { prng });
    value = sampler();
    nextState = prng.getState();
  }

  let logp = 0;
  if (ctx.tally === 'all' || (ctx.tally === 'clamped' && observed != null)) {
    logp = entry.logpdfFn(value, ...params);
  }
  return { value, logp, state: nextState };
}

function walkJoint(state, ir, env, observed, ctx) {
  // Two surface forms:
  //
  //   * kwarg-joint / record: ir.fields = [{ name, value }, ...]. The
  //     output variate is keyed by field name; `observed` is a record
  //     (plain object) keyed the same way. A missing key means "not
  //     observed at that field" (sampled fresh, no clamped tally).
  //
  //   * positional joint:    ir.args   = [M1, M2, ...]. The output
  //     variate is a vector built by concatenating component variates
  //     (cat semantics per spec §06). `observed`, if provided, is a
  //     flat array consumed left-to-right by each component's
  //     footprint. We split it by trial-walking each component
  //     against the *remaining* tail and threading the unused
  //     suffix to the next.
  if (Array.isArray(ir.fields)) {
    const fields = ir.fields;
    const out = {};
    let logp = 0;
    let st = state;
    // Env-threading for scoring: when an observation pins each
    // field's value, write the observed value into env keyed by field
    // name BEFORE walking subsequent fields. This is the joint-density
    // semantics — `logdensityof(joint(theta1=M1, theta2=M2, y=K(theta1,
    // theta2)), {theta1=0, theta2=1, y=0})` evaluates y's leaf at the
    // OBSERVED θ values, not at the prior per-atom θ samples.
    // Variate name === binding name in FlatPPL by construction (the
    // jointchain rewrite preserves names through extractRecordFields).
    //
    // We don't mutate the caller's env — shallow-copy on demand so
    // independent atoms or sibling walks aren't entangled.
    const walkEnv = (observed != null) ? Object.assign({}, env) : env;
    for (let i = 0; i < fields.length; i++) {
      const f = fields[i];
      const sub = observed != null && Object.prototype.hasOwnProperty.call(observed, f.name)
        ? observed[f.name]
        : undefined;
      const r = walkInner(st, f.value, walkEnv, sub, ctx);
      out[f.name] = r.value;
      logp += r.logp;
      st = r.state;
      // Thread the consumed value into env for subsequent fields.
      // We write under TWO keys: the surface field name (so a leaf
      // refs the binding by name) and the source binding name (so a
      // leaf refs the anon binding that the orchestrator's
      // expandMeasureIR pass attached as `source` — typically a
      // kernel-substituted anon whose name doesn't match any
      // user-visible binding). Both writes are local to the joint
      // walk via the shadowed env copy above; structured field
      // values still get written but downstream refs to them would
      // be uncommon.
      if (observed != null && Object.prototype.hasOwnProperty.call(observed, f.name)) {
        walkEnv[f.name] = r.value;
        if (f.source && f.source !== f.name) {
          walkEnv[f.source] = r.value;
        }
      }
    }
    return { value: out, logp, state: st };
  }
  if (Array.isArray(ir.args)) {
    const components = ir.args;
    // For sampling (observed == null) we don't need to split — each
    // sub-walk just samples and we concatenate values into an array.
    // For scoring (observed != null) we slice the observation per
    // component's footprint, computed by inferComponentSize.
    const out = new Array(components.length);
    let logp = 0;
    let st = state;
    let cursor = 0;
    for (let i = 0; i < components.length; i++) {
      let sub;
      if (observed != null) {
        const k = inferComponentSize(components[i], env, ctx);
        sub = Array.isArray(observed) || (observed && observed.BYTES_PER_ELEMENT)
          ? sliceObserved(observed, cursor, cursor + k)
          : undefined;
        cursor += k;
      }
      const r = walkInner(st, components[i], env, sub, ctx);
      out[i] = r.value;
      logp += r.logp;
      st = r.state;
    }
    return { value: out, logp, state: st };
  }
  throw new Error("traceeval: joint with neither fields nor args");
}

// Footprint inference for positional joint's observation-splitting.
// Each measure kind contributes a fixed-or-derivable number of scalar
// entries to the cat'd output: scalar leaf → 1, iid(M, n) → n × M's
// footprint, joint(args) → sum of components, joint(fields) → sum of
// fields, weighted/logweighted → same as base. Refs resolve through
// the same resolveRef the walker uses for measure refs.
function inferComponentSize(ir, env, ctx) {
  if (!ir) return 1;
  if (ir.kind === 'ref' && ir.ns === 'self' && ctx && ctx.resolveRef) {
    const inner = ctx.resolveRef(ir.name);
    if (inner) return inferComponentSize(inner, env, ctx);
  }
  if (ir.kind !== 'call') return 1;
  const op = ir.op;
  if (samplerLib.isKnownDistribution(op)) return 1;
  if (op === 'iid' && Array.isArray(ir.args) && ir.args.length === 2) {
    const n = samplerLib.evaluateExpr(ir.args[1], env) | 0;
    return Math.max(0, n) * inferComponentSize(ir.args[0], env, ctx);
  }
  if (op === 'weighted' || op === 'logweighted') {
    return inferComponentSize(ir.args[1], env, ctx);
  }
  if (op === 'truncate' || op === 'normalize') {
    return inferComponentSize(ir.args[0], env, ctx);
  }
  if ((op === 'joint' || op === 'record') && Array.isArray(ir.fields)) {
    let s = 0;
    for (const f of ir.fields) s += inferComponentSize(f.value, env, ctx);
    return s;
  }
  if (op === 'joint' && Array.isArray(ir.args)) {
    let s = 0;
    for (const a of ir.args) s += inferComponentSize(a, env, ctx);
    return s;
  }
  return 1;
}

function sliceObserved(obs, lo, hi) {
  if (obs && obs.BYTES_PER_ELEMENT) return obs.subarray(lo, hi);
  return obs.slice(lo, hi);
}

function walkIid(state, ir, env, observed, ctx) {
  // iid(M, n): n iid draws of measure M sharing params. observed, if
  // present, must be array-like of length n. Inner env is shared —
  // params do NOT change across the n inner draws (that's what makes
  // it 'iid' vs a vectorised call with per-index params).
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`traceeval: iid expected 2 args (measure, count), got ${args.length}`);
  }
  const M = args[0];
  // iid count may reference bindings (e.g. `iid(M, n)` where n is a
  // fixed-phase value binding) — pre-fill before evaluating.
  state = fillEnvFromRefs(state, args[1], env, ctx);
  const n = samplerLib.evaluateExpr(args[1], env) | 0;
  if (n < 0) throw new Error(`traceeval: iid count must be non-negative, got ${n}`);

  if (observed != null) {
    const obsLen = observed.length;
    if (obsLen !== n) {
      throw new Error(
        `traceeval: iid observed length ${obsLen} does not match count ${n}`
      );
    }
  }

  const out = new Array(n);
  let logp = 0;
  let st = state;
  for (let j = 0; j < n; j++) {
    const sub = observed != null ? observed[j] : undefined;
    const r = walkInner(st, M, env, sub, ctx);
    out[j] = r.value;
    logp += r.logp;
    st = r.state;
  }
  return { value: out, logp, state: st };
}

function walkWeighted(state, ir, env, observed, ctx) {
  // weighted(w, M): density p_w(x) = w * p_M(x). On the log scale:
  //   log p_w(x) = log(w) + log p_M(x)
  // The weight expression is evaluated against the current env (not
  // against the trace value of M). Function-of-variate weights are
  // NOT handled here — see header comment.
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`traceeval: weighted expected 2 args (weight, measure), got ${args.length}`);
  }
  const r = walkInner(state, args[1], env, observed, ctx);
  let logp = r.logp;
  let st = r.state;
  if (ctx.tally !== 'none') {
    // Pre-fill env so refs inside the weight expression resolve.
    st = fillEnvFromRefs(st, args[0], env, ctx);
    const w = samplerLib.evaluateExpr(args[0], env);
    if (Number.isNaN(w)) {
      throw new Error('traceeval: weighted weight evaluated to NaN');
    }
    if (w < 0) {
      throw new Error(`traceeval: weighted weight must be non-negative, got ${w}`);
    }
    if (w === 0) {
      logp += -Infinity;
    } else {
      logp += Math.log(w);
    }
  }
  return { value: r.value, logp, state: st };
}

function walkLogWeighted(state, ir, env, observed, ctx) {
  // logweighted(g, M): log-domain weight g added directly. -Infinity
  // is permitted (zero-mass atom). NaN is a hard error.
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`traceeval: logweighted expected 2 args, got ${args.length}`);
  }
  const r = walkInner(state, args[1], env, observed, ctx);
  let logp = r.logp;
  let st = r.state;
  if (ctx.tally !== 'none') {
    st = fillEnvFromRefs(st, args[0], env, ctx);
    const g = samplerLib.evaluateExpr(args[0], env);
    if (Number.isNaN(g)) {
      throw new Error('traceeval: logweighted weight evaluated to NaN');
    }
    logp += g;
  }
  return { value: r.value, logp, state: st };
}

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
function fillEnvFromRefs(state, ir, env, ctx) {
  if (!ctx || !ctx.resolveValueRef) return state;
  const refs = new Set();
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

function collectValueRefs(ir, out) {
  if (ir == null || typeof ir !== 'object') return;
  if (ir.kind === 'ref' && ir.ns === 'self') { out.add(ir.name); return; }
  if (Array.isArray(ir.args)) for (const a of ir.args) collectValueRefs(a, out);
  if (ir.kwargs) for (const k in ir.kwargs) collectValueRefs(ir.kwargs[k], out);
  // Don't descend into `fields` / `body` — those are measure / scope
  // boundaries handled by the surrounding walker's recursion.
}

// =====================================================================
// Measure-op walkers
// =====================================================================
//
// One entry per IR op the walker handles structurally (above and
// beyond leaf distributions, which dispatch via samplerLib's REGISTRY).
// Adding a new measure op (pushfwd, truncate, relabel, …) is one
// entry here + one handler function — no edits to walkInner.

/**
 * lawof(M): per spec §sec:lawof, the law of a variate equals the
 * measure it was drawn from — `lawof(draw(M)) ≡ M`. From the walker's
 * perspective lawof is a pass-through wrapper: sampling from
 * `lawof(X)` is the same as sampling from X (after any self-ref
 * inside resolves through the orchestrator's measure resolver).
 *
 * This lets inline forms like `rand(rstate, lawof(obs))` walk
 * without first being canonicalised to the corresponding measure
 * binding — the named form `d = lawof(obs); rand(rstate, d)` worked
 * already because expandMeasureIR resolves d's alias derivation
 * before traceeval sees the IR; the inline form needs the walker
 * to do the same unwrapping itself.
 */
function walkLawof(state, ir, env, observed, ctx) {
  const args = ir.args || [];
  if (args.length !== 1) {
    throw new Error(`traceeval: lawof expected 1 arg, got ${args.length}`);
  }
  return walkInner(state, args[0], env, observed, ctx);
}

/**
 * draw(M): unlike lawof, draw is value-position — its result is a
 * variate, not a measure. The walker doesn't normally encounter
 * draw at measure-position; when it does (e.g. an inlined `draw(M)`
 * surfacing in a kernel body the orchestrator hasn't canonicalised),
 * fall back to the spec identity `lawof(draw(M)) ≡ M` and treat draw
 * as a pass-through to its inner measure. Same handler shape as
 * walkLawof for symmetry.
 */
function walkDraw(state, ir, env, observed, ctx) {
  const args = ir.args || [];
  if (args.length !== 1) {
    throw new Error(`traceeval: draw expected 1 arg, got ${args.length}`);
  }
  return walkInner(state, args[0], env, observed, ctx);
}

const MEASURE_OP_WALKERS = {
  joint:       walkJoint,
  record:      walkJoint,
  iid:         walkIid,
  weighted:    walkWeighted,
  logweighted: walkLogWeighted,
  lawof:       walkLawof,
  draw:        walkDraw,
};

module.exports = { walk };

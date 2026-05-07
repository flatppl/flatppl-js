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

  switch (op) {
    case 'joint':
    case 'record':
      return walkJoint(state, ir, env, observed, ctx);
    case 'iid':
      return walkIid(state, ir, env, observed, ctx);
    case 'weighted':
      return walkWeighted(state, ir, env, observed, ctx);
    case 'logweighted':
      return walkLogWeighted(state, ir, env, observed, ctx);
    default:
      throw new Error(
        `traceeval: op '${op}' is not a measure expression we can ` +
        `sample or score. Known: leaf distributions, joint, record, ` +
        `iid, weighted, logweighted.`
      );
  }
}

function walkLeaf(state, ir, env, observed, ctx) {
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
  // joint/record fields: [{ name, value }, ...]. observed is split per
  // field name (a missing key means "not observed at that field").
  const fields = ir.fields || [];
  const out = {};
  let logp = 0;
  let st = state;
  for (let i = 0; i < fields.length; i++) {
    const f = fields[i];
    const sub = observed != null && Object.prototype.hasOwnProperty.call(observed, f.name)
      ? observed[f.name]
      : undefined;
    const r = walkInner(st, f.value, env, sub, ctx);
    out[f.name] = r.value;
    logp += r.logp;
    st = r.state;
  }
  return { value: out, logp, state: st };
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
  if (ctx.tally !== 'none') {
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
  return { value: r.value, logp, state: r.state };
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
  if (ctx.tally !== 'none') {
    const g = samplerLib.evaluateExpr(args[0], env);
    if (Number.isNaN(g)) {
      throw new Error('traceeval: logweighted weight evaluated to NaN');
    }
    logp += g;
  }
  return { value: r.value, logp, state: r.state };
}

module.exports = { walk };

'use strict';

// profile-plan.js — profile-plot UI support extracted from the
// orchestrator. Maps a callable's input signature to sweepable axis
// ranges: base-set resolution, preset/domain matching, 4-sigma
// quantile clipping, and per-profile IR inlining. A leaf w.r.t. the
// split: depends only on ir-shared (resolveConstant, parseSetIR,
// lowerSafe, NAMED_SET_NAMES) and histogram (quantileSorted), and has
// ZERO internal callers in orchestrator (only reached via the public
// API) — the facade re-bind exists purely for module.exports parity.

const { quantileSorted } = require('./histogram.ts');
const {
  resolveConstant,
  parseSetIR,
  lowerSafe,
  NAMED_SET_NAMES,
} = require('./ir-shared.ts');

/**
 * Resolve the value-set of a paramSources entry to a structural
 * descriptor the profile-plot UI can use to pick an axis range.
 *
 * Returns one of:
 *
 *   { kind: 'interval', lo, hi }   — bounded set; viewer uses [lo, hi]
 *   { kind: 'reals' / 'posreals' / 'nonnegreals' / 'unitinterval' }
 *   { kind: 'integers' / 'posintegers' / 'nonnegintegers' / 'booleans' }
 *   { kind: 'empirical', name }    — binding ref; viewer materialises
 *                                    the binding and computes a 4-σ
 *                                    quantile range
 *   null                           — couldn't resolve; UI falls back
 *                                    to default range for the leaf type
 */
function resolveAxisBaseSet(source, bindings) {
  if (!source) return null;
  // Anonymous placeholder boundaries (`par = _par_`) aren't bound to
  // any elementof — per spec they're equivalent to
  // elementof(anything), so we have no set restriction to surface.
  // The viewer falls back to its leaf-type-based default range.
  if (source.kind === 'placeholder') return null;
  if (source.kind === 'binding') {
    if (!bindings) return { kind: 'empirical', name: source.name };
    const target = bindings.get(source.name);
    if (!target) return { kind: 'empirical', name: source.name };
    // elementof bindings (`x_set = elementof(reals)` /
    // `x_set = elementof(interval(0, 1))`) carry a structural set
    // restriction; surface it so the viewer can use the bounds
    // directly. The `ir.op === 'elementof'` check below is the
    // real discriminator — external / load_data bindings also
    // share binding.type='input' but their RHS op isn't elementof,
    // so they fall through to the empirical fallback below.
    const ir = target.ir
      || (target.effectiveValue && lowerSafe(target.effectiveValue))
      || (target.node && target.node.value && lowerSafe(target.node.value));
    if (ir && ir.kind === 'call' && ir.op === 'elementof'
        && Array.isArray(ir.args) && ir.args.length === 1) {
      const setDescr = parseSetIR(ir.args[0]);
      if (setDescr) return setDescr;
    }
    // Anything else (variates, derived deterministic bindings):
    // there's no static set, but the binding has empirical samples
    // we can quantile-clip into a range at materialise time.
    return { kind: 'empirical', name: source.name };
  }
  return null;
}

/**
 * Find record bindings that look like preset points for a callable's
 * input signature. A "preset point" is any global record(...) binding
 * whose kwarg shape matches the callable's input kwargs (spec §03
 * value types: "Any literal (or fixed, in general) global binding
 * `some_name = record(name1=val1, ...)` can be interpreted as a
 * possibly suitable input"). The profile-plot UI uses this to
 * populate its preset-point dropdown — selecting one fills fixedEnv
 * with its values for non-swept axes.
 *
 * Match rule (Phase 1 — strict, top-level scalars only):
 *   - b.ir.op === 'record'
 *   - the set of record kwarg names equals the set of signature
 *     input kwargNames (no missing inputs, no extra record fields)
 *   - every value is constant-resolvable to a finite number via
 *     resolveConstant, after first unwrapping any fixed(...) marker.
 *     resolveConstant folds literals, named constants, and simple
 *     arithmetic (e.g. `-3.5` lowers to `neg(lit 3.5)`).
 *
 * Returns an array of { name, values, fixedNames } where:
 *   - values    : kwargName → JS number (held-constant + sweepable)
 *   - fixedNames: Set<kwargName> for kwargs wrapped in `fixed(...)` —
 *                 the spec's "hold constant during optimization" hint.
 *                 Tooling uses this to e.g. exclude these kwargs from
 *                 the x-axis sweep selector.
 *
 * Future work (deferred): unify nested record / array preset
 * shapes against record-input signatures.
 */
function findMatchingPresets(signature, bindings) {
  if (!signature || !bindings || !Array.isArray(signature.inputs)) return [];
  const expected = new Set();
  for (const inp of signature.inputs) {
    if (inp.kwargName) expected.add(inp.kwargName);
  }
  if (expected.size === 0) return [];
  const out: any[] = [];
  for (const [name, b] of bindings) {
    if (!b || !b.ir || b.ir.kind !== 'call' || b.ir.op !== 'record') continue;
    // record's IR carries fields (FIELD_FORM in lower.js), not kwargs.
    // Each field's value is typically a ref to an anon-lifted binding
    // (the lift pre-pass moves literals into __anon* bindings);
    // resolveConstant chases refs through the bindings map to recover
    // the underlying value. Before constant-folding, peek through any
    // `fixed(...)` wrapper so the hint doesn't block the match.
    const fields = Array.isArray(b.ir.fields) ? b.ir.fields : [];
    if (fields.length !== expected.size) continue;
    let allMatch = true;
    const values = {};
    const fixedNames = new Set();
    for (const f of fields) {
      if (!expected.has(f.name)) { allMatch = false; break; }
      let inner = f.value;
      // Unwrap fixed(...) at the top of the field value. The wrapper
      // may be a direct call or a ref to a lifted __anon binding
      // whose IR is the fixed() call. resolveConstantInner handles
      // both because the IR-level resolveConstant chases refs.
      if (inner && inner.kind === 'call' && inner.op === 'fixed'
          && Array.isArray(inner.args) && inner.args.length === 1) {
        fixedNames.add(f.name);
        inner = inner.args[0];
      } else if (inner && inner.kind === 'ref' && inner.ns === 'self') {
        const refTarget = bindings.get(inner.name);
        if (refTarget && refTarget.ir && refTarget.ir.kind === 'call'
            && refTarget.ir.op === 'fixed'
            && Array.isArray(refTarget.ir.args)
            && refTarget.ir.args.length === 1) {
          fixedNames.add(f.name);
          inner = refTarget.ir.args[0];
        }
      }
      const v = resolveConstant(inner, bindings, new Set());
      if (v == null) { allMatch = false; break; }
      values[f.name] = v;
    }
    if (!allMatch) continue;
    out.push({ name, values, fixedNames });
  }
  return out;
}

/**
 * Find cartprod bindings that look like preset domains for a
 * callable's input signature. A "preset domain" is any global
 * cartprod(...) binding whose kwarg shape matches the callable's
 * input kwargs (spec §03 value types: "Any literal/fixed global
 * binding like `some_name = cartprod(name1=some_set, ...)` can be
 * interpreted as a possibly suitable domain"). The viewer uses
 * this to populate a "Domain" dropdown — selecting one sets the
 * x-axis range per kwarg, or falls back to the per-binding auto-
 * fit for kwargs whose field is a bare set name rather than a
 * bounded interval.
 *
 * Match rule:
 *   - b.ir.op === 'cartprod'
 *   - the set of cartprod kwarg names equals the set of signature
 *     input kwargNames
 *   - every field is one of
 *       (a) `interval(lo, hi)` with constant-resolvable numeric
 *           bounds (lo < hi), OR
 *       (b) a bare named-set reference: `reals`, `posreals`,
 *           `nonnegreals`, `unitinterval`, `integers`,
 *           `posintegers`, `nonnegintegers`, `booleans`.
 *     (a) contributes a {lo, hi} entry to `ranges`; (b) does not —
 *     the named set is recorded in `setNames` so tooling can display
 *     it, but it's unbounded for axis-fit purposes, and the viewer
 *     uses the per-axis auto-fit instead.
 *
 * Returns an array of { name, ranges, setNames } where:
 *   - ranges:   kwargName → { lo, hi }       (interval bounds only)
 *   - setNames: kwargName → 'reals' | …      (named-set fields only)
 *
 * Future work (deferred): cartpow for vector inputs; unwrapping
 * fixed(...) wrappers around set fields.
 */
function findMatchingDomains(signature, bindings) {
  if (!signature || !bindings || !Array.isArray(signature.inputs)) return [];
  const expected = new Set();
  for (const inp of signature.inputs) {
    if (inp.kwargName) expected.add(inp.kwargName);
  }
  if (expected.size === 0) return [];
  const out: any[] = [];
  for (const [name, b] of bindings) {
    if (!b || !b.ir || b.ir.kind !== 'call' || b.ir.op !== 'cartprod') continue;
    const fields = Array.isArray(b.ir.fields) ? b.ir.fields : [];
    if (fields.length !== expected.size) continue;
    let allMatch = true;
    const ranges = {};
    const setNames = {};
    for (const f of fields) {
      if (!expected.has(f.name)) { allMatch = false; break; }
      // Chase a single ref through lifted __anon bindings so the
      // surface form `cartprod(x = reals)` matches whether the lift
      // pass moved the value out or not.
      let inner = f.value;
      if (inner && inner.kind === 'ref' && inner.ns === 'self') {
        const refTarget = bindings.get(inner.name);
        if (refTarget && refTarget.ir) inner = refTarget.ir;
      }
      // (a) interval(lo, hi) with literal-resolvable bounds.
      if (inner && inner.kind === 'call' && inner.op === 'interval'
          && Array.isArray(inner.args) && inner.args.length === 2) {
        const lo = resolveConstant(inner.args[0], bindings, new Set());
        const hi = resolveConstant(inner.args[1], bindings, new Set());
        if (lo == null || hi == null || !(lo < hi)) { allMatch = false; break; }
        ranges[f.name] = { lo, hi };
        continue;
      }
      // (b) bare named-set reference (parser emits these as const
      // refs in the IR — see builtins.SETS).
      const setName = inner && (inner.kind === 'const' || inner.kind === 'ref')
        ? inner.name : null;
      if (setName && NAMED_SET_NAMES.has(setName)) {
        setNames[f.name] = setName;
        continue;
      }
      allMatch = false; break;
    }
    if (!allMatch) continue;
    out.push({ name, ranges, setNames });
  }
  return out;
}

/**
 * Compute a 4-σ-equivalent central quantile range from a sample
 * array. Returns [lo, hi] or null for an empty input.
 *
 * 4-σ on a unit Gaussian covers central probability erf(4/√2) ≈
 * 0.999937, leaving a tail of ~3.17e-5 per side. With sample sizes
 * typical of the visualizer (5000 / 100000) this is essentially
 * min/max; under heavy-tailed empirical distributions it drops the
 * thinnest tails. Used by the profile-plot UI to set an axis range
 * from a binding-source backref's empirical samples.
 */
function fourSigmaQuantileRange(samples) {
  if (!samples || samples.length === 0) return null;
  if (samples.length === 1) return [samples[0], samples[0]];
  const sorted = Float64Array.from(samples);
  sorted.sort();
  // Two-sided tail mass for ±4σ on a unit Normal: (1 - erf(4/√2)) / 2
  // ≈ 3.1671241833e-5. We use the exact constant rather than
  // computing it inline — it's spec-stable per our docs.
  const ALPHA = 3.1671241833e-5;
  return [
    quantileSorted(sorted, ALPHA),
    quantileSorted(sorted, 1 - ALPHA),
  ];
}

/**
 * Substitute IR for the profile-plot evaluator. Two transformations:
 *
 *   1. (ref self <name>) where <name> is a swept input parameter →
 *      (ref %local <name>). The body uses %local refs for its own
 *      params; transitive deps that surface a self-ref to the same
 *      param need to be rewritten so they pick up the swept value
 *      from the worker's env.
 *
 *   2. (ref self <name>) where <name>'s derivation is evaluate-kind
 *      → substitute the binding's lowered IR inline (recursively
 *      processed). Pulls deterministic transforms (e.g. `a = c *
 *      theta1` or `b = abs(theta1) * theta2`) into the body so the
 *      swept axis propagates through them. Constants and other
 *      truly-self-referential bindings (literals, prior atoms) are
 *      left as-is for the viewer's pre-materialise step to bind via
 *      fixedEnv.
 *
 * Used by the profile-plot UI before sending IR to worker.profileN.
 * Without this pass, sweeping `theta1` through a kernel body whose
 * `mu = a` (with `a = c * theta1`) leaves `a` materialised at a
 * single fixed value — the plot shows a flat line because the swept
 * axis doesn't reach the leaf distributions.
 */
function inlineForProfile(ir, paramNames, bindings, derivations) {
  if (!ir) return ir;
  const paramSet = new Set(paramNames || []);
  const visiting = new Set();
  return walk(ir);

  function walk(node) {
    if (node == null || typeof node !== 'object') return node;
    if (Array.isArray(node)) return node.map(walk);
    if (node.kind === 'ref' && node.ns === 'self') {
      // Swept input → %local rewrite.
      if (paramSet.has(node.name)) {
        return { ...node, ns: '%local' };
      }
      // Evaluable binding → inline. Cycle guard: if we're already
      // expanding this name, leave the ref intact (the cycle would
      // be the analyzer's bug, not ours to mask).
      //
      // Two routes to "evaluable":
      //   1. The derivation table tags this binding as evaluate-kind
      //      (the common case for non-parameterized chains).
      //   2. buildDerivations pruned the binding because it transitively
      //      depends on a parameterized (elementof) ancestor — exactly
      //      the situation profile-plotting is here to handle. Fall
      //      back on the binding's static shape: type='call' with a
      //      lowered IR is evaluable. Without this fallback, sweeping
      //      `f_mu2 = functionof(mu2)` (mu2 = mu^2) leaves the body as
      //      `ref self mu2` and the profile evaluator never reaches
      //      `ref self mu` to rewrite it as `ref %local mu`, producing
      //      an empty line plot.
      if (!visiting.has(node.name)) {
        const target = bindings && bindings.get(node.name);
        const drv = derivations && Object.prototype.hasOwnProperty.call(derivations, node.name)
          ? derivations[node.name] : null;
        const isEvaluate =
          (drv && drv.kind === 'evaluate')
          || (!drv && target && target.type === 'call' && target.ir);
        if (isEvaluate && target && target.ir) {
          visiting.add(node.name);
          const expanded = walk(target.ir);
          visiting.delete(node.name);
          return expanded;
        }
      }
      // Constant / stochastic / opaque ref — leave for fixedEnv.
      return node;
    }
    // Recurse into structural children: args, fields, kwargs, body.
    const out = { ...node };
    if (Array.isArray(node.args))   out.args   = node.args.map(walk);
    if (Array.isArray(node.fields)) out.fields = node.fields.map(f => ({ ...f, value: walk(f.value) }));
    if (node.kwargs && typeof node.kwargs === 'object') {
      out.kwargs = {};
      for (const k in node.kwargs) out.kwargs[k] = walk(node.kwargs[k]);
    }
    if (node.body) out.body = walk(node.body);
    return out;
  }
}

module.exports = {
  resolveAxisBaseSet,
  findMatchingPresets,
  findMatchingDomains,
  fourSigmaQuantileRange,
  inlineForProfile,
};

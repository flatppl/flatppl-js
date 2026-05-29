'use strict';

// profile-plan.js — profile-plot UI support extracted from the
// orchestrator. Maps a callable's input signature to sweepable axis
// ranges: base-set resolution, preset/domain matching, 4-sigma
// quantile clipping, and per-profile IR inlining. A leaf w.r.t. the
// split: depends only on ir-shared (resolveConstant, parseSetIR,
// lowerSafe, NAMED_SET_NAMES) and histogram (quantileSorted), and has
// ZERO internal callers in orchestrator (only reached via the public
// API) — the facade re-bind exists purely for module.exports parity.

import type { IRNode } from './engine-types';

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
// Spec §08 distribution supports. Each entry names the distribution's
// natural value-set; used by `resolveAxisBaseSet` to surface tighter
// auto-domain ranges than the loose `empirical` fallback. Adding a
// new distribution: pair its spec-canonical name with the right set
// descriptor (the same shape `parseSetIR` returns for explicit
// `elementof(set)` bindings).
const DISTRIBUTION_SUPPORT: Record<string, any> = {
  // Continuous over all reals.
  Normal:            { kind: 'reals' },
  Cauchy:            { kind: 'reals' },
  StudentT:          { kind: 'reals' },
  Logistic:          { kind: 'reals' },
  Laplace:           { kind: 'reals' },
  GeneralizedNormal: { kind: 'reals' },
  // Continuous over the positive half-line.
  Exponential:  { kind: 'posreals' },
  Gamma:        { kind: 'posreals' },
  LogNormal:    { kind: 'posreals' },
  Weibull:      { kind: 'posreals' },
  InverseGamma: { kind: 'posreals' },
  ChiSquared:   { kind: 'posreals' },
  // Continuous over a bounded interval.
  Beta:     { kind: 'unitinterval' },
  // VonMises — angles on (-π, π]; surfaced as a bounded interval so
  // the viewer's auto-range is meaningful even without samples.
  VonMises: { kind: 'interval', lo: -Math.PI, hi: Math.PI },
  // Discrete over the non-negative integers.
  Poisson:           { kind: 'nonnegintegers' },
  Binomial:          { kind: 'nonnegintegers' },
  NegativeBinomial:  { kind: 'nonnegintegers' },
  NegativeBinomial2: { kind: 'nonnegintegers' },
  Categorical0:      { kind: 'nonnegintegers' },
  // Discrete over the positive integers (Categorical: 1..K; Geometric:
  // 1..∞).
  Categorical: { kind: 'posintegers' },
  Geometric:   { kind: 'posintegers' },
  // Boolean.
  Bernoulli: { kind: 'booleans' },
};

/**
 * For a `~`-bound variate, look up the distribution's natural value
 * set instead of falling through to `empirical`. Walks the binding's
 * IR to recover the call to a known distribution constructor and
 * consults `DISTRIBUTION_SUPPORT`. Two surface shapes covered:
 *
 *   theta = draw(Exponential(rate = 1))     // direct
 *   theta ~ Exponential(rate = 1)           // ~ desugars to draw
 *
 * `Uniform(support = S)` is a special case: its support is given by a
 * kwarg, not implicit in the constructor name; the same `parseSetIR`
 * the explicit `elementof(set)` path uses applies. Other constructors
 * with explicit support (Lebesgue / Counting) are also handled.
 *
 * Returns null when the binding isn't a draw or the distribution isn't
 * in `DISTRIBUTION_SUPPORT`; the caller falls back to `empirical`.
 */
function resolveDrawSupport(target: any, bindings: any): any {
  if (!target || target.type !== 'draw') return null;
  const ir = target.ir;
  if (!ir || ir.kind !== 'call' || ir.op !== 'draw') return null;
  const inner = ir.args && ir.args[0];
  if (!inner) return null;
  // `~`-form usually lifts the dist call to an anon binding: the
  // draw's arg is `(ref self __anonN)` pointing at the actual
  // `Exponential(rate = 1)` call. Chase the ref.
  let distIR = inner;
  if (distIR.kind === 'ref' && distIR.ns === 'self' && bindings
      && bindings.has(distIR.name)) {
    const anon = bindings.get(distIR.name);
    if (anon && anon.ir && anon.ir.kind === 'call') distIR = anon.ir;
  }
  if (!distIR || distIR.kind !== 'call') return null;
  const op = distIR.op;
  // `iid(Dist, n)` produces an array-shaped variate whose PER-SLOT
  // support is Dist's support. Surface that — the caller (`compute-
  // AutoDomain`) wraps it in `cartpow(base, n)` when the input type
  // is array-shaped. Without this peek, a `~ iid(Normal, 4)` source
  // degrades to `empirical` and the persisted domain emits a
  // `cartpow(reals, 4)` only because computeAutoDomain falls back
  // when base is null; an Exponential / Beta-cored iid would lose
  // its support refinement.
  if (op === 'iid' && Array.isArray(distIR.args) && distIR.args.length >= 1) {
    const inner = distIR.args[0];
    // Synthesise a draw-shaped target for the inner distribution so we
    // can recurse uniformly through the same support logic.
    const innerTarget = { type: 'draw', ir: { kind: 'call', op: 'draw', args: [inner] } };
    return resolveDrawSupport(innerTarget, bindings);
  }
  // Explicit-support constructors: prefer the user's support kwarg
  // over the distribution-name default.
  if (op === 'Uniform' || op === 'Lebesgue' || op === 'Counting') {
    const supportIR = distIR.kwargs && distIR.kwargs.support;
    if (supportIR) {
      const d = parseSetIR(supportIR);
      if (d) return d;
    }
    if (op === 'Lebesgue') return { kind: 'reals' };
    if (op === 'Counting') return { kind: 'integers' };
    // Uniform without support kwarg — fall through to empirical.
    return null;
  }
  if (Object.prototype.hasOwnProperty.call(DISTRIBUTION_SUPPORT, op)) {
    return DISTRIBUTION_SUPPORT[op];
  }
  return null;
}

function resolveAxisBaseSet(source: any, bindings: any) {
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
    // Drawn variates: look up the distribution's natural support set
    // instead of degrading to `empirical` (spec §08). Tighter ranges
    // give the viewer's auto-domain a meaningful starting point even
    // before sample materialisation runs.
    const drawn = resolveDrawSupport(target, bindings);
    if (drawn) return drawn;
    // Anything else (derived deterministic bindings, lawof-defined
    // measures, untyped sources): there's no static set, but the
    // binding has empirical samples we can quantile-clip into a range
    // at materialise time.
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
 * Match rule (strict — top-level scalars only):
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
function findMatchingPresets(signature: any, bindings: any) {
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
    const values: Record<string, any> = {};
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
      // Array-typed value: `vector(lit, lit, ...)` IR (lowered from
      // surface `[v1, v2, ...]` literals). The lift pass typically
      // hoists array literals into an anon binding, so the field's
      // immediate value here is a (ref self __anonN). Chase that
      // ref once to see whether the target IR is a vector call.
      // Each vector element must itself be constant-resolvable.
      // Surfacing array values lets a saved `record(theta = [...])`
      // show up in the preset dropdown for an array-typed input.
      let vectorIR: any = null;
      if (inner && inner.kind === 'call' && inner.op === 'vector'
          && Array.isArray(inner.args)) {
        vectorIR = inner;
      } else if (inner && inner.kind === 'ref' && inner.ns === 'self') {
        const refTarget2 = bindings.get(inner.name);
        if (refTarget2 && refTarget2.ir && refTarget2.ir.kind === 'call'
            && refTarget2.ir.op === 'vector'
            && Array.isArray(refTarget2.ir.args)) {
          vectorIR = refTarget2.ir;
        }
      }
      if (vectorIR) {
        const arr: any[] = [];
        let arrayMatch = true;
        for (const el of vectorIR.args) {
          const elV = resolveConstant(el, bindings, new Set());
          if (elV == null) { arrayMatch = false; break; }
          arr.push(elV);
        }
        if (arrayMatch) {
          values[f.name] = arr;
          continue;
        }
        allMatch = false; break;
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
function findMatchingDomains(signature: any, bindings: any) {
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
    const ranges: Record<string, any> = {};
    const setNames: Record<string, any> = {};
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
      // (c) cartpow(<base>, n) for array-typed kwargs. Per the
      // linked-cartpow domain semantics, the interior `<base>`
      // applies uniformly to every slot; expose it as a single
      // {lo, hi} range entry so the viewer renders identical
      // bounds across the per-slot axes. `<base>` may itself be
      // an `interval(lo, hi)` (literal-resolvable) or a bare
      // named set (no range entry, just a setName tag).
      if (inner && inner.kind === 'call' && inner.op === 'cartpow'
          && Array.isArray(inner.args) && inner.args.length === 2) {
        let basePart: any = inner.args[0];
        if (basePart && basePart.kind === 'ref' && basePart.ns === 'self') {
          const refTarget = bindings.get(basePart.name);
          if (refTarget && refTarget.ir) basePart = refTarget.ir;
        }
        if (basePart && basePart.kind === 'call' && basePart.op === 'interval'
            && Array.isArray(basePart.args) && basePart.args.length === 2) {
          const lo = resolveConstant(basePart.args[0], bindings, new Set());
          const hi = resolveConstant(basePart.args[1], bindings, new Set());
          if (lo == null || hi == null || !(lo < hi)) { allMatch = false; break; }
          ranges[f.name] = { lo, hi };
          continue;
        }
        const baseSetName = basePart && (basePart.kind === 'const' || basePart.kind === 'ref')
          ? basePart.name : null;
        if (baseSetName && NAMED_SET_NAMES.has(baseSetName)) {
          setNames[f.name] = baseSetName;
          continue;
        }
        allMatch = false; break;
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
/**
 * Static slot count of an array-typed callable input, or null if the
 * type isn't a statically-shaped array. Used by the auto-input /
 * auto-domain derivation paths to decide whether to populate a kwarg
 * entry with a single scalar (samples[0] / interval) or a J-element
 * array (atom-0 slice / cartpow wrap).
 */
function arrayInputLength(type: any): number | null {
  if (!type || type.kind !== 'array') return null;
  if (!Array.isArray(type.shape)) return null;
  let total = 1;
  for (const d of type.shape) {
    if (d === '%dynamic' || typeof d !== 'number') return null;
    total *= d;
  }
  return total;
}

/**
 * Per-leaf-type default for an unresolved auto-input slot. Mirrors
 * the viewer's `defaultValueForLeafType`; centralised here so the
 * engine alone owns the "what value should the viewer show before
 * the user picks one?" decision.
 *
 * Conservative defaults: integer / boolean / real all collapse to 0
 * / false. The "swept-axis range" decision is separate (lives in
 * `resolveAxisBaseSet` + `fourSigmaQuantileRange`).
 */
function defaultValueForLeafType(leafType: any): number | boolean {
  if (!leafType) return 0;
  if (leafType.kind === 'scalar') {
    if (leafType.prim === 'boolean') return false;
    return 0;
  }
  return 0;
}

/**
 * Compute per-kwarg "auto" input values for a callable signature.
 *
 * For each kwarg, the returned record entry is:
 *   - a J-element array, if the input type is `array(rank=1,
 *     shape=[J])` — atom-0 slice from `getAtomZero(sourceName)` when
 *     the input's source is a binding, else J copies of the
 *     leaf-type default.
 *   - a scalar (number / boolean), otherwise — `getAtomZero
 *     (sourceName)[0]`-equivalent when the source is a binding (the
 *     callback may return a scalar directly for scalar measures), else
 *     the leaf-type default.
 *
 * `getAtomZero(name)` is the consumer-supplied callback for fetching
 * atom-0 of a binding's empirical measure. It returns:
 *   - a single number for scalar measures,
 *   - a number[] / TypedArray of length ≥ arrayLen for array measures
 *     (only the first arrayLen entries are consumed),
 *   - null when the measure isn't materialised / isn't applicable.
 * The engine never reaches into the viewer's measure cache directly;
 * the caller routes through this hook.
 *
 * Used by the viewer's preset dropdown ("auto" row), persist
 * (`persistAutoAsNewBinding`), and any future render path that
 * needs deterministic per-kwarg defaults. Engine-side ownership
 * makes the resolution rules testable in isolation, mirrors
 * `resolveAxisBaseSet` (per-axis) at the per-kwarg level, and
 * single-sources the array-vs-scalar dispatch.
 */
function computeAutoInputs(
  signature: any,
  bindings: any,
  fixedValues: any,
  getAtomZero: ((name: string) => any) | null,
): Record<string, any> {
  void bindings; void fixedValues;
  const out: Record<string, any> = {};
  if (!signature || !Array.isArray(signature.inputs)) return out;
  const seen = new Set<string>();
  for (const inp of signature.inputs) {
    const kw = inp.kwargName;
    if (!kw || seen.has(kw)) continue;
    seen.add(kw);
    const arrayLen = arrayInputLength(inp.type);
    const leafType = arrayLen != null && inp.type && inp.type.elem
      ? inp.type.elem
      : inp.type;
    const leafDef = defaultValueForLeafType(leafType);
    let def: any = arrayLen != null
      ? new Array(arrayLen).fill(leafDef)
      : leafDef;
    if (inp.source && inp.source.kind === 'binding'
        && typeof getAtomZero === 'function') {
      const v = getAtomZero(inp.source.name);
      const isArrayLike = Array.isArray(v) || (v != null
        && typeof v === 'object'
        && typeof v.length === 'number'
        && (typeof v.BYTES_PER_ELEMENT === 'number'
            || v[0] !== undefined));
      if (v != null) {
        if (arrayLen != null) {
          if (isArrayLike && v.length >= arrayLen) {
            def = Array.from(v).slice(0, arrayLen);
          }
          // If the callback returned a scalar for an array input,
          // ignore it — atom-0 of a J-vector measure should itself
          // be J entries; a scalar result indicates the source
          // doesn't match the signature shape.
        } else if (typeof v === 'number' || typeof v === 'boolean') {
          def = v;
        } else if (isArrayLike && v.length > 0
                   && (typeof v[0] === 'number' || typeof v[0] === 'boolean')) {
          // Array-like passed for scalar input — convention is "take
          // the first atom's scalar". Lets callers hand the full
          // sample buffer through without branching on arity per-call.
          def = v[0];
        }
      }
    }
    out[kw] = def;
  }
  return out;
}

/**
 * Compute per-kwarg "auto" domain descriptors for a callable
 * signature. Each entry is a `SetDescriptor` — the same shape
 * `resolveAxisBaseSet` returns for scalar axes, with the additional
 * `{kind: 'cartpow', base: SetDescriptor, n: number}` variant for
 * array-typed inputs.
 *
 * For array kwargs the resolution rule: walk the per-slot axes (all
 * slots share a source), resolve `resolveAxisBaseSet` once, wrap in
 * `cartpow(base, J)` where `J = arrayInputLength(input.type)`.
 *
 * Used by `persistAutoDomainAsNewBinding` (the viewer's "save as
 * new domain binding" path); the viewer's `defaultSetSourceForKwarg`
 * becomes a thin formatter over the engine's structured descriptor.
 */
function computeAutoDomain(
  signature: any,
  bindings: any,
): Record<string, any> {
  const out: Record<string, any> = {};
  if (!signature || !Array.isArray(signature.inputs)) return out;
  for (const inp of signature.inputs) {
    const kw = inp.kwargName;
    if (!kw || Object.prototype.hasOwnProperty.call(out, kw)) continue;
    const arrayLen = arrayInputLength(inp.type);
    let base: any = null;
    try {
      base = resolveAxisBaseSet(inp.source, bindings);
    } catch (_) { base = null; }
    if (!base) base = { kind: 'reals' };
    if (arrayLen != null) {
      out[kw] = { kind: 'cartpow', base, n: arrayLen };
    } else {
      out[kw] = base;
    }
  }
  return out;
}

/**
 * Effective per-kwarg input values for a callable signature, after
 * merging the precedence chain:
 *
 *   effective[kwarg]  =  override[kwarg]  ??  base[kwarg]  ??  auto[kwarg]
 *
 * `auto` is computed via `computeAutoInputs(signature, bindings,
 * fixedValues, getAtomZero)`; `base` and `override` are passed in
 * verbatim (the consumer supplies them — typically the active
 * preset's `values` and the user's override entry). Returns one
 * entry per kwarg in the signature, with scalar OR array shape
 * matching the input's static type (no per-slot kwarg keys).
 *
 * Centralises the merge logic that was previously sprinkled across
 * commitSliceX, computeAutoValues consumers, and the persist path.
 * The smell flagged in 4575b35: every site re-derived "what's the
 * current value of kwarg X for this plan?" by hand; pull it into
 * one engine call.
 */
function effectiveInputValues(
  signature: any,
  bindings: any,
  fixedValues: any,
  baseValues: Record<string, any> | null | undefined,
  overrideValues: Record<string, any> | null | undefined,
  getAtomZero: ((name: string) => any) | null,
): Record<string, any> {
  const auto = computeAutoInputs(signature, bindings, fixedValues, getAtomZero);
  const base = baseValues || {};
  const over = overrideValues || {};
  const out: Record<string, any> = {};
  if (!signature || !Array.isArray(signature.inputs)) return out;
  const seen = new Set<string>();
  for (const inp of signature.inputs) {
    const kw = inp.kwargName;
    if (!kw || seen.has(kw)) continue;
    seen.add(kw);
    if (Object.prototype.hasOwnProperty.call(over, kw)) {
      out[kw] = over[kw];
    } else if (Object.prototype.hasOwnProperty.call(base, kw)) {
      out[kw] = base[kw];
    } else if (Object.prototype.hasOwnProperty.call(auto, kw)) {
      out[kw] = auto[kw];
    }
  }
  return out;
}

/**
 * Effective per-kwarg SetDescriptor for a callable signature's
 * input domain, after merging:
 *
 *   effective[kwarg]  =
 *     overrideRange[kwarg] applied to auto-domain shape ??
 *     baseRange[kwarg]     applied to auto-domain shape ??
 *     baseSetName[kwarg]   applied to auto-domain shape ??
 *     auto[kwarg]
 *
 * "Applied to auto-domain shape" preserves the cartpow wrap for
 * array-typed kwargs: an override interval becomes the base of
 * `cartpow(<base>, n)` rather than collapsing to bare `interval`.
 * Without this, the viewer's persist path lost the array shape
 * whenever a user edited a per-slot range.
 *
 * `overrideRanges` and `baseRanges` are maps `kwarg → {lo, hi}`;
 * `baseSetNames` is a map `kwarg → 'reals' | 'posreals' | ...`
 * (the named-set form `findMatchingDomains` surfaces). The caller
 * passes these in (typically from `activeDomainRangesFor` /
 * matched-domain entries).
 */
function effectiveInputDomain(
  signature: any,
  bindings: any,
  baseRanges: Record<string, { lo: number; hi: number }> | null | undefined,
  baseSetNames: Record<string, string> | null | undefined,
  overrideRanges: Record<string, { lo: number; hi: number }> | null | undefined,
): Record<string, any> {
  const auto = computeAutoDomain(signature, bindings);
  const br = baseRanges || {};
  const bn = baseSetNames || {};
  const or = overrideRanges || {};
  const out: Record<string, any> = {};
  if (!signature || !Array.isArray(signature.inputs)) return out;
  const seen = new Set<string>();
  for (const inp of signature.inputs) {
    const kw = inp.kwargName;
    if (!kw || seen.has(kw)) continue;
    seen.add(kw);
    const autoDesc = auto[kw];
    // Resolve the active SetDescriptor with precedence override > base
    // range > base named-set > auto. Each "range" / "named-set" entry
    // is wrapped to preserve the auto descriptor's cartpow shape when
    // the input is array-typed.
    let raw: any = null;
    if (Object.prototype.hasOwnProperty.call(or, kw)) {
      const r = or[kw];
      if (r && Number.isFinite(r.lo) && Number.isFinite(r.hi)) {
        raw = { kind: 'interval', lo: r.lo, hi: r.hi };
      }
    } else if (Object.prototype.hasOwnProperty.call(br, kw)) {
      const r = br[kw];
      if (r && Number.isFinite(r.lo) && Number.isFinite(r.hi)) {
        raw = { kind: 'interval', lo: r.lo, hi: r.hi };
      }
    } else if (Object.prototype.hasOwnProperty.call(bn, kw)) {
      // Named set: build a bare descriptor with the set name as kind.
      // (The format mirrors the descriptors `resolveAxisBaseSet` /
      // `computeAutoDomain` return for built-in sets.)
      const n = bn[kw];
      if (typeof n === 'string') raw = { kind: n };
    }
    if (raw == null) {
      out[kw] = autoDesc || { kind: 'reals' };
      continue;
    }
    if (autoDesc && autoDesc.kind === 'cartpow') {
      // Linked-cartpow: preserve the wrap, replace the base.
      out[kw] = { kind: 'cartpow', base: raw, n: autoDesc.n };
    } else {
      out[kw] = raw;
    }
  }
  return out;
}

function fourSigmaQuantileRange(samples: ArrayLike<number>) {
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
function inlineForProfile(ir: IRNode | null | undefined, paramNames: any, bindings: any, derivations: any) {
  if (!ir) return ir;
  const paramSet = new Set(paramNames || []);
  const visiting = new Set();
  return walk(ir);

  function walk(node: any): any {
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
          const expanded: any = walk(target.ir);
          visiting.delete(node.name);
          return expanded;
        }
      }
      // Constant / stochastic / opaque ref — leave for fixedEnv.
      return node;
    }
    // Recurse into structural children: args, fields, kwargs, body,
    // branches (the `select` IR's per-branch measures — without this
    // a `self.<kernel_input>` ref inside a `weighted(w, m)` branch
    // stays unrewritten when the swept input appears inside a
    // superpose / mixture / ifelse, surfacing downstream as
    // "select branch weight must reduce to a number; got self.psi"
    // at materialiseSelectIR's fold step).
    const out = { ...node };
    if (Array.isArray(node.args))   out.args   = node.args.map(walk);
    if (Array.isArray(node.fields)) out.fields = node.fields.map((f: any) => ({ ...f, value: walk(f.value) }));
    if (node.kwargs && typeof node.kwargs === 'object') {
      out.kwargs = {};
      for (const k in node.kwargs) out.kwargs[k] = walk(node.kwargs[k]);
    }
    if (node.body) out.body = walk(node.body);
    if (Array.isArray(node.branches)) {
      out.branches = node.branches.map(walk);
    }
    if (node.selector)   out.selector   = walk(node.selector);
    if (node.logweights) out.logweights = walk(node.logweights);
    return out;
  }
}

module.exports = {
  resolveAxisBaseSet,
  findMatchingPresets,
  findMatchingDomains,
  fourSigmaQuantileRange,
  inlineForProfile,
  arrayInputLength,
  defaultValueForLeafType,
  computeAutoInputs,
  computeAutoDomain,
  effectiveInputValues,
  effectiveInputDomain,
};

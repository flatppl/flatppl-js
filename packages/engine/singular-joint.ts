'use strict';

// ════════════════════════════════════════════════════════════════════════
// Singular joints — the STATIC gate (spec §06 "Singular joints")
// ════════════════════════════════════════════════════════════════════════
//
// §06 "Singular joints", quoted verbatim from flatppl-design 9e35262:
//
//   "When one component's variate is determined by the others given the shared
//   ancestors (the same draw referenced twice, a deterministic transform of
//   another component), the joint law has no density w.r.t. the product
//   reference measure. Sampling is well-defined; a density query is a static
//   error where statically detectable, and is otherwise refused by the engine."
//
// This module is the "static error where statically detectable" half.
//
// ── THE CRITERION IS CONDITIONAL DETERMINISM, NOT A SHARED ANCESTOR ─────
//
// §06's operative words are "determined by the others given the shared
// ancestors". Sharing an ancestor is NOT the criterion, and an earlier revision
// of this pass got that wrong: it fired on any non-empty overlap between two
// components' noise-source sets, which made
//
//   y ~ Normal(0, 1); n1 ~ Normal(0, 1); n2 ~ Normal(0, 1)
//   u = y + n1; v = y + n2; joint(a = lawof(u), b = lawof(v))
//
// a compile error. That law is Gaussian with covariance [[2,1],[1,2]],
// determinant 3 — full rank, absolutely continuous, and `u` is emphatically NOT
// determined by `v` given `y`. Overlap is necessary for singularity, never
// sufficient.
//
// What IS sufficient, at pair granularity: both components are deterministic
// functions of ONE SHARED SCALAR VALUE — a single common "generator". Then the
// pair's support is the image of a 1-dimensional variable in the 2-dimensional
// product space, a curve, which is Lebesgue-null. Both of §06's named classes are
// this case (the same draw twice is `f = g = id`; a deterministic transform of
// another component is `g = h ∘ f`).
//
// `_commonGenerator` looks for that value: a binding `X` of scalar type such that
// each component's noise roots, computed with `X` treated as terminal, are exactly
// {X} — i.e. neither component carries any randomness that does not come through
// `X`. Two conditions guard it, each with a verified false positive behind it:
//
//   the generator must exist  — `u = y + n1; v = y + n2` has no common generator
//                       (each component carries its own noise), and the law is
//                       Gaussian with covariance [[2,1],[1,2]], determinant 3.
//                       Neither does `joint(a = lawof(y), b = lawof(y + n))`,
//                       where `b` carries `n` on top of `a`. An any-overlap test
//                       flagged both.
//   the generator must be SCALAR — a multi-coordinate draw is excluded because
//                       two components can read different coordinates of it
//                       (`v ~ MvNormal(...); a = v[1]; b = v[2]` is two
//                       independent standard normals, and root identity is
//                       per-draw-binding, so it cannot see the difference).
//
// ── THE REFERENCE MEASURE IS FIXED BY THE COMPONENTS, NOT THE DRAW ──────
//
// Nullity is meaningless without naming the measure it is null against, and §06
// "Reference measure for product measures" fixes it per COMPONENT (verbatim,
// flatppl-design 9e35262):
//
//   "When `joint(M1, M2, ...)` (or `iid(M, size)`, `jointchain(M, K1, ...)` etc.)
//   combines components with individual reference measures ρ1, ρ2, … (each either
//   `Lebesgue` or `Counting` on the corresponding component support), the
//   reference measure of the product is the product ρ1 ⊗ ρ2 ⊗ ⋯ on the joint
//   variate space."
//
// So the curve-is-null argument needs BOTH components to have continuous
// (Lebesgue) support, and that is a property of the COMPONENTS' types, not of the
// shared draw. Keying it on the draw is wrong in both directions, and an earlier
// revision did exactly that:
//
//   `lawof(record(a = y, b = floor(y)))` for `y ~ Normal` — the draw is
//   continuous, but `floor(y)` is scalar INTEGER, so the reference is
//   Lebesgue ⊗ Counting and the law HAS a density w.r.t. it:
//   f(t, n) = φ(t)·1[⌊t⌋ = n], since μ(A × {n}) = ∫_A 1[⌊t⌋ = n] φ(t) dt.
//   Equivalently the support's slices {t : ⌊t⌋ = n} = [n, n+1) are not
//   Lebesgue-null. NOT singular; the draw-keyed test made it a static error.
//
//   `c ~ Poisson(2); a = c * 1.0; b = c * 2.0` — the draw is DISCRETE, but both
//   components are scalar real, so the reference is Lebesgue ⊗ Lebesgue and the
//   support {(n, 2n)} is countable, hence 2-D Lebesgue-null. Genuinely singular;
//   the draw-keyed test stayed silent on it.
//
// Hence the gate below: the generator must be scalar of ANY prim (a discrete draw
// generates a perfectly good singular pair, as the Poisson case shows), and BOTH
// COMPONENTS must be `scalar real`. An all-discrete joint can never be singular —
// on a countable support with counting reference the only null set is ∅, so every
// measure there is absolutely continuous — which is why `joint(a = lawof(k), b =
// lawof(k))` for `k ~ Bernoulli(p)` is correctly silent: it has an ordinary pmf.
//
// ── THE GENERAL CRITERION, STATED HONESTLY ──────────────────────────────
//
// The joint's Jacobian dropping rank is the real criterion. Hall's condition on
// the component→root bipartite graph (some subset S with |⋃_{i∈S} R_i| < |S|) is
// the GENERIC rank criterion: it assumes the component functions are
// algebraically independent, so it is NECESSARY for full rank but NOT SUFFICIENT.
// Rank can drop inside a Hall-satisfying graph when the components are
// functionally dependent — `u = y + n; joint(a = lawof(u), b = lawof(2*u))` has
// R₁ = R₂ = {y, n}, so |⋃ R_i| = 2 = |S|, yet the Jacobian has rank 1. A full
// Hall check would therefore NOT complete this pass.
//
// This pass implements the |S| = 2 case via the common-generator test, which
// catches both the equal-singleton shapes and that dependent-function case. A
// deficiency needing a subset of size ≥ 3 — `lawof(record(inner = record(a = y,
// b = t), c = y))`, whose support is {(y,t,y)} ⊂ R³ — is a MISS. See the
// fail-silent note.
//
// ── WHAT ELSE MAKES A JOINT SINGULAR: INHERITANCE ───────────────────────
//
// Singularity propagates outward, and two constructs carry it:
//   - a COMPONENT that is itself a singular joint. `joint(inner = S, c =
//     lawof(t))` with `S` singular concentrates on {(u, u, t)} ⊂ R³.
//   - `iid(M, size)` over a singular `M`. §06 `iid` is the product measure
//     M^⊗N and "never shares nodes between copies", so the copies are
//     independent and the product is null exactly when M is.
// Both are checked recursively. They matter more than they look: before they
// were covered, all three shapes returned a finite WRONG number rather than
// refusing (see the fail-silent note).
//
// ── WHY THE DENSITY QUERY, NOT THE JOINT ────────────────────────────────
//
// §06 makes sampling well-defined — the singular joint is a legal binding, and
// `joint(a = lawof(y), b = lawof(y))` samples on the diagonal. Only a DENSITY
// query over it is the error. So this pass starts from `logdensityof` call sites
// and works inward; a model that merely builds or samples the joint is silent.
//
// ── RELATION TO clm._refuseIfSingular ───────────────────────────────────
//
// `clm._refuseIfSingular` is the "otherwise refused by the engine" half. It runs
// at density time on the post-lift DERIVATION table; this pass runs on the
// LOWERED IR, because a pass that had to wait for `buildDerivations` could not be
// static. The two are NOT the same predicate and deliberately so:
//
//   clm still uses the coarse any-overlap test. It therefore REFUSES the
//   full-rank shapes above, with a reason that is factually wrong for them.
//   Narrowing it is not this pass's business and is not obviously safe: a refusal
//   returns no number, so it is sound-but-incapable, whereas narrowing it would
//   hand those shapes to a scoring path nobody has verified. Trading a safe
//   refusal for an unverified number is the wrong direction. Tracked in
//   flatppl-dev/TODO-flatppl-js.md.
//
// The two therefore diverge in BOTH directions, and neither contains the other:
//
//   pair rule (§06's named classes) — clm also refuses, so the static error and
//   the refusal agree, and the query genuinely reaches no number.
//
//   inherited cases (iid, nested) — clm does NOT refuse. At 61c29f0 it scores a
//   finite wrong number, and it still does: this pass makes the error VISIBLE
//   but cannot prevent the number, because a diagnostic is not a gate. A user
//   who ignores the diagnostic still gets -2.8268155996140187 out of a law that
//   has no density. That is an improvement on silence, not a fix.
//
// Anything narrower than clm is safe to flag. The inherited cases are flagged
// anyway, because a visible wrong answer beats a silent one and §06 wants the
// static error regardless of what the runtime does with it.
//
// ── FAIL-SILENT, AND WHAT THAT DOES AND DOES NOT BUY ────────────────────
//
// Every unrecognised shape yields no diagnostic rather than a guess, because a
// false positive refuses a model the engine answers exactly (§06's correlated
// case, which the linear-Gaussian recogniser closes in closed form) and the user
// has no way around a compile error.
//
// Be precise about what a miss costs, though — the earlier revision of this
// header claimed "a miss still refuses at density time", and that is FALSE. The
// runtime backstop does not cover every shape either: at 61c29f0 an `iid` over a
// singular joint, and a nested singular joint or record, each returned a finite
// wrong number. This pass now catches those three statically, but the underlying
// density bug is unfixed and other unrecognised shapes may share it. A miss here
// may therefore cost a wrong number, not merely a late message. The
// silent-number holes are tracked in flatppl-dev/TODO-flatppl-js.md with their
// closed forms.

const { forEachIRChild } = require('./ir-walk.ts');

// Guard against a malformed cyclic IR. `buildDerivations` has a real cycle
// detector; this pass runs before it, so it bounds its own recursion instead.
const MAX_DEPTH = 64;

// The binding `ir` names, or null when `ir` is not a self-ref to one.
function _refTarget(ir: any, loweredModule: any): { name: string; binding: any } | null {
  if (!ir || ir.kind !== 'ref' || ir.ns === 'module') return null;
  const name = ir.name;
  if (typeof name !== 'string' || !loweredModule.bindings.has(name)) return null;
  return { name, binding: loweredModule.bindings.get(name) };
}

// Is this binding a `~`-draw (or `draw(m)`)? Its variate IS a noise source, and
// the walk terminates there — a draw's own ancestors are shared context, not
// shared noise, which is what keeps `a, b ~ Normal(z, 1)` separable.
function _isDrawBinding(binding: any): boolean {
  const rhs = binding && binding.rhs;
  return !!(rhs && rhs.kind === 'call' && rhs.op === 'draw');
}

function _isMeasureTyped(binding: any): boolean {
  const t = binding && binding.inferredType;
  return !!(t && t.kind === 'measure');
}

// The variate type, unwrapping a measure to its domain. Null when unavailable —
// every caller treats that as "cannot classify", i.e. stay silent.
function _variateType(t: any): any {
  if (!t) return null;
  return t.kind === 'measure' ? (t.domain || null) : t;
}

// A single CONTINUOUS real coordinate. §06 "Reference measure for product
// measures" makes this the COMPONENT-level test: only a Lebesgue-referenced
// component can sit on a Lebesgue-null curve.
function _isScalarReal(t: any): boolean {
  const v = _variateType(t);
  return !!(v && v.kind === 'scalar' && v.prim === 'real');
}

// A single coordinate of ANY prim. The generator may be discrete — `c ~ Poisson;
// a = c*1.0; b = c*2.0` is singular w.r.t. Lebesgue ⊗ Lebesgue because its support
// is countable — so this deliberately does not require `real`. What it does
// require is ONE coordinate: an array-valued generator can feed two components
// different coordinates, and then they are not functions of one value.
function _isScalarBinding(name: string, loweredModule: any): boolean {
  const b = loweredModule.bindings.get(name);
  if (!b) return false;
  const v = _variateType(b.inferredType);
  return !!(v && v.kind === 'scalar');
}

// The independent noise sources the VALUE expression `ir` is a deterministic
// function of. A ref to a draw binding is a root; a ref to any other value
// binding is followed into its RHS; a measure-typed ref contributes nothing
// (a measure is not a variate, so it carries no coordinate of its own).
function _noiseRoots(ir: any, loweredModule: any, out: Set<string>,
  visited: Set<string>, depth: number): void {
  if (!ir || typeof ir !== 'object' || depth > MAX_DEPTH) return;
  const target = _refTarget(ir, loweredModule);
  if (target) {
    if (visited.has(target.name)) return;
    visited.add(target.name);
    if (_isDrawBinding(target.binding)) { out.add(target.name); return; }
    if (_isMeasureTyped(target.binding)) return;
    _noiseRoots(target.binding.rhs, loweredModule, out, visited, depth + 1);
    return;
  }
  // A nested `lawof` / `draw` inside a value expression reifies or consumes a
  // measure rather than reading a variate — neither is this component's noise.
  if (ir.kind === 'call' && (ir.op === 'lawof' || ir.op === 'draw')) return;
  forEachIRChild(ir, (child: any) =>
    _noiseRoots(child, loweredModule, out, visited, depth + 1));
}

function _noiseRootsOf(ir: any, loweredModule: any): Set<string> {
  const out = new Set<string>();
  _noiseRoots(ir, loweredModule, out, new Set<string>(), 0);
  return out;
}

// The same walk, but `barrier` is treated as a terminal noise source instead of
// being descended through. So `_rootsWithBarrier(expr, X) === {X}` says exactly
// "expr is a deterministic function of X and of nothing else random" — which is
// the property the common-generator test needs.
function _rootsWithBarrier(ir: any, barrier: string, loweredModule: any): Set<string> {
  const out = new Set<string>();
  const walk = (node: any, visited: Set<string>, depth: number): void => {
    if (!node || typeof node !== 'object' || depth > MAX_DEPTH) return;
    const target = _refTarget(node, loweredModule);
    if (target) {
      if (target.name === barrier) { out.add(barrier); return; }
      if (visited.has(target.name)) return;
      visited.add(target.name);
      if (_isDrawBinding(target.binding)) { out.add(target.name); return; }
      if (_isMeasureTyped(target.binding)) return;
      walk(target.binding.rhs, visited, depth + 1);
      return;
    }
    if (node.kind === 'call' && (node.op === 'lawof' || node.op === 'draw')) return;
    forEachIRChild(node, (child: any) => walk(child, visited, depth + 1));
  };
  walk(ir, new Set<string>(), 0);
  return out;
}

// Every binding the value expression `ir` passes through, draws included. These
// are the candidate generators: a common generator, if one exists, is always a
// binding one of the components is built from.
function _visitedBindings(ir: any, loweredModule: any): string[] {
  const order: string[] = [];
  const seen = new Set<string>();
  const walk = (node: any, depth: number): void => {
    if (!node || typeof node !== 'object' || depth > MAX_DEPTH) return;
    const target = _refTarget(node, loweredModule);
    if (target) {
      if (seen.has(target.name)) return;
      seen.add(target.name);
      order.push(target.name);
      if (_isDrawBinding(target.binding) || _isMeasureTyped(target.binding)) return;
      walk(target.binding.rhs, depth + 1);
      return;
    }
    if (node.kind === 'call' && (node.op === 'lawof' || node.op === 'draw')) return;
    forEachIRChild(node, (child: any) => walk(child, depth + 1));
  };
  walk(ir, 0);
  return order;
}

// A scalar binding both expressions are deterministic functions of, and of nothing
// else random. Prefers a DRAW when several qualify, purely so the diagnostic can
// name the draw — the more recognisable object to a reader.
function _commonGenerator(exprA: any, exprB: any, loweredModule: any):
{ name: string; isDraw: boolean } | null {
  const only = (roots: Set<string>, x: string) => roots.size === 1 && roots.has(x);
  let fallback: { name: string; isDraw: boolean } | null = null;
  for (const x of _visitedBindings(exprA, loweredModule)) {
    if (!_isScalarBinding(x, loweredModule)) continue;
    if (!only(_rootsWithBarrier(exprA, x, loweredModule), x)) continue;
    if (!only(_rootsWithBarrier(exprB, x, loweredModule), x)) continue;
    const isDraw = _isDrawBinding(loweredModule.bindings.get(x));
    if (isDraw) return { name: x, isDraw: true };
    if (!fallback) fallback = { name: x, isDraw: false };
  }
  return fallback;
}

// Follow a chain of bare-ref bindings to the expression that DEFINES the
// measure. `joint(a = Ly, b = Ly)` with `Ly = lawof(y)` must see the `lawof`,
// and alias-resolution has already collapsed the pure-ref hops, so this handles
// what remains: a named measure binding reached by one ref.
function _resolveMeasureExpr(ir: any, loweredModule: any, depth: number): any {
  if (!ir || typeof ir !== 'object' || depth > MAX_DEPTH) return ir;
  const target = _refTarget(ir, loweredModule);
  if (!target || !target.binding.rhs) return ir;
  return _resolveMeasureExpr(target.binding.rhs, loweredModule, depth + 1);
}

// ── the singularity judgement ────────────────────────────────────────────
//
// A `Reason` records WHY, so the diagnostic can name the offending pair and the
// path to it. `via` / `iid` wrap an inner reason for the inherited cases.
type Reason =
  | { kind: 'pair'; a: string; b: string; generator: string; generatorIsDraw: boolean }
  | { kind: 'via'; label: string; inner: Reason }
  | { kind: 'iid'; inner: Reason };

// A component, as the two rules need it.
//   roots    — its noise sources, for the cheap equal-sets pre-filter
//   value    — the VALUE expression the generator test walks (for a `lawof(V)`
//              component that is `V`; for a record field, the field itself)
//   type     — its variate type, per §06's per-component reference measure
//   measure  — its measure expression, for the inheritance rule
type Component = {
  label: string;
  roots: Set<string>;
  value: any;
  type: any;
  measure: any;
};

// Component variate types keyed by label, read off the enclosing measure's own
// inferred domain — §06's "corresponding component support". This is the reliable
// source: a bare-ref field carries no `meta.type` of its own.
function _componentTypeAt(container: any, label: string, index: number): any {
  const dom = _variateType(container && container.meta && container.meta.type);
  if (!dom) return null;
  if (dom.kind === 'record' && dom.fields) return dom.fields[label] || null;
  // A positional joint cats its components into one array; every component
  // carries the element type.
  if (dom.kind === 'array') return dom.elem || null;
  if (dom.kind === 'table' && dom.columns) return dom.columns[label] || null;
  return null;
}

// The components of a record-valued joint law as { label, roots, measure }:
// `roots` feeds the pair rule, `measure` (when present) is the component's own
// measure expression, which the inheritance rule recurses into. Null when
// `measureIR` is not a shape this pass classifies — the fail-silent exit.
function _componentsOf(measureIR: any, loweredModule: any): Component[] | null {
  const m = _resolveMeasureExpr(measureIR, loweredModule, 0);
  if (!m || m.kind !== 'call') return null;

  if (m.op === 'joint') {
    const parts: Array<{ label: string; value: any }> = Array.isArray(m.fields)
      ? m.fields.map((f: any) => ({ label: f.name, value: f.value }))
      : Array.isArray(m.args)
        ? m.args.map((a: any, i: number) => ({ label: '#' + (i + 1), value: a }))
        : [];
    if (parts.length < 2) return null;
    return parts.map((p, i) => {
      const resolved = _resolveMeasureExpr(p.value, loweredModule, 0);
      const isLawof = !!(resolved && resolved.kind === 'call' && resolved.op === 'lawof');
      // Only a REIFIED component reads a variate. A constructor contributes a
      // fresh coordinate (§06 "Joint composition"), so it gets no value
      // expression and no roots, and can never pair-match.
      const value = isLawof ? (resolved.args || [])[0] : null;
      return {
        label: p.label,
        roots: value ? _noiseRootsOf(value, loweredModule) : new Set<string>(),
        value,
        type: _componentTypeAt(m, p.label, i),
        measure: p.value,
      };
    });
  }

  // `lawof(record(...))` — §06 "Equivalent record law". Its fields are VALUE
  // expressions, so each field's noise sources are read directly, and a field
  // that is itself a `record(...)` is a nested record law.
  if (m.op === 'lawof') {
    const inner = _resolveMeasureExpr((m.args || [])[0], loweredModule, 0);
    return _recordFieldComponents(inner, loweredModule, m);
  }

  if (m.op === 'record') return _recordFieldComponents(m, loweredModule, m);

  return null;
}

// `typeSource` is the node whose inferred domain names the component supports:
// the `lawof` wrapper when there is one (its domain is the record type), else the
// record node itself.
function _recordFieldComponents(recordIR: any, loweredModule: any,
  typeSource: any): Component[] | null {
  if (!recordIR || recordIR.kind !== 'call' || recordIR.op !== 'record') return null;
  const fields = Array.isArray(recordIR.fields) ? recordIR.fields : [];
  if (fields.length < 2) return null;
  return fields.map((f: any, i: number) => ({
    label: f.name,
    roots: _noiseRootsOf(f.value, loweredModule),
    value: f.value,
    type: _componentTypeAt(typeSource, f.name, i)
      || _componentTypeAt(recordIR, f.name, i),
    // A record FIELD is a value, not a measure — except when it is itself a
    // `record(...)`, which is the nested record law and recurses.
    measure: f.value,
  }));
}

// Why `measureIR` is singular, or null. Order matters only for the message: the
// direct pair is §06's named case, so it is reported in preference to an
// inherited one.
function _singularityOf(measureIR: any, loweredModule: any, depth: number): Reason | null {
  if (depth > MAX_DEPTH) return null;
  const m = _resolveMeasureExpr(measureIR, loweredModule, 0);
  if (!m || m.kind !== 'call') return null;

  // `iid(M, size)` — the product measure inherits M's singularity.
  if (m.op === 'iid') {
    const inner = _singularityOf((m.args || [])[0], loweredModule, depth + 1);
    return inner ? { kind: 'iid', inner } : null;
  }

  const comps = _componentsOf(m, loweredModule);
  if (!comps) return null;

  const pair = _firstSingularPair(comps, loweredModule);
  if (pair) return pair;

  // Inheritance: a component that is itself a singular joint.
  for (const c of comps) {
    const inner = _singularityOf(c.measure, loweredModule, depth + 1);
    if (inner) return { kind: 'via', label: c.label, inner };
  }
  return null;
}

function _setsEqual(a: Set<string>, b: Set<string>): boolean {
  if (a.size !== b.size) return false;
  for (const x of a) if (!b.has(x)) return false;
  return true;
}

// Two components that are deterministic functions of one shared scalar value, both
// with continuous (Lebesgue-referenced) support — so their pair lies on a curve in
// R², which is 2-D Lebesgue-null.
function _firstSingularPair(comps: Component[], loweredModule: any): Reason | null {
  for (let i = 0; i < comps.length; i++) {
    if (!comps[i].value || comps[i].roots.size === 0) continue;
    // §06 "Reference measure for product measures": a component with COUNTING
    // reference cannot make the pair null, whatever the shared draw is.
    if (!_isScalarReal(comps[i].type)) continue;
    for (let j = i + 1; j < comps.length; j++) {
      if (!comps[j].value || comps[j].roots.size === 0) continue;
      if (!_isScalarReal(comps[j].type)) continue;
      // Cheap necessary pre-filter: a common generator forces equal root sets,
      // so unequal sets can skip the walk. `{y}` vs `{y,n}` exits here.
      if (!_setsEqual(comps[i].roots, comps[j].roots)) continue;
      const gen = _commonGenerator(comps[i].value, comps[j].value, loweredModule);
      if (!gen) continue;
      return {
        kind: 'pair',
        a: comps[i].label,
        b: comps[j].label,
        generator: gen.name,
        generatorIsDraw: gen.isDraw,
      };
    }
  }
  return null;
}

// ── diagnostics ─────────────────────────────────────────────────────────

function _describe(reason: Reason): string {
  if (reason.kind === 'pair') {
    return "components '" + reason.a + "' and '" + reason.b
      + "' are both deterministic functions of the single "
      + (reason.generatorIsDraw ? 'draw' : 'value') + " '"
      + reason.generator + "', so each is determined by the other";
  }
  if (reason.kind === 'via') {
    return "component '" + reason.label + "' is itself a singular joint ("
      + _describe(reason.inner) + ')';
  }
  return 'it is an iid product over a singular joint (' + _describe(reason.inner) + ')';
}

function _message(reason: Reason): string {
  return 'singular joint: ' + _describe(reason)
    + '. The joint law therefore has no density w.r.t. the product reference '
    + 'measure (it concentrates on a lower-dimensional subset), so this density '
    + 'query is a static error (spec §06 "Singular joints"). Sampling this joint '
    + 'stays well-defined; give each component its own draw to score it';
}

// The measure argument of every DENSITY QUERY in `ir`, with the node to locate a
// diagnostic on. Three surfaces, all of which reach `builtin_logdensityof`:
// `logdensityof(M, x)`, the `fn(logdensityof(M, _))` body a broadcast wraps, and
// the 3-arg `broadcast(logdensityof, M, points)` form where `logdensityof` rides
// as a bare ref.
function _densityQueryMeasures(ir: any, out: Array<{ measure: any; loc: any }>): void {
  if (!ir || typeof ir !== 'object') return;
  if (ir.kind === 'call' && ir.op === 'logdensityof') {
    const M = (ir.args || [])[0];
    if (M) out.push({ measure: M, loc: M.loc || ir.loc });
  } else if (ir.kind === 'call' && ir.op === 'broadcast') {
    const head = (ir.args || [])[0];
    const M = (ir.args || [])[1];
    if (head && head.kind === 'ref' && head.name === 'logdensityof' && M) {
      out.push({ measure: M, loc: M.loc || ir.loc });
    }
  }
  forEachIRChild(ir, (child: any) => _densityQueryMeasures(child, out));
}

/**
 * Error diagnostics for every statically detectable singular-joint density
 * query in `loweredModule`. Runs after type inference (it reads `inferredType`
 * to tell a measure from a variate and a continuous scalar draw from anything
 * else) and emits nothing for a model that only samples the joint.
 *
 * One diagnostic per density QUERY. A joint with three same-draw components is
 * one modelling mistake, not three (`_firstSingularPair` stops at the first
 * offending pair), but two queries over the same singular joint each report,
 * since each is its own ill-formed query.
 */
function checkSingularJoints(loweredModule: any): any[] {
  const diagnostics: any[] = [];
  for (const [, binding] of loweredModule.bindings) {
    const queries: Array<{ measure: any; loc: any }> = [];
    _densityQueryMeasures(binding.rhs, queries);
    for (const q of queries) {
      const reason = _singularityOf(q.measure, loweredModule, 0);
      if (!reason) continue;
      diagnostics.push({ severity: 'error', message: _message(reason), loc: q.loc });
    }
  }
  return diagnostics;
}

module.exports = { checkSingularJoints };

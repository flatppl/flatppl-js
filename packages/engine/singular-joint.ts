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
// What IS sufficient, at pair granularity: the two components are deterministic
// functions of the SAME SINGLE continuous scalar draw. Then the pair's support is
// the image of a 1-dimensional variable in R², a curve, which is 2-D
// Lebesgue-null — so no density w.r.t. the product reference. Both of §06's named
// classes are this case (the same draw twice is `f = g = id`; a deterministic
// transform of another component is `g = h ∘ f`).
//
// Three conditions, each load-bearing, each with a false positive behind it:
//   equal root sets   — `{y} vs {y,n}` means the second component carries noise
//                       the first does not, so it is not determined by it;
//   singleton         — a strict subset relation among larger sets is Hall's
//                       condition territory (see the deficiency note below);
//   continuous scalar — a MULTI-coordinate draw fails because two components can
//                       read different coordinates of it (`v ~ MvNormal(...);
//                       a = v[1]; b = v[2]` is two independent standard normals,
//                       and root identity is per-draw-binding, so it cannot see
//                       the difference). A DISCRETE draw fails for a deeper
//                       reason: `joint(a = lawof(k), b = lawof(k))` for `k ~
//                       Bernoulli(p)` is not singular at all. Its reference
//                       measure is counting ⊗ counting, and the diagonal of
//                       {0,1}² is NOT null w.r.t. counting measure — the law has
//                       a perfectly good pmf. Lebesgue-nullity is a
//                       continuous-support argument and does not transfer.
//
// The general rule is Hall's condition on the component→root bipartite graph (the
// joint's Jacobian drops rank exactly when some subset S has |⋃_{i∈S} R_i| < |S|).
// This pass implements only the |S| = 2 case with equal singletons. A deficiency
// needing a subset of size ≥ 3 — `lawof(record(inner = record(a = y, b = t), c =
// y))`, whose support is {(y,t,y)} ⊂ R³ — is a MISS. See the fail-silent note.
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

// A draw whose variate is a single CONTINUOUS real coordinate — the only kind for
// which "both components are functions of this one draw" implies Lebesgue-null.
// An absent type answers false: unclassifiable means silent.
function _isContinuousScalarDraw(name: string, loweredModule: any): boolean {
  const b = loweredModule.bindings.get(name);
  if (!b || !_isDrawBinding(b)) return false;
  const t = b.inferredType;
  return !!(t && t.kind === 'scalar' && t.prim === 'real');
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
  | { kind: 'pair'; a: string; b: string; ancestor: string }
  | { kind: 'via'; label: string; inner: Reason }
  | { kind: 'iid'; inner: Reason };

// The components of a record-valued joint law as { label, roots, measure }:
// `roots` feeds the pair rule, `measure` (when present) is the component's own
// measure expression, which the inheritance rule recurses into. Null when
// `measureIR` is not a shape this pass classifies — the fail-silent exit.
function _componentsOf(measureIR: any, loweredModule: any):
Array<{ label: string; roots: Set<string>; measure: any }> | null {
  const m = _resolveMeasureExpr(measureIR, loweredModule, 0);
  if (!m || m.kind !== 'call') return null;

  if (m.op === 'joint') {
    const parts: Array<{ label: string; value: any }> = Array.isArray(m.fields)
      ? m.fields.map((f: any) => ({ label: f.name, value: f.value }))
      : Array.isArray(m.args)
        ? m.args.map((a: any, i: number) => ({ label: '#' + (i + 1), value: a }))
        : [];
    if (parts.length < 2) return null;
    return parts.map((p) => {
      const resolved = _resolveMeasureExpr(p.value, loweredModule, 0);
      const isLawof = !!(resolved && resolved.kind === 'call' && resolved.op === 'lawof');
      return {
        label: p.label,
        // Only a REIFIED component carries noise roots. A constructor
        // contributes a fresh coordinate (§06 "Joint composition": "A component
        // contributes a fresh coordinate"), so it gets the empty set and can
        // never pair-match.
        roots: isLawof
          ? _noiseRootsOf((resolved.args || [])[0], loweredModule)
          : new Set<string>(),
        measure: p.value,
      };
    });
  }

  // `lawof(record(...))` — §06 "Equivalent record law". Its fields are VALUE
  // expressions, so each field's noise sources are read directly, and a field
  // that is itself a `record(...)` is a nested record law.
  if (m.op === 'lawof') {
    const inner = _resolveMeasureExpr((m.args || [])[0], loweredModule, 0);
    return _recordFieldComponents(inner, loweredModule);
  }

  if (m.op === 'record') return _recordFieldComponents(m, loweredModule);

  return null;
}

function _recordFieldComponents(recordIR: any, loweredModule: any):
Array<{ label: string; roots: Set<string>; measure: any }> | null {
  if (!recordIR || recordIR.kind !== 'call' || recordIR.op !== 'record') return null;
  const fields = Array.isArray(recordIR.fields) ? recordIR.fields : [];
  if (fields.length < 2) return null;
  return fields.map((f: any) => ({
    label: f.name,
    roots: _noiseRootsOf(f.value, loweredModule),
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

// The |S| = 2 Hall deficiency: two components that are deterministic functions of
// the SAME single continuous scalar draw.
function _firstSingularPair(comps: Array<{ label: string; roots: Set<string> }>,
  loweredModule: any): Reason | null {
  for (let i = 0; i < comps.length; i++) {
    const ri = comps[i].roots;
    if (ri.size !== 1) continue;
    const r = [...ri][0];
    if (!_isContinuousScalarDraw(r, loweredModule)) continue;
    for (let j = i + 1; j < comps.length; j++) {
      const rj = comps[j].roots;
      if (rj.size !== 1 || !rj.has(r)) continue;
      return { kind: 'pair', a: comps[i].label, b: comps[j].label, ancestor: r };
    }
  }
  return null;
}

// ── diagnostics ─────────────────────────────────────────────────────────

function _describe(reason: Reason): string {
  if (reason.kind === 'pair') {
    return "components '" + reason.a + "' and '" + reason.b
      + "' are both deterministic functions of the single draw '"
      + reason.ancestor + "', so each is determined by the other";
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

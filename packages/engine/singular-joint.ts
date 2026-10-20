'use strict';

// A single law may put continuous coordinates on a lower-dimensional support.
// Detect shared scalar generators within that law and recurse into singular
// components. Independent product components never share their absorbed draws.
// The reference measure follows each coordinate's type: a discrete coordinate
// does not make a continuous/discrete law singular. Unknown shapes remain for
// the density evaluator to handle or refuse.

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
      return {
        label: p.label,
        roots: new Set<string>(),
        value: null,
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

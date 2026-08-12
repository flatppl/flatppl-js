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
// This module is the "static error where statically detectable" half. The other
// half — "otherwise refused by the engine" — is `clm._refuseIfSingular`, which
// stays as the backstop for every shape this pass declines to classify. The two
// share a doctrine but not a substrate: clm reads the post-lift DERIVATION
// table, this reads the LOWERED IR, which is what the analyzer has. The
// duplication is deliberate — a static pass that had to wait for
// `buildDerivations` could not be a static pass.
//
// The identity that decides singularity is the component's set of INDEPENDENT
// NOISE SOURCES: the draws it is a deterministic function of. Two components are
// singular exactly when those sets overlap, because then one is a function of
// the other given the shared ancestors. `lawof(a)` and `lawof(b)` for `a, b ~
// Normal(z, 1)` do NOT overlap — each carries its own noise — even though both
// trace through `z`; that is §06's correlated, absolutely-continuous case, and
// firing on it would refuse a model the engine answers exactly.
//
// A CONSTRUCTOR component contributes no noise source at all, per §06 "Joint
// composition": "A component contributes a fresh coordinate". So `joint(a = q,
// b = q)` with `q = Normal(mu = z, sigma = 0.6)` is the correlated compound law,
// not the singular diagonal, and must not fire here. At this layer the
// discriminator is structural: a component is a reified draw only when it is
// spelled `lawof(<value>)`, and `q` is a distribution call.
//
// WHY THE DENSITY QUERY, NOT THE JOINT. §06 makes sampling well-defined — the
// singular joint itself is a legal binding, and `joint(a = lawof(y), b =
// lawof(y))` samples on the diagonal. Only a DENSITY query over it is the error.
// So this pass starts from `logdensityof` call sites and works inward; a model
// that merely builds the joint gets no diagnostic.
//
// FAIL-SILENT BY DESIGN. Every unrecognised shape yields no diagnostic rather
// than a guess. A missed static diagnostic still refuses at density time
// (clm's backstop), so the cost of a miss is a late error message; the cost of a
// false positive is a legal model the user cannot score. The asymmetry decides
// every judgement call below.

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
// shared noise (which is exactly what keeps `a, b ~ Normal(z, 1)` separable).
function _isDrawBinding(binding: any): boolean {
  const rhs = binding && binding.rhs;
  return !!(rhs && rhs.kind === 'call' && rhs.op === 'draw');
}

function _isMeasureTyped(binding: any): boolean {
  const t = binding && binding.inferredType;
  return !!(t && t.kind === 'measure');
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

// The components of a record-valued joint law, as { label, roots } — `roots`
// being the component's noise sources. Null when `measureIR` is not a shape
// this pass classifies, which is the fail-silent exit.
//
// Two spellings reach here, and §06 "Equivalent record law" says they are the
// same measure: "`joint(a = lawof(a), b = lawof(b))` is equivalent to
// `lawof(record(a = a, b = b))`; the positional form is the corresponding `cat`
// law". Both are classified, so both diagnose.
function _componentRoots(measureIR: any, loweredModule: any):
Array<{ label: string; roots: Set<string> }> | null {
  const m = _resolveMeasureExpr(measureIR, loweredModule, 0);
  if (!m || m.kind !== 'call') return null;

  if (m.op === 'joint') {
    const parts: Array<{ label: string; value: any }> = Array.isArray(m.fields)
      ? m.fields.map((f: any) => ({ label: f.name, value: f.value }))
      : Array.isArray(m.args)
        ? m.args.map((a: any, i: number) => ({ label: '#' + (i + 1), value: a }))
        : [];
    if (parts.length < 2) return null;
    return parts.map((p) => ({
      label: p.label,
      roots: _jointComponentRoots(p.value, loweredModule),
    }));
  }

  // `lawof(record(...))` — the equivalent record law. Its fields are VALUE
  // expressions, so each field's noise sources are read directly.
  if (m.op === 'lawof') {
    const inner = _resolveMeasureExpr((m.args || [])[0], loweredModule, 0);
    if (!inner || inner.kind !== 'call' || inner.op !== 'record') return null;
    const fields = Array.isArray(inner.fields) ? inner.fields : [];
    if (fields.length < 2) return null;
    return fields.map((f: any) => ({
      label: f.name,
      roots: _noiseRootsOf(f.value, loweredModule),
    }));
  }

  return null;
}

// A `joint` component's noise sources. Only a REIFIED component carries any: a
// constructor contributes a fresh coordinate (§06 "Joint composition"), so it
// gets the empty set and can never overlap with anything.
function _jointComponentRoots(value: any, loweredModule: any): Set<string> {
  const resolved = _resolveMeasureExpr(value, loweredModule, 0);
  if (resolved && resolved.kind === 'call' && resolved.op === 'lawof') {
    return _noiseRootsOf((resolved.args || [])[0], loweredModule);
  }
  return new Set<string>();
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
 * query in `loweredModule`. Runs after type inference (it reads
 * `inferredType` to tell a measure binding from a variate) and emits nothing
 * for a model that only samples the joint.
 */
function checkSingularJoints(loweredModule: any): any[] {
  const diagnostics: any[] = [];
  for (const [, binding] of loweredModule.bindings) {
    const queries: Array<{ measure: any; loc: any }> = [];
    _densityQueryMeasures(binding.rhs, queries);
    for (const q of queries) {
      const comps = _componentRoots(q.measure, loweredModule);
      if (!comps) continue;
      const hit = _firstOverlap(comps);
      if (!hit) continue;
      diagnostics.push({
        severity: 'error',
        message: "singular joint: components '" + hit.a + "' and '" + hit.b
          + "' are determined by the same draw '" + hit.ancestor + "' — no "
          + 'independent noise separates them, so the joint law has no density '
          + 'w.r.t. the product reference measure (it concentrates on a '
          + 'lower-dimensional subset) and this density query is a static error '
          + '(spec §06 "Singular joints"). Sampling this joint stays '
          + 'well-defined; give each component its own draw to score it',
        loc: q.loc,
      });
      // One diagnostic per query: a joint with three same-draw components is
      // one modelling mistake, not three.
      break;
    }
  }
  return diagnostics;
}

function _firstOverlap(comps: Array<{ label: string; roots: Set<string> }>):
{ a: string; b: string; ancestor: string } | null {
  for (let i = 0; i < comps.length; i++) {
    for (let j = i + 1; j < comps.length; j++) {
      for (const r of comps[i].roots) {
        if (comps[j].roots.has(r)) {
          return { a: comps[i].label, b: comps[j].label, ancestor: r };
        }
      }
    }
  }
  return null;
}

module.exports = {
  checkSingularJoints,
  _internal: { _componentRoots, _noiseRootsOf, _densityQueryMeasures },
};

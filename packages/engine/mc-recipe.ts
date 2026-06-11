'use strict';

// =====================================================================
// mc-recipe — functor-law composition of a generative-composite measure
// into the canonical MC marginalising-pushforward density form.
// =====================================================================
//
// A generative likelihood whose per-event observation is a bijective
// pushforward of one RETAINED internal innovation but MARGINALISES a
// latent draw (the transport model) reaches density as a tree of nested
// broadcasts over an iid collection, e.g.
//
//   broadcast(post,                       # deterministic ÷2  (functionof)
//     broadcast(transport,                # generative kernel (internal Uniform)
//       iid(Normal(pars.mu, σ), n),       # latent collection to MARGINALISE
//       [pars]))
//
// `buildMcMarginalForm` re-expresses that tree as the canonical density
// form — applying the pushforward functor law (broadcast-over-iid =
// iid-of-pushforward; deterministic outer maps fold into the per-event
// map):
//
//   iid( mcmarginal{ inverseIR, ladjIR, outName, retainedRef,
//                    retainedInterval, marginalRef, marginalDistIR }, n )
//
// which density.walkIid factorises per-event into density.walkMcMarginal
// (spec §06 case-3 opt-in; engine-concepts §6). The retained inverse +
// forward LADJ are derived EXACTLY by bijection-registry.invertExpr.
//
// This is a pure IR→IR transform with NO sampling/materialiser
// dependency — reused by the density path (mat-density.matLogdensityof)
// now and by the dissolver later (the sampling-side functor-law
// canonicalisation, so sampling consumes the same per-event form).
//
// Returns null when the tree isn't the recognised v1 shape (exactly one
// invertible Uniform internal innovation + exactly one iid-collection
// latent); the caller then refuses (or falls through to closed-form).

const bijReg = require('./bijection-registry.ts');
const kbShape = require('./kernel-broadcast-shape.ts');
const shared = require('./ir-shared.ts');

const OUT_NAME = '__mc_z__';

// Function/kernel-like bindings are left as refs by the inliner (mirrors
// materialiser-shared.isFunctionLikeBinding — inlined here to keep this
// transform free of any materialiser dependency).
function _isFunctionLike(binding: any): boolean {
  if (!binding) return false;
  switch (binding.type) {
    case 'fn': case 'functionof': case 'kernelof': case 'bijection': case 'fchain':
      return true;
    default: return false;
  }
}

// Resolve a measure-position node to its FULLY-EXPANDED measure call.
// A `self`-ref is followed through a `draw(M)` to M, then deep-expanded
// via `expand` (orchestrator.expandMeasure) so kernel applications resolve
// to leaf dists — e.g. `xs = draw(__anon8)`, `__anon8 = iid(generator(pars), n)`
// expands to `iid(Normal(pars.mu, σ), n)`. Falls back to the raw binding
// IR when `expand` can't (non-measure bindings like the `[pars]` arg).
function _resolveMeasure(node: any, bindings: any, expand: any): any {
  if (!node || node.kind !== 'ref' || node.ns !== 'self') return node;
  const b = bindings.has(node.name) ? bindings.get(node.name) : null;
  let nameToExpand = node.name;
  if (b && b.ir && b.ir.kind === 'call' && b.ir.op === 'draw'
      && Array.isArray(b.ir.args) && b.ir.args[0] && b.ir.args[0].kind === 'ref') {
    nameToExpand = b.ir.args[0].name;            // draw(M) → expand M
  }
  const expanded = (typeof expand === 'function') ? expand(nameToExpand) : null;
  if (expanded) return expanded;
  return (b && b.ir) || node;
}

// Inline module value bindings into a self-contained expression: draws,
// boundary formals, fixed-phase and function-like bindings stay as refs;
// every other value binding is inlined (cf. mat-broadcast._inlineGenerative-
// Body). Bounded recursion (spec §04 modules are DAGs).
function _inline(ir: any, bindings: any, drawNames: Set<string>, boundary: Set<string>): any {
  if (!ir || typeof ir !== 'object') return ir;
  if (ir.kind === 'ref' && ir.ns === 'self') {
    const nm = ir.name;
    if (drawNames.has(nm) || boundary.has(nm)) return ir;
    if (!bindings.has(nm)) return ir;
    const b = bindings.get(nm);
    if (!b || !b.ir || _isFunctionLike(b) || b.phase === 'fixed') return ir;
    return _inline(b.ir, bindings, drawNames, boundary);
  }
  if (ir.kind !== 'call') return ir;
  const out: any = { kind: 'call' };
  if (ir.op) out.op = ir.op;
  if (ir.target) out.target = ir.target;
  if (Array.isArray(ir.args)) out.args = ir.args.map((a: any) => _inline(a, bindings, drawNames, boundary));
  if (ir.kwargs) { out.kwargs = {}; for (const k in ir.kwargs) out.kwargs[k] = _inline(ir.kwargs[k], bindings, drawNames, boundary); }
  if (Array.isArray(ir.fields)) out.fields = ir.fields.map((f: any) => ({ ...f, value: _inline(f && f.value, bindings, drawNames, boundary) }));
  return out;
}

// [lo,hi] from a Uniform(interval(lo,hi)) measure IR (constant bounds), else null.
function _uniformInterval(mIR: any, bindings: any): number[] | null {
  if (!mIR || mIR.kind !== 'call' || mIR.op !== 'Uniform') return null;
  let supp = (mIR.args && mIR.args[0]) || (mIR.kwargs && (mIR.kwargs.support || mIR.kwargs.set));
  if (supp && supp.kind === 'ref' && supp.ns === 'self' && bindings.has(supp.name)) {
    const sb = bindings.get(supp.name); if (sb && sb.ir) supp = sb.ir;
  }
  if (!supp || supp.kind !== 'call' || supp.op !== 'interval') return null;
  const lo = shared.resolveConstant(supp.args[0], bindings, new Set());
  const hi = shared.resolveConstant(supp.args[1], bindings, new Set());
  if (typeof lo !== 'number' || typeof hi !== 'number') return null;
  return [lo, hi];
}

// Follow a draw binding `name = draw(ref M)` to M's measure IR, else null.
function _drawMeasureIR(name: string, bindings: any): any {
  const b = bindings.has(name) ? bindings.get(name) : null;
  if (!b || !b.ir || b.ir.kind !== 'call' || b.ir.op !== 'draw') return null;
  const arg = b.ir.args && b.ir.args[0];
  if (arg && arg.kind === 'ref' && arg.ns === 'self' && bindings.has(arg.name)) {
    const mb = bindings.get(arg.name);
    return (mb && mb.ir) ? mb.ir : arg;
  }
  return arg || null;
}

// Collect the draw-binding names reachable from an expression's `self`-refs
// (stopping AT a draw binding — its own measure is the leaf). Bounded by a
// visited set; spec §04 modules are DAGs.
function _collectDrawAncestors(ir: any, bindings: any, draws: Set<string>, seen: Set<string>): void {
  if (!ir || typeof ir !== 'object') return;
  if (ir.kind === 'ref' && ir.ns === 'self') {
    const nm = ir.name;
    if (seen.has(nm)) return;
    seen.add(nm);
    const b = bindings.has(nm) ? bindings.get(nm) : null;
    if (b && b.ir && b.ir.kind === 'call' && b.ir.op === 'draw') { draws.add(nm); return; }
    if (b && b.ir) _collectDrawAncestors(b.ir, bindings, draws, seen);
    return;
  }
  if (ir.kind !== 'call') return;
  if (Array.isArray(ir.args)) for (const a of ir.args) _collectDrawAncestors(a, bindings, draws, seen);
  if (ir.kwargs) for (const k in ir.kwargs) _collectDrawAncestors(ir.kwargs[k], bindings, draws, seen);
  if (Array.isArray(ir.fields)) for (const f of ir.fields) _collectDrawAncestors(f && f.value, bindings, draws, seen);
}

// True if `ir` contains a `draw` op anywhere (a non-deterministic body).
function _containsDraw(ir: any): boolean {
  if (!ir || typeof ir !== 'object') return false;
  if (ir.kind === 'call' && ir.op === 'draw') return true;
  if (Array.isArray(ir.args)) for (const a of ir.args) if (_containsDraw(a)) return true;
  if (ir.kwargs) for (const k in ir.kwargs) if (_containsDraw(ir.kwargs[k])) return true;
  if (Array.isArray(ir.fields)) for (const f of ir.fields) if (_containsDraw(f && f.value)) return true;
  return false;
}

// Substitute a `%local` parameter ref (a deterministic functionof formal)
// with `arg` throughout `body`.
function _substLocal(body: any, paramName: string, arg: any): any {
  if (!body || typeof body !== 'object') return body;
  if (body.kind === 'ref' && body.ns === '%local' && body.name === paramName) return arg;
  if (body.kind !== 'call') return body;
  const out: any = { kind: 'call' };
  if (body.op) out.op = body.op;
  if (body.target) out.target = body.target;
  if (Array.isArray(body.args)) out.args = body.args.map((a: any) => _substLocal(a, paramName, arg));
  if (body.kwargs) { out.kwargs = {}; for (const k in body.kwargs) out.kwargs[k] = _substLocal(body.kwargs[k], paramName, arg); }
  if (Array.isArray(body.fields)) out.fields = body.fields.map((f: any) => ({ ...f, value: _substLocal(f && f.value, paramName, arg) }));
  return out;
}

/**
 * Re-express a generative-composite measure tree as
 * `iid(mcmarginal{…}, n)`, or return null if it isn't the v1 shape.
 *
 * `expand(name)` deep-expands a measure binding (orchestrator.expand-
 * Measure) so sub-measures resolve to leaf dists; supplied by the caller
 * (which holds the derivations).
 */
function buildMcMarginalForm(measureIR: any, bindings: any, expand: any): any {
  if (!bindings || !bindings.has) return null;
  // 1. Peel deterministic broadcast(f_det, inner) layers (outermost first),
  //    collecting their functionof bodies to fold into the per-event map.
  const detMaps: Array<{ paramName: string; body: any }> = [];
  let cur = _resolveMeasure(measureIR, bindings, expand);
  let guard = 0;
  while (cur && cur.kind === 'call' && cur.op === 'broadcast' && guard++ < 64) {
    const head = cur.args && cur.args[0];
    const headName = head && head.kind === 'ref' ? head.name : null;
    if (!headName) return null;
    // Generative kernel head → the per-event generative layer; build here.
    const gen = kbShape.detectGenerativeKernelBinding(headName, bindings);
    if (gen) return _buildFromGenerative(cur, gen, detMaps, bindings, expand);
    // Deterministic functionof head (no internal draw) → a pushforward map
    // to fold in; recurse into its single collection arg.
    const hb = bindings.get(headName);
    const hir = hb && hb.ir;
    if (!hir || hir.kind !== 'call' || hir.op !== 'functionof'
        || !Array.isArray(hir.params) || hir.params.length !== 1
        || !hir.body || _containsDraw(hir.body)) {
      return null;
    }
    detMaps.push({ paramName: hir.params[0], body: hir.body });
    cur = _resolveMeasure(cur.args[1], bindings, expand);
  }
  // No broadcast tree: the bare single-event form `m = lawof(z)` where z is a
  // value transform of stochastic ancestors. Only attempt it when nothing was
  // peeled (guard===0 — a measure with a half-peeled broadcast prefix is a
  // malformed generative composite, not a bare transform) and `cur` is a value
  // call (not a leaf measure/distribution — those have no draw ancestors and
  // _buildFromValueExpr returns null anyway, but the guard keeps intent clear).
  if (guard === 0 && cur && cur.kind === 'call') {
    return _buildFromValueExpr(cur, bindings);
  }
  return null;
}

// Bare single-event form: `m = lawof(z)` where z is a deterministic value
// transform of stochastic ancestors — at least one internal draw (the
// retained innovation, bijective in z) and exactly one MARGINALISED latent.
// This is the un-broadcasted sibling of _buildFromGenerative: there is no iid
// collection / broadcast axis (z is one scalar event), so the result is the
// bare `mcmarginal` node (NOT iid-wrapped) — density.walkMcMarginal scores the
// single event, marginalising the latent over its M in-worker MC samples
// (sampleLeafN + batched evaluateExprN). The broadcast/iid generative
// composites still take the batched per-event path above; this fallback only
// fires when there is no broadcast tree, so it never bypasses that batching.
//
// Draw classification is by-construction, not heuristic: invertExpr succeeds
// only for the draw z is a straight-line bijection of (it appears once on the
// path) — that is the retained innovation; the rest are marginalised.
function _buildFromValueExpr(exprIR: any, bindings: any): any {
  const draws = new Set<string>();
  _collectDrawAncestors(exprIR, bindings, draws, new Set<string>());
  if (draws.size < 2) return null;               // need ≥1 retained + ≥1 marginal
  // Inline the value closure down to its draw leaves (draws stay refs).
  const recipe = _inline(exprIR, bindings, draws, new Set<string>());
  const OUT = { kind: 'ref', ns: '%mc', name: OUT_NAME };
  let retainedRef: string | null = null;
  let retainedInterval: number[] | null = null;
  let inv: any = null;
  const marginals: string[] = [];
  for (const dn of draws) {
    if (!retainedRef) {
      const r = bijReg.invertExpr({ outputExpr: recipe, freeRef: { name: dn }, outputValue: OUT });
      const ivl = r ? _uniformInterval(_drawMeasureIR(dn, bindings), bindings) : null;
      if (r && ivl) { retainedRef = dn; retainedInterval = ivl; inv = r; continue; }
    }
    marginals.push(dn);
  }
  if (!retainedRef || !inv || marginals.length !== 1) return null;   // v1 shape
  const marginalRef = marginals[0];
  const marginalDistIR = _drawMeasureIR(marginalRef, bindings);
  if (!marginalDistIR) return null;
  const ext = new Set<string>();
  for (const r of shared.collectSelfRefs(inv.inverseIR)) ext.add(r);
  for (const r of shared.collectSelfRefs(inv.ladjIR)) ext.add(r);
  for (const r of shared.collectSelfRefs(marginalDistIR)) ext.add(r);
  ext.delete(retainedRef);
  ext.delete(marginalRef);
  return {
    kind: 'call', op: 'mcmarginal',
    inverseIR: inv.inverseIR, ladjIR: inv.ladjIR, outName: OUT_NAME,
    retainedRef, retainedInterval, marginalRef, marginalDistIR,
    externalRefs: Array.from(ext),
  };
}

function _buildFromGenerative(genBcastIR: any, gen: any, detMaps: Array<{ paramName: string; body: any }>, bindings: any, expand: any): any {
  const params: string[] = gen.params || [];
  const internalDraws: Array<{ bindingName: string; distIR: any }> = gen.internalDraws || [];
  // v1: exactly one internal innovation (the retained draw).
  if (internalDraws.length !== 1) return null;
  const retainedRef = internalDraws[0].bindingName;
  const retainedInterval = _uniformInterval(internalDraws[0].distIR, bindings);
  if (!retainedInterval) return null;            // v1: Uniform innovation

  // Map broadcast positional args to kernel params; find the iid-collection
  // latent (the marginalised draw) — exactly one in v1.
  const args: any[] = (genBcastIR.args || []).slice(1);
  let marginalRef: string | null = null;
  let marginalDistIR: any = null;
  let nIR: any = null;
  for (let i = 0; i < params.length && i < args.length; i++) {
    const m = _resolveMeasure(args[i], bindings, expand);
    if (m && m.kind === 'call' && m.op === 'iid') {
      if (marginalRef) return null;              // v1: a single marginalised latent
      marginalRef = params[i];
      marginalDistIR = m.args[0];                // per-event base measure (self-contained vs env)
      nIR = m.args[1];                           // iid count
    }
    // Otherwise an atom-constant boundary ([pars]) — resolved from the
    // session env at evaluation; left implicit (no recipe ref rewrite).
  }
  if (!marginalRef || !marginalDistIR || !nIR) return null;

  // Per-event recipe: inline the generative body, then fold the
  // deterministic outer maps in (functor law, innermost-first).
  const drawNames = new Set<string>(internalDraws.map((d) => d.bindingName));
  const boundary = new Set<string>(params);
  let recipe = _inline(gen.bodyValueExprIR, bindings, drawNames, boundary);
  for (let i = detMaps.length - 1; i >= 0; i--) {
    recipe = _substLocal(detMaps[i].body, detMaps[i].paramName, recipe);
  }

  // Exact inverse + forward LADJ in the retained innovation.
  const OUT = { kind: 'ref', ns: '%mc', name: OUT_NAME };
  const inv = bijReg.invertExpr({ outputExpr: recipe, freeRef: { name: retainedRef }, outputValue: OUT });
  if (!inv) return null;                          // not bijective in the retained draw

  // EXTERNAL refs the worker resolves from its env (session-fixed θ like
  // `pars`/`sigma`, or per-atom in a posterior): the free `self`-refs of the
  // recipe MINUS the internally-supplied retained / marginal draws (the
  // output placeholder is `%mc`-namespaced, never collected). The node's
  // recipe lives in custom fields `walkIR`/`collectSelfRefs` do not descend,
  // so this list is how the main-thread ref-collection (prepareDensityRefs)
  // sees through the otherwise-opaque node — see ir-shared.collectSelfRefs.
  const ext = new Set<string>();
  for (const r of shared.collectSelfRefs(inv.inverseIR)) ext.add(r);
  for (const r of shared.collectSelfRefs(inv.ladjIR)) ext.add(r);
  for (const r of shared.collectSelfRefs(marginalDistIR)) ext.add(r);
  ext.delete(retainedRef);
  ext.delete(marginalRef);

  const node = {
    kind: 'call', op: 'mcmarginal',
    inverseIR: inv.inverseIR, ladjIR: inv.ladjIR, outName: OUT_NAME,
    retainedRef, retainedInterval,
    marginalRef, marginalDistIR,
    externalRefs: Array.from(ext),
  };
  return { kind: 'call', op: 'iid', args: [node, nIR] };
}

module.exports = { buildMcMarginalForm, OUT_NAME };

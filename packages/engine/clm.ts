'use strict';

// ════════════════════════════════════════════════════════════════════════
// Canonical Lowered Measure (CLM) — the single measure-lowering pass both
// the SAMPLE walker (materialiser / sampler) and the DENSITY walker
// (density.ts) will consume unchanged, so "sampling ≡ density" becomes a
// STRUCTURAL property rather than a hoped-for test invariant.
// Design: flatppl-dev/measure-lowering-unification-plan.md.
//
// Today the engine walks the same measure IR along two independent paths
// (plus a third inliner in the viewer), each re-deriving the same five
// normalisation steps INCONSISTENTLY, so they drift (the measure-algebra
// audit's central finding). `lowerMeasure` performs those steps ONCE and
// emits one tagged node:
//
//   clm { kind:'call', op:'clm', body, inputs, reduce, mcOpts? }
//
//   body    — the expanded measure IR (existing node vocabulary only),
//             after, in FIXED ORDER: (1) draw/lawof peeled + (2) measure
//             refs expanded by name — both already done by expandMeasure;
//             (3) derived VALUE bindings transitively inlined to the
//             boundary set (computeClosureIR); (4) buildMcMarginalForm
//             applied LAST when body matches the generative-composite
//             shape.
//   inputs  — ordered { name, ns, source, shape }[]; `source` is one of
//             {kind:'boundary', from, field?, splat?} (fed from a caller
//             measure's atoms — the prior field / integration variable /
//             history column / product factor), {kind:'shared', ref} (the
//             only surviving getMeasure-as-boundary path, now EXPLICIT), or
//             {kind:'fixed', ref}. Each carries a shape/axis descriptor
//             (critique C — repeat-axis awareness is NOT optional).
//   reduce  — null for a plain product measure, or {kind:'marginal', method}
//             for kchain / lawof-of-record (H8) / mc-generative — replacing
//             the scattered isChain/naryKchain/mcForm booleans. `method` says
//             HOW the marginal is evaluated — the shared-ancestor marginal
//             admits no stochastic answer (see the H8 branch below for the
//             §06 anchor and the owner decision):
//               'logsumexp-logN'   — the kchain reduction over the fed prior
//                                    atoms, and the vacuous no-op case where
//                                    every ancestor is a threaded record field
//               'analytic-gaussian'— the shared-ancestor closed form
//                                    (linear-gaussian.ts), carried on
//                                    reduce.gaussian
//               'analytic-mixture' — the same closed form per atom of an
//                                    ENUMERATED finite discrete ancestor,
//                                    carried on reduce.gaussian.mixture
//                                    (deterministic and exact, not MC)
//               'refuse'           — a marginal this engine cannot close;
//                                    reduce.reason says why, and the density
//                                    consumer throws it
//
// Structural invariant the pass asserts: collectSelfRefs(body) ⊆
// {i.name for i in inputs} (callable refs excluded — they resolve by name,
// not by feeding), descending into .bijection (ir-walk prereq D / H4).
//
// Phases 2–6 LANDED: feedInputs + the consumers (matScore, the sampler/density
// clm branches) are live — clm is the production jointchain/kchain path. The
// ⊆ check THROWS (the Phase-6 assertion flip, main-thread per plan critique F):
// a lowering that cannot declare every body self-ref is not self-contained,
// and feeding it would silently re-materialise like-named module bindings
// (the audit-§3 boundary-conflation class). The dual guarantee:
//   lowerMeasure — every body ref is DECLARED (⊆ throw, here);
//   matScore + assertFedCoverage — every declared boundary the body
//   references is FED (coverage throw, after the extraRefArrays merge);
//   matClm — the body walk re-checks via the ctx._boundaryNames net.
// ════════════════════════════════════════════════════════════════════════

const orchestrator = require('./orchestrator.ts');
const shared = require('./materialiser-shared.ts');
const builtins = require('./builtins.ts');
const valueLib = require('./value.ts');

// A scorable measure node (a distribution or measure-algebra op) vs a value
// transform. The stochastic-ancestor marginal (H8) applies only to a measure
// body — `lawof(Normal(theta,1))` → marginalise theta. A deterministic
// transform body, `lawof(z)` with z = f(stochastic…), is a PUSHFORWARD whose
// density needs the mc-marginal recipe (inverse + LADJ), not a plain
// logsumexp; applying the marginal there would mis-score it, so it is left to
// the mc-form path (and stays a loud refusal until that recogniser lands).
function _isMeasureNode(ir: any): boolean {
  return !!(ir && ir.kind === 'call' && ir.op
    && (builtins.DISTRIBUTIONS.has(ir.op)
      || builtins.MEASURE_PRODUCING.has(ir.op)
      || builtins.MEASURE_OPS.has(ir.op)));
}


// ── shape / axis descriptor (critique C) ────────────────────────────────
//
// Each clm input carries enough shape information for feedInputs (Phase 2)
// to lay it out correctly — crucially under the inflated repeat-block child
// ctx (iid(generative_kernel, n): sampleCount = N·k, repeatBlock = k) where
// VALUE-typed boundaries must tile atom-major and MEASURE-typed ones must
// fresh-redraw. The descriptor is best-effort static; feedInputs reconciles
// against the runtime measure it fetches.
//
//   kind        'scalar' | 'vector' | 'matrix' | 'record' | 'opaque'
//   perAtom     does the value vary per atom (a [N,…] column) or is it
//               atom-independent (a single value broadcast)?
//   repeatTile  'value'  — tile atom-major across the [N,k] repeat block
//               'measure'— the measure subtree redraws per (atom,cell)
//               'fixed'  — a fixed-phase value (session env; no tiling)
//   fields?     for kind:'record', the field labels (declaration order)
function describeInputShape(source: any, name: string, ctx: any): any {
  // Explicit boundary (a viewer preset / selected kernel input) carries its
  // value directly — describe it off the value, not off a binding. A free
  // input (the profile sweep axis) is left opaque (it is not fed).
  if (source.kind === 'explicit') return _describeValueShape(source.value);
  if (source.kind === 'free') {
    return { kind: 'opaque', perAtom: false, repeatTile: 'value', fields: null };
  }
  const bindings = ctx && ctx.bindings;
  const repeatTile = source.kind === 'fixed' ? 'fixed'
    : source.kind === 'shared' ? 'measure'
    : 'value';                                   // boundary atoms are values
  // Read the value kind off the source binding's inferred type where we can.
  let typeRef: string | null = null;
  if (source.kind === 'boundary') typeRef = source.from;
  else if (source.kind === 'shared' || source.kind === 'fixed') typeRef = source.ref;
  const b = typeRef != null && bindings && bindings.get ? bindings.get(typeRef) : null;
  const t = b && b.inferredType;
  let kind = 'scalar';
  let fields: string[] | null = null;
  const dom = t && t.kind === 'measure' ? t.domain : t;
  if (dom && dom.kind === 'record' && dom.fields) {
    // A boundary into a single record field is that field's kind; a whole
    // record source is 'record'.
    if (source.field != null && dom.fields[source.field]) {
      kind = _domKind(dom.fields[source.field]);
    } else {
      kind = 'record';
      fields = Object.keys(dom.fields);
    }
  } else if (dom) {
    kind = _domKind(dom);
  }
  return { kind, perAtom: source.kind !== 'fixed', repeatTile, fields };
}

function _isStochastic(b: any): boolean {
  return !!(b && b.phase === 'stochastic');
}

// ── singular joints ──────────────────────────────────────────────────────
//
// spec §06 "Singular joints", quoted verbatim from flatppl-design 52df5de:
//
//   "When one component's variate is determined by the others given the shared
//   ancestors (the same draw referenced twice, a deterministic transform of
//   another component), the joint law has no density w.r.t. the product
//   reference measure. Sampling is well-defined; a density query is a static
//   error where statically detectable, and is otherwise refused by the engine."
//
// The identity that decides this is the component's set of INDEPENDENT NOISE
// SOURCES: the draws it is a deterministic function of. Two components are
// singular exactly when those sets overlap, because then one is a function of
// the other given the shared ancestors. `lawof(a)` and `lawof(b)` for `a, b ~
// Normal(z, 1)` do NOT overlap (each carries its own noise) even though both
// trace through `z` — that is the correlated, absolutely-continuous case §06
// "Equivalent record law" wants retained.
//
// `joint(m, m)` over a CONSTRUCTOR measure does not overlap either, and this
// holds whether or not `m`'s parameters reach a draw. §06 "Joint composition"
// (verbatim, flatppl-design 9e35262): "A component contributes a fresh
// coordinate; a stochastic node shared between component traces (through a
// reified component — `lawof`, `kernelof` — or a stochastic constructor
// parameter) remains a single node of the composed trace." So for `z ~
// Normal(0, 1); q = Normal(mu = z, sigma = 0.6); joint(a = q, b = q)` the shared
// node is `z`, not the coordinate: the two coordinates are conditionally
// independent given `z`, and the law is the correlated compound one, not the
// singular diagonal. Naming the constructor once and using it twice must
// therefore behave exactly as writing it out twice does.
//
// The refusal is a RUNTIME one. §06 prefers "a static error where statically
// detectable", which this shape is; an analyzer diagnostic is a follow-up
// tracked in flatppl-dev/TODO-flatppl-js.md.
//
// This gate runs only on the DENSITY path (lowerMeasure is not the sampling
// route for a record/tuple measure), so sampling a singular joint stays legal.
//
// ── SINGULARITY IS INHERITED, SO THE CHECK RECURSES ─────────────────────
//
// The pair test above reads ONE component→binding map, so a singular joint one
// level down used to be invisible and the query answered with a finite number for
// a law that has none: `iid(S, 2)` over a singular `S` scored -3.8257541328186915
// (four independent standard normals), and `joint(inner = S, c = lawof(t))` scored
// -2.8268155996140187 (three of them) — the singular pair scored as two
// independent coordinates. Two constructs carry singularity outward:
//
//   `iid(M, size)` — §06 `iid` (verbatim, flatppl-design 9e35262) is the product
//   measure M^⊗N and "never shares nodes between copies", so the copies are
//   independent and the product is null exactly when M is.
//
//   a COMPONENT that is itself a singular joint or record law — §06 "Joint
//   composition" makes a record-valued component "a nested record under its
//   name", so the outer support is the inner curve crossed with the remaining
//   coordinates, still null in the product space.
//
// Hence `_refuseIfSingular` walks the derivation table: through an `iid`
// derivation to its base measure, and into every component binding. The path is
// carried into the message, because the offending pair is not at the level the
// user queried.
//
// This is the "otherwise refused by the engine" half only. `singular-joint.ts`
// implements the same inheritance rule statically, on the lowered IR, with a
// NARROWER pair predicate (a common scalar generator, both components Lebesgue-
// referenced). The two are deliberately not the same predicate — see that
// module's "RELATION TO clm._refuseIfSingular".

// Whether `name` binds a VARIATE of a draw rather than a measure. Both a `~`-draw
// and a constructor measure whose parameters reach one carry phase 'stochastic'
// (`q = Normal(mu = z, sigma = 0.6)` is stochastic-phase), so phase alone cannot
// separate them; the domain does. A variate has a value type (scalar / array /
// record) and a measure has `kind === 'measure'`, so a binding that HAS a type is
// classified by `kind !== 'measure'`.
//
// The untyped case is separate, and is why this is not simply that one test. A
// LIFTED anon constructor binding has no inferred type at all, and is a measure;
// answering it by absence-of-type alone would be FAIL-OPEN for anything else that
// arrived untyped, since a variate that lost its type would be read as a
// constructor, get a fresh coordinate, and score a singular joint instead of
// refusing it. Every stochastic binding without a type is an internal lifted name
// today, so the guard asserts exactly that rather than trusting it — a
// user-facing name arriving here untyped is a classification this function cannot
// make, and must not answer.
function _namesADraw(name: string, b: any): boolean {
  if (!_isStochastic(b)) return false;
  if (!b.inferredType) {
    if (!_isInternalName(name)) {
      throw new Error("clm: stochastic binding '" + name + "' has no inferred type, "
        + 'so it cannot be classified as a draw variate or a constructor measure. '
        + 'Only internal lifted names (__anon…, %…) are untyped here; a user-facing '
        + 'name reaching this point would be read as a constructor and would score '
        + 'a singular joint instead of refusing it');
    }
    // A closure synthesized at a kernel APPLICATION point has no inferred type
    // — typeinfer ran before lift's boundary substitution created it. `draw(M)`
    // is unambiguously a variate whatever else is unknown, so read the IR head
    // instead of defaulting to "constructor": defaulting there let the singular
    // fan-out `logdensityof(joint(K, K)(v), y)` SCORE a product of independent
    // coordinates where §06 "Singular joints" refuses. Only this one IR shape is
    // recognised — a deterministic transform of a draw stays a constructor here,
    // per the classifier note on `_noiseRoots`.
    return !!(b.ir && b.ir.kind === 'call' && b.ir.op === 'draw');
  }
  return b.inferredType.kind !== 'measure';
}

// Lifted/internal binding names, as `_displayName` and derivations.ts spell them.
function _isInternalName(name: string): boolean {
  return typeof name === 'string' && (name.startsWith('__anon') || name.startsWith('%'));
}

// The alias chain from `name` to its terminal binding, and whether any name on
// it binds a draw's variate — i.e. whether the component reifies a DRAW rather
// than naming a constructor measure. Phase lives on the user binding, not on the
// lifted anon sample, so the whole chain is inspected.
function _aliasChain(name: string, ctx: any): { root: string; isDraw: boolean } {
  const derivations = ctx && ctx.derivations;
  let n = name;
  let isDraw = _namesADraw(n, ctx.bindings && ctx.bindings.get ? ctx.bindings.get(n) : null);
  for (let guard = 0; guard < 64; guard++) {
    const d = derivations && derivations[n];
    if (!d || d.kind !== 'alias') break;
    n = d.from;
    if (_namesADraw(n, ctx.bindings && ctx.bindings.get ? ctx.bindings.get(n) : null)) isDraw = true;
  }
  return { root: n, isDraw };
}

// The child bindings a structural (record/tuple) derivation is built from, or
// null when `d` is not one.
function _structuralChildren(d: any): string[] | null {
  if (!d) return null;
  if (d.kind === 'record' && d.fields) return Object.keys(d.fields).map((k) => d.fields[k]);
  if (d.kind === 'tuple' && Array.isArray(d.elems)) return d.elems;
  return null;
}

// The independent noise sources `name` is a deterministic function of. Walks
// through `evaluate` (deterministic value) bindings only — a `sample` binding
// IS its own noise source and terminates the walk.
//
// A STRUCTURAL (record/tuple) binding is not a noise source either: it is a
// container, so its noise is the union of its children's. Without that branch a
// nested record contributed only its own lifted name, and a shared draw split
// across nesting levels was invisible to the pair test — `lawof(record(inner =
// record(a = y, b = t), c = y))` (support {(y,t,y)} ⊂ R³, genuinely singular)
// scored -2.7868155996140187, three normals with `y` counted twice. That is
// §06's "determined by the others given the shared ancestors" across a level,
// and it is a Hall deficiency needing a subset of size 3, which no PAIR of
// components can express until the container reports its children's roots.
//
// This is the COARSE pair predicate reaching one level further, so it inherits
// that predicate's overstatement: a FULL-RANK nested law (`u = y+n1; w = y+n2;
// record(inner = record(a = u, b = t), c = w)`) now also refuses, with the
// nullity claim that is false of it. Every such shape already refused for the
// ≥2-latent pushforward or marginalisation reason, so no number regresses — only
// the wording moves, and it moves onto clm's pre-existing wrong message (the
// factually-false-reason gap tracked in flatppl-dev/TODO-flatppl-js.md).
//
// What is NOT widened is the component CLASSIFIER. An INLINE field expression
// lifts to an untyped internal binding that `_namesADraw` reads as a constructor,
// so it reports no roots, and a Hall size-3 shape whose outer sibling is inline
// (`c = y + 0.0`) still scores. Widening that would drag in the Lebesgue ⊗
// Counting exemptions — `floor(y)`/`round(y)`/a boolean beside `y`, whose laws §06
// "Reference measure for product measures" gives a real density — and telling
// those they have no density contradicts the very section they rest on. That is
// the cost that makes the classifier the wrong thing to widen here.
function _noiseRoots(name: string, ctx: any, seen?: Set<string>): Set<string> {
  const out = new Set<string>();
  const visited = seen || new Set<string>();
  const { root } = _aliasChain(name, ctx);
  if (visited.has(root)) return out;
  visited.add(root);
  const d = ctx.derivations && ctx.derivations[root];
  if (d && d.kind === 'evaluate') {
    for (const ref of orchestrator.collectSelfRefs(d.ir)) {
      if (!_isStochastic(ctx.bindings && ctx.bindings.get ? ctx.bindings.get(ref) : null)) continue;
      for (const r of _noiseRoots(ref, ctx, visited)) out.add(r);
    }
    return out;
  }
  const children = _structuralChildren(d);
  if (children) {
    for (const child of children) {
      for (const r of _noiseRoots(child, ctx, visited)) out.add(r);
    }
    return out;
  }
  out.add(root);
  return out;
}

// The record/tuple derivation `input` names, following aliases (`L =
// lawof(record(…))` is an alias to the lifted record binding). Null when the
// input is not a named record/tuple measure.
function _jointComponents(input: any, ctx: any): Array<{ label: string; binding: string }> | null {
  if (typeof input !== 'string' || !ctx || !ctx.derivations) return null;
  const { root } = _aliasChain(input, ctx);
  const d = ctx.derivations[root];
  if (!d) return null;
  if (d.kind === 'record' && d.fields) {
    return Object.keys(d.fields).map((k) => ({ label: k, binding: d.fields[k] }));
  }
  if (d.kind === 'tuple' && Array.isArray(d.elems)) {
    return d.elems.map((e: string, i: number) => ({ label: '#' + (i + 1), binding: e }));
  }
  return null;
}

// A user-facing name for a noise root. A `~`-draw lowers to an anonymous
// sample binding with the user's name as a pure alias to it, so the root a
// draw-identity walk lands on is usually the internal name.
function _displayName(root: string, ctx: any): string {
  if (!root.startsWith('__anon') && !root.startsWith('%')) return root;
  const derivations = (ctx && ctx.derivations) || {};
  for (const nm of Object.keys(derivations)) {
    if (nm.startsWith('__anon') || nm.startsWith('%')) continue;
    if (derivations[nm].kind === 'alias' && _aliasChain(nm, ctx).root === root) return nm;
  }
  /* c8 ignore start -- unexercised: a genuinely anonymous intermediate no user
     binding aliases. Mirrors derivations.ts's _displayNameForRoot fallback; a
     `~`-draw always leaves the user's name as an alias to its anon sample. */
  return root;
  /* c8 ignore stop */
}

// The `iid` base measure `input` is a product of, or null when it is not an iid.
function _iidBase(input: any, ctx: any): string | null {
  if (typeof input !== 'string' || !ctx || !ctx.derivations) return null;
  const d = ctx.derivations[_aliasChain(input, ctx).root];
  return (d && d.kind === 'iid' && typeof d.from === 'string') ? d.from : null;
}

// Bound the walk the way the rest of this module does — `buildDerivations` has a
// real cycle detector, but an alias chain reaching here malformed must not hang.
const _SINGULAR_MAX_DEPTH = 64;

function _refuseIfSingular(input: any, ctx: any, depth?: number, via?: string[]): void {
  const d = depth || 0;
  if (d > _SINGULAR_MAX_DEPTH) return;
  const path = via || [];

  const base = _iidBase(input, ctx);
  if (base != null) {
    _refuseIfSingular(base, ctx, d + 1, path.concat(["the iid product's inner measure"]));
    return;
  }

  const comps = _jointComponents(input, ctx);
  if (!comps) return;
  // A component reports noise when it names a VARIATE (a draw or a transform of
  // one) or when it is a structural container of them. A CONSTRUCTOR measure
  // reports none — §06 "Joint composition": "A component contributes a fresh
  // coordinate" — which is what keeps `joint(a = q, b = q)` scoring.
  const noise = comps.map((c) => ((_aliasChain(c.binding, ctx).isDraw
    || _structuralChildren(ctx.derivations[_aliasChain(c.binding, ctx).root]))
    ? _noiseRoots(c.binding, ctx) : new Set<string>()));
  for (let i = 0; i < comps.length; i++) {
    for (let j = i + 1; j < comps.length; j++) {
      for (const r of noise[i]) {
        if (!noise[j].has(r)) continue;
        // `path` is outermost-first, so it is reversed for the message: the
        // reader wants the offending level named first, then its containers.
        const where = path.length
          ? path.slice().reverse().join(', inside ') + ': ' : '';
        const err: any = new Error('density: ' + where
          + "joint components '" + comps[i].label
          + "' and '" + comps[j].label + "' are reified laws of the same draw — they "
          + "share the ancestor '" + _displayName(r, ctx) + "' with no independent "
          + 'noise separating them, so the joint law has no density w.r.t. the '
          + 'product reference measure (it concentrates on a lower-dimensional '
          + 'subset) — refused per spec §06 "Singular joints". Sampling this '
          + 'joint stays well-defined.');
        err.code = 'CLM_SINGULAR_JOINT';
        throw err;
      }
    }
  }

  for (const c of comps) {
    _refuseIfSingular(c.binding, ctx, d + 1, path.concat(["component '" + c.label + "'"]));
  }
}

// The COORDINATE identity of each of the body's joint components, aligned with
// the body's component order — the key the linear-Gaussian recogniser builds its
// node table on. A field carries its own `source`; a positional body carries
// none, so the tuple derivation's element order supplies them. `null` for a
// component whose identity is not recoverable: the recogniser refuses a
// multi-component body rather than risk double-counting a shared draw.
//
// A REIFIED component (`lawof(x)`) has the coordinate of the draw it reifies, so
// a component and a marginalised ancestor that are the same draw collapse to one
// node. A CONSTRUCTOR component instead contributes a fresh coordinate (§06
// "Joint composition": "A component contributes a fresh coordinate"), so it gets
// a synthesized key. Handing back the constructor's own binding name would make
// `joint(a = q, b = q)` two components on ONE key, which the recogniser reads as
// the same draw and refuses as singular — the shared node is `q`'s parameter, not
// the coordinate.
function _componentKeys(body: any, input: any, ctx: any): Array<string | null> | null {
  const isJoint = body && body.kind === 'call' && body.op === 'joint';
  const fields = isJoint && Array.isArray(body.fields) ? body.fields : null;
  const args = isJoint && Array.isArray(body.args) ? body.args : null;
  if (!fields && !args) return null;                 // a scalar body — one component
  const comps = _jointComponents(input, ctx);
  const coordinate = (binding: any, i: number): string | null => {
    if (binding == null) return null;
    return _aliasChain(binding, ctx).isDraw ? binding : '%coord' + i;
  };
  if (fields) {
    const byLabel = new Map<string, string>();
    for (const c of (comps || [])) byLabel.set(c.label, c.binding);
    return fields.map((f: any, i: number) => coordinate(
      f.source != null ? f.source : byLabel.get(f.name), i));
  }
  if (!comps || comps.length !== args.length) return null;
  return comps.map((c, i) => coordinate(c.binding, i));
}

function _domKind(dom: any): string {
  if (!dom) return 'opaque';
  if (dom.kind === 'scalar') return 'scalar';
  if (dom.kind === 'array') return (dom.rank && dom.rank >= 2) ? 'matrix' : 'vector';
  if (dom.kind === 'record') return 'record';
  return 'opaque';
}

// Shape descriptor for an EXPLICIT boundary VALUE (a raw JS preset / selected
// kernel input, not a binding). A plain object that is not a Value (no
// shape/data fields) is a record param (the transport `pars` case); an array /
// Float64Array / Value is a vector; anything else a scalar. perAtom:false — the
// one value is broadcast across every atom.
function _describeValueShape(v: any): any {
  const base = { perAtom: false, repeatTile: 'value' as const, fields: null as any };
  if (v != null && typeof v === 'object'
      && !Array.isArray(v) && !(v.data instanceof Float64Array) && v.shape === undefined) {
    return Object.assign({}, base, { kind: 'record', fields: Object.keys(v) });
  }
  if (Array.isArray(v) || v instanceof Float64Array
      || (v && Array.isArray(v.shape) && v.data instanceof Float64Array)) {
    return Object.assign({}, base, { kind: 'vector' });
  }
  return Object.assign({}, base, { kind: 'scalar' });
}

// ── body construction ────────────────────────────────────────────────────
//
// (1)+(2) peel + expand-refs are done by expandMeasure. (3) computeClosureIR
// inlines derived value bindings to the boundary set. (4) buildMcMarginalForm
// rewrites a generative-composite tree into the canonical mcmarginal form.
function _buildBody(input: any, deriv: any, ctx: any, opts?: any): { body: any; boundarySet: Set<string>; mc: boolean } | null {
  const expanded = orchestrator.expandMeasure(input, ctx);
  if (!expanded) return null;

  // The boundary set: refs that are FED (not inlined). For bayesupdate the
  // kernel's parametric inputs (paramKwargs); for kchain/jointchain the
  // prior step variates; otherwise empty (a plain product measure inlines
  // its derived values down to its own leaf refs, which become inputs).
  const boundarySet = _boundarySet(deriv, ctx);
  // EXPLICIT boundaries / free inputs (the viewer kernel/profile path —
  // spec §04 functionof boundary substitution): the caller-named inputs
  // are the cut exactly like a derivation's parametric inputs — derived
  // value bindings between the body and these names must inline DOWN TO
  // them (H5/H3: sweeping/feeding `theta` through `a = 5*theta` must
  // reach the leaf), and the names themselves are fed, never inlined.
  if (opts && opts.boundaries) {
    for (const nm in opts.boundaries) {
      if (Object.prototype.hasOwnProperty.call(opts.boundaries, nm)) boundarySet.add(nm);
    }
  }
  if (opts && Array.isArray(opts.freeInputs)) {
    for (const nm of opts.freeInputs) boundarySet.add(nm);
  }

  // (3)+(4) Generative-composite → canonical mcmarginal form, OR closed-form
  // boundary inlining — mutually exclusive, matching the legacy
  // `densIR = mcForm || inlineBoundaryDerivations(...)`. The mc-recipe peels
  // and FOLDS the deterministic layers into the per-event recipe itself
  // ("mc-form last" in spirit — it internalises the closure), so it consumes
  // the EXPANDED body; only when it doesn't fire do we inline derived value
  // bindings transitively down to the boundary set (multi-hop b ← a ← theta,
  // the single IR-space closure — critique E). On every non-generative,
  // no-derived-param shape both are no-ops, so body === expandMeasure(name)
  // (the green-fixture snapshot).
  let body = expanded;
  let mc = false;
  try {
    const mcr = require('./mc-recipe.ts');
    const form = mcr.buildMcMarginalForm(expanded, ctx.bindings,
      (nm: any) => orchestrator.expandMeasure(nm, ctx));
    if (form) { body = form; mc = true; }
  } catch (_) { /* mc-recipe is best-effort here; density owns the hard path */ }
  if (!mc) {
    // Inline derived value bindings transitively down to the boundary set
    // (multi-hop b ← a ← theta) so a deterministic field-dependent param
    // (`Beta(phi*kappa, …)`) is expressed via the fed/observed atoms rather
    // than fed as an independent `shared` column materialised from the prior
    // (which decouples it from the point being scored — a silent wrong
    // number for `logdensityof(lawof(record(...)), point)` whose boundary
    // set is empty). With no derived value bindings this is a no-op, so
    // body === expandMeasure(name) for the green-fixture shapes.
    body = shared.inlineBoundaryDerivations(expanded, boundarySet, ctx);
  }

  return { body, boundarySet, mc };
}

// The fed-boundary names for a derivation, by kind.
function _boundarySet(deriv: any, ctx: any): Set<string> {
  const s = new Set<string>();
  if (!deriv) return s;
  if ((deriv.kind === 'bayesupdate' || deriv.kind === 'likelihood_density')
      && Array.isArray(deriv.paramKwargs)) {
    // likelihood_density (standalone pdf(κ(θ), obs), audit H2): the same
    // parametric-input boundary set as bayesupdate — derived value bindings
    // inline down to the kernel's inputs (H5/H3) — but the inputs are fed
    // EXPLICITLY from the given θ (opts.boundaries), not from a prior.
    for (const k of deriv.paramKwargs) s.add(k);
    // The kernel body's refs name the PARAMS (node names — `%local`
    // placeholders or, spec-shaped §11, plain `self` refs), which differ
    // from the kwarg names when the boundary renames (`functionof(e,
    // p = a)`). Both are the fed cut — inlining must stop at either
    // (feedInputs binds the columns under both via `localAlias`).
    if (Array.isArray(deriv.params)) for (const p of deriv.params) s.add(p);
  } else if (deriv.kind === 'jointchain' && Array.isArray(deriv.steps)) {
    // Prior step variates: the base var (+ its record fields) and every
    // non-final kernel step's variate are integration variables the kernel
    // bodies splat over. The 2-step kchain rewires its hole to the base
    // BINDING ref, so include that too.
    const base = deriv.steps[0];
    if (base) {
      if (base.ref != null) s.add(base.ref);
      s.add(base.var);
      if (Array.isArray(base.baseFields)) for (const f of base.baseFields) s.add(f);
    }
    for (let i = 1; i < deriv.steps.length; i++) s.add(deriv.steps[i].var);
  }
  return s;
}

// reduce: a marginalisation reduction lives on the node so consumers stop
// re-deriving it from isChain/naryKchain (and H10's samples[0] collapse goes
// away as a side effect of declaring it here). `over` names the prior measure
// integrated out (steps[0].ref for a 2-step kchain; the retained history for
// N-ary, reconstructed from steps by the consumer). bayesupdate is NOT a
// marginal reduce — it reweights prior atoms; its per-event MC marginal (when
// the likelihood is generative) rides mcmarginal inside body.
//
// This kchain/jointchain marginal is a MONTE-CARLO estimator: applyReduce
// averages the per-atom scores over the prior's SAMPLED atoms, which is the one
// place §06's `kchain` sentence applies literally ("an engine evaluates it in
// closed form, or by enumeration of a discrete latent, and otherwise reports a
// static error") and the engine does neither. Out of scope for the wave that
// made the shared-ancestor marginal exact; tracked as the third open MC density
// site in flatppl-dev/TODO-flatppl-js.md. `test/kchain-density-relabelled-prior.
// test.ts` pins it at 8000 prior atoms against a 0.1-nat tolerance, which is the
// tell: an exact density would not need either number.
function _reduce(deriv: any): any {
  if (deriv && deriv.kind === 'jointchain' && deriv.marginalize) {
    const base = Array.isArray(deriv.steps) ? deriv.steps[0] : null;
    return {
      kind: 'marginal', method: 'logsumexp-logN',
      over: (base && base.ref != null) ? base.ref : null,
    };
  }
  return null;
}

// Structural boundary inputs declared by the derivation kind, independent of
// whether they surface as body self-refs. The named record-kernel case threads
// its params as `%local` (so collectSelfRefs can't see them) yet the prior and
// its fields must still be declared so feedInputs materialises them and the
// marginal knows what to integrate over. ⊆ allows inputs ⊇ body self-refs, so
// declaring extras is sound.
function _structuralBoundaries(deriv: any): any[] {
  const out: any[] = [];
  if (!deriv) return out;
  const push = (name: any, from: any, field?: any) => {
    const src: any = { kind: 'boundary', from };
    if (field != null) src.field = field;
    out.push({ name, source: src });
  };
  if (deriv.kind === 'bayesupdate' && Array.isArray(deriv.paramKwargs)) {
    // Each kernel param is fed from the like-named prior record field. The
    // bayesupdate expansion also references the boundary under a `%local`
    // placeholder (deriv.params[i], e.g. `_pars_`); feedInputs binds the
    // per-atom value under BOTH names, so record the alias on the source.
    const plc: string[] = Array.isArray(deriv.params) ? deriv.params : [];
    for (let i = 0; i < deriv.paramKwargs.length; i++) {
      const k = deriv.paramKwargs[i];
      const src: any = { kind: 'boundary', from: deriv.from, field: k };
      if (plc[i] && plc[i] !== k) src.localAlias = plc[i];
      out.push({ name: k, source: src });
    }
  } else if (deriv.kind === 'jointchain' && Array.isArray(deriv.steps)) {
    const base = deriv.steps[0];
    if (base && base.ref != null) {
      push(base.ref, base.ref);                  // the prior measure itself
      if (Array.isArray(base.baseFields)) {
        for (const f of base.baseFields) push(f, base.ref, f);
      }
    }
    // Intermediate step variates are history columns; their precise `from`
    // (the materialised retained history) is reconstructed by the Phase-3
    // consumer — declare them against base.ref as a best-effort anchor.
    for (let i = 1; i < deriv.steps.length - 1; i++) {
      const v = deriv.steps[i].var;
      if (v != null) push(v, base && base.ref != null ? base.ref : null, v);
    }
  }
  return out;
}

// ── input enumeration + classification ────────────────────────────────────
//
// Every self-ref in body that is not a callable (resolved by name) becomes a
// declared input. The classification mirrors prepareDensityRefs (the function
// feedInputs replaces): fixed-phase → fixed; a fed boundary → boundary;
// otherwise a measure ref → shared (today's getMeasure path, now explicit).
function _enumerateInputs(body: any, deriv: any, boundarySet: Set<string>, ctx: any, opts?: any): { inputs: any[]; missing: string[] } {
  const bindings = ctx && ctx.bindings;
  const fixedValues = ctx && ctx.fixedValues;
  const inputs: any[] = [];
  const seen = new Set<string>();
  const missing: string[] = [];
  const priorFrom = _priorFrom(deriv);

  const add = (name: string, source: any) => {
    if (seen.has(name)) return;
    seen.add(name);
    inputs.push({ name, ns: 'self', source, shape: describeInputShape(source, name, ctx) });
  };

  // 0. Explicit boundaries + free inputs (the viewer kernel/profile plot —
  //    spec §04 functionof boundary inputs). The caller supplies VALUES for
  //    named kernel inputs (fed via the ONE feedInputs contract,
  //    source.kind:'explicit', replacing the legacy viewer bake)
  //    and names a FREE input — the profile sweep axis — declared but left
  //    UNFED so the worker varies it per grid point. Added FIRST so a
  //    same-named body self-ref is already `seen` and not re-classified as
  //    shared/missing.
  const explicit = opts && opts.boundaries;
  if (explicit) {
    for (const nm in explicit) {
      if (Object.prototype.hasOwnProperty.call(explicit, nm)) {
        add(nm, { kind: 'explicit', value: explicit[nm] });
      }
    }
  }
  if (opts && Array.isArray(opts.freeInputs)) {
    for (const nm of opts.freeInputs) add(nm, { kind: 'free' });
  }

  // 1. Declared structural boundaries (independent of body self-refs).
  //    A source's `localAlias` (the param node name when it differs from
  //    the kwarg name) is marked seen too: feedInputs binds the column
  //    under BOTH names, so a body ref to the alias is covered by the
  //    declared input — it must not re-classify as shared/missing.
  for (const sb of _structuralBoundaries(deriv)) {
    add(sb.name, sb.source);
    if (sb.source && sb.source.localAlias) seen.add(sb.source.localAlias);
  }

  // 2. Remaining body self-refs: classify fixed / shared (the explicit
  //    surviving getMeasure path) / boundary, mirroring prepareDensityRefs.
  orchestrator.collectSelfRefs(body).forEach((n: string) => {
    if (seen.has(n)) return;
    const b = bindings && bindings.get ? bindings.get(n) : null;
    // Callable refs (fn / functionof / kernelof / bijection) resolve by name
    // at dispatch — they are NOT fed inputs and are excluded from the ⊆ set.
    if (shared.isFunctionLikeBinding(b)) return;

    if (fixedValues && fixedValues.has(n)) {
      add(n, { kind: 'fixed', ref: n });
    } else if (boundarySet.has(n)) {
      add(n, { kind: 'boundary', from: priorFrom, field: n });
    } else if (b) {
      add(n, { kind: 'shared', ref: n });
    } else if (priorFrom != null) {
      // A synthetic prior-variate (s0 / __jc$j) the chain feeds.
      add(n, { kind: 'boundary', from: priorFrom, field: n });
    } else if (builtins.ALL_KNOWN.has(n)) {
      // A BUILT-IN name riding as a `self` ref (engine IR keeps builtins in
      // ref form — the §11 export divergence): `broadcast(Normal, …)` heads,
      // kernel constructors. Resolved BY NAME at dispatch (worker REGISTRY /
      // walker tables), never fed — excluded from the ⊆ set exactly like
      // callable bindings. Checked AFTER the bindings branch so a user
      // binding that shadows a builtin (spec §04) still classifies as
      // shared/boundary.
    } else {
      missing.push(n);     // names nothing we can feed — a real ⊆ gap.
    }
  });
  return { inputs, missing };
}

// The caller measure a derivation's boundaries are fed from.
function _priorFrom(deriv: any): string | null {
  if (!deriv) return null;
  if (deriv.kind === 'bayesupdate') return deriv.from || null;
  if (deriv.kind === 'jointchain' && Array.isArray(deriv.steps)) {
    const base = deriv.steps[0];
    return (base && base.ref != null) ? base.ref : null;
  }
  return null;
}

// ── the pass ──────────────────────────────────────────────────────────────

/**
 * Lower a measure (by binding name or by IR) to its canonical clm node.
 * `ctx` is the materialiser/derivation context — { derivations, bindings,
 * fixedValues }. `opts.derivation` overrides the derivation lookup (for a
 * by-IR call). Returns null when the input doesn't expand to a measure.
 */
function lowerMeasure(input: any, ctx: any, opts?: any): any {
  const deriv = (opts && opts.derivation)
    || (typeof input === 'string' && ctx && ctx.derivations ? ctx.derivations[input] : null);
  // Singular joints are refused BEFORE the body is built: inlining erases the
  // shared draw identity (two `lawof(y)` components both inline to a bare
  // `Normal(0, 1)` call, indistinguishable from independent draws), so the
  // check must read the derivation's component→binding map while it survives.
  _refuseIfSingular(input, ctx);
  const built = _buildBody(input, deriv, ctx, opts);
  if (!built) return null;
  const { body, boundarySet, mc } = built;
  let reduce = _reduce(deriv);
  const { inputs, missing } = _enumerateInputs(body, deriv, boundarySet, ctx, opts);

  // Marginalised stochastic ancestor (H8 — the kchain/lawof unification). A
  // standalone measure whose body references a STOCHASTIC binding that is not
  // a retained variate (e.g. `pp = lawof(obs)`, obs ~ Normal(theta,1),
  // theta ~ Normal(0,1)) is the marginal law: p(x) = ∫ p(x|theta) p(theta) dθ,
  // and so is a shared-ancestor joint. §06 "Equivalent record law" (verbatim,
  // flatppl-design 52df5de): "`joint(a = lawof(a), b = lawof(b))` is equivalent
  // to `lawof(record(a = a, b = b))`; the positional form is the corresponding
  // `cat` law", and §06 "Density of composed measures": "A `joint` with shared
  // ancestry reduces as its equivalent record law".
  //
  // HOW that marginal may be evaluated is NOT ruled. §06 at 52df5de states no
  // evaluation rule for the record law's marginal: the joint bullet stops at
  // "reduces as its equivalent record law", and the closed-form/enumeration/
  // static-error sentence one paragraph later has `kchain` as its subject
  // ("`kchain` marginalizes the intermediate variate, so its density is the
  // marginal integral … This is generally intractable; an engine evaluates it in
  // closed form, or by enumeration of a discrete latent, and otherwise reports a
  // static error"). That rule is a supporting ANALOGY here — the same marginal
  // integral reached through the equivalent record law — not the authority.
  // flatppl-design#72 would have made it general but was CLOSED UNMERGED, and the
  // owner's 2026-08-06 call leaves the method unruled for now
  // (flatppl-dev/decisions-log.md), superseding the earlier 2026-08-05
  // no-stochastic-estimate decision. An MC marginal would therefore be conformant
  // today, and three other density paths in this engine do estimate
  // (TODO-flatppl-js.md). This branch nonetheless answers exactly or refuses, as
  // its own engineering choice: the reduce declares WHICH exact device it used —
  // 'analytic-gaussian' or 'analytic-mixture' — and refuses when neither applies,
  // so a caller can tell a closed form from an estimate.
  //
  // bayesupdate is excluded by the `deriv.kind` guard below, and that guard is
  // load-bearing: matBayesupdate DOES route through matScore, so a non-null
  // marginal reduce on its node would make matScore answer analytically or throw
  // instead of returning the per-atom log-likelihoods it reweights with. The
  // protection is that its node carries reduce === null — `_reduce` returns a
  // marginal only for `jointchain`, and this branch declines bayesupdate — so
  // `_analyticMarginalReply` returns null and the worker scores as before. That
  // keeps bayesupdate on the pure structural sum §06 mandates ("`logdensityof`
  // reduces structurally to the densities of its operands, terminating at the
  // per-kernel primitive `builtin_logdensityof`").
  if (!reduce && (!deriv || deriv.kind !== 'bayesupdate') && _isMeasureNode(body)) {
    const latents = inputs.filter((i: any) => i.source.kind === 'shared'
      && _isStochastic(ctx.bindings && ctx.bindings.get(i.source.ref)));
    if (latents.length > 0) {
      const over = latents.map((i: any) => i.source.ref);
      // An ancestor that is itself one of the record's FIELDS is not
      // integrated out: the density walker threads that field's observed
      // value into the overlay by name/source (the hierarchical record law
      // p(a)·p(b|a)), making the per-atom scores constant and the reduction a
      // no-op. Only genuinely unexposed ancestors need a marginal.
      const exposed = new Set<string>();
      for (const f of (body.fields || [])) {
        exposed.add(f.name);
        if (f.source != null) exposed.add(f.source);
        if (f.threadAs != null) exposed.add(f.threadAs);
      }
      const marg = over.filter((nm: string) => !exposed.has(nm));
      // An mc-form body carries an `mcmarginal` recipe the worker integrates
      // in-batch (buildMcMarginalForm) — one of the THREE MC density sites this
      // wave leaves alone (the others: the bayesupdate/mc-generative routing
      // that consumes the same form, and `_reduce`'s own kchain/jointchain
      // marginal, which averages sampled prior atoms). All three are tracked in
      // flatppl-dev/TODO-flatppl-js.md; this branch keeps their reduction
      // untouched.
      if (marg.length === 0 || mc) {
        reduce = { kind: 'marginal', method: 'logsumexp-logN', over };
      } else {
        // Inputs the CALLER feeds instead of the model's own value: an explicit
        // boundary (the viewer's kernel/profile route) or the free profile axis.
        // The closed form reads constants out of bindings/fixedValues, so a
        // marginal that depends on one of these must refuse rather than answer
        // from the un-substituted value — the recogniser owns that check because
        // it is the one that knows which names the moments were built from.
        const substituted = new Set<string>();
        for (const inp of inputs) {
          if (inp.source.kind !== 'explicit' && inp.source.kind !== 'free') continue;
          substituted.add(inp.name);
          if (inp.source.localAlias) substituted.add(inp.source.localAlias);
        }
        const lg = require('./linear-gaussian.ts');
        const identity = {
          keyOf: (nm: string) => _aliasChain(nm, ctx).root,
          componentKeys: _componentKeys(body, input, ctx),
        };
        const g = lg.recogniseGaussianMarginal(body, marg, ctx, substituted, identity);
        reduce = g.refuse
          ? { kind: 'marginal', method: 'refuse', over, marginalize: marg, reason: g.refuse }
          : {
            kind: 'marginal',
            method: g.mixture ? 'analytic-mixture' : 'analytic-gaussian',
            over, marginalize: marg, gaussian: g,
          };
      }
    }
  }

  // ⊆ invariant (Phase 6: THROW). With prereq D landed, collectSelfRefs
  // descends .bijection, so a non-empty `missing` is a REAL gap: a body
  // self-ref no declared input covers. Feeding such a body would fall back to
  // module-graph getMeasure resolution of the like-named binding — exactly the
  // boundary-conflation class the audit (§3) traced through ~10 operators.
  // Loud HERE, at the lowering on the main thread (plan critique F — the
  // worker cannot distinguish a shared ref from an unfed boundary), where
  // every caller (materialiser pipeline, viewer plot routing) already catches
  // and degrades per-target.
  if (missing.length > 0) {
    const label = typeof input === 'string' ? input : (deriv && deriv.kind) || '<ir>';
    const err: any = new Error('clm.lowerMeasure(' + label + '): body self-refs not covered '
      + 'by declared inputs: ' + JSON.stringify(missing)
      + ' — the lowering is not self-contained (⊆ invariant, audit §3); '
      + 'feeding it would re-materialise like-named module bindings');
    // Structured marker: the cascade-prune (derivationRefsValid) treats the
    // ⊆ violation as the authoritative not-plottable verdict, while every
    // OTHER lowering throw falls back to the legacy walk — matched by code,
    // not by message text.
    err.code = 'CLM_SUBSET_VIOLATION';
    throw err;
  }

  const node: any = { kind: 'call', op: 'clm', body, inputs, reduce };
  if (mc) node.mc = true;          // body carries an mcmarginal recipe → MC opts
  // Explicit-boundary lowering (viewer kernel/profile plot): the boundary
  // VALUES are supplied by the caller and must be FED (matClm's feed path),
  // even when reduce is null — unlike a jointchain RETAIN, whose prior is
  // inlined into the body and needs no external feed. The flag forces the feed
  // path so the body's `%local`/`self` boundary refs resolve to the fed values.
  if (opts && (opts.boundaries || opts.freeInputs)) node.fed = true;

  // N-ary kchain marginal (engine-concepts §6 chain-associativity): the body is
  // the FINAL kernel only, with its hole rewired to a cat over the retained
  // history variates s0..s_{n-2} (`vector(ref s0, ref s1, …)`). Those variates
  // are NOT in `body`, so the sample side must reconstruct the retained
  // (n−1)-joint history and bind its columns (the density side does the same
  // via extraRefArrays). Expand a synthetic RETAIN history derivation through
  // the existing jointchain expander (no inline-logic duplication) and attach
  // its body IR; matClm materialises it (the dependent-threaded retain joint)
  // and binds s_i. 2-step marginals rewire their hole to base.ref directly and
  // need no history (matClm's boundary path covers them).
  const hist = _marginalHistoryBody(deriv, ctx);
  if (hist) node.marginalHistoryBody = hist;
  return node;
}

// Build the retained (n−1)-joint history body IR for an N-ary marginal kchain
// by expanding a synthetic RETAIN jointchain derivation through the existing
// expander. Mirrors matLogdensityof's `steps.slice(0, -1)` history. Returns
// null for non-chains, 2-step chains (base.ref suffices), labelled chains
// (matches expandMeasure's labelled-N-ary deferral), or when expansion fails.
function _marginalHistoryBody(deriv: any, ctx: any): any {
  if (!deriv || deriv.kind !== 'jointchain' || !deriv.marginalize) return null;
  const steps = deriv.steps;
  if (!Array.isArray(steps) || steps.length <= 2 || deriv.labels) return null;
  const histName = '__clmhist';
  const derivs = Object.assign({}, ctx.derivations, {
    [histName]: { kind: 'jointchain', marginalize: false, labels: null,
                  steps: steps.slice(0, -1) },
  });
  return orchestrator.expandMeasure(histName,
    { derivations: derivs, bindings: ctx.bindings });
}

// ── feedInputs: the ONE feeding contract (Phase 2) ────────────────────────
//
// Materialise each clm input and bind it into refArrays / fixedEnv exactly as
// the legacy matBayesupdate boundRefArrays loop + prepareDensityRefs did, so
// the result is byte-identical on the bayesupdate fixtures — but driven by the
// declared `inputs` instead of caller-specific knowledge, so sampler and
// density (and the viewer) feed identically. Reference-identity preserving:
// columns are bound BY REFERENCE (never `.slice()`d — a clone silently breaks
// propagateLogWeights' independence dedupe and reintroduces M4-class bugs).
//
// Per input source:
//   boundary — materialise `from` ONCE (getMeasure is cached); then per the
//              shape: 'record' (whole-record param) → measureToPerAtomRecords;
//              a present record field → measureToRefValue(parent.fields[field]);
//              a scalar parent → measureToRefValue(parent). Bound under the
//              input name AND its `localAlias` (the bayesupdate `%local`
//              placeholder), matching the old dual-key feed.
//   shared   — getMeasure(ref) → measureToRefValue (the explicit getMeasure path).
//   fixed    — fixedValues.get(ref) → fixedEnv (pushed via setEnv merge by the
//              caller / matScore).
// An EXPLICIT boundary value (a viewer preset / selected kernel input) → the
// refArray entry the body's `%local`/`self` ref resolves to. Atom-independent
// (one value for every atom): a record becomes the per-atom-record array
// get_field consumes (matching measureToPerAtomRecords — N identical refs, the
// preset is constant across atoms); a scalar/vector becomes a shape-tagged
// Value (rank-0 broadcasts; rank-1 is an atom-indep vector).
function _explicitToRefValue(value: any, shape: any, ctx: any): any {
  const N = (ctx && ctx.sampleCount) || 1;
  // A record param → the per-atom-record array get_field consumes (atom-
  // independent: the same preset record for every atom).
  if (shape && shape.kind === 'record') {
    return new Array(N).fill(value);
  }
  // Scalar / vector → BROADCAST the atom-independent constant to a per-atom
  // batched column ([N] or [N, k]). A refArray entry consumed as a distribution
  // parameter must be the per-atom shape measureToRefValue produces (a rank-0
  // Value is the FIXED-value/session-env contract, not the refArray one — the
  // worker's param path would read it as NaN).
  const v = valueLib.asValue(value);
  if (!v.shape || v.shape.length === 0) {
    return valueLib.batchedScalar(new Float64Array(N).fill(v.data[0]));
  }
  if (v.shape.length === 1) {
    const k = v.shape[0];
    const data = new Float64Array(N * k);
    for (let i = 0; i < N; i++) data.set(v.data, i * k);
    const out: any = { shape: [N, k], data };
    if (v.outerRank != null) out.outerRank = v.outerRank;
    return out;
  }
  return v;                              // higher-rank constant: pass through
}

function feedInputs(node: any, ctx: any): Promise<{ refArrays: any; fixedEnv: any }> {
  const refArrays: Record<string, any> = {};
  const fixedEnv: Record<string, any> = {};
  const byFrom = new Map<string, any[]>();
  const sharedRefs: string[] = [];

  // Referenced boundary inputs (C2, #73): a fieldless (positional-product)
  // parent variate has no per-field key to match a named boundary input
  // against, so bindOne's record/field branches below never fire for it —
  // only a whole-parent bind can feed it. That is unambiguous exactly when
  // the kernel body references a SINGLE boundary input (computed lazily,
  // reusing the same walk assertFedCoverage uses): gate on the REFERENCED
  // count, not the declared-input count, because a record-base prior
  // declares every field but the kernel may reference only some. Arity ≥2
  // is a later fix (#73) and is left to fall through to the existing throw.
  let _referencedBoundaryInputs: any[] | null = null;
  const referencedBoundaryInputs = (): any[] => {
    if (_referencedBoundaryInputs) return _referencedBoundaryInputs;
    const referenced = _referencedRefNames(node.body);
    const found = (node.inputs || []).filter((i: any) => {
      const s = i.source;
      return s && s.kind === 'boundary'
        && (referenced.has(i.name) || (s.localAlias && referenced.has(s.localAlias)));
    });
    _referencedBoundaryInputs = found;
    return found;
  };

  for (const inp of node.inputs || []) {
    const src = inp.source;
    if (!src) continue;
    if (src.kind === 'free') {
      continue;                       // the profile sweep axis — fed by the worker, not here
    } else if (src.kind === 'explicit') {
      // A viewer preset / selected kernel input: an atom-independent VALUE,
      // bound under the bare name (collectRefArrays merges _extraRefArrays by
      // name, so the body's `%local`/`self` ref resolves to it).
      refArrays[inp.name] = _explicitToRefValue(src.value, inp.shape, ctx);
    } else if (src.kind === 'fixed') {
      const nm = src.ref != null ? src.ref : inp.name;
      if (ctx.fixedValues && ctx.fixedValues.has(nm)) fixedEnv[nm] = ctx.fixedValues.get(nm);
    } else if (src.kind === 'shared') {
      sharedRefs.push(src.ref != null ? src.ref : inp.name);
    } else if (src.kind === 'boundary' && src.from != null) {
      if (!byFrom.has(src.from)) byFrom.set(src.from, []);
      byFrom.get(src.from)!.push(inp);
    }
  }

  const bindOne = (parent: any, inp: any) => {
    const src = inp.source;
    const name = inp.name;
    let val: any = null;
    if (inp.shape && inp.shape.kind === 'record' && parent && parent.fields) {
      // Whole-record param: each per-atom record object (field access resolves
      // per atom). e.g. transport `pars` = record(a,b,mu).
      val = shared.measureToPerAtomRecords(parent, name, 'feedInputs');
    } else if (parent && parent.fields && src.field != null
        && parent.fields[src.field] && parent.fields[src.field].samples) {
      val = shared.measureToRefValue(parent.fields[src.field], src.field, 'feedInputs');
    } else if (parent && parent.samples) {
      val = shared.measureToRefValue(parent, name, 'feedInputs');
    } else if (parent && !parent.fields && referencedBoundaryInputs().length === 1) {
      // Whole non-record parent variate feeding a kernel's lone referenced
      // boundary input — e.g. a positional `joint(...)` prior, which
      // materialises to a tuple (`.elems`, no top-level `.samples`/`.value`),
      // not the record (`.fields`) or scalar-ensemble (`.samples`) shapes the
      // branches above cover. measureToParamValue flattens `.value` /
      // composite-`.elems` / scalar-`.samples` alike into one atom-major
      // Value, so it's the general whole-variate bind for this case.
      val = shared.measureToParamValue(parent, name, 'feedInputs');
    } else {
      return;                         // nothing materialisable for this input
    }
    refArrays[name] = val;            // by reference — never clone
    if (src.localAlias && src.localAlias !== name) refArrays[src.localAlias] = val;
  };

  const froms = Array.from(byFrom.keys());
  return Promise.all(froms.map((f) => Promise.resolve(ctx.getMeasure(f))))
    .then((parents: any[]) => {
      for (let i = 0; i < froms.length; i++) {
        for (const inp of byFrom.get(froms[i])!) bindOne(parents[i], inp);
      }
      return Promise.all(sharedRefs.map((r) => Promise.resolve(ctx.getMeasure(r))));
    })
    .then((measures: any[]) => {
      for (let i = 0; i < sharedRefs.length; i++) {
        refArrays[sharedRefs[i]] =
          shared.measureToRefValue(measures[i], sharedRefs[i], 'feedInputs');
      }
      return { refArrays, fixedEnv };
    });
}

// ── assertFedCoverage: the Phase-6 unfed-boundary throw (critique F) ─────
//
// matScore sends `body` + `refArrays` straight to the worker, which cannot
// distinguish a shared ref from an unfed boundary — so an unfed boundary
// there either dies with a cryptic "unbound self reference" or, worse,
// silently resolves from the CUMULATIVE WORKER SESSION ENV (the audit-H4
// ambient-state leak: crash-vs-wrong depended on materialisation order).
// This main-thread check closes that hole: every declared boundary /
// explicit input the body actually REFERENCES must have a fed column after
// the caller's overlay merge. Declared-but-unreferenced inputs are vacuous
// (no ref ⇒ no conflation) — a record-base prior declares every field but a
// kernel may consume only some. References are collected over BOTH
// namespaces (`self` and `%local`) because the `%local` param refs are
// exactly the ones collectSelfRefs — and hence the ⊆ check — cannot see.
function _referencedRefNames(body: any): Set<string> {
  const { walkIR } = require('./ir-walk.ts');
  const out = new Set<string>();
  walkIR(body, (n: any) => {
    if (n && n.kind === 'ref' && (n.ns === 'self' || n.ns === '%local')) out.add(n.name);
  });
  return out;
}

function assertFedCoverage(node: any, refArrays: any, where: string): void {
  if (!node || !Array.isArray(node.inputs)) return;
  let referenced: Set<string> | null = null;   // lazy — most paths fully fed
  for (const inp of node.inputs) {
    const src = inp.source;
    if (!src) continue;
    // shared/fixed resolve via getMeasure / session env; free is the worker-
    // fed profile sweep axis. Only fed-by-the-caller kinds can gap.
    if (src.kind !== 'boundary' && src.kind !== 'explicit') continue;
    if (refArrays && refArrays[inp.name] != null) continue;
    if (src.localAlias && refArrays && refArrays[src.localAlias] != null) continue;
    if (!referenced) referenced = _referencedRefNames(node.body);
    if (!referenced.has(inp.name)
        && !(src.localAlias && referenced.has(src.localAlias))) continue;
    throw new Error(where + ": declared boundary input '" + inp.name + "' is "
      + 'referenced by the lowered body but no fed column covers it — '
      + 'boundary inputs are fed by the caller (spec §04), never resolved '
      + 'from the worker session env or a like-named module binding '
      + '(audit §3/H4); this is a feed gap in the enclosing materialiser');
  }
}

module.exports = {
  lowerMeasure, feedInputs, describeInputShape, assertFedCoverage,
};

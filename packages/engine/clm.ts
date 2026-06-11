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
//   reduce  — null for a plain product measure, or {kind:'marginal',
//             method:'logsumexp-logN'} for kchain / lawof-of-record (H8) /
//             mc-generative — replacing the scattered isChain/naryKchain/
//             mcForm booleans.
//
// Structural invariant the pass asserts: collectSelfRefs(body) ⊆
// {i.name for i in inputs} (callable refs excluded — they resolve by name,
// not by feeding), descending into .bijection (ir-walk prereq D / H4).
//
// Phases 2–4 LANDED: feedInputs + the consumers (matScore, the sampler/density
// clm branches) are live — clm is the production jointchain/kchain path, not
// test-only. The ⊆ check still LOGS, it does not throw (the throw is the
// main-thread Phase-6 flip, gated on Phase 5b — see flatppl-dev/TODO-flatppl-js.md).
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
function _buildBody(input: any, deriv: any, ctx: any): { body: any; boundarySet: Set<string>; mc: boolean } | null {
  const expanded = orchestrator.expandMeasure(input, ctx);
  if (!expanded) return null;

  // The boundary set: refs that are FED (not inlined). For bayesupdate the
  // kernel's parametric inputs (paramKwargs); for kchain/jointchain the
  // prior step variates; otherwise empty (a plain product measure inlines
  // its derived values down to its own leaf refs, which become inputs).
  const boundarySet = _boundarySet(deriv, ctx);

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
  if (!mc && boundarySet.size > 0) {
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
  //    source.kind:'explicit', replacing the viewer's substituteLocals bake)
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
  const built = _buildBody(input, deriv, ctx);
  if (!built) return null;
  const { body, boundarySet, mc } = built;
  let reduce = _reduce(deriv);
  const { inputs, missing } = _enumerateInputs(body, deriv, boundarySet, ctx, opts);

  // Marginalised stochastic ancestor (H8 — the kchain/lawof unification). A
  // standalone measure whose body references a STOCHASTIC binding that is not
  // a retained variate (e.g. `pp = lawof(obs)`, obs ~ Normal(theta,1),
  // theta ~ Normal(0,1)) is the marginal law: p(x) = ∫ p(x|theta) p(theta) dθ.
  // The ancestor is already fed from its own law (the `shared` getMeasure path
  // yields theta ~ prior atoms), so scoring per atom then logsumexp − logN is
  // the MC marginal — the SAME reduction kchain uses, reached structurally
  // instead of via a derivation flag. Sampling draws obs ancestrally and is
  // already the marginal, so this makes density agree with the histogram
  // (the conditional Normal(·,1) vs the marginal Normal(0,√2), audit H8).
  // bayesupdate is excluded — it reweights prior atoms, not logsumexp, and
  // ignores `reduce` anyway (matBayesupdate owns its reduction).
  if (!reduce && (!deriv || deriv.kind !== 'bayesupdate') && _isMeasureNode(body)) {
    const latents = inputs.filter((i: any) => i.source.kind === 'shared'
      && _isStochastic(ctx.bindings && ctx.bindings.get(i.source.ref)));
    if (latents.length > 0) {
      reduce = {
        kind: 'marginal', method: 'logsumexp-logN',
        over: latents.map((i: any) => i.source.ref),
      };
    }
  }

  // ⊆ invariant (Phase 1: LOG, do not throw). With prereq D landed,
  // collectSelfRefs descends .bijection, so a non-empty `missing` is a REAL
  // gap (an undeclared body ref) rather than the former vacuous pass.
  if (missing.length > 0) {
    const label = typeof input === 'string' ? input : (deriv && deriv.kind) || '<ir>';
    // eslint-disable-next-line no-console
    console.warn('[clm] lowerMeasure(' + label + '): body self-refs not covered '
      + 'by declared inputs: ' + JSON.stringify(missing)
      + ' — ⊆ invariant gap (Phase 1 advisory).');
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

module.exports = {
  lowerMeasure, feedInputs, describeInputShape,
};

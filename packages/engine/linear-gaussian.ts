'use strict';

// ════════════════════════════════════════════════════════════════════════
// Linear-Gaussian marginal recognition — the ANALYTIC closed form for a
// shared-ancestor joint law's density (spec §06 "Reified components share
// their ancestry" + "Density of composed measures": "A `joint` with shared
// ancestry reduces as its equivalent record law").
//
// `lawof(record(a = a, b = b))` — equivalently `joint(a = lawof(a), b =
// lawof(b))` — over draws that share a stochastic ancestor is the MARGINAL
// law of the component vector, ∫ p(a, b | z) p(z) dz. §06 "Density of
// composed measures" gives an engine exactly two ways to evaluate such a
// marginal — "This is generally intractable; an engine evaluates it in closed
// form, or by enumeration of a discrete latent, and otherwise reports a static
// error." A Monte-Carlo estimate is not one of them, so refusing is spec
// CONFORMANCE here, not a local policy choice.
//
// When every node of the shared sub-DAG is Normal with an AFFINE location
// in its ancestors and a CONSTANT scale, the marginal is exactly
// multivariate normal and this module returns its (mean, cov). Anything
// else returns a refusal string — the caller throws it. There is no
// approximate branch here on purpose: a shape this module cannot prove
// linear-Gaussian is refused, not estimated.
//
// The moment propagation is the textbook one, over the latents in
// topological order: for a node x_i = c_i + Σ_j b_ij x_j + ε_i with
// ε_i ~ Normal(0, σ_i²) independent of every ancestor,
//     mean_i    = c_i + Σ_j b_ij mean_j
//     cov[i][k] = Σ_j b_ij cov[j][k]        (k already placed, k ≠ i)
//     cov[i][i] = Σ_j b_ij cov[i][j] + σ_i²
// and each component y = c + Σ_j b_j x_j + ε likewise, giving
//     cov[y][y'] = Σ_j Σ_l b_j b'_l cov_L[j][l] + δ_{yy'} σ_y².
// Pinned against Distributions.jl in test/joint-shared-ancestor-density.test.ts.
// ════════════════════════════════════════════════════════════════════════

const orchestrator = require('./orchestrator.ts');

// An affine form over the marginalised latents: value = c + Σ coefs[name]·name.
type Affine = { c: number; coefs: Map<string, number> };

const ZERO = (): Affine => ({ c: 0, coefs: new Map() });

function _addAffine(x: Affine, y: Affine, sign: number): Affine {
  const out: Affine = { c: x.c + sign * y.c, coefs: new Map(x.coefs) };
  y.coefs.forEach((v, k) => out.coefs.set(k, (out.coefs.get(k) || 0) + sign * v));
  return out;
}

function _scaleAffine(x: Affine, k: number): Affine {
  const out: Affine = { c: x.c * k, coefs: new Map() };
  x.coefs.forEach((v, n) => out.coefs.set(n, v * k));
  return out;
}

const _isConst = (a: Affine): boolean => a.coefs.size === 0;

// A numeric constant an IR expression resolves to, or null. A ref to a
// STOCHASTIC binding is never a constant — refuse rather than let
// resolveIRToValue walk into its law and evaluate a distribution call.
function _constant(ir: any, ctx: any): number | null {
  if (_refsStochastic(ir, ctx)) return null;
  let v: any;
  try { v = orchestrator.resolveIRToValue(ir, ctx.bindings, ctx.fixedValues); }
  catch (_) { return null; }
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function _refsStochastic(ir: any, ctx: any): boolean {
  const { walkIR } = require('./ir-walk.ts');
  let found = false;
  walkIR(ir, (n: any) => {
    if (found || !n || n.kind !== 'ref' || n.ns !== 'self') return;
    const b = ctx.bindings && ctx.bindings.get ? ctx.bindings.get(n.name) : null;
    if (b && b.phase === 'stochastic') found = true;
  });
  return found;
}

// Parse `ir` as an affine form over `latents`. Returns null when the
// expression is not affine in them (a product of two latents, a nonlinear
// op, a ref to a stochastic name that is not a marginalised latent).
function _affine(ir: any, latents: Set<string>, ctx: any): Affine | null {
  if (!ir || typeof ir !== 'object') return null;
  if (ir.kind === 'lit') {
    return typeof ir.value === 'number' && Number.isFinite(ir.value)
      ? { c: ir.value, coefs: new Map() } : null;
  }
  if (ir.kind === 'ref') {
    if (ir.ns === 'self' && latents.has(ir.name)) {
      return { c: 0, coefs: new Map([[ir.name, 1]]) };
    }
    /* c8 ignore start -- unexercised: a fixed-phase ref the fold left in place.
       Every fixed binding in the tested models reaches _affine already folded
       to a `lit`, so this is the capability rather than the common path;
       dropping it would refuse an otherwise-analytic marginal. */
    const k = _constant(ir, ctx);
    return k == null ? null : { c: k, coefs: new Map() };
    /* c8 ignore stop */
  }
  if (ir.kind !== 'call' || !Array.isArray(ir.args)) return null;
  const parts: Affine[] = [];
  for (const a of ir.args) {
    const p = _affine(a, latents, ctx);
    if (!p) return null;
    parts.push(p);
  }
  switch (ir.op) {
    case 'add':
      return parts.length >= 1
        ? parts.reduce((acc, p) => _addAffine(acc, p, 1), ZERO()) : null;
    case 'sub':
      return parts.length === 2 ? _addAffine(parts[0], parts[1], -1) : null;
    case 'neg':
      return parts.length === 1 ? _scaleAffine(parts[0], -1) : null;
    case 'mul': {
      if (parts.length !== 2) return null;
      if (_isConst(parts[1])) return _scaleAffine(parts[0], parts[1].c);
      if (_isConst(parts[0])) return _scaleAffine(parts[1], parts[0].c);
      return null;                       // latent × latent — not affine
    }
    case 'divide': {
      if (parts.length !== 2 || !_isConst(parts[1]) || parts[1].c === 0) return null;
      return _scaleAffine(parts[0], 1 / parts[1].c);
    }
    default:
      return null;
  }
}

// A distribution call's parameter IR by declared name, accepting both the
// keyword and the positional spelling. The name→position mapping comes from
// the sampler registry (the ONE catalogue), mirroring density.ts's broadcast
// parameter resolution rather than restating the order here.
function _distParamIR(ir: any, param: string): any {
  const samplerLib = require('./sampler.ts');
  // Throws for an unknown op; callers reach here only for a known distribution.
  const entry = samplerLib.lookupDistribution(ir);
  const kwargs = ir.kwargs || {};
  if (Object.prototype.hasOwnProperty.call(kwargs, param)) return kwargs[param];
  /* c8 ignore start -- unexercised: the registry keyword-alias path. `Normal`,
     the only distribution this module recognises, declares no aliases; the
     lookup stays generic so adding one cannot silently read the wrong
     keyword. */
  const alias = (entry.aliases || {})[param];
  if (alias && Object.prototype.hasOwnProperty.call(kwargs, alias)) return kwargs[alias];
  /* c8 ignore stop */
  if (Object.keys(kwargs).length > 0) return null;
  const idx = entry.params.indexOf(param);
  const args = Array.isArray(ir.args) ? ir.args : [];
  return idx >= 0 && idx < args.length ? args[idx] : null;
}

// A Normal node's (location affine, scale) pair, or a refusal reason.
function _normalNode(ir: any, latents: Set<string>, ctx: any, what: string):
{ loc: Affine; sd: number } | { refuse: string } {
  if (!ir || ir.kind !== 'call' || ir.op !== 'Normal') {
    return { refuse: what + ' is ' + _describeOp(ir) + ', not a Normal — the '
      + 'shared-ancestor marginal is analytic only for a linear-Gaussian '
      + 'sub-DAG' };
  }
  const loc = _affine(_distParamIR(ir, 'mu'), latents, ctx);
  if (!loc) {
    return { refuse: what + "'s location is not an affine function of the "
      + 'marginalised latents with constant coefficients' };
  }
  const sd = _constant(_distParamIR(ir, 'sigma'), ctx);
  if (sd == null || !(sd > 0)) {
    return { refuse: what + "'s scale is not a positive constant (a latent in "
      + 'the scale is not a linear-Gaussian shape)' };
  }
  return { loc, sd };
}

function _describeOp(ir: any): string {
  if (ir && ir.kind === 'call' && typeof ir.op === 'string') return "a '" + ir.op + "'";
  /* c8 ignore start -- unexercised: a component that is not a call at all.
     clm's _isMeasureNode gate keeps the BODY a call and every joint field in
     the tested models is one; kept so the refusal still reads sanely. */
  return 'not a distribution call';
  /* c8 ignore stop */
}

// The components of a clm body: the joint's fields (named) / args
// (positional), or the body itself as a single unnamed component.
function _components(body: any): { labels: string[] | null; irs: any[] } {
  if (body && body.kind === 'call' && body.op === 'joint') {
    if (Array.isArray(body.fields) && body.fields.length > 0) {
      return { labels: body.fields.map((f: any) => f.name), irs: body.fields.map((f: any) => f.value) };
    }
    if (Array.isArray(body.args) && body.args.length > 0) {
      return { labels: null, irs: body.args };
    }
  }
  return { labels: null, irs: [body] };
}

/**
 * Recognise the marginal of a shared-ancestor joint law as a multivariate
 * normal, or refuse.
 *
 * `body` is a clm body (a Normal call, or a `joint` over Normal calls);
 * `marginalize` names the stochastic ancestors integrated out. Returns
 * `{ labels, mean, cov }` — labels null for a positional/scalar variate —
 * or `{ refuse: <reason> }`.
 */
function recogniseGaussianMarginal(body: any, marginalize: string[], ctx: any): any {
  const latents = new Set(marginalize);
  // Each latent's own law, plus its affine dependence on the other latents
  // (a hierarchy z2 ~ Normal(z1, s) is linear-Gaussian too).
  const laws = new Map<string, { loc: Affine; sd: number }>();
  for (const nm of marginalize) {
    let law: any = null;
    try { law = orchestrator.expandMeasure(nm, ctx); } catch (_) { /* refused below */ }
    if (!law) return { refuse: "the marginalised ancestor '" + nm + "' has no expandable law" };
    const node = _normalNode(law, latents, ctx, "the marginalised ancestor '" + nm + "'");
    if ((node as any).refuse) return node;
    laws.set(nm, node as { loc: Affine; sd: number });
  }

  // Topological order over the latent-to-latent affine dependencies.
  const order: string[] = [];
  const state = new Map<string, number>();      // 1 in progress, 2 done
  const visit = (nm: string): string | null => {
    if (state.get(nm) === 2) return null;
    /* c8 ignore start -- defensive: a cyclic latent graph. A binding cycle
       (including a mutual stochastic one) is dropped with a diagnostic long
       before any derivation is buildable
       (self-referential-derivation.test.ts), so this turns an unreachable
       infinite recursion into a refusal. */
    if (state.get(nm) === 1) {
      return "the marginalised ancestors' dependency graph is cyclic through '" + nm + "'";
    }
    /* c8 ignore stop */
    state.set(nm, 1);
    const coefs = (laws.get(nm) as any).loc.coefs as Map<string, number>;
    for (const p of coefs.keys()) {
      const bad = visit(p);
      if (bad) return bad;
    }
    state.set(nm, 2);
    order.push(nm);
    return null;
  };
  for (const nm of marginalize) {
    const bad = visit(nm);
    if (bad) return { refuse: bad };
  }

  // Latent moments, parents first.
  const idx = new Map<string, number>();
  order.forEach((nm, i) => idx.set(nm, i));
  const n = order.length;
  const mean = new Array(n).fill(0);
  const cov: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const { loc, sd } = laws.get(order[i]) as { loc: Affine; sd: number };
    let m = loc.c;
    loc.coefs.forEach((b, p) => { m += b * mean[idx.get(p) as number]; });
    mean[i] = m;
    for (let k = 0; k < i; k++) {
      let c = 0;
      loc.coefs.forEach((b, p) => { c += b * cov[idx.get(p) as number][k]; });
      cov[i][k] = c;
      cov[k][i] = c;
    }
    let vi = sd * sd;
    loc.coefs.forEach((b, p) => { vi += b * cov[i][idx.get(p) as number]; });
    cov[i][i] = vi;
  }

  // Components: each an affine function of the latents plus its own noise.
  const { labels, irs } = _components(body);
  const comps: Array<{ loc: Affine; sd: number }> = [];
  for (let i = 0; i < irs.length; i++) {
    const what = labels ? "component '" + labels[i] + "'" : 'the component measure';
    const node = _normalNode(irs[i], latents, ctx, what);
    if ((node as any).refuse) return node;
    comps.push(node as { loc: Affine; sd: number });
  }

  const k = comps.length;
  const cMean = new Array(k).fill(0);
  const cCov: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < k; i++) {
    let m = comps[i].loc.c;
    comps[i].loc.coefs.forEach((b, p) => { m += b * mean[idx.get(p) as number]; });
    cMean[i] = m;
  }
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      let c = i === j ? comps[i].sd * comps[i].sd : 0;
      comps[i].loc.coefs.forEach((bi, pi) => {
        comps[j].loc.coefs.forEach((bj, pj) => {
          c += bi * bj * cov[idx.get(pi) as number][idx.get(pj) as number];
        });
      });
      cCov[i][j] = c;
    }
  }
  // A component that does not depend on any marginalised latent still
  // belongs in the block (its row is diagonal) — nothing to check. A
  // component pair with |corr| = 1 would be a singular joint, which the
  // singularity gate refuses before this point.
  return { labels, mean: cMean, cov: cCov };
}

/**
 * Score a recognised Gaussian marginal at `observed`, reusing the engine's
 * canonical MvNormal / Normal closed forms (density-prims) rather than a
 * second implementation of the same maths.
 */
function scoreGaussianMarginal(g: any, observed: any): number {
  const k = g.mean.length;
  const x = _observedVector(observed, g.labels, k);
  const prims = require('./density-prims.ts');
  if (k === 1) {
    return prims.builtinLogdensityof(
      'Normal', { mu: g.mean[0], sigma: Math.sqrt(g.cov[0][0]) }, x[0]);
  }
  const flat = new Float64Array(k * k);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) flat[i * k + j] = g.cov[i][j];
  return prims.builtinLogdensityof('MvNormal', {
    mu: { shape: [k], data: new Float64Array(g.mean) },
    cov: { shape: [k, k], data: flat },
  }, { shape: [k], data: new Float64Array(x) });
}

// The observed variate as a plain number[] in component order.
function _observedVector(observed: any, labels: string[] | null, k: number): number[] {
  const out = new Array(k);
  if (labels) {
    if (observed == null || typeof observed !== 'object' || Array.isArray(observed)) {
      throw new Error('density: the shared-ancestor joint law has a record variate {'
        + labels.join(', ') + '}, but the point scored is not a record');
    }
    for (let i = 0; i < k; i++) {
      const v = observed[labels[i]];
      if (typeof v !== 'number' || !Number.isFinite(v)) {
        throw new Error("density: the point scored has no finite field '" + labels[i]
          + "' for the shared-ancestor joint law's variate");
      }
      out[i] = v;
    }
    return out;
  }
  const flat = _flattenNumbers(observed);
  if (flat.length !== k) {
    throw new Error('density: the shared-ancestor joint law has ' + k
      + ' components, but the point scored has ' + flat.length + ' element(s)');
  }
  return flat;
}

function _flattenNumbers(v: any): number[] {
  if (typeof v === 'number') return [v];
  if (Array.isArray(v)) {
    const out: number[] = [];
    for (const e of v) out.push(..._flattenNumbers(e));
    return out;
  }
  if (v && v.data && typeof v.data.length === 'number') return Array.from(v.data as any);
  throw new Error('density: cannot read the point scored as a numeric variate '
    + 'for the shared-ancestor joint law');
}

module.exports = { recogniseGaussianMarginal, scoreGaussianMarginal };

'use strict';

// ════════════════════════════════════════════════════════════════════════
// Linear-Gaussian marginal recognition — the ANALYTIC closed form for a
// shared-ancestor joint law's density, optionally mass-weighted over a
// FINITE DISCRETE latent's support. Spec anchors, verbatim from
// flatppl-design 52df5de:
//
//   §06 "Equivalent record law" — "`joint(a = lawof(a), b = lawof(b))` is
//   equivalent to `lawof(record(a = a, b = b))`; the positional form is the
//   corresponding `cat` law". (The traced-once statement lives in §04 "Trace of
//   the reified law", a different section.)
//
//   §06 "Density of composed measures" — "A `joint` with shared ancestry
//   reduces as its equivalent record law; a singular joint has no density and
//   the query is refused."
//
// `lawof(record(a = a, b = b))` over draws that share a stochastic ancestor is
// therefore the MARGINAL law of the component vector, ∫ p(a, b | z) p(z) dz.
// The same section states how such a marginal may be evaluated, for `kchain`:
// "This is generally intractable; an engine evaluates it in closed form, or by
// enumeration of a discrete latent, and otherwise reports a static error." That
// wording is scoped to `kchain` at 52df5de, and reaches this construct only as
// an ANALOGY — the same marginal integral, arrived at through the equivalent
// record law.
//
// Nothing NORMATIVE forces exact-or-refuse here. flatppl-design#72 (which would
// have made "closed form, enumeration, or static error" general) was closed
// unmerged, and the owner's call of 2026-08-06 leaves the evaluation method
// unruled for now: an engine may evaluate a density however it likes, so a
// Monte-Carlo marginal would be conformant today. Exact-or-refuse is this
// module's own choice, kept because a caller cannot otherwise tell a closed form
// from an estimate. See flatppl-dev/decisions-log.md 2026-08-06 and
// TODO-flatppl-js.md's three remaining MC density sites. Both exact devices the
// §06 sentence names are implemented:
//
//   closed form   — every node of the shared sub-DAG is Normal with an AFFINE
//                   location in its ancestors and a CONSTANT scale, so the
//                   marginal is exactly multivariate normal: returns
//                   { labels, mean, cov }.
//   enumeration   — some ancestors are finite discrete (Bernoulli, Categorical,
//                   Categorical0, Binomial) and the rest are linear-Gaussian
//                   given their atoms, so the marginal is a finite mixture:
//                   returns { labels, mixture: [{ logw, mean, cov }, …] }.
//                   Deterministic and exact — NOT Monte Carlo.
//
// Anything else returns a refusal string — the caller throws it. There is no
// approximate branch here on purpose.
//
// ── the linear-Gaussian sub-DAG ────────────────────────────────────────
//
// Recognition runs over ONE node table covering the joint's COMPONENTS and
// every stochastic ancestor they reach, closed transitively: a location that
// references a stochastic name adds that name's own law as a node. Composing
// affine maps through the intermediate nodes is what makes a hierarchical CHAIN
// analytic — `z ~ Normal(0,1); a ~ Normal(z,1); b ~ Normal(a,1)` scored as
// `joint(a = lawof(a), b = lawof(b))` is exactly MvNormal([0,0], [[2,2],[2,3]]),
// with `a` a component AND `b`'s parent. Propagating moments over the whole
// table (rather than substituting a sibling's location into `b`) is what keeps
// Cov(a, b) = Var(a) = 2 instead of dropping the noise `a` and `b` share.
//
// Nodes are keyed by DRAW IDENTITY (the caller's `identity.keyOf` — clm's alias
// root), not by name, so a component and a marginalised ancestor that are the
// SAME draw collapse to one node. Without that they double-count: the component
// contributes fresh noise while the ancestor is integrated out separately, which
// silently under-states the cross-covariance of exactly the chain shape above.
//
// The moment propagation is the textbook one, over the nodes in topological
// order: for a node x_i = c_i + Σ_j b_ij x_j + ε_i with ε_i ~ Normal(0, σ_i²)
// independent of every ancestor,
//     mean_i    = c_i + Σ_j b_ij mean_j
//     cov[i][k] = Σ_j b_ij cov[j][k]        (k already placed, k ≠ i)
//     cov[i][i] = Σ_j b_ij cov[i][j] + σ_i²
// and the component block is the sub-matrix of the component rows — the
// remaining nodes are integrated out by construction. Pinned against
// Distributions.jl / scipy in test/joint-shared-ancestor-density.test.ts.
//
// KNOWN GAP: a caller-substituted boundary (`opts.boundaries` / `opts.freeInputs`)
// refuses rather than threading the fed value into the moments — see
// `_reachesBlocked`. Tracked in flatppl-dev/TODO-flatppl-js.md.
// ════════════════════════════════════════════════════════════════════════

const orchestrator = require('./orchestrator.ts');

// The enumeration cap, mirroring flatppl-rust's determiniser
// (`crates/determinizer/src/marginal.rs`, `MAX_ATOMS`): an enumerated logsumexp
// must stay small, so a finite but large support is treated as non-enumerable.
// Applied to each latent's own support AND to the product over several latents,
// which is the number of Gaussian blocks the mixture carries.
const MAX_ATOMS = 256;

// The finite-discrete distributions this module enumerates, with their atom sets
// per spec §08: `Bernoulli` support `booleans` ⇒ {0,1}; `Categorical(p)` support
// `interval(1, n)` ⇒ {1,…,n} ("Categories are numbered starting from 1,
// consistent with FlatPPL's 1-based indexing convention"); `Categorical0(p)`
// support `interval(0, n-1)`; `Binomial(n, p)` support `interval(0, n)`.
const DISCRETE_OPS = new Set(['Bernoulli', 'Categorical', 'Categorical0', 'Binomial']);

// An affine form over the sub-DAG's nodes: value = c + Σ coefs[key]·node(key).
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

// A constant numeric VECTOR an IR resolves to (a `Categorical`'s `p`), or null.
function _constantVector(ir: any, ctx: any): number[] | null {
  if (!ir || _refsStochastic(ir, ctx)) return null;
  let v: any;
  try { v = orchestrator.resolveIRToValue(ir, ctx.bindings, ctx.fixedValues); }
  catch (_) { return null; }
  const arr = Array.isArray(v) ? v
    : (v && v.data && typeof v.data.length === 'number' ? Array.from(v.data as any) : null);
  if (!arr || arr.length === 0) return null;
  return arr.every((e: any) => typeof e === 'number' && Number.isFinite(e))
    ? (arr as number[]) : null;
}

function _isStochasticBinding(b: any): boolean {
  return !!(b && b.phase === 'stochastic');
}

function _stochasticName(name: string, ctx: any): boolean {
  const b = ctx.bindings && ctx.bindings.get ? ctx.bindings.get(name) : null;
  return _isStochasticBinding(b);
}

function _refsStochastic(ir: any, ctx: any): boolean {
  const { walkIR } = require('./ir-walk.ts');
  let found = false;
  walkIR(ir, (n: any) => {
    if (found || !n || n.kind !== 'ref' || n.ns !== 'self') return;
    if (_stochasticName(n.name, ctx)) found = true;
  });
  return found;
}

// The refs an IR mentions, in either namespace. `%local` counts: an explicit
// boundary is fed under both its own name and its `%local` param alias.
function _refNames(ir: any): string[] {
  const { walkIR } = require('./ir-walk.ts');
  const out: string[] = [];
  walkIR(ir, (n: any) => {
    if (n && n.kind === 'ref' && (n.ns === 'self' || n.ns === '%local')) out.push(n.name);
  });
  return out;
}

// The STOCHASTIC names an IR references directly, deduplicated. Drives the
// transitive closure of the node table.
function _stochasticRefs(ir: any, ctx: any): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const nm of _refNames(ir)) {
    if (seen.has(nm) || !_stochasticName(nm, ctx)) continue;
    seen.add(nm);
    out.push(nm);
  }
  return out;
}

// The first `blocked` name the given IRs reach, transitively through binding
// bodies, or null.
//
// This is the CALLER-SUBSTITUTED-VALUE guard. Every constant in the closed form
// is resolved from the model's own `bindings`/`fixedValues`, so a caller that
// substitutes a boundary — `lowerMeasure`'s `opts.boundaries` (the viewer's
// kernel/profile route, spec §04 functionof boundary substitution) or
// `opts.freeInputs` (the profile sweep axis the worker varies per grid point) —
// would get a number computed from the ORIGINAL value, silently ignoring the
// substitution. Refusing is the only sound answer available here: the previous
// MC reduce at least fed the boundary to the worker, so an exact-looking number
// that does not track the override would be a regression dressed as a fix.
function _reachesBlocked(irs: any[], blocked: Set<string>, ctx: any): string | null {
  if (blocked.size === 0) return null;
  const seen = new Set<string>();
  const stack = irs.slice();
  while (stack.length > 0) {
    const ir = stack.pop();
    if (!ir) continue;
    for (const ref of _refNames(ir)) {
      if (blocked.has(ref)) return ref;
      if (seen.has(ref)) continue;
      seen.add(ref);
      const b = ctx.bindings && ctx.bindings.get ? ctx.bindings.get(ref) : null;
      if (b && b.ir) stack.push(b.ir);
    }
  }
  return null;
}

// Parse `ir` as an affine form over the sub-DAG's nodes. `keyFor` maps a
// stochastic name to its node key; `atoms` pins the nodes an enumeration has
// fixed to a value, which enter as plain constants. Returns null when the
// expression is not affine (a product of two nodes, a nonlinear op, a ref to a
// stochastic name with no node).
function _affine(ir: any, keyFor: (nm: string) => string | null,
                 atoms: Map<string, number>, ctx: any): Affine | null {
  if (!ir || typeof ir !== 'object') return null;
  if (ir.kind === 'lit') {
    return typeof ir.value === 'number' && Number.isFinite(ir.value)
      ? { c: ir.value, coefs: new Map() } : null;
  }
  if (ir.kind === 'ref') {
    if (ir.ns === 'self' && _stochasticName(ir.name, ctx)) {
      const key = keyFor(ir.name);
      if (key == null) return null;
      const pinned = atoms.get(key);
      if (pinned !== undefined) return { c: pinned, coefs: new Map() };
      return { c: 0, coefs: new Map([[key, 1]]) };
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
    const p = _affine(a, keyFor, atoms, ctx);
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
      return null;                       // node × node — not affine
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
  /* c8 ignore start -- unexercised: the registry keyword-alias path. None of the
     distributions this module reads (`Normal`, `Bernoulli`, `Categorical`,
     `Categorical0`, `Binomial`) declares an alias; the lookup stays generic so
     adding one cannot silently read the wrong keyword. */
  const alias = (entry.aliases || {})[param];
  if (alias && Object.prototype.hasOwnProperty.call(kwargs, alias)) return kwargs[alias];
  /* c8 ignore stop */
  if (Object.keys(kwargs).length > 0) return null;
  const idx = entry.params.indexOf(param);
  const args = Array.isArray(ir.args) ? ir.args : [];
  return idx >= 0 && idx < args.length ? args[idx] : null;
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
function _components(body: any): { labels: string[] | null; irs: any[]; sources: any[] } {
  if (body && body.kind === 'call' && body.op === 'joint') {
    if (Array.isArray(body.fields) && body.fields.length > 0) {
      return {
        labels: body.fields.map((f: any) => f.name),
        irs: body.fields.map((f: any) => f.value),
        sources: body.fields.map((f: any) => (f.source != null ? f.source : null)),
      };
    }
    if (Array.isArray(body.args) && body.args.length > 0) {
      return { labels: null, irs: body.args, sources: body.args.map(() => null) };
    }
  }
  return { labels: null, irs: [body], sources: [null] };
}

// ── the node table ────────────────────────────────────────────────────────

// One node of the shared sub-DAG. `component` marks the observed coordinates;
// every other node is integrated out (Gaussian) or enumerated (discrete).
type TableNode = {
  key: string;
  what: string;              // how a refusal names this node
  ir: any;                   // the distribution call
  component: number | null;  // component index, or null
  atoms?: number[];          // discrete only
  logMass?: number[];        // discrete only
};

// Classify a distribution call as a finite discrete latent and return its atoms
// with their log-masses. Returns null when the call is not one of the four
// enumerable distributions, or `{ refuse }` when it is one but its support is
// not a statically-resolvable size within the cap.
function _discreteNode(ir: any, what: string, ctx: any): any {
  if (!ir || ir.kind !== 'call' || !DISCRETE_OPS.has(ir.op)) return null;
  const prims = require('./density-prims.ts');
  const tooBig = (n: number) => ({ refuse: what + ' has ' + n + ' atoms, above the '
    + 'enumeration cap of ' + MAX_ATOMS + ' — a finite but large discrete latent is '
    + 'treated as non-enumerable (the cap matches the determiniser\'s)' });
  let atoms: number[];
  let params: any;
  if (ir.op === 'Bernoulli') {
    const p = _constant(_distParamIR(ir, 'p'), ctx);
    if (p == null || !(p >= 0 && p <= 1)) {
      return { refuse: what + "'s `p` is not a constant in [0, 1], so its support "
        + 'cannot be enumerated' };
    }
    atoms = [0, 1];
    params = { p };
  } else if (ir.op === 'Binomial') {
    const n = _constant(_distParamIR(ir, 'n'), ctx);
    const p = _constant(_distParamIR(ir, 'p'), ctx);
    if (n == null || !Number.isInteger(n) || n < 1 || p == null || !(p >= 0 && p <= 1)) {
      return { refuse: what + "'s `n`/`p` are not constants with an integer `n` ≥ 1, "
        + 'so its support cannot be enumerated' };
    }
    if (n + 1 > MAX_ATOMS) return tooBig(n + 1);
    atoms = [];
    for (let k = 0; k <= n; k++) atoms.push(k);
    params = { n, p };
  } else {
    const p = _constantVector(_distParamIR(ir, 'p'), ctx);
    if (!p) {
      return { refuse: what + "'s `p` is not a constant probability vector, so its "
        + 'category count is not statically known' };
    }
    if (p.length > MAX_ATOMS) return tooBig(p.length);
    const base = ir.op === 'Categorical' ? 1 : 0;   // §08: Categorical is 1-based
    atoms = [];
    for (let i = 0; i < p.length; i++) atoms.push(base + i);
    params = { p };
  }
  const logMass = atoms.map((k) => prims.builtinLogdensityof(ir.op, params, k));
  return { atoms, logMass };
}

/**
 * Build the node table: the components plus every stochastic ancestor they
 * reach, closed transitively and keyed by draw identity. Returns
 * `{ nodes, compKeys }` or `{ refuse }`.
 */
function _buildTable(comps: any, marginalize: string[], keyOf: (nm: string) => string,
                     ctx: any): any {
  const nodes = new Map<string, TableNode>();
  const compKeys: string[] = [];

  // Components first, so a marginalised ancestor that IS one of them merges
  // into the component's node instead of being integrated out separately.
  for (let i = 0; i < comps.irs.length; i++) {
    const what = comps.labels ? "component '" + comps.labels[i] + "'" : 'the component measure';
    const src = comps.sources[i];
    // A component with no recoverable draw identity gets a private key. Safe for
    // a single component (nothing can reference it), but with siblings an
    // unidentified component could BE one of the ancestors — refuse rather than
    // risk double-counting it.
    const key = src != null ? keyOf(src) : '%component' + i;
    if (src == null && comps.irs.length > 1) {
      return { refuse: what + ' carries no draw identity, so the recogniser cannot '
        + 'tell whether it is one of the marginalised ancestors — refused rather '
        + 'than risk double-counting a shared draw' };
    }
    if (nodes.has(key)) {
      return { refuse: what + ' and ' + (nodes.get(key) as TableNode).what
        + ' are the same draw, so the joint law is singular' };
    }
    nodes.set(key, { key, what, ir: comps.irs[i], component: i });
    compKeys.push(key);
  }

  // The marginalised ancestors, plus the transitive closure over every node's
  // location — a chain's intermediate node is not in `marginalize` when it is a
  // component's parent only, and composing through it is what makes the chain
  // analytic. `keyOfName` records how each name resolved, for the affine parse.
  const keyOfName = new Map<string, string>();
  const pending: Array<{ name: string; what: string }> = marginalize.map((nm) => ({
    name: nm, what: "the marginalised ancestor '" + nm + "'",
  }));
  for (const n of nodes.values()) {
    for (const ref of _stochasticRefs(n.ir, ctx)) {
      pending.push({ name: ref, what: "the stochastic ancestor '" + ref + "'" });
    }
  }
  while (pending.length > 0) {
    const { name, what } = pending.shift() as { name: string; what: string };
    const key = keyOf(name);
    keyOfName.set(name, key);
    if (nodes.has(key)) continue;
    let law: any = null;
    try { law = orchestrator.expandMeasure(name, ctx); } catch (_) { /* refused below */ }
    if (!law) return { refuse: what + ' has no expandable law' };
    nodes.set(key, { key, what, ir: law, component: null });
    for (const ref of _stochasticRefs(law, ctx)) {
      pending.push({ name: ref, what: "the stochastic ancestor '" + ref + "'" });
    }
  }

  // Classify: Normal (Gaussian) / finite discrete (enumerable) / neither.
  const discrete: TableNode[] = [];
  for (const n of nodes.values()) {
    if (n.ir && n.ir.kind === 'call' && n.ir.op === 'Normal') continue;
    // An observed coordinate is never enumerated: its variate is the point
    // scored, w.r.t. the product reference measure the Gaussian block assumes.
    if (n.component == null) {
      const d = _discreteNode(n.ir, n.what, ctx);
      if (d && d.refuse) return d;
      if (d) {
        n.atoms = d.atoms;
        n.logMass = d.logMass;
        discrete.push(n);
        continue;
      }
    }
    return { refuse: n.what + ' is ' + _describeOp(n.ir) + ', not a Normal — the '
      + 'shared-ancestor marginal is analytic only for a linear-Gaussian sub-DAG, '
      + 'or a linear-Gaussian one conditioned on a finite discrete latent' };
  }

  const combos = discrete.reduce((acc, n) => acc * (n.atoms as number[]).length, 1);
  if (combos > MAX_ATOMS) {
    return { refuse: 'enumerating ' + discrete.length + ' discrete ancestors takes '
      + combos + ' atom combinations, above the cap of ' + MAX_ATOMS };
  }
  return { nodes, compKeys, discrete, keyOfName };
}

// ── moments ───────────────────────────────────────────────────────────────

// The component block's (mean, cov) for one pinned assignment of the discrete
// ancestors. Returns `{ refuse }` when a location or scale is not linear-Gaussian.
function _blockMoments(nodes: Map<string, TableNode>, compKeys: string[],
                       keyOfName: Map<string, string>, atoms: Map<string, number>,
                       ctx: any): any {
  // Every Gaussian node's (affine location, constant scale) under this pinning.
  // The null return is defensive: `_buildTable` registers a node for every
  // stochastic ref it finds in every node's IR, so a ref reaching the affine
  // parse always has one. Returning null makes an unregistered ref a refusal
  // rather than a coefficient on a key with no moments.
  const keyFor = (nm: string): string | null => {
    const k = keyOfName.get(nm);
    return k !== undefined && nodes.has(k) ? k : null;
  };
  const laws = new Map<string, { loc: Affine; sd: number }>();
  for (const n of nodes.values()) {
    if (atoms.has(n.key)) continue;                 // enumerated, not Gaussian
    const loc = _affine(_distParamIR(n.ir, 'mu'), keyFor, atoms, ctx);
    if (!loc) {
      return { refuse: n.what + "'s location is not an affine function of the "
        + 'sub-DAG\'s stochastic nodes with constant coefficients' };
    }
    const sd = _constant(_distParamIR(n.ir, 'sigma'), ctx);
    if (sd == null || !(sd > 0)) {
      return { refuse: n.what + "'s scale is not a positive constant (a latent in "
        + 'the scale is not a linear-Gaussian shape)' };
    }
    laws.set(n.key, { loc, sd });
  }

  // Topological order over the node-to-node affine dependencies.
  const order: string[] = [];
  const state = new Map<string, number>();          // 1 in progress, 2 done
  const visit = (key: string): string | null => {
    if (state.get(key) === 2) return null;
    /* c8 ignore start -- defensive: a cyclic node graph. A binding cycle
       (including a mutual stochastic one) is dropped with a diagnostic long
       before any derivation is buildable (self-referential-derivation.test.ts),
       so this turns an unreachable infinite recursion into a refusal. */
    if (state.get(key) === 1) {
      return "the sub-DAG's dependency graph is cyclic through "
        + (nodes.get(key) as TableNode).what;
    }
    /* c8 ignore stop */
    state.set(key, 1);
    for (const p of (laws.get(key) as { loc: Affine }).loc.coefs.keys()) {
      const bad = visit(p);
      if (bad) return bad;
    }
    state.set(key, 2);
    order.push(key);
    return null;
  };
  for (const key of laws.keys()) {
    const bad = visit(key);
    if (bad) return { refuse: bad };
  }

  // Node moments, parents first.
  const idx = new Map<string, number>();
  order.forEach((key, i) => idx.set(key, i));
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

  // The component block: the observed rows/columns. The remaining nodes are
  // integrated out by having contributed their moments to these.
  const k = compKeys.length;
  const rows = compKeys.map((key) => idx.get(key) as number);
  const cMean = rows.map((r) => mean[r]);
  const cCov: number[][] = Array.from({ length: k },
    (_v, i) => Array.from({ length: k }, (_w, j) => cov[rows[i]][rows[j]]));
  return { mean: cMean, cov: cCov };
}

/**
 * Recognise the marginal of a shared-ancestor joint law, or refuse.
 *
 * `body` is a clm body (a Normal call, or a `joint` over Normal calls);
 * `marginalize` names the stochastic ancestors integrated out. `substituted`
 * names the inputs the CALLER feeds instead of the model's own values (clm's
 * `explicit` / `free` input kinds); the closed form reads constants from the
 * model, so any dependence on one of those is refused rather than answered from
 * the un-substituted value.
 *
 * `identity` is REQUIRED and throws when absent. Its `keyOf` maps a binding name
 * to its draw identity (clm's alias root) so a component and an ancestor that are
 * the same draw collapse to one node. Defaulting it to the identity function
 * `nm => nm` would be name keying — precisely the defect that made
 * `joint(lawof(a), lawof(b))` over a hierarchical chain SCORE `[[2,1],[1,3]]`
 * instead of `[[2,2],[2,3]]`, silently, before this recogniser was rewritten. A
 * caller that cannot supply draw identities must get an error, never a number.
 *
 * Returns `{ labels, mean, cov }` for a pure linear-Gaussian sub-DAG,
 * `{ labels, mixture: [{ logw, mean, cov }, …] }` when a finite discrete
 * ancestor is enumerated, or `{ refuse: <reason> }`. `labels` is null for a
 * positional/scalar variate.
 */
function recogniseGaussianMarginal(
  body: any, marginalize: string[], ctx: any, substituted: Set<string> | undefined,
  identity: { keyOf: (nm: string) => string; componentKeys?: Array<string | null> | null },
): any {
  if (!identity || typeof identity.keyOf !== 'function') {
    throw new Error('linear-gaussian.recogniseGaussianMarginal: identity.keyOf is '
      + 'required — without draw identities a component and a marginalised ancestor '
      + 'that are the same draw are double-counted, which scores a wrong covariance '
      + 'instead of refusing');
  }
  const keyOf = identity.keyOf;
  const comps = _components(body);
  if (Array.isArray(identity.componentKeys)) {
    for (let i = 0; i < comps.sources.length; i++) {
      if (identity.componentKeys[i] != null) comps.sources[i] = identity.componentKeys[i];
    }
  }

  const table = _buildTable(comps, marginalize, keyOf, ctx);
  if (table.refuse) return table;
  const { nodes, compKeys, discrete, keyOfName } = table as {
    nodes: Map<string, TableNode>; compKeys: string[]; discrete: TableNode[];
    keyOfName: Map<string, string>;
  };

  // Caller-substituted inputs (see _reachesBlocked): checked over the body AND
  // every node's law, since a substituted value most often reaches the marginal
  // through an ancestor's parameters (`z ~ Normal(m, 1)` with `m` overridden)
  // rather than through a component directly.
  const irs: any[] = [body];
  nodes.forEach((n) => irs.push(n.ir));
  const hit = _reachesBlocked(irs, substituted || new Set<string>(), ctx);
  if (hit) {
    return { refuse: "the marginal depends on '" + hit + "', which the caller "
      + 'substitutes (an explicit boundary value or a free profile axis). The '
      + "closed form is built from the model's own constants, so it would ignore "
      + 'the substitution — refused rather than returning an exact-looking number '
      + 'that does not track the fed value' };
  }

  if (discrete.length === 0) {
    const b = _blockMoments(nodes, compKeys, keyOfName, new Map<string, number>(), ctx);
    if (b.refuse) return b;
    return { labels: comps.labels, mean: b.mean, cov: b.cov };
  }

  // Enumeration (§06's second permitted device): one Gaussian block per atom
  // combination, weighted by the atoms' joint mass. Deterministic and exact —
  // density(y) = Σ_k P(z = k) · density(y | z = k).
  const mixture: Array<{ logw: number; mean: number[]; cov: number[][] }> = [];
  const sizes = discrete.map((n) => (n.atoms as number[]).length);
  const total = sizes.reduce((a, b) => a * b, 1);
  for (let c = 0; c < total; c++) {
    const atoms = new Map<string, number>();
    let logw = 0;
    let rest = c;
    for (let d = 0; d < discrete.length; d++) {
      const j = rest % sizes[d];
      rest = (rest - j) / sizes[d];
      atoms.set(discrete[d].key, (discrete[d].atoms as number[])[j]);
      logw += (discrete[d].logMass as number[])[j];
    }
    if (logw === -Infinity) continue;               // a zero-mass atom drops out
    const b = _blockMoments(nodes, compKeys, keyOfName, atoms, ctx);
    if (b.refuse) return b;
    mixture.push({ logw, mean: b.mean, cov: b.cov });
  }
  return { labels: comps.labels, mixture };
}

/**
 * Score a recognised marginal at `observed`, reusing the engine's canonical
 * MvNormal / Normal closed forms (density-prims) rather than a second
 * implementation of the same maths. A mixture logsumexps its mass-weighted
 * blocks (§06's enumeration device).
 */
function scoreGaussianMarginal(g: any, observed: any): number {
  if (!g.mixture) return _scoreBlock(g.mean, g.cov, g.labels, observed);
  let max = -Infinity;
  const terms = g.mixture.map((b: any) => {
    const t = b.logw + _scoreBlock(b.mean, b.cov, g.labels, observed);
    if (t > max) max = t;
    return t;
  });
  /* c8 ignore start -- unexercised: every block underflowing to -∞ at once.
     Zero-mass atoms are dropped at recognition and each surviving block is a
     proper Gaussian, so a finite point always leaves one finite term. Guards
     the logsumexp shift against NaN if that ever stops holding. */
  if (max === -Infinity) return -Infinity;
  /* c8 ignore stop */
  let sum = 0;
  for (const t of terms) sum += Math.exp(t - max);
  return max + Math.log(sum);
}

function _scoreBlock(mean: number[], cov: number[][], labels: string[] | null,
                     observed: any): number {
  const k = mean.length;
  const x = _observedVector(observed, labels, k);
  const prims = require('./density-prims.ts');
  if (k === 1) {
    return prims.builtinLogdensityof(
      'Normal', { mu: mean[0], sigma: Math.sqrt(cov[0][0]) }, x[0]);
  }
  const flat = new Float64Array(k * k);
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) flat[i * k + j] = cov[i][j];
  return prims.builtinLogdensityof('MvNormal', {
    mu: { shape: [k], data: new Float64Array(mean) },
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

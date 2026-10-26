'use strict';
// piecewise-mass — a per-θ normalizer for a weight with DISCONTINUITIES, built
// as an IR expression in θ.
//
// crn-normalize (a fixed 128-point sample of a Lebesgue box) and leaf-mass-quad
// (a fixed graded rule over a probability leaf's quantile scale) both fix their
// nodes at rewrite time, before θ is known. A weight whose window moves with a
// latent — `1 + 100 · [x ∈ (c, c + w)]` with `c` a latent — then falls between
// nodes: at c = 0.5, w = 1e-4 over Lebesgue([0, 1]) the box estimator scored
// log-density 0 against −0.00995, and over Normal(0, 1) the leaf rule scored
// log φ(0.8) as if the window were absent. Neither can tell, so the number was
// silently wrong.
//
// THE RULE. Collect the weight's breakpoints as expressions in θ
// (weight-breakpoints.ts: affine tests of x, `abs(affine)`), clamp and sort
// them (a min/max network), and cut the support into pieces. The path is taken
// only when every occurrence of x sits inside one of those tests, so the weight
// is CONSTANT on each piece, and Z(θ) = Σ width · w(midpoint) is exact.
//
// WHAT IT REFUSES. A weight that also varies with x between its breakpoints
// (`ifelse(x < c, 1, 2) + exp(-x²)`): no rule with nodes fixed before θ is
// known is exact for it, since a feature narrower than the node spacing falls
// between nodes and every node count misses some width. A jump
// weight-breakpoints.ts cannot solve (`x*x < c`, `mod` of the variate) has no
// breakpoint to split at, a step of the variate has a θ-dependent number of
// them, and more than MAX_BREAKPOINTS would make the sorting network too large.
// All THROW rather than fall back to an estimator that cannot see the window.

const { mapIR } = require('./ir-walk.ts');

const MAX_BREAKPOINTS = 4;

const _lit = (v: number) => ({ kind: 'lit', value: v });
const _call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });

function mentions(n: any, param: string): boolean {
  if (!n || typeof n !== 'object') return false;
  if (Array.isArray(n)) return n.some((c) => mentions(c, param));
  if (n.kind === 'ref' && n.name === param) return true;
  for (const k in n) if (k !== 'loc' && k !== 'meta' && mentions(n[k], param)) return true;
  return false;
}

// The variate as the weight sees it: the bare parameter, or `v[1]` when the
// support makes it a one-component vector.
function isVariate(n: any, param: string, arrayVariate: boolean): boolean {
  if (!arrayVariate) return !!n && n.kind === 'ref' && n.name === param;
  return !!n && n.kind === 'call' && n.op === 'get' && Array.isArray(n.args) && n.args.length === 2
    && n.args[0]?.kind === 'ref' && n.args[0].name === param
    && n.args[1]?.kind === 'lit' && n.args[1].value === 1;
}

// Breakpoint expressions of a one-axis weight, or `refuse` naming the first
// jump of the variate that is not solved (weight-breakpoints.ts). `exprs` is
// empty for a weight with no jump, which the callers leave on their existing
// rule. A step (`floor` of the variate) refuses too: the number of its jumps
// depends on θ, and an IR expression has a fixed number of pieces.
function weightBreakpoints(body: any, param: string, arrayVariate: boolean):
  { exprs: any[]; refuse: string | null } {
  const { weightDiscontinuities } = require('./weight-breakpoints.ts');
  const d = weightDiscontinuities(body, (n: any) => isVariate(n, param, arrayVariate),
    (n: any) => mentions(n, param));
  let refuse = d.refuse;
  if (!refuse && d.steps.length > 0) refuse = '`' + d.steps[0].op + '` of the variate, whose number of steps depends on θ';
  if (!refuse && d.cuts.length > MAX_BREAKPOINTS) {
    refuse = d.cuts.length + ' breakpoints (more than ' + MAX_BREAKPOINTS + ')';
  }
  return { exprs: d.cuts, refuse };
}

// `exprs` clamped to [lo, hi] and sorted ascending. Two values sort with one
// min/max pair. More would nest min/max over min/max and grow the tree
// exponentially, so they use order statistics instead: the k-th smallest is the
// value whose rank (how many others are smaller, ties broken by position) is k,
// which costs O(K²) copies of the inputs.
function sortedClamped(exprs: any[], lo: any, hi: any): any[] {
  const c = exprs.map((e) => _call('max', lo, _call('min', hi, e)));
  if (c.length <= 1) return c;
  if (c.length === 2) return [_call('min', c[0], c[1]), _call('max', c[0], c[1])];
  const rank = (j: number) => balancedAdd(c.map((ci, i) => (i === j ? _lit(0)
    : _call('ifelse', i < j ? _call('le', ci, c[j]) : _call('lt', ci, c[j]), _lit(1), _lit(0)))));
  return c.map((_, k) => balancedAdd(c.map((cj, j) =>
    _call('ifelse', _call('equal', rank(j), _lit(k)), cj, _lit(0)))));
}

// Bind the weight's variate to an IR expression.
function bindVariate(body: any, param: string, x: any, arrayVariate: boolean): any {
  const value = arrayVariate ? _call('vector', x) : x;
  return mapIR(body, (n: any) => (n && n.kind === 'ref' && n.name === param ? value : n));
}

function balancedAdd(terms: any[]): any {
  let level = terms;
  while (level.length > 1) {
    const next: any[] = [];
    for (let i = 0; i < level.length; i += 2) {
      next.push(i + 1 < level.length ? _call('add', level[i], level[i + 1]) : level[i]);
    }
    level = next;
  }
  return level[0];
}

// Σ over pieces [cuts[i], cuts[i+1]] of width · f(midpoint). Exact when the
// integrand is constant on each piece. `cuts` are IR expressions.
function piecewiseConstant(cuts: any[], f: (t: any) => any): any {
  const terms: any[] = [];
  for (let i = 0; i + 1 < cuts.length; i++) {
    const width = _call('sub', cuts[i + 1], cuts[i]);
    const mid = _call('mul', _lit(0.5), _call('add', cuts[i], cuts[i + 1]));
    terms.push(_call('mul', width, f(mid)));
  }
  return balancedAdd(terms);
}

// Z(θ) = ∫_lo^hi w(x; θ) dx over a Lebesgue box of one axis.
function boxMassExpr(param: string, body: any, lo: number, hi: number, breakpoints: any[],
  arrayVariate: boolean): any {
  const cuts = [_lit(lo), ...sortedClamped(breakpoints, _lit(lo), _lit(hi)), _lit(hi)];
  return piecewiseConstant(cuts, (t: any) => bindVariate(body, param, t, arrayVariate));
}

// Z(θ) = ∫ w(x; θ) dB(x) = ∫₀¹ w(F⁻¹(u); θ) du for a probability leaf B, split
// at u = F(b(θ)) for each breakpoint b. F and F⁻¹ are the §07 transports, so the
// pieces move with θ at evaluation time.
function leafMassExpr(param: string, body: any, kernel: string, input: Record<string, number>,
  breakpoints: any[], logSpace: boolean): any {
  const record = {
    kind: 'call', op: 'record',
    fields: Object.keys(input).map((name) => ({ name, value: _lit(input[name]) })),
  };
  const kernelLit = { kind: 'lit', value: kernel };
  const toU = (x: any) => _call('builtin_touniform', kernelLit, record, x);
  const fromU = (u: any) => _call('builtin_fromuniform', kernelLit, record, u);
  const cuts = [_lit(0), ...sortedClamped(breakpoints.map(toU), _lit(0), _lit(1)), _lit(1)];
  return piecewiseConstant(cuts, (u: any) => {
    const w = bindVariate(body, param, fromU(u), false);
    return logSpace ? _call('exp', w) : w;
  });
}

// Where the weight varies with the variate outside its tests, as " at line L,
// column C", or null when every occurrence of the variate sits inside a
// comparison or an `in` (weightBreakpoints has solved each of those for its
// breakpoints, so the weight is constant between them).
function variesBetweenCuts(body: any, param: string, arrayVariate: boolean): string | null {
  let found: any = null;
  const visit = (n: any, near: any) => {
    if (found || !n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach((c) => visit(c, near)); return; }
    const here = n.loc ? n : near;
    if (n.kind === 'call' && (n.op === 'in' || ['lt', 'le', 'gt', 'ge'].includes(n.op))) return;
    if (isVariate(n, param, arrayVariate) || (n.kind === 'ref' && n.name === param)) { found = here || {}; return; }
    for (const k in n) if (k !== 'loc' && k !== 'meta') visit(n[k], here);
  };
  visit(body, null);
  if (!found) return null;
  const s = found.loc && found.loc.start;
  return s ? ' at line ' + (s.line + 1) + ', column ' + (s.col + 1) : '';
}

function variesError(base: string, at: string): Error {
  return new Error('normalize(weighted(w, ' + base + ')): the weight depends on a latent and varies '
    + 'with the variate' + at + ' between its breakpoints. An x-varying weight inside a θ-dependent '
    + 'window has no exact rule: nodes fixed before θ is known miss any feature narrower than their '
    + 'spacing. Z(θ) is unresolved (spec §06).');
}

// Does the weight jump in the variate anywhere? Over an N-D variate no jump
// can be split at, since the pieces would not be boxes.
function jumpsInVariate(body: any, param: string): boolean {
  const { weightDiscontinuities } = require('./weight-breakpoints.ts');
  const d = weightDiscontinuities(body, () => false, (n: any) => mentions(n, param));
  return !!d.refuse || d.cuts.length > 0 || d.steps.length > 0;
}

module.exports = {
  weightBreakpoints, variesBetweenCuts, variesError, jumpsInVariate, sortedClamped, boxMassExpr, leafMassExpr, MAX_BREAKPOINTS,
};

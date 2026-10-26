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
// THE RULE. Collect the weight's breakpoints as expressions in θ: the bounds of
// every `x in interval(a, b)` and the threshold of every comparison of `x` with
// a value. Emitted as clamped and sorted (a min/max network), they cut the
// support into pieces on which the weight is continuous, and each piece gets its
// own composite Gauss-Legendre rule mapped onto it at evaluation time. An
// indicator weight is then integrated exactly, and a smooth one to the rule's
// accuracy. The expression carries an embedded check — a coarser rule over the
// same pieces — and hands back +∞ when the two disagree, which the callers'
// `fromNormalize` mark turns into a loud refusal (§06: Z = ∞ leaves normalize
// undefined).
//
// WHAT IT REFUSES. A jump weight-breakpoints.ts cannot solve (`x*x < c`, `mod`
// of the variate) has no breakpoint to split at, a step of the variate has a
// θ-dependent number of them, and more than MAX_BREAKPOINTS would make the
// sorting network too large. All THROW rather than fall back to an estimator
// that cannot see the window.

const { mapIR } = require('./ir-walk.ts');

const MAX_BREAKPOINTS = 4;
const LEAF_GRADE_LEVELS = 40;
// Relative disagreement between the fine and the coarse rule above which the
// normalizer is unresolved.
const PIECE_GATE = 1e-6;

const _lit = (v: number) => ({ kind: 'lit', value: v });
const _call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });

// 5-point Gauss-Legendre on [0, 1].
const GL5 = [
  { x: 0.5 * (1 - 0.9061798459386640), w: 0.5 * 0.2369268850561891 },
  { x: 0.5 * (1 - 0.5384693101056831), w: 0.5 * 0.4786286704993665 },
  { x: 0.5, w: 0.5 * 0.5688888888888889 },
  { x: 0.5 * (1 + 0.5384693101056831), w: 0.5 * 0.4786286704993665 },
  { x: 0.5 * (1 + 0.9061798459386640), w: 0.5 * 0.2369268850561891 },
];

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

// A composite GL5 rule on [0, 1] with `cells` equal cells.
function uniformRule(cells: number): Array<{ x: number; w: number }> {
  const out: Array<{ x: number; w: number }> = [];
  for (let c = 0; c < cells; c++) for (const g of GL5) out.push({ x: (c + g.x) / cells, w: g.w / cells });
  return out;
}

// A composite GL5 rule on [0, 1] graded geometrically towards one end, `end`:
// cells [0, 2^-L], [2^-L, 2^-(L-1)], …, [1/2, 1], mirrored for end = 1. A tail
// piece [0, F(b)] of a leaf reaches the leaf's tail at that end only, and its
// cells then match those the two-sided 40-level rule had there.
function gradedRule(levels: number, end: 0 | 1): Array<{ x: number; w: number }> {
  const bps = [0];
  for (let k = levels; k >= 1; k--) bps.push(Math.pow(2, -k));
  bps.push(1);
  const out: Array<{ x: number; w: number }> = [];
  for (let i = 0; i + 1 < bps.length; i++) {
    const h = bps[i + 1] - bps[i];
    for (const g of GL5) {
      const x = bps[i] + g.x * h;
      out.push({ x: end === 0 ? x : 1 - x, w: g.w * h });
    }
  }
  return out;
}

// Σ over pieces [cuts[i], cuts[i+1]] of `rule(i)` mapped onto the piece. `cuts`
// are IR expressions; `integrand(t)` gives the integrand at the IR point t.
function piecewiseRule(cuts: any[], rule: (piece: number) => Array<{ x: number; w: number }>,
  integrand: (t: any) => any): any {
  const terms: any[] = [];
  for (let i = 0; i + 1 < cuts.length; i++) {
    const a = cuts[i], width = _call('sub', cuts[i + 1], a);
    for (const g of rule(i)) {
      const t = _call('add', a, _call('mul', _lit(g.x), width));
      terms.push(_call('mul', _call('mul', _lit(g.w), width), integrand(t)));
    }
  }
  return balancedAdd(terms);
}

// fine if the coarse rule agrees with it, else +∞.
function gated(fine: () => any, coarse: any): any {
  const ok = _call('le', _call('abs', _call('sub', fine(), coarse)),
    _call('mul', _lit(PIECE_GATE), _call('abs', fine())));
  return _call('ifelse', ok, fine(), _lit(Infinity));
}

// Z(θ) = ∫_lo^hi w(x; θ) dx over a Lebesgue box of one axis.
function boxMassExpr(param: string, body: any, lo: number, hi: number, breakpoints: any[],
  arrayVariate: boolean): any {
  const cuts = [_lit(lo), ...sortedClamped(breakpoints, _lit(lo), _lit(hi)), _lit(hi)];
  const f = (t: any) => bindVariate(body, param, t, arrayVariate);
  // 26 cells of 5 nodes put 130 nodes on every piece, no fewer than the
  // 128-point sample this replaces put on the whole box.
  return gated(() => piecewiseRule(cuts, () => uniformRule(26), f),
    piecewiseRule(cuts, () => uniformRule(13), f));
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
  const f = (u: any) => {
    const w = bindVariate(body, param, fromU(u), false);
    return logSpace ? _call('exp', w) : w;
  };
  // The two outer pieces reach the leaf's tails, so they take the graded rule.
  const last = cuts.length - 2;
  // The tail pieces grade as deep as the 40-level rule this replaces.
  const rule = (levels: number, cells: number) => (i: number) =>
    (i === 0 ? gradedRule(levels, 0) : i === last ? gradedRule(levels, 1) : uniformRule(cells));
  const fine = rule(LEAF_GRADE_LEVELS, 4);
  const coarse = rule(LEAF_GRADE_LEVELS / 2, 2);
  return gated(() => piecewiseRule(cuts, fine, f), piecewiseRule(cuts, coarse, f));
}

// Does the weight jump in the variate anywhere? Over an N-D variate no jump
// can be split at, since the pieces would not be boxes.
function jumpsInVariate(body: any, param: string): boolean {
  const { weightDiscontinuities } = require('./weight-breakpoints.ts');
  const d = weightDiscontinuities(body, () => false, (n: any) => mentions(n, param));
  return !!d.refuse || d.cuts.length > 0 || d.steps.length > 0;
}

module.exports = {
  weightBreakpoints, jumpsInVariate, sortedClamped, boxMassExpr, leafMassExpr, MAX_BREAKPOINTS, PIECE_GATE,
};

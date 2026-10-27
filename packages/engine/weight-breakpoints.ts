'use strict';
// weight-breakpoints — where a normalizer's weight w(x; θ) may jump in x.
//
// A quadrature rule resolves a jump only when the integral is split at it. A
// window narrower than the node spacing is otherwise never sampled, and the
// rule's own error estimate cannot see it: `1 + 100 · [abs(x - 0.5) < 5e-5]`
// over [0, 1] scored log-density 0 against −0.00995.
//
// The weight's IR is walked once. A test whose sides are affine in x gives one
// cut, `abs(affine)` against a value gives two, and floor, ceil or round of an
// affine argument gives a step at every integer (half-integer for round) it
// crosses. The cuts are IR expressions in θ, so a caller with θ fixed
// evaluates them and a caller building a per-θ expression embeds them.
//
// Any other way x reaches a jump REFUSES: a test on a non-affine expression
// (`x*x < c`), mod, integer division, a predicate, a density, or an op this
// file does not know to be continuous. Guessing a window there would score a
// silently wrong normalizer.

const _lit = (v: number) => ({ kind: 'lit', value: v });
const _call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });

// Ops continuous in every real argument (domain edges aside: a NaN region of
// `sqrt` or `log` is one jump, which an adaptive rule detects and bisects).
// `equal` and `unequal` jump on a set of measure zero, which no integral sees.
// A standard-module function (`target` instead of `op`) is one of the
// engine's closed-form special functions, and is continuous the same way.
const CONTINUOUS_OPS = new Set([
  'add', 'sub', 'mul', 'divide', 'neg', 'pow',
  'exp', 'log', 'log2', 'log10', 'log1p', 'expm1', 'sqrt', 'abs', 'abs2',
  'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
  'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
  'min', 'max', 'gamma', 'loggamma', 'logit', 'invlogit', 'probit', 'invprobit',
  'polynomial', 'bernstein', 'real',
  // Reductions over an array, and selecting one element.
  'sum', 'prod', 'mean', 'l1norm', 'l2norm', 'linfnorm', 'logsumexp', 'maximum', 'minimum', 'get',
  // Constructors and complex parts.
  'vector', 'tuple', 'record', 'cat', 'rowstack', 'colstack', 'transpose', 'adjoint',
  'complex', 'imag', 'conj', 'cis',
  'ifelse', 'land', 'lor', 'lnot', 'lxor', 'equal', 'unequal',
]);
const COMPARISONS = new Set(['lt', 'le', 'gt', 'ge']);
const STEP_OPS = new Set(['floor', 'ceil', 'round']);

interface Affine { a: any; b: any }
interface Step { op: string; a: any; b: any }
interface Discontinuities {
  // Points where w may jump, as IR expressions in θ.
  cuts: any[];
  // `op(a·x + b)` for op in floor/ceil/round: a jump at every integer
  // (half-integer for round) that a·x + b crosses.
  steps: Step[];
  // The first x-dependent jump that is not solved, as a message naming it.
  refuse: string | null;
}

const isLit = (n: any, v: number) => n && n.kind === 'lit' && n.value === v;

function add(p: any, q: any): any {
  if (isLit(p, 0)) return q;
  if (isLit(q, 0)) return p;
  return _call('add', p, q);
}
function sub(p: any, q: any): any {
  if (isLit(q, 0)) return p;
  return _call('sub', p, q);
}
function mul(p: any, q: any): any {
  if (isLit(p, 0) || isLit(q, 0)) return _lit(0);
  if (isLit(p, 1)) return q;
  if (isLit(q, 1)) return p;
  return _call('mul', p, q);
}

// x where a·x + b = c.
function root(f: Affine, c: any): any {
  return _call('divide', sub(c, f.b), f.a);
}

function where(n: any): string {
  const s = n && n.loc && n.loc.start;
  // loc is 0-based.
  return s ? ' at line ' + (s.line + 1) + ', column ' + (s.col + 1) : '';
}

// Is `f`, applied elementwise by `broadcast`, continuous in its arguments? A
// `functionof` (an operator `.+`, or `fn(...)`) is when its body has no jump
// in its parameters. A standard-module function is (see CONTINUOUS_OPS).
function continuousFunction(f: any): boolean {
  if (f && f.kind === 'ref') return f.ns !== 'self' && f.ns !== '%local';
  if (!f || f.kind !== 'call' || f.op !== 'functionof' || !Array.isArray(f.params)) return false;
  const params = new Set(f.params);
  const isParam = (n: any) => !!n && n.kind === 'ref' && params.has(n.name);
  const mentions = (n: any): boolean => {
    if (!n || typeof n !== 'object') return false;
    if (Array.isArray(n)) return n.some(mentions);
    if (isParam(n)) return true;
    for (const k in n) if (k !== 'loc' && k !== 'meta' && mentions(n[k])) return true;
    return false;
  };
  const d = weightDiscontinuities(f.body, () => false, mentions);
  return !d.refuse && d.cuts.length === 0 && d.steps.length === 0;
}

function weightDiscontinuities(body: any, isVariate: (n: any) => boolean,
  mentions: (n: any) => boolean): Discontinuities {
  const cuts: any[] = [];
  const steps: Step[] = [];
  let refuse: string | null = null;

  const affine = (n: any): Affine | null => {
    if (isVariate(n)) return { a: _lit(1), b: _lit(0) };
    if (!mentions(n)) return { a: _lit(0), b: n };
    if (n.kind !== 'call' || !Array.isArray(n.args)) return null;
    const [p, q] = n.args;
    if (n.op === 'neg' && n.args.length === 1) {
      const f = affine(p);
      return f && { a: mul(_lit(-1), f.a), b: mul(_lit(-1), f.b) };
    }
    if (n.args.length !== 2) return null;
    if (n.op === 'add' || n.op === 'sub') {
      const f = affine(p), g = affine(q);
      if (!f || !g) return null;
      return n.op === 'add' ? { a: add(f.a, g.a), b: add(f.b, g.b) } : { a: sub(f.a, g.a), b: sub(f.b, g.b) };
    }
    if (n.op === 'mul') {
      const [k, v] = mentions(p) ? [q, p] : [p, q];
      if (mentions(k)) return null;
      const f = affine(v);
      return f && { a: mul(k, f.a), b: mul(k, f.b) };
    }
    if (n.op === 'divide' && !mentions(q)) {
      const f = affine(p);
      return f && { a: _call('divide', f.a, q), b: _call('divide', f.b, q) };
    }
    return null;
  };

  // `l <op> r` jumps where l = r. `abs(f) <op> c` jumps where f = ±c.
  const test = (l: any, r: any): boolean => {
    const f = affine(l), g = affine(r);
    if (f && g) { cuts.push(root({ a: sub(f.a, g.a), b: sub(f.b, g.b) }, _lit(0))); return true; }
    for (const [side, other] of [[l, r], [r, l]]) {
      if (side && side.kind === 'call' && side.op === 'abs' && side.args?.length === 1 && !mentions(other)) {
        const h = affine(side.args[0]);
        if (h) { cuts.push(root(h, other), root(h, _call('neg', other))); return true; }
      }
    }
    return false;
  };

  const visit = (n: any) => {
    if (refuse || !n || typeof n !== 'object') return;
    if (Array.isArray(n)) { n.forEach(visit); return; }
    if (n.kind !== 'call' || isVariate(n) || !mentions(n)) return;
    if (COMPARISONS.has(n.op) && n.args?.length === 2) {
      if (!test(n.args[0], n.args[1])) refuse = 'a `' + n.op + '` test on an expression that is not affine in the variate' + where(n);
      return;
    }
    if (n.op === 'in' && n.args?.length === 2) {
      const [v, set] = n.args;
      // §03's named real intervals have the same finite cuts. Open/closed
      // endpoints remain owned by `in`; isolated points have zero Lebesgue mass.
      const bounds = set?.kind === 'const'
        ? set.name === 'posreals' || set.name === 'nonnegreals' ? [_lit(0)]
          : set.name === 'unitinterval' ? [_lit(0), _lit(1)] : null
        : set?.kind === 'call' && set.op === 'interval' && set.args?.length === 2 ? set.args : null;
      const ok = bounds && bounds.every((bound: any) => test(v, bound));
      if (!ok) refuse = 'an `in` test on an expression that is not affine in the variate, or on a set that is not an interval' + where(n);
      return;
    }
    if (STEP_OPS.has(n.op)) {
      const f = n.args?.length === 1 ? affine(n.args[0]) : null;
      if (!f) { refuse = '`' + n.op + '` of an expression that is not affine in the variate' + where(n); return; }
      steps.push({ op: n.op, a: f.a, b: f.b });
      return;
    }
    if (n.op === 'broadcast' && n.args?.length >= 2) {
      if (!continuousFunction(n.args[0])) {
        refuse = 'a `broadcast` of the variate through a function that may jump' + where(n);
        return;
      }
      n.args.slice(1).forEach(visit);
      return;
    }
    if (n.op !== undefined && !CONTINUOUS_OPS.has(n.op)) {
      refuse = '`' + n.op + '` of the variate, which may jump' + where(n);
      return;
    }
    for (const k in n) if (k !== 'loc' && k !== 'meta') visit(n[k]);
  };
  visit(body);
  return { cuts, steps, refuse };
}

module.exports = { weightDiscontinuities, CONTINUOUS_OPS };

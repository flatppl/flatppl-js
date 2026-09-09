'use strict';

// §06 reweighting retains the base reference. §08 discrete kernels therefore
// normalize by a support sum. Tail bounds below bound every remaining adjacent
// ratio, not just the last term. An anchor changes cost, never the answer.
const { builtinLogdensityof } = require('./density-prims.ts');

type Factor = { kernel: string; input: Record<string, any> };
type Support = {
  lo: number; hi: number;
  right?: (k: number) => number;
  left?: (k: number) => number;
};
const LOG_TOL = Math.log(1e-14);
const MAX_TERMS = 1000000;

function logadd(a: number, b: number): number {
  if (a === Infinity || b === Infinity) return Infinity;
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const hi = Math.max(a, b);
  return hi + Math.log1p(Math.exp(Math.min(a, b) - hi));
}

// Finite supports include degenerate parameter endpoints. Infinite supports
// carry log bounds for forward and reverse ratios over their entire tails.
function support(f: Factor): Support {
  const p = f.input;
  const check = (valid: boolean): void => {
    if (!valid) throw new Error('normalize density: invalid ' + f.kernel + ' parameters');
  };
  switch (f.kernel) {
    case 'Bernoulli':
      check(p.p >= 0 && p.p <= 1);
      return { lo: p.p === 1 ? 1 : 0, hi: p.p === 0 ? 0 : 1 };
    case 'Binomial':
      check(Number.isSafeInteger(p.n) && p.n > 0 && p.p >= 0 && p.p <= 1);
      return { lo: p.p === 1 ? p.n : 0, hi: p.p === 0 ? 0 : p.n };
    case 'Categorical':
    case 'Categorical0': {
      check(p.p.length > 0 && p.p.every((v: number) => v >= 0 && v <= 1)
        && Math.abs(p.p.reduce((s: number, v: number) => s + v, 0) - 1) <= 1e-12);
      const start = f.kernel === 'Categorical' ? 1 : 0;
      return { lo: start, hi: start + p.p.length - 1 };
    }
    case 'Poisson':
      check(p.rate >= 0);
      if (p.rate === 0) return { lo: 0, hi: 0 };
      return { lo: 0, hi: Infinity,
        right: k => Math.log(p.rate) - Math.log(k + 1),
        left: k => Math.log(k) - Math.log(p.rate) };
    case 'Geometric': {
      check(p.p > 0 && p.p <= 1);
      if (p.p === 1) return { lo: 0, hi: 0 };
      const q = Math.log1p(-p.p);
      return { lo: 0, hi: Infinity, right: () => q, left: () => -q };
    }
    case 'NegativeBinomial':
    case 'NegativeBinomial2': {
      const r = f.kernel === 'NegativeBinomial' ? p.alpha : p.psi;
      check(r > 0 && (f.kernel === 'NegativeBinomial' ? p.beta > 0 : p.mu > 0));
      const q = f.kernel === 'NegativeBinomial' ? -Math.log1p(p.beta) : -Math.log1p(p.psi / p.mu);
      return { lo: 0, hi: Infinity,
        right: k => q + Math.max(0, Math.log1p((r - 1) / (k + 1))),
        left: k => -q + Math.max(-Math.log(r), -Math.log1p((r - 1) / k)) };
    }
    // A continuous density may be a weight on a counting base. Normal ratios
    // decrease monotonically, so their boundary values bound the whole tail.
    case 'Normal':
      check(p.sigma > 0);
      return { lo: 0, hi: Infinity,
      right: k => (p.mu - k - 0.5) / (p.sigma * p.sigma),
      left: k => (k - p.mu - 0.5) / (p.sigma * p.sigma) };
    default: return { lo: 0, hi: Infinity };
  }
}

/** Log of the counting-reference product mass, with omitted tail mass ≤1e-14
 * of the sum. This certificate does not bound floating-point primitive error.
 * Finite intersections sum exactly. Infinite sums use two geometric bounds.
 * Invalid primitives retain their own parameter errors; zero mass is undefined.
 */
function discreteProductLogZ(factors: Factor[]): number {
  const supports = factors.map(support);
  const lo = Math.max(...supports.map(s => s.lo));
  const hi = Math.min(...supports.map(s => s.hi));
  const logterm = (k: number): number => {
    let sum = 0;
    for (const f of factors) {
      const lp = builtinLogdensityof(f.kernel, f.input, k);
      // A null atom stays null even if another density is infinite there.
      if (lp === -Infinity) return -Infinity;
      sum += lp;
    }
    return sum;
  };
  const finish = (sum: number): number => {
    if (!Number.isFinite(sum)) throw new Error('normalize density: product mass is zero or infinite');
    return sum;
  };
  if (!Number.isSafeInteger(lo) || (hi !== Infinity && !Number.isSafeInteger(hi))) {
    throw new Error('normalize density: discrete support exceeds safe integer indexing');
  }
  if (hi < lo) return finish(-Infinity);
  if (hi !== Infinity) {
    if (hi - lo >= MAX_TERMS) throw new Error('normalize density: discrete support sum exceeds the work limit');
    let sum = -Infinity;
    for (let k = lo; k <= hi; k++) sum = logadd(sum, logterm(k));
    return finish(sum);
  }
  if (supports.some(s => !s.right || !s.left)) {
    throw new Error('normalize density: this infinite-support product has no implemented tail bound');
  }
  // For a Poisson product the mode is near the geometric mean of its rates.
  // Else start at zero. No unimodality assumption enters either certificate.
  const anchor = factors.every(f => f.kernel === 'Poisson')
    ? Math.max(lo, Math.floor(Math.exp(factors.reduce((s, f) => s + Math.log(f.input.rate), 0) / factors.length)))
    : lo;
  if (!Number.isSafeInteger(anchor)) throw new Error('normalize density: product mode exceeds safe integer indexing');
  let a = anchor, b = anchor;
  let ta = logterm(a), tb = ta, sum = ta;
  const tail = (term: number, k: number, side: 'left' | 'right'): number => {
    const ratio = supports.reduce((s, bounds) => s + bounds[side]!(k), 0);
    return ratio < 0 ? term + ratio - Math.log(-Math.expm1(ratio)) : Infinity;
  };
  for (let count = 1; count < MAX_TERMS; count++) {
    const left = a === lo ? -Infinity : tail(ta, a, 'left');
    const right = tail(tb, b, 'right');
    if (Number.isFinite(sum) && logadd(left, right) <= sum + LOG_TOL) return finish(sum);
    if (left > right) { a--; ta = logterm(a); sum = logadd(sum, ta); }
    else {
      if (!Number.isSafeInteger(b + 1)) break;
      b++; tb = logterm(b); sum = logadd(sum, tb);
    }
  }
  throw new Error('normalize density: discrete product tail did not converge within the work limit');
}

module.exports = { discreteProductLogZ };

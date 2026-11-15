'use strict';

// §09 regression: keep the linear predictor in link scale and score one
// vector in a single row loop. Shared normalizers are computed once.
const V = require('./value.ts');
const lgamma = require('@stdlib/math-base-special-gammaln');
const betaln = require('@stdlib/math-base-special-betaln');

const PARAMS: Record<string, string[]> = {
  BernoulliLogitGLM: ['x', 'alpha', 'beta'],
  BinomialLogitGLM: ['x', 'n', 'alpha', 'beta'],
  CategoricalLogitGLM: ['x', 'alpha', 'beta'],
  NormalGLM: ['x', 'alpha', 'beta', 'sigma'],
  PoissonLogGLM: ['x', 'alpha', 'beta'],
  NegativeBinomial2LogGLM: ['x', 'alpha', 'beta', 'psi'],
  StudentTGLM: ['x', 'nu', 'alpha', 'beta', 'sigma'],
  GammaLogGLM: ['x', 'alpha', 'beta', 'shape'],
  OrderedLogitGLM: ['x', 'beta', 'cutpoints'],
  BetaLogitGLM: ['x', 'alpha', 'beta', 'precision'],
};
const NAMES = new Set(Object.keys(PARAMS));
const DISCRETE = new Set(['BernoulliLogitGLM', 'BinomialLogitGLM', 'CategoricalLogitGLM',
  'PoissonLogGLM', 'NegativeBinomial2LogGLM', 'OrderedLogitGLM']);

function signature(T: any, name: string): any {
  const vec = T.array(1, ['%dynamic'], T.REAL);
  const mat = T.array(2, ['%dynamic', '%dynamic'], T.REAL);
  const kwargs: Record<string, any> = {};
  for (const p of PARAMS[name]) kwargs[p] = p === 'x' ? mat
    : p === 'beta' ? (name === 'CategoricalLogitGLM' ? mat : vec)
    : p === 'cutpoints' || (p === 'alpha' && name === 'CategoricalLogitGLM') ? vec
    : p === 'n' ? T.INTEGER : T.REAL;
  return { args: null, kwargs, result: T.measure(T.array(1, ['%dynamic'],
    name === 'BernoulliLogitGLM' ? T.BOOLEAN : DISCRETE.has(name) ? T.INTEGER : T.REAL)) };
}

function bind(name: string, args: any[] = [], kwargs: any = {}): any {
  const names = PARAMS[name];
  if (args.length > names.length) throw new Error(`${name}: too many parameters`);
  const out = { ...kwargs };
  for (let i = 0; i < args.length; i++) {
    if (out[names[i]] !== undefined) throw new Error(`${name}: duplicate parameter '${names[i]}'`);
    out[names[i]] = args[i];
  }
  return out;
}

function array(v: any, rank: number, label: string): any {
  const a = V.asValue(v);
  if (a.shape.length !== rank || V.outerRankOf(a) !== rank || V.isComplexValue(a)) {
    throw new Error(`regression.${label}: expected a real rank-${rank} array`);
  }
  const d = V._logicalDense(a);
  for (let i = 0; i < d.data.length; i++) {
    if (!Number.isFinite(d.data[i])) throw new Error(`regression.${label}: entries must be finite`);
  }
  return d;
}

function scalar(v: any, label: string, positive = false, allowInfinity = false): number {
  if (V.isValue(v) && v.shape.length === 0 && !V.isComplexValue(v)) v = +v.data[0];
  if (typeof v !== 'number' || (!Number.isFinite(v) && !(allowInfinity && v === Infinity))
      || (positive && !(v > 0))) throw new Error(`regression.${label}: invalid parameter`);
  return v;
}

function softplus(x: number): number {
  return Math.max(0, x) + Math.log1p(Math.exp(-Math.abs(x)));
}
function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}
function logGammaFromLog(x: number): number {
  // Gamma(z)=Gamma(1+z)/z retains the log normalizer even when z underflows.
  return x < 0 ? lgamma(1 + Math.exp(x)) - x : lgamma(Math.exp(x));
}
function xlog(a: number, b: number): number { return a === 0 ? 0 : a * Math.log(b); }

function prepare(name: string, kw: any): any {
  const x = array(kw.x, 2, `${name} x`);
  const [rows, cols] = x.shape;
  if (rows < 1 || cols < 1) throw new Error(`${name}: x must have positive dimensions`);
  const categorical = name === 'CategoricalLogitGLM';
  const ordered = name === 'OrderedLogitGLM';
  const beta = array(kw.beta, categorical ? 2 : 1, `${name} beta`);
  if (beta.shape[0] !== cols) throw new Error(`${name}: x columns and beta length must agree`);
  const alpha = categorical ? array(kw.alpha, 1, `${name} alpha`)
    : ordered ? 0 : scalar(kw.alpha, `${name} alpha`);
  const classes = categorical ? beta.shape[1] : 0;
  if (categorical && (classes < 1 || alpha.shape[0] !== classes)) {
    throw new Error(`${name}: alpha length and beta classes must agree`);
  }
  const cut = ordered ? array(kw.cutpoints, 1, `${name} cutpoints`) : null;
  if (ordered && cut.shape[0] < 1) throw new Error(`${name}: at least one cutpoint is required`);
  if (ordered) for (let i = 1; i < cut.data.length; i++) {
    if (!(cut.data[i] > cut.data[i - 1])) throw new Error(`${name}: cutpoints must increase strictly`);
  }
  const n = name === 'BinomialLogitGLM' ? scalar(kw.n, `${name} n`, true) : 0;
  if (n && !Number.isInteger(n)) throw new Error(`${name}: n must be a positive integer`);
  const sigma = name === 'NormalGLM' || name === 'StudentTGLM'
    ? scalar(kw.sigma, `${name} sigma`, true) : 1;
  const nu = name === 'StudentTGLM' ? scalar(kw.nu, `${name} nu`, true, true) : Infinity;
  const psi = name === 'NegativeBinomial2LogGLM' ? scalar(kw.psi, `${name} psi`, true) : 1;
  const shape = name === 'GammaLogGLM' ? scalar(kw.shape, `${name} shape`, true) : 1;
  const precision = name === 'BetaLogitGLM' ? scalar(kw.precision, `${name} precision`, true) : 1;
  const logSigma = Math.log(sigma), logPsi = Math.log(psi), logShape = Math.log(shape);
  const normalizer = name === 'NormalGLM' || (name === 'StudentTGLM' && nu === Infinity)
    ? -0.5 * Math.log(2 * Math.PI) - logSigma
    : name === 'StudentTGLM' ? -Math.log(nu) / 2 - betaln(nu / 2, 0.5) - logSigma
    : name === 'BinomialLogitGLM' ? lgamma(n + 1)
    : name === 'NegativeBinomial2LogGLM' ? -lgamma(psi)
    : name === 'GammaLogGLM' ? shape * logShape - lgamma(shape)
    : name === 'BetaLogitGLM' ? lgamma(precision) : 0;
  const logits = new Float64Array(classes);

  function predictor(i: number): number {
    let eta = alpha;
    const base = i * cols;
    for (let j = 0; j < cols; j++) eta += x.data[base + j] * beta.data[j];
    return eta;
  }
  function categoryLogits(i: number): number {
    let max = -Infinity;
    for (let c = 0; c < classes; c++) {
      let z = alpha.data[c];
      for (let j = 0; j < cols; j++) z += x.data[i * cols + j] * beta.data[j * classes + c];
      logits[c] = z;
      if (z > max) max = z;
    }
    return max;
  }

  function score(value: any): number {
    const y = V._logicalDense(V.asValue(value));
    if (y.shape.length !== 1 || V.outerRankOf(y) !== 1 || y.shape[0] !== rows || V.isComplexValue(y)) {
      throw new Error(`${name}: response must be a length-${rows} vector`);
    }
    let total = 0;
    for (let i = 0; i < rows; i++) {
      const yi = +y.data[i];
      if (Number.isNaN(yi)) return NaN;
      if (!Number.isFinite(yi)) return -Infinity;
      if (DISCRETE.has(name) && !Number.isInteger(yi)) return -Infinity;
      if (categorical) {
        if (yi < 1 || yi > classes) return -Infinity;
        const max = categoryLogits(i);
        let sum = 0;
        for (let c = 0; c < classes; c++) sum += Math.exp(logits[c] - max);
        total += logits[yi - 1] - max - Math.log(sum);
        continue;
      }
      const eta = predictor(i);
      switch (name) {
        case 'BernoulliLogitGLM':
          if (yi !== 0 && yi !== 1) return -Infinity;
          total -= softplus(yi ? -eta : eta);
          break;
        case 'BinomialLogitGLM':
          if (yi < 0 || yi > n) return -Infinity;
          total += normalizer - lgamma(yi + 1) - lgamma(n - yi + 1)
            - (yi === 0 ? 0 : yi * softplus(-eta)) - (yi === n ? 0 : (n - yi) * softplus(eta));
          break;
        case 'NormalGLM': {
          const z = (yi - eta) / sigma;
          total += normalizer - z * z / 2;
          break;
        }
        case 'PoissonLogGLM':
          if (yi < 0) return -Infinity;
          total += (yi === 0 ? 0 : yi * eta) - Math.exp(eta) - lgamma(yi + 1);
          break;
        case 'NegativeBinomial2LogGLM': {
          if (yi < 0) return -Infinity;
          const q = eta - logPsi;
          total += lgamma(yi + psi) + normalizer - lgamma(yi + 1)
            - psi * softplus(q) - (yi === 0 ? 0 : yi * softplus(-q));
          break;
        }
        case 'StudentTGLM': {
          const z = (yi - eta) / sigma;
          total += normalizer - (nu === Infinity ? z * z / 2
            : (nu + 1) / 2 * softplus(2 * Math.log(Math.abs(z)) - Math.log(nu)));
          break;
        }
        case 'GammaLogGLM':
          if (yi < 0) return -Infinity;
          total += normalizer - shape * eta + xlog(shape - 1, yi)
            - (yi === 0 ? 0 : Math.exp(logShape + Math.log(yi) - eta));
          break;
        case 'OrderedLogitGLM': {
          const c = yi - 1, count = cut.data.length;
          if (c < 0 || c > count) return -Infinity;
          total += c === 0 ? -softplus(eta - cut.data[0])
            : c === count ? -softplus(cut.data[count - 1] - eta)
            : -softplus(eta - cut.data[c]) - softplus(cut.data[c - 1] - eta)
              + Math.log(-Math.expm1(cut.data[c - 1] - cut.data[c]));
          break;
        }
        case 'BetaLogitGLM': {
          if (yi < 0 || yi > 1) return -Infinity;
          const loga = Math.log(precision) - softplus(-eta);
          const logb = Math.log(precision) - softplus(eta);
          const a = Math.exp(loga), b = Math.exp(logb);
          total += normalizer - logGammaFromLog(loga) - logGammaFromLog(logb)
            + xlog(a - 1, yi) + (b === 1 ? 0 : (b - 1) * Math.log1p(-yi));
          break;
        }
      }
    }
    return total;
  }

  function sample(draw: (name: string, ...params: number[]) => number, uniform: () => number): any {
    const out = new Float64Array(rows);
    for (let i = 0; i < rows; i++) {
      if (categorical) {
        const max = categoryLogits(i);
        let sum = 0;
        for (let c = 0; c < classes; c++) sum += Math.exp(logits[c] - max);
        const target = uniform() * sum;
        let c = 0, acc = Math.exp(logits[0] - max);
        while (c < classes - 1 && target >= acc) acc += Math.exp(logits[++c] - max);
        out[i] = c + 1;
        continue;
      }
      const eta = predictor(i);
      switch (name) {
        case 'BernoulliLogitGLM': out[i] = draw('Bernoulli', sigmoid(eta)); break;
        case 'BinomialLogitGLM': out[i] = draw('Binomial', n, sigmoid(eta)); break;
        case 'NormalGLM': out[i] = draw('Normal', eta, sigma); break;
        case 'PoissonLogGLM': out[i] = draw('Poisson', Math.exp(eta)); break;
        case 'NegativeBinomial2LogGLM': out[i] = draw('NegativeBinomial2', Math.exp(eta), psi); break;
        case 'StudentTGLM': out[i] = nu === Infinity ? draw('Normal', eta, sigma) : eta + sigma * draw('StudentT', nu); break;
        case 'GammaLogGLM': out[i] = draw('Gamma', shape, shape * Math.exp(-eta)); break;
        case 'BetaLogitGLM': out[i] = draw('Beta', precision * sigmoid(eta), precision * sigmoid(-eta)); break;
        case 'OrderedLogitGLM': {
          const u = uniform(), latent = eta + Math.log(u) - Math.log1p(-u);
          let c = 0;
          while (c < cut.data.length && latent > cut.data[c]) c++;
          out[i] = c + 1;
          break;
        }
      }
    }
    return { shape: [rows], data: out };
  }
  return { rows, score, sample };
}

module.exports = { PARAMS, NAMES, DISCRETE, signature, bind, prepare };

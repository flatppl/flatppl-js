'use strict';

// Constrained <-> unconstrained bijections and their log-Jacobians, for the
// gradient-free (MH) and gradient (NUTS) samplers. Standard Stan transforms.
// Scalar transforms here; simplex (vector) is `simplexTransform`.

function transformFor(support: any) {
  switch (support.kind) {
    case 'real':
      return {
        unconstrain: (theta: number) => theta,
        constrain:   (y: number) => y,
        logDetJ:     (_y: number) => 0,
      };
    case 'positive':
      // y = log theta ; theta = exp y ; d theta/dy = exp y.
      // Floor theta at a tiny positive so unconstraining a value AT the
      // boundary (a prior draw of 0, or a proposal that lands there) stays
      // finite rather than -Infinity.
      return {
        unconstrain: (theta: number) => Math.log(theta > 1e-300 ? theta : 1e-300),
        constrain:   (y: number) => Math.exp(y),
        logDetJ:     (y: number) => y,
      };
    case 'interval': {
      const a = support.a, b = support.b, w = b - a;
      // theta = a + w*sigmoid(y) ; y = logit((theta-a)/w)
      // log|d theta/dy| = log w + log sigmoid(y) + log sigmoid(-y)
      const logSig = (x: number) => -Math.log1p(Math.exp(-x)); // log sigmoid(x), stable
      return {
        unconstrain: (theta: number) => {
          // Clamp the interior proportion away from {0,1} so a boundary value
          // unconstrains to a large-but-FINITE y (not ±Infinity → NaN logπ).
          let p = (theta - a) / w;
          if (!(p > 1e-12)) p = 1e-12;
          if (p > 1 - 1e-12) p = 1 - 1e-12;
          return Math.log(p / (1 - p));
        },
        constrain: (y: number) => a + w / (1 + Math.exp(-y)),
        logDetJ:   (y: number) => Math.log(w) + logSig(y) + logSig(-y),
      };
    }
    default:
      throw new Error(`transforms: no scalar transform for support kind '${support.kind}'`);
  }
}

function simplexTransform(K: number) {
  // Stan stick-breaking: K-simplex <-> R^{K-1}.
  const sig = (x: number) => 1 / (1 + Math.exp(-x));
  const logSig = (x: number) => -Math.log1p(Math.exp(-x));
  return {
    unconstrain(p: Float64Array): Float64Array {
      const y = new Float64Array(K - 1);
      let remaining = 1;
      for (let k = 0; k < K - 1; k++) {
        const zk = p[k] / remaining;             // breaking proportion in (0,1)
        y[k] = Math.log(zk / (1 - zk)) + Math.log(K - 1 - k); // offset so y=0 is uniform
        remaining -= p[k];
      }
      return y;
    },
    constrain(y: Float64Array): Float64Array {
      const p = new Float64Array(K);
      let remaining = 1;
      for (let k = 0; k < K - 1; k++) {
        const zk = sig(y[k] - Math.log(K - 1 - k));
        p[k] = remaining * zk;
        remaining -= p[k];
      }
      p[K - 1] = remaining;
      return p;
    },
    logDetJ(y: Float64Array): number {
      // sum over breaks: log zk + log(1-zk) + log(remaining before break k)
      let remaining = 1, acc = 0;
      for (let k = 0; k < K - 1; k++) {
        const x = y[k] - Math.log(K - 1 - k);
        const zk = sig(x);
        acc += logSig(x) + logSig(-x) + Math.log(remaining);
        remaining -= remaining * zk;
      }
      return acc;
    },
  };
}

const SUPPORT_BY_DIST: Record<string, (params?: any) => any> = {
  Normal:       () => ({ kind: 'real' }),
  Cauchy:       () => ({ kind: 'real' }),
  Logistic:     () => ({ kind: 'real' }),
  Laplace:      () => ({ kind: 'real' }),
  StudentT:     () => ({ kind: 'real' }),
  Exponential:  () => ({ kind: 'positive' }),
  Gamma:        () => ({ kind: 'positive' }),
  InverseGamma: () => ({ kind: 'positive' }),
  LogNormal:    () => ({ kind: 'positive' }),
  HalfCauchy:   () => ({ kind: 'positive' }),
  HalfNormal:   () => ({ kind: 'positive' }),
  Weibull:      () => ({ kind: 'positive' }),
  ChiSquared:   () => ({ kind: 'positive' }),
  Beta:         () => ({ kind: 'interval', a: 0, b: 1 }),
  // Uniform handled below (needs bounds from params).
};

function supportOf(distOp: string, params?: any) {
  if (distOp === 'Uniform') {
    const a = params && typeof params.a === 'number' ? params.a : 0;
    const b = params && typeof params.b === 'number' ? params.b : 1;
    return { kind: 'interval', a, b };
  }
  if (distOp === 'Dirichlet') return { kind: 'simplex' };
  const f = SUPPORT_BY_DIST[distOp];
  if (!f) throw new Error(`transforms: no continuous support known for distribution '${distOp}' (NUTS/MH-continuous cannot transform it)`);
  return f(params);
}

module.exports = { transformFor, simplexTransform, supportOf, SUPPORT_BY_DIST };

'use strict';
const T = require('./transforms.ts');

// Closed-form prior log-densities for the continuous dists we transform.
// (Mirrors density-prims; kept local so ModelView's prior term is independent
// of the observation-consuming density walker.)
const LN_2PI = Math.log(2 * Math.PI);
const PRIOR_LOGPDF: Record<string, (x: number, p: any) => number> = {
  Normal:      (x, p) => -0.5 * LN_2PI - Math.log(p.sigma) - 0.5 * ((x - p.mu) / p.sigma) ** 2,
  Exponential: (x, p) => Math.log(p.rate) - p.rate * x,
  HalfCauchy:  (x, p) => Math.log(2) - Math.log(Math.PI * p.scale) - Math.log1p((x / p.scale) ** 2),
  HalfNormal:  (x, p) => 0.5 * Math.log(2 / Math.PI) - Math.log(p.sigma) - 0.5 * (x / p.sigma) ** 2,
  // extend as dists are added; absence => throw in buildModelView
};

function priorLogpdf(distOp: string, x: number, params: any): number {
  const f = PRIOR_LOGPDF[distOp];
  if (!f) throw new Error(`model-view: no prior log-density for '${distOp}'`);
  return f(x, params);
}

function buildModelView(spec: any) {
  const latents = spec.latents;
  const names = latents.map((l: any) => l.name);
  const transforms = latents.map((l: any) => T.transformFor(l.support));
  const hasDiscrete = latents.some((l: any) => l.discrete);
  const dim = latents.length;

  function constrainAll(y: Float64Array) {
    const theta: any = {};
    for (let i = 0; i < dim; i++) theta[names[i]] = transforms[i].constrain(y[i]);
    return theta;
  }
  function unconstrainAll(theta: any) {
    const y = new Float64Array(dim);
    for (let i = 0; i < dim; i++) y[i] = transforms[i].unconstrain(theta[names[i]]);
    return y;
  }
  function logPosteriorConstrained(theta: any) {
    let lp = 0;
    for (let i = 0; i < dim; i++) lp += priorLogpdf(latents[i].distOp, theta[names[i]], latents[i].params);
    lp += spec.logLikelihood(theta);
    return lp;
  }
  function logPosterior(y: Float64Array) {
    const theta = constrainAll(y);
    let lp = logPosteriorConstrained(theta);
    for (let i = 0; i < dim; i++) lp += transforms[i].logDetJ(y[i]);   // change of variables
    return lp;
  }
  return { dim, names, hasDiscrete, constrainAll, unconstrainAll, logPosterior, logPosteriorConstrained };
}

module.exports = { buildModelView, priorLogpdf };

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

// ─── Legacy sync ModelView (OLD path, used by model-view.test.ts and the
//     mh-backend.test.ts single-latent path via mat-density.ts).
// ─────────────────────────────────────────────────────────────────────────
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

// ─── New async vector-aware ModelView (wires mcmc-density.ts scorer) ─────
//
// Contract:
//   constrainAll(y: Float64Array) → FLAT record:
//     scalars:  { mu: 0.5, tau: 1.2 }
//     vectors:  { 'theta[0]': 1.0, 'theta[1]': 2.0, … }
//   This flat layout satisfies the driver's  `theta[mv.names[d]]`  indexing.
//
//   logPosteriorConstrained(scorerPt) accepts the SCORER format (with
//     Float64Array vectors), calls logPi(scorerPt) directly.
//
//   logPosterior(y) → constrainAll(y) → reassemble scorer format → logPi + Jacobians.
//
//   unconstrainAll(scorerPt) accepts scorer format, returns Float64Array(dim).
//
//   latentNames   — unique latent names in declaration order.
//   latentShapes  — { kind: 'scalar' | 'vector', dims?: [d] } per latent.
// ─────────────────────────────────────────────────────────────────────────
async function buildModelViewFromCtx(ctx: any, posteriorDeriv: any): Promise<any> {
  const { buildLogPi } = require('./mcmc-density.ts');
  const modelSpec = require('./model-spec.ts');

  // 1. Enumerate latents in declaration order.
  const leafLatents = modelSpec.enumerateLatents(posteriorDeriv, ctx);

  // 2. Reject discrete latents.
  for (const l of leafLatents) {
    if (l.discrete) {
      throw new Error(
        `backend 'mh' (continuous): latent '${l.name}' is discrete; `
        + `use backend 'is' or a discrete MH path`,
      );
    }
  }

  // 3. Build per-coordinate structures (coordinates = one slot per scalar
  //    or per element of a vector latent).
  const coordNames: string[]  = [];   // e.g. ['mu','tau','theta[0]','theta[1]',…]
  const coordSupports: any[]  = [];   // per-coordinate support
  const latentNames: string[] = [];   // unique latent names
  const latentShapes: any[]   = [];   // { kind:'scalar'|'vector', dims?:[d] }

  for (const l of leafLatents) {
    latentNames.push(l.name);
    latentShapes.push(l.shape);
    if (l.shape.kind === 'scalar') {
      coordNames.push(l.name);
      coordSupports.push(l.support);
    } else {
      // vector — expand each coordinate
      const d = l.shape.dims[0];
      for (let i = 0; i < d; i++) {
        coordNames.push(`${l.name}[${i}]`);
        coordSupports.push(l.support);
      }
    }
  }

  const dim = coordNames.length;
  const coordTransforms = coordSupports.map((s: any) => T.transformFor(s));

  // 4. Async setup: build logPi / priorOf / likOf from mcmc-density.ts.
  const { logPi } = await buildLogPi(ctx, posteriorDeriv);

  // Helper: flat coord record → scorer-format record { name: scalar | Float64Array }.
  function flatToScorerPt(flatPt: Record<string, any>): Record<string, any> {
    const rec: Record<string, any> = {};
    for (let li = 0; li < latentNames.length; li++) {
      const nm  = latentNames[li];
      const shp = latentShapes[li];
      if (shp.kind === 'scalar') {
        rec[nm] = flatPt[nm];
      } else {
        const d = shp.dims[0];
        const vec = new Float64Array(d);
        for (let j = 0; j < d; j++) vec[j] = flatPt[`${nm}[${j}]`];
        rec[nm] = vec;
      }
    }
    return rec;
  }

  // 5. Driver-compatible constrainAll: Float64Array(dim) → flat record.
  function constrainAll(y: Float64Array): Record<string, any> {
    const flat: Record<string, any> = {};
    for (let i = 0; i < dim; i++) {
      flat[coordNames[i]] = coordTransforms[i].constrain(y[i]);
    }
    return flat;
  }

  // 6. unconstrainAll: scorer-format record → Float64Array(dim).
  function unconstrainAll(scorerPt: Record<string, any>): Float64Array {
    const y = new Float64Array(dim);
    let i = 0;
    for (let li = 0; li < latentNames.length; li++) {
      const nm  = latentNames[li];
      const shp = latentShapes[li];
      if (shp.kind === 'scalar') {
        const v = scorerPt[nm];
        y[i] = coordTransforms[i].unconstrain(typeof v === 'number' ? v : +v);
        i++;
      } else {
        const d   = shp.dims[0];
        const vec = scorerPt[nm];
        for (let j = 0; j < d; j++) {
          const v = vec instanceof Float64Array ? vec[j]
            : Array.isArray(vec) ? vec[j] : +vec;
          y[i] = coordTransforms[i].unconstrain(v);
          i++;
        }
      }
    }
    return y;
  }

  // 7. logPosteriorConstrained: scorer-format record → scalar (no Jacobian).
  function logPosteriorConstrained(scorerPt: Record<string, any>): number {
    return logPi(scorerPt);
  }

  // 8. logPosterior: unconstrained Float64Array → scalar (includes Jacobian).
  function logPosterior(y: Float64Array): number {
    const flatPt = constrainAll(y);
    const scorerPt = flatToScorerPt(flatPt);
    let lp = logPi(scorerPt);
    for (let i = 0; i < dim; i++) lp += coordTransforms[i].logDetJ(y[i]);
    return lp;
  }

  return {
    dim,
    names: coordNames,        // per-coordinate expanded names (length = dim)
    hasDiscrete: false,       // only continuous latents reach here
    constrainAll,
    unconstrainAll,
    logPosterior,
    logPosteriorConstrained,
    latentNames,
    latentShapes,
  };
}

module.exports = { buildModelView, buildModelViewFromCtx, priorLogpdf };

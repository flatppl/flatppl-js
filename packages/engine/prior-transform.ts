// packages/engine/prior-transform.ts
'use strict';
// Unit-cube prior transform T:[0,1]^d → θ for nested sampling. Walks the prior's
// latents in declaration (= topological) order; for each, expands its measure IR,
// unwraps to a base distribution, resolves that dist's params in a running env
// holding already-realised parents (via sampler.evaluateExpr — the same param
// resolution the density path uses), and maps the latent's cube coordinate(s)
// through the inverse-CDF ladder. Deterministic, monotone, bijective — the hard
// requirement nested sampling places on the transform.
const orchestrator = require('./orchestrator.ts');
const sampler      = require('./sampler.ts');
const modelSpec    = require('./model-spec.ts');
const { quantile, hasQuantile } = require('./inverse-cdf.ts');

// Evaluate a param IR to a number in env; null on failure.
function evalParam(ir: any, env: Record<string, any>): number | null {
  try { const v = sampler.evaluateExpr(ir, env); return typeof v === 'number' ? v : +v; } catch (_) { return null; }
}

// Map a base distribution's arg IRs → the params object the ladder expects.
// Positional-arg convention per dist (matches inverse-cdf.ts param field names).
const ARG_NAMES: Record<string, string[]> = {
  Normal: ['mu', 'sigma'], LogNormal: ['mu', 'sigma'], Exponential: ['rate'],
  Uniform: ['lo', 'hi'], Beta: ['alpha', 'beta'], Gamma: ['shape', 'rate'],
  HalfNormal: ['sigma'], HalfCauchy: ['scale'], Cauchy: ['location', 'scale'],
  Logistic: ['mu', 's'], Weibull: ['shape', 'scale'], Pareto: ['shape', 'scale'],
  Laplace: ['location', 'scale'], Dirac: ['value'],
};
function resolveParams(distOp: string, argIRs: any[], env: Record<string, any>): any {
  const names = ARG_NAMES[distOp];
  if (!names) throw new Error(`prior-transform: unsupported base distribution '${distOp}'`);
  const q: any = {};
  for (let i = 0; i < names.length; i++) {
    const v = evalParam(argIRs[i], env);
    if (v == null) throw new Error(`prior-transform: cannot resolve param '${names[i]}' of ${distOp}`);
    q[names[i]] = v;
  }
  return q;
}

// A per-latent plan: how many cube coords it consumes and how to realise it from
// (coords, env). Built once per latent; `realise` returns the latent's value and
// its cube-coord count. Extended by later tasks (iid/truncate/pushfwd/composite).
function planLatent(measureIR: any): { count: number; realise: (u: Float64Array, off: number, env: Record<string, any>) => number } {
  if (measureIR.kind !== 'call') throw new Error('prior-transform: non-call measure IR');
  const op = measureIR.op;
  if (hasQuantile(op) || ARG_NAMES[op]) {
    // Scalar base distribution: one coord.
    return {
      count: 1,
      realise: (u, off, env) => {
        const params = resolveParams(op, measureIR.args, env);
        return quantile(op, u[off], params);
      },
    };
  }
  throw new Error(`prior-transform: latent measure op '${op}' not yet supported by the nested backend`);
}

function buildPriorTransform(ctx: any, d: any) {
  if (!d || d.kind !== 'bayesupdate') throw new Error('buildPriorTransform: derivation must be kind=bayesupdate');
  const sources = modelSpec.priorLatentSources(d, ctx);
  if (sources.length === 0) throw new Error('buildPriorTransform: prior has no stochastic draws');
  const leaf = modelSpec.enumerateLatents(d, ctx);
  const shapeOf: Record<string, any> = {};
  for (const l of leaf) { if (l.discrete) throw new Error(`nested backend: latent '${l.name}' is discrete`); shapeOf[l.name] = l.shape; }

  // Fixed-value env seed (data constants).
  const fixedEnv: Record<string, any> = {};
  if (ctx.fixedValues) ctx.fixedValues.forEach((v: any, k: string) => { fixedEnv[k] = v; });

  const plans: Array<{ name: string; plan: ReturnType<typeof planLatent> }> = [];
  const latentNames: string[] = [];
  const coordSupports: any[] = [];
  let dim = 0;
  for (const { latentName, measureName } of sources) {
    const measureIR = orchestrator.expandMeasure(measureName, { derivations: ctx.derivations, bindings: ctx.bindings });
    const plan = planLatent(measureIR);
    plans.push({ name: latentName, plan });
    latentNames.push(latentName);
    const shp = shapeOf[latentName] || { kind: 'scalar' };
    for (let i = 0; i < plan.count; i++) coordSupports.push(shp.support || { kind: 'real' });
    dim += plan.count;
  }

  function transform(u: Float64Array): Record<string, any> {
    const env: Record<string, any> = Object.assign({}, fixedEnv);
    const rec: Record<string, any> = {};
    let off = 0;
    for (const { name, plan } of plans) {
      if (plan.count === 1) {
        const x = plan.realise(u, off, env);
        rec[name] = x; env[name] = x;
      } else {
        const v = new Float64Array(plan.count);
        for (let i = 0; i < plan.count; i++) v[i] = plan.realise(u, off + i, env);
        rec[name] = v; env[name] = Array.from(v);
      }
      off += plan.count;
    }
    return rec;
  }

  return { dim, latentNames, coordSupports, transform };
}
module.exports = { buildPriorTransform };

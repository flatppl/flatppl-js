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
const { truncatedQuantile } = require('./forward-cdf.ts');
const { recognizeCompositeIidDraw } = require('./composite-prior.ts');

// Evaluate a param IR to a number in env; null on failure.
function evalParam(ir: any, env: Record<string, any>): number | null {
  try { const v = sampler.evaluateExpr(ir, env); return typeof v === 'number' ? v : +v; } catch (_) { return null; }
}

// Evaluate a truncation bound IR — handles the ±inf sentinel that a plain
// evaluateExpr can't (there's no numeric literal for infinity in the IR). An
// infinite bound is legitimate and returns ±Infinity; a bound that fails to
// resolve to a finite number (unresolvable param, or NaN) throws rather than
// silently falling back to a sentinel — the nested-sampling invariant demands
// the transform fail loudly, never mis-map a coordinate (see resolveParams,
// which throws the same way on an unresolvable dist param).
function evalBound(ir: any, env: Record<string, any>): number {
  if (ir && ir.kind === 'const' && ir.name === 'inf') return Infinity;
  if (ir && ir.kind === 'call' && ir.op === 'neg' && ir.args && ir.args[0] && ir.args[0].kind === 'const' && ir.args[0].name === 'inf') return -Infinity;
  const v = evalParam(ir, env);
  if (v == null || Number.isNaN(v)) throw new Error('prior-transform: cannot resolve truncate bound');
  return v;
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
function planLatent(measureIR: any, ctx: any): { count: number; realise: (u: Float64Array, off: number, env: Record<string, any>, localIdx?: number) => number } {
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
  if (op === 'iid') {
    const inner = measureIR.args[0];
    const nVal = measureIR.args[1];
    const n = (nVal && nVal.kind === 'lit') ? nVal.value : null;
    if (!Number.isInteger(n) || n <= 0) throw new Error('prior-transform: iid size must be a positive integer literal');
    const innerPlan = planLatent(inner, ctx);     // recurse — inner is a base dist (or truncate/pushfwd via later tasks)
    if (innerPlan.count !== 1) throw new Error('prior-transform: iid inner measure must be scalar per element');
    return {
      count: n,
      realise: (u, off, env) => innerPlan.realise(u, off, env),  // each coord independently through the inner F⁻¹
    };
  }

  if (op === 'normalize') {
    return planLatent(measureIR.args[0], ctx);   // normalizing a truncation → its truncated quantile
  }
  if (op === 'truncate') {
    const D = measureIR.args[0];
    const region = measureIR.args[1];
    if (!D || D.kind !== 'call' || !ARG_NAMES[D.op]) throw new Error(`prior-transform: truncate base '${D && D.op}' unsupported`);
    if (!region || region.kind !== 'call' || region.op !== 'interval') throw new Error('prior-transform: truncate region must be interval(lo,hi)');
    return {
      count: 1,
      realise: (u, off, env) => {
        const params = resolveParams(D.op, D.args, env);
        const lo = evalBound(region.args[0], env);
        const hi = evalBound(region.args[1], env);
        if (!(lo <= hi)) throw new Error(`prior-transform: truncate lo>hi (lo=${lo},hi=${hi})`);
        return truncatedQuantile(D.op, u[off], params, lo, hi);
      },
    };
  }

  if (op === 'pushfwd') {
    const fnRef = measureIR.args[0];
    const base = measureIR.args[1];
    const basePlan = planLatent(base, ctx);
    if (basePlan.count !== 1) throw new Error('prior-transform: pushfwd base must be scalar');
    // The forward function is a `functionof` binding referenced by name.
    const fb = ctx.bindings && ctx.bindings.get && ctx.bindings.get(fnRef && fnRef.name);
    const fir = fb && (fb.ir || fb.node || fb);
    if (!fir || fir.op !== 'functionof' || !fir.body || !fir.params || !fir.params[0]) {
      throw new Error(`prior-transform: cannot resolve pushfwd forward function '${fnRef && fnRef.name}'`);
    }
    const placeholder = fir.params[0];
    const body = fir.body;
    return {
      count: 1,
      realise: (u, off, env) => {
        const baseVal = basePlan.realise(u, off, env);
        const y = sampler.evaluateExpr(body, Object.assign({}, env, { [placeholder]: baseVal }));
        return typeof y === 'number' ? y : +y;
      },
    };
  }

  throw new Error(`nested backend: prior form '${op}' is not invertible; not eligible for nested sampling`);
}

function buildPriorTransform(ctx: any, d: any) {
  if (!d || d.kind !== 'bayesupdate') throw new Error('buildPriorTransform: derivation must be kind=bayesupdate');
  const sources = modelSpec.priorLatentSources(d, ctx);
  if (sources.length === 0) throw new Error('buildPriorTransform: prior has no stochastic draws');
  const leaf = modelSpec.enumerateLatents(d, ctx);
  const shapeOf: Record<string, any> = {};
  const supportOf: Record<string, any> = {};
  for (const l of leaf) {
    // Discrete latents have no continuous quantile; planLatent throws 'op not
    // yet supported' for their distribution op (nested refusal message refined
    // in the composite/refusal task).
    shapeOf[l.name] = l.shape;
    supportOf[l.name] = l.support;
  }

  // Fixed-value env seed (data constants).
  const fixedEnv: Record<string, any> = {};
  if (ctx.fixedValues) ctx.fixedValues.forEach((v: any, k: string) => { fixedEnv[k] = v; });

  // Derived (non-draw) bindings — e.g. `logm = log(m)` sitting between a
  // parent draw and a child latent's dist params (`b ~ LogNormal(logm, 0.5)`).
  // buildDerivations' lift pass already cached each binding's lowered IR on
  // `.ir` (module-context-aware — reusing it beats re-lowering the raw AST
  // ourselves), so collecting derived bindings is just: skip draws (handled
  // by `plans` below) and anything already fixed, keep everything else that
  // lowered cleanly. `refreshDerived` evaluates each into `env` and is called
  // once after seeding `fixedEnv` and again after every realised latent, so a
  // derived binding that depends on a just-realised parent becomes visible to
  // the next latent's params. A binding not yet resolvable (still missing a
  // later parent) is silently skipped — it will resolve on a later call.
  const derivedBindings: Array<{ name: string; ir: any }> = [];
  if (ctx.bindings) {
    ctx.bindings.forEach((b: any, k: string) => {
      if (!b || b.type === 'draw' || (k in fixedEnv) || !b.ir) return;
      derivedBindings.push({ name: k, ir: b.ir });
    });
  }
  function refreshDerived(env: Record<string, any>): void {
    for (const db of derivedBindings) {
      try { env[db.name] = sampler.evaluateExpr(db.ir, env); } catch (_) { /* not yet resolvable */ }
    }
  }

  // Coerce a resolved value (JS array | Value{shape,data} | scalar) to a flat JS array.
  function asArr(x: any): number[] {
    if (Array.isArray(x)) return x;
    if (x && x.data && (x.data instanceof Float64Array || Array.isArray(x.data))) return Array.from(x.data);
    return [x];
  }

  // A composite-iid draw (kernel-broadcast over a user function returning an
  // iid block, e.g. `p ~ beta_row_K.(a,b)` where each row has its own params)
  // has no single measure IR `planLatent` can walk — `enumerateLatents`
  // reports it as a fromPool placeholder shape (dims:[0], not the real size).
  // Build its plan directly from the recognizer: `comp.D` is the inner iid
  // axis length (N); `G` (row count) is the length of the first resolved
  // sub-param vector. The block is [G,D] row-major — element `i` maps to row
  // `g = ⌊i/D⌋`, whose per-row params come from `comp.sub[name][g]`.
  function makeCompositePlan(comp: any, buildEnv: Record<string, any>) {
    const D = comp.D;
    const g0 = asArr(sampler.evaluateExpr(comp.sub[comp.paramNames[0]], buildEnv));
    const G = g0.length;
    return {
      count: G * D,
      realise: (u: Float64Array, off: number, env: Record<string, any>, localIdx?: number) => {
        const g = Math.floor((localIdx || 0) / D);
        const rowParams: any = {};
        for (const pn of comp.paramNames) {
          const pv = asArr(sampler.evaluateExpr(comp.sub[pn], env));
          rowParams[pn] = pv[g < pv.length ? g : pv.length - 1];
        }
        return quantile(comp.distOp, u[off], rowParams);
      },
    };
  }

  const plans: Array<{ name: string; plan: ReturnType<typeof planLatent> }> = [];
  const latentNames: string[] = [];
  const coordSupports: any[] = [];
  let dim = 0;
  for (const { latentName, measureName } of sources) {
    const comp = recognizeCompositeIidDraw(measureName, ctx);
    let plan: ReturnType<typeof planLatent>;
    if (comp) {
      const seedEnv = Object.assign({}, fixedEnv); refreshDerived(seedEnv);
      plan = makeCompositePlan(comp, seedEnv);
    } else {
      const measureIR = orchestrator.expandMeasure(measureName, { derivations: ctx.derivations, bindings: ctx.bindings });
      plan = planLatent(measureIR, ctx);
    }
    plans.push({ name: latentName, plan });
    latentNames.push(latentName);
    for (let i = 0; i < plan.count; i++) coordSupports.push(supportOf[latentName] || { kind: 'real' });
    dim += plan.count;
  }

  function transform(u: Float64Array): Record<string, any> {
    const env: Record<string, any> = Object.assign({}, fixedEnv);
    refreshDerived(env);
    const rec: Record<string, any> = {};
    let off = 0;
    for (const { name, plan } of plans) {
      if (plan.count === 1) {
        const x = plan.realise(u, off, env);
        rec[name] = x; env[name] = x;
      } else {
        const v = new Float64Array(plan.count);
        for (let i = 0; i < plan.count; i++) v[i] = plan.realise(u, off + i, env, i);
        rec[name] = v; env[name] = Array.from(v);
      }
      off += plan.count;
      refreshDerived(env);
    }
    return rec;
  }

  return { dim, latentNames, coordSupports, transform };
}
module.exports = { buildPriorTransform };

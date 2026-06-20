'use strict';

// mcmc-density.ts — SYNCHRONOUS posterior-numerator log-density for MCMC.
//
// Provides buildLogPi(ctx, d) → { logPi, priorOf, likOf }, where all three
// returned functions are SYNCHRONOUS.  The async SETUP resolves everything
// that would otherwise block an MCMC inner loop:
//
//   • Prior IR:     expandMeasure + resolveNormalizeMasses (handles
//                   normalize(truncate(Cauchy,...)) as in eight-schools tau)
//   • Likelihood IR: clm.lowerMeasure (same call as matBayesupdate IS path)
//   • Observed:     orchestrator.resolveIRToValue
//
// SCORE (sync, per call):
//   priorOf(pt) = density.logDensityN(priorIR, pt, refArrays+derivedEnv, 1, opts)
//   likOf(pt)   = density.logDensityN(likBodyIR, observed, refArrays+derivedEnv, 1, opts)
//   logPi(pt)   = priorOf(pt) + likOf(pt)
//
// opts carries parseSet (required by walkTruncate) and resolveMeasureRef.
// Both are rebuilt each call with the current point's derived env.

const orchestrator  = require('./orchestrator.ts');
const clm           = require('./clm.ts');
const density       = require('./density.ts');
const irShared      = require('./ir-shared.ts');
const sampler       = require('./sampler.ts');
const lower         = require('./lower.ts');

// ---------------------------------------------------------------------------
// parseSet — mirrors worker.ts makeParseSet (worker.ts:112-126).
// ---------------------------------------------------------------------------
function makeParseSet(env: Record<string, any>) {
  return (setIR: any) => {
    const d = irShared.parseSetIR(setIR, null);
    if (d) return d;
    if (setIR && setIR.kind === 'call' && setIR.op === 'interval'
        && Array.isArray(setIR.args) && setIR.args.length === 2) {
      try {
        const lo = +sampler.evaluateExpr(setIR.args[0], env);
        const hi = +sampler.evaluateExpr(setIR.args[1], env);
        if (Number.isFinite(lo) || Number.isFinite(hi)) return { kind: 'interval', lo, hi };
      } catch (_) { /* fall through */ }
    }
    return null;
  };
}

// ---------------------------------------------------------------------------
// resolveNormalizeMasses — inline version of the unexported function in
// mat-density.ts (mat-density.ts:730).
//
// Walk measureIR and collect all normalize nodes with a massFrom spec.
// Materialise the inner measure via ctx.getMeasure (cached in ctx), read the
// tracked logTotalmass, and rewrite the node in-place to
// logweighted(-logZ, inner) — the form the density walker expects
// (mat-density.ts:730-754).
// ---------------------------------------------------------------------------
function collectNormalizeMassNodes(node: any, out: any[], seen: Set<any>) {
  if (!node || typeof node !== 'object') return;
  if (seen.has(node)) return;
  seen.add(node);
  if (Array.isArray(node)) {
    for (const x of node) collectNormalizeMassNodes(x, out, seen);
    return;
  }
  if (node.kind === 'call' && node.op === 'normalize' && node.massFrom) out.push(node);
  for (const k in node) {
    const v = node[k];
    if (v && typeof v === 'object') collectNormalizeMassNodes(v, out, seen);
  }
}

async function resolveNormalizeMasses(measureIR: any, ctx: any): Promise<any> {
  const nodes: any[] = [];
  collectNormalizeMassNodes(measureIR, nodes, new Set());
  if (nodes.length === 0) return measureIR;
  const refNames = Array.from(new Set(nodes.map((n: any) => n.massFrom.ref as string)));
  const measures = await Promise.all(refNames.map((r) => Promise.resolve(ctx.getMeasure(r))));
  const byName: Record<string, any> = {};
  refNames.forEach((nm, i) => { byName[nm] = measures[i]; });
  for (const node of nodes) {
    const M = byName[node.massFrom.ref];
    const logZ = M && typeof M.logTotalmass === 'number' ? M.logTotalmass : null;
    if (logZ == null || !Number.isFinite(logZ)) {
      throw new Error('normalize density: inner measure "' + node.massFrom.ref
        + '" did not materialise to a finite tracked logTotalmass'
        + ' (Z = 0 or ∞ is undefined per spec §06)');
    }
    const inner = node.args[0];
    node.op = 'logweighted';
    node.args = [{ kind: 'lit', value: -logZ }, inner];
    delete node.massFrom;
  }
  return measureIR;
}

// ---------------------------------------------------------------------------
// evaluateDerivedBindings — resolve all non-stochastic bindings in the
// model at the given point.
//
// The density walker resolves kwarg refs via `evaluateExpr(exprIR, callEnv)`.
// For derived bindings (type='value', e.g. `alpha = phi * kappa`), the ref
// `alpha` needs to be in callEnv.  The fix: traverse bindings in declaration
// order, evaluate each non-draw binding's lowered RHS in the running env
// (which starts with fixedValues + the point), and collect the result.
//
// The point record provides values for the stochastic latents; lowerExpr
// converts the AST node to IR; evaluateExpr evaluates it.  Errors (e.g. a
// function binding whose body can't be evaluated as a plain value) are
// silently skipped — the binding may not be needed for scoring.
// ---------------------------------------------------------------------------
function evaluateDerivedBindings(
  ctx: any,
  pt: Record<string, any>,
): Record<string, any> {
  const env: Record<string, any> = {};
  // Seed with fixed values (data constants, etc.)
  if (ctx.fixedValues) {
    ctx.fixedValues.forEach((v: any, k: string) => { env[k] = v; });
  }
  // Seed with the point values (stochastic latents).
  // Convert Float64Array to plain Array so evaluateExpr index operations work.
  for (const [k, v] of Object.entries(pt)) {
    if (typeof v === 'number') {
      env[k] = v;
    } else if (v instanceof Float64Array) {
      env[k] = Array.from(v);
    } else if (Array.isArray(v)) {
      env[k] = v;
    }
  }
  // Evaluate derived (non-draw) bindings in declaration order
  if (ctx.bindings) {
    ctx.bindings.forEach((binding: any, k: string) => {
      if (!binding || binding.type === 'draw' || k in env) return;
      // Only process value-type bindings with a lowerable RHS node
      const node = binding.node || binding;
      const rhsNode = node.value || node.effectiveValue;
      if (!rhsNode) return;
      try {
        const ir = lower.lowerExpr(rhsNode);
        const val = sampler.evaluateExpr(ir, env);
        env[k] = val;
      } catch (_) {
        // Function binding, stochastic ref, or unresolvable — skip
      }
    });
  }
  return env;
}

// ---------------------------------------------------------------------------
// Value → refArrays suitable for density.logDensityN(count=1).
//
// Scalar latents go into refArrays as Float64Array([v]) — the density walker
// does refArrays[k][i] at atom i=0 and puts it in callEnv[k].
//
// Vector latents (Float64Array or Array with length > 1) CANNOT go into
// refArrays: the walker would do refArrays[k][0] and get a single scalar,
// losing the vector.  Instead they are injected directly into baseEnv
// (additionalEnv) so the walker finds the full array for indexing operations
// like `a[group_data]` or `theta[person]`.  evaluateExpr resolves plain
// Array indexing correctly; Float64Array indexing also works for integer
// indices but we convert to Array to be safe with arbitrary index expressions.
// ---------------------------------------------------------------------------
function pointToRefArrays(
  pt: Record<string, any>,
): { refArrays: Record<string, Float64Array>; vectorEnv: Record<string, any> } {
  const refArrays: Record<string, Float64Array> = {};
  const vectorEnv: Record<string, any> = {};
  for (const [k, v] of Object.entries(pt)) {
    if (typeof v === 'number') {
      refArrays[k] = new Float64Array([v]);
    } else if (v instanceof Float64Array) {
      if (v.length === 1) {
        refArrays[k] = v;
      } else {
        // Full vector → env so indexing works at atom i=0
        vectorEnv[k] = Array.from(v);
      }
    } else if (Array.isArray(v)) {
      if (v.length === 1) {
        refArrays[k] = new Float64Array([+(v as any)[0]]);
      } else {
        vectorEnv[k] = v;
      }
    } else {
      refArrays[k] = new Float64Array([+(v as any)]);
    }
  }
  return { refArrays, vectorEnv };
}

// ---------------------------------------------------------------------------
// PUBLIC: buildLogPi
//
// async setup → sync logPi / priorOf / likOf.
//
// ctx  — materialiser context (as built by ctxFor in regression-baseline.test.ts)
// d    — the bayesupdate derivation (ctx.derivations[posteriorName])
// ---------------------------------------------------------------------------
async function buildLogPi(
  ctx: any,
  d: any,
): Promise<{
  logPi:   (pt: Record<string, any>) => number;
  priorOf: (pt: Record<string, any>) => number;
  likOf:   (pt: Record<string, any>) => number;
}> {
  if (!d || d.kind !== 'bayesupdate') {
    throw new Error('buildLogPi: derivation must be kind=bayesupdate');
  }

  // ── SETUP: prior IR ───────────────────────────────────────────────────────
  // expandMeasure: same call as matBroadcastLogdensity (mat-density.ts:865)
  // resolveNormalizeMasses: inline copy of mat-density.ts:730
  const priorName = d.from as string;
  const rawPriorIR = orchestrator.expandMeasure(
    priorName, { derivations: ctx.derivations, bindings: ctx.bindings },
  );
  if (!rawPriorIR) throw new Error(`buildLogPi: cannot expand prior '${priorName}'`);
  const priorIR = await resolveNormalizeMasses(rawPriorIR, ctx);

  // ── SETUP: likelihood body IR ─────────────────────────────────────────────
  // clm.lowerMeasure: same call as mat-density.ts:210 (IS path in matBayesupdate)
  const likNode = clm.lowerMeasure(d.bodyIR || d.bodyName, ctx, { derivation: d });
  if (!likNode) {
    throw new Error('buildLogPi: cannot expand the likelihood body into measure IR');
  }
  const likBodyIR = likNode.body;

  // ── SETUP: observed ───────────────────────────────────────────────────────
  const observed = orchestrator.resolveIRToValue(d.obsIR, ctx.bindings, ctx.fixedValues);
  if (observed == null) {
    throw new Error('buildLogPi: cannot resolve observed value from obsIR');
  }

  // ── SYNC closures ─────────────────────────────────────────────────────────

  function priorOf(pt: Record<string, any>): number {
    // Build an env that includes: fixedValues + stochastic latents at pt
    // + all derived bindings evaluated at pt.
    const callEnv = evaluateDerivedBindings(ctx, pt);
    if (ctx.moduleRegistry) callEnv.__moduleRegistry = ctx.moduleRegistry;
    const { refArrays, vectorEnv } = pointToRefArrays(pt);
    // Merge vector latents into baseEnv (they can't go through refArrays[k][i]).
    Object.assign(callEnv, vectorEnv);
    const opts = {
      parseSet: makeParseSet(callEnv),
      resolveMeasureRef: (_name: string) => null as any,
      baseEnv: callEnv,
    };
    try {
      const logps = density.logDensityN(priorIR, pt, refArrays, 1, opts);
      const v = logps[0];
      return Number.isFinite(v) ? v : -Infinity;
    } catch (_err) {
      return -Infinity;
    }
  }

  function likOf(pt: Record<string, any>): number {
    const callEnv = evaluateDerivedBindings(ctx, pt);
    if (ctx.moduleRegistry) callEnv.__moduleRegistry = ctx.moduleRegistry;
    const { refArrays, vectorEnv } = pointToRefArrays(pt);
    Object.assign(callEnv, vectorEnv);
    const opts = {
      parseSet: makeParseSet(callEnv),
      resolveMeasureRef: (_name: string) => null as any,
      baseEnv: callEnv,
    };
    try {
      const logps = density.logDensityN(likBodyIR, observed, refArrays, 1, opts);
      const v = logps[0];
      return Number.isFinite(v) ? v : -Infinity;
    } catch (_err) {
      return -Infinity;
    }
  }

  function logPi(pt: Record<string, any>): number {
    const lp = priorOf(pt) + likOf(pt);
    return Number.isFinite(lp) ? lp : -Infinity;
  }

  return { logPi, priorOf, likOf };
}

module.exports = { buildLogPi };

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
const { totalMassExpr } = require('./normalize-mass.ts');

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
  // First pass: rewrite every node whose inner mass we can express as a
  // θ-dependent expression (the closed-form superpose-mixture mass). These need
  // NO materialisation and are correct at every θ.
  const needMaterialise: any[] = [];
  for (const node of nodes) {
    const inner = node.args[0];
    const massExpr = totalMassExpr(inner);
    if (massExpr != null) {
      // normalize(inner) → logweighted(−log Z(θ), inner). walkLogWeighted
      // evaluates the (non-functionof) weight per atom against θ.
      node.op = 'logweighted';
      node.args = [{ kind: 'call', op: 'neg', args: [{ kind: 'call', op: 'log', args: [massExpr] }] }, inner];
      delete node.massFrom;
    } else {
      needMaterialise.push(node);
    }
  }
  if (needMaterialise.length === 0) return measureIR;
  // Fallback: a normalize whose mass we can't express symbolically. Materialise
  // the inner measure once and bake the constant −log Z (the pre-existing
  // approximation; correct only when Z is θ-independent).
  const refNames = Array.from(new Set(needMaterialise.map((n: any) => n.massFrom.ref as string)));
  const measures = await Promise.all(refNames.map((r) => Promise.resolve(ctx.getMeasure(r))));
  const byName: Record<string, any> = {};
  refNames.forEach((nm, i) => { byName[nm] = measures[i]; });
  for (const node of needMaterialise) {
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
    } else if (v && (v as any).data instanceof Float64Array && Array.isArray((v as any).shape)) {
      // Multi-axis (matrix) latent — a Value {shape,data}, e.g. a [G,N] block
      // `p ~ beta_row_K.(a,b)`. It must reach the density walker with its FULL
      // shape (the likelihood broadcasts it against an equally-ranked data
      // matrix; flattening it to a 1-axis vector — or the +Value=NaN fallback
      // below — makes the broadcast axis count mismatch). Keep the Value in the
      // env (same form buildEnv threads), never a scalar refArray.
      vectorEnv[k] = v;
    } else {
      refArrays[k] = new Float64Array([+(v as any)]);
    }
  }
  return { refArrays, vectorEnv };
}

// Collect the names of all `{kind:'ref'}` nodes in an IR (free identifiers the
// density walker will resolve from callEnv). Used to partition atom-dependent
// from atom-independent bindings for the batched scorer.
function collectRefNames(node: any, out: Set<string>): void {
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node)) { for (const x of node) collectRefNames(x, out); return; }
  if (node.kind === 'ref' && typeof node.name === 'string') out.add(node.name);
  for (const k in node) { const v = node[k]; if (v && typeof v === 'object') collectRefNames(v, out); }
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
  logPi:         (pt: Record<string, any>) => number;
  logPiBatch:    (pts: Array<Record<string, any>>) => Float64Array;
  priorLikBatch: (pts: Array<Record<string, any>>) => { prior: Float64Array; lik: Float64Array };
  priorOf:       (pt: Record<string, any>) => number;
  likOf:         (pt: Record<string, any>) => number;
  probePrior:    (pt: Record<string, any>) => void;
}> {
  if (!d || d.kind !== 'bayesupdate') {
    throw new Error('buildLogPi: derivation must be kind=bayesupdate');
  }

  // ── SETUP: per-draw prior measures ────────────────────────────────────────
  // Parameterise the prior by its stochastic draws and score each draw's OWN
  // measure: logπ_prior(θ) = Σ_draw logdensityof(draw_i's measure | env). This
  // equals logdensityof(prior_record) for an all-stochastic record (the joint
  // factorises — the regression-baseline oracle notes ARE these per-draw sums),
  // AND it handles a prior record that contains a DERIVED field (e.g.
  // sigma=sqrt(sigma2)): we never densify the record (the walker can't walk
  // `sqrt` as a measure op) — we score sigma2's InverseGamma directly and feed
  // the derived sigma through the env. Each draw's measure has its normalize
  // masses resolved once (e.g. tau's normalize(truncate(Cauchy,…))).
  const priorName = d.from as string;
  // The prior's free variates as { latentName, measureName } pairs (model-spec):
  // identity on the canonical draw path, but for a joint-of-distributions prior
  // (`prior = joint(theta1 = Normal(...), …)`, variates supplied as `elementof`
  // boundary inputs — 06-measure-algebra.md §joint) the latent is the field
  // LABEL and its measure is the distribution binding the label points at.
  const modelSpec = require('./model-spec.ts');
  const sources: Array<{ latentName: string; measureName: string }> = modelSpec.priorLatentSources(d, ctx);
  if (sources.length === 0) {
    throw new Error(`buildLogPi: prior '${priorName}' has no stochastic draws to sample`);
  }
  // drawNames are the point-record keys (latent names); measureOf maps each to
  // the binding to expand/recognize for its prior measure.
  const drawNames: string[] = sources.map((s) => s.latentName);
  const measureOf: Record<string, string> = {};
  for (const s of sources) measureOf[s.latentName] = s.measureName;
  // A draw whose measure is a kernel-broadcast over a user function returning
  // an iid block (e.g. p ~ beta_row_K.(a,b)) cannot be scored by expandMeasure's
  // IR directly (its head is a user function, not a built-in distribution). For
  // those we score the inner built-in as a flat independent product with the
  // per-row params expanded across the inner iid axis. compositeDesc[nm] holds
  // { distOp, D, paramNames, sub } when nm is such a draw, else is absent.
  const { recognizeCompositeIidDraw } = require('./composite-prior.ts');
  const compositeDesc: Record<string, any> = {};
  const drawMeasures: Record<string, any> = {};
  for (const nm of drawNames) {
    const msrc = measureOf[nm];   // measure-source binding (= nm on the draw path)
    const comp = recognizeCompositeIidDraw(msrc, ctx);
    if (comp) { compositeDesc[nm] = comp; continue; }
    const raw = orchestrator.expandMeasure(
      msrc, { derivations: ctx.derivations, bindings: ctx.bindings },
    );
    if (!raw) throw new Error(`buildLogPi: cannot expand draw measure '${msrc}'`);
    drawMeasures[nm] = await resolveNormalizeMasses(raw, ctx);
  }

  // ── SETUP: likelihood body IR ─────────────────────────────────────────────
  // clm.lowerMeasure: same call as mat-density.ts:210 (IS path in matBayesupdate)
  const likNode = clm.lowerMeasure(d.bodyIR || d.bodyName, ctx, { derivation: d });
  if (!likNode) {
    throw new Error('buildLogPi: cannot expand the likelihood body into measure IR');
  }
  // Resolve any normalize-mass spec in the LIKELIHOOD body, exactly as the IS
  // path does (mat-density.matScore: "resolve on every matScore path including
  // bayesupdate"). A normalize(...) in the body (e.g. a normalized mixture
  // `normalize(superpose(weighted(theta, …), weighted(1-theta, …)))`) lowers to
  // a node carrying massFrom; left unresolved it reaches walkNormalize, which
  // refuses an unresolved spec — swallowed to −∞ by likWith, making logπ = −∞
  // everywhere (every sampler stuck at init; SMC's 0·(−∞)=NaN at β=0 wipes all
  // weights). Mirrors the per-draw prior resolution above.
  const likBodyIR = await resolveNormalizeMasses(likNode.body, ctx);

  // ── SETUP: observed ───────────────────────────────────────────────────────
  const observed = orchestrator.resolveIRToValue(d.obsIR, ctx.bindings, ctx.fixedValues);
  if (observed == null) {
    throw new Error('buildLogPi: cannot resolve observed value from obsIR');
  }

  // ── SETUP: hot-path precompute ────────────────────────────────────────────
  // The per-step cost is dominated by (a) re-lowering every derived binding's
  // RHS on every call and (b) rebuilding the env twice per logπ. Hoist both:
  // lower each derived (non-draw value) binding's IR ONCE here; per call only
  // evaluateExpr them into a fresh env, built ONCE per logπ and shared by the
  // prior and likelihood terms.
  const fixedEnvObj: Record<string, any> = {};
  if (ctx.fixedValues) ctx.fixedValues.forEach((v: any, k: string) => { fixedEnvObj[k] = v; });
  const derivedBindings: Array<{ name: string; ir: any }> = [];
  if (ctx.bindings) {
    ctx.bindings.forEach((b: any, k: string) => {
      if (!b || b.type === 'draw' || (k in fixedEnvObj)) return;
      const rhsNode = (b.node || b).value || (b.node || b).effectiveValue;
      if (!rhsNode) return;
      try { derivedBindings.push({ name: k, ir: lower.lowerExpr(rhsNode) }); } catch (_) { /* skip */ }
    });
  }
  const moduleReg = ctx.moduleRegistry;
  const NULL_REF = (_n: string) => null as any;

  // Build the scoring env at a point: fixed + draws + derived (eval only, no
  // lowering). Vector draws land as plain arrays so index ops resolve.
  function buildEnv(pt: Record<string, any>): Record<string, any> {
    const env: Record<string, any> = Object.assign({}, fixedEnvObj);
    for (const k in pt) {
      const v = pt[k];
      env[k] = (v instanceof Float64Array) ? Array.from(v) : v;
    }
    for (let i = 0; i < derivedBindings.length; i++) {
      const db = derivedBindings[i];
      if (db.name in env) continue;
      try { env[db.name] = sampler.evaluateExpr(db.ir, env); } catch (_) { /* skip */ }
    }
    if (moduleReg) env.__moduleRegistry = moduleReg;
    return env;
  }

  // ── BATCH setup: partition bindings into atom-dependent vs -independent ───
  // The likelihood term is the costly one (it walks every observation). It is
  // batchable exactly like the IS path: the observed value is atom-independent,
  // and per-atom variation enters only through the latents (and any derived
  // binding that depends on them). So score N points in ONE logDensityN(count=N)
  // pass instead of N scalar calls. The prior term stays scalar (a few cheap
  // logpdfs per atom). `likBodyIR` has derived bindings inlined by clm, so the
  // names it actually references are typically just the draws.
  // Atom-dependent derived = transitive closure of bindings whose lowered RHS
  // references a draw or another atom-dependent name.
  const atomDep = new Set<string>(drawNames);
  let grew = true;
  while (grew) {
    grew = false;
    for (const db of derivedBindings) {
      if (atomDep.has(db.name)) continue;
      const refs = new Set<string>(); collectRefNames(db.ir, refs);
      for (const r of refs) { if (atomDep.has(r)) { atomDep.add(db.name); grew = true; break; } }
    }
  }
  // Atom-independent base env: fixed values + derived that depend only on them,
  // evaluated ONCE. (Atom-dependent values are supplied per call via refArrays.)
  const baseEnvConst: Record<string, any> = Object.assign({}, fixedEnvObj);
  for (const db of derivedBindings) {
    if (atomDep.has(db.name) || db.name in baseEnvConst) continue;
    try { baseEnvConst[db.name] = sampler.evaluateExpr(db.ir, baseEnvConst); } catch (_) { /* skip */ }
  }
  if (moduleReg) baseEnvConst.__moduleRegistry = moduleReg;
  // Which atom-dependent names the likelihood body actually references — only
  // those need a per-atom refArray (avoids replicating unused intermediates).
  const likRefSet = new Set<string>(); collectRefNames(likBodyIR, likRefSet);
  const likPerAtom: string[] = Array.from(likRefSet).filter((n) => atomDep.has(n));
  const constParseSet = makeParseSet(baseEnvConst);

  // Fast-batch eligibility (lets the batched scorer skip per-atom buildEnv —
  // the interpreted re-evaluation of every derived binding, N times, which the
  // breakdown showed dominates BOTH the prior and likelihood terms):
  //   • likPerAtomAllDraws — the likelihood references only draws (derived are
  //     inlined by clm), so its refArrays can be packed straight from the
  //     points, no env.
  //   • priorNeedsDerived — some prior draw measure (or any composite draw)
  //     references an atom-dependent derived value, so the prior needs the full
  //     env; otherwise a lean env (fixed + const-derived + draws) suffices.
  const drawSet = new Set<string>(drawNames);
  const likPerAtomAllDraws = likPerAtom.every((n) => drawSet.has(n));
  let priorNeedsDerived = false;
  for (const nm of drawNames) {
    if (compositeDesc[nm]) { priorNeedsDerived = true; break; }   // scoreComposite needs derived params
    const refs = new Set<string>(); collectRefNames(drawMeasures[nm], refs);
    for (const r of refs) { if (atomDep.has(r) && !drawSet.has(r)) { priorNeedsDerived = true; break; } }
    if (priorNeedsDerived) break;
  }
  // Lean per-atom env: fixed + atom-independent derived + this point's draws,
  // skipping the (expensive) atom-dependent derived evaluation.
  function leanEnvOf(pt: Record<string, any>): Record<string, any> {
    const env: Record<string, any> = Object.assign({}, baseEnvConst);
    for (const k in pt) {
      const v = pt[k];
      if (v instanceof Float64Array) {
        env[k] = Array.from(v);
      } else if (v && v.data instanceof Float64Array && Array.isArray(v.shape) && v.shape.length > 1) {
        // Multi-axis (matrix) latent: keep the Value so the likelihood broadcast
        // sees its full rank. Flattening to v.data would collapse [G,M] to 1 axis.
        env[k] = v;
      } else if (v && v.data instanceof Float64Array) {
        env[k] = Array.from(v.data);   // rank-1 vector Value → plain array
      } else {
        env[k] = v;
      }
    }
    return env;
  }

  // ── SYNC closures ─────────────────────────────────────────────────────────

  // Score a composite-iid draw (a [G,N] block of `distOp` whose per-row params
  // are the broadcast args): logp = Σ_{g,j} logpdf_distOp(val[g·N+j]; params_g).
  // Expand each per-row param across the inner iid axis to a flat length-K
  // vector, then reuse the built-in `broadcast` density path.
  // Unwrap a Value {shape,data} | Float64Array | Array → a flat numeric array.
  function flatData(v: any): any {
    if (v instanceof Float64Array) return v;
    if (v && v.data instanceof Float64Array) return v.data;   // Value
    if (Array.isArray(v)) return v;
    return null;
  }

  function scoreComposite(desc: any, val: any, env: any): number {
    const arr = flatData(val);
    if (!arr) return -Infinity;
    const K = arr.length, D = desc.D | 0;
    if (!(D > 0) || K % D !== 0) return -Infinity;
    const G = K / D;
    const benv: Record<string, any> = {};
    const kwargs: Record<string, any> = {};
    for (const pn of desc.paramNames) {
      if (!(pn in desc.sub)) return -Infinity;
      let pvRaw: any;
      try { pvRaw = sampler.evaluateExpr(desc.sub[pn], env); } catch (_) { return -Infinity; }
      const pv = (typeof pvRaw === 'number') ? pvRaw : flatData(pvRaw);
      const flat = new Float64Array(K);
      if (typeof pv === 'number') { flat.fill(pv); }
      else if (pv && pv.length === G) { for (let g = 0; g < G; g++) { const x = +pv[g]; for (let j = 0; j < D; j++) flat[g * D + j] = x; } }
      else if (pv && pv.length === K) { for (let k = 0; k < K; k++) flat[k] = +pv[k]; }
      else if (pv && pv.length === D) { for (let g = 0; g < G; g++) for (let j = 0; j < D; j++) flat[g * D + j] = +pv[j]; }
      else return -Infinity;
      benv['__cb_' + pn] = { shape: [K], data: flat };
      kwargs[pn] = { kind: 'ref', ns: 'self', name: '__cb_' + pn };
    }
    if (moduleReg) benv.__moduleRegistry = moduleReg;
    const ir = { kind: 'call', op: 'broadcast', args: [{ kind: 'ref', ns: 'self', name: desc.distOp }], kwargs };
    const opts = { parseSet: makeParseSet(benv), resolveMeasureRef: NULL_REF, baseEnv: benv };
    try {
      const r = density.logDensityN(ir, arr, {}, 1, opts);
      const v = r[0];
      return Number.isFinite(v) ? v : -Infinity;
    } catch (_err) { return -Infinity; }
  }

  function priorWith(env: any, pt: Record<string, any>): number {
    const { refArrays } = pointToRefArrays(pt);
    const opts = { parseSet: makeParseSet(env), resolveMeasureRef: NULL_REF, baseEnv: env };
    let total = 0;
    for (const nm of drawNames) {
      const val = pt[nm];
      if (val === undefined) return -Infinity;
      if (compositeDesc[nm]) {
        const v = scoreComposite(compositeDesc[nm], val, env);
        if (!Number.isFinite(v)) return -Infinity;
        total += v;
        continue;
      }
      try {
        const logps = density.logDensityN(drawMeasures[nm], val, refArrays, 1, opts);
        const v = logps[0];
        if (!Number.isFinite(v)) return -Infinity;
        total += v;
      } catch (_err) { return -Infinity; }
    }
    return total;
  }

  function likWith(env: any, pt: Record<string, any>): number {
    const { refArrays } = pointToRefArrays(pt);
    const opts = { parseSet: makeParseSet(env), resolveMeasureRef: NULL_REF, baseEnv: env };
    try {
      const logps = density.logDensityN(likBodyIR, observed, refArrays, 1, opts);
      const v = logps[0];
      return Number.isFinite(v) ? v : -Infinity;
    } catch (_err) { return -Infinity; }
  }

  // One-time tractability probe at an in-support point: lets the density
  // walker's errors propagate (annotated with the draw name) instead of being
  // swallowed to −∞ as priorWith does at runtime. A prior whose density is a
  // static error (e.g. an unannotated pushfwd, spec §06) makes MCMC/AMIS
  // impossible — the backend must refuse with the reason, not run stuck chains.
  function probePrior(pt: Record<string, any>): void {
    const env = buildEnv(pt);
    const { refArrays } = pointToRefArrays(pt);
    const opts = { parseSet: makeParseSet(env), resolveMeasureRef: NULL_REF, baseEnv: env };
    for (const nm of drawNames) {
      if (compositeDesc[nm]) continue;          // composite-iid is scorable
      const val = pt[nm];
      if (val === undefined) continue;
      try {
        density.logDensityN(drawMeasures[nm], val, refArrays, 1, opts);
      } catch (err: any) {
        throw new Error(`prior draw '${nm}' has no tractable density: ${err && err.message ? err.message : String(err)}`);
      }
    }
  }

  function priorOf(pt: Record<string, any>): number { return priorWith(buildEnv(pt), pt); }
  function likOf(pt: Record<string, any>): number { return likWith(buildEnv(pt), pt); }

  function logPi(pt: Record<string, any>): number {
    const env = buildEnv(pt);                 // built ONCE, shared by both terms
    const prior = priorWith(env, pt);
    if (!Number.isFinite(prior)) return -Infinity;
    const lik = likWith(env, pt);
    const lp = prior + lik;
    return Number.isFinite(lp) ? lp : -Infinity;
  }

  // Batched likelihood over N points: when the body references only draws (the
  // common case — clm inlines derived), pack its refArrays straight from the
  // points and score all N in ONE logDensityN(count=N) pass with NO per-atom
  // buildEnv. Returns null when the body needs a derived value (caller falls
  // back to the scalar likWith).
  function batchLik(pts: Array<Record<string, any>>): Float64Array | null {
    const N = pts.length;
    if (!likPerAtomAllDraws) return null;
    const refArrays: Record<string, any> = {};
    for (const nm of likPerAtom) {
      const v0 = pts[0][nm];
      if (typeof v0 === 'number') {
        const a = new Float64Array(N);
        for (let i = 0; i < N; i++) a[i] = +pts[i][nm];
        refArrays[nm] = a;
      } else {
        const d0 = flatData(v0);
        if (!d0) return null;
        const D = d0.length | 0;
        // Per-atom rank: a matrix latent carries shape (e.g. [G,M]); preserve it
        // so the batched value is [N, ...atomShape], not a collapsed [N, D].
        const atomShape = (v0 && Array.isArray(v0.shape) && v0.shape.length > 0) ? v0.shape.slice() : [D];
        const data = new Float64Array(N * D);
        for (let i = 0; i < N; i++) {
          const vi = flatData(pts[i][nm]);
          if (!vi || vi.length !== D) return null;
          for (let j = 0; j < D; j++) data[i * D + j] = +vi[j];
        }
        refArrays[nm] = { shape: [N, ...atomShape], data };
      }
    }
    try {
      const opts = { parseSet: constParseSet, resolveMeasureRef: NULL_REF, baseEnv: baseEnvConst };
      const r = density.logDensityN(likBodyIR, observed, refArrays, N, opts);
      // If any per-atom result is NaN the density walker hit an unresolvable shape
      // conflict (e.g. atom-batched vs atom-indep ambiguity in an atom-indep param)
      // — fall back to the scalar path rather than propagating NaN as −∞.
      if (!r || r.length !== N) return null;
      for (let i = 0; i < N; i++) { if (r[i] !== r[i]) return null; }  // NaN check
      return r as Float64Array;
    } catch (_err) { return null; }
  }

  // Batched prior + likelihood over N points, as SEPARATE per-atom arrays (no
  // Jacobian). Tempering — π_β = prior + β·lik — needs the two terms apart so a
  // population can be re-tempered to a new β by arithmetic on cached `lik`, with
  // no re-evaluation (SMC). Same batched-likelihood / lean-prior path as
  // logPiBatch; non-finite entries are −Infinity.
  function priorLikBatch(pts: Array<Record<string, any>>): { prior: Float64Array; lik: Float64Array } {
    const N = pts.length;
    const prior = new Float64Array(N);
    const lik = new Float64Array(N);
    if (N === 0) return { prior, lik };
    const logL = batchLik(pts);
    for (let i = 0; i < N; i++) {
      // One buildEnv shared by both terms on the fallback path; lean env for the
      // prior when the likelihood was batched and no prior draw needs derived.
      const env = (logL === null || priorNeedsDerived) ? buildEnv(pts[i]) : leanEnvOf(pts[i]);
      const pr = priorWith(env, pts[i]);
      prior[i] = Number.isFinite(pr) ? pr : -Infinity;
      const lk = (logL !== null) ? logL[i] : likWith(env, pts[i]);
      lik[i] = Number.isFinite(lk) ? lk : -Infinity;
    }
    return { prior, lik };
  }

  // Batched logπ over N points = prior + lik. Equals [logPi(p) for p in pts]
  // (verified by batched-scorer.test.ts).
  function logPiBatch(pts: Array<Record<string, any>>): Float64Array {
    const { prior, lik } = priorLikBatch(pts);
    const out = new Float64Array(prior.length);
    for (let i = 0; i < out.length; i++) {
      const lp = prior[i] + lik[i];
      out[i] = Number.isFinite(lp) ? lp : -Infinity;
    }
    return out;
  }

  return { logPi, logPiBatch, priorLikBatch, priorOf, likOf, probePrior };
}

module.exports = { buildLogPi, evaluateDerivedBindings };

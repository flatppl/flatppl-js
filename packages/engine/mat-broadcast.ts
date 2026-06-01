'use strict';

// matKernelBroadcast — array-valued independent-product measure
// `broadcast(Dist, c1, c2, …)` (spec §04 sec:higher-order).
//
// Element j of every atom is drawn from Dist(params_j), where
// params_j is the j-th element of each collection arg (rank-1) or the
// held-constant scalar (rank-0 / length-1 singleton). Result is a
// vector-atom measure shape=[N, K], atom-major, mirroring matIid.
//
// Supports atom-independent params (fixed-phase μ, σ — closed-form
// Normal MvNormal pipeline takes the hot path) AND per-atom params
// (stochastic-ancestor expressions like `means = α .+ β .* x_data` —
// evaluated batched via `evaluateExprN`, then K worker sampleN calls
// with per-atom refArrays carrying each param's atom-i value).
// Symmetric with the density side (`density.walkBroadcast`) which
// dispatches off the same shape contract.

import type {
  DerivationKernelBroadcast,
} from './engine-types';

const empirical    = require('./empirical.ts');
const orchestrator = require('./orchestrator.ts');
const valueLib     = require('./value.ts');
const shared       = require('./materialiser-shared.ts');
const axisStackMod = require('./axis-stack.ts');
// Loading this module registers the built-in KERNEL_BROADCAST_FAST_PATHS
// handlers (Normal etc.) as a side effect. The dispatch helper is
// looked up below before falling through to the general per-cell path.
const fastPaths    = require('./kernel-broadcast-handlers.ts');
// Loading this module registers the built-in COMPOSITE_BODY_RECOGNIZERS
// (iid etc.) as a side effect. matKernelBroadcast dispatches into the
// table when d.distOp resolves to a user kernel binding (i.e. not in
// sampler.REGISTRY); the table grows in Phase 4 with joint /
// jointchain / nested broadcast / vector-per-cell recognizers.
const compositeBodies = require('./composite-body-recognizers.ts');

const { nameSeed, measureFromValue, prepareDensityRefs, pushFixedEnv } = shared;

function matKernelBroadcast(name: string, d: DerivationKernelBroadcast, ctx: any) {
  const sampler = require('./sampler.ts');
  const irShared = require('./ir-shared.ts');
  // Phase 5.1 Session 4: bare vector-output dist case (MvNormal etc.).
  // These distributions sit OUTSIDE the worker's scalar-only sampleN
  // REGISTRY (engine-concepts §22.2(a)); we per-cell dispatch through
  // the bijection registry's atom-batched affine forward — same path
  // matMvNormal uses, but iterated over the outer broadcast cell axis.
  // The branch runs BEFORE the REGISTRY lookup so we don't shadow it
  // with a "not in REGISTRY" composite-body recogniser dispatch.
  if (irShared.VECTOR_OUTPUT_DISTRIBUTIONS
      && irShared.VECTOR_OUTPUT_DISTRIBUTIONS.has(d.distOp)) {
    return _executeBareVectorOutputBroadcast(name, d, ctx);
  }
  let params: string[] = (sampler._internal.REGISTRY[d.distOp] || {}).params;
  // Composite kernel-body case: when d.distOp resolves to a user-
  // defined kernel binding (not a builtin REGISTRY entry), dispatch
  // through the COMPOSITE_BODY_RECOGNIZERS table. Today (Phase 1.2)
  // the table holds only the iid recognizer — `lawof(iid(<BuiltinDist>
  // (<kw>), <n>))` — which unfolds the placeholders and routes through
  // the per-cell sampleN loop with `repeat = n_inner` so each cell
  // produces an iid block. Phase 4 adds joint / jointchain / nested-
  // broadcast / vector-per-cell recognizers; each gets its own
  // execution branch keyed on body.kind.
  let compositeBody: any = null;
  if (!params) {
    compositeBody = compositeBodies.tryRecognizeCompositeBody(d, ctx);
    if (!compositeBody) {
      // Phase 4.5: emit the near-miss diagnostic naming the closest
      // recogniser shape + structural issues that kept the body from
      // matching. Beats the bare "unknown kernel" error today's
      // codepath emitted — users get an actionable pointer to either
      // rewrite the model OR file a precise feature request. The
      // per-atom catch-all walker (slow always-correct path) is
      // deferred to a follow-up; landing it without a motivating
      // fixture would add unused worker / materialiser surface.
      const shapeLib = require('./kernel-broadcast-shape.ts');
      const diag = shapeLib.diagnoseKernelBodyNearMiss(d.distOp, ctx.bindings);
      return Promise.reject(new Error(
        'broadcast: no composite-body recogniser matches kernel \''
        + d.distOp + '\'. ' + diag.message));
    }
    // Phase 4.2: joint-bodied composite — execute via the per-cell
    // multi-component pathway. paramIRs assembly + the per-cell loop
    // live in `_executeJointComposite`; we route there before touching
    // paramIRs since joint's component model differs from iid's single
    // inner-dist model.
    if (compositeBody.kind === 'joint') {
      return _executeJointComposite(name, d, ctx, compositeBody);
    }
    // Phase 4.3: jointchain-bodied composite — execute via the per-cell
    // state-threaded chain walker. Step k's sampleN consumes step k-1's
    // per-atom column as a refArray, so the inner loop is sequential
    // (within a cell). See `_executeJointChainComposite`.
    if (compositeBody.kind === 'jointchain') {
      return _executeJointChainComposite(name, d, ctx, compositeBody);
    }
    // Phase 4.4: nested-broadcast-bodied composite — outer body is
    // itself `broadcast(<bare_dist>, kw)`. Executor walks two per-cell
    // axes (outer j × inner k); see `_executeNestedBroadcastComposite`.
    if (compositeBody.kind === 'nested_broadcast') {
      return _executeNestedBroadcastComposite(name, d, ctx, compositeBody);
    }
    if (compositeBody.kind !== 'iid') {
      return Promise.reject(new Error(
        'broadcast: composite kernel-body kind \'' + compositeBody.kind
        + '\' recognised but not yet executable (Phase 4 work)'));
    }
    params = compositeBody.distParams;
  }
  // Map parameter slot → its IR (kwarg form wins, else positional).
  // For the iid-composite case, the params come from the INNER dist
  // call's kwargs (with kernel-param placeholders substituted by the
  // broadcast args); other kwargs (closed-over self-refs) pass
  // through unchanged.
  const paramIRs: Record<string, any> = {};
  if (compositeBody && compositeBody.kind === 'iid') {
    // Substitute kernel placeholders in distCall's kwargs with the
    // broadcast args (d.kwargIRs maps surface kw → broadcast arg IR).
    for (const pn of Object.keys(compositeBody.distKwargs)) {
      paramIRs[pn] = _substituteKernelParams(
        compositeBody.distKwargs[pn],
        compositeBody.params, compositeBody.paramKwargs, d.kwargIRs || {});
    }
  } else if (d.kwargIRs && Object.keys(d.kwargIRs).length > 0) {
    for (const pn of params) {
      if (Object.prototype.hasOwnProperty.call(d.kwargIRs, pn)) paramIRs[pn] = d.kwargIRs[pn];
    }
    for (const pn of Object.keys(d.kwargIRs)) {
      if (paramIRs[pn] === undefined) paramIRs[pn] = d.kwargIRs[pn];
    }
  } else if (Array.isArray(d.argIRs)) {
    if (d.argIRs.length !== params.length) {
      return Promise.reject(new Error('broadcast(' + d.distOp + '): expected '
        + params.length + ' parameter args (' + params.join(', ')
        + '), got ' + d.argIRs.length));
    }
    for (let i = 0; i < params.length; i++) paramIRs[params[i]] = d.argIRs[i];
  }
  // Effective distOp for runtime worker calls. In the composite case
  // we sample from the inner builtin distribution (e.g. Normal) with
  // the substituted kwargs; the outer "iid" structure is realised by
  // the per-cell `repeat = n` argument to sampleN.
  const effectiveDistOp = (compositeBody && compositeBody.kind === 'iid')
    ? compositeBody.distOp : d.distOp;
  const iidN = (compositeBody && compositeBody.kind === 'iid')
    ? compositeBody.n : 1;

  // Phase 4.1 fast-detect: iid-composite vec-per-cell case.
  //
  // If the kernel binding is iid-composite AND any BROADCAST ARG has
  // shape [K, D] (atom-indep) or [N, K, D] (atom-batched) where D ==
  // iidN, then the body's per-cell parameters are length-D vectors —
  // each iid component within a cell uses a DIFFERENT param value.
  // The existing per-cell loop's `repeat = iidN` assumes shared
  // per-cell params across the iid axis; that's not valid here.
  //
  // Branch to the new `_executeIidCompositeVecPerCell` path BEFORE
  // batched evaluation of paramIRs[*] (which would otherwise fail —
  // the substituted body mixes atom-batched scalar refs with rank-2
  // atom-indep refs, and `value-ops.addN` rejects the per-atom shape
  // mismatch). The new path substitutes broadcast args per (j, r) so
  // every leaf in the body resolves to per-atom scalar before
  // evaluation.
  if (compositeBody && compositeBody.kind === 'iid'
      && _shouldUseVecPerCell(d, ctx, iidN)) {
    return _executeIidCompositeVecPerCell(name, d, ctx, compositeBody, paramIRs);
  }
  const pnames = Object.keys(paramIRs);
  // Synthesize a containing IR so prepareDensityRefs collects every
  // free name across all parameter expressions in one pass.
  const aggregateIR: any = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: d.distOp }],
    kwargs: paramIRs,
  };
  return prepareDensityRefs(aggregateIR, ctx, 'broadcast').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const baseEnv: Record<string, any> = {};
    for (const k in fixedEnv) baseEnv[k] = fixedEnv[k];
    const refNames = Object.keys(refArrays);
    const N = ctx.sampleCount;
    // Evaluate each parameter expression once over the atom batch.
    // The shape disambiguates: per-atom vs atom-indep; scalar vs
    // length-K vector. Mirrors density.walkBroadcast's classification.
    //
    // **P4 (advisory):** if the binding's IR carries an axisStack
    // entry from kernel-broadcast (`source: 'kernel_broadcast'` with
    // a literal-integer size), we initialise K from it — bypassing
    // the runtime shape ladder for the K determination. The ladder
    // still runs (it also enforces per-param shape consistency); the
    // axisStack only seeds an *expected* K that the ladder validates
    // against. When the axisStack is absent or symbolic, behaviour is
    // identical to today's (the ladder discovers K from the first
    // param with intrinsicK > 1).
    const _bindingStack = axisStackMod.bindingAxisStack(name, ctx);
    const _axisStackK = axisStackMod.outerAxisSize(_bindingStack, 'kernel_broadcast');
    const usesAtomBy: Record<string, boolean> = {};
    const paramVals: Record<string, any> = {};
    let K = (_axisStackK && _axisStackK > 1) ? _axisStackK : 1;
    for (const pn of pnames) {
      const paramRefs = orchestrator.collectSelfRefs(paramIRs[pn]);
      const usesAtom = refNames.some((n) => paramRefs.has(n));
      usesAtomBy[pn] = usesAtom;
      let v: any;
      try {
        v = sampler.evaluateExprN(paramIRs[pn], refArrays, N, baseEnv, undefined);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
      paramVals[pn] = v;
      // Determine intrinsic K from this param (the non-batch axis).
      let intrinsicK = 1;
      if (typeof v === 'number' || typeof v === 'boolean') {
        intrinsicK = 1;
      } else if (valueLib.isValue(v)) {
        const shape = v.shape;
        if (shape.length === 0) intrinsicK = 1;
        else if (shape.length === 1) {
          // shape=[N] is atom-batched scalar (K=1) ONLY when this param
          // references an atom-batched ref (`usesAtom`); a stand-alone
          // length-N vector that doesn't ref any atom-batched ref is
          // K=N (a true length-N collection arg). The canonical
          // `value.isAtomBatched` would flag both as "shape match";
          // here the name-based guard is the disambiguator.
          intrinsicK = (usesAtom && valueLib.isAtomBatched(v, N)) ? 1 : shape[0];
        } else if (valueLib.isAtomBatched(v, N) && shape.length === 2) {
          intrinsicK = shape[1];
        } else {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): parameter \'' + pn + '\' resolved to unsupported shape '
            + JSON.stringify(shape) + ' (v1 supports scalar / [K] / [N] / [N, K])'));
        }
      } else if (v && v.BYTES_PER_ELEMENT !== undefined
                  && typeof v.length === 'number') {
        intrinsicK = (usesAtom && valueLib.isAtomBatched(v, N)) ? 1 : v.length;
      } else if (Array.isArray(v)) {
        intrinsicK = v.length;
      } else {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): parameter \'' + pn + '\' resolved to non-numeric value'));
      }
      if (intrinsicK > 1) {
        if (K === 1) K = intrinsicK;
        else if (K !== intrinsicK) {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): incompatible collection lengths (' + K + ' vs '
            + intrinsicK + ') on parameter \'' + pn + '\''));
        }
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty collection argument'));
    }
    const anyAtomDep = pnames.some((pn) => usesAtomBy[pn]);

    // CLOSED-FORM FAST PATHS via the KERNEL_BROADCAST_FAST_PATHS
    // registry (kernel-broadcast-handlers.ts). Each per-distOp handler
    // declares its own match predicate (typically: atom-indep params,
    // specific param presence) and execute body. Returns null if no
    // registered handler matches; falls through to the general per-
    // cell path below. See kernel-broadcast-handlers.ts for the
    // registry surface + currently-registered handlers (Normal as of
    // Phase 1.1).
    const fastResult = fastPaths.tryKernelBroadcastFastPath({
      d, ctx, name, K, N, paramVals, anyAtomDep,
    });
    if (fastResult) return fastResult;

    // GENERAL PATH: K worker sampleN calls, each with per-atom
    // refArrays carrying the (i, j) value of each per-atom parameter.
    // Atom-indep params pass as lit kwargs. Param-shape contract
    // (scalar / [K] / [N] / [N, K]) and the per-atom column / scalar
    // extractors live in materialiser-shared so the density side and
    // any other per-atom materialiser path can share them.
    //
    // For the iid-composite case (Phase F): each cell's sampleN
    // additionally uses `repeat: iidN` to produce iidN samples per
    // atom per cell. Result per cell: shape=[N, iidN]; stacked across
    // K cells: shape=[N, K, iidN] atom-major.
    const cols = new Array(K);
    let chain = Promise.resolve();
    for (let j = 0; j < K; j++) {
      const jj = j;
      chain = chain.then(() => {
        const kwargs: Record<string, any> = {};
        const perAtomRefs: Record<string, any> = {};
        let safeIdx = 0;
        for (const pn of pnames) {
          if (usesAtomBy[pn]) {
            const refName = '__bc_' + pn + '_' + (safeIdx++);
            try {
              const col = shared.perAtomColumnAtJ(paramVals[pn], jj, N);
              perAtomRefs[refName] = valueLib.batchedScalar(col);
            } catch (err) {
              throw err instanceof Error ? err : new Error(String(err));
            }
            kwargs[pn] = { kind: 'ref', ns: 'self', name: refName };
          } else {
            kwargs[pn] = { kind: 'lit', value: shared.elemScalarAtJ(paramVals[pn], jj, N) };
          }
        }
        const distIR = { kind: 'call', op: effectiveDistOp, kwargs: kwargs };
        const workerMsg: any = {
          type: 'sampleN', ir: distIR, count: N,
          refArrays: perAtomRefs,
          seed: nameSeed(name + ':' + jj, ctx.rootKey),
        };
        if (iidN > 1) workerMsg.repeat = iidN;
        return ctx.sendWorker(workerMsg).then((reply: any) => {
          cols[jj] = reply.samples;
        });
      });
    }
    return pushFixedEnv(ctx, fixedEnv).then(() => chain).then(() => {
      // Non-iid case: out shape [N, K] atom-major. Iid case: out
      // shape [N, K, iidN] atom-major — for each (atom, cell), pack
      // the iidN per-cell samples consecutively.
      if (iidN > 1) {
        const out = new Float64Array(N * K * iidN);
        // cols[j] is a flat Float64Array of length N * iidN (sampleN
        // with count=N + repeat=iidN packs N atoms × iidN samples
        // atom-major: cols[j][i * iidN + r] = atom i, sample r).
        for (let j = 0; j < K; j++) {
          const col = cols[j];
          for (let i = 0; i < N; i++) {
            for (let r = 0; r < iidN; r++) {
              out[i * K * iidN + j * iidN + r] = col[i * iidN + r];
            }
          }
        }
        const value = { shape: [N | 0, K, iidN], data: out };
        return Object.assign(
          empirical.arrayMeasure(out, [K, iidN], null),
          { value: value, logTotalmass: 0, n_eff: N },
        );
      }
      const out = new Float64Array(N * K);
      for (let j = 0; j < K; j++) {
        const col = cols[j];
        for (let i = 0; i < N; i++) out[i * K + j] = col[i];
      }
      const value = { shape: [N | 0, K], data: out };
      return Object.assign(
        empirical.arrayMeasure(out, [K], null),
        { value: value, logTotalmass: 0, n_eff: N },
      );
    });
  });
}

// =====================================================================
// Kernel parameter substitution
// =====================================================================
//
// Composite kernel-body recognition has moved to the
// COMPOSITE_BODY_RECOGNIZERS table in composite-body-recognizers.ts
// (which wraps the structural detectors in kernel-broadcast-shape.ts).
// matKernelBroadcast above dispatches through that table; on a hit it
// uses _substituteKernelParams below to bind the broadcast args into
// the inner dist call's kwargs.

// Substitute kernel placeholders in an IR expression with the
// corresponding broadcast arg from `bcKwargs`. Matches BOTH `%local.
// <paramName>` (the canonical placeholder form) AND `self.<paramName>`
// when the name appears in the kernel's params (the lift pass strips
// %local when hoisting inline measure calls to anon bindings —
// inside the anon, placeholder refs come out as `self.<paramName>`
// referencing a non-existent module binding; the substitution
// rebinds them to the broadcast args). Other refs (closed-over
// self-refs from the kernel definition's outer scope) and literals
// pass through unchanged.
function _substituteKernelParams(
  expr: any, params: string[], paramKwargs: string[], bcKwargs: any,
): any {
  if (!expr || typeof expr !== 'object') return expr;
  if (expr.kind === 'ref'
      && (expr.ns === '%local' || expr.ns === 'self')) {
    const idx = params.indexOf(expr.name);
    if (idx < 0) return expr;
    const surfaceKw = paramKwargs[idx];
    if (Object.prototype.hasOwnProperty.call(bcKwargs, surfaceKw)) {
      return bcKwargs[surfaceKw];
    }
    return expr;
  }
  if (expr.kind === 'call') {
    const out: any = { kind: 'call' };
    if (expr.op) out.op = expr.op;
    if (expr.target) out.target = expr.target;
    if (Array.isArray(expr.args)) {
      out.args = expr.args.map((a: any) =>
        _substituteKernelParams(a, params, paramKwargs, bcKwargs));
    }
    if (expr.kwargs) {
      out.kwargs = {};
      for (const k in expr.kwargs) {
        out.kwargs[k] = _substituteKernelParams(
          expr.kwargs[k], params, paramKwargs, bcKwargs);
      }
    }
    if (expr.op === 'functionof') {
      if (expr.body) out.body = expr.body;
      if (expr.params) out.params = expr.params;
      if (expr.paramKwargs) out.paramKwargs = expr.paramKwargs;
    }
    if (expr.loc) out.loc = expr.loc;
    return out;
  }
  return expr;
}

// =====================================================================
// Phase 4.1 — vec-per-cell iid-composite execution
// =====================================================================
//
// When a kernel-broadcast wraps `kernelof(iid(<Dist>, n), …)` and at
// least one broadcast arg is rank-2 atom-indep `[K, D]` or rank-3
// atom-batched `[N, K, D]` (with D == iidN), the per-cell parameter
// expression — e.g. `intercept + slope .* x_group` in random-
// intercepts — evaluates to a length-D vector per cell. The inner
// iid axis IS the vector D axis: each iid component within a cell
// uses a DIFFERENT param value (varies along r ∈ [0, D)).
//
// The existing per-cell loop in `matKernelBroadcast` pre-evaluates
// the substituted body in batched mode and then slices per cell.
// That fails on these inputs because the substituted body's batched
// evaluation mixes atom-batched scalar refs `[N]` with rank-2 atom-
// indep refs `[K, D]`, and `value-ops.addN` doesn't broadcast across
// such mixed shapes.
//
// The vec-per-cell path inverts the order: substitute per (j, r)
// FIRST (so each broadcast arg becomes a per-atom scalar or a
// literal), then evaluate the substituted body in batched mode. All
// the body's arithmetic reduces to per-atom-scalar operations —
// shapes that `evaluateExprN` handles cleanly.
//
// Cost: K * D worker sampleN calls instead of K calls with
// `repeat = D`. For typical hierarchical models (K, D small; N >> 1)
// the overhead is negligible.

// Should we route to the vec-per-cell path? True iff ANY broadcast
// arg classifies as `KD` (atom-indep [K, D]) or `NKD` (atom-batched
// [N, K, D]) where D == iidN. We do a CHEAP shape probe (evaluate
// each broadcast arg once, classify) — paramVals get reused inside
// `_executeIidCompositeVecPerCell` so the cost amortises.
function _shouldUseVecPerCell(d: any, ctx: any, iidN: number): boolean {
  if (iidN <= 1) return false;
  if (!d.kwargIRs) return false;
  const sampler = require('./sampler.ts');
  const N = ctx.sampleCount;
  // Inspect each broadcast arg's inferred shape via a quick type
  // probe — read the binding's inferred type when the arg is a
  // simple `self`-ref, falling back to a value-eval if needed.
  // We're conservative: only fire when at least one arg has a
  // statically observable rank-2 (atom-indep) or rank-3 (atom-
  // batched) shape whose last dim equals iidN.
  for (const argName of Object.keys(d.kwargIRs)) {
    const argIR = d.kwargIRs[argName];
    if (!argIR || argIR.kind !== 'ref' || argIR.ns !== 'self') continue;
    if (!ctx.bindings || !ctx.bindings.has(argIR.name)) continue;
    const b = ctx.bindings.get(argIR.name);
    const t = b && b.inferredType;
    if (!t || t.kind !== 'array' || !Array.isArray(t.shape)) continue;
    const shape = t.shape;
    if (shape.length === 2 && shape[1] === iidN) return true;
    if (shape.length === 3 && shape[2] === iidN) return true;
  }
  // Fallback: probe by evaluating each broadcast arg once and
  // inspecting the runtime shape. This catches cases where typeinfer
  // hasn't populated the full shape (or where evaluateExprN runs a
  // shape-folding pass that produces a more-precise concrete shape
  // than the static type carries).
  //
  // We don't reuse these evaluations here — `_executeIidComposite-
  // VecPerCell` re-evaluates them through `prepareDensityRefs` so
  // worker / fixedEnv plumbing stays in one place. This duplicate
  // eval is bounded by the per-broadcast-arg cost (small ops) and
  // only fires on the iid-composite path; fine for now, can be
  // memoised later if it shows in profiles.
  for (const argName of Object.keys(d.kwargIRs)) {
    try {
      const argIR = d.kwargIRs[argName];
      const refs = require('./orchestrator.ts').collectSelfRefs(argIR);
      const fixedValues = ctx.fixedValues || new Map();
      const env: Record<string, any> = {};
      for (const refName of refs) {
        if (fixedValues.has && fixedValues.has(refName)) {
          env[refName] = fixedValues.get(refName);
        }
      }
      const v = sampler.evaluateExpr(argIR, env);
      if (v && Array.isArray(v.shape)) {
        const shape = v.shape;
        if (shape.length === 2 && shape[1] === iidN) return true;
        if (shape.length === 3 && shape[2] === iidN) return true;
      }
    } catch (_) {
      // Couldn't evaluate this arg statically — fall through. The
      // existing batched path will run and either succeed (scalar-
      // per-cell case) or surface a clearer error.
    }
  }
  return false;
}

function _executeIidCompositeVecPerCell(
  name: string, d: any, ctx: any, compositeBody: any,
  paramIRs: Record<string, any>,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const N = ctx.sampleCount;
  const iidN = compositeBody.n;

  // Build an aggregateIR that mirrors the existing scalar-per-cell
  // path: `broadcast(<distOp_ref>, paramIRs)` where `paramIRs` is
  // the body kwargs WITH kernel placeholders pre-substituted to the
  // broadcast args' IRs (built by matKernelBroadcast above). This
  // surfaces every closed-over self-ref (`slope`, `sigma`, …) AND
  // every broadcast arg ref (`intercepts`, `x_per_group`, …) to
  // `prepareDensityRefs`'s collectSelfRefs walk — yielding a
  // refArrays map populated with all upstream values.
  //
  // The per-(j, r) loop below RE-substitutes the body kwargs with
  // cell-sliced bcKwargs (a literal scalar for atom-indep args; a
  // ref to a `__bc_<arg>_j_r` per-atom column for atom-batched args)
  // so each iteration evaluates with concrete per-atom-scalar shapes.
  const aggregateIR: any = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: d.distOp }],
    kwargs: paramIRs,
  };

  return prepareDensityRefs(aggregateIR, ctx, 'broadcast').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const baseEnv: Record<string, any> = {};
    for (const k in fixedEnv) baseEnv[k] = fixedEnv[k];
    const refNames = Object.keys(refArrays);

    // Evaluate each broadcast arg once to determine shape + atom-dep.
    const argVals: Record<string, any> = {};
    const argClass: Record<string, any> = {};
    let K = 1;
    for (const argName of Object.keys(d.kwargIRs)) {
      const argIR = d.kwargIRs[argName];
      const argRefs = orchestrator.collectSelfRefs(argIR);
      const usesAtom = refNames.some((n) => argRefs.has(n));
      let v: any;
      try {
        v = sampler.evaluateExprN(argIR, refArrays, N, baseEnv, undefined);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
      argVals[argName] = v;
      const cls = shared.classifyBroadcastArg(v, N, usesAtom);
      argClass[argName] = cls;
      const argK = cls.K || 1;
      if (argK > 1) {
        if (K === 1) K = argK;
        else if (K !== argK) {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): incompatible cell-axis lengths (' + K + ' vs '
            + argK + ') on arg \'' + argName + '\''));
        }
      }
      const argD = cls.D || 1;
      if (argD > 1 && argD !== iidN) {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): vec-per-cell dim ' + argD + ' on arg \''
          + argName + '\' must equal the inner iid count ' + iidN));
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty collection argument'));
    }

    // Per (j, r) loop: substitute each broadcast arg with its
    // (j, r) per-atom slice and evaluate the substituted body
    // kwargs in batched mode. Each substituted kwarg evaluates to
    // a per-atom scalar Value or a plain scalar; worker sampleN
    // takes those as the inner builtin dist's params.
    const cells = new Array(K * iidN);
    let chain = Promise.resolve();
    for (let j = 0; j < K; j++) {
      for (let r = 0; r < iidN; r++) {
        const jj = j, rr = r;
        chain = chain.then(() => {
          // Build per-(j, r) bcKwargs: each surface kwarg maps to
          // either a lit scalar (atom-indep) or a ref to a per-atom
          // column refArray (atom-batched).
          const bcKwargs: Record<string, any> = {};
          const perAtomRefs: Record<string, any> = {};
          for (const argName of Object.keys(d.kwargIRs)) {
            const cls = argClass[argName];
            const v = argVals[argName];
            const isAtomDep = cls.kind === 'N' || cls.kind === 'NK'
                              || cls.kind === 'NKD';
            if (isAtomDep) {
              const col = shared.perAtomColumnAtJR(v, jj, rr, N);
              const refName = '__bc_' + argName + '_' + jj + '_' + rr;
              perAtomRefs[refName] = valueLib.batchedScalar(col);
              bcKwargs[argName] = { kind: 'ref', ns: 'self', name: refName };
            } else {
              const s = shared.elemScalarAtJR(v, jj, rr, N);
              bcKwargs[argName] = { kind: 'lit', value: s };
            }
          }

          // Substitute the kernel placeholders in the body's kwargs
          // and evaluate each substituted body-kwarg in batched mode.
          // The result is per-atom scalar — we pass it to the worker
          // either as a `lit` (closed-form scalar) or as a fresh
          // `__body_<pn>_j_r` refArray (atom-batched scalar). The
          // worker's sampleN re-evaluates the IR; we resolve the
          // per-atom value up-front so the IR is just a ref/lit.
          const distKwargs: Record<string, any> = {};
          for (const pn of Object.keys(compositeBody.distKwargs)) {
            const subIR = _substituteKernelParams(
              compositeBody.distKwargs[pn],
              compositeBody.params, compositeBody.paramKwargs, bcKwargs);
            // Batched-evaluate the substituted body kwarg. We extend
            // refArrays with the per-(j, r) cell refs so the body's
            // refs to the synthetic __bc_*_j_r names resolve.
            const cellRefArrays: Record<string, any> = {};
            for (const k in refArrays) cellRefArrays[k] = refArrays[k];
            for (const k in perAtomRefs) cellRefArrays[k] = perAtomRefs[k];
            let bodyVal: any;
            try {
              bodyVal = sampler.evaluateExprN(
                subIR, cellRefArrays, N, baseEnv, undefined);
            } catch (err) {
              throw err instanceof Error ? err : new Error(String(err));
            }
            // Convert the per-atom scalar Value into a worker kwarg:
            //   - rank-0 Value or plain number → lit scalar
            //   - atom-batched scalar [N] → ref entry
            //   - JS array of length N (per-atom fallback result) → ref
            //     entry; the engine's batched `broadcast` cold path
            //     returns a per-atom array rather than packing into a
            //     flat Value, so we re-pack here.
            if (typeof bodyVal === 'number' || typeof bodyVal === 'boolean') {
              distKwargs[pn] = { kind: 'lit', value: +bodyVal };
            } else if (valueLib.isValue(bodyVal)) {
              const bs = bodyVal.shape;
              if (bs.length === 0) {
                distKwargs[pn] = { kind: 'lit', value: bodyVal.data[0] };
              } else if (bs.length === 1 && bs[0] === N) {
                const refName = '__body_' + pn + '_' + jj + '_' + rr;
                perAtomRefs[refName] = valueLib.batchedScalar(bodyVal.data);
                distKwargs[pn] = { kind: 'ref', ns: 'self', name: refName };
              } else if (bs.length === 1 && bs[0] === 1) {
                // Singleton-vector eval result; treat as scalar.
                distKwargs[pn] = { kind: 'lit', value: bodyVal.data[0] };
              } else {
                throw new Error('broadcast(' + d.distOp
                  + '): vec-per-cell body kwarg \'' + pn
                  + '\' evaluated to unsupported shape '
                  + JSON.stringify(bs) + ' at (j=' + jj + ',r=' + rr + ')');
              }
            } else if (bodyVal && bodyVal.BYTES_PER_ELEMENT !== undefined
                       && bodyVal.length === N) {
              const refName = '__body_' + pn + '_' + jj + '_' + rr;
              perAtomRefs[refName] = valueLib.batchedScalar(bodyVal);
              distKwargs[pn] = { kind: 'ref', ns: 'self', name: refName };
            } else if (Array.isArray(bodyVal) && bodyVal.length === N) {
              // Per-atom fallback result: array of per-atom values
              // (numbers OR rank-0 Values). Pack into a flat
              // Float64Array(N).
              const flat = new Float64Array(N);
              for (let i = 0; i < N; i++) {
                const cell = bodyVal[i];
                if (typeof cell === 'number') flat[i] = cell;
                else if (typeof cell === 'boolean') flat[i] = cell ? 1 : 0;
                else if (valueLib.isValue(cell) && cell.shape.length === 0) {
                  flat[i] = cell.data[0];
                } else if (valueLib.isValue(cell) && cell.shape.length === 1
                           && cell.shape[0] === 1) {
                  flat[i] = cell.data[0];
                } else {
                  throw new Error('broadcast(' + d.distOp
                    + '): vec-per-cell body kwarg \'' + pn
                    + '\' produced per-atom non-scalar at (j='
                    + jj + ',r=' + rr + ',i=' + i + ')');
                }
              }
              const refName = '__body_' + pn + '_' + jj + '_' + rr;
              perAtomRefs[refName] = valueLib.batchedScalar(flat);
              distKwargs[pn] = { kind: 'ref', ns: 'self', name: refName };
            } else {
              throw new Error('broadcast(' + d.distOp
                + '): vec-per-cell body kwarg \'' + pn
                + '\' evaluated to unexpected type at (j='
                + jj + ',r=' + rr + ')');
            }
          }

          const distIR = {
            kind: 'call', op: compositeBody.distOp, kwargs: distKwargs,
          };
          const workerMsg: any = {
            type: 'sampleN', ir: distIR, count: N,
            refArrays: perAtomRefs,
            seed: nameSeed(name + ':' + jj + ':' + rr, ctx.rootKey),
          };
          return ctx.sendWorker(workerMsg).then((reply: any) => {
            cells[jj * iidN + rr] = reply.samples;
          });
        });
      }
    }
    return pushFixedEnv(ctx, fixedEnv).then(() => chain).then(() => {
      // Stitch results: shape=[N, K, iidN] atom-major.
      const out = new Float64Array(N * K * iidN);
      for (let j = 0; j < K; j++) {
        for (let r = 0; r < iidN; r++) {
          const col = cells[j * iidN + r];
          for (let i = 0; i < N; i++) {
            out[i * K * iidN + j * iidN + r] = col[i];
          }
        }
      }
      const value = { shape: [N | 0, K, iidN], data: out };
      return Object.assign(
        empirical.arrayMeasure(out, [K, iidN], null),
        { value: value, logTotalmass: 0, n_eff: N },
      );
    });
  });
}

// =====================================================================
// Phase 4.2 — joint-composite execution
// =====================================================================
//
// When a kernel-broadcast wraps `kernelof(joint(<components>), …)`,
// each cell of the outer broadcast produces ONE draw from the joint
// product measure. Per spec §06 the components are independent: each
// produces its own variate slice and the joint variate is the
// concat-vector (positional layout) or named record (keyword layout)
// of those slices.
//
// Phase 4.2 MVP restricts each component to a built-in sampleable
// scalar distribution (Normal, Beta, Gamma, …). Vector-valued
// components (MvNormal) wait for Phase 5.1, where MvNormal joins the
// sampler REGISTRY so the same kwarg-driven sampleN dispatch handles
// it uniformly.
//
// Execution shape per (j, c):
//   1. Build per-cell `bcKwargs` — atom-indep broadcast args become
//      `lit` kwargs; atom-batched args become per-atom refArrays
//      (`__bc_<argName>_<j>`).
//   2. Substitute kernel placeholders in component c's `distKwargs`
//      with those bcKwargs (via `_substituteKernelParams`).
//   3. Send one sampleN worker call for component c with the
//      substituted IR; the worker returns a length-N column of per-
//      atom scalars.
//
// Stitching: out[i * K * C + j * C + c] = col_jc[i], where
// C = components.length. Result Value shape = [N, K, C] atom-major,
// matching the iid-composite layout for downstream broadcast / density
// consumers.
//
// Cost: K * C worker calls. For typical multi-output regressions
// (K small, C small, N large) the per-cell sampling dominates, not
// the dispatch overhead.

function _executeJointComposite(
  name: string, d: any, ctx: any, compositeBody: any,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const N = ctx.sampleCount;
  const components = compositeBody.components;
  const C = components.length;

  // Synthesize a containing IR so prepareDensityRefs sees every free
  // name across all components' substituted kwargs PLUS every
  // broadcast arg. We pre-substitute the kernel placeholders in each
  // component's kwargs so the aggregateIR closes over the broadcast
  // args (not the kernel-local placeholder names like `mu` that
  // collectSelfRefs would otherwise try — and fail — to materialise).
  const substComponentKwargs: Array<Record<string, any>> = [];
  for (const comp of components) {
    const subst: Record<string, any> = {};
    for (const pn of Object.keys(comp.distKwargs)) {
      subst[pn] = _substituteKernelParams(
        comp.distKwargs[pn],
        compositeBody.params, compositeBody.paramKwargs, d.kwargIRs || {});
    }
    substComponentKwargs.push(subst);
  }
  // Aggregate the substituted kwargs into a single IR for refArrays
  // discovery. Wrap as a broadcast(...) so the structure mirrors what
  // the iid-composite executor passes — both consumers of
  // `prepareDensityRefs` see the same shape.
  const flatKwargs: Record<string, any> = {};
  for (let c = 0; c < C; c++) {
    for (const pn of Object.keys(substComponentKwargs[c])) {
      flatKwargs['__joint_c' + c + '_' + pn] = substComponentKwargs[c][pn];
    }
  }
  const aggregateIR: any = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: d.distOp }],
    kwargs: flatKwargs,
  };

  return prepareDensityRefs(aggregateIR, ctx, 'broadcast').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const baseEnv: Record<string, any> = {};
    for (const k in fixedEnv) baseEnv[k] = fixedEnv[k];
    const refNames = Object.keys(refArrays);

    // Evaluate each broadcast arg once to determine cell-axis length K
    // and atom-dep classification. This mirrors the iid-composite
    // vec-per-cell pathway; the same `classifyBroadcastArg` shape
    // contract applies — joint just doesn't have an inner D axis (each
    // component yields a scalar variate in Phase 4.2 scope).
    const argVals: Record<string, any> = {};
    const argClass: Record<string, any> = {};
    let K = 1;
    for (const argName of Object.keys(d.kwargIRs || {})) {
      const argIR = d.kwargIRs[argName];
      const argRefs = orchestrator.collectSelfRefs(argIR);
      const usesAtom = refNames.some((n) => argRefs.has(n));
      let v: any;
      try {
        v = sampler.evaluateExprN(argIR, refArrays, N, baseEnv, undefined);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
      argVals[argName] = v;
      // Pass usesAtom hint so a stand-alone length-N atom-indep vector
      // (rare for joint args but legal) doesn't get mis-classified as
      // atom-batched scalar.
      const cls = shared.classifyBroadcastArg(v, N, usesAtom);
      argClass[argName] = cls;
      const argK = cls.K || 1;
      if (argK > 1) {
        if (K === 1) K = argK;
        else if (K !== argK) {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): incompatible cell-axis lengths (' + K + ' vs '
            + argK + ') on arg \'' + argName + '\''));
        }
      }
      // Inner-vector args (D > 1) used to be rejected as "vec-per-cell
      // iid territory". Phase 5.1 Session 5a admits them when ANY
      // joint component is vector-output: an MvNormal component's
      // `mu` parameter IS a per-cell n-vector, not a scalar. The
      // per-cell slicing below builds a literal `vector(...)` IR for
      // such args at each j, so the substituted MvNormal kwargs end
      // up as `mu = [m_j0, m_j1, …]` (closed at runtime).
      const argD = cls.D || 1;
      const anyVectorComponent = components.some((c: any) => c.isVectorOutput);
      if (argD > 1 && !anyVectorComponent) {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): joint-composite broadcast arg \'' + argName
          + '\' has inner dim ' + argD + ' (vec-per-cell shapes are '
          + 'iid-composite territory unless a vector-output component '
          + 'consumes them, which this joint kernel has none of)'));
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty collection argument'));
    }

    // Per (j, c) loop: build bcKwargs cell-slice, re-substitute the
    // component's distKwargs with the cell-sliced bcKwargs, then
    // sample. Scalar components go through the worker's sampleN
    // (Float64Array(N) of per-atom scalars). Vector-output components
    // (Phase 5.1 Session 5a) materialise via _sampleVectorOutputAtCell,
    // which consumes the bijection registry directly — returning a
    // Float64Array(N * eventDim) of per-atom vectors.
    //
    // Each per-cell cell entry carries `{data: Float64Array, eventDim}`
    // so the stitcher knows how to splat. Component event-dims are
    // resolved at execute-time here (not classify-time) because the
    // dim depends on the substituted-kwargs Value shape, not just the
    // surface IR.
    type JointCell = { data: Float64Array; eventDim: number };
    const cells: JointCell[] = new Array(K * C);
    let chain = Promise.resolve();
    for (let j = 0; j < K; j++) {
      for (let c = 0; c < C; c++) {
        const jj = j, cc = c;
        chain = chain.then(() => {
          const bcKwargs: Record<string, any> = {};
          const perAtomRefs: Record<string, any> = {};
          for (const argName of Object.keys(d.kwargIRs || {})) {
            const cls = argClass[argName];
            const v = argVals[argName];
            const argD = cls.D || 1;
            const isAtomDep = cls.kind === 'N' || cls.kind === 'NK'
                              || cls.kind === 'NKD';
            if (isAtomDep) {
              if (argD > 1) {
                // Atom-dep per-cell vector — defers to Session 5+
                // alongside matPushfwd vector-base (atom-batched
                // affine forward needs walker symmetry).
                throw new Error('joint composite: atom-dep per-cell '
                  + 'vector broadcast args (' + argName
                  + ' with D=' + argD + ') deferred to Session 5+; '
                  + 'use atom-indep mu / cov in Session 5a scope');
              }
              const col = shared.perAtomColumnAtJ(v, jj, N);
              const refName = '__bc_' + argName + '_' + jj;
              perAtomRefs[refName] = valueLib.batchedScalar(col);
              bcKwargs[argName] = { kind: 'ref', ns: 'self', name: refName };
            } else if (argD > 1) {
              // Atom-indep per-cell vector — extract v[jj, :] as a
              // literal `vector(…)` IR so the substituted MvNormal
              // kwarg becomes a concrete n-vector. v.shape is [K, D]
              // or [D] (singleton K=1 broadcast).
              const baseOff = (v.shape && v.shape.length === 2)
                ? jj * argD : 0;
              const litArgs: any[] = [];
              for (let k = 0; k < argD; k++) {
                litArgs.push({ kind: 'lit', value: v.data[baseOff + k] });
              }
              bcKwargs[argName] = { kind: 'call', op: 'vector', args: litArgs };
            } else {
              const s = shared.elemScalarAtJ(v, jj, N);
              bcKwargs[argName] = { kind: 'lit', value: s };
            }
          }

          const comp = components[cc];
          const distKwargs: Record<string, any> = {};
          for (const pn of Object.keys(comp.distKwargs)) {
            distKwargs[pn] = _substituteKernelParams(
              comp.distKwargs[pn],
              compositeBody.params, compositeBody.paramKwargs, bcKwargs);
          }

          if (comp.isVectorOutput) {
            // Per-cell vector-output dispatch through the registry.
            // The substituted distKwargs are now closed expressions
            // resolvable through the standard orchestrator helpers
            // (atom-dep refs end up in perAtomRefs — Session 5a's
            // shared-cov / closed-mu MVP scope rejects atom-dep here
            // because the affine forward needs an atom-indep L per
            // cell; per-atom-stochastic mu/cov defers to Session 5+
            // alongside the matPushfwd vector-base extension).
            return _sampleVectorOutputAtCell(
              name, comp.distOp, distKwargs, perAtomRefs, jj, cc, ctx, N
            ).then((res: any) => {
              cells[jj * C + cc] = res;
            });
          }

          const distIR = {
            kind: 'call', op: comp.distOp, kwargs: distKwargs,
          };
          const workerMsg: any = {
            type: 'sampleN', ir: distIR, count: N,
            refArrays: perAtomRefs,
            seed: nameSeed(name + ':j' + jj + ':c' + cc, ctx.rootKey),
          };
          return ctx.sendWorker(workerMsg).then((reply: any) => {
            cells[jj * C + cc] = { data: reply.samples, eventDim: 1 };
          });
        });
      }
    }
    return pushFixedEnv(ctx, fixedEnv).then(() => chain).then(() => {
      // Stitch with per-component event-dim awareness. Output width
      // along the joint axis = sum_c(eventDim_c); scalar components
      // contribute 1 each (compatibility with Phase 4.2 layout for
      // all-scalar joints), vector components contribute their n.
      const eventDims = new Array(C);
      let totalEventDim = 0;
      const eventOffsets = new Array(C);
      for (let c = 0; c < C; c++) {
        // Read per-cell eventDim from cell j=0 (executor guarantees
        // consistent dim across cells for a given component).
        eventDims[c] = cells[c].eventDim;
        eventOffsets[c] = totalEventDim;
        totalEventDim += eventDims[c];
      }
      const out = new Float64Array(N * K * totalEventDim);
      for (let j = 0; j < K; j++) {
        for (let c = 0; c < C; c++) {
          const ed = eventDims[c];
          const off = eventOffsets[c];
          const cell = cells[j * C + c];
          const col = cell.data;
          if (ed === 1) {
            // Scalar component — col is Float64Array(N) of scalars.
            for (let i = 0; i < N; i++) {
              out[i * K * totalEventDim + j * totalEventDim + off] = col[i];
            }
          } else {
            // Vector-output component — col is Float64Array(N*ed)
            // atom-major (each atom's ed-vector is contiguous).
            for (let i = 0; i < N; i++) {
              for (let k = 0; k < ed; k++) {
                out[i * K * totalEventDim + j * totalEventDim + off + k]
                  = col[i * ed + k];
              }
            }
          }
        }
      }
      const value = { shape: [N | 0, K, totalEventDim], data: out };
      return Object.assign(
        empirical.arrayMeasure(out, [K, totalEventDim], null),
        { value: value, logTotalmass: 0, n_eff: N },
      );
    });
  });
}

// =====================================================================
// Phase 5.1 Session 5a — per-cell vector-output dispatch for joint
// composite components. Returns `{data: Float64Array(N*eventDim),
// eventDim}` — atom-major per-cell n-vector samples. Today (Session
// 5a MVP) only MvNormal is supported; the helper extends naturally
// to any VECTOR_OUTPUT_DISTRIBUTIONS entry whose materialiser path
// the registry covers.
// =====================================================================

function _sampleVectorOutputAtCell(
  name: string, distOp: string, distKwargs: Record<string, any>,
  perAtomRefs: Record<string, any>, jj: number, cc: number,
  ctx: any, N: number,
): Promise<{ data: Float64Array; eventDim: number }> {
  if (distOp !== 'MvNormal') {
    return Promise.reject(new Error('joint composite: vector-output '
      + 'component dist \'' + distOp + '\' not supported (Session 5a '
      + 'ships MvNormal only; further multivariates land alongside '
      + 'their registry entries in Phase 5.x)'));
  }
  return _sampleMvNormalAtCell(name, distKwargs, perAtomRefs, jj, cc, ctx, N);
}

function _sampleMvNormalAtCell(
  name: string, distKwargs: Record<string, any>,
  perAtomRefs: Record<string, any>, jj: number, cc: number,
  ctx: any, N: number,
): Promise<{ data: Float64Array; eventDim: number }> {
  const sampler = require('./sampler.ts');
  const bijRegistry = require('./bijection-registry.ts');
  const muIR = distKwargs.mu;
  const covIR = distKwargs.cov;
  if (!muIR || !covIR) {
    return Promise.reject(new Error('joint composite: MvNormal component '
      + 'at cell j=' + jj + ' c=' + cc + ' requires mu and cov kwargs'));
  }
  // Cell-substituted distKwargs may still reference fixed-phase
  // bindings via self-refs. resolveIRToValue handles literal + ref
  // forms uniformly; per-atom refs (which perAtomRefs would expose)
  // are NOT supported in this MVP — atom-batched mu/cov needs the
  // matPushfwd vector-base extension (Session 5c) for walker
  // symmetry. Reject up front when perAtomRefs has any entry — the
  // current substituted kwargs would otherwise silently miss them.
  if (perAtomRefs && Object.keys(perAtomRefs).length > 0) {
    return Promise.reject(new Error('joint composite: MvNormal component '
      + 'with per-atom-stochastic mu/cov is not yet supported (atom-dep '
      + 'broadcast args produced ' + Object.keys(perAtomRefs).length
      + ' perAtomRefs at cell j=' + jj + '). Per-atom-stochastic '
      + 'multivariate params land with Session 5c (matPushfwd '
      + 'vector-base extension).'));
  }
  const muVal = orchestrator.resolveIRToValue(muIR, ctx.bindings, ctx.fixedValues);
  const covVal = orchestrator.resolveIRToValue(covIR, ctx.bindings, ctx.fixedValues);
  if (muVal == null) {
    return Promise.reject(new Error('joint composite: MvNormal component '
      + 'cannot resolve mu at cell j=' + jj));
  }
  if (covVal == null) {
    return Promise.reject(new Error('joint composite: MvNormal component '
      + 'cannot resolve cov at cell j=' + jj));
  }
  const valueOps = require('./value-ops.ts');
  const muValue = valueLib.asValue(muVal);
  if (muValue.shape.length !== 1) {
    return Promise.reject(new Error('joint composite: MvNormal component '
      + 'mu must be a vector at cell j=' + jj + '; got shape='
      + JSON.stringify(muValue.shape)));
  }
  const n = muValue.shape[0];
  const covValue = Array.isArray(covVal) && covVal.length > 0 && Array.isArray(covVal[0])
    ? valueOps._nestedToValue(covVal)
    : valueLib.asValue(covVal);
  if (covValue.shape.length !== 2 || covValue.shape[0] !== n
      || covValue.shape[1] !== n) {
    return Promise.reject(new Error('joint composite: MvNormal cov must '
      + 'be ' + n + 'x' + n + ' at cell j=' + jj + '; got shape='
      + JSON.stringify(covValue.shape)));
  }
  let L: any;
  try {
    L = sampler._internal.ARITH_OPS.lower_cholesky(covValue);
  } catch (err) {
    return Promise.reject(new Error('joint composite: MvNormal Cholesky '
      + 'failed at cell j=' + jj + ': ' + (err as any).message));
  }
  // Draw N atoms of n standard normals → flat Float64Array(N*n) atom-
  // major. The registry's affineAtomBatchedForward then computes
  // L·z + mu per atom — same hot path matMvNormal consumes for the
  // single-cell case.
  return ctx.sendWorker({
    type: 'sampleN',
    ir: { kind: 'call', op: 'Normal',
          kwargs: { mu: { kind: 'lit', value: 0 },
                    sigma: { kind: 'lit', value: 1 } } },
    count: N, repeat: n,
    refArrays: {},
    seed: nameSeed(name + ':j' + jj + ':c' + cc + ':iidbase', ctx.rootKey),
  }).then((reply: any) => {
    const z = { shape: [N, n], data: reply.samples };
    const y = bijRegistry.affineAtomBatchedForward(
      z, { L: L, b: muValue }, N);
    return { data: y.data as Float64Array, eventDim: n };
  });
}

// =====================================================================
// Phase 4.3 — jointchain-composite execution
// =====================================================================
//
// Markov-chain kernel-broadcast: `kernelof(jointchain(<base>, <K_1>,
// <K_2>, …), …)`. Per spec §06 the chain factorises as
//
//     p(v_0, v_1, …, v_n) = p_base(v_0) · K_1(v_0)(v_1) · K_2(v_1)(v_2) · …
//
// Phase 4.3 MVP scope (matches the recogniser's gate):
// - Step 0 is a closed-first base measure (anon-ref to sampleable
//   DistCall, kernel placeholders embedded).
// - Each subsequent step is a single-input kernel binding whose body
//   is `lawof(<sampleable DistCall>)`. The kernel's single param
//   receives the previous step's variate.
// - Positional layout only; chain length ≥ 2.
//
// Execution per cell j (K cells in total):
//   1. Build bcKwargs from cell-sliced broadcast args (one slice per
//      broadcast arg — lit for atom-indep, per-atom refArray for
//      atom-batched). These resolve outer kernel placeholders inside
//      the base measure's distKwargs.
//   2. Step 0: substitute outer placeholders in the base distKwargs,
//      sampleN → column[j, 0] (Float64Array(N) of per-atom scalars).
//   3. Step k > 0: build per-step substitution:
//        - The step kernel's input param resolves to a refArray named
//          `__prev_<j>_<k-1>` whose data is column[j, k-1].
//        - Substitute outer placeholders + the prev-variate refArray
//          into the step's distKwargs.
//      sampleN → column[j, k].
//   4. Stitch: out[i*K*C + j*C + k] = column[j, k][i].
//
// State threading is the new ingredient vs Phase 4.2: each step k's
// sampleN waits on step k-1's reply (per cell). Cells could run
// concurrently but we sequence them too to match Phase 4.2's chain
// promise shape; ordering doesn't affect correctness (each cell has
// its own seed namespace).
//
// Cost: K * C worker calls, K cells × C steps each. For modest chain
// length (typical MCMC fragments) this is the natural granularity.
// A backend lowering to vectorised codegen folds the inner loop into
// a scan; the JS engine's sequential walk is the interpretation
// analogue.

function _executeJointChainComposite(
  name: string, d: any, ctx: any, compositeBody: any,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const N = ctx.sampleCount;
  const steps = compositeBody.steps;
  const C = steps.length;
  if (C < 2) {
    return Promise.reject(new Error(
      'broadcast: jointchain composite requires chain length ≥ 2, got ' + C));
  }

  // Pre-substitute outer kernel placeholders in every step's distKwargs
  // so the aggregateIR sees the broadcast args (not the kernel's local
  // placeholder names). For the kernel-step case the inputParam stays
  // unbound here — it gets a refArray substitution per (j, k) in the
  // inner loop.
  const substBaseKwargs: Record<string, any> = {};
  for (const pn of Object.keys(steps[0].base.distKwargs)) {
    substBaseKwargs[pn] = _substituteKernelParams(
      steps[0].base.distKwargs[pn],
      compositeBody.params, compositeBody.paramKwargs, d.kwargIRs || {});
  }
  const substStepKwargs: Array<Record<string, any>> = [substBaseKwargs];
  for (let k = 1; k < C; k++) {
    const subst: Record<string, any> = {};
    for (const pn of Object.keys(steps[k].kernel.distKwargs)) {
      subst[pn] = _substituteKernelParams(
        steps[k].kernel.distKwargs[pn],
        compositeBody.params, compositeBody.paramKwargs, d.kwargIRs || {});
    }
    substStepKwargs.push(subst);
  }

  // Aggregate IR: flatten every step's substituted kwargs into a
  // single broadcast-shaped node so `prepareDensityRefs` discovers all
  // outer refs in one pass (closed-over fixed-phase values like
  // `sigma_init`, `sigma_step`).
  const flatKwargs: Record<string, any> = {};
  for (let k = 0; k < C; k++) {
    for (const pn of Object.keys(substStepKwargs[k])) {
      flatKwargs['__chain_s' + k + '_' + pn] = substStepKwargs[k][pn];
    }
  }
  const aggregateIR: any = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: d.distOp }],
    kwargs: flatKwargs,
  };

  return prepareDensityRefs(aggregateIR, ctx, 'broadcast').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const baseEnv: Record<string, any> = {};
    for (const k in fixedEnv) baseEnv[k] = fixedEnv[k];
    const refNames = Object.keys(refArrays);

    // Determine cell count K + atom-dep classification per broadcast arg.
    // Same shape contract as Phase 4.2's joint executor — broadcast args
    // can be lit-scalar, [K] atom-indep, [N] atom-batched, or [N, K]
    // atom-batched; inner-D shapes reject (vec-per-cell is iid territory).
    const argVals: Record<string, any> = {};
    const argClass: Record<string, any> = {};
    let K = 1;
    for (const argName of Object.keys(d.kwargIRs || {})) {
      const argIR = d.kwargIRs[argName];
      const argRefs = orchestrator.collectSelfRefs(argIR);
      const usesAtom = refNames.some((n) => argRefs.has(n));
      let v: any;
      try {
        v = sampler.evaluateExprN(argIR, refArrays, N, baseEnv, undefined);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
      argVals[argName] = v;
      const cls = shared.classifyBroadcastArg(v, N, usesAtom);
      argClass[argName] = cls;
      const argK = cls.K || 1;
      if (argK > 1) {
        if (K === 1) K = argK;
        else if (K !== argK) {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): incompatible cell-axis lengths (' + K + ' vs '
            + argK + ') on arg \'' + argName + '\''));
        }
      }
      const argD = cls.D || 1;
      if (argD > 1) {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): jointchain-composite broadcast arg \'' + argName
          + '\' has inner dim ' + argD + ' (vec-per-cell shapes belong '
          + 'to iid-composite, not jointchain)'));
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty collection argument'));
    }

    // Per-cell sequential chain walk. cells[j*C + k] holds step k's
    // sampled per-atom column for cell j.
    const cells = new Array(K * C);
    let chain = Promise.resolve();
    for (let j = 0; j < K; j++) {
      const jj = j;
      chain = chain.then(() => {
        // Build the cell-j bcKwargs once — these resolve OUTER kernel
        // placeholders inside each step's distKwargs. The bcKwargs IR
        // shape mirrors Phase 4.2's joint executor.
        const bcKwargs: Record<string, any> = {};
        const cellRefs: Record<string, any> = {};
        for (const argName of Object.keys(d.kwargIRs || {})) {
          const cls = argClass[argName];
          const v = argVals[argName];
          const isAtomDep = cls.kind === 'N' || cls.kind === 'NK'
                            || cls.kind === 'NKD';
          if (isAtomDep) {
            const col = shared.perAtomColumnAtJ(v, jj, N);
            const refName = '__bc_' + argName + '_' + jj;
            cellRefs[refName] = valueLib.batchedScalar(col);
            bcKwargs[argName] = { kind: 'ref', ns: 'self', name: refName };
          } else {
            const s = shared.elemScalarAtJ(v, jj, N);
            bcKwargs[argName] = { kind: 'lit', value: s };
          }
        }

        // Sequential chain walk for this cell. Each step's refArrays
        // include cellRefs PLUS the prev-variate ref accumulated so
        // far (step k sees __prev_<j>_<0…k-1>).
        const cellStepRefs: Record<string, any> = {};
        for (const rk in cellRefs) cellStepRefs[rk] = cellRefs[rk];

        let stepChain = Promise.resolve();
        for (let k = 0; k < C; k++) {
          const kk = k;
          stepChain = stepChain.then(() => {
            const step = steps[kk];
            let distOp: string;
            let stepKwargsRaw: Record<string, any>;
            if (kk === 0) {
              // Base step — re-substitute outer kernel placeholders
              // with this cell's bcKwargs. (Pre-substitution above used
              // d.kwargIRs which is the IR-level binding; the per-cell
              // pass uses the lit/ref bcKwargs that carry the j-slice.)
              distOp = step.base.distOp;
              stepKwargsRaw = {};
              for (const pn of Object.keys(step.base.distKwargs)) {
                stepKwargsRaw[pn] = _substituteKernelParams(
                  step.base.distKwargs[pn],
                  compositeBody.params, compositeBody.paramKwargs, bcKwargs);
              }
            } else {
              // Kernel step — outer placeholders + the prev-variate
              // refArray. inputParam is the kernel's single param name
              // (e.g. `prev`); we bind it to the refArray that holds
              // step kk-1's sampled column for this cell.
              distOp = step.kernel.distOp;
              const prevRefName = '__prev_' + jj + '_' + (kk - 1);
              // cellStepRefs[prevRefName] was set in the kk-1 iteration
              // below. Build a kernel-input map so substituteKernelParams
              // resolves the step kernel's single param.
              const stepInputKwargs: Record<string, any> = Object.assign(
                {}, bcKwargs);
              stepInputKwargs[step.kernel.inputParam] = {
                kind: 'ref', ns: 'self', name: prevRefName,
              };
              // The step kernel uses inputParam as its sole param. The
              // outer kernel's params (e.g. `x0`) might ALSO appear in
              // the step kwargs (closed-over) — bcKwargs already binds
              // them. Substitute against BOTH name sets in one pass.
              const stepParamNames = [step.kernel.inputParam].concat(
                compositeBody.params);
              const stepParamKwargs = [step.kernel.inputParam].concat(
                compositeBody.paramKwargs);
              stepKwargsRaw = {};
              for (const pn of Object.keys(step.kernel.distKwargs)) {
                stepKwargsRaw[pn] = _substituteKernelParams(
                  step.kernel.distKwargs[pn],
                  stepParamNames, stepParamKwargs, stepInputKwargs);
              }
            }
            const distIR = {
              kind: 'call', op: distOp, kwargs: stepKwargsRaw,
            };
            const workerMsg: any = {
              type: 'sampleN', ir: distIR, count: N,
              refArrays: cellStepRefs,
              seed: nameSeed(name + ':j' + jj + ':k' + kk, ctx.rootKey),
            };
            return ctx.sendWorker(workerMsg).then((reply: any) => {
              cells[jj * C + kk] = reply.samples;
              // Stage the just-sampled column as the prev-variate ref
              // for step kk+1. cellStepRefs persists across iterations
              // within this cell.
              const prevRefName = '__prev_' + jj + '_' + kk;
              cellStepRefs[prevRefName] = valueLib.batchedScalar(reply.samples);
            });
          });
        }
        return stepChain;
      });
    }
    return pushFixedEnv(ctx, fixedEnv).then(() => chain).then(() => {
      // Stitch: shape [N, K, C] atom-major.
      const out = new Float64Array(N * K * C);
      for (let j = 0; j < K; j++) {
        for (let k = 0; k < C; k++) {
          const col = cells[j * C + k];
          for (let i = 0; i < N; i++) {
            out[i * K * C + j * C + k] = col[i];
          }
        }
      }
      const value = { shape: [N | 0, K, C], data: out };
      return Object.assign(
        empirical.arrayMeasure(out, [K, C], null),
        { value: value, logTotalmass: 0, n_eff: N },
      );
    });
  });
}

// =====================================================================
// Phase 4.4 — nested-broadcast-composite execution
// =====================================================================
//
// Outer kernel-broadcast body is itself `broadcast(<bare_dist>, kw)`:
//
//     broadcast(<outer_kernel>, outer_kwargs)
//        with outer_kernel = kernelof(broadcast(D, inner_kw), outer_kw)
//
// Each atom yields a [K_outer, K_inner] block. The new architectural
// ingredient vs Phases 4.1-4.3 is **two-axis per-cell iteration**:
//
//   for outer j in [0, K_outer):
//     substitute outer placeholders into inner kwargs (per outer cell)
//     evaluate inner kwargs to discover K_inner + per-arg shape
//     for inner k in [0, K_inner):
//       slice inner kwargs at index k (lit or per-atom refArray)
//       sampleN with the substituted inner distOp → column[j, k]
//
// Inner kwargs may mix `%local`-as-outer-placeholder refs with `self`-
// as-closed-over refs; `_substituteKernelParams` handles both via the
// same `params` / `paramKwargs` mapping used by Phases 4.1-4.3.
//
// MVP scope (matches recogniser gate):
// - Inner broadcast head is a bare sampler-REGISTRY scalar dist.
// - Inner kwargs use kwarg form (positional defers).
// - No vec-per-cell inner D axis (the inner cells are scalars).
//
// Cost: K_outer * K_inner worker calls. A future optimisation can
// detect when the inner cell axis has no atom-dep AFTER outer
// substitution and route through `KERNEL_BROADCAST_FAST_PATHS` for a
// single vectorised inner pass per outer cell; the recogniser surface
// stays unchanged.

function _executeNestedBroadcastComposite(
  name: string, d: any, ctx: any, compositeBody: any,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const N = ctx.sampleCount;
  const innerKwargs = compositeBody.innerKwargs;

  // Pre-substitute outer placeholders in inner kwargs so the aggregate
  // IR closes over the right refs (broadcast args + outer-scope refs).
  const substInnerKwargs: Record<string, any> = {};
  for (const pn of Object.keys(innerKwargs)) {
    substInnerKwargs[pn] = _substituteKernelParams(
      innerKwargs[pn],
      compositeBody.params, compositeBody.paramKwargs, d.kwargIRs || {});
  }
  const aggregateIR: any = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: d.distOp }],
    kwargs: substInnerKwargs,
  };

  return prepareDensityRefs(aggregateIR, ctx, 'broadcast').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const baseEnv: Record<string, any> = {};
    for (const k in fixedEnv) baseEnv[k] = fixedEnv[k];
    const refNames = Object.keys(refArrays);

    // Determine OUTER cell axis K from the outer broadcast args (same
    // shape contract Phase 4.2 uses).
    const argVals: Record<string, any> = {};
    const argClass: Record<string, any> = {};
    let K = 1;
    for (const argName of Object.keys(d.kwargIRs || {})) {
      const argIR = d.kwargIRs[argName];
      const argRefs = orchestrator.collectSelfRefs(argIR);
      const usesAtom = refNames.some((n) => argRefs.has(n));
      let v: any;
      try {
        v = sampler.evaluateExprN(argIR, refArrays, N, baseEnv, undefined);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
      argVals[argName] = v;
      const cls = shared.classifyBroadcastArg(v, N, usesAtom);
      argClass[argName] = cls;
      const argK = cls.K || 1;
      if (argK > 1) {
        if (K === 1) K = argK;
        else if (K !== argK) {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): incompatible outer cell-axis lengths (' + K + ' vs '
            + argK + ') on arg \'' + argName + '\''));
        }
      }
      const argD = cls.D || 1;
      if (argD > 1) {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): nested-broadcast outer arg \'' + argName
          + '\' has inner dim ' + argD + ' (vec-per-cell shapes belong '
          + 'to iid-composite)'));
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty outer collection argument'));
    }

    // Build the cell-j bcKwargs map once per outer j; then per
    // outer cell we determine K_inner by evaluating the substituted
    // inner kwargs and per inner cell we materialise (sampleN for
    // scalar inner dists; registry-backed _sampleVectorOutputAtCell
    // for VECTOR_OUTPUT inner dists — Phase 5.1 Session 5b).
    //
    // Two passes within each outer cell:
    //   (a) evaluate each inner kwarg against this cell's bcKwargs to
    //       extract the per-(j, k) value (scalar or per-atom column).
    //   (b) materialise per (j, k).
    //
    // The "K_inner" is the largest cell-axis size across the inner
    // kwargs (mirrors the standalone broadcast shape ladder); inner
    // kwargs that resolve to scalars or atom-batched-scalars pass
    // through unchanged.
    //
    // Per-inner-cell output is `{data: Float64Array, eventDim}` so
    // the stitcher knows the per-cell width — scalar inner dists
    // produce eventDim=1 (compat with Phase 4.4 layout), vector
    // inner dists produce their n. Output shape becomes
    // [N, K_outer, K_inner * eventDim] atom-major.
    type InnerCellResult = { data: Float64Array; eventDim: number };
    type CellResult = { K_inner: number; cells: InnerCellResult[] };
    const outerCells: CellResult[] = new Array(K);
    let chain = Promise.resolve();
    for (let j = 0; j < K; j++) {
      const jj = j;
      chain = chain.then(() => {
        const bcKwargs: Record<string, any> = {};
        const outerRefs: Record<string, any> = {};
        for (const argName of Object.keys(d.kwargIRs || {})) {
          const cls = argClass[argName];
          const v = argVals[argName];
          const isAtomDep = cls.kind === 'N' || cls.kind === 'NK'
                            || cls.kind === 'NKD';
          if (isAtomDep) {
            const col = shared.perAtomColumnAtJ(v, jj, N);
            const refName = '__bc_' + argName + '_' + jj;
            outerRefs[refName] = valueLib.batchedScalar(col);
            bcKwargs[argName] = { kind: 'ref', ns: 'self', name: refName };
          } else {
            const s = shared.elemScalarAtJ(v, jj, N);
            bcKwargs[argName] = { kind: 'lit', value: s };
          }
        }

        // Inner kwargs after substituting outer placeholders with the
        // cell-j bcKwargs (refs to per-atom columns or lit scalars).
        const innerKwargsCellJ: Record<string, any> = {};
        for (const pn of Object.keys(innerKwargs)) {
          innerKwargsCellJ[pn] = _substituteKernelParams(
            innerKwargs[pn],
            compositeBody.params, compositeBody.paramKwargs, bcKwargs);
        }

        // Evaluate each inner kwarg to determine K_inner + per-arg shape.
        // Inner refArrays = refArrays (outer prep) + outerRefs (this
        // cell's per-atom slices).
        const cellRefArrays: Record<string, any> = {};
        for (const rk in refArrays) cellRefArrays[rk] = refArrays[rk];
        for (const rk in outerRefs) cellRefArrays[rk] = outerRefs[rk];
        const innerVals: Record<string, any> = {};
        const innerClass: Record<string, any> = {};
        let K_inner = 1;
        for (const pn of Object.keys(innerKwargsCellJ)) {
          let v: any;
          try {
            v = sampler.evaluateExprN(
              innerKwargsCellJ[pn], cellRefArrays, N, baseEnv, undefined);
          } catch (err) {
            throw err instanceof Error ? err : new Error(String(err));
          }
          innerVals[pn] = v;
          // For shape classification we need to know if the kwarg
          // references any per-atom refs (i.e. is atom-batched). After
          // the cell-j substitution, atom-batched is propagated through
          // the outerRefs entries — collectSelfRefs would still see
          // those names so we can classify.
          const refs = orchestrator.collectSelfRefs(innerKwargsCellJ[pn]);
          const usesAtom = Object.keys(cellRefArrays).some((n) =>
            refs.has(n));
          const cls = shared.classifyBroadcastArg(v, N, usesAtom);
          innerClass[pn] = cls;
          const innerK = cls.K || 1;
          if (innerK > 1) {
            if (K_inner === 1) K_inner = innerK;
            else if (K_inner !== innerK) {
              throw new Error('broadcast(' + d.distOp
                + '): incompatible inner cell-axis lengths (' + K_inner
                + ' vs ' + innerK + ') on inner kwarg \'' + pn
                + '\' at outer cell ' + jj);
            }
          }
          const innerD = cls.D || 1;
          if (innerD > 1 && !compositeBody.innerIsVectorOutput) {
            throw new Error('broadcast(' + d.distOp
              + '): nested-broadcast inner kwarg \'' + pn
              + '\' has inner D ' + innerD + ' (vec-per-cell shapes '
              + 'belong to iid-composite unless inner is a vector-'
              + 'output dist, which this nested-broadcast has none of)');
          }
        }
        if (K_inner < 1) {
          throw new Error('broadcast(' + d.distOp
            + '): empty inner collection argument at outer cell ' + jj);
        }

        // Per inner cell k: build inner-cell kwargs (lit, refArray, or
        // per-cell literal vector for D > 1 atom-indep kwargs) and
        // materialise. Scalar inner dists go through sampleN; vector-
        // output inner dists (Session 5b — MvNormal) dispatch through
        // _sampleVectorOutputAtCell. Each cell's result carries an
        // eventDim so the final stitch knows the per-cell width.
        const innerCells: InnerCellResult[] = new Array(K_inner);
        let innerChain = Promise.resolve();
        for (let k = 0; k < K_inner; k++) {
          const kk = k;
          innerChain = innerChain.then(() => {
            const distKwargs: Record<string, any> = {};
            const innerCellRefs: Record<string, any> = {};
            for (const rk in outerRefs) innerCellRefs[rk] = outerRefs[rk];
            for (const pn of Object.keys(innerVals)) {
              const cls = innerClass[pn];
              const v = innerVals[pn];
              const innerD = cls.D || 1;
              const isAtomDep = cls.kind === 'N' || cls.kind === 'NK'
                                || cls.kind === 'NKD';
              if (isAtomDep) {
                if (innerD > 1) {
                  throw new Error('nested-broadcast: atom-dep per-cell '
                    + 'vector inner kwarg \'' + pn + '\' (D=' + innerD
                    + ') deferred to Session 5c+ (matPushfwd vector-base '
                    + 'extension)');
                }
                const col = shared.perAtomColumnAtJ(v, kk, N);
                const refName = '__inbc_' + pn + '_' + jj + '_' + kk;
                innerCellRefs[refName] = valueLib.batchedScalar(col);
                distKwargs[pn] = { kind: 'ref', ns: 'self', name: refName };
              } else if (innerD > 1) {
                // Atom-indep per-cell vector — extract v[kk, :] as a
                // literal `vector(...)` IR (e.g. inner MvNormal's mu
                // per inner cell).
                const baseOff = (v.shape && v.shape.length === 2)
                  ? kk * innerD : 0;
                const litArgs: any[] = [];
                for (let z = 0; z < innerD; z++) {
                  litArgs.push({ kind: 'lit', value: v.data[baseOff + z] });
                }
                distKwargs[pn] = {
                  kind: 'call', op: 'vector', args: litArgs };
              } else {
                const s = shared.elemScalarAtJ(v, kk, N);
                distKwargs[pn] = { kind: 'lit', value: s };
              }
            }

            if (compositeBody.innerIsVectorOutput) {
              // Phase 5.1 Session 5b — MvNormal (or future vector-
              // output dist) as the inner dist. Dispatch through the
              // shared per-cell helper that consumes the bijection
              // registry. The helper rejects perAtomRefs in the MVP
              // (atom-dep mu/cov defers to Session 5c+); the outer-
              // cell's outerRefs ARE atom-dep refs, so we filter them
              // out — they don't bind any inner kwarg in this scope
              // (atom-dep outer ARGS feed via per-atom slices into mu
              // / cov only if the inner MvNormal references them
              // through outer placeholders, which Session 5b's MVP
              // doesn't yet handle).
              const atomIndepRefs = {};   // empty by design — MVP gate
              const refKeys = Object.keys(innerCellRefs);
              if (refKeys.length > 0) {
                // Detect whether any innerCellRef is actually used by
                // a remaining distKwarg ref (post-substitution). If
                // it is, we'd need the matPushfwd vector-base
                // extension — reject up front.
                const usedRefs = new Set<string>();
                for (const pn of Object.keys(distKwargs)) {
                  const refs = orchestrator.collectSelfRefs(distKwargs[pn]);
                  refs.forEach((n: string) => usedRefs.add(n));
                }
                for (const rk of refKeys) {
                  if (usedRefs.has(rk)) {
                    throw new Error('nested-broadcast: inner MvNormal '
                      + 'with atom-dep mu/cov via outer-cell per-atom '
                      + 'refs is not supported in Session 5b MVP. '
                      + 'Outer arg threading into MvNormal params '
                      + 'defers to Session 5c+ (matPushfwd '
                      + 'vector-base extension).');
                  }
                }
              }
              return _sampleVectorOutputAtCell(
                name, compositeBody.innerDistOp, distKwargs,
                atomIndepRefs, jj, kk, ctx, N
              ).then((res: any) => {
                innerCells[kk] = res;
              });
            }
            const distIR = {
              kind: 'call', op: compositeBody.innerDistOp, kwargs: distKwargs,
            };
            const workerMsg: any = {
              type: 'sampleN', ir: distIR, count: N,
              refArrays: innerCellRefs,
              seed: nameSeed(name + ':j' + jj + ':k' + kk, ctx.rootKey),
            };
            return ctx.sendWorker(workerMsg).then((reply: any) => {
              innerCells[kk] = { data: reply.samples, eventDim: 1 };
            });
          });
        }
        return innerChain.then(() => {
          outerCells[jj] = { K_inner, cells: innerCells };
        });
      });
    }

    return pushFixedEnv(ctx, fixedEnv).then(() => chain).then(() => {
      // Enforce uniform K_inner across outer cells (the recogniser's
      // contract). Different K_inner per outer cell would be a ragged
      // shape — defer to a follow-up if motivated.
      let K_inner = outerCells[0].K_inner;
      for (let j = 1; j < K; j++) {
        if (outerCells[j].K_inner !== K_inner) {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): ragged inner cell-axis lengths across outer cells ('
            + K_inner + ' vs ' + outerCells[j].K_inner + ' at outer '
            + j + ') — Phase 4.4 MVP requires uniform inner K'));
        }
      }
      // Stitch with per-(j, k) event-dim awareness. For scalar inner
      // dists every (j, k) cell contributes 1 scalar (compat with the
      // Phase 4.4 layout). For vector-output inner dists every (j, k)
      // cell contributes an eventDim-vector; we splat them along a
      // unified inner axis to keep the output rank-3 (the trailing n
      // multiplies into the K_inner axis):
      //   shape = [N, K_outer, K_inner * eventDim]   atom-major
      // Sane downstream consumption (reductions, slicing) treats the
      // last axis as the natural value axis of each (outer, inner)
      // cell — matches the bare-vector-output kernel-broadcast output.
      const eventDim = outerCells[0].cells[0].eventDim;
      for (let j = 0; j < K; j++) {
        for (let k = 0; k < K_inner; k++) {
          if (outerCells[j].cells[k].eventDim !== eventDim) {
            return Promise.reject(new Error('broadcast(' + d.distOp
              + '): ragged inner event-dim across (j=' + j + ', k='
              + k + '): expected ' + eventDim + ', got '
              + outerCells[j].cells[k].eventDim));
          }
        }
      }
      const innerWidth = K_inner * eventDim;
      const out = new Float64Array(N * K * innerWidth);
      for (let j = 0; j < K; j++) {
        for (let k = 0; k < K_inner; k++) {
          const cell = outerCells[j].cells[k];
          const col = cell.data;
          if (eventDim === 1) {
            for (let i = 0; i < N; i++) {
              out[i * K * innerWidth + j * innerWidth + k] = col[i];
            }
          } else {
            // Vector inner — each atom contributes an eventDim-vector
            // at inner cell k; splat into the [k*eventDim, (k+1)*eventDim)
            // slot of the inner axis.
            const off = k * eventDim;
            for (let i = 0; i < N; i++) {
              for (let z = 0; z < eventDim; z++) {
                out[i * K * innerWidth + j * innerWidth + off + z]
                  = col[i * eventDim + z];
              }
            }
          }
        }
      }
      const value = { shape: [N | 0, K, innerWidth], data: out };
      return Object.assign(
        empirical.arrayMeasure(out, [K, innerWidth], null),
        { value: value, logTotalmass: 0, n_eff: N },
      );
    });
  });
}

// =====================================================================
// Phase 5.1 Session 4 — bare vector-output dist broadcast (MvNormal)
// =====================================================================
//
// `broadcast(MvNormal, mu = mu_arr, cov = cov_arg)` — per spec §04
// kernel-broadcast over a multivariate distribution. Engine-concepts
// §22 frames this case as the canonical decomposition
//     broadcast(pushfwd(affine_per_cell, iid(Normal, n)), …)
// without yet committing to surface lowering (deliverable a). Instead,
// this executor materialises per outer cell via the SAME registry
// entry matMvNormal consumes: build per-cell (L, mu) from the
// broadcast args, draw an iid Normal base, apply
// `bijectionRegistry.affineAtomBatchedForward`. Output is a
// [N, K, n] atom-major Value mirroring the joint-composite stitcher.
//
// Scope (Session 4 MVP):
//   - `mu` may be per-cell (rank-2 [K, n] when atom-indep, rank-3
//     [N, K, n] when atom-batched) or shared (rank-1 [n] for atom-
//     indep, rank-2 [N, n] for atom-batched broadcasted scalars on a
//     per-atom random mu).
//   - `cov` is SHARED across cells in Session 4 (rank-2 [n, n]
//     atom-indep). Per-cell cov adds (a) per-cell Cholesky and
//     (b) shape-2 broadcast-arg slicing — deferred to Session 5+ once
//     a fixture motivates it.
//   - Both kwarg and positional `mu / cov` admitted (kwarg form is
//     the canonical FlatPPL surface for MvNormal).
//
// Deferred (no forcing fixture yet):
//   - Per-cell `cov_arr` of shape [K, n, n] — needs per-cell Cholesky
//     plus cell-axis slicing of rank-3 broadcast args.
//   - Per-atom stochastic `mu` / `cov` (atom-batched broadcast args
//     where the value's leading atom axis aligns with the per-atom
//     refArrays).
//   - Per-cell Cholesky factor `chol = lower_cholesky(cov)` supplied
//     directly to spare the per-cell Cholesky — falls out of the
//     pushfwd-of-iid lift-time decomposition once landed.

function _executeBareVectorOutputBroadcast(
  name: string, d: any, ctx: any,
): Promise<any> {
  if (d.distOp !== 'MvNormal') {
    return Promise.reject(new Error(
      'broadcast: bare vector-output distribution \'' + d.distOp
      + '\' not yet supported (Phase 5.1 Session 4 ships MvNormal only)'));
  }
  return _executeMvNormalBroadcast(name, d, ctx);
}

function _executeMvNormalBroadcast(
  name: string, d: any, ctx: any,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const bijRegistry = require('./bijection-registry.ts');
  const N = ctx.sampleCount;

  // Resolve mu / cov IRs. Both kwargs and positional admitted: MvNormal's
  // canonical kw layout is (mu, cov).
  const kw = d.kwargIRs || {};
  let muIR = kw.mu;
  let covIR = kw.cov;
  if ((!muIR || !covIR) && Array.isArray(d.argIRs) && d.argIRs.length >= 2) {
    if (!muIR) muIR = d.argIRs[0];
    if (!covIR) covIR = d.argIRs[1];
  }
  if (!muIR || !covIR) {
    return Promise.reject(new Error('broadcast(MvNormal): requires mu and cov '
      + '(kwarg or positional)'));
  }

  // Static resolution — Session 4 scope assumes mu/cov are atom-indep
  // (literals or fixed-phase / closed-form refs). Per-atom stochastic
  // mu/cov defers to Session 5+ alongside the matPushfwd vector-base
  // extension that gives walker symmetry.
  const muVal = orchestrator.resolveIRToValue(muIR, ctx.bindings, ctx.fixedValues);
  const covVal = orchestrator.resolveIRToValue(covIR, ctx.bindings, ctx.fixedValues);
  if (muVal == null) {
    return Promise.reject(new Error('broadcast(MvNormal): cannot resolve mu '
      + '(per-atom mu deferred to Session 5+)'));
  }
  if (covVal == null) {
    return Promise.reject(new Error('broadcast(MvNormal): cannot resolve cov '
      + '(per-atom cov / per-cell cov deferred to Session 5+)'));
  }

  // mu shape: [n] (shared) or [K, n] (per-cell). cov shape: [n, n]
  // (Session 4 shared-cov scope). Determine n and K.
  const muValue = valueLib.asValue(muVal);
  if (!muValue.shape || muValue.shape.length < 1 || muValue.shape.length > 2) {
    return Promise.reject(new Error('broadcast(MvNormal): mu must be rank-1 '
      + '(shared) or rank-2 [K, n] (per-cell); got shape='
      + JSON.stringify(muValue.shape)));
  }
  const muIsPerCell = muValue.shape.length === 2;
  const n = muIsPerCell ? muValue.shape[1] : muValue.shape[0];
  const K = muIsPerCell ? muValue.shape[0] : 1;

  // cov to a square Value — accept nested-array form (the historical
  // matMvNormal shape) as well as a direct shape-rank-2 Value.
  const valueOps = require('./value-ops.ts');
  const covValue = Array.isArray(covVal) && covVal.length > 0 && Array.isArray(covVal[0])
    ? valueOps._nestedToValue(covVal)
    : valueLib.asValue(covVal);
  if (covValue.shape.length !== 2 || covValue.shape[0] !== n
      || covValue.shape[1] !== n) {
    return Promise.reject(new Error('broadcast(MvNormal): cov must be ' + n
      + 'x' + n + ' (Session 4 shared-cov scope); got shape='
      + JSON.stringify(covValue.shape)));
  }

  // Single Cholesky for the shared cov — the per-cell Cholesky case
  // lands when per-cell cov support arrives.
  let L: any;
  try {
    L = sampler._internal.ARITH_OPS.lower_cholesky(covValue);
  } catch (err) {
    return Promise.reject(new Error('broadcast(MvNormal): ' + (err as any).message));
  }

  // Shared-mu degenerate case: K=1, every cell has the same mu and
  // cov, so the output is just a [N, 1, n] block. Materialise once.
  // Per-cell case: K cells × n-vector samples each.
  return ctx.sendWorker({
    type: 'sampleN',
    ir: { kind: 'call', op: 'Normal',
          kwargs: { mu: { kind: 'lit', value: 0 },
                    sigma: { kind: 'lit', value: 1 } } },
    count: N, repeat: K * n,
    refArrays: {},
    seed: nameSeed(name + ':iidbase', ctx.rootKey),
  }).then((reply: any) => {
    // Reply.samples is Float64Array(N * K * n) atom-major, iid Normal.
    // For each cell j, apply affine(L, mu_j) to the [N, n] slice of z.
    const z = reply.samples;
    const out = new Float64Array(N * K * n);
    for (let j = 0; j < K; j++) {
      // Slice z into per-cell [N, n] atom-batched view, then apply
      // affine. Cheapest construction: build a contiguous [N, n]
      // Float64Array per cell so the registry's atom-batched paths
      // accept it directly.
      const zCell = new Float64Array(N * n);
      for (let i = 0; i < N; i++) {
        for (let k = 0; k < n; k++) {
          zCell[i * n + k] = z[i * K * n + j * n + k];
        }
      }
      const zCellValue = { shape: [N, n], data: zCell };
      // mu_j slice: rank-1 [n] (shared) or rank-1 [n] sliced from
      // rank-2 [K, n] at row j.
      let muSliceData: Float64Array;
      if (muIsPerCell) {
        muSliceData = new Float64Array(n);
        for (let k = 0; k < n; k++) {
          muSliceData[k] = muValue.data[j * n + k];
        }
      } else {
        muSliceData = muValue.data instanceof Float64Array
          ? muValue.data
          : new Float64Array(muValue.data);
      }
      const muSlice = { shape: [n], data: muSliceData };
      const yCell = bijRegistry.affineAtomBatchedForward(
        zCellValue, { L: L, b: muSlice }, N);
      // Splat yCell into out at cell j: out[i, j, :] = yCell[i, :].
      const yData = yCell.data;
      for (let i = 0; i < N; i++) {
        for (let k = 0; k < n; k++) {
          out[i * K * n + j * n + k] = yData[i * n + k];
        }
      }
    }
    const value = { shape: [N | 0, K, n], data: out };
    return Object.assign(
      empirical.arrayMeasure(out, [K, n], null),
      { value: value, logTotalmass: 0, n_eff: N },
    );
  });
}

module.exports = { matKernelBroadcast };

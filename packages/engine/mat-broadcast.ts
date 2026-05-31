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
      return Promise.reject(new Error(
        'broadcast: unknown distribution kernel ' + d.distOp));
    }
    // Phase 1.2: only the 'iid' kind is registered. Future kinds get
    // their own execution branches before this point.
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

module.exports = { matKernelBroadcast };

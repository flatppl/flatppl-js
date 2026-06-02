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
    // Phase 8 batch-flatten: an iid-bodied composite folds the
    // (atom N, cell K, inner-iid D) axes into ONE count = N·K·D batch
    // (engine-concepts §20.10 "atom axis = just another batch axis").
    // A single worker call replaces the former K (scalar-per-cell) or
    // K·D (vector-per-cell) per-cell calls. This subsumes both the old
    // scalar-iid `repeat` loop below and `_executeIidCompositeVecPerCell`.
    if (compositeBody.kind === 'iid') {
      return _executeIidCompositeBatchFlatten(name, d, ctx, compositeBody);
    }
    return Promise.reject(new Error(
      'broadcast: composite kernel-body kind \'' + compositeBody.kind
      + '\' recognised but not yet executable (Phase 4 work)'));
  }
  // Map parameter slot → its IR (kwarg form wins, else positional).
  // Reached only for a BUILTIN-distribution head now (composite-iid
  // routes to `_executeIidCompositeBatchFlatten` above); `params` came
  // from the sampler REGISTRY lookup.
  const paramIRs: Record<string, any> = {};
  if (d.kwargIRs && Object.keys(d.kwargIRs).length > 0) {
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
  // Builtin-distribution head: sample directly from `d.distOp` with the
  // mapped param kwargs (per-cell scalar params over the K cell axis).
  const effectiveDistOp = d.distOp;
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
    // **axisStack (authoritative):** if the binding's IR carries a
    // kernel-broadcast axisStack entry (`source: 'kernel_broadcast'`
    // with a literal-integer size), that K is the TRUSTED static fact
    // — we seed K from it and the per-param ladder below VALIDATES
    // against it. A param with intrinsicK > 1 that disagrees is then a
    // producer/runtime drift (engine invariant violation), reported as
    // such rather than as a user collection-length error. When the
    // axisStack is absent or symbolic ('%dynamic' / a binding-ref name
    // — e.g. dynamic-D, PoissonProcess ragged), there is no static
    // fact to trust and the ladder discovers K from the first param
    // with intrinsicK > 1 (the runtime shape-sniff fallback).
    const _bindingStack = axisStackMod.bindingAxisStack(name, ctx);
    const _axisStackK = axisStackMod.outerAxisSize(_bindingStack, 'kernel_broadcast');
    const _kFromAxisStack = !!(_axisStackK && _axisStackK > 1);
    const usesAtomBy: Record<string, boolean> = {};
    const paramVals: Record<string, any> = {};
    let K = _kFromAxisStack ? (_axisStackK as number) : 1;
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
          return Promise.reject(new Error('broadcast(' + d.distOp + '): '
            + (_kFromAxisStack
              ? 'internal: static axisStack kernel-broadcast axis (' + K
                + ') disagrees with runtime collection length (' + intrinsicK
                + ') on parameter \'' + pn + '\' — producer/runtime drift'
              : 'incompatible collection lengths (' + K + ' vs '
                + intrinsicK + ') on parameter \'' + pn + '\'')));
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
    // any other per-atom materialiser path can share them. Result is a
    // [N, K] vector-atom measure (builtin-distribution head; the
    // iid-composite [N, K, D] case is handled by batch-flatten above).
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
        return ctx.sendWorker(workerMsg).then((reply: any) => {
          cols[jj] = reply.samples;
        });
      });
    }
    return pushFixedEnv(ctx, fixedEnv).then(() => chain).then(() => {
      // out shape [N, K] atom-major (builtin-distribution head).
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
// Phase 8 — batch-flatten: the uniform iid-composite executor
// =====================================================================
//
// `broadcast(kernelof(iid(<BuiltinDist>(<body kwargs>), D), …), args…)`
// produces, per atom, a [K, D] block: K outer broadcast cells × D inner
// iid draws. The former code split this across two executors — a per-
// cell `repeat = D` loop (scalar-per-cell params) and a per-(j, r) loop
// (`_executeIidCompositeVecPerCell`, for `[K, D]` / `[N, K, D]` args) —
// issuing K or K·D worker calls.
//
// Batch-flatten is the vmap/XLA move (engine-concepts §20.10 "atom axis
// = just another batch axis"): FOLD the three parallel axes (atom N,
// cell K, inner D) into ONE batch of `count = N·K·D`, lay every input
// out as a single flat `[count]` buffer in (i, j, r) row-major order,
// evaluate the inner dist's params ONCE over that batch, and issue ONE
// `sampleN`. The worker's output is already in [N, K, D] atom-major
// order — no per-cell restitch, no `__bc_*` / `__body_*` synthetic refs.
//
// The per-axis layout (with size-1 broadcast on each axis) is what lets
// one code path cover scalar-per-cell, vector-per-cell, AND the within-
// atom conditional independence of the repeat axis (spec §06) at once:
//   - a closed-over per-atom draw (a hierarchical `mu`, kind 'N') lays
//     out CONSTANT along the (j, r) axes, so atom i's D inner draws all
//     see mu_i — the shared-value requirement — while the draws are
//     fresh (distinct positions in the count = N·K·D RNG batch);
//   - a `[K, D]` regressor lays out per (j, r); a `[K]` collection
//     broadcasts across the inner axis; etc.
//
// §03 invariant (load-bearing): K and D are PARALLEL axes folded ABOVE
// the variate's intrinsic structure. Here the inner dist is scalar so
// there is none; the fold never reshapes an intrinsic vec-of-vec axis.

// Lay a value out as a single flat Float64Array of length `lead·∏axes`
// in row-major (lead, axes…) order — the general N-axis fold primitive
// (vmap-like). `lead` is the leading-axis size (the atom count N).
// `axes` are the parallel axis sizes being folded in. `atomVaries` says
// whether `v` carries a leading `lead` axis; `axisSizes[a]` is `v`'s
// source size along parallel axis `a` (1 ⇒ `v` is constant on that axis
// and broadcasts). `v`'s data is row-major over `[ (lead?), axisSizes… ]`.
// One odometer walk copies each source element into every broadcast
// position. The iid-composite fold uses axes = [K, D]; the nested fold
// uses [K_outer, K_inner] — same primitive, different axis tuple.
function _layoutFlat(
  v: any, lead: number, axes: number[],
  atomVaries: boolean, axisSizes: number[],
): Float64Array {
  const m = axes.length;
  let pProd = 1;
  for (let a = 0; a < m; a++) pProd *= axes[a];
  const out = new Float64Array(lead * pProd);
  const data: any = valueLib.isValue(v) ? v.data
    : (v && v.BYTES_PER_ELEMENT !== undefined) ? v
    : Array.isArray(v) ? v
    : null;
  if (data == null) {                 // plain scalar / boolean
    out.fill(typeof v === 'boolean' ? (v ? 1 : 0) : v);
    return out;
  }
  // Source strides over [ (lead?), axisSizes… ] row-major.
  const srcStride = new Array(m);
  let acc = 1;
  for (let a = m - 1; a >= 0; a--) { srcStride[a] = acc; acc *= axisSizes[a]; }
  const leadStride = acc;             // = ∏ axisSizes
  const tIdx = new Array(m).fill(0);
  for (let i = 0; i < lead; i++) {
    const leadBase = atomVaries ? i * leadStride : 0;
    const tBase = i * pProd;
    for (let a = 0; a < m; a++) tIdx[a] = 0;
    for (let p = 0; p < pProd; p++) {
      let srcOff = leadBase;
      for (let a = 0; a < m; a++) {
        if (axisSizes[a] !== 1) srcOff += tIdx[a] * srcStride[a];
      }
      out[tBase + p] = data[srcOff];
      for (let a = m - 1; a >= 0; a--) {     // odometer, last axis fastest
        if (++tIdx[a] < axes[a]) break;
        tIdx[a] = 0;
      }
    }
  }
  return out;
}

// Translate a classifyBroadcastArg verdict into the (atomVaries,
// axisSizes) descriptor `_layoutFlat` consumes, mapping the verdict's
// cell axis (cls.K) onto parallel-axis `kAxis` and its inner axis (cls.D)
// onto `dAxis` of a length-`m` tuple; unmapped axes stay size-1. The
// iid-composite fold maps {K→0, D→1} (m=2); the nested fold maps an
// OUTER arg's cell axis to {K→0} and an INNER collection's to {K→1}.
function _spanOf(
  cls: { kind: string; K?: number; D?: number },
  m: number, kAxis: number, dAxis: number,
): { atomVaries: boolean; axisSizes: number[] } {
  const atomVaries = cls.kind === 'N' || cls.kind === 'NK' || cls.kind === 'NKD';
  const axisSizes = new Array(m).fill(1);
  if (cls.kind === 'K' || cls.kind === 'KD'
      || cls.kind === 'NK' || cls.kind === 'NKD') {
    axisSizes[kAxis] = cls.K || 1;
  }
  if (cls.kind === 'KD' || cls.kind === 'NKD') {
    axisSizes[dAxis] = cls.D || 1;
  }
  return { atomVaries, axisSizes };
}

// Resolve a batch-flatten param value (an inner-dist kwarg evaluated over
// the count batch) to a worker kwarg: a constant → `{kind:'lit'}`; a
// length-`count` column → `{kind:'col'}`. Returns null for any other
// shape/type — the caller decides (the iid fold rejects; the nested fold
// falls back to its per-cell reference). Handles number / boolean / Value
// / typed-array / per-position JS-array (cold-path) forms.
function _resolveBatchFlattenParam(
  pv: any, count: number,
): { kind: 'lit'; value: number } | { kind: 'col'; data: Float64Array } | null {
  if (typeof pv === 'number' || typeof pv === 'boolean') return { kind: 'lit', value: +pv };
  if (valueLib.isValue(pv)) {
    const bs = pv.shape;
    if (bs.length === 0 || (bs.length === 1 && bs[0] === 1)) return { kind: 'lit', value: pv.data[0] };
    if (bs.length === 1 && bs[0] === count) return { kind: 'col', data: pv.data };
    return null;
  }
  if (pv && pv.BYTES_PER_ELEMENT !== undefined && pv.length === count) {
    return { kind: 'col', data: pv };
  }
  if (Array.isArray(pv) && pv.length === count) {
    const buf = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      const c = pv[i];
      if (typeof c === 'number') buf[i] = c;
      else if (typeof c === 'boolean') buf[i] = c ? 1 : 0;
      else if (valueLib.isValue(c) && (c.shape.length === 0
               || (c.shape.length === 1 && c.shape[0] === 1))) buf[i] = c.data[0];
      else return null;
    }
    return { kind: 'col', data: buf };
  }
  return null;
}

function _executeIidCompositeBatchFlatten(
  name: string, d: any, ctx: any, compositeBody: any,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const N = ctx.sampleCount;
  const D = compositeBody.n;          // inner iid axis size
  const distOp = compositeBody.distOp;

  // Surface every free name (broadcast args + closed-over self-refs) to
  // prepareDensityRefs in one pass via the body kwargs with kernel
  // formals substituted to the broadcast args.
  const subBodyKwargs: Record<string, any> = {};
  for (const pn of Object.keys(compositeBody.distKwargs)) {
    subBodyKwargs[pn] = _substituteKernelParams(
      compositeBody.distKwargs[pn], compositeBody.params,
      compositeBody.paramKwargs, d.kwargIRs || {});
  }
  const aggregateIR: any = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: distOp }],
    kwargs: subBodyKwargs,
  };

  return prepareDensityRefs(aggregateIR, ctx, 'broadcast').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const baseEnv: Record<string, any> = {};
    for (const k in fixedEnv) baseEnv[k] = fixedEnv[k];
    const refNames = Object.keys(refArrays);

    // Determine K (outer broadcast-cell axis). axisStack is
    // AUTHORITATIVE for a literal kernel_broadcast size; otherwise
    // discover K from the first broadcast arg carrying a cell axis > 1.
    const _bindingStack = axisStackMod.bindingAxisStack(name, ctx);
    const _axisStackK = axisStackMod.outerAxisSize(_bindingStack, 'kernel_broadcast');
    const _kFromAxisStack = !!(_axisStackK && _axisStackK > 1);
    let K = _kFromAxisStack ? (_axisStackK as number) : 1;

    // Evaluate + classify each broadcast arg over the atom batch.
    const argVals: Record<string, any> = {};
    const argCls: Record<string, any> = {};
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
      const cls = shared.classifyBroadcastArg(v, N, usesAtom);
      argVals[argName] = v;
      argCls[argName] = cls;
      const argK = cls.K || 1;
      if (argK > 1) {
        if (K === 1) K = argK;
        else if (K !== argK) {
          return Promise.reject(new Error('broadcast(' + d.distOp + '): '
            + (_kFromAxisStack
              ? 'internal: static axisStack kernel-broadcast axis (' + K
                + ') disagrees with runtime cell length (' + argK
                + ') on arg \'' + argName + '\' — producer/runtime drift'
              : 'incompatible cell-axis lengths (' + K + ' vs ' + argK
                + ') on arg \'' + argName + '\'')));
        }
      }
      const argD = cls.D || 1;
      if (argD > 1 && argD !== D) {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): vec-per-cell dim ' + argD + ' on arg \'' + argName
          + '\' must equal the inner iid count ' + D));
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty collection argument'));
    }

    const count = N * K * D;

    // Lay the broadcast args out as `__bf_<formal>` buffers — these are
    // the cell/inner-axis carriers (K / KD / N / NK / NKD), so their
    // intrinsic dims ARE the (K, D) axes by construction. They feed the
    // kernel formals.
    const axes = [K, D];                // parallel axes: cell K, inner-iid D
    const flatRefs: Record<string, any> = {};
    const bcKwargs: Record<string, any> = {};
    for (const argName of Object.keys(d.kwargIRs || {})) {
      const flatName = '__bf_' + argName;
      const sp = _spanOf(argCls[argName], 2, 0, 1);
      flatRefs[flatName] = valueLib.batchedScalar(
        _layoutFlat(argVals[argName], N, axes, sp.atomVaries, sp.axisSizes));
      bcKwargs[argName] = { kind: 'ref', ns: 'self', name: flatName };
    }

    // Substitute formals → `__bf_` refs in the inner-dist body kwargs and
    // collect the body's free names, so we lay out EXACTLY the closed-over
    // refs it touches — not every name prepareDensityRefs surfaced (the
    // raw broadcast-arg bindings are superseded by their `__bf_` layouts).
    const subKwargs: Record<string, any> = {};
    const bodyRefNames = new Set<string>();
    for (const pn of Object.keys(compositeBody.distKwargs)) {
      const sub = _substituteKernelParams(
        compositeBody.distKwargs[pn], compositeBody.params,
        compositeBody.paramKwargs, bcKwargs);
      subKwargs[pn] = sub;
      orchestrator.collectSelfRefs(sub).forEach((n: string) => bodyRefNames.add(n));
    }

    // A closed-over ref the body uses is a per-atom SCALAR: it broadcasts
    // across the cell (K) and inner (D) axes (kind 'N'). Fold each to the
    // count axis. A per-atom VECTOR closed-over ref ([N, m] reduced or
    // indexed inside the body) is outside the scalar-inner body class —
    // refuse it loudly rather than fold its intrinsic m axis into the cell
    // axis (which would read out of bounds / corrupt silently). Fixed-phase
    // refs resolve via baseEnv and aren't in refArrays (rv === undefined).
    for (const rn of bodyRefNames) {
      if (Object.prototype.hasOwnProperty.call(flatRefs, rn)) continue;   // `__bf_*`
      const rv = refArrays[rn];
      if (rv === undefined) continue;                                     // baseEnv (fixed)
      const cls = shared.classifyBroadcastArg(rv, N, true);
      if (cls.kind !== 'N' && cls.kind !== 'scalar') {
        const shp = valueLib.isValue(rv) ? JSON.stringify(rv.shape) : typeof rv;
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): closed-over ref \'' + rn + '\' resolved to ' + shp
          + ' (kind ' + cls.kind + ') — batch-flatten scalar-inner bodies'
          + ' take per-atom scalars (a per-atom vector reduced/indexed in'
          + ' the body is unsupported; the lift normally hoists such a'
          + ' reduction to its own per-atom-scalar binding)'));
      }
      const sp = _spanOf(cls, 2, 0, 1);
      flatRefs[rn] = valueLib.batchedScalar(
        _layoutFlat(rv, N, axes, sp.atomVaries, sp.axisSizes));
    }

    // Evaluate each inner-dist param ONCE over the count = N·K·D batch.
    // Resolve each to a worker kwarg: a constant → lit; a [count] column →
    // ref (carried in sampleRefs, the only refs the worker needs).
    const distKwargs: Record<string, any> = {};
    const sampleRefs: Record<string, any> = {};
    for (const pn of Object.keys(subKwargs)) {
      let pv: any;
      try {
        pv = sampler.evaluateExprN(subKwargs[pn], flatRefs, count, baseEnv, undefined);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
      const r = _resolveBatchFlattenParam(pv, count);
      if (!r) {
        const shp = valueLib.isValue(pv) ? JSON.stringify(pv.shape) : typeof pv;
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): batch-flatten param \'' + pn + '\' resolved to ' + shp
          + ' (expected scalar or [' + count + '])'));
      }
      if (r.kind === 'col') {
        const pr = '__p_' + pn;
        sampleRefs[pr] = valueLib.batchedScalar(r.data);
        distKwargs[pn] = { kind: 'ref', ns: 'self', name: pr };
      } else {
        distKwargs[pn] = { kind: 'lit', value: r.value };
      }
    }

    const distIR = { kind: 'call', op: distOp, kwargs: distKwargs };
    const workerMsg: any = {
      type: 'sampleN', ir: distIR, count: count,
      refArrays: sampleRefs,
      seed: nameSeed(name, ctx.rootKey),
    };
    return pushFixedEnv(ctx, fixedEnv)
      .then(() => ctx.sendWorker(workerMsg))
      .then((reply: any) => {
        // reply.samples is flat [count] in (i, j, r) = [N, K, D] order.
        const out = reply.samples;
        if (D > 1) {
          const value = { shape: [N | 0, K, D], data: out };
          return Object.assign(
            empirical.arrayMeasure(out, [K, D], null),
            { value: value, logTotalmass: 0, n_eff: N });
        }
        const value = { shape: [N | 0, K], data: out };
        return Object.assign(
          empirical.arrayMeasure(out, [K], null),
          { value: value, logTotalmass: 0, n_eff: N });
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
// Phase 8 — nested-broadcast de-nesting (single-shot two-axis fold)
// =====================================================================
//
// A nested kernel-broadcast `broadcast(kernelof(broadcast(D, inner_kw),
// outer_kw), outer_args)` yields a [K_outer, K_inner] block per atom.
// Batch-flatten folds BOTH parallel axes (outer K_outer × inner K_inner)
// into ONE count = N·K_outer·K_inner batch — the iid leg's move with two
// user axes (engine-concepts §20.10). It reads the static axis ladder
// `[kernel_broadcast K_outer, broadcast K_inner]` the dissolver now
// records on the binding (enabler 2f1a542), lays every input out to the
// count via the shared `_layoutFlat` (outer args span axis 0, inner
// collections span axis 1, closed-over per-atom scalars span neither),
// evaluates the inner dist's params ONCE, issues ONE sampleN, reshapes.
//
// Laying every input out BEFORE evaluating is what makes it MIXING-
// TOLERANT: an inner kwarg like `mu = patient_intercept .+ visit_effect`
// combines an outer-axis and an inner-axis term, which only resolves once
// both are materialised on the common count axis — the per-cell loop
// couldn't (it evaluated inner kwargs per outer cell). The vector-output-
// inner sub-case (MvNormal → bijection registry, §22) and any symbolic /
// unrecognised shape fall back to the per-cell reference below.

function _executeNestedBroadcastComposite(
  name: string, d: any, ctx: any, compositeBody: any,
): Promise<any> {
  // Dispatch: a scalar-inner nested broadcast with a statically-known
  // axis ladder folds; everything else uses the per-cell reference.
  if (!compositeBody.innerIsVectorOutput) {
    const stack = axisStackMod.bindingAxisStack(name, ctx);
    const Kout = axisStackMod.outerAxisSize(stack, 'kernel_broadcast');
    const Kin = axisStackMod.outerAxisSize(stack, 'broadcast');
    if (Kout && Kout >= 1 && Kin && Kin >= 1) {
      return _executeNestedBroadcastBatchFlatten(name, d, ctx, compositeBody, Kout, Kin);
    }
  }
  return _executeNestedBroadcastPerCell(name, d, ctx, compositeBody);
}

function _executeNestedBroadcastBatchFlatten(
  name: string, d: any, ctx: any, compositeBody: any, Kout: number, Kin: number,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const N = ctx.sampleCount;
  const innerKwargs = compositeBody.innerKwargs;
  const distOp = compositeBody.innerDistOp;
  const axes = [Kout, Kin];           // parallel axes: outer cell, inner cell
  const count = N * Kout * Kin;

  // Surface every free name (outer args + inner collections + closed-over
  // refs) to prepareDensityRefs via the inner kwargs with outer formals
  // substituted to the outer broadcast args.
  const substForRefs: Record<string, any> = {};
  for (const pn of Object.keys(innerKwargs)) {
    substForRefs[pn] = _substituteKernelParams(
      innerKwargs[pn], compositeBody.params, compositeBody.paramKwargs, d.kwargIRs || {});
  }
  const aggregateIR: any = {
    kind: 'call', op: 'broadcast',
    args: [{ kind: 'ref', ns: 'self', name: distOp }],
    kwargs: substForRefs,
  };

  return prepareDensityRefs(aggregateIR, ctx, 'broadcast').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const baseEnv: Record<string, any> = {};
    for (const k in fixedEnv) baseEnv[k] = fixedEnv[k];
    const refNames = Object.keys(refArrays);
    const fallback = () => _executeNestedBroadcastPerCell(name, d, ctx, compositeBody);

    const flatRefs: Record<string, any> = {};

    // Outer args → `__o_<formal>` refs spanning the OUTER axis (0).
    const outerBcKwargs: Record<string, any> = {};
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
      const cls = shared.classifyBroadcastArg(v, N, usesAtom);
      if (((cls.K || 1) > 1 && cls.K !== Kout) || (cls.D || 1) > 1) return fallback();
      const sp = _spanOf(cls, 2, 0, 1);                 // cell axis → outer (0)
      const flatName = '__o_' + argName;
      flatRefs[flatName] = valueLib.batchedScalar(
        _layoutFlat(v, N, axes, sp.atomVaries, sp.axisSizes));
      outerBcKwargs[argName] = { kind: 'ref', ns: 'self', name: flatName };
    }

    // Substitute outer formals → `__o_` refs; collect the inner body's
    // free names so we lay out exactly the inner collections + closed-over
    // refs it uses.
    const subKwargs: Record<string, any> = {};
    const bodyRefNames = new Set<string>();
    for (const pn of Object.keys(innerKwargs)) {
      const sub = _substituteKernelParams(
        innerKwargs[pn], compositeBody.params, compositeBody.paramKwargs, outerBcKwargs);
      subKwargs[pn] = sub;
      orchestrator.collectSelfRefs(sub).forEach((n: string) => bodyRefNames.add(n));
    }

    // Inner collections span the INNER axis (1); closed-over per-atom
    // scalars span neither. A non-`__o_` ref whose cell length is K_inner
    // is an inner collection; an 'N'/'scalar' is closed over; anything
    // else falls back to the per-cell reference.
    for (const rn of bodyRefNames) {
      if (Object.prototype.hasOwnProperty.call(flatRefs, rn)) continue;   // `__o_*`
      const fromRefs = refArrays[rn] !== undefined;
      const v = fromRefs ? refArrays[rn] : baseEnv[rn];
      if (v === undefined) continue;                                      // resolved elsewhere
      const cls = shared.classifyBroadcastArg(v, N, fromRefs);
      if (cls.kind === 'N' || cls.kind === 'scalar') {
        const sp = _spanOf(cls, 2, 0, 1);                                 // spans neither
        flatRefs[rn] = valueLib.batchedScalar(
          _layoutFlat(v, N, axes, sp.atomVaries, sp.axisSizes));
      } else if ((cls.kind === 'K' || cls.kind === 'NK')
                 && cls.K === Kin && (cls.D || 1) === 1) {
        const sp = _spanOf(cls, 2, 1, 0);                                 // cell axis → inner (1)
        flatRefs[rn] = valueLib.batchedScalar(
          _layoutFlat(v, N, axes, sp.atomVaries, sp.axisSizes));
      } else {
        return fallback();
      }
    }

    // Evaluate each inner-dist param ONCE over the count batch.
    const distKwargs: Record<string, any> = {};
    const sampleRefs: Record<string, any> = {};
    for (const pn of Object.keys(subKwargs)) {
      let pv: any;
      try {
        pv = sampler.evaluateExprN(subKwargs[pn], flatRefs, count, baseEnv, undefined);
      } catch (err) {
        return Promise.reject(err instanceof Error ? err : new Error(String(err)));
      }
      const r = _resolveBatchFlattenParam(pv, count);
      if (!r) return fallback();
      if (r.kind === 'col') {
        const pr = '__p_' + pn;
        sampleRefs[pr] = valueLib.batchedScalar(r.data);
        distKwargs[pn] = { kind: 'ref', ns: 'self', name: pr };
      } else {
        distKwargs[pn] = { kind: 'lit', value: r.value };
      }
    }

    const distIR = { kind: 'call', op: distOp, kwargs: distKwargs };
    const workerMsg: any = {
      type: 'sampleN', ir: distIR, count: count,
      refArrays: sampleRefs,
      seed: nameSeed(name, ctx.rootKey),
    };
    return pushFixedEnv(ctx, fixedEnv)
      .then(() => ctx.sendWorker(workerMsg))
      .then((reply: any) => {
        // reply.samples is flat [count] in (i, j_outer, k_inner) order
        // = [N, K_outer, K_inner].
        const out = reply.samples;
        const value = { shape: [N | 0, Kout, Kin], data: out };
        return Object.assign(
          empirical.arrayMeasure(out, [Kout, Kin], null),
          { value: value, logTotalmass: 0, n_eff: N });
      });
  });
}

// =====================================================================
// Phase 4.4 — nested-broadcast per-cell execution (reference + vector-
// output-inner path; scalar-inner with a static ladder folds above)
// =====================================================================
//
// Outer kernel-broadcast body is itself `broadcast(<bare_dist>, kw)`:
//
//     broadcast(<outer_kernel>, outer_kwargs)
//        with outer_kernel = kernelof(broadcast(D, inner_kw), outer_kw)
//
// Each atom yields a [K_outer, K_inner] block via **two-axis per-cell
// iteration**:
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
// This is the dual-mode reference (engine-concepts §15) for the scalar-
// inner case (folded above when the ladder is static) AND the live path
// for the vector-output-inner (MvNormal) sub-case, which dispatches per
// (j, k) through the bijection registry rather than scalar sampleN.

function _executeNestedBroadcastPerCell(
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

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
// Phase 8 — joint-composite de-nesting (per-component fold = product)
// =====================================================================
//
// `broadcast(kernelof(joint(<components>), …), args)` draws, per atom, ONE
// joint product variate per cell: K cells × C independent components
// (spec §06), variate = the concat of the components' slices. The
// components are the variate STRUCTURE (eventShape), not a parallel axis,
// so they are NOT folded into the batch — each is its own draw. But the
// outer cell axis K IS a parallel axis: batch-flatten folds it into each
// component (count = N·K, one draw per component) and the joint is the
// product (concat) of the per-component results → [N, K, Σ eventDim].
// That turns the per-cell K·C worker calls into C.
//
// SCALAR components fold via one `sampleN` over count; VECTOR-OUTPUT
// components (MvNormal, eventDim n) fold via `_mvNormalFoldOverCells` (one
// affine forward over count, §22). Atom-dep mu·cov is rejected (Session-5b
// scope — `resolveIRToValue` returns null), as the former per-cell path
// did. The per-cell executor is gone; this is the only joint path.

// Fold a single VECTOR-OUTPUT (MvNormal) joint component over the cell
// axis K. Resolves the component's per-cell mu ([n] shared or [K, n]) and
// shared cov ([n, n]) from its outer-formal-substituted kwargs, Choleskys
// once, and runs `_mvNormalFoldOverCells` → `{data [N,K,n], eventDim n}`.
function _jointVectorComponentCol(
  comp: any, compositeBody: any, d: any, ctx: any,
  N: number, K: number, seed: [number, number],
): Promise<{ data: Float64Array; eventDim: number }> {
  const sampler = require('./sampler.ts');
  const valueOps = require('./value-ops.ts');
  if (comp.distOp !== 'MvNormal') {
    return Promise.reject(new Error('broadcast: joint vector-output component \''
      + comp.distOp + '\' not supported (MvNormal only; further multivariates '
      + 'land alongside their registry entries)'));
  }
  const subst = (ir: any) => ir == null ? null : _substituteKernelParams(
    ir, compositeBody.params, compositeBody.paramKwargs, d.kwargIRs || {});
  const muIR = subst(comp.distKwargs.mu);
  const covIR = subst(comp.distKwargs.cov);
  if (!muIR || !covIR) {
    return Promise.reject(new Error('broadcast: joint MvNormal component requires mu and cov'));
  }
  const muVal = orchestrator.resolveIRToValue(muIR, ctx.bindings, ctx.fixedValues);
  const covVal = orchestrator.resolveIRToValue(covIR, ctx.bindings, ctx.fixedValues);
  if (muVal == null || covVal == null) {
    return Promise.reject(new Error('broadcast: joint MvNormal component with '
      + 'atom-dep / unresolved mu·cov is not supported (Session-5b scope: '
      + 'atom-indep per-cell mu·cov only — per-atom-stochastic lands with the '
      + 'matPushfwd vector-base extension)'));
  }
  const muValue = valueLib.asValue(muVal);
  let muIsPerCell: boolean;
  let n: number;
  if (muValue.shape.length === 1) { muIsPerCell = false; n = muValue.shape[0]; }
  else if (muValue.shape.length === 2) {
    muIsPerCell = true; n = muValue.shape[1];
    if (muValue.shape[0] !== K) {
      return Promise.reject(new Error('broadcast: joint MvNormal per-cell mu rows ('
        + muValue.shape[0] + ') must equal the cell count K=' + K));
    }
  } else {
    return Promise.reject(new Error('broadcast: joint MvNormal mu must be [n] or '
      + '[K, n]; got shape=' + JSON.stringify(muValue.shape)));
  }
  const covValue = Array.isArray(covVal) && covVal.length > 0 && Array.isArray(covVal[0])
    ? valueOps._nestedToValue(covVal)
    : valueLib.asValue(covVal);
  if (covValue.shape.length !== 2 || covValue.shape[0] !== n || covValue.shape[1] !== n) {
    return Promise.reject(new Error('broadcast: joint MvNormal cov must be ' + n + 'x'
      + n + '; got shape=' + JSON.stringify(covValue.shape)));
  }
  let L: any;
  try {
    L = sampler._internal.ARITH_OPS.lower_cholesky(covValue);
  } catch (err) {
    return Promise.reject(new Error('broadcast: joint MvNormal Cholesky failed: '
      + (err as any).message));
  }
  return _mvNormalFoldOverCells(muValue, L, n, muIsPerCell, N, K, seed, ctx);
}

function _executeJointComposite(
  name: string, d: any, ctx: any, compositeBody: any,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const N = ctx.sampleCount;
  const components = compositeBody.components;
  const C = components.length;

  // Formals consumed by VECTOR-OUTPUT components — their args (per-cell
  // mu, etc.) are resolved as vector/matrix values by
  // `_jointVectorComponentCol`, NOT laid out via the scalar `__o_` path.
  const vectorFormals = new Set<string>();
  const formalSet = new Set<string>(compositeBody.params);
  for (let c = 0; c < C; c++) {
    if (!components[c].isVectorOutput) continue;
    for (const pn of Object.keys(components[c].distKwargs)) {
      orchestrator.collectSelfRefs(components[c].distKwargs[pn]).forEach((r: string) => {
        if (formalSet.has(r)) vectorFormals.add(r);
      });
    }
  }

  // Surface scalar components' free names (broadcast args + closed-over
  // refs) to prepareDensityRefs. Vector components resolve their own
  // mu·cov via resolveIRToValue, so they aren't surfaced here.
  const flatKwargs: Record<string, any> = {};
  for (let c = 0; c < C; c++) {
    if (components[c].isVectorOutput) continue;
    for (const pn of Object.keys(components[c].distKwargs)) {
      flatKwargs['__joint_c' + c + '_' + pn] = _substituteKernelParams(
        components[c].distKwargs[pn], compositeBody.params,
        compositeBody.paramKwargs, d.kwargIRs || {});
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

    // Determine the cell axis K — axisStack authoritative (a joint body
    // adds no inner axis, so the ladder is `[kernel_broadcast K]`); else
    // discover from the first SCALAR-component broadcast arg with a cell
    // axis. Vector-component args (per-cell mu, etc.) are skipped here.
    const stack = axisStackMod.bindingAxisStack(name, ctx);
    const axisStackK = axisStackMod.outerAxisSize(stack, 'kernel_broadcast');
    const kFromStack = !!(axisStackK && axisStackK > 1);
    let K = kFromStack ? (axisStackK as number) : 1;
    const argVals: Record<string, any> = {};
    const argCls: Record<string, any> = {};
    for (const argName of Object.keys(d.kwargIRs || {})) {
      if (vectorFormals.has(argName)) continue;       // resolved by the vector path
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
      if ((cls.D || 1) > 1) {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): scalar-component broadcast arg \'' + argName + '\' has inner dim '
          + cls.D + ' (vec-per-cell shapes are iid-composite territory)'));
      }
      argVals[argName] = v;
      argCls[argName] = cls;
      const argK = cls.K || 1;
      if (argK > 1) {
        if (K === 1) K = argK;
        else if (K !== argK) {
          return Promise.reject(new Error('broadcast(' + d.distOp + '): '
            + (kFromStack
              ? 'internal: static axisStack kernel-broadcast axis (' + K
                + ') disagrees with runtime cell length (' + argK
                + ') on arg \'' + argName + '\' — producer/runtime drift'
              : 'incompatible cell-axis lengths (' + K + ' vs ' + argK
                + ') on arg \'' + argName + '\'')));
        }
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty collection argument'));
    }

    const axes = [K];                   // single parallel axis: the cell K
    const count = N * K;

    // Lay scalar-component broadcast args → __o_<formal> (span the cell
    // axis), shared across scalar components.
    const flatRefs: Record<string, any> = {};
    const outerBcKwargs: Record<string, any> = {};
    for (const argName of Object.keys(argCls)) {
      const sp = _spanOf(argCls[argName], 1, 0, 0);
      const flatName = '__o_' + argName;
      flatRefs[flatName] = valueLib.batchedScalar(
        _layoutFlat(argVals[argName], N, axes, sp.atomVaries, sp.axisSizes));
      outerBcKwargs[argName] = { kind: 'ref', ns: 'self', name: flatName };
    }

    // Substitute outer formals → __o_ in SCALAR components; collect the
    // closed-over refs they use (per-atom scalars, span neither).
    const subComps: Record<number, Record<string, any>> = {};
    const bodyRefNames = new Set<string>();
    for (let c = 0; c < C; c++) {
      if (components[c].isVectorOutput) continue;
      const sub: Record<string, any> = {};
      for (const pn of Object.keys(components[c].distKwargs)) {
        const s = _substituteKernelParams(
          components[c].distKwargs[pn], compositeBody.params,
          compositeBody.paramKwargs, outerBcKwargs);
        sub[pn] = s;
        orchestrator.collectSelfRefs(s).forEach((rn: string) => bodyRefNames.add(rn));
      }
      subComps[c] = sub;
    }
    for (const rn of bodyRefNames) {
      if (Object.prototype.hasOwnProperty.call(flatRefs, rn)) continue;   // __o_*
      const v = refArrays[rn] !== undefined ? refArrays[rn] : baseEnv[rn];
      if (v === undefined) continue;                                      // baseEnv (fixed)
      const cls = shared.classifyBroadcastArg(v, N, refArrays[rn] !== undefined);
      if (cls.kind !== 'N' && cls.kind !== 'scalar') {
        const shp = valueLib.isValue(v) ? JSON.stringify(v.shape) : typeof v;
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): joint closed-over ref \'' + rn + '\' resolved to ' + shp
          + ' (kind ' + cls.kind + ') — scalar-component bodies take per-atom scalars'));
      }
      const sp = _spanOf(cls, 1, 0, 0);
      flatRefs[rn] = valueLib.batchedScalar(
        _layoutFlat(v, N, axes, sp.atomVaries, sp.axisSizes));
    }

    // One draw per component over count = N·K (independent → independent
    // seeds): a scalar component via `sampleN`, a vector-output component
    // via the affine-forward-over-count. `cells[c] = { data, eventDim }`;
    // scalar data is [count]=(i,j), vector data is [count·n]=(i,j,:).
    const cells: Array<{ data: Float64Array; eventDim: number }> = new Array(C);
    const tasks: Array<Promise<void>> = [];
    for (let c = 0; c < C; c++) {
      const cc = c;
      if (components[cc].isVectorOutput) {
        tasks.push(_jointVectorComponentCol(
          components[cc], compositeBody, d, ctx, N, K,
          nameSeed(name + ':c' + cc, ctx.rootKey),
        ).then((res) => { cells[cc] = res; }));
        continue;
      }
      const distKwargs: Record<string, any> = {};
      const sampleRefs: Record<string, any> = {};
      for (const pn of Object.keys(subComps[cc])) {
        let pv: any;
        try {
          pv = sampler.evaluateExprN(subComps[cc][pn], flatRefs, count, baseEnv, undefined);
        } catch (err) {
          return Promise.reject(err instanceof Error ? err : new Error(String(err)));
        }
        const r = _resolveBatchFlattenParam(pv, count);
        if (!r) {
          return Promise.reject(new Error('broadcast(' + d.distOp
            + '): joint component ' + cc + ' param \'' + pn + '\' resolved to an '
            + 'unsupported shape (expected scalar or [' + count + '])'));
        }
        if (r.kind === 'col') {
          const pr = '__p_c' + cc + '_' + pn;
          sampleRefs[pr] = valueLib.batchedScalar(r.data);
          distKwargs[pn] = { kind: 'ref', ns: 'self', name: pr };
        } else {
          distKwargs[pn] = { kind: 'lit', value: r.value };
        }
      }
      const distIR = { kind: 'call', op: components[cc].distOp, kwargs: distKwargs };
      tasks.push(ctx.sendWorker({
        type: 'sampleN', ir: distIR, count: count, refArrays: sampleRefs,
        seed: nameSeed(name + ':c' + cc, ctx.rootKey),
      }).then((reply: any) => { cells[cc] = { data: reply.samples, eventDim: 1 }; }));
    }

    return pushFixedEnv(ctx, fixedEnv).then(() => Promise.all(tasks)).then(() => {
      // Stitch by per-component eventDim → [N, K, E], E = Σ_c eventDim_c
      // (matches the former per-cell joint layout). out[i·K·E + j·E +
      // off_c + e] = cells[c].data[(i·K+j)·ed_c + e].
      let E = 0;
      const offsets: number[] = new Array(C);
      for (let c = 0; c < C; c++) { offsets[c] = E; E += cells[c].eventDim; }
      const out = new Float64Array(N * K * E);
      for (let c = 0; c < C; c++) {
        const ed = cells[c].eventDim, off = offsets[c], data = cells[c].data;
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < K; j++) {
            const src = (i * K + j) * ed;
            const dst = i * K * E + j * E + off;
            for (let e = 0; e < ed; e++) out[dst + e] = data[src + e];
          }
        }
      }
      const value = { shape: [N | 0, K, E], data: out };
      return Object.assign(
        empirical.arrayMeasure(out, [K, E], null),
        { value: value, logTotalmass: 0, n_eff: N });
    });
  });
}

// =====================================================================
// Phase 8 — jointchain-composite de-nesting (scan over steps)
// =====================================================================
//
// A jointchain-bodied kernel-broadcast is a Markov chain per cell: step 0
// is a base measure, each step k>0 a kernel consuming step k-1's variate
// (the carry). The steps are SEQUENTIAL (a scan), the cell axis K is
// parallel. Batch-flatten folds K into each step's sampleN (count = N·K,
// one call per step) while scanning the steps: step k-1's flat [count]
// result is ALREADY in (i, j) order, so it binds directly as step k's
// input refArray — no re-layout. K·C per-cell worker calls collapse to C
// (sequential across steps, parallel across atom×cell). This is the
// lax.scan analogue (carry = previous variate).
//
// jointchain steps are scalar dists (recogniser scope) — there is no
// vector-output sub-case, so the scan covers the FULL scalar jointchain
// class and is the only path (the former per-cell executor is deleted).
// A per-cell vector broadcast arg (D>1, vec-per-cell = iid territory) or
// an ill-formed param is a clear error, not a fallback.

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
  const inputParams = new Set<string>();
  for (let k = 1; k < C; k++) inputParams.add(steps[k].kernel.inputParam);

  // Surface every free name (broadcast args + closed-over refs across all
  // steps) via the steps' kwargs with outer formals substituted to the
  // broadcast args; the inputParam stays unbound (it's the scan carry).
  const flatKwargs: Record<string, any> = {};
  const rawOf = (k: number) => (k === 0 ? steps[0].base : steps[k].kernel);
  for (let k = 0; k < C; k++) {
    const raw = rawOf(k).distKwargs;
    for (const pn of Object.keys(raw)) {
      flatKwargs['__chain_s' + k + '_' + pn] = _substituteKernelParams(
        raw[pn], compositeBody.params, compositeBody.paramKwargs, d.kwargIRs || {});
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

    // Cell axis K (axisStack authoritative; else discover from args).
    const stack = axisStackMod.bindingAxisStack(name, ctx);
    const axisStackK = axisStackMod.outerAxisSize(stack, 'kernel_broadcast');
    const kFromStack = !!(axisStackK && axisStackK > 1);
    let K = kFromStack ? (axisStackK as number) : 1;
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
      if ((cls.D || 1) > 1) {
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): jointchain-composite broadcast arg \'' + argName + '\' has inner '
          + 'dim ' + cls.D + ' (vec-per-cell shapes are iid-composite territory)'));
      }
      argVals[argName] = v;
      argCls[argName] = cls;
      const argK = cls.K || 1;
      if (argK > 1) {
        if (K === 1) K = argK;
        else if (K !== argK) {
          return Promise.reject(new Error('broadcast(' + d.distOp + '): '
            + (kFromStack
              ? 'internal: static axisStack kernel-broadcast axis (' + K
                + ') disagrees with runtime cell length (' + argK
                + ') on arg \'' + argName + '\' — producer/runtime drift'
              : 'incompatible cell-axis lengths (' + K + ' vs ' + argK
                + ') on arg \'' + argName + '\'')));
        }
      }
    }
    if (K < 1) {
      return Promise.reject(new Error('broadcast(' + d.distOp
        + '): empty collection argument'));
    }

    const axes = [K];
    const count = N * K;

    // Broadcast args → __o_<formal> (span the cell axis); shared by steps.
    const flatRefs: Record<string, any> = {};
    const outerBcKwargs: Record<string, any> = {};
    for (const argName of Object.keys(d.kwargIRs || {})) {
      const sp = _spanOf(argCls[argName], 1, 0, 0);
      const flatName = '__o_' + argName;
      flatRefs[flatName] = valueLib.batchedScalar(
        _layoutFlat(argVals[argName], N, axes, sp.atomVaries, sp.axisSizes));
      outerBcKwargs[argName] = { kind: 'ref', ns: 'self', name: flatName };
    }

    // Closed-over refs across all steps (per-atom scalars, span neither);
    // the inputParam (carry) is bound per step during the scan, not here.
    const bodyRefNames = new Set<string>();
    for (let k = 0; k < C; k++) {
      const raw = rawOf(k).distKwargs;
      for (const pn of Object.keys(raw)) {
        const s = _substituteKernelParams(
          raw[pn], compositeBody.params, compositeBody.paramKwargs, outerBcKwargs);
        orchestrator.collectSelfRefs(s).forEach((n: string) => bodyRefNames.add(n));
      }
    }
    for (const rn of bodyRefNames) {
      if (Object.prototype.hasOwnProperty.call(flatRefs, rn)) continue;   // __o_*
      if (inputParams.has(rn)) continue;                                  // scan carry
      const v = refArrays[rn] !== undefined ? refArrays[rn] : baseEnv[rn];
      if (v === undefined) continue;
      const cls = shared.classifyBroadcastArg(v, N, refArrays[rn] !== undefined);
      if (cls.kind !== 'N' && cls.kind !== 'scalar') {
        const shp = valueLib.isValue(v) ? JSON.stringify(v.shape) : typeof v;
        return Promise.reject(new Error('broadcast(' + d.distOp
          + '): jointchain closed-over ref \'' + rn + '\' resolved to ' + shp
          + ' (kind ' + cls.kind + ') — scalar-step bodies take per-atom scalars'
          + ' (the lift normally hoists an in-body reduction to its own binding)'));
      }
      const sp = _spanOf(cls, 1, 0, 0);
      flatRefs[rn] = valueLib.batchedScalar(
        _layoutFlat(v, N, axes, sp.atomVaries, sp.axisSizes));
    }

    // Scan: one sampleN per step over count = N·K, threading the carry.
    // cols[k] is flat [count] in (i, j) order = [N, K]; step k binds its
    // inputParam to cols[k-1] directly (already count-aligned).
    const cols: Float64Array[] = new Array(C);
    let acc = Promise.resolve();
    for (let k = 0; k < C; k++) {
      const kk = k;
      acc = acc.then(() => {
        const step = rawOf(kk);
        const subMap: Record<string, any> = Object.assign({}, outerBcKwargs);
        const stepRefs: Record<string, any> = Object.assign({}, flatRefs);
        let paramNames = compositeBody.params;
        let paramKwargs = compositeBody.paramKwargs;
        if (kk > 0) {
          subMap[steps[kk].kernel.inputParam] = { kind: 'ref', ns: 'self', name: '__carry' };
          stepRefs.__carry = valueLib.batchedScalar(cols[kk - 1]);
          paramNames = [steps[kk].kernel.inputParam].concat(compositeBody.params);
          paramKwargs = [steps[kk].kernel.inputParam].concat(compositeBody.paramKwargs);
        }
        const distKwargs: Record<string, any> = {};
        const sampleRefs: Record<string, any> = {};
        for (const pn of Object.keys(step.distKwargs)) {
          const subIR = _substituteKernelParams(
            step.distKwargs[pn], paramNames, paramKwargs, subMap);
          let pv: any;
          try {
            pv = sampler.evaluateExprN(subIR, stepRefs, count, baseEnv, undefined);
          } catch (err) {
            throw err instanceof Error ? err : new Error(String(err));
          }
          const r = _resolveBatchFlattenParam(pv, count);
          if (!r) {
            const shp = valueLib.isValue(pv) ? JSON.stringify(pv.shape) : typeof pv;
            throw new Error('broadcast(' + d.distOp + '): jointchain step ' + kk
              + ' param \'' + pn + '\' resolved to ' + shp
              + ' (expected scalar or [' + count + '])');
          }
          if (r.kind === 'col') {
            const pr = '__p_s' + kk + '_' + pn;
            sampleRefs[pr] = valueLib.batchedScalar(r.data);
            distKwargs[pn] = { kind: 'ref', ns: 'self', name: pr };
          } else {
            distKwargs[pn] = { kind: 'lit', value: r.value };
          }
        }
        const distIR = { kind: 'call', op: step.distOp, kwargs: distKwargs };
        const workerMsg: any = {
          type: 'sampleN', ir: distIR, count: count,
          refArrays: sampleRefs,
          seed: nameSeed(name + ':s' + kk, ctx.rootKey),
        };
        return ctx.sendWorker(workerMsg).then((reply: any) => { cols[kk] = reply.samples; });
      });
    }
    return pushFixedEnv(ctx, fixedEnv).then(() => acc).then(() => {
      // Stitch the C steps into [N, K, C] atom-major
      // (out[i*K*C + j*C + k] = cols[k][i*K + j]).
      const out = new Float64Array(N * K * C);
      for (let k = 0; k < C; k++) {
        const col = cols[k];
        for (let i = 0; i < N; i++) {
          for (let j = 0; j < K; j++) {
            out[i * K * C + j * C + k] = col[i * K + j];
          }
        }
      }
      const value = { shape: [N | 0, K, C], data: out };
      return Object.assign(
        empirical.arrayMeasure(out, [K, C], null),
        { value: value, logTotalmass: 0, n_eff: N });
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
  // Both nested-broadcast classes fold through the shared layout/affine
  // primitives — there is no per-cell path (Phase 8 de-nesting endgame).
  //
  //   - VECTOR-OUTPUT inner (MvNormal): the two-axis vector fold derives
  //     K_outer / K_inner from the resolved (atom-indep) outer args + mu
  //     grid, so it doesn't lean on the static ladder — the inner [K_in,n]
  //     `mu` is a KD-shaped arg that would muddy a ladder read anyway.
  //   - SCALAR inner: the authoritative `[kernel_broadcast K_out,
  //     broadcast K_in]` ladder (dissolver enabler) drives the fold.
  if (compositeBody.innerIsVectorOutput) {
    return _executeNestedBroadcastVectorFold(name, d, ctx, compositeBody);
  }
  const stack = axisStackMod.bindingAxisStack(name, ctx);
  const Kout = axisStackMod.outerAxisSize(stack, 'kernel_broadcast');
  const Kin = axisStackMod.outerAxisSize(stack, 'broadcast');
  if (Kout && Kout >= 1 && Kin && Kin >= 1) {
    return _executeNestedBroadcastBatchFlatten(name, d, ctx, compositeBody, Kout, Kin);
  }
  // A non-static ladder (a symbolic axis length) is not folded — no silent
  // per-cell fallback (the fold is the only path, as for jointchain). The
  // concrete-need-gated follow-up is to resolve symbolic axis sizes from
  // fixedValues at materialise time and feed the fold (TODO Phase 8).
  return Promise.reject(new Error('broadcast: nested-broadcast \'' + name
    + '\' has a non-static axis ladder (outer=' + Kout + ', inner=' + Kin
    + ') — folding a symbolic nested ladder is not yet supported; the axis '
    + 'sizes must be statically resolvable (use literal / fixed-phase '
    + 'collection lengths).'));
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
    // No per-cell fallback — the fold is the only path (Phase 8). An arg
    // shape the fold can't lay out is a clear error with the offending
    // reason, not a silent reroute.
    const unfoldable = (reason: string) => Promise.reject(new Error(
      'broadcast: nested-broadcast scalar fold for \'' + name
      + '\' cannot fold: ' + reason));

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
      if (((cls.K || 1) > 1 && cls.K !== Kout) || (cls.D || 1) > 1) {
        return unfoldable('outer arg \'' + argName + '\' has cell axis '
          + (cls.K || 1) + ' / inner dim ' + (cls.D || 1) + ' (expected a '
          + 'scalar-per-cell arg on the K_outer=' + Kout + ' axis; vec-per-'
          + 'cell shapes belong to iid-composite)');
      }
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
    // else is unfoldable (a clear error — no per-cell fallback).
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
        return unfoldable('inner-body ref \'' + rn + '\' is neither closed-'
          + 'over (N/scalar) nor a K_inner=' + Kin + ' collection (got cell '
          + 'axis ' + (cls.K || 1) + ' / inner dim ' + (cls.D || 1) + ')');
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
      if (!r) {
        return unfoldable('inner-dist param \'' + pn + '\' did not resolve to '
          + 'a constant or a length-' + count + ' column over the count batch');
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
// Phase 8 — nested-broadcast de-nesting: VECTOR-OUTPUT inner (MvNormal)
// =====================================================================
//
// Outer kernel-broadcast body is itself `broadcast(MvNormal, mu, cov)`:
//
//     broadcast(<outer_kernel>, outer_args)
//        with outer_kernel = kernelof(broadcast(MvNormal, mu, cov), outer_kw)
//
// The two parallel axes (outer cell K_out × inner cell K_in) fold into ONE
// affine forward over count = N·K_out·K_in — the vector-output twin of the
// scalar nested fold (`_executeNestedBroadcastBatchFlatten`), built on the
// shared `_mvNormalFoldOverCells` primitive (engine-concepts §20.10 + §22).
//
// Scope — atom-INDEPENDENT per-cell mean grid + a SHARED cov, matching the
// bare/joint vector-output folds: the inner MvNormal's mu/cov resolve
// statically (`resolveIRToValue`), so the outer kernel param threads into
// `mu` only through atom-indep collections / fixed values. Per-atom-
// stochastic mu/cov (a latent outer mean threaded into the inner draw) and
// per-cell cov (one Cholesky per cell) defer to the matPushfwd vector-base
// extension — the same boundary the bare / joint vector-output folds hold.
//
// We walk the OUTER axis once (cheap — no worker calls): per outer cell j we
// substitute the kernel formals with that cell's arg slice and resolve the
// inner `mu` to [K_in, n] (inner-cell-varying) or [n] (inner-shared), laying
// it into row (j·K_in + k) of a [K_out·K_in, n] grid. The shared cov resolves
// once → one Cholesky → L. `_mvNormalFoldOverCells` over count = N·K_out·K_in
// then writes its [count, n] output in (i, j_out, k_in, :) order — EXACTLY
// the [N, K_out, K_in·n] atom-major contract, so no restitch is needed.
// (§03: K_out, K_in are PARALLEL axes folded ABOVE the intrinsic n axis.)
function _executeNestedBroadcastVectorFold(
  name: string, d: any, ctx: any, compositeBody: any,
): Promise<any> {
  const sampler = require('./sampler.ts');
  const valueOps = require('./value-ops.ts');
  const N = ctx.sampleCount;
  const distOp = compositeBody.innerDistOp;
  if (distOp !== 'MvNormal') {
    return Promise.reject(new Error('broadcast: nested-broadcast vector-output '
      + 'inner \'' + distOp + '\' not supported (MvNormal only; further '
      + 'multivariates land alongside their bijection-registry entries)'));
  }
  const innerKwargs = compositeBody.innerKwargs || {};
  const muInnerIR = innerKwargs.mu;
  const covInnerIR = innerKwargs.cov;
  if (!muInnerIR || !covInnerIR) {
    return Promise.reject(new Error('broadcast: nested-broadcast MvNormal inner '
      + 'requires mu and cov kwargs'));
  }

  // (1) Outer broadcast args → atom-indep values (scope); derive K_outer.
  //     A scalar-per-cell collection [K_out] supplies the outer axis; a
  //     plain scalar broadcasts. Atom-dep / vec-per-cell / unresolved outer
  //     args are out of scope (a clear error, not a silent reroute).
  const outerArgVals: Record<string, any> = {};
  let Kout = 1;
  for (const argName of Object.keys(d.kwargIRs || {})) {
    const rv = orchestrator.resolveIRToValue(
      d.kwargIRs[argName], ctx.bindings, ctx.fixedValues);
    if (rv == null) {
      return Promise.reject(new Error('broadcast: nested-broadcast MvNormal '
        + 'inner with atom-dep / unresolved outer arg \'' + argName + '\' is '
        + 'not supported (atom-indep scope; per-atom-stochastic threading '
        + 'defers to the matPushfwd vector-base extension)'));
    }
    const av = valueLib.asValue(rv);
    if (av.shape.length > 1) {
      return Promise.reject(new Error('broadcast: nested-broadcast outer arg \''
        + argName + '\' is vec-per-cell (shape ' + JSON.stringify(av.shape)
        + ') — vec-per-cell shapes belong to iid-composite'));
    }
    const len = av.shape.length === 0 ? 1 : av.shape[0];
    if (len > 1) {
      if (Kout === 1) Kout = len;
      else if (Kout !== len) {
        return Promise.reject(new Error('broadcast: nested-broadcast '
          + 'incompatible outer cell-axis lengths (' + Kout + ' vs ' + len
          + ') on arg \'' + argName + '\''));
      }
    }
    outerArgVals[argName] = av;
  }

  // (2) Shared cov — resolve ONCE (no per-cell substitution). A cov that
  //     references the outer kernel formal is per-cell cov (one Cholesky per
  //     cell) → does not resolve here → out of scope, like the bare gate.
  const covVal = orchestrator.resolveIRToValue(
    covInnerIR, ctx.bindings, ctx.fixedValues);
  if (covVal == null) {
    return Promise.reject(new Error('broadcast: nested-broadcast MvNormal cov '
      + 'did not resolve to a shared value (per-cell / outer-param-dependent '
      + 'cov defers to per-cell Cholesky support)'));
  }
  const covValue = Array.isArray(covVal) && covVal.length > 0 && Array.isArray(covVal[0])
    ? valueOps._nestedToValue(covVal)
    : valueLib.asValue(covVal);
  if (covValue.shape.length !== 2 || covValue.shape[0] !== covValue.shape[1]) {
    return Promise.reject(new Error('broadcast: nested-broadcast MvNormal cov '
      + 'must be a square n×n matrix; got shape='
      + JSON.stringify(covValue.shape)));
  }
  const n = covValue.shape[0];
  let L: any;
  try {
    L = sampler._internal.ARITH_OPS.lower_cholesky(covValue);
  } catch (err) {
    return Promise.reject(new Error('broadcast: nested-broadcast MvNormal '
      + 'Cholesky failed: ' + (err as any).message));
  }

  // (3) Per outer cell j: substitute the kernel formals with cell j's arg
  //     slice, resolve the inner mu, and lay it into the [K_out·K_in, n]
  //     grid in (j_out, k_in) row-major order. K_inner is established from
  //     the first cell's mu and required uniform across cells (rectangular
  //     nesting — a ragged inner axis is rejected, not silently flattened).
  let Kin = -1;
  let grid: Float64Array | null = null;
  for (let j = 0; j < Kout; j++) {
    const bcKwargs: Record<string, any> = {};
    for (const argName of Object.keys(d.kwargIRs || {})) {
      const av = outerArgVals[argName];
      const s = av.data.length === 1 ? av.data[0] : av.data[j];
      bcKwargs[argName] = { kind: 'lit', value: s };
    }
    const muSub = _substituteKernelParams(
      muInnerIR, compositeBody.params, compositeBody.paramKwargs, bcKwargs);
    const mv = orchestrator.resolveIRToValue(muSub, ctx.bindings, ctx.fixedValues);
    if (mv == null) {
      return Promise.reject(new Error('broadcast: nested-broadcast MvNormal '
        + 'inner mu did not resolve at outer cell ' + j + ' (atom-indep '
        + 'scope; per-atom-stochastic mu defers to the matPushfwd vector-base '
        + 'extension)'));
    }
    const muValue = valueLib.asValue(mv);
    let cellKin: number;
    if (muValue.shape.length === 2) {
      if (muValue.shape[1] !== n) {
        return Promise.reject(new Error('broadcast: nested-broadcast MvNormal '
          + 'mu event-dim (' + muValue.shape[1] + ') != cov n (' + n
          + ') at outer cell ' + j));
      }
      cellKin = muValue.shape[0];
    } else if (muValue.shape.length === 1) {
      if (muValue.shape[0] !== n) {
        return Promise.reject(new Error('broadcast: nested-broadcast MvNormal '
          + 'shared mu length (' + muValue.shape[0] + ') != cov n (' + n
          + ') at outer cell ' + j));
      }
      cellKin = 1;            // inner-shared mu (degenerate K_inner = 1)
    } else {
      return Promise.reject(new Error('broadcast: nested-broadcast MvNormal '
        + 'mu must be [n] or [K_inner, n]; got shape='
        + JSON.stringify(muValue.shape) + ' at outer cell ' + j));
    }
    if (Kin < 0) { Kin = cellKin; grid = new Float64Array(Kout * Kin * n); }
    else if (cellKin !== Kin) {
      return Promise.reject(new Error('broadcast: nested-broadcast ragged '
        + 'inner cell-axis lengths across outer cells (' + Kin + ' vs '
        + cellKin + ' at outer ' + j + ') — rectangular nesting required'));
    }
    const perInner = muValue.shape.length === 2 && cellKin > 1;
    for (let k = 0; k < Kin; k++) {
      const src = perInner ? k * n : 0;
      const dst = (j * Kin + k) * n;
      for (let l = 0; l < n; l++) grid![dst + l] = muValue.data[src + l];
    }
  }

  // (4) Fold over count = N · K_out · K_in: one shared Cholesky, the per-cell
  //     means laid out, ONE affine forward. `_mvNormalFoldOverCells` returns
  //     [count, n] in (i, jk, :) order = the [N, K_out, K_in·n] contract.
  const muGrid = { shape: [Kout * Kin, n], data: grid as Float64Array };
  return _mvNormalFoldOverCells(
    muGrid, L, n, true, N, Kout * Kin,
    nameSeed(name + ':iidbase', ctx.rootKey), ctx,
  ).then((res: { data: Float64Array }) => {
    const out = res.data;
    const innerWidth = Kin * n;
    const value = { shape: [N | 0, Kout, innerWidth], data: out };
    return Object.assign(
      empirical.arrayMeasure(out, [Kout, innerWidth], null),
      { value: value, logTotalmass: 0, n_eff: N },
    );
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

// Fold a per-cell MvNormal over the cell axis K into ONE affine forward
// over count = N·K (engine-concepts §20.10 + §22). `muValue` is the
// per-cell mean ([n] shared, muIsPerCell=false, or [K, n]); `L` is the
// shared lower-Cholesky of the cov ([n, n]). Returns the flat [count·n]
// buffer = [N, K, n] in (i, j, :) order with its eventDim n: the shared L
// matvecs every (i, j) position, the atom-indep per-cell mean lays out
// repeated across atoms. The intrinsic n axis is NOT folded (§03). Reused
// by the bare MvNormal broadcast and the joint vector-output-component
// fold (both shared-cov, Session-5b scope).
function _mvNormalFoldOverCells(
  muValue: any, L: any, n: number, muIsPerCell: boolean,
  N: number, K: number, seed: [number, number], ctx: any,
): Promise<{ data: Float64Array; eventDim: number }> {
  const bijRegistry = require('./bijection-registry.ts');
  const count = N * K;
  const muFlat = new Float64Array(count * n);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < K; j++) {
      const dst = (i * K + j) * n;
      const src = muIsPerCell ? j * n : 0;
      for (let l = 0; l < n; l++) muFlat[dst + l] = muValue.data[src + l];
    }
  }
  return ctx.sendWorker({
    type: 'sampleN',
    ir: { kind: 'call', op: 'Normal',
          kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } } },
    count: count, repeat: n,
    refArrays: {},
    seed: seed,
  }).then((reply: any) => {
    const z = { shape: [count, n], data: reply.samples };
    const y = bijRegistry.affineAtomBatchedForward(
      z, { L: L, b: { shape: [count, n], data: muFlat } }, count);
    return { data: y.data as Float64Array, eventDim: n };
  });
}

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

  // Batch-flatten the cell axis K into ONE affine forward over count =
  // N·K (Phase 8 §20.10): see `_mvNormalFoldOverCells`. Replaces the
  // former per-cell loop (K affine forwards + z-slicing) with one call.
  return _mvNormalFoldOverCells(
    muValue, L, n, muIsPerCell, N, K,
    nameSeed(name + ':iidbase', ctx.rootKey), ctx,
  ).then((res: { data: Float64Array }) => {
    const out = res.data;
    const value = { shape: [N | 0, K, n], data: out };
    return Object.assign(
      empirical.arrayMeasure(out, [K, n], null),
      { value: value, logTotalmass: 0, n_eff: N },
    );
  });
}

module.exports = { matKernelBroadcast };

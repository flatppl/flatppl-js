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

const { nameSeed, measureFromValue, prepareDensityRefs, pushFixedEnv } = shared;

function matKernelBroadcast(name: string, d: DerivationKernelBroadcast, ctx: any) {
  const sampler = require('./sampler.ts');
  let params: string[] = (sampler._internal.REGISTRY[d.distOp] || {}).params;
  // Composite iid-bodied kernel (fusion (b) Phase F, engine-concepts
  // §20.10.11). When d.distOp resolves to a user-defined kernel
  // binding whose body is `lawof(iid(<BuiltinDistCall>, <n_literal>))`,
  // unfold the placeholders and route through the existing per-cell
  // sampleN loop with `repeat = n_inner` so each broadcast cell
  // produces an iid block of size n. Result shape per atom: [G, n].
  let iidComposite: any = null;
  if (!params) {
    iidComposite = _detectIidKernelBody(d, ctx);
    if (!iidComposite) {
      return Promise.reject(new Error(
        'broadcast: unknown distribution kernel ' + d.distOp));
    }
    params = iidComposite.distParams;
  }
  // Map parameter slot → its IR (kwarg form wins, else positional).
  // For the iid-composite case, the params come from the INNER dist
  // call's kwargs (with kernel-param placeholders substituted by the
  // broadcast args); other kwargs (closed-over self-refs) pass
  // through unchanged.
  const paramIRs: Record<string, any> = {};
  if (iidComposite) {
    // Substitute kernel placeholders in distCall's kwargs with the
    // broadcast args (d.kwargIRs maps surface kw → broadcast arg IR).
    for (const pn of Object.keys(iidComposite.distKwargs)) {
      paramIRs[pn] = _substituteKernelParams(
        iidComposite.distKwargs[pn],
        iidComposite.params, iidComposite.paramKwargs, d.kwargIRs || {});
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
  const effectiveDistOp = iidComposite ? iidComposite.distOp : d.distOp;
  const iidN = iidComposite ? iidComposite.n : 1;
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
// Composite iid-bodied kernel detection + param substitution
// (engine-concepts §20.10.11 / fusion (b) Phase F)
// =====================================================================
//
// Detects when `d.distOp` is a user binding whose IR is
//   functionof(lawof(iid(<BuiltinDistCall>(<kw>), <n_literal>)),
//              <kernel_params>)
// — the kernelof(iid(...)) pattern motivating hierarchical models
// (Pyro plate-over-plate, lme4 / Bambi random-intercepts).
//
// Returns the unpacked descriptor on match (param names, the inner
// builtin distOp, its kwargs IR, the literal iid count, etc.); null
// otherwise.
// Delegate to the shared classifier in `kernel-broadcast-shape.ts`.
// P7 (LANDED 2026-05-30) hoisted the structural recognition into a
// single source of truth; the runtime form passes `ctx.fixedValues`
// to admit ref-to-fixed-integer n (in addition to literal n the
// classify-time path accepts).
const _kernelBroadcastShape = require('./kernel-broadcast-shape.ts');
function _detectIidKernelBody(d: any, ctx: any): any | null {
  if (!ctx || !ctx.bindings) return null;
  return _kernelBroadcastShape.detectIidKernelBinding(
    d.distOp, ctx.bindings, ctx.fixedValues);
}

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

module.exports = { matKernelBroadcast };

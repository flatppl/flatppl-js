'use strict';

// =====================================================================
// KERNEL_BROADCAST_FAST_PATHS — per-distOp closed-form handler registry
// =====================================================================
//
// `broadcast(Dist, args…)` produces a vector-atom independent-product
// measure of shape [N, K]. For some distributions, certain parameter-
// shape combinations admit closed-form vectorised generation that
// beats the general per-cell `sampleN` loop by orders of magnitude.
// (Example: `broadcast(Normal, mu_vec, sigma_vec)` with atom-indep μ
// and σ → MvNormal with a diagonal covariance, sampled via one worker
// `sampleN(Normal(0,1))` call + diag-Cholesky transform.)
//
// This registry is the home for those handlers. Each handler declares:
//
//   - `label`: human-readable identifier for diagnostics.
//   - `match(ctx)`: predicate over the materialiser call context. Must
//     be cheap (no IR rewrites, no worker dispatch); just shape /
//     phase checks.
//   - `execute(ctx)`: produces the EmpiricalMeasure. Called only after
//     `match` returned true.
//
// Multiple handlers may register against the same distOp (first match
// wins). This lets a future atom-aware handler co-exist with an
// atom-indep handler without a sibling registry — the atom-aware one
// registers FIRST so it gets first pick; the atom-indep one is the
// fallback.
//
// `tryKernelBroadcastFastPath(ctx)` is the dispatch entry. Returns the
// EmpiricalMeasure promise if a handler matched; null otherwise.
// `matKernelBroadcast` (in `mat-broadcast.ts`) calls it after param
// evaluation, K determination, and the K < 1 check; on null, it falls
// through to the general per-cell `sampleN` loop.
//
// === Cross-engine architecture note ===========================
// The fast-path registry surface (per-distOp handlers + match/execute
// shape) is cross-engine: any FlatPPL engine that wants to vectorise
// kernel-broadcasts beyond per-cell calls needs an analogous registry
// keyed by the distribution name + shape predicates. The handlers
// themselves are engine-specific (closed-form formulas vs MLIR
// codegen rules vs Reactant.jl compile rules); the SURFACE is shared.

import type { DerivationKernelBroadcast } from './engine-types';

// ---------------------------------------------------------------------
// Handler interface
// ---------------------------------------------------------------------
//
// Type-only declarations (no `export` keyword — esbuild would
// classify this file as ESM and break the `module.exports` at the
// bottom in CSP-strict bundles; see engine/ARCHITECTURE.md
// "Bundle build gotchas — ESM/CJS mixing"). Consumers in TypeScript
// can import via `import type` on the *.d.ts barrel; the runtime
// API is the `module.exports` shape below.

/**
 * Context passed to every fast-path `match` and `execute` call.
 * Materialised once by `matKernelBroadcast` after it has finished
 * its setup pass (param evaluation, K determination, etc.).
 */
interface KernelBroadcastFastPathCtx {
  /** The kernel-broadcast derivation. `d.distOp` is the surface name
   *  (registry key); `d.argIRs` / `d.kwargIRs` carry the broadcast
   *  args' IR. */
  d: DerivationKernelBroadcast;
  /** The materialiser context — handlers use `ctx.sendWorker`,
   *  `ctx.rootKey`, `ctx.sampleCount`, etc. */
  ctx: any;
  /** The binding name being materialised. Used for `nameSeed`. */
  name: string;
  /** The broadcast width — number of independent draws per atom. */
  K: number;
  /** Engine atom count — leading axis of the result Value. */
  N: number;
  /** Per-param resolved values (output of `evaluateExprN`). Keys are
   *  parameter names (kwarg form). Values are Float64Array / Value /
   *  number / boolean depending on resolution. */
  paramVals: Record<string, any>;
  /** True if any parameter expression references an atom-batched
   *  ref. Closed-form handlers typically require false. */
  anyAtomDep: boolean;
}

interface KernelBroadcastFastPath {
  /** Human-readable label used in diagnostics / logging. */
  label: string;
  /** Cheap predicate — returns true if this handler can handle the
   *  given context. */
  match: (ctx: KernelBroadcastFastPathCtx) => boolean;
  /** Produces the EmpiricalMeasure. Only called when `match` returned
   *  true. Returns a Promise (kernel-broadcast materialisation is
   *  asynchronous via `ctx.sendWorker`). */
  execute: (ctx: KernelBroadcastFastPathCtx) => Promise<any>;
}

// ---------------------------------------------------------------------
// Registry + dispatch
// ---------------------------------------------------------------------

/** distOp name → ordered list of handlers (first-match-wins). */
const REGISTRY: Map<string, KernelBroadcastFastPath[]> = new Map();

/**
 * Register a fast-path handler under `distOp`. Handlers registered
 * earlier are tried first.
 *
 * Typical pattern: the most-specific (atom-aware, structured-cov, etc.)
 * handler registers first; the broadest atom-indep handler registers
 * last as the fallback.
 */
function registerKernelBroadcastFastPath(
  distOp: string, handler: KernelBroadcastFastPath,
): void {
  let list = REGISTRY.get(distOp);
  if (!list) { list = []; REGISTRY.set(distOp, list); }
  list.push(handler);
}

/**
 * Dispatch entry. Walks registered handlers for `ctx.d.distOp`,
 * invokes the first whose `match` returns true. Returns the handler's
 * execute promise, or `null` if no handler matched (caller falls
 * through to the general per-cell path).
 */
function tryKernelBroadcastFastPath(
  ctx: KernelBroadcastFastPathCtx,
): Promise<any> | null {
  const list = REGISTRY.get(ctx.d.distOp);
  if (!list || list.length === 0) return null;
  for (const h of list) {
    if (h.match(ctx)) return h.execute(ctx);
  }
  return null;
}

/** Test-only: list registered distOps with handler counts. Used by
 *  invariants tests to confirm built-in handlers loaded. */
function _testListKernelBroadcastFastPaths(): Array<{ distOp: string, count: number, labels: string[] }> {
  const out: Array<{ distOp: string, count: number, labels: string[] }> = [];
  REGISTRY.forEach((list, distOp) => {
    out.push({ distOp, count: list.length, labels: list.map((h) => h.label) });
  });
  return out;
}

// ---------------------------------------------------------------------
// Built-in handlers
// ---------------------------------------------------------------------
//
// Each built-in handler is registered eagerly at module load. Adding a
// new built-in handler: write the handler object, then
// `registerKernelBroadcastFastPath('<distOp>', handler)` below.
//
// Handlers lazy-require sibling modules (`sampler`, `value-ops`, `ops`)
// inside `execute` to break the require cycle (this file is required
// from `mat-broadcast.ts`, which is in turn required from the
// materialiser pipeline).

const shared       = require('./materialiser-shared.ts');
const valueLib     = require('./value.ts');
const { nameSeed, measureFromValue } = shared;

// ---------- Normal: atom-indep μ, σ → MvNormal(μ, diag(σ²)) ----------
//
// The closed-form hot path. Both μ and σ must be atom-independent
// (no per-atom references); each is a scalar or length-K vector.
// Generates K samples per atom in one worker `sampleN(Normal(0,1))`
// call, then applies the diag-Cholesky transform `L·z + μ` via the
// atom-aware op dispatcher.
//
// Output: vector-atom measure shape=[N, K] atom-major.

registerKernelBroadcastFastPath('Normal', {
  label: 'Normal: atom-indep μ, σ → diag-Cholesky MvNormal',
  match: (c) => !c.anyAtomDep
              && c.paramVals.mu !== undefined
              && c.paramVals.sigma !== undefined,
  execute: (c) => {
    const sampler  = require('./sampler.ts');
    const valueOps = require('./value-ops.ts');
    const ops      = require('./ops.ts');
    const { ctx, name, K, N, paramVals } = c;

    const muV  = valueLib.asValue(paramVals.mu);
    const sigV = valueLib.asValue(paramVals.sigma);
    const muLen  = muV.shape.length === 0 ? 1 : muV.shape[0];
    const sigLen = sigV.shape.length === 0 ? 1 : sigV.shape[0];
    const muVec  = new Float64Array(K);
    const sigSq  = new Float64Array(K);
    for (let j = 0; j < K; j++) {
      muVec[j] = muV.data[muLen === 1 ? 0 : j];
      const s = sigV.data[sigLen === 1 ? 0 : j];
      sigSq[j] = s * s;
    }
    const cov = valueLib.diagMatrix(sigSq);
    let L;
    try {
      L = sampler._internal.ARITH_OPS.lower_cholesky(cov);
    } catch (err) {
      return Promise.reject(new Error('broadcast(Normal): ' + (err as any).message));
    }
    const stdNormalIR = {
      kind: 'call', op: 'Normal',
      kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
    };
    return ctx.sendWorker({
      type: 'sampleN', ir: stdNormalIR, count: N, repeat: K,
      refArrays: {}, seed: nameSeed(name, ctx.rootKey),
    }).then((reply: any) => {
      const z = { shape: [N, K], data: reply.samples };
      // Atom-batched L·z + μ via the registry's atom-aware variants.
      // L is shape=[K, K] (Cholesky factor of diag(σ²) — diag-stored
      // on this hot path); z is atom-batched rank-1 shape=[N, K]; μ
      // is rank-1 shape=[K]. Diag-stored L stays on value-ops.mulN
      // (its null-fallthrough fast-path doesn't fit variant dispatch).
      const Lz = (valueLib.isDiagStored && valueLib.isDiagStored(L))
        ? valueOps.mulN(L, z, N)
        : ops.dispatch('mul', [L, z], { atomN: N });
      const result = ops.dispatch('add',
        [{ shape: [K], data: muVec }, Lz], { atomN: N });
      return measureFromValue(result, {
        logWeights: null, logTotalmass: 0, n_eff: N,
      });
    });
  },
});

// =====================================================================
// CJS facade
// =====================================================================

module.exports = {
  registerKernelBroadcastFastPath,
  tryKernelBroadcastFastPath,
  _testListKernelBroadcastFastPaths,
};

'use strict';

// FlatPPL sampler-worker message handler.
//
// The worker is a *stateless math kernel*: given a distribution IR or
// arithmetic IR plus arrays of upstream samples (one entry per Monte
// Carlo draw), it produces an output sample array. It owns no model
// state — the main thread holds the binding graph, the sample cache,
// and the RNG seeding policy. This separation is deliberate:
//
//   * The cache lives where the model lives. Main-thread cache is
//     resilient to worker restarts, can be shared across workers (if
//     we ever parallelise), and is naturally invalidated by source
//     edits.
//   * The worker becomes a small, testable surface — three primitive
//     ops cover everything the Plot panel needs.
//   * Future ports (Rust, Wasm) implement the same primitive set
//     without inheriting JS-specific session state.
//
// Why is this module factored as createWorkerHandler() rather than a
// top-level message listener?
//   * Tests can drive `handle(msg)` synchronously without a Worker
//     constructor or postMessage round-trip.
//   * The same handler can run in-process on the main thread as a
//     fallback if the worker bundle hasn't loaded yet.
//   * worker-entry.js is the thin shim that wires either a Web Worker
//     or Node worker_threads transport to the handler.
//
// Message protocol — request 'type' → reply 'type':
//
//   sample   { ir, env?, count }
//        → samples { samples: Float64Array }
//        Single-distribution sampling with a flat env. Used by tests
//        and the legacy direct-density path.
//
//   density  { ir, env?, opts? }
//        → density { xs, ys, support, reference }
//        Analytical PDF/PMF via stdlib.
//
//   evaluate { ir, env? }
//        → value { value }
//        Single-shot scalar eval — used by tests.
//
//   sampleN { distIR, count, refArrays?, seed? }
//        → samples { samples: Float64Array }
//        N-sample draw with per-i env: each row of refArrays supplies
//        the ref values for one Monte Carlo draw. The orchestrator's
//        sample-step IRs feed into here. `seed` (if provided) re-keys
//        Philox before drawing so per-binding seeding is deterministic.
//
//   evaluateN { ir, count, refArrays }
//        → samples { samples: Float64Array }
//        Element-wise deterministic compute. Used for binding RHSs
//        like 's = mu + 1' where mu's samples are passed as refArrays.mu.
//
//   logDensityN { ir, count, refArrays?, observed?, tally?, seed? }
//        → samples { samples: Float64Array, logWeights: null }
//        Per-i density evaluation via traceeval.walk. The same trace
//        walker that sampleN dispatches into for structural cases (joint,
//        iid, weighted, …) is invoked once per atom with `tally` mode
//        set ('all' for full joint log-density of a sampled trace,
//        'clamped' for likelihood / bayesupdate scoring). The reply's
//        `samples` array carries one log-density per outer atom — we
//        reuse the 'samples' reply shape so the main-thread cache and
//        zero-copy transfer logic don't need a new path.
//
//        `observed` (optional) is a JSON-encoded value-shape mirroring
//        the measure IR: a number for a leaf, a plain object keyed by
//        field name for joint/record, an array for iid. `null` /
//        `undefined` at any site means "not observed at this site"
//        (sample fresh; needed only when tally='all'). For uniform
//        observation across atoms (e.g. bayesupdate with a single obs
//        record), `observed` is shared across all i. For per-i
//        observed values, callers should pass them via refArrays and
//        reference them inside the IR.
//
//        Used by the orchestrator's bayesupdate / likelihoodof paths
//        to construct importance log-weights for posterior samples,
//        and by future diagnostics that want per-trace joint logp.
//
//   init { seed?, env? }                  → ready    (resets state — kept for compat)
//   setSeed { seed }                      → ok       (resets RNG only)
//   setEnv  { env, merge? }               → ok       (legacy session env)
//   dispose {}                            → (no reply)
//
// Every request may carry an `id`; the reply echoes it. Errors come
// back as `{ type: 'error', id, message, stack? }`.

const rngLib = require('./rng');
const samplerLib = require('./sampler');
const densityLib = require('./density');

function createWorkerHandler(opts = {}) {
  // Session RNG state: used by the legacy `sample` / chain primitives
  // when no explicit seed is passed. The new sampleN path takes a per-
  // call seed and doesn't touch this.
  let philox = rngLib.stateFromKey(opts.seed ?? 0);
  let env = { ...(opts.env ?? {}) };

  function handle(msg) {
    const id = msg.id;
    try {
      switch (msg.type) {
        case 'init': {
          philox = rngLib.stateFromKey(msg.seed ?? 0);
          env = { ...(msg.env ?? {}) };
          return { type: 'ready', id };
        }
        case 'setEnv': {
          env = msg.merge === false ? { ...(msg.env ?? {}) } : { ...env, ...(msg.env ?? {}) };
          return { type: 'ok', id };
        }
        case 'setSeed': {
          philox = rngLib.stateFromKey(msg.seed ?? 0);
          return { type: 'ok', id };
        }
        case 'sample': {
          const callEnv = msg.env ? { ...msg.env, ...env } : env;
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`sample.count must be positive integer (got ${msg.count})`);
          const out = new Float64Array(count);
          for (let i = 0; i < count; i++) {
            const [v, next] = samplerLib.rand(philox, msg.ir, callEnv);
            philox = next;
            out[i] = v;
          }
          return { type: 'samples', id, samples: out };
        }
        case 'density': {
          const callEnv = msg.env ? { ...msg.env, ...env } : env;
          const d = samplerLib.density(msg.ir, callEnv, msg.opts ?? {});
          return { type: 'density', id, ...d };
        }
        case 'evaluate': {
          const callEnv = msg.env ? { ...msg.env, ...env } : env;
          const value = samplerLib.evaluateExpr(msg.ir, callEnv);
          return { type: 'value', id, value };
        }
        case 'sampleN': {
          // Per-binding sampling primitive. distIR may have refs in
          // its kwargs; refArrays maps each ref-name to a pre-computed
          // Float64Array of length `count` holding the upstream's
          // samples. For draw i, env[k] = refArrays[k][i].
          //
          // `seed` re-keys Philox locally for this call, so the same
          // (distIR, refArrays, seed) always produce the same output.
          // Per-binding seeding is the main thread's job — this just
          // honours whatever seed it sends.
          //
          // `repeat: k` (optional, default 1) draws k iid samples per
          // outer atom — the iid(M, k) sampling primitive. Output is
          // length `count * repeat`, atom-major (atom i's slot is
          // [i*repeat, (i+1)*repeat)). When repeat===1 the layout
          // collapses to today's flat samples array.
          //
          // PERF: there are two paths.
          //   * static-params (refArrays empty) — kwargs resolve to
          //     literals so the stdlib factory is constant. Build it
          //     once via samplerLib.makeSampler() and call its
          //     `draw()` count*repeat times. This avoids ~N factory
          //     allocations and brings 1M Normal draws from ~10s down
          //     to ~tens of ms; the factory build is by far the
          //     dominant cost.
          //   * per-i-params (refArrays non-empty) — at least one
          //     kwarg references an upstream sample, so params change
          //     per outer atom. We build ONE parametric sampler for
          //     the whole call (factory closure with prng bound but
          //     params unbound) and call .drawWith(env) per atom,
          //     resolving the per-i env into stdlib's params on each
          //     draw. This is generic across every distribution in
          //     the registry — the stdlib `factory(opts)` form
          //     returns a closure that accepts params per call, so
          //     the expensive factory setup runs once instead of N
          //     times. For repeat>1, atom i's k inner draws share
          //     atom i's env so we just call drawWith k times in
          //     succession with the same env.
          const count  = msg.count  | 0;
          const repeat = (msg.repeat | 0) || 1;
          if (count  <= 0) throw new Error(`sampleN.count must be positive integer (got ${msg.count})`);
          if (repeat <= 0) throw new Error(`sampleN.repeat must be positive integer (got ${msg.repeat})`);
          const refArrays = msg.refArrays || {};
          const refKeys = Object.keys(refArrays);
          let state = msg.seed != null ? rngLib.stateFromKey(msg.seed) : philox;
          const total = count * repeat;
          const out = new Float64Array(total);

          if (refKeys.length === 0) {
            // Static-params fast path: one sampler instance for the
            // whole call, regardless of repeat. Pass session env so
            // measures whose params reference fixed-phase bindings
            // (e.g. `Normal(mean(random_data), 1)`) resolve at
            // factory-build time rather than failing as unbound.
            const s = samplerLib.makeSampler(state, msg.ir, env);
            for (let i = 0; i < total; i++) out[i] = s.draw();
            state = s.getState();
          } else {
            // Per-i-params path. One parametric sampler for the whole
            // call; params resolved per draw via drawWith(env).
            // drawEnv merges session env (fixed-phase bindings) with
            // the per-atom refArrays slice — same precedence as
            // evaluateN above.
            const s = samplerLib.makeParametricSampler(state, msg.ir);
            const drawEnv = { ...env };
            if (repeat === 1) {
              for (let i = 0; i < count; i++) {
                for (const k of refKeys) drawEnv[k] = refArrays[k][i];
                out[i] = s.drawWith(drawEnv);
              }
            } else {
              for (let i = 0; i < count; i++) {
                for (const k of refKeys) drawEnv[k] = refArrays[k][i];
                const base = i * repeat;
                for (let j = 0; j < repeat; j++) out[base + j] = s.drawWith(drawEnv);
              }
            }
            state = s.getState();
          }

          // Only update session RNG if no explicit seed was given. Per-
          // binding seeded calls leave the session state alone so the
          // calls are independent of arrival order.
          if (msg.seed == null) philox = state;
          // EmpiricalMeasure shape: samples + logWeights. Variates and
          // independent draws come back unweighted (null = uniform 1/N);
          // weighted operations attach explicit per-atom weights later.
          return { type: 'samples', id, samples: out, logWeights: null };
        }
        case 'evaluateN': {
          // Element-wise deterministic compute. Thin wrapper around
          // sampler.evaluateExprN — the single batched-evaluation
          // primitive. Scalar-arith ops dispatch through ARITH_OPS_N
          // (one Float64Array result for the whole batch); non-scalar
          // ops fall back to per-atom dispatch internally.
          //
          // Env precedence at refs: refArrays (per-atom) > baseEnv
          // (session). Same layering as the old hand-coded loop —
          // expressions like `mean(random_data)` resolve `random_data`
          // through baseEnv when refArrays doesn't carry it.
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`evaluateN.count must be positive integer (got ${msg.count})`);
          const refArrays = msg.refArrays || null;
          const result = samplerLib.evaluateExprN(msg.ir, refArrays, count, env);
          let out;
          if (result && result.BYTES_PER_ELEMENT !== undefined
              && result.length === count) {
            out = result;  // Float64Array(count) — happy path
          } else if (typeof result === 'number' || typeof result === 'boolean') {
            // Atom-independent scalar — broadcast to the batch.
            out = new Float64Array(count);
            out.fill(+result);
          } else {
            throw new Error('evaluateN: expression produced non-scalar per-atom '
              + 'result (got ' + (typeof result) + '); only scalar exprs are '
              + 'supported on this path');
          }
          // Deterministic transforms preserve their parents' weights.
          // The main thread is responsible for plumbing the parent's
          // logWeights through; the worker just emits null here and
          // lets that wrap-up happen at the cache boundary.
          return { type: 'samples', id, samples: out, logWeights: null };
        }
        case 'logDensityN': {
          // Batched density evaluation — thin shell around
          // density.logDensityBatch (the single density implementation
          // for the whole engine; see density.js for the consume/rest
          // primitive and batch loop). The worker hosts this so the
          // main-thread cache and postMessage protocol stay stable
          // even though density is pure: future parallelisation, or
          // running density across multiple workers, only changes
          // this handler.
          //
          // `observed` is shared across atoms (typical bayesupdate /
          // likelihoodof case: one obs, N prior atoms). Per-atom
          // variation comes from `refArrays` — the prior's per-atom
          // value-position refs.
          //
          // Reply uses the 'samples' shape so the main-thread cache
          // and zero-copy transfer logic don't need a new path; the
          // numbers are log-densities, not draws.
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`logDensityN.count must be positive integer (got ${msg.count})`);
          const logps = densityLib.logDensityN(
            msg.ir,
            msg.observed,
            msg.refArrays || null,
            count,
            { baseEnv: env });
          return { type: 'samples', id, samples: logps, logWeights: null };
        }
        case 'truncateSampleN': {
          // Truncated-distribution sampling primitive for matTruncate
          // (spec §06). Two modes:
          //
          //   'cdf'       — inverse-CDF sampling against an interval set.
          //                 Draws u ~ U(0,1), maps to F^{-1}(F(lo) + u·(F(hi)−F(lo))).
          //                 Exact, uniform-weight output. Requires the
          //                 measure IR to be a known stdlib distribution
          //                 with finite-or-resolved bounds; no NaN slots.
          //                 logShift = log(F(hi) − F(lo)) — caller adds
          //                 this to the parent's logTotalmass.
          //
          //   'rejection' — per-atom rejection-redraw, configurable per-
          //                 atom budget. Draws from the distribution
          //                 until the value falls inside `setDescr`, or
          //                 the budget runs out; budget-exhausted atoms
          //                 become NaN. Generic over any sampleable IR
          //                 + any set descriptor with a numeric-bounds
          //                 surface. logShift = log(n_eff / totalDraws)
          //                 — empirical acceptance probability.
          //
          // setDescr is the structural set descriptor from orchestrator's
          // parseSetIR (`{ kind: 'interval', lo, hi }` etc.). Anything
          // we can't map to numeric (lo, hi) bounds becomes a worker
          // error — the materialiser handles fallback decisions.
          const count   = msg.count   | 0;
          const seed    = msg.seed;
          const setDescr = msg.setDescr || null;
          const mode    = msg.mode    || 'rejection';
          if (count <= 0) throw new Error(`truncateSampleN.count must be positive integer (got ${msg.count})`);
          if (!setDescr) throw new Error('truncateSampleN.setDescr is required');
          const [lo, hi] = setBoundsFor(setDescr);
          if (!(hi >= lo)) {
            throw new Error('truncateSampleN: degenerate set bounds [' + lo + ', ' + hi + ']');
          }
          let state = seed != null ? rngLib.stateFromKey(seed) : philox;
          const out = new Float64Array(count);

          if (mode === 'cdf') {
            // Inverse-CDF path. Build the analytical distribution once
            // (static params required — refArrays not supported here),
            // pre-compute F(lo) and F(hi), then per-atom Q(F(lo) + u·Δ).
            const callEnv = msg.env ? { ...env, ...msg.env } : env;
            const dist = samplerLib.makeAnalytical(msg.ir, callEnv);
            const Flo = isFinite(lo) ? dist.cdf(lo) : 0;
            const Fhi = isFinite(hi) ? dist.cdf(hi) : 1;
            const dF = Fhi - Flo;
            if (!(dF > 0)) {
              // Empty intersection M ∩ S: every atom NaN, mass shift = -Inf.
              for (let i = 0; i < count; i++) out[i] = NaN;
              if (seed == null) philox = state;
              return { type: 'samples', id, samples: out, logWeights: null,
                       logShift: -Infinity, n_eff: 0 };
            }
            for (let i = 0; i < count; i++) {
              const pair = rngLib.nextUniform(state);
              state = pair[1];
              out[i] = dist.quantile(Flo + pair[0] * dF);
            }
            if (seed == null) philox = state;
            return { type: 'samples', id, samples: out, logWeights: null,
                     logShift: Math.log(dF), n_eff: count };
          }

          // Rejection-redraw path. Build one sampler (static or
          // per-i parametric) and loop per atom, redrawing until the
          // value lands in [lo, hi] or the per-atom budget is spent.
          const budget = Math.max(1, msg.budget | 0);
          const refArrays = msg.refArrays || {};
          const refKeys = Object.keys(refArrays);
          let totalDraws = 0;
          let n_eff = 0;
          if (refKeys.length === 0) {
            const s = samplerLib.makeSampler(state, msg.ir, env);
            for (let i = 0; i < count; i++) {
              let v = NaN;
              for (let t = 0; t < budget; t++) {
                const draw = s.draw();
                totalDraws++;
                if (draw >= lo && draw <= hi) { v = draw; n_eff++; break; }
              }
              out[i] = v;
            }
            state = s.getState();
          } else {
            const s = samplerLib.makeParametricSampler(state, msg.ir);
            const drawEnv = { ...env };
            for (let i = 0; i < count; i++) {
              for (const k of refKeys) drawEnv[k] = refArrays[k][i];
              let v = NaN;
              for (let t = 0; t < budget; t++) {
                const draw = s.drawWith(drawEnv);
                totalDraws++;
                if (draw >= lo && draw <= hi) { v = draw; n_eff++; break; }
              }
              out[i] = v;
            }
            state = s.getState();
          }
          if (seed == null) philox = state;
          // Empirical acceptance probability → log-mass shift. When no
          // atom accepted (n_eff == 0), report -Infinity so downstream
          // sees zero truncated mass cleanly.
          const logShift = n_eff > 0
            ? Math.log(n_eff / totalDraws)
            : -Infinity;
          return { type: 'samples', id, samples: out, logWeights: null,
                   logShift, n_eff };
        }
        case 'profileN': {
          // Profile-plot evaluator. Sweeps a single scalar input over
          // a [lo, hi] range at N evenly-spaced points, with all other
          // inputs held at their fixed values. Two evaluation modes:
          //
          //   'function'   — out[i] = evaluateExpr(ir, env_i)
          //                  for fn / functionof bindings. ir is the
          //                  reified body. env_i = fixedEnv with the
          //                  swept axis name set to the i-th sample.
          //   'logdensity' — out[i] = density.logDensity(ir, observed,
          //                  env_i) for kernelof / likelihoodof bindings.
          //                  Density routes through density.js (the
          //                  single density implementation); we pass
          //                  the swept axis as a length-N refArray and
          //                  the surrounding fixedEnv as baseEnv, then
          //                  scatter the result through a per-atom
          //                  try/catch so domain-of-definition errors
          //                  become NaN gaps instead of aborting.
          //
          // %local refs in reified bodies look up via the same env as
          // 'self' refs (sampler.evaluateExpr handles both namespaces
          // uniformly), so populating env keyed by param name works
          // for both reified-function and measure paths.
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`profileN.count must be positive integer (got ${msg.count})`);
          const range = msg.range || [0, 1];
          const lo = +range[0], hi = +range[1];
          if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
            throw new Error(`profileN.range must be finite numbers, got [${range[0]}, ${range[1]}]`);
          }
          const sweepName = msg.sweepName;
          if (!sweepName) throw new Error('profileN.sweepName is required');
          const fixedEnv = msg.fixedEnv || {};
          const mode     = msg.mode     || 'function';
          const out      = new Float64Array(count);
          // Layer the message's per-call fixedEnv over the session env
          // (fixed-phase module bindings pushed via setEnv). Per-call
          // values win, so a sweep can locally override a module
          // binding without mutating session state.
          const baseEnv = Object.assign({}, env, fixedEnv);
          if (mode === 'function') {
            const evalEnv = Object.assign({}, baseEnv);
            for (let i = 0; i < count; i++) {
              const t = count === 1 ? 0 : i / (count - 1);
              evalEnv[sweepName] = lo + t * (hi - lo);
              try {
                out[i] = samplerLib.evaluateExpr(msg.ir, evalEnv);
              } catch (_) {
                out[i] = NaN;
              }
            }
          } else if (mode === 'logdensity') {
            // Build a per-atom refArray for the swept axis and call
            // density.logDensityN once for the whole sweep. Per-atom
            // domain errors become NaN: density.logDensityN throws on
            // the whole batch if any atom hits an unrecoverable error,
            // so we fall back to per-atom calls when the batch throws.
            const sweepArr = new Float64Array(count);
            for (let i = 0; i < count; i++) {
              const t = count === 1 ? 0 : i / (count - 1);
              sweepArr[i] = lo + t * (hi - lo);
            }
            const refArrays = { [sweepName]: sweepArr };
            try {
              const logps = densityLib.logDensityN(
                msg.ir, msg.observed, refArrays, count, { baseEnv });
              out.set(logps);
            } catch (_) {
              // Per-atom fallback for NaN-gap behaviour.
              const callEnv = Object.assign({}, baseEnv);
              for (let i = 0; i < count; i++) {
                callEnv[sweepName] = sweepArr[i];
                try {
                  out[i] = densityLib.logDensity(msg.ir, msg.observed, callEnv);
                } catch (_) {
                  out[i] = NaN;
                }
              }
            }
          } else {
            throw new Error(`profileN.mode must be 'function' or 'logdensity' (got '${mode}')`);
          }
          return { type: 'samples', id, samples: out, logWeights: null };
        }
        case 'dispose': {
          philox = null;
          env = null;
          return null;
        }
        default:
          throw new Error(`unknown message type: ${msg.type}`);
      }
    } catch (err) {
      return {
        type: 'error',
        id,
        message: err && err.message ? err.message : String(err),
        stack: err && err.stack ? err.stack : undefined,
      };
    }
  }

  // Test-only inspector. Not part of the production surface.
  function _inspect() {
    return { philox, env };
  }

  return { handle, _inspect };
}

// Map a structural set descriptor (orchestrator.parseSetIR shape) to
// numeric [lo, hi] bounds suitable for interval membership tests and
// CDF clipping. ±Infinity is allowed for half-open / unbounded sets.
// Throws on shapes that don't reduce to an interval surface (e.g.
// integer / boolean sets — those defer to a discrete-aware path).
function setBoundsFor(setDescr) {
  if (!setDescr || typeof setDescr !== 'object') {
    throw new Error('setBoundsFor: missing set descriptor');
  }
  switch (setDescr.kind) {
    case 'interval':    return [+setDescr.lo, +setDescr.hi];
    case 'reals':       return [-Infinity, Infinity];
    case 'posreals':    return [0, Infinity];
    case 'nonnegreals': return [0, Infinity];
    default:
      throw new Error('setBoundsFor: unsupported set kind \'' + setDescr.kind + '\'');
  }
}

// Helper: collect transferable buffers in a reply. The browser shim
// uses this to populate postMessage's transferList so large sample
// arrays move zero-copy across the worker boundary.
function transferablesOf(reply) {
  if (!reply) return [];
  if (reply.type === 'samples') {
    const out = [];
    if (reply.samples    instanceof Float64Array) out.push(reply.samples.buffer);
    // logWeights is null for unweighted measures (which is everything
    // until the weighted-ops land). When it becomes a typed array,
    // ship its buffer too so weighted draws stay zero-copy.
    if (reply.logWeights instanceof Float64Array) out.push(reply.logWeights.buffer);
    return out;
  }
  if (reply.type === 'density') {
    const out = [];
    if (reply.xs instanceof Float64Array) out.push(reply.xs.buffer);
    if (reply.ys instanceof Float64Array) out.push(reply.ys.buffer);
    return out;
  }
  return [];
}

module.exports = {
  createWorkerHandler,
  transferablesOf,
};

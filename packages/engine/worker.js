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
const traceevalLib = require('./traceeval');

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
          // Element-wise deterministic compute. ir typically a (call op …)
          // node from the orchestrator; refArrays is the per-i env.
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`evaluateN.count must be positive integer (got ${msg.count})`);
          const refArrays = msg.refArrays || {};
          const out = new Float64Array(count);
          // callEnv merges three layers in priority order:
          //   1. session env (fixed-phase bindings the orchestrator
          //      pre-evaluated and pushed via setEnv — full values,
          //      not per-atom)
          //   2. refArrays per-atom slice (stochastic deps)
          // Per-atom keys take precedence so a stochastic ref of the
          // same name as a fixed binding (shouldn't happen in practice)
          // wouldn't collide silently. The session-env layer lets
          // expressions like `mean(random_data)` resolve `random_data`
          // to its full fixed array rather than the per-i undefined
          // slice that refArrays would produce.
          const callEnv = { ...env };
          for (let i = 0; i < count; i++) {
            for (const k in refArrays) callEnv[k] = refArrays[k][i];
            out[i] = samplerLib.evaluateExpr(msg.ir, callEnv);
          }
          // Deterministic transforms preserve their parents' weights.
          // The main thread is responsible for plumbing the parent's
          // logWeights through; the worker just emits null here and
          // lets that wrap-up happen at the cache boundary.
          return { type: 'samples', id, samples: out, logWeights: null };
        }
        case 'logDensityN': {
          // Per-i density evaluation via traceeval.walk. The walker is
          // a single recursion that handles leaf distributions,
          // joint/record, iid, weighted, logweighted — same code as
          // sampling, just with `tally` set so log-densities accumulate.
          //
          // Two tally modes are useful here:
          //   'clamped' — only score observed leaves (likelihoodof,
          //               bayesupdate). Latents either don't appear in
          //               the IR (already substituted) or are
          //               re-sampled freely; their logp doesn't enter
          //               the accumulator.
          //   'all'     — joint log-density of the whole sampled
          //               trace, useful for diagnostics.
          //
          // For 'clamped' against a single observation shared across
          // atoms (the typical bayesupdate case: one obs, N priors),
          // pass `observed` once and we reuse it for every i. For
          // per-i observations, encode them as refArrays and have the
          // measure IR reference them.
          //
          // Reply uses the 'samples' shape so the main-thread cache
          // and zero-copy transfer logic don't need a new path; the
          // numbers are log-densities, not draws.
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`logDensityN.count must be positive integer (got ${msg.count})`);
          const refArrays = msg.refArrays || {};
          const observed = msg.observed; // may be undefined; null means same
          const tally = msg.tally || 'clamped';
          const out = new Float64Array(count);
          // Layer session env (fixed-phase bindings pushed via setEnv)
          // under the per-atom refArrays. Same precedence as evaluateN.
          const callEnv = { ...env };
          // Walker needs a state even when no sampling occurs; reuse
          // session philox (or a per-call seed if supplied) for any
          // free latents the caller leaves unobserved.
          let state = msg.seed != null ? rngLib.stateFromKey(msg.seed) : philox;
          for (let i = 0; i < count; i++) {
            for (const k in refArrays) callEnv[k] = refArrays[k][i];
            const r = traceevalLib.walk(state, msg.ir, callEnv, observed, { tally });
            out[i] = r.logp;
            state = r.state;
          }
          if (msg.seed == null) philox = state;
          return { type: 'samples', id, samples: out, logWeights: null };
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
          //   'logdensity' — out[i] = traceeval.walk(...).logp
          //                  for kernelof / likelihoodof bindings. ir
          //                  is the expanded measure IR; observed is
          //                  the obs value (constant across i for
          //                  likelihoods).
          //
          // Domain-of-definition errors (log of negative, division
          // by zero, etc.) become NaN entries — the plot pane shows
          // them as gaps rather than aborting the whole sweep.
          //
          // Note: %local refs in reified bodies look up via the same
          // env as 'self' refs (sampler.evaluateExpr handles both
          // namespaces uniformly), so we can populate env keyed by
          // param name and the body's lookups Just Work.
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
          const observed = msg.observed;
          const tally    = msg.tally || 'clamped';
          const out      = new Float64Array(count);
          // Layer the message's per-call fixedEnv over the session env
          // (fixed-phase module bindings pushed via setEnv). Per-call
          // values win, so a sweep can locally override a module
          // binding without mutating session state.
          const evalEnv  = Object.assign({}, env, fixedEnv);
          let state = msg.seed != null ? rngLib.stateFromKey(msg.seed) : philox;
          for (let i = 0; i < count; i++) {
            const t = count === 1 ? 0 : i / (count - 1);
            evalEnv[sweepName] = lo + t * (hi - lo);
            try {
              if (mode === 'function') {
                out[i] = samplerLib.evaluateExpr(msg.ir, evalEnv);
              } else if (mode === 'logdensity') {
                const r = traceevalLib.walk(state, msg.ir, evalEnv, observed, { tally });
                out[i] = r.logp;
                state = r.state;
              } else {
                throw new Error(`profileN.mode must be 'function' or 'logdensity' (got '${mode}')`);
              }
            } catch (_) {
              out[i] = NaN;
            }
          }
          if (msg.seed == null) philox = state;
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

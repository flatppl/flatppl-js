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
//   drawN { distIR, count, refArrays?, seed? }
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
//   init { seed?, env? }                  → ready    (resets state — kept for compat)
//   setSeed { seed }                      → ok       (resets RNG only)
//   setEnv  { env, merge? }               → ok       (legacy session env)
//   dispose {}                            → (no reply)
//
// Every request may carry an `id`; the reply echoes it. Errors come
// back as `{ type: 'error', id, message, stack? }`.

const rngLib = require('./rng');
const samplerLib = require('./sampler');

function createWorkerHandler(opts = {}) {
  // Session RNG state: used by the legacy `sample` / chain primitives
  // when no explicit seed is passed. The new drawN path takes a per-
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
        case 'drawN': {
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
          // PERF: there are two paths.
          //   * static-params (refArrays empty) — kwargs resolve to
          //     literals so the stdlib factory is constant. Build it
          //     once via samplerLib.makeSampler() and call its
          //     `draw()` N times. This avoids ~N factory allocations
          //     and brings 1M Normal draws from ~10s down to ~tens of
          //     ms; the factory build is by far the dominant cost.
          //   * per-i-params (refArrays non-empty) — at least one
          //     kwarg references an upstream sample, so params change
          //     per draw and we have to call rand() per i. Slower,
          //     but unavoidable without batched-parameter stdlib APIs
          //     (which random-array-* offers but at the cost of
          //     additional dependencies).
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`drawN.count must be positive integer (got ${msg.count})`);
          const refArrays = msg.refArrays || {};
          const refKeys = Object.keys(refArrays);
          let state = msg.seed != null ? rngLib.stateFromKey(msg.seed) : philox;
          const out = new Float64Array(count);

          if (refKeys.length === 0) {
            // Static-params fast path.
            const s = samplerLib.makeSampler(state, msg.ir, {});
            for (let i = 0; i < count; i++) out[i] = s.draw();
            state = s.getState();
          } else {
            // Per-i-params path.
            const drawEnv = {};
            for (let i = 0; i < count; i++) {
              for (const k of refKeys) drawEnv[k] = refArrays[k][i];
              const [v, next] = samplerLib.rand(state, msg.ir, drawEnv);
              state = next;
              out[i] = v;
            }
          }

          // Only update session RNG if no explicit seed was given. Per-
          // binding seeded calls leave the session state alone so the
          // calls are independent of arrival order.
          if (msg.seed == null) philox = state;
          return { type: 'samples', id, samples: out };
        }
        case 'evaluateN': {
          // Element-wise deterministic compute. ir typically a (call op …)
          // node from the orchestrator; refArrays is the per-i env.
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`evaluateN.count must be positive integer (got ${msg.count})`);
          const refArrays = msg.refArrays || {};
          const out = new Float64Array(count);
          const callEnv = {};
          for (let i = 0; i < count; i++) {
            for (const k in refArrays) callEnv[k] = refArrays[k][i];
            out[i] = samplerLib.evaluateExpr(msg.ir, callEnv);
          }
          return { type: 'samples', id, samples: out };
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
  if (reply.type === 'samples' && reply.samples instanceof Float64Array) {
    return [reply.samples.buffer];
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

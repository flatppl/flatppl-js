'use strict';

// FlatPPL sampler-worker message handler.
//
// This module is the *transport-agnostic* heart of the sampling worker. It
// owns a small piece of mutable state (RNG, env) and exposes a `handle(msg)`
// function that takes a request object and returns a reply object. Wiring
// it up to a Web Worker, Node worker_threads, or an in-process call is the
// caller's job — see `worker-entry.js` for the browser/Node shim.
//
// Why factor it this way?
//   * Tests can drive the handler with plain object literals — no Worker
//     constructor, no postMessage round-trips, no SharedArrayBuffer setup.
//   * The same logic can run synchronously on the main thread when a
//     worker isn't available (e.g. the UI's first render before the worker
//     bundle has finished loading).
//   * Future Rust port can implement the same protocol without inheriting
//     any JS-specific transport details.
//
// State held by a handler instance:
//   - philox  : pure-functional RNG state (rng.js); threaded through every
//               draw so the worker is fully reproducible from its seed.
//   - env     : { name → number } map of values bound on the main thread
//               (parents already sampled, fixed inputs, etc.). The IR
//               coming in has free identifiers that look up here.
//
// Message protocol (request `type` → reply `type`):
//   init     { seed?, env? }              → ready
//   setEnv   { env, merge? }              → ok
//   setSeed  { seed }                     → ok
//   sample   { ir, env?, count }          → samples  { samples: Float64Array }
//   density  { ir, env?, opts? }          → density  { xs, ys, support, reference }
//   evaluate { ir, env? }                 → value    { value }
//   dispose  {}                           → (no reply; worker terminates)
//
// Every request may carry an `id` field; the reply echoes it. Errors come
// back as `{ type: 'error', id, message, stack? }`. Sample arrays are
// returned as Float64Array so the entry shim can transferList them.

const rngLib = require('./rng');
const samplerLib = require('./sampler');

function createWorkerHandler(opts = {}) {
  // Default seed: 0n. Callers should always send `init { seed }` before the
  // first `sample`. Using 0 as the "uninitialized" seed makes accidental
  // missed-init runs deterministic instead of silently random.
  let philox = rngLib.stateFromKey(opts.seed ?? 0);
  let env = { ...(opts.env ?? {}) };

  function handle(msg) {
    const id = msg.id;
    try {
      switch (msg.type) {
        case 'init': {
          // Reset both RNG and env. `init` is the canonical "fresh start"
          // so it always replaces both — `setEnv` and `setSeed` exist for
          // partial updates without a full reset.
          philox = rngLib.stateFromKey(msg.seed ?? 0);
          env = { ...(msg.env ?? {}) };
          return { type: 'ready', id };
        }
        case 'setEnv': {
          // merge=true keeps existing entries (default), merge=false replaces.
          env = msg.merge === false ? { ...(msg.env ?? {}) } : { ...env, ...(msg.env ?? {}) };
          return { type: 'ok', id };
        }
        case 'setSeed': {
          philox = rngLib.stateFromKey(msg.seed ?? 0);
          return { type: 'ok', id };
        }
        case 'sample': {
          // Per-request env overlay: caller can pass extra bindings without
          // clobbering the worker-level env. Worker env wins on conflict so
          // expensive fixed setups don't get accidentally overwritten by a
          // stray per-request key with the same name.
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
        case 'dispose': {
          // Caller (entry shim) is expected to close the worker after this.
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

  // Test-only inspector. Exposed so tests can assert on RNG advancement
  // without sampling. Don't rely on this from production code.
  function _inspect() {
    return { philox, env };
  }

  return { handle, _inspect };
}

// Helper: collect the transferable buffers in a reply. The browser shim
// uses this to populate the `transferList` argument of postMessage so
// large sample arrays move zero-copy across the worker boundary.
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

// @ts-nocheck — test support only; never bundled (index.ts does not import it).
//
// The engine's sampler-worker handler run in-process behind the Worker
// interface, with structured cloning at the boundary as in a browser.

import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { createWorkerHandler } = require('../../engine/worker.ts');

export class InProcessWorker {
  /** Message types posted to any instance, for tests that check which path ran. */
  static posted = [];
  constructor() {
    this.handler = createWorkerHandler();
    this.listeners = [];
    this.terminated = false;
    this.handler.setProgressSink?.((m) => this.emit(m));
  }
  addEventListener(type, fn) { if (type === 'message') this.listeners.push(fn); }
  removeEventListener() {}
  emit(reply) {
    if (this.terminated) return;
    const data = structuredClone(reply);
    this.listeners.forEach((fn) => fn({ data }));
  }
  postMessage(msg) {
    InProcessWorker.posted.push(msg && msg.type);
    const copy = structuredClone(msg);
    setImmediate(() => Promise.resolve(this.handler.handle(copy)).then((r) => { if (r) this.emit(r); }));
  }
  terminate() { this.terminated = true; }
}

/** Let queued worker replies and promise chains run. */
export async function settle(turns = 200) {
  for (let i = 0; i < turns; i++) await new Promise((resolve) => setImmediate(resolve));
}

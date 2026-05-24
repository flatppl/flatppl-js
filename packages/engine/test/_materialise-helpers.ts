'use strict';

// Shared helpers for integration-style tests that drive a FlatPPL
// source end-to-end through the engine and assert properties of the
// resulting measures.
//
// Why this exists
// ===============
// The earlier test sweep covered each pipeline layer (parser, lowering,
// analyzer, typeinfer, density math) in isolation but left the
// **materialiser ↔ viewer contract** untested for unusual fixed-phase
// shapes. A bug shipped where the viewer's array-mode renderer read
// `measure.samples` (undefined) and threw "Cannot read properties of
// undefined" for `A, _ = rand(rstate, iid(Normal(0,1), [3, 3]))`.
//
// These helpers close the gap: every binding-shape test that goes
// through here also drives materialiseMeasure(name), so a "viewer
// can plot this" assertion is one line of test code.
//
// The contract: for any binding the viewer might be asked to plot,
// `materialiseMeasure` must succeed and the returned measure must
// have a shape the viewer recognises (.samples / .value / .fields /
// .elems consistent with the binding's inferred type / phase).

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

interface MatCtx {
  derivations: any;
  bindings:    any;
  fixedValues: any;
  sampleCount: number;
  rootSeed:    number;
  getMeasure:  (name: string) => Promise<any>;
  sendWorker:  (msg: any) => Promise<any>;
}

/**
 * Build a fully wired context against a FlatPPL source. Returns
 * { ctx, lifted, built } so tests can also inspect bindings /
 * derivations / fixedValues directly.
 */
function makeMatCtx(source: string, opts?: { sampleCount?: number; rootSeed?: number }): {
  ctx: MatCtx; lifted: any; built: any;
} {
  const sampleCount = opts?.sampleCount ?? 1024;
  const rootSeed = opts?.rootSeed ?? 0xC0FFEEEE;
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  if (errs.length > 0) {
    throw new Error('makeMatCtx: source has parse/analyze errors: '
      + errs.map((d: any) => d.message).join('; '));
  }
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: rootSeed });
  const cache = new Map();
  const ctx: MatCtx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount,
    rootSeed,
    getMeasure(name: string) {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker(msg: any) {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') {
        return Promise.reject(new Error(reply.message));
      }
      return Promise.resolve(reply);
    },
  };
  return { ctx, lifted, built };
}

/**
 * Materialise a single binding and assert it has a shape the viewer
 * could plot — at minimum one of .samples / .value / .fields / .elems
 * with internally consistent length / shape.
 *
 * Returns the measure so callers can probe specific fields.
 */
async function expectPlottable(ctx: MatCtx, name: string): Promise<any> {
  const m = await ctx.getMeasure(name);
  if (!m) {
    throw new Error(`expectPlottable: ${name} produced no measure`);
  }
  const hasSamples = m.samples != null;
  const hasValue   = m.value != null;
  const hasFields  = m.fields != null;
  const hasElems   = Array.isArray(m.elems) && m.elems.length > 0;
  if (!hasSamples && !hasValue && !hasFields && !hasElems) {
    throw new Error(`expectPlottable: ${name} measure has none of `
      + `.samples/.value/.fields/.elems (keys: ${Object.keys(m).join(', ')})`);
  }
  // If `.samples` is exposed it MUST be a real typed array — that's
  // what the viewer's renderArrayStepPlot / renderEmpiricalMeasure
  // read. Catches the bug class this helper was built for.
  if (hasSamples) {
    if (typeof m.samples.length !== 'number') {
      throw new Error(`expectPlottable: ${name}.samples lacks .length`);
    }
  }
  // If `.value` is exposed it MUST have shape + data with prod(shape)
  // === data.length. Without this the viewer's shape-explicit paths
  // mis-stride.
  if (hasValue) {
    const v = m.value;
    if (!Array.isArray(v.shape)) {
      throw new Error(`expectPlottable: ${name}.value lacks .shape array`);
    }
    if (!v.data || typeof v.data.length !== 'number') {
      throw new Error(`expectPlottable: ${name}.value lacks .data with length`);
    }
    const prod = v.shape.reduce((a: number, b: number) => a * b, 1);
    if (prod !== v.data.length) {
      throw new Error(`expectPlottable: ${name}.value shape=[${v.shape}] `
        + `but data.length=${v.data.length} (expected ${prod})`);
    }
  }
  return m;
}

module.exports = { makeMatCtx, expectPlottable };

'use strict';

const regression = require('./regression.ts');
const shared = require('./materialiser-shared.ts');
const V = require('./value.ts');
const sampler = require('./sampler.ts');

/** Batched vector draws share the parameter feed and its importance weights. */
async function materialise(name: string, d: any, ctx: any): Promise<any> {
  const parents: any[] = [];
  const refs = await shared.collectRefArrays(d.distIR, ctx, parents);
  const env: Record<string, any> = Object.fromEntries(ctx.fixedValues || []);
  const names = Object.keys(refs);
  const ir = d.distIR;
  const paramsIR = regression.bind(ir.op, ir.args, ir.kwargs);
  const prng = shared.makeMainThreadPrng(shared.nameSeed(name, ctx.rootKey));
  const factories: Record<string, any> = {};
  const draw = (kind: string, ...args: number[]) => {
    factories[kind] ||= sampler._internal.REGISTRY[kind].randFn.factory({ prng });
    return factories[kind](...args);
  };
  let kernel: any = null, data: Float64Array | null = null, rows = 0;
  for (let i = 0; i < ctx.sampleCount; i++) {
    if (!kernel || names.length) {
      for (const key of names) {
        const v = refs[key];
        env[key] = V.isValue(v) ? V._sliceRow(v, i) : v[i];
      }
      const params: Record<string, any> = {};
      for (const key of regression.PARAMS[ir.op]) params[key] = sampler.evaluateExpr(paramsIR[key], env);
      kernel = regression.prepare(ir.op, params);
      if (!data) { rows = kernel.rows; data = new Float64Array(ctx.sampleCount * rows); }
      if (kernel.rows !== rows) throw new Error(`${ir.op}: response shape must stay fixed across draws`);
    }
    data!.set(kernel.sample(draw, prng).data, i * rows);
  }
  return shared.withParameterWeights(shared.measureFromValue({ shape: [ctx.sampleCount, rows], data }, {}), parents);
}

async function broadcast(name: string, d: any, ctx: any): Promise<any> {
  const axesLib = require('./kernel-broadcast-axes.ts');
  const orchestrator = require('./orchestrator.ts');
  const paramsIR = regression.bind(d.distOp, d.argIRs, d.kwargIRs || {});
  const names = regression.PARAMS[d.distOp];
  const { refArrays, fixedEnv, parents } = await shared.prepareDensityRefs(
    { kind: 'call', op: 'record', kwargs: paramsIR }, ctx, 'regression broadcast');
  const refs = Object.keys(refArrays), count = ctx.sampleCount;
  const cells: Record<string, any> = {};
  for (const key of names) {
    const usesAtom = refs.some((n: string) => orchestrator.collectSelfRefs(paramsIR[key]).has(n));
    const v = sampler.evaluateExprN(paramsIR[key], refArrays, count, fixedEnv);
    cells[key] = axesLib.cellAccessor(v, usesAtom, count);
  }
  const { axes, strides } = axesLib.resolveBroadcastAxes(names.map((key: string) =>
    ({ name: key, collectionAxes: cells[key].axes })));
  const size = axes.reduce((a: number, b: number) => a * b, 1);
  const prng = shared.makeMainThreadPrng(shared.nameSeed(name, ctx.rootKey));
  const factories: Record<string, any> = {};
  const draw = (kind: string, ...args: number[]) => {
    factories[kind] ||= sampler._internal.REGISTRY[kind].randFn.factory({ prng });
    return factories[kind](...args);
  };
  let rows = 0, data: Float64Array | null = null;
  for (let i = 0; i < count; i++) for (let cell = 0; cell < size; cell++) {
    const coord = axesLib.cellToCoord(cell, axes), params: Record<string, any> = {};
    for (const key of names) params[key] = cells[key].accessor(i, axesLib.coordToOffset(coord, strides[key]));
    const kernel = regression.prepare(d.distOp, params);
    if (!data) { rows = kernel.rows; data = new Float64Array(count * size * rows); }
    if (rows !== kernel.rows) throw new Error(`${d.distOp}: response shape must stay fixed across broadcast cells`);
    data!.set(kernel.sample(draw, prng).data, (i * size + cell) * rows);
  }
  const value = V.setLevels({ shape: [count, ...axes, rows], data: data || new Float64Array(0) },
    axes.length ? [1, axes.length] : [1]);
  return shared.withParameterWeights(shared.measureFromValue(value, {}), parents);
}

module.exports = { materialise, broadcast };

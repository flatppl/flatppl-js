'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, resolveBundle } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function dataContext(source: string, sources: Record<string, string>) {
  return makeMatCtx(source, { path: 'model.flatppl', bundle: { sources }, sampleCount: 4 });
}

test('load_data supplies dimensions and observations to a module-relative Normal score', async () => {
  const source = 'm=load_module("sub/model.flatppl")\nscore=logdensityof(m.P,m.d.y)';
  const files: Record<string, string> = {
    'models/sub/model.flatppl': `
      sizes=load_data(source="../data/model.json",valueset=cartprod(n=posintegers))
      n=sizes.n
      d=base.load_data("../data/model.json",cartprod(y=cartpow(reals,n)))
      x~iid(Normal(0,1),n)
      P=lawof(x)
    `,
    'models/data/model.json': '{"n":2,"y":[1,2,99],"unused":null}',
  };
  const reads: string[] = [];
  const bundle = await resolveBundle('models/main.flatppl', source, async (path: string) => {
    reads.push(path); return files[path];
  });
  const { ctx, lifted, built } = makeMatCtx(source, { path: bundle.primaryPath, bundle, sampleCount: 4 });
  assert.deepEqual(built.diagnostics, []);
  assert.deepEqual(reads, ['models/sub/model.flatppl', 'models/data/model.json']);
  assert.equal(lifted.modules.size, 1, 'JSON text is not compiled as a FlatPPL module');
  assert.equal(lifted.modules.get('models/sub/model.flatppl').bindings.get('d').node.value.callee.name, 'load_data');
  const scores = (await ctx.getMeasure('score')).samples;
  for (const score of scores) assert.ok(Math.abs(score - (-Math.log(2 * Math.PI) - 2.5)) < 1e-12);
});

test('load_data schema distinguishes tensors, nested vectors and unequal record fields', async () => {
  const { ctx } = dataContext(`
    d=load_data("data.json",cartprod(A=cartpow(reals,[2,3]),v=cartpow(integers,2),ok=booleans))
    A=d.A
    v=d.v
    product=A*[1.,2.,3.]
    ok=d.ok
    nested=load_data("nested.json",cartpow(cartpow(reals,2),2))
    item=nested[2][1]
    parts=load_data("parts.json",cartprod(cartpow(reals,2),cartpow(posreals,1)))
    total=sum(parts)
  `, { 'data.json': '{"A":[[1,2,3],[4,5,6]],"v":[7,8,9],"ok":true,"ignored":"anything"}',
    'nested.json': '[[2,3],[4,5],[6,7]]', 'parts.json': '[1,2,3]' });
  assert.deepEqual((await ctx.getMeasure('A')).intrinsicShape, [2, 3]);
  assert.deepEqual(Array.from((await ctx.getMeasure('product')).value.data), [14, 32]);
  assert.deepEqual(Array.from((await ctx.getMeasure('v')).value.data), [7, 8]);
  assert.equal((await ctx.getMeasure('ok')).samples[0], 1);
  assert.equal((await ctx.getMeasure('item')).samples[0], 4);
  assert.equal((await ctx.getMeasure('total')).samples[0], 6);
});

test('load_data selects table fields and row prefixes from both JSON layouts and text headers', async () => {
  const schema = 'cartpow(cartprod(y=reals,v=cartpow(reals,2)),2)';
  const { ctx } = dataContext(`
    rows=load_data("rows.json",${schema})
    cols=load_data("cols.json",${schema})
    a=sum(rows.y)+sum(rows.v[2])
    b=sum(cols.y)+sum(cols.v[2])
    csv=load_data("data.csv",cartpow(cartprod(y=reals),2))
    c=sum(csv.y)
    wsv=load_data("data.wsv",cartpow(reals,2))
    d=sum(wsv)
  `, { 'rows.json': '[{"y":1,"v":[2,3]},{"y":4,"v":[5,6]},{"ignored":null}]',
    'cols.json': '{"y":[1,4,99],"v":[[2,3],[5,6]],"ignored":[null]}',
    'data.csv': 'y,ignored\n"1.5",text\n2.5,other\n99,unused\n',
    'data.wsv': 'y\n1.5\n2.5\n99\n' });
  for (const name of ['a', 'b']) assert.equal((await ctx.getMeasure(name)).samples[0], 16);
  for (const name of ['c', 'd']) assert.equal((await ctx.getMeasure(name)).samples[0], 4);
});

test('load_data validates selected shapes and scalar/vector domains before scoring', async () => {
  const valid = dataContext(`
    d=load_data("data.json",cartprod(z=zerosum(3),p=stdsimplex(2),o=ordered(2),v=cartpow(interval(0,1),2)))
    score=sum(d.z)+sum(d.p)+sum(d.o)+sum(d.v)
  `, { 'data.json': '{"z":[-1,0,1],"p":[0.25,0.75],"o":[1,2],"v":[0,1]}' });
  assert.equal((await valid.ctx.getMeasure('score')).samples[0], 5);
  const cases: Array<[string, string]> = [
    ['cartprod(n=posintegers)', '{"n":1.5}'],
    ['cartprod(y=reals)', '{"other":1}'],
    ['cartpow(reals,3)', '[1,2]'],
    ['cartpow(reals,[2,2])', '[[1,2],[3]]'],
    ['reals', 'null'],
    ['booleans', '1'],
    ['zerosum(3)', '[1,1,1]'],
    ['ordered(2)', '[2,1]'],
    ['cartpow(interval(0,1),2)', '[0,2]'],
  ];
  for (const [schema, data] of cases) {
    const result = processSource(`d=load_data("data.json",${schema})`, { bundle: { sources: { 'data.json': data } } });
    assert.ok(result.diagnostics.some((d: any) => d.severity === 'error' && /load_data/.test(d.message)), schema);
  }
  assert.throws(() => dataContext('d=load_data("x.csv",cartpow(reals,1))', { 'x.csv': 'y\n""\n' }), /load_data/);
});

test('load_data anything stays opaque until access, including invalid source text', async () => {
  const { ctx, built } = dataContext('opaque=load_data("data.json",anything)\nx=2+3', { 'data.json': '{not JSON' });
  assert.deepEqual(built.diagnostics, []);
  assert.equal((await ctx.getMeasure('x')).samples[0], 5);
  await assert.rejects(ctx.getMeasure('opaque'), /explicit valueset/);
  assert.throws(() => dataContext('d=load_data("data.json",reals)', { 'data.json': '{not JSON' }), /JSON/);
  assert.throws(() => dataContext('d=load_data("data.arrow",reals)', { 'data.arrow': 'binary' }), /Arrow IPC/);
  assert.throws(() => dataContext('d=load_data("data.parquet",reals)', { 'data.parquet': 'x' }),
    /unsupported file extension `\.parquet`.*`\.arrow`, `\.arrows` and `\.safetensors`/);
  assert.throws(() => dataContext('d=load_data("absent.json",reals)', {}), /not found in bundle/);
});

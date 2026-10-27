'use strict';

// §03 literal vectors contain realized values; §07 matrix-vector multiplication
// acts on those same draws. Oracles are direct complex linear combinations.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const M = 'M = rowstack([[complex(1.0,2.0),complex(0.0,1.0)],[complex(3.0,0.0),complex(1.0,-1.0)]])\n';
const N = 5;
function close(got: number, want: number) { assert.ok(Math.abs(got - want) < 1e-13, `${got} vs ${want}`); }

for (const [label, inputs, vector, right] of [
  ['mixed scalar and literal', 'x ~ Normal(0.0,1.0)\n', '[x,1.0]', (_x: number, _z: number) => 1],
  ['computed scalar', 'x ~ Normal(0.0,1.0)\n', '[x,1.0-x]', (x: number, _z: number) => 1 - x],
  ['two scalar draws', 'x ~ Normal(0.0,1.0)\nz ~ Normal(0.0,1.0)\n', '[x,z]', (_x: number, z: number) => z],
] as const) {
  test(`complex matrix consumes the ${label} vector as one value`, async () => {
    const { ctx } = ctxFor(M + inputs + `v = ${vector}\ny = M*v\n`, N);
    const x = (await ctx.getMeasure('x')).samples;
    const z = label === 'two scalar draws' ? (await ctx.getMeasure('z')).samples : [];
    const v = await ctx.getMeasure('v'), y = await ctx.getMeasure('y');
    for (let i = 0; i < N; i++) {
      const r = right(x[i], z[i]);
      close(v.samples[2 * i], x[i]); close(v.samples[2 * i + 1], r);
      close(y.samples[2 * i], x[i]); close(y.imag[2 * i], 2 * x[i] + r);
      close(y.samples[2 * i + 1], 3 * x[i] + r); close(y.imag[2 * i + 1], -r);
    }
  });
}

test('the packed stochastic vector also obeys ordinary real matrix arithmetic', async () => {
  const { ctx } = ctxFor('x ~ Normal(0.0,1.0)\nv=[x,1.0-x]\nM=rowstack([[1.0,2.0],[3.0,4.0]])\ny=M*v\n', N);
  const x = (await ctx.getMeasure('x')).samples;
  const y = (await ctx.getMeasure('y')).samples;
  for (let i = 0; i < N; i++) { close(y[2 * i], 2 - x[i]); close(y[2 * i + 1], 4 - x[i]); }
});

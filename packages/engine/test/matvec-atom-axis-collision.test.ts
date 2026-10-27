'use strict';

// The sample axis is provenance, not an intrinsic matrix dimension (§03/§07).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const complexMatrix = 'rowstack([[complex(1.0,2.0),complex(0.0,1.0)],[complex(3.0,0.0),complex(1.0,-1.0)]])';
function close(got: number, want: number) { assert.ok(Math.abs(got - want) < 1e-13, `${got} vs ${want}`); }

for (const N of [1, 2, 3]) {
  test(`shared complex matrix times iid vector at sample count ${N}`, async () => {
    const { ctx } = ctxFor(`M=${complexMatrix}\nv~iid(Normal(0.0,1.0),2)\ny=M*v\n`, N);
    const v = (await ctx.getMeasure('v')).samples, y = await ctx.getMeasure('y');
    for (let i = 0; i < N; i++) {
      const x = v[2*i], z = v[2*i+1];
      close(y.samples[2*i], x); close(y.imag[2*i], 2*x+z);
      close(y.samples[2*i+1], 3*x+z); close(y.imag[2*i+1], -z);
    }
  });

  test(`shared real matrix times iid vector at sample count ${N}`, async () => {
    const { ctx } = ctxFor('M=rowstack([[1.0,2.0],[3.0,4.0]])\nv~iid(Normal(0.0,1.0),2)\ny=M*v\n', N);
    const v = (await ctx.getMeasure('v')).samples, y = (await ctx.getMeasure('y')).samples;
    for (let i = 0; i < N; i++) {
      close(y[2*i], v[2*i]+2*v[2*i+1]); close(y[2*i+1], 3*v[2*i]+4*v[2*i+1]);
    }
  });
}

test('shared computed adjoint matrix retains complex orientation at the collision', async () => {
  const { ctx } = ctxFor(`M=${complexMatrix}\nv~iid(Normal(0.0,1.0),2)\ny=adjoint(M+M)*v\n`, 2);
  const v = (await ctx.getMeasure('v')).samples, y = await ctx.getMeasure('y');
  for (let i = 0; i < 2; i++) {
    const x = v[2*i], z = v[2*i+1];
    close(y.samples[2*i], 2*x+6*z); close(y.imag[2*i], -4*x);
    close(y.samples[2*i+1], 2*z); close(y.imag[2*i+1], -2*x+2*z);
  }
});

test('both complex planes survive the collision for a complex sampled vector', async () => {
  const { ctx } = ctxFor(`M=${complexMatrix}\nv~iid(Dirac(complex(2.0,-1.0)),2)\ny=M*v\n`, 2);
  const y = await ctx.getMeasure('y');
  assert.deepEqual(Array.from(y.samples), [5, 7, 5, 7]);
  assert.deepEqual(Array.from(y.imag), [5, -6, 5, -6]);
});

test('a shared rectangular matrix keeps its full contracted axis at the collision', async () => {
  const { ctx } = ctxFor('M=rowstack([[1.0,2.0,3.0],[4.0,5.0,6.0]])\nv~iid(Normal(0.0,1.0),3)\ny=M*v\n', 2);
  const v = (await ctx.getMeasure('v')).samples, y = (await ctx.getMeasure('y')).samples;
  for (let i = 0; i < 2; i++) {
    const x = v[3*i], z = v[3*i+1], w = v[3*i+2];
    close(y[2*i], x+2*z+3*w); close(y[2*i+1], 4*x+5*z+6*w);
  }
});

test('two stochastic column vectors retain the scalar multiplication domain refusal', async () => {
  const { ctx } = ctxFor('v~iid(Normal(0.0,1.0),2)\nw~iid(Normal(0.0,1.0),2)\ny=v*w\n', 2);
  await assert.rejects(() => ctx.getMeasure('y'), /vector.*vector|column.*column/);
});

'use strict';

// §06: `bijection(f, f_inv, logvolume)` "is semantically `f`", so
// `pushfwd(b, M)` has the variate type of f applied to M's variate. The
// codomain query read only a `functionof`, so a bijection fell back to a
// scalar variate. Over an iid base that typed the pushforward as `real` and
// raised a false "expects real, got array of real" on a point it scores.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, types: T } = require('..');

function infer(src: string) {
  const { bindings, diagnostics } = processSource(src + '\n');
  const errors = diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
  return { bindings, errors };
}

const VEC3 = T.array(1, [3], T.REAL);
const B = 'b = bijection(fn(2.0 * _), fn(_ / 2.0), 3.0 * log(2.0))\n';
const BASE = 'iid(Normal(1.0, 2.0), 3)';
const POINT = '\nlp = logdensityof(lawof(y), [3.0, 1.0, -2.0])';

const DRAWS: [string, string][] = [
  ['a named bijection', `${B}y ~ pushfwd(b, ${BASE})${POINT}`],
  ['an inline bijection', `y ~ pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), 3.0 * log(2.0)), ${BASE})${POINT}`],
  ['an alias of a bijection', `${B}c = b\ny ~ pushfwd(c, ${BASE})${POINT}`],
  ['a bijection whose f is an alias',
    `g = fn(2.0 * _)\nh = g\nb = bijection(h, fn(_ / 2.0), 3.0 * log(2.0))\ny ~ pushfwd(b, ${BASE})${POINT}`],
  ['an elementwise exp bijection',
    'b = bijection(fn(exp.(_)), fn(log.(_)), fn(sum(_)))\ny ~ pushfwd(b, iid(Normal(0.5, 1.5), 3))'
    + '\nlp = logdensityof(lawof(y), [2.0, 1.0, 0.5])'],
];

for (const [label, src] of DRAWS) {
  test(`pushfwd of ${label} over an iid base has an array variate`, () => {
    const { bindings, errors } = infer(src);
    assert.deepEqual(errors, []);
    assert.ok(T.equal(bindings.get('y').inferredType, VEC3),
      JSON.stringify(bindings.get('y').inferredType));
  });
}

const KERNEL_B = `mu = elementof(reals)\n${B}`;
const KERNELS: [string, string][] = [
  ['a functionof kernel body', `${KERNEL_B}K = functionof(pushfwd(b, iid(Normal(mu, 2.0), 3)), mu = mu)`],
  ['kernelof of a pushfwd draw', `${KERNEL_B}y ~ pushfwd(b, iid(Normal(mu, 2.0), 3))\nK = kernelof(y, mu = mu)`],
  ['a pushfwd of a kernel', `${KERNEL_B}K0 = functionof(iid(Normal(mu, 2.0), 3), mu = mu)\nK = pushfwd(b, K0)`],
];

for (const [label, model] of KERNELS) {
  test(`${label} through a bijection has an array output variate`, () => {
    const { bindings, errors } = infer(`${model}\nlp = logdensityof(K(mu = 1.0), [3.0, 1.0, -2.0])`);
    assert.deepEqual(errors, []);
    const K = bindings.get('K').inferredType;
    assert.equal(K.kind, 'kernel');
    assert.ok(T.equal(K.result, T.measure(VEC3)), JSON.stringify(K.result));
  });
}

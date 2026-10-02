'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const { toJS } = require('./_value-helpers.ts');

function fixed(source: string) {
  const parsed = processSource(source);
  const state = orchestrator.buildDerivations(parsed.bindings, {
    moduleRegistry: parsed.loweredModule.moduleRegistry,
  });
  return (name: string) => {
    const value = state.fixedValues.get(name);
    const errors = parsed.diagnostics.concat(state.diagnostics)
      .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
    assert.deepEqual(errors, []);
    assert.notEqual(value, undefined, name);
    return value;
  };
}

function close(actual: number, expected: number) {
  assert.ok(Math.abs(actual - expected) <= 1e-12 * Math.max(Math.abs(expected), Number.MIN_VALUE),
    `${actual} != ${expected}`);
}

test('diagonal broadcasts evaluate implicit zeros and reductions include them', () => {
  const get = fixed(`D = diagmat([2,3])
    comparison = broadcast(le, D, D)
    power = D .^ D
    average = mean(D)
    product = prod(D)
    smallest = minimum(D)
    middle = median(D)`);
  assert.deepEqual(toJS(get('comparison')), [[1,1],[1,1]]);
  assert.deepEqual(toJS(get('power')), [[4,1],[1,27]]);
  assert.equal(get('average'), 1.25);
  for (const name of ['product', 'smallest']) assert.equal(get(name), 0);
  assert.equal(get('middle'), 1);
});

test('matrix algorithms read transpose and diagonal storage logically', () => {
  const get = fixed(`e = standard_module("ext-linear-algebra", "0.1")
    A = rowstack([[1,2],[3,4]])
    inverse = inv(transpose(A))
    upper = diag(transpose(A), 1)
    kron_view = e.kron(transpose(A), rowstack([[1]]))
    kron_diag = e.kron(diagmat([2,3]), rowstack([[1]]))`);
  const inverse = toJS(get('inverse')).flat();
  [-2,1.5,1,-0.5].forEach((x, i) => close(inverse[i], x));
  assert.deepEqual(toJS(get('upper')), [3]);
  assert.deepEqual(toJS(get('kron_view')), [[1,3],[2,4]]);
  assert.deepEqual(toJS(get('kron_diag')), [[2,0],[0,3]]);
});

test('matrix addition accepts opposite storage orientations', () => {
  const get = fixed('x = transpose(rowstack([[1,2],[3,4]])) + rowstack([[1,3],[2,4]])');
  assert.deepEqual(toJS(get('x')), [[2,6],[4,8]]);
});

test('diagonal multiplication preserves complex vectors and scalars', () => {
  const get = fixed(`D = diagmat([2,3])
    literal = broadcast(imag, D * [complex(0,1),complex(1,2)])
    planar = broadcast(imag, D * broadcast(complex,[0,1],[1,2]))
    scalar = broadcast(imag, D * complex(0,1))`);
  assert.deepEqual(toJS(get('literal')), [2,6]);
  assert.deepEqual(toJS(get('planar')), [2,6]);
  assert.deepEqual(toJS(get('scalar')), [[2,0],[0,3]]);
});

test('complex scalar trig functions and predicates use their specified domain', () => {
  const get = fixed(`s = sin(complex(0,1))
    c = cos(complex(0,1))
    finite = isfinite(complex(0,1))
    zero = iszero(complex(0,0))
    predicates = broadcast(iszero, broadcast(complex,[0,0],[0,1]))`);
  close(get('s').im, Math.sinh(1));
  close(get('c').re, Math.cosh(1));
  assert.equal(get('finite'), true);
  assert.equal(get('zero'), true);
  assert.deepEqual(toJS(get('predicates')), [1,0]);
});

test('cat preserves nested cells and imaginary components', () => {
  const get = fixed(`nested = cat([[1,2]], [[3,4]])
    imaginary = broadcast(imag, cat(broadcast(complex,[0],[1]), broadcast(complex,[2],[3])))`);
  assert.deepEqual(toJS(get('nested')), [[1,2],[3,4]]);
  assert.deepEqual(toJS(get('imaginary')), [1,3]);
});

test('Euclidean norm and unit direction survive finite scale changes', () => {
  for (const scale of ['1e200', '1e-200']) {
    const get = fixed(`norm = l2norm([${scale}])
      unit = l2unit([${scale},${scale}])`);
    close(get('norm'), Number(scale));
    for (const x of toJS(get('unit'))) close(x, Math.SQRT1_2);
  }
  assert.ok(Number.isNaN(fixed('x = l2norm([inf,0/0])')('x')));
});

test('logsoftmax retains its normalization under a large common offset', () => {
  for (const offset of ['1e20', '-1e20']) {
    const get = fixed(`x = logsoftmax([${offset},${offset}])`);
    for (const x of toJS(get('x'))) close(x, -Math.LN2);
  }
});

test('broadcast domain restrictions reject fractional integers and non-booleans', () => {
  const get = fixed('ints = broadcast(integer,[1,2])\nbools = broadcast(boolean,[0,1])');
  assert.deepEqual(toJS(get('ints')), [1,2]);
  assert.deepEqual(toJS(get('bools')), [0,1]);
  assert.throws(() => fixed('x = broadcast(integer,[1.5])')('x'), /not an integer/);
  assert.throws(() => fixed('x = broadcast(boolean,[2])')('x'), /not a boolean/);
});

test('filter preserves complex elements and nested table columns', () => {
  const get = fixed(`zs = broadcast(complex,[1,3,5],[2,-4,6])
    imaginary = broadcast(imag, filter(fn(imag(_) > 0), zs))
    selected = filter(fn(_.a > 2), table(a=[1,3], b=table(z=[4,8])))
    nested = selected.b.z
    count = lengthof(selected)`);
  assert.deepEqual(toJS(get('imaginary')), [2,6]);
  assert.deepEqual(toJS(get('nested')), [8]);
  assert.equal(get('count'), 1);
});

test('complex literal reductions preserve both components', () => {
  const get = fixed(`zs = [complex(1,2),complex(3,4)]
    total = sum(zs)
    average = mean(zs)
    product = prod(zs)`);
  assert.deepEqual(get('total'), { re:4, im:6 });
  assert.deepEqual(get('average'), { re:2, im:3 });
  assert.deepEqual(get('product'), { re:-5, im:10 });
});

test('reduce and scan accept named builtins while broadcast respects user bindings', () => {
  const get = fixed(`total = reduce(add,[1,2,3])
    cumulative = scan(add,0,[1,2,3])
    exp(x) = x + 10
    custom = broadcast(exp,[1,2])`);
  assert.equal(get('total'), 6);
  assert.deepEqual(toJS(get('cumulative')), [1,3,6]);
  assert.deepEqual(toJS(get('custom')), [11,12]);
});

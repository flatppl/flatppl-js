'use strict';

// =====================================================================
// broadcastn-bench.ts — hot-path gate for `broadcastN`
// =====================================================================
//
// `broadcastN` (sampler-eval-batched.ts) is the batched scalar-primitive
// dispatcher every elementwise op shares, so a per-call regression there
// is paid by the whole engine. This harness times it directly at the
// shapes the runtime actually hits: arity 1/2/3, atom-batched and
// atom-indep operands, raw Float64Array and wrapped Value inputs.
//
// Run:  node --experimental-strip-types packages/engine/bench/broadcastn-bench.ts
//
// NOT a test — it prints a table and exits 0. Re-run before and after a
// change to `broadcastN`; the "ns/atom" column is the comparison.

const valueLib = require('../value.ts');
const { broadcastN } = require('../sampler-eval-batched.ts');

const N = 4096;
const REPS = 400;
const BATCHES = 25;

// Minimum, not mean or median: every sample is the true cost plus
// scheduler noise, so the smallest batch is the closest estimate of the
// cost itself. Medians on this harness swing by 2× run to run.
function best(xs: number[]): number {
  let m = xs[0];
  for (let i = 1; i < xs.length; i++) if (xs[i] < m) m = xs[i];
  return m;
}

function filled(n: number): Float64Array {
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = 1 + (i % 7) * 0.125;
  return a;
}

const f1 = (x: number) => Math.exp(x);
const f2 = (x: number, y: number) => x * y;
const f3 = (c: number, x: number, y: number) => (c ? x : y);

const raw = filled(N);
const wrapped = valueLib.batchedScalar(filled(N));
const cell = valueLib.withShape(filled(N * 4), [N, 4]);
cell.outerRank = 1;

const CASES: [string, any, any[]][] = [
  ['arity1 raw batched', f1, [raw]],
  ['arity1 Value batched', f1, [wrapped]],
  ['arity1 rank-0', f1, [valueLib.scalar(2.5)]],
  ['arity2 raw × raw', f2, [raw, filled(N)]],
  ['arity2 raw × scalar', f2, [raw, 2.5]],
  ['arity2 Value × Value', f2, [wrapped, valueLib.batchedScalar(filled(N))]],
  ['arity3 all batched', f3, [raw, filled(N), filled(N)]],
  ['arity3 mixed', f3, [raw, 2.5, filled(N)]],
  ['arity1 cell [N,4]', f1, [cell]],
  ['arity2 cell × scalar', f2, [cell, 2.5]],
];

console.log('broadcastN — N=' + N + ', ' + REPS + ' reps, best of '
  + BATCHES + ' batches\n');
console.log('case'.padEnd(24) + 'best ms'.padStart(11)
  + 'ns/elem'.padStart(11) + 'elems'.padStart(8));
console.log('-'.repeat(54));

for (const [label, fn, args] of CASES) {
  let out: any;
  for (let w = 0; w < 80; w++) out = broadcastN(fn, args, N);
  const elems = (out && out.data ? out.data.length
    : (out && out.length ? out.length : 1));
  const batches: number[] = [];
  for (let b = 0; b < BATCHES; b++) {
    const t0 = process.hrtime.bigint();
    for (let r = 0; r < REPS; r++) out = broadcastN(fn, args, N);
    batches.push(Number(process.hrtime.bigint() - t0) / 1e6);
  }
  const ms = best(batches);
  const nsPer = (ms * 1e6) / (REPS * elems);
  console.log(label.padEnd(24) + ms.toFixed(3).padStart(11)
    + nsPer.toFixed(2).padStart(11) + String(elems).padStart(8));
}

const test = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// §04 aggregation resolves axis lengths from each application's arrays.
// A scan reuses its callback IR; one call must not fix that IR's dimensions.
test('scan applications resolve their own aggregate dimensions', async () => {
  const { ctx } = makeMatCtx(`
    column_sums(X) = aggregate(sum, [.j], exp(X[.i, .j]))
    step(acc, i, p) = column_sums(p.X)
    f(X) = scan(step, X[1, :], [1], record(X = X))
    first = sum(f(log.(rowstack([[1, 2], [3, 4]])))[1])
    larger = sum(f(log.(rowstack([[1, 2, 3], [4, 5, 6], [7, 8, 9]])))[1])
    smaller = sum(f(log.(rowstack([[7]])))[1])
  `, { sampleCount: 1 });
  // exp(log(X)) followed by column sums and a total sums every matrix entry.
  for (const [name, expected] of [['first', 10], ['larger', 45], ['smaller', 7]] as const) {
    const actual = (await ctx.getMeasure(name)).samples[0];
    assert.ok(Math.abs(actual - expected) < 1e-12, `${name}: ${actual} vs ${expected}`);
  }
});

// Directly enumerate latent paths, independently of the forward algorithm.
function hmmOracle(initial: number[], transition: number[][], emissions: number[][]): number {
  if (initial.length === 1) return emissions.reduce((sum, row) => sum + row[0], 0);
  let mass = 0;
  function visit(t: number, previous: number, weight: number) {
    if (t === emissions.length) { mass += weight; return; }
    for (let state = 0; state < initial.length; state++) {
      const probability = t === 0 ? initial[state] : transition[previous][state];
      visit(t + 1, state, weight * probability * Math.exp(emissions[t][state]));
    }
  }
  visit(0, 0, 1);
  return Math.log(mass);
}

test('HMM forward calls preserve each state count and log scale', async () => {
  const cases = [
    { initial: [0.6, 0.4], transition: [[0.8, 0.2], [0.3, 0.7]], emissions: [[-0.2, -1.5]] },
    { initial: [0.2, 0.3, 0.5], transition: [[0.6, 0.3, 0.1], [0.1, 0.7, 0.2], [0.3, 0.2, 0.5]],
      emissions: [[-1, -0.2, -1.8], [-0.3, -1.4, -0.7], [-0.8, -0.5, -1.1]] },
    { initial: [1], transition: [[1]], emissions: [[-700], [-700]] },
  ];
  const source = `
    maxima(scores) = aggregate(maximum, [.j], scores[.i, .j])
    logsums(scores, shifts) = log.(aggregate(sum, [.j], exp(scores[.i, .j] - shifts[.j]))) .+ shifts
    predict(scores) = logsums(scores, maxima(scores))
    step(weights, i, p) = ifelse(i == 1, weights,
      predict(addaxes(weights, 0, 1) .+ p.log_transition)) .+ p.log_emissions[i, :]
    hmm(initial, transition, emissions) = logsumexp(scan(step, log.(initial),
      indicesof(emissions[:, 1]), record(log_transition = log.(transition), log_emissions = emissions))[
        lengthof(emissions[:, 1])])
  ` + cases.map((c, i) => `result${i} = hmm(${JSON.stringify(c.initial)},
    rowstack(${JSON.stringify(c.transition)}), rowstack(${JSON.stringify(c.emissions)}))`).join('\n');
  const { ctx } = makeMatCtx(source, { sampleCount: 1 });
  for (let i = 0; i < cases.length; i++) {
    const actual = (await ctx.getMeasure(`result${i}`)).samples[0];
    const expected = hmmOracle(cases[i].initial, cases[i].transition, cases[i].emissions);
    assert.ok(Math.abs(actual - expected) < 1e-12, `case ${i}: ${actual} vs ${expected}`);
  }
});

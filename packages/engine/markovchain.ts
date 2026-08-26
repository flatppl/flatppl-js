'use strict';

// =====================================================================
// markovchain.ts — spec §06 `markovchain(kernel, init, n)`
// =====================================================================
//
// §06 dependent composition, verbatim: "`kernel` is a Markov kernel
// `(state) -> measure_over_state`; `init` is a value in the state space;
// `n` is a positive integer. Step $i$ is $\text{traj}_i \sim
// \kappa(\text{traj}_{i-1})$ with $\text{traj}_0 = \text{init}$. The
// initial value is not part of the trajectory. The resulting measure is a
// measure over arrays `[traj[1], ..., traj[n]]`, excluding the initial
// state."
//
// So the variate is an ARRAY of n states, `init` is a VALUE and carries no
// density of its own, and the §06 "Density of composed measures" reduction
// for the constituent conditionals gives
//
//   logdensityof(markovchain(K, init, n), [x_1, …, x_n])
//     = Σ_{j=1..n} logdensityof(K(x_{j-1}), x_j),   x_0 = init
//
// n terms, not n+1 — there is no base-measure term, because §06 makes the
// initial state a value rather than a measure.
//
// WHY THIS IS ITS OWN OP RATHER THAN A `jointchain` EXPANSION.
// `jointchain(K(init), K, …, K)` has the same factorisation on paper, and
// reusing the landed chain machinery was the first design tried. Three
// measured facts ruled it out on this engine at `14d20337`:
//
//   1. A bare (non-broadcast) positional `jointchain` of ≥ 3 components
//      samples NaN and scores NaN, for EVERY step spelling — `fn(_)`
//      holes and by-name `kernelof(…, prev = prev)` alike. Its derivation
//      declares step i's inputs as EVERY variate to its left
//      (`inputs: ["s0","s1"]`), so the single-input step body receives the
//      cat pair and the arithmetic goes NaN. That is the open jointchain
//      half of the two-feeds divergence in TODO-flatppl-js.md, whose
//      kchain half branch `kchain-inline-kernel-arity` fixed for the
//      MARGINAL chain only. The one correct sequential executor today is
//      `_executeJointChainComposite` (mat-broadcast.ts), reachable only
//      through the kernel-broadcast path — which is why
//      `test/fixtures/hierarchical-state-space.flatppl` samples calibrated
//      while the same chain written bare does not.
//   2. `jointchain` requires ≥ 2 components, so it cannot spell n = 1,
//      which §06 admits (`n` is any positive integer).
//   3. A positional `jointchain` retaining all its variates materialises
//      to a TUPLE measure, and §06 says markovchain's variate is an ARRAY.
//      Whether a positional chain variate is a tuple or an array is an
//      unsettled cross-engine ruling (TODO-flatppl-js.md); §06 leaves
//      markovchain no such latitude, so inheriting the tuple would import
//      an open question into an op that does not have one.
//
// The Markov feed — the previous state ALONE — is what makes a native
// scan short: there is no cat to build and no history to thread.
//
// SCOPE. The kernel must be a single-input kernel whose body is a
// sampleable scalar distribution (`fn(Normal(mu = _, sigma = s))`,
// `x -> Normal(x, s)`, `kernelof(Normal(mu = p, sigma = s), p = p)`),
// which is the same "scalar dist step" scope the jointchain composite
// recogniser takes. §06 also admits record states (whose trajectories are
// tables) and composite kernel bodies; both are a located refusal here
// rather than a silent mis-lowering.
//
// SAMPLING ≡ DENSITY. Both sides read ONE producer, `stepDistIR`, which
// substitutes the step kernel's input param with a given IR. The sampler
// substitutes a ref to the previous sampled column; the density walker
// substitutes the previous OBSERVED element (`s{j-1}`, the name
// `walkJointFieldsOrPositional` threads positionally). So the two paths
// cannot drift in which distribution they place at step j.

const valueLib = require('./value.ts');

/** `lawof(<x>)` → `<x>`; anything else unchanged. */
function _peelLawof(ir: any): any {
  return (ir && ir.kind === 'call' && ir.op === 'lawof'
    && Array.isArray(ir.args) && ir.args.length === 1) ? ir.args[0] : ir;
}

/** REGISTRY param names for a distribution op, or [] when the sampler
 *  is not loadable (classify time). Null when the sampler is loaded and
 *  does not know the op — which callers reach only through the
 *  `SAMPLEABLE_DISTRIBUTIONS` gate, and every member of that set carries a
 *  REGISTRY `params` list (checked: the set difference is empty), so there is
 *  no loaded-but-unknown arm to write. */
function _distParamsOf(distOp: string): string[] {
  let registry: any = null;
  try {
    const sampler = require('./sampler.ts');
    if (sampler && sampler._internal && sampler._internal.REGISTRY) {
      registry = sampler._internal.REGISTRY;
    }
  } catch (_) { /* classify-time conservatism */ }
  const entry = registry && registry[distOp];
  return (entry && Array.isArray(entry.params)) ? entry.params : [];
}

/**
 * The step descriptor for a markovchain's kernel argument, or a
 * `{ reason }` refusal naming what the shape is instead.
 *
 * `compIR` is either a self-ref to a kernel binding or an inline
 * `functionof` (`fn` / `kernelof` / `->` all lower to one).
 */
function describeStepKernel(compIR: any, bindings: any): any {
  const SAMPLEABLE = require('./ir-shared.ts').SAMPLEABLE_DISTRIBUTIONS;
  let ir = compIR;
  let shownAs = 'the kernel argument';
  if (ir && ir.kind === 'ref' && ir.ns === 'self') {
    shownAs = '`' + ir.name + '`';
    if (!bindings || !bindings.has || !bindings.has(ir.name)) {
      return { reason: shownAs + ' names no binding' };
    }
    const b = bindings.get(ir.name);
    ir = b && b.ir;
  }
  if (!ir || ir.kind !== 'call' || ir.op !== 'functionof') {
    return {
      reason: shownAs + ' is not a kernel — §06 markovchain takes a Markov '
        + 'kernel `(state) -> measure_over_state` as its first argument',
    };
  }
  const params: string[] = Array.isArray(ir.params) ? ir.params : [];
  if (params.length !== 1) {
    return {
      reason: shownAs + ' has ' + params.length + ' inputs; §06 markovchain '
        + 'takes a single-input kernel `(state) -> measure_over_state`',
    };
  }
  const dist = _peelLawof(ir.body);
  if (!dist || dist.kind !== 'call' || !dist.op
      || !SAMPLEABLE || !SAMPLEABLE.has(dist.op)) {
    return {
      reason: shownAs + '\'s body is not a sampleable distribution call; this '
        + 'engine implements markovchain over a scalar-distribution step '
        + 'kernel only (a composite or record-state step is not lowered)',
    };
  }
  const distParams = _distParamsOf(dist.op);
  // §05 lets a distribution take its parameters positionally
  // (`Normal(x, sqrt(2*D*dt))`, the §06 markovchain example itself). Name them
  // from the REGISTRY order so the two execution paths only ever see kwargs.
  const distKwargs: Record<string, any> = Object.assign({}, dist.kwargs || {});
  const positional: any[] = Array.isArray(dist.args) ? dist.args : [];
  if (positional.length > 0) {
    if (positional.length > distParams.length) {
      return {
        reason: shownAs + '\'s body passes ' + positional.length + ' positional '
          + 'arguments to `' + dist.op + '`, which takes ' + distParams.length,
      };
    }
    for (let i = 0; i < positional.length; i++) distKwargs[distParams[i]] = positional[i];
  }
  return {
    inputParam: params[0],
    distOp: dist.op,
    distParams,
    distKwargs,
  };
}

/**
 * Step j's distribution IR, with the kernel's input param replaced by
 * `prevIR`. The single producer both the sampler and the density walker
 * read, so step j is the same distribution on both paths.
 */
function stepDistIR(step: any, prevIR: any): any {
  const substituteKernelParams = require('./mat-broadcast.ts').substituteKernelParams;
  const kwargs: Record<string, any> = {};
  const subMap: Record<string, any> = {};
  subMap[step.inputParam] = prevIR;
  for (const pn of Object.keys(step.distKwargs)) {
    kwargs[pn] = substituteKernelParams(
      step.distKwargs[pn], [step.inputParam], [step.inputParam], subMap);
  }
  return { kind: 'call', op: step.distOp, kwargs };
}

/**
 * The density-side canonical IR: a POSITIONAL `joint` of the n step
 * distributions, step j's input bound to `s{j-1}` (step 0's to `init`).
 *
 * `walkJointFieldsOrPositional` consumes a positional joint's components
 * left to right off a flat array value, threading the scalar each
 * component consumed into the overlay under `s{i}` — which is exactly the
 * §06 Markov conditioning, so the n transition densities sum with no
 * markovchain-specific density arm.
 */
function densityIR(d: any): any {
  const args: any[] = [];
  const selfThreaded: string[] = [];
  for (let j = 0; j < d.n; j++) {
    const prevIR = (j === 0)
      ? d.initIR
      : { kind: 'ref', ns: 'self', name: 's' + (j - 1) };
    if (j > 0) selfThreaded.push('s' + (j - 1));
    args.push(stepDistIR(d.step, prevIR));
  }
  // `selfThreaded` rides ON THE NODE, naming the refs this joint satisfies from
  // its own observed value, so clm's ⊆ check can exclude them wherever the node
  // ends up. Keying that exclusion off the enclosing derivation's kind instead
  // does not survive any indirection, and `x ~ markovchain(…)` is already one
  // — `~` makes `x` an alias to an anon markovchain binding, so the enclosing
  // kind at lowering time is 'alias', 'record', or 'likelihood_density'.
  //
  // The list is carried rather than re-derived from the `s{i}` convention,
  // because a `jointchain` names its FED step variates `s{i}` too: a blanket
  // exclusion of that spelling would silently drop real ⊆ gaps there. Carrying
  // it also makes nesting work — two trajectories in one body each declare
  // their own, and the density walker's per-level copy-on-write overlay keeps
  // an inner `s0` from colliding with an outer one.
  return { kind: 'call', op: 'joint', args, selfThreaded };
}

/**
 * A scalar or per-atom value as a length-N column. Null when neither, which
 * every caller turns into a located refusal.
 *
 * An atom-independent scalar arrives as a plain number (measured across the
 * init spellings: a literal, a fixed binding ref, an arithmetic expression);
 * only a per-atom quantity arrives as a Value, with shape [N]. A Value of any
 * OTHER shape is an array-valued state or parameter, and this returns null for
 * it rather than reading `data[0]` — reading element 0 of a length-1 array
 * would silently lower `markovchain(f, [2.0], n)`, an ARRAY state §06 admits
 * and this engine does not implement, as though it were the scalar 2.0.
 */
function _asColumn(pv: any, N: number): Float64Array | null {
  if (typeof pv === 'number' || typeof pv === 'boolean') {
    const buf = new Float64Array(N);
    buf.fill(+pv);
    return buf;
  }
  if (valueLib.isValue(pv)) {
    return (pv.shape.length === 1 && pv.shape[0] === N) ? pv.data : null;
  }
  if (pv && pv.BYTES_PER_ELEMENT !== undefined && pv.length === N) return pv;
  return null;
}

/**
 * Sample the trajectory: one `sampleN` per step over N atoms, threading
 * step j-1's column as step j's input (the scan carry; `init` is the
 * carry at j = 0). Returns an ARRAY measure with per-atom dims `[n]`,
 * atom-major — §06's "measure over arrays `[traj[1], …, traj[n]]`".
 */
function matMarkovchain(name: string, d: any, ctx: any): Promise<any> {
  const sampler = require('./sampler.ts');
  const shared = require('./materialiser-shared.ts');
  const empirical = require('./empirical.ts');
  const N = ctx.sampleCount;
  const n = d.n;

  // One aggregate node carrying every free expression (the init value and
  // each step kwarg) so per-atom refs resolve in a single pass.
  const aggregate: any = {
    kind: 'call', op: 'joint',
    args: [d.initIR].concat(Object.keys(d.step.distKwargs)
      .map((pn: string) => d.step.distKwargs[pn])),
  };

  return shared.prepareDensityRefs(aggregate, ctx, 'markovchain').then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const initVal = sampler.evaluateExprN(d.initIR, refArrays, N, fixedEnv, undefined);
    const initCol = _asColumn(initVal, N);
    if (!initCol) {
      const shp = valueLib.isValue(initVal) ? JSON.stringify(initVal.shape) : typeof initVal;
      return Promise.reject(new Error('markovchain: `init` resolved to ' + shp
        + ' (expected a scalar or one value per atom) — §06 makes `init` a '
        + 'value in the state space, and this engine lowers a scalar state'));
    }

    const cols: Float64Array[] = new Array(n);
    let acc: Promise<any> = Promise.resolve();
    for (let j = 0; j < n; j++) {
      const jj = j;
      acc = acc.then(() => {
        const prev = (jj === 0) ? initCol : cols[jj - 1];
        const stepRefs: Record<string, any> = Object.assign({}, refArrays);
        stepRefs.__mc_prev = valueLib.batchedScalar(prev);
        const ir = stepDistIR(d.step, { kind: 'ref', ns: 'self', name: '__mc_prev' });
        const distKwargs: Record<string, any> = {};
        const sampleRefs: Record<string, any> = {};
        for (const pn of Object.keys(ir.kwargs)) {
          const pv = sampler.evaluateExprN(ir.kwargs[pn], stepRefs, N, fixedEnv, undefined);
          const col = _asColumn(pv, N);
          if (!col) {
            const shp = valueLib.isValue(pv) ? JSON.stringify(pv.shape) : typeof pv;
            throw new Error('markovchain: step ' + (jj + 1) + ' param \'' + pn
              + '\' resolved to ' + shp + ' (expected a scalar or [' + N + '])');
          }
          const rn = '__mc_p_' + pn;
          sampleRefs[rn] = valueLib.batchedScalar(col);
          distKwargs[pn] = { kind: 'ref', ns: 'self', name: rn };
        }
        return ctx.sendWorker({
          type: 'sampleN',
          ir: { kind: 'call', op: d.step.distOp, kwargs: distKwargs },
          count: N,
          refArrays: sampleRefs,
          seed: shared.nameSeed(name + ':mc' + jj, ctx.rootKey),
        }).then((reply: any) => { cols[jj] = reply.samples; });
      });
    }

    return acc.then(() => {
      // Atom-major: samples[i*n + j] is atom i's traj[j+1].
      const out = new Float64Array(N * n);
      for (let j = 0; j < n; j++) {
        const c = cols[j];
        for (let i = 0; i < N; i++) out[i * n + j] = c[i];
      }
      return empirical.arrayMeasure(out, [n], null);
    });
  });
}

/**
 * Located refusals for `markovchain` calls this engine will not lower.
 * Without this the classifier's `null` leaves the binding with no derivation
 * and no message, so a composite or record-state step kernel would fail
 * silently instead of saying what it is. Reads the same `bindings` map the
 * classifier reads, so the two cannot disagree about what is lowerable.
 */
function checkMarkovchain(bindings: any): any[] {
  const out: any[] = [];
  if (!bindings || !bindings.entries) return out;
  for (const [name, b] of bindings.entries()) {
    const ir = b && b.ir;
    if (!ir || ir.kind !== 'call' || ir.op !== 'markovchain') continue;
    // Arity is typeinfer's (it reports `markovchain expects 3`), so only the
    // step-kernel shape is judged here.
    const args = Array.isArray(ir.args) ? ir.args : [];
    if (args.length !== 3) continue;
    const step = describeStepKernel(args[0], bindings);
    if (step && step.reason) {
      out.push({
        name,
        severity: 'error',
        message: 'markovchain: ' + step.reason,
        loc: (args[0] && args[0].loc) || ir.loc,
      });
    }
  }
  return out;
}

module.exports = {
  describeStepKernel, stepDistIR, densityIR, matMarkovchain, checkMarkovchain,
};

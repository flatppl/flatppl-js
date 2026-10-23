'use strict';

// =====================================================================
// markovchain.ts — spec §06 `markovchain(kernel, init, n)` and
//                  spec §06 `kscan(kernel, init, xs)`
// Both accept an optional fourth input p, passed unchanged after the step's
// state (and exogenous input for kscan). The chain is Markov conditional on p.
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
//   1. A positional `jointchain` cannot express a prev-only step past two
//      components at all. §06 gives it the same lowering as kchain —
//      `c ~ K3([a, b])` — so step i binds the `cat` of every variate to its
//      left, which a prev-only step body cannot consume. That is now a
//      located error naming this op (density-prims `catBoundary`,
//      typeinfer `checkChainStepBodies`), and
//      `test/fixtures/hierarchical-state-space.flatppl` is spelled with
//      `markovchain` for exactly that reason.
//   2. `jointchain` requires ≥ 2 components, so it cannot spell n = 1,
//      which §06 admits (`n` is any positive integer).
//   3. A positional `jointchain` retaining all its variates materialises
//      to a TUPLE measure, and §06 says markovchain's variate is an ARRAY.
//      Whether a positional chain variate is a tuple or an array is an
//      unsettled cross-engine ruling (TODO-flatppl-js.md); §06 leaves
//      markovchain no such latitude, so inheriting the tuple would import
//      an open question into an op that does not have one.
//
// Each step receives the previous state and, when supplied, the same p.
// There is no concatenated history to build.
//
// SCOPE. The kernel must be a single-input kernel whose body is a
// sampleable scalar distribution (`fn(Normal(mu = _, sigma = s))`,
// `x -> Normal(x, s)`, `functionof(Normal(mu = _p_, sigma = s), p = _p_)`
// — `functionof`, not `kernelof`, because a distribution call is a measure
// and spec §04 §sec:kernelof bars a measure from `kernelof`),
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
//
// `kscan`, §06 verbatim: "`kernel` is a Markov kernel `(state, x) ->
// measure_over_state`; step $i$ is $\text{traj}_i \sim
// \kappa(\text{traj}_{i-1}, \text{xs}_i)$ with $\text{traj}_0 = \text{init}$.
// Trajectories have length `lengthof(xs)`. As with `markovchain`, `init` is a
// value in the state space and not part of the trajectory."
//
// So `kscan` is this same scan with a SECOND substitution at each step: the
// kernel's exogenous param takes `xs[i]`. Everything else — the array variate,
// the value-not-measure `init`, the n-term density with no base term — is
// markovchain's, which is why the two share every function here and differ
// only in the step kernel's arity, where the length comes from (`lengthof(xs)`
// rather than the literal `n`), and one extra entry in `stepDistIR`'s
// substitution map. `xs[i]` is spelled as the 1-based `get(xs, i)` on BOTH
// paths, so an exogenous column is the same value in the sampler and in the
// density.

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
 * The two §06 trajectory ops, by the only thing that differs in their step
 * kernel: its arity, and the arrow §06 spells for it.
 */
const TRAJECTORY_OPS: Record<string, { arity: number; arrow: string }> = {
  markovchain: { arity: 1, arrow: '`(state, ...) -> measure_over_state`' },
  kscan:       { arity: 1, arrow: '`(state, ...) -> measure_over_state`' },
};

/**
 * The static field IRs of a record (or table) argument, following a binding ref:
 * `record(a = .., b = ..)` gives `{ a, b }`. Null when the fields are not visible.
 */
function recordFieldIRs(ir: any, bindings: any, kind: 'record' | 'table'): Record<string, any> | null {
  const seen = new Set<string>();
  while (ir && ir.kind === 'ref' && ir.ns === 'self' && !seen.has(ir.name)) {
    seen.add(ir.name);
    ir = bindings?.get?.(ir.name)?.ir;
  }
  if (ir?.kind !== 'call' || ir.op !== kind || !Array.isArray(ir.fields)) return null;
  return Object.fromEntries(ir.fields.map((f: any) => [f.name, f.value]));
}

/**
 * §04 "Additional arguments": bind the arguments after the operation's own to the
 * kernel's remaining inputs, positionally or by name (a sole record, or for kscan a
 * sole table, splats its fields by name when two or more inputs remain). Returns the
 * IR bound to each remaining input, or a `{ reason }` refusal.
 */
function bindAdditionalIRs(op: string, ir: any, bindings: any, extras: { args: any[]; kwargs: Record<string, any> }):
  { bound: Record<string, any> } | { reason: string } {
  const params: string[] = ir.params;
  const rest = params.slice(1);
  const restNames: string[] = (ir.paramKwargs || params).slice(1);
  const bound: Record<string, any> = {};
  const byName = (entries: Array<[string, any]>) => {
    for (const [name, value] of entries) {
      const index = restNames.indexOf(name);
      if (index < 0) return { reason: 'the kernel has no input named "' + name + '" after its state' };
      bound[rest[index]] = value;
    }
    return null;
  };
  const named = Object.keys(extras.kwargs);
  let problem: { reason: string } | null = null;
  if (named.length) {
    if (extras.args.length) return { reason: 'additional arguments are all positional or all named, not both' };
    problem = byName(named.map((name): [string, any] => [name, extras.kwargs[name]]));
  } else if (extras.args.length === 1 && rest.length >= 2) {
    const fields = recordFieldIRs(extras.args[0], bindings, op === 'kscan' ? 'table' : 'record')
      || recordFieldIRs(extras.args[0], bindings, 'record');
    if (!fields) {
      return { reason: 'a sole record or table argument must have statically visible fields to bind '
        + 'the ' + rest.length + ' remaining inputs by name' };
    }
    problem = byName(Object.entries(fields));
  } else {
    if (extras.args.length !== rest.length) {
      return { reason: 'the kernel has ' + rest.length + ' inputs after its state, but ' + extras.args.length
        + ' additional arguments were given' };
    }
    extras.args.forEach((value, i) => { bound[rest[i]] = value; });
  }
  if (problem) return problem;
  const unbound = rest.filter(param => !(param in bound));
  if (unbound.length) return { reason: 'a kernel input is not bound by the additional arguments' };
  return { bound };
}

/**
 * The step descriptor for a `markovchain` / `kscan` kernel argument, or a
 * `{ reason }` refusal naming what the shape is instead.
 *
 * `compIR` is either a self-ref to a kernel binding or an inline
 * `functionof` (`fn` / `kernelof` / `->` all lower to one). `op` selects the
 * §06 arity: one input for `markovchain`, two for `kscan`, whose second is
 * the exogenous input.
 */
function describeStepKernel(compIR: any, bindings: any, op?: string,
  extras: { args: any[]; kwargs: Record<string, any> } = { args: [], kwargs: {} }): any {
  const opName = op || 'markovchain';
  const spec = TRAJECTORY_OPS[opName];
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
      reason: shownAs + ' is not a kernel — §06 ' + opName + ' takes a Markov '
        + 'kernel ' + spec.arrow + ' as its first argument',
    };
  }
  const params: string[] = Array.isArray(ir.params) ? ir.params : [];
  if (params.length < 1) {
    return { reason: shownAs + ' has no inputs; §06 ' + opName + ' needs the state input first' };
  }
  if (opName === 'kscan' && extras.args.length + Object.keys(extras.kwargs).length === 0) {
    return { reason: 'kscan needs at least one array or table input after `init` (§06)' };
  }
  const additional = bindAdditionalIRs(opName, ir, bindings, extras);
  if ('reason' in additional) {
    return { reason: shownAs + ': ' + additional.reason + ' (§04 "Additional arguments")' };
  }
  const dist = _peelLawof(ir.body);
  if (!dist || dist.kind !== 'call' || !dist.op
      || !SAMPLEABLE || !SAMPLEABLE.has(dist.op)) {
    return {
      reason: shownAs + '\'s body is not a sampleable distribution call; this '
        + 'engine implements ' + opName + ' over a scalar-distribution step '
        + 'kernel only (a composite or record-state step is not lowered)',
    };
  }
  const distParams = _distParamsOf(dist.op);
  // §05 lets a distribution take its parameters positionally
  // (`Normal(x, sqrt(2*D*dt))`, §06's markovchain and kscan examples both). Name them
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
  const rest = params.slice(1);
  // `markovchain` passes the additional arguments whole: substitute them now. `kscan`
  // traverses them as sequences: its descriptor lists them for the per-step element.
  if (opName === 'markovchain' && rest.length) {
    const substitute = require('./mat-broadcast.ts').substituteKernelParams;
    for (const name of Object.keys(distKwargs)) {
      distKwargs[name] = substitute(distKwargs[name], rest, rest, additional.bound);
    }
  }
  return {
    inputParam: params[0],
    xParams: opName === 'kscan' ? rest : [],
    xSequences: opName === 'kscan' ? rest.map(param => ({ param, ir: additional.bound[param] })) : [],
    distOp: dist.op,
    distParams,
    distKwargs,
  };
}

/**
 * Step j's distribution IR, with the kernel's state param replaced by
 * `prevIR` and — for a `kscan` step — its exogenous param replaced by `xIR`.
 * The single producer both the sampler and the density walker read, so step j
 * is the same distribution on both paths.
 */
function stepDistIR(step: any, prevIR: any, xIRs?: Record<string, any>): any {
  const substituteKernelParams = require('./mat-broadcast.ts').substituteKernelParams;
  const kwargs: Record<string, any> = {};
  const params: string[] = [step.inputParam];
  const subMap: Record<string, any> = {};
  subMap[step.inputParam] = prevIR;
  for (const param of step.xParams) {
    params.push(param);
    subMap[param] = xIRs![param];
  }
  for (const pn of Object.keys(step.distKwargs)) {
    kwargs[pn] = substituteKernelParams(
      step.distKwargs[pn], params, params, subMap);
  }
  return { kind: 'call', op: step.distOp, kwargs };
}

/**
 * The per-step elements of a kscan's sequences, by input: step `j` reads element
 * `j + 1`, or element 1 of a length-one sequence, which repeats (§04). `index` is
 * the IR of the 1-based step index.
 */
function sequenceElementIRs(d: any, index: any): Record<string, any> {
  const out: Record<string, any> = {};
  for (const sequence of d.xs || []) {
    out[sequence.param] = { kind: 'call', op: 'get',
      args: [sequence.ir, sequence.length === 1 ? { kind: 'lit', value: 1 } : index] };
  }
  return out;
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
    args.push(stepDistIR(d.step, prevIR, sequenceElementIRs(d, { kind: 'lit', value: j + 1 })));
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
  const op = d.kind;

  // One aggregate node carrying every free expression (the init value, `xs`
  // where there is one, and each step kwarg) so per-atom refs resolve in a
  // single pass.
  const aggregate: any = {
    kind: 'call', op: 'joint',
    args: [d.initIR].concat((d.xs || []).map((sequence: any) => sequence.ir)).concat(
      Object.keys(d.step.distKwargs).map((pn: string) => d.step.distKwargs[pn])),
  };

  return shared.prepareDensityRefs(aggregate, ctx, op).then((prep: any) => {
    const { refArrays, fixedEnv } = prep;
    const initVal = sampler.evaluateExprN(d.initIR, refArrays, N, fixedEnv, undefined);
    const initCol = _asColumn(initVal, N);
    if (!initCol) {
      const shp = valueLib.isValue(initVal) ? JSON.stringify(initVal.shape) : typeof initVal;
      return Promise.reject(new Error(op + ': `init` resolved to ' + shp
        + ' (expected a scalar or one value per atom) — §06 makes `init` a '
        + 'value in the state space, and this engine lowers a scalar state'));
    }

    // Keep step IR stable so evaluateExprN reuses its compiled plans. Only
    // the carry and kscan index change; neither belongs in the IR cache key.
    const stepRefs: Record<string, any> = Object.assign({}, refArrays);
    const stepEnv = Object.assign({}, fixedEnv);
    let indexName = '__mc_index';
    while (indexName in stepRefs || indexName in stepEnv) indexName += '_';
    const ir = stepDistIR(d.step, { kind: 'ref', ns: 'self', name: '__mc_prev' },
      sequenceElementIRs(d, { kind: 'ref', ns: 'self', name: indexName }));
    const cols: Float64Array[] = new Array(n);
    let acc: Promise<any> = Promise.resolve();
    for (let j = 0; j < n; j++) {
      const jj = j;
      acc = acc.then(() => {
        const prev = (jj === 0) ? initCol : cols[jj - 1];
        stepRefs.__mc_prev = valueLib.batchedScalar(prev);
        stepEnv[indexName] = jj + 1;
        const distKwargs: Record<string, any> = {};
        const sampleRefs: Record<string, any> = {};
        for (const pn of Object.keys(ir.kwargs)) {
          const pv = sampler.evaluateExprN(ir.kwargs[pn], stepRefs, N, stepEnv, undefined);
          const col = _asColumn(pv, N);
          if (!col) {
            const shp = valueLib.isValue(pv) ? JSON.stringify(pv.shape) : typeof pv;
            throw new Error(op + ': step ' + (jj + 1) + ' param \'' + pn
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
 * `lengthof(xs)` — §06 kscan's trajectory length — as a positive integer, or
 * null when it does not resolve before materialisation.
 *
 * An inline array literal lowers to `vector(…)`, whose argument count IS the
 * length. Otherwise `resolveConstant`'s `lengthof` arm reads the pre-evaluated
 * fixed-phase value, which is how a data-loaded or computed `xs` resolves.
 * THE one length reader, shared by `classifyKscan` and the refusal check, so a
 * length that does not resolve is a message rather than a silent non-lowering.
 */
function resolveXsLength(xsIR: any, bindings: any, fixedValues?: any): number | null {
  const seen = new Set<string>();
  while (xsIR?.kind === 'ref' && xsIR.ns === 'self' && !seen.has(xsIR.name)) {
    seen.add(xsIR.name);
    const binding = bindings?.get(xsIR.name);
    const size = binding?.inferredType?.shape?.[0];
    if (Number.isInteger(size) && size > 0) return size;
    if (!binding?.ir) break;
    xsIR = binding.ir;
  }
  if (xsIR && xsIR.kind === 'call' && xsIR.op === 'vector' && Array.isArray(xsIR.args)) {
    return xsIR.args.length > 0 ? xsIR.args.length : null;
  }
  // A table's length is its columns' common length; read the first column's.
  if (xsIR?.kind === 'call' && xsIR.op === 'table' && Array.isArray(xsIR.fields) && xsIR.fields.length) {
    return resolveXsLength(xsIR.fields[0].value, bindings, fixedValues);
  }
  const resolveConstant = require('./ir-shared.ts').resolveConstant;
  if (xsIR?.kind === 'call' && xsIR.op === 'fill' && xsIR.args?.length === 2) {
    const size = resolveConstant(xsIR.args[1], bindings, new Set(), fixedValues);
    if (Number.isInteger(size) && size > 0) return size;
  }
  const n = resolveConstant(
    { kind: 'call', op: 'lengthof', args: [xsIR] },
    bindings || new Map(), new Set(), fixedValues);
  return (Number.isInteger(n) && n > 0) ? n : null;
}

/**
 * Located refusals for `markovchain` / `kscan` calls this engine will not
 * lower. Without this the classifier's `null` leaves the binding with no
 * derivation and no message, so a composite or record-state step kernel would
 * fail silently instead of saying what it is. Reads the same `bindings` map
 * and the same readers the classifiers read, so the two cannot disagree about
 * what is lowerable.
 */
function checkTrajectoryKernels(bindings: any, fixedValues?: any): any[] {
  const out: any[] = [];
  if (!bindings || !bindings.entries) return out;
  for (const [name, b] of bindings.entries()) {
    const ir = b && b.ir;
    if (!ir || ir.kind !== 'call' || !ir.op
        || !Object.prototype.hasOwnProperty.call(TRAJECTORY_OPS, ir.op)) continue;
    const op = ir.op;
    // Arity is typeinfer's, so only the step-kernel shape and kscan's sequence lengths
    // are judged here.
    const args = Array.isArray(ir.args) ? ir.args : [];
    const leading = op === 'markovchain' ? 3 : 2;
    if (args.length < leading) continue;
    const extras = { args: args.slice(leading), kwargs: ir.kwargs || {} };
    const step = describeStepKernel(args[0], bindings, op, extras);
    if (step && step.reason) {
      out.push({
        name,
        severity: 'error',
        message: op + ': ' + step.reason,
        loc: (args[0] && args[0].loc) || ir.loc,
      });
      continue;
    }
    if (op === 'kscan') {
      const lengths = step.xSequences.map((sequence: any) => resolveXsLength(sequence.ir, bindings, fixedValues));
      if (lengths.some((n: any) => n == null)) {
        out.push({
          name,
          severity: 'error',
          message: 'kscan: a sequence length does not resolve before sampling — §06 '
            + 'gives the trajectory length from the sequences, so each must be an '
            + 'array literal or a fixed-phase array value',
          loc: ir.loc,
        });
      } else if (new Set(lengths.filter((n: number) => n !== 1)).size > 1) {
        out.push({
          name,
          severity: 'error',
          message: 'kscan: the sequence inputs have unequal lengths ' + lengths.join(', ')
            + '; only length-one inputs repeat (spec §04)',
          loc: ir.loc,
        });
      }
    }
  }
  return out;
}

module.exports = {
  describeStepKernel, stepDistIR, densityIR, matMarkovchain,
  checkTrajectoryKernels, resolveXsLength,
};

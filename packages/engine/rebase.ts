'use strict';

// Spec §06 rebase: retain the density expression and the explicit reference
// measure until scoring/sampling. In particular, an inferred Normal scale must
// never be replaced by a normalizer evaluated at a materialised parameter draw.

const call = (op: string, args: any[]) => ({ kind: 'call', op, args });

/** Recognise the two exact iid-Normal vector laws required by §06. */
function normalVector(ir: any, resolveValue?: (ir: any) => any): any {
  if (ir?.op !== 'rebase' || ir.args?.length !== 2) return null;
  const [m, b] = ir.args;
  if (m?.op !== 'iid' || m.args?.length !== 2 || m.args[0]?.op !== 'Normal'
      || b?.op !== 'Lebesgue') return null;
  const fixed = (x: any) => x?.kind === 'lit' ? x.value : resolveValue?.(x);
  const n = fixed(m.args[1]);
  const support = b.kwargs?.support || b.args?.[0];
  const set = b.vectorSet?.set || support?.op;
  const size = b.vectorSet?.n ?? fixed(support?.args?.[0]);
  if (!Number.isInteger(n) || n < 1 || n !== size
      || (set !== 'ordered' && set !== 'zerosum')) return null;
  const normal = m.args[0];
  const mu = normal.kwargs?.mu || normal.args?.[0];
  const sigma = normal.kwargs?.sigma || normal.args?.[1];
  return mu && sigma ? { n, set, mu, sigma } : null;
}

/** Defining measure algebra for cases served by the existing normalizer. */
function definition(ir: any): any {
  const point = { kind: 'ref', ns: '%local', name: '%rebase_variate' };
  const weight = {
    kind: 'call', op: 'functionof', params: [point.name], paramKwargs: [point.name],
    body: call('logdensityof', [ir.args[0], point]),
  };
  return call('normalize', [call('logweighted', [weight, ir.args[1]])]);
}

/** Validate the source law even where its common mean cancels algebraically. */
function checkParameters(mu: number, sigma: number) {
  if (!Number.isFinite(mu) || !Number.isFinite(sigma) || !(sigma > 0)) {
    throw new Error('rebase: exact Normal laws require finite mu and finite sigma > 0');
  }
}

/** Sort or center iid Normal draws; centering avoids cancellation in the mean. */
function transform(x: Float64Array, set: string) {
  if (set === 'ordered') x.sort();
  else {
    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    for (let j = 0; j < x.length; j++) x[j] -= mean;
  }
}

module.exports = { normalVector, definition, checkParameters, transform };

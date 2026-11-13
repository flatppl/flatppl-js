'use strict';

// §04 "Interface adaptation": a relabelled vector remains one joint prior.
// Adapt its public fields to the existing vector inference path, then restore
// those fields after sampling. The vector's chart and prior transform stay intact.
function adaptRelabelledPrior(d: any, ctx: any): any {
  let name = d.from;
  const seen = new Set<string>();
  while (!seen.has(name) && ctx.bindings.get(name)?.ir?.kind === 'ref'
      && ctx.derivations[name]?.kind === 'alias') {
    seen.add(name);
    name = ctx.derivations[name].from;
  }
  const relabel = ctx.derivations[name];
  if (relabel?.kind !== 'relabel') return null;
  // Existing draw-based laws own their latent coordinates. This adapter adds
  // only the missing constructor-prior case, without replacing that mapping.
  if (require('./model-spec.ts').priorLatentSources(d, ctx).length) return null;

  const vectorName = '%relabel:vector', priorName = '%relabel:prior';
  const { names, from } = relabel;
  const bindings = new Map(ctx.bindings);
  const fields = { [vectorName]: from };
  bindings.set(priorName, { name: priorName, type: 'joint', synthetic: true,
    ir: { kind: 'call', op: 'joint', fields: [
      { name: vectorName, value: { kind: 'ref', ns: 'self', name: from } },
    ] } });
  const derivations = { ...ctx.derivations, [priorName]: { kind: 'record', fields } };
  const empirical = require('./empirical.ts');
  const shared = require('./materialiser-shared.ts');
  let prior: Promise<any> | undefined;
  const child = { ...ctx, bindings, derivations, getMeasure(n: string) {
    if (n !== priorName) return ctx.getMeasure(n);
    return prior || (prior = ctx.getMeasure(from).then((m: any) => Object.assign(
      empirical.recordMeasure({ [vectorName]: m }, m.logWeights), {
        logTotalmass: shared.massOf(m), n_eff: m.n_eff,
      })));
  } };

  // Expand under the original boundary cut before changing coordinates. CLM's
  // argument substitution also rewrites captured bijections and normalizers.
  const node = require('./clm.ts').lowerMeasure(d.bodyIR || d.bodyName, ctx, { derivation: d });
  if (!node) throw new Error('relabel prior: cannot lower the likelihood body');
  const boundaryArgs: Record<string, any> = {};
  for (let i = 0; i < d.paramKwargs.length; i++) {
    const index = names.indexOf(d.paramKwargs[i]);
    if (index < 0) throw new Error(`relabel prior: no field '${d.paramKwargs[i]}'`);
    boundaryArgs[d.params[i]] = { kind: 'call', op: 'get', args: [
      { kind: 'ref', ns: 'self', name: vectorName },
      { kind: 'lit', value: index + 1, numType: 'integer' },
    ] };
  }
  const adapted = { ...d, from: priorName, bodyIR: node.body, bodyName: null,
    params: [vectorName], paramKwargs: [vectorName], boundaryArgs };

  function restore(result: any): any {
    const m = require('./mat-transformations.ts').relabelMeasure(result.fields[vectorName], names);
    const diagnosticNames = new Map(names.map((label: string, i: number) => [`${vectorName}[${i}]`, label]));
    let diagnostics = result.diagnostics;
    if (diagnostics) {
      diagnostics = { ...diagnostics };
      for (const key of ['perParam', 'rawChains']) {
        if (diagnostics[key]) diagnostics[key] = Object.fromEntries(Object.entries(diagnostics[key])
          .map(([key, value]) => [diagnosticNames.get(key) || key, value]));
      }
      for (const field of Object.values(m.fields) as any[]) field.diagnostics = diagnostics;
    }
    return Object.assign(m, { logWeights: result.logWeights, logTotalmass: shared.massOf(result),
      n_eff: result.n_eff, diagnostics });
  }
  return { derivation: adapted, ctx: child, restore };
}

module.exports = { adaptRelabelledPrior };

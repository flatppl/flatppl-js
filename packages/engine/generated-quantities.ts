// @flatppl/engine — generated quantities (Stan-style).
//
// Evaluate a deterministic binding's RHS samplewise over a record measure's
// per-field sample columns, producing a derived column that inherits the
// measure's importance weights. Pure post-processing — no pushfwd / density
// machinery; reuses the sampler's vectorized evaluateExprN.

// sampler.ts re-exports evaluateExprN and calls initARITHOPSN at load time,
// so ARITH_OPS_N is populated when evaluateExprN is first invoked.
const sampler = require('./sampler.ts');

/** N atoms in a record measure (first field's sample length). */
function atomCount(recordMeasure: any): number {
  const fields = recordMeasure && recordMeasure.fields;
  if (!fields) return 0;
  const k = Object.keys(fields)[0];
  return k ? (fields[k].samples ? fields[k].samples.length : 0) : 0;
}

/** Evaluate `ir` (a deterministic binding RHS referencing record field names
 *  as ns:'self' refs) samplewise over `recordMeasure`'s field columns. */
function deriveColumn(ir: any, recordMeasure: any, baseEnv: any): Float64Array {
  const fields = recordMeasure.fields || {};
  const N = atomCount(recordMeasure);
  const refArrays: Record<string, any> = {};
  for (const name in fields) {
    if (fields[name] && fields[name].samples) refArrays[name] = fields[name].samples;
  }
  const out = sampler.evaluateExprN(ir, refArrays, N, baseEnv || {}, null);
  return out;
}

/** Return a shallow-cloned record measure with each spec appended as a field
 *  carrying the measure's logWeights. specs: Array<{name, ir}>. */
function appendGeneratedQuantities(recordMeasure: any, specs: any[], baseEnv: any): any {
  if (!specs || specs.length === 0) return recordMeasure;
  const lw = recordMeasure.logWeights || null;
  const fields = Object.assign({}, recordMeasure.fields);
  for (let i = 0; i < specs.length; i++) {
    const s = specs[i];
    const col = deriveColumn(s.ir, recordMeasure, baseEnv);
    fields[s.name] = { samples: col, logWeights: lw };
  }
  return Object.assign({}, recordMeasure, { fields: fields });
}

module.exports = { deriveColumn, appendGeneratedQuantities };

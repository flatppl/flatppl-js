// @flatppl/viewer — detect model deterministic bindings that qualify as
// generated quantities for a focused posterior record: kind 'evaluate'
// bindings whose self-refs are all variates of the record.

declare const FlatPPLEngine: any;

/** Collect ns:'self' ref names in an IR using a local recursive walk.
 *  ir-shared.ts exports collectSelfRefs but it is an engine-internal
 *  symbol not exposed through the viewer bundle; we use a local walk
 *  that covers the same IR child positions as forEachIRChild in ir-walk.ts. */
function selfRefs(ir: any, acc: Set<string>): Set<string> {
  if (!ir || typeof ir !== 'object') return acc;
  if (ir.kind === 'ref' && ir.ns === 'self' && ir.name) acc.add(ir.name);
  if (ir.callee) selfRefs(ir.callee, acc);
  if (Array.isArray(ir.args)) ir.args.forEach((a: any) => selfRefs(a, acc));
  if (ir.kwargs) for (const k in ir.kwargs) selfRefs(ir.kwargs[k], acc);
  if (Array.isArray(ir.fields)) ir.fields.forEach((f: any) => { if (f && f.value) selfRefs(f.value, acc); });
  if (Array.isArray(ir.assigns)) ir.assigns.forEach((a: any) => { if (a && a.value) selfRefs(a.value, acc); });
  if (ir.body) selfRefs(ir.body, acc);
  if (Array.isArray(ir.branches)) ir.branches.forEach((b: any) => {
    if (!b || typeof b !== 'object') return;
    if (b.ir) selfRefs(b.ir, acc);
    else if (b.kind) selfRefs(b, acc);
  });
  if (ir.selector) selfRefs(ir.selector, acc);
  if (Array.isArray(ir.logweights)) ir.logweights.forEach((w: any) => { if (w) selfRefs(w, acc); });
  return acc;
}

/** Qualifying deterministic bindings for a focused posterior record.
 *  Returns derivations of kind 'evaluate' whose ns:'self' refs each resolve
 *  to either a field of `recordMeasure` (a posterior variate) or a fixed-phase
 *  constant (resolvable via fixedValues — evaluateExprN gets it through the
 *  baseEnv), with at least one variate ref (a binding of constants alone is not
 *  a function of the posterior). Excludes bindings that are themselves a field. */
export function detectGeneratedQuantities(ctx: any, recordMeasure: any): Array<{ name: string; ir: any }> {
  const ds = ctx.derivationsState;
  const derivs = ds && ds.derivations;
  if (!derivs || !recordMeasure || !recordMeasure.fields) return [];
  const variates = new Set(Object.keys(recordMeasure.fields));
  const fixedValues = ds && ds.fixedValues;
  function isFixedConst(r: string): boolean {
    try { return !!(fixedValues && typeof fixedValues.has === 'function' && fixedValues.has(r)); }
    catch (_) { return false; }
  }
  const out: Array<{ name: string; ir: any }> = [];
  for (const name in derivs) {
    const d = derivs[name];
    if (!d || d.kind !== 'evaluate' || !d.ir) continue;  // deterministic only
    if (variates.has(name)) continue;                     // already a field
    const refs = selfRefs(d.ir, new Set<string>());
    if (refs.size === 0) continue;                        // not a function of anything
    let ok = true;
    let anyVariate = false;
    refs.forEach((r) => {
      if (variates.has(r)) anyVariate = true;
      else if (!isFixedConst(r)) ok = false;              // a non-variate, non-constant ref disqualifies
    });
    if (ok && anyVariate) out.push({ name, ir: d.ir });   // ≥1 variate + every other ref a fixed constant
  }
  return out;
}

/** Build the plain `{name: value}` baseEnv that `evaluateExprN` needs for the
 *  fixed-phase constants referenced by `specs` (the non-variate refs). The
 *  engine resolves a baseEnv ref via `name in baseEnv`, so it must be a plain
 *  object — NOT the FixedValues resolver (whose names live behind `.get`/`.has`,
 *  not as own-properties). Variate refs come from the per-atom sample columns
 *  instead and are omitted here. */
export function fixedEnvFor(ctx: any, recordMeasure: any, specs: Array<{ name: string; ir: any }>): Record<string, any> {
  const env: Record<string, any> = {};
  const ds = ctx.derivationsState;
  const fixedValues = ds && ds.fixedValues;
  if (!fixedValues || typeof fixedValues.get !== 'function' || !recordMeasure || !recordMeasure.fields) return env;
  const variates = new Set(Object.keys(recordMeasure.fields));
  for (let i = 0; i < specs.length; i++) {
    const refs = selfRefs(specs[i].ir, new Set<string>());
    refs.forEach((r) => {
      if (variates.has(r) || (r in env)) return;          // variate → per-atom column; already done
      try { if (fixedValues.has(r)) env[r] = fixedValues.get(r); } catch (_) { /* leave unresolved → eval guarded */ }
    });
  }
  return env;
}

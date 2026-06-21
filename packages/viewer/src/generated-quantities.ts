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
 *  Returns derivations of kind 'evaluate' whose ns:'self' refs are
 *  all field names of `recordMeasure`, and which are not themselves
 *  already fields of the record. */
export function detectGeneratedQuantities(ctx: any, recordMeasure: any): Array<{ name: string; ir: any }> {
  const derivs = ctx.derivationsState && ctx.derivationsState.derivations;
  if (!derivs || !recordMeasure || !recordMeasure.fields) return [];
  const variates = new Set(Object.keys(recordMeasure.fields));
  const out: Array<{ name: string; ir: any }> = [];
  for (const name in derivs) {
    const d = derivs[name];
    if (!d || d.kind !== 'evaluate' || !d.ir) continue;  // deterministic only
    if (variates.has(name)) continue;                     // already a field
    const refs = selfRefs(d.ir, new Set<string>());
    if (refs.size === 0) continue;                        // not a function of variates
    let ok = true;
    refs.forEach((r) => { if (!variates.has(r)) ok = false; });
    if (ok) out.push({ name, ir: d.ir });
  }
  return out;
}

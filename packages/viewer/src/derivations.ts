// @flatppl/viewer — derivation rebuild —
//
// rebuildDerivations parses the current source via
// FlatPPLEngine.orchestrator, builds the derivations map + fixed
// values, reconciles the override stores against the new bindings,
// and primes the cytoscape DAG's element list. Called on every
// sourceUpdate and configUpdate.

import { sendWorkerNow } from './worker.js';
import { ensureSamplerWorker } from './worker.js';
import type { Ctx } from './types';
export function rebuildDerivations(ctx: Ctx) {
  if (!ctx.currentBindings) {
    ctx.derivationsState = null;
    ctx.measureCache = new Map();
    ctx.histogramCache = new Map();
    ctx.profileRangeCache = new Map();
    return;
  }
  try {
    ctx.derivationsState = FlatPPLEngine.orchestrator.buildDerivations(ctx.currentBindings);
    // Thread the module registry from the lowered module into
    // derivationsState so the cross-module call dispatch path
    // (sampler.ts's `_evaluateStandardModuleCall`) finds it via
    // the setEnv push below. `currentLoweredModule.moduleRegistry`
    // is populated by `pir.lowerToModule` for every `standard
    // _module(name, compat)` binding.
    if (ctx.derivationsState && ctx.currentLoweredModule
        && ctx.currentLoweredModule.moduleRegistry) {
      ctx.derivationsState.moduleRegistry = ctx.currentLoweredModule.moduleRegistry;
    }
    // Surface classification diagnostics instead of letting a
    // silently-dropped binding turn into a confusing plot-time
    // error far from its cause. buildDerivations only emits the
    // unambiguous fixed-phase-dead-end case (an engine gap on a
    // deterministic expression), so any entry here is actionable.
    const bdiags = (ctx.derivationsState && ctx.derivationsState.diagnostics) || [];
    for (let bi = 0; bi < bdiags.length; bi++) {
      console.warn('FlatPPL: ' + bdiags[bi].message);
    }
  } catch (e) {
    console.error('FlatPPL: buildDerivations failed:', e);
    ctx.derivationsState = null;
  }
  // Source change invalidates every cached measure — derivations
  // (or just signatures) may have shifted under any of them. Drop
  // the histogram cache too since histograms are downstream of
  // measures.
  ctx.measureCache = new Map();
  ctx.histogramCache = new Map();
  ctx.profileRangeCache = new Map();

  // Reconcile module-wide preset overrides against the new
  // source. For each existing override:
  //   - If the preset binding is gone, drop the override.
  //   - For each kwarg in override.values, if the new source
  //     value matches the override (or the kwarg is gone),
  //     drop it from the override (matched → redundant,
  //     gone → not applicable). Persist-button writes lean on
  //     this — after writing the override into the source,
  //     the next rebuildDerivations finds equal values and
  //     prunes the override automatically.
  //   - If the override is empty after pruning, retire it.
  ctx.presetOverrides.forEach(function(entry: any, name: any) {
    const b = ctx.currentBindings!.get(name);
    let curValues: Record<string, number> | null = null;
    // analyzer.classifyStatement only returns 'literal' for
    // primitive literal RHS (NumberLiteral etc.); record(...)
    // binds with type='call'. So gate on the AST callee name
    // alone — that's the structural property we actually need.
    if (b && b.node && b.node.value
        && b.node.value.type === 'CallExpr' && b.node.value.callee
        && b.node.value.callee.name === 'record') {
      // Best-effort literal extraction; bail to no-match if
      // anything looks non-trivial (which falls into the
      // "kwarg unknown" branch below). A NumberLiteral wrapped
      // in fixed(...) (spec §03: "held constant during
      // optimization" hint) is identity at runtime, so we look
      // through the wrapper to read the underlying number —
      // otherwise persist-time reconciliation would not see the
      // override's value matching source and would fail to prune.
      curValues = {};
      const args = b.node.value.args || [];
      for (let ai = 0; ai < args.length; ai++) {
        const arg = args[ai];
        if (arg.type !== 'KeywordArg' || !arg.value) continue;
        let v = arg.value;
        if (v.type === 'CallExpr' && v.callee && v.callee.name === 'fixed'
            && Array.isArray(v.args) && v.args.length === 1) {
          v = v.args[0];
        }
        if (v && v.type === 'NumberLiteral') {
          curValues[arg.name] = v.value;
        }
      }
    }
    if (!curValues) {
      ctx.presetOverrides.delete(name);
      return;
    }
    const vs = entry.values || {};
    for (const k in vs) {
      if (!Object.prototype.hasOwnProperty.call(vs, k)) continue;
      if (!Object.prototype.hasOwnProperty.call(curValues, k)) {
        delete vs[k];                          // kwarg gone
      } else if (vs[k] === curValues[k]) {
        delete vs[k];                          // override matches source
      }
    }
    if (Object.keys(vs).length === 0) {
      ctx.presetOverrides.delete(name);
    }
  });

  // Reconcile module-wide domain overrides against the new
  // source. Mirrors the preset-override loop above:
  //   - If the cartprod binding is gone, drop the override.
  //   - sourceKwargs tracks every kwarg the source's cartprod
  //     mentions, regardless of whether its value is an interval
  //     or a bare named set; an override on a kwarg the source
  //     doesn't mention at all is "kwarg gone" → drop it.
  //   - sourceIntervals[k] is set only when the source field is
  //     interval(lit, lit). When set and the override's [lo,hi]
  //     equals it, the override is redundant → drop it. When the
  //     source field is a bare named set (no bounds) we leave
  //     the override in place: the user's range overrides the
  //     unbounded source.
  //   - If the override is empty after pruning, retire it.
  ctx.domainOverrides.forEach(function(entry: any, name: any) {
    const b = ctx.currentBindings!.get(name);
    let sourceKwargs: Set<string> | null = null;
    let sourceIntervals: Record<string, { lo: number; hi: number }> | null = null;
    if (b && b.node && b.node.value
        && b.node.value.type === 'CallExpr' && b.node.value.callee
        && b.node.value.callee.name === 'cartprod') {
      sourceKwargs = new Set();
      sourceIntervals = {};
      const args = b.node.value.args || [];
      for (let ai = 0; ai < args.length; ai++) {
        const arg = args[ai];
        if (arg.type !== 'KeywordArg' || !arg.value) continue;
        sourceKwargs.add(arg.name);
        const ic = arg.value;
        if (ic.type === 'CallExpr' && ic.callee
            && ic.callee.name === 'interval'
            && Array.isArray(ic.args) && ic.args.length === 2
            && ic.args[0].type === 'NumberLiteral'
            && ic.args[1].type === 'NumberLiteral') {
          sourceIntervals[arg.name] = {
            lo: ic.args[0].value, hi: ic.args[1].value,
          };
        }
      }
    }
    if (!sourceKwargs) {
      ctx.domainOverrides.delete(name);
      return;
    }
    const rs = entry.ranges || {};
    for (const k in rs) {
      if (!Object.prototype.hasOwnProperty.call(rs, k)) continue;
      if (!sourceKwargs.has(k)) {
        delete rs[k];                                // kwarg gone
      } else if (sourceIntervals && Object.prototype.hasOwnProperty.call(sourceIntervals, k)
                 && rs[k].lo === sourceIntervals[k].lo
                 && rs[k].hi === sourceIntervals[k].hi) {
        delete rs[k];                                // matches source interval
      }
      // Otherwise: source uses a bare named set, the override's
      // explicit range still adds information — keep it.
    }
    if (Object.keys(rs).length === 0) {
      ctx.domainOverrides.delete(name);
    }
  });

  // Push fixed-phase pre-evaluated values into the worker's
  // session env. The orchestrator computed these once at module-
  // build time (rnginit / rand results, fixed scalar reductions,
  // etc.); the worker resolves refs to them via env rather than
  // through per-atom refArrays — the only correct semantics for
  // non-scalar fixed values like a length-10 `random_data` array.
  // setEnv with merge=false replaces (so a stale fixedValues map
  // from the previous source can't leak into the new one).
  //
  // Module registry threaded under `__moduleRegistry` for
  // cross-module call dispatch (sampler.ts's
  // `_evaluateStandardModuleCall`). One push per derivations
  // rebuild — replaces the previous module registry on every
  // source-change so a removed `standard_module(...)` binding
  // can't leave stale entries behind.
  const fixedValues = ctx.derivationsState && ctx.derivationsState.fixedValues;
  const moduleRegistry = ctx.derivationsState && ctx.derivationsState.moduleRegistry;
  const hasFV = fixedValues && fixedValues.size > 0;
  const hasMR = moduleRegistry && Object.keys(moduleRegistry).length > 0;
  if (hasFV || hasMR) {
    const envObj: Record<string, any> = {};
    if (hasFV) fixedValues.forEach(function(v: any, k: any) { envObj[k] = v; });
    if (hasMR) envObj.__moduleRegistry = moduleRegistry;
    ensureSamplerWorker(ctx).then(function(w: any) {
      sendWorkerNow(ctx, w, { type: 'setEnv', env: envObj, merge: false });
    }).catch(function(err: any) {
      console.error('FlatPPL: setEnv push failed:', err);
    });
  }
}

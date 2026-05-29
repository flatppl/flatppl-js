'use strict';

// =====================================================================
// alias-resolution.ts — post-lower IR rewrite pass that canonicalises
// references through alias bindings (engine-concepts §19 callable-layer;
// spec §04 "Aliasing is just assignment ... a reference to the same
// underlying object in the loaded module's DAG, not a clone").
// =====================================================================
//
// **Background.** FlatPPL bindings of the shape
//
//   breit_wigner = hepphys.breit_wigner_lineshape    % alias to a std-module callable
//   sig_model    = sig_module.model                  % alias to a loaded-module binding
//   theta        = some_param                        % local rename
//
// are pure aliases per spec §04 — they introduce a new name for the
// SAME underlying object. Their lowered IR has a single-ref RHS:
//
//   { kind: 'ref', ns: <hepphys | self | …>, name: <breit_wigner_lineshape | model | …> }
//
// **Architectural choice (matches Rust HIR / LLVM @alias / MLIR symbol
// resolution).** Resolve aliases to canonical form ONCE here, at IR
// time, then leave the canonical IR alone. Every downstream consumer
// (typeinfer, derivations, materialiser, sampler) sees one ref shape
// per object — no symbol-table dance, no per-pass alias-chain walking.
//
// Aliases survive in the LoweredModule's binding map (the viewer's DAG
// renders them as 'alias' nodes; signature presets reach them by
// name) but every IR-INTERNAL `(%ref self <alias>)` rewrites to the
// canonical ref. The non-alias bindings are tagged `isAlias: true` so
// downstream consumers that iterate bindings can skip them when
// appropriate.
//
// **Why not "follow aliases in every consumer"?** Two costs:
//   1. Every consumer's ref-resolution code path grows an alias-chain
//      walk + cycle guard. Drift surface scales linearly with consumers.
//   2. Pattern-rewriters and cross-engine FlatPIR emitters would need
//      the chain walk too — and they consume the IR as data, not via
//      our symbol-table API.
// Canonicalising once at IR time keeps the IR self-contained.
//
// **What counts as an alias.** A binding's RHS is a SINGLE ref node:
//
//   { kind: 'ref', ns, name, loc? }
//
// Anything else — calls, lits, consts, holes, reified bodies — is not
// an alias. References to `%local` (only valid inside reified scope)
// can't appear at top level, so the canonicalised RHS must have
// `ns ∈ {'self', <module-alias>}`.

const { mapIR } = require('./ir-walk.ts');

/**
 * Recognise the IR shape of an alias binding. Returns the RHS ref node
 * when `rhs` is a pure ref (the alias's canonical target); otherwise
 * returns null.
 */
function _aliasTarget(rhs: any): any | null {
  if (rhs && rhs.kind === 'ref' && typeof rhs.name === 'string'
      && (rhs.ns === 'self' || (typeof rhs.ns === 'string' && rhs.ns !== '%local'))) {
    return rhs;
  }
  return null;
}

/**
 * Walk an alias chain to its canonical ref. `aliases` is the map of
 * local-alias-name → immediate target ref. Returns the canonical
 * (non-self-alias) ref, or null if the chain is cyclic or escapes
 * into a missing binding.
 *
 * Examples:
 *   B = (%ref self A); A = (%ref hepphys X)         → resolve(B) = (%ref hepphys X)
 *   B = (%ref self A); A = (%ref self C); C = lit 3 → resolve(B) = (%ref self C)   [C is not an alias]
 *   B = (%ref self B)                                → resolve(B) = null            [cycle]
 *   B = (%ref hepphys X)                             → resolve(B) = (%ref hepphys X) [already canonical]
 */
function _canonicalise(target: any, aliases: Map<string, any>): any | null {
  const seen = new Set<string>();
  let cur = target;
  while (cur && cur.ns === 'self' && aliases.has(cur.name)) {
    if (seen.has(cur.name)) return null;
    seen.add(cur.name);
    cur = aliases.get(cur.name);
  }
  return cur;
}

/**
 * Run alias resolution over a Map<name, {<irField>: IR, isAlias?: bool}>
 * — the structural shape shared by LoweredModule (irField='rhs') AND
 * the analyzer's lift-output bindings (irField='ir'). Both forms need
 * canonicalisation: LoweredModule.bindings.X.rhs flows into typeinfer
 * + the materialiser; the analyzer's bindings.get(name).ir flows into
 * derivation classifiers + signatureOf + the sampler. The two are
 * derived from the same source AST but live in parallel structures
 * (the lift re-lowers from AST, so it doesn't inherit
 * loweredModule.rhs's alias resolution). Running the same pass on
 * both keeps them in lockstep.
 *
 * Behavior:
 *   - alias bindings (single-ref RHS in <irField>) get `isAlias: true`
 *     and their RHS is updated to the canonical ref;
 *   - non-alias bindings have their RHS `mapIR`-rewritten so every
 *     `(%ref self <alias>)` ref AND `call.target = {ns:'self',
 *     name:<alias>}` is replaced by the alias's canonical ref/target.
 *
 * Cycles are detected and left untouched. Returns the same Map
 * (mutation in place).
 *
 * IR purity: rewritten bindings get a NEW IR object (shallow-rebuilt
 * via mapIR). Caller-held references to pre-resolution IR stay valid;
 * the resolution writes to `binding.<irField>` so subsequent reads
 * see the canonical form.
 */
function resolveAliasesOnBindings(bindings: any, irField: string): any {
  if (!bindings) return bindings;

  // Pass 1: collect every binding whose RHS is a pure ref. These are
  // the alias candidates.
  const aliases = new Map<string, any>();
  for (const [name, lb] of bindings) {
    const target = lb && _aliasTarget(lb[irField]);
    if (target) aliases.set(name, target);
  }

  // Pass 2: compute the canonical ref for each alias (walking through
  // self-alias chains transitively). Cycle-skipped aliases keep their
  // original target.
  const canonicalByName = new Map<string, any>();
  for (const [name, target] of aliases) {
    const canon = _canonicalise(target, aliases);
    if (canon !== null) canonicalByName.set(name, canon);
  }

  // Pass 3: rewrite every NON-ALIAS binding's RHS in place. The
  // mapIR walker (ir-walk.ts) visits every IR sub-position; we
  // rewrite two distinct alias-bearing shapes:
  //   - value-position refs:   { kind: 'ref',  ns: 'self', name: X, … }
  //   - call-position targets: { kind: 'call', target: { ns: 'self',
  //                                                       name: X }, …}
  //
  // The two shapes are structurally distinct in the engine's IR (a
  // ref node has no `target`; a call node's target is a string-pair
  // descriptor, not a child IR node — so mapIR's default sub-position
  // walk doesn't visit it). Both need rewriting for the alias to
  // disappear from downstream consumers.
  //
  // Identity-preserving — bindings whose IR contains no alias-resolved
  // refs / targets are untouched (no copy allocated).
  for (const [name, lb] of bindings) {
    if (aliases.has(name)) continue;  // alias bindings handled in pass 4
    if (!lb || lb[irField] == null) continue;
    lb[irField] = mapIR(lb[irField], (n: any) => {
      if (!n || typeof n !== 'object') return n;
      if (n.kind === 'ref' && n.ns === 'self'
          && canonicalByName.has(n.name)) {
        const canon = canonicalByName.get(n.name);
        // Preserve source-location for downstream diagnostics.
        return { kind: 'ref', ns: canon.ns, name: canon.name, loc: n.loc };
      }
      if (n.kind === 'call' && n.target && n.target.ns === 'self'
          && canonicalByName.has(n.target.name)) {
        const canon = canonicalByName.get(n.target.name);
        return Object.assign({}, n, {
          target: { ns: canon.ns, name: canon.name },
        });
      }
      return n;
    });
  }

  // Pass 4: update each alias binding to its canonical target and
  // mark it. Subsequent passes (derivations / signatures / DAG) can
  // skip-or-special-case alias bindings via `lb.isAlias`. Cyclic
  // aliases stay un-marked so the cycle surfaces as the original
  // self-reference downstream (the cleanest place to error).
  for (const [name, lb] of bindings) {
    if (!aliases.has(name)) continue;
    const canon = canonicalByName.get(name);
    if (canon == null) continue;  // cyclic — leave untouched
    if (!lb) continue;
    const oldLoc = lb[irField] && lb[irField].loc;
    lb[irField] = { kind: 'ref', ns: canon.ns, name: canon.name, loc: oldLoc };
    lb.isAlias = true;
  }

  return bindings;
}

/**
 * LoweredModule entry point — runs alias resolution over `.rhs`.
 * The original API the analyzer pipeline uses (analyzer.ts:2534).
 */
function resolveAliases(loweredModule: any): any {
  if (!loweredModule || !loweredModule.bindings) return loweredModule;
  resolveAliasesOnBindings(loweredModule.bindings, 'rhs');
  return loweredModule;
}

module.exports = {
  resolveAliases,
  resolveAliasesOnBindings,
  // Exported for tests / introspection.
  _internal: {
    _aliasTarget,
    _canonicalise,
  },
};

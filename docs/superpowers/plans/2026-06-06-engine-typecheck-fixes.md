# Engine + Viewer Typecheck Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive `npm run typecheck` to zero errors (engine 36 + viewer 4 = 40 pre-existing), then make the CI typecheck step blocking.

**Architecture:** Proportionate fixes matching the codebase's existing loose typing — minimal interface extensions for dynamically-attached properties, one real runtime guard, boundary casts where strict-mode `Array.from` widens `any` to `unknown[]`. No type-system overhaul. Each group is one commit; `npx tsc` error count is the red/green signal; `npm test` must stay green throughout.

**Tech Stack:** TypeScript 5.7+ (strict), `tsc --noEmit` lint-only typecheck, Node `--test`, npm workspaces.

**Branch:** `typecheck-fixes` (already created off synced main `17ec32a`).

**Root-cause reference (from investigation):**
- **Group B** — objects get props bolted on at runtime that aren't in their inferred/declared type.
- **Group C** — `ops.ts:1091` calls optional `decl.logical` without a guard (real latent bug); `ops-declarations.ts` `s` is a false-positive narrowing gap.
- **Group A** — `Array.from(x)` with `x: any` returns `unknown[]` under strict mode (generic `T` defaults to `unknown`); `unknown[]` then fails `number[]` params and breaks `.sort`/`.reduce` comparator contravariance. `_lu`/`_matValue` return `any` (`ext-linalg.ts:22,44`).
- **Group D** — TS 5.7 made TypedArrays generic over buffer (`Float64Array<ArrayBuffer>` vs `<ArrayBufferLike>`); plus viewer browser-path `require` + one implicit-any param.

**Baseline command (run once before starting, to confirm the starting count):**

Run: `cd packages/engine && npx tsc -p tsconfig.json 2>&1 | grep -c "error TS"`
Expected: `36`

Run: `cd packages/viewer && npx tsc -p tsconfig.json 2>&1 | grep -c "error TS"`
Expected: `4`

---

### Task 1: Group B — type dynamically-attached properties (6 errors: engine 4 + viewer 2)

Kills: viewer `derivations.ts:30,195`; engine `pir.ts:139`, `derivations.ts:204,205`, `lift.ts:1041`.

**Files:**
- Modify: `packages/engine/engine-types.d.ts:620-628` (DerivationsState interface)
- Modify: `packages/engine/pir.ts:68-75` (loweredModule factory)
- Modify: `packages/engine/derivations.ts:179` and `:204-208` (bijection metadata)
- Modify: `packages/engine/lift.ts:1041` (mvnormal marker assignment)

- [ ] **Step 1: Add `moduleRegistry` to DerivationsState**

In `packages/engine/engine-types.d.ts`, the interface currently reads:

```ts
export interface DerivationsState {
  bindings: Map<string, BindingInfo>;
  derivations: Record<string, Derivation>;
  fixedValues: Map<string, any>;
  discrete: Record<string, boolean>;
  /** Classifier diagnostics surfaced by buildDerivations (e.g.
   *  fixed-phase dead ends). Empty when nothing to report. */
  diagnostics?: Array<{ message: string; [extra: string]: any }>;
}
```

Add the optional field (populated at runtime by `pir.lowerToModule`, read by the viewer's derivations bridge):

```ts
export interface DerivationsState {
  bindings: Map<string, BindingInfo>;
  derivations: Record<string, Derivation>;
  fixedValues: Map<string, any>;
  discrete: Record<string, boolean>;
  /** Classifier diagnostics surfaced by buildDerivations (e.g.
   *  fixed-phase dead ends). Empty when nothing to report. */
  diagnostics?: Array<{ message: string; [extra: string]: any }>;
  /** alias → resolved module descriptor, copied from the lowered
   *  module by the viewer bridge. Populated by pir.lowerToModule. */
  moduleRegistry?: Record<string, any>;
}
```

- [ ] **Step 2: Annotate the loweredModule factory return**

In `packages/engine/pir.ts`, the factory currently reads:

```ts
function loweredModule(opts?: any) {
  opts = opts || {};
  return {
    bindings: new Map(),       // name → LoweredBinding (insertion-ordered)
    publicSet: new Set(),      // export list; populated during lowering
    source: opts.source || null,
  };
}
```

Give it an explicit return type declaring the post-construction `moduleRegistry` field (set at `pir.ts:139`):

```ts
function loweredModule(opts?: any): {
  bindings: Map<string, any>;
  publicSet: Set<string>;
  source: any;
  moduleRegistry?: Record<string, any>;
} {
  opts = opts || {};
  return {
    bindings: new Map(),       // name → LoweredBinding (insertion-ordered)
    publicSet: new Set(),      // export list; populated during lowering
    source: opts.source || null,
  };
}
```

- [ ] **Step 3: Type the bijection-metadata literal in derivations.ts**

In `packages/engine/derivations.ts:179` the literal is assigned:

```ts
    binding.bijection = { fName: fA.name, fInvName: fIA.name, logVolume };
```

The §22 registry block at `:204-208` later adds `.registryName` and `.paramIRs`. Declare those as optional on the literal so the later assignments typecheck. Replace line 179 with:

```ts
    binding.bijection = { fName: fA.name, fInvName: fIA.name, logVolume } as {
      fName: string; fInvName: string; logVolume: any;
      registryName?: string;
      paramIRs?: Record<string, any>;
    };
```

Leave the `:204-208` assignment block unchanged — it now matches the widened type.

- [ ] **Step 4: Match the existing `as any` for the mvnormal marker write**

In `packages/engine/lift.ts:1041`, the assignment currently reads:

```ts
    bijBinding.__mvnormalLowering = {
```

`bijBinding`'s inferred literal type lacks `__mvnormalLowering`. Cast at the write to match the `(binding as any).__mvnormalLowering` reads used in derivations.ts. Replace with:

```ts
    (bijBinding as any).__mvnormalLowering = {
```

- [ ] **Step 5: Verify engine + viewer error counts dropped**

Run: `cd packages/engine && npx tsc -p tsconfig.json 2>&1 | grep -c "error TS"`
Expected: `32` (36 − 4: pir.ts:139 (1) + derivations.ts:204,205 (2) + lift.ts:1041 (1) = 4 engine Group B errors).

> Treat **any decrease with no NEW errors** as success; the absolute target for the whole plan is 0.

Run: `cd packages/viewer && npx tsc -p tsconfig.json 2>&1 | grep -c "error TS"`
Expected: `2` (viewer Group B = derivations.ts:30,195 = 2; viewer goes 4 → 2)

- [ ] **Step 6: Confirm no behavior change**

Run: `cd packages/engine && npm test 2>&1 | tail -5`
Expected: all tests pass (no `not ok` lines; final summary `# fail 0`)

- [ ] **Step 7: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add packages/engine/engine-types.d.ts packages/engine/pir.ts packages/engine/derivations.ts packages/engine/lift.ts
git commit -m "types: declare dynamically-attached module/bijection fields

pir.lowerToModule attaches moduleRegistry to the lowered module and
the viewer copies it onto DerivationsState; the §22 registry block
adds registryName/paramIRs to binding.bijection; lift marks synthetic
MvNormal bindings with __mvnormalLowering. None were in the declared
types. Declare the optional fields (and match the existing as-any for
the lift marker) so strict tsc accepts the runtime shape."
```

---

### Task 2: Group C — close the real guard gap + the narrowing false-positive (4 errors)

Kills: engine `ops.ts:1091`, `ops-declarations.ts:1139,1140` (x2 lines, 3 error instances).

**Files:**
- Modify: `packages/engine/ops.ts:1076-1091` (dispatchHigherOrder)
- Modify: `packages/engine/ops-declarations.ts:1138` (add post-chain null guard)

- [ ] **Step 1: Add the missing `decl.logical` guard in dispatchHigherOrder**

In `packages/engine/ops.ts`, the function ends:

```ts
  if (!ctx || typeof ctx.evaluateExpr !== 'function'
      || typeof ctx.resolveFn !== 'function') {
    throw new Error('ops.dispatchHigherOrder: ctx must provide env, ' +
      'evaluateExpr, and resolveFn');
  }
  return decl.logical(ir, ctx);
```

`logical?` is optional (`ops.ts:98`). Sibling dispatchers guard it (`ops.ts:965,982`); this one does not — a higher-order op registered without a `logical` impl throws an opaque "not a function" at runtime. Add the guard, matching the sibling style:

```ts
  if (!ctx || typeof ctx.evaluateExpr !== 'function'
      || typeof ctx.resolveFn !== 'function') {
    throw new Error('ops.dispatchHigherOrder: ctx must provide env, ' +
      'evaluateExpr, and resolveFn');
  }
  if (typeof decl.logical !== 'function') {
    throw new Error("ops.dispatchHigherOrder: op '" + name +
      "' has no logical impl");
  }
  return decl.logical(ir, ctx);
```

- [ ] **Step 2: Add the post-chain null guard in ops-declarations.ts**

In `packages/engine/ops-declarations.ts`, `s` is declared `let s: number[] | null = null;` (line 1122). Every non-returning branch assigns it, but tsc won't carry that narrowing past the if-chain, so `s.length` / `s[a]` at `:1139-1140` flag "possibly null". The chain currently ends and flows into:

```ts
    if (innerShape === null) { innerShape = s; stackOuterRank = nestedTag; }
    else {
```

Add an explicit guard immediately before that `if`, after the if/else-if chain closes (i.e. right after the final `else { return xs; }` block and before `if (innerShape === null)`):

```ts
    if (s === null) return xs;
    if (innerShape === null) { innerShape = s; stackOuterRank = nestedTag; }
    else {
```

This both satisfies the narrower and is defensively correct (s being null here would mean an unhandled arg shape — bail to JS-array form, consistent with the other `return xs` fallbacks).

- [ ] **Step 3: Verify engine error count dropped**

Run: `cd packages/engine && npx tsc -p tsconfig.json 2>&1 | grep -c "error TS"`
Expected: `28` (32 − 4: ops.ts:1091 = 1, ops-declarations.ts:1139,1140 = 3)

- [ ] **Step 4: Confirm no behavior change**

Run: `cd packages/engine && npm test 2>&1 | tail -5`
Expected: all tests pass (`# fail 0`). The two new guards are throw-on-impossible paths not hit by existing tests.

- [ ] **Step 5: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add packages/engine/ops.ts packages/engine/ops-declarations.ts
git commit -m "fix: guard optional decl.logical in dispatchHigherOrder

decl.logical is optional (variant-only ops omit it) but
dispatchHigherOrder called it unguarded, unlike its siblings — a
higher-order op without a logical impl would throw an opaque 'not a
function'. Add the guard. Also add an explicit s-null bail in the
stack-shape loop (ops-declarations) to satisfy strict narrowing and
match the existing return-xs fallbacks."
```

---

### Task 3: Group A — boundary casts where strict `Array.from` widens to `unknown[]` (27 errors)

Kills: engine `test/ext-linalg.test.ts` (23) + `test/kernel-chain-first-class.test.ts` (4). Test-only; zero runtime risk.

**Files:**
- Modify: `packages/engine/test/ext-linalg.test.ts` (multiple sites)
- Modify: `packages/engine/test/kernel-chain-first-class.test.ts:408-415`

**The pattern (apply uniformly):** `Array.from(X)` where `X` is `any`/`unknown` produces `unknown[]`. Type the source so `X` is a `Float64Array` (→ `Array.from` yields `number[]`), or cast the source to `ArrayLike<number>`. Casting the source is the smaller, local edit.

- [ ] **Step 1: Cast the matrix `.data` reads in ext-linalg.test.ts**

The recurring shape is, e.g. at `test/ext-linalg.test.ts:59-62`:

```ts
  const { fields } = extLinalg._lu(A);
  const P = Array.from(fields.P.data);
  const L = Array.from(fields.L.data);
  const U = Array.from(fields.U.data);
```

Wrap each `_lu`/matrix destructure so `.data` is typed. Replace the destructure line with a typed cast and leave the `Array.from` calls as-is:

```ts
  const { fields } = extLinalg._lu(A) as {
    fields: Record<string, { data: ArrayLike<number> }>
  };
  const P = Array.from(fields.P.data);
  const L = Array.from(fields.L.data);
  const U = Array.from(fields.U.data);
```

Apply the same `as { fields: Record<string, { data: ArrayLike<number> }> }` cast at **every** `const { fields } = extLinalg.<fn>(...)` site in the file (the LU, matexp, and matmul tests all share this shape). For standalone `Array.from(<expr>.data)` calls whose source is not a `fields` destructure, cast the source inline: `Array.from((expr).data as ArrayLike<number>)`.

For the comparator errors (`:277,284` — `(a: number, b: number) => number` not assignable to `(a: unknown, b: unknown) => number`): these are `.sort`/comparator calls on arrays that were `unknown[]`. Once the `.data` sources above are `ArrayLike<number>`, the arrays are `number[]` and the comparators typecheck — re-run tsc after Step 1 before adding any comparator-specific cast; only annotate a comparator (`(a: number, b: number)` already present) if an error genuinely remains.

- [ ] **Step 2: Cast `components[i]` in kernel-chain-first-class.test.ts**

At `test/kernel-chain-first-class.test.ts:408-415`:

```ts
    const k0 = components[0];
    assert.ok(k0.samples && k0.samples.length === 256);
    const mean0 = Array.from(k0.samples).reduce((a: number, b: number) => a + b, 0) / k0.samples.length;
    ...
    const k1 = components[1];
    const mean1 = Array.from(k1.samples).reduce((a: number, b: number) => a + b, 0) / k1.samples.length;
```

`components[i]` is `unknown`. Cast at the two extraction sites so `.samples` is a typed numeric buffer:

```ts
    const k0 = components[0] as { samples: ArrayLike<number> };
    assert.ok(k0.samples && k0.samples.length === 256);
    const mean0 = Array.from(k0.samples).reduce((a: number, b: number) => a + b, 0) / k0.samples.length;
    ...
    const k1 = components[1] as { samples: ArrayLike<number> };
    const mean1 = Array.from(k1.samples).reduce((a: number, b: number) => a + b, 0) / k1.samples.length;
```

- [ ] **Step 3: Iterate to zero on the two test files**

Run: `cd packages/engine && npx tsc -p tsconfig.json 2>&1 | grep -E "ext-linalg.test.ts|kernel-chain-first-class.test.ts"`
Expected: no output (0 errors in both test files).

If any remain, they are the same root (an untyped value flowing into `Array.from`/`.length`/index) — apply the same `as ArrayLike<number>` (for buffers) or `as { ... }` (for objects) cast at the named line. Do not introduce `as any` blanket casts; keep each cast at the narrowest site.

- [ ] **Step 4: Verify total engine count + tests green**

Run: `cd packages/engine && npx tsc -p tsconfig.json 2>&1 | grep -c "error TS"`
Expected: `1` (only `ext-linalg.ts:227` Group D remains)

Run: `cd packages/engine && npm test 2>&1 | tail -5`
Expected: all pass (`# fail 0`). Casts are type-only; runtime unchanged.

- [ ] **Step 5: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add packages/engine/test/ext-linalg.test.ts packages/engine/test/kernel-chain-first-class.test.ts
git commit -m "test: type the buffers fed to Array.from in engine tests

Under strict mode Array.from(x) with x: any returns unknown[] (the
generic T defaults to unknown), which then fails number[] params and
breaks comparator contravariance in .sort/.reduce. _lu/_matValue
return any, so the matrix .data reads and measure components were
untyped. Cast the sources to ArrayLike<number> / minimal object
shapes at each read site — type-only, no runtime change."
```

---

### Task 4: Group D — TypedArray buffer variance + viewer trivia (3 errors)

Kills: engine `ext-linalg.ts:227`; viewer `plot-plan.ts:369`, `render-profile.ts:218`. (Viewer derivations.ts:30,195 already fixed in Task 1.)

**Files:**
- Modify: `packages/engine/ext-linalg.ts:207` (widen `R`'s declared type)
- Modify: `packages/viewer/src/plot-plan.ts` (ambient `require` declaration)
- Modify: `packages/viewer/src/render-profile.ts:218` (annotate param)

- [ ] **Step 1: Widen `R` in ext-linalg.ts so `_matmul`'s return assigns**

At `packages/engine/ext-linalg.ts:207`:

```ts
  let R = new Float64Array(n * n);
```

`new Float64Array(number)` infers `Float64Array<ArrayBuffer>`, but `_matmul` returns the wider `Float64Array` (= `Float64Array<ArrayBufferLike>`), so `R = _matmul(R, R, n)` at `:227` fails. Annotate `R` with the wider type:

```ts
  let R: Float64Array = new Float64Array(n * n);
```

- [ ] **Step 2: Declare `require` for the viewer's type-only pass**

`packages/viewer/src/plot-plan.ts:369` uses `require('./engine-facade.js')` (esbuild rewrites it at bundle time). The viewer tsconfig is `module: es2020` with no node types, so `require` is undefined to tsc. Do NOT add `"types": ["node"]` — that would wrongly pull node globals into browser code. Instead add a file-local ambient declaration near the top of `plot-plan.ts`, after the existing imports:

```ts
// esbuild rewrites this CommonJS require at bundle time; declare it for
// the type-only tsc pass (the viewer tsconfig omits node types on
// purpose — this is browser code). The require here is a pre-existing
// lazy-load of engine-facade; not refactored as part of the typecheck
// cleanup.
declare function require(id: string): any;
```

- [ ] **Step 3: Annotate the implicit-any param in render-profile.ts**

At `packages/viewer/src/render-profile.ts:218`:

```ts
    Promise.all(selfRefs.map(function(n) { return tryGetMeasure(ctx, n); })),
```

`n` is implicitly `any`. Annotate it (sibling callbacks in this file use explicit types):

```ts
    Promise.all(selfRefs.map(function(n: any) { return tryGetMeasure(ctx, n); })),
```

- [ ] **Step 4: Verify both packages are at zero**

Run: `cd packages/engine && npx tsc -p tsconfig.json 2>&1 | grep -c "error TS"`
Expected: `0`

Run: `cd packages/viewer && npx tsc -p tsconfig.json 2>&1 | grep -c "error TS"`
Expected: `0`

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js && npm run typecheck 2>&1 | tail -3; echo "EXIT:$?"`
Expected: `EXIT:0` (all four workspaces clean: engine, viewer, web, vscode-extension)

- [ ] **Step 5: Confirm engine tests still green**

Run: `cd packages/engine && npm test 2>&1 | tail -5`
Expected: `# fail 0`

- [ ] **Step 6: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add packages/engine/ext-linalg.ts packages/viewer/src/plot-plan.ts packages/viewer/src/render-profile.ts
git commit -m "types: TypedArray buffer variance + viewer require/param annotations

ext-linalg: widen R to Float64Array so _matmul's ArrayBufferLike
return assigns (TS 5.7 made TypedArrays buffer-generic). viewer:
declare require locally for plot-plan's bundle-time lazy load (no
node types in the browser tsconfig), and annotate an implicit-any
callback param in render-profile."
```

---

### Task 5: Make the typecheck CI step blocking

**Files:**
- Modify: `.github/workflows/test.yml` (remove `continue-on-error` from the Typecheck step)

- [ ] **Step 1: Remove the non-blocking escape hatch**

In `.github/workflows/test.yml` the Typecheck step currently reads:

```yaml
      - name: Typecheck
        # tsc across every workspace with a typecheck script
        # (--workspaces --if-present). Catches type errors the esbuild
        # bundle step below would miss — esbuild strips types without
        # checking them.
        #
        # NON-BLOCKING for now: engine + viewer carry ~40 pre-existing
        # type errors. continue-on-error surfaces the signal (the step
        # shows red in the run) without failing the job. Drop this
        # once those packages typecheck clean.
        continue-on-error: true
        run: npm run typecheck
```

All 40 errors are now fixed, so make it a real gate. Replace with:

```yaml
      - name: Typecheck
        # tsc across every workspace with a typecheck script
        # (--workspaces --if-present). Catches type errors the esbuild
        # bundle step below would miss — esbuild strips types without
        # checking them. Blocking: the tree typechecks clean.
        run: npm run typecheck
```

- [ ] **Step 2: Verify the workflow still parses and the gate is on**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js && node -e "const s=require('fs').readFileSync('.github/workflows/test.yml','utf8'); if(/^\t/m.test(s)) throw 'tabs'; if(s.includes('continue-on-error')) throw 'still non-blocking'; console.log('test.yml: blocking typecheck, no tabs')"`
Expected: `test.yml: blocking typecheck, no tabs`

- [ ] **Step 3: Final full-repo verification**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js && npm run typecheck 2>&1 | tail -3; echo "TYPECHECK_EXIT:$?"`
Expected: `TYPECHECK_EXIT:0`

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js && npm test 2>&1 | tail -5`
Expected: all workspaces' tests pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add .github/workflows/test.yml
git commit -m "ci: make typecheck blocking now that the tree is clean

All 40 pre-existing engine + viewer type errors are fixed, so drop
continue-on-error from the Typecheck step — type regressions now fail
the Test workflow."
```

---

## Done criteria

- `npm run typecheck` exits 0 across all four workspaces.
- `npm test` passes (engine behavior unchanged; only type-level edits + two throw-on-impossible guards).
- `.github/workflows/test.yml` Typecheck step has no `continue-on-error`.
- Five commits on `typecheck-fixes`, no push (await instruction).

## Notes for the executor

- **Do not push.** The user controls push/PR (per repo convention).
- Error-count "Expected" numbers are guide rails; the absolute gate is **0 at the end** with **no NEW errors introduced** at any step. If a step's count differs but is strictly decreasing with no new error codes, proceed.
- Keep casts at the narrowest site; never blanket-`as any` a whole module.
- If any `npm test` run goes red, STOP — a type fix changed behavior (most likely the Group C guards hit a real path). Investigate before continuing.

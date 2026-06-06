# Vectorized `broadcast(fn(logdensityof(M, _)), grid)` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the idiomatic broadcasting surface `broadcast(fn(logdensityof(M, _)), grid)` and its dot-sugar `fn(logdensityof(M, _)).(grid)` evaluate a closed measure's log-density at every grid point in one binding — by routing them to the *existing* `broadcast_logdensity` materialise path.

**Architecture:** The engine ALREADY vectorizes `broadcast(logdensityof, M, pts)` (3-arg, bare-builtin head) — `classifyBroadcastLogdensity` (`derivations.ts:1241`) produces a `broadcast_logdensity` derivation that `matBroadcastLogdensity` (`mat-density.ts:334`) evaluates via the worker's `pointsBatched` path, bit-exact to the per-point scalar loop. The fn-wrapped/dot surface currently falls through to `kind:'evaluate'` and crashes at runtime (`call op 'logdensityof' not evaluable in sampler context`). This plan adds ONE new sub-classifier, `classifyBroadcastFnLogdensity`, that recognizes the fn-wrapped shape and returns the **same** `broadcast_logdensity` derivation record — so no materialise/worker/density code changes. Everything else (value-fn broadcast, kernel broadcast, 3-arg logdensity broadcast) is untouched.

**Tech Stack:** TypeScript (strict, `tsc --noEmit` lint-only), Node `--test`, npm workspaces. All work in `packages/engine`.

**Branch:** create `broadcast-fn-logdensity` off synced `main` as the first action (see Task 0).

---

## Background facts (verified against the running engine — do not re-derive)

- `broadcast(logdensityof, mix, grid)` **already works** and is **bit-exact** to the scalar per-point loop (verified: max abs diff = 0 over a 4-point grid on `normalize(superpose(weighted(0.4,Normal),weighted(0.6,Normal)))`).
- The fn-wrapped form lowers to this exact IR (captured from the engine):
  ```json
  {"kind":"call","op":"broadcast","args":[
    {"kind":"call","op":"functionof","params":["_arg1_"],
     "body":{"kind":"call","op":"logdensityof","args":[
        {"kind":"ref","ns":"self","name":"mix"},
        {"kind":"ref","ns":"%local","name":"_arg1_"}]}},
    {"kind":"ref","ns":"self","name":"grid"}]}
  ```
  i.e. `args.length === 2`; `args[0]` is an inline `functionof` with a single `params` entry; its `body` is `logdensityof(<self-ref measure>, <ref to that same param>)`; `args[1]` is the points expression.
- Current derivation kind for the fn-form: `evaluate` (broken). For the 3-arg form: `broadcast_logdensity` (works). This is the entire gap.
- `classifyBroadcast` (`derivations.ts:1262`) is the SOLE classifier for `{op:'broadcast'}` (registered at `derivations.ts:1647`). It chains sub-classifiers: `classifyBroadcastLogdensity(3-arg) || classifyKernelBroadcast`. The new sub-classifier slots **between** them.
- Existing classifier return shape (copy exactly): `{ kind: 'broadcast_logdensity', measureName: <string>, pointsIR: <IRNode> }` — type `DerivationBroadcastLogdensity` (`engine-types.d.ts:420`).
- `matBroadcastLogdensity` already handles closed (batched, fast) AND non-closed (per-point fallback loop) measures, and resolves `pointsIR` for both array literals and `linspace` Values. No changes needed there.
- `isSelfRef(node)` helper exists in `derivations.ts` (used at `derivations.ts:1257`).
- Regression baseline: on current `main`, `cd packages/engine && npm test` ⇒ `# pass 3117  # fail 0`. `npm run typecheck` ⇒ exit 0.

---

### Task 0: Branch and baseline

**Files:** none (git + verification only)

- [ ] **Step 1: Sync main and branch**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git fetch origin && git checkout main && git pull --ff-only origin main
git checkout -b broadcast-fn-logdensity main
git branch --show-current
```
Expected: `broadcast-fn-logdensity`

- [ ] **Step 2: Capture the regression baseline**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && node --test 'test/*.test.ts' 2>&1 | grep -E "^ℹ (tests|pass|fail) "`
Expected:
```
ℹ tests 3117
ℹ pass 3117
ℹ fail 0
```

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js && npm run typecheck 2>&1 | tail -1; echo "EXIT:$?"`
Expected: `EXIT:0`

If the baseline differs (someone changed main), record the actual numbers and use them as the target for the regression checks below.

---

### Task 1: New classifier `classifyBroadcastFnLogdensity` + failing test

**Files:**
- Create: `packages/engine/test/broadcast-fn-logdensity.test.ts`
- Modify: `packages/engine/derivations.ts` (add `classifyBroadcastFnLogdensity`; wire into `classifyBroadcast` at `derivations.ts:1262-1265`)

- [ ] **Step 1: Write the failing test**

Create `packages/engine/test/broadcast-fn-logdensity.test.ts` with this exact content (a self-contained harness — proven to work; does not depend on any test-only export):

```ts
'use strict';

// broadcast(fn(logdensityof(M, _)), grid) and its dot-sugar must
// evaluate M's log-density at every grid point in ONE binding,
// bit-exact to the per-point scalar logdensityof loop. Routes through
// the existing broadcast_logdensity materialise path.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

function mk(src: string): any {
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: [1, 2, 3] });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      if (r && r.type === 'error') return Promise.reject(new Error(r.message));
      return Promise.resolve(r);
    },
    sampleCount: 1000,
    rootSeed: [1, 2, 3],
  };
  return ctx;
}

const MODEL = `
A = Normal(mu = -2.0, sigma = 0.8)
B = Normal(mu =  3.0, sigma = 1.3)
mix = normalize(superpose(weighted(0.4, A), weighted(0.6, B)))
`;
const PTS = [-2.0, 0.0, 3.0, 7.5];

async function scalarRef(): Promise<number[]> {
  let src = MODEL;
  PTS.forEach((p, i) => { src += `p${i} = logdensityof(mix, ${p})\n`; });
  const ctx = mk(src);
  const out: number[] = [];
  for (let i = 0; i < PTS.length; i++) out.push((await ctx.getMeasure('p' + i)).samples[0]);
  return out;
}

test('broadcast(fn(logdensityof(M,_)), grid): bit-exact to scalar loop', async () => {
  const ctx = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = broadcast(fn(logdensityof(mix, _)), grid)`);
  const m = await ctx.getMeasure('gv');
  assert.ok(m && m.samples, 'gv must materialise to a measure with samples');
  assert.equal(m.samples.length, PTS.length, `expected ${PTS.length} densities`);
  const ref = await scalarRef();
  for (let i = 0; i < PTS.length; i++) {
    assert.equal(m.samples[i], ref[i],
      `point ${PTS[i]}: broadcast=${m.samples[i]} scalar=${ref[i]}`);
  }
});

test('broadcast(fn(logdensityof(M,_)), grid): agrees with the existing 3-arg form', async () => {
  const fn3 = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = broadcast(logdensityof, mix, grid)`);
  const fnW = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = broadcast(fn(logdensityof(mix, _)), grid)`);
  const a = await fn3.getMeasure('gv');
  const b = await fnW.getMeasure('gv');
  assert.equal(b.samples.length, a.samples.length);
  for (let i = 0; i < a.samples.length; i++) assert.equal(b.samples[i], a.samples[i]);
});
```

- [ ] **Step 2: Run the test to verify it FAILS**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && node --test 'test/broadcast-fn-logdensity.test.ts' 2>&1 | grep -E "not ok|# fail|not evaluable|samples"`
Expected: FAIL — the first test errors (the fn-wrapped binding currently classifies as `evaluate` and throws at materialise, e.g. `call op 'logdensityof' not evaluable in sampler context`, or `gv` has no `samples`). The 3-arg comparison test's `fn3` side works but `fnW` side fails the same way.

- [ ] **Step 3: Add the new sub-classifier**

In `packages/engine/derivations.ts`, immediately ABOVE the existing `classifyBroadcast` function (which begins at `derivations.ts:1262` with the `// broadcast is overloaded:` comment), insert:

```ts
// broadcast(fn(logdensityof(M, _)), pts) — the idiomatic broadcasting
// surface (and its dot-sugar fn(logdensityof(M,_)).(pts)) for grid
// density evaluation. Lowers to a 2-arg broadcast whose head is an
// inline functionof of one parameter whose body is EXACTLY
// logdensityof(<self-measure>, <that param>). Recognise that precise
// shape and emit the SAME broadcast_logdensity derivation the bare
// 3-arg broadcast(logdensityof, M, pts) form produces, so it reuses
// matBroadcastLogdensity verbatim. Anything richer (a transformed
// point, a scaled/offset density, multiple params) does NOT match and
// falls through unchanged.
function classifyBroadcastFnLogdensity(
  rhsIR: IRNode, ast: any, bindings: any,
): DerivationBroadcastLogdensity | null {
  if (rhsIR.op !== 'broadcast' || !Array.isArray(rhsIR.args)
      || rhsIR.args.length !== 2) return null;
  const headIR: any = rhsIR.args[0];
  const pIR: any = rhsIR.args[1];
  // Head: inline functionof with exactly one parameter.
  if (!headIR || headIR.kind !== 'call' || headIR.op !== 'functionof'
      || !Array.isArray(headIR.params) || headIR.params.length !== 1) {
    return null;
  }
  const paramName = headIR.params[0];
  const body: any = headIR.body;
  // Body: exactly logdensityof(<measure>, <param>).
  if (!body || body.kind !== 'call' || body.op !== 'logdensityof'
      || !Array.isArray(body.args) || body.args.length !== 2) {
    return null;
  }
  const mIR: any = body.args[0];
  const xIR: any = body.args[1];
  if (!isSelfRef(mIR) || !bindings.has(mIR.name)) return null;
  // The point must be the functionof's own parameter, untouched — not
  // a transformed expression. Match by name (ns is the local param ns).
  if (!xIR || xIR.kind !== 'ref' || xIR.name !== paramName) return null;
  return { kind: 'broadcast_logdensity', measureName: mIR.name, pointsIR: pIR };
}
```

- [ ] **Step 4: Wire it into `classifyBroadcast`**

In `packages/engine/derivations.ts`, the `classifyBroadcast` function currently reads:

```ts
function classifyBroadcast(rhsIR: IRNode, ast: any, bindings: any): DerivationBroadcastLogdensity | DerivationKernelBroadcast | null {
  return classifyBroadcastLogdensity(rhsIR, ast, bindings)
      || classifyKernelBroadcast(rhsIR, ast, bindings);
}
```

Change it to insert the new sub-classifier between the two (after the 3-arg logdensity form, before kernel-broadcast):

```ts
function classifyBroadcast(rhsIR: IRNode, ast: any, bindings: any): DerivationBroadcastLogdensity | DerivationKernelBroadcast | null {
  return classifyBroadcastLogdensity(rhsIR, ast, bindings)
      || classifyBroadcastFnLogdensity(rhsIR, ast, bindings)
      || classifyKernelBroadcast(rhsIR, ast, bindings);
}
```

- [ ] **Step 5: Run the test to verify it PASSES**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && node --test 'test/broadcast-fn-logdensity.test.ts' 2>&1 | grep -E "^ℹ (tests|pass|fail) "`
Expected:
```
ℹ tests 2
ℹ pass 2
ℹ fail 0
```

- [ ] **Step 6: Typecheck still clean**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js && npm run typecheck 2>&1 | tail -1; echo "EXIT:$?"`
Expected: `EXIT:0`

- [ ] **Step 7: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add packages/engine/derivations.ts packages/engine/test/broadcast-fn-logdensity.test.ts
git commit -m "feat: recognize broadcast(fn(logdensityof(M,_)), grid)

The bare 3-arg broadcast(logdensityof, M, pts) already vectorizes grid
density evaluation, but the idiomatic fn-wrapped surface
broadcast(fn(logdensityof(M,_)), grid) fell through to kind=evaluate and
crashed at materialise. Add classifyBroadcastFnLogdensity: it matches the
exact fn(logdensityof(self-M, param)) shape and emits the SAME
broadcast_logdensity derivation, reusing matBroadcastLogdensity verbatim.
No materialise/worker/density changes. Bit-exact to the scalar loop."
```

---

### Task 2: Dot-sugar `fn(logdensityof(M,_)).(grid)` coverage

The dot-call `f.(args)` lowers to `broadcast(f, args)` (per spec `04-design.md` → "Broadcasting"), so `fn(logdensityof(mix,_)).(grid)` produces the same 2-arg functionof-head IR as Task 1 and is already handled. This task only PROVES it with a test (no code change expected).

**Files:**
- Modify: `packages/engine/test/broadcast-fn-logdensity.test.ts`

- [ ] **Step 1: Add the dot-sugar test**

Append to `packages/engine/test/broadcast-fn-logdensity.test.ts`:

```ts
test('dot-sugar fn(logdensityof(M,_)).(grid): bit-exact to scalar loop', async () => {
  const ctx = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = (fn(logdensityof(mix, _))).(grid)`);
  const m = await ctx.getMeasure('gv');
  assert.ok(m && m.samples, 'gv must materialise to a measure with samples');
  assert.equal(m.samples.length, PTS.length);
  const ref = await scalarRef();
  for (let i = 0; i < PTS.length; i++) {
    assert.equal(m.samples[i], ref[i],
      `point ${PTS[i]}: dot=${m.samples[i]} scalar=${ref[i]}`);
  }
});
```

- [ ] **Step 2: Run it**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && node --test 'test/broadcast-fn-logdensity.test.ts' 2>&1 | grep -E "^ℹ (tests|pass|fail) "`
Expected: `ℹ tests 3  ℹ pass 3  ℹ fail 0`

> If the dot-sugar test FAILS (the dot-call lowered to a different IR shape than the Task-1 form), STOP. Dump its lowered IR with the probe pattern from the plan's Background section (a `struct`-copy of `built.bindings.get('gv').ir`), compare to the Task-1 IR, and widen `classifyBroadcastFnLogdensity` to also match that shape (same return value). Do not loosen the body match (still require body op `logdensityof` and point === the param).

- [ ] **Step 3: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add packages/engine/test/broadcast-fn-logdensity.test.ts
git commit -m "test: dot-sugar fn(logdensityof(M,_)).(grid) vectorizes too"
```

---

### Task 3: Regression guards — every other broadcast shape unchanged

The new sub-classifier must not perturb value-fn broadcast, kernel broadcast, or the 3-arg logdensity broadcast. Pin them explicitly, then run the whole suite.

**Files:**
- Modify: `packages/engine/test/broadcast-fn-logdensity.test.ts`

- [ ] **Step 1: Add negative/regression tests**

Append to `packages/engine/test/broadcast-fn-logdensity.test.ts`:

```ts
// --- Regression: shapes that must NOT be captured by the new classifier ---

test('regression: value-fn broadcast still maps elementwise', async () => {
  // broadcast(fn(2*_ + 1), A) is a pure-arithmetic value function — body
  // op is `add`/`mul`, not logdensityof, so the new classifier returns
  // null and the existing value-fn path handles it.
  const ctx = mk(`A = [1.0, 2.0, 3.0]\ngv = broadcast(fn(2 * _ + 1), A)`);
  const m = await ctx.getMeasure('gv');
  assert.deepEqual(Array.from(m.samples), [3.0, 5.0, 7.0]);
});

test('regression: kernel broadcast (Normal over params) still samples', async () => {
  // broadcast(Normal, mus, sigmas) is a kernel broadcast — head is a
  // bare distribution ref, not a functionof, so the new classifier
  // returns null. Must still produce a [3]-shaped stochastic measure.
  const ctx = mk(`
mus = [-5.0, 0.0, 5.0]
sigmas = [0.1, 0.1, 0.1]
gv = broadcast(Normal, mus, sigmas)
`);
  const m = await ctx.getMeasure('gv');
  assert.ok(m && (m.samples || m.fields || m.dims), 'kernel broadcast must materialise');
  // mean of each atom-batch near its mu (loose envelope; just a liveness check)
});

test('regression: bare 3-arg broadcast(logdensityof, M, pts) unchanged', async () => {
  const ctx = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = broadcast(logdensityof, mix, grid)`);
  const m = await ctx.getMeasure('gv');
  const ref = await scalarRef();
  for (let i = 0; i < PTS.length; i++) assert.equal(m.samples[i], ref[i]);
});
```

- [ ] **Step 2: Run the new file**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && node --test 'test/broadcast-fn-logdensity.test.ts' 2>&1 | grep -E "^ℹ (tests|pass|fail) "`
Expected: `ℹ tests 6  ℹ pass 6  ℹ fail 0`

> If the kernel-broadcast liveness test fails because the exact `Normal`-broadcast surface differs, replace it with whatever the codebase's existing kernel-broadcast test uses (grep `kernel-broadcast` / `broadcast(Normal` under `test/`) — the point is only to confirm a kernel broadcast still classifies and materialises, not to pin its numerics here.

- [ ] **Step 3: Run the FULL engine suite (the real regression gate)**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && node --test 'test/*.test.ts' 2>&1 | grep -E "^ℹ (tests|pass|fail) "`
Expected: `# fail 0`, and `# pass` = baseline (3117) + 6 new = **3123** (adjust if Task-0 baseline differed). Any pre-existing test that now fails is a regression — STOP and investigate before committing.

- [ ] **Step 4: Verify the cascade-prune guard covers the new bindings**

The fn-form reuses the `broadcast_logdensity` kind, so it inherits whatever `derivationRefsValid` does for that kind. Confirm there is no NEW unguarded path: 

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && grep -n "broadcast_logdensity" derivations.ts`
Expected: the kind is referenced in classification and (if the 3-arg form was guarded) in `derivationRefsValid`/cascade-prune. Since the new classifier produces an IDENTICAL record, no additional guard is required. Record the grep output in the commit/PR notes. (No code change in this step unless the grep shows the 3-arg form had a kind-specific guard that reads a field the fn-form doesn't set — it sets the same two fields, so this is a confirmation step only.)

- [ ] **Step 5: Typecheck**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js && npm run typecheck 2>&1 | tail -1; echo "EXIT:$?"`
Expected: `EXIT:0`

- [ ] **Step 6: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add packages/engine/test/broadcast-fn-logdensity.test.ts
git commit -m "test: regression-guard value-fn / kernel / 3-arg broadcast unchanged"
```

---

### Task 4: Dogfood — speed up the slow integration test

The slowest engine test, `normalized mixture: integrates to 1` (`closed-form-measure-algebra.test.ts`, ~1.5 s standalone), builds 6001 scalar `logdensityof` bindings. Rewrite it to evaluate the density over the grid in ONE broadcast binding — proving the feature and removing the slowest test's cost. Numerics are bit-identical (the density catalogue is the same), so the assertion is unchanged.

**Files:**
- Modify: `packages/engine/test/closed-form-measure-algebra.test.ts` (the `integrates to 1` test, starting `test('normalized mixture: integrates to 1 (trapezoid over a wide grid)'` )

- [ ] **Step 1: Read the current test to get the exact harness it uses**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && sed -n '/integrates to 1/,/^});/p' test/closed-form-measure-algebra.test.ts`
Confirm it uses the file-local `makeCtx(source)` helper (defined at `test/closed-form-measure-algebra.test.ts:30`) and `await c2.getMeasure('p'+i)`.

- [ ] **Step 2: Replace the test body with the vectorized form**

Replace the entire `test('normalized mixture: integrates to 1 (trapezoid over a wide grid)', async () => { ... });` block with:

```ts
test('normalized mixture: integrates to 1 (trapezoid over a wide grid)', async () => {
  // ∫ p_mix(x) dx ≈ 1 for a proper normalized mixture. Evaluate the
  // closed-form density on a grid in ONE binding via vectorized
  // broadcast(fn(logdensityof(mix, _)), grid) — bit-identical to a
  // per-point loop (same logpdf catalogue), but one materialise instead
  // of thousands. Then trapezoid-integrate in JS.
  const lo = -12, hi = 15, n = 1500, h = (hi - lo) / n;
  const ctx = makeCtx(`
A = Normal(mu = -2.0, sigma = 0.8)
B = Normal(mu =  3.0, sigma = 1.3)
mix = normalize(superpose(weighted(0.4, A), weighted(0.6, B)))
grid = linspace(${lo.toFixed(1)}, ${hi.toFixed(1)}, ${n + 1})
logd = broadcast(fn(logdensityof(mix, _)), grid)
`);
  const lp = (await ctx.getMeasure('logd')).samples;
  assert.equal(lp.length, n + 1, `expected ${n + 1} grid densities`);
  let integral = 0;
  for (let i = 0; i <= n; i++) {
    const w = (i === 0 || i === n) ? 0.5 : 1.0;
    integral += w * Math.exp(lp[i]);
  }
  integral *= h;
  assert.ok(Math.abs(integral - 1.0) < 2e-3,
    `normalized mixture must integrate to 1, got ${integral}`);
});
```

Notes for the implementer:
- `linspace(lo, hi, n+1)` yields `n+1` endpoint-exact points (the engine's `linspace` is endpoint-exact; `ext-linalg`/`sampler.ts:1017`). `h = (hi-lo)/n` matches the trapezoid spacing.
- `n = 1500` (down from 6000) keeps the integral error at ~machine epsilon (verified: error stays < 1e-12 for n ≥ 200; tolerance is 2e-3). Do not raise the tolerance.
- The grid binding must be named (`grid = linspace(...)`) so it is a fixed-phase binding the points-resolver can read.

- [ ] **Step 3: Run the rewritten test**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && node --test --test-reporter=spec 'test/closed-form-measure-algebra.test.ts' 2>&1 | grep "integrates to 1"`
Expected: `✔ normalized mixture: integrates to 1 (trapezoid over a wide grid) (Xms)` where **X is well under 200 ms** (was ~1534 ms). It must PASS.

- [ ] **Step 4: Run the whole file to confirm no neighbour broke**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-js/packages/engine && node --test 'test/closed-form-measure-algebra.test.ts' 2>&1 | grep -E "^ℹ (tests|pass|fail) "`
Expected: `# fail 0`.

- [ ] **Step 5: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add packages/engine/test/closed-form-measure-algebra.test.ts
git commit -m "test: vectorize the mixture-integration test via broadcast logdensityof

Was 6001 scalar logdensityof bindings (~1.5s, the slowest engine test).
Now one broadcast(fn(logdensityof(mix,_)), grid) binding over a 1501-pt
linspace — bit-identical densities, ~25x faster. Tolerance unchanged
(integral error stays at machine epsilon)."
```

---

### Task 5 (optional, low-priority): Document the surface

Add a one-paragraph note + example so the vectorized density-over-grid surface is discoverable. Keep it minimal; skip if docs scope is undesired.

**Files:**
- Modify: `flatppl-design/docs/06-measure-algebra.md` (the section that introduces `logdensityof`, under "Operations that map measures to values")

- [ ] **Step 1: Add a note after the `logdensityof(M, x)` description**

After the sentence ``logdensityof(M, x)` evaluate the density of a measure at a point`` (06-measure-algebra.md → "Operations that map measures to values"), add:

```markdown

To evaluate a density at many points (e.g. a grid for numerical
integration or plotting), broadcast the operation rather than calling it
per point: `broadcast(fn(logdensityof(M, _)), grid)` (equivalently
`fn(logdensityof(M, _)).(grid)`) returns one log-density per grid
element. This is the standard [broadcasting](04-design.md#sec:broadcasting)
mechanism applied to a measure→value op; the point argument stays scalar.
```

- [ ] **Step 2: Sanity-check the doc renders / no broken anchor**

Run: `cd /Users/benjamincox/Code/work/flatppl/flatppl-design/docs && grep -n "sec:broadcasting" 04-design.md | head -1`
Expected: a line showing the `sec:broadcasting` anchor exists (so the link target is valid).

- [ ] **Step 3: Commit**

```bash
cd /Users/benjamincox/Code/work/flatppl/flatppl-js
git add ../flatppl-design/docs/06-measure-algebra.md
git commit -m "docs: note broadcast(fn(logdensityof(M,_)), grid) for grid density eval"
```

> NOTE: `flatppl-design` may be a separate git repo from `flatppl-js`. If `git add` fails because the docs live outside this repo, commit there separately (and only if pushing that repo is in scope — otherwise skip Task 5 and just mention the surface in the PR description).

---

## Done criteria

- `broadcast(fn(logdensityof(M, _)), grid)` and `fn(logdensityof(M, _)).(grid)` materialise to a length-`|grid|` value measure, bit-exact to the scalar loop AND to the existing 3-arg form.
- Value-fn broadcast, kernel broadcast, and 3-arg `broadcast(logdensityof, M, pts)` are unchanged (explicit regression tests + full-suite green).
- Full engine suite `# fail 0`, pass = baseline + 6; `npm run typecheck` exit 0.
- The mixture-integration test runs in <200 ms (was ~1534 ms) with the same assertion.
- Commits on `broadcast-fn-logdensity`; **no push** (await instruction).

## No-regression rationale (why this is low-risk)

- The ONLY production change is one new pure function (`classifyBroadcastFnLogdensity`) and a one-line insertion into `classifyBroadcast`. It can only ADD a derivation for a shape that currently ERRORS (`kind:'evaluate'` → runtime throw); it removes no behavior.
- The new classifier matches an extremely narrow shape (2-arg broadcast; inline functionof of one param; body is exactly `logdensityof(self-measure, that-param)`). Value-fn bodies (`add`/`mul`/…), kernel-ref heads, and the 3-arg bare-builtin form all return `null` from it and reach their existing classifiers untouched.
- It returns the IDENTICAL `DerivationBroadcastLogdensity` record the 3-arg form already produces, so `matBroadcastLogdensity`, the worker `logDensityN`/`pointsBatched` path, and `measureFromValue` run unchanged and already-tested.

## Notes for the executor

- **Do not push.** The user controls push/PR.
- TDD: write each test, watch it fail (Task 1 Step 2), then implement. Don't implement before the red.
- If any full-suite test regresses, STOP — the narrow classifier should make that impossible; a regression means the match is too broad. Re-check the `body.op === 'logdensityof'` and `xIR.name === paramName` guards.
- Keep `classifyBroadcastFnLogdensity` a pure function (no side effects, no `bindings` mutation), matching its siblings.

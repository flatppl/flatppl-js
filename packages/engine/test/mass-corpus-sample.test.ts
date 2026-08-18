'use strict';

// =====================================================================
// mass-corpus-sample.test.ts — a sample of the defect-2 fix's real blast
// radius, pinned against the actual corpus fixtures
// =====================================================================
//
// `mass-published-object.test.ts` pins the two synthetic repros the original
// finding and the wave report describe. Neither touches the real corpus.
// Swept over all 38 `test/fixtures/*.flatppl`, the copy-on-write fix changes
// the PUBLISHED mass class of 29 of 130 measure/kernel bindings — a change to
// FlatPIR `%meta` output for most corpus models, not only to the two
// synthetic bindings the report names. This file pins three of those, one
// per distinct kind of change, so a partial revert of the copy-on-write
// cannot go undetected by the rest of the suite:
//
//   - `finite` → `normalized` (18+2 of the 29 are a class getting corrected
//     upward once it is no longer being overwritten by an unrelated sibling
//     that shared its pre-fix type object)
//   - `unknown` → `locallyfinite` (an improper-flat prior joint, sharper)
//   - a kernel's `result.mass` UNSET → `normalized` (base emitted no `%mass`
//     slot on the kernel at all)
//
// Every direction below was checked against spec §11's own class
// definitions, not assumed correct because it changed — see
// flatppl-dev/TODO-flatppl-js.md's defect-2 entry for the derivation of each.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource } = require('../index.ts');

const FIXTURES = path.join(__dirname, 'fixtures');

function bindingOf(file: string, name: string): any {
  const src = fs.readFileSync(path.join(FIXTURES, file), 'utf8');
  return processSource(src).loweredModule.bindings.get(name);
}

test('hadron-physics-resonance.flatppl: D1 = normalize(weighted(...)) '
  + 'publishes %normalized, not the enclosing mixture\'s %finite', () => {
  // `normalize` yields total mass one by construction (spec §06) whatever the
  // class of its argument — D1's own class does not depend on anything else
  // in the module. Pre-fix it read `finite`, stamped by `mixture`'s own
  // `weighted(f1, D1)` classifying AFTER D1 in the same shared type object.
  const b = bindingOf('hadron-physics-resonance.flatppl', 'D1');
  assert.equal(b.inferredType.kind, 'measure');
  assert.equal(b.inferredType.mass, 'normalized');
});

test('flatppl-uncorrelated_background-ma-priors.flatppl: the improper-flat '
  + 'prior joint publishes %locallyfinite, sharper than the pre-fix %unknown',
  () => {
    // The fixture's own comment calls this "improper flat on mu" —
    // §11: %locallyfinite is "infinite total mass, but finite mass on every
    // bounded set (e.g. Lebesgue(reals))", which is exactly what an
    // unbounded-support component of a joint gives. `unknown` claims nothing;
    // `locallyfinite` is a strictly stronger, correct classification.
    const b = bindingOf('flatppl-uncorrelated_background-ma-priors.flatppl',
      'prior');
    assert.equal(b.inferredType.kind, 'measure');
    assert.equal(b.inferredType.mass, 'locallyfinite');
  });

test('bayesian_inference_4.flatppl: a restrict-derived kernel publishes '
  + 'result.mass = %normalized instead of leaving the slot unset', () => {
  // Base emitted this kernel with NO %mass slot at all (not %deferred,
  // literally absent from the object) — the aliased result object was never
  // reached by a stamp before something else's mutation raced it. Any
  // binding matching this synthetic name is the same shape; the exact
  // counter suffix is not load-bearing.
  const src = fs.readFileSync(
    path.join(FIXTURES, 'bayesian_inference_4.flatppl'), 'utf8');
  const { loweredModule } = processSource(src);
  const restrictKernels = [...loweredModule.bindings.entries()]
    .filter(([name, b]: [string, any]) => /^__restrict_kernel_\d+$/.test(name)
      && b.inferredType && b.inferredType.kind === 'kernel');
  assert.ok(restrictKernels.length > 0, 'fixture no longer synthesizes a '
    + 'restrict-derived kernel binding — pick a new witness');
  for (const [name, b] of restrictKernels) {
    assert.equal(b.inferredType.result && b.inferredType.result.mass,
      'normalized', name + ' should publish result.mass = %normalized');
  }
});

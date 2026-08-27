'use strict';

// =====================================================================
// corpus-diagnostics.test.ts — every fixture analyses without errors
// =====================================================================
//
// The rest of the fixture suite asserts SHAPES: a derivation kind, an
// atom shape, a density number. None of it reads `diagnostics`, so a
// fixture can acquire a static error and stay green indefinitely — which
// is how five models came to draw from an unnormalized measure (spec §04)
// without any test noticing, and it took a hand-run sweep to find them.
//
// This closes that: `processSource` over every `test/fixtures/*.flatppl`,
// asserting the exact set of error-severity messages. Clean is the
// default; the exceptions below are pinned WITH their messages, so a
// known-bad fixture cannot quietly acquire a NEW error either.
//
// When a fixture legitimately changes, update its entry (or delete it, if
// the fix made it clean). When a fixture starts erroring, the model or the
// engine is wrong — do not add it here to silence the test.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource } = require('../index.ts');

const FIXTURES = path.join(__dirname, 'fixtures');

// filename → the sorted unique error messages it is KNOWN to produce.
// Every one of these is a pre-existing defect unrelated to what any model
// says about measures; none is a mass or normalization diagnostic.
// The map is EMPTY and should stay that way: the last four entries (stacked
// `%` doc lines, and prose whose `;` ended a `#` line comment and left the
// rest to parse as a binding) were fixed with the residual `kernelof`
// boundary-input migration. A new entry needs the same justification the
// header demands — a defect the fixture cannot yet avoid, never a silencer.
const KNOWN_ERRORS: Record<string, string[]> = {};

function errorMessagesOf(file: string): string[] {
  const src = fs.readFileSync(path.join(FIXTURES, file), 'utf8');
  const { diagnostics } = processSource(src);
  const messages: string[] = (diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => String(d.message));
  return [...new Set(messages)].sort();
}

const FIXTURE_FILES = fs.readdirSync(FIXTURES)
  .filter((f: string) => f.endsWith('.flatppl')).sort();

test('the fixture corpus is non-empty (the walk itself must not silently pass)', () => {
  assert.ok(FIXTURE_FILES.length >= 30,
    `expected the fixture corpus, found ${FIXTURE_FILES.length} .flatppl files`);
});

for (const file of FIXTURE_FILES) {
  const expected = KNOWN_ERRORS[file] || [];
  const label = expected.length
    ? `${file} produces only its ${expected.length} known error(s)`
    : `${file} analyses with no error diagnostics`;
  test(label, () => {
    assert.deepEqual(errorMessagesOf(file), expected);
  });
}

// Spec §11 "Reified callables" admits two spellings for a boundary input: a
// ref into the ancestor subgraph, or a placeholder within the reified output.
// A bare unbound name is neither, and `lower.ts` still accepts it as an
// off-spec shorthand behind an `info` diagnostic. The corpus carries no
// remaining instance, so pin that — a new fixture must not reintroduce one.
test('no fixture spells a kernel boundary input as a bare unbound name', () => {
  const offenders: string[] = [];
  for (const file of FIXTURE_FILES) {
    const src = fs.readFileSync(path.join(FIXTURES, file), 'utf8');
    const { diagnostics } = processSource(src);
    for (const d of diagnostics || []) {
      if (/Bare-name boundary input/.test(String(d.message))) {
        offenders.push(`${file}: ${d.message}`);
      }
    }
  }
  assert.deepEqual(offenders, []);
});

test('every KNOWN_ERRORS entry names a fixture that still exists', () => {
  for (const file of Object.keys(KNOWN_ERRORS)) {
    assert.ok(FIXTURE_FILES.includes(file),
      `KNOWN_ERRORS names ${file}, which is not in test/fixtures/ — drop the entry`);
  }
});

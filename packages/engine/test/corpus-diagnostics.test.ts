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
const KNOWN_ERRORS: Record<string, string[]> = {
  // Doc-comment style: consecutive single-line `#` comments where the
  // parser wants one `%%%` block.
  'hierarchical-state-space.flatppl': [
    'Only one doc-comment may precede a binding (use a `%%%` block for multi-line content)',
  ],
  'joint-obs-regression.flatppl': [
    'Only one doc-comment may precede a binding (use a `%%%` block for multi-line content)',
  ],
  'nested-broadcast.flatppl': [
    'Only one doc-comment may precede a binding (use a `%%%` block for multi-line content)',
  ],
  'random-intercepts.flatppl': [
    'Only one doc-comment may precede a binding (use a `%%%` block for multi-line content)',
  ],
  // Prose in the header comment block parses as bindings.
  'nested-broadcast-mvnormal-inner.flatppl': [
    "Expected EQUALS, got IDENT 'lands'",
    "Expected EQUALS, got NUMBER '5'",
  ],
};

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

test('every KNOWN_ERRORS entry names a fixture that still exists', () => {
  for (const file of Object.keys(KNOWN_ERRORS)) {
    assert.ok(FIXTURE_FILES.includes(file),
      `KNOWN_ERRORS names ${file}, which is not in test/fixtures/ — drop the entry`);
  }
});

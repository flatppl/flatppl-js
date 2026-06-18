// Unit tests for the pure inference-lens title helper. Lives in
// src/lspHelpers.ts (no vscode import), so it is required directly with no
// VS Code host stubbing — same pattern as the other helper tests.
const test = require('node:test');
const assert = require('node:assert/strict');
const { inferenceLensTitle } = require('../src/lspHelpers.js');

test('string label: strips leading ": " and prefixes the glyph', () => {
  assert.equal(inferenceLensTitle(': Real'), '▷ Real');
});

test('label parts: joins values then strips and prefixes', () => {
  assert.equal(
    inferenceLensTitle([{ value: ': ' }, { value: 'Real' }]),
    '▷ Real',
  );
});

test('tolerates a bare colon with no space', () => {
  assert.equal(inferenceLensTitle(':Real'), '▷ Real');
});

test('tolerates a label with no leading colon', () => {
  assert.equal(inferenceLensTitle('Real'), '▷ Real');
});

test('custom glyph is honored', () => {
  assert.equal(inferenceLensTitle(': Real', '>'), '> Real');
});

test('empty / whitespace-only type returns empty string', () => {
  assert.equal(inferenceLensTitle(': '), '');
  assert.equal(inferenceLensTitle(''), '');
  assert.equal(inferenceLensTitle([]), '');
});

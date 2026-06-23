// Tests for the shared file-type / tree-bucket decisions (src/file-types.mjs).
// This module is the single source of truth consumed by both build.mjs
// (manifest generation) and the runtime scripts (surfaces.ts, app.ts), so
// locking its behaviour here guards the drift ../../CLAUDE.md warns about.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  MODEL_EXTENSIONS,
  typeForPath,
  bucketKeyForPath,
  defaultFolderOpen,
  hideTestExamplesEnv,
} from '../src/file-types.mjs';

test('typeForPath maps extensions to surface types', () => {
  assert.equal(typeForPath('examples/minimal.flatppl'), 'flatppl');
  assert.equal(typeForPath('demo/notes.md'), 'markdown');
  assert.equal(typeForPath('readme.markdown'), 'markdown');
  // .hs3.json / .pyhf.json are distinct types (NOT generic json), and the
  // double extension must win over a naive `.json` check.
  assert.equal(typeForPath('examples/hs3/gaussian.hs3.json'), 'hs3');
  assert.equal(typeForPath('uploads/model.pyhf.json'), 'pyhf');
  // Plain .json / unrecognised extensions are 'unknown' (placeholder surface).
  assert.equal(typeForPath('data.json'), 'unknown');
  assert.equal(typeForPath('notes.txt'), 'unknown');
  // A bare .hs3 (no .json) is NOT recognised — documents the current corpus
  // mismatch rather than silently classifying it.
  assert.equal(typeForPath('examples/hs3/gaussian.hs3'), 'unknown');
});

test('typeForPath is case-insensitive and FlatPPL-defaults on empty', () => {
  assert.equal(typeForPath('MODEL.FLATPPL'), 'flatppl');
  assert.equal(typeForPath('A.HS3.JSON'), 'hs3');
  assert.equal(typeForPath(''), 'flatppl');
  assert.equal(typeForPath(null), 'flatppl');
  assert.equal(typeForPath(undefined), 'flatppl');
});

test('every MODEL_EXTENSIONS entry classifies to a non-unknown type', () => {
  // The build only lists files whose extension is in MODEL_EXTENSIONS; each
  // such file must route to a real surface type, or the gallery would surface
  // a file it can't place. (Guards adding an extension without a type arm.)
  for (const ext of MODEL_EXTENSIONS) {
    assert.notEqual(typeForPath('x' + ext), 'unknown', `extension ${ext}`);
  }
});

test('bucketKeyForPath groups manifest entries into folders', () => {
  assert.equal(bucketKeyForPath('examples/minimal.flatppl'), 'examples');
  assert.equal(bucketKeyForPath('examples/hs3/gaussian.flatppl'), 'hs3');
  // HS3 must win over the generic examples/ prefix.
  assert.equal(bucketKeyForPath('examples/linear-regression.flatppl'), 'examples');
  // demo/ is the legacy prefix for the "Test cases" folder; test-cases/ is
  // the current one — both land in test-cases.
  assert.equal(bucketKeyForPath('demo/feature-test1.flatppl'), 'test-cases');
  assert.equal(bucketKeyForPath('test-cases/whatever.flatppl'), 'test-cases');
  assert.equal(bucketKeyForPath(''), 'examples');
});

test('defaultFolderOpen: examples open, user opens with content, rest closed', () => {
  assert.equal(defaultFolderOpen('examples', 0), true);
  assert.equal(defaultFolderOpen('examples', 5), true);
  assert.equal(defaultFolderOpen('user', 0), false);
  assert.equal(defaultFolderOpen('user', 3), true);
  assert.equal(defaultFolderOpen('hs3', 9), false);
  assert.equal(defaultFolderOpen('test-cases', 9), false);
});

test('hideTestExamplesEnv treats truthy non-off values as on', () => {
  assert.equal(hideTestExamplesEnv('1'), true);
  assert.equal(hideTestExamplesEnv('true'), true);
  assert.equal(hideTestExamplesEnv('yes'), true);
  // explicit off-ish values read as off even though non-empty
  assert.equal(hideTestExamplesEnv('0'), false);
  assert.equal(hideTestExamplesEnv('off'), false);
  assert.equal(hideTestExamplesEnv('false'), false);
  assert.equal(hideTestExamplesEnv(''), false);
  assert.equal(hideTestExamplesEnv(undefined), false);
});

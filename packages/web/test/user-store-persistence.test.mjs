import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

const source = stripTypeScriptTypes(readFileSync(new URL('../src/user-store.ts', import.meta.url), 'utf8'));
const indexKey = 'flatppl-web:user-files';

function storage() {
  const values = new Map();
  return {
    fail: () => false,
    getItem: key => values.get(key) ?? null,
    setItem(key, value) {
      if (this.fail(key)) throw new Error('QuotaExceededError');
      values.set(key, value);
    },
    removeItem: key => values.delete(key),
  };
}

function userStore(localStorage) {
  const window = { localStorage };
  vm.runInNewContext(source, { window, console });
  return window.FlatPPLWebUserStore;
}

for (const boundary of ['entry', 'index']) {
  test('a failed new ' + boundary + ' write retries the same source durably', () => {
    const ls = storage();
    const store = userStore(ls);
    ls.fail = key => boundary === 'index' ? key === indexKey : key !== indexKey;
    assert.equal(store.save('user/new.flatppl', 'x = 1'), false);
    assert.equal(store.getSource('user/new.flatppl'), 'x = 1');
    assert.equal(userStore(ls).getSource('user/new.flatppl'), null);
    assert.equal(store.isPersisted('user/new.flatppl'), false);
    assert.equal(store.hasUnpersisted(), true);
    ls.fail = () => false;
    assert.equal(store.updateSource('user/new.flatppl', 'x = 1'), true);
    assert.equal(userStore(ls).getSource('user/new.flatppl'), 'x = 1');
    assert.equal(store.isPersisted('user/new.flatppl'), true);
    assert.equal(store.hasUnpersisted(), false);
  });

  test('a failed existing ' + boundary + ' write preserves the saved source for Revert', () => {
    const ls = storage();
    const store = userStore(ls);
    store.save('user/model.flatppl', 'x = 1', { parent: 'examples/model.flatppl' });
    ls.fail = key => boundary === 'index' ? key === indexKey : key !== indexKey;
    assert.equal(store.updateSource('user/model.flatppl', 'x = 2'), false);
    assert.equal(store.getSource('user/model.flatppl'), 'x = 2');
    assert.equal(userStore(ls).getSource('user/model.flatppl'), 'x = 1');
    assert.equal(store.isPersisted('user/model.flatppl'), false);
    assert.equal(store.revert('user/model.flatppl'), true);
    assert.equal(store.getSource('user/model.flatppl'), 'x = 1');
    assert.equal(store.get('user/model.flatppl').parent, 'examples/model.flatppl');
    assert.equal(store.hasUnpersisted(), false);
  });
}

test('saving another file does not publish a failed draft', () => {
  const ls = storage();
  const store = userStore(ls);
  ls.fail = () => true;
  store.save('user/draft.flatppl', 'x = 1');
  ls.fail = () => false;
  assert.equal(store.save('user/saved.flatppl', 'x = 2'), true);
  assert.deepEqual(Array.from(userStore(ls).list(), entry => entry.path), ['user/saved.flatppl']);
  assert.equal(store.hasUnpersisted(), true);
  assert.equal(store.updateSource('user/draft.flatppl', 'x = 1'), true);
  assert.deepEqual(Array.from(userStore(ls).list(), entry => entry.path), ['user/draft.flatppl', 'user/saved.flatppl']);
  assert.equal(store.hasUnpersisted(), false);
});

test('Revert removes a failed new save and its unload warning', () => {
  const store = userStore(null);
  store.save('user/draft.flatppl', 'x = 1');
  assert.equal(store.hasUnpersisted(), true);
  assert.equal(store.revert('user/draft.flatppl'), false);
  assert.equal(store.getSource('user/draft.flatppl'), null);
  assert.equal(store.list().length, 0);
  assert.equal(store.hasUnpersisted(), false);
});

test('rename preserves a pending update on failure and saves it on success', () => {
  const ls = storage();
  const store = userStore(ls);
  store.save('user/old.flatppl', 'x = 1');
  ls.fail = () => true;
  store.updateSource('user/old.flatppl', 'x = 2');
  assert.equal(store.rename('user/old.flatppl', 'user/new.flatppl'), false);
  assert.equal(store.getSource('user/old.flatppl'), 'x = 2');
  assert.equal(userStore(ls).getSource('user/old.flatppl'), 'x = 1');
  assert.equal(store.hasUnpersisted(), true);
  ls.fail = () => false;
  assert.equal(store.rename('user/old.flatppl', 'user/new.flatppl'), true);
  assert.equal(userStore(ls).getSource('user/new.flatppl'), 'x = 2');
  assert.equal(userStore(ls).getSource('user/old.flatppl'), null);
  assert.equal(store.isPersisted('user/new.flatppl'), true);
  assert.equal(store.hasUnpersisted(), false);
});

test('session-only rename retains the draft warning until Revert', () => {
  const store = userStore(null);
  store.save('user/old.flatppl', 'x = 1');
  assert.equal(store.rename('user/old.flatppl', 'user/new.flatppl'), true);
  assert.equal(store.getSource('user/new.flatppl'), 'x = 1');
  assert.equal(store.isPersisted('user/new.flatppl'), false);
  assert.equal(store.hasUnpersisted(), true);
  assert.equal(store.revert('user/new.flatppl'), false);
  assert.equal(store.hasUnpersisted(), false);
});

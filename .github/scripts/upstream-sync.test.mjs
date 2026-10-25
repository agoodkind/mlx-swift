#!/usr/bin/env node

import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const helper = fileURLToPath(new URL('./upstream-sync.mjs', import.meta.url));
const branch = 'sync/upstream-latest';
const state = '.github/upstream-sync-state';

function git(directory, ...arguments_) {
  return execFileSync('git', ['-C', directory, ...arguments_], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null' },
  }).trim();
}

function initialize(directory, bare = false) {
  mkdirSync(directory);
  const arguments_ = ['init', '--initial-branch=main'];
  if (bare) arguments_.push('--bare');
  git(directory, ...arguments_);
  if (!bare) {
    git(directory, 'config', 'user.name', 'Sync Test');
    git(directory, 'config', 'user.email', 'sync-test@example.invalid');
  }
}

function commit(directory, filename, contents) {
  writeFileSync(join(directory, filename), contents);
  git(directory, 'add', filename);
  git(directory, 'commit', '-m', `Update ${filename}`);
  return git(directory, 'rev-parse', 'HEAD');
}

test('synchronization preserves repairs and rejects conflicts before publishing', () => {
  const root = mkdtempSync(join(tmpdir(), 'upstream-sync-'));
  const origin = join(root, 'origin.git');
  const upstream = join(root, 'upstream');
  const main = join(root, 'main');
  const worker = join(root, 'worker');
  initialize(origin, true);
  initialize(upstream);
  initialize(main);
  mkdirSync(join(main, '.github'));
  commit(main, 'fork.txt', 'fork\n');
  commit(main, 'shared.txt', 'original\n');
  git(main, 'remote', 'add', 'origin', origin);
  git(main, 'push', 'origin', 'main');
  commit(upstream, 'upstream.txt', 'upstream\n');
  const firstUpstream = commit(upstream, 'shared.txt', 'original\n');
  git(root, 'clone', origin, worker);
  git(worker, 'config', 'user.name', 'Sync Test');
  git(worker, 'config', 'user.email', 'sync-test@example.invalid');

  const run = () => spawnSync(process.execPath, [helper, 'sync'], {
    cwd: worker,
    encoding: 'utf8',
    env: { ...process.env, SYNC_UPSTREAM_URL: upstream, GITHUB_OUTPUT: '',
      GIT_CONFIG_NOSYSTEM: '1', GIT_CONFIG_GLOBAL: '/dev/null' },
  });
  const succeed = () => {
    const result = run();
    assert.equal(result.status, 0, result.stderr);
    return result;
  };
  const remoteHead = () => git(origin, 'rev-parse', `refs/heads/${branch}`);

  succeed();
  assert.equal(readFileSync(join(worker, state), 'utf8').trim(), firstUpstream);
  assert.equal(git(worker, 'show', 'HEAD:fork.txt'), 'fork');
  assert.equal(git(worker, 'show', 'HEAD:upstream.txt'), 'upstream');
  assert.equal(remoteHead(), git(worker, 'rev-parse', 'HEAD'));
  assert.doesNotMatch(git(worker, 'log', '--format=%B', 'origin/main..HEAD'), /\[skip ci\]/);

  const repair = commit(worker, 'manual-repair.txt', 'preserve this repair\n');
  git(worker, 'push', 'origin', `HEAD:refs/heads/${branch}`);
  const mainUpdate = commit(main, 'main-update.txt', 'current main\n');
  git(main, 'push', 'origin', 'main');
  const nextUpstream = commit(upstream, 'upstream-update.txt', 'new upstream\n');
  succeed();
  git(worker, 'merge-base', '--is-ancestor', repair, 'HEAD');
  git(worker, 'merge-base', '--is-ancestor', mainUpdate, 'HEAD');
  git(worker, 'merge-base', '--is-ancestor', nextUpstream, 'HEAD');
  assert.equal(git(worker, 'rev-parse', 'HEAD~1^2'), nextUpstream);
  assert.equal(git(worker, 'rev-parse', 'HEAD~2^2'), mainUpdate);
  assert.equal(git(worker, 'show', 'HEAD:manual-repair.txt'), 'preserve this repair');
  assert.equal(readFileSync(join(worker, state), 'utf8').trim(), nextUpstream);

  const unchangedHead = remoteHead();
  succeed();
  assert.equal(remoteHead(), unchangedHead);
  assert.equal(git(worker, 'rev-parse', 'HEAD'), unchangedHead);

  const conflictingRepair = commit(worker, 'shared.txt', 'fork repair\n');
  git(worker, 'push', 'origin', `HEAD:refs/heads/${branch}`);
  commit(upstream, 'shared.txt', 'conflicting upstream\n');
  const failed = run();
  assert.notEqual(failed.status, 0);
  assert.match(failed.stderr, /CONFLICT/);
  assert.match(failed.stderr, /Upstream synchronization failed/);
  assert.equal(remoteHead(), conflictingRepair);
  assert.equal(readFileSync(join(worker, state), 'utf8').trim(), nextUpstream);
  assert.notEqual(git(worker, 'ls-files', '--unmerged'), '');
});

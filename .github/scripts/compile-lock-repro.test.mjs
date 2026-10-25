#!/usr/bin/env node

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

const binary = process.env.COMPILE_LOCK_REPRO_BINARY;
assert.ok(binary, 'Set COMPILE_LOCK_REPRO_BINARY to the built diagnostic executable.');
const outputDirectory = mkdtempSync(join(tmpdir(), 'compile-lock-repro-'));

function run(arguments_, disabled) {
  const environment = { ...process.env, LLVM_PROFILE_FILE: join(outputDirectory, 'profile-%p.profraw') };
  delete environment.MLX_DISABLE_COMPILE;
  if (disabled) environment.MLX_DISABLE_COMPILE = '1';
  const result = spawnSync(binary, arguments_, { env: environment, encoding: 'utf8', timeout: 30000 });
  if (result.error) throw result.error;
  return result;
}

for (const disabled of [false, true]) {
  test(`completed stress reports inconclusive when compilation is ${disabled ? 'disabled' : 'enabled'}`, () => {
    const result = run(['--device', 'cpu'], disabled);
    assert.equal(result.status, 2, result.stderr + result.stdout);
    assert.match(result.stdout, /INCONCLUSIVE:/);
    assert.match(result.stdout, /20 stress rounds completed/);
    assert.doesNotMatch(result.stdout, /^PASS:/m);
    assert.doesNotMatch(result.stdout, /enteredTrace/);
  });
}

test('the device option rejects a missing value', () => {
  const result = run(['--device'], false);
  assert.equal(result.status, 2, result.stderr + result.stdout);
  assert.match(result.stdout, /--device requires cpu or gpu/);
});

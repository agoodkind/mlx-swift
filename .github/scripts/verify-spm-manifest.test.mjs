#!/usr/bin/env node

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const verifier = fileURLToPath(new URL('../../scripts/verify-spm-manifest.py', import.meta.url));
const fixture = readFileSync(new URL('../../Tests/ManifestValidationTests/Package.swift', import.meta.url), 'utf8');
const linux = process.platform === 'linux';

function verify(pluginName) {
  const directory = mkdtempSync(join(tmpdir(), 'manifest-validation-'));
  writeFileSync(join(directory, 'Package.swift'), fixture.replaceAll('__PLUGIN_NAME__', pluginName));
  const result = spawnSync('python3', [verifier], { cwd: directory, encoding: 'utf8' });
  if (result.error) throw result.error;
  return result;
}

test('the platform manifest accepts the required plugin identity', () => {
  const result = verify(linux ? 'CudaBuild' : 'OtherBuild');
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /SwiftPM manifest CUDA containment passed/);
});

test('the platform manifest rejects an invalid CUDA plugin binding', () => {
  const result = verify(linux ? 'OtherBuild' : 'CudaBuild');
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /CUDA build (plugin|targets)/);
});

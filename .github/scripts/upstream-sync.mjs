#!/usr/bin/env node

import { execFileSync } from 'node:child_process';
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';

const candidateBranch = 'sync/upstream-latest';
const stateFile = '.github/upstream-sync-state';

function command(executable, arguments_) {
  try {
    return execFileSync(executable, arguments_, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'inherit'],
    }).trim();
  } catch (error) {
    if (error.stdout) process.stderr.write(error.stdout);
    throw error;
  }
}

function git(...arguments_) {
  return command('git', arguments_);
}

function requiredEnvironment(name) {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Set ${name} before running upstream synchronization.`);
  }
  return value;
}

function readState() {
  try {
    return readFileSync(stateFile, 'utf8').trim();
  } catch (error) {
    if (error.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

function output(name, value) {
  console.log(`${name}=${value}`);
  if (process.env.GITHUB_OUTPUT) {
    appendFileSync(process.env.GITHUB_OUTPUT, `${name}=${value}\n`);
  }
}

function sync() {
  const upstreamURL = requiredEnvironment('SYNC_UPSTREAM_URL');
  if (git('status', '--porcelain')) {
    throw new Error('Commit or remove working tree changes before synchronizing.');
  }

  git('fetch', '--no-tags', 'origin');
  const candidateRef = `refs/remotes/origin/${candidateBranch}`;
  const previousCandidate = git('for-each-ref', '--format=%(objectname)', candidateRef);
  git('fetch', '--no-tags', upstreamURL, 'main');
  const upstreamSHA = git('rev-parse', 'FETCH_HEAD');
  git('switch', '--detach', previousCandidate || 'origin/main');
  git('merge', '--no-edit', '-m', 'Merge main into upstream candidate', 'origin/main');

  if (readState() !== upstreamSHA) {
    git('merge', '--no-edit', '--allow-unrelated-histories',
      '-m', 'Merge upstream main into sync candidate', upstreamSHA);
    mkdirSync('.github', { recursive: true });
    writeFileSync(stateFile, `${upstreamSHA}\n`);
    git('add', stateFile);
    git('commit', '-m', `Record upstream main at ${upstreamSHA}`);
  }

  const headSHA = git('rev-parse', 'HEAD');
  const hasChanges = git('rev-list', '--count', 'origin/main..HEAD') !== '0';
  if (hasChanges && headSHA !== previousCandidate) {
    git('push', 'origin', `HEAD:refs/heads/${candidateBranch}`);
  }
  output('has_changes', hasChanges);
  output('head_sha', headSHA);
  output('upstream_sha', upstreamSHA);
}

function updatePullRequest() {
  requiredEnvironment('GH_TOKEN');
  const repository = requiredEnvironment('GITHUB_REPOSITORY');
  const pullRequests = JSON.parse(command('gh', [
    'pr', 'list', '--repo', repository, '--base', 'main', '--head', candidateBranch,
    '--state', 'open', '--json', 'number',
  ]));
  if (pullRequests.length > 1) {
    throw new Error(`Resolve duplicate open pull requests for ${candidateBranch}.`);
  }
  const title = 'Sync Apple mlx-swift updates';
  const body = `Merge Apple mlx-swift main at ${readFileSync(stateFile, 'utf8').trim()}.\n\n`
    + 'The candidate preserves existing repair commits and merges current main before upstream. '
    + 'Conflicts stop synchronization. Review the diff and require successful build and integration checks.';
  if (pullRequests.length === 1) {
    command('gh', ['pr', 'edit', String(pullRequests[0].number), '--repo', repository,
      '--title', title, '--body', body]);
  } else {
    command('gh', ['pr', 'create', '--repo', repository, '--base', 'main',
      '--head', candidateBranch, '--title', title, '--body', body]);
  }
}

try {
  if (process.argv[2] === 'sync') {
    sync();
  } else if (process.argv[2] === 'pr') {
    updatePullRequest();
  } else {
    throw new Error('Run upstream-sync.mjs with sync or pr.');
  }
} catch (error) {
  console.error(`Upstream synchronization failed: ${error.message}`);
  process.exitCode = 1;
}

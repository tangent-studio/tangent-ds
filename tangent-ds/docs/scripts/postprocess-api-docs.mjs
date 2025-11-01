import {promises as fs} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const apiRoot = path.resolve(__dirname, '..', 'docs', 'api');

const SPECIAL_LABELS = new Map([
  ['api', 'API Reference'],
  ['ds', '@tangent.to/ds'],
  ['ml', 'ML'],
  ['mva', 'MVA'],
]);

async function listMarkdownFiles(dir) {
  const entries = await fs.readdir(dir, {withFileTypes: true});
  const files = await Promise.all(
    entries.map(async entry => {
      const resolved = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        return listMarkdownFiles(resolved);
      }
      return entry.isFile() && entry.name.endsWith('.md') ? resolved : [];
    }),
  );
  return files.flat();
}

function buildDocId(relativePath) {
  const segments = relativePath.replace(/\\/g, '/').replace(/\.md$/, '').split('/');
  if (segments[segments.length - 1] === 'index') {
    segments.pop();
  }
  const filtered = segments.filter(Boolean);
  if (filtered.length === 0) {
    return 'api';
  }
  // Use dashes instead of slashes to avoid Docusaurus error
  return ['api', ...filtered].join('-');
}

function labelFromSegments(relativePath) {
  const segments = relativePath.replace(/\\/g, '/').replace(/\.md$/, '').split('/');
  let labelSegment = segments[segments.length - 1] ?? 'api';
  if (labelSegment === 'index' && segments.length > 1) {
    labelSegment = segments[segments.length - 2];
  } else if (labelSegment === 'index') {
    labelSegment = 'api';
  }
  const normalized = labelSegment.toLowerCase();
  if (SPECIAL_LABELS.has(normalized)) {
    return SPECIAL_LABELS.get(normalized);
  }
  return labelSegment;
}

function escapeYaml(value) {
  const needsQuotes = /[:{}\[\],&*#?|\-<>=!%@`]/.test(value);
  const escaped = value.replace(/"/g, '\\"');
  return needsQuotes ? `"${escaped}"` : escaped;
}

async function addFrontMatter(file) {
  const content = await fs.readFile(file, 'utf8');
  if (content.startsWith('---\n')) {
    return;
  }
  const relativePath = path.relative(apiRoot, file);
  const docId = buildDocId(relativePath);
  const label = labelFromSegments(relativePath);
  const title = label;
  const frontMatter = [
    '---',
    `id: ${docId}`,
    `title: ${escapeYaml(title)}`,
    `sidebar_label: ${escapeYaml(label)}`,
    'hide_title: true',
    '---',
    '',
  ].join('\n');
  await fs.writeFile(file, frontMatter + content);
}

async function main() {
  try {
    const stats = await fs.stat(apiRoot);
    if (!stats.isDirectory()) {
      return;
    }
  } catch {
    return;
  }
  const files = await listMarkdownFiles(apiRoot);
  await Promise.all(files.map(addFrontMatter));
}

await main();

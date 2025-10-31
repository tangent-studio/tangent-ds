#!/usr/bin/env node

/**
 * Verify package structure before publishing
 */

import { readFileSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

console.log('Verifying package structure...\n');

let errors = 0;
let warnings = 0;

// Check package.json
const pkgPath = resolve(__dirname, 'package.json');
if (!existsSync(pkgPath)) {
  console.error('✗ package.json not found');
  errors++;
} else {
  const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'));
  console.log(`✓ package.json found`);
  console.log(`  Name: ${pkg.name}`);
  console.log(`  Version: ${pkg.version}`);

  // Check required fields
  const requiredFields = ['name', 'version', 'description', 'type', 'exports'];
  requiredFields.forEach(field => {
    if (!pkg[field]) {
      console.error(`  ✗ Missing field: ${field}`);
      errors++;
    }
  });
}

// Check source files
const requiredFiles = [
  'src/index.js',
  'src/core/index.js',
  'src/stats/index.js',
  'src/ml/index.js',
  'src/mva/index.js',
  'src/plot/index.js',
  'README.md',
  'LICENSE'
];

console.log('\nChecking required files:');
requiredFiles.forEach(file => {
  const filePath = resolve(__dirname, file);
  if (existsSync(filePath)) {
    console.log(`  ✓ ${file}`);
  } else {
    console.error(`  ✗ ${file} not found`);
    errors++;
  }
});

// Check dist build
const distPath = resolve(__dirname, 'dist/index.js');
if (existsSync(distPath)) {
  console.log(`\n✓ Browser bundle exists: dist/index.js`);
} else {
  console.warn(`\n⚠ Browser bundle not found. Run: npm run build:browser`);
  warnings++;
}

// Check tests
const testsPath = resolve(__dirname, 'tests');
if (existsSync(testsPath)) {
  console.log(`✓ Tests directory exists`);
} else {
  console.error(`✗ Tests directory not found`);
  errors++;
}

// Summary
console.log('\n' + '='.repeat(50));
if (errors === 0 && warnings === 0) {
  console.log('✓ Package verification passed!\n');
  console.log('Ready to publish:');
  console.log('  npm version patch|minor|major');
  console.log('  git push origin main --tags');
  process.exit(0);
} else {
  console.log(`Verification completed with ${errors} error(s) and ${warnings} warning(s)\n`);
  if (errors > 0) {
    console.error('Fix errors before publishing.');
    process.exit(1);
  }
  if (warnings > 0) {
    console.warn('Address warnings if needed.');
    process.exit(0);
  }
}

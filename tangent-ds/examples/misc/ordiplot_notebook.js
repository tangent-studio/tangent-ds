/**
 * Unified Ordiplot Examples
 * Demonstrates the ordiplot() function for PCA, LDA, and RDA
 */

import { PCA, LDA } from '../src/mva/index.js';
import { ordiplot } from '../src/plot/index.js';

// ============= Generate synthetic data =============
function generateData(n = 100, seed = 42) {
  let rng = seed;
  const random = () => {
    rng = (rng * 9301 + 49297) % 233280;
    return rng / 233280;
  };

  const X = [];
  const groups = [];
  const labels = [];

  // Generate 3 groups with different means
  const groupMeans = [
    [0, 0, 0],
    [3, 1, 2],
    [-2, 2, -1]
  ];

  for (let i = 0; i < n; i++) {
    const group = Math.floor(i / (n / 3));
    const mean = groupMeans[group];

    const x = [
      mean[0] + (random() - 0.5) * 2,
      mean[1] + (random() - 0.5) * 2,
      mean[2] + (random() - 0.5) * 2
    ];

    X.push(x);
    groups.push(group);
    labels.push(`S${i + 1}`);
  }

  return { X, groups, labels };
}

const { X, groups, labels } = generateData(99, 42);

// ============= Example 1: Ordiplot with PCA =============
console.log('=== Ordiplot with PCA ===\n');

const pcaEstimator = new PCA({ scale: true, center: true });
pcaEstimator.fit(X);
const pcaResult = pcaEstimator.model;

// Basic ordiplot (auto-detects PCA)
const pcaPlot1 = ordiplot(pcaResult, {
  width: 640,
  height: 400,
  showLoadings: true
});

console.log('PCA Ordiplot Configuration:');
console.log(`  Type: ${pcaPlot1.ordinationType}`);
console.log(`  Axes: ${pcaPlot1.axes.x.label} vs ${pcaPlot1.axes.y.label}`);
console.log(`  Number of points: ${pcaPlot1.data.scores.length}`);
console.log(`  Number of loadings: ${pcaPlot1.data.loadings?.length || 0}`);

// With color grouping
const pcaPlot2 = ordiplot(pcaResult, {
  colorBy: groups,
  showLoadings: true,
  showConvexHulls: true, // Show convex hulls around groups
  width: 640,
  height: 400
});

console.log('\nPCA Ordiplot with grouping:');
console.log(`  Color groups: ${new Set(groups).size} groups`);
console.log(`  Convex hulls: ${pcaPlot2.data.hulls ? 'yes' : 'no'}`);

// With labels
const pcaPlot3 = ordiplot(pcaResult, {
  colorBy: groups,
  labels: labels.slice(0, 10), // Label first 10 points
  showLoadings: false,
  axis1: 1,
  axis2: 2
});

console.log('\nPCA Ordiplot with labels:');
console.log(`  Labeled points: ${labels.slice(0, 10).length}`);

// ============= Example 2: Ordiplot with LDA =============
console.log('\n=== Ordiplot with LDA ===\n');

const ldaEstimator = new LDA();
ldaEstimator.fit(X, groups);
const ldaResult = ldaEstimator.model;

// LDA ordiplot (auto-detects LDA)
const ldaPlot1 = ordiplot(ldaResult, {
  width: 640,
  height: 400,
  showCentroids: true // Show class centroids for LDA
});

console.log('LDA Ordiplot Configuration:');
console.log(`  Type: ${ldaPlot1.ordinationType}`);
console.log(`  Axes: ${ldaPlot1.axes.x.label} vs ${ldaPlot1.axes.y.label}`);
console.log(`  Number of points: ${ldaPlot1.data.scores.length}`);
console.log(`  Number of centroids: ${ldaPlot1.data.centroids?.length || 0}`);

// With convex hulls
const ldaPlot2 = ordiplot(ldaResult, {
  showCentroids: true,
  showConvexHulls: true,
  width: 640,
  height: 400
});

console.log('\nLDA Ordiplot with convex hulls and centroids:');
console.log(`  Shows group separation clearly`);

// ============= Example 3: Comparing ordinations side-by-side =============
console.log('\n=== Comparing PCA vs LDA ===\n');

// Same data, different ordination methods
const pcaForComparison = ordiplot(pcaResult, {
  colorBy: groups,
  showLoadings: false,
  width: 400,
  height: 400
});

const ldaForComparison = ordiplot(ldaResult, {
  showCentroids: true,
  width: 400,
  height: 400
});

console.log('Side-by-side comparison:');
console.log(`  PCA: Maximizes variance`);
console.log(`  LDA: Maximizes class separation`);

// ============= Example 4: Custom axes =============
console.log('\n=== Custom Axes Selection ===\n');

// Plot PC2 vs PC3 instead of PC1 vs PC2
const pcaCustomAxes = ordiplot(pcaResult, {
  axis1: 2,
  axis2: 3,
  colorBy: groups,
  showLoadings: true,
  width: 640,
  height: 400
});

console.log('PCA with custom axes:');
console.log(`  Axes: ${pcaCustomAxes.axes.x.label} vs ${pcaCustomAxes.axes.y.label}`);
console.log(`  Useful for exploring additional components`);

// ============= Example 5: Loading vectors scale =============
console.log('\n=== Loading Vector Scaling ===\n');

const pcaSmallLoadings = ordiplot(pcaResult, {
  showLoadings: true,
  loadingScale: 1, // Smaller scale
  width: 640,
  height: 400
});

const pcaLargeLoadings = ordiplot(pcaResult, {
  showLoadings: true,
  loadingScale: 5, // Larger scale
  width: 640,
  height: 400
});

console.log('Loading vector scaling:');
console.log(`  Small scale (1): More compact`);
console.log(`  Large scale (5): More visible`);

// ============= Usage Summary =============
console.log('\n=== Ordiplot Usage Summary ===\n');
console.log('The ordiplot() function provides a unified interface for:');
console.log('  • PCA - Principal Component Analysis');
console.log('  • LDA - Linear Discriminant Analysis');
console.log('  • RDA - Redundancy Analysis');
console.log('\nKey features:');
console.log('  • Auto-detects ordination type');
console.log('  • Color grouping with colorBy option');
console.log('  • Convex hulls around groups (showConvexHulls)');
console.log('  • Loading vectors for PCA/RDA (showLoadings)');
console.log('  • Class centroids for LDA (showCentroids)');
console.log('  • Point labels (labels option)');
console.log('  • Custom axis selection (axis1, axis2)');
console.log('  • Consistent API across all ordination methods');
console.log('\nExample usage:');
console.log('  const plot = ordiplot(pcaResult, {');
console.log('    colorBy: groups,');
console.log('    showLoadings: true,');
console.log('    showConvexHulls: true');
console.log('  });');

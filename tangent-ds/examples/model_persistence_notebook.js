/**
 * Model Persistence Examples
 * Demonstrates saving and loading models in JSON format
 */

import { KMeans, KNNClassifier, DecisionTreeClassifier } from '../src/ml/index.js';
import { PCA, LDA } from '../src/mva/index.js';
import { lm } from '../src/stats/index.js';
import { saveModel, loadModel } from '../src/core/persistence.js';
import * as fs from 'fs';

// ============= Helper: Generate sample data =============
function generateClassificationData(n = 100, seed = 42) {
  let rng = seed;
  const random = () => {
    rng = (rng * 9301 + 49297) % 233280;
    return rng / 233280;
  };

  const X = [];
  const y = [];

  for (let i = 0; i < n; i++) {
    const x1 = random() * 10 - 5;
    const x2 = random() * 10 - 5;
    const label = (x1 + x2) > 0 ? 1 : 0;

    X.push([x1, x2]);
    y.push(label);
  }

  return { X, y };
}

// ============= Example 1: KMeans Save/Load =============
console.log('=== Example 1: KMeans Persistence ===\n');

const { X, y } = generateClassificationData(100, 42);

// Train KMeans model
const kmeans = new KMeans({ k: 2, seed: 42 });
kmeans.fit(X);

console.log('Original KMeans model:');
console.log(`  Centroids: ${kmeans.centroids.length}`);
console.log(`  Inertia: ${kmeans.inertia.toFixed(4)}`);
console.log(`  Converged: ${kmeans.converged}`);

// Save to JSON string
const kmeansJSON = kmeans.save();
console.log('\nSaved model (first 200 chars):');
console.log(kmeansJSON.substring(0, 200) + '...');

// Load from JSON string
const loadedKMeans = KMeans.load(kmeansJSON);
console.log('\nLoaded KMeans model:');
console.log(`  Centroids: ${loadedKMeans.centroids.length}`);
console.log(`  Inertia: ${loadedKMeans.inertia.toFixed(4)}`);
console.log(`  Converged: ${loadedKMeans.converged}`);

// Test predictions match
const testPoint = [[1, 1]];
const pred1 = kmeans.predict(testPoint);
const pred2 = loadedKMeans.predict(testPoint);
console.log(`\nPrediction on [1, 1]:`);
console.log(`  Original: ${pred1}`);
console.log(`  Loaded: ${pred2}`);
console.log(`  Match: ${pred1[0] === pred2[0] ? '✓' : '✗'}`);

// ============= Example 2: PCA Save/Load =============
console.log('\n=== Example 2: PCA Persistence ===\n');

// Generate data for PCA
const pcaX = X.map(row => [...row, row[0] * 0.5 + row[1] * 0.3]);

const pca = new PCA({ scale: true, center: true });
pca.fit(pcaX);

console.log('Original PCA model:');
console.log(`  Components: ${pca.model.eigenvalues.length}`);
console.log(`  Variance explained: [${pca.model.varianceExplained.map(v => v.toFixed(3)).join(', ')}]`);

// Save to JSON
const pcaJSON = pca.save();
console.log(`\nSaved PCA model size: ${pcaJSON.length} bytes`);

// Load from JSON
const loadedPCA = PCA.load(pcaJSON);
console.log('\nLoaded PCA model:');
console.log(`  Components: ${loadedPCA.model.eigenvalues.length}`);
console.log(`  Variance explained: [${loadedPCA.model.varianceExplained.map(v => v.toFixed(3)).join(', ')}]`);

// Test transformation
const testData = [[1, 2, 3]];
const trans1 = pca.transform(testData);
const trans2 = loadedPCA.transform(testData);
console.log(`\nTransformation matches: ${JSON.stringify(trans1) === JSON.stringify(trans2) ? '✓' : '✗'}`);

// ============= Example 3: LDA Save/Load =============
console.log('\n=== Example 3: LDA Persistence ===\n');

// Generate multi-class data
const groups = X.map(([x1, x2]) => {
  if (x1 > 1) return 0;
  if (x2 > 1) return 1;
  return 2;
});

const lda = new LDA();
lda.fit(X, groups);

console.log('Original LDA model:');
console.log(`  Classes: ${lda.model.classes.length}`);
console.log(`  Discriminant axes: ${lda.model.discriminantAxes.length}`);

// Save and load
const ldaJSON = lda.save();
const loadedLDA = LDA.load(ldaJSON);

console.log('\nLoaded LDA model:');
console.log(`  Classes: ${loadedLDA.model.classes.length}`);
console.log(`  Discriminant axes: ${loadedLDA.model.discriminantAxes.length}`);

// Test transformation
const ldaTestPoint = [[2, 2]];
const ldaTrans1 = lda.transform(ldaTestPoint);
const ldaTrans2 = loadedLDA.transform(ldaTestPoint);
console.log(`\nTransformations match: ${JSON.stringify(ldaTrans1) === JSON.stringify(ldaTrans2) ? '✓' : '✗'}`);

// ============= Example 4: Multiple Models =============
console.log('\n=== Example 4: Save Multiple Models ===\n');

const models = {
  kmeans: kmeans.save(),
  pca: pca.save(),
  lda: lda.save()
};

console.log('Saved models:');
console.log(`  KMeans: ${models.kmeans.length} bytes`);
console.log(`  PCA: ${models.pca.length} bytes`);
console.log(`  LDA: ${models.lda.length} bytes`);

// You can store multiple models in a single object
const multiModelJSON = JSON.stringify(models, null, 2);
console.log(`\nCombined size: ${multiModelJSON.length} bytes`);

// ============= Example 5: Saving to File =============
console.log('\n=== Example 5: Save/Load from File ===\n');

const modelPath = '/tmp/tangent_ds_model.json';

// Save KMeans to file
fs.writeFileSync(modelPath, kmeansJSON);
console.log(`Model saved to: ${modelPath}`);
console.log(`File size: ${fs.statSync(modelPath).size} bytes`);

// Load from file
const loadedFromFile = fs.readFileSync(modelPath, 'utf8');
const kmeansFromFile = KMeans.load(loadedFromFile);

console.log('\nLoaded from file:');
console.log(`  Centroids: ${kmeansFromFile.centroids.length}`);
console.log(`  Inertia: ${kmeansFromFile.inertia.toFixed(4)}`);

// Clean up
fs.unlinkSync(modelPath);
console.log('Temp file cleaned up ✓');

// ============= Example 6: Using standalone persistence functions =============
console.log('\n=== Example 6: Standalone Persistence Functions ===\n');

// You can also use the standalone save/load functions
const model = {
  type: 'custom_model',
  coefficients: [1.5, -0.3, 2.1],
  intercept: 0.5,
  metadata: {
    trained_on: '2024-01-01',
    accuracy: 0.95
  }
};

const saved = saveModel(model);
console.log('Saved custom model:');
console.log(saved.substring(0, 150) + '...');

const loaded = loadModel(saved);
console.log('\nLoaded custom model:');
console.log(JSON.stringify(loaded, null, 2));

// ============= Best Practices Summary =============
console.log('\n=== Best Practices for Model Persistence ===\n');
console.log('1. Use estimator.save() and Estimator.load() for built-in models');
console.log('2. Models are saved in human-readable JSON format');
console.log('3. Saved models include metadata (version, timestamp, type)');
console.log('4. Write to file using fs.writeFileSync(path, jsonString)');
console.log('5. Load from file using Estimator.load(fs.readFileSync(path, "utf8"))');
console.log('6. Always version your models to handle format changes');
console.log('7. For production, consider compressing large models');
console.log('\nCurrently supported models with serialization:');
console.log('  • KMeans (clustering)');
console.log('  • PCA, LDA, RDA, HCA (multivariate analysis)');
console.log('  • Other models may need custom toJSON/fromJSON implementation');
console.log('\nNote: Models store all learned parameters (centroids, loadings, etc.)');
console.log('so they can make predictions without retraining.');

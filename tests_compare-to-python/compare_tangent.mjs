#!/usr/bin/env node
/**
 * Helper script to compute reference results using @tangent.to/ds for comparison
 * with scikit-learn. Supports PCA, linear regression, KMeans, LDA,
 * logistic regression, polynomial regression, MLP regression,
 * KNN, decision trees, random forests, and GAMs.
 *
 * Usage:
 *   node tests_compare-to-python/compare_tangent.mjs pca /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs lm /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs kmeans /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs lda /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs logit /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs polynomial /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs mlp /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs knn_classifier /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs knn_regressor /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs decision_tree_classifier /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs decision_tree_regressor /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs random_forest_classifier /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs random_forest_regressor /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs gam_regressor /path/to/data.json
 *   node tests_compare-to-python/compare_tangent.mjs gam_classifier /path/to/data.json
 *
 * PCA input JSON shape:
 *   { "X": [[...], [...], ...], "options": { "center": true, "scale": false } }
 *
 * Linear regression (lm) input JSON shape:
 *   { "X": [[...], [...], ...], "y": [...], "options": { "intercept": true } }
 *
 * The script prints a JSON object with the relevant results to stdout.
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const mode = process.argv[2];
const dataPath = process.argv[3];

if (!mode || !dataPath) {
  console.error('Usage: node scripts/compare_tangent.mjs <pca|lm> <data.json>');
  process.exit(1);
}

const resolvePackagePath = (relativePath) =>
  path.resolve(__dirname, '..', 'tangent-ds', relativePath);

async function run() {
  const raw = await fs.readFile(dataPath, 'utf8');
  const payload = JSON.parse(raw);

  if (mode === 'pca') {
    const { PCA } = await import(resolvePackagePath('src/mva/index.js'));
    const { X, options = {} } = payload;
    const estimator = new PCA(options);
    estimator.fit(X);
    const model = estimator.model;
    const cumulative = estimator.cumulativeVariance();

    const response = {
      eigenvalues: model.eigenvalues,
      varianceExplained: model.varianceExplained,
      cumulativeVariance: cumulative,
      scores: model.scores.slice(0, 5) // include a handful for spot checking
    };
    console.log(JSON.stringify(response));
    return;
  }

  if (mode === 'lm') {
    const stats = await import(resolvePackagePath('src/stats/index.js'));
    const LmEstimator = stats.lm;
    const { X, y, options = {} } = payload;
    const estimator = new LmEstimator(options);
    estimator.fit(X, y);
    const summary = estimator.summary();
    const response = {
      coefficients: Array.from(estimator.coef),
      rSquared: summary.rSquared,
      adjRSquared: summary.adjRSquared,
      residualStandardError: summary.residualStandardError
    };
    console.log(JSON.stringify(response));
    return;
  }

  if (mode === 'kmeans') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, options = {} } = payload;
    const estimator = new ml.KMeans(options);
    estimator.fit(X);
    const response = {
      centroids: estimator.centroids,
      inertia: estimator.inertia,
      labels: estimator.labels
    };
    console.log(JSON.stringify(response));
    return;
  }

  if (mode === 'lda') {
    const mva = await import(resolvePackagePath('src/mva/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new mva.LDA(options);
    estimator.fit(X, y);
    const model = estimator.model;
    const response = {
      eigenvalues: model.eigenvalues,
      discriminantAxes: model.discriminantAxes,
      scores: model.scores.slice(0, 5),
      classes: model.classes,
      classMeanScores: model.classMeanScores,
      classStdScores: model.classStdScores
    };
    console.log(JSON.stringify(response));
    return;
  }

  if (mode === 'logit') {
    const stats = await import(resolvePackagePath('src/stats/index.js'));
    const LogitEstimator = stats.logit;
    const { X, y, options = {} } = payload;
    const estimator = new LogitEstimator(options);
    estimator.fit(X, y);
    const response = {
      coefficients: estimator.coefficients,
      summary: estimator.summary(),
      probs: estimator.predictProba(X)
    };
    console.log(JSON.stringify(response));
    return;
  }

  if (mode === 'polynomial') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.PolynomialRegressor(options);
    estimator.fit(X, y);
    const summary = estimator.summary();
    const response = {
      coefficients: Array.from(summary.coefficients),
      degree: summary.degree,
      rSquared: summary.rSquared,
      fitted: summary.fitted.slice(0, 10)
    };
    console.log(JSON.stringify(response));
    return;
  }

  if (mode === 'mlp') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.MLPRegressor(options);
    estimator.fit(X, y);
    const preds = estimator.predict(X);
    const response = {
      predictions: preds.slice(0, 20),
      summary: estimator.summary()
    };
    console.log(JSON.stringify(response));
    return;
  }

  if (mode === 'knn_classifier') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.KNNClassifier(options);
    estimator.fit(X, y);
    const preds = estimator.predict(payload.Xtest || X.slice(0, 10));
    const proba = estimator.predictProba(payload.Xtest || X.slice(0, 10));
    const response = {
      predictions: preds,
      probabilities: proba
    };
    console.log(JSON.stringify(response));
    return;
  }

  if (mode === 'knn_regressor') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.KNNRegressor(options);
    estimator.fit(X, y);
    const preds = estimator.predict(payload.Xtest || X.slice(0, 10));
    console.log(JSON.stringify({ predictions: preds }));
    return;
  }

  if (mode === 'decision_tree_classifier') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.DecisionTreeClassifier(options);
    estimator.fit(X, y);
    const preds = estimator.predict(payload.Xtest || X.slice(0, 10));
    console.log(JSON.stringify({ predictions: preds }));
    return;
  }

  if (mode === 'decision_tree_regressor') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.DecisionTreeRegressor(options);
    estimator.fit(X, y);
    const preds = estimator.predict(payload.Xtest || X.slice(0, 10));
    console.log(JSON.stringify({ predictions: preds }));
    return;
  }

  if (mode === 'random_forest_classifier') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.RandomForestClassifier(options);
    estimator.fit(X, y);
    const preds = estimator.predict(payload.Xtest || X.slice(0, 10));
    console.log(JSON.stringify({ predictions: preds }));
    return;
  }

  if (mode === 'random_forest_regressor') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.RandomForestRegressor(options);
    estimator.fit(X, y);
    const preds = estimator.predict(payload.Xtest || X.slice(0, 10));
    console.log(JSON.stringify({ predictions: preds }));
    return;
  }

  if (mode === 'gam_regressor') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.GAMRegressor(options);
    estimator.fit(X, y);
    const preds = estimator.predict(payload.Xtest || X.slice(0, 10));
    console.log(JSON.stringify({ predictions: preds, coefficients: estimator.gam?.coef }));
    return;
  }

  if (mode === 'gam_classifier') {
    const ml = await import(resolvePackagePath('src/ml/index.js'));
    const { X, y, options = {} } = payload;
    const estimator = new ml.GAMClassifier(options);
    estimator.fit(X, y);
    const preds = estimator.predict(payload.Xtest || X.slice(0, 10));
    const proba = estimator.predictProba(payload.Xtest || X.slice(0, 10));
    console.log(JSON.stringify({ predictions: preds, probabilities: proba }));
    return;
  }

  console.error(`Unsupported mode: ${mode}`);
  process.exit(1);
}

run().catch((err) => {
  console.error('Error running tangent comparison:', err);
  process.exit(1);
});

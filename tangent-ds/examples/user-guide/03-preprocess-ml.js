// ---
// title: 03 — Preprocess for ML
// id: preprocess-ml
// ---

// %% [markdown]
/*
# 3 — Preprocess for ML

Purpose:
- Split early, fit transformers on training data only, avoid leakage.
- Persist pipeline parameters (means/stds, categorical maps) so transforms are reproducible.
*/

// %% [javascript]
/*
Simple train/validation/test split and lightweight transformers (standard scaler, one-hot mapping).
*/
const trainValTestSplit = (X, y, testRatio = 0.2) => {
  const n = X.length;
  const split = Math.floor(n * (1 - testRatio));
  return {
    X_train: X.slice(0, split),
    X_test: X.slice(split),
    y_train: y.slice(0, split),
    y_test: y.slice(split)
  };
};

// Use iris from explore notebook
globalThis.y_iris = globalThis.irisData.map(d => d.species);
const split = trainValTestSplit(globalThis.X_iris, globalThis.y_iris, 0.2);
globalThis.X_train = split.X_train;
globalThis.X_test = split.X_test;
globalThis.y_train = split.y_train;
globalThis.y_test = split.y_test;

console.log(`Train size: ${globalThis.X_train.length}, Test size: ${globalThis.X_test.length}`);

// Standard scaler
const fitStandardScaler = (X) => {
  const cols = X[0].length;
  const n = X.length;
  const mean = Array(cols).fill(0);
  const std = Array(cols).fill(0);
  for (const r of X) for (let j = 0; j < cols; j++) mean[j] += r[j];
  mean.forEach((m, j) => mean[j] = m / n);
  for (const r of X) for (let j = 0; j < cols; j++) std[j] += (r[j] - mean[j]) ** 2;
  std.forEach((s, j) => std[j] = Math.sqrt(s / Math.max(1, n - 1)));
  return { mean, std };
};
globalThis.scaler = fitStandardScaler(globalThis.X_train);

const transformScale = (X, scaler) => X.map(r => r.map((v, j) => (v - scaler.mean[j]) / (scaler.std[j] || 1)));
globalThis.X_train_scaled = transformScale(globalThis.X_train, globalThis.scaler);
globalThis.X_test_scaled = transformScale(globalThis.X_test, globalThis.scaler);

// Simple categorical encoder (species -> one-hot index map)
const buildCategoryMap = (labels) => {
  const uniq = [...new Set(labels)];
  const map = Object.fromEntries(uniq.map((v, i) => [v, i]));
  return { map, categories: uniq };
};
globalThis.labelEncoder = buildCategoryMap(globalThis.y_train);

// Encoded labels (numeric)
globalThis.y_train_encoded = globalThis.y_train.map(l => globalThis.labelEncoder.map[l]);
globalThis.y_test_encoded = globalThis.y_test.map(l => globalThis.labelEncoder.map[l]);

// Persistable pipeline (JSON-safe)
globalThis.preprocessPipeline = {
  scaler: { mean: globalThis.scaler.mean, std: globalThis.scaler.std },
  labelEncoder: { map: globalThis.labelEncoder.map, categories: globalThis.labelEncoder.categories }
};
console.log("Preprocessing pipeline ready. Save preprocessPipeline for deployment.");

// ---
// title: 01 — Explore data
// id: explore-data
// ---

// %% [markdown]
/*
# 1 — Explore data: descriptive statistics, MVA and visualization

Purpose:
- Load data, compute descriptive statistics and correlations.
- Inspect missingness and run PCA / ordination for multivariate structure.
*/

// %% [javascript]
import * as ds from "@tangent.to/ds";

const fetchJson = async (url) => (await fetch(url)).json();

// Load datasets
globalThis.irisData = await fetchJson("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/iris.json");
globalThis.carsData = (await fetchJson("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json"))
  .filter(d => d.Horsepower != null && d.Miles_per_Gallon != null);

// Feature matrix helpers
const irisFeatures = d => [d.sepalLength, d.sepalWidth, d.petalLength, d.petalWidth];
globalThis.X_iris = globalThis.irisData.map(irisFeatures);

// Simple descriptive stats
const summarize = (X) => {
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

globalThis.irisSummary = summarize(globalThis.X_iris);
console.log("Iris means:", globalThis.irisSummary.mean.map(v => v.toFixed(3)));
console.log("Iris stds :", globalThis.irisSummary.std.map(v => v.toFixed(3)));

// Missingness (cars)
const countMissing = (arr, accessor) => arr.reduce((acc, r) => acc + (accessor(r) == null ? 1 : 0), 0);
console.log("Cars missing Horsepower:", countMissing(globalThis.carsData, d => d.Horsepower));

// Correlation matrix (Pearson)
const corrMatrix = (X) => {
  const cols = X[0].length;
  const { mean, std } = summarize(X);
  const C = Array.from({ length: cols }, () => Array(cols).fill(0));
  for (const r of X) for (let i = 0; i < cols; i++) for (let j = 0; j < cols; j++) C[i][j] += ((r[i] - mean[i]) * (r[j] - mean[j]));
  const n = X.length;
  for (let i = 0; i < cols; i++) for (let j = 0; j < cols; j++) {
    const denom = (std[i] * std[j] * (n - 1)) || 1;
    C[i][j] = (C[i][j] / denom);
  }
  return C;
};
globalThis.irisCorr = corrMatrix(globalThis.X_iris);
console.log("Iris correlation[0]:", globalThis.irisCorr[0].map(v => v.toFixed(3)));

// PCA (if available)
if (ds.mva && ds.mva.PCA) {
  globalThis.pca = new ds.mva.PCA({ nComponents: 2 });
  globalThis.pca.fit(globalThis.X_iris);
  globalThis.irisScores = globalThis.pca.transform(globalThis.X_iris);
  console.log("PCA scores sample:", globalThis.irisScores[0]);
} else {
  console.log("PCA not available in ds.mva — skip PCA demo.");
}

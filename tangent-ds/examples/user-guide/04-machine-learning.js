// ---
// title: Machine Learning with @tangent.to/ds — User Guide Pipeline
// id: machine-learning
// ---

// %% [markdown]
/*
# Machine Learning with @tangent.to/ds

This notebook is reorganized as a sequential data-science pipeline:

1. Explore data — descriptive statistics, MVA and visualization
2. Statistical tests — ANOVA, LM, LMM (illustrative)
3. Preprocess for ML — splitting, imputation, scaling, encoding
4. ML — core algorithms, model selection and evaluation
5. Advanced ML — MLP (tensorflow.js) with recommended IO

Use the cells below to run each stage. Variables that should be accessible across cells are stored on globalThis.
*/

// %% [markdown]
/*
## 1 — Explore data: descriptive statistics, MVA and visualization

Purpose:
- Quickly inspect distributions, missingness, correlations.
- Use PCA / ordination to reveal major multivariate structure.
*/

// %% [javascript]
/*
Load datasets (iris and cars) and compute basic summaries.
*/
import * as ds from "@tangent.to/ds";

const fetchJson = async (url) => (await fetch(url)).json();

// Load datasets
globalThis.irisData = await fetchJson("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/iris.json");
globalThis.carsData = (await fetchJson("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json"))
  .filter(d => d.Horsepower != null && d.Miles_per_Gallon != null);

console.log(`Loaded ${globalThis.irisData.length} iris samples and ${globalThis.carsData.length} cars samples`);

// Descriptive stats (example)
const numericFeatures = d => [d.sepalLength, d.sepalWidth, d.petalLength, d.petalWidth];
globalThis.X_iris = globalThis.irisData.map(numericFeatures);

// Simple summary: mean per column
const columnMeans = (X) => {
  const cols = X[0].length;
  const sums = Array(cols).fill(0);
  for (const r of X) for (let j = 0; j < cols; j++) sums[j] += r[j];
  return sums.map(s => s / X.length);
};
console.log("Iris feature means:", columnMeans(globalThis.X_iris).map(v => v.toFixed(3)));

// PCA (if available) — reduce to 2 components for visualization
if (ds.mva && ds.mva.PCA) {
  globalThis.pca = new ds.mva.PCA({ nComponents: 2 });
  globalThis.pca.fit(globalThis.X_iris);
  globalThis.irisScores = globalThis.pca.transform(globalThis.X_iris);
  console.log("PCA done. First score sample:", globalThis.irisScores[0]);
} else {
  console.log("PCA not available in ds.mva — skip PCA demo.");
}

// %% [markdown]
/*
## 2 — Statistical tests: ANOVA, LM, LMM (illustrative)

Purpose:
- Formally test hypotheses and inspect effect sizes before predictive modeling.

Notes:
- The following is illustrative pseudocode: adapt to your project's stats API (e.g., LinearModel, anova, MixedModel).
*/

// %% [javascript]
// Illustrative linear model / anova usage (replace with concrete API as available)
try {
  if (ds.stats && ds.stats.LinearModel) {
    // Example: model species numeric response from first petal length (dummy demo)
    globalThis.irisDF = globalThis.irisData.map(d => ({ sepalLength: d.sepalLength, petalLength: d.petalLength, species: d.species }));
    // Fit a simple linear model (replace with real API)
    globalThis.lm = new ds.stats.LinearModel({ formula: "sepalLength ~ petalLength" });
    await globalThis.lm.fit(globalThis.irisDF);
    console.log("LinearModel summary:", globalThis.lm.summary());
  } else {
    console.log("Statistical model APIs not present (ds.stats.LinearModel). Illustrative only.");
  }
} catch (e) {
  console.log("Stat tests skipped (no suitable API):", e.message);
}

// %% [markdown]
/*
## 3 — Preprocess for ML

Purpose:
- Split early, fit transformers on training set only, and persist the pipeline.
*/

// %% [javascript]
// Split data (train/val/test) and demonstrate scaling/encoding
const trainTestSplit = (X, y, testRatio = 0.2, seed = 42) => {
  const n = X.length;
  const split = Math.floor(n * (1 - testRatio));
  return {
    X_train: X.slice(0, split),
    X_test: X.slice(split),
    y_train: y.slice(0, split),
    y_test: y.slice(split)
  };
};

globalThis.y_iris = globalThis.irisData.map(d => d.species);
const split = trainTestSplit(globalThis.X_iris, globalThis.y_iris, 0.2, 42);
globalThis.X_train = split.X_train;
globalThis.X_test = split.X_test;
globalThis.y_train = split.y_train;
globalThis.y_test = split.y_test;

console.log(`Train size: ${globalThis.X_train.length}, Test size: ${globalThis.X_test.length}`);

// Example scaling (simple standardizer)
const fitStandardScaler = (X) => {
  const cols = X[0].length;
  const mean = Array(cols).fill(0);
  const std = Array(cols).fill(0);
  for (const r of X) for (let j = 0; j < cols; j++) mean[j] += r[j];
  mean.forEach((m, j) => mean[j] = m / X.length);
  for (const r of X) for (let j = 0; j < cols; j++) std[j] += (r[j] - mean[j]) ** 2;
  std.forEach((s, j) => std[j] = Math.sqrt(s

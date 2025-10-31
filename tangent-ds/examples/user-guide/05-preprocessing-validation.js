// ---
// title: Preprocessing & Validation with @tangent/ds
// id: preprocessing-validation
// ---

// %% [markdown]
/*
# Preprocessing & Validation with @tangent/ds

This notebook covers data preprocessing and model validation:

**Preprocessing**
- Feature scaling: StandardScaler, MinMaxScaler, Normalizer
- Encoding: LabelEncoder, OneHotEncoder
- Feature engineering: PolynomialFeatures

**Validation**
- Train-test split
- Cross-validation: K-Fold, Stratified K-Fold, Leave-One-Out
- Hyperparameter tuning: GridSearchCV, RandomSearchCV
- Model persistence: Save and load models

These are essential for building robust, production-ready models.
*/

// %% [javascript]
import * as ds from "@tangent/ds";

// Load iris dataset
globalThis.irisData = await fetch("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/iris.json")
  .then(r => r.json());

globalThis.X_iris = irisData.map(d => [
  d.sepalLength,
  d.sepalWidth,
  d.petalLength,
  d.petalWidth
]);
globalThis.species = irisData.map(d => d.species);

console.log(`Loaded ${irisData.length} iris samples`);

// %% [markdown]
/*
## 1. Feature Scaling

Scaling ensures all features are on similar scales, important for many algorithms.
*/

// %% [javascript]
// StandardScaler: zero mean, unit variance
globalThis.scaler = new ds.ml.StandardScaler();
scaler.fit(X_iris);

console.log("StandardScaler:");
console.log(`  Means: [${scaler.means.map(m => m.toFixed(2)).join(", ")}]`);
console.log(`  Stds: [${scaler.stds.map(s => s.toFixed(2)).join(", ")}]`);

// Transform data
globalThis.X_scaled = scaler.transform(X_iris);
console.log(`\nOriginal range: [${Math.min(...X_iris.flat()).toFixed(2)}, ${Math.max(...X_iris.flat()).toFixed(2)}]`);
console.log(`Scaled range: [${Math.min(...X_scaled.flat()).toFixed(2)}, ${Math.max(...X_scaled.flat()).toFixed(2)}]`);

// Verify scaling: mean should be ~0, std should be ~1
const scaled_mean = X_scaled.flat().reduce((a, b) => a + b, 0) / X_scaled.flat().length;
console.log(`Scaled mean: ${scaled_mean.toFixed(6)} (should be ~0)`);

// %% [markdown]
/*
### MinMaxScaler

Scale features to a specific range (default [0, 1]).
*/

// %% [javascript]
globalThis.minmax_scaler = new ds.ml.MinMaxScaler({ featureRange: [0, 1] });
minmax_scaler.fit(X_iris);

globalThis.X_minmax = minmax_scaler.transform(X_iris);
console.log("\nMinMaxScaler ([0, 1]):");
console.log(`  Min: ${Math.min(...X_minmax.flat()).toFixed(2)}`);
console.log(`  Max: ${Math.max(...X_minmax.flat()).toFixed(2)}`);

// Inverse transform to get back original values
globalThis.X_inv = minmax_scaler.inverseTransform(X_minmax);
globalThis.max_diff = Math.max(...X_iris.flat().map((v, i) => Math.abs(v - X_inv.flat()[i])));
console.log(`  Reconstruction error: ${max_diff.toFixed(10)}`);

// %% [markdown]
/*
### Normalizer

Normalize each sample (row) to unit norm.
*/

// %% [javascript]
globalThis.normalizer = new ds.ml.Normalizer({ norm: "l2" });
globalThis.X_normalized = normalizer.transform(X_iris);

// Check that each row has unit L2 norm
globalThis.norms = X_normalized.map(row =>
  Math.sqrt(row.reduce((sum, val) => sum + val ** 2, 0))
);
console.log("\nNormalizer (L2):");
console.log(`  Sample norms (first 5): [${norms.slice(0, 5).map(n => n.toFixed(3)).join(", ")}]`);
console.log(`  All should be 1.0`);

// %% [markdown]
/*
## 2. Categorical Encoding

Convert categorical variables to numeric form.
*/

// %% [javascript]
// LabelEncoder: Convert labels to integers
globalThis.label_encoder = new ds.ml.LabelEncoder();
label_encoder.fit(species);

globalThis.species_encoded = label_encoder.transform(species);
console.log("LabelEncoder:");
console.log(`  Classes: [${label_encoder.classes.join(", ")}]`);
console.log(`  Encoded (first 5): [${species_encoded.slice(0, 5).join(", ")}]`);

// Inverse transform
globalThis.species_decoded = label_encoder.inverseTransform(species_encoded.slice(0, 5));
console.log(`  Decoded (first 5): [${species_decoded.join(", ")}]`);

// %% [markdown]
/*
### OneHotEncoder

Create binary columns for each category.
*/

// %% [javascript]
// OneHotEncoder works on arrays of arrays
globalThis.categories = species.map(s => [s]); // Wrap in arrays
globalThis.onehot_encoder = new ds.ml.OneHotEncoder();
onehot_encoder.fit(categories);

globalThis.species_onehot = onehot_encoder.transform(categories);
console.log("\nOneHotEncoder:");
console.log(`  Feature names: [${onehot_encoder.getFeatureNames().join(", ")}]`);
console.log(`  Original shape: ${categories.length} × ${categories[0].length}`);
console.log(`  Encoded shape: ${species_onehot.length} × ${species_onehot[0].length}`);
console.log(`  First sample: [${species_onehot[0].join(", ")}]`);

// %% [markdown]
/*
## 3. Feature Engineering

Create new features from existing ones.
*/

// %% [javascript]
// PolynomialFeatures: Create interaction terms and powers
globalThis.poly = new ds.ml.PolynomialFeatures({ degree: 2, includeBias: false });
poly.fit(X_iris);

globalThis.X_poly = poly.transform(X_iris);
console.log("PolynomialFeatures (degree 2):");
console.log(`  Input features: ${X_iris[0].length}`);
console.log(`  Output features: ${X_poly[0].length}`);
console.log(`  Includes: original features + interactions + squares`);
console.log(`  Example: [a, b] → [a, b, a², ab, b²]`);

// %% [markdown]
/*
## 4. Train-Test Split

Split data into training and testing sets.
*/

// %% [javascript]
globalThis.split = ds.ml.trainTestSplit(X_iris, species, {
  ratio: 0.8,  // 80% train, 20% test
  shuffle: true,
  seed: 42
});

console.log("\nTrain-Test Split:");
console.log(`  Training samples: ${split.XTrain.length}`);
console.log(`  Testing samples: ${split.XTest.length}`);
console.log(`  Ratio: ${(split.XTrain.length / X_iris.length * 100).toFixed(0)}% / ${(split.XTest.length / X_iris.length * 100).toFixed(0)}%`);

// %% [markdown]
/*
## 5. Cross-Validation

Evaluate model performance using multiple train-test splits.
*/

// %% [javascript]
// K-Fold Cross-Validation
globalThis.folds = ds.ml.kFold(X_iris, species, 5, true);

console.log("5-Fold Cross-Validation:");
console.log(`  Number of folds: ${folds.length}`);
console.log(`  Fold sizes:`);
folds.forEach((fold, i) => {
  console.log(`    Fold ${i + 1}: ${fold.trainIndices.length} train, ${fold.testIndices.length} test`);
});

// %% [markdown]
/*
### Stratified K-Fold

Maintain class proportions in each fold (important for imbalanced data).
*/

// %% [javascript]
globalThis.stratified_folds = ds.ml.stratifiedKFold(X_iris, species, 5);

console.log("\nStratified 5-Fold:");
// Check class distribution in first fold
const fold0_labels = stratified_folds[0].testIndices.map(i => species[i]);
const counts = {};
fold0_labels.forEach(label => counts[label] = (counts[label] || 0) + 1);
console.log(`  Fold 1 test set class distribution:`);
Object.entries(counts).forEach(([label, count]) => {
  console.log(`    ${label}: ${count}`);
});

// %% [markdown]
/*
### Leave-One-Out Cross-Validation

Use n-1 samples for training, 1 for testing (n times).
Expensive but maximizes training data.
*/

// %% [javascript]
// Use small subset for demo
globalThis.X_small = X_iris.slice(0, 10);
globalThis.y_small = species.slice(0, 10);
globalThis.loo_folds = ds.ml.leaveOneOut(X_small, y_small);

console.log("\nLeave-One-Out:");
console.log(`  Samples: ${X_small.length}`);
console.log(`  Folds: ${loo_folds.length} (one per sample)`);
console.log(`  Each fold: ${loo_folds[0].trainIndices.length} train, ${loo_folds[0].testIndices.length} test`);

// %% [markdown]
/*
### Cross-Validation Execution

Run cross-validation with a model.
*/

// %% [javascript]
// Define fit and score functions
const fitFn = (XTrain, yTrain) => {
  const knn = new ds.ml.KNNClassifier({ k: 5 });
  knn.fit(XTrain, yTrain);
  return knn;
};

const scoreFn = (model, XTest, yTest) => {
  const yPred = model.predict(XTest);
  return yTest.filter((y, i) => y === yPred[i]).length / yTest.length;
};

// Run cross-validation
globalThis.cv_results = ds.ml.crossValidate(fitFn, scoreFn, X_iris, species, folds);

console.log("Cross-Validation Results:");
console.log(`  Scores: [${cv_results.scores.map(s => s.toFixed(3)).join(", ")}]`);
console.log(`  Mean accuracy: ${cv_results.meanScore.toFixed(3)}`);
console.log(`  Std deviation: ${cv_results.stdScore.toFixed(3)}`);

// %% [markdown]
/*
## 6. Hyperparameter Tuning

Find the best hyperparameters using grid or random search.
*/

// %% [javascript]
// Grid Search: Try all combinations
globalThis.param_grid = {
  k: [3, 5, 7, 9],
  weight: ["uniform", "distance"]
};

const fitFnWithParams = (XTrain, yTrain, params) => {
  const knn = new ds.ml.KNNClassifier(params);
  knn.fit(XTrain, yTrain);
  return knn;
};

console.log("GridSearchCV:");
console.log(`  Parameter grid: k=${param_grid.k.length} values, weight=${param_grid.weight.length} values`);
console.log(`  Total combinations: ${param_grid.k.length * param_grid.weight.length}`);

// Note: Actual execution commented out to save time
// globalThis.grid_search = ds.ml.GridSearchCV(
//   fitFnWithParams,
//   scoreFn,
//   X_iris,
//   species,
//   param_grid,
//   { k: 3, verbose: false }
// );
// console.log(`  Best params: k=${grid_search.bestParams.k}, weight=${grid_search.bestParams.weight}`);
// console.log(`  Best score: ${grid_search.bestScore.toFixed(3)}`);

// %% [markdown]
/*
### Random Search

Sample random combinations (faster for large parameter spaces).
*/

// %% [javascript]
// Random Search with distributions
globalThis.param_distributions = {
  k: [3, 5, 7, 9, 11, 13, 15],
  weight: ["uniform", "distance"]
};

console.log("\nRandomSearchCV:");
console.log(`  Parameter space: k=${param_distributions.k.length} values, weight=${param_distributions.weight.length} values`);
console.log(`  Will try random combinations instead of all`);

// Note: Actual execution commented out
// globalThis.random_search = ds.ml.RandomSearchCV(
//   fitFnWithParams,
//   scoreFn,
//   X_iris,
//   species,
//   param_distributions,
//   { nIter: 10, k: 3, seed: 42, verbose: false }
// );

// %% [markdown]
/*
## 7. Model Persistence

Save and load trained models.
*/

// %% [javascript]
// Train a model
globalThis.knn_model = new ds.ml.KNNClassifier({ k: 5, weight: "distance" });
knn_model.fit(split.XTrain, split.yTrain);

// Save to JSON
globalThis.model_json = knn_model.save();
console.log("Model Persistence:");
console.log(`  Saved model size: ${model_json.length} bytes`);

// Load from JSON
globalThis.loaded_model = ds.ml.KNNClassifier.load(model_json);
console.log(`  Model loaded successfully`);

// Verify predictions match
globalThis.pred_original = knn_model.predict(split.XTest.slice(0, 5));
globalThis.pred_loaded = loaded_model.predict(split.XTest.slice(0, 5));
globalThis.predictions_match = JSON.stringify(pred_original) === JSON.stringify(pred_loaded);
console.log(`  Predictions match: ${predictions_match ? "✓" : "✗"}`);

// %% [markdown]
/*
### Saving Multiple Models

Store a collection of models together.
*/

// %% [javascript]
// Train multiple models
globalThis.models_collection = {
  knn: knn_model.save(),
  scaler: JSON.stringify({
    type: "StandardScaler",
    means: scaler.means,
    stds: scaler.stds
  })
};

console.log("\nModel Collection:");
console.log(`  Models saved: ${Object.keys(models_collection).length}`);
console.log(`  Total size: ${JSON.stringify(models_collection).length} bytes`);

// In production, save to file:
// import * as fs from 'fs';
// fs.writeFileSync('models.json', JSON.stringify(models_collection, null, 2));

// %% [markdown]
/*
## 8. Complete ML Pipeline

Put it all together: preprocessing → training → validation → persistence.
*/

// %% [javascript]
console.log("\nComplete ML Pipeline:");

// 1. Split data
const pipeline_split = ds.ml.trainTestSplit(X_iris, species, {
  ratio: 0.8,
  shuffle: true,
  seed: 42
});
console.log(`  1. Split: ${pipeline_split.XTrain.length} train, ${pipeline_split.XTest.length} test`);

// 2. Scale features
const pipeline_scaler = new ds.ml.StandardScaler();
pipeline_scaler.fit(pipeline_split.XTrain);
const X_train_scaled = pipeline_scaler.transform(pipeline_split.XTrain);
const X_test_scaled = pipeline_scaler.transform(pipeline_split.XTest);
console.log(`  2. Scaled features`);

// 3. Train model
const pipeline_model = new ds.ml.KNNClassifier({ k: 5 });
pipeline_model.fit(X_train_scaled, pipeline_split.yTrain);
console.log(`  3. Trained KNN (k=5)`);

// 4. Evaluate
const y_pred_pipeline = pipeline_model.predict(X_test_scaled);
const accuracy_pipeline = pipeline_split.yTest.filter(
  (y, i) => y === y_pred_pipeline[i]
).length / pipeline_split.yTest.length;
console.log(`  4. Test accuracy: ${(accuracy_pipeline * 100).toFixed(1)}%`);

// 5. Save everything
const pipeline_bundle = {
  scaler: {
    means: pipeline_scaler.means,
    stds: pipeline_scaler.stds
  },
  model: pipeline_model.save(),
  metadata: {
    features: ["sepalLength", "sepalWidth", "petalLength", "petalWidth"],
    target: "species",
    accuracy: accuracy_pipeline,
    date: new Date().toISOString()
  }
};
console.log(`  5. Saved complete pipeline (${JSON.stringify(pipeline_bundle).length} bytes)`);

// %% [markdown]
/*
## Summary

@tangent/ds provides comprehensive preprocessing and validation tools:

**Preprocessing**
- `StandardScaler()` - Zero mean, unit variance
- `MinMaxScaler({ featureRange })` - Scale to [min, max]
- `Normalizer({ norm })` - Normalize samples to unit norm
- `LabelEncoder()` - Encode categorical labels
- `OneHotEncoder()` - Create binary indicator features
- `PolynomialFeatures({ degree })` - Create polynomial terms

**Validation**
- `trainTestSplit(X, y, { ratio, shuffle, seed })` - Split data
- `kFold(X, y, k, shuffle)` - K-fold CV
- `stratifiedKFold(X, y, k)` - Stratified K-fold
- `leaveOneOut(X, y)` - LOO CV
- `crossValidate(fitFn, scoreFn, X, y, folds)` - Execute CV

**Hyperparameter Tuning**
- `GridSearchCV(fitFn, scoreFn, X, y, paramGrid, options)` - Grid search
- `RandomSearchCV(fitFn, scoreFn, X, y, paramDist, options)` - Random search

**Model Persistence**
- `model.save()` - Save to JSON string
- `Model.load(jsonString)` - Load from JSON
- Works with KMeans, PCA, LDA, RDA, HCA and more

All tools follow scikit-learn-like API for familiarity and ease of use.
*/

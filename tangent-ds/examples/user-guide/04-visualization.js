// ---
// title: Visualization with @tangent/ds
// id: visualization
// ---

// %% [markdown]
/*
# Visualization with @tangent/ds

@tangent/ds provides plot configuration generators for Observable Plot.
This notebook covers:

- **Ordination plots**: PCA, LDA, RDA biplots with unified `ordiplot()`
- **Classification metrics**: ROC curves, confusion matrices, calibration plots
- **Model interpretation**: Feature importance, partial dependence, residuals
- **Dendrograms**: Hierarchical clustering trees

All functions return Observable Plot configurations, not actual plots.
You can use them with `Plot.plot(config)` in Observable or similar environments.
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

// %% [markdown]
/*
## 1. Ordination Plots (Ordiplot)

The unified `ordiplot()` function works with PCA, LDA, and RDA results.
*/

// %% [javascript]
// Fit PCA
globalThis.pca = new ds.mva.PCA({ scale: true, center: true });
pca.fit(X_iris);

// Create PCA ordiplot with loadings and color grouping
globalThis.pca_config = ds.plot.ordiplot(pca.model, {
  colorBy: species,
  showLoadings: true,
  showConvexHulls: true,
  loadingScale: 3,
  width: 640,
  height: 400
});

console.log("PCA Ordiplot Configuration:");
console.log(`  Type: ${pca_config.ordinationType}`);
console.log(`  Points: ${pca_config.data.scores.length}`);
console.log(`  Loadings: ${pca_config.data.loadings.length}`);
console.log(`  Convex hulls: ${pca_config.data.hulls ? "yes" : "no"}`);
console.log(`  Axes: ${pca_config.axes.x.label} vs ${pca_config.axes.y.label}`);

// %% [markdown]
/*
### LDA Ordiplot with Centroids

LDA plots show class separation with optional centroids.
*/

// %% [javascript]
// Fit LDA
globalThis.lda = new ds.mva.LDA();
lda.fit(X_iris, species);

// Create LDA ordiplot with centroids
globalThis.lda_config = ds.plot.ordiplot(lda.model, {
  showCentroids: true,
  showConvexHulls: true,
  width: 640,
  height: 400
});

console.log("\nLDA Ordiplot Configuration:");
console.log(`  Type: ${lda_config.ordinationType}`);
console.log(`  Points: ${lda_config.data.scores.length}`);
console.log(`  Centroids: ${lda_config.data.centroids.length}`);
console.log(`  Marks: ${lda_config.marks.length} layers`);

// %% [markdown]
/*
### Scree Plot

Visualize variance explained by each component.
*/

// %% [javascript]
globalThis.scree_config = ds.plot.plotScree(pca.model, {
  width: 640,
  height: 300
});

console.log("\nScree Plot Configuration:");
console.log(`  Components: ${scree_config.data.components.length}`);
console.log(`  PC1 variance: ${scree_config.data.components[0].variance.toFixed(1)}%`);
console.log(`  PC2 variance: ${scree_config.data.components[1].variance.toFixed(1)}%`);

// %% [markdown]
/*
## 2. Classification Metrics

Visualize model performance with ROC curves, confusion matrices, and calibration plots.
*/

// %% [javascript]
// Train logistic regression for binary classification
globalThis.y_binary = species.map(s => s === "setosa" ? 1 : 0);
globalThis.logit = ds.stats.logit.fit(X_iris, y_binary, { intercept: true });
globalThis.y_prob = logit.fitted;
globalThis.y_pred = y_prob.map(p => p > 0.5 ? 1 : 0);

// %% [markdown]
/*
### ROC Curve

Receiver Operating Characteristic curve shows trade-off between TPR and FPR.
*/

// %% [javascript]
globalThis.roc_config = ds.plot.plotROC(y_binary, y_prob, {
  width: 500,
  height: 500,
  showDiagonal: true
});

console.log("ROC Curve Configuration:");
console.log(`  AUC: ${roc_config.data.auc.toFixed(3)}`);
console.log(`  Points: ${roc_config.data.curve.length}`);
console.log(`  Title: ${roc_config.title}`);

// %% [markdown]
/*
### Precision-Recall Curve

Shows trade-off between precision and recall.
*/

// %% [javascript]
globalThis.pr_config = ds.plot.plotPrecisionRecall(y_binary, y_prob, {
  width: 500,
  height: 500,
  showBaseline: true
});

console.log("\nPrecision-Recall Configuration:");
console.log(`  Average Precision: ${pr_config.data.avgPrecision.toFixed(3)}`);
console.log(`  Points: ${pr_config.data.curve.length}`);

// %% [markdown]
/*
### Confusion Matrix

Visual representation of classification performance.
*/

// %% [javascript]
globalThis.cm_config = ds.plot.plotConfusionMatrix(y_binary, y_pred, {
  width: 400,
  height: 400,
  normalize: false,
  labels: ["Not Setosa", "Setosa"]
});

console.log("\nConfusion Matrix Configuration:");
console.log(`  Cells: ${cm_config.data.cells.length}`);
console.log(`  Normalized: ${cm_config.data.normalized}`);
console.log(`  Classes: [${cm_config.data.classes.join(", ")}]`);

// Display confusion matrix values
console.log("\nConfusion Matrix Values:");
cm_config.data.cells.forEach(cell => {
  console.log(`  ${cell.true} → ${cell.predicted}: ${cell.count}`);
});

// %% [markdown]
/*
### Calibration Plot

Shows how well predicted probabilities match actual frequencies.
*/

// %% [javascript]
globalThis.cal_config = ds.plot.plotCalibration(y_binary, y_prob, {
  width: 500,
  height: 500,
  nBins: 10
});

console.log("\nCalibration Plot Configuration:");
console.log(`  Bins: ${cal_config.data.curve.length}`);
console.log(`  Title: ${cal_config.title}`);

// %% [markdown]
/*
## 3. Model Interpretation

Understand what drives model predictions.
*/

// %% [javascript]
// Train a random forest for feature importance
globalThis.rf = new ds.ml.RandomForestClassifier({
  nTrees: 50,
  maxDepth: 5,
  seed: 42
});
rf.fit(X_iris, species);

// Compute feature importance using permutation
globalThis.importance = ds.ml.interpret.featureImportance(
  rf,
  X_iris,
  species,
  (yTrue, yPred) => {
    return yTrue.filter((y, i) => y === yPred[i]).length / yTrue.length;
  },
  { nRepeats: 5, seed: 42 }
);

console.log("Feature Importance (Random Forest):");
importance.forEach((imp, i) => {
  const featureNames = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"];
  console.log(`  ${featureNames[imp.feature]}: ${imp.importance.toFixed(4)} ± ${imp.std.toFixed(4)}`);
});

// %% [markdown]
/*
### Feature Importance Plot

Visualize which features contribute most to predictions.
*/

// %% [javascript]
// Add feature names
const featureNames = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"];
globalThis.importance_with_names = importance.map((imp, i) => ({
  ...imp,
  feature: featureNames[i]
}));

globalThis.fi_config = ds.plot.plotFeatureImportance(importance_with_names, {
  width: 640,
  height: 400,
  topN: 4
});

console.log("\nFeature Importance Plot Configuration:");
console.log(`  Features: ${fi_config.data.features.length}`);
console.log(`  Top feature: ${fi_config.data.features[0].feature}`);

// %% [markdown]
/*
### Partial Dependence Plot

Shows how predictions change with one feature, holding others constant.
*/

// %% [javascript]
// Compute partial dependence for petal length (feature 2)
globalThis.pd = ds.ml.interpret.partialDependence(rf, X_iris, 2, {
  gridSize: 20,
  percentiles: [0.05, 0.95]
});

globalThis.pd_config = ds.plot.plotPartialDependence(pd, {
  width: 640,
  height: 400,
  featureName: "Petal Length"
});

console.log("\nPartial Dependence Configuration:");
console.log(`  Feature: ${pd.feature}`);
console.log(`  Grid points: ${pd.values.length}`);
console.log(`  Range: [${pd.range[0].toFixed(2)}, ${pd.range[1].toFixed(2)}]`);

// %% [markdown]
/*
## 4. Regression Diagnostics

Visualize residuals and model fit.
*/

// %% [javascript]
// Load cars data for regression
globalThis.carsData = await fetch("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json")
  .then(r => r.json())
  .then(data => data.filter(d => d.Horsepower != null && d.Miles_per_Gallon != null));

globalThis.X_cars = carsData.map(d => [d.Horsepower, d.Weight_in_lbs]);
globalThis.y_cars = carsData.map(d => d.Miles_per_Gallon);

// Fit linear model
globalThis.lm = ds.stats.lm.fit(X_cars, y_cars, { intercept: true });

// Get residual data
globalThis.residual_data = ds.ml.interpret.residualPlotData(lm, X_cars, y_cars);

// %% [markdown]
/*
### Residual Plot

Check for patterns in residuals (should be random).
*/

// %% [javascript]
globalThis.resid_config = ds.plot.plotResiduals(residual_data, {
  width: 640,
  height: 400,
  standardized: false
});

console.log("Residual Plot Configuration:");
console.log(`  Points: ${resid_config.data.points.length}`);
console.log(`  Residual mean: ${residual_data.residuals.reduce((a,b) => a+b, 0) / residual_data.residuals.length}`);
console.log(`  Residual std: ${Math.sqrt(residual_data.residuals.reduce((s,r) => s + r**2, 0) / residual_data.residuals.length).toFixed(2)}`);

// %% [markdown]
/*
### Q-Q Plot

Check if residuals are normally distributed.
*/

// %% [javascript]
globalThis.qq_config = ds.plot.plotQQ(residual_data, {
  width: 400,
  height: 400
});

console.log("\nQ-Q Plot Configuration:");
console.log(`  Points: ${qq_config.data.points.length}`);
console.log(`  Purpose: Check normality of residuals`);

// %% [markdown]
/*
## 5. Correlation Matrix

Visualize relationships between all variables.
*/

// %% [javascript]
globalThis.corr = ds.ml.interpret.correlationMatrix(X_iris, [
  "Sepal Length",
  "Sepal Width",
  "Petal Length",
  "Petal Width"
]);

globalThis.corr_config = ds.plot.plotCorrelationMatrix(corr, {
  width: 640,
  height: 600
});

console.log("Correlation Matrix Configuration:");
console.log(`  Features: ${corr.features.length}`);
console.log(`  Cells: ${corr_config.data.cells.length}`);

// Show highest correlations
console.log("\nHighest correlations:");
const correlations = [];
for (let i = 0; i < corr.features.length; i++) {
  for (let j = i + 1; j < corr.features.length; j++) {
    correlations.push({
      pair: `${corr.features[i]} - ${corr.features[j]}`,
      value: Math.abs(corr.matrix[i][j])
    });
  }
}
correlations.sort((a, b) => b.value - a.value);
correlations.slice(0, 3).forEach(c => {
  console.log(`  ${c.pair}: ${c.value.toFixed(3)}`);
});

// %% [markdown]
/*
## 6. Hierarchical Clustering Dendrogram

Visualize hierarchical relationships.
*/

// %% [javascript]
// Fit HCA on subset
globalThis.X_subset = X_iris.slice(0, 30);
globalThis.hca = new ds.mva.HCA({ linkage: "average" });
hca.fit(X_subset);

globalThis.dendro_config = ds.plot.plotHCA(hca.model, {
  width: 800,
  height: 400
});

console.log("\nDendrogram Configuration:");
console.log(`  Samples: ${X_subset.length}`);
console.log(`  Merges: ${hca.model.dendrogram.length}`);
console.log(`  Max height: ${hca.model.dendrogram[hca.model.dendrogram.length - 1].distance.toFixed(2)}`);

// %% [markdown]
/*
## Summary

@tangent/ds provides comprehensive visualization configurations:

**Ordination Plots**
- `ordiplot(result, options)` - Unified plot for PCA, LDA, RDA
- `plotPCA(result, options)` - Specific PCA biplot
- `plotScree(result, options)` - Variance explained plot
- `plotHCA(result, options)` - Dendrogram

**Classification Metrics**
- `plotROC(yTrue, yProb, options)` - ROC curve with AUC
- `plotPrecisionRecall(yTrue, yProb, options)` - PR curve
- `plotConfusionMatrix(yTrue, yPred, options)` - Confusion matrix
- `plotCalibration(yTrue, yProb, options)` - Calibration curve

**Model Interpretation**
- `plotFeatureImportance(importance, options)` - Feature importance bars
- `plotPartialDependence(pd, options)` - Partial dependence curve
- `plotResiduals(residualData, options)` - Residual plot
- `plotQQ(residualData, options)` - Q-Q plot for normality
- `plotCorrelationMatrix(corr, options)` - Correlation heatmap

All functions return Observable Plot configurations that can be:
- Rendered with `Plot.plot(config)` in Observable
- Converted to other plotting libraries
- Inspected for data and styling
*/

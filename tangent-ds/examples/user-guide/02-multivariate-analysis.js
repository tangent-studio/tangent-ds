// ---
// title: Multivariate Analysis with @tangent/ds
// id: multivariate-analysis
// ---

// %% [markdown]
/*
# Multivariate Analysis with @tangent/ds

This notebook covers multivariate statistical methods for exploring and understanding
multidimensional data:

- **PCA (Principal Component Analysis)**: Dimensionality reduction and variance explanation
- **LDA (Linear Discriminant Analysis)**: Supervised dimensionality reduction
- **RDA (Redundancy Analysis)**: Constrained ordination
- **HCA (Hierarchical Clustering)**: Agglomerative clustering with dendrograms

These methods are essential for:
- Exploratory data analysis
- Data visualization
- Feature extraction
- Understanding relationships in high-dimensional data
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

console.log(`Loaded ${irisData.length} iris samples with ${X_iris[0].length} features`);

// %% [markdown]
/*
## 1. Principal Component Analysis (PCA)

PCA finds orthogonal directions (principal components) that maximize variance.
It's useful for:
- Dimensionality reduction
- Data visualization
- Noise reduction
- Understanding data structure
*/

// %% [javascript]
// Fit PCA with scaling
globalThis.pca = new ds.mva.PCA({ scale: true, center: true });
pca.fit(X_iris);

console.log("PCA Results:");
console.log(`  Components: ${pca.model.eigenvalues.length}`);
console.log(`  Variance explained: [${pca.model.varianceExplained.map(v => (v*100).toFixed(1) + '%').join(", ")}]`);
console.log(`  Cumulative variance: [${pca.cumulativeVariance().map(v => (v*100).toFixed(1) + '%').join(", ")}]`);

// Transform data to PC space
globalThis.pca_scores = pca.transform(X_iris);
console.log(`\nTransformed shape: ${pca_scores.length} × ${pca_scores[0].length}`);
console.log(`First sample in PC space: [${pca_scores[0].map(v => v.toFixed(2)).join(", ")}]`);

// %% [markdown]
/*
### PCA Loadings

Loadings show how original variables contribute to each principal component.
*/

// %% [javascript]
globalThis.loadings = pca.model.loadings;
const featureNames = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"];

console.log("\nPC1 Loadings (direction of maximum variance):");
featureNames.forEach((name, i) => {
  console.log(`  ${name.padEnd(13)}: ${loadings[i].pc1.toFixed(3)}`);
});

console.log("\nPC2 Loadings (second most important direction):");
featureNames.forEach((name, i) => {
  console.log(`  ${name.padEnd(13)}: ${loadings[i].pc2.toFixed(3)}`);
});

// %% [markdown]
/*
## 2. Linear Discriminant Analysis (LDA)

LDA finds linear combinations of features that best separate classes.
Unlike PCA, LDA is supervised and maximizes between-class variance.
*/

// %% [javascript]
globalThis.lda = new ds.mva.LDA();
lda.fit(X_iris, species);

console.log("LDA Results:");
console.log(`  Classes: ${lda.model.classes.length} (${lda.model.classes.join(", ")})`);
console.log(`  Discriminant axes: ${lda.model.discriminantAxes.length}`);
console.log(`  Variance explained: [${lda.model.varianceExplained.map(v => (v*100).toFixed(1) + '%').join(", ")}]`);

// Transform to discriminant space
globalThis.lda_scores = lda.transform(X_iris);
console.log(`\nLD1 range: [${Math.min(...lda_scores.map(s => s[0])).toFixed(2)}, ${Math.max(...lda_scores.map(s => s[0])).toFixed(2)}]`);
console.log(`LD2 range: [${Math.min(...lda_scores.map(s => s[1])).toFixed(2)}, ${Math.max(...lda_scores.map(s => s[1])).toFixed(2)}]`);

// %% [markdown]
/*
### LDA for Classification

LDA can also predict class labels for new observations.
*/

// %% [javascript]
// Test classification on a few samples
globalThis.test_samples = X_iris.slice(0, 5);
globalThis.test_labels = species.slice(0, 5);
globalThis.lda_predictions = lda.predict(test_samples);

console.log("LDA Classification:");
test_labels.forEach((actual, i) => {
  const pred = lda_predictions[i];
  const match = actual === pred ? "✓" : "✗";
  console.log(`  ${match} Actual: ${actual.padEnd(11)} Predicted: ${pred}`);
});

// %% [markdown]
/*
## 3. Redundancy Analysis (RDA)

RDA is a constrained ordination method that relates two sets of variables (Y ~ X).
It's commonly used in ecology to relate species composition to environmental variables.
*/

// %% [javascript]
// For RDA, let's use petal measurements as response and sepal measurements as predictors
globalThis.Y_rda = X_iris.map(row => [row[2], row[3]]); // Petal length & width
globalThis.X_rda = X_iris.map(row => [row[0], row[1]]); // Sepal length & width

globalThis.rda = new ds.mva.RDA();
rda.fit(Y_rda, X_rda);

console.log("RDA Results:");
console.log(`  Constrained variance: ${(rda.model.constrainedVariance * 100).toFixed(1)}%`);
console.log(`  R² (variance explained by X): ${rda.model.constrainedVariance.toFixed(3)}`);
console.log(`  Canonical axes: ${rda.model.canonicalEigenvalues.length}`);

// The constrained variance tells us how much of Y's variation is explained by X
console.log(`\nInterpretation: ${(rda.model.constrainedVariance * 100).toFixed(1)}% of petal variation is explained by sepal measurements`);

// %% [markdown]
/*
## 4. Hierarchical Clustering Analysis (HCA)

HCA builds a tree (dendrogram) showing hierarchical relationships.
Useful for discovering natural groupings in data.
*/

// %% [javascript]
// Use a subset for clearer visualization
globalThis.X_subset = X_iris.slice(0, 50); // First 50 samples (setosa only)

globalThis.hca = new ds.mva.HCA({
  linkage: "average",  // Can be: single, complete, average
  metric: "euclidean"
});
hca.fit(X_subset);

console.log("HCA Results:");
console.log(`  Samples: ${X_subset.length}`);
console.log(`  Dendrogram merges: ${hca.model.dendrogram.length}`);
console.log(`  Max height: ${hca.model.dendrogram[hca.model.dendrogram.length - 1].distance.toFixed(2)}`);

// %% [markdown]
/*
### Cutting the Dendrogram

Cut the dendrogram at different levels to get different numbers of clusters.
*/

// %% [javascript]
// Cut into 3 clusters
globalThis.clusters_3 = hca.cut(3);
console.log("3 clusters:");
console.log(`  Cluster sizes: [${[0,1,2].map(c => clusters_3.filter(x => x === c).length).join(", ")}]`);

// Cut into 5 clusters
globalThis.clusters_5 = hca.cut(5);
console.log("\n5 clusters:");
console.log(`  Cluster sizes: [${[0,1,2,3,4].map(c => clusters_5.filter(x => x === c).length).join(", ")}]`);

// Cut at specific height
globalThis.clusters_height = hca.cutHeight(2.0);
globalThis.n_clusters_height = new Set(clusters_height).size;
console.log(`\nCutting at height 2.0 gives ${n_clusters_height} clusters`);

// %% [markdown]
/*
## 5. Unified Ordination Plot (Ordiplot)

Use the `ordiplot()` function for consistent visualization across all ordination methods.
*/

// %% [javascript]
// Create ordination plots for PCA
globalThis.pca_plot = ds.plot.ordiplot(pca.model, {
  colorBy: species,
  showLoadings: true,
  showConvexHulls: true,
  loadingScale: 3
});

console.log("PCA Ordiplot:");
console.log(`  Type: ${pca_plot.ordinationType}`);
console.log(`  Axes: ${pca_plot.axes.x.label} vs ${pca_plot.axes.y.label}`);
console.log(`  Points: ${pca_plot.data.scores.length}`);
console.log(`  Loadings: ${pca_plot.data.loadings?.length || 0}`);
console.log(`  Convex hulls: ${pca_plot.data.hulls ? "yes" : "no"}`);

// Create ordination plot for LDA
globalThis.lda_plot = ds.plot.ordiplot(lda.model, {
  showCentroids: true,
  showConvexHulls: true
});

console.log("\nLDA Ordiplot:");
console.log(`  Type: ${lda_plot.ordinationType}`);
console.log(`  Points: ${lda_plot.data.scores.length}`);
console.log(`  Centroids: ${lda_plot.data.centroids?.length || 0}`);

// %% [markdown]
/*
## 6. Comparing Ordination Methods

Let's compare PCA and LDA on the same data.
*/

// %% [javascript]
console.log("Comparison of PCA vs LDA:\n");

console.log("PCA (Unsupervised):");
console.log(`  - Maximizes total variance`);
console.log(`  - PC1 explains ${(pca.model.varianceExplained[0] * 100).toFixed(1)}% of variance`);
console.log(`  - PC2 explains ${(pca.model.varianceExplained[1] * 100).toFixed(1)}% of variance`);
console.log(`  - Ignores class labels`);

console.log("\nLDA (Supervised):");
console.log(`  - Maximizes between-class separation`);
console.log(`  - LD1 explains ${(lda.model.varianceExplained[0] * 100).toFixed(1)}% of between-class variance`);
console.log(`  - LD2 explains ${(lda.model.varianceExplained[1] * 100).toFixed(1)}% of between-class variance`);
console.log(`  - Uses class labels for better separation`);

console.log("\nWhen to use:");
console.log(`  - PCA: Exploratory analysis, dimensionality reduction, feature extraction`);
console.log(`  - LDA: Classification, supervised dimensionality reduction`);
console.log(`  - RDA: Relating two data matrices (e.g., species ~ environment)`);
console.log(`  - HCA: Discovering hierarchical structure, taxonomic relationships`);

// %% [markdown]
/*
## Summary

@tangent/ds provides comprehensive multivariate analysis tools:

**PCA (Principal Component Analysis)**
- `PCA({ scale, center })` - Initialize with options
- `pca.fit(X)` - Fit on data
- `pca.transform(X)` - Project to PC space
- `pca.cumulativeVariance()` - Get cumulative variance explained

**LDA (Linear Discriminant Analysis)**
- `LDA()` - Initialize
- `lda.fit(X, y)` - Fit with class labels
- `lda.transform(X)` - Project to LD space
- `lda.predict(X)` - Classify new observations

**RDA (Redundancy Analysis)**
- `RDA()` - Initialize
- `rda.fit(Y, X)` - Fit Y ~ X relationship
- `rda.transform(X)` - Project X to canonical space

**HCA (Hierarchical Clustering)**
- `HCA({ linkage, metric })` - Initialize with options
- `hca.fit(X)` - Build dendrogram
- `hca.cut(k)` - Cut into k clusters
- `hca.cutHeight(h)` - Cut at height h

**Visualization**
- `ordiplot(result, options)` - Unified ordination plotting for all methods
*/

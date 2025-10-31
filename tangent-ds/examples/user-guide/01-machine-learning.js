// ---
// title: Machine Learning with @tangent/ds
// id: machine-learning
// ---

// %% [markdown]
/*
# Machine Learning with @tangent/ds

This notebook demonstrates the machine learning capabilities of @tangent/ds including:
- **Clustering**: K-Means, Hierarchical Clustering
- **Classification**: KNN, Decision Trees, Random Forests, Logistic Regression
- **Regression**: KNN Regressor, Decision Trees, GAMs
- **Neural Networks**: Multi-Layer Perceptrons (MLPs)

We'll use real datasets from the Vega Datasets collection.
*/

// %% [javascript]
// Import the library
import * as ds from "@tangent/ds";

// Fetch iris dataset
globalThis.irisData = await fetch("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/iris.json")
  .then(r => r.json());

console.log(`Loaded ${irisData.length} iris samples`);
console.log("First sample:", irisData[0]);

// %% [markdown]
/*
## 1. K-Means Clustering

K-Means is an unsupervised learning algorithm that groups similar data points into clusters.
*/

// %% [javascript]
// Extract numeric features for clustering
globalThis.X_iris = irisData.map(d => [
  d.sepalLength,
  d.sepalWidth,
  d.petalLength,
  d.petalWidth
]);

// Fit K-Means with k=3 (we know there are 3 species)
globalThis.kmeans = new ds.ml.KMeans({ k: 3, seed: 42 });
kmeans.fit(X_iris);

console.log("K-Means Results:");
console.log(`  Converged: ${kmeans.converged}`);
console.log(`  Iterations: ${kmeans.iterations}`);
console.log(`  Inertia: ${kmeans.inertia.toFixed(2)}`);
console.log("\nCentroids:");
kmeans.centroids.forEach((c, i) => {
  console.log(`  Cluster ${i}: [${c.map(v => v.toFixed(2)).join(", ")}]`);
});

// %% [markdown]
/*
## 2. Hierarchical Clustering

Hierarchical clustering builds a tree of clusters (dendrogram) and allows cutting at different heights.
*/

// %% [javascript]
// Use a subset for visualization clarity
globalThis.X_subset = X_iris.slice(0, 30);

globalThis.hca = new ds.mva.HCA({ linkage: "average" });
hca.fit(X_subset);

// Cut the dendrogram into 3 clusters
globalThis.hca_labels = hca.cut(3);

console.log("Hierarchical Clustering:");
console.log(`  Dendrogram height: ${hca.model.dendrogram[hca.model.dendrogram.length - 1].distance.toFixed(2)}`);
console.log(`  Cluster labels (first 10): [${hca_labels.slice(0, 10).join(", ")}]`);

// %% [markdown]
/*
## 3. K-Nearest Neighbors (KNN) Classification

KNN is a simple, instance-based learning algorithm that classifies based on the k nearest neighbors.
*/

// %% [javascript]
// Prepare classification data
globalThis.X_train = X_iris.slice(0, 120);
globalThis.y_train = irisData.slice(0, 120).map(d => d.species);
globalThis.X_test = X_iris.slice(120);
globalThis.y_test = irisData.slice(120).map(d => d.species);

// Train KNN classifier
globalThis.knn = new ds.ml.KNNClassifier({ k: 5, weight: "distance" });
knn.fit(X_train, y_train);

// Make predictions
globalThis.y_pred = knn.predict(X_test);

// Calculate accuracy
globalThis.accuracy = y_test.filter((y, i) => y === y_pred[i]).length / y_test.length;
console.log(`KNN Accuracy: ${(accuracy * 100).toFixed(1)}%`);

// Show some predictions
console.log("\nSample predictions:");
for (let i = 0; i < 5; i++) {
  console.log(`  Actual: ${y_test[i].padEnd(15)} Predicted: ${y_pred[i]}`);
}

// %% [markdown]
/*
## 4. Decision Tree Classification

Decision trees learn a tree of if-then-else rules to make predictions.
*/

// %% [javascript]
globalThis.tree = new ds.ml.DecisionTreeClassifier({
  maxDepth: 5,
  minSamplesSplit: 4
});
tree.fit(X_train, y_train);

globalThis.tree_pred = tree.predict(X_test);
globalThis.tree_accuracy = y_test.filter((y, i) => y === tree_pred[i]).length / y_test.length;

console.log(`Decision Tree Accuracy: ${(tree_accuracy * 100).toFixed(1)}%`);

// %% [markdown]
/*
## 5. Random Forest Classification

Random Forests are ensemble methods that combine multiple decision trees for better predictions.
*/

// %% [javascript]
globalThis.rf = new ds.ml.RandomForestClassifier({
  nTrees: 50,
  maxDepth: 5,
  seed: 42
});
rf.fit(X_train, y_train);

globalThis.rf_pred = rf.predict(X_test);
globalThis.rf_accuracy = y_test.filter((y, i) => y === rf_pred[i]).length / y_test.length;

console.log(`Random Forest Accuracy: ${(rf_accuracy * 100).toFixed(1)}%`);
console.log(`Number of trees: ${rf.nTrees}`);

// %% [markdown]
/*
## 6. KNN Regression

KNN can also be used for regression tasks by averaging the values of nearest neighbors.
*/

// %% [javascript]
// Fetch cars dataset for regression
globalThis.carsData = await fetch("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json")
  .then(r => r.json())
  .then(data => data.filter(d => d.Horsepower != null && d.Miles_per_Gallon != null));

// Prepare regression data (predict MPG from Horsepower and Weight)
globalThis.X_cars = carsData.map(d => [d.Horsepower, d.Weight_in_lbs]);
globalThis.y_cars = carsData.map(d => d.Miles_per_Gallon);

// Split data
globalThis.split_idx = Math.floor(X_cars.length * 0.8);
globalThis.X_train_cars = X_cars.slice(0, split_idx);
globalThis.y_train_cars = y_cars.slice(0, split_idx);
globalThis.X_test_cars = X_cars.slice(split_idx);
globalThis.y_test_cars = y_cars.slice(split_idx);

// Train KNN regressor
globalThis.knn_reg = new ds.ml.KNNRegressor({ k: 5, weight: "distance" });
knn_reg.fit(X_train_cars, y_train_cars);

// Evaluate
globalThis.y_pred_cars = knn_reg.predict(X_test_cars);
globalThis.mae = y_test_cars.reduce((sum, y, i) => sum + Math.abs(y - y_pred_cars[i]), 0) / y_test_cars.length;
globalThis.r2 = knn_reg.score(y_test_cars, y_pred_cars);

console.log("KNN Regression Results:");
console.log(`  MAE: ${mae.toFixed(2)} MPG`);
console.log(`  R²: ${r2.toFixed(3)}`);

// %% [markdown]
/*
## 7. Generalized Additive Models (GAMs)

GAMs model non-linear relationships using smooth functions.
*/

// %% [javascript]
globalThis.gam = new ds.ml.GAMRegressor({
  nSplines: 10,
  lambda: 0.01
});
gam.fit(X_train_cars, y_train_cars);

globalThis.gam_pred = gam.predict(X_test_cars);
globalThis.gam_mae = y_test_cars.reduce((sum, y, i) => sum + Math.abs(y - gam_pred[i]), 0) / y_test_cars.length;
globalThis.gam_r2 = gam.score(y_test_cars, gam_pred);

console.log("GAM Regression Results:");
console.log(`  MAE: ${gam_mae.toFixed(2)} MPG`);
console.log(`  R²: ${gam_r2.toFixed(3)}`);

// %% [markdown]
/*
## 8. Multi-Layer Perceptron (MLP) - Neural Network

MLPs are feedforward neural networks that can learn complex non-linear patterns.
*/

// %% [javascript]
// Create a simple neural network for iris classification
// First, encode species as numbers
globalThis.species_map = { "setosa": 0, "versicolor": 1, "virginica": 2 };
globalThis.y_train_encoded = y_train.map(s => species_map[s]);
globalThis.y_test_encoded = y_test.map(s => species_map[s]);

globalThis.mlp = new ds.ml.MLPClassifier({
  hiddenLayers: [10, 5],
  activation: "relu",
  epochs: 100,
  learningRate: 0.01,
  seed: 42
});

mlp.fit(X_train, y_train_encoded);

globalThis.mlp_pred = mlp.predict(X_test);
globalThis.mlp_accuracy = y_test_encoded.filter((y, i) => y === mlp_pred[i]).length / y_test_encoded.length;

console.log(`MLP Accuracy: ${(mlp_accuracy * 100).toFixed(1)}%`);
console.log(`Network architecture: [${mlp.hiddenLayers.join(", ")}]`);

// %% [markdown]
/*
## 9. Model Persistence

Save and load trained models in JSON format.
*/

// %% [javascript]
// Save the KMeans model
globalThis.kmeans_json = kmeans.save();
console.log("Saved KMeans model:");
console.log(`  Size: ${kmeans_json.length} bytes`);

// Load it back
globalThis.kmeans_loaded = ds.ml.KMeans.load(kmeans_json);
console.log("\nLoaded model:");
console.log(`  Centroids: ${kmeans_loaded.centroids.length}`);
console.log(`  Predictions match: ${JSON.stringify(kmeans.predict([[5, 3, 1.5, 0.2]])) === JSON.stringify(kmeans_loaded.predict([[5, 3, 1.5, 0.2]]))}`);

// %% [markdown]
/*
## Summary

@tangent/ds provides a comprehensive set of machine learning algorithms:

**Clustering**
- K-Means: Fast, scalable clustering
- HCA: Hierarchical clustering with dendrograms

**Classification**
- KNN: Instance-based learning
- Decision Trees: Interpretable rules
- Random Forests: Ensemble of trees
- MLP: Neural networks

**Regression**
- KNN Regressor: Non-parametric regression
- Decision Tree Regressor: Tree-based regression
- GAM: Non-linear smooth functions
- MLP: Neural network regression

All models support:
- Standard scikit-learn-like API (fit, predict, score)
- Model persistence (save/load)
- Both array and table-style inputs
*/

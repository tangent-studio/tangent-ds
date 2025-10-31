// ---
// title: Machine Learning
// id: machine-learning
// ---

// %% [markdown]
/*
Machine Learning module examples for @tangent.to/ds
Uses real datasets from Vega Datasets: https://cdn.jsdelivr.net/npm/vega-datasets@2/data/
*/

// %% [markdown]
// # Example 1: K-means Clustering - Penguins Dataset

// %% [javascript]
import * as ds from "@tangent.to/ds";

const { ml } = ds;

// Fetch penguins dataset
const penguinsResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/penguins.json",
);
const penguinsData = await penguinsResponse.json();

penguinsData.slice(0, 5);

// %%

// Prepare clustering data using flipper length and body mass (numeric features)
const validForClustering = penguinsData.filter(
  (d) =>
    d["Flipper Length (mm)"] != null &&
    d["Body Mass (g)"] != null &&
    !isNaN(d["Flipper Length (mm)"]) &&
    !isNaN(d["Body Mass (g)"]),
);

const clusterData = validForClustering.map((d) => [
  d["Flipper Length (mm)"],
  d["Body Mass (g)"],
]);

// %% [javascript]
// Use the class-based KMeans estimator with declarative table-style inputs
const km = new ds.ml.KMeans({ k: 3 });
// validForClustering is an array of row objects (from earlier). Fit using column names.
km.fit({
  data: validForClustering,
  columns: ["Flipper Length (mm)", "Body Mass (g)"],
  omit_missing: true,
});

console.log("K-means clustering on Penguin flipper length vs body mass");
console.log("Cluster labels (first 20):", km.labels.slice(0, 20));
console.log(
  "Centroids:",
  km.centroids.map((c) => c.map((v) => v.toFixed(2))),
);
console.log("Inertia:", km.inertia.toFixed(4));
console.log("Converged:", km.converged);

// %% [javascript]
// Predict cluster for new penguins (flipper_length_mm, body_mass_g)
const newPoints = [
  [180, 3700], // smaller penguin
  [200, 5000], // larger penguin
  [210, 5600], // very large penguin
];
const predictions = km.predict(newPoints);
console.log("Predicted cluster indices for new penguins:", predictions);

// %% [javascript]
// Compute silhouette score (use declarative form to re-use same columns)
const silhouette = km.silhouetteScore({
  data: validForClustering,
  columns: ["Flipper Length (mm)", "Body Mass (g)"],
});
console.log("Silhouette score:", silhouette.toFixed(4));

// %% [javascript]
// Example 2: Polynomial Regression - Cars Dataset
console.log("\n📈 Example 2: Polynomial Regression - Cars Dataset");
console.log("-".repeat(50));

// %% [javascript]
// Fetch cars dataset
const carsResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json",
);
const carsData = await carsResponse.json();

// Use Horsepower to predict acceleration (non-linear relationship)
const validCars = carsData.filter((c) => c.Horsepower && c.Acceleration);
const xPoly = validCars.slice(0, 60).map((c) => c.Horsepower);
const yPoly = validCars.slice(0, 60).map((c) => c.Acceleration);

// %% [javascript]
// Fit polynomial regression using class-based estimator
const polyReg = new ds.ml.PolynomialRegressor({ degree: 2 });
polyReg.fit(xPoly, yPoly);
const polyModel = polyReg.summary();
console.log(
  "Polynomial coefficients (degree 2):",
  polyModel.coefficients.map((c) => c.toFixed(4)),
);
console.log("R-squared:", polyModel.rSquared.toFixed(4));

// %% [javascript]
// Make predictions
const xNew = [100, 150, 200];
const yPred = polyReg.predict(xNew);
console.log(`Predictions for HP=[${xNew}]:`, yPred.map((y) => y.toFixed(2)));

// %% [javascript]
// Example 3: Polynomial Feature Generation
console.log("\n📊 Example 3: Polynomial Features");
console.log("-".repeat(50));

// %% [javascript]
const xFeatures = [[100], [150], [200]];
const poly2 = ml.polynomial.polynomialFeatures(xFeatures, 2);
const poly3 = ml.polynomial.polynomialFeatures(xFeatures, 3);

// %% [javascript]
console.log("Original HP:", xFeatures.flat());
console.log("Degree 2 features (HP, HP^2):", poly2);
console.log("Degree 3 features (HP, HP^2, HP^3):", poly3);

// %% [javascript]
// Example 4: Multilayer Perceptron - Weather Pattern Classification
console.log("\n🧠 Example 4: MLP for Weather Pattern Classification");
console.log("-".repeat(50));

// %% [javascript]
// Fetch Seattle weather data
const weatherResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/seattle-weather.csv",
);
const weatherText = await weatherResponse.text();
const weatherData = weatherText.split("\n").slice(1, 151).map((line) => {
  const parts = line.split(",");
  return {
    precipitation: parseFloat(parts[1]),
    temp_max: parseFloat(parts[2]),
    temp_min: parseFloat(parts[3]),
    hasRain: parseFloat(parts[1]) > 0 ? 1 : 0,
  };
}).filter((d) => !isNaN(d.temp_max) && !isNaN(d.temp_min));

// Normalize features
const tempMaxs = weatherData.map((d) => d.temp_max);
const tempMins = weatherData.map((d) => d.temp_min);
const maxTemp = Math.max(...tempMaxs);
const minTemp = Math.min(...tempMins);

const xMLP = weatherData.map((d) => [
  (d.temp_max - minTemp) / (maxTemp - minTemp),
  (d.temp_min - minTemp) / (maxTemp - minTemp),
]);
const yMLP = weatherData.map((d) => d.hasRain);

const mlpEstimator = new ds.ml.MLPRegressor({
  layerSizes: [2, 6, 1], // 2 inputs (temp max, temp min), 6 hidden, 1 output
  activation: "relu",
  epochs: 100,
  learningRate: 0.05,
  batchSize: 32,
});
mlpEstimator.fit(xMLP, yMLP);
const mlpSummary = mlpEstimator.summary();

console.log(
  "Training complete. Loss decreased from",
  mlpSummary.initialLoss.toFixed(4),
  "to",
  mlpSummary.finalLoss.toFixed(4),
);

// %% [javascript]
// Make predictions on weather patterns
const mlpPredictions = mlpEstimator.predict(xMLP.slice(0, 10));
console.log("Rain predictions (first 10 days):");
mlpPredictions.slice(0, 10).forEach((pred, i) => {
  console.log(
    `  Day ${i + 1}: ${(pred[0] * 100).toFixed(1)}% chance of rain (actual: ${
      yMLP[i] ? "rain" : "no rain"
    })`,
  );
});

// %% [javascript]
// Evaluate on training data
const mlpMetrics = mlpEstimator.evaluate(xMLP, yMLP);
console.log("Training MSE:", mlpMetrics.mse.toFixed(4));
console.log("Training MAE:", mlpMetrics.mae.toFixed(4));

// %% [javascript]
// Example 5: MLP for Non-linear Pattern - Stock Price Movement
console.log("\n🎯 Example 5: MLP for Stock Returns");
console.log("-".repeat(50));

// Fetch stock data (S&P 500)
const stockResponse = await fetch(
  "https://cdn.jsdelivr.net/npm/vega-datasets@2/data/sp500.csv",
);
const stockText = await stockResponse.text();
const stockData = stockText.split("\n").slice(1, 201).map((line) => {
  const parts = line.split(",");
  return {
    date: parts[0],
    price: parseFloat(parts[1]),
  };
}).filter((d) => !isNaN(d.price));

// Create features: price change patterns
const stockFeatures = [];
const stockTargets = [];

for (let i = 3; i < stockData.length; i++) {
  const p1 = stockData[i - 3].price;
  const p2 = stockData[i - 2].price;
  const p3 = stockData[i - 1].price;
  const p4 = stockData[i].price;

  // Features: 3-day price changes (normalized)
  stockFeatures.push([
    (p2 - p1) / p1,
    (p3 - p2) / p2,
  ]);

  // Target: next day up (1) or down (0)
  stockTargets.push(p4 > p3 ? 1 : 0);
}

const xNonlinear = stockFeatures.slice(0, 150);
const yNonlinear = stockTargets.slice(0, 150);

const nonlinearEstimator = new ds.ml.MLPRegressor({
  layerSizes: [2, 8, 1],
  activation: "relu",
  epochs: 200,
  learningRate: 0.05,
  batchSize: 16,
});
nonlinearEstimator.fit(xNonlinear, yNonlinear);
const nonlinearSummary = nonlinearEstimator.summary();

console.log("Stock movement predictor trained");
console.log("Training epochs:", nonlinearSummary.epochs);
console.log(
  "Final loss:",
  nonlinearSummary.finalLoss.toFixed(4),
);

const nonlinearPred = nonlinearEstimator.predict(xNonlinear.slice(0, 5));
console.log("Predictions (first 5):");
xNonlinear.slice(0, 5).forEach((x, i) => {
  console.log(
    `  Pattern ${i + 1} => Prediction: ${
      (nonlinearPred[i][0] * 100).toFixed(1)
    }% (actual: ${yNonlinear[i] ? "up" : "down"})`,
  );
});

// %% [javascript]
// Example 6: Using Penguins Data for Classification
console.log("\n🐧 Example 6: ML with Penguins Data - Species Classification");
console.log("-".repeat(50));

// Use penguinsData (already fetched above). Filter complete cases and take a subset.
const validPenguins = penguinsData.filter((p) =>
  p["Beak Length (mm)"] != null &&
  p["Beak Depth (mm)"] != null &&
  p["Flipper Length (mm)"] != null &&
  p["Body Mass (g)"] != null &&
  p["Species"]
);

// %% [javascript]
// Extract features and encode species as target
const speciesMap = { "Adelie": 0, "Chinstrap": 1, "Gentoo": 2 };
const billLengths = validPenguins.map((p) => p["Beak Length (mm)"]);
const billDepths = validPenguins.map((p) => p["Beak Depth (mm)"]);
const maxLength = Math.max(...billLengths);
const minLength = Math.min(...billLengths);
const maxDepth = Math.max(...billDepths);
const minDepth = Math.min(...billDepths);

const X = validPenguins.map((p) => [
  (p["Beak Length (mm)"] - minLength) / (maxLength - minLength),
  (p["Beak Depth (mm)"] - minDepth) / (maxDepth - minDepth),
]);
const y = validPenguins.map((p) => speciesMap[p["Species"]]);

// %% [javascript]
// Train MLP classifier (simple single-output encoding used for demonstration)
const classifier = new ds.ml.MLPRegressor({
  layerSizes: [2, 8, 1],
  epochs: 150,
  learningRate: 0.1,
});
classifier.fit(X, y);

const classifierSummary = classifier.summary();

console.log("Penguin species classifier trained");
console.log("Final loss:", classifierSummary.finalLoss.toFixed(4));

// %% [javascript]
// Predict for new penguin (normalized bill length & depth)
const newPenguin = [[0.5, 0.5]]; // Middle-range bill measurements
const penguinPred = classifier.predict(newPenguin);
const speciesNames = ["Adelie", "Chinstrap", "Gentoo"];
const predictedSpeciesIdx = Math.round(penguinPred[0][0]);
const clampedIdx = Math.max(
  0,
  Math.min(speciesNames.length - 1, predictedSpeciesIdx),
);
console.log(
  `New penguin prediction score: ${penguinPred[0][0].toFixed(3)} (likely ${
    speciesNames[clampedIdx]
  })`,
);

// %% [javascript]
console.log("\n🧭 Example 7: K-Nearest Neighbors");
console.log("-".repeat(50));

const knnClassifier = new ds.ml.KNNClassifier({ k: 5, weight: "distance" });
knnClassifier.fit({
  data: validPenguins,
  X: ["bill_length_mm", "bill_depth_mm"],
  y: "species",
  omit_missing: true,
});

const knnClassPreds = knnClassifier.predict([
  [40, 18],
  [50, 14],
]);
console.log("KNN class predictions:", knnClassPreds);

const knnRegressor = new ds.ml.KNNRegressor({ k: 4, weight: "distance" });
knnRegressor.fit(xPoly.map((x) => [x]), yPoly);
const knnRegPreds = knnRegressor.predict([[125], [185]]);
console.log("KNN regression predictions:", knnRegPreds.map((v) => v.toFixed(2)));

// %% [javascript]
console.log("\n🌲 Example 8: Decision Trees & Random Forests");
console.log("-".repeat(50));

const treeClf = new ds.ml.DecisionTreeClassifier({ maxDepth: 4, minSamplesSplit: 4 });
treeClf.fit({
  data: validPenguins,
  X: ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"],
  y: "species",
  omit_missing: true,
});
const treePreds = treeClf.predict([[42, 17, 200], [49, 15, 220]]);
console.log("Decision tree predictions:", treePreds);

const treeReg = new ds.ml.DecisionTreeRegressor({ maxDepth: 5, minSamplesSplit: 3 });
treeReg.fit(xPoly.map((x) => [x]), yPoly);
const treeRegPreds = treeReg.predict([[110], [190]]);
console.log("Decision tree regression predictions:", treeRegPreds.map((v) => v.toFixed(2)));

const forestClf = new ds.ml.RandomForestClassifier({
  nEstimators: 50,
  maxDepth: 6,
  seed: 42,
});
forestClf.fit({
  data: validPenguins,
  X: ["bill_length_mm", "bill_depth_mm", "flipper_length_mm"],
  y: "species",
  omit_missing: true,
});
const forestClassPreds = forestClf.predict([[44, 18, 210], [50, 16, 225]]);
console.log("Random forest predictions:", forestClassPreds);

const forestReg = new ds.ml.RandomForestRegressor({
  nEstimators: 60,
  maxDepth: 5,
  seed: 7,
});
forestReg.fit(xPoly.map((x) => [x]), yPoly);
const forestRegPreds = forestReg.predict([[120], [195]]);
console.log("Random forest regression predictions:", forestRegPreds.map((v) => v.toFixed(2)));

// %% [javascript]
console.log("\n📈 Example 9: Generalized Additive Models");
console.log("-".repeat(50));

const smoothX = [];
const smoothY = [];
for (let i = 0; i < 120; i++) {
  const x1 = -2 + (4 * i) / 119;
  const x2 = Math.sin(x1);
  smoothX.push([x1]);
  smoothY.push(x2 + 0.1 * (Math.random() - 0.5));
}

const gamReg = new ds.ml.GAMRegressor({ nSplines: 5 });
gamReg.fit(smoothX, smoothY);
const gamRegPreds = gamReg.predict([[-1.5], [0], [1.5]]);
console.log("GAM regression predictions:", gamRegPreds.map((v) => v.toFixed(3)));

const gamClassifier = new ds.ml.GAMClassifier({ nSplines: 5, maxIter: 80 });
const binaryY = smoothY.map((v) => (v > 0 ? "high" : "low"));
gamClassifier.fit(smoothX, binaryY);
const gamClassPreds = gamClassifier.predict([[-1.5], [0], [1.5]]);
console.log("GAM classification predictions:", gamClassPreds);

console.log("\n" + "=".repeat(70));
console.log("✅ All ML examples completed successfully!");
console.log("=".repeat(70) + "\n");

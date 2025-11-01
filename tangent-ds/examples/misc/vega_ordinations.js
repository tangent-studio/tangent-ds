/**
 * Examples of LDA, CCA, and RDA using datasets from Vega Datasets.
 *
 * Data source: https://cdn.jsdelivr.net/npm/vega-datasets@3.2.1/data/
 *
 * Run with:
 *    node examples/misc/vega_ordinations.js
 */

import { LDA, CCA, RDA } from '../../src/mva/index.js';
import { prepareX } from '../../src/core/table.js';

const DATA_BASE =
  'https://cdn.jsdelivr.net/npm/vega-datasets@3.2.1/data/';

async function fetchJSON(name) {
  const res = await fetch(`${DATA_BASE}${name}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${name}: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function runLDA() {
  console.log('\n=== Linear Discriminant Analysis (LDA) on iris.json ===');
  const iris = await fetchJSON('iris.json');

  const numeric = iris
    .filter((row) =>
      row.sepalLength != null &&
      row.sepalWidth != null &&
      row.petalLength != null &&
      row.petalWidth != null
    );

  const X = numeric.map((row) => [
    row.sepalLength,
    row.sepalWidth,
    row.petalLength,
    row.petalWidth,
  ]);
  const y = numeric.map((row) => row.species);

  const lda = new LDA({ scaling: 0 });
  lda.fit(X, y);

  const model = lda.model;
  console.log('Classes:', model.classes);
  console.log('Eigenvalues:', model.eigenvalues.map((v) => v.toFixed(4)));
  console.log('First 3 scores:', model.scores.slice(0, 3));
}

async function runCCA() {
  console.log('\n=== Canonical Correlation Analysis (CCA) on cars.json ===');
  const carsRaw = await fetchJSON('cars.json');

  const cars = carsRaw.filter((row) =>
    row.Horsepower != null &&
    row.Weight_in_lbs != null &&
    row.Acceleration != null &&
    row.Displacement != null &&
    row.Miles_per_Gallon != null &&
    row.Cylinders != null
  );

  // X: engine / performance characteristics
  const X = cars.map((row) => [
    row.Horsepower,
    row.Weight_in_lbs,
    row.Displacement,
  ]);

  // Y: outcomes of interest
  const Y = cars.map((row) => [
    row.Miles_per_Gallon,
    row.Acceleration,
    row.Cylinders,
  ]);

  const cca = new CCA({ center: true, scale: true });
  cca.fit(X, Y);

  const model = cca.model;
  console.log('Canonical correlations:', model.correlations.map((c) => c.toFixed(4)));
  console.log('Canonical weights (first 3 variables of X):');
  model.xWeights.slice(0, 3).forEach((row) => console.log(row));
  console.log('Canonical weights (Y variables):');
  model.yWeights.forEach((row) => console.log(row));
}

async function runRDA() {
  console.log('\n=== Redundancy Analysis (RDA) on penguins.json ===');
  const penguinsRaw = await fetchJSON('penguins.json');

  const penguins = penguinsRaw.filter((row) =>
    row["Body Mass (g)"] != null &&
    row["Flipper Length (mm)"] != null &&
    row["Bill Length (mm)"] != null &&
    row["Bill Depth (mm)"] != null &&
    row.Species != null &&
    row.Island != null
  );

  const responseCols = [
    "Body Mass (g)",
    "Flipper Length (mm)",
    "Bill Length (mm)",
    "Bill Depth (mm)"
  ];
  const responsePrep = prepareX({
    columns: responseCols,
    data: penguins,
    omit_missing: true
  });

  const predictorPrep = prepareX({
    columns: ['Species', 'Island'],
    data: penguins,
    omit_missing: true,
    encode: {
      Species: 'onehot',
      Island: 'onehot'
    }
  });

  const Y = responsePrep.X;
  const X = predictorPrep.X;

  const rda = new RDA({ scale: true });
  rda.fit(Y, X);

  const model = rda.model;
  console.log('Constrained variance:', model.constrainedVariance.toFixed(4));
  console.log('Eigenvalues:', model.eigenvalues.map((v) => v.toFixed(4)));
  console.log('Predictor columns after encoding:', predictorPrep.columns);
  console.log('First 3 canonical scores (sites):', model.canonicalScores.slice(0, 3));
}

await runLDA();
await runCCA();
await runRDA();

console.log('\nExamples completed.');

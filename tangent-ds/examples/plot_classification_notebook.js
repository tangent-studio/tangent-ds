/**
 * Classification Plotting Examples
 * Demonstrates ROC curves, precision-recall, confusion matrices, and calibration plots
 */

import { logit } from '../src/stats/index.js';
import {
  plotROC,
  plotPrecisionRecall,
  plotConfusionMatrix,
  plotCalibration
} from '../src/plot/index.js';

// ============= Generate synthetic binary classification data =============
function generateClassificationData(n = 200, seed = 42) {
  // Simple PRNG for reproducibility
  let rng = seed;
  const random = () => {
    rng = (rng * 9301 + 49297) % 233280;
    return rng / 233280;
  };

  const X = [];
  const y = [];

  for (let i = 0; i < n; i++) {
    const x1 = random() * 10 - 5;
    const x2 = random() * 10 - 5;

    // True decision boundary: x1 + x2 > 0
    const trueProb = 1 / (1 + Math.exp(-(x1 + x2 + random() * 2 - 1)));
    const label = random() < trueProb ? 1 : 0;

    X.push([x1, x2]);
    y.push(label);
  }

  return { X, y };
}

// ============= Example 1: ROC Curve =============
console.log('=== ROC Curve Example ===\n');

const { X, y } = generateClassificationData(200, 42);

// Fit logistic regression
const model = logit.fit(X, y, { intercept: true });

// Get predicted probabilities
const yProb = model.fitted;

// Generate ROC curve plot configuration
const rocConfig = plotROC(y, yProb, {
  width: 500,
  height: 500,
  showDiagonal: true
});

console.log('ROC Curve Configuration:');
console.log(JSON.stringify(rocConfig, null, 2));
console.log(`\nAUC: ${rocConfig.data.auc.toFixed(3)}`);

// ============= Example 2: Precision-Recall Curve =============
console.log('\n=== Precision-Recall Curve Example ===\n');

const prConfig = plotPrecisionRecall(y, yProb, {
  width: 500,
  height: 500,
  showBaseline: true
});

console.log('Precision-Recall Configuration:');
console.log(`Average Precision: ${prConfig.data.avgPrecision.toFixed(3)}`);
console.log(`Number of points: ${prConfig.data.curve.length}`);

// ============= Example 3: Confusion Matrix =============
console.log('\n=== Confusion Matrix Example ===\n');

// Get predicted labels (threshold = 0.5)
const yPred = yProb.map(p => p > 0.5 ? 1 : 0);

const confMatrixConfig = plotConfusionMatrix(y, yPred, {
  width: 400,
  height: 400,
  normalize: false,
  labels: ['Negative', 'Positive']
});

console.log('Confusion Matrix Configuration:');
console.log(`Number of cells: ${confMatrixConfig.data.cells.length}`);
console.log('Matrix:');
confMatrixConfig.data.cells.forEach(cell => {
  console.log(`  ${cell.true} -> ${cell.predicted}: ${cell.count}`);
});

// Normalized version
const confMatrixNormConfig = plotConfusionMatrix(y, yPred, {
  width: 400,
  height: 400,
  normalize: true,
  labels: ['Negative', 'Positive']
});

console.log('\nNormalized Confusion Matrix:');
confMatrixNormConfig.data.cells.forEach(cell => {
  console.log(`  ${cell.true} -> ${cell.predicted}: ${cell.count.toFixed(2)}`);
});

// ============= Example 4: Calibration Curve =============
console.log('\n=== Calibration Curve Example ===\n');

const calibConfig = plotCalibration(y, yProb, {
  width: 500,
  height: 500,
  nBins: 10
});

console.log('Calibration Curve Configuration:');
console.log(`Number of bins: ${calibConfig.data.curve.length}`);
console.log('Calibration data:');
calibConfig.data.curve.forEach((point, i) => {
  console.log(`  Bin ${i + 1}: Predicted=${point.predicted.toFixed(3)}, Observed=${point.observed.toFixed(3)}`);
});

// ============= Example 5: Multi-class Classification =============
console.log('\n=== Multi-class Confusion Matrix Example ===\n');

// Generate multi-class data
function generateMultiClassData(n = 150, seed = 100) {
  let rng = seed;
  const random = () => {
    rng = (rng * 9301 + 49297) % 233280;
    return rng / 233280;
  };

  const yTrue = [];
  const yPred = [];

  for (let i = 0; i < n; i++) {
    const trueClass = Math.floor(random() * 3); // 0, 1, or 2

    // Simulate predictions with some noise
    const r = random();
    let predClass;
    if (r < 0.7) {
      predClass = trueClass; // 70% correct
    } else if (r < 0.85) {
      predClass = (trueClass + 1) % 3; // 15% off by 1
    } else {
      predClass = (trueClass + 2) % 3; // 15% off by 2
    }

    yTrue.push(trueClass);
    yPred.push(predClass);
  }

  return { yTrue, yPred };
}

const { yTrue: yTrueMulti, yPred: yPredMulti } = generateMultiClassData(150, 100);

const multiConfMatrix = plotConfusionMatrix(yTrueMulti, yPredMulti, {
  width: 500,
  height: 500,
  normalize: false,
  labels: ['Class A', 'Class B', 'Class C']
});

console.log('Multi-class Confusion Matrix:');
console.log(`Classes: ${multiConfMatrix.data.classes.join(', ')}`);

// Calculate accuracy
const correct = yTrueMulti.filter((y, i) => y === yPredMulti[i]).length;
const accuracy = correct / yTrueMulti.length;
console.log(`\nAccuracy: ${(accuracy * 100).toFixed(1)}%`);

// ============= Usage Notes =============
console.log('\n=== Usage Notes ===\n');
console.log('All plot functions return configuration objects for Observable Plot.');
console.log('These can be rendered in Observable notebooks or used with Plot.plot().');
console.log('\nAvailable classification plots:');
console.log('  • plotROC - ROC curve with AUC');
console.log('  • plotPrecisionRecall - Precision-Recall curve with Average Precision');
console.log('  • plotConfusionMatrix - Confusion matrix (binary or multi-class)');
console.log('  • plotCalibration - Calibration curve for probability predictions');

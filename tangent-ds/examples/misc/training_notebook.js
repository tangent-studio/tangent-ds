/**
 * Training Utilities Example
 * Demonstrates unified training loop with loss functions
 * Using Tangent Notebook format
 */

import { ml, stats, core } from '@tangent.to/ds';

console.log('=== Training Utilities Demo ===\n');

// ## Generate Training Data

function generateLinearData(n = 100, noise = 0.5) {
  const X = [];
  const y = [];
  
  for (let i = 0; i < n; i++) {
    const x1 = Math.random() * 10;
    const x2 = Math.random() * 5;
    
    // y = 3*x1 + 2*x2 + 5 + noise
    const target = 3 * x1 + 2 * x2 + 5 + (Math.random() - 0.5) * noise;
    
    X.push([x1, x2]);
    y.push(target);
  }
  
  return { X, y };
}

console.log('=== Linear Regression with Training Loop ===\n');

const { X, y } = generateLinearData(200, 2.0);
console.log(`Generated dataset: ${X.length} samples, ${X[0].length} features`);

// Split data
const split = ml.validation.trainTestSplit(X, y, { testSize: 0.3, shuffle: true });
console.log(`Train: ${split.XTrain.length}, Test: ${split.XTest.length}\n`);

// ## Define Loss Function for Linear Model

function createLinearLoss(XTrain, yTrain) {
  return (params) => {
    // params = [intercept, coef1, coef2]
    const predictions = XTrain.map((x, i) => 
      params[0] + x[0] * params[1] + x[1] * params[2]
    );
    
    const { loss, gradient: predGrad } = ml.loss.mseLoss(yTrain, predictions);
    
    // Compute parameter gradient
    const gradient = [0, 0, 0];
    for (let i = 0; i < XTrain.length; i++) {
      gradient[0] += predGrad[i]; // intercept
      gradient[1] += predGrad[i] * XTrain[i][0]; // coef1
      gradient[2] += predGrad[i] * XTrain[i][1]; // coef2
    }
    
    return { loss, gradient };
  };
}

// ## Train with Different Optimizers

console.log('--- Training with Different Optimizers ---\n');

const optimizers = ['adam', 'momentum', 'rmsprop'];
const results = {};

optimizers.forEach(optName => {
  const lossFn = createLinearLoss(split.XTrain, split.yTrain);
  
  const result = ml.train.trainFunction(lossFn, [0, 0, 0], {
    optimizer: optName,
    optimizerOptions: { learningRate: 0.01 },
    maxIter: 500,
    tol: 1e-6,
    verbose: false
  });
  
  results[optName] = result;
  
  console.log(`${optName.toUpperCase()}:`);
  console.log(`  Parameters: [${result.params.map(p => p.toFixed(3)).join(', ')}]`);
  console.log(`  Final loss: ${result.history.loss[result.history.loss.length - 1].toFixed(6)}`);
  console.log(`  Iterations: ${result.history.loss.length}\n`);
});

// ## Evaluate on Test Set

console.log('--- Test Set Evaluation ---\n');

Object.entries(results).forEach(([name, result]) => {
  const testPred = split.XTest.map(x => 
    result.params[0] + x[0] * result.params[1] + x[1] * result.params[2]
  );
  
  const testMSE = ml.metrics.mse(split.yTest, testPred);
  const testR2 = ml.metrics.r2(split.yTest, testPred);
  
  console.log(`${name.toUpperCase()}:`);
  console.log(`  Test MSE: ${testMSE.toFixed(4)}`);
  console.log(`  Test R²: ${testR2.toFixed(4)}\n`);
});

// ## Training with Callbacks

console.log('=== Training with Callbacks ===\n');

const lossFn = createLinearLoss(split.XTrain, split.yTrain);

let epochLosses = [];
const callbacks = {
  onEpochEnd: (epoch, logs) => {
    epochLosses.push(logs.loss);
    if (epoch % 100 === 0) {
      console.log(`Epoch ${epoch}: loss=${logs.loss.toFixed(6)}`);
    }
  }
};

const finalResult = ml.train.trainFunction(lossFn, [0, 0, 0], {
  optimizer: 'adam',
  optimizerOptions: { learningRate: 0.01 },
  maxIter: 500,
  verbose: false,
  callbacks
});

console.log(`\nFinal parameters: [${finalResult.params.map(p => p.toFixed(3)).join(', ')}]`);
console.log(`True parameters: ~[5.000, 3.000, 2.000]`);

// ## Loss Function Comparison

console.log('\n=== Loss Function Comparison ===\n');

const yTrue = [1, 2, 3, 4, 5];
const yPred1 = [1.1, 2.1, 3.1, 4.1, 5.1]; // Good predictions
const yPred2 = [2, 3, 4, 5, 6]; // Offset predictions

console.log('Good Predictions vs Offset Predictions:\n');

const mse1 = ml.loss.mseLoss(yTrue, yPred1);
const mse2 = ml.loss.mseLoss(yTrue, yPred2);

console.log('MSE Loss:');
console.log(`  Good: ${mse1.loss.toFixed(6)}`);
console.log(`  Offset: ${mse2.loss.toFixed(6)}\n`);

const mae1 = ml.loss.maeLoss(yTrue, yPred1);
const mae2 = ml.loss.maeLoss(yTrue, yPred2);

console.log('MAE Loss:');
console.log(`  Good: ${mae1.loss.toFixed(6)}`);
console.log(`  Offset: ${mae2.loss.toFixed(6)}\n`);

const huber1 = ml.loss.huberLoss(yTrue, yPred1, 1.0);
const huber2 = ml.loss.huberLoss(yTrue, yPred2, 1.0);

console.log('Huber Loss (delta=1.0):');
console.log(`  Good: ${huber1.loss.toFixed(6)}`);
console.log(`  Offset: ${huber2.loss.toFixed(6)}\n`);

// ## Binary Classification Loss

console.log('=== Binary Classification (Log Loss) ===\n');

const yTrueBinary = [1, 1, 0, 0, 1];
const yPredGood = [0.9, 0.85, 0.1, 0.15, 0.88];
const yPredBad = [0.4, 0.6, 0.6, 0.4, 0.5];

const logLoss1 = ml.loss.logLoss(yTrueBinary, yPredGood);
const logLoss2 = ml.loss.logLoss(yTrueBinary, yPredBad);

console.log('Good Predictions:');
console.log(`  Log Loss: ${logLoss1.loss.toFixed(6)}\n`);

console.log('Poor Predictions:');
console.log(`  Log Loss: ${logLoss2.loss.toFixed(6)}\n`);

// ## Convergence Visualization

console.log('=== Convergence Analysis ===\n');

const convergenceData = results.adam.history.loss;

console.log('Loss progression (every 50 iterations):');
for (let i = 0; i < convergenceData.length; i += 50) {
  const loss = convergenceData[i];
  const improvement = i > 0 ? ((convergenceData[0] - loss) / convergenceData[0] * 100) : 0;
  console.log(`  Iter ${i}: loss=${loss.toFixed(6)}, improvement=${improvement.toFixed(1)}%`);
}

console.log('\n✓ Training utilities demo complete');
console.log('\nKey features demonstrated:');
console.log('- Unified training interface');
console.log('- Multiple optimizers');
console.log('- Loss function selection');
console.log('- Training callbacks');
console.log('- Convergence monitoring');

/**
 * Model interpretation utilities
 * Feature importance, partial dependence, residuals, correlation
 */

import { mean } from '../core/math.js';
import { shuffle, setSeed } from './utils.js';

/**
 * Compute feature importance via permutation
 * @param {Object} model - Fitted model with predict method
 * @param {Array<Array<number>>} X - Feature matrix
 * @param {Array<number>} y - Target values
 * @param {Function} scoreFn - Scoring function (yTrue, yPred) => score
 * @param {Object} options - {nRepeats, seed}
 * @returns {Array<Object>} Feature importance scores
 */
export function featureImportance(model, X, y, scoreFn, { nRepeats = 10, seed = null } = {}) {
  if (seed !== null) {
    setSeed(seed);
  }
  
  const nFeatures = X[0].length;
  const baselinePred = model.predict(X);
  const baselineScore = scoreFn(y, baselinePred);
  
  const importances = [];
  
  for (let feature = 0; feature < nFeatures; feature++) {
    const scores = [];
    
    for (let repeat = 0; repeat < nRepeats; repeat++) {
      // Permute this feature
      const XPermuted = X.map(row => [...row]);
      const featureColumn = X.map(row => row[feature]);
      const shuffledFeature = shuffle(featureColumn);
      
      XPermuted.forEach((row, i) => {
        row[feature] = shuffledFeature[i];
      });
      
      // Score with permuted feature
      const permutedPred = model.predict(XPermuted);
      const permutedScore = scoreFn(y, permutedPred);
      
      // Importance = drop in score
      scores.push(baselineScore - permutedScore);
    }
    
    const meanImportance = mean(scores);
    const stdImportance = Math.sqrt(
      scores.reduce((sum, s) => sum + (s - meanImportance) ** 2, 0) / scores.length
    );
    
    importances.push({
      feature,
      importance: meanImportance,
      std: stdImportance
    });
  }
  
  return importances.sort((a, b) => b.importance - a.importance);
}

/**
 * Compute coefficient-based feature importance (for linear models)
 * @param {Object} model - Fitted linear model with coefficients
 * @param {Array<string>} featureNames - Feature names (optional)
 * @returns {Array<Object>} Feature importance based on coefficients
 */
export function coefficientImportance(model, featureNames = null) {
  if (!model.coefficients) {
    throw new Error('Model must have coefficients property');
  }
  
  const coeffs = model.coefficients;
  const nFeatures = model.intercept !== undefined ? coeffs.length - 1 : coeffs.length;
  const startIdx = model.intercept !== undefined ? 1 : 0;
  
  const importances = [];
  
  for (let i = 0; i < nFeatures; i++) {
    importances.push({
      feature: featureNames ? featureNames[i] : i,
      importance: Math.abs(coeffs[startIdx + i]),
      coefficient: coeffs[startIdx + i]
    });
  }
  
  return importances.sort((a, b) => b.importance - a.importance);
}

/**
 * Compute partial dependence for a feature
 * @param {Object} model - Fitted model with predict method
 * @param {Array<Array<number>>} X - Feature matrix
 * @param {number} feature - Feature index
 * @param {Object} options - {gridSize, percentiles}
 * @returns {Object} {values, predictions}
 */
export function partialDependence(model, X, feature, { 
  gridSize = 20, 
  percentiles = [0.05, 0.95] 
} = {}) {
  const nSamples = X.length;
  
  // Get feature values and range
  const featureValues = X.map(row => row[feature]).sort((a, b) => a - b);
  const minIdx = Math.floor(nSamples * percentiles[0]);
  const maxIdx = Math.floor(nSamples * percentiles[1]);
  const minVal = featureValues[minIdx];
  const maxVal = featureValues[maxIdx];
  
  // Create grid
  const grid = [];
  for (let i = 0; i < gridSize; i++) {
    grid.push(minVal + (maxVal - minVal) * i / (gridSize - 1));
  }
  
  // Compute predictions for each grid point
  const predictions = [];
  
  for (const value of grid) {
    // Create modified dataset with feature set to value
    const XModified = X.map(row => {
      const newRow = [...row];
      newRow[feature] = value;
      return newRow;
    });
    
    // Predict and average
    const preds = model.predict(XModified);
    const avgPred = Array.isArray(preds[0]) 
      ? mean(preds.map(p => p[0])) 
      : mean(preds);
    
    predictions.push(avgPred);
  }
  
  return {
    feature,
    values: grid,
    predictions,
    range: [minVal, maxVal]
  };
}

/**
 * Compute residual plot data
 * @param {Object} model - Fitted model with predict method
 * @param {Array<Array<number>>} X - Feature matrix
 * @param {Array<number>} y - Target values
 * @returns {Object} {fitted, residuals, standardized}
 */
export function residualPlotData(model, X, y) {
  const predictions = model.predict(X);
  const fitted = Array.isArray(predictions[0]) 
    ? predictions.map(p => p[0]) 
    : predictions;
  
  const residuals = y.map((yi, i) => yi - fitted[i]);
  
  // Standardized residuals
  const residualMean = mean(residuals);
  const residualStd = Math.sqrt(
    residuals.reduce((sum, r) => sum + (r - residualMean) ** 2, 0) / residuals.length
  );
  
  const standardized = residuals.map(r => (r - residualMean) / residualStd);
  
  return {
    fitted,
    residuals,
    standardized,
    yTrue: y
  };
}

/**
 * Compute correlation matrix
 * @param {Array<Array<number>>} X - Feature matrix
 * @param {Array<string>} featureNames - Feature names (optional)
 * @returns {Object} {matrix, features}
 */
export function correlationMatrix(X, featureNames = null) {
  const n = X.length;
  const p = X[0].length;
  
  // Compute means
  const means = [];
  for (let j = 0; j < p; j++) {
    const col = X.map(row => row[j]);
    means.push(mean(col));
  }
  
  // Compute standard deviations
  const stds = [];
  for (let j = 0; j < p; j++) {
    const col = X.map(row => row[j]);
    const variance = col.reduce((sum, val) => sum + (val - means[j]) ** 2, 0) / (n - 1);
    stds.push(Math.sqrt(variance));
  }
  
  // Compute correlation matrix
  const matrix = [];
  for (let i = 0; i < p; i++) {
    const row = [];
    for (let j = 0; j < p; j++) {
      if (i === j) {
        row.push(1);
      } else {
        let cov = 0;
        for (let k = 0; k < n; k++) {
          cov += (X[k][i] - means[i]) * (X[k][j] - means[j]);
        }
        cov /= (n - 1);
        
        const corr = stds[i] > 0 && stds[j] > 0 
          ? cov / (stds[i] * stds[j]) 
          : 0;
        row.push(corr);
      }
    }
    matrix.push(row);
  }
  
  const features = featureNames || Array.from({ length: p }, (_, i) => `Feature ${i}`);
  
  return {
    matrix,
    features
  };
}

/**
 * Learning curve data (performance vs training size)
 * @param {Function} fitFn - Function to fit model: (X, y) => model
 * @param {Function} scoreFn - Scoring function: (yTrue, yPred) => score
 * @param {Array<Array<number>>} X - Feature matrix
 * @param {Array} y - Target values
 * @param {Object} options - {trainSizes, cv}
 * @returns {Object} {trainSizes, trainScores, testScores}
 */
export function learningCurve(fitFn, scoreFn, X, y, { 
  trainSizes = [0.1, 0.3, 0.5, 0.7, 0.9], 
  cv = 5 
} = {}) {
  const n = X.length;
  const foldSize = Math.floor(n / cv);
  
  const results = {
    trainSizes: [],
    trainScores: [],
    testScores: []
  };
  
  for (const ratio of trainSizes) {
    const trainSize = Math.floor(n * ratio);
    results.trainSizes.push(trainSize);
    
    const cvTrainScores = [];
    const cvTestScores = [];
    
    // Cross-validation at this training size
    for (let fold = 0; fold < cv; fold++) {
      const testStart = fold * foldSize;
      const testEnd = fold === cv - 1 ? n : (fold + 1) * foldSize;
      
      const trainIndices = [];
      const testIndices = [];
      
      for (let i = 0; i < n; i++) {
        if (i >= testStart && i < testEnd) {
          testIndices.push(i);
        } else if (trainIndices.length < trainSize) {
          trainIndices.push(i);
        }
      }
      
      const XTrain = trainIndices.map(i => X[i]);
      const yTrain = trainIndices.map(i => y[i]);
      const XTest = testIndices.map(i => X[i]);
      const yTest = testIndices.map(i => y[i]);
      
      const model = fitFn(XTrain, yTrain);
      
      const trainPred = model.predict(XTrain);
      const testPred = model.predict(XTest);
      
      cvTrainScores.push(scoreFn(yTrain, trainPred));
      cvTestScores.push(scoreFn(yTest, testPred));
    }
    
    results.trainScores.push(mean(cvTrainScores));
    results.testScores.push(mean(cvTestScores));
  }
  
  return results;
}

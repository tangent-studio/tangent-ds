/**
 * Model evaluation metrics
 * Regression and classification metrics
 */

import { mean } from '../core/math.js';

// ============= Regression Metrics =============

/**
 * Mean Squared Error
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @returns {number} MSE
 */
export function mse(yTrue, yPred) {
  if (yTrue.length !== yPred.length) {
    throw new Error('yTrue and yPred must have same length');
  }
  
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    sum += (yTrue[i] - yPred[i]) ** 2;
  }
  return sum / yTrue.length;
}

/**
 * Root Mean Squared Error
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @returns {number} RMSE
 */
export function rmse(yTrue, yPred) {
  return Math.sqrt(mse(yTrue, yPred));
}

/**
 * Mean Absolute Error
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @returns {number} MAE
 */
export function mae(yTrue, yPred) {
  if (yTrue.length !== yPred.length) {
    throw new Error('yTrue and yPred must have same length');
  }
  
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    sum += Math.abs(yTrue[i] - yPred[i]);
  }
  return sum / yTrue.length;
}

/**
 * R² (coefficient of determination)
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @returns {number} R²
 */
export function r2(yTrue, yPred) {
  if (yTrue.length !== yPred.length) {
    throw new Error('yTrue and yPred must have same length');
  }
  
  const yMean = mean(yTrue);
  
  let ssRes = 0;
  let ssTot = 0;
  
  for (let i = 0; i < yTrue.length; i++) {
    ssRes += (yTrue[i] - yPred[i]) ** 2;
    ssTot += (yTrue[i] - yMean) ** 2;
  }
  
  return 1 - (ssRes / ssTot);
}

// ============= Classification Metrics =============

/**
 * Accuracy score
 * @param {Array} yTrue - True labels
 * @param {Array} yPred - Predicted labels
 * @returns {number} Accuracy
 */
export function accuracy(yTrue, yPred) {
  if (yTrue.length !== yPred.length) {
    throw new Error('yTrue and yPred must have same length');
  }
  
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === yPred[i]) correct++;
  }
  
  return correct / yTrue.length;
}

/**
 * Confusion matrix
 * @param {Array} yTrue - True labels
 * @param {Array} yPred - Predicted labels
 * @returns {Object} {matrix, labels}
 */
export function confusionMatrix(yTrue, yPred) {
  const labels = [...new Set([...yTrue, ...yPred])].sort();
  const n = labels.length;
  const matrix = Array(n).fill(null).map(() => Array(n).fill(0));
  
  const labelIndex = new Map(labels.map((label, i) => [label, i]));
  
  for (let i = 0; i < yTrue.length; i++) {
    const trueIdx = labelIndex.get(yTrue[i]);
    const predIdx = labelIndex.get(yPred[i]);
    matrix[trueIdx][predIdx]++;
  }
  
  return { matrix, labels };
}

/**
 * Precision score (for binary classification)
 * @param {Array} yTrue - True labels
 * @param {Array} yPred - Predicted labels
 * @param {*} positiveLabel - Label to consider as positive
 * @returns {number} Precision
 */
export function precision(yTrue, yPred, positiveLabel = 1) {
  let tp = 0;
  let fp = 0;
  
  for (let i = 0; i < yTrue.length; i++) {
    if (yPred[i] === positiveLabel) {
      if (yTrue[i] === positiveLabel) {
        tp++;
      } else {
        fp++;
      }
    }
  }
  
  return tp + fp === 0 ? 0 : tp / (tp + fp);
}

/**
 * Recall score (for binary classification)
 * @param {Array} yTrue - True labels
 * @param {Array} yPred - Predicted labels
 * @param {*} positiveLabel - Label to consider as positive
 * @returns {number} Recall
 */
export function recall(yTrue, yPred, positiveLabel = 1) {
  let tp = 0;
  let fn = 0;
  
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === positiveLabel) {
      if (yPred[i] === positiveLabel) {
        tp++;
      } else {
        fn++;
      }
    }
  }
  
  return tp + fn === 0 ? 0 : tp / (tp + fn);
}

/**
 * F1 score
 * @param {Array} yTrue - True labels
 * @param {Array} yPred - Predicted labels
 * @param {*} positiveLabel - Label to consider as positive
 * @returns {number} F1 score
 */
export function f1(yTrue, yPred, positiveLabel = 1) {
  const prec = precision(yTrue, yPred, positiveLabel);
  const rec = recall(yTrue, yPred, positiveLabel);
  
  return prec + rec === 0 ? 0 : 2 * (prec * rec) / (prec + rec);
}

/**
 * Log loss (cross-entropy loss)
 * @param {Array} yTrue - True labels (0 or 1)
 * @param {Array<number>} yPred - Predicted probabilities
 * @param {number} eps - Small constant to avoid log(0)
 * @returns {number} Log loss
 */
export function logLoss(yTrue, yPred, eps = 1e-15) {
  if (yTrue.length !== yPred.length) {
    throw new Error('yTrue and yPred must have same length');
  }
  
  let sum = 0;
  for (let i = 0; i < yTrue.length; i++) {
    const p = Math.max(eps, Math.min(1 - eps, yPred[i]));
    sum += yTrue[i] * Math.log(p) + (1 - yTrue[i]) * Math.log(1 - p);
  }
  
  return -sum / yTrue.length;
}

/**
 * ROC AUC score (simplified for binary classification)
 * @param {Array} yTrue - True labels (0 or 1)
 * @param {Array<number>} yPred - Predicted probabilities
 * @returns {number} AUC
 */
export function rocAuc(yTrue, yPred) {
  if (yTrue.length !== yPred.length) {
    throw new Error('yTrue and yPred must have same length');
  }
  
  // Create pairs of (probability, true_label)
  const pairs = yTrue.map((label, i) => ({ prob: yPred[i], label }));
  
  // Sort by probability descending
  pairs.sort((a, b) => b.prob - a.prob);
  
  let auc = 0;
  let tpCount = 0;
  let fpCount = 0;
  
  const nPositive = yTrue.filter(y => y === 1).length;
  const nNegative = yTrue.length - nPositive;
  
  if (nPositive === 0 || nNegative === 0) {
    return 0.5; // Undefined, return 0.5
  }
  
  for (const pair of pairs) {
    if (pair.label === 1) {
      tpCount++;
    } else {
      auc += tpCount;
      fpCount++;
    }
  }
  
  return auc / (nPositive * nNegative);
}

/**
 * Cohen's Kappa coefficient
 * @param {Array} yTrue - True labels
 * @param {Array} yPred - Predicted labels
 * @returns {number} Kappa
 */
export function cohenKappa(yTrue, yPred) {
  const { matrix, labels } = confusionMatrix(yTrue, yPred);
  const n = yTrue.length;
  const k = labels.length;
  
  // Observed accuracy
  let po = 0;
  for (let i = 0; i < k; i++) {
    po += matrix[i][i];
  }
  po /= n;
  
  // Expected accuracy
  let pe = 0;
  for (let i = 0; i < k; i++) {
    const rowSum = matrix[i].reduce((a, b) => a + b, 0);
    const colSum = matrix.reduce((sum, row) => sum + row[i], 0);
    pe += (rowSum * colSum) / (n * n);
  }
  
  return (po - pe) / (1 - pe);
}

/**
 * Adjusted Rand Index
 * @param {Array} yTrue - True labels
 * @param {Array} yPred - Predicted labels
 * @returns {number} Adjusted Rand Index
 */
export function adjustedRandIndex(yTrue, yPred) {
  const n = yTrue.length;
  
  // Build contingency table
  const trueLabels = [...new Set(yTrue)];
  const predLabels = [...new Set(yPred)];
  
  const contingency = {};
  for (const tl of trueLabels) {
    contingency[tl] = {};
    for (const pl of predLabels) {
      contingency[tl][pl] = 0;
    }
  }
  
  for (let i = 0; i < n; i++) {
    contingency[yTrue[i]][yPred[i]]++;
  }
  
  // Compute sums
  const trueSums = {};
  const predSums = {};
  
  for (const tl of trueLabels) {
    trueSums[tl] = Object.values(contingency[tl]).reduce((a, b) => a + b, 0);
  }
  
  for (const pl of predLabels) {
    predSums[pl] = 0;
    for (const tl of trueLabels) {
      predSums[pl] += contingency[tl][pl];
    }
  }
  
  // Compute index
  const comb2 = x => x * (x - 1) / 2;
  
  let sumComb = 0;
  for (const tl of trueLabels) {
    for (const pl of predLabels) {
      sumComb += comb2(contingency[tl][pl]);
    }
  }
  
  const sumTrueComb = Object.values(trueSums).reduce((sum, x) => sum + comb2(x), 0);
  const sumPredComb = Object.values(predSums).reduce((sum, x) => sum + comb2(x), 0);
  
  const expectedIndex = sumTrueComb * sumPredComb / comb2(n);
  const maxIndex = (sumTrueComb + sumPredComb) / 2;
  const index = sumComb;
  
  return (index - expectedIndex) / (maxIndex - expectedIndex);
}

/**
 * Model validation utilities
 * Cross-validation, train-test split, and other validation strategies
 */

import { shuffle as shuffleUtil, setSeed } from './utils.js';

/**
 * Split data into train and test sets
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values (optional)
 * @param {Object} options - {ratio, shuffle, seed}
 * @returns {Object} {XTrain, XTest, yTrain, yTest, trainIndices, testIndices}
 */
export function trainTestSplit(X, y = null, { 
  ratio = 0.8, 
  shuffle = true, 
  seed = null 
} = {}) {
  const n = X.length;
  
  if (seed !== null) {
    setSeed(seed);
  }
  
  let indices = Array.from({ length: n }, (_, i) => i);
  
  if (shuffle) {
    indices = shuffleUtil(indices);
  }
  
  const splitPoint = Math.floor(n * ratio);
  const trainIndices = indices.slice(0, splitPoint);
  const testIndices = indices.slice(splitPoint);
  
  const XTrain = trainIndices.map(i => X[i]);
  const XTest = testIndices.map(i => X[i]);
  
  if (y === null) {
    return { XTrain, XTest, trainIndices, testIndices };
  }
  
  const yTrain = trainIndices.map(i => y[i]);
  const yTest = testIndices.map(i => y[i]);
  
  return { XTrain, XTest, yTrain, yTest, trainIndices, testIndices };
}

/**
 * K-Fold cross-validation generator
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {number} k - Number of folds
 * @param {boolean} shuffle - Whether to shuffle data
 * @returns {Array<Object>} Array of fold objects
 */
export function kFold(X, y, k = 5, shuffle = false) {
  const n = X.length;
  
  if (k < 2 || k > n) {
    throw new Error(`k must be between 2 and ${n}`);
  }
  
  let indices = Array.from({ length: n }, (_, i) => i);
  
  if (shuffle) {
    indices = shuffleUtil(indices);
  }
  
  const foldSize = Math.floor(n / k);
  const folds = [];
  
  for (let i = 0; i < k; i++) {
    const start = i * foldSize;
    const end = i === k - 1 ? n : (i + 1) * foldSize;
    
    const testIndices = indices.slice(start, end);
    const trainIndices = [...indices.slice(0, start), ...indices.slice(end)];
    
    folds.push({
      trainIndices,
      testIndices,
      fold: i + 1,
      nSplits: k,
      description: 'KFold'
    });
  }
  
  return folds;
}

/**
 * Stratified K-Fold for classification with balanced class distribution
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target labels
 * @param {number} k - Number of folds
 * @returns {Array<Object>} Array of fold objects
 */
export function stratifiedKFold(X, y, k = 5) {
  const n = X.length;
  
  if (k < 2 || k > n) {
    throw new Error(`k must be between 2 and ${n}`);
  }
  
  // Group indices by class
  const classIndices = new Map();
  y.forEach((label, idx) => {
    if (!classIndices.has(label)) {
      classIndices.set(label, []);
    }
    classIndices.get(label).push(idx);
  });
  
  // Shuffle indices within each class
  for (const [label, indices] of classIndices) {
    classIndices.set(label, shuffleUtil(indices));
  }
  
  // Distribute samples to folds maintaining class proportions
  const folds = Array.from({ length: k }, () => ({ train: [], test: [] }));
  
  for (const [label, indices] of classIndices) {
    const classSize = indices.length;
    const foldSize = Math.floor(classSize / k);
    
    for (let i = 0; i < k; i++) {
      const start = i * foldSize;
      const end = i === k - 1 ? classSize : (i + 1) * foldSize;
      const testIndices = indices.slice(start, end);
      const trainIndices = [...indices.slice(0, start), ...indices.slice(end)];
      
      folds[i].test.push(...testIndices);
      folds[i].train.push(...trainIndices);
    }
  }
  
  return folds.map((fold, i) => ({
    trainIndices: fold.train,
    testIndices: fold.test,
    fold: i + 1,
    nSplits: k,
    description: 'StratifiedKFold'
  }));
}

/**
 * Group K-Fold keeping group membership intact
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {Array} groups - Group labels
 * @param {number} k - Number of folds
 * @returns {Array<Object>} Array of fold objects
 */
export function groupKFold(X, y, groups, k = 5) {
  const n = X.length;
  
  if (groups.length !== n) {
    throw new Error('groups must have same length as X');
  }
  
  // Get unique groups
  const uniqueGroups = [...new Set(groups)];
  const nGroups = uniqueGroups.length;
  
  if (k > nGroups) {
    throw new Error(`k (${k}) cannot be greater than number of groups (${nGroups})`);
  }
  
  // Shuffle groups
  const shuffledGroups = shuffleUtil(uniqueGroups);
  
  // Distribute groups to folds
  const groupsPerFold = Math.floor(nGroups / k);
  const folds = [];
  
  for (let i = 0; i < k; i++) {
    const start = i * groupsPerFold;
    const end = i === k - 1 ? nGroups : (i + 1) * groupsPerFold;
    const testGroups = new Set(shuffledGroups.slice(start, end));
    
    const trainIndices = [];
    const testIndices = [];
    
    groups.forEach((group, idx) => {
      if (testGroups.has(group)) {
        testIndices.push(idx);
      } else {
        trainIndices.push(idx);
      }
    });
    
    folds.push({
      trainIndices,
      testIndices,
      fold: i + 1,
      nSplits: k,
      description: 'GroupKFold'
    });
  }
  
  return folds;
}

/**
 * Leave-One-Out cross-validation
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @returns {Array<Object>} Array of fold objects
 */
export function leaveOneOut(X, y) {
  const n = X.length;
  const folds = [];
  
  for (let i = 0; i < n; i++) {
    const testIndices = [i];
    const trainIndices = [...Array.from({ length: i }, (_, j) => j), 
                          ...Array.from({ length: n - i - 1 }, (_, j) => i + j + 1)];
    
    folds.push({
      trainIndices,
      testIndices,
      fold: i + 1,
      nSplits: n,
      description: 'LeaveOneOut'
    });
  }
  
  return folds;
}

/**
 * Shuffle Split - repeated random train-test splits
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {Object} options - {nSplits, testRatio, seed}
 * @returns {Array<Object>} Array of split objects
 */
export function shuffleSplit(X, y, { 
  nSplits = 5, 
  testRatio = 0.2, 
  seed = null 
} = {}) {
  const n = X.length;
  
  if (seed !== null) {
    setSeed(seed);
  }
  
  const testSize = Math.floor(n * testRatio);
  const splits = [];
  
  for (let i = 0; i < nSplits; i++) {
    const indices = shuffleUtil(Array.from({ length: n }, (_, j) => j));
    const testIndices = indices.slice(0, testSize);
    const trainIndices = indices.slice(testSize);
    
    splits.push({
      trainIndices,
      testIndices,
      split: i + 1,
      nSplits,
      description: 'ShuffleSplit'
    });
  }
  
  return splits;
}

/**
 * Execute cross-validation with a model
 * @param {Function} fitFn - Function that fits model: (XTrain, yTrain) => model
 * @param {Function} scoreFn - Function that scores model: (model, XTest, yTest) => score
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {Array<Object>} folds - Fold objects from kFold, etc.
 * @returns {Object} {scores, meanScore, stdScore}
 */
export function crossValidate(fitFn, scoreFn, X, y, folds) {
  const scores = [];
  
  for (const fold of folds) {
    const { trainIndices, testIndices } = fold;
    
    const XTrain = trainIndices.map(i => X[i]);
    const yTrain = trainIndices.map(i => y[i]);
    const XTest = testIndices.map(i => X[i]);
    const yTest = testIndices.map(i => y[i]);
    
    const model = fitFn(XTrain, yTrain);
    const score = scoreFn(model, XTest, yTest);
    scores.push(score);
  }
  
  const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance = scores.reduce((sum, score) => sum + (score - meanScore) ** 2, 0) / scores.length;
  const stdScore = Math.sqrt(variance);
  
  return {
    scores,
    meanScore,
    stdScore,
    nFolds: folds.length
  };
}

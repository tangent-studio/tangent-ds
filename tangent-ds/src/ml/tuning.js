/**
 * Hyperparameter tuning utilities
 * GridSearchCV and RandomSearchCV for model selection
 */

import { kFold, crossValidate } from './validation.js';
import { random as seededRandom, setSeed } from './utils.js';

/**
 * Grid Search Cross-Validation
 * @param {Function} fitFn - Function (X, y, params) => model
 * @param {Function} scoreFn - Function (model, X, y) => score
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {Object} paramGrid - Parameter grid {param1: [values], param2: [values]}
 * @param {Object} options - {k, shuffle, metric}
 * @returns {Object} {bestParams, bestScore, bestModel, results}
 */
export function GridSearchCV(fitFn, scoreFn, X, y, paramGrid, options = {}) {
  const {
    k = 5,
    shuffle = true,
    verbose = true
  } = options;

  // Generate all parameter combinations
  const paramCombinations = generateParamCombinations(paramGrid);

  if (verbose) {
    console.log(`GridSearchCV: ${paramCombinations.length} parameter combinations to try`);
  }

  let bestScore = -Infinity;
  let bestParams = null;
  let bestModel = null;
  const results = [];

  // Try each parameter combination
  for (let i = 0; i < paramCombinations.length; i++) {
    const params = paramCombinations[i];

    if (verbose) {
      console.log(`\n[${i + 1}/${paramCombinations.length}] Testing params:`, params);
    }

    try {
      // Create folds for cross-validation
      const folds = kFold(X, y, k, shuffle);

      // Cross-validate with these parameters
      const cvResult = crossValidate(
        (Xtrain, ytrain) => fitFn(Xtrain, ytrain, params),
        scoreFn,
        X,
        y,
        folds
      );

      const meanScore = cvResult.meanScore;
      const stdScore = cvResult.stdScore;

      results.push({
        params: { ...params },
        meanScore,
        stdScore,
        scores: cvResult.scores
      });

      if (verbose) {
        console.log(`  Mean score: ${meanScore.toFixed(6)} ± ${stdScore.toFixed(6)}`);
      }

      // Update best if better
      if (meanScore > bestScore) {
        bestScore = meanScore;
        bestParams = { ...params };
        // Fit model on full dataset with best params
        bestModel = fitFn(X, y, params);

        if (verbose) {
          console.log(`  *** New best score! ***`);
        }
      }
    } catch (error) {
      if (verbose) {
        console.log(`  Error with params:`, error.message);
      }
      results.push({
        params: { ...params },
        meanScore: -Infinity,
        stdScore: 0,
        error: error.message
      });
    }
  }

  if (verbose) {
    console.log(`\nBest parameters found:`, bestParams);
    console.log(`Best cross-validation score: ${bestScore.toFixed(6)}`);
  }

  return {
    bestParams,
    bestScore,
    bestModel,
    results
  };
}

/**
 * Random Search Cross-Validation
 * @param {Function} fitFn - Function (X, y, params) => model
 * @param {Function} scoreFn - Function (model, X, y) => score
 * @param {Array} X - Feature matrix
 * @param {Array} y - Target values
 * @param {Object} paramDistributions - Parameter distributions
 * @param {Object} options - {nIter, k, shuffle, seed}
 * @returns {Object} {bestParams, bestScore, bestModel, results}
 */
export function RandomSearchCV(fitFn, scoreFn, X, y, paramDistributions, options = {}) {
  const {
    nIter = 20,
    k = 5,
    shuffle = true,
    seed = null,
    verbose = true
  } = options;

  if (seed !== null) {
    setSeed(seed);
  }

  if (verbose) {
    console.log(`RandomSearchCV: ${nIter} random parameter combinations to try`);
  }

  let bestScore = -Infinity;
  let bestParams = null;
  let bestModel = null;
  const results = [];

  // Try random parameter combinations
  for (let i = 0; i < nIter; i++) {
    // Sample random parameters
    const params = sampleParams(paramDistributions);

    if (verbose) {
      console.log(`\n[${i + 1}/${nIter}] Testing params:`, params);
    }

    try {
      // Create folds for cross-validation
      const folds = kFold(X, y, k, shuffle);

      // Cross-validate with these parameters
      const cvResult = crossValidate(
        (Xtrain, ytrain) => fitFn(Xtrain, ytrain, params),
        scoreFn,
        X,
        y,
        folds
      );

      const meanScore = cvResult.meanScore;
      const stdScore = cvResult.stdScore;

      results.push({
        params: { ...params },
        meanScore,
        stdScore,
        scores: cvResult.scores
      });

      if (verbose) {
        console.log(`  Mean score: ${meanScore.toFixed(6)} ± ${stdScore.toFixed(6)}`);
      }

      // Update best if better
      if (meanScore > bestScore) {
        bestScore = meanScore;
        bestParams = { ...params };
        // Fit model on full dataset with best params
        bestModel = fitFn(X, y, params);

        if (verbose) {
          console.log(`  *** New best score! ***`);
        }
      }
    } catch (error) {
      if (verbose) {
        console.log(`  Error with params:`, error.message);
      }
      results.push({
        params: { ...params },
        meanScore: -Infinity,
        stdScore: 0,
        error: error.message
      });
    }
  }

  if (verbose) {
    console.log(`\nBest parameters found:`, bestParams);
    console.log(`Best cross-validation score: ${bestScore.toFixed(6)}`);
  }

  return {
    bestParams,
    bestScore,
    bestModel,
    results
  };
}

/**
 * Generate all combinations from parameter grid
 * @param {Object} paramGrid
 * @returns {Array<Object>} Array of parameter combinations
 */
function generateParamCombinations(paramGrid) {
  const keys = Object.keys(paramGrid);
  if (keys.length === 0) return [{}];

  const combinations = [];

  function recurse(index, current) {
    if (index === keys.length) {
      combinations.push({ ...current });
      return;
    }

    const key = keys[index];
    const values = paramGrid[key];

    for (const value of values) {
      current[key] = value;
      recurse(index + 1, current);
    }
  }

  recurse(0, {});
  return combinations;
}

/**
 * Sample parameters from distributions
 * @param {Object} paramDistributions
 * @returns {Object} Sampled parameters
 */
function sampleParams(paramDistributions) {
  const params = {};

  for (const [key, distribution] of Object.entries(paramDistributions)) {
    if (Array.isArray(distribution)) {
      // Discrete uniform distribution
      const idx = Math.floor(seededRandom() * distribution.length);
      params[key] = distribution[idx];
    } else if (typeof distribution === 'object' && distribution.type) {
      // Distribution object
      params[key] = sampleFromDistribution(distribution);
    } else {
      throw new Error(`Invalid distribution for parameter ${key}`);
    }
  }

  return params;
}

/**
 * Sample from a distribution object
 * @param {Object} distribution - {type, ...params}
 * @returns {number} Sampled value
 */
function sampleFromDistribution(distribution) {
  const { type } = distribution;

  switch (type) {
    case 'uniform':
      return distribution.low + seededRandom() * (distribution.high - distribution.low);

    case 'loguniform':
      const logLow = Math.log(distribution.low);
      const logHigh = Math.log(distribution.high);
      return Math.exp(logLow + seededRandom() * (logHigh - logLow));

    case 'randint':
      return distribution.low + Math.floor(seededRandom() * (distribution.high - distribution.low));

    case 'choice':
      const idx = Math.floor(seededRandom() * distribution.options.length);
      return distribution.options[idx];

    default:
      throw new Error(`Unknown distribution type: ${type}`);
  }
}

/**
 * Create parameter distribution objects
 */
export const distributions = {
  /**
   * Uniform distribution
   * @param {number} low
   * @param {number} high
   * @returns {Object}
   */
  uniform: (low, high) => ({ type: 'uniform', low, high }),

  /**
   * Log-uniform distribution (for learning rates, etc.)
   * @param {number} low
   * @param {number} high
   * @returns {Object}
   */
  loguniform: (low, high) => ({ type: 'loguniform', low, high }),

  /**
   * Random integer
   * @param {number} low
   * @param {number} high
   * @returns {Object}
   */
  randint: (low, high) => ({ type: 'randint', low, high }),

  /**
   * Choice from options
   * @param {Array} options
   * @returns {Object}
   */
  choice: (options) => ({ type: 'choice', options })
};

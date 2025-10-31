/**
 * Linear Mixed Models (LMM)
 * Simple random-intercept model using REML
 */

import { toMatrix, Matrix } from '../core/linalg.js';
import { mean, sum } from '../core/math.js';

/**
 * Fit simple random-intercept linear mixed model
 * @param {Array<Array<number>>|Matrix} X - Fixed effects design matrix
 * @param {Array<number>} y - Response vector
 * @param {Array<number|string>} groups - Group indicators
 * @param {Object} options - {intercept: boolean, maxIter: number, tol: number}
 * @returns {Object} {fixedEffects, randomEffects, varFixed, varRandom, logLikelihood}
 */
export function fit(X, y, groups, { intercept = true, maxIter = 100, tol = 1e-6 } = {}) {
  let designMatrix = toMatrix(X);
  const n = designMatrix.rows;
  const responseVector = Array.isArray(y) ? y : Array.from(y);
  
  if (n !== responseVector.length || n !== groups.length) {
    throw new Error('X, y, and groups must have same length');
  }
  
  // Add intercept column if requested
  if (intercept) {
    const withIntercept = [];
    for (let i = 0; i < n; i++) {
      const row = [1];
      for (let j = 0; j < designMatrix.columns; j++) {
        row.push(designMatrix.get(i, j));
      }
      withIntercept.push(row);
    }
    designMatrix = new Matrix(withIntercept);
  }
  
  const p = designMatrix.columns;
  
  // Map groups to indices
  const uniqueGroups = [...new Set(groups)];
  const nGroups = uniqueGroups.length;
  const groupMap = new Map(uniqueGroups.map((g, i) => [g, i]));
  const groupIndices = groups.map(g => groupMap.get(g));
  
  // Initialize variance components
  let sigmaE2 = 1; // residual variance
  let sigmaU2 = 1; // random intercept variance
  
  let fixedEffects = new Array(p).fill(0);
  let randomEffects = new Array(nGroups).fill(0);
  
  // Simplified REML iteration
  for (let iter = 0; iter < maxIter; iter++) {
    const prevSigmaE2 = sigmaE2;
    const prevSigmaU2 = sigmaU2;
    
    // E-step: compute conditional expectations
    const yResid = responseVector.map((yi, i) => {
      let pred = randomEffects[groupIndices[i]];
      for (let j = 0; j < p; j++) {
        pred += designMatrix.get(i, j) * fixedEffects[j];
      }
      return yi - pred;
    });
    
    // M-step: update parameters
    
    // Update fixed effects (weighted least squares)
    const XtX = designMatrix.transpose().mmul(designMatrix);
    const Xty = designMatrix.transpose().mmul(Matrix.columnVector(responseVector));
    
    try {
      const betaSolution = XtX.solve(Xty);
      fixedEffects = betaSolution.to1DArray();
    } catch (e) {
      // If singular, keep previous estimates
    }
    
    // Update random effects (group means of residuals)
    const groupResiduals = new Array(nGroups).fill(0);
    const groupCounts = new Array(nGroups).fill(0);
    
    for (let i = 0; i < n; i++) {
      let resid = responseVector[i];
      for (let j = 0; j < p; j++) {
        resid -= designMatrix.get(i, j) * fixedEffects[j];
      }
      const gIdx = groupIndices[i];
      groupResiduals[gIdx] += resid;
      groupCounts[gIdx]++;
    }
    
    for (let g = 0; g < nGroups; g++) {
      if (groupCounts[g] > 0) {
        // Shrinkage estimate
        const groupMean = groupResiduals[g] / groupCounts[g];
        const shrinkage = sigmaU2 / (sigmaU2 + sigmaE2 / groupCounts[g]);
        randomEffects[g] = shrinkage * groupMean;
      }
    }
    
    // Update variance components
    let ssResid = 0;
    for (let i = 0; i < n; i++) {
      let pred = randomEffects[groupIndices[i]];
      for (let j = 0; j < p; j++) {
        pred += designMatrix.get(i, j) * fixedEffects[j];
      }
      ssResid += (responseVector[i] - pred) ** 2;
    }
    sigmaE2 = ssResid / (n - p);
    
    const ssRandom = sum(randomEffects.map(u => u ** 2));
    sigmaU2 = ssRandom / nGroups;
    
    // Check convergence
    const deltaE = Math.abs(sigmaE2 - prevSigmaE2);
    const deltaU = Math.abs(sigmaU2 - prevSigmaU2);
    
    if (deltaE < tol && deltaU < tol) {
      break;
    }
  }
  
  // Compute log-likelihood (approximation)
  let logLik = 0;
  for (let i = 0; i < n; i++) {
    let pred = randomEffects[groupIndices[i]];
    for (let j = 0; j < p; j++) {
      pred += designMatrix.get(i, j) * fixedEffects[j];
    }
    const resid = responseVector[i] - pred;
    logLik -= 0.5 * Math.log(2 * Math.PI * sigmaE2) - 0.5 * (resid ** 2) / sigmaE2;
  }
  
  return {
    fixedEffects,
    randomEffects,
    varResidual: sigmaE2,
    varRandom: sigmaU2,
    logLikelihood: logLik,
    nGroups,
    groupMap: Object.fromEntries(groupMap)
  };
}

/**
 * Predict using fitted LMM
 * @param {Object} model - Fitted model
 * @param {Array<Array<number>>|Matrix} X - Design matrix
 * @param {Array<number|string>} groups - Group indicators
 * @param {Object} options - {intercept: boolean}
 * @returns {Array<number>} Predictions
 */
export function predict(model, X, groups, { intercept = true } = {}) {
  let designMatrix = toMatrix(X);
  const n = designMatrix.rows;
  
  if (n !== groups.length) {
    throw new Error('X and groups must have same length');
  }
  
  // Add intercept column if requested
  if (intercept) {
    const withIntercept = [];
    for (let i = 0; i < n; i++) {
      const row = [1];
      for (let j = 0; j < designMatrix.columns; j++) {
        row.push(designMatrix.get(i, j));
      }
      withIntercept.push(row);
    }
    designMatrix = new Matrix(withIntercept);
  }
  
  const predictions = new Array(n);
  const { fixedEffects, randomEffects, groupMap } = model;
  
  for (let i = 0; i < n; i++) {
    let pred = 0;
    
    // Add fixed effects
    for (let j = 0; j < fixedEffects.length; j++) {
      pred += designMatrix.get(i, j) * fixedEffects[j];
    }
    
    // Add random effect if group is known
    const groupKey = groups[i];
    if (groupMap[groupKey] !== undefined) {
      pred += randomEffects[groupMap[groupKey]];
    }
    // If group is new, only use fixed effects (pred stays as is)
    
    predictions[i] = pred;
  }
  
  return predictions;
}

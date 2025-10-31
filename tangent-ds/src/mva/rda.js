/**
 * Redundancy Analysis (RDA)
 * Constrained ordination - PCA on fitted values from multiple regression
 */

import { toMatrix, solveLeastSquares, Matrix } from '../core/linalg.js';
import * as pca from './pca.js';
import { mean } from '../core/math.js';

/**
 * Fit RDA model
 * @param {Array<Array<number>>} Y - Response matrix (n x q)
 * @param {Array<Array<number>>} X - Explanatory matrix (n x p)
 * @param {Object} options - {scale: boolean}
 * @returns {Object} RDA model
 */
export function fit(Y, X, { scale = false } = {}) {
  const responseData = Y.map(row => Array.isArray(row) ? row : [row]);
  const explData = X.map(row => Array.isArray(row) ? row : [row]);
  
  const n = responseData.length;
  const q = responseData[0].length;
  const p = explData[0].length;
  
  if (n !== explData.length) {
    throw new Error('Y and X must have same number of rows');
  }
  
  if (n < p + 2) {
    throw new Error('Need more samples than explanatory variables');
  }
  
  // Center Y and X
  const YMeans = [];
  const XMeans = [];
  
  for (let j = 0; j < q; j++) {
    const col = responseData.map(row => row[j]);
    YMeans.push(mean(col));
  }
  
  for (let j = 0; j < p; j++) {
    const col = explData.map(row => row[j]);
    XMeans.push(mean(col));
  }
  
  const YCentered = responseData.map(row => 
    row.map((val, j) => val - YMeans[j])
  );
  
  const XCentered = explData.map(row => 
    row.map((val, j) => val - XMeans[j])
  );
  
  // Fit Y ~ X using multiple regression for each column of Y
  const YFitted = [];
  const YResiduals = [];
  const coefficients = [];
  
  for (let j = 0; j < q; j++) {
    const yCol = YCentered.map(row => row[j]);
    
    // Solve: X * beta = y
    const XMat = new Matrix(XCentered);
    const yVec = Matrix.columnVector(yCol);
    const betaVec = solveLeastSquares(XMat, yVec);
    const beta = betaVec.to1DArray();
    
    coefficients.push(beta);
    
    // Compute fitted and residuals
    const fitted = [];
    const residuals = [];
    for (let i = 0; i < n; i++) {
      let yhat = 0;
      for (let k = 0; k < p; k++) {
        yhat += XCentered[i][k] * beta[k];
      }
      fitted.push(yhat);
      residuals.push(yCol[i] - yhat);
    }
    
    YFitted.push(fitted);
    YResiduals.push(residuals);
  }
  
  // Transpose fitted values to get n x q matrix
  const fittedMatrix = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < q; j++) {
      row.push(YFitted[j][i]);
    }
    fittedMatrix.push(row);
  }
  
  // Perform PCA on fitted values
  const pcaModel = pca.fit(fittedMatrix, { scale, center: false });
  
  // Rename PC scores to canonical axes
  const canonicalScores = pcaModel.scores.map(score => {
    const newScore = {};
    Object.keys(score).forEach(key => {
      if (key.startsWith('pc')) {
        const num = key.slice(2);
        newScore[`rda${num}`] = score[key];
      }
    });
    return newScore;
  });
  
  // Rename loadings
  const canonicalLoadings = pcaModel.loadings.map(loading => {
    const newLoading = { variable: loading.variable };
    Object.keys(loading).forEach(key => {
      if (key.startsWith('pc')) {
        const num = key.slice(2);
        newLoading[`rda${num}`] = loading[key];
      }
    });
    return newLoading;
  });
  
  // Compute total and explained inertia
  let totalInertia = 0;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < n; i++) {
      totalInertia += YCentered[i][j] ** 2;
    }
  }
  totalInertia /= n;
  
  let explainedInertia = 0;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < n; i++) {
      explainedInertia += fittedMatrix[i][j] ** 2;
    }
  }
  explainedInertia /= n;
  
  const constrainedVariance = explainedInertia / totalInertia;
  
  return {
    canonicalScores,
    canonicalLoadings,
    eigenvalues: pcaModel.eigenvalues,
    varianceExplained: pcaModel.varianceExplained,
    constrainedVariance,
    coefficients,
    YMeans,
    XMeans,
    n,
    p,
    q
  };
}

/**
 * Transform new data using fitted RDA model
 * @param {Object} model - Fitted RDA model
 * @param {Array<Array<number>>} Y - New response data
 * @param {Array<Array<number>>} X - New explanatory data
 * @returns {Array<Object>} Canonical scores
 */
export function transform(model, Y, X) {
  const { coefficients, YMeans, XMeans, canonicalLoadings } = model;
  
  const responseData = Y.map(row => Array.isArray(row) ? row : [row]);
  const explData = X.map(row => Array.isArray(row) ? row : [row]);
  
  const n = responseData.length;
  const q = responseData[0].length;
  const p = explData[0].length;
  
  // Center data
  const YCentered = responseData.map(row => 
    row.map((val, j) => val - YMeans[j])
  );
  
  const XCentered = explData.map(row => 
    row.map((val, j) => val - XMeans[j])
  );
  
  // Compute fitted values
  const fittedMatrix = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < q; j++) {
      let yhat = 0;
      for (let k = 0; k < p; k++) {
        yhat += XCentered[i][k] * coefficients[j][k];
      }
      row.push(yhat);
    }
    fittedMatrix.push(row);
  }
  
  // Extract loading matrix
  const nAxes = canonicalLoadings[0] ? Object.keys(canonicalLoadings[0]).length - 1 : 0;
  const loadingMatrix = [];
  for (let j = 0; j < nAxes; j++) {
    const col = canonicalLoadings.map(l => l[`rda${j + 1}`]);
    loadingMatrix.push(col);
  }
  
  // Project onto canonical axes
  const scores = [];
  for (const row of fittedMatrix) {
    const score = {};
    for (let j = 0; j < nAxes; j++) {
      let sum = 0;
      for (let k = 0; k < q; k++) {
        sum += row[k] * loadingMatrix[j][k];
      }
      score[`rda${j + 1}`] = sum;
    }
    scores.push(score);
  }
  
  return scores;
}

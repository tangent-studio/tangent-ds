/**
 * Ordinary Least Squares (OLS) Linear Regression
 * Uses normal equations: β = (X'X)^-1 X'y
 */

import { Matrix, solveLeastSquares, toMatrix } from "../core/linalg.js";
import { mean, sum } from "../core/math.js";
import { prepareXY } from "../core/table.js";

/**
 * Fit OLS linear regression
 * Supports two calling styles:
 *  - Array/matrix style (backward compatible): fit(X, y, { intercept })
 *  - Declarative table style: fit({ X: 'col' | ['col1','col2'], y: 'target', data, omit_missing, intercept })
 *
 * @param {Array<Array<number>>|Matrix|Object} X - Design matrix or options object when using table style
 * @param {Array<number>|string} y - Response vector or (when using array/matrix style) ignored if options object provided
 * @param {Object} options - { intercept: boolean, omit_missing: boolean } (when using array/matrix style)
 * @returns {Object} {coefficients, fitted, residuals, rSquared, adjRSquared, se, n, p}
 */
export function fit(X, y, { intercept = true, omit_missing = true } = {}) {
  // Support declarative options object: fit({ X: ..., y: ..., data: ..., intercept, omit_missing })
  if (
    X && typeof X === "object" && !Array.isArray(X) && "X" in X && "y" in X &&
    "data" in X
  ) {
    const prepared = prepareXY({
      X: X.X,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined
        ? X.omit_missing
        : omit_missing,
    });
    // replace X and y with prepared numeric arrays
    y = prepared.y;
    X = prepared.X;
    // allow intercept override from options object
    interceptor: void 0;
    intercept = X.intercept !== undefined
      ? X.intercept
      : (X.intercept === undefined ? intercept : X.intercept);
    // Note: above line intentionally preserves provided intercept flag if present
  }

  let designMatrix = toMatrix(X);
  const n = designMatrix.rows;
  const responseVector = Array.isArray(y) ? y : Array.from(y);

  if (n !== responseVector.length) {
    throw new Error("X and y must have same number of rows");
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

  // Solve using least squares
  const coeffMatrix = solveLeastSquares(designMatrix, responseVector);
  const coefficients = coeffMatrix.to1DArray();

  // Compute fitted values
  const fittedMatrix = designMatrix.mmul(coeffMatrix);
  const fitted = fittedMatrix.to1DArray();

  // Compute residuals
  const residuals = responseVector.map((yi, i) => yi - fitted[i]);

  // R-squared
  const yMean = mean(responseVector);
  const sst = sum(responseVector.map((yi) => (yi - yMean) ** 2));
  const sse = sum(residuals.map((r) => r ** 2));
  const rSquared = 1 - sse / sst;

  // Adjusted R-squared
  const p = coefficients.length;
  const adjRSquared = 1 - (1 - rSquared) * (n - 1) / (n - p);

  // Standard error of regression
  const se = Math.sqrt(sse / (n - p));

  return {
    coefficients,
    fitted,
    residuals,
    rSquared,
    adjRSquared,
    se,
    n,
    p,
  };
}

/**
 * Predict using fitted model
 * @param {Array<number>} coefficients - Model coefficients
 * @param {Array<Array<number>>|Matrix} X - New design matrix
 * @param {Object} options - {intercept: boolean}
 * @returns {Array<number>} Predictions
 */
export function predict(coefficients, X, { intercept = true } = {}) {
  let designMatrix = toMatrix(X);
  const n = designMatrix.rows;

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

  if (designMatrix.columns !== coefficients.length) {
    throw new Error(
      `Coefficient length (${coefficients.length}) must match design matrix columns (${designMatrix.columns})`,
    );
  }

  const coeffMatrix = Matrix.columnVector(coefficients);
  const predictions = designMatrix.mmul(coeffMatrix);

  return predictions.to1DArray();
}

/**
 * Summary statistics for regression
 * @param {Object} model - Fitted model from fit()
 * @returns {Object} Summary information
 */
export function summary(model) {
  const { coefficients, rSquared, adjRSquared, se, n, p } = model;

  return {
    coefficients,
    nObservations: n,
    nPredictors: p,
    rSquared,
    adjRSquared,
    residualStandardError: se,
    fStatistic: ((rSquared / (p - 1)) / ((1 - rSquared) / (n - p))),
  };
}

/**
 * Logistic Regression using Iteratively Reweighted Least Squares (IRLS)
 */

import { Matrix, toMatrix, solveLeastSquares } from "../core/linalg.js";
import { approxEqual, sum } from "../core/math.js";

/**
 * Logistic (sigmoid) function
 * @param {number} z - Input value
 * @returns {number} Probability between 0 and 1
 */
function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

/**
 * Fit logistic regression using IRLS
 * @param {Array<Array<number>>|Matrix} X - Design matrix
 * @param {Array<number>} y - Binary response (0 or 1)
 * @param {Object} options - {intercept: boolean, maxIter: number, tol: number}
 * @returns {Object} {coefficients, fitted, logLikelihood, iterations, converged}
 */
export function fit(
  X,
  y,
  { intercept = true, maxIter = 100, tol = 1e-8 } = {},
) {
  // Support declarative table input:
  // fit({ X: 'col' | ['col1','col2'], y: 'target', data: tableLike, omit_missing = true, intercept = true })
  if (
    X &&
    typeof X === "object" &&
    !Array.isArray(X) &&
    ("X" in X) &&
    ("y" in X) &&
    ("data" in X)
  ) {
    const opts = X;
    const data = opts.data;
    const colsX = typeof opts.X === "string"
      ? [opts.X]
      : Array.isArray(opts.X)
      ? opts.X
      : null;
    const colY = opts.y;
    const omit_missing = opts.omit_missing !== undefined
      ? opts.omit_missing
      : true;

    if (!colsX || typeof colY !== "string") {
      throw new Error(
        "When using declarative form, X must be string or array of strings and y must be a string",
      );
    }

    // Normalize data: accept array-of-objects or Arquero-like (has .objects())
    let rows;
    if (Array.isArray(data)) {
      rows = data;
    } else if (data && typeof data.objects === "function") {
      rows = data.objects();
    } else {
      throw new Error("Data must be array of objects or Arquero-like table");
    }

    // Validate columns presence
    if (rows.length > 0) {
      for (const c of [...colsX, colY]) {
        if (!(c in rows[0])) {
          throw new Error(`Column ${c} not found in data`);
        }
      }
    }

    // Optionally omit rows with missing / NaN values in any requested columns
    const filtered = omit_missing
      ? rows.filter((r) =>
        colsX.every((c) => r[c] != null && !Number.isNaN(r[c])) &&
        r[colY] != null && !Number.isNaN(r[colY])
      )
      : rows;

    // Build numeric X and y arrays
    const Xarr = filtered.map((r) =>
      colsX.map((c) => {
        const v = r[c];
        if (typeof v !== "number") {
          throw new Error(`Column ${c} contains non-numeric value: ${v}`);
        }
        return v;
      })
    );

    const yarr = filtered.map((r) => {
      const v = r[colY];
      if (typeof v !== "number") {
        throw new Error(`Column ${colY} contains non-numeric value: ${v}`);
      }
      return v;
    });

    X = Xarr;
    y = yarr;
    // allow intercept override from opts
    if (opts.intercept !== undefined) intercept = opts.intercept;
  }

  let designMatrix = toMatrix(X);
  const n = designMatrix.rows;
  const responseVector = Array.isArray(y) ? y : Array.from(y);

  if (n !== responseVector.length) {
    throw new Error("X and y must have same number of rows");
  }

  // Validate binary response
  for (const yi of responseVector) {
    if (yi !== 0 && yi !== 1) {
      throw new Error("Response must be binary (0 or 1)");
    }
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

  // Initialize coefficients
  let beta = new Array(p).fill(0);
  let converged = false;
  let iter = 0;

  for (iter = 0; iter < maxIter; iter++) {
    // Compute linear predictor and probabilities
    const eta = new Array(n);
    const mu = new Array(n);
    const weights = new Array(n);

    for (let i = 0; i < n; i++) {
      eta[i] = 0;
      for (let j = 0; j < p; j++) {
        eta[i] += designMatrix.get(i, j) * beta[j];
      }
      mu[i] = sigmoid(eta[i]);
      // Avoid division by zero
      mu[i] = Math.max(1e-10, Math.min(1 - 1e-10, mu[i]));
      weights[i] = mu[i] * (1 - mu[i]);
    }

    // Compute working response
    const z = new Array(n);
    for (let i = 0; i < n; i++) {
      z[i] = eta[i] + (responseVector[i] - mu[i]) / weights[i];
    }

    // Weighted least squares: X'WX β = X'Wz
    const W = Matrix.diag(weights);
    const XtW = designMatrix.transpose().mmul(W);
    const XtWX = XtW.mmul(designMatrix);
    const XtWz = XtW.mmul(Matrix.columnVector(z));

    let betaNew;
    try {
      const solution = solveLeastSquares(XtWX, XtWz);
      betaNew = solution.to1DArray();
    } catch (e) {
      converged = false;
      break;
    }

    // Check convergence
    let maxDiff = 0;
    for (let j = 0; j < p; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(betaNew[j] - beta[j]));
    }

    beta = betaNew;

    if (maxDiff < tol) {
      converged = true;
      break;
    }
  }

  // Compute final fitted values
  const fitted = new Array(n);
  for (let i = 0; i < n; i++) {
    let eta = 0;
    for (let j = 0; j < p; j++) {
      eta += designMatrix.get(i, j) * beta[j];
    }
    const prob = sigmoid(eta);
    fitted[i] = Math.max(1e-10, Math.min(1 - 1e-10, prob));
  }

  // Log-likelihood
  let logLikelihood = 0;
  for (let i = 0; i < n; i++) {
    const p = fitted[i];
    logLikelihood += responseVector[i] * Math.log(p) +
      (1 - responseVector[i]) * Math.log(1 - p);
  }

  return {
    coefficients: beta,
    fitted,
    logLikelihood,
    iterations: iter + 1,
    converged,
  };
}

/**
 * Predict probabilities using fitted model
 * @param {Array<number>} coefficients - Model coefficients
 * @param {Array<Array<number>>|Matrix} X - New design matrix
 * @param {Object} options - {intercept: boolean}
 * @returns {Array<number>} Predicted probabilities
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
    throw new Error(`Coefficient length must match design matrix columns`);
  }

  const predictions = new Array(n);
  for (let i = 0; i < n; i++) {
    let eta = 0;
    for (let j = 0; j < coefficients.length; j++) {
      eta += designMatrix.get(i, j) * coefficients[j];
    }
    predictions[i] = sigmoid(eta);
  }

  return predictions;
}

/**
 * Classify based on threshold
 * @param {Array<number>} probabilities - Predicted probabilities
 * @param {number} threshold - Classification threshold
 * @returns {Array<number>} Binary predictions (0 or 1)
 */
export function classify(probabilities, threshold = 0.5) {
  return probabilities.map((p) => (p >= threshold ? 1 : 0));
}

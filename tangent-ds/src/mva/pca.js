/**
 * Principal Component Analysis (PCA)
 * Dimensionality reduction via Singular Value Decomposition (SVD)
 * More numerically stable than eigendecomposition of covariance matrix
 */

import { svd, Matrix, toMatrix } from "../core/linalg.js";
import { mean, stddev } from "../core/math.js";
import { prepareX } from "../core/table.js";

/**
 * Standardize data (center and optionally scale)
 * @param {Array<Array<number>>} data - Data matrix
 * @param {boolean} scale - If true, scale to unit variance
 * @returns {Object} {standardized, means, sds}
 */
function standardize(data, scale = false) {
  const n = data.length;
  const p = data[0].length;

  // Compute means
  const means = [];
  for (let j = 0; j < p; j++) {
    const col = data.map((row) => row[j]);
    means.push(mean(col));
  }

  // Center data
  const centered = data.map((row) => row.map((val, j) => val - means[j]));

  if (!scale) {
    return { standardized: centered, means, sds: null };
  }

  // Compute standard deviations (using population std to match sklearn)
  const sds = [];
  for (let j = 0; j < p; j++) {
    const col = centered.map((row) => row[j]);
    sds.push(stddev(col, false));
  }

  // Scale data
  const scaled = centered.map((row) =>
    row.map((val, j) => sds[j] > 0 ? val / sds[j] : 0)
  );

  return { standardized: scaled, means, sds };
}

/**
 * Fit PCA model
 * @param {Array<Array<number>>|Matrix} X - Data matrix (n x p)
 * @param {Object} options - {scale: boolean, center: boolean}
 * @returns {Object} PCA model
 */
export function fit(
  X,
  {
    scale = false,
    center = true,
    columns = null,
    data: data_in = null,
    omit_missing = true,
  } = {},
) {
  // Support declarative style: fit({ data, columns, scale, center, omit_missing })
  if (
    X && typeof X === "object" && !Array.isArray(X) && (X.data || X.columns)
  ) {
    const opts = X;
    data_in = opts.data !== undefined ? opts.data : data_in;
    columns = opts.columns !== undefined ? opts.columns : columns;
    scale = opts.scale !== undefined ? opts.scale : scale;
    center = opts.center !== undefined ? opts.center : center;
    omit_missing = opts.omit_missing !== undefined
      ? opts.omit_missing
      : omit_missing;
  }

  // Convert to array format
  let data;
  if (data_in) {
    // Prepare numeric matrix from table-like input
    const prepared = prepareX({ columns, data: data_in, omit_missing });
    data = prepared.X;
  } else if (Array.isArray(X)) {
    data = X.map((row) => Array.isArray(row) ? row : [row]);
  } else {
    const mat = toMatrix(X);
    data = [];
    for (let i = 0; i < mat.rows; i++) {
      const row = [];
      for (let j = 0; j < mat.columns; j++) {
        row.push(mat.get(i, j));
      }
      data.push(row);
    }
  }

  const n = data.length;
  const p = data[0].length;

  if (n < 2) {
    throw new Error("Need at least 2 samples for PCA");
  }

  // Standardize if requested
  let processedData = data;
  let means = null;
  let sds = null;

  if (center) {
    const result = standardize(data, scale);
    processedData = result.standardized;
    means = result.means;
    sds = result.sds;
  }

  // Use SVD for better numerical stability
  // X = U * S * V^T, where X is n x p
  // The principal components are the columns of V
  // The eigenvalues are s^2 / (n-1)
  const mat = new Matrix(processedData);
  const { U, s, V } = svd(mat);

  // Compute eigenvalues from singular values
  // eigenvalue = s^2 / (n-1)
  const eigenvalues = s.map((sv) => (sv * sv) / (n - 1));

  // The columns of V are already the principal axes (eigenvectors)
  // They are already sorted by decreasing singular values
  const sortedEigenvalues = eigenvalues;

  // Extract eigenvectors from V (columns of V matrix)
  const sortedEigenvectors = [];
  for (let j = 0; j < p; j++) {
    const col = [];
    for (let i = 0; i < p; i++) {
      col.push(V.get(i, j));
    }
    sortedEigenvectors.push(col);
  }

  // Compute variance explained
  const totalVar = sortedEigenvalues.reduce((a, b) => a + b, 0);
  const varianceExplained = sortedEigenvalues.map((val) => val / totalVar);

  // Compute scores (projections)
  const scores = [];
  for (let i = 0; i < n; i++) {
    const score = {};
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) {
        sum += processedData[i][k] * sortedEigenvectors[j][k];
      }
      score[`pc${j + 1}`] = sum;
    }
    scores.push(score);
  }

  // Create loadings (eigenvectors as columns)
  const loadings = [];
  for (let i = 0; i < p; i++) {
    const loading = { variable: `var${i + 1}` };
    for (let j = 0; j < p; j++) {
      loading[`pc${j + 1}`] = sortedEigenvectors[j][i];
    }
    loadings.push(loading);
  }

  return {
    scores,
    loadings,
    eigenvalues: sortedEigenvalues,
    varianceExplained,
    means,
    sds,
    scale,
    center,
  };
}

/**
 * Transform new data using fitted PCA model
 * @param {Object} model - Fitted PCA model
 * @param {Array<Array<number>>} X - New data
 * @returns {Array<Object>} Transformed scores
 */
export function transform(model, X) {
  const { loadings, means, sds, scale, center } = model;

  let data = X.map((row) => Array.isArray(row) ? [...row] : [row]);
  const p = data[0].length;

  // Apply same standardization as training
  if (center && means) {
    data = data.map((row) => row.map((val, j) => val - means[j]));
  }

  if (scale && sds) {
    data = data.map((row) =>
      row.map((val, j) => sds[j] > 0 ? val / sds[j] : 0)
    );
  }

  // Extract loading matrix
  const nPCs = loadings[0] ? Object.keys(loadings[0]).length - 1 : 0;
  const loadingMatrix = [];
  for (let j = 0; j < nPCs; j++) {
    const col = loadings.map((l) => l[`pc${j + 1}`]);
    loadingMatrix.push(col);
  }

  // Compute scores
  const scores = [];
  for (const row of data) {
    const score = {};
    for (let j = 0; j < nPCs; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) {
        sum += row[k] * loadingMatrix[j][k];
      }
      score[`pc${j + 1}`] = sum;
    }
    scores.push(score);
  }

  return scores;
}

/**
 * Get cumulative variance explained
 * @param {Object} model - Fitted PCA model
 * @returns {Array<number>} Cumulative variance explained
 */
export function cumulativeVariance(model) {
  const { varianceExplained } = model;
  const cumulative = [];
  let sum = 0;
  for (const ve of varianceExplained) {
    sum += ve;
    cumulative.push(sum);
  }
  return cumulative;
}

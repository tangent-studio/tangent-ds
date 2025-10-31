/**
 * K-means clustering algorithm
 * Uses k-means++ initialization for better convergence
 */

import { toMatrix } from "../core/linalg.js";
import { mean } from "../core/math.js";
import { prepareX } from "../core/table.js";

/**
 * Euclidean distance between two points
 * @param {Array<number>} a - First point
 * @param {Array<number>} b - Second point
 * @returns {number} Distance
 */
function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

/**
 * Initialize centroids using k-means++ algorithm
 * @param {Array<Array<number>>} data - Data points
 * @param {number} k - Number of clusters
 * @returns {Array<Array<number>>} Initial centroids
 */
function createRandomGenerator(seed) {
  if (seed === null || seed === undefined) {
    return Math.random;
  }

  let state = seed >>> 0;
  return () => {
    // Linear congruential generator
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * Initialize centroids using k-means++ algorithm
 * @param {Array<Array<number>>} data - Data points
 * @param {number} k - Number of clusters
 * @param {Function} random - Random number generator returning [0,1)
 * @returns {Array<Array<number>>} Initial centroids
 */
function initializeCentroidsKMeansPlusPlus(data, k, random) {
  const n = data.length;
  const centroids = [];

  // Choose first centroid randomly
  const firstIdx = Math.floor(random() * n);
  centroids.push([...data[firstIdx]]);

  // Choose remaining centroids
  for (let c = 1; c < k; c++) {
    const distances = new Array(n);

    // Compute distance to nearest centroid
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (const centroid of centroids) {
        const dist = euclideanDistance(data[i], centroid);
        minDist = Math.min(minDist, dist);
      }
      distances[i] = minDist ** 2;
    }

    // Choose next centroid with probability proportional to distance^2
    const totalDist = distances.reduce((a, b) => a + b, 0);
    let threshold = random() * totalDist;

    for (let i = 0; i < n; i++) {
      threshold -= distances[i];
      if (threshold <= 0) {
        centroids.push([...data[i]]);
        break;
      }
    }
  }

  return centroids;
}

/**
 * Assign each point to nearest centroid
 * @param {Array<Array<number>>} data - Data points
 * @param {Array<Array<number>>} centroids - Current centroids
 * @returns {Array<number>} Cluster assignments
 */
function assignClusters(data, centroids) {
  return data.map((point) => {
    let minDist = Infinity;
    let cluster = 0;

    centroids.forEach((centroid, idx) => {
      const dist = euclideanDistance(point, centroid);
      if (dist < minDist) {
        minDist = dist;
        cluster = idx;
      }
    });

    return cluster;
  });
}

/**
 * Update centroids based on cluster assignments
 * @param {Array<Array<number>>} data - Data points
 * @param {Array<number>} labels - Cluster assignments
 * @param {number} k - Number of clusters
 * @returns {Array<Array<number>>} New centroids
 */
function updateCentroids(data, labels, k) {
  const dim = data[0].length;
  const centroids = Array(k).fill(null).map(() => new Array(dim).fill(0));
  const counts = new Array(k).fill(0);

  // Sum points in each cluster
  data.forEach((point, idx) => {
    const cluster = labels[idx];
    counts[cluster]++;
    point.forEach((val, d) => {
      centroids[cluster][d] += val;
    });
  });

  // Compute means
  centroids.forEach((centroid, c) => {
    if (counts[c] > 0) {
      centroid.forEach((val, d) => {
        centroid[d] = val / counts[c];
      });
    }
  });

  return centroids;
}

/**
 * Compute within-cluster sum of squares (inertia)
 * @param {Array<Array<number>>} data - Data points
 * @param {Array<number>} labels - Cluster assignments
 * @param {Array<Array<number>>} centroids - Centroids
 * @returns {number} Inertia
 */
function computeInertia(data, labels, centroids) {
  let inertia = 0;
  data.forEach((point, idx) => {
    const centroid = centroids[labels[idx]];
    inertia += euclideanDistance(point, centroid) ** 2;
  });
  return inertia;
}

/**
 * Fit k-means clustering model
 * @param {Array<Array<number>>|Matrix} X - Data matrix (n samples × d features)
 * @param {Object} options - {k: number of clusters, maxIter: max iterations, tol: tolerance}
 * @returns {Object} {labels, centroids, inertia, iterations, converged}
 */
export function fit(
  X,
  {
    k = 3,
    maxIter = 300,
    tol = 1e-4,
    seed = null,
    columns = null,
    data: data_in = null,
  } = {},
) {
  // Accept either:
  //  - legacy numeric input: fit(X_array_or_matrix, { k, ... })
  //  - declarative options-object as first arg: fit({ data, columns, k, ... })
  //
  // If first arg is an options-object (contains `data` or `columns`), treat it as such.
  let data;
  if (
    X && typeof X === "object" && !Array.isArray(X) && (X.data || X.columns)
  ) {
    const opts = X;
    data_in = opts.data !== undefined ? opts.data : data_in;
    columns = opts.columns !== undefined ? opts.columns : columns;
    k = opts.k !== undefined ? opts.k : k;
    maxIter = opts.maxIter !== undefined ? opts.maxIter : maxIter;
    tol = opts.tol !== undefined ? opts.tol : tol;
    seed = opts.seed !== undefined ? opts.seed : seed;
  }

  // If declarative data provided, prepare numeric matrix via prepareX
  if (data_in) {
    const prepared = prepareX({ columns, data: data_in });
    // prepared.X is array of arrays (rows)
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

  if (data.length < k) {
    throw new Error(
      `Cannot fit ${k} clusters with only ${data.length} samples`,
    );
  }

  // Initialize centroids
  const random = createRandomGenerator(seed);
  let centroids = initializeCentroidsKMeansPlusPlus(data, k, random);
  let labels = new Array(data.length).fill(0);
  let prevInertia = Infinity;
  let converged = false;
  let iter = 0;

  for (iter = 0; iter < maxIter; iter++) {
    // Assignment step
    labels = assignClusters(data, centroids);

    // Update step
    centroids = updateCentroids(data, labels, k);

    // Check convergence
    const inertia = computeInertia(data, labels, centroids);

    if (Math.abs(prevInertia - inertia) < tol) {
      converged = true;
      break;
    }

    prevInertia = inertia;
  }

  const finalInertia = computeInertia(data, labels, centroids);

  return {
    labels,
    centroids,
    inertia: finalInertia,
    iterations: iter + 1,
    converged,
  };
}

/**
 * Predict cluster labels for new data
 * @param {Object} model - Fitted model from fit()
 * @param {Array<Array<number>>} X - New data points
 * @returns {Array<number>} Cluster labels
 */
export function predict(model, X) {
  const { centroids } = model;
  const data = Array.isArray(X[0]) ? X : X.map((x) => [x]);
  return assignClusters(data, centroids);
}

/**
 * Compute silhouette score for clustering quality
 * @param {Array<Array<number>>} X - Data points
 * @param {Array<number>} labels - Cluster assignments
 * @returns {number} Silhouette score (range: -1 to 1, higher is better)
 */
export function silhouetteScore(X, labels) {
  const n = X.length;
  if (n === 0) return 0;

  const k = Math.max(...labels) + 1;
  if (k === 1) return 0;

  const scores = new Array(n);

  for (let i = 0; i < n; i++) {
    const clusterI = labels[i];

    // a(i): mean distance to points in same cluster
    let sumA = 0;
    let countA = 0;

    // b(i): mean distance to points in nearest other cluster
    const sumB = new Array(k).fill(0);
    const countB = new Array(k).fill(0);

    for (let j = 0; j < n; j++) {
      if (i === j) continue;

      const dist = euclideanDistance(X[i], X[j]);
      const clusterJ = labels[j];

      if (clusterJ === clusterI) {
        sumA += dist;
        countA++;
      } else {
        sumB[clusterJ] += dist;
        countB[clusterJ]++;
      }
    }

    const a = countA > 0 ? sumA / countA : 0;

    // Find nearest cluster
    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c !== clusterI && countB[c] > 0) {
        b = Math.min(b, sumB[c] / countB[c]);
      }
    }

    if (b === Infinity) b = 0;

    scores[i] = (b - a) / Math.max(a, b);
  }

  return mean(scores);
}

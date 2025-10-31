/**
 * Hierarchical Clustering Analysis (HCA)
 * Agglomerative clustering with various linkage methods
 */

import { mean } from "../core/math.js";
import { prepareX } from "../core/table.js";

/**
 * Compute Euclidean distance between two points
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
 * Compute pairwise distance matrix
 * @param {Array<Array<number>>} data - Data points
 * @returns {Array<Array<number>>} Distance matrix
 */
function computeDistanceMatrix(data) {
  const n = data.length;
  const distances = Array(n).fill(null).map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dist = euclideanDistance(data[i], data[j]);
      distances[i][j] = dist;
      distances[j][i] = dist;
    }
  }

  return distances;
}

/**
 * Find minimum distance between two clusters
 * @param {Array<number>} cluster1 - Indices in cluster 1
 * @param {Array<number>} cluster2 - Indices in cluster 2
 * @param {Array<Array<number>>} distances - Distance matrix
 * @param {string} linkage - Linkage method
 * @returns {number} Distance between clusters
 */
function clusterDistance(cluster1, cluster2, distances, linkage) {
  const dists = [];

  for (const i of cluster1) {
    for (const j of cluster2) {
      dists.push(distances[i][j]);
    }
  }

  if (linkage === "single") {
    return Math.min(...dists);
  } else if (linkage === "complete") {
    return Math.max(...dists);
  } else if (linkage === "average") {
    return mean(dists);
  }

  throw new Error(`Unknown linkage: ${linkage}`);
}

/**
 * Fit hierarchical clustering
 * @param {Array<Array<number>>} X - Data matrix
 * @param {Object} options - {linkage: 'single'|'complete'|'average'}
 * @returns {Object} {dendrogram, distances}
 */
export function fit(X, { linkage = "average" } = {}) {
  // Support declarative usage:
  // - fit({ data: tableLike, columns: ['col1','col2'], linkage: 'average', omit_missing: true })
  // - fit(arrayOfRows)
  let data;

  // Declarative object form (first argument is an options object containing data or columns)
  if (
    X && typeof X === "object" && !Array.isArray(X) && (X.data || X.columns)
  ) {
    const opts = X;
    const data_in = opts.data;
    const columns = opts.columns;
    const omit_missing = opts.omit_missing !== undefined
      ? opts.omit_missing
      : true;

    if (!data_in) {
      throw new Error(
        "Declarative call requires a `data` property (array-of-objects or Arquero-like table)",
      );
    }

    const prepared = prepareX({ columns, data: data_in, omit_missing });
    data = prepared.X;

    // allow overriding linkage from options object
    linkage = opts.linkage !== undefined ? opts.linkage : linkage;
  } else if (Array.isArray(X)) {
    // Legacy array-of-rows form
    data = X.map((row) => Array.isArray(row) ? row : [row]);
  } else {
    throw new Error(
      "X must be an array of rows or an options object containing `data`/`columns`",
    );
  }

  const n = data.length;

  if (n < 2) {
    throw new Error("Need at least 2 samples for clustering");
  }

  // Compute distance matrix
  const distances = computeDistanceMatrix(data);

  // Initialize clusters (each point is a cluster)
  const clusters = data.map((_, i) => [i]);
  const merges = [];

  // Agglomerative clustering
  while (clusters.length > 1) {
    // Find closest pair of clusters
    let minDist = Infinity;
    let minI = 0;
    let minJ = 1;

    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const dist = clusterDistance(
          clusters[i],
          clusters[j],
          distances,
          linkage,
        );
        if (dist < minDist) {
          minDist = dist;
          minI = i;
          minJ = j;
        }
      }
    }

    // Merge clusters
    const merged = [...clusters[minI], ...clusters[minJ]];

    merges.push({
      cluster1: clusters[minI],
      cluster2: clusters[minJ],
      distance: minDist,
      size: merged.length,
    });

    // Remove old clusters and add merged cluster
    const newClusters = [];
    for (let i = 0; i < clusters.length; i++) {
      if (i !== minI && i !== minJ) {
        newClusters.push(clusters[i]);
      }
    }
    newClusters.push(merged);
    clusters.splice(0, clusters.length, ...newClusters);
  }

  return {
    dendrogram: merges,
    linkage,
    n,
  };
}

/**
 * Cut dendrogram to get k clusters
 * @param {Object} model - Fitted HCA model
 * @param {number} k - Number of clusters
 * @returns {Array<number>} Cluster assignments
 */
export function cut(model, k) {
  const { dendrogram, n } = model;

  if (k < 1 || k > n) {
    throw new Error(`k must be between 1 and ${n}`);
  }

  if (k === n) {
    // Each point is its own cluster
    return Array.from({ length: n }, (_, i) => i);
  }

  // Start with individual clusters
  const clusters = Array.from({ length: n }, (_, i) => [i]);

  // Apply merges until we have k clusters
  for (let i = 0; i < dendrogram.length && clusters.length > k; i++) {
    const merge = dendrogram[i];

    // Find which clusters to merge
    let idx1 = -1;
    let idx2 = -1;

    for (let j = 0; j < clusters.length; j++) {
      if (clusters[j].some((x) => merge.cluster1.includes(x))) {
        idx1 = j;
      }
      if (clusters[j].some((x) => merge.cluster2.includes(x))) {
        idx2 = j;
      }
    }

    if (idx1 !== -1 && idx2 !== -1 && idx1 !== idx2) {
      // Merge
      const merged = [...clusters[idx1], ...clusters[idx2]];
      const newClusters = [];
      for (let j = 0; j < clusters.length; j++) {
        if (j !== idx1 && j !== idx2) {
          newClusters.push(clusters[j]);
        }
      }
      newClusters.push(merged);
      clusters.splice(0, clusters.length, ...newClusters);
    }
  }

  // Create labels array
  const labels = new Array(n);
  clusters.forEach((cluster, clusterIdx) => {
    cluster.forEach((pointIdx) => {
      labels[pointIdx] = clusterIdx;
    });
  });

  return labels;
}

/**
 * Cut dendrogram at specific height
 * @param {Object} model - Fitted HCA model
 * @param {number} height - Height to cut at
 * @returns {Array<number>} Cluster assignments
 */
export function cutHeight(model, height) {
  const { dendrogram, n } = model;

  // Start with individual clusters
  const clusters = Array.from({ length: n }, (_, i) => [i]);

  // Apply merges until height threshold
  for (const merge of dendrogram) {
    if (merge.distance > height) {
      break;
    }

    // Find which clusters to merge
    let idx1 = -1;
    let idx2 = -1;

    for (let j = 0; j < clusters.length; j++) {
      if (clusters[j].some((x) => merge.cluster1.includes(x))) {
        idx1 = j;
      }
      if (clusters[j].some((x) => merge.cluster2.includes(x))) {
        idx2 = j;
      }
    }

    if (idx1 !== -1 && idx2 !== -1 && idx1 !== idx2) {
      // Merge
      const merged = [...clusters[idx1], ...clusters[idx2]];
      const newClusters = [];
      for (let j = 0; j < clusters.length; j++) {
        if (j !== idx1 && j !== idx2) {
          newClusters.push(clusters[j]);
        }
      }
      newClusters.push(merged);
      clusters.splice(0, clusters.length, ...newClusters);
    }
  }

  // Create labels array
  const labels = new Array(n);
  clusters.forEach((cluster, clusterIdx) => {
    cluster.forEach((pointIdx) => {
      labels[pointIdx] = clusterIdx;
    });
  });

  return labels;
}

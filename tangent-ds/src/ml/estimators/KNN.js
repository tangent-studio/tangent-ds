/**
 * K-Nearest Neighbors estimators (classifier & regressor).
 *
 * Provides scikit-learn style interfaces with support for both numeric-array
 * inputs and declarative table usage through prepareX/prepareXY helpers.
 */

import { Classifier, Regressor } from '../../core/estimators/estimator.js';
import { prepareXY, prepareX } from '../../core/table.js';
import { mean } from '../../core/math.js';

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function buildDataset(X, y, opts = {}) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareXY({
      X: X.X || X.columns,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
    });
    return {
      X: prepared.X,
      y: prepared.y,
      columns: prepared.columnsX
    };
  }

  if (!Array.isArray(X) || !Array.isArray(y)) {
    throw new Error('KNN fit expects arrays for X and y (or a table object).');
  }

  return { X, y, columns: null };
}

function preparePredictInput(X, storedColumns, opts = {}) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareX({
      columns: X.X || X.columns || storedColumns,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
    });
    return prepared.X;
  }

  return Array.isArray(X[0]) ? X : X.map((row) => [row]);
}

function getNeighbors(trainX, trainY, point, k, metric) {
  const distances = [];
  for (let i = 0; i < trainX.length; i++) {
    const dist = metric(trainX[i], point);
    distances.push({ dist, label: trainY[i] });
  }

  distances.sort((a, b) => a.dist - b.dist);
  return distances.slice(0, k);
}

class KNNBase {
  constructor({ k = 5, weight = 'uniform', metric = euclideanDistance } = {}) {
    if (k <= 0) throw new Error('k must be positive');
    this.k = k;
    this.weight = weight;
    this.metric = metric;
    this.X = null;
    this.y = null;
    this.columns = null;
  }

  _fitBase(X, y) {
    const prepared = buildDataset(X, y);
    this.X = prepared.X.map((row) =>
      Array.isArray(row) ? row.map(Number) : [Number(row)]
    );
    this.y = prepared.y;
    this.columns = prepared.columns;
    this.fitted = true;
  }

  _preparePredict(X) {
    if (!this.fitted) {
      throw new Error('KNN estimator not fitted.');
    }
    return preparePredictInput(X, this.columns);
  }

  _neighborWeights(neighbors) {
    if (this.weight === 'distance') {
      return neighbors.map(({ dist, label }) => ({
        label,
        weight: dist === 0 ? 1e9 : 1 / dist
      }));
    }
    return neighbors.map(({ label }) => ({ label, weight: 1 }));
  }
}

export class KNNClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.knn = new KNNBase(opts);
  }

  fit(X, y = null) {
    this.knn._fitBase(X, y);
    return this;
  }

  predict(X) {
    const data = this.knn._preparePredict(X);
    const predictions = [];
    const { X: trainX, y: trainY } = this.knn;

    for (const point of data) {
      const neighbors = getNeighbors(trainX, trainY, point, this.knn.k, this.knn.metric);
      const weighted = this.knn._neighborWeights(neighbors);
      const votes = new Map();
      for (const { label, weight } of weighted) {
        votes.set(label, (votes.get(label) || 0) + weight);
      }
      let bestLabel = null;
      let bestScore = -Infinity;
      for (const [label, score] of votes.entries()) {
        if (score > bestScore) {
          bestScore = score;
          bestLabel = label;
        }
      }
      predictions.push(bestLabel);
    }
    return predictions;
  }

  predictProba(X) {
    const data = this.knn._preparePredict(X);
    const result = [];
    const { X: trainX, y: trainY } = this.knn;
    const labels = Array.from(new Set(trainY));

    for (const point of data) {
      const neighbors = getNeighbors(trainX, trainY, point, this.knn.k, this.knn.metric);
      const weighted = this.knn._neighborWeights(neighbors);
      const votes = new Map();
      let total = 0;
      for (const { label, weight } of weighted) {
        votes.set(label, (votes.get(label) || 0) + weight);
        total += weight;
      }
      const proba = {};
      labels.forEach((label) => {
        proba[label] = total === 0 ? 0 : (votes.get(label) || 0) / total;
      });
      result.push(proba);
    }
    return result;
  }
}

export class KNNRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.knn = new KNNBase(opts);
  }

  fit(X, y = null) {
    this.knn._fitBase(X, y);
    return this;
  }

  predict(X) {
    const data = this.knn._preparePredict(X);
    const predictions = [];
    const { X: trainX, y: trainY } = this.knn;

    for (const point of data) {
      const neighbors = getNeighbors(trainX, trainY, point, this.knn.k, this.knn.metric);
      const weighted = this.knn._neighborWeights(neighbors);
      const totalWeight = weighted.reduce((acc, w) => acc + w.weight, 0);

      if (totalWeight === 0) {
        predictions.push(mean(neighbors.map((n) => Number(n.label))));
        continue;
      }

      let sum = 0;
      for (const { label, weight } of weighted) {
        sum += Number(label) * weight;
      }
      predictions.push(sum / totalWeight);
    }

    return predictions;
  }
}

export default {
  KNNClassifier,
  KNNRegressor
};

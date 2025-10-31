/**
 * Decision Tree estimators (classification & regression) using CART-style splits.
 */

import { Classifier, Regressor } from '../../core/estimators/estimator.js';
import { prepareXY, prepareX } from '../../core/table.js';

function toNumericMatrix(X) {
  return X.map((row) => Array.isArray(row) ? row.map(Number) : [Number(row)]);
}

function toArray(y) {
  return Array.isArray(y) ? y.slice() : Array.from(y);
}

function variance(values) {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  let sum = 0;
  for (const v of values) {
    const diff = v - mean;
    sum += diff * diff;
  }
  return sum / values.length;
}

function gini(labels) {
  if (labels.length === 0) return 0;
  const counts = new Map();
  labels.forEach((label) => {
    counts.set(label, (counts.get(label) || 0) + 1);
  });
  const total = labels.length;
  let sum = 0;
  for (const count of counts.values()) {
    const p = count / total;
    sum += p * p;
  }
  return 1 - sum;
}

function majorityVote(labels) {
  const counts = new Map();
  labels.forEach((label) => {
    counts.set(label, (counts.get(label) || 0) + 1);
  });
  let bestLabel = null;
  let bestCount = -Infinity;
  for (const [label, count] of counts.entries()) {
    if (count > bestCount) {
      bestCount = count;
      bestLabel = label;
    }
  }
  return bestLabel;
}

function meanValue(values) {
  if (values.length === 0) return 0;
  const sum = values.reduce((a, b) => a + b, 0);
  return sum / values.length;
}

function buildPreparedData(X, y) {
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
      X: toNumericMatrix(prepared.X),
      y: prepared.y,
      columns: prepared.columnsX
    };
  }

  return {
    X: toNumericMatrix(X),
    y: toArray(y),
    columns: null
  };
}

function preparePredictInput(X, storedColumns) {
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
    return toNumericMatrix(prepared.X);
  }
  return toNumericMatrix(X);
}

function featureSubset(allFeatures, maxFeatures, random) {
  if (!maxFeatures || maxFeatures >= allFeatures.length) {
    return allFeatures;
  }
  const features = allFeatures.slice();
  for (let i = features.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [features[i], features[j]] = [features[j], features[i]];
  }
  return features.slice(0, maxFeatures);
}

class DecisionTreeBase {
  constructor({
    maxDepth = 10,
    minSamplesSplit = 2,
    minGain = 1e-7,
    task = 'classification',
    maxFeatures = null,
    random = Math.random
  } = {}) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.minGain = minGain;
    this.task = task;
    this.maxFeatures = maxFeatures;
    this.random = random;

    this.root = null;
    this.columns = null;
  }

  fit(X, y) {
    const prepared = buildPreparedData(X, y);
    this.columns = prepared.columns;

    const features = Array.from({ length: prepared.X[0].length }, (_, i) => i);
    this.root = this._buildTree(prepared.X, prepared.y, 0, features);
    this.fitted = true;
    this.trainX = prepared.X;
    this.trainY = prepared.y;
  }

  _buildTree(X, y, depth, allFeatures) {
    if (
      depth >= this.maxDepth ||
      X.length < this.minSamplesSplit ||
      new Set(y).size === 1
    ) {
      return this._createLeaf(y);
    }

    const subset = featureSubset(allFeatures, this.maxFeatures, this.random);
    const { feature, threshold, gain } = this._bestSplit(X, y, subset);

    if (gain < this.minGain || feature === null) {
      return this._createLeaf(y);
    }

    const { leftX, leftY, rightX, rightY } = this._splitDataset(X, y, feature, threshold);

    if (leftX.length === 0 || rightX.length === 0) {
      return this._createLeaf(y);
    }

    return {
      type: 'internal',
      feature,
      threshold,
      left: this._buildTree(leftX, leftY, depth + 1, allFeatures),
      right: this._buildTree(rightX, rightY, depth + 1, allFeatures)
    };
  }

  _splitDataset(X, y, feature, threshold) {
    const leftX = [];
    const leftY = [];
    const rightX = [];
    const rightY = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return { leftX, leftY, rightX, rightY };
  }

  _bestSplit(X, y, features) {
    const p = X[0].length;
    let bestFeature = null;
    let bestThreshold = null;
    let bestGain = -Infinity;

    const impurityFunc = this.task === 'classification' ? gini : variance;
    const parentImpurity = impurityFunc(y);

    for (const feature of features) {
      const values = X.map((row) => row[feature]);
      const uniqueValues = Array.from(new Set(values)).sort((a, b) => a - b);
      if (uniqueValues.length <= 1) continue;

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        const { leftY, rightY } = this._splitDataset(X, y, feature, threshold);

        if (leftY.length === 0 || rightY.length === 0) continue;

        const leftImpurity = impurityFunc(leftY);
        const rightImpurity = impurityFunc(rightY);
        const gain = parentImpurity - (
          (leftY.length / y.length) * leftImpurity +
          (rightY.length / y.length) * rightImpurity
        );

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = feature;
          bestThreshold = threshold;
        }
      }
    }

    return { feature: bestFeature, threshold: bestThreshold, gain: bestGain };
  }

  _createLeaf(y) {
    if (this.task === 'classification') {
      const counts = new Map();
      y.forEach((label) => counts.set(label, (counts.get(label) || 0) + 1));
      const total = y.length;
      const distribution = {};
      counts.forEach((count, label) => {
        distribution[label] = count / total;
      });
      return { type: 'leaf', value: majorityVote(y), distribution };
    }
    return { type: 'leaf', value: meanValue(y) };
  }

  predict(X) {
    if (!this.fitted) {
      throw new Error('DecisionTree estimator not fitted.');
    }
    const data = preparePredictInput(X, this.columns);
    return data.map((row) => this._predictRow(row, this.root));
  }

  _predictRow(row, node) {
    if (node.type === 'leaf') {
      return node.value;
    }
    if (row[node.feature] <= node.threshold) {
      return this._predictRow(row, node.left);
    }
    return this._predictRow(row, node.right);
  }
}

export class DecisionTreeClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.tree = new DecisionTreeBase({ ...opts, task: 'classification' });
  }

  fit(X, y = null) {
    this.tree.fit(X, y);
    this.fitted = true;
    return this;
  }

  predict(X) {
    return this.tree.predict(X);
  }

  predictProba(X) {
    if (!this.fitted) {
      throw new Error('DecisionTree: estimator not fitted.');
    }
    const data = preparePredictInput(X, this.tree.columns);
    return data.map((row) => {
      let node = this.tree.root;
      while (node.type !== 'leaf') {
        if (row[node.feature] <= node.threshold) {
          node = node.left;
        } else {
          node = node.right;
        }
      }
      return node.distribution;
    });
  }
}

export class DecisionTreeRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.tree = new DecisionTreeBase({ ...opts, task: 'regression' });
  }

  fit(X, y = null) {
    this.tree.fit(X, y);
    this.fitted = true;
    return this;
  }

  predict(X) {
    return this.tree.predict(X);
  }
}

export default {
  DecisionTreeClassifier,
  DecisionTreeRegressor
};

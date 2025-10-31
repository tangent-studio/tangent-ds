/**
 * Random Forest estimators built on top of Decision Trees.
 */

import { Classifier, Regressor } from '../../core/estimators/estimator.js';
import { prepareXY, prepareX } from '../../core/table.js';
import { DecisionTreeClassifier, DecisionTreeRegressor } from './DecisionTree.js';

function toNumericMatrix(X) {
  return X.map((row) => Array.isArray(row) ? row.map(Number) : [Number(row)]);
}

function bootstrapSample(X, y, random) {
  const n = X.length;
  const XSample = [];
  const ySample = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(random() * n);
    XSample.push(X[idx]);
    ySample.push(y[idx]);
  }
  return { XSample, ySample };
}

function createRandomGenerator(seed) {
  if (seed === null || seed === undefined) {
    return Math.random;
  }
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function prepareDataset(X, y) {
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
    y: Array.isArray(y) ? y.slice() : Array.from(y),
    columns: null
  };
}

function preparePredict(X, columns) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.columns)
  ) {
    const prepared = prepareX({
      columns: X.X || X.columns || columns,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true
    });
    return toNumericMatrix(prepared.X);
  }
  return toNumericMatrix(X);
}

class RandomForestBase {
  constructor({
    nEstimators = 50,
    maxDepth = 10,
    minSamplesSplit = 2,
    maxFeatures = null,
    task = 'classification',
    seed = null
  } = {}) {
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.maxFeatures = maxFeatures;
    this.task = task;
    this.seed = seed;
    this.trees = [];
    this.columns = null;
    this.random = createRandomGenerator(seed);
  }

  fit(X, y) {
    const prepared = prepareDataset(X, y);
    this.columns = prepared.columns;

    const featureCount = prepared.X[0].length;
    const defaultMaxFeatures = this.task === 'classification'
      ? Math.max(1, Math.floor(Math.sqrt(featureCount)))
      : Math.max(1, Math.floor(featureCount / 3));
    const featureBagSize = this.maxFeatures || defaultMaxFeatures;
    this.classes = this.task === 'classification' ? Array.from(new Set(prepared.y)) : null;

    this.trees = [];

    for (let i = 0; i < this.nEstimators; i++) {
      const { XSample, ySample } = bootstrapSample(prepared.X, prepared.y, this.random);

      const treeOpts = {
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        maxFeatures: featureBagSize,
        random: this.random
      };

      const tree = this.task === 'classification'
        ? new DecisionTreeClassifier(treeOpts)
        : new DecisionTreeRegressor(treeOpts);

      tree.fit(XSample, ySample);
      this.trees.push(tree);
    }
    this.fitted = true;
  }

  _predictRaw(X) {
    const data = preparePredict(X, this.columns);
    const predictions = [];

    for (const row of data) {
      if (this.task === 'classification') {
        const votes = new Map();
        this.trees.forEach((tree) => {
          const pred = tree.predict([row])[0];
          votes.set(pred, (votes.get(pred) || 0) + 1);
        });
        let bestLabel = null;
        let bestCount = -Infinity;
        for (const [label, count] of votes.entries()) {
          if (count > bestCount) {
            bestCount = count;
            bestLabel = label;
          }
        }
        predictions.push(bestLabel);
      } else {
        let sum = 0;
        this.trees.forEach((tree) => {
          sum += tree.predict([row])[0];
        });
        predictions.push(sum / this.trees.length);
      }
    }

    return predictions;
  }
}

export class RandomForestClassifier extends Classifier {
  constructor(opts = {}) {
    super(opts);
    this.forest = new RandomForestBase({ ...opts, task: 'classification' });
  }

  fit(X, y = null) {
    this.forest.fit(X, y);
    this.fitted = true;
    return this;
  }

  predict(X) {
    return this.forest._predictRaw(X);
  }

  predictProba(X) {
    if (!this.fitted) throw new Error('RandomForest: estimator not fitted.');
    const data = preparePredict(X, this.forest.columns);
    const proba = [];
    const labels = this.forest.classes;

    for (const row of data) {
      const votes = new Map();
      this.forest.trees.forEach((tree) => {
        const leafProba = tree.predictProba([row])[0];
        Object.keys(leafProba).forEach((label) => {
          votes.set(label, (votes.get(label) || 0) + leafProba[label]);
        });
      });
      const total = this.forest.trees.length;
      const dist = {};
      labels.forEach((label) => {
        dist[label] = (votes.get(label) || 0) / total;
      });
      proba.push(dist);
    }
    return proba;
  }
}

export class RandomForestRegressor extends Regressor {
  constructor(opts = {}) {
    super(opts);
    this.forest = new RandomForestBase({ ...opts, task: 'regression' });
  }

  fit(X, y = null) {
    this.forest.fit(X, y);
    this.fitted = true;
    return this;
  }

  predict(X) {
    return this.forest._predictRaw(X);
  }
}

export default {
  RandomForestClassifier,
  RandomForestRegressor
};

/**
 * MLPRegressor - class wrapper around ml/mlp utilities.
 */

import { Regressor } from '../../core/estimators/estimator.js';
import { prepareXY, prepareX } from '../../core/table.js';
import * as mlp from '../mlp.js';

const DEFAULT_PARAMS = {
  layerSizes: null,
  activation: 'relu',
  learningRate: 0.01,
  epochs: 100,
  batchSize: 32,
  verbose: false,
  omit_missing: true
};

export class MLPRegressor extends Regressor {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
  }

  fit(X, y = null, opts = {}) {
    let dataX = X;
    let dataY = y;
    const merged = { ...this.params, ...opts };

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.X || X.columns)
    ) {
      const callOpts = { ...DEFAULT_PARAMS, ...this.params, ...X };
      const prepared = prepareXY({
        X: callOpts.X || callOpts.columns,
        y: callOpts.y,
        data: callOpts.data,
        omit_missing: callOpts.omit_missing
      });
      dataX = prepared.X;
      dataY = prepared.y;
      Object.assign(merged, {
        layerSizes: callOpts.layerSizes,
        activation: callOpts.activation,
        learningRate: callOpts.learningRate,
        epochs: callOpts.epochs,
        batchSize: callOpts.batchSize,
        verbose: callOpts.verbose
      });
    }

    if (!dataX || !dataY) {
      throw new Error('MLPRegressor.fit requires X and y.');
    }

    const result = mlp.fit(dataX, dataY, {
      layerSizes: merged.layerSizes,
      activation: merged.activation,
      learningRate: merged.learningRate,
      epochs: merged.epochs,
      batchSize: merged.batchSize,
      verbose: merged.verbose
    });

    this.model = result;
    this.fitted = true;
    this.params = {
      ...this.params,
      layerSizes: result.layerSizes,
      activation: merged.activation,
      learningRate: merged.learningRate,
      epochs: merged.epochs,
      batchSize: merged.batchSize,
      verbose: merged.verbose
    };
    return this;
  }

  predict(X) {
    if (!this.fitted || !this.model) {
      throw new Error('MLPRegressor: estimator not fitted.');
    }

    let matrix = X;
    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.X || X.columns)
    ) {
      const prepared = prepareX({
        columns: X.X || X.columns,
        data: X.data,
        omit_missing: X.omit_missing !== undefined
          ? X.omit_missing
          : this.params.omit_missing
      });
      matrix = prepared.X;
    }

    return mlp.predict(this.model, matrix);
  }

  evaluate(X, y) {
    if (!this.fitted || !this.model) {
      throw new Error('MLPRegressor: estimator not fitted.');
    }
    return mlp.evaluate(this.model, X, y);
  }

  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('MLPRegressor: estimator not fitted.');
    }
    const { losses, epochs, layerSizes } = this.model;
    return {
      epochs,
      layerSizes,
      finalLoss: losses[losses.length - 1],
      initialLoss: losses[0],
      losses
    };
  }

  toJSON() {
    return {
      __class__: 'MLPRegressor',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  static fromJSON(obj = {}) {
    const inst = new MLPRegressor(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default MLPRegressor;

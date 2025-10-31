/**
 * lm - Ordinary Least Squares Linear Regression estimator
 *
 * Provides a scikit-learn style API atop the functional helpers in src/stats/lm.js.
 * Supports both numeric array inputs and declarative table-style inputs:
 *
 *   const model = new lm({ intercept: true });
 *   model.fit({ X: ['feat1', 'feat2'], y: 'target', data: table });
 *   const preds = model.predict({ X: ['feat1', 'feat2'], data: table });
 *
 * Static helpers (lm.fit / lm.predict / lm.summary) remain available by
 * attaching the original functional exports to the class.
 */

import { Regressor } from '../../core/estimators/estimator.js';
import {
  fit as lmFit,
  predict as lmPredict,
  summary as lmSummary
} from '../lm.js';

const DEFAULT_PARAMS = {
  intercept: true
};

export class lm extends Regressor {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;
    this.model = null;
    this.coef = null;
  }

  /**
   * Fit the linear model.
   *
   * Accepts either:
   *  - fit(Xarray, yarray, { intercept })
   *  - fit({ X: 'col' | ['col1','col2'], y: 'target', data, omit_missing, intercept })
   *
   * Returns this for chaining.
   */
  fit(X, y = null, opts = {}) {
    let result;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      ('X' in X || 'columns' in X) &&
      'data' in X
    ) {
      const callOpts = { ...X };
      if (callOpts.intercept === undefined) {
        callOpts.intercept = this.params.intercept;
      }
      result = lmFit(callOpts);
    } else {
      const callOpts = { ...opts };
      if (callOpts.intercept === undefined) {
        callOpts.intercept = this.params.intercept;
      }
      result = lmFit(X, y, callOpts);
    }

    this.model = result;
    this.coef = result.coefficients;
    this.fitted = true;

    return this;
  }

  /**
   * Predict using the fitted coefficients.
   *
   * Accepts numeric arrays or declarative objects { X, data }.
   */
  predict(X, { intercept = undefined, data = null } = {}) {
    if (!this.fitted || !this.coef) {
      throw new Error('lm: estimator is not fitted. Call fit() before predict().');
    }

    const useIntercept = intercept === undefined ? this.params.intercept : intercept;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      ('data' in X || 'X' in X || 'columns' in X)
    ) {
      return lmPredict(this.coef, X, { intercept: useIntercept });
    }

    return lmPredict(this.coef, X, { intercept: useIntercept, data });
  }

  /**
   * Return summary statistics for the fitted model.
   */
  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('lm: estimator is not fitted. Call fit() before summary().');
    }
    return lmSummary(this.model);
  }

  /**
   * Serialize estimator state.
   */
  toJSON() {
    return {
      __class__: 'lm',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  /**
   * Restore from serialized representation.
   */
  static fromJSON(obj = {}) {
    const inst = new lm(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.coef = obj.model.coefficients || null;
      inst.fitted = true;
    }
    return inst;
  }
}

// Preserve functional API as static helpers on the class.
Object.assign(lm, {
  fit: lmFit,
  predict: lmPredict,
  summary: lmSummary
});

export default lm;

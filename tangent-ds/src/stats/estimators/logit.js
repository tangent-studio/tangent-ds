/**
 * logit - Logistic Regression estimator (IRLS)
 *
 * Scikit-learn style wrapper around src/stats/logit.js providing a class-based API.
 * Supports both numeric arrays and declarative table-style inputs.
 */

import { Classifier } from '../../core/estimators/estimator.js';
import * as logitFn from '../logit.js';

const DEFAULT_PARAMS = {
  intercept: true,
  maxIter: 100,
  tol: 1e-8
};

export class logit extends Classifier {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;

    this.model = null;
    this.coefficients = null;
  }

  /**
   * Fit logistic regression.
   *
   * Accepts numeric arrays or declarative form:
   *   fit({ X: ['feat1'], y: 'label', data, intercept, maxIter, tol })
   */
  fit(X, y = null, opts = {}) {
    let result;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      ('X' in X || 'data' in X || 'columns' in X)
    ) {
      const callOpts = { ...X };
      if (callOpts.intercept === undefined) callOpts.intercept = this.params.intercept;
      if (callOpts.maxIter === undefined) callOpts.maxIter = this.params.maxIter;
      if (callOpts.tol === undefined) callOpts.tol = this.params.tol;

      result = logitFn.fit(callOpts);
    } else {
      const callOpts = { ...opts };
      if (callOpts.intercept === undefined) callOpts.intercept = this.params.intercept;
      if (callOpts.maxIter === undefined) callOpts.maxIter = this.params.maxIter;
      if (callOpts.tol === undefined) callOpts.tol = this.params.tol;

      result = logitFn.fit(X, y, callOpts);
    }

    this.model = result;
    this.coefficients = result.coefficients;
    this.fitted = true;

    return this;
  }

  /**
   * Predict probabilities for new data.
   */
  predictProba(X, { intercept = undefined } = {}) {
    if (!this.fitted || !this.coefficients) {
      throw new Error('logit: estimator not fitted. Call fit() first.');
    }

    const useIntercept = intercept === undefined ? this.params.intercept : intercept;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      ('data' in X || 'X' in X || 'columns' in X)
    ) {
      return logitFn.predict(this.coefficients, X, { intercept: useIntercept });
    }

    return logitFn.predict(this.coefficients, X, { intercept: useIntercept });
  }

  /**
   * Predict discrete classes by thresholding probabilities (default 0.5).
   */
  predict(X, { threshold = 0.5, intercept = undefined } = {}) {
    const probs = this.predictProba(X, { intercept });
    return logitFn.classify(probs, threshold);
  }

  /**
   * Convenience summary of fitted model diagnostics.
   */
  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('logit: estimator not fitted. Call fit() first.');
    }

    const { coefficients, fitted, logLikelihood, iterations, converged } = this.model;
    return {
      coefficients,
      nObservations: fitted ? fitted.length : 0,
      logLikelihood,
      iterations,
      converged
    };
  }

  toJSON() {
    return {
      __class__: 'logit',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  static fromJSON(obj = {}) {
    const inst = new logit(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.coefficients = obj.model.coefficients || null;
      inst.fitted = true;
    }
    return inst;
  }
}

// Preserve functional helpers as static attachments.
Object.assign(logit, logitFn);

export default logit;

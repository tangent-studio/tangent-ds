/**
 * LDA - Linear Discriminant Analysis estimator (class wrapper)
 *
 * Provides a scikit-learn style estimator around the functional LDA implementation
 * in src/mva/lda.js. Accepts both numeric arrays and declarative table-style inputs:
 *
 *  - Numeric style:
 *      const lda = new LDA();
 *      lda.fit(X, y);
 *      const preds = lda.predict(Xnew);
 *
 *  - Declarative style (uses core table helpers via underlying functions):
 *      lda.fit({ X: ['col1','col2'], y: 'label', data: tableLike, omit_missing: true });
 *      lda.predict({ X: ['col1','col2'], data: otherTable });
 *
 * The wrapper stores the fitted internal model returned by the functional API and
 * exposes .predict(), .transform(), .summary(), .toJSON()/fromJSON().
 */

import { Classifier } from '../../core/estimators/estimator.js';
import * as ldaFn from '../lda.js';

export class LDA extends Classifier {
  /**
   * @param {Object} params - optional hyperparameters (none required for basic LDA)
   */
  constructor(params = {}) {
    super(params);
    // The fitted model object returned by ldaFn.fit(...)
    this.model = null;
    this.fitted = false;
  }

  /**
   * Fit the LDA model.
   *
   * Supports:
   *  - fit(Xarray, yarray)
   *  - fit({ X: 'col'|'[cols]', y: 'label', data: tableLike, omit_missing })
   *
   * Returns: this
   */
  fit(X, y = null, opts = {}) {
    let result;

    // If first argument is a declarative options object (contains data/X/y), forward directly
    if (X && typeof X === 'object' && !Array.isArray(X) && ('data' in X || 'X' in X || 'x' in X)) {
      // Merge instance params with provided options when appropriate
      const callOpts = { ...X };
      // If intercept-like or other params existed, they'd be merged here.
      // Underlying ldaFn.fit supports receiving a single object { X, y, data, ... }
      result = ldaFn.fit(callOpts);
    } else {
      // Positional numeric call: fit(Xarray, yarray, opts)
      // pass opts through if provided
      result = ldaFn.fit(X, y);
    }

    // Save model and metadata
    this.model = result;
    this.fitted = true;
    return this;
  }

  /**
   * Transform input X to discriminant scores (delegates to functional transform).
   *
   * Accepts:
   *  - numeric array X
   *  - declarative object { X: cols, data: tableLike }
   */
  transform(X) {
    if (!this.fitted || !this.model) {
      throw new Error('LDA: estimator not fitted. Call fit() first.');
    }

    return ldaFn.transform(this.model, X);
  }

  /**
   * Predict class labels for X.
   *
   * Accepts:
   *  - numeric array X
   *  - declarative object { X: cols, data: tableLike }
   */
  predict(X) {
    if (!this.fitted || !this.model) {
      throw new Error('LDA: estimator not fitted. Call fit() first.');
    }

    return ldaFn.predict(this.model, X);
  }

  /**
   * Return a small summary of the fitted model.
   */
  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('LDA: estimator not fitted. Call fit() first.');
    }

    const { classes, discriminantAxes, eigenvalues } = this.model;
    return {
      classes,
      nComponents: discriminantAxes ? discriminantAxes.length : 0,
      eigenvalues,
    };
  }

  /**
   * JSON serialization helper.
   */
  toJSON() {
    return {
      __class__: 'LDA',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model,
    };
  }

  /**
   * Restore an instance from JSON produced by toJSON().
   */
  static fromJSON(obj = {}) {
    const inst = new LDA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = true;
    }
    return inst;
  }
}

export default LDA;

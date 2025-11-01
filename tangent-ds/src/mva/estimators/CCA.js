/**
 * Canonical Correlation Analysis (CCA) estimator
 *
 * Provides a scikit-learn style interface around the functional CCA utilities.
 */

import { Transformer } from '../../core/estimators/estimator.js';
import * as ccaFn from '../cca.js';

export class CCA extends Transformer {
  constructor(params = {}) {
    super(params);
    this.model = null;
  }

  fit(X, Y = null, opts = {}) {
    let result;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      (X.data || X.X || X.Y || X.columnsX || X.columnsY)
    ) {
      const callOpts = { ...this.params, ...X };
      result = ccaFn.fit(callOpts);
    } else {
      const options = { ...this.params, ...opts };
      result = ccaFn.fit(X, Y, options);
    }

    this.model = result;
    this.fitted = true;
    return this;
  }

  transformX(X, opts = {}) {
    if (!this.fitted || !this.model) {
      throw new Error('CCA: estimator not fitted. Call fit() before transformX().');
    }
    return ccaFn.transformX(this.model, X, opts);
  }

  transformY(Y, opts = {}) {
    if (!this.fitted || !this.model) {
      throw new Error('CCA: estimator not fitted. Call fit() before transformY().');
    }
    return ccaFn.transformY(this.model, Y, opts);
  }

  transform(X, Y, opts = {}) {
    if (!this.fitted || !this.model) {
      throw new Error('CCA: estimator not fitted. Call fit() before transform().');
    }
    return ccaFn.transform(this.model, X, Y, opts);
  }

  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('CCA: estimator not fitted. Call fit() before summary().');
    }
    return {
      nSamples: this.model.nSamples,
      nComponents: this.model.nComponents,
      correlations: this.model.correlations.slice()
    };
  }

  toJSON() {
    return {
      __class__: 'CCA',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  static fromJSON(obj = {}) {
    const inst = new CCA(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = !!obj.fitted;
    }
    return inst;
  }
}

export default CCA;

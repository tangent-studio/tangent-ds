import { Regressor } from '../../core/estimators/estimator.js';
import * as lmmFn from '../lmm.js';
import { prepareXY, prepareX, normalize } from '../../core/table.js';

const DEFAULT_PARAMS = {
  intercept: true,
  maxIter: 100,
  tol: 1e-6
};

export class lmm extends Regressor {
  constructor(params = {}) {
    const merged = { ...DEFAULT_PARAMS, ...params };
    super(merged);
    this.params = merged;

    this.model = null;
  }

  fit(X, y = null, groups = null, opts = {}) {
    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      ('data' in X || 'X' in X || 'columns' in X)
    ) {
      const args = X;
      if (!('data' in args)) {
        throw new Error('lmm.fit declarative usage requires a `data` property.');
      }
      if (!args.X || !args.y || !args.groups) {
        throw new Error('lmm.fit declarative usage requires X, y, and groups keys.');
      }

      const prepared = prepareXY({
        X: args.X,
        y: args.y,
        data: args.data,
        omit_missing: args.omit_missing !== undefined ? args.omit_missing : true
      });

      const rows = prepared.rows || normalize(args.data);
      const groupCol = args.groups;
      const groupsArr = rows.map((r) => r[groupCol]);

      const callOpts = {
        intercept: args.intercept !== undefined ? args.intercept : this.params.intercept,
        maxIter: args.maxIter !== undefined ? args.maxIter : this.params.maxIter,
        tol: args.tol !== undefined ? args.tol : this.params.tol
      };

      const result = lmmFn.fit(prepared.X, prepared.y, groupsArr, callOpts);
      this.model = result;
      this.fitted = true;
      return this;
    }

    const callOpts = {
      intercept: opts.intercept !== undefined ? opts.intercept : this.params.intercept,
      maxIter: opts.maxIter !== undefined ? opts.maxIter : this.params.maxIter,
      tol: opts.tol !== undefined ? opts.tol : this.params.tol
    };

    const result = lmmFn.fit(X, y, groups, callOpts);
    this.model = result;
    this.fitted = true;
    return this;
  }

  predict(X, groups = null, { intercept = undefined } = {}) {
    if (!this.fitted || !this.model) {
      throw new Error('lmm: estimator not fitted. Call fit() first.');
    }

    const useIntercept = intercept === undefined ? this.params.intercept : intercept;

    if (
      X &&
      typeof X === 'object' &&
      !Array.isArray(X) &&
      ('data' in X || 'X' in X || 'columns' in X)
    ) {
      const args = X;
      if (!('data' in args)) {
        throw new Error('lmm.predict declarative usage requires a `data` property.');
      }
      if (!args.X || !args.groups) {
        throw new Error('lmm.predict declarative usage requires X and groups keys.');
      }

      const preparedX = prepareX({
        columns: args.X,
        data: args.data,
        omit_missing: args.omit_missing !== undefined ? args.omit_missing : true
      });

      const rows = preparedX.rows || normalize(args.data);
      const groupsArr = rows.map((r) => r[args.groups]);

      return lmmFn.predict(this.model, preparedX.X, groupsArr, { intercept: useIntercept });
    }

    return lmmFn.predict(this.model, X, groups, { intercept: useIntercept });
  }

  summary() {
    if (!this.fitted || !this.model) {
      throw new Error('lmm: estimator not fitted. Call fit() first.');
    }

    const {
      fixedEffects,
      randomEffects,
      varResidual,
      varRandom,
      logLikelihood,
      nGroups
    } = this.model;

    return {
      nGroups,
      fixedEffects,
      randomEffects,
      varResidual,
      varRandom,
      logLikelihood
    };
  }

  toJSON() {
    return {
      __class__: 'lmm',
      params: this.getParams(),
      fitted: !!this.fitted,
      model: this.model
    };
  }

  static fromJSON(obj = {}) {
    const inst = new lmm(obj.params || {});
    if (obj.model) {
      inst.model = obj.model;
      inst.fitted = true;
    }
    return inst;
  }
}

Object.assign(lmm, lmmFn);

export default lmm;

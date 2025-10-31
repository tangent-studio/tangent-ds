import { describe, it, expect } from 'vitest';
import { lm } from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

describe('OLS linear regression (class API)', () => {
  describe('fit', () => {
    it('should fit simple linear model', () => {
      // y = 2x + 1
      const X = [[0], [1], [2], [3], [4]];
      const y = [1, 3, 5, 7, 9];

      const model = new lm({ intercept: true });
      model.fit(X, y);

      // coefficients: [intercept, slope]
      expect(approxEqual(model.coef[0], 1, 0.001)).toBe(true); // intercept
      expect(approxEqual(model.coef[1], 2, 0.001)).toBe(true); // slope
      expect(approxEqual(model.model.rSquared, 1, 0.001)).toBe(true); // perfect fit
    });

    it('should fit multiple regression', () => {
      // y = 1 + 2*x1 + 3*x2
      const X = [
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2]
      ];
      const y = [6, 8, 9, 11];

      const model = new lm({ intercept: true });
      model.fit(X, y);

      expect(model.coef.length).toBe(3);
      expect(approxEqual(model.coef[0], 1, 0.001)).toBe(true);
      expect(approxEqual(model.coef[1], 2, 0.001)).toBe(true);
      expect(approxEqual(model.coef[2], 3, 0.001)).toBe(true);
    });

    it('should compute residuals and fitted values', () => {
      const X = [[1], [2], [3]];
      const y = [2, 4, 6];

      const model = new lm({ intercept: true });
      model.fit(X, y);

      // fitted/residuals are stored on model.model (the underlying lm result)
      expect(model.model.fitted.length).toBe(3);
      expect(model.model.residuals.length).toBe(3);

      // Check residuals are small for good fit
      const maxResid = Math.max(...model.model.residuals.map(Math.abs));
      expect(maxResid).toBeLessThan(0.1);
    });
  });

  describe('predict', () => {
    it('should make predictions', () => {
      const X = [[1], [2], [3]];
      const y = [3, 5, 7];

      const lr = new lm({ intercept: true });
      lr.fit(X, y);

      const Xnew = [[4], [5]];
      const predictions = lr.predict(Xnew);

      expect(predictions.length).toBe(2);
      expect(approxEqual(predictions[0], 9, 0.001)).toBe(true);
      expect(approxEqual(predictions[1], 11, 0.001)).toBe(true);
    });
  });

  describe('summary', () => {
    it('should provide model summary', () => {
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 5, 8];

      const lr = new lm({ intercept: true });
      lr.fit(X, y);
      const summ = lr.summary();

      expect(summ.nObservations).toBe(4);
      expect(summ.nPredictors).toBe(2);
      expect(summ.rSquared).toBeGreaterThan(0);
      expect(summ.fStatistic).toBeGreaterThan(0);
    });
  });
});

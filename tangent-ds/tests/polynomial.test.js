import { describe, it, expect } from 'vitest';
import { fit, predict, polynomialFeatures } from '../src/ml/polynomial.js';
import { PolynomialRegressor } from '../src/ml/index.js';
import { approxEqual } from '../src/core/math.js';

describe('polynomial regression', () => {
  describe('polynomialFeatures', () => {
    it('should create polynomial features', () => {
      const X = [[1], [2], [3]];
      const poly = polynomialFeatures(X, 2);
      
      expect(poly).toEqual([
        [1, 1],   // x^1, x^2
        [2, 4],
        [3, 9]
      ]);
    });

    it('should handle degree 3', () => {
      const X = [[2]];
      const poly = polynomialFeatures(X, 3);
      
      expect(poly).toEqual([[2, 4, 8]]); // x, x^2, x^3
    });
  });

  describe('fit', () => {
    it('should fit quadratic function', () => {
      // y = x^2
      const X = [1, 2, 3, 4];
      const y = [1, 4, 9, 16];
      
      const model = fit(X, y, { degree: 2 });
      
      expect(model.degree).toBe(2);
      expect(model.coefficients.length).toBe(3); // intercept + 2 coefficients
      expect(model.rSquared).toBeGreaterThan(0.99); // Nearly perfect fit
    });

    it('should fit cubic function', () => {
      // y = x^3 - x
      const X = [-2, -1, 0, 1, 2];
      const y = [-6, 0, 0, 0, 6];
      
      const model = fit(X, y, { degree: 3 });
      
      expect(model.degree).toBe(3);
      expect(model.fitted.length).toBe(5);
      // Check that it fits reasonably well
      const maxError = Math.max(...model.residuals.map(r => Math.abs(r)));
      expect(maxError).toBeLessThan(2);
    });

    it('should work with 2D input', () => {
      const X = [[1], [2], [3]];
      const y = [1, 4, 9];
      
      const model = fit(X, y, { degree: 2 });
      
      expect(model.coefficients.length).toBe(3);
    });
  });

  describe('predict', () => {
    it('should predict new values', () => {
      const X = [1, 2, 3];
      const y = [1, 4, 9]; // y = x^2
      
      const model = fit(X, y, { degree: 2 });
      const predictions = predict(model, [4, 5]);
      
      expect(predictions.length).toBe(2);
      // Should be close to 16 and 25
      expect(approxEqual(predictions[0], 16, 1)).toBe(true);
      expect(approxEqual(predictions[1], 25, 2)).toBe(true);
    });

    it('should handle 2D input', () => {
      const X = [[1], [2], [3]];
      const y = [1, 4, 9];
      
      const model = fit(X, y, { degree: 2 });
      const predictions = predict(model, [[4]]);
      
      expect(predictions.length).toBe(1);
      expect(approxEqual(predictions[0], 16, 1)).toBe(true);
    });
  });
});

describe('PolynomialRegressor (class API)', () => {
  it('should fit and predict using class interface', () => {
    const X = [1, 2, 3, 4];
    const y = [1, 4, 9, 16];

    const reg = new PolynomialRegressor({ degree: 2 });
    reg.fit(X, y);
    const preds = reg.predict([5, 6]);

    expect(preds.length).toBe(2);
    expect(approxEqual(preds[0], 25, 2)).toBe(true);
    expect(approxEqual(preds[1], 36, 3)).toBe(true);
  });

  it('should expose summary information', () => {
    const X = [1, 2, 3];
    const y = [1, 4, 9];
    const reg = new PolynomialRegressor({ degree: 2 });
    reg.fit(X, y);

    const summary = reg.summary();
    expect(summary.degree).toBe(2);
    expect(summary.coefficients.length).toBeGreaterThan(0);
  });
});

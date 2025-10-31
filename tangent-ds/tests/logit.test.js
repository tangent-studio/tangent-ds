import { describe, it, expect } from 'vitest';
import { logit } from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

describe('logistic regression', () => {
  describe('fit', () => {
    it('should fit logistic regression model', () => {
      // Simple separable data
      const X = [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6]
      ];
      const y = [0, 0, 0, 1, 1, 1];
      
      const clf = new logit({ intercept: true, maxIter: 100 });
      clf.fit(X, y);
      const model = clf.model;

      expect(clf.coefficients.length).toBe(2);
      expect(model.fitted.length).toBe(6);
      expect(model.iterations).toBeGreaterThan(0);
      expect(model.logLikelihood).toBeLessThan(0);
    });

    it('should produce probabilities between 0 and 1', () => {
      const X = [[1], [2], [3], [4]];
      const y = [0, 0, 1, 1];
      const clf = new logit({ intercept: true });
      clf.fit(X, y);

      for (const p of clf.model.fitted) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
    });

    it('should throw for non-binary response', () => {
      const X = [[1], [2]];
      const y = [0, 2];
      
      const clf = new logit();
      expect(() => clf.fit(X, y)).toThrow();
    });
  });

  describe('predict', () => {
    it('should predict probabilities for new data', () => {
      const X = [[1], [2], [3], [4]];
      const y = [0, 0, 1, 1];
      const clf = new logit({ intercept: true });
      clf.fit(X, y);

      const Xnew = [[2.5], [3.5]];
      const predictions = clf.predictProba(Xnew);

      expect(predictions.length).toBe(2);
      expect(predictions[0]).toBeGreaterThan(0);
      expect(predictions[0]).toBeLessThan(1);
    });
  });

  describe('classify', () => {
    it('should classify based on threshold', () => {
      const probabilities = [0.2, 0.4, 0.6, 0.8];
      const classes = logit.classify(probabilities, 0.5);
      
      expect(classes).toEqual([0, 0, 1, 1]);
    });

    it('should use custom threshold', () => {
      const probabilities = [0.2, 0.4, 0.6, 0.8];
      const classes = logit.classify(probabilities, 0.7);
      
      expect(classes).toEqual([0, 0, 0, 1]);
    });
  });
});

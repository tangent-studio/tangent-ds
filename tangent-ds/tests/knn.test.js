import { describe, it, expect } from 'vitest';
import { KNNClassifier, KNNRegressor } from '../src/ml/index.js';

describe('KNN estimators', () => {
  describe('KNNClassifier', () => {
    it('should classify simple dataset', () => {
      const X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [5, 5],
        [5, 6],
        [6, 5],
        [6, 6]
      ];
      const y = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'];

      const clf = new KNNClassifier({ k: 3 });
      clf.fit(X, y);

      const preds = clf.predict([[0.2, 0.3], [5.5, 5.2]]);
      expect(preds).toEqual(['A', 'B']);

      const proba = clf.predictProba([[0.2, 0.3]])[0];
      expect(proba.A).toBeGreaterThan(proba.B);
    });
  });

  describe('KNNRegressor', () => {
    it('should predict average of neighbors', () => {
      const X = [[0], [1], [2], [3]];
      const y = [0, 2, 4, 6];

      const reg = new KNNRegressor({ k: 2 });
      reg.fit(X, y);

      const preds = reg.predict([[1.5], [2.5]]);
      expect(preds[0]).toBeCloseTo(3, 5);
      expect(preds[1]).toBeCloseTo(5, 5);
    });
  });
});

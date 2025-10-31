import { describe, it, expect } from 'vitest';
import { lmm } from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

describe('linear mixed models', () => {
  describe('fit', () => {
    it('should fit random-intercept model', () => {
      // Simple grouped data
      const X = [
        [1],
        [2],
        [1],
        [2],
        [1],
        [2]
      ];
      const y = [2, 4, 3, 5, 2.5, 4.5];
      const groups = ['A', 'A', 'B', 'B', 'C', 'C'];
      
      const model = new lmm({ intercept: true });
      model.fit(X, y, groups);
      const result = model.model;

      expect(result.fixedEffects.length).toBe(2);
      expect(result.randomEffects.length).toBe(3); // 3 groups
      expect(result.nGroups).toBe(3);
      expect(result.varResidual).toBeGreaterThan(0);
      expect(result.varRandom).toBeGreaterThanOrEqual(0);
    });

    it('should estimate group effects', () => {
      // Data where groups have clear differences
      const X = [[1], [1], [1], [1]];
      const y = [10, 11, 20, 21];
      const groups = ['A', 'A', 'B', 'B'];
      
      const model = new lmm({ intercept: true });
      model.fit(X, y, groups);
      
      // Group B should have higher random effect than group A
      const groupA = model.model.randomEffects[model.model.groupMap['A']];
      const groupB = model.model.randomEffects[model.model.groupMap['B']];
      
      expect(groupB).toBeGreaterThan(groupA);
    });
  });

  describe('predict', () => {
    it('should predict for known groups', () => {
      const X = [[1], [2], [1], [2]];
      const y = [2, 4, 3, 5];
      const groups = ['A', 'A', 'B', 'B'];
      
      const model = new lmm({ intercept: true });
      model.fit(X, y, groups);

      const Xnew = [[1.5], [1.5]];
      const groupsNew = ['A', 'B'];
      const predictions = model.predict(Xnew, groupsNew);
      
      expect(predictions.length).toBe(2);
      // Predictions for different groups should differ
      expect(Math.abs(predictions[0] - predictions[1])).toBeGreaterThan(0);
    });

    it('should predict for new groups using only fixed effects', () => {
      const X = [[1], [2]];
      const y = [2, 4];
      const groups = ['A', 'A'];
      
      const model = new lmm({ intercept: true });
      model.fit(X, y, groups);
      
      const Xnew = [[1.5]];
      const groupsNew = ['NewGroup'];
      const predictions = model.predict(Xnew, groupsNew);
      
      expect(predictions.length).toBe(1);
      // Prediction should be reasonable (not necessarily > 0 for all cases)
      expect(typeof predictions[0]).toBe('number');
      expect(Number.isFinite(predictions[0])).toBe(true);
    });
  });
});

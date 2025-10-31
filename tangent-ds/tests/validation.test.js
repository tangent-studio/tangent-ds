import { describe, it, expect, beforeEach } from 'vitest';
import { trainTestSplit, kFold, stratifiedKFold, groupKFold, leaveOneOut, shuffleSplit, crossValidate } from '../src/ml/validation.js';
import { setSeed } from '../src/ml/utils.js';

describe('ML Validation', () => {
  beforeEach(() => {
    setSeed(42); // Ensure reproducibility
  });

  describe('trainTestSplit', () => {
    it('should split data with default ratio', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [1, 2, 3, 4, 5];
      
      const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X, y, { shuffle: false });
      
      expect(XTrain.length).toBe(4);
      expect(XTest.length).toBe(1);
      expect(yTrain.length).toBe(4);
      expect(yTest.length).toBe(1);
    });

    it('should respect custom ratio', () => {
      const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
      const y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      
      const { XTrain, XTest } = trainTestSplit(X, y, { ratio: 0.7, shuffle: false });
      
      expect(XTrain.length).toBe(7);
      expect(XTest.length).toBe(3);
    });

    it('should work without y', () => {
      const X = [[1], [2], [3]];
      const { XTrain, XTest, yTrain, yTest } = trainTestSplit(X);
      
      expect(XTrain.length).toBeGreaterThan(0);
      expect(XTest.length).toBeGreaterThan(0);
      expect(yTrain).toBeUndefined();
      expect(yTest).toBeUndefined();
    });

    it('should be reproducible with seed', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [1, 2, 3, 4, 5];
      
      const split1 = trainTestSplit(X, y, { seed: 123 });
      const split2 = trainTestSplit(X, y, { seed: 123 });
      
      expect(split1.trainIndices).toEqual(split2.trainIndices);
      expect(split1.testIndices).toEqual(split2.testIndices);
    });
  });

  describe('kFold', () => {
    it('should create k folds', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [1, 2, 3, 4, 5];
      
      const folds = kFold(X, y, 5);
      
      expect(folds.length).toBe(5);
      expect(folds[0].description).toBe('KFold');
      expect(folds[0].nSplits).toBe(5);
    });

    it('should have non-overlapping test sets', () => {
      const X = [[1], [2], [3], [4]];
      const y = [1, 2, 3, 4];
      
      const folds = kFold(X, y, 2, false);
      
      const allTestIndices = folds.flatMap(f => f.testIndices);
      const uniqueTestIndices = new Set(allTestIndices);
      
      expect(uniqueTestIndices.size).toBe(X.length);
    });

    it('should throw error for invalid k', () => {
      const X = [[1], [2]];
      const y = [1, 2];
      
      expect(() => kFold(X, y, 1)).toThrow();
      expect(() => kFold(X, y, 3)).toThrow();
    });
  });

  describe('stratifiedKFold', () => {
    it('should maintain class proportions', () => {
      const X = [[1], [2], [3], [4], [5], [6]];
      const y = ['A', 'A', 'A', 'B', 'B', 'B'];
      
      const folds = stratifiedKFold(X, y, 2);
      
      expect(folds.length).toBe(2);
      
      // Each fold should have samples from both classes
      folds.forEach(fold => {
        const testLabels = fold.testIndices.map(i => y[i]);
        const hasA = testLabels.includes('A');
        const hasB = testLabels.includes('B');
        expect(hasA).toBe(true);
        expect(hasB).toBe(true);
      });
    });
  });

  describe('groupKFold', () => {
    it('should keep groups together', () => {
      const X = [[1], [2], [3], [4]];
      const y = [1, 2, 3, 4];
      const groups = ['A', 'A', 'B', 'B'];
      
      const folds = groupKFold(X, y, groups, 2);
      
      expect(folds.length).toBe(2);
      
      // Check that groups are not split
      folds.forEach(fold => {
        const testGroups = fold.testIndices.map(i => groups[i]);
        const uniqueTestGroups = new Set(testGroups);
        
        fold.trainIndices.forEach(i => {
          expect(uniqueTestGroups.has(groups[i])).toBe(false);
        });
      });
    });

    it('should throw error if k > number of groups', () => {
      const X = [[1], [2]];
      const y = [1, 2];
      const groups = ['A', 'A'];
      
      expect(() => groupKFold(X, y, groups, 3)).toThrow();
    });
  });

  describe('leaveOneOut', () => {
    it('should create n folds for n samples', () => {
      const X = [[1], [2], [3]];
      const y = [1, 2, 3];
      
      const folds = leaveOneOut(X, y);
      
      expect(folds.length).toBe(3);
      expect(folds[0].description).toBe('LeaveOneOut');
    });

    it('should have single test sample per fold', () => {
      const X = [[1], [2], [3]];
      const y = [1, 2, 3];
      
      const folds = leaveOneOut(X, y);
      
      folds.forEach(fold => {
        expect(fold.testIndices.length).toBe(1);
        expect(fold.trainIndices.length).toBe(2);
      });
    });
  });

  describe('shuffleSplit', () => {
    it('should create n splits', () => {
      const X = [[1], [2], [3], [4], [5]];
      const y = [1, 2, 3, 4, 5];
      
      const splits = shuffleSplit(X, y, { nSplits: 3 });
      
      expect(splits.length).toBe(3);
      expect(splits[0].description).toBe('ShuffleSplit');
    });

    it('should respect test ratio', () => {
      const X = Array.from({ length: 10 }, (_, i) => [i]);
      const y = Array.from({ length: 10 }, (_, i) => i);
      
      const splits = shuffleSplit(X, y, { nSplits: 2, testRatio: 0.3 });
      
      splits.forEach(split => {
        expect(split.testIndices.length).toBe(3);
        expect(split.trainIndices.length).toBe(7);
      });
    });
  });

  describe('crossValidate', () => {
    it('should perform cross-validation', () => {
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 6, 8];
      
      const fitFn = (XTrain, yTrain) => {
        // Simple mean model
        const meanY = yTrain.reduce((a, b) => a + b, 0) / yTrain.length;
        return { predict: () => [meanY, meanY] };
      };
      
      const scoreFn = (model, XTest, yTest) => {
        const predictions = model.predict();
        // Simple accuracy (within 2 of true value)
        let correct = 0;
        for (let i = 0; i < yTest.length; i++) {
          if (Math.abs(predictions[i] - yTest[i]) < 2) correct++;
        }
        return correct / yTest.length;
      };
      
      const folds = kFold(X, y, 2, false);
      const result = crossValidate(fitFn, scoreFn, X, y, folds);
      
      expect(result.scores.length).toBe(2);
      expect(result.meanScore).toBeGreaterThanOrEqual(0);
      expect(result.meanScore).toBeLessThanOrEqual(1);
      expect(result.stdScore).toBeGreaterThanOrEqual(0);
    });
  });
});

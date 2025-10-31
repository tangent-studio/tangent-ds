import { describe, it, expect } from 'vitest';
import { LDA } from '../src/mva/index.js';
import { approxEqual } from '../src/core/math.js';

describe('LDA - Linear Discriminant Analysis (class API)', () => {
  describe('fit', () => {
    it('should fit LDA on 2-class data', () => {
      const X = [
        [1, 2], [2, 3],    // Class 0
        [8, 9], [9, 10]    // Class 1
      ];
      const y = [0, 0, 1, 1];

      const lda = new LDA();
      lda.fit(X, y);
      const model = lda.model;

      expect(model.scores.length).toBe(4);
      expect(model.loadings.length).toBe(2);
      expect(model.classes.length).toBe(2);
      expect(model.eigenvalues.length).toBe(1); // k-1 components
    });

    it('should fit LDA on 3-class data', () => {
      const X = [
        [0, 0], [0, 1],       // Class 0
        [5, 5], [5, 6],       // Class 1
        [10, 10], [10, 11]    // Class 2
      ];
      const y = [0, 0, 1, 1, 2, 2];

      const lda = new LDA();
      lda.fit(X, y);
      const model = lda.model;

      expect(model.classes.length).toBe(3);
      expect(model.eigenvalues.length).toBe(2); // k-1 = 2 components
      expect(model.scores.length).toBe(6);
    });

    it('should include class labels in scores', () => {
      const X = [[1, 2], [8, 9]];
      const y = ['A', 'B'];

      const lda = new LDA();
      lda.fit(X, y);
      const model = lda.model;

      expect(model.scores[0].class).toBe('A');
      expect(model.scores[1].class).toBe('B');
    });

    it('should compute discriminant scores', () => {
      const X = [[1, 2], [2, 3], [8, 9], [9, 10]];
      const y = [0, 0, 1, 1];

      const lda = new LDA();
      lda.fit(X, y);
      const model = lda.model;

      // Classes should be separated in discriminant space
      const class0Scores = model.scores.filter(s => s.class === 0).map(s => s.ld1);
      const class1Scores = model.scores.filter(s => s.class === 1).map(s => s.ld1);

      const mean0 = class0Scores.reduce((a, b) => a + b, 0) / class0Scores.length;
      const mean1 = class1Scores.reduce((a, b) => a + b, 0) / class1Scores.length;

      // Class means should be different
      expect(Math.abs(mean1 - mean0)).toBeGreaterThan(0);
    });

    it('should throw error for single class', () => {
      const X = [[1], [2]];
      const y = [0, 0];

      const lda = new LDA();
      expect(() => lda.fit(X, y)).toThrow();
    });

    it('should throw error for mismatched dimensions', () => {
      const X = [[1], [2]];
      const y = [0];

      const lda = new LDA();
      expect(() => lda.fit(X, y)).toThrow();
    });
  });

  describe('transform', () => {
    it('should transform new data', () => {
      const X = [[1, 2], [2, 3], [8, 9], [9, 10]];
      const y = [0, 0, 1, 1];

      const lda = new LDA();
      lda.fit(X, y);

      const Xnew = [[1.5, 2.5], [8.5, 9.5]];
      const transformed = lda.transform(Xnew);

      expect(transformed.length).toBe(2);
      expect(transformed[0].ld1).toBeDefined();
    });
  });

  describe('predict', () => {
    it('should predict class labels', () => {
      const X = [[1, 2], [2, 3], [8, 9], [9, 10]];
      const y = [0, 0, 1, 1];

      const lda = new LDA();
      lda.fit(X, y);

      // Points close to training data
      const Xnew = [[1.5, 2.5], [8.5, 9.5]];
      const predictions = lda.predict(Xnew);

      expect(predictions.length).toBe(2);
      expect(predictions[0]).toBe(0); // Should predict class 0
      expect(predictions[1]).toBe(1); // Should predict class 1
    });

    it('should work with string labels', () => {
      const X = [[1, 2], [8, 9]];
      const y = ['A', 'B'];

      const lda = new LDA();
      lda.fit(X, y);
      const predictions = lda.predict([[1.5, 2.5]]);

      expect(predictions[0]).toBe('A');
    });
  });
});

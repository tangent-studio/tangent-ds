import { describe, it, expect } from 'vitest';
import { pca as pcaFns, PCA } from '../src/mva/index.js';
import { approxEqual } from '../src/core/math.js';

describe('PCA - Principal Component Analysis (class API)', () => {
  describe('fit', () => {
    it('should fit PCA on simple 2D data', () => {
      const X = [
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8]
      ];

      const p = new PCA();
      p.fit(X);
      const model = p.model;

      expect(model.scores.length).toBe(4);
      expect(model.loadings.length).toBe(2);
      expect(model.eigenvalues.length).toBe(2);
      expect(model.varianceExplained.length).toBe(2);
    });

    it('should explain 100% variance with all components', () => {
      const X = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1]
      ];

      const p = new PCA();
      p.fit(X);
      const totalVariance = p.model.varianceExplained.reduce((a, b) => a + b, 0);

      expect(approxEqual(totalVariance, 1, 0.001)).toBe(true);
    });

    it('should center data by default', () => {
      const X = [[10, 20], [11, 21], [12, 22]];
      const p = new PCA({ center: true });
      p.fit(X);

      expect(p.model.means).toBeTruthy();
      expect(p.model.means.length).toBe(2);
      expect(approxEqual(p.model.means[0], 11, 0.001)).toBe(true);
      expect(approxEqual(p.model.means[1], 21, 0.001)).toBe(true);
    });

    it('should scale data when requested', () => {
      const X = [[1, 10], [2, 20], [3, 30]];
      const p = new PCA({ scale: true });
      p.fit(X);

      expect(p.model.sds).toBeTruthy();
      expect(p.model.sds.length).toBe(2);
    });

    it('should sort components by variance', () => {
      const X = [
        [1, 0, 0],
        [2, 0, 0],
        [3, 0.1, 0],
        [4, 0, 0.1]
      ];

      const p = new PCA();
      p.fit(X);
      const model = p.model;

      // First eigenvalue should be largest
      for (let i = 0; i < model.eigenvalues.length - 1; i++) {
        expect(model.eigenvalues[i]).toBeGreaterThanOrEqual(model.eigenvalues[i + 1]);
      }
    });
  });

  describe('transform', () => {
    it('should transform new data', () => {
      const X = [[1, 2], [2, 4], [3, 6]];
      const p = new PCA();
      p.fit(X);
      const model = p.model;

      const Xnew = [[4, 8], [5, 10]];
      const transformed = p.transform(Xnew);

      expect(transformed.length).toBe(2);
      expect(transformed[0].pc1).toBeDefined();
      expect(transformed[0].pc2).toBeDefined();
    });

    it('should apply same standardization as training', () => {
      const X = [[10, 20], [11, 21], [12, 22]];
      const p = new PCA({ center: true, scale: true });
      p.fit(X);

      const Xnew = [[11, 21]]; // Same as mean
      const transformed = p.transform(Xnew);

      // Should be close to origin in PC space
      expect(Math.abs(transformed[0].pc1)).toBeLessThan(1);
    });
  });

  describe('cumulativeVariance', () => {
    it('should compute cumulative variance explained', () => {
      const X = [[1, 0], [2, 0], [3, 0.1]];
      const p = new PCA();
      p.fit(X);
      const model = p.model;

      const cumulative = pcaFns.cumulativeVariance(model);

      expect(cumulative.length).toBe(model.varianceExplained.length);
      expect(cumulative[cumulative.length - 1]).toBeLessThanOrEqual(1.001);

      // Should be increasing
      for (let i = 0; i < cumulative.length - 1; i++) {
        expect(cumulative[i + 1]).toBeGreaterThanOrEqual(cumulative[i]);
      }
    });
  });

  describe('edge cases', () => {
    it('should throw error for insufficient samples', () => {
      const X = [[1, 2]];
      const p = new PCA();
      expect(() => p.fit(X)).toThrow();
    });

    it('should handle 1D data', () => {
      const X = [[1], [2], [3]];
      const p = new PCA();
      p.fit(X);
      const model = p.model;

      expect(model.scores.length).toBe(3);
      expect(model.eigenvalues.length).toBe(1);
    });
  });
});

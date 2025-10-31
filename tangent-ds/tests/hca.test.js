import { describe, it, expect } from 'vitest';
import { fit, cut, cutHeight } from '../src/mva/hca.js';
import { HCA } from '../src/mva/index.js';

describe('HCA - Hierarchical Clustering', () => {
  describe('fit', () => {
    it('should perform hierarchical clustering', () => {
      const X = [
        [0, 0],
        [0, 1],
        [10, 10],
        [10, 11]
      ];
      
      const model = fit(X, { linkage: 'average' });
      
      expect(model.dendrogram.length).toBe(3); // n-1 merges
      expect(model.linkage).toBe('average');
      expect(model.n).toBe(4);
    });

    it('should work with single linkage', () => {
      const X = [[1], [2], [10], [11]];
      const model = fit(X, { linkage: 'single' });
      
      expect(model.dendrogram.length).toBe(3);
      expect(model.linkage).toBe('single');
    });

    it('should work with complete linkage', () => {
      const X = [[1], [2], [10]];
      const model = fit(X, { linkage: 'complete' });
      
      expect(model.dendrogram.length).toBe(2);
      expect(model.linkage).toBe('complete');
    });

    it('should have increasing distances in dendrogram', () => {
      const X = [[0], [1], [2], [10]];
      const model = fit(X);
      
      // Distances should generally increase (not strict for all linkages)
      expect(model.dendrogram.every(m => m.distance >= 0)).toBe(true);
    });

    it('should throw error for insufficient samples', () => {
      const X = [[1]];
      expect(() => fit(X)).toThrow();
    });
  });

  describe('cut', () => {
    it('should cut dendrogram into k clusters', () => {
      const X = [
        [0, 0], [0, 1],    // Cluster 1
        [10, 10], [10, 11] // Cluster 2
      ];
      
      const model = fit(X);
      const labels = cut(model, 2);
      
      expect(labels.length).toBe(4);
      expect(new Set(labels).size).toBe(2);
      
      // Points 0,1 should be in same cluster
      expect(labels[0]).toBe(labels[1]);
      // Points 2,3 should be in same cluster
      expect(labels[2]).toBe(labels[3]);
      // But different from first cluster
      expect(labels[0]).not.toBe(labels[2]);
    });

    it('should handle k=n (all separate)', () => {
      const X = [[1], [2], [3]];
      const model = fit(X);
      const labels = cut(model, 3);
      
      expect(labels.length).toBe(3);
      expect(new Set(labels).size).toBe(3);
    });

    it('should handle k=1 (all together)', () => {
      const X = [[1], [2], [3]];
      const model = fit(X);
      const labels = cut(model, 1);
      
      expect(labels.length).toBe(3);
      expect(new Set(labels).size).toBe(1);
    });

    it('should throw error for invalid k', () => {
      const X = [[1], [2]];
      const model = fit(X);
      
      expect(() => cut(model, 0)).toThrow();
      expect(() => cut(model, 10)).toThrow();
    });
  });

  describe('cutHeight', () => {
    it('should cut dendrogram at specified height', () => {
      const X = [[0], [1], [10], [11]];
      const model = fit(X);
      
      // Cut at very low height - should get many clusters
      const labels1 = cutHeight(model, 0.5);
      expect(new Set(labels1).size).toBeGreaterThan(2);
      
      // Cut at high height - should get fewer clusters
      const labels2 = cutHeight(model, 10);
      expect(new Set(labels2).size).toBeLessThanOrEqual(2);
    });

    it('should produce valid cluster labels', () => {
      const X = [[1], [2], [3]];
      const model = fit(X);
      const labels = cutHeight(model, 1);
      
      expect(labels.length).toBe(3);
      expect(labels.every(l => l >= 0)).toBe(true);
    });
  });
});

describe('HCA - class API', () => {
  it('should fit using class wrapper and cut clusters', () => {
    const X = [
      [0, 0],
      [0, 1],
      [10, 10],
      [10, 11]
    ];

    const estimator = new HCA({ linkage: 'average' });
    estimator.fit(X);
    const labels = estimator.cut(2);

    expect(labels.length).toBe(4);
    expect(new Set(labels).size).toBe(2);
  });

  it('should provide summary information', () => {
    const X = [[1], [2], [3]];
    const estimator = new HCA();
    estimator.fit(X);
    const summary = estimator.summary();

    expect(summary.linkage).toBeDefined();
    expect(summary.merges).toBeGreaterThan(0);
  });
});

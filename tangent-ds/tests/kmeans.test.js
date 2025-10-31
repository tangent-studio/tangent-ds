import { describe, it, expect } from 'vitest';
import { fit as kfit, predict as kpredict, silhouetteScore as ksilhouette } from '../src/ml/kmeans.js';
import { KMeans } from '../src/ml/index.js';

describe('k-means clustering', () => {
  describe('functional API', () => {
    it('should cluster simple 2D data', () => {
      // Create two well-separated clusters
      const data = [
        [0, 0], [0.5, 0], [0, 0.5],      // Cluster 1
        [10, 10], [10.5, 10], [10, 10.5] // Cluster 2
      ];

      const model = kfit(data, { k: 2, maxIter: 100 });

      expect(model.labels.length).toBe(6);
      expect(model.centroids.length).toBe(2);
      expect(model.inertia).toBeGreaterThan(0);
      expect(model.iterations).toBeGreaterThan(0);
    });

    it('should converge for well-separated clusters', () => {
      const data = [
        [0, 0], [1, 0], [0, 1],
        [10, 10], [11, 10], [10, 11],
        [20, 0], [21, 0], [20, 1]
      ];

      const model = kfit(data, { k: 3 });

      expect(model.converged).toBe(true);
      expect(model.centroids.length).toBe(3);
    });

    it('should handle 1D data', () => {
      const data = [[1], [2], [3], [10], [11], [12]];
      const model = kfit(data, { k: 2 });

      expect(model.labels.length).toBe(6);
      expect(model.centroids.length).toBe(2);
    });

    it('should throw error if k > n samples', () => {
      const data = [[1], [2]];
      expect(() => kfit(data, { k: 3 })).toThrow();
    });

    it('should predict cluster labels for new data', () => {
      const trainData = [
        [0, 0], [1, 0], [0, 1],
        [10, 10], [11, 10], [10, 11]
      ];

      const model = kfit(trainData, { k: 2 });

      const newData = [[0.5, 0.5], [10.5, 10.5]];
      const labels = kpredict(model, newData);

      expect(labels.length).toBe(2);
      expect(labels[0]).toBeGreaterThanOrEqual(0);
      expect(labels[0]).toBeLessThan(2);
    });

    it('should compute silhouette score', () => {
      const data = [
        [0, 0], [1, 0], [0, 1],
        [10, 10], [11, 10], [10, 11]
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const score = ksilhouette(data, labels);

      expect(score).toBeGreaterThan(0); // Well-separated clusters
      expect(score).toBeLessThanOrEqual(1);
    });

    it('should return 0 for single cluster', () => {
      const data = [[1], [2], [3]];
      const labels = [0, 0, 0];

      const score = ksilhouette(data, labels);
      expect(score).toBe(0);
    });
  });

  describe('class API (KMeans)', () => {
    it('should cluster simple 2D data', () => {
      const data = [
        [0, 0], [0.5, 0], [0, 0.5],
        [10, 10], [10.5, 10], [10, 10.5]
      ];

      const km = new KMeans({ k: 2, maxIter: 100 });
      km.fit(data);

      expect(km.labels.length).toBe(6);
      expect(km.centroids.length).toBe(2);
      expect(km.inertia).toBeGreaterThan(0);
      expect(km.iterations).toBeGreaterThan(0);
    });

    it('should converge for well-separated clusters', () => {
      const data = [
        [0, 0], [1, 0], [0, 1],
        [10, 10], [11, 10], [10, 11],
        [20, 0], [21, 0], [20, 1]
      ];

      const km = new KMeans({ k: 3 });
      km.fit(data);

      expect(km.converged).toBe(true);
      expect(km.centroids.length).toBe(3);
    });

    it('should handle 1D data', () => {
      const data = [[1], [2], [3], [10], [11], [12]];
      const km = new KMeans({ k: 2 });
      km.fit(data);

      expect(km.labels.length).toBe(6);
      expect(km.centroids.length).toBe(2);
    });

    it('should throw error if k > n samples', () => {
      const data = [[1], [2]];
      const km = new KMeans({ k: 3 });
      expect(() => km.fit(data)).toThrow();
    });

    it('should predict cluster labels for new data', () => {
      const trainData = [
        [0, 0], [1, 0], [0, 1],
        [10, 10], [11, 10], [10, 11]
      ];

      const km = new KMeans({ k: 2 });
      km.fit(trainData);

      const newData = [[0.5, 0.5], [10.5, 10.5]];
      const labels = km.predict(newData);

      expect(labels.length).toBe(2);
      expect(labels[0]).toBeGreaterThanOrEqual(0);
      expect(labels[0]).toBeLessThan(2);
    });

    it('should compute silhouette score (instance method)', () => {
      const data = [
        [0, 0], [1, 0], [0, 1],
        [10, 10], [11, 10], [10, 11]
      ];
      const labels = [0, 0, 0, 1, 1, 1];

      const km = new KMeans({ k: 2 });
      const score = km.silhouetteScore(data, labels);

      expect(score).toBeGreaterThan(0);
      expect(score).toBeLessThanOrEqual(1);
    });

    it('should return 0 for single cluster via instance method', () => {
      const data = [[1], [2], [3]];
      const labels = [0, 0, 0];

      const km = new KMeans({ k: 1 });
      const score = km.silhouetteScore(data, labels);
      expect(score).toBe(0);
    });
  });
});

import { describe, it, expect } from 'vitest';
import { normal, uniform, gamma, beta } from '../src/stats/distribution.js';
import { approxEqual } from '../src/core/math.js';

describe('distributions', () => {
  describe('normal distribution', () => {
    it('should compute PDF', () => {
      const pdf0 = normal.pdf(0, { mean: 0, sd: 1 });
      expect(approxEqual(pdf0, 0.3989, 0.001)).toBe(true);
    });

    it('should compute CDF', () => {
      const cdf0 = normal.cdf(0, { mean: 0, sd: 1 });
      expect(approxEqual(cdf0, 0.5, 0.001)).toBe(true);
    });

    it('should compute quantiles', () => {
      const q50 = normal.quantile(0.5, { mean: 0, sd: 1 });
      expect(approxEqual(q50, 0, 0.001)).toBe(true);
    });

    it('should work with non-standard parameters', () => {
      const pdf = normal.pdf(10, { mean: 10, sd: 2 });
      expect(pdf).toBeGreaterThan(0);
    });
  });

  describe('uniform distribution', () => {
    it('should compute PDF', () => {
      const pdf = uniform.pdf(0.5, { min: 0, max: 1 });
      expect(pdf).toBe(1);
      
      const pdfOut = uniform.pdf(2, { min: 0, max: 1 });
      expect(pdfOut).toBe(0);
    });

    it('should compute CDF', () => {
      const cdf = uniform.cdf(0.5, { min: 0, max: 1 });
      expect(cdf).toBe(0.5);
    });

    it('should compute quantiles', () => {
      const q = uniform.quantile(0.25, { min: 0, max: 10 });
      expect(q).toBe(2.5);
    });
  });

  describe('gamma distribution', () => {
    it('should compute PDF', () => {
      const pdf = gamma.pdf(1, { shape: 2, scale: 1 });
      expect(pdf).toBeGreaterThan(0);
    });

    it('should return 0 for negative values', () => {
      const pdf = gamma.pdf(-1, { shape: 2, scale: 1 });
      expect(pdf).toBe(0);
    });
  });

  describe('beta distribution', () => {
    it('should compute PDF', () => {
      const pdf = beta.pdf(0.5, { alpha: 2, beta: 2 });
      expect(pdf).toBeGreaterThan(0);
    });

    it('should return 0 outside [0,1]', () => {
      const pdf1 = beta.pdf(-0.1, { alpha: 2, beta: 2 });
      const pdf2 = beta.pdf(1.1, { alpha: 2, beta: 2 });
      expect(pdf1).toBe(0);
      expect(pdf2).toBe(0);
    });
  });
});

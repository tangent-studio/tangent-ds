import { describe, it, expect } from 'vitest';
import { approxEqual, guardFinite, guardPositive, guardProbability, sum, mean, variance, stddev } from '../src/core/math.js';

describe('math utilities', () => {
  describe('approxEqual', () => {
    it('should return true for approximately equal numbers', () => {
      expect(approxEqual(1.0, 1.00000001, 1e-7)).toBe(true);
      expect(approxEqual(0.1 + 0.2, 0.3, 1e-9)).toBe(true);
    });

    it('should return false for different numbers', () => {
      expect(approxEqual(1.0, 1.1)).toBe(false);
    });
  });

  describe('guard functions', () => {
    it('guardFinite should throw for non-finite values', () => {
      expect(() => guardFinite(Infinity)).toThrow();
      expect(() => guardFinite(NaN)).toThrow();
      expect(guardFinite(42)).toBe(42);
    });

    it('guardPositive should throw for non-positive values', () => {
      expect(() => guardPositive(0)).toThrow();
      expect(() => guardPositive(-1)).toThrow();
      expect(guardPositive(1)).toBe(1);
    });

    it('guardProbability should throw for values outside [0,1]', () => {
      expect(() => guardProbability(-0.1)).toThrow();
      expect(() => guardProbability(1.1)).toThrow();
      expect(guardProbability(0.5)).toBe(0.5);
    });
  });

  describe('basic statistics', () => {
    const data = [1, 2, 3, 4, 5];

    it('should compute sum', () => {
      expect(sum(data)).toBe(15);
    });

    it('should compute mean', () => {
      expect(mean(data)).toBe(3);
    });

    it('should compute variance', () => {
      const sampleVar = variance(data, true);
      expect(approxEqual(sampleVar, 2.5)).toBe(true);
    });

    it('should compute standard deviation', () => {
      const sd = stddev(data, true);
      expect(approxEqual(sd, Math.sqrt(2.5))).toBe(true);
    });
  });
});

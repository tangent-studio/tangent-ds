import { describe, it, expect } from 'vitest';
import { OneSampleTTest, TwoSampleTTest, ChiSquareTest, OneWayAnova } from '../src/stats/index.js';
import { approxEqual } from '../src/core/math.js';

describe('statistical tests', () => {
  describe('oneSampleTTest', () => {
    it('should perform one-sample t-test', () => {
      const sample = [5, 6, 7, 8, 9];
      const test = new OneSampleTTest();
      test.fit(sample, { mu: 5 });
      const result = test.summary();
      
      expect(result.mean).toBe(7);
      expect(result.df).toBe(4);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pValue).toBeGreaterThan(0);
      expect(result.pValue).toBeLessThan(1);
    });

    it('should test against different hypothesized mean', () => {
      const sample = [10, 11, 12, 13, 14];
      const test = new OneSampleTTest();
      test.fit(sample, { mu: 12 });
      const result = test.summary();
      
      expect(result.mean).toBe(12);
      expect(approxEqual(result.pValue, 1, 0.5)).toBe(true); // should be close to 1
    });

    it('should support one-sided tests', () => {
      const sample = [10, 11, 12, 13, 14];
      const test = new OneSampleTTest();
      test.fit(sample, { mu: 8, alternative: 'greater' });
      const resultLess = test.summary();
      expect(resultLess.pValue).toBeLessThan(0.5);
    });
  });

  describe('twoSampleTTest', () => {
    it('should perform two-sample t-test', () => {
      const sample1 = [1, 2, 3, 4, 5];
      const sample2 = [3, 4, 5, 6, 7];
      const test = new TwoSampleTTest();
      test.fit(sample1, sample2);
      const result = test.summary();
      
      expect(result.mean1).toBe(3);
      expect(result.mean2).toBe(5);
      expect(result.df).toBe(8);
      expect(result.statistic).toBeLessThan(0); // mean1 < mean2
    });

    it('should detect no difference when samples are similar', () => {
      const sample1 = [5, 6, 7, 8, 9];
      const sample2 = [5, 6, 7, 8, 9];
      const test = new TwoSampleTTest();
      test.fit(sample1, sample2);
      const result = test.summary();
      
      expect(approxEqual(result.statistic, 0, 0.001)).toBe(true);
    });
  });

  describe('chiSquareTest', () => {
    it('should perform chi-square goodness of fit test', () => {
      const observed = [10, 20, 30];
      const expected = [15, 20, 25];
      const test = new ChiSquareTest();
      test.fit(observed, expected);
      const result = test.summary();
      
      expect(result.df).toBe(2);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.pValue).toBeGreaterThan(0);
      expect(result.pValue).toBeLessThan(1);
    });

    it('should return perfect fit for identical distributions', () => {
      const observed = [10, 20, 30];
      const expected = [10, 20, 30];
      const test = new ChiSquareTest();
      test.fit(observed, expected);
      const result = test.summary();
      
      expect(result.statistic).toBe(0);
    });
  });

  describe('oneWayAnova', () => {
    it('should perform one-way ANOVA', () => {
      const group1 = [1, 2, 3];
      const group2 = [4, 5, 6];
      const group3 = [7, 8, 9];
      const test = new OneWayAnova();
      test.fit([group1, group2, group3]);
      const result = test.summary();
      
      expect(result.dfBetween).toBe(2);
      expect(result.dfWithin).toBe(6);
      expect(result.statistic).toBeGreaterThan(0);
      expect(result.MSbetween).toBeGreaterThan(result.MSwithin);
    });

    it('should detect no difference for identical groups', () => {
      const group1 = [5, 5, 5];
      const group2 = [5, 5, 5];
      const group3 = [5, 5, 5];
      const test = new OneWayAnova();
      test.fit([group1, group2, group3]);
      const result = test.summary();
      
      expect(result.MSbetween).toBe(0);
    });
  });
});

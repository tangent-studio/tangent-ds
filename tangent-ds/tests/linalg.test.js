import { describe, it, expect } from 'vitest';
import { toMatrix, solveLeastSquares, covarianceMatrix, svd, mmul, transpose, Matrix } from '../src/core/linalg.js';
import { approxEqual } from '../src/core/math.js';

describe('linalg', () => {
  describe('toMatrix', () => {
    it('should convert 2D array to Matrix', () => {
      const arr = [[1, 2], [3, 4]];
      const mat = toMatrix(arr);
      expect(mat instanceof Matrix).toBe(true);
      expect(mat.rows).toBe(2);
      expect(mat.columns).toBe(2);
    });

    it('should pass through Matrix objects', () => {
      const mat = new Matrix([[1, 2], [3, 4]]);
      const result = toMatrix(mat);
      expect(result).toBe(mat);
    });
  });

  describe('solveLeastSquares', () => {
    it('should solve simple linear system', () => {
      // y = 2x + 1, data: (0,1), (1,3), (2,5)
      const X = [[1, 0], [1, 1], [1, 2]];
      const y = [1, 3, 5];
      const solution = solveLeastSquares(X, y);
      const coeffs = solution.to1DArray();
      
      expect(approxEqual(coeffs[0], 1, 0.001)).toBe(true); // intercept
      expect(approxEqual(coeffs[1], 2, 0.001)).toBe(true); // slope
    });
  });

  describe('covarianceMatrix', () => {
    it('should compute covariance matrix', () => {
      const data = [
        [1, 2],
        [2, 4],
        [3, 6]
      ];
      const cov = covarianceMatrix(data);
      
      expect(cov.rows).toBe(2);
      expect(cov.columns).toBe(2);
      // Perfect positive correlation
      expect(cov.get(0, 1)).toBeGreaterThan(0);
    });
  });

  describe('matrix operations', () => {
    it('should multiply matrices', () => {
      const A = [[1, 2], [3, 4]];
      const B = [[2, 0], [1, 2]];
      const result = mmul(A, B);
      
      expect(result.get(0, 0)).toBe(4);
      expect(result.get(0, 1)).toBe(4);
      expect(result.get(1, 0)).toBe(10);
      expect(result.get(1, 1)).toBe(8);
    });

    it('should transpose matrix', () => {
      const A = [[1, 2, 3], [4, 5, 6]];
      const At = transpose(A);
      
      expect(At.rows).toBe(3);
      expect(At.columns).toBe(2);
      expect(At.get(0, 0)).toBe(1);
      expect(At.get(2, 1)).toBe(6);
    });
  });

  describe('svd', () => {
    it('should decompose matrix', () => {
      const A = [[1, 2], [3, 4], [5, 6]];
      const { U, s, V } = svd(A);
      
      expect(U.rows).toBe(3);
      expect(s.length).toBe(2);
      expect(V.rows).toBe(2);
      expect(V.columns).toBe(2);
    });
  });
});

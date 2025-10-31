import { describe, it, expect } from 'vitest';
import { normalize, toMatrix, toVector, toColumns, getColumns, filter, select } from '../src/core/table.js';

describe('table', () => {
  const sampleData = [
    { x: 1, y: 2, group: 'A' },
    { x: 2, y: 4, group: 'B' },
    { x: 3, y: 6, group: 'A' }
  ];

  describe('normalize', () => {
    it('should pass through array of objects', () => {
      const result = normalize(sampleData);
      expect(result).toBe(sampleData);
    });

    it('should throw for invalid input', () => {
      expect(() => normalize(null)).toThrow();
      expect(() => normalize(42)).toThrow();
    });
  });

  describe('toMatrix', () => {
    it('should convert specified columns to matrix', () => {
      const mat = toMatrix(sampleData, ['x', 'y']);
      expect(mat.rows).toBe(3);
      expect(mat.columns).toBe(2);
      expect(mat.get(0, 0)).toBe(1);
      expect(mat.get(2, 1)).toBe(6);
    });

    it('should auto-select numeric columns if none specified', () => {
      const mat = toMatrix(sampleData);
      expect(mat.columns).toBe(2); // x and y
    });

    it('should throw for non-numeric columns', () => {
      expect(() => toMatrix(sampleData, ['group'])).toThrow();
    });
  });

  describe('toVector', () => {
    it('should extract column as array', () => {
      const vec = toVector(sampleData, 'x');
      expect(vec).toEqual([1, 2, 3]);
    });

    it('should throw for non-numeric column', () => {
      expect(() => toVector(sampleData, 'group')).toThrow();
    });
  });

  describe('toColumns', () => {
    it('should extract multiple columns', () => {
      const cols = toColumns(sampleData, ['x', 'y']);
      expect(cols.x).toEqual([1, 2, 3]);
      expect(cols.y).toEqual([2, 4, 6]);
    });
  });

  describe('getColumns', () => {
    it('should return column names', () => {
      const names = getColumns(sampleData);
      expect(names).toEqual(['x', 'y', 'group']);
    });
  });

  describe('filter', () => {
    it('should filter rows', () => {
      const filtered = filter(sampleData, row => row.group === 'A');
      expect(filtered.length).toBe(2);
      expect(filtered[0].x).toBe(1);
    });
  });

  describe('select', () => {
    it('should select specific columns', () => {
      const selected = select(sampleData, ['x', 'group']);
      expect(Object.keys(selected[0])).toEqual(['x', 'group']);
      expect(selected[0].x).toBe(1);
    });
  });
});

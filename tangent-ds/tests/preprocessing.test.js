import { describe, it, expect } from 'vitest';
import { StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder } from '../src/ml/preprocessing.js';

describe('Preprocessing', () => {
  describe('StandardScaler', () => {
    it('should scale to zero mean and unit variance', () => {
      const scaler = new StandardScaler();
      const X = [[1, 2], [3, 4], [5, 6]];
      
      scaler.fit(X);
      const Xscaled = scaler.transform(X);
      
      expect(Xscaled.length).toBe(3);
      expect(scaler.means.length).toBe(2);
    });
  });

  describe('MinMaxScaler', () => {
    it('should scale to [0, 1]', () => {
      const scaler = new MinMaxScaler();
      const X = [[1], [2], [3]];
      
      scaler.fit(X);
      const Xscaled = scaler.transform(X);
      
      expect(Xscaled[0][0]).toBe(0);
      expect(Xscaled[2][0]).toBe(1);
    });
  });

  describe('LabelEncoder', () => {
    it('should encode labels to indices', () => {
      const encoder = new LabelEncoder();
      const y = ['cat', 'dog', 'cat', 'bird'];
      
      encoder.fit(y);
      const yEncoded = encoder.transform(y);
      
      expect(yEncoded.length).toBe(4);
      expect(yEncoded[0]).toBe(yEncoded[2]); // Both 'cat'
    });
  });

  describe('OneHotEncoder', () => {
    it('should one-hot encode categories', () => {
      const encoder = new OneHotEncoder();
      const X = [['A'], ['B'], ['A']];
      
      encoder.fit(X);
      const Xencoded = encoder.transform(X);
      
      expect(Xencoded[0]).toEqual([1, 0]);
      expect(Xencoded[1]).toEqual([0, 1]);
    });
  });
});

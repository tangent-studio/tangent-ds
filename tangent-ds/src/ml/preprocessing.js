/**
 * Preprocessing utilities
 * Scalers, encoders, and feature transformers
 */

import { mean, stddev } from '../core/math.js';

// ============= StandardScaler =============

/**
 * Standardize features by removing mean and scaling to unit variance
 */
export class StandardScaler {
  constructor() {
    this.means = null;
    this.stds = null;
    this.nFeatures = null;
  }
  
  /**
   * Compute mean and standard deviation
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {StandardScaler} this
   */
  fit(X) {
    const n = X.length;
    this.nFeatures = X[0].length;
    
    this.means = [];
    this.stds = [];
    
    for (let j = 0; j < this.nFeatures; j++) {
      const col = X.map(row => row[j]);
      this.means.push(mean(col));
      // Use population std (ddof=0) to match sklearn's StandardScaler
      this.stds.push(stddev(col, false));
    }
    
    return this;
  }
  
  /**
   * Standardize features
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Scaled features
   */
  transform(X) {
    if (this.means === null) {
      throw new Error('Scaler not fitted. Call fit() first.');
    }
    
    return X.map(row => 
      row.map((val, j) => {
        const std = this.stds[j] > 0 ? this.stds[j] : 1;
        return (val - this.means[j]) / std;
      })
    );
  }
  
  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Scaled features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }
  
  /**
   * Inverse transform (unscale)
   * @param {Array<Array<number>>} X - Scaled features
   * @returns {Array<Array<number>>} Original scale features
   */
  inverseTransform(X) {
    if (this.means === null) {
      throw new Error('Scaler not fitted. Call fit() first.');
    }
    
    return X.map(row =>
      row.map((val, j) => {
        const std = this.stds[j] > 0 ? this.stds[j] : 1;
        return val * std + this.means[j];
      })
    );
  }
}

// ============= MinMaxScaler =============

/**
 * Scale features to a given range [min, max]
 */
export class MinMaxScaler {
  constructor({ featureRange = [0, 1] } = {}) {
    this.featureRange = featureRange;
    this.dataMin = null;
    this.dataMax = null;
    this.nFeatures = null;
  }
  
  /**
   * Compute min and max for scaling
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {MinMaxScaler} this
   */
  fit(X) {
    const n = X.length;
    this.nFeatures = X[0].length;
    
    this.dataMin = [];
    this.dataMax = [];
    
    for (let j = 0; j < this.nFeatures; j++) {
      const col = X.map(row => row[j]);
      this.dataMin.push(Math.min(...col));
      this.dataMax.push(Math.max(...col));
    }
    
    return this;
  }
  
  /**
   * Scale features to range
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Scaled features
   */
  transform(X) {
    if (this.dataMin === null) {
      throw new Error('Scaler not fitted. Call fit() first.');
    }
    
    const [minRange, maxRange] = this.featureRange;
    
    return X.map(row =>
      row.map((val, j) => {
        const dataRange = this.dataMax[j] - this.dataMin[j];
        if (dataRange === 0) return minRange;
        const scaled = (val - this.dataMin[j]) / dataRange;
        return scaled * (maxRange - minRange) + minRange;
      })
    );
  }
  
  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Scaled features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }
  
  /**
   * Inverse transform
   * @param {Array<Array<number>>} X - Scaled features
   * @returns {Array<Array<number>>} Original scale features
   */
  inverseTransform(X) {
    if (this.dataMin === null) {
      throw new Error('Scaler not fitted. Call fit() first.');
    }
    
    const [minRange, maxRange] = this.featureRange;
    
    return X.map(row =>
      row.map((val, j) => {
        const dataRange = this.dataMax[j] - this.dataMin[j];
        if (dataRange === 0) return this.dataMin[j];
        const unscaled = (val - minRange) / (maxRange - minRange);
        return unscaled * dataRange + this.dataMin[j];
      })
    );
  }
}

// ============= Normalizer =============

/**
 * Normalize samples individually to unit norm
 */
export class Normalizer {
  constructor({ norm = 'l2' } = {}) {
    this.norm = norm;
  }
  
  /**
   * Normalizer doesn't need fitting
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Normalizer} this
   */
  fit(X) {
    return this;
  }
  
  /**
   * Normalize each sample
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Normalized features
   */
  transform(X) {
    return X.map(row => {
      if (this.norm === 'l2') {
        const norm = Math.sqrt(row.reduce((sum, val) => sum + val ** 2, 0));
        return norm > 0 ? row.map(val => val / norm) : row;
      } else if (this.norm === 'l1') {
        const norm = row.reduce((sum, val) => sum + Math.abs(val), 0);
        return norm > 0 ? row.map(val => val / norm) : row;
      } else if (this.norm === 'max') {
        const norm = Math.max(...row.map(Math.abs));
        return norm > 0 ? row.map(val => val / norm) : row;
      }
      throw new Error(`Unknown norm: ${this.norm}`);
    });
  }
  
  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Normalized features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }
}

// ============= LabelEncoder =============

/**
 * Encode target labels with value between 0 and n_classes-1
 */
export class LabelEncoder {
  constructor() {
    this.classes = null;
    this.classMap = null;
  }
  
  /**
   * Fit label encoder
   * @param {Array} y - Target labels
   * @returns {LabelEncoder} this
   */
  fit(y) {
    this.classes = [...new Set(y)].sort();
    this.classMap = new Map(this.classes.map((c, i) => [c, i]));
    return this;
  }
  
  /**
   * Transform labels to indices
   * @param {Array} y - Target labels
   * @returns {Array<number>} Encoded labels
   */
  transform(y) {
    if (this.classMap === null) {
      throw new Error('LabelEncoder not fitted. Call fit() first.');
    }
    
    return y.map(label => {
      if (!this.classMap.has(label)) {
        throw new Error(`Unknown label: ${label}`);
      }
      return this.classMap.get(label);
    });
  }
  
  /**
   * Fit and transform in one step
   * @param {Array} y - Target labels
   * @returns {Array<number>} Encoded labels
   */
  fitTransform(y) {
    return this.fit(y).transform(y);
  }
  
  /**
   * Transform indices back to original labels
   * @param {Array<number>} y - Encoded labels
   * @returns {Array} Original labels
   */
  inverseTransform(y) {
    if (this.classes === null) {
      throw new Error('LabelEncoder not fitted. Call fit() first.');
    }
    
    return y.map(idx => {
      if (idx < 0 || idx >= this.classes.length) {
        throw new Error(`Invalid index: ${idx}`);
      }
      return this.classes[idx];
    });
  }
}

// ============= OneHotEncoder =============

/**
 * Encode categorical features as one-hot numeric array
 */
export class OneHotEncoder {
  constructor() {
    this.categories = null;
    this.nFeatures = null;
  }
  
  /**
   * Fit encoder by discovering categories
   * @param {Array<Array>} X - Categorical features
   * @returns {OneHotEncoder} this
   */
  fit(X) {
    this.nFeatures = X[0].length;
    this.categories = [];
    
    for (let j = 0; j < this.nFeatures; j++) {
      const col = X.map(row => row[j]);
      const unique = [...new Set(col)].sort();
      this.categories.push(unique);
    }
    
    return this;
  }
  
  /**
   * Transform to one-hot encoding
   * @param {Array<Array>} X - Categorical features
   * @returns {Array<Array<number>>} One-hot encoded features
   */
  transform(X) {
    if (this.categories === null) {
      throw new Error('OneHotEncoder not fitted. Call fit() first.');
    }
    
    return X.map(row => {
      const encoded = [];
      
      for (let j = 0; j < this.nFeatures; j++) {
        const value = row[j];
        const categories = this.categories[j];
        const idx = categories.indexOf(value);
        
        if (idx === -1) {
          throw new Error(`Unknown category: ${value} in feature ${j}`);
        }
        
        // Add one-hot vector for this feature
        for (let k = 0; k < categories.length; k++) {
          encoded.push(k === idx ? 1 : 0);
        }
      }
      
      return encoded;
    });
  }
  
  /**
   * Fit and transform in one step
   * @param {Array<Array>} X - Categorical features
   * @returns {Array<Array<number>>} One-hot encoded features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }
  
  /**
   * Get feature names after one-hot encoding
   * @returns {Array<string>} Feature names
   */
  getFeatureNames() {
    if (this.categories === null) {
      throw new Error('OneHotEncoder not fitted. Call fit() first.');
    }
    
    const names = [];
    for (let j = 0; j < this.nFeatures; j++) {
      for (const category of this.categories[j]) {
        names.push(`feature${j}_${category}`);
      }
    }
    return names;
  }
}

// ============= PolynomialFeatures =============

/**
 * Generate polynomial and interaction features
 */
export class PolynomialFeatures {
  constructor({ degree = 2, includeBias = false } = {}) {
    this.degree = degree;
    this.includeBias = includeBias;
    this.nInputFeatures = null;
    this.nOutputFeatures = null;
  }
  
  /**
   * Fit by determining input/output dimensions
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {PolynomialFeatures} this
   */
  fit(X) {
    this.nInputFeatures = X[0].length;
    
    // Calculate number of output features
    // For degree d and n features: (n+d choose d)
    let count = this.includeBias ? 1 : 0;
    
    // Add original features and powers
    for (let d = 1; d <= this.degree; d++) {
      count += this._countTerms(this.nInputFeatures, d);
    }
    
    this.nOutputFeatures = count;
    return this;
  }
  
  /**
   * Count number of polynomial terms
   * @private
   */
  _countTerms(n, d) {
    // Combinations with replacement: (n+d-1 choose d)
    if (d === 0) return 1;
    if (d === 1) return n;
    let result = 1;
    for (let i = 0; i < d; i++) {
      result *= (n + d - 1 - i);
      result /= (i + 1);
    }
    return Math.round(result);
  }
  
  /**
   * Transform to polynomial features
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Polynomial features
   */
  transform(X) {
    return X.map(row => {
      const features = [];
      
      if (this.includeBias) {
        features.push(1);
      }
      
      // Generate polynomial features up to degree
      this._generatePolynomial(row, this.degree, features);
      
      return features;
    });
  }
  
  /**
   * Generate polynomial features recursively
   * @private
   */
  _generatePolynomial(row, maxDegree, features) {
    const n = row.length;
    
    // Degree 1: original features
    for (let i = 0; i < n; i++) {
      features.push(row[i]);
    }
    
    if (maxDegree === 1) return;
    
    // Degree 2 and higher
    for (let d = 2; d <= maxDegree; d++) {
      this._generateDegree(row, d, features);
    }
  }
  
  /**
   * Generate features of specific degree
   * @private
   */
  _generateDegree(row, degree, features) {
    const n = row.length;
    
    const generateCombinations = (start, current, power) => {
      if (power === 0) {
        features.push(current);
        return;
      }
      
      for (let i = start; i < n; i++) {
        generateCombinations(i, current * row[i], power - 1);
      }
    };
    
    generateCombinations(0, 1, degree);
  }
  
  /**
   * Fit and transform in one step
   * @param {Array<Array<number>>} X - Feature matrix
   * @returns {Array<Array<number>>} Polynomial features
   */
  fitTransform(X) {
    return this.fit(X).transform(X);
  }
}

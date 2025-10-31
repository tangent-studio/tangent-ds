/**
 * Statistical distributions
 * Provides PDF, CDF, and quantile functions
 */

import { 
  cumulativeStdNormalProbability,
  probit,
  gamma as gammaFunc
} from 'simple-statistics';
import { guardPositive, guardProbability } from '../core/math.js';

// ============= Normal Distribution =============

/**
 * Standard normal PDF
 * @param {number} x - Value
 * @returns {number} Probability density
 */
function stdNormalPDF(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Normal distribution
 */
export const normal = {
  /**
   * Probability density function
   * @param {number} x - Value
   * @param {Object} params - {mean, sd}
   * @returns {number} Probability density
   */
  pdf(x, { mean = 0, sd = 1 } = {}) {
    guardPositive(sd, 'sd');
    const z = (x - mean) / sd;
    return stdNormalPDF(z) / sd;
  },

  /**
   * Cumulative distribution function
   * @param {number} x - Value
   * @param {Object} params - {mean, sd}
   * @returns {number} Cumulative probability
   */
  cdf(x, { mean = 0, sd = 1 } = {}) {
    guardPositive(sd, 'sd');
    const z = (x - mean) / sd;
    return cumulativeStdNormalProbability(z);
  },

  /**
   * Quantile function (inverse CDF)
   * @param {number} p - Probability
   * @param {Object} params - {mean, sd}
   * @returns {number} Quantile
   */
  quantile(p, { mean = 0, sd = 1 } = {}) {
    guardProbability(p, 'p');
    guardPositive(sd, 'sd');
    return mean + sd * probit(p);
  }
};

// ============= Uniform Distribution =============

export const uniform = {
  /**
   * Probability density function
   * @param {number} x - Value
   * @param {Object} params - {min, max}
   * @returns {number} Probability density
   */
  pdf(x, { min = 0, max = 1 } = {}) {
    if (max <= min) {
      throw new Error('max must be greater than min');
    }
    return (x >= min && x <= max) ? 1 / (max - min) : 0;
  },

  /**
   * Cumulative distribution function
   * @param {number} x - Value
   * @param {Object} params - {min, max}
   * @returns {number} Cumulative probability
   */
  cdf(x, { min = 0, max = 1 } = {}) {
    if (max <= min) {
      throw new Error('max must be greater than min');
    }
    if (x < min) return 0;
    if (x > max) return 1;
    return (x - min) / (max - min);
  },

  /**
   * Quantile function
   * @param {number} p - Probability
   * @param {Object} params - {min, max}
   * @returns {number} Quantile
   */
  quantile(p, { min = 0, max = 1 } = {}) {
    guardProbability(p, 'p');
    if (max <= min) {
      throw new Error('max must be greater than min');
    }
    return min + p * (max - min);
  }
};

// ============= Gamma Distribution =============

/**
 * Log gamma function
 */
function logGamma(x) {
  return Math.log(gammaFunc(x));
}

export const gamma = {
  /**
   * Probability density function
   * @param {number} x - Value (x > 0)
   * @param {Object} params - {shape, scale}
   * @returns {number} Probability density
   */
  pdf(x, { shape = 1, scale = 1 } = {}) {
    guardPositive(shape, 'shape');
    guardPositive(scale, 'scale');
    
    if (x <= 0) return 0;
    
    const logPdf = (shape - 1) * Math.log(x) - x / scale - shape * Math.log(scale) - logGamma(shape);
    return Math.exp(logPdf);
  },

  /**
   * Cumulative distribution function (approximation)
   * @param {number} x - Value
   * @param {Object} params - {shape, scale}
   * @returns {number} Cumulative probability
   */
  cdf(x, { shape = 1, scale = 1 } = {}) {
    guardPositive(shape, 'shape');
    guardPositive(scale, 'scale');
    
    if (x <= 0) return 0;
    
    // Use regularized incomplete gamma function approximation
    return incompleteGamma(shape, x / scale) / gammaFunc(shape);
  },

  /**
   * Quantile function (numerical approximation)
   * @param {number} p - Probability
   * @param {Object} params - {shape, scale}
   * @returns {number} Quantile
   */
  quantile(p, { shape = 1, scale = 1 } = {}) {
    guardProbability(p, 'p');
    guardPositive(shape, 'shape');
    guardPositive(scale, 'scale');
    
    // Simple numerical inversion using bisection
    let low = 0;
    let high = shape * scale * 10; // reasonable upper bound
    const tolerance = 1e-8;
    
    while (high - low > tolerance) {
      const mid = (low + high) / 2;
      const cdfVal = gamma.cdf(mid, { shape, scale });
      if (cdfVal < p) {
        low = mid;
      } else {
        high = mid;
      }
    }
    
    return (low + high) / 2;
  }
};

// Helper: incomplete gamma function (lower)
function incompleteGamma(s, x, maxIter = 100) {
  if (x === 0) return 0;
  
  // Series expansion
  let sum = 1.0;
  let term = 1.0;
  
  for (let n = 1; n < maxIter; n++) {
    term *= x / (s + n);
    sum += term;
    if (Math.abs(term) < 1e-10) break;
  }
  
  return Math.exp(-x + s * Math.log(x)) * sum;
}

// ============= Beta Distribution =============

/**
 * Beta function B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
 */
function betaFunc(a, b) {
  return Math.exp(logGamma(a) + logGamma(b) - logGamma(a + b));
}

export const beta = {
  /**
   * Probability density function
   * @param {number} x - Value (0 <= x <= 1)
   * @param {Object} params - {alpha, beta}
   * @returns {number} Probability density
   */
  pdf(x, { alpha = 1, beta: betaParam = 1 } = {}) {
    guardPositive(alpha, 'alpha');
    guardPositive(betaParam, 'beta');
    
    if (x < 0 || x > 1) return 0;
    if (x === 0) return alpha === 1 ? 1 / betaFunc(alpha, betaParam) : 0;
    if (x === 1) return betaParam === 1 ? 1 / betaFunc(alpha, betaParam) : 0;
    
    const logPdf = (alpha - 1) * Math.log(x) + (betaParam - 1) * Math.log(1 - x) - Math.log(betaFunc(alpha, betaParam));
    return Math.exp(logPdf);
  },

  /**
   * Cumulative distribution function (approximation)
   * @param {number} x - Value
   * @param {Object} params - {alpha, beta}
   * @returns {number} Cumulative probability
   */
  cdf(x, { alpha = 1, beta: betaParam = 1 } = {}) {
    guardPositive(alpha, 'alpha');
    guardPositive(betaParam, 'beta');
    
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    
    // Regularized incomplete beta function approximation
    return incompleteBeta(x, alpha, betaParam) / betaFunc(alpha, betaParam);
  },

  /**
   * Quantile function (numerical approximation)
   * @param {number} p - Probability
   * @param {Object} params - {alpha, beta}
   * @returns {number} Quantile
   */
  quantile(p, { alpha = 1, beta: betaParam = 1 } = {}) {
    guardProbability(p, 'p');
    guardPositive(alpha, 'alpha');
    guardPositive(betaParam, 'beta');
    
    // Numerical inversion using bisection
    let low = 0;
    let high = 1;
    const tolerance = 1e-8;
    
    while (high - low > tolerance) {
      const mid = (low + high) / 2;
      const cdfVal = beta.cdf(mid, { alpha, beta: betaParam });
      if (cdfVal < p) {
        low = mid;
      } else {
        high = mid;
      }
    }
    
    return (low + high) / 2;
  }
};

// Helper: incomplete beta function
function incompleteBeta(x, a, b, maxIter = 100) {
  if (x === 0) return 0;
  if (x === 1) return betaFunc(a, b);
  
  // Series expansion
  const logBeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - logBeta) / a;
  
  let sum = 1.0;
  let term = 1.0;
  
  for (let n = 0; n < maxIter; n++) {
    term *= (a + b + n) * x / (a + 1 + n);
    sum += term;
    if (Math.abs(term) < 1e-10) break;
  }
  
  return front * sum;
}

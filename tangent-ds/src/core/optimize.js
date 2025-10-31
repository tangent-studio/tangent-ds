/**
 * Optimization algorithms for gradient-based minimization
 * Supports: Gradient Descent, Adam, Momentum, RMSProp
 */

import { mean } from './math.js';

/**
 * Base Optimizer class
 */
class Optimizer {
  constructor(options = {}) {
    this.learningRate = options.learningRate || 0.01;
    this.maxIter = options.maxIter || 1000;
    this.tol = options.tol || 1e-6;
    this.verbose = options.verbose || false;
  }

  /**
   * Minimize a loss function
   * @param {Function} lossFn - Function that returns {loss, gradient}
   * @param {Array<number>} x0 - Initial parameters
   * @param {Object} options - Additional options
   * @returns {Object} {x, history}
   */
  minimize(lossFn, x0, options = {}) {
    throw new Error('minimize() must be implemented by subclass');
  }

  /**
   * Check convergence based on gradient norm
   * @param {Array<number>} gradient
   * @returns {boolean}
   */
  checkConvergence(gradient) {
    const norm = Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0));
    return norm < this.tol;
  }
}

/**
 * Gradient Descent Optimizer (Batch and Stochastic)
 */
export class GradientDescent extends Optimizer {
  constructor(options = {}) {
    super(options);
    this.stochastic = options.stochastic || false;
    this.batchSize = options.batchSize || 32;
    this.lineSearch = options.lineSearch || false;
  }

  minimize(lossFn, x0, options = {}) {
    const maxIter = options.maxIter || this.maxIter;
    const tol = options.tol || this.tol;
    
    let x = [...x0];
    const history = {
      loss: [],
      gradNorm: [],
      learningRate: []
    };

    for (let iter = 0; iter < maxIter; iter++) {
      // Compute loss and gradient
      const { loss, gradient } = lossFn(x);
      
      // Store history
      history.loss.push(loss);
      const gradNorm = Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0));
      history.gradNorm.push(gradNorm);

      // Check convergence
      if (gradNorm < tol) {
        if (this.verbose) {
          console.log(`Converged at iteration ${iter}, loss: ${loss.toFixed(6)}`);
        }
        break;
      }

      // Line search or fixed learning rate
      let lr = this.learningRate;
      if (this.lineSearch) {
        lr = this.backtrackingLineSearch(lossFn, x, gradient, loss);
      }
      history.learningRate.push(lr);

      // Update parameters
      for (let i = 0; i < x.length; i++) {
        x[i] -= lr * gradient[i];
      }

      if (this.verbose && iter % 100 === 0) {
        console.log(`Iter ${iter}: loss=${loss.toFixed(6)}, grad_norm=${gradNorm.toFixed(6)}`);
      }
    }

    return { x, history };
  }

  /**
   * Backtracking line search
   * @param {Function} lossFn
   * @param {Array<number>} x
   * @param {Array<number>} gradient
   * @param {number} currentLoss
   * @returns {number} Step size
   */
  backtrackingLineSearch(lossFn, x, gradient, currentLoss) {
    const alpha = 0.3; // Armijo condition constant
    const beta = 0.8;  // Reduction factor
    let t = 1.0;
    
    // Compute gradient norm squared
    const gradNormSq = gradient.reduce((sum, g) => sum + g * g, 0);
    
    // Try decreasing step sizes
    for (let i = 0; i < 20; i++) {
      // Compute new point
      const xNew = x.map((xi, j) => xi - t * gradient[j]);
      
      // Evaluate loss at new point
      const { loss: newLoss } = lossFn(xNew);
      
      // Check Armijo condition
      if (newLoss <= currentLoss - alpha * t * gradNormSq) {
        return t;
      }
      
      t *= beta;
    }
    
    return t;
  }
}

/**
 * Momentum Optimizer
 */
export class MomentumOptimizer extends Optimizer {
  constructor(options = {}) {
    super(options);
    this.momentum = options.momentum || 0.9;
  }

  minimize(lossFn, x0, options = {}) {
    const maxIter = options.maxIter || this.maxIter;
    const tol = options.tol || this.tol;
    
    let x = [...x0];
    let velocity = new Array(x.length).fill(0);
    
    const history = {
      loss: [],
      gradNorm: []
    };

    for (let iter = 0; iter < maxIter; iter++) {
      const { loss, gradient } = lossFn(x);
      
      history.loss.push(loss);
      const gradNorm = Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0));
      history.gradNorm.push(gradNorm);

      if (gradNorm < tol) {
        if (this.verbose) {
          console.log(`Converged at iteration ${iter}, loss: ${loss.toFixed(6)}`);
        }
        break;
      }

      // Update velocity and parameters
      for (let i = 0; i < x.length; i++) {
        velocity[i] = this.momentum * velocity[i] + this.learningRate * gradient[i];
        x[i] -= velocity[i];
      }

      if (this.verbose && iter % 100 === 0) {
        console.log(`Iter ${iter}: loss=${loss.toFixed(6)}, grad_norm=${gradNorm.toFixed(6)}`);
      }
    }

    return { x, history };
  }
}

/**
 * RMSProp Optimizer
 */
export class RMSProp extends Optimizer {
  constructor(options = {}) {
    super(options);
    this.decay = options.decay || 0.9;
    this.epsilon = options.epsilon || 1e-8;
  }

  minimize(lossFn, x0, options = {}) {
    const maxIter = options.maxIter || this.maxIter;
    const tol = options.tol || this.tol;
    
    let x = [...x0];
    let cache = new Array(x.length).fill(0);
    
    const history = {
      loss: [],
      gradNorm: []
    };

    for (let iter = 0; iter < maxIter; iter++) {
      const { loss, gradient } = lossFn(x);
      
      history.loss.push(loss);
      const gradNorm = Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0));
      history.gradNorm.push(gradNorm);

      if (gradNorm < tol) {
        if (this.verbose) {
          console.log(`Converged at iteration ${iter}, loss: ${loss.toFixed(6)}`);
        }
        break;
      }

      // Update cache and parameters
      for (let i = 0; i < x.length; i++) {
        cache[i] = this.decay * cache[i] + (1 - this.decay) * gradient[i] * gradient[i];
        x[i] -= this.learningRate * gradient[i] / (Math.sqrt(cache[i]) + this.epsilon);
      }

      if (this.verbose && iter % 100 === 0) {
        console.log(`Iter ${iter}: loss=${loss.toFixed(6)}, grad_norm=${gradNorm.toFixed(6)}`);
      }
    }

    return { x, history };
  }
}

/**
 * Adam Optimizer (Adaptive Moment Estimation)
 */
export class AdamOptimizer extends Optimizer {
  constructor(options = {}) {
    super(options);
    this.beta1 = options.beta1 || 0.9;
    this.beta2 = options.beta2 || 0.999;
    this.epsilon = options.epsilon || 1e-8;
  }

  minimize(lossFn, x0, options = {}) {
    const maxIter = options.maxIter || this.maxIter;
    const tol = options.tol || this.tol;
    
    let x = [...x0];
    let m = new Array(x.length).fill(0); // First moment estimate
    let v = new Array(x.length).fill(0); // Second moment estimate
    
    const history = {
      loss: [],
      gradNorm: []
    };

    for (let iter = 0; iter < maxIter; iter++) {
      const { loss, gradient } = lossFn(x);
      
      history.loss.push(loss);
      const gradNorm = Math.sqrt(gradient.reduce((sum, g) => sum + g * g, 0));
      history.gradNorm.push(gradNorm);

      if (gradNorm < tol) {
        if (this.verbose) {
          console.log(`Converged at iteration ${iter}, loss: ${loss.toFixed(6)}`);
        }
        break;
      }

      // Update biased first moment estimate
      for (let i = 0; i < x.length; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * gradient[i];
      }

      // Update biased second raw moment estimate
      for (let i = 0; i < x.length; i++) {
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * gradient[i] * gradient[i];
      }

      // Compute bias-corrected moment estimates
      const t = iter + 1;
      const mHat = m.map(mi => mi / (1 - Math.pow(this.beta1, t)));
      const vHat = v.map(vi => vi / (1 - Math.pow(this.beta2, t)));

      // Update parameters
      for (let i = 0; i < x.length; i++) {
        x[i] -= this.learningRate * mHat[i] / (Math.sqrt(vHat[i]) + this.epsilon);
      }

      if (this.verbose && iter % 100 === 0) {
        console.log(`Iter ${iter}: loss=${loss.toFixed(6)}, grad_norm=${gradNorm.toFixed(6)}`);
      }
    }

    return { x, history };
  }
}

/**
 * Convenience function to create optimizer by name
 * @param {string} name - Optimizer name
 * @param {Object} options - Optimizer options
 * @returns {Optimizer} Optimizer instance
 */
export function createOptimizer(name, options = {}) {
  const optimizers = {
    'gd': GradientDescent,
    'gradient_descent': GradientDescent,
    'sgd': GradientDescent,
    'momentum': MomentumOptimizer,
    'rmsprop': RMSProp,
    'adam': AdamOptimizer
  };

  const OptimizerClass = optimizers[name.toLowerCase()];
  if (!OptimizerClass) {
    throw new Error(`Unknown optimizer: ${name}. Available: ${Object.keys(optimizers).join(', ')}`);
  }

  return new OptimizerClass(options);
}

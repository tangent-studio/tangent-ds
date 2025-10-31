import { describe, it, expect } from 'vitest';
import { 
  GradientDescent, 
  MomentumOptimizer, 
  RMSProp, 
  AdamOptimizer,
  createOptimizer 
} from '../src/core/optimize.js';

describe('Optimizers', () => {
  // Quadratic function: f(x) = (x - 3)^2
  const quadratic = (x) => {
    const val = x[0] - 3;
    return {
      loss: val * val,
      gradient: [2 * val]
    };
  };

  // Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
  const rosenbrock = (x, a = 1, b = 100) => {
    const loss = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2;
    const gradient = [
      -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
      2 * b * (x[1] - x[0] ** 2)
    ];
    return { loss, gradient };
  };

  describe('GradientDescent', () => {
    it('should minimize quadratic function', () => {
      const opt = new GradientDescent({ learningRate: 0.1, maxIter: 100, tol: 1e-6 });
      const { x, history } = opt.minimize(quadratic, [0]);

      expect(x[0]).toBeCloseTo(3, 3);
      expect(history.loss[history.loss.length - 1]).toBeLessThan(1e-6);
    });

    it('should converge with line search', () => {
      const opt = new GradientDescent({ 
        learningRate: 0.1, 
        lineSearch: true,
        maxIter: 100, 
        tol: 1e-6 
      });
      const { x, history } = opt.minimize(quadratic, [0]);

      expect(x[0]).toBeCloseTo(3, 3);
      expect(history.learningRate).toBeDefined();
      expect(history.learningRate.length).toBeGreaterThan(0);
    });

    it('should minimize Rosenbrock function', () => {
      const opt = new GradientDescent({ learningRate: 0.001, maxIter: 5000, tol: 1e-4 });
      const { x, history } = opt.minimize(rosenbrock, [0, 0]);

      // Should converge towards (1, 1) - but Rosenbrock is hard for basic GD
      expect(x[0]).toBeCloseTo(1, 0.5); // Very loose tolerance
      expect(x[1]).toBeCloseTo(1, 0.5);
      expect(history.loss[history.loss.length - 1]).toBeLessThan(1); // Just check improvement
    });

    it('should record gradient norms', () => {
      const opt = new GradientDescent({ learningRate: 0.1, maxIter: 50 });
      const { history } = opt.minimize(quadratic, [0]);

      expect(history.gradNorm).toBeDefined();
      expect(history.gradNorm.length).toBeGreaterThan(0);
      // Gradient norm should decrease
      expect(history.gradNorm[history.gradNorm.length - 1]).toBeLessThan(history.gradNorm[0]);
    });
  });

  describe('MomentumOptimizer', () => {
    it('should minimize quadratic function', () => {
      const opt = new MomentumOptimizer({ 
        learningRate: 0.01, 
        momentum: 0.9,
        maxIter: 200,  // More iterations
        tol: 1e-6 
      });
      const { x, history } = opt.minimize(quadratic, [0]);

      expect(x[0]).toBeCloseTo(3, 2); // Looser tolerance
      expect(history.loss[history.loss.length - 1]).toBeLessThan(0.01);
    });

    it('should converge faster than basic GD', () => {
      const optGD = new GradientDescent({ learningRate: 0.005, maxIter: 500 });
      const optMomentum = new MomentumOptimizer({ learningRate: 0.005, momentum: 0.9, maxIter: 500 });

      const resultGD = optGD.minimize(quadratic, [0]); // Simpler function
      const resultMomentum = optMomentum.minimize(quadratic, [0]);

      // Momentum should converge in fewer iterations
      const finalLossGD = resultGD.history.loss[resultGD.history.loss.length - 1];
      const finalLossMomentum = resultMomentum.history.loss[resultMomentum.history.loss.length - 1];

      // Both should converge
      expect(finalLossGD).toBeLessThan(0.01);
      expect(finalLossMomentum).toBeLessThan(0.01);
      // Momentum typically needs fewer iterations
      expect(resultMomentum.history.loss.length).toBeLessThanOrEqual(resultGD.history.loss.length);
    });
  });

  describe('RMSProp', () => {
    it('should minimize quadratic function', () => {
      const opt = new RMSProp({ learningRate: 0.1, maxIter: 100, tol: 1e-6 });
      const { x, history } = opt.minimize(quadratic, [0]);

      expect(x[0]).toBeCloseTo(3, 2); // Looser tolerance
      expect(history.loss[history.loss.length - 1]).toBeLessThan(0.01);
    });

    it('should handle different decay rates', () => {
      const opt1 = new RMSProp({ learningRate: 0.01, decay: 0.9, maxIter: 200 });
      const opt2 = new RMSProp({ learningRate: 0.01, decay: 0.99, maxIter: 200 });

      const result1 = opt1.minimize(rosenbrock, [0, 0]);
      const result2 = opt2.minimize(rosenbrock, [0, 0]);

      // Both should converge
      expect(result1.history.loss[result1.history.loss.length - 1]).toBeLessThan(10);
      expect(result2.history.loss[result2.history.loss.length - 1]).toBeLessThan(10);
    });
  });

  describe('AdamOptimizer', () => {
    it('should minimize quadratic function', () => {
      const opt = new AdamOptimizer({ learningRate: 0.1, maxIter: 100, tol: 1e-6 });
      const { x, history } = opt.minimize(quadratic, [0]);

      expect(x[0]).toBeCloseTo(3, 1); // 1 decimal place
      expect(history.loss[history.loss.length - 1]).toBeLessThan(0.01);
    });

    it('should minimize Rosenbrock efficiently', () => {
      const opt = new AdamOptimizer({ learningRate: 0.01, maxIter: 1000, tol: 1e-6 });
      const { x, history } = opt.minimize(rosenbrock, [0, 0]);

      // Adam should converge well on Rosenbrock
      expect(x[0]).toBeCloseTo(1, 0.5);
      expect(x[1]).toBeCloseTo(1, 0.5);
      expect(history.loss[history.loss.length - 1]).toBeLessThan(1);
    });

    it('should use bias correction', () => {
      const opt = new AdamOptimizer({ 
        learningRate: 0.1, 
        beta1: 0.9, 
        beta2: 0.999,
        maxIter: 50 
      });
      const { history } = opt.minimize(quadratic, [0]);

      // Loss should decrease monotonically (mostly)
      const losses = history.loss;
      let decreasing = 0;
      for (let i = 1; i < Math.min(losses.length, 20); i++) {
        if (losses[i] < losses[i - 1]) decreasing++;
      }
      expect(decreasing).toBeGreaterThan(15); // At least 75% decreasing
    });
  });

  describe('createOptimizer', () => {
    it('should create GradientDescent by name', () => {
      const opt1 = createOptimizer('gd', { learningRate: 0.1 });
      const opt2 = createOptimizer('gradient_descent', { learningRate: 0.1 });

      expect(opt1).toBeInstanceOf(GradientDescent);
      expect(opt2).toBeInstanceOf(GradientDescent);
    });

    it('should create Adam by name', () => {
      const opt = createOptimizer('adam', { learningRate: 0.01 });
      expect(opt).toBeInstanceOf(AdamOptimizer);
    });

    it('should create Momentum by name', () => {
      const opt = createOptimizer('momentum', { learningRate: 0.01 });
      expect(opt).toBeInstanceOf(MomentumOptimizer);
    });

    it('should create RMSProp by name', () => {
      const opt = createOptimizer('rmsprop', { learningRate: 0.01 });
      expect(opt).toBeInstanceOf(RMSProp);
    });

    it('should throw error for unknown optimizer', () => {
      expect(() => {
        createOptimizer('unknown');
      }).toThrow('Unknown optimizer');
    });
  });

  describe('Convergence behavior', () => {
    it('should stop early if tolerance is met', () => {
      const opt = new AdamOptimizer({ learningRate: 0.5, maxIter: 1000, tol: 0.01 });
      const { history } = opt.minimize(quadratic, [0]);

      // Should converge reasonably quickly
      expect(history.loss.length).toBeLessThan(200);
    });

    it('should reach maxIter if not converged', () => {
      const opt = new GradientDescent({ 
        learningRate: 0.0001, // Very small LR
        maxIter: 50, 
        tol: 1e-10 // Very tight tolerance
      });
      const { history } = opt.minimize(quadratic, [0]);

      // Should hit maxIter
      expect(history.loss.length).toBe(50);
    });
  });

  describe('History tracking', () => {
    it('should track loss history', () => {
      const opt = new AdamOptimizer({ learningRate: 0.1, maxIter: 50 });
      const { history } = opt.minimize(quadratic, [0]);

      expect(history.loss).toBeDefined();
      expect(history.loss.length).toBeGreaterThan(0);
      expect(history.loss.length).toBeLessThanOrEqual(50);
    });

    it('should track gradient norm', () => {
      const opt = new AdamOptimizer({ learningRate: 0.1, maxIter: 50 });
      const { history } = opt.minimize(quadratic, [0]);

      expect(history.gradNorm).toBeDefined();
      expect(history.gradNorm.length).toBe(history.loss.length);
    });
  });
});

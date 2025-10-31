import { describe, it, expect } from 'vitest';
import { createNetwork, fit, predict, evaluate } from '../src/ml/mlp.js';
import { MLPRegressor } from '../src/ml/index.js';

describe('multilayer perceptron', () => {
  describe('createNetwork', () => {
    it('should create network with correct structure', () => {
      const layers = createNetwork([2, 3, 1], 'relu');
      
      expect(layers.length).toBe(2); // 2 layers: input->hidden, hidden->output
      expect(layers[0].weights.length).toBe(3); // 3 hidden neurons
      expect(layers[0].weights[0].length).toBe(2); // 2 inputs
      expect(layers[1].weights.length).toBe(1); // 1 output
      expect(layers[1].weights[0].length).toBe(3); // from 3 hidden
    });

    it('should initialize with different activations', () => {
      // Use 3-layer networks to test hidden layer activations
      const layers1 = createNetwork([2, 3, 2], 'sigmoid');
      const layers2 = createNetwork([2, 3, 2], 'relu');
      const layers3 = createNetwork([2, 3, 2], 'tanh');
      
      // Hidden layer should have specified activation
      expect(layers1[0].activation).toBe('sigmoid');
      expect(layers2[0].activation).toBe('relu');
      expect(layers3[0].activation).toBe('tanh');
      
      // Output layer should always be linear
      expect(layers1[1].activation).toBe('linear');
      expect(layers2[1].activation).toBe('linear');
      expect(layers3[1].activation).toBe('linear');
    });

    it('should throw error for invalid network', () => {
      expect(() => createNetwork([2], 'relu')).toThrow();
      expect(() => createNetwork([2, 3], 'invalid')).toThrow();
    });
  });

  describe('fit', () => {
    it('should train on simple linear data', () => {
      // y = 2x
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 6, 8];
      
      const model = fit(X, y, {
        layerSizes: [1, 4, 1],
        epochs: 100,
        learningRate: 0.01
      });
      
      expect(model.layers.length).toBe(2);
      expect(model.losses.length).toBe(100);
      // Loss should decrease
      expect(model.losses[model.losses.length - 1]).toBeLessThan(model.losses[0]);
    });

    it('should train on XOR-like data', () => {
      // Simple non-linear pattern
      const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
      const y = [0, 1, 1, 0];
      
      const model = fit(X, y, {
        layerSizes: [2, 4, 1],
        activation: 'relu',
        epochs: 200,
        learningRate: 0.1,
        batchSize: 4
      });
      
      expect(model.epochs).toBe(200);
      expect(model.losses.length).toBe(200);
    });

    it('should handle 2D output', () => {
      const X = [[1], [2], [3]];
      const y = [[1, 2], [2, 4], [3, 6]];
      
      const model = fit(X, y, {
        layerSizes: [1, 3, 2],
        epochs: 50
      });
      
      expect(model.layerSizes).toEqual([1, 3, 2]);
    });
  });

  describe('predict', () => {
    it('should make predictions', () => {
      const X = [[1], [2], [3]];
      const y = [2, 4, 6];
      
      const model = fit(X, y, {
        layerSizes: [1, 4, 1],
        epochs: 100,
        learningRate: 0.01
      });
      
      const predictions = predict(model, [[4]]);
      
      expect(predictions.length).toBe(1);
      expect(predictions[0].length).toBe(1);
      expect(typeof predictions[0][0]).toBe('number');
    });

    it('should handle multiple predictions', () => {
      const X = [[1], [2]];
      const y = [1, 2];
      
      const model = fit(X, y, {
        layerSizes: [1, 2, 1],
        epochs: 50
      });
      
      const predictions = predict(model, [[3], [4]]);
      
      expect(predictions.length).toBe(2);
    });
  });

  describe('evaluate', () => {
    it('should evaluate model performance', () => {
      const X = [[1], [2], [3], [4]];
      const y = [2, 4, 6, 8];
      
      const model = fit(X, y, {
        layerSizes: [1, 4, 1],
        epochs: 100,
        learningRate: 0.01
      });
      
      const metrics = evaluate(model, X, y);
      
      expect(metrics.mse).toBeGreaterThanOrEqual(0);
      expect(metrics.mae).toBeGreaterThanOrEqual(0);
    });
  });
});

describe('MLPRegressor (class API)', () => {
  it('should train and predict via class wrapper', () => {
    const X = [[1], [2], [3], [4]];
    const y = [2, 4, 6, 8];

    const mlpReg = new MLPRegressor({
      layerSizes: [1, 4, 1],
      epochs: 50,
      learningRate: 0.05,
      batchSize: 2
    });

    mlpReg.fit(X, y);
    const preds = mlpReg.predict([[5]]);

    expect(preds.length).toBe(1);
    expect(preds[0].length).toBe(1);
    expect(typeof preds[0][0]).toBe('number');
  });

  it('should expose training summary', () => {
    const X = [[1], [2]];
    const y = [2, 4];

    const mlpReg = new MLPRegressor({ layerSizes: [1, 3, 1], epochs: 20 });
    mlpReg.fit(X, y);

    const summary = mlpReg.summary();
    expect(summary.epochs).toBe(20);
    expect(summary.layerSizes).toEqual([1, 3, 1]);
    expect(summary.finalLoss).toBeDefined();
  });
});

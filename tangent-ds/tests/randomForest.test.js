import { describe, it, expect } from 'vitest';
import { RandomForestClassifier, RandomForestRegressor } from '../src/ml/index.js';

describe('RandomForest estimators', () => {
  it('should classify blobs', () => {
    const X = [];
    const y = [];
    const random = () => Math.random();

    const addPoints = (centerX, centerY, label) => {
      for (let i = 0; i < 30; i++) {
        const x = centerX + (random() - 0.5);
        const yVal = centerY + (random() - 0.5);
        X.push([x, yVal]);
        y.push(label);
      }
    };

    addPoints(0, 0, 'A');
    addPoints(5, 5, 'B');

    const forest = new RandomForestClassifier({ nEstimators: 10, maxDepth: 5, seed: 42 });
    forest.fit(X, y);

    const preds = forest.predict([[0.1, -0.2], [5.2, 5.1]]);
    expect(preds).toEqual(['A', 'B']);
  });

  it('should regress smooth function', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 50; i++) {
      const x = -2 + (4 * i) / 49;
      X.push([x]);
      y.push(Math.sin(x));
    }

    const forest = new RandomForestRegressor({ nEstimators: 20, maxDepth: 6, seed: 7 });
    forest.fit(X, y);

    const preds = forest.predict([[0], [Math.PI / 2]]);
    expect(preds[0]).toBeCloseTo(0, 0);
    expect(preds[1]).toBeGreaterThan(0.7);
  });
});

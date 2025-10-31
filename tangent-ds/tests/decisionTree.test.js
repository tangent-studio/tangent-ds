import { describe, it, expect } from 'vitest';
import { DecisionTreeClassifier, DecisionTreeRegressor } from '../src/ml/index.js';

describe('DecisionTree estimators', () => {
  it('should classify axis-aligned regions', () => {
    const X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
      [3, 3],
      [3, 4],
      [4, 3],
      [4, 4]
    ];
    const y = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'];

    const tree = new DecisionTreeClassifier({ maxDepth: 3 });
    tree.fit(X, y);

    const preds = tree.predict([[0.2, 0.3], [3.5, 3.2]]);
    expect(preds).toEqual(['A', 'B']);
  });

  it('should regress simple function', () => {
    const X = [];
    const y = [];
    for (let i = 0; i <= 10; i++) {
      const x = i / 10;
      X.push([x]);
      y.push(x * x);
    }

    const tree = new DecisionTreeRegressor({ maxDepth: 4, minSamplesSplit: 2 });
    tree.fit(X, y);

    const preds = tree.predict([[0.15], [0.85]]);
    expect(preds[0]).toBeCloseTo(0.15 * 0.15, 1);
    expect(preds[1]).toBeCloseTo(0.85 * 0.85, 0);
  });
});

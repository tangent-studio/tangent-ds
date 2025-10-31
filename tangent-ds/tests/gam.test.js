import { describe, it, expect } from 'vitest';
import { GAMRegressor, GAMClassifier } from '../src/ml/index.js';

describe('GAM estimators', () => {
  it('should fit smooth regression curve', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 80; i++) {
      const x = -2 + (4 * i) / 79;
      X.push([x]);
      y.push(Math.sin(x));
    }

    const gam = new GAMRegressor({ nSplines: 5 });
    gam.fit(X, y);

    const preds = gam.predict([[0], [Math.PI / 2]]);
    expect(preds[0]).toBeCloseTo(0, 1);
    expect(preds[1]).toBeGreaterThan(0.7);
  });

  it('should separate two classes', () => {
    const X = [];
    const y = [];
    for (let i = 0; i < 50; i++) {
      const x = i / 50;
      X.push([x]);
      y.push(x > 0.5 ? 'B' : 'A');
    }

    const gam = new GAMClassifier({ nSplines: 4, maxIter: 50 });
    gam.fit(X, y);

    const preds = gam.predict([[0.2], [0.8]]);
    expect(preds).toEqual(['A', 'B']);
  });
});

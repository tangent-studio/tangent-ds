/**
 * Optimization Example
 * Demonstrates gradient-based optimizers and convergence visualization
 * Using Tangent Notebook format
 */

import { core } from '@tangent.to/ds';

console.log('=== Optimization Algorithms Demo ===\n');

// ## Test Functions

// Quadratic function: f(x) = (x - 3)^2
function quadratic(x) {
  const val = x[0] - 3;
  return {
    loss: val * val,
    gradient: [2 * val]
  };
}

// Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
function rosenbrock(x, a = 1, b = 100) {
  const loss = (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2;
  const gradient = [
    -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2),
    2 * b * (x[1] - x[0] ** 2)
  ];
  return { loss, gradient };
}

// Beale function: f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
function beale(x) {
  const [a, b] = x;
  const term1 = (1.5 - a + a * b) ** 2;
  const term2 = (2.25 - a + a * b ** 2) ** 2;
  const term3 = (2.625 - a + a * b ** 3) ** 2;
  const loss = term1 + term2 + term3;

  const grad_a = 2 * (1.5 - a + a * b) * (-1 + b) +
                 2 * (2.25 - a + a * b ** 2) * (-1 + b ** 2) +
                 2 * (2.625 - a + a * b ** 3) * (-1 + b ** 3);

  const grad_b = 2 * (1.5 - a + a * b) * a +
                 2 * (2.25 - a + a * b ** 2) * (2 * a * b) +
                 2 * (2.625 - a + a * b ** 3) * (3 * a * b ** 2);

  return { loss, gradient: [grad_a, grad_b] };
}

console.log('Test functions defined:');
console.log('- Quadratic: f(x) = (x - 3)²');
console.log('- Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²');
console.log('- Beale: Complex 2D function\n');

// ## Test 1: Quadratic Minimization

console.log('=== Test 1: Quadratic Function (1D) ===\n');

const optimizers = [
  { name: 'Gradient Descent', opt: new core.optimize.GradientDescent({ learningRate: 0.1, maxIter: 100 }) },
  { name: 'Momentum', opt: new core.optimize.MomentumOptimizer({ learningRate: 0.1, momentum: 0.9, maxIter: 100 }) },
  { name: 'RMSProp', opt: new core.optimize.RMSProp({ learningRate: 0.1, maxIter: 100 }) },
  { name: 'Adam', opt: new core.optimize.AdamOptimizer({ learningRate: 0.1, maxIter: 100 }) }
];

optimizers.forEach(({ name, opt }) => {
  const { x, history } = opt.minimize(quadratic, [0]);
  
  console.log(`${name}:`);
  console.log(`  Final x: ${x[0].toFixed(6)} (target: 3.0)`);
  console.log(`  Final loss: ${history.loss[history.loss.length - 1].toFixed(8)}`);
  console.log(`  Iterations: ${history.loss.length}`);
  console.log(`  Convergence rate: ${(history.loss[0] / history.loss[history.loss.length - 1]).toFixed(2)}x\n`);
});

// ## Test 2: Rosenbrock Function

console.log('=== Test 2: Rosenbrock Function (2D) ===\n');
console.log('Global minimum at (1, 1)\n');

const rosOpts = [
  { name: 'GD', opt: new core.optimize.GradientDescent({ learningRate: 0.001, maxIter: 2000 }) },
  { name: 'Momentum', opt: new core.optimize.MomentumOptimizer({ learningRate: 0.001, momentum: 0.9, maxIter: 2000 }) },
  { name: 'Adam', opt: new core.optimize.AdamOptimizer({ learningRate: 0.01, maxIter: 1000 }) }
];

rosOpts.forEach(({ name, opt }) => {
  const { x, history } = opt.minimize(rosenbrock, [-1, -1]);
  
  console.log(`${name}:`);
  console.log(`  Final x: (${x[0].toFixed(4)}, ${x[1].toFixed(4)})`);
  console.log(`  Distance from optimum: ${Math.sqrt((x[0] - 1)**2 + (x[1] - 1)**2).toFixed(4)}`);
  console.log(`  Final loss: ${history.loss[history.loss.length - 1].toFixed(6)}`);
  console.log(`  Iterations: ${history.loss.length}\n`);
});

// ## Test 3: Learning Rate Comparison

console.log('=== Test 3: Learning Rate Impact (Adam on Beale) ===\n');

const learningRates = [0.001, 0.01, 0.1, 0.5];

learningRates.forEach(lr => {
  const opt = new core.optimize.AdamOptimizer({ learningRate: lr, maxIter: 500 });
  const { x, history } = opt.minimize(beale, [0, 0]);
  
  console.log(`LR = ${lr}:`);
  console.log(`  Final x: (${x[0].toFixed(4)}, ${x[1].toFixed(4)})`);
  console.log(`  Final loss: ${history.loss[history.loss.length - 1].toFixed(6)}`);
  console.log(`  Iterations to converge: ${history.loss.length}\n`);
});

// ## Test 4: Line Search

console.log('=== Test 4: Backtracking Line Search ===\n');

const gdBasic = new core.optimize.GradientDescent({ 
  learningRate: 0.1, 
  lineSearch: false,
  maxIter: 200 
});

const gdLineSearch = new core.optimize.GradientDescent({ 
  learningRate: 1.0, // Can use larger initial LR
  lineSearch: true,
  maxIter: 200 
});

const result1 = gdBasic.minimize(rosenbrock, [0, 0]);
const result2 = gdLineSearch.minimize(rosenbrock, [0, 0]);

console.log('Fixed Learning Rate:');
console.log(`  Final loss: ${result1.history.loss[result1.history.loss.length - 1].toFixed(6)}`);
console.log(`  Iterations: ${result1.history.loss.length}`);

console.log('\nWith Line Search:');
console.log(`  Final loss: ${result2.history.loss[result2.history.loss.length - 1].toFixed(6)}`);
console.log(`  Iterations: ${result2.history.loss.length}`);
console.log(`  Avg step size: ${(result2.history.learningRate.reduce((a,b) => a+b, 0) / result2.history.learningRate.length).toFixed(4)}`);

// ## Convergence Analysis

console.log('\n=== Convergence Analysis ===\n');

const adamOpt = new core.optimize.AdamOptimizer({ 
  learningRate: 0.1, 
  maxIter: 50,
  verbose: false
});

const { history } = adamOpt.minimize(quadratic, [10]);

console.log('Loss progression (first 10 iterations):');
history.loss.slice(0, 10).forEach((loss, i) => {
  console.log(`  Iter ${i}: loss=${loss.toFixed(6)}, grad_norm=${history.gradNorm[i].toFixed(6)}`);
});

console.log(`\n... (${history.loss.length - 10} more iterations)\n`);

console.log(`Final 3 iterations:`);
history.loss.slice(-3).forEach((loss, i) => {
  const idx = history.loss.length - 3 + i;
  console.log(`  Iter ${idx}: loss=${loss.toFixed(6)}, grad_norm=${history.gradNorm[idx].toFixed(6)}`);
});

console.log('\n✓ Optimization examples complete');
console.log('\nKey takeaways:');
console.log('- Adam typically converges fastest');
console.log('- Momentum helps with ill-conditioned problems');
console.log('- Line search can improve robustness');
console.log('- Learning rate is crucial for convergence');

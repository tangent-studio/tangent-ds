/**
 * Differentiable loss functions for optimization
 * Each function returns {loss, gradient}
 */

/**
 * Mean Squared Error Loss
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @returns {Object} {loss, gradient}
 */
export function mseLoss(yTrue, yPred) {
  const n = yTrue.length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const diff = yPred[i] - yTrue[i];
    loss += diff * diff;
  }
  loss /= n;
  
  // Compute gradient
  const gradient = [];
  for (let i = 0; i < n; i++) {
    gradient.push((2 / n) * (yPred[i] - yTrue[i]));
  }
  
  return { loss, gradient };
}

/**
 * Mean Absolute Error Loss
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @returns {Object} {loss, gradient}
 */
export function maeLoss(yTrue, yPred) {
  const n = yTrue.length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    loss += Math.abs(yPred[i] - yTrue[i]);
  }
  loss /= n;
  
  // Compute gradient (subgradient at 0)
  const gradient = [];
  for (let i = 0; i < n; i++) {
    const diff = yPred[i] - yTrue[i];
    gradient.push((1 / n) * Math.sign(diff));
  }
  
  return { loss, gradient };
}

/**
 * Binary Cross-Entropy Loss (Log Loss)
 * @param {Array<number>} yTrue - True labels (0 or 1)
 * @param {Array<number>} yPred - Predicted probabilities
 * @param {number} epsilon - Small value to avoid log(0)
 * @returns {Object} {loss, gradient}
 */
export function logLoss(yTrue, yPred, epsilon = 1e-15) {
  const n = yTrue.length;
  
  // Clip predictions to avoid log(0)
  const yPredClipped = yPred.map(p => Math.max(epsilon, Math.min(1 - epsilon, p)));
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    loss += -(yTrue[i] * Math.log(yPredClipped[i]) + 
              (1 - yTrue[i]) * Math.log(1 - yPredClipped[i]));
  }
  loss /= n;
  
  // Compute gradient
  const gradient = [];
  for (let i = 0; i < n; i++) {
    gradient.push((1 / n) * ((yPredClipped[i] - yTrue[i]) / 
                  (yPredClipped[i] * (1 - yPredClipped[i]))));
  }
  
  return { loss, gradient };
}

/**
 * Categorical Cross-Entropy Loss
 * @param {Array<Array<number>>} yTrue - One-hot encoded true labels
 * @param {Array<Array<number>>} yPred - Predicted probabilities
 * @param {number} epsilon - Small value to avoid log(0)
 * @returns {Object} {loss, gradient}
 */
export function crossEntropy(yTrue, yPred, epsilon = 1e-15) {
  const n = yTrue.length;
  const k = yTrue[0].length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      const pred = Math.max(epsilon, Math.min(1 - epsilon, yPred[i][j]));
      loss += -yTrue[i][j] * Math.log(pred);
    }
  }
  loss /= n;
  
  // Compute gradient
  const gradient = [];
  for (let i = 0; i < n; i++) {
    const gradRow = [];
    for (let j = 0; j < k; j++) {
      const pred = Math.max(epsilon, Math.min(1 - epsilon, yPred[i][j]));
      gradRow.push((1 / n) * (-yTrue[i][j] / pred));
    }
    gradient.push(gradRow);
  }
  
  return { loss, gradient };
}

/**
 * Hinge Loss (for SVM)
 * @param {Array<number>} yTrue - True labels (-1 or 1)
 * @param {Array<number>} yPred - Predicted scores
 * @returns {Object} {loss, gradient}
 */
export function hingeLoss(yTrue, yPred) {
  const n = yTrue.length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    loss += Math.max(0, 1 - yTrue[i] * yPred[i]);
  }
  loss /= n;
  
  // Compute gradient (subgradient)
  const gradient = [];
  for (let i = 0; i < n; i++) {
    if (yTrue[i] * yPred[i] < 1) {
      gradient.push((1 / n) * (-yTrue[i]));
    } else {
      gradient.push(0);
    }
  }
  
  return { loss, gradient };
}

/**
 * Huber Loss (robust to outliers)
 * @param {Array<number>} yTrue - True values
 * @param {Array<number>} yPred - Predicted values
 * @param {number} delta - Threshold for switching from quadratic to linear
 * @returns {Object} {loss, gradient}
 */
export function huberLoss(yTrue, yPred, delta = 1.0) {
  const n = yTrue.length;
  
  // Compute loss
  let loss = 0;
  for (let i = 0; i < n; i++) {
    const diff = Math.abs(yPred[i] - yTrue[i]);
    if (diff <= delta) {
      loss += 0.5 * diff * diff;
    } else {
      loss += delta * (diff - 0.5 * delta);
    }
  }
  loss /= n;
  
  // Compute gradient
  const gradient = [];
  for (let i = 0; i < n; i++) {
    const diff = yPred[i] - yTrue[i];
    const absDiff = Math.abs(diff);
    if (absDiff <= delta) {
      gradient.push((1 / n) * diff);
    } else {
      gradient.push((1 / n) * delta * Math.sign(diff));
    }
  }
  
  return { loss, gradient };
}

/**
 * Get loss function by name
 * @param {string} name - Loss function name
 * @returns {Function} Loss function
 */
export function getLossFunction(name) {
  const losses = {
    'mse': mseLoss,
    'mean_squared_error': mseLoss,
    'mae': maeLoss,
    'mean_absolute_error': maeLoss,
    'log': logLoss,
    'log_loss': logLoss,
    'binary_crossentropy': logLoss,
    'crossentropy': crossEntropy,
    'categorical_crossentropy': crossEntropy,
    'hinge': hingeLoss,
    'huber': huberLoss
  };

  const lossFn = losses[name.toLowerCase()];
  if (!lossFn) {
    throw new Error(`Unknown loss function: ${name}. Available: ${Object.keys(losses).join(', ')}`);
  }

  return lossFn;
}

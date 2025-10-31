/**
 * Model training utilities
 * Unified training loop with callbacks and history tracking
 */

import { createOptimizer } from '../core/optimize.js';
import { getLossFunction } from './loss.js';

/**
 * Train a model with gradient-based optimization
 * @param {Object} model - Model with forward and backward methods
 * @param {Array<Array<number>>} X - Feature matrix
 * @param {Array<number>|Array<Array<number>>} y - Target values
 * @param {Object} options - Training options
 * @returns {Object} Training history
 */
export function train(model, X, y, options = {}) {
  const {
    optimizer = 'adam',
    optimizerOptions = {},
    loss = 'mse',
    epochs = 100,
    batchSize = 32,
    validationSplit = 0.0,
    shuffle = true,
    verbose = true,
    callbacks = {}
  } = options;

  // Get optimizer and loss function
  const opt = typeof optimizer === 'string' 
    ? createOptimizer(optimizer, optimizerOptions)
    : optimizer;
  
  const lossFn = typeof loss === 'string' 
    ? getLossFunction(loss)
    : loss;

  // Split data if validation is requested
  let XTrain = X;
  let yTrain = y;
  let XVal = null;
  let yVal = null;

  if (validationSplit > 0) {
    const splitIdx = Math.floor(X.length * (1 - validationSplit));
    XTrain = X.slice(0, splitIdx);
    yTrain = y.slice(0, splitIdx);
    XVal = X.slice(splitIdx);
    yVal = y.slice(splitIdx);
  }

  const nSamples = XTrain.length;
  const nBatches = Math.ceil(nSamples / batchSize);

  // Training history
  const history = {
    loss: [],
    valLoss: [],
    epoch: []
  };

  // Training loop
  for (let epoch = 0; epoch < epochs; epoch++) {
    let epochLoss = 0;
    let batchCount = 0;

    // Shuffle data if requested
    let indices = Array.from({ length: nSamples }, (_, i) => i);
    if (shuffle) {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }

    // Mini-batch training
    for (let batch = 0; batch < nBatches; batch++) {
      const batchStart = batch * batchSize;
      const batchEnd = Math.min(batchStart + batchSize, nSamples);
      const batchIndices = indices.slice(batchStart, batchEnd);

      // Get batch data
      const XBatch = batchIndices.map(i => XTrain[i]);
      const yBatch = batchIndices.map(i => yTrain[i]);

      // Forward pass
      const yPred = model.predict(XBatch);

      // Compute loss
      const { loss: batchLoss, gradient } = lossFn(yBatch, yPred);
      epochLoss += batchLoss;
      batchCount++;

      // Backward pass and update (if model supports it)
      if (model.backward && model.update) {
        model.backward(gradient);
        model.update(opt);
      }

      // Batch callback
      if (callbacks.onBatchEnd) {
        callbacks.onBatchEnd(batch, { loss: batchLoss });
      }
    }

    // Average epoch loss
    epochLoss /= batchCount;
    history.loss.push(epochLoss);
    history.epoch.push(epoch);

    // Validation
    if (XVal !== null) {
      const yPredVal = model.predict(XVal);
      const { loss: valLoss } = lossFn(yVal, yPredVal);
      history.valLoss.push(valLoss);
    }

    // Epoch callback
    if (callbacks.onEpochEnd) {
      const info = { loss: epochLoss };
      if (history.valLoss.length > 0) {
        info.valLoss = history.valLoss[history.valLoss.length - 1];
      }
      callbacks.onEpochEnd(epoch, info);
    }

    // Verbose output
    if (verbose && epoch % Math.max(1, Math.floor(epochs / 10)) === 0) {
      let msg = `Epoch ${epoch}/${epochs}: loss=${epochLoss.toFixed(6)}`;
      if (history.valLoss.length > 0) {
        msg += `, val_loss=${history.valLoss[history.valLoss.length - 1].toFixed(6)}`;
      }
      console.log(msg);
    }
  }

  return history;
}

/**
 * Simple training loop for functions (not models)
 * @param {Function} lossFn - Loss function that returns {loss, gradient}
 * @param {Array<number>} params0 - Initial parameters
 * @param {Object} options - Training options
 * @returns {Object} {params, history}
 */
export function trainFunction(lossFn, params0, options = {}) {
  const {
    optimizer = 'adam',
    optimizerOptions = {},
    maxIter = 1000,
    tol = 1e-6,
    verbose = false,
    callbacks = {}
  } = options;

  // Get optimizer
  const opt = typeof optimizer === 'string'
    ? createOptimizer(optimizer, { ...optimizerOptions, maxIter, tol, verbose })
    : optimizer;

  // Minimize
  const result = opt.minimize(lossFn, params0, { maxIter, tol });

  // Call epoch callbacks if provided
  if (callbacks.onEpochEnd) {
    result.history.loss.forEach((loss, epoch) => {
      callbacks.onEpochEnd(epoch, { loss });
    });
  }

  return {
    params: result.x,
    history: result.history
  };
}

/**
 * Early stopping callback
 * @param {number} patience - Number of epochs to wait for improvement
 * @param {number} minDelta - Minimum change to qualify as improvement
 * @returns {Object} Callback object with state
 */
export function earlyStopping(patience = 10, minDelta = 0) {
  let bestLoss = Infinity;
  let wait = 0;
  let stopped = false;

  return {
    onEpochEnd: (epoch, logs) => {
      const currentLoss = logs.valLoss !== undefined ? logs.valLoss : logs.loss;

      if (currentLoss < bestLoss - minDelta) {
        bestLoss = currentLoss;
        wait = 0;
      } else {
        wait += 1;
        if (wait >= patience) {
          stopped = true;
          console.log(`Early stopping at epoch ${epoch}`);
        }
      }
    },
    stopped: () => stopped,
    reset: () => {
      bestLoss = Infinity;
      wait = 0;
      stopped = false;
    }
  };
}

/**
 * Learning rate scheduler callback
 * @param {Function} scheduleFn - Function (epoch) => learningRate
 * @param {Optimizer} optimizer - Optimizer to update
 * @returns {Object} Callback object
 */
export function learningRateScheduler(scheduleFn, optimizer) {
  return {
    onEpochEnd: (epoch, logs) => {
      const newLR = scheduleFn(epoch);
      optimizer.learningRate = newLR;
      if (logs.verbose) {
        console.log(`Learning rate updated to ${newLR}`);
      }
    }
  };
}

/**
 * Model checkpoint callback
 * @param {string} metric - Metric to monitor ('loss' or 'valLoss')
 * @returns {Object} Callback object with state
 */
export function modelCheckpoint(metric = 'valLoss') {
  let bestMetric = Infinity;
  let bestParams = null;

  return {
    onEpochEnd: (epoch, logs) => {
      const currentMetric = logs[metric];
      if (currentMetric !== undefined && currentMetric < bestMetric) {
        bestMetric = currentMetric;
        // In a real implementation, would save model here
        console.log(`New best ${metric}: ${currentMetric.toFixed(6)} at epoch ${epoch}`);
      }
    },
    getBest: () => ({ metric: bestMetric, params: bestParams })
  };
}

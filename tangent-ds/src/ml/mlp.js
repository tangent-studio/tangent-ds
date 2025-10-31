/**
 * Multilayer Perceptron (MLP)
 * Simple feedforward neural network with backpropagation
 */

import { toMatrix, Matrix } from '../core/linalg.js';
import { mean } from '../core/math.js';

// ============= Activation Functions =============

const activations = {
  sigmoid: {
    fn: (x) => 1 / (1 + Math.exp(-x)),
    derivative: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    }
  },
  
  relu: {
    fn: (x) => Math.max(0, x),
    derivative: (x) => x > 0 ? 1 : 0
  },
  
  tanh: {
    fn: (x) => Math.tanh(x),
    derivative: (x) => {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
  },
  
  linear: {
    fn: (x) => x,
    derivative: (x) => 1
  }
};

/**
 * Initialize weights using Xavier/Glorot initialization
 * @param {number} nIn - Input size
 * @param {number} nOut - Output size
 * @returns {Array<Array<number>>} Weight matrix
 */
function initializeWeights(nIn, nOut) {
  const limit = Math.sqrt(6 / (nIn + nOut));
  const weights = [];
  for (let i = 0; i < nOut; i++) {
    const row = [];
    for (let j = 0; j < nIn; j++) {
      row.push((Math.random() * 2 - 1) * limit);
    }
    weights.push(row);
  }
  return weights;
}

/**
 * Forward pass through network
 * @param {Array<number>} input - Input vector
 * @param {Array<Object>} layers - Network layers
 * @returns {Object} {activations, outputs}
 */
function forward(input, layers) {
  const layerActivations = [input];
  const outputs = [input];
  
  let currentActivation = input;
  
  for (const layer of layers) {
    const { weights, bias, activation } = layer;
    
    // Linear transformation: z = W * a + b
    const z = [];
    for (let i = 0; i < weights.length; i++) {
      let sum = bias[i];
      for (let j = 0; j < currentActivation.length; j++) {
        sum += weights[i][j] * currentActivation[j];
      }
      z.push(sum);
    }
    
    outputs.push(z);
    
    // Apply activation function
    const a = z.map(val => activations[activation].fn(val));
    layerActivations.push(a);
    
    currentActivation = a;
  }
  
  return { activations: layerActivations, outputs };
}

/**
 * Backward pass (backpropagation)
 * @param {Array<number>} target - Target output
 * @param {Object} forwardResult - Result from forward pass
 * @param {Array<Object>} layers - Network layers
 * @returns {Array<Object>} Gradients for each layer
 */
function backward(target, forwardResult, layers) {
  const { activations: acts, outputs } = forwardResult;
  const nLayers = layers.length;
  const gradients = [];
  
  // Output layer error
  const outputActivation = acts[nLayers];
  let delta = outputActivation.map((a, i) => a - target[i]);
  
  // Backpropagate through layers
  for (let l = nLayers - 1; l >= 0; l--) {
    const layer = layers[l];
    const { weights, activation } = layer;
    const z = outputs[l + 1];
    const prevActivation = acts[l];
    
    // Compute gradient for activation
    const activationGrad = z.map((val, i) => 
      delta[i] * activations[activation].derivative(val)
    );
    
    // Compute weight gradients
    const weightGrad = [];
    for (let i = 0; i < weights.length; i++) {
      const row = [];
      for (let j = 0; j < weights[i].length; j++) {
        row.push(activationGrad[i] * prevActivation[j]);
      }
      weightGrad.push(row);
    }
    
    // Compute bias gradients
    const biasGrad = [...activationGrad];
    
    gradients.unshift({ weightGrad, biasGrad });
    
    // Propagate error to previous layer
    if (l > 0) {
      const newDelta = new Array(prevActivation.length).fill(0);
      for (let j = 0; j < weights[0].length; j++) {
        for (let i = 0; i < weights.length; i++) {
          newDelta[j] += weights[i][j] * activationGrad[i];
        }
      }
      delta = newDelta;
    }
  }
  
  return gradients;
}

/**
 * Create MLP architecture
 * @param {Array<number>} layerSizes - Size of each layer [input, hidden1, ..., output]
 * @param {string} activation - Activation function ('sigmoid', 'relu', 'tanh')
 * @returns {Array<Object>} Initialized layers
 */
export function createNetwork(layerSizes, activation = 'relu') {
  if (layerSizes.length < 2) {
    throw new Error('Network must have at least input and output layers');
  }
  
  if (!activations[activation]) {
    throw new Error(`Unknown activation: ${activation}. Use 'sigmoid', 'relu', or 'tanh'`);
  }
  
  const layers = [];
  
  for (let i = 1; i < layerSizes.length; i++) {
    const nIn = layerSizes[i - 1];
    const nOut = layerSizes[i];
    
    // Use linear activation for output layer, specified activation for hidden layers
    const layerActivation = i === layerSizes.length - 1 ? 'linear' : activation;
    
    layers.push({
      weights: initializeWeights(nIn, nOut),
      bias: new Array(nOut).fill(0),
      activation: layerActivation
    });
  }
  
  return layers;
}

/**
 * Train MLP using mini-batch gradient descent
 * @param {Array<Array<number>>} X - Training data
 * @param {Array<Array<number>>|Array<number>} y - Target values
 * @param {Object} options - Training options
 * @returns {Object} {layers, losses, epochs}
 */
export function fit(X, y, {
  layerSizes = null,
  activation = 'relu',
  learningRate = 0.01,
  epochs = 100,
  batchSize = 32,
  verbose = false
} = {}) {
  // Convert data
  let data = Array.isArray(X[0]) ? X : X.map(x => [x]);
  let targets;
  
  if (Array.isArray(y[0])) {
    targets = y;
  } else {
    // Convert 1D targets to 2D
    targets = y.map(val => [val]);
  }
  
  const nSamples = data.length;
  const nFeatures = data[0].length;
  const nOutputs = targets[0].length;
  
  // Create network if not provided
  if (!layerSizes) {
    // Default: one hidden layer with size = mean(input, output)
    const hiddenSize = Math.floor((nFeatures + nOutputs) / 2);
    layerSizes = [nFeatures, hiddenSize, nOutputs];
  } else {
    // Validate layer sizes
    if (layerSizes[0] !== nFeatures) {
      throw new Error(`First layer size (${layerSizes[0]}) must match input features (${nFeatures})`);
    }
    if (layerSizes[layerSizes.length - 1] !== nOutputs) {
      throw new Error(`Last layer size (${layerSizes[layerSizes.length - 1]}) must match output size (${nOutputs})`);
    }
  }
  
  const layers = createNetwork(layerSizes, activation);
  const losses = [];
  
  // Training loop
  for (let epoch = 0; epoch < epochs; epoch++) {
    let epochLoss = 0;
    
    // Shuffle data
    const indices = Array.from({ length: nSamples }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    // Mini-batch training
    for (let batch = 0; batch < nSamples; batch += batchSize) {
      const batchEnd = Math.min(batch + batchSize, nSamples);
      const batchIndices = indices.slice(batch, batchEnd);
      
      // Accumulate gradients over batch
      const batchGradients = layers.map(layer => ({
        weightGrad: layer.weights.map(row => row.map(() => 0)),
        biasGrad: layer.bias.map(() => 0)
      }));
      
      let batchLoss = 0;
      
      for (const idx of batchIndices) {
        const input = data[idx];
        const target = targets[idx];
        
        // Forward pass
        const forwardResult = forward(input, layers);
        const output = forwardResult.activations[forwardResult.activations.length - 1];
        
        // Compute loss (MSE)
        const sampleLoss = output.reduce((sum, val, i) => 
          sum + (val - target[i]) ** 2, 0
        ) / output.length;
        batchLoss += sampleLoss;
        
        // Backward pass
        const gradients = backward(target, forwardResult, layers);
        
        // Accumulate gradients
        gradients.forEach((grad, l) => {
          grad.weightGrad.forEach((row, i) => {
            row.forEach((val, j) => {
              batchGradients[l].weightGrad[i][j] += val;
            });
          });
          grad.biasGrad.forEach((val, i) => {
            batchGradients[l].biasGrad[i] += val;
          });
        });
      }
      
      // Update weights and biases
      const batchCount = batchIndices.length;
      batchGradients.forEach((grad, l) => {
        // Update weights
        layers[l].weights.forEach((row, i) => {
          row.forEach((val, j) => {
            layers[l].weights[i][j] -= (learningRate / batchCount) * grad.weightGrad[i][j];
          });
        });
        
        // Update biases
        layers[l].bias.forEach((val, i) => {
          layers[l].bias[i] -= (learningRate / batchCount) * grad.biasGrad[i];
        });
      });
      
      epochLoss += batchLoss / batchCount;
    }
    
    epochLoss /= Math.ceil(nSamples / batchSize);
    losses.push(epochLoss);
    
    if (verbose && epoch % 10 === 0) {
      console.log(`Epoch ${epoch}: Loss = ${epochLoss.toFixed(6)}`);
    }
  }
  
  return {
    layers,
    losses,
    epochs,
    layerSizes
  };
}

/**
 * Predict using trained MLP
 * @param {Object} model - Trained model from fit()
 * @param {Array<Array<number>>|Array<number>} X - Input data
 * @returns {Array<Array<number>>} Predictions
 */
export function predict(model, X) {
  const { layers } = model;
  const data = Array.isArray(X[0]) ? X : X.map(x => [x]);
  
  const predictions = [];
  
  for (const input of data) {
    const result = forward(input, layers);
    const output = result.activations[result.activations.length - 1];
    predictions.push(output);
  }
  
  return predictions;
}

/**
 * Evaluate model on test data
 * @param {Object} model - Trained model
 * @param {Array<Array<number>>} X - Test inputs
 * @param {Array<Array<number>>|Array<number>} y - Test targets
 * @returns {Object} {mse, mae}
 */
export function evaluate(model, X, y) {
  const predictions = predict(model, X);
  const targets = Array.isArray(y[0]) ? y : y.map(val => [val]);
  
  let mse = 0;
  let mae = 0;
  const n = predictions.length;
  
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < predictions[i].length; j++) {
      const error = predictions[i][j] - targets[i][j];
      mse += error ** 2;
      mae += Math.abs(error);
    }
  }
  
  const totalOutputs = n * predictions[0].length;
  
  return {
    mse: mse / totalOutputs,
    mae: mae / totalOutputs
  };
}
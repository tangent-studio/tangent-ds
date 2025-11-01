import { attachShow } from './show.js';

/**
 * Visualization utilities for model interpretation
 * Returns Observable Plot configuration objects
 */

/**
 * Generate feature importance bar plot configuration
 * @param {Array<Object>} importances - Feature importance array
 * @param {Object} options - {width, height, topN}
 * @returns {Object} Plot configuration
 */
export function plotFeatureImportance(importances, { 
  width = 640, 
  height = 400,
  topN = 10
} = {}) {
  // Take top N features
  const topFeatures = importances.slice(0, topN);
  
  const data = topFeatures.map((imp, i) => ({
    feature: imp.feature?.toString() || `Feature ${imp.feature}`,
    importance: imp.importance,
    std: imp.std,
    rank: i + 1
  }));
  
  return attachShow({
    type: 'featureImportance',
    width,
    height,
    data: { features: data },
    axes: {
      x: { label: 'Importance' },
      y: { label: 'Feature' }
    },
    marks: [
      {
        type: 'barX',
        data: 'features',
        x: 'importance',
        y: 'feature',
        fill: 'steelblue',
        sort: { y: '-x' }
      }
    ]
  });
}

/**
 * Generate partial dependence plot configuration
 * @param {Object} pdResult - Result from partialDependence()
 * @param {Object} options - {width, height, featureName}
 * @returns {Object} Plot configuration
 */
export function plotPartialDependence(pdResult, { 
  width = 640, 
  height = 400,
  featureName = null
} = {}) {
  const { values, predictions, feature } = pdResult;
  
  const data = values.map((val, i) => ({
    value: val,
    prediction: predictions[i]
  }));
  
  const xLabel = featureName || `Feature ${feature}`;
  
  return attachShow({
    type: 'partialDependence',
    width,
    height,
    data: { points: data },
    axes: {
      x: { label: xLabel },
      y: { label: 'Partial Dependence', grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'points',
        x: 'value',
        y: 'prediction',
        stroke: 'steelblue',
        strokeWidth: 2
      },
      {
        type: 'dot',
        data: 'points',
        x: 'value',
        y: 'prediction',
        fill: 'steelblue',
        r: 3
      }
    ]
  });
}

/**
 * Generate correlation matrix heatmap configuration
 * @param {Object} corrResult - Result from correlationMatrix()
 * @param {Object} options - {width, height}
 * @returns {Object} Plot configuration
 */
export function plotCorrelationMatrix(corrResult, { 
  width = 640, 
  height = 600
} = {}) {
  const { matrix, features } = corrResult;
  
  // Flatten matrix into data points
  const data = [];
  for (let i = 0; i < features.length; i++) {
    for (let j = 0; j < features.length; j++) {
      data.push({
        feature1: features[i],
        feature2: features[j],
        correlation: matrix[i][j],
        i,
        j
      });
    }
  }
  
  return attachShow({
    type: 'correlationMatrix',
    width,
    height,
    data: { cells: data },
    axes: {
      x: { label: null },
      y: { label: null }
    },
    marks: [
      {
        type: 'cell',
        data: 'cells',
        x: 'feature1',
        y: 'feature2',
        fill: 'correlation',
        fillScale: {
          domain: [-1, 1],
          scheme: 'RdBu',
          reverse: true
        }
      },
      {
        type: 'text',
        data: 'cells',
        x: 'feature1',
        y: 'feature2',
        text: d => d.correlation.toFixed(2),
        fontSize: 10
      }
    ]
  });
}

/**
 * Generate residual plot configuration
 * @param {Object} residualData - Result from residualPlotData()
 * @param {Object} options - {width, height, standardized}
 * @returns {Object} Plot configuration
 */
export function plotResiduals(residualData, { 
  width = 640, 
  height = 400,
  standardized = false
} = {}) {
  const { fitted, residuals, standardized: stdResiduals } = residualData;
  
  const yValues = standardized ? stdResiduals : residuals;
  const yLabel = standardized ? 'Standardized Residuals' : 'Residuals';
  
  const data = fitted.map((fit, i) => ({
    fitted: fit,
    residual: yValues[i]
  }));
  
  return attachShow({
    type: 'residuals',
    width,
    height,
    data: { points: data },
    axes: {
      x: { label: 'Fitted Values', grid: true },
      y: { label: yLabel, grid: true }
    },
    marks: [
      {
        type: 'dot',
        data: 'points',
        x: 'fitted',
        y: 'residual',
        fill: 'steelblue',
        r: 3,
        fillOpacity: 0.6
      },
      {
        type: 'ruleY',
        y: 0,
        stroke: 'red',
        strokeDasharray: '4,4'
      }
    ]
  });
}

/**
 * Generate Q-Q plot configuration for normality check
 * @param {Object} residualData - Result from residualPlotData()
 * @param {Object} options - {width, height}
 * @returns {Object} Plot configuration
 */
export function plotQQ(residualData, { 
  width = 400, 
  height = 400
} = {}) {
  const { standardized } = residualData;
  
  // Sort standardized residuals
  const sorted = [...standardized].sort((a, b) => a - b);
  const n = sorted.length;
  
  // Theoretical quantiles (standard normal)
  const theoreticalQuantiles = [];
  for (let i = 0; i < n; i++) {
    // Use approximate quantile for standard normal
    const p = (i + 0.5) / n;
    // Approximate inverse CDF of standard normal
    const q = Math.sqrt(2) * erfInv(2 * p - 1);
    theoreticalQuantiles.push(q);
  }
  
  const data = sorted.map((obs, i) => ({
    theoretical: theoreticalQuantiles[i],
    observed: obs
  }));
  
  // Add reference line
  const minVal = Math.min(...theoreticalQuantiles, ...sorted);
  const maxVal = Math.max(...theoreticalQuantiles, ...sorted);
  
  return attachShow({
    type: 'qq',
    width,
    height,
    data: { 
      points: data,
      reference: [
        { x: minVal, y: minVal },
        { x: maxVal, y: maxVal }
      ]
    },
    axes: {
      x: { label: 'Theoretical Quantiles', grid: true },
      y: { label: 'Sample Quantiles', grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'reference',
        x: 'x',
        y: 'y',
        stroke: 'red',
        strokeDasharray: '4,4'
      },
      {
        type: 'dot',
        data: 'points',
        x: 'theoretical',
        y: 'observed',
        fill: 'steelblue',
        r: 3
      }
    ]
  });
}

/**
 * Approximate error function inverse (for Q-Q plot)
 * @param {number} x - Input value
 * @returns {number} Inverse erf
 */
function erfInv(x) {
  const a = 0.147;
  const b = 2 / (Math.PI * a) + Math.log(1 - x * x) / 2;
  const sqrt1 = Math.sqrt(b * b - Math.log(1 - x * x) / a);
  const sqrt2 = Math.sqrt(sqrt1 - b);
  return sqrt2 * Math.sign(x);
}

/**
 * Generate learning curve plot configuration
 * @param {Object} lcResult - Result from learningCurve()
 * @param {Object} options - {width, height}
 * @returns {Object} Plot configuration
 */
export function plotLearningCurve(lcResult, { 
  width = 640, 
  height = 400
} = {}) {
  const { trainSizes, trainScores, testScores } = lcResult;
  
  const data = trainSizes.map((size, i) => [
    { size, score: trainScores[i], type: 'train' },
    { size, score: testScores[i], type: 'test' }
  ]).flat();
  
  return attachShow({
    type: 'learningCurve',
    width,
    height,
    data: { points: data },
    axes: {
      x: { label: 'Training Set Size', grid: true },
      y: { label: 'Score', grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'points',
        x: 'size',
        y: 'score',
        stroke: 'type',
        strokeWidth: 2
      },
      {
        type: 'dot',
        data: 'points',
        x: 'size',
        y: 'score',
        fill: 'type',
        r: 4
      }
    ],
    legend: {
      color: { domain: ['train', 'test'], label: 'Type' }
    }
  });
}

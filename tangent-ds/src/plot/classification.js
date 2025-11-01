import { attachShow } from './show.js';

/**
 * Classification visualization utilities
 * ROC curves, confusion matrices, precision-recall curves
 */

/**
 * Generate ROC curve plot configuration
 * @param {Array<number>} yTrue - True binary labels (0 or 1)
 * @param {Array<number>} yProb - Predicted probabilities for positive class
 * @param {Object} options - {width, height, showDiagonal}
 * @returns {Object} Plot configuration with ROC curve data and AUC
 */
export function plotROC(yTrue, yProb, {
  width = 500,
  height = 500,
  showDiagonal = true
} = {}) {
  // Compute ROC curve points
  const { fpr, tpr, thresholds, auc } = computeROC(yTrue, yProb);

  const data = fpr.map((fp, i) => ({
    fpr: fp,
    tpr: tpr[i],
    threshold: thresholds[i]
  }));

  const config = {
    type: 'roc',
    width,
    height,
    data: {
      curve: data,
      auc
    },
    axes: {
      x: { label: 'False Positive Rate', domain: [0, 1], grid: true },
      y: { label: 'True Positive Rate', domain: [0, 1], grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'curve',
        x: 'fpr',
        y: 'tpr',
        stroke: 'steelblue',
        strokeWidth: 2.5
      }
    ],
    title: `ROC Curve (AUC = ${auc.toFixed(3)})`
  };

  // Add diagonal reference line
  if (showDiagonal) {
    config.data.diagonal = [
      { x: 0, y: 0 },
      { x: 1, y: 1 }
    ];
    config.marks.push({
      type: 'line',
      data: 'diagonal',
      x: 'x',
      y: 'y',
      stroke: 'gray',
      strokeDasharray: '4,4',
      strokeWidth: 1
    });
  }

  return attachShow(config);
}

/**
 * Compute ROC curve data
 * @private
 */
function computeROC(yTrue, yProb) {
  // Sort by predicted probability (descending)
  const sorted = yTrue.map((y, i) => ({ y, prob: yProb[i] }))
    .sort((a, b) => b.prob - a.prob);

  const n = yTrue.length;
  const nPos = yTrue.filter(y => y === 1).length;
  const nNeg = n - nPos;

  const fpr = [0];
  const tpr = [0];
  const thresholds = [Infinity];

  let tp = 0;
  let fp = 0;

  for (let i = 0; i < n; i++) {
    if (sorted[i].y === 1) {
      tp++;
    } else {
      fp++;
    }

    // Add point when threshold changes or at end
    if (i === n - 1 || sorted[i].prob !== sorted[i + 1].prob) {
      fpr.push(fp / nNeg);
      tpr.push(tp / nPos);
      thresholds.push(sorted[i].prob);
    }
  }

  // Compute AUC using trapezoidal rule
  let auc = 0;
  for (let i = 1; i < fpr.length; i++) {
    auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2;
  }

  return { fpr, tpr, thresholds, auc };
}

/**
 * Generate precision-recall curve plot configuration
 * @param {Array<number>} yTrue - True binary labels (0 or 1)
 * @param {Array<number>} yProb - Predicted probabilities for positive class
 * @param {Object} options - {width, height, showBaseline}
 * @returns {Object} Plot configuration with precision-recall curve and average precision
 */
export function plotPrecisionRecall(yTrue, yProb, {
  width = 500,
  height = 500,
  showBaseline = true
} = {}) {
  // Compute precision-recall curve
  const { precision, recall, thresholds, avgPrecision } = computePrecisionRecall(yTrue, yProb);

  const data = recall.map((rec, i) => ({
    recall: rec,
    precision: precision[i],
    threshold: thresholds[i]
  }));

  const config = {
    type: 'precisionRecall',
    width,
    height,
    data: {
      curve: data,
      avgPrecision
    },
    axes: {
      x: { label: 'Recall', domain: [0, 1], grid: true },
      y: { label: 'Precision', domain: [0, 1], grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'curve',
        x: 'recall',
        y: 'precision',
        stroke: 'steelblue',
        strokeWidth: 2.5
      }
    ],
    title: `Precision-Recall (AP = ${avgPrecision.toFixed(3)})`
  };

  // Add baseline
  if (showBaseline) {
    const baseline = yTrue.filter(y => y === 1).length / yTrue.length;
    config.marks.push({
      type: 'ruleY',
      y: baseline,
      stroke: 'gray',
      strokeDasharray: '4,4',
      strokeWidth: 1
    });
  }

  return attachShow(config);
}

/**
 * Compute precision-recall curve
 * @private
 */
function computePrecisionRecall(yTrue, yProb) {
  // Sort by predicted probability (descending)
  const sorted = yTrue.map((y, i) => ({ y, prob: yProb[i] }))
    .sort((a, b) => b.prob - a.prob);

  const n = yTrue.length;
  const precision = [];
  const recall = [];
  const thresholds = [];

  let tp = 0;
  let fp = 0;
  const totalPos = yTrue.filter(y => y === 1).length;

  for (let i = 0; i < n; i++) {
    if (sorted[i].y === 1) {
      tp++;
    } else {
      fp++;
    }

    // Add point when threshold changes or at end
    if (i === n - 1 || sorted[i].prob !== sorted[i + 1].prob) {
      const prec = tp / (tp + fp);
      const rec = tp / totalPos;
      precision.push(prec);
      recall.push(rec);
      thresholds.push(sorted[i].prob);
    }
  }

  // Compute average precision (area under PR curve)
  let avgPrecision = 0;
  for (let i = 1; i < recall.length; i++) {
    avgPrecision += (recall[i] - recall[i - 1]) * precision[i];
  }

  return { precision, recall, thresholds, avgPrecision };
}

/**
 * Generate confusion matrix plot configuration
 * @param {Array} yTrue - True labels
 * @param {Array} yPred - Predicted labels
 * @param {Object} options - {width, height, normalize, labels}
 * @returns {Object} Plot configuration with confusion matrix
 */
export function plotConfusionMatrix(yTrue, yPred, {
  width = 500,
  height = 500,
  normalize = false,
  labels = null
} = {}) {
  // Compute confusion matrix
  const { matrix, classes } = computeConfusionMatrix(yTrue, yPred);

  // Normalize if requested
  let displayMatrix = matrix;
  if (normalize) {
    displayMatrix = matrix.map(row => {
      const sum = row.reduce((a, b) => a + b, 0);
      return row.map(val => sum > 0 ? val / sum : 0);
    });
  }

  // Prepare class labels
  const classLabels = labels || classes.map(c => String(c));

  // Flatten matrix into data points
  const data = [];
  for (let i = 0; i < classes.length; i++) {
    for (let j = 0; j < classes.length; j++) {
      data.push({
        true: classLabels[i],
        predicted: classLabels[j],
        count: displayMatrix[i][j],
        trueIdx: i,
        predIdx: j
      });
    }
  }

  const config = {
    type: 'confusionMatrix',
    width,
    height,
    data: {
      cells: data,
      classes: classLabels,
      normalized: normalize
    },
    axes: {
      x: { label: 'Predicted Label' },
      y: { label: 'True Label' }
    },
    marks: [
      {
        type: 'cell',
        data: 'cells',
        x: 'predicted',
        y: 'true',
        fill: 'count',
        fillScale: {
          scheme: 'Blues',
          domain: [0, normalize ? 1 : Math.max(...data.map(d => d.count))]
        },
        inset: 0.5
      },
      {
        type: 'text',
        data: 'cells',
        x: 'predicted',
        y: 'true',
        text: d => normalize ? d.count.toFixed(2) : String(Math.round(d.count)),
        fill: d => d.count > (normalize ? 0.5 : Math.max(...data.map(d => d.count)) / 2) ? 'white' : 'black',
        fontSize: 12
      }
    ],
    title: normalize ? 'Normalized Confusion Matrix' : 'Confusion Matrix'
  };

  return attachShow(config);
}

/**
 * Compute confusion matrix
 * @private
 */
function computeConfusionMatrix(yTrue, yPred) {
  // Get unique classes
  const classes = [...new Set([...yTrue, ...yPred])].sort();
  const n = classes.length;

  // Initialize matrix
  const matrix = Array.from({ length: n }, () => Array(n).fill(0));

  // Fill matrix
  for (let i = 0; i < yTrue.length; i++) {
    const trueIdx = classes.indexOf(yTrue[i]);
    const predIdx = classes.indexOf(yPred[i]);
    matrix[trueIdx][predIdx]++;
  }

  return { matrix, classes };
}

/**
 * Generate calibration curve plot configuration
 * Shows how well predicted probabilities match actual frequencies
 * @param {Array<number>} yTrue - True binary labels (0 or 1)
 * @param {Array<number>} yProb - Predicted probabilities
 * @param {Object} options - {width, height, nBins}
 * @returns {Object} Plot configuration with calibration curve
 */
export function plotCalibration(yTrue, yProb, {
  width = 500,
  height = 500,
  nBins = 10
} = {}) {
  // Compute calibration curve
  const { probMean, fracPositive } = computeCalibration(yTrue, yProb, nBins);

  const data = probMean.map((prob, i) => ({
    predicted: prob,
    observed: fracPositive[i]
  }));

  const config = {
    type: 'calibration',
    width,
    height,
    data: {
      curve: data,
      perfect: [
        { x: 0, y: 0 },
        { x: 1, y: 1 }
      ]
    },
    axes: {
      x: { label: 'Mean Predicted Probability', domain: [0, 1], grid: true },
      y: { label: 'Fraction of Positives', domain: [0, 1], grid: true }
    },
    marks: [
      {
        type: 'line',
        data: 'perfect',
        x: 'x',
        y: 'y',
        stroke: 'gray',
        strokeDasharray: '4,4',
        strokeWidth: 1
      },
      {
        type: 'line',
        data: 'curve',
        x: 'predicted',
        y: 'observed',
        stroke: 'steelblue',
        strokeWidth: 2.5
      },
      {
        type: 'dot',
        data: 'curve',
        x: 'predicted',
        y: 'observed',
        fill: 'steelblue',
        r: 5
      }
    ],
    title: 'Calibration Curve'
  };

  return attachShow(config);
}

/**
 * Compute calibration curve
 * @private
 */
function computeCalibration(yTrue, yProb, nBins) {
  // Sort by probability
  const sorted = yTrue.map((y, i) => ({ y, prob: yProb[i] }))
    .sort((a, b) => a.prob - b.prob);

  const binSize = Math.ceil(sorted.length / nBins);
  const probMean = [];
  const fracPositive = [];

  for (let i = 0; i < nBins; i++) {
    const start = i * binSize;
    const end = Math.min((i + 1) * binSize, sorted.length);
    const bin = sorted.slice(start, end);

    if (bin.length === 0) continue;

    const meanProb = bin.reduce((sum, item) => sum + item.prob, 0) / bin.length;
    const nPositive = bin.filter(item => item.y === 1).length;
    const frac = nPositive / bin.length;

    probMean.push(meanProb);
    fracPositive.push(frac);
  }

  return { probMean, fracPositive };
}

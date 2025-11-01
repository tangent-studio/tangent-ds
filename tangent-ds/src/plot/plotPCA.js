import { attachShow } from './show.js';

/**
 * PCA visualization helpers
 * Returns Observable Plot configuration objects
 */

/**
 * Generate PCA biplot configuration
 * @param {Object} result - PCA result from mva.pca.fit()
 * @param {Object} options - {colorBy, showLoadings, width, height}
 * @returns {Object} Plot configuration
 */
export function plotPCA(result, { 
  colorBy = null, 
  showLoadings = true, 
  width = 640, 
  height = 400,
  pcX = 1,
  pcY = 2
} = {}) {
  const { scores, loadings } = result;
  
  // Prepare scores data
  const scoresData = scores.map((score, i) => ({
    x: score[`pc${pcX}`],
    y: score[`pc${pcY}`],
    index: i,
    color: colorBy ? colorBy[i] : 'default'
  }));
  
  const config = {
    type: 'pca',
    width,
    height,
    data: {
      scores: scoresData
    },
    axes: {
      x: { label: `PC${pcX}`, grid: true },
      y: { label: `PC${pcY}`, grid: true }
    },
    marks: [
      {
        type: 'dot',
        data: 'scores',
        x: 'x',
        y: 'y',
        fill: 'color',
        r: 4
      }
    ]
  };
  
  // Add loadings if requested
  if (showLoadings && loadings) {
    const loadingsData = loadings.map((loading, i) => ({
      x: 0,
      y: 0,
      dx: loading[`pc${pcX}`] * 3,
      dy: loading[`pc${pcY}`] * 3,
      variable: loading.variable,
      index: i
    }));
    
    config.data.loadings = loadingsData;
    config.marks.push({
      type: 'arrow',
      data: 'loadings',
      x1: 'x',
      y1: 'y',
      x2: d => d.x + d.dx,
      y2: d => d.y + d.dy,
      stroke: 'red',
      strokeWidth: 2
    });
  }
  
  return attachShow(config);
}

/**
 * Generate PCA scree plot configuration
 * @param {Object} result - PCA result
 * @param {Object} options - {width, height}
 * @returns {Object} Plot configuration
 */
export function plotScree(result, { width = 640, height = 300 } = {}) {
  const { varianceExplained } = result;
  
  const data = varianceExplained.map((variance, i) => ({
    component: i + 1,
    variance: variance * 100,
    label: `PC${i + 1}`
  }));
  
  return attachShow({
    type: 'scree',
    width,
    height,
    data: { components: data },
    axes: {
      x: { label: 'Principal Component' },
      y: { label: 'Variance Explained (%)' }
    },
    marks: [
      {
        type: 'line',
        data: 'components',
        x: 'component',
        y: 'variance',
        stroke: 'steelblue',
        strokeWidth: 2
      },
      {
        type: 'dot',
        data: 'components',
        x: 'component',
        y: 'variance',
        fill: 'steelblue',
        r: 5
      }
    ]
  });
}

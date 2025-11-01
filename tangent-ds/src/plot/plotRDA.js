import { attachShow } from './show.js';

/**
 * RDA visualization helpers
 */

/**
 * Generate RDA triplot configuration
 * @param {Object} result - RDA result from mva.rda.fit()
 * @param {Object} options - {width, height, axis1, axis2}
 * @returns {Object} Plot configuration
 */
export function plotRDA(result, { 
  width = 640, 
  height = 400,
  axis1 = 1,
  axis2 = 2
} = {}) {
  const { canonicalScores, canonicalLoadings } = result;
  
  // Site scores
  const siteData = canonicalScores.map((score, i) => ({
    x: score[`rda${axis1}`],
    y: score[`rda${axis2}`],
    type: 'site',
    index: i
  }));
  
  // Species loadings
  const speciesData = canonicalLoadings.map((loading, i) => ({
    x: loading[`rda${axis1}`] * 2,
    y: loading[`rda${axis2}`] * 2,
    type: 'species',
    variable: loading.variable
  }));
  
  return attachShow({
    type: 'rda',
    width,
    height,
    data: {
      sites: siteData,
      species: speciesData
    },
    axes: {
      x: { label: `RDA${axis1}`, grid: true },
      y: { label: `RDA${axis2}`, grid: true }
    },
    marks: [
      {
        type: 'dot',
        data: 'sites',
        x: 'x',
        y: 'y',
        fill: 'steelblue',
        r: 5
      },
      {
        type: 'arrow',
        data: 'species',
        x1: 0,
        y1: 0,
        x2: 'x',
        y2: 'y',
        stroke: 'red',
        strokeWidth: 2
      }
    ]
  });
}

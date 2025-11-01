import { attachShow } from './show.js';

/**
 * LDA visualization helpers
 */

/**
 * Generate LDA scatter plot configuration
 * @param {Object} result - LDA result from mva.lda.fit()
 * @param {Object} options - {width, height, ldX, ldY}
 * @returns {Object} Plot configuration
 */
export function plotLDA(result, { 
  width = 640, 
  height = 400,
  ldX = 1,
  ldY = 2
} = {}) {
  const { scores } = result;
  
  // Check if we have 2 discriminant axes
  const hasLD2 = scores[0] && scores[0].ld2 !== undefined;
  
  if (!hasLD2 && ldY === 2) {
    // Fall back to 1D plot
    const data = scores.map((score, i) => ({
      x: score.ld1,
      y: 0,
      class: score.class,
      index: i
    }));
    
    return attachShow({
      type: 'lda',
      width,
      height: height / 2,
      data: { scores: data },
      axes: {
        x: { label: 'LD1' },
        y: { label: '', domain: [-1, 1] }
      },
      marks: [
        {
          type: 'dot',
          data: 'scores',
          x: 'x',
          y: 'y',
          fill: 'class',
          r: 5
        }
      ]
    });
  }
  
  // 2D plot
  const data = scores.map((score, i) => ({
    x: score[`ld${ldX}`],
    y: score[`ld${ldY}`],
    class: score.class,
    index: i
  }));
  
  return attachShow({
    type: 'lda',
    width,
    height,
    data: { scores: data },
    axes: {
      x: { label: `LD${ldX}`, grid: true },
      y: { label: `LD${ldY}`, grid: true }
    },
    marks: [
      {
        type: 'dot',
        data: 'scores',
        x: 'x',
        y: 'y',
        fill: 'class',
        r: 5
      }
    ]
  });
}

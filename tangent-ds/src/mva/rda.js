/**
 * Redundancy Analysis (RDA)
 * Constrained ordination - PCA on fitted values from multiple regression
 */

import { toMatrix, solveLeastSquares, Matrix } from '../core/linalg.js';
import * as pca from './pca.js';
import { mean } from '../core/math.js';
import { prepareX } from '../core/table.js';

/**
 * Fit RDA model
 * @param {Array<Array<number>>} Y - Response matrix (n x q)
 * @param {Array<Array<number>>} X - Explanatory matrix (n x p)
 * @param {Object} options - {scale: boolean}
 * @returns {Object} RDA model
 */
export function fit(Y, X, options = {}) {
  let scale = options.scale ?? false;
  let responseMatrix = Y;
  let predictorMatrix = X;
  let responseNames = Array.isArray(options.responseNames)
    ? options.responseNames.map((name) => String(name))
    : null;
  let predictorNames = Array.isArray(options.predictorNames)
    ? options.predictorNames.map((name) => String(name))
    : null;

  if (
    Y &&
    typeof Y === "object" &&
    !Array.isArray(Y) &&
    (Y.data || Y.response || Y.responses || Y.predictors || Y.Y)
  ) {
    const opts = Y;
    const data = opts.data;
    const responseCols = opts.response || opts.responses || opts.Y;
    const predictorCols = opts.predictors || opts.X;
    if (!data || !responseCols || !predictorCols) {
      throw new Error("RDA.fit requires data, response columns, and predictor columns.");
    }
    const omitMissing = opts.omit_missing !== undefined ? opts.omit_missing : true;
    scale = opts.scale !== undefined ? opts.scale : scale;

    const responseList = Array.isArray(responseCols) ? responseCols : [responseCols];
    const predictorList = Array.isArray(predictorCols) ? predictorCols : [predictorCols];

    const responsePrepInitial = prepareX({
      columns: responseList,
      data,
      omit_missing: omitMissing,
    });
    const predictorPrep = prepareX({
      columns: predictorList,
      data: responsePrepInitial.rows,
      omit_missing: omitMissing,
    });
    const responsePrepAligned = prepareX({
      columns: responseList,
      data: predictorPrep.rows,
      omit_missing: false,
    });

    responseMatrix = responsePrepAligned.X;
    predictorMatrix = predictorPrep.X;
    responseNames = responsePrepAligned.columns.map((name) => String(name));
    predictorNames = predictorPrep.columns.map((name) => String(name));
  }

  if (!responseMatrix || !predictorMatrix) {
    throw new Error('RDA.fit requires response and predictor matrices.');
  }

  const responseData = responseMatrix.map(row => Array.isArray(row) ? row : [row]);
  const explData = predictorMatrix.map(row => Array.isArray(row) ? row : [row]);

  const n = responseData.length;
  const q = responseData[0].length;
  const p = explData[0].length;

  if (n !== explData.length) {
    throw new Error('Y and X must have same number of rows');
  }

  if (n < p + 2) {
    throw new Error('Need more samples than explanatory variables');
  }

  const YMeans = [];
  const XMeans = [];

  for (let j = 0; j < q; j++) {
    const col = responseData.map(row => row[j]);
    YMeans.push(mean(col));
  }

  for (let j = 0; j < p; j++) {
    const col = explData.map(row => row[j]);
    XMeans.push(mean(col));
  }

  const YCentered = responseData.map(row =>
    row.map((val, j) => val - YMeans[j])
  );

  const XCentered = explData.map(row =>
    row.map((val, j) => val - XMeans[j])
  );

  const YFitted = [];
  const YResiduals = [];
  const coefficients = [];

  for (let j = 0; j < q; j++) {
    const yCol = YCentered.map(row => row[j]);

    const XMat = new Matrix(XCentered);
    const yVec = Matrix.columnVector(yCol);
    const betaVec = solveLeastSquares(XMat, yVec);
    const beta = betaVec.to1DArray();

    coefficients.push(beta);

    const fitted = [];
    const residuals = [];
    for (let i = 0; i < n; i++) {
      let yhat = 0;
      for (let k = 0; k < p; k++) {
        yhat += XCentered[i][k] * beta[k];
      }
      fitted.push(yhat);
      residuals.push(yCol[i] - yhat);
    }

    YFitted.push(fitted);
    YResiduals.push(residuals);
  }

  const fittedMatrix = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < q; j++) {
      row.push(YFitted[j][i]);
    }
    fittedMatrix.push(row);
  }

  const pcaModel = pca.fit(fittedMatrix, {
    scale,
    center: false,
    columns: responseNames || undefined,
  });

  const canonicalScores = pcaModel.scores.map(score => {
    const newScore = {};
    Object.keys(score).forEach(key => {
      if (key.startsWith('pc')) {
        const num = key.slice(2);
        newScore[`rda${num}`] = score[key];
      }
    });
    return newScore;
  });

  const canonicalLoadings = pcaModel.loadings.map(loading => {
    const newLoading = { variable: loading.variable };
    Object.keys(loading).forEach(key => {
      if (key.startsWith('pc')) {
        const num = key.slice(2);
        newLoading[`rda${num}`] = loading[key];
      }
    });
    return newLoading;
  });

  let totalInertia = 0;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < n; i++) {
      totalInertia += YCentered[i][j] ** 2;
    }
  }
  totalInertia /= n;

  let explainedInertia = 0;
  for (let j = 0; j < q; j++) {
    for (let i = 0; i < n; i++) {
      explainedInertia += fittedMatrix[i][j] ** 2;
    }
  }
  explainedInertia /= n;

  const constrainedVariance = explainedInertia / totalInertia;

  const model = {
    canonicalScores,
    canonicalLoadings,
    eigenvalues: pcaModel.eigenvalues,
    varianceExplained: pcaModel.varianceExplained,
    constrainedVariance,
    coefficients,
    YMeans,
    XMeans,
    n,
    p,
    q
  };

  if (responseNames) {
    model.responseNames = responseNames;
  }
  if (predictorNames) {
    model.predictorNames = predictorNames;
  }

  return model;
}

/**
 * Transform new data using fitted RDA model
 * @param {Object} model - Fitted RDA model
 * @param {Array<Array<number>>} Y - New response data
 * @param {Array<Array<number>>} X - New explanatory data
 * @returns {Array<Object>} Canonical scores
 */
export function transform(model, Y, X) {
  const { coefficients, YMeans, XMeans, canonicalLoadings } = model;
  
  const responseData = Y.map(row => Array.isArray(row) ? row : [row]);
  const explData = X.map(row => Array.isArray(row) ? row : [row]);
  
  const n = responseData.length;
  const q = responseData[0].length;
  const p = explData[0].length;
  
  // Center data
  const YCentered = responseData.map(row => 
    row.map((val, j) => val - YMeans[j])
  );
  
  const XCentered = explData.map(row => 
    row.map((val, j) => val - XMeans[j])
  );
  
  // Compute fitted values
  const fittedMatrix = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < q; j++) {
      let yhat = 0;
      for (let k = 0; k < p; k++) {
        yhat += XCentered[i][k] * coefficients[j][k];
      }
      row.push(yhat);
    }
    fittedMatrix.push(row);
  }
  
  // Extract loading matrix
  const nAxes = canonicalLoadings[0] ? Object.keys(canonicalLoadings[0]).length - 1 : 0;
  const loadingMatrix = [];
  for (let j = 0; j < nAxes; j++) {
    const col = canonicalLoadings.map(l => l[`rda${j + 1}`]);
    loadingMatrix.push(col);
  }
  
  // Project onto canonical axes
  const scores = [];
  for (const row of fittedMatrix) {
    const score = {};
    for (let j = 0; j < nAxes; j++) {
      let sum = 0;
      for (let k = 0; k < q; k++) {
        sum += row[k] * loadingMatrix[j][k];
      }
      score[`rda${j + 1}`] = sum;
    }
    scores.push(score);
  }
  
  return scores;
}

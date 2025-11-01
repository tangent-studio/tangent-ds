/**
 * Canonical Correlation Analysis (CCA)
 *
 * Computes pairs of canonical variates that maximise correlation between
 * two multivariate datasets X and Y.
 */

import { Matrix, covarianceMatrix, eig, svd } from '../core/linalg.js';
import { normalize } from '../core/table.js';

const EPSILON = 1e-10;

/**
 * Fit CCA model.
 *
 * Accepts either numeric matrices (fit(XMatrix, YMatrix, options)) or a declarative
 * object: fit({ X: ['col1', ...], Y: ['colA', ...], data, omit_missing, center, scale }).
 */
export function fit(X, Y = null, options = {}) {
  if (
    X &&
    typeof X === 'object' &&
    !Array.isArray(X) &&
    (X.data || X.X || X.Y || X.columnsX || X.columnsY)
  ) {
    const opts = { ...X };
    return fitFromDeclarative(opts);
  }

  if (!Array.isArray(X) || !Array.isArray(Y)) {
    throw new Error('CCA.fit expects numeric arrays or a declarative object with data/X/Y.');
  }

  return computeCCA(
    X,
    Y,
    {
      center: options.center !== undefined ? options.center : true,
      scale: options.scale !== undefined ? options.scale : false,
      columnsX: options.columnsX || null,
      columnsY: options.columnsY || null
    }
  );
}

function fitFromDeclarative({
  data,
  X,
  Y,
  columnsX,
  columnsY,
  omit_missing = true,
  center = true,
  scale = false
} = {}) {
  if (!data) {
    throw new Error('CCA.fit declarative usage requires `data`.');
  }

  const xCols = normalizeColumns(X || columnsX);
  const yCols = normalizeColumns(Y || columnsY);

  const rows = normalize(data);
  const filtered = [];

  for (const row of rows) {
    if (!row) continue;
    let skip = false;
    for (const col of xCols) {
      const value = row[col];
      if (!isFiniteNumber(value)) {
        skip = true;
        break;
      }
    }
    if (skip && omit_missing) continue;

    for (const col of yCols) {
      const value = row[col];
      if (!isFiniteNumber(value)) {
        skip = true;
        break;
      }
    }

    if (skip) {
      if (!omit_missing) {
        throw new Error('CCA.fit found missing or non-numeric values. Set omit_missing=true to skip rows.');
      }
      continue;
    }

    filtered.push(row);
  }

  if (filtered.length === 0) {
    throw new Error('CCA.fit: no valid rows after filtering.');
  }

  const XMatrix = filtered.map(row => xCols.map(col => row[col]));
  const YMatrix = filtered.map(row => yCols.map(col => row[col]));

  return computeCCA(XMatrix, YMatrix, {
    center,
    scale,
    columnsX: xCols,
    columnsY: yCols
  });
}

function computeCCA(X, Y, {
  center = true,
  scale = false,
  columnsX = null,
  columnsY = null
} = {}) {
  if (!Array.isArray(X) || !Array.isArray(Y)) {
    throw new Error('CCA.fit expects numeric matrices.');
  }
  if (X.length !== Y.length) {
    throw new Error('CCA.fit requires X and Y to have the same number of observations.');
  }
  if (X.length < 2) {
    throw new Error('CCA.fit requires at least two observations.');
  }

  const processedX = preprocessMatrix(X, center, scale);
  const processedY = preprocessMatrix(Y, center, scale);

  const n = processedX.matrix.length;
  const p = processedX.matrix[0].length;
  const q = processedY.matrix[0].length;

  if (n <= Math.max(p, q)) {
    throw new Error('CCA.fit requires more observations than variables in each set.');
  }

  const XMat = new Matrix(processedX.matrix);
  const YMat = new Matrix(processedY.matrix);

  const Sxx = covarianceMatrix(XMat, false);
  const Syy = covarianceMatrix(YMat, false);
  const Sxy = XMat.transpose().mmul(YMat).div(n - 1);
  const Syx = Sxy.transpose();

  const invSqrtSxx = symmetricInverseSqrt(Sxx);
  const invSqrtSyy = symmetricInverseSqrt(Syy);

  const T = invSqrtSxx.mmul(Sxy).mmul(invSqrtSyy);
  const { U, s, V } = svd(T);

  const components = Math.min(p, q);
  const correlations = s.slice(0, components).map((value) => clamp(Math.abs(value), 0, 1));

  const Ax = invSqrtSxx.mmul(U);
  const Ay = invSqrtSyy.mmul(V);

  const xWeightsMat = Ax.subMatrix(0, Ax.rows - 1, 0, components - 1);
  const yWeightsMat = Ay.subMatrix(0, Ay.rows - 1, 0, components - 1);

  const xScoresMat = XMat.mmul(xWeightsMat);
  const yScoresMat = YMat.mmul(yWeightsMat);

  const columnNamesX = columnsX || defaultNames('x', p);
  const columnNamesY = columnsY || defaultNames('y', q);

  const xWeights = matrixToVariableLoadings(xWeightsMat, columnNamesX, components, 'cca');
  const yWeights = matrixToVariableLoadings(yWeightsMat, columnNamesY, components, 'cca');

  const xScores = matrixToScoreObjects(xScoresMat, components, 'cca');
  const yScores = matrixToScoreObjects(yScoresMat, components, 'cca');

  return {
    type: 'cca',
    nSamples: n,
    nFeaturesX: p,
    nFeaturesY: q,
    nComponents: components,
    correlations,
    xWeights,
    yWeights,
    xScores,
    yScores,
    xMeans: processedX.means,
    xSds: processedX.sds,
    yMeans: processedY.means,
    ySds: processedY.sds,
    center,
    scale,
    columnsX: columnNamesX,
    columnsY: columnNamesY
  };
}

export function transformX(model, X, options = {}) {
  ensureModel(model);
  const matrix = prepareNewData(
    X,
    model.columnsX,
    model.xMeans,
    model.xSds,
    model.center,
    model.scale,
    options
  );

  const weights = loadingsMatrix(model.xWeights, model.columnsX.length, model.nComponents, 'cca');
  const scores = matrix.mmul(weights);
  return matrixToScoreObjects(scores, model.nComponents, 'cca');
}

export function transformY(model, Y, options = {}) {
  ensureModel(model);
  const matrix = prepareNewData(
    Y,
    model.columnsY,
    model.yMeans,
    model.ySds,
    model.center,
    model.scale,
    options
  );

  const weights = loadingsMatrix(model.yWeights, model.columnsY.length, model.nComponents, 'cca');
  const scores = matrix.mmul(weights);
  return matrixToScoreObjects(scores, model.nComponents, 'cca');
}

export function transform(model, X, Y, options = {}) {
  ensureModel(model);
  const xScores = transformX(model, X, options);
  const yScores = transformY(model, Y, options);
  return { xScores, yScores };
}

function preprocessMatrix(data, center, scale) {
  const n = data.length;
  const p = data[0].length;

  const means = new Array(p).fill(0);
  data.forEach(row => {
    row.forEach((value, idx) => {
      means[idx] += value;
    });
  });
  if (center) {
    for (let j = 0; j < p; j++) {
      means[j] /= n;
    }
  } else {
    for (let j = 0; j < p; j++) {
      means[j] = 0;
    }
  }

  const centered = data.map(row =>
    row.map((value, idx) => value - means[idx])
  );

  const sds = new Array(p).fill(1);
  if (scale) {
    for (let j = 0; j < p; j++) {
      let sumSq = 0;
      for (let i = 0; i < n; i++) {
        sumSq += centered[i][j] ** 2;
      }
      const variance = sumSq / (n - 1);
      sds[j] = variance > EPSILON ? Math.sqrt(variance) : 1;
    }
  }

  const processed = centered.map(row =>
    row.map((value, idx) =>
      scale ? value / sds[idx] : value
    )
  );

  return { matrix: processed, means, sds };
}

function symmetricInverseSqrt(matrix) {
  const { values, vectors } = eig(matrix);
  const diagValues = values.map((value) =>
    value > EPSILON ? 1 / Math.sqrt(value) : 0
  );

  const diag = Matrix.diag(diagValues);
  return vectors.mmul(diag).mmul(vectors.transpose());
}

function matrixToVariableLoadings(matrix, names, components, prefix) {
  const result = [];
  for (let i = 0; i < matrix.rows; i++) {
    const entry = { variable: names[i] || `${prefix}${i + 1}` };
    for (let j = 0; j < components; j++) {
      entry[`${prefix}${j + 1}`] = matrix.get(i, j);
    }
    result.push(entry);
  }
  return result;
}

function matrixToScoreObjects(matrix, components, prefix) {
  const scores = [];
  for (let i = 0; i < matrix.rows; i++) {
    const entry = {};
    for (let j = 0; j < components; j++) {
      entry[`${prefix}${j + 1}`] = matrix.get(i, j);
    }
    scores.push(entry);
  }
  return scores;
}

function loadingsMatrix(loadings, rows, cols, prefix) {
  const data = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) =>
      loadings[r][`${prefix}${c + 1}`]
    )
  );
  return new Matrix(data);
}

function prepareNewData(raw, columns, means, sds, center, scale, options) {
  if (
    raw &&
    typeof raw === 'object' &&
    !Array.isArray(raw) &&
    (raw.data || raw.X || raw.columns)
  ) {
    const opts = { ...raw };
    const data = opts.data;
    if (!data) {
      throw new Error('CCA.transform requires `data` when using declarative format.');
    }
    const cols = normalizeColumns(opts.X || opts.columns || columns);
    const rows = normalize(data);
    const matrix = rows.map(row => cols.map(col => row[col]));
    return preprocessForTransform(matrix, columns, means, sds, center, scale);
  }

  if (!Array.isArray(raw)) {
    throw new Error('CCA.transform expects numeric arrays or a declarative object with data.');
  }
  return preprocessForTransform(raw, columns, means, sds, center, scale);
}

function preprocessForTransform(data, columns, means, sds, center, scale) {
  if (!Array.isArray(data) || data.length === 0) {
    throw new Error('CCA.transform received empty data.');
  }
  if (data[0].length !== columns.length) {
    throw new Error('CCA.transform data has incorrect number of columns.');
  }

  const processed = data.map(row =>
    row.map((value, idx) => {
      let centered = value;
      if (center) centered -= means[idx];
      if (scale) centered /= sds[idx] || 1;
      return centered;
    })
  );

  return new Matrix(processed);
}

function normalizeColumns(columns) {
  if (!columns) {
    throw new Error('CCA.fit requires column names for both X and Y.');
  }
  if (typeof columns === 'string') {
    return [columns];
  }
  return columns.slice();
}

function isFiniteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function defaultNames(prefix, count) {
  return Array.from({ length: count }, (_, idx) => `${prefix}${idx + 1}`);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function ensureModel(model) {
  if (!model || model.type !== 'cca') {
    throw new Error('CCA model is invalid. Fit before calling transform.');
  }
}

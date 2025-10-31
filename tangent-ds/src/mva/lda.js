/**
 * Linear Discriminant Analysis (LDA)
 * Implementation mirrors scikit-learn's SVD solver to ensure comparable results.
 */

import { svd, Matrix, solveLeastSquares, eig } from "../core/linalg.js";
import { mean } from "../core/math.js";
import { prepareXY } from "../core/table.js";

function toNumericMatrix(X) {
  return X.map((row) => Array.isArray(row) ? row.map(Number) : [Number(row)]);
}

export function fit(X, y) {
  if (
    X && typeof X === "object" && !Array.isArray(X) &&
    ("X" in X) && ("y" in X) && ("data" in X)
  ) {
    const prepared = prepareXY({
      X: X.X,
      y: X.y,
      data: X.data,
      omit_missing: X.omit_missing !== undefined ? X.omit_missing : true,
    });
    X = prepared.X;
    y = prepared.y;
  }

  const data = toNumericMatrix(X);
  const n = data.length;
  if (n === 0) {
    throw new Error("LDA: empty dataset");
  }
  const p = data[0].length;

  if (n !== y.length) {
    throw new Error("X and y must have same number of samples");
  }

  const classes = [...new Set(y)];
  const k = classes.length;
  if (k < 2) {
    throw new Error("Need at least 2 classes for LDA");
  }

  const classIndices = [];
  const classCounts = [];
  const classMeansOriginal = [];
  const SwOriginal = Array(p).fill(null).map(() => Array(p).fill(0));
  for (const c of classes) {
    const indices = y.map((label, i) => label === c ? i : -1).filter((i) => i !== -1);
    classIndices.push(indices);
    classCounts.push(indices.length);
    const meanVec = new Array(p).fill(0);
    for (const idx of indices) {
      for (let j = 0; j < p; j++) {
        meanVec[j] += data[idx][j];
      }
    }
    for (let j = 0; j < p; j++) {
      meanVec[j] /= indices.length;
    }
    classMeansOriginal.push(meanVec);

    for (const idx of indices) {
      const point = data[idx];
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          SwOriginal[i][j] += (point[i] - meanVec[i]) * (point[j] - meanVec[j]);
        }
      }
    }
  }

  const overallMean = [];
  for (let j = 0; j < p; j++) {
    const col = data.map((row) => row[j]);
    overallMean.push(mean(col));
  }

  const centered = data.map((row) => row.map((val, j) => val - overallMean[j]));
  const centeredMatrix = new Matrix(centered);
  const scaleFactor = 1 / Math.sqrt(Math.max(n - 1, 1));
  const scaledMatrix = centeredMatrix.clone().mul(scaleFactor);

  const { V, s } = svd(scaledMatrix);
  const tol = 1e-12;
  const validIndices = [];
  for (let i = 0; i < s.length; i++) {
    if (s[i] > tol) validIndices.push(i);
  }
  const rank = validIndices.length;
  if (rank === 0) {
    throw new Error("LDA: singular data matrix.");
  }

  const projector = new Matrix(p, rank);
  const invScales = new Array(rank);
  for (let idx = 0; idx < rank; idx++) {
    const col = validIndices[idx];
    const scale = s[col];
    invScales[idx] = scale > tol ? 1 / scale : 0;
    for (let i = 0; i < p; i++) {
      projector.set(i, idx, V.get(i, col));
    }
  }

  const whitenedMatrix = centeredMatrix.mmul(projector);
  for (let i = 0; i < whitenedMatrix.rows; i++) {
    for (let j = 0; j < whitenedMatrix.columns; j++) {
      whitenedMatrix.set(i, j, whitenedMatrix.get(i, j) * invScales[j]);
    }
  }
  const whitenedData = whitenedMatrix.to2DArray();

  const whitenedOverallMean = new Array(rank).fill(0);
  for (const row of whitenedData) {
    for (let j = 0; j < rank; j++) {
      whitenedOverallMean[j] += row[j];
    }
  }
  for (let j = 0; j < rank; j++) {
    whitenedOverallMean[j] /= n;
  }

  const classMeansWhitened = [];
  for (let c = 0; c < k; c++) {
    const indices = classIndices[c];
    const meanW = new Array(rank).fill(0);
    for (const idx of indices) {
      for (let j = 0; j < rank; j++) {
        meanW[j] += whitenedData[idx][j];
      }
    }
    for (let j = 0; j < rank; j++) {
      meanW[j] /= indices.length;
    }
    classMeansWhitened.push(meanW);
  }

  const Sw_w = Array(rank).fill(null).map(() => Array(rank).fill(0));
  const Sb_w = Array(rank).fill(null).map(() => Array(rank).fill(0));

  for (let c = 0; c < k; c++) {
    const indices = classIndices[c];
    const meanW = classMeansWhitened[c];
    for (const idx of indices) {
      for (let i = 0; i < rank; i++) {
        for (let j = 0; j < rank; j++) {
          Sw_w[i][j] += (whitenedData[idx][i] - meanW[i]) *
            (whitenedData[idx][j] - meanW[j]);
        }
      }
    }
    const prior = classCounts[c] / n;
    for (let i = 0; i < rank; i++) {
      for (let j = 0; j < rank; j++) {
        Sb_w[i][j] += prior *
          (meanW[i] - whitenedOverallMean[i]) *
          (meanW[j] - whitenedOverallMean[j]);
      }
    }
  }

  const SwMatrix = new Matrix(Sw_w);
 const SbMatrix = new Matrix(Sb_w);
  const SwInvSb = solveLeastSquares(SwMatrix, SbMatrix);
  const { values: eigenvaluesRaw, vectors: eigenvectorsRaw } = eig(SwInvSb);

  const eigenPairs = eigenvaluesRaw.map((val, idx) => ({
    value: val,
    vector: eigenvectorsRaw.getColumn(idx),
  }));
  eigenPairs.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const nComponents = Math.min(k - 1, eigenPairs.length);
  const sortedEigenvalues = eigenPairs.slice(0, nComponents).map((pair) => pair.value);
  const selectedEigenvectors = new Matrix(rank, nComponents);
  for (let j = 0; j < nComponents; j++) {
    const vec = eigenPairs[j].vector;
    for (let i = 0; i < rank; i++) {
      selectedEigenvectors.set(i, j, vec[i]);
    }
  }

  const scalingMatrix = projector.clone();
  for (let j = 0; j < rank; j++) {
    for (let i = 0; i < p; i++) {
      scalingMatrix.set(i, j, scalingMatrix.get(i, j) * invScales[j]);
    }
  }
  const scalingsMatrix = scalingMatrix.mmul(selectedEigenvectors);

 const discriminantAxes = [];
 for (let j = 0; j < nComponents; j++) {
    const axis = [];
    for (let i = 0; i < p; i++) {
      axis.push(scalingsMatrix.get(i, j));
    }

    // Orientation normalization: align with original eigenvector direction
    const sign = Math.sign(axis.reduce((acc, val) => acc + val, 0)) || 1;
    const normalized = axis.map((v) => v * sign);
    // Scale so within-class variance approximately 1
    const axisMat = new Matrix([normalized]);
    const denom = axisMat.mmul(new Matrix(SwOriginal)).mmul(axisMat.transpose()).get(0, 0);
    const scale = denom > 1e-12 ? 1 / Math.sqrt(denom) : 1;
    discriminantAxes.push(normalized.map((v) => v * scale));
 }

  const projectedMatrix = whitenedMatrix.mmul(selectedEigenvectors);
  const projectedData = projectedMatrix.to2DArray();
  const scores = projectedData.map((row, i) => {
    const score = { class: y[i] };
    row.forEach((value, idx) => {
      score[`ld${idx + 1}`] = value;
    });
    return score;
  });

  const classMeanScores = classIndices.map((indices) => {
    const meanVec = new Array(nComponents).fill(0);
    for (const idx of indices) {
      for (let j = 0; j < nComponents; j++) {
        meanVec[j] += projectedData[idx][j];
      }
    }
    for (let j = 0; j < nComponents; j++) {
      meanVec[j] /= indices.length;
    }
    return meanVec;
  });

  const classStdScores = classIndices.map((indices, classIdx) => {
    const stdVec = new Array(nComponents).fill(0);
    for (const idx of indices) {
      for (let j = 0; j < nComponents; j++) {
        const diff = projectedData[idx][j] - classMeanScores[classIdx][j];
        stdVec[j] += diff * diff;
      }
    }
    for (let j = 0; j < nComponents; j++) {
      stdVec[j] = Math.sqrt(stdVec[j] / Math.max(indices.length, 1));
    }
    return stdVec;
  });

  const loadings = [];
  for (let i = 0; i < p; i++) {
    const loading = { variable: `var${i + 1}` };
    for (let j = 0; j < nComponents; j++) {
      loading[`ld${j + 1}`] = discriminantAxes[j][i];
    }
    loadings.push(loading);
  }

  return {
    scores,
    loadings,
    eigenvalues: sortedEigenvalues,
    discriminantAxes,
    classMeans: classMeansOriginal,
    classes,
    overallMean,
    projector: projector.to2DArray(),
    invScales,
    eigenvectors: selectedEigenvectors.to2DArray(),
    classMeanScores,
    classStdScores,
  };
}

export function transform(model, X) {
  const {
    projector,
    invScales,
    eigenvectors,
    overallMean,
  } = model;

  const data = toNumericMatrix(X);
  const centered = data.map((row) => row.map((val, j) => val - overallMean[j]));
  let projected = new Matrix(centered).mmul(new Matrix(projector));
  for (let i = 0; i < projected.rows; i++) {
    for (let j = 0; j < projected.columns; j++) {
      projected.set(i, j, projected.get(i, j) * invScales[j]);
    }
  }
  projected = projected.mmul(new Matrix(eigenvectors));
  const projectedData = projected.to2DArray();

  return projectedData.map((row) => {
    const score = {};
    row.forEach((value, idx) => {
      score[`ld${idx + 1}`] = value;
    });
    return score;
  });
}

export function predict(model, X) {
  const { classes, classMeanScores } = model;
  const transformed = transform(model, X);
  const scoreVectors = transformed.map((score) =>
    Array.from({ length: Object.keys(score).length }, (_, idx) => score[`ld${idx + 1}`])
  );

  const predictions = [];
  for (const point of scoreVectors) {
    let minDist = Infinity;
    let predictedClass = classes[0];
    for (let c = 0; c < classes.length; c++) {
      const meanVec = classMeanScores[c];
      let dist = 0;
      for (let j = 0; j < point.length; j++) {
        dist += (point[j] - meanVec[j]) ** 2;
      }
      dist = Math.sqrt(dist);
      if (dist < minDist) {
        minDist = dist;
        predictedClass = classes[c];
      }
    }
    predictions.push(predictedClass);
  }

  return predictions;
}

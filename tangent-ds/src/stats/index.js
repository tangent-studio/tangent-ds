/**
 * Stats module exports
 */

import { normal, uniform, gamma, beta } from './distribution.js';
import {
  oneSampleTTest as oneSampleTTestFn,
  twoSampleTTest as twoSampleTTestFn,
  chiSquareTest as chiSquareTestFn,
  oneWayAnova as oneWayAnovaFn
} from './tests.js';

import lmClass from './estimators/lm.js';
import logitClass from './estimators/logit.js';
import lmmClass from './estimators/lmm.js';
import {
  OneSampleTTest,
  TwoSampleTTest,
  ChiSquareTest,
  OneWayAnova
} from './estimators/tests.js';

const lm = lmClass;
const logit = logitClass;
const lmm = lmmClass;

// Alias classes under camelCase names for ergonomic construction
const oneSampleTTest = OneSampleTTest;
const twoSampleTTest = TwoSampleTTest;
const chiSquareTest = ChiSquareTest;
const oneWayAnova = OneWayAnova;

// Preserve functional helpers grouped under a namespace for direct usage
const hypothesis = {
  oneSampleTTest: oneSampleTTestFn,
  twoSampleTTest: twoSampleTTestFn,
  chiSquareTest: chiSquareTestFn,
  oneWayAnova: oneWayAnovaFn
};

export {
  // Distributions
  normal,
  uniform,
  gamma,
  beta,

  // Hypothesis test helper namespace (functional)
  hypothesis,

  // Estimator-style classes
  lm,
  logit,
  lmm,
  OneSampleTTest,
  TwoSampleTTest,
  ChiSquareTest,
  OneWayAnova,
  oneSampleTTest,
  twoSampleTTest,
  chiSquareTest,
  oneWayAnova
};

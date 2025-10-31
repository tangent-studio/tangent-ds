/**
 * Core mathematical utilities and constants
 */

// Mathematical constants
export const EPSILON = 1e-10;
export const PI = Math.PI;
export const E = Math.E;

/**
 * Approximate equality comparison for floating point numbers
 * @param {number} a - First number
 * @param {number} b - Second number
 * @param {number} tolerance - Tolerance for comparison
 * @returns {boolean} True if approximately equal
 */
export function approxEqual(a, b, tolerance = EPSILON) {
  return Math.abs(a - b) < tolerance;
}

/**
 * Guard against non-finite values
 * @param {number} value - Value to check
 * @param {string} name - Name for error message
 * @returns {number} The value if valid
 * @throws {Error} If value is not finite
 */
export function guardFinite(value, name = 'value') {
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be finite, got ${value}`);
  }
  return value;
}

/**
 * Guard against negative values
 * @param {number} value - Value to check
 * @param {string} name - Name for error message
 * @returns {number} The value if valid
 * @throws {Error} If value is negative
 */
export function guardPositive(value, name = 'value') {
  if (value <= 0) {
    throw new Error(`${name} must be positive, got ${value}`);
  }
  return value;
}

/**
 * Guard against values outside [0, 1]
 * @param {number} value - Value to check
 * @param {string} name - Name for error message
 * @returns {number} The value if valid
 * @throws {Error} If value is outside [0, 1]
 */
export function guardProbability(value, name = 'value') {
  if (value < 0 || value > 1) {
    throw new Error(`${name} must be between 0 and 1, got ${value}`);
  }
  return value;
}

/**
 * Sum of array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Sum
 */
export function sum(arr) {
  return arr.reduce((a, b) => a + b, 0);
}

/**
 * Mean of array
 * @param {number[]} arr - Array of numbers
 * @returns {number} Mean
 */
export function mean(arr) {
  return sum(arr) / arr.length;
}

/**
 * Variance of array
 * @param {number[]} arr - Array of numbers
 * @param {boolean} sample - If true, use sample variance (n-1)
 * @returns {number} Variance
 */
export function variance(arr, sample = true) {
  const m = mean(arr);
  const squaredDiffs = arr.map(x => (x - m) ** 2);
  const divisor = sample ? arr.length - 1 : arr.length;
  return sum(squaredDiffs) / divisor;
}

/**
 * Standard deviation of array
 * @param {number[]} arr - Array of numbers
 * @param {boolean} sample - If true, use sample variance (n-1)
 * @returns {number} Standard deviation
 */
export function stddev(arr, sample = true) {
  return Math.sqrt(variance(arr, sample));
}

/**
 * ML utilities including seeded random number generation
 */

let seed = Date.now();

/**
 * Set random seed for reproducibility
 * @param {number} value - Seed value
 */
export function setSeed(value) {
  seed = value;
}

/**
 * Seeded random number generator (LCG)
 * @returns {number} Random number between 0 and 1
 */
export function random() {
  seed = (seed * 9301 + 49297) % 233280;
  return seed / 233280;
}

/**
 * Random integer in range [min, max)
 * @param {number} min - Minimum value (inclusive)
 * @param {number} max - Maximum value (exclusive)
 * @returns {number} Random integer
 */
export function randomInt(min, max) {
  return Math.floor(random() * (max - min)) + min;
}

/**
 * Shuffle array in place using seeded random
 * @param {Array} arr - Array to shuffle
 * @returns {Array} Shuffled array
 */
export function shuffle(arr) {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = randomInt(0, i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/**
 * Sample without replacement
 * @param {Array} arr - Array to sample from
 * @param {number} k - Number of samples
 * @returns {Array} Sampled elements
 */
export function sample(arr, k) {
  const shuffled = shuffle(arr);
  return shuffled.slice(0, k);
}

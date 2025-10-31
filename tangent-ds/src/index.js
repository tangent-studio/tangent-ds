/**
 * @tangent.to/ds - A minimalist, browser-friendly data-science library
 * 
 * Main entry point exporting core, stats, ml, mva, and plot namespaces
 */

import * as core from './core/index.js';
import * as stats from './stats/index.js';
import * as ml from './ml/index.js';
import * as mva from './mva/index.js';
import * as plot from './plot/index.js';

export { core, stats, ml, mva, plot };

// Also export individual namespaces for convenience
export default { core, stats, ml, mva, plot };

# Introduction to @tangent.to/ds

## Purpose and Philosophy

`@tangent.to/ds` is a minimalist, browser-friendly data-science library built in pure JavaScript (ESM). It provides essential statistical, machine learning, and visualization tools without heavy dependencies or build complexity.

### Design Principles

**1. Framework-Agnostic**
Works seamlessly in browsers, Node.js, Deno, and Observable Notebook without transpilation or bundling.

**2. ESM-Only**
Uses modern JavaScript modules for clean imports and tree-shaking.

**3. Zero Build Step**
Import and use directly - no webpack, babel, or TypeScript compilation required.

**4. Minimal Dependencies**
Only `ml-matrix` and `simple-statistics` for core functionality. Visualization uses standard Observable Plot.

**5. Declarative APIs**
Functions return configuration objects, not side-effects. You control when and how to render.

## What's Included

### Core Modules

**`core`**: Linear algebra, data tables, and mathematical utilities
**`stats`**: Statistical distributions, hypothesis tests, and regression models
**`ml`**: Machine learning algorithms (clustering, classification, regression)
**`mva`**: Multivariate analysis (PCA, HCA, LDA, RDA)
**`plot`**: Observable Plot configuration generators
**`ml.interpret`**: Model interpretation tools
**`ml.train`**: Training utilities and optimizers
**`ml.tuning`**: Hyperparameter search

### Key Features

- Statistical distributions (normal, gamma, beta, uniform)
- Hypothesis testing (t-tests, ANOVA, chi-square)
- Regression models (OLS, logistic, mixed effects)
- Clustering (K-means, hierarchical)
- Dimensionality reduction (PCA, LDA, RDA)
- Neural networks (MLP with backpropagation)
- Gradient-based optimizers (Adam, RMSProp, Momentum)
- Cross-validation and hyperparameter tuning
- Model interpretation (feature importance, partial dependence)
- Visualization helpers for all analyses

## Installation

```bash
npm install @tangent.to/ds
# or
yarn add @tangent.to/ds
```

## Quick Start

```javascript
import { stats, ml, plot } from '@tangent.to/ds';

// Linear regression
const X = [[1], [2], [3], [4], [5]];
const y = [2, 4, 6, 8, 10];
const model = new stats.lm();
model.fit(X, y);

// Make predictions
const predictions = model.predict([[6], [7]]);
console.log(predictions); // [12, 14]

// K-means clustering
const data = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]];
const clusters = ml.kmeans.fit(data, { k: 2 });
console.log(clusters.labels); // [0, 0, 1, 1, 0, 1]
```

## Use Cases

### Scientific Data Analysis
Explore ecological datasets, climate models, and experimental results with statistical rigor.

### Machine Learning Prototyping
Quickly test hypotheses and build models without heavy frameworks.

### Education
Learn data science concepts with transparent, readable implementations.

### Browser-Based Tools
Build interactive data applications that run entirely client-side.

## Comparison with Other Libraries

### vs TensorFlow.js
- **@tangent.to/ds**: Lightweight, interpretable, statistical focus
- **TensorFlow.js**: Heavy, GPU-accelerated, deep learning focus

### vs scikit-learn (Python)
- **@tangent.to/ds**: JavaScript, browser-compatible, minimal
- **scikit-learn**: Python-only, comprehensive, battle-tested

### vs D3.js
- **@tangent.to/ds**: Analysis and modeling, declarative plots
- **D3.js**: Visualization-first, imperative rendering

## Browser Compatibility

Works in all modern browsers with ES2015+ support:
- Chrome 61+
- Firefox 60+
- Safari 11+
- Edge 79+

## Next Steps

- Read the [Data Structures](./02-data-structures.md) chapter
- Explore [Tangent Notebook examples](../../examples/)
- Check the [API Reference](../api/)

## Getting Help

- **Documentation**: [https://js.tangent.to/docs](https://js.tangent.to/docs)
- **Examples**: See `examples/` directory
- **Issues**: Report bugs or request features on GitHub

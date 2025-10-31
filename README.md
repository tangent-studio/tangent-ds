# @tangent.to/ds

A browser-friendly data science library in modern JavaScript (ESM).

## Package

The npm package is in [`tangent-ds/`](tangent-ds/).

See [tangent-ds/README.md](tangent-ds/README.md) for full API documentation.

## Quick Start

```bash
cd tangent-ds/
npm install
npm run test:run           # 259 tests
node examples/quick-test.js # Smoke test
```

## Repository Structure

```
tangent-ds/               # npm package
├── src/                  # Source code
│   ├── core/            # Linear algebra, tables, math
│   ├── stats/           # Distributions, regression, tests
│   ├── ml/              # Machine learning algorithms
│   ├── mva/             # Multivariate analysis (PCA, LDA, RDA)
│   └── plot/            # Observable Plot configurations
├── tests/               # Test suite (259 tests)
├── examples/            # Working examples
│   ├── quick-test.js   # Smoke test
│   ├── misc/           # Full API demos
│   └── user-guide/     # Sequential pipeline examples
└── dist/               # Browser bundle (generated)

tests_compare-to-python/  # Python/scikit-learn compatibility tests
```

## Features

- **Linear Algebra**: Matrix operations, decompositions
- **Statistics**: Distributions (Normal, Uniform, Gamma, Beta), OLS, Logistic, LMM
- **Hypothesis Tests**: t-tests, Chi-square, ANOVA
- **Machine Learning**: K-Means, KNN, Decision Trees, Random Forest, GAM, MLP
- **Multivariate Analysis**: PCA, LDA, RDA, Hierarchical Clustering
- **Model Selection**: Cross-validation, Grid Search, Metrics
- **Visualization**: Observable Plot configurations

## Publishing

### First Release (v0.1.0)

```bash
cd tangent-ds/
git add .
git commit -m "Release v0.1.0"
git tag v0.1.0
git push origin main --tags
```

### Subsequent Releases

```bash
cd tangent-ds/
npm version patch  # or minor/major
git push origin main --tags
```

GitHub Actions workflow automatically:
1. Runs 259 tests
2. Builds browser bundle
3. Publishes to npmjs.com
4. Deploys documentation site

## License

GPL-3.0

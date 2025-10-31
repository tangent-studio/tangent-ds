# @tangent/ds Examples

This directory contains comprehensive examples for all @tangent/ds functionality.

## 📁 Directory Structure

```
examples/
├── tangent/              # Tangent Notebook examples (⭐ START HERE)
│   ├── 01-machine-learning.js
│   ├── 02-multivariate-analysis.js
│   ├── 03-statistics.js
│   ├── 04-visualization.js
│   └── 05-preprocessing-validation.js
│
├── api_overview.js       # Quick API reference
├── ml_usage.js          # Machine learning examples
├── mva_usage.js         # Multivariate analysis examples
├── stats_usage.js       # Statistical modeling examples
├── interpretation_notebook.js    # Model interpretation
├── tuning_notebook.js   # Hyperparameter tuning
└── model_persistence_notebook.js # Save/load models
```

## 🎯 Recommended: Tangent Notebook Examples

The **`tangent/`** directory contains comprehensive, interactive notebooks designed for Tangent:

### 01. Machine Learning
Covers all ML algorithms with real datasets:
- **Clustering**: K-Means, Hierarchical Clustering
- **Classification**: KNN, Decision Trees, Random Forests
- **Regression**: KNN Regressor, GAMs
- **Neural Networks**: MLPs
- **Model Persistence**: Save/load trained models

### 02. Multivariate Analysis
Ordination methods for high-dimensional data:
- **PCA**: Principal Component Analysis with loadings
- **LDA**: Linear Discriminant Analysis for classification
- **RDA**: Redundancy Analysis for constrained ordination
- **HCA**: Hierarchical Clustering with dendrograms
- **Unified ordiplot**: Consistent visualization across all methods

### 03. Statistics
Statistical modeling and inference:
- **Linear Regression**: Simple, multiple, polynomial
- **Logistic Regression**: Binary classification with probabilities
- **Linear Mixed Models**: Random and fixed effects
- **Statistical Tests**: t-tests, ANOVA, chi-square
- **Model Diagnostics**: Residuals, R², significance testing

### 04. Visualization
All plot types (requires Tangent Notebook to render):
- **Ordination Plots**: PCA biplots, LDA plots, scree plots
- **Classification Metrics**: ROC curves, precision-recall, confusion matrices, calibration
- **Model Interpretation**: Feature importance, partial dependence
- **Diagnostics**: Residual plots, Q-Q plots, correlation matrices
- **Dendrograms**: Hierarchical clustering trees

### 05. Preprocessing & Validation
Data preparation and model evaluation:
- **Scaling**: StandardScaler, MinMaxScaler, Normalizer
- **Encoding**: LabelEncoder, OneHotEncoder
- **Feature Engineering**: PolynomialFeatures
- **Cross-Validation**: K-Fold, Stratified, Leave-One-Out
- **Hyperparameter Tuning**: GridSearchCV, RandomSearchCV
- **Complete Pipelines**: End-to-end workflows

## 🚀 How to Run

### Option 1: Tangent Notebook (Recommended for visualizations)

**This is the ONLY way to see actual plot visualizations!**

```bash
# Terminal 1: Watch tangent-ds for live updates
cd tangent-ds
npm run watch

# Terminal 2: Link and run Tangent
npm link
cd ../tangent-notebook/frontend
npm link @tangent/ds
npm run dev
```

Then in browser (http://localhost:5173):
1. Open any example from `tangent-ds/examples/tangent/`
2. Run cells with `Cmd/Ctrl + Enter`
3. See beautiful interactive plots!

### Option 2: Zed + Deno REPL (Fast testing without plots)

**Best for rapid algorithm testing and debugging:**

```bash
# Terminal 1: Watch mode
cd tangent-ds
npm run watch

# Terminal 2: Serve bundle
cd tangent-ds/dist
npx http-server -p 8080 --cors -c-1
```

In Zed:
1. Create a new `.js` file
2. Import: `const ds = await import('http://localhost:8080/index.js');`
3. Run cells with Deno kernel
4. Get instant inline results!

**Note**: Plot functions only return config objects in Deno REPL, not rendered plots.

### Option 3: Node.js (For legacy examples)

```bash
# Terminal 1: Watch mode
cd tangent-ds
npm run watch

# Terminal 2: Run example
node examples/ml_usage.js

# Or with auto-reload:
npx nodemon --watch src --watch examples examples/ml_usage.js
```

## 📊 What Each Example Demonstrates

### Tangent Format Examples (tangent/)

**Interactive notebooks with:**
- ✅ Markdown cells with explanations
- ✅ Executable code cells
- ✅ Real datasets from Vega
- ✅ Progressive learning (simple → complex)
- ✅ Complete coverage of @tangent/ds

**Variables:**
- Use `globalThis.variable` for cross-cell persistence
- Compatible with Zed's Deno REPL

### Legacy Examples (root directory)

**Simple Node.js scripts:**
- Quick reference for specific features
- Can be run with `node examples/filename.js`
- Good for CI/testing

## 🎨 Visualization Examples

**All visualization examples are in `tangent/04-visualization.js`**

This includes:
- 15+ plot types
- Classification metrics (ROC, confusion matrix, calibration)
- Model interpretation (feature importance, partial dependence)
- Regression diagnostics (residuals, Q-Q plots)
- Correlation heatmaps

**Important**: These only render in Tangent Notebook! In Deno REPL or Node.js, you'll only see the configuration objects.

## 📚 Learning Path

**Recommended order for learning @tangent/ds:**

1. **Start with** `tangent/01-machine-learning.js`
   - Get familiar with basic ML algorithms
   - Learn the fit/predict API pattern

2. **Then** `tangent/02-multivariate-analysis.js`
   - Understand PCA, LDA for dimensionality reduction
   - Learn ordination methods

3. **Next** `tangent/03-statistics.js`
   - Statistical modeling and inference
   - Hypothesis testing

4. **After that** `tangent/05-preprocessing-validation.js`
   - Data preparation workflows
   - Model validation and tuning

5. **Finally** `tangent/04-visualization.js`
   - Visualize everything you've learned
   - Create publication-ready plots

## 🔧 Troubleshooting

### Plots don't show up
- ✅ **Solution**: Use Tangent Notebook, not Deno REPL or Node.js
- Plot functions return Observable Plot configs that only render in Tangent

### Import errors in Tangent
- ✅ **Solution**: Make sure you ran `npm link` in tangent-ds and linked it in Tangent frontend
- Restart Vite dev server after linking

### Module not found in Node.js
- ✅ **Solution**: Run `npm run build` first to generate dist files
- Or use `npm run watch` for live updates

### Deno can't import from localhost
- ✅ **Solution**: Make sure http-server is running on port 8080
- Check CORS is enabled with `--cors` flag

## 💡 Tips

**For algorithm development:**
- Use Zed + Deno REPL for fastest iteration
- Test individual functions in isolation
- No need for plots, just data validation

**For creating documentation:**
- Use Tangent Notebook examples
- Include markdown explanations
- Show actual plot visualizations

**For testing in production:**
- Use `npm link` workflow
- Test in actual Tangent Notebook environment
- Verify plots render correctly

## 📖 API Reference

See individual example files for comprehensive API documentation:
- Each function is demonstrated with real data
- Expected inputs/outputs are shown
- Common use cases are covered
- Edge cases are handled

## 🆘 Need Help?

1. Check `WORKFLOW_GUIDE.md` for setup instructions
2. Look at similar examples in `tangent/` directory
3. Test small pieces in Deno REPL first
4. Use Tangent Notebook for full integration testing

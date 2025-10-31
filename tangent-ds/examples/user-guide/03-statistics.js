// ---
// title: Statistical Modeling with @tangent/ds
// id: statistics
// ---

// %% [markdown]
/*
# Statistical Modeling with @tangent/ds

This notebook covers statistical modeling methods:

- **Linear Regression**: OLS, polynomial, regularization
- **Logistic Regression**: Binary classification with probabilities
- **Linear Mixed Models**: Random and fixed effects
- **Statistical Tests**: t-tests, ANOVA, chi-square
- **Model Diagnostics**: Residuals, R², coefficient significance

These are fundamental for hypothesis testing, inference, and prediction.
*/

// %% [javascript]
import * as ds from "@tangent/ds";

// Load cars dataset
globalThis.carsData = await fetch("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/cars.json")
  .then(r => r.json())
  .then(data => data.filter(d =>
    d.Horsepower != null &&
    d.Miles_per_Gallon != null &&
    d.Weight_in_lbs != null
  ));

console.log(`Loaded ${carsData.length} car records`);

// %% [markdown]
/*
## 1. Linear Regression

Linear regression models the relationship between predictors and a continuous response.
*/

// %% [javascript]
// Simple linear regression: MPG ~ Horsepower
globalThis.X_simple = carsData.map(d => [d.Horsepower]);
globalThis.y_mpg = carsData.map(d => d.Miles_per_Gallon);

globalThis.lm_simple = ds.stats.lm.fit(X_simple, y_mpg, { intercept: true });

console.log("Simple Linear Regression (MPG ~ Horsepower):");
console.log(`  Intercept: ${lm_simple.coefficients[0].toFixed(2)}`);
console.log(`  Horsepower coefficient: ${lm_simple.coefficients[1].toFixed(4)}`);
console.log(`  R²: ${lm_simple.r2.toFixed(3)}`);
console.log(`  Adjusted R²: ${lm_simple.adjustedR2.toFixed(3)}`);

// Interpretation
const hp_effect = lm_simple.coefficients[1];
console.log(`\nInterpretation: Each additional horsepower reduces MPG by ${Math.abs(hp_effect).toFixed(3)}`);

// %% [markdown]
/*
### Multiple Linear Regression

Include multiple predictors for better predictions.
*/

// %% [javascript]
// Multiple regression: MPG ~ Horsepower + Weight
globalThis.X_multiple = carsData.map(d => [d.Horsepower, d.Weight_in_lbs]);

globalThis.lm_multiple = ds.stats.lm.fit(X_multiple, y_mpg, { intercept: true });

console.log("\nMultiple Linear Regression (MPG ~ Horsepower + Weight):");
console.log(`  Intercept: ${lm_multiple.coefficients[0].toFixed(2)}`);
console.log(`  Horsepower: ${lm_multiple.coefficients[1].toFixed(4)}`);
console.log(`  Weight: ${lm_multiple.coefficients[2].toFixed(6)}`);
console.log(`  R²: ${lm_multiple.r2.toFixed(3)}`);
console.log(`  Adjusted R²: ${lm_multiple.adjustedR2.toFixed(3)}`);

console.log(`\nR² improved from ${lm_simple.r2.toFixed(3)} to ${lm_multiple.r2.toFixed(3)}`);

// %% [markdown]
/*
### Model Diagnostics

Examine residuals to check model assumptions.
*/

// %% [javascript]
globalThis.residuals = lm_multiple.residuals;
globalThis.fitted = lm_multiple.fitted;

// Residual statistics
globalThis.residual_mean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
globalThis.residual_std = Math.sqrt(
  residuals.reduce((sum, r) => sum + (r - residual_mean) ** 2, 0) / residuals.length
);

console.log("Residual Diagnostics:");
console.log(`  Mean: ${residual_mean.toFixed(6)} (should be ~0)`);
console.log(`  Std: ${residual_std.toFixed(2)}`);
console.log(`  Min: ${Math.min(...residuals).toFixed(2)}`);
console.log(`  Max: ${Math.max(...residuals).toFixed(2)}`);

// Check for patterns in residuals
console.log("\nResidual plot data available for visualization:");
console.log(`  Fitted values: ${fitted.length} points`);
console.log(`  Residuals: ${residuals.length} points`);

// %% [markdown]
/*
## 2. Polynomial Regression

Model non-linear relationships using polynomial features.
*/

// %% [javascript]
// Fit quadratic model: MPG ~ Horsepower + Horsepower²
globalThis.poly_model = ds.stats.polynomial.fit(
  carsData.map(d => [d.Horsepower]),
  y_mpg,
  { degree: 2, intercept: true }
);

console.log("Polynomial Regression (degree 2):");
console.log(`  Coefficients: [${poly_model.coefficients.map(c => c.toFixed(4)).join(", ")}]`);
console.log(`  R²: ${poly_model.r2.toFixed(3)}`);

// Compare with linear model
console.log(`\nPolynomial R² (${poly_model.r2.toFixed(3)}) > Linear R² (${lm_simple.r2.toFixed(3)})`);
console.log(`Improvement: ${((poly_model.r2 - lm_simple.r2) * 100).toFixed(1)}% more variance explained`);

// %% [markdown]
/*
## 3. Logistic Regression

Logistic regression predicts binary outcomes with probabilities.
*/

// %% [javascript]
// Load iris for binary classification
globalThis.irisData = await fetch("https://cdn.jsdelivr.net/npm/vega-datasets@2/data/iris.json")
  .then(r => r.json());

// Binary classification: setosa vs others
globalThis.X_iris = irisData.map(d => [
  d.sepalLength,
  d.sepalWidth,
  d.petalLength,
  d.petalWidth
]);
globalThis.y_binary = irisData.map(d => d.species === "setosa" ? 1 : 0);

globalThis.logit_model = ds.stats.logit.fit(X_iris, y_binary, {
  intercept: true,
  maxIter: 100,
  tol: 1e-6
});

console.log("Logistic Regression (setosa vs others):");
console.log(`  Intercept: ${logit_model.coefficients[0].toFixed(3)}`);
console.log(`  Coefficients: [${logit_model.coefficients.slice(1).map(c => c.toFixed(3)).join(", ")}]`);
console.log(`  Log-likelihood: ${logit_model.logLikelihood.toFixed(2)}`);
console.log(`  Converged: ${logit_model.converged}`);

// Make predictions
globalThis.test_sample = [[5.1, 3.5, 1.4, 0.2]]; // Typical setosa
globalThis.prob = logit_model.predict(test_sample)[0];
console.log(`\nPrediction for setosa-like sample: ${(prob * 100).toFixed(1)}% probability of being setosa`);

// %% [markdown]
/*
### Classification Performance

Evaluate logistic regression as a classifier.
*/

// %% [javascript]
// Get predictions for all samples
globalThis.probs = logit_model.fitted;
globalThis.y_pred_logit = probs.map(p => p > 0.5 ? 1 : 0);

// Calculate accuracy
globalThis.logit_accuracy = y_binary.filter((y, i) => y === y_pred_logit[i]).length / y_binary.length;
console.log(`Logistic Regression Accuracy: ${(logit_accuracy * 100).toFixed(1)}%`);

// Confusion matrix
globalThis.tp = y_binary.filter((y, i) => y === 1 && y_pred_logit[i] === 1).length;
globalThis.tn = y_binary.filter((y, i) => y === 0 && y_pred_logit[i] === 0).length;
globalThis.fp = y_binary.filter((y, i) => y === 0 && y_pred_logit[i] === 1).length;
globalThis.fn = y_binary.filter((y, i) => y === 1 && y_pred_logit[i] === 0).length;

console.log("\nConfusion Matrix:");
console.log(`  True Positives:  ${tp}`);
console.log(`  True Negatives:  ${tn}`);
console.log(`  False Positives: ${fp}`);
console.log(`  False Negatives: ${fn}`);

// %% [markdown]
/*
## 4. Linear Mixed Models (LMM)

LMMs handle grouped/hierarchical data with random effects.
*/

// %% [javascript]
// Simulate grouped data (e.g., students nested in schools)
globalThis.n_groups = 5;
globalThis.n_per_group = 20;
globalThis.X_lmm = [];
globalThis.y_lmm = [];
globalThis.groups_lmm = [];

for (let g = 0; g < n_groups; g++) {
  const group_effect = (g - 2) * 2; // Random intercept per group
  for (let i = 0; i < n_per_group; i++) {
    const x = Math.random() * 10;
    const y = 2 + 1.5 * x + group_effect + (Math.random() - 0.5) * 2;
    X_lmm.push([x]);
    y_lmm.push(y);
    groups_lmm.push(g);
  }
}

globalThis.lmm = ds.stats.lmm.fit(X_lmm, y_lmm, groups_lmm, {
  intercept: true,
  maxIter: 50
});

console.log("Linear Mixed Model:");
console.log(`  Fixed effects (intercept, slope): [${lmm.fixedEffects.map(c => c.toFixed(2)).join(", ")}]`);
console.log(`  Random effects (group intercepts): [${lmm.randomEffects.map(c => c.toFixed(2)).join(", ")}]`);
console.log(`  Residual variance: ${lmm.residualVariance.toFixed(3)}`);
console.log(`  Random effect variance: ${lmm.randomEffectVariance.toFixed(3)}`);

// %% [markdown]
/*
## 5. Statistical Tests

Perform hypothesis tests for comparing groups and distributions.
*/

// %% [javascript]
// One-sample t-test: Is mean sepal length different from 6.0?
globalThis.sepal_lengths = irisData.map(d => d.sepalLength);
globalThis.ttest_result = ds.stats.tests.oneSampleTTest(sepal_lengths, {
  mu: 6.0,
  alternative: "two-sided"
});

console.log("One-Sample t-test (H₀: μ = 6.0):");
console.log(`  t-statistic: ${ttest_result.statistic.toFixed(3)}`);
console.log(`  p-value: ${ttest_result.pValue.toFixed(4)}`);
console.log(`  Mean: ${ttest_result.mean.toFixed(2)}`);
console.log(`  Decision: ${ttest_result.pValue < 0.05 ? "Reject H₀" : "Fail to reject H₀"}`);

// %% [markdown]
/*
### Two-Sample t-test

Compare two groups.
*/

// %% [javascript]
// Compare sepal length between setosa and versicolor
globalThis.setosa_sl = irisData.filter(d => d.species === "setosa").map(d => d.sepalLength);
globalThis.versicolor_sl = irisData.filter(d => d.species === "versicolor").map(d => d.sepalLength);

globalThis.ttest2_result = ds.stats.tests.twoSampleTTest(setosa_sl, versicolor_sl, {
  alternative: "two-sided"
});

console.log("\nTwo-Sample t-test (setosa vs versicolor):");
console.log(`  t-statistic: ${ttest2_result.statistic.toFixed(3)}`);
console.log(`  p-value: ${ttest2_result.pValue.toFixed(6)}`);
console.log(`  Mean difference: ${(ttest2_result.mean1 - ttest2_result.mean2).toFixed(3)}`);
console.log(`  Decision: ${ttest2_result.pValue < 0.05 ? "Significantly different" : "Not significantly different"}`);

// %% [markdown]
/*
### ANOVA (Analysis of Variance)

Compare means across multiple groups.
*/

// %% [javascript]
// Compare sepal length across all three species
globalThis.setosa_vals = irisData.filter(d => d.species === "setosa").map(d => d.sepalLength);
globalThis.versicolor_vals = irisData.filter(d => d.species === "versicolor").map(d => d.sepalLength);
globalThis.virginica_vals = irisData.filter(d => d.species === "virginica").map(d => d.sepalLength);

globalThis.anova_result = ds.stats.tests.oneWayAnova([setosa_vals, versicolor_vals, virginica_vals]);

console.log("\nOne-Way ANOVA (sepal length across species):");
console.log(`  F-statistic: ${anova_result.statistic.toFixed(3)}`);
console.log(`  p-value: ${anova_result.pValue.toFixed(6)}`);
console.log(`  Between-group variance: ${anova_result.msBetween.toFixed(2)}`);
console.log(`  Within-group variance: ${anova_result.msWithin.toFixed(2)}`);
console.log(`  Decision: ${anova_result.pValue < 0.05 ? "Groups differ significantly" : "No significant difference"}`);

// %% [markdown]
/*
## Summary

@tangent/ds provides comprehensive statistical modeling tools:

**Linear Models**
- `lm.fit(X, y, { intercept })` - Ordinary least squares regression
- `polynomial.fit(X, y, { degree })` - Polynomial regression
- `lmm.fit(X, y, groups, { intercept })` - Linear mixed models

**Generalized Linear Models**
- `logit.fit(X, y, { intercept, maxIter })` - Logistic regression

**Statistical Tests**
- `tests.oneSampleTTest(data, { mu, alternative })` - One-sample t-test
- `tests.twoSampleTTest(data1, data2, { alternative })` - Two-sample t-test
- `tests.oneWayAnova(groups)` - ANOVA for multiple groups
- `tests.chiSquareTest(observed, expected)` - Chi-square goodness of fit

**Model Diagnostics**
- `model.residuals` - Model residuals
- `model.fitted` - Fitted values
- `model.r2` - R-squared
- `model.adjustedR2` - Adjusted R-squared

All models provide:
- Coefficient estimates
- Statistical significance
- Predictions on new data
- Diagnostic measures
*/

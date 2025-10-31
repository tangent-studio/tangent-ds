TODO
- implement Gaussian processes inspired by https://github.com/jmonlabs/jmon-algo/tree/main/src/algorithms/generative/gaussian-processes, then generate python tests and examples
- fix LDA
- manual tests in tangent-notebooks

DOING
- the workflow publishes to npm and pushes the site on tag

Before the workflow can run successfully, you'll need to:
- Add NPM_TOKEN secret to GitHub repository settings (see RELEASE.md for details)
- Ensure GitHub Pages is enabled in repository settings
- Test by pushing a tag: git tag v0.7.1 && git push origin v0.7.1

DONE
- scikit-learn's-like API across the whole pakcage, ml, mvs and plot
- fix logistic regression
- fix kmeans
- test Python similarity
- implement KNN, decision trees, random forest and generalized additive models (both in classification and regression)
- implement python tests and examples for KNN, decision trees, random forest and generalized additive models
- fix PCA, implement SVD
- review train, validation, preprocessing, tunig and interpret
- review plot, one ordiplot module for ordinations (PCA, LDA, RDA), add plot for feature importance, add ROC curves, add other ploting facilitators you might think useful...
- model persistance in json format
- MLP (tensorflowjs) save/load differently
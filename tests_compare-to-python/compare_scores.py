import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=150,
    n_classes=3,
    n_features=5,
    n_informative=3,
    random_state=42,
)

model = LinearDiscriminantAnalysis()
model.fit(X, y)
projected = model.transform(X)

means = projected.mean(axis=0)
stds = projected.std(axis=0)
print('Means:', means)
print('Stds:', stds)

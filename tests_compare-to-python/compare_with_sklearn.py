#!/usr/bin/env python3
"""
Compare key numerical outputs between @tangent/ds estimators (via Node)
and reference implementations built on scikit-learn.

Requires numpy, scikit-learn (already installed in the UV environment)
and Node.js for invoking the tangent-ds estimators.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SkPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans as SkKMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor as SkMLPRegressor

ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = ROOT / "tests_compare-to-python" / "compare_tangent.mjs"


def run_node(mode: str, payload: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        temp_path = fh.name

    try:
        result = subprocess.run(
            ["node", str(NODE_SCRIPT), mode, temp_path],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Node comparison script failed (mode={mode}):\n{exc.stderr}"
        ) from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return json.loads(result.stdout)


def compare_pca(seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    options = {"scale": True, "center": True}

    tangent = run_node("pca", {"X": X.tolist(), "options": options})

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skl_pca = SkPCA(n_components=X.shape[1], svd_solver="full")
    skl_pca.fit(X_scaled)

    eig_diff = np.max(
        np.abs(np.array(tangent["eigenvalues"]) - skl_pca.explained_variance_)
    )
    var_diff = np.max(
        np.abs(
            np.array(tangent["varianceExplained"])
            - skl_pca.explained_variance_ratio_
        )
    )
    cum_diff = np.max(
        np.abs(
            np.array(tangent["cumulativeVariance"])
            - np.cumsum(skl_pca.explained_variance_ratio_)
        )
    )

    return {
        "dataset_shape": X.shape,
        "eig_max_abs_diff": float(eig_diff),
        "variance_ratio_max_abs_diff": float(var_diff),
        "cumulative_max_abs_diff": float(cum_diff),
    }


def compare_linear_regression(seed: int = 24):
    rng = np.random.default_rng(seed)
    n, p = 200, 3
    X = rng.normal(size=(n, p))
    true_coef = np.array([1.5, -2.0, 0.75])
    y = X @ true_coef + 0.5
    y += rng.normal(scale=0.1, size=n)

    tangent = run_node("lm", {"X": X.tolist(), "y": y.tolist(), "options": {"intercept": True}})

    skl_lr = LinearRegression()
    skl_lr.fit(X, y)

    tangent_coef = np.array(tangent["coefficients"])
    skl_coef = np.concatenate(([skl_lr.intercept_], skl_lr.coef_))
    coef_diff = np.max(np.abs(tangent_coef - skl_coef))

    tangent_r2 = tangent["rSquared"]
    skl_r2 = r2_score(y, skl_lr.predict(X))

    return {
        "coef_max_abs_diff": float(coef_diff),
        "r2_diff": float(abs(tangent_r2 - skl_r2)),
        "tangent_summary": tangent,
        "sklearn_intercept": float(skl_lr.intercept_),
        "sklearn_coefficients": skl_lr.coef_.tolist(),
    }


def compare_kmeans(seed: int = 100):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [5, 5], [-4, 3]])
    X = np.vstack([
        center + rng.normal(scale=0.5, size=(80, centers.shape[1]))
        for center in centers
    ])
    options = {"k": len(centers), "seed": seed, "maxIter": 300}
    tangent = run_node("kmeans", {"X": X.tolist(), "options": options})

    skl_kmeans = SkKMeans(
        n_clusters=len(centers),
        n_init=10,
        max_iter=300,
        random_state=seed,
    )
    skl_kmeans.fit(X)

    def sort_rows(arr):
        return np.asarray(sorted(arr, key=lambda row: tuple(row)))

    tangent_centroids = sort_rows(np.array(tangent["centroids"]))
    sklearn_centroids = sort_rows(skl_kmeans.cluster_centers_)

    centroid_diff = np.max(np.abs(tangent_centroids - sklearn_centroids))
    inertia_diff = abs(tangent["inertia"] - skl_kmeans.inertia_)

    return {
        "centroid_max_abs_diff": float(centroid_diff),
        "inertia_diff": float(inertia_diff)
    }


def compare_lda(seed: int = 7):
    rng = np.random.default_rng(seed)
    n_per_class = 60
    means = np.array([[0, 0, 0], [2, 2, -1], [-2, 1, 2]])
    cov = np.array([
        [1.0, 0.2, 0.1],
        [0.2, 1.2, 0.05],
        [0.1, 0.05, 0.8]
    ])
    X = np.vstack([
        rng.multivariate_normal(mean, cov, size=n_per_class)
        for mean in means
    ])
    y = np.repeat(np.arange(len(means)), n_per_class)

    tangent = run_node("lda", {"X": X.tolist(), "y": y.tolist(), "options": {}})

    skl_lda = LinearDiscriminantAnalysis()
    skl_lda.fit(X, y)

    tangent_eigs = np.array(tangent["eigenvalues"])
    total = np.sum(tangent_eigs)
    tangent_ratios = tangent_eigs / total if total > 0 else tangent_eigs
    skl_ratios = skl_lda.explained_variance_ratio_
    eig_diff = np.max(np.abs(tangent_ratios - skl_ratios[: len(tangent_ratios)]))

    classes = tangent.get("classes", list(range(len(skl_lda.classes_))))
    tangent_means = np.array(tangent.get("classMeanScores", []), dtype=float)
    tangent_axes = np.array(tangent.get("discriminantAxes", []), dtype=float).T  # shape (p, components)
    skl_scores = skl_lda.transform(X)

    # Recover tangent scalings in original space
    tangent_scalings = tangent_axes

    # Solve Procrustes alignment between tangent axes and sklearn scalings
    skl_scalings = skl_lda.scalings_
    # Truncate to same number of components
    n_components = min(tangent_scalings.shape[1], skl_scalings.shape[1])
    tangent_scalings = tangent_scalings[:, :n_components]
    skl_scalings = skl_scalings[:, :n_components]

    # Procrustes alignment: find orthogonal R minimizing ||A R - B||_F
    U, _, Vt = np.linalg.svd(tangent_scalings.T @ skl_scalings, full_matrices=False)
    R = U @ Vt
    aligned_axes = tangent_scalings @ R

    # Align tangent scores using the same rotation
    tangent_scores_arr = np.array([[row.get(f"ld{j + 1}", 0) for j in range(n_components)] for row in tangent.get("scores", [])])
    aligned_scores = tangent_scores_arr @ R if tangent_scores_arr.size else np.empty((0, n_components))

    # --- NEW: per-component sign correction to resolve sign-ambiguity ---
    # compute a sign per component from alignment with sklearn scalings
    dot = np.sum(aligned_axes * skl_scalings, axis=0)
    signs = np.sign(dot)
    signs[signs == 0] = 1.0
    # apply signs to aligned axes and per-sample aligned scores
    aligned_axes = aligned_axes * signs
    if aligned_scores.size:
        aligned_scores = aligned_scores * signs

    # --- NEW: per-component scale correction (tangent may use different axis scaling) ---
    # compute scale factors c_j to best match aligned_axes * c_j ~ skl_scalings (least-squares on each component)
    num = np.sum(aligned_axes * skl_scalings, axis=0)
    den = np.sum(aligned_axes * aligned_axes, axis=0)
    scales = np.ones_like(num)
    nonzero = den != 0
    scales[nonzero] = num[nonzero] / den[nonzero]
    # avoid pathological scales
    scales = np.where(np.isfinite(scales), scales, 1.0)
    # apply scales to axes and scores
    aligned_axes = aligned_axes * scales
    if aligned_scores.size:
        aligned_scores = aligned_scores * scales

    tangent_class_means = np.array(tangent.get("classMeanScores", []), dtype=float)
    tangent_class_stds = np.array(tangent.get("classStdScores", []), dtype=float) if tangent.get("classStdScores") is not None else None

    aligned_class_means = None
    aligned_class_stds = None
    if tangent_class_means.size:
        # Normalize orientation to (n_classes, n_components)
        if tangent_class_means.ndim == 1:
            tangent_class_means = tangent_class_means.reshape(-1, n_components)
        if tangent_class_means.shape[1] != n_components and tangent_class_means.shape[0] == n_components:
            tangent_class_means = tangent_class_means.T
        tangent_class_means = tangent_class_means[:, :n_components]
        aligned_class_means = tangent_class_means @ R

        if aligned_class_means is not None:
            # apply same per-component scales to class means
            aligned_class_means = aligned_class_means * scales

        if tangent_class_stds is not None and tangent_class_stds.size:
            if tangent_class_stds.ndim == 1:
                tangent_class_stds = tangent_class_stds.reshape(-1, n_components)
            if tangent_class_stds.shape[1] != n_components and tangent_class_stds.shape[0] == n_components:
                tangent_class_stds = tangent_class_stds.T
            aligned_class_stds = tangent_class_stds[:, :n_components]

        # Reorder aligned_class_means/stds to match sklearn class ordering (skl_lda.classes_)
        tangent_classes = list(tangent.get("classes", []))
        if tangent_classes:
            # build arrays in sklearn class order
            ordered_means = []
            ordered_stds = []
            for skl_cls in skl_lda.classes_:
                if skl_cls in tangent_classes:
                    idx = tangent_classes.index(skl_cls)
                    ordered_means.append(aligned_class_means[idx])
                    if aligned_class_stds is not None:
                        ordered_stds.append(aligned_class_stds[idx])
                    else:
                        ordered_stds.append(np.full(n_components, np.nan, dtype=float))
                else:
                    # missing class in tangent output -> pad with NaN
                    ordered_means.append(np.full(n_components, np.nan, dtype=float))
                    ordered_stds.append(np.full(n_components, np.nan, dtype=float))
            aligned_class_means = np.vstack(ordered_means)
            aligned_class_stds = np.vstack(ordered_stds) if aligned_class_stds is not None else None

            # --- NEW: apply the same per-component sign flips to the reordered class-means ---
            if 'signs' in locals() and aligned_class_means is not None:
                aligned_class_means = aligned_class_means * signs
                if aligned_class_stds is not None:
                    aligned_class_stds = aligned_class_stds

    # Map classes to indices (classes may be provided by tangent)
    class_list = list(classes)
    class_index = {lab: i for i, lab in enumerate(class_list)}

    # Compute class means/stds for both aligned tangent scores and sklearn scores
    skl_means = []
    skl_stds = []
    aligned_means = []
    aligned_stds = []

    for label in classes:
        mask = (y == label)
        skl_proj = skl_scores[mask][:, :n_components]

        # Preferred: use provided per-class means
        if aligned_class_means is not None:
            idx = class_index[label]
            aligned_means.append(aligned_class_means[idx])
            if aligned_class_stds is not None:
                aligned_stds.append(aligned_class_stds[idx])
            else:
                aligned_stds.append(np.full(n_components, np.nan, dtype=float))
        else:
            # If aligned_scores has one row per sample, index by mask
            if aligned_scores.shape[0] == X.shape[0]:
                aligned_proj = aligned_scores[mask]
                aligned_means.append(aligned_proj.mean(axis=0))
                aligned_stds.append(aligned_proj.std(axis=0))
            # If aligned_scores has one row per class, use that as class means
            elif aligned_scores.shape[0] == len(classes):
                # Determine index for this class (tangent ordering might match classes variable)
                idx = class_index[label]
                aligned_means.append(aligned_scores[idx])
                aligned_stds.append(np.full(n_components, np.nan, dtype=float))
            else:
                # Cannot recover per-class projections from tangent output: fill with NaN to avoid crashing
                aligned_means.append(np.full(n_components, np.nan, dtype=float))
                aligned_stds.append(np.full(n_components, np.nan, dtype=float))

        skl_means.append(skl_proj.mean(axis=0))
        skl_stds.append(skl_proj.std(axis=0))

    skl_means = np.array(skl_means)
    skl_stds = np.array(skl_stds)
    aligned_means = np.array(aligned_means)
    aligned_stds = np.array(aligned_stds)

    # Compute diffs using nan-aware maxima (in case some classes lack tangent info)
    with np.errstate(invalid='ignore'):
        mean_diff = float(np.nanmax(np.abs(aligned_means - skl_means)))
        std_diff = float(np.nanmax(np.abs(aligned_stds - skl_stds)))

    # Add lightweight diagnostics to help inspect remaining mismatches
    diagnostics = {
        "sklearn_classes": list(skl_lda.classes_),
        "tangent_classes": list(tangent.get("classes", [])),
        "n_components": int(n_components),
        "aligned_scores_shape": tuple(aligned_scores.shape),
        "aligned_class_means_shape": None if aligned_class_means is None else tuple(aligned_class_means.shape),
        "signs": None if 'signs' not in locals() else signs.tolist(),
        "scales": None if 'scales' not in locals() else scales.tolist()
    }

    return {
        "variance_ratio_max_abs_diff": float(eig_diff),
        "class_mean_max_abs_diff": float(mean_diff),
        "class_std_max_abs_diff": float(std_diff),
        "diagnostics": diagnostics
    }


def compare_logistic(seed: int = 101):
    rng = np.random.default_rng(seed)
    n, p = 150, 3
    X = rng.normal(size=(n, p))
    true_coef = np.array([1.2, -0.8, 0.6])
    logits = 0.3 + X @ true_coef
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(int)

    tangent = run_node(
        "logit",
        {
            "X": X.tolist(),
            "y": y.tolist(),
            "options": {"intercept": True, "maxIter": 200}
        }
    )

    skl_logit = LogisticRegression(
        solver="lbfgs",
        penalty=None,
        max_iter=200,
        random_state=seed,
    )
    skl_logit.fit(X, y)

    tangent_coef = np.array(tangent["coefficients"])
    skl_coef = np.concatenate(([skl_logit.intercept_[0]], skl_logit.coef_[0]))
    coef_diff = np.max(np.abs(tangent_coef - skl_coef))

    tangent_probs = np.array(tangent["probs"]).flatten()
    skl_probs = skl_logit.predict_proba(X)[: len(tangent["probs"]), 1]
    prob_diff = np.max(np.abs(tangent_probs - skl_probs))

    return {
        "coef_max_abs_diff": float(coef_diff),
        "prob_max_abs_diff": float(prob_diff),
        "tangent_coefficients": tangent["coefficients"],
        "sklearn_intercept": float(skl_logit.intercept_[0]),
        "sklearn_coefficients": skl_logit.coef_[0].tolist(),
    }


def compare_polynomial(seed: int = 321):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.uniform(-3, 3, size=(n, 1))
    noise = rng.normal(scale=0.2, size=n)
    y = 1.0 + 2.5 * X[:, 0] + 0.75 * X[:, 0] ** 2 + noise

    tangent = run_node(
        "polynomial",
        {
            "X": X.tolist(),
            "y": y.tolist(),
            "options": {"degree": 2, "intercept": True}
        }
    )

    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    skl_lr = LinearRegression(fit_intercept=False)
    skl_lr.fit(X_poly, y)

    tangent_coef = np.array(tangent["coefficients"])
    skl_coef = skl_lr.coef_
    coef_diff = np.max(np.abs(tangent_coef - skl_coef))

    tangent_r2 = tangent["rSquared"]
    skl_r2 = r2_score(y, skl_lr.predict(X_poly))

    return {
        "coef_max_abs_diff": float(coef_diff),
        "r2_diff": float(abs(tangent_r2 - skl_r2))
    }


def compare_mlp(seed: int = 555):
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.uniform(-2, 2, size=(n, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=n)

    options = {
        "layerSizes": [1, 20, 1],
        "activation": "relu",
        "learningRate": 0.05,
        "epochs": 200,
        "batchSize": 32,
        "seed": seed
    }
    tangent = run_node(
        "mlp",
        {"X": X.tolist(), "y": y.tolist(), "options": options}
    )
    tangent_preds = np.array(tangent["predictions"]).reshape(-1)

    skl_mlp = SkMLPRegressor(
        hidden_layer_sizes=(20,),
        activation="relu",
        learning_rate_init=0.05,
        max_iter=200,
        batch_size=32,
        random_state=seed
    )
    skl_mlp.fit(X, y)
    skl_preds = skl_mlp.predict(X)

    mae = np.mean(np.abs(tangent_preds - skl_preds[: len(tangent_preds)]))

    return {
        "prediction_mae": float(mae)
    }


def compare_knn(seed: int = 123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 3))
    y_clf = (X[:, 0] + X[:, 1] > 0).astype(int)
    y_reg = X[:, 0] - 2 * X[:, 1] + 0.5 * rng.normal(size=120)
    Xtest = rng.normal(size=(20, 3))

    knn_clf = run_node(
        "knn_classifier",
        {
            "X": X.tolist(),
            "y": y_clf.tolist(),
            "Xtest": Xtest.tolist(),
            "options": {"k": 5, "weight": "distance"}
        }
    )
    skl_knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance')
    skl_knn_clf.fit(X, y_clf)
    skl_preds_clf = skl_knn_clf.predict(Xtest)

    clf_diff = np.mean(np.array(knn_clf["predictions"]) != skl_preds_clf)

    knn_reg = run_node(
        "knn_regressor",
        {
            "X": X.tolist(),
            "y": y_reg.tolist(),
            "Xtest": Xtest.tolist(),
            "options": {"k": 4, "weight": "distance"}
        }
    )
    skl_knn_reg = KNeighborsRegressor(n_neighbors=4, weights='distance')
    skl_knn_reg.fit(X, y_reg)
    skl_preds_reg = skl_knn_reg.predict(Xtest)

    reg_mae = np.mean(np.abs(np.array(knn_reg["predictions"]) - skl_preds_reg))

    return {
        "classification_error_rate": float(clf_diff),
        "regression_mae": float(reg_mae)
    }


def compare_decision_tree(seed: int = 246):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(150, 4))
    y_clf = (X[:, 0] * X[:, 1] > 0).astype(int)
    y_reg = X[:, 0] ** 2 - X[:, 1] + rng.normal(scale=0.1, size=150)
    Xtest = rng.normal(size=(30, 4))

    tree_clf = run_node(
        "decision_tree_classifier",
        {
            "X": X.tolist(),
            "y": y_clf.tolist(),
            "Xtest": Xtest.tolist(),
            "options": {"maxDepth": 5, "minSamplesSplit": 4}
        }
    )
    skl_tree_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=seed)
    skl_tree_clf.fit(X, y_clf)
    skl_preds_clf = skl_tree_clf.predict(Xtest)
    clf_err = np.mean(np.array(tree_clf["predictions"]) != skl_preds_clf)

    tree_reg = run_node(
        "decision_tree_regressor",
        {
            "X": X.tolist(),
            "y": y_reg.tolist(),
            "Xtest": Xtest.tolist(),
            "options": {"maxDepth": 5, "minSamplesSplit": 4}
        }
    )
    skl_tree_reg = DecisionTreeRegressor(max_depth=5, min_samples_split=4, random_state=seed)
    skl_tree_reg.fit(X, y_reg)
    skl_preds_reg = skl_tree_reg.predict(Xtest)
    reg_mae = np.mean(np.abs(np.array(tree_reg["predictions"]) - skl_preds_reg))

    return {
        "classification_error_rate": float(clf_err),
        "regression_mae": float(reg_mae)
    }


def compare_random_forest(seed: int = 369):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 5))
    y_clf = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    y_reg = np.sin(X[:, 0]) + 0.3 * X[:, 1] + rng.normal(scale=0.1, size=200)
    Xtest = rng.normal(size=(40, 5))

    forest_clf = run_node(
        "random_forest_classifier",
        {
            "X": X.tolist(),
            "y": y_clf.tolist(),
            "Xtest": Xtest.tolist(),
            "options": {"nEstimators": 50, "maxDepth": 7, "seed": seed}
        }
    )
    skl_forest_clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=7,
        random_state=seed
    )
    skl_forest_clf.fit(X, y_clf)
    skl_preds_clf = skl_forest_clf.predict(Xtest)
    clf_err = np.mean(np.array(forest_clf["predictions"]) != skl_preds_clf)

    forest_reg = run_node(
        "random_forest_regressor",
        {
            "X": X.tolist(),
            "y": y_reg.tolist(),
            "Xtest": Xtest.tolist(),
            "options": {"nEstimators": 60, "maxDepth": 6, "seed": seed}
        }
    )
    skl_forest_reg = RandomForestRegressor(
        n_estimators=60,
        max_depth=6,
        random_state=seed
    )
    skl_forest_reg.fit(X, y_reg)
    skl_preds_reg = skl_forest_reg.predict(Xtest)
    reg_mae = np.mean(np.abs(np.array(forest_reg["predictions"]) - skl_preds_reg))

    return {
        "classification_error_rate": float(clf_err),
        "regression_mae": float(reg_mae)
    }


def compute_knots(values, n_splines):
    sorted_vals = np.sort(values)
    knots = []
    for i in range(1, n_splines):
        idx = int(np.floor((i / (n_splines + 1)) * (len(sorted_vals) - 1)))
        knots.append(float(sorted_vals[idx]))
    return knots


def spline_basis(value, knots):
    basis = [value]
    for knot in knots:
        diff = value - knot
        basis.append(diff ** 3 if diff > 0 else 0.0)
    return basis


def build_design_matrix(X, knots_per_feature, include_intercept=True):
    rows = []
    for row in X:
        features = []
        if include_intercept:
            features.append(1.0)
        for j, value in enumerate(row):
            features.extend(spline_basis(value, knots_per_feature[j]))
        rows.append(features)
    return np.array(rows)


def compare_gam(seed: int = 777):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(150, 2))
    y_reg = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.1 * rng.normal(size=150)
    y_clf_labels = np.where(np.sin(X[:, 0]) + X[:, 1] > 0, 'B', 'A')
    Xtest = rng.uniform(-2, 2, size=(30, 2))

    options = {"nSplines": 5, "maxIter": 100}

    gam_reg = run_node(
        "gam_regressor",
        {"X": X.tolist(), "y": y_reg.tolist(), "Xtest": Xtest.tolist(), "options": options}
    )

    knots = [compute_knots(X[:, j], options["nSplines"]) for j in range(X.shape[1])]
    design = build_design_matrix(X, knots)
    design_test = build_design_matrix(Xtest, knots)
    skl_lr = LinearRegression(fit_intercept=False)
    skl_lr.fit(design, y_reg)
    skl_preds = skl_lr.predict(design_test)
    reg_mae = np.mean(np.abs(np.array(gam_reg["predictions"]) - skl_preds))

    mapping = {'A': 0, 'B': 1}
    numeric_y = np.array([mapping[label] for label in y_clf_labels])

    gam_clf = run_node(
        "gam_classifier",
        {"X": X.tolist(), "y": y_clf_labels.tolist(), "Xtest": Xtest.tolist(), "options": options}
    )

    skl_logit = LogisticRegression(fit_intercept=False, penalty=None, max_iter=options["maxIter"])
    skl_logit.fit(design, numeric_y)
    skl_probs = skl_logit.predict_proba(design_test)[:, 1]
    tangent_probs = np.array([prob['B'] for prob in gam_clf["probabilities"]])
    clf_mae = np.mean(np.abs(tangent_probs - skl_probs))

    return {
        "regression_mae": float(reg_mae),
        "classification_proba_mae": float(clf_mae)
    }


def main():
    pca_result = compare_pca()
    lm_result = compare_linear_regression()
    kmeans_result = compare_kmeans()
    lda_result = compare_lda()
    logit_result = compare_logistic()
    poly_result = compare_polynomial()
    mlp_result = compare_mlp()
    knn_result = compare_knn()
    tree_result = compare_decision_tree()
    forest_result = compare_random_forest()
    gam_result = compare_gam()

    print("=== PCA comparison (tangent vs scikit-learn) ===")
    for key, value in pca_result.items():
        print(f"{key}: {value}")

    print("\n=== Linear regression comparison ===")
    print(f"Coefficient max abs diff: {lm_result['coef_max_abs_diff']:.4e}")
    print(f"R² difference: {lm_result['r2_diff']:.4e}")
    print("tangent coefficients:", lm_result["tangent_summary"]["coefficients"])
    print("sklearn intercept / coefficients:",
          lm_result["sklearn_intercept"],
          lm_result["sklearn_coefficients"])

    print("\n=== KMeans comparison ===")
    for key, value in kmeans_result.items():
        print(f"{key}: {value}")

    print("\n=== LDA comparison ===")
    for key, value in lda_result.items():
        print(f"{key}: {value}")

    print("\n=== Logistic regression comparison ===")
    print(f"Coefficient max abs diff: {logit_result['coef_max_abs_diff']:.4e}")
    print(f"Probability max abs diff: {logit_result['prob_max_abs_diff']:.4e}")
    print("tangent coefficients:", logit_result["tangent_coefficients"])
    print("sklearn intercept / coefficients:",
          logit_result["sklearn_intercept"],
          logit_result["sklearn_coefficients"])

    print("\n=== Polynomial regression comparison ===")
    for key, value in poly_result.items():
        print(f"{key}: {value}")

    print("\n=== MLP regression comparison ===")
    for key, value in mlp_result.items():
        print(f"{key}: {value}")

    print("\n=== KNN comparison ===")
    for key, value in knn_result.items():
        print(f"{key}: {value}")

    print("\n=== Decision Tree comparison ===")
    for key, value in tree_result.items():
        print(f"{key}: {value}")

    print("\n=== Random Forest comparison ===")
    for key, value in forest_result.items():
        print(f"{key}: {value}")

    print("\n=== GAM comparison ===")
    for key, value in gam_result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

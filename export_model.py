"""Reproduce the validated notebook OFFLINE and export a frozen serving bundle.

Never imported by Streamlit. The notebook and its saved outputs are read only.
Assertions intentionally stop export if data, numerical results or labels drift.
"""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from src.config import ROOT, DATA_PATH, BUNDLE_PATH, BUNDLE_VERSION, FEATURES, PCA_COLUMNS
from src.inference import validate_bundle, predict_customer_segment


def notebook_reference():
    """Extract validation references from saved notebook text, without executing it."""
    notebook = json.loads((ROOT / "customer_segments_completed.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    def source(cell):
        return "".join(cell["source"])
    def find(fragment):
        return next(c for c in cells if fragment in source(c))
    def output(cell):
        return "\n".join("".join(o.get("text", o.get("data", {}).get("text/plain", [])))
                         for o in cell.get("outputs", []))
    sample_source = source(find("indices = ["))
    samples = ast.literal_eval(re.search(r"indices = (\[[^\]]+\])", sample_source)[1])
    centers_text = output(find("log_centers = pca.inverse_transform"))
    centers = [[float(v) for v in line.split()[1:]] for line in centers_text.splitlines()
               if re.match(r"^[01]\s+\d", line)]
    scores_text = source(find("K-Means silhouettes for k=2"))
    km = [float(x) for x in re.findall(r"0\.\d+", scores_text.split("GMM silhouettes")[0])]
    gm = [float(x) for x in re.findall(r"0\.\d+", scores_text.split("GMM silhouettes")[1].split("Two-cluster")[0])]
    predictions_text = output(find('sample_results = samples.copy()'))
    predictions = [int(x) for x in re.findall(r"^\s*(?:3|28|34)\s+([01])\s*$", predictions_text, re.M)]
    assert samples == [3, 28, 34] and predictions == [1, 0, 1]
    assert np.shape(centers) == (2, 6) and len(km) == 5 and len(gm) == 3
    return {"samples": samples, "predictions": predictions, "centers": centers, "km": km, "gm": gm}


def export():
    reference = notebook_reference()
    environment = {p: metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]}
    assert environment["scikit-learn"] == "1.7.2", "Install the pinned requirements before exporting."
    full = pd.read_csv(DATA_PATH)
    assert full.shape == (440, 8) and full.isna().sum().sum() == 0 and full.duplicated().sum() == 0
    raw = full.loc[:, FEATURES]
    assert np.isfinite(raw).all().all() and (raw > 0).all().all()
    logged = np.log(raw)
    q1, q3 = logged.quantile(.25), logged.quantile(.75)
    flags = ((logged < q1 - 1.5 * (q3 - q1)) | (logged > q3 + 1.5 * (q3 - q1))).sum(axis=1)
    removed = flags.index[flags >= 2].tolist()
    assert removed == [65, 66, 75, 128, 154]
    retained = raw.index[~raw.index.isin(removed)]
    good = logged.loc[retained].reset_index(drop=True)
    with threadpool_limits(limits=1):
        full_pca = PCA(n_components=6, svd_solver="full").fit(good)
        pca = PCA(n_components=2, svd_solver="full").fit(good)
        reduced = pd.DataFrame(pca.transform(good), columns=PCA_COLUMNS)
        model = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(reduced)
        labels = model.predict(reduced)
        score = silhouette_score(reduced, labels)
        comparison = []
        for k, expected in zip(range(2, 7), reference["km"]):
            candidate = model if k == 2 else KMeans(n_clusters=k, random_state=42, n_init="auto").fit(reduced)
            value = silhouette_score(reduced, candidate.predict(reduced))
            assert np.isclose(value, expected, atol=5e-7, rtol=0)
            comparison.append({"Algorithm": "K-Means", "Clusters": k, "Silhouette": value})
        for k, expected in zip(range(2, 5), reference["gm"]):
            candidate = GaussianMixture(n_components=k, covariance_type="full", n_init=10, random_state=42).fit(reduced)
            value = silhouette_score(reduced, candidate.predict(reduced))
            assert candidate.converged_ and np.isclose(value, expected, atol=5e-7, rtol=0)
            comparison.append({"Algorithm": "GMM", "Clusters": k, "Silhouette": value})
    centers = np.exp(pca.inverse_transform(model.cluster_centers_))
    np.testing.assert_allclose(centers, reference["centers"], atol=.0051, rtol=0)
    np.testing.assert_allclose(full_pca.explained_variance_ratio_,
                               [.44302505, .26379218, .12306380, .10120908, .04850196, .02040793], atol=5e-9, rtol=0)
    assert np.isclose(score, .426072168, atol=1e-9, rtol=0)
    assert np.bincount(labels).tolist() == [175, 260]
    # Freeze spending-based interpretations before the external Channel join.
    assert (centers[0, [1, 2, 4]] > centers[1, [1, 2, 4]]).all()
    assert (centers[1, [0, 3]] > centers[0, [0, 3]]).all()
    profiles = {
        0: {"name": "Grocery / Retail-Oriented Purchasing",
            "description": "Higher grocery, milk, and detergents/paper spending relative to the other segment. The pattern is more consistent with retail-oriented purchasing; it does not establish business identity."},
        1: {"name": "Fresh / Frozen / Food-Service-Oriented Purchasing",
            "description": "Higher fresh and frozen spending relative to the other segment, with lower grocery and household-product spending. The pattern is more consistent with food-service-oriented purchasing; it does not establish business identity."},
    }
    external = pd.crosstab(pd.Series(labels, index=retained, name="Segment"), full.loc[retained, "Channel"])
    external = external.rename(columns={1: "HoReCa", 2: "Retail"})
    assert external.to_numpy().tolist() == [[40, 135], [254, 6]]
    stats = pd.DataFrame({"min": raw.loc[retained].min(), "max": raw.loc[retained].max(),
                          "q01": raw.loc[retained].quantile(.01), "q99": raw.loc[retained].quantile(.99),
                          "median": raw.median(), "mean": raw.mean()})
    bundle = {
        "version": BUNDLE_VERSION, "pca": pca, "kmeans": model, "features": FEATURES,
        "centers": centers, "profiles": profiles, "pca_variance": full_pca.explained_variance_ratio_,
        "pca_loadings": full_pca.components_, "silhouette": float(score), "statistics": stats,
        "training_coordinates": reduced.to_numpy(), "training_indices": retained.to_numpy(),
        "training_labels": labels, "removed_indices": removed, "outlier_counts": flags.loc[removed].to_dict(),
        "channel_composition": external, "comparison": pd.DataFrame(comparison),
        "sample_indices": reference["samples"], "sample_predictions": reference["predictions"],
        "presets": {"Grocery-oriented · customer 28": raw.loc[28].to_dict(),
                    "Fresh-food-oriented · customer 3": raw.loc[3].to_dict(),
                    "Typical customer · population medians": raw.median().to_dict()},
        "metadata": {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
                     "environment": environment, "training_rows": len(good), "dataset_rows": len(full),
                     "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                     "notebook_sha256": hashlib.sha256((ROOT / "customer_segments_completed.ipynb").read_bytes()).hexdigest(),
                     "source": "customer_segments_completed.ipynb", "random_state": 42, "n_init": "auto",
                     "preprocessing": "np.log; no scaling; PCA(2, svd_solver=full)",
                     "outlier_policy": "Remove rows with >=2 log-space Tukey flags during fitting only"},
    }
    validate_bundle(bundle)
    for idx, expected in zip(reference["samples"], reference["predictions"]):
        assert predict_customer_segment(raw.loc[[idx]], bundle)["segment"] == expected
    BUNDLE_PATH.parent.mkdir(exist_ok=True)
    temporary = BUNDLE_PATH.with_suffix(".tmp")
    joblib.dump(bundle, temporary, compress=3)
    restored = validate_bundle(joblib.load(temporary))
    np.testing.assert_array_equal(restored["kmeans"].predict(reduced), labels)
    temporary.replace(BUNDLE_PATH)
    print(f"Export validated: {len(good)} rows | silhouette {score:.9f} | PCA {pca.explained_variance_ratio_.sum():.6%}")
    print(f"Samples: {dict(zip(reference['samples'], reference['predictions']))}")
    print(f"Saved {BUNDLE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    export()

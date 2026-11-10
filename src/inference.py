"""Pure inference. No fitting, feature selection, or outlier removal occurs here."""
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import InconsistentVersionWarning
from src.config import BUNDLE_PATH, BUNDLE_VERSION, FEATURES, PCA_COLUMNS


def validate_bundle(bundle):
    """Fail closed when the frozen model does not match the serving contract."""
    try:
        if bundle["version"] != BUNDLE_VERSION or bundle["features"] != FEATURES:
            raise ValueError("Unsupported bundle version or feature order.")
        if bundle["metadata"]["environment"]["scikit-learn"] != sklearn.__version__:
            raise ValueError("scikit-learn version differs from the exported model; install requirements.txt.")
        pca, model = bundle["pca"], bundle["kmeans"]
        if not isinstance(pca, PCA) or not isinstance(model, KMeans):
            raise ValueError("Expected fitted PCA and KMeans objects.")
        if (pca.n_components_ != 2 or pca.n_features_in_ != 6 or
                model.n_clusters != 2 or model.n_features_in_ != 2):
            raise ValueError("Model dimensions do not match the validated pipeline.")
        if list(pca.feature_names_in_) != FEATURES or list(model.feature_names_in_) != PCA_COLUMNS:
            raise ValueError("Fitted feature names do not match the serving contract.")
        if pca.svd_solver != "full" or model.random_state != 42 or model.n_init != "auto":
            raise ValueError("Unexpected model configuration.")
        expected_shapes = {"centers": (2, 6), "pca_variance": (6,), "pca_loadings": (6, 6)}
        for key, shape in expected_shapes.items():
            if np.shape(bundle[key]) != shape:
                raise ValueError(f"Invalid {key} dimensions.")
        arrays = [pca.components_, pca.mean_, pca.explained_variance_ratio_,
                  model.cluster_centers_, bundle["centers"], bundle["pca_variance"],
                  bundle["pca_loadings"], bundle["training_coordinates"],
                  bundle["statistics"].to_numpy(), bundle["silhouette"]]
        if not all(np.isfinite(a).all() for a in arrays):
            raise ValueError("Non-finite numerical model artifacts.")
        if set(bundle["profiles"]) != {0, 1} or not (-1 <= bundle["silhouette"] <= 1):
            raise ValueError("Invalid segment metadata.")
        if not np.allclose(bundle["centers"], np.exp(pca.inverse_transform(model.cluster_centers_))):
            raise ValueError("Representative centers do not match the frozen model.")
        n = bundle["metadata"]["training_rows"]
        if np.shape(bundle["training_coordinates"]) != (n, 2):
            raise ValueError("Training coordinates have an invalid shape.")
        if len(bundle["training_indices"]) != n or len(bundle["training_labels"]) != n:
            raise ValueError("Training row alignment is invalid.")
        if not set(np.unique(bundle["training_labels"])).issubset({0, 1}):
            raise ValueError("Invalid historical segment IDs.")
        if list(bundle["statistics"].index) != FEATURES:
            raise ValueError("Distribution metadata has incorrect feature order.")
    except (KeyError, AttributeError, TypeError) as exc:
        raise ValueError(f"Incomplete or incompatible model bundle: {exc}") from exc
    return bundle


def load_bundle(path=BUNDLE_PATH):
    # joblib is pickle-based: load only the trusted, repository-owned artifact.
    if not path.is_file():
        raise FileNotFoundError("Model artifact missing. Run python export_model.py offline, then restart the app.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        return validate_bundle(joblib.load(path))


def predict_customer_segment(customer_df, bundle):
    """Validate exactly one ordered customer and return a structured prediction."""
    if not isinstance(customer_df, pd.DataFrame) or list(customer_df.columns) != FEATURES:
        raise ValueError("Provide exactly these columns in order: " + ", ".join(FEATURES))
    if len(customer_df) != 1:
        raise ValueError("Provide exactly one customer at a time.")
    try:
        values = customer_df.astype(float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("All spending values must be numeric.") from exc
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("All six spending values must be present and finite (no NaN or infinity).")
    if not (values.to_numpy() > 0).all():
        raise ValueError("All annual spending values must be greater than zero for the natural logarithm.")
    coordinates = bundle["pca"].transform(np.log(values))
    reduced = pd.DataFrame(coordinates, columns=PCA_COLUMNS)
    segment = int(bundle["kmeans"].predict(reduced)[0])
    distances = bundle["kmeans"].transform(reduced)[0]
    total = distances.sum()
    proximity = float(distances[1 - segment] / total) if total else 0.5
    stats = bundle["statistics"]
    unusual = (values.iloc[0] < stats["q01"]) | (values.iloc[0] > stats["q99"])
    return {
        "segment": segment, "segment_name": bundle["profiles"][segment]["name"],
        "pca_coordinates": coordinates[0], "centroid_distances": distances,
        "relative_proximity": proximity, "representative_spending": bundle["centers"][segment],
        "unusual_features": unusual.index[unusual].tolist(),
    }

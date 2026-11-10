import ast
import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.config import ROOT, DATA_PATH, FEATURES
from src.inference import load_bundle, predict_customer_segment, validate_bundle


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()
        cls.raw = pd.read_csv(DATA_PATH)[FEATURES]

    def test_artifact_loads_and_matches_notebook(self):
        self.assertEqual(self.bundle["metadata"]["training_rows"], 435)
        self.assertAlmostEqual(self.bundle["silhouette"], .426072168, places=8)
        np.testing.assert_array_equal(np.bincount(self.bundle["training_labels"]), [175, 260])

    def test_valid_prediction_and_distances(self):
        customer = self.raw.median().to_frame().T
        result = predict_customer_segment(customer, self.bundle)
        self.assertIn(result["segment"], [0, 1])
        expected = np.linalg.norm(self.bundle["kmeans"].cluster_centers_ - result["pca_coordinates"], axis=1)
        np.testing.assert_allclose(result["centroid_distances"], expected)
        self.assertEqual(result["segment"], int(np.argmin(expected)))
        self.assertGreaterEqual(result["relative_proximity"], .5)
        self.assertLessEqual(result["relative_proximity"], 1.)

    def test_zero_negative_nonfinite_missing_nonnumeric_fail(self):
        for value in [0, -1, np.nan, np.inf, -np.inf, None, "not a number"]:
            with self.subTest(value=value):
                customer = self.raw.iloc[[0]].astype(object)
                customer.iloc[0, 0] = value
                with self.assertRaises(ValueError):
                    predict_customer_segment(customer, self.bundle)

    def test_exact_schema_and_order(self):
        valid = self.raw.iloc[[0]]
        invalid = [valid.rename(columns={"Fresh": "Other"}), valid[FEATURES[::-1]],
                   valid.drop(columns="Milk"), valid.assign(Channel=1), valid.assign(Region=1),
                   valid.rename(columns={"Milk": "Fresh"}), valid.iloc[:0], self.raw.iloc[:2]]
        for customer in invalid:
            with self.subTest(columns=list(customer.columns), rows=len(customer)):
                with self.assertRaises(ValueError):
                    predict_customer_segment(customer, self.bundle)

    def test_samples_reproduce_notebook(self):
        for index, expected in [(3, 1), (28, 0), (34, 1)]:
            self.assertEqual(predict_customer_segment(self.raw.loc[[index]], self.bundle)["segment"], expected)

    def test_unusual_profiles_are_not_deleted(self):
        result = predict_customer_segment(self.raw.iloc[[0]].astype(float) * 1e8, self.bundle)
        self.assertEqual(result["unusual_features"], FEATURES)
        for index in self.bundle["removed_indices"]:
            self.assertIn(predict_customer_segment(self.raw.loc[[index]], self.bundle)["segment"], [0, 1])

    def test_no_fitting_in_serving_source_or_execution(self):
        for path in [ROOT / "app.py", *(ROOT / "src").glob("*.py")]:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            calls = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
            self.assertFalse({"fit", "fit_transform", "fit_predict", "partial_fit"}.intersection(calls), path.name)
        with patch.object(PCA, "fit", side_effect=AssertionError("PCA fit forbidden")), patch.object(KMeans, "fit", side_effect=AssertionError("KMeans fit forbidden")):
            predict_customer_segment(self.raw.iloc[[3]], load_bundle())

    def test_missing_and_incompatible_artifact(self):
        with self.assertRaises(FileNotFoundError):
            load_bundle(ROOT / "artifacts" / "does-not-exist.joblib")
        for key, value in [("features", FEATURES[::-1]), ("version", "unknown"), ("centers", np.full((2, 6), np.nan))]:
            broken = copy.deepcopy(self.bundle)
            broken[key] = value
            with self.assertRaises(ValueError):
                validate_bundle(broken)


if __name__ == "__main__":
    unittest.main()

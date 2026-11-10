import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.config import ROOT, PAGES


class ApplicationTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def assert_clean(self, app):
        self.assertFalse(app.exception, [str(e.message) for e in app.exception])
        self.assertFalse(app.error, [e.value for e in app.error])

    def test_all_pages_render_without_training(self):
        with patch.object(PCA, "fit", side_effect=AssertionError("Training forbidden")), patch.object(KMeans, "fit", side_effect=AssertionError("Training forbidden")):
            app = self.app()
            self.assert_clean(app)
            for page in PAGES:
                with self.subTest(page=page):
                    app.radio(key="navigation").set_value(page).run()
                    self.assert_clean(app)

    def test_prediction_presets_views_and_stale_results(self):
        app = self.app()
        app.button[0].click().run()
        self.assertEqual(app.radio(key="navigation").value, "Customer Segmentation")
        self.assert_clean(app)
        for preset, expected in [("Grocery-oriented · customer 28", 0), ("Fresh-food-oriented · customer 3", 1)]:
            app.selectbox(key="preset").select(preset).run()
            next(b for b in app.button if b.label == "Load Example Customer").click().run()
            next(b for b in app.button if b.label == "Analyze Customer").click().run()
            self.assert_clean(app)
            self.assertEqual(app.session_state["analysis"][1]["segment"], expected)
            next(r for r in app.radio if r.label == "Spending view").set_value("Relative to population median").run()
            self.assert_clean(app)
        app.slider(key="whatif_multiplier").set_value(2.0).run()
        self.assert_clean(app)
        app.number_input(key="input_Fresh").set_value(12000.).run()
        self.assertTrue(any("Inputs changed" in message.value for message in app.info))

    def test_model_loading_error_is_visible(self):
        with patch("src.data.get_bundle", side_effect=ValueError("Incompatible model artifact")):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertIn("Incompatible model artifact", app.error[0].value)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "customers.csv"
BUNDLE_PATH = ROOT / "artifacts" / "customer_segmentation_bundle.joblib"
FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
PCA_COLUMNS = ["Dimension 1", "Dimension 2"]
BUNDLE_VERSION = "1.0.0"
PAGES = ["Overview", "Customer Segmentation", "Segment Explorer", "Data Insights",
         "PCA & Clustering", "Model Performance", "About Project"]

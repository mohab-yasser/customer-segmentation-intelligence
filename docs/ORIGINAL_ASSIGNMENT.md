# Content: Unsupervised Learning
## Project: Creating Customer Segments

## Customer Segmentation Intelligence — Streamlit application

The completed project now includes a dark, interactive analytics application for business users and technical reviewers. The validated notebook and its HTML export remain unchanged. The original assignment documentation below is retained for context; its Python 2 instructions are historical. Use the pinned modern environment for the completed project and app.

### Run locally

Tested locally on Python 3.10.11 with the pinned numerical stack.

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The frozen artifact is included, so no training step is required to launch the application.

### Frozen model and reproducibility

Inference is strictly:

```text
Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen (exact order)
  → validate one row of positive, finite numeric values
  → np.log
  → fitted PCA.transform (2 components, full SVD)
  → fitted KMeans.predict (2 clusters, random_state=42, n_init="auto")
```

**Channel and Region are excluded from prediction.** Training removed five observations (original zero-based indices 65, 66, 75, 128, 154) with Tukey flags in at least two log-spending dimensions. New customers are never removed by that policy. Values outside the retained training population's 1st–99th percentiles produce a warning, not rejection.

The final baseline is preserved: **435 training rows**, segment sizes **175 / 260**, silhouette **0.426072168**, and two-component retained variance **70.681723%**. Samples **3 → 1, 28 → 0, 34 → 1** are verified against saved notebook outputs. Segment 0 has grocery/retail-oriented purchasing; Segment 1 has relatively fresh/frozen/food-service-oriented purchasing. These are behavioral interpretations, not verified business identities.

`artifacts/customer_segmentation_bundle.joblib` contains the fitted PCA and K-Means, exact feature order, recovered centers, segment descriptions, all six PCA variance ratios and component weights, silhouette comparisons, training coordinates/labels/original row indices, outlier metadata, external Channel counts, real-data presets, distribution statistics, sample predictions, version information and dataset/notebook SHA-256 hashes.

To regenerate the artifact **offline**, using the pinned requirements:

```bash
python export_model.py
python -m unittest discover -s tests -v
```

Export reproduces the original methodology, evaluates the original K-Means/GMM configurations, and asserts agreement with the saved analysis before replacing the artifact. It does not execute or modify the notebook. Recovered centers are checked within the notebook's displayed rounding precision; silhouettes within its six-decimal precision. The final score and full PCA ratios have tighter audit-reference checks.

`app.py` and `src/` contain no model fitting calls and never import the exporter. Missing or incompatible artifacts cause a visible error, not automatic retraining. Artifact loading is cached with `st.cache_resource`; dataset loading and display summaries use `st.cache_data`. Only load the trusted repository artifact: joblib files use pickle serialization. Keep the pinned scikit-learn version when serving it.

### Pages and features

1. **Overview:** verified KPIs, pipeline, customer map and educational explanation.
2. **Customer Segmentation:** six inputs, real customer/median presets, distribution warnings, centroid distances, grouped spending charts, comparison table and highlighted PCA position.
3. **Segment Explorer:** recovered profiles, mean/median-relative comparisons, sizes and external Channel composition.
4. **Data Insights:** live data audit, distributions, raw/log skewness, boxplots, correlation and outlier explanation.
5. **PCA & Clustering:** all six explained variances, cumulative variance, component weights and actual PC1/PC2 interpretation.
6. **Model Performance:** original K-Means/GMM silhouette comparison, selected model and artifact provenance.
7. **About Project:** training vs inference, business applications, delivery experiments and limitations.

Optional tools include a normalized radar profile, empirical population percentiles, nearest historical customers in PCA space, a one-category what-if scenario, and downloadable analysis CSV. Changed inputs invalidate the displayed result until Analyze Customer is selected again.

**Relative cluster proximity** = distance to alternative centroid / sum of both distances. It is a geometric indicator, not probability, confidence or accuracy. Representative spending is `exp(pca.inverse_transform(centroid))`, not an arithmetic mean. Display normalization never changes prediction. Monetary values use the dataset's unspecified monetary units.

### Project structure

```text
app.py                         Streamlit entry point
export_model.py                Offline reproduction and validated artifact export
src/config.py                  Portable paths and serving contract
src/inference.py               Bundle checks and pure inference
src/data.py                    Cached data and artifact loaders
src/charts.py                  Consistent interactive Plotly charts
src/ui.py                      Theme, cards and presentation helpers
src/pages.py                   Seven dashboard views
tests/test_inference.py        Model, validation and no-fitting checks
tests/test_app.py              Page navigation and interaction checks
.streamlit/config.toml         Dark theme and server defaults
artifacts/customer_segmentation_bundle.joblib
customers.csv
customer_segments_completed.ipynb
customer_segments_completed.html
visuals.py
```

### Screenshots

Browser captures are available in `assets/screenshots/`: Overview, customer analysis, model performance, and mobile layouts. The application has no screenshot placeholders in its interface.

![Customer Segmentation Intelligence overview](assets/screenshots/overview-desktop.png)

To refresh screenshots and run browser interaction checks, start the app and run `python tests/browser_smoke.py` in a development environment with Playwright and Chromium installed. Browser tooling is optional and is not part of the deployment requirements.

### Verification performed

- All 11 unittest tests pass: artifact contract, notebook sample predictions, positive/finite validation, schema/order rejection, unusual-customer handling, and static/runtime guards against fitting.
- Streamlit AppTest renders every page and exercises presets, predictions, display views, what-if controls, stale results and visible artifact errors.
- Local Streamlit server starts successfully and returns `ok` at its health endpoint.
- Headless Chromium checks pass for all seven pages, prediction, CSV download and 390px mobile layouts, without browser exceptions or page-level horizontal overflow. Desktop and mobile screenshots were inspected.
- `python -m pip check` reports no broken requirements. Dataset and notebook hashes still match export provenance.

### Streamlit Community Cloud deployment

The project is prepared for Community Cloud; publishing to an account is a separate action.

1. Push this project to a GitHub repository, including the **joblib artifact**, `customers.csv`, `src/`, `app.py`, `.streamlit/config.toml`, and `requirements.txt`.
2. In Community Cloud choose **Create app**, select the repository/branch, and set the entry point to `app.py`.
3. Select a Python version compatible with the pinned packages in Advanced settings (Python 3.11 is a suitable deployment target; the local validation environment used Python 3.10.11). Avoid choosing a newer Python release without checking wheel availability for the pinned numerical stack.
4. Deploy and verify all pages and the three known sample assignments. The platform installs `requirements.txt`; no secrets, external services, OS-specific paths, or startup training commands are required.

See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guidance](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). A Linux/Cloud deployment has not been executed as part of local development; repeat the included checks when changing Python or dependency versions.

### Scope and limitations

This is a polished portfolio serving application, not evidence of prospective business performance. The small historical dataset has moderate cluster overlap; PCA(2) omits 29.32% of log-spending variance. Names and Channel composition support interpretation only. Unusual customers can be poorly represented even when centroid proximity is high. Delivery effects were not measured: assess a 5-day to 3-day delivery change through a randomized A/B experiment within segments. Authentication, monitoring and business-system integration are outside this portfolio scope.

---

## Original assignment documentation

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/index.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

### Code

Template code is provided in the `customer_segments.ipynb` notebook file. You will also be required to use the included `visuals.py` Python file and the `customers.csv` dataset file to complete your work. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project. Note that the code included in `visuals.py` is meant to be used out-of-the-box and not intended for students to manipulate. If you are interested in how the visualizations are created in the notebook, please feel free to explore this Python file.

### Run

In a terminal or command window, navigate to the top-level project directory `customer_segments/` (that contains this README) and run one of the following commands:

```bash
ipython notebook customer_segments.ipynb
```  
or
```bash
jupyter notebook customer_segments.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

## Data

The customer segments data is included as a selection of 440 data points collected on data found from clients of a wholesale distributor in Lisbon, Portugal. More information can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers).

Note (m.u.) is shorthand for *monetary units*.

**Features**
1) `Fresh`: annual spending (m.u.) on fresh products (Continuous); 
2) `Milk`: annual spending (m.u.) on milk products (Continuous); 
3) `Grocery`: annual spending (m.u.) on grocery products (Continuous); 
4) `Frozen`: annual spending (m.u.) on frozen products (Continuous);
5) `Detergents_Paper`: annual spending (m.u.) on detergents and paper products (Continuous);
6) `Delicatessen`: annual spending (m.u.) on and delicatessen products (Continuous); 
7) `Channel`: {Hotel/Restaurant/Cafe - 1, Retail - 2} (Nominal)
8) `Region`: {Lisbon - 1, Oporto - 2, or Other - 3} (Nominal) 

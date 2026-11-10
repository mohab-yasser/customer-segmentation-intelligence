# Creating Customer Segments — Completed Project

This project explores natural purchasing groups among 440 wholesale customers. It completes every original notebook implementation task and all twelve questions, with executed plots, numerical answers, and external Channel comparison.

## Start here

- `customer_segments_completed.html`: read the executed analysis in a browser.
- `customer_segments_completed.ipynb`: review or run the submission notebook.
- `project_audit.md`: requirement coverage, compatibility changes, metrics, checks, and limitations.

Use this file as the main guide for the completed project. The original `customer_segments.ipynb` and instructor `README.md` are retained for assignment context, dataset provenance, and the original requirements. The instructor README describes the historical Python 2 environment; use the modern instructions below to run the completed notebook. Redundant backups, preliminary work, and development tools have been removed.

## Dataset and pipeline

The six input features are annual monetary spending on Fresh, Milk, Grocery, Frozen, Detergents_Paper, and Delicatessen. Channel and Region are categorical metadata, excluded from model inputs. Channel is joined to assignments only after the model and spending-based interpretations are finalized.

Pipeline: data audit → six spending features → natural logarithm → multi-feature Tukey outlier removal → full PCA analysis → two-component PCA → K-Means/GMM comparison → final two-cluster K-Means → inverse-PCA/exponential center recovery → business profiling → external Channel comparison.

No StandardScaler is used. The DecisionTreeRegressor experiment is auxiliary; it uses a 75/25 train/test split to predict Grocery and does not remove any clustering features. The main descriptive clustering task has no artificial supervised holdout set.

## Main results

- Verified 440 × 8 records/columns, no missing values and no duplicates.
- Log-space Tukey analysis removes indices 65, 66, 75, 128, and 154 because each is unusual in at least two features; 435 customers remain. Outliers are not assumed to be errors.
- Grocery test R²: **0.681884**.
- Two PCA components preserve **70.6817%** of variance; four preserve **93.1090%**.
- Final model: **K-Means, two clusters, random_state=42, n_init="auto"**; silhouette **0.426072**.
- K-Means k=2–6 and full-covariance GMM with 2–4 components were compared. Two-component GMM scores **0.421917**. Ten-start K-Means checks score **0.426281**, with ARI **0.981630** versus the retained baseline; this small difference does not materially change the interpretation.
- Segment 0: 175 grocery/retail-oriented customers; 135 Retail and 40 HoReCa (**77.14% Retail**).
- Segment 1: 260 customers with relatively fresh/frozen-focused spending; 254 HoReCa and 6 Retail (**97.69% HoReCa**).

Labels are arbitrary identifiers, and establishment descriptions are hypotheses. Channel proportions are external comparisons, not supervised accuracy. Delivery frequency changes require randomized evaluation within segments.

## Run and reproduce

Tested on Windows with **Python 3.10.11**. Run from this project directory and install the pinned dependencies in your preferred environment:

```powershell
python -m pip install -r requirements.txt
```

Open `customer_segments_completed.ipynb` in a Jupyter-capable editor such as VS Code, select the environment containing these dependencies, restart the kernel, and run all cells. The working directory must contain `customers.csv` and `visuals.py`. No build scripts, validation scripts, backups, hashes, or intermediate JSON files are required.

To review without running Python, open `customer_segments_completed.html`. To refresh HTML after intentionally editing or rerunning the notebook, export its saved outputs without executing it again:

```powershell
python -m nbconvert --to html customer_segments_completed.ipynb
```

The notebook retains its validation assertions and an in-memory `results` dictionary. It no longer writes an intermediate results JSON file. Numerical prose describes this dataset and pinned environment; review it if data or dependencies change. Python/Jupyter may recreate normal caches during future use.

For submission, include the completed notebook and HTML; keep `customers.csv` and `visuals.py` alongside the notebook when sharing a runnable project. Original assignment instructions remain in `customer_segments.ipynb` and `README.md`.

## Organization and compatibility

`visuals.py` retains all four instructor functions. Its only changes guard IPython initialization outside a notebook and replace two deprecated colormap calls. The completed notebook also replaces the obsolete pandas scatter-matrix entry point.

The notebook contains an `assign_new_customers` function applying the frozen log → PCA → cluster-prediction path to positive finite inputs. There is no Streamlit deployment or saved model artifact: optional deployment is outside this analysis deliverable, and the project does not claim production readiness.

## Limitations

Moderate overlap, 29.32% variance loss in PCA2, sensitivity to outlier policy, limited historical observations, and uncertain establishment labels limit operational conclusions. Initialization checks are not sampling-stability checks. Compare full-log-space clustering, resampled stability, and customer outcomes before deploying segmentation or reducing delivery service.

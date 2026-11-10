# Project audit

## Files reviewed and preservation

- Reviewed all five initial files: `customer_segments.ipynb` (all cells), `customers.csv` (all records and schema), `visuals.py` (all four helpers), `README.md`, and `p3.ipynb` (audit and histogram work).
- Original instructor notebook, dataset, and instructor README are retained. The preliminary audit/histogram notebook was removed after confirming that its useful work is included in the completed notebook.
- Redundant backups, integrity manifests, intermediate JSON results, and development build/validation scripts were removed after successful validation. Instructor helper compatibility changes remain documented below.

## Compatibility changes

- Guarded `get_ipython()` before enabling inline plotting; ordinary Python imports now work.
- Replaced the two deprecated `cm.get_cmap` calls with `plt.get_cmap`; all instructor plotting logic is preserved.
- Updated the notebook’s deprecated `pd.scatter_matrix` to `pd.plotting.scatter_matrix`.
- Uses modern `GaussianMixture`, fixed random states, and version-aware K-Means `n_init`.
- Retained original-row identities separately and reset retained plotting indices to prevent helper/crosstab alignment errors.
- Uses one numerical thread and a Windows core-count environment fallback; no global warning suppression was added.
- Installed nbconvert for the requested HTML export. The instructor README remains unchanged.

## Original implementation coverage

All 22 original code TODO markers are implemented; no unfinished code placeholders remain. Original cell numbers below are zero-based.

| Original cell | Required implementation | Completed section |
|---|---|---|
| 7 | Select three samples | 8 |
| 11 | Feature removal, split, regression fit and R² | 9 |
| 20 | Natural log of data and samples; scatter matrix | 12 |
| 24 | Q1/Q3, IQR step, outlier investigation/removal | 13 |
| 29 | Full PCA and sample transform | 14 |
| 35 | Two-component PCA, reduced data, sample transform | 15 |
| 45 | Clustering fit, predictions, centers, sample predictions, silhouette | 16–19 |
| 51 | Inverse PCA and exponentiation | 21 |

All 12 questions have explicit, nonempty Markdown answers. Coverage: Q1 samples; Q2 regression; Q3 EDA; Q4 outliers; Q5 PCA; Q6 algorithm choice; Q7 silhouette comparisons; Q8 profiles; Q9 sample assignments; Q10 randomized delivery experiment; Q11 direct prediction and supervised pseudo-label alternative; Q12 external Channel comparison.

All instructor functions execute: `pca_results`, `biplot`, `cluster_results`, `channel_results`. Raw/log scatter matrices and sample transformation observations are retained.

## Pipeline and leakage audit

- Six spending columns only → natural logarithm → remove customers flagged in at least two log-space Tukey tests → PCA(2) → K-Means(2).
- No StandardScaler; the primary pipeline follows the instructor methodology. Log variances are not assumed equal.
- Full six-component PCA precedes dimensionality reduction. Recovery is PCA inverse → exponential.
- Channel and Region are metadata only. Initial value counts are schema checks; row-level Channel joins occur only after the final model and spending-based profiles are frozen.
- No model is selected or retuned using Channel. The only train/test split is the auxiliary Grocery regression.
- Samples 3, 28, 34 have no Tukey flags and are retained. Outliers are calculated, not hard-coded in the pipeline.

## Executed findings

- Dataset: [440, 8]; missing values 0; duplicates 0.
- Removed indices and feature counts: {'65': 2, '66': 2, '75': 2, '128': 2, '154': 3}. These are unusual observations, not proven errors.
- Grocery test R²: 0.681884009.
- PCA variance ratios: 0.44302505, 0.26379218, 0.12306380, 0.10120908, 0.04850196, 0.02040793.
- Cumulative two/four component variance: 70.681723% / 93.109011%.

| Algorithm | Clusters | Silhouette | Sizes |
|---|---:|---:|---|
| K-Means | 2 | 0.426072 | [175, 260] |
| K-Means | 3 | 0.397423 | [80, 215, 140] |
| K-Means | 4 | 0.331767 | [69, 144, 109, 113] |
| K-Means | 5 | 0.318854 | [69, 159, 81, 49, 77] |
| K-Means | 6 | 0.321841 | [74, 152, 73, 53, 71, 12] |
| GMM (full covariance, 10 starts) | 2 | 0.421917 | [281, 154] |
| GMM (full covariance, 10 starts) | 3 | 0.394442 | [262, 31, 142] |
| GMM (full covariance, 10 starts) | 4 | 0.301212 | [203, 132, 49, 51] |

Final: K-Means, k=2, random_state=42, n_init='auto'; silhouette 0.426072168. Selected using internal metrics, sizable groups, visual structure, initialization sensitivity, and business interpretability.

- Segment 0: grocery/retail-oriented purchases; 175 customers, 135 Retail and 40 HoReCa (77.14% Retail).
- Segment 1: relatively fresh/frozen-focused purchases; 260 customers, 254 HoReCa and 6 Retail (97.69% HoReCa).
- Neither segment is pure. Channel is an external comparison, not supervised accuracy.
- Sample predictions: customer 3 → 1; 28 → 0; 34 → 1.
- Ten-start checks across five seeds score 0.426281 with ARI 0.981630 versus the baseline. This marginal difference is not treated as a meaningful improvement; the requested baseline is retained. This is not a bootstrap stability study.

## Execution and deliverables

- Clean current-Python kernel: Python 3.10.11; 29 code cells executed sequentially without errors in 6.7 seconds.
- 12 embedded PNG figure outputs; all preserved in the exported HTML.
- Runtime assertions verify feature exclusion, original-row alignment, inverse transforms, frozen predictions, finite centers, sample assignment consistency, and dataset integrity.
- Before cleanup, the development validator checked all answers, TODO completion, sequential execution, numerical statements, backups, and embedded HTML plots. These results describe the validated analysis; cleanup did not rerun or retrain it.
- Final deliverables: `customer_segments_completed.ipynb`, `customer_segments_completed.html`, `customers.csv`, `visuals.py`, `README_COMPLETED.md`, `requirements.txt`, and this audit. The original instructor notebook and README are retained as assignment references.
- Run the completed notebook directly using `README_COMPLETED.md`; no development scripts or intermediate files are needed.

## Limitations and scope

- Moderate cluster overlap; 2D PCA discards 29.32% of variance. Full-log-space clustering, resampling stability, and outlier-policy sensitivity are future work.
- One historical dataset and one regression split do not establish future performance. Cluster names are hypotheses, and delivery effects require an experiment.
- Static numerical prose is validated against this dataset and pinned environment; changing data or dependencies requires reviewing the prose and rerunning validation.
- Optional Streamlit deployment is not included; the requested validated analysis and submission artifacts are complete.

## Final checklist

- [x] Dataset loads successfully
- [x] Correct clustering features used
- [x] Channel excluded from clustering
- [x] Region excluded from clustering
- [x] Missing values checked
- [x] Duplicates checked
- [x] EDA completed
- [x] Skewness analyzed
- [x] Log transformation performed
- [x] Outliers investigated
- [x] PCA completed
- [x] Dimensionality reduction completed
- [x] Biplot created
- [x] K-Means/GMM discussed
- [x] Multiple cluster counts evaluated
- [x] Silhouette scores reported
- [x] Final model justified
- [x] Cluster centers recovered
- [x] Customer segments interpreted
- [x] Sample customers evaluated
- [x] Business delivery question answered
- [x] New-customer question answered
- [x] Channel comparison performed
- [x] All 12 original questions answered
- [x] Every original implementation task completed
- [x] No data leakage
- [x] Notebook runs top-to-bottom without errors

## Final folder cleanup

- Preserved the model pipeline, numerical outputs, all 12 answers, all figures, and the original instructor notebook.
- Dataset and visualization helper were not modified during cleanup.
- Removed redundant backups, build/validation scripts, intermediate JSON files, and the preliminary notebook after inspecting their content.
- Updated documentation references and removed only the optional JSON file-writing side effect from the completed notebook; the results dictionary and pipeline assertions remain.
- Refreshed HTML from existing saved notebook outputs without executing cells or retraining models.
- Retained both READMEs for assignment context; README_COMPLETED.md is the current run guide.

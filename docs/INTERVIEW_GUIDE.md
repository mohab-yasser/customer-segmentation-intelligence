# Interview guide — Customer Segmentation Intelligence

Use this as a way to explain choices in your own words. All figures below are drawn from the project's validated saved outputs and audit.

## One-minute project pitch

"I built an unsupervised wholesale customer segmentation pipeline on 440 historical customers. I investigated the spending distribution, applied a log transform to six purchasing categories, examined multivariate outliers, and used PCA to explore two-dimensional structure. I compared K-Means and Gaussian Mixture Models using silhouette scores and chose a two-cluster K-Means model with a silhouette of about 0.426. I packaged the trained PCA and K-Means into a validated joblib artifact, then built a seven-page Streamlit dashboard that lets a user explore customer behavior and run frozen-model inference. Channel and Region were excluded from fitting and used for external analysis only."

## Questions you should be able to answer

1. **Why unsupervised learning?** The task is to discover patterns in spending rather than predict an existing labeled target.
2. **Why log-transform?** The spending features are strongly right-skewed; the log compresses large values. That requires strictly positive input values.
3. **Why didn't you use Channel or Region?** They were excluded from clustering to avoid steering the result toward known categories. Channel is compared with clusters afterward.
4. **Why remove five rows?** The predefined policy flags observations as outliers in at least two log-space spending variables. The excluded rows are unusual, not proven data errors.
5. **Why PCA?** To summarize six logged variables for visual exploration and clustering. Two components retain 70.6817% of the variance, so some information is lost.
6. **Why K-Means over GMM?** Two-cluster K-Means gave an approximately 0.426072 silhouette compared with ~0.421917 for two-component GMM; the final rationale also considered interpretability and sensible cluster sizes.
7. **Does silhouette 0.426 mean 42.6% accuracy?** No. Silhouette measures within-cluster cohesion vs separation, not supervised accuracy.
8. **How do predictions work?** Input validation → natural log → fitted PCA.transform → fitted KMeans.predict. No fitting or outlier dropping at serving time.
9. **What does centroid proximity mean?** Relative geometry in the projected PC space; it is not a calibrated class probability or business confidence score.
10. **Are the segment names definitive?** No. They describe relative spending patterns. Channel distributions support interpretation but are not ground-truth labels.
11. **What about stability?** Initialization checks were performed, but there is no extensive resampling stability analysis. A stronger follow-up would evaluate clustering across bootstrap samples and alternative outlier policies.
12. **How would this help a business?** Generate segment-specific hypotheses, design differentiated experiences, and measure impact via controlled experiments rather than claiming guaranteed outcomes.
13. **How did you deploy safely?** A pinned environment and bundled model artifact, deterministic input order, dataset integrity check, model validation, no training in the app, and automated tests.
14. **What would you improve?** Out-of-sample segment monitoring, handling concept drift, segment stability checks, adding a downstream decision process, and real-world validation of business outcomes.

## Verified facts to keep accurate

- Dataset: **440 rows × 8 columns**, 0 blanks, 0 duplicates.
- Six spending variables: Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen.
- Five outliers removed for training (indices 65, 66, 75, 128, 154) → **435 rows**.
- Final k=2 clusters: **175 and 260**.
- Silhouette: **0.426072168**.
- PCA two-component retained variance: **70.681723%**.
- Sample assignments: **customer 3 → 1, customer 28 → 0, customer 34 → 1**.
- Auxiliary Grocery regression test R²: **0.681884**, separate from the unsupervised segmentation quality.

## Be careful not to claim

- That 0.426 silhouette is prediction accuracy.
- That clusters are known customer types or that 100% of customers fit a segment cleanly.
- That the historical dataset proves an intervention increases revenue or reduces costs.
- That the app is a production-ready customer analytics system.

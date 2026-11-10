"""Dashboard views, all backed by frozen artifacts and the original dataset."""
from html import escape
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import charts
from src.config import FEATURES
from src.inference import predict_customer_segment
from src.ui import chart, flow, heading, metrics


def overview(bundle, data, derived):
    retained = bundle["pca_variance"][:2].sum() * 100
    st.html('''<div class="hero"><div class="eyebrow">PURCHASING PATTERNS. CLEARER DECISIONS.</div>
    <span class="badge">UNSUPERVISED LEARNING · FROZEN MODEL</span>
    <h1>Customer Segmentation<br>Intelligence</h1>
    <p>Understand the patterns behind every purchase. An unsupervised Machine Learning system
    that groups wholesale customers by their annual purchasing behavior.</p></div>''')
    metrics([("Customers analyzed", len(data), f"{bundle['metadata']['training_rows']} used in fitting"),
             ("Spending features", len(FEATURES), "Annual product categories"),
             ("Customer segments", bundle["kmeans"].n_clusters, "Distinct purchasing profiles"),
             ("PCA variance retained", f"{retained:.2f}%", "Two principal components"),
             ("Silhouette score", f"{bundle['silhouette']:.3f}", "Internal cohesion / separation")])
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("What does this system do?")
        st.write("Find natural purchasing patterns across six categories, then place a new customer within those patterns. Explore the evidence behind each segment and use it to shape business experiments.")
        flow(["Raw spending", "Natural log", "PCA · 2D", "K-Means", "Segment"])
        st.info("Channel was used only after clustering as external interpretation/reference information. Neither Channel nor Region is a model input.")
        if st.button("Analyze a customer →", type="primary", width="stretch"):
            st.session_state["navigation"] = "Customer Segmentation"
            st.rerun()
        st.caption("Annual spending is measured in the dataset's monetary units; no currency is assumed.")
    with right:
        with st.container(border=True):
            st.subheader("A map of purchasing behavior")
            chart(charts.pca_scatter(bundle))
            st.caption(f"{len(bundle['training_labels'])} retained historical customers · each point is one customer.")
    with st.expander("A quick guide to the model"):
        st.write("**Clustering** finds similar observations without training labels. **PCA** summarizes correlated log-spending features into two axes. **Silhouette** measures within-cluster cohesion and between-cluster separation from −1 to 1; it is not accuracy. **Natural log** compresses extreme spending and reduces strong right skew.")


def _preset(bundle):
    for feature, value in bundle["presets"][st.session_state["preset"]].items():
        st.session_state[f"input_{feature}"] = float(value)
    st.session_state.pop("analysis", None)


def prediction(bundle, data, derived):
    heading("Analyze a customer", "Six spending inputs. One interpretable purchasing profile.", "CUSTOMER SEGMENTATION")
    st.caption("Enter annual spending in monetary units. Presets come from actual historical customers or population medians.")
    a, b = st.columns([3, 1], vertical_alignment="bottom")
    with a:
        st.selectbox("Example customer", list(bundle["presets"]), key="preset")
    with b:
        st.button("Load Example Customer", on_click=_preset, args=(bundle,), width="stretch")
    with st.container(border=True):
        st.subheader("Annual purchasing profile")
        columns = st.columns(3)
        values = {}
        for i, feature in enumerate(FEATURES):
            stats = bundle["statistics"].loc[feature]
            key = f"input_{feature}"
            if key not in st.session_state:
                st.session_state[key] = float(stats["median"])
            with columns[i % 3]:
                values[feature] = st.number_input(feature.replace("_", " & "), min_value=1.0, step=100.0,
                    format="%.2f", key=key,
                    help=f"Annual {feature.replace('_', ' / ').lower()} spending. Historical retained range: {stats['min']:,.0f}–{stats['max']:,.0f}. Larger values are allowed.")
        submitted = st.button("Analyze Customer", type="primary", width="stretch")
    customer = pd.DataFrame([values], columns=FEATURES)
    if submitted:
        try:
            st.session_state["analysis"] = (customer, predict_customer_segment(customer, bundle))
        except ValueError as exc:
            st.session_state.pop("analysis", None)
            st.error(str(exc))
    if "analysis" not in st.session_state:
        st.info("Choose an example or enter your customer's spending, then select Analyze Customer.")
        return
    analyzed, result = st.session_state["analysis"]
    if not np.array_equal(customer.to_numpy(), analyzed.to_numpy()):
        st.info("Inputs changed. Select Analyze Customer to refresh your analysis.")
        return
    segment = result["segment"]
    profile = bundle["profiles"][segment]
    st.html(f'<div class="result"><div class="eyebrow">CUSTOMER SEGMENT · {segment:02d}</div>'
            f'<h2>Segment {segment} · {escape(profile["name"])}</h2><p>{escape(profile["description"])}</p></div>')
    if result["unusual_features"]:
        st.warning("This spending profile is outside or near the edge of the historical training distribution. Interpret the prediction carefully. Categories outside the retained 1st–99th percentile: " + ", ".join(result["unusual_features"]) + ". This customer is still analyzed; unusual does not mean invalid.")
    d = result["centroid_distances"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Distance to assigned centroid", f"{d[segment]:.3f}", help="Euclidean distance in the fitted two-dimensional PCA space. Lower means closer.")
    c2.metric("Distance to alternative centroid", f"{d[1-segment]:.3f}")
    c3.metric("Relative cluster proximity", f"{result['relative_proximity']:.1%}",
              help="Alternative distance / (assigned distance + alternative distance). A geometric comparison only, not probability, confidence or accuracy. 50% means equal distance.")
    st.caption("Relative proximity describes the two centroid distances only. Even an unusual customer can be closer to one centroid.")
    st.subheader("Customer vs. segment representative")
    view = st.radio("Spending view", ["Raw monetary units", "Relative to population median"], horizontal=True)
    median = bundle["statistics"]["median"].to_numpy()
    raw = analyzed.iloc[0].to_numpy()
    representative = result["representative_spending"]
    denominator = median if view.startswith("Relative") else None
    chart(charts.spending_chart({"Customer": raw, f"Segment {segment} representative": representative}, denominator,
                               "Multiple of population median" if denominator is not None else "Annual spending · monetary units"))
    st.caption("Representative spending = exp(PCA inverse transform of the centroid). It is an approximate profile, not the segment's arithmetic mean.")
    table = pd.DataFrame({"Feature": FEATURES, "Customer": raw, "Segment Representative": representative,
                          "Difference %": (raw / representative - 1) * 100})
    st.dataframe(table.style.format({"Customer": "{:,.2f}", "Segment Representative": "{:,.2f}", "Difference %": "{:+.1f}%"}),
                 hide_index=True, width="stretch")
    st.subheader("Where this customer sits")
    chart(charts.pca_scatter(bundle, result["pca_coordinates"]), key="customer_position")
    st.caption("The gold star is your customer. Distance and assignment are computed in this same frozen PCA space.")
    with st.expander("Profile shape & population percentiles"):
        left, right = st.columns(2)
        with left:
            chart(charts.radar(raw, representative, median))
            st.caption("Radar uses log(1 + spending / population median) for readability. This visualization normalization does not affect prediction.")
        with right:
            percentiles = (derived["raw"].le(raw, axis=1).mean() * 100)
            fig = px.bar(x=percentiles.values, y=FEATURES, orientation="h", labels={"x": "Historical percentile", "y": "Category"}, range_x=[0, 100])
            chart(charts.polish(fig))
            st.caption(f"Empirical percentile: percentage of all {len(data)} historical customers with spending at or below this value.")
    with st.expander("Nearest historical customers"):
        distances = np.linalg.norm(bundle["training_coordinates"] - result["pca_coordinates"], axis=1)
        nearest = np.argsort(distances)[:5]
        examples = data.loc[bundle["training_indices"][nearest], FEATURES].copy()
        examples.insert(0, "PCA distance", distances[nearest])
        examples.index.name = "Original customer index"
        st.dataframe(examples, width="stretch")
        st.caption("Similarity examples among retained customers in PCA space. These can include the input customer when a historical preset is used.")
    with st.expander("What-if analysis"):
        feature = st.selectbox("Adjust one category", FEATURES, key="whatif_feature")
        multiplier = st.slider("Spending multiplier", .1, 3.0, 1.0, .1, key="whatif_multiplier")
        changed = analyzed.copy()
        changed[feature] *= multiplier
        alternative = predict_customer_segment(changed, bundle)
        st.write(f"At {changed[feature].iloc[0]:,.2f} monetary units in {feature}, the profile belongs to **Segment {alternative['segment']}**. Relative cluster proximity: **{alternative['relative_proximity']:.1%}**.")
        chart(charts.pca_scatter(bundle, alternative["pca_coordinates"]), key="whatif_position")
        if alternative["unusual_features"]:
            st.warning("This scenario is outside the retained 1st–99th percentile in: " + ", ".join(alternative["unusual_features"]))
        st.caption("A descriptive sensitivity check with the frozen model; it does not estimate a causal business effect or change the original analysis.")
    download = table.assign(segment=segment, segment_name=profile["name"],
                            pc1=result["pca_coordinates"][0], pc2=result["pca_coordinates"][1],
                            distance_segment_0=d[0], distance_segment_1=d[1],
                            relative_cluster_proximity=result["relative_proximity"], model_version=bundle["version"])
    st.download_button("Download Customer Analysis · CSV", download.to_csv(index=False).encode("utf-8"),
                       "customer_analysis.csv", "text/csv", width="stretch")


def segment_explorer(bundle, data, derived):
    heading("Two segments. Different priorities.", "Compare purchasing profiles and the external evidence behind their interpretation.", "SEGMENT EXPLORER")
    sizes = np.bincount(bundle["training_labels"])
    for segment, col in enumerate(st.columns(2, gap="large")):
        with col, st.container(border=True):
            st.caption(f"SEGMENT {segment:02d}")
            st.subheader(bundle["profiles"][segment]["name"])
            st.metric("Retained customers", int(sizes[segment]))
            st.write(bundle["profiles"][segment]["description"])
            ratios = bundle["centers"][segment] / bundle["centers"][1-segment]
            top = np.argsort(ratios)[::-1][:3 if segment == 0 else 2]
            st.markdown("**Distinguishing categories**")
            st.write(" · ".join(f"{FEATURES[i]} ({ratios[i]:.1f}×)" for i in top))
            st.caption("Representative spending relative to the other segment.")
            channel = bundle["channel_composition"].loc[segment]
            st.write(f"External Channel: **{channel['Retail']} Retail** ({channel['Retail']/sizes[segment]:.2%}) · **{channel['HoReCa']} HoReCa** ({channel['HoReCa']/sizes[segment]:.2%}).")
    view = st.radio("Compare representatives", ["Monetary units", "Overall median", "Overall mean"], horizontal=True)
    denom = None if view == "Monetary units" else bundle["statistics"]["median" if view == "Overall median" else "mean"].to_numpy()
    chart(charts.spending_chart({f"Segment {i}": bundle["centers"][i] for i in range(2)}, denom,
                               "Annual spending · monetary units" if denom is None else f"Multiple of {view.lower()}"))
    st.caption("Recovered centers use inverse PCA followed by exponentiation. Overall mean and median refer to all original customers.")
    st.dataframe(pd.DataFrame(bundle["centers"], columns=FEATURES, index=["Segment 0", "Segment 1"]).style.format("{:,.2f}"), width="stretch")
    left, right = st.columns(2)
    with left:
        chart(charts.polish(px.pie(names=["Segment 0", "Segment 1"], values=sizes, hole=.72,
                                  color_discrete_sequence=charts.COLORS, title="Retained customer mix")))
        st.caption("Segment sizes are not expected to be equal.")
    with right:
        counts = bundle["channel_composition"]
        fig = go.Figure([go.Bar(name=channel, x=["Segment 0", "Segment 1"], y=counts[channel], marker_color=charts.COLORS[i])
                         for i, channel in enumerate(counts.columns)])
        fig.update_layout(barmode="stack", title="External Channel composition", yaxis_title="Customers")
        chart(charts.polish(fig))
    st.info("External comparison only, not cluster accuracy. Channel was joined after the spending-based model and profiles were frozen. HoReCa means Hotel / Restaurant / Café; neither segment is a verified business identity.")


def data_insights(bundle, data, derived):
    heading("Explore the evidence", "The original dataset, its spending distributions, and the training-only outlier policy.", "DATA INSIGHTS")
    metrics([("Original rows", len(data), "Historical customers"), ("Original columns", data.shape[1], "6 spending + 2 metadata"),
             ("Missing values", derived["missing"], "Calculated from customers.csv"), ("Duplicate rows", derived["duplicates"], "Full-row duplicates")])
    feature = st.selectbox("Spending category", FEATURES)
    st.subheader("Before & after the natural logarithm")
    for col, label, values, skew in zip(st.columns(2), ["Raw spending", "Natural-log spending"],
                                       [derived["raw"][feature], derived["log"][feature]],
                                       [derived["raw_skew"][feature], derived["log_skew"][feature]]):
        with col:
            st.metric(f"{label} skewness", f"{skew:.3f}")
            fig = px.histogram(x=values, nbins=35, labels={"x": label, "y": "Customers"})
            chart(charts.polish(fig, 320))
    st.write("The natural logarithm compresses extreme spending and reduces strong right skew. It does not make every category normally distributed. The model uses these log values without standard scaling.")
    with st.expander("Spending spread & correlation", expanded=True):
        box = px.box(derived["log"].melt(var_name="Category", value_name="Natural-log spending"), x="Category", y="Natural-log spending", points="outliers")
        chart(charts.polish(box, 350))
        chart(charts.heatmap(derived["correlation"], "Raw spending · Pearson correlation"))
    st.subheader("Unusual observations, not invalid customers")
    st.write(f"The training policy removed {len(bundle['removed_indices'])} observations flagged by Tukey's 1.5×IQR rule in at least two spending dimensions **after log transformation**. This reduces disproportionate influence on distance-based clustering; it does not establish that the records are incorrect.")
    removed = data.loc[bundle["removed_indices"], FEATURES].copy()
    removed.insert(0, "Flagged dimensions", pd.Series(bundle["outlier_counts"]))
    removed.index.name = "Original row index (zero-based)"
    st.dataframe(removed, width="stretch")
    st.info("New customers are never deleted by this policy. Inference validates positive, finite spending, applies the frozen transforms, and warns about unusual profiles.")
    with st.expander("Original dataset & summary statistics"):
        st.dataframe(data, width="stretch")
        st.dataframe(derived["raw"].describe(), width="stretch")


def pca_clustering(bundle, data, derived):
    heading("Six dimensions, one readable map", "Understand what PCA preserves, what it loses, and how the segments occupy the reduced space.", "PCA & CLUSTERING")
    ratios = bundle["pca_variance"]
    metrics([("Two components", f"{ratios[:2].sum():.2%}", "Variance retained"),
             ("Four components", f"{ratios[:4].sum():.2%}", "Variance retained"),
             ("Variance omitted by PCA(2)", f"{1-ratios[:2].sum():.2%}", "Information outside the map")])
    chart(charts.variance(bundle))
    st.info(f"PCA(2) was retained primarily for the original project pathway and visualization. It loses {(1-ratios[:2].sum()):.2%} of total log-spending variance; this map cannot show every purchasing difference.")
    loadings = pd.DataFrame(bundle["pca_loadings"], columns=FEATURES, index=[f"PC{i}" for i in range(1, 7)])
    chart(charts.heatmap(loadings, "PCA component weights · all six components"))
    for i, col in enumerate(st.columns(2)):
        with col, st.container(border=True):
            st.subheader(f"Reading PC{i+1}")
            weights = loadings.iloc[i]
            dominant = weights.abs().nlargest(3).index
            st.write("Largest absolute weights: " + ", ".join(f"**{f} ({weights[f]:+.3f})**" for f in dominant) + ".")
            st.write("PC1 contrasts grocery, milk and household supplies with fresh/frozen purchasing." if i == 0 else
                     "PC2 mainly tracks fresh, frozen and delicatessen spending in the positive direction.")
            st.caption("Weights apply to natural-log spending. Component signs are coordinate conventions, not positive or negative business outcomes.")
    chart(charts.pca_scatter(bundle))


def performance(bundle, data, derived):
    heading("Model quality, without shortcuts", "The actual tested configurations from the validated project, reproduced and checked during offline export.", "MODEL PERFORMANCE")
    metrics([("Selected model", "K-Means", "2 clusters · seed 42"),
             ("Silhouette", f"{bundle['silhouette']:.6f}", "Measured in PCA space"),
             ("Training observations", bundle["metadata"]["training_rows"], "After the outlier policy")])
    comparison = bundle["comparison"]
    fig = px.line(comparison, x="Clusters", y="Silhouette", color="Algorithm", markers=True, color_discrete_sequence=charts.COLORS)
    fig.add_scatter(x=[2], y=[bundle["silhouette"]], mode="markers", name="Selected configuration",
                    marker=dict(symbol="star", size=18, color=charts.COLORS[2]))
    fig.update_xaxes(dtick=1)
    chart(charts.polish(fig))
    st.write("Higher silhouette indicates stronger within-cluster cohesion and between-cluster separation. It is **not accuracy**. The chosen model leads this primary comparison, but the small difference from GMM does not prove general superiority.")
    st.dataframe(comparison.style.format({"Silhouette": "{:.6f}"}), hide_index=True, width="stretch")
    st.subheader("Frozen model specification")
    info = {"Algorithm": "K-Means", "Clusters": bundle["kmeans"].n_clusters, "Input features": len(FEATURES),
            "Preprocessing": "Natural logarithm (np.log), no scaling", "Dimensionality reduction": "PCA (2 components), full SVD",
            "Silhouette": f"{bundle['silhouette']:.9f}", "Random state": bundle["kmeans"].random_state,
            "Initialization": bundle["kmeans"].n_init, "Training rows": bundle["metadata"]["training_rows"], "Bundle version": bundle["version"]}
    st.dataframe(pd.DataFrame({"Property": info.keys(), "Value": [str(v) for v in info.values()]}), hide_index=True, width="stretch")
    st.caption("This is historical, in-sample cluster quality. No supervised accuracy or future business uplift has been established.")
    with st.expander("Reproducibility & artifact provenance"):
        st.json(bundle["metadata"])
        st.write("Exporter assertions verify sample assignments, recovered centers, feature order, dimensions, finite artifacts, comparison scores, and external counts before writing the bundle.")


def about(bundle, data, derived):
    heading("Creating Customer Segments", "An interpretable unsupervised learning portfolio project for wholesale customer purchasing.", "ABOUT THE PROJECT")
    st.write(f"**Problem:** A wholesale distributor needs to understand different customer purchasing patterns. **Learning type:** Unsupervised learning. **Dataset:** {len(data)} historical customers with six annual product-spending categories. **Goal:** Discover natural purchasing groups that may support differentiated business strategies.")
    st.subheader("From source data to a frozen model")
    flow(["customers.csv", "Data audit", "Six spending features", "Natural log", "Tukey analysis", "PCA", "K-Means / GMM evaluation", "Final K-Means", "Segments", "External Channel comparison"])
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Training · offline only")
        flow(["Data", "Log", "Outlier policy", "PCA fit", "K-Means fit", "Export artifact"])
        st.write("The separate exporter recreates the validated baseline and checks it against notebook results. The app only loads the resulting artifact.")
    with right, st.container(border=True):
        st.subheader("Inference · in the application")
        flow(["New customer", "Validate", "Log", "Frozen PCA transform", "Frozen K-Means predict"])
        st.write("Exactly six positive, finite spending values. No training, no outlier deletion, and no Channel or Region inputs.")
    st.subheader("Business questions this can support")
    st.markdown("- **Delivery strategy experiments:** stratify trials by spending segment.\n- **Customer service differentiation:** investigate different service needs.\n- **Marketing targeting:** test relevant category offers.\n- **Purchasing analysis:** compare accounts with representative patterns.\n- **Account management:** use similarity examples to guide conversations.")
    st.info("Considering a change from 5-day to 3-day delivery? Run an A/B experiment within each segment, randomly assigning delivery schedules and tracking satisfaction, retention, purchasing, and operational costs. This dataset did not measure delivery-strategy effects and cannot prove which strategy increases revenue.")
    st.subheader("Key limitations")
    st.markdown(f"- Relatively small historical dataset; no temporal or prospective validation.\n- Moderate cluster overlap and sensitivity to unusual customers.\n- PCA(2) omits {(1-bundle['pca_variance'][:2].sum()):.2%} of log-spending variance.\n- Cluster names are business interpretations, not ground truth.\n- Channel comparison is an external reference only.\n- Initialization checks do not establish resampling stability.\n- Delivery and revenue effects require a controlled experiment.")
    st.caption("Original notebook, HTML report, dataset and instructor visualization helpers are preserved. Spending is expressed in monetary units, as in the source dataset.")


VIEWS = {"Overview": overview, "Customer Segmentation": prediction, "Segment Explorer": segment_explorer,
         "Data Insights": data_insights, "PCA & Clustering": pca_clustering,
         "Model Performance": performance, "About Project": about}

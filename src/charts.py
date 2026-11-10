"""Consistent Plotly charts; all normalization here is visualization only."""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import FEATURES

COLORS = ["#68DFCE", "#A99BFF", "#EEC578"]


def polish(fig, height=410):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial, sans-serif", color="#CBD5E1", size=12),
                      colorway=COLORS, margin=dict(l=20, r=20, t=45, b=45),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
                      hoverlabel=dict(bgcolor="#19283A", font_size=13))
    fig.update_xaxes(gridcolor="#243247", zerolinecolor="#35445B")
    fig.update_yaxes(gridcolor="#243247", zerolinecolor="#35445B")
    return fig


def pca_scatter(bundle, customer=None):
    fig = go.Figure()
    xy, labels = bundle["training_coordinates"], bundle["training_labels"]
    for segment in range(2):
        mask = labels == segment
        fig.add_trace(go.Scatter(x=xy[mask, 0], y=xy[mask, 1], mode="markers", name=f"Segment {segment}",
                                customdata=bundle["training_indices"][mask],
                                marker=dict(color=COLORS[segment], size=7, opacity=.65),
                                hovertemplate="Customer %{customdata}<br>PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra>%{fullData.name}</extra>"))
    centers = bundle["kmeans"].cluster_centers_
    fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1], name="Centroids", mode="markers+text",
                            text=["0", "1"], textposition="top center",
                            marker=dict(symbol="diamond", color="white", size=14, line=dict(color="#0C1424", width=2))))
    if customer is not None:
        fig.add_trace(go.Scatter(x=[customer[0]], y=[customer[1]], name="Your customer", mode="markers",
                                marker=dict(symbol="star", color=COLORS[2], size=22, line=dict(color="white", width=1))))
    fig.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    return polish(fig, 460)


def spending_chart(profiles, denominator=None, ylabel="Annual spending · monetary units"):
    fig = go.Figure()
    for i, (name, values) in enumerate(profiles.items()):
        values = np.asarray(values) / (np.asarray(denominator) if denominator is not None else 1)
        fig.add_bar(name=name, x=[f.replace("_", " & ") for f in FEATURES], y=values,
                    marker_color=COLORS[i], hovertemplate="%{x}<br>%{y:,.2f}<extra>%{fullData.name}</extra>")
    fig.update_layout(barmode="group", yaxis_title=ylabel)
    if denominator is not None:
        fig.add_hline(y=1, line_dash="dot", line_color="#7D8CA4")
    return polish(fig)


def radar(customer, representative, medians):
    fig = go.Figure()
    for i, (name, values) in enumerate([("Customer", customer), ("Segment representative", representative)]):
        # Log1p of median ratios keeps extreme radar profiles readable.
        r = np.log1p(np.asarray(values) / np.asarray(medians))
        fig.add_trace(go.Scatterpolar(r=np.r_[r, r[0]], theta=FEATURES + [FEATURES[0]],
                                     name=name, fill="toself", line_color=COLORS[i], opacity=.75))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(title="log(1 + ratio)", gridcolor="#35445B"),
                                 angularaxis=dict(gridcolor="#35445B")))
    return polish(fig, 410)


def variance(bundle):
    x = [f"PC{i}" for i in range(1, 7)]
    values = bundle["pca_variance"] * 100
    fig = go.Figure(go.Bar(x=x, y=values, name="Individual variance", marker_color=[COLORS[0]] * 2 + ["#44546D"] * 4))
    fig.add_scatter(x=x, y=np.cumsum(values), name="Cumulative variance", mode="lines+markers", line_color=COLORS[1])
    fig.update_layout(yaxis_title="Explained variance (%)", yaxis_range=[0, 105])
    return polish(fig)


def heatmap(frame, title):
    fig = px.imshow(frame, text_auto=".2f", color_continuous_scale=["#9C8AF5", "#172538", "#64DCC9"],
                    zmin=-1, zmax=1, aspect="auto", title=title)
    return polish(fig, 440)

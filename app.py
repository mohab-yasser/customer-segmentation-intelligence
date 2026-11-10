"""Streamlit entry point: load -> transform -> predict. Training lives offline."""
import streamlit as st
from src.config import PAGES
from src.data import get_bundle, get_data, describe_data
from src.pages import VIEWS
from src.ui import styles

st.set_page_config(page_title="Customer Segmentation Intelligence", page_icon="◈", layout="wide")
styles()
try:
    bundle = get_bundle()
    data = get_data(bundle["metadata"]["dataset_sha256"])
    derived = describe_data(data)
except Exception as exc:
    st.error(f"Unable to load the validated project: {exc}")
    st.info("Install the pinned requirements and restore the trusted artifact and dataset. If the artifact has not been generated, run `python export_model.py` offline, then restart Streamlit. The app never retrains automatically.")
    st.stop()

# Render content before the keyed navigation widget so the Overview CTA can
# safely change the active page before that widget is instantiated on this run.
selected = st.session_state.get("navigation", PAGES[0])
with st.sidebar:
    st.html('<div class="eyebrow">WHOLESALE ANALYTICS</div><div class="brand">Customer Segmentation<br>Intelligence</div><div class="fine">From spending to understanding.</div>')
    navigation_slot = st.empty()
    st.divider()
    st.caption("FROZEN MODEL · v" + bundle["version"])
    st.markdown("**K-Means · 2 Segments**")
    st.metric("Silhouette", f"{bundle['silhouette']:.3f}")
    st.metric("PCA retained", f"{bundle['pca_variance'][:2].sum():.2%}")
    st.divider()
    st.caption("Unsupervised ML Portfolio Project")
    st.caption("Six categories. Two purchasing patterns.")

VIEWS[selected](bundle, data, derived)
with navigation_slot.container():
    st.radio("Navigation", PAGES, key="navigation", label_visibility="collapsed")
st.divider()
st.caption("CUSTOMER SEGMENTATION INTELLIGENCE  /  Historical patterns, informed experiments.")

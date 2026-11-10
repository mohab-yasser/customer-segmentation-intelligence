from html import escape
import streamlit as st


def styles():
    st.html("""<style>
    .stApp {background: radial-gradient(ellipse at 85% 0%, #16283b 0%, #0b1220 52%);}
    h1,h2,h3 {letter-spacing:-.035em;}
    h1 {font-weight:750 !important;}
    .eyebrow {color:#68dfce;font-size:.72rem;letter-spacing:.17em;font-weight:700;margin:0 0 12px;}
    .hero {padding:30px 34px;border:1px solid #2a3d52;border-radius:20px;
      background:linear-gradient(120deg,#182d3e,#172037 70%,#25213d);margin:8px 0 24px;}
    .hero h1 {font-size:clamp(2rem,3.3vw,3.5rem);line-height:1.12;max-width:850px;margin:8px 0 16px;}
    .hero p {color:#b6c6d7;max-width:740px;font-size:1rem;line-height:1.7;}
    .badge {display:inline-block;border:1px solid #426c70;border-radius:30px;padding:5px 11px;
      color:#93e6d7;font-size:.72rem;letter-spacing:.04em;}
    .metric-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:8px 0 26px;}
    .metric-card {background:#121e2f;border:1px solid #27384d;border-radius:14px;padding:20px;}
    .metric-label {color:#9eb0c6;font-size:.76rem;min-height:22px;}
    .metric-value {color:#eff6ff;font-size:1.85rem;font-weight:650;letter-spacing:-.04em;}
    .metric-note {color:#7e95ae;font-size:.7rem;margin-top:6px;}
    .flow {display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:16px 0 24px;}
    .flow span {padding:12px 16px;border:1px solid #31455b;background:#142333;border-radius:10px;color:#d4e5ee;font-size:.83rem;}
    .flow b {color:#6b8299;}
    .result {padding:25px 28px;border:1px solid #467f7b;border-left:4px solid #68dfce;
      border-radius:14px;background:linear-gradient(110deg,#173037,#172139);margin:18px 0;}
    .result h2 {font-size:1.5rem;color:#f0f8ff;margin:4px 0 12px;}
    .result p {color:#b8cddd;line-height:1.7;}
    .brand {font-size:1.14rem;font-weight:700;color:#eef7ff;line-height:1.4;margin-bottom:8px;}
    .fine {color:#8da3ba;font-size:.78rem;line-height:1.65;}
    @media(max-width:600px){.hero{padding:22px}.metric-card{padding:16px}.result{padding:20px}}
    </style>""")


def metrics(items):
    cards = "".join(f'<div class="metric-card"><div class="metric-label">{escape(str(label))}</div>'
                    f'<div class="metric-value">{escape(str(value))}</div><div class="metric-note">{escape(str(note))}</div></div>'
                    for label, value, note in items)
    st.html(f'<div class="metric-grid">{cards}</div>')


def heading(title, subtitle, eyebrow="CUSTOMER INTELLIGENCE"):
    st.html(f'<div class="eyebrow">{escape(eyebrow)}</div>')
    st.title(title)
    st.caption(subtitle)


def flow(steps):
    st.html('<div class="flow">' + '<b>→</b>'.join(f'<span>{escape(step)}</span>' for step in steps) + '</div>')


def chart(fig, key=None):
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "responsive": True, "scrollZoom": False})

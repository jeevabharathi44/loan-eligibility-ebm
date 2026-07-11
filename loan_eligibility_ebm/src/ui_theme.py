"""
ui_theme.py
------------
Shared visual identity for the Explainable Loan Eligibility app.

Design concept: a bank "case file" — ledger-navy backgrounds, paper-cream
cards, brass accents, and a rotated stamp motif that stands in for the
idea of a transparent, accountable decision (as opposed to a black-box
rubber stamp). Import `inject_theme()` at the top of every page.
"""

import streamlit as st

COLORS = {
    "navy": "#0F1C2B",
    "navy_light": "#16283C",
    "navy_border": "#26405C",
    "paper": "#F4EFE2",
    "paper_dim": "#E7DFC9",
    "ink": "#1B2430",
    "ink_soft": "#4B5563",
    "brass": "#C6A15B",
    "brass_dark": "#9C7C3F",
    "approved": "#3F6B4F",
    "approved_soft": "#DCE7DE",
    "rejected": "#A8432E",
    "rejected_soft": "#F0DCD4",
}


def inject_theme():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: {COLORS['navy']};
        }}

        section[data-testid="stSidebar"] {{
            background: {COLORS['navy_light']};
            border-right: 1px solid {COLORS['navy_border']};
        }}
        section[data-testid="stSidebar"] * {{
            color: {COLORS['paper']} !important;
        }}

        h1, h2, h3 {{
            font-family: 'Source Serif 4', serif;
            color: {COLORS['paper']};
        }}
        p, li, span, label, div {{
            color: {COLORS['paper']};
        }}

        .eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.75rem;
            color: {COLORS['brass']};
            margin-bottom: 0.4rem;
        }}

        .case-card {{
            background: {COLORS['paper']};
            border-radius: 4px;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.28);
            border-top: 3px solid {COLORS['brass']};
        }}
        .case-card, .case-card p, .case-card li, .case-card span, .case-card label {{
            color: {COLORS['ink']} !important;
        }}
        .case-card h1, .case-card h2, .case-card h3, .case-card h4 {{
            color: {COLORS['ink']} !important;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {COLORS['paper']};
            border-radius: 4px !important;
            border: 1px solid {COLORS['paper_dim']} !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] * {{
            color: {COLORS['ink']} !important;
        }}

        .stButton > button {{
            background: {COLORS['brass']};
            color: {COLORS['navy']};
            border: none;
            border-radius: 4px;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
            letter-spacing: 0.02em;
            transition: background 0.15s ease;
        }}
        .stButton > button:hover {{
            background: {COLORS['brass_dark']};
            color: {COLORS['paper']};
        }}
        .stButton > button[kind="secondary"] {{
            background: transparent;
            color: {COLORS['brass']};
            border: 1px solid {COLORS['brass']};
        }}

        hr {{
            border-color: {COLORS['navy_border']};
        }}

        .feature-strip {{
            display: flex;
            gap: 1.2rem;
        }}
        .feature-block {{
            flex: 1;
            background: {COLORS['navy_light']};
            border: 1px solid {COLORS['navy_border']};
            border-radius: 4px;
            padding: 1.2rem 1.3rem;
        }}
        .feature-block .mark {{
            font-family: 'IBM Plex Mono', monospace;
            color: {COLORS['brass']};
            font-size: 0.8rem;
            letter-spacing: 0.1em;
        }}

        .reason-list {{
            list-style: none;
            padding-left: 0;
            margin: 0;
        }}
        .reason-list li {{
            padding: 0.5rem 0 0.5rem 1.2rem;
            border-bottom: 1px solid {COLORS['paper_dim']};
            position: relative;
            font-size: 0.94rem;
        }}
        .reason-list.support li::before {{
            content: "+";
            position: absolute;
            left: 0;
            color: {COLORS['approved']};
            font-weight: 700;
        }}
        .reason-list.oppose li::before {{
            content: "–";
            position: absolute;
            left: 0;
            color: {COLORS['rejected']};
            font-weight: 700;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_stamp(decision: str, subtext: str = "EBM ASSESSED") -> str:
    """Returns HTML for a rotated stamp badge: 'Approved' or 'Rejected'."""
    is_approved = decision.lower().startswith("approv")
    color = COLORS["approved"] if is_approved else COLORS["rejected"]
    label = "APPROVED" if is_approved else "REJECTED"
    return f"""
    <div style="
        width: 168px; height: 168px;
        border: 3px solid {color};
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        transform: rotate(-9deg);
        margin: 0 auto;
        position: relative;
        background: repeating-radial-gradient(circle, transparent, transparent 78px, {color}11 80px);
    ">
        <div style="
            width: 148px; height: 148px;
            border: 1px solid {color};
            border-radius: 50%;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            text-align: center;
        ">
            <div style="
                font-family: 'Source Serif 4', serif;
                font-weight: 700;
                font-size: 1.3rem;
                letter-spacing: 0.05em;
                color: {color};
                line-height: 1.1;
            ">{label}</div>
            <div style="
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.55rem;
                letter-spacing: 0.14em;
                color: {color};
                margin-top: 6px;
            ">{subtext}</div>
        </div>
    </div>
    """


def render_seal(text_top: str = "TRANSPARENT", text_bottom: str = "BY DESIGN") -> str:
    """Decorative (non-decision) seal used on the landing page hero."""
    return f"""
    <div style="
        width: 210px; height: 210px;
        border: 2px solid {COLORS['brass']};
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        transform: rotate(6deg);
        margin: 0 auto;
        background: {COLORS['navy_light']};
    ">
        <div style="
            width: 182px; height: 182px;
            border: 1px dashed {COLORS['brass']};
            border-radius: 50%;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            text-align: center;
        ">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:0.16em;color:{COLORS['brass']};">{text_top}</div>
            <div style="font-family:'Source Serif 4',serif;font-weight:600;font-size:1.05rem;color:{COLORS['paper']};margin:6px 0;">EBM</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;letter-spacing:0.16em;color:{COLORS['brass']};">{text_bottom}</div>
        </div>
    </div>
    """


def plotly_theme_layout(height: int = 320) -> dict:
    """Common Plotly layout kwargs so every chart matches the case-file theme."""
    return dict(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["ink"], size=13),
    )

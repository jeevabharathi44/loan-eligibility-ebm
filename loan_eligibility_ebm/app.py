"""
app.py
--------
Landing page for "Explainable Loan Eligibility Prediction".

This page introduces the tool and shows what the model weighs most
overall. The actual applicant form now lives on its own page
(pages/1_Check_Eligibility.py), and results are shown on a dedicated
results page (pages/2_Result.py).

Run with:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go

from src.model_utils import load_model_and_columns
from src.explain_utils import get_global_feature_importance
from src.ui_theme import inject_theme, render_seal, plotly_theme_layout, COLORS

st.set_page_config(
    page_title="Explainable Loan Eligibility Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()


def render_hero():
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown('<div class="eyebrow">EXPLAINABLE CREDIT DECISIONS</div>', unsafe_allow_html=True)
        st.markdown("## Every approval, accountable.")
        st.markdown(
            "An Explainable Boosting Machine reviews each application and shows "
            "its full reasoning — not an approximation of it. No black box, "
            "no post-hoc guesswork: the number on the screen **is** the model's "
            "actual decision math, translated into plain English for the desk."
        )
        st.write("")
        if st.button("Begin Assessment →", type="primary"):
            st.switch_page("pages/1_Check_Eligibility.py")
    with right:
        st.markdown(render_seal(), unsafe_allow_html=True)


def render_feature_strip():
    st.write("")
    st.markdown(
        f"""
        <div class="feature-strip">
            <div class="feature-block">
                <div class="mark">GLASSBOX MODEL</div>
                <p style="margin-top:0.5rem;">Built on an EBM, whose additive structure means every
                prediction can be decomposed feature-by-feature — no SHAP or LIME
                approximation required.</p>
            </div>
            <div class="feature-block">
                <div class="mark">CASE-LEVEL REASONING</div>
                <p style="margin-top:0.5rem;">Each assessment names the specific factors that helped
                and hurt an application, in language built for the desk, not the lab.</p>
            </div>
            <div class="feature-block">
                <div class="mark">AUDIT READY</div>
                <p style="margin-top:0.5rem;">Because the reasoning is exact rather than approximated,
                every decision holds up the same way on review as it did at approval.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_global_importance(model):
    st.write("")
    st.write("")
    st.markdown('<div class="eyebrow">MODEL OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown("### What the model weighs most, overall")
    st.markdown(
        '<div class="case-card">',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Averaged across every applicant the model has seen — not any single case. "
        "This is the model's general policy, before any one applicant's details are applied."
    )

    importance_df = get_global_feature_importance(model)
    importance_df["feature"] = importance_df["feature"].str.replace("_", " ").str.title()
    importance_df = importance_df.sort_values("importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker_color=COLORS["brass"],
        )
    )
    fig.update_layout(**plotly_theme_layout(height=380))
    fig.update_xaxes(title="Average impact on the decision", gridcolor=COLORS["paper_dim"])
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    render_hero()
    render_feature_strip()
    model, _columns_info = load_model_and_columns()
    render_global_importance(model)


if __name__ == "__main__":
    main()

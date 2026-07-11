"""
pages/2_Result.py
-------------------
The results page. Reads the applicant's assessment out of
session_state (written by pages/1_Check_Eligibility.py) and renders:
  1. A stamp (Approved/Rejected) + confidence gauge.
  2. A tornado chart of the strongest per-applicant contributions.
  3. A donut chart showing the overall balance of supporting vs.
     opposing evidence.
  4. The plain-English reason lists.

This page never runs a prediction itself — it only ever displays one
that pages/1_Check_Eligibility.py already computed, so refreshing this
page can't silently recompute a stale form state.
"""

import streamlit as st
import plotly.graph_objects as go

from src.explain_utils import get_contribution_chart_data, get_evidence_balance
from src.ui_theme import inject_theme, render_stamp, plotly_theme_layout, COLORS

st.set_page_config(
    page_title="Assessment Result",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_theme()


def render_empty_state():
    st.markdown('<div class="eyebrow">NO ASSESSMENT YET</div>', unsafe_allow_html=True)
    st.markdown("## Nothing to show here")
    st.markdown(
        "There's no applicant case loaded. Start a new assessment to see a result."
    )
    if st.button("Start New Assessment →", type="primary"):
        st.switch_page("pages/1_Check_Eligibility.py")


def render_confidence_gauge(proba: float, is_approved: bool):
    color = COLORS["approved"] if is_approved else COLORS["rejected"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 34, "family": "IBM Plex Mono"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["ink_soft"]},
                "bar": {"color": color},
                "bgcolor": COLORS["paper_dim"],
                "borderwidth": 0,
                "threshold": {
                    "line": {"color": COLORS["ink"], "width": 2},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(**plotly_theme_layout(height=220))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_tornado_chart(chart_df):
    if chart_df.empty:
        st.info("No strong individual factors were found for this applicant.")
        return
    colors = [COLORS["approved"] if d == "Helped" else COLORS["rejected"] for d in chart_df["direction"]]
    fig = go.Figure(
        go.Bar(
            x=chart_df["score"],
            y=chart_df["feature"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(**plotly_theme_layout(height=340))
    fig.add_vline(x=0, line_color=COLORS["ink_soft"], line_width=1)
    fig.update_xaxes(title="Contribution to decision", gridcolor=COLORS["paper_dim"], zeroline=False)
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_donut_chart(balance: dict):
    total = balance["Helped"] + balance["Hurt"]
    if total <= 0:
        st.info("Not enough signal to show an evidence balance for this applicant.")
        return
    fig = go.Figure(
        go.Pie(
            labels=["Helped", "Hurt"],
            values=[balance["Helped"], balance["Hurt"]],
            hole=0.62,
            marker=dict(colors=[COLORS["approved"], COLORS["rejected"]]),
            textinfo="percent",
            textfont=dict(family="IBM Plex Mono", size=13, color=COLORS["paper"]),
            sort=False,
        )
    )
    fig.update_layout(
        **plotly_theme_layout(height=260),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, font=dict(color=COLORS["ink"])),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_reason_lists(result):
    col_support, col_oppose = st.columns(2)
    with col_support:
        st.markdown("**Factors that helped**")
        if result["supporting_reasons"]:
            items = "".join(f"<li>{r}</li>" for r in result["supporting_reasons"])
            st.markdown(f'<ul class="reason-list support">{items}</ul>', unsafe_allow_html=True)
        else:
            st.caption("No strongly positive factors found.")
    with col_oppose:
        st.markdown("**Factors that hurt**")
        if result["opposing_reasons"]:
            items = "".join(f"<li>{r}</li>" for r in result["opposing_reasons"])
            st.markdown(f'<ul class="reason-list oppose">{items}</ul>', unsafe_allow_html=True)
        else:
            st.caption("No strongly negative factors found.")


def main():
    if "assessment_result" not in st.session_state:
        render_empty_state()
        return

    result = st.session_state["assessment_result"]
    X_row = st.session_state["assessment_X_row"]
    model = st.session_state["assessment_model"]

    is_approved = result["decision"] == "Approved"

    top_left, top_right = st.columns([1, 3])
    with top_left:
        if st.button("← New Assessment", type="secondary"):
            st.switch_page("pages/1_Check_Eligibility.py")
    with top_right:
        st.markdown('<div class="eyebrow">ASSESSMENT RESULT</div>', unsafe_allow_html=True)

    st.write("")
    stamp_col, gauge_col, summary_col = st.columns([1, 1.3, 1.7], gap="large")
    with stamp_col:
        st.markdown(render_stamp(result["decision"]), unsafe_allow_html=True)
    with gauge_col:
        render_confidence_gauge(result["probability"], is_approved)
    with summary_col:
        st.markdown('<div class="case-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown(f"#### {result['decision']}")
        st.write(result["summary_sentence"])
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.divider()
    st.markdown("### How the model got here")

    chart_col, donut_col = st.columns([2, 1], gap="large")
    with chart_col:
        st.markdown('<div class="case-card">', unsafe_allow_html=True)
        st.markdown("**Strongest individual factors**")
        chart_df = get_contribution_chart_data(model, X_row, top_k=6)
        render_tornado_chart(chart_df)
        st.markdown("</div>", unsafe_allow_html=True)
    with donut_col:
        st.markdown('<div class="case-card">', unsafe_allow_html=True)
        st.markdown("**Overall weight of evidence**")
        balance = get_evidence_balance(model, X_row)
        render_donut_chart(balance)
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="case-card">', unsafe_allow_html=True)
    st.markdown("### Why this decision, in plain English")
    render_reason_lists(result)
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

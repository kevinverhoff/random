import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px

st.set_page_config(page_title="Survey Analysis Tool", layout="wide")

st.title("📊 Survey Analysis Tool")

st.markdown("""
Upload a CSV of survey responses (one row per participant, one column per question).  
You’ll be able to:
- Select **up to three attributes** (e.g., Region, Location, or Role)
- Classify question columns as **numeric** or **categorical**
- Identify **positive responses** for categorical questions
- View **charts and significance tests** (ANOVA or Chi²)
""")

# --- Session reset ---
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if st.button("🔄 Reset App"):
    reset_app()

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your survey CSV", type=["csv"], key="file_uploader")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.uploaded_file = uploaded_file

    st.success(f"✅ Loaded {df.shape[0]} responses and {df.shape[1]} columns.")
    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head())

    # --- Attribute selection ---
    st.header("Step 1: Identify Attributes")
    attrs = st.multiselect(
        "Select up to three attribute columns (for grouping)",
        options=df.columns.tolist(),
        max_selections=3,
        key="attr_select"
    )

    # --- Identify question columns ---
    st.header("Step 2: Identify Question Columns")
    question_cols = [c for c in df.columns if c not in attrs]
    st.write(f"Detected {len(question_cols)} potential question columns.")

    # --- Auto-detect types ---
    auto_types = {
        c: "numeric" if np.issubdtype(df[c].dropna().dtype, np.number) else "categorical"
        for c in question_cols
    }

    st.subheader("Question Type Overrides")
    q_types = {}
    for col in question_cols:
        q_types[col] = st.selectbox(
            f"Type of question '{col}'",
            ["categorical", "numeric"],
            index=0 if auto_types[col] == "categorical" else 1,
            key=f"type_{col}"
        )

    # --- Define positive responses ---
    st.header("Step 3: Define Positive Responses (Categorical)")
    positive_map = {}
    for col in [c for c in question_cols if q_types[c] == "categorical"]:
        unique_vals = sorted(df[col].dropna().unique())
        pos_vals = st.multiselect(
            f"Select positive responses for '{col}'",
            options=unique_vals,
            key=f"pos_{col}"
        )
        positive_map[col] = pos_vals

    # --- Run Analysis ---
    st.header("Step 4: Run Analysis")
    col_run, col_reset = st.columns([1, 1])
    with col_run:
        run = st.button("🚀 Run Survey Analysis", use_container_width=True)
    with col_reset:
        st.button("🔄 Reset All", use_container_width=True, on_click=reset_app)

    if run:
        st.divider()
        for q in question_cols:
            st.subheader(f"🧩 Question: {q}")
            qtype = q_types[q]

            for attr in attrs:
                st.markdown(f"**Breakdown by:** `{attr}`")

                if qtype == "numeric":
                    # --- Numeric Analysis ---
                    fig = px.box(df, x=attr, y=q, points="all", title=f"{q} by {attr}")
                    st.plotly_chart(fig, use_container_width=True)

                    groups = [group[q].dropna() for _, group in df.groupby(attr)]
                    if len(groups) > 1 and all(len(g) > 1 for g in groups):
                        f_val, p_val = stats.f_oneway(*groups)
                        st.write(f"**ANOVA:** F = {f_val:.3f}, p = {p_val:.3g}")
                    else:
                        st.info("Not enough groups or data for ANOVA.")

                elif qtype == "categorical":
                    # --- Convert to Positive / Not Positive ---
                    df["_positive_"] = df[q].isin(positive_map[q]).astype(int)

                    summary = df.groupby(attr)["_positive_"].mean().reset_index()
                    summary["_positive_"] *= 100
                    fig = px.bar(
                        summary,
                        x=attr,
                        y="_positive_",
                        title=f"Percent Positive for '{q}' by {attr}",
                        labels={"_positive_": "% Positive"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Chi-square test
                    contingency = pd.crosstab(df[attr], df["_positive_"])
                    if contingency.shape[0] > 1:
                        chi2, p, dof, exp = stats.chi2_contingency(contingency)
                        st.write(f"**Chi² test:** χ² = {chi2:.3f}, p = {p:.3g}")
                    else:
                        st.info("Not enough groups for Chi² test.")

                    df.drop(columns="_positive_", inplace=True)

            st.divider()
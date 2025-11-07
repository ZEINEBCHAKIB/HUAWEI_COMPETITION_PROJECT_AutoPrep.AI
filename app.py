import os
import io
import json
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

from pipeline import AutoPreprocessor
from report import generate_pdf_report

st.set_page_config(page_title="AUTOPREP.AI - Smart Data Preprocessing", layout="wide", page_icon="☁️")

# Display logo if exists
logo_path = "assets/logo.png"
if os.path.exists(logo_path):
    col_logo, col_spacer = st.columns([1, 3])
    with col_logo:
        st.image(logo_path, width=300)
else:
    st.title("AUTOPREP.AI")
    st.subheader("Smart Data Preprocessing")

st.caption("Upload a CSV, let AI decide the best cleaning strategies, and get a ready-to-use dataset with a detailed PDF report.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    target_col = st.text_input("Target column (optional)", value="")
    low_card_threshold = st.number_input("Low cardinality threshold (One-Hot)", min_value=2, max_value=200, value=10)
    high_missing_threshold = st.slider("Missing columns removal threshold (%)", min_value=0, max_value=100, value=30)
    apply_outlier_treatment = st.checkbox("Treat outliers (IQR)", value=True)
    scaling_enabled = st.checkbox("Enable scaling (numeric)", value=True)

uploaded = st.file_uploader("Upload CSV (UTF-8)", type=["csv"]) 

if 'state' not in st.session_state:
    st.session_state['state'] = {
        'df_raw': None,
        'df_processed': None,
        'decisions': None,
        'before_stats': None,
        'after_stats': None,
        'pdf_bytes': None,
        'csv_bytes': None,
    }

col1, col2, col3 = st.columns([1,1,1])

with col1:
    analyze = st.button("Analyze & Preprocess", use_container_width=True, type="primary")
with col2:
    download_csv_btn = st.button("Prepare Preprocessed CSV", use_container_width=True)
with col3:
    download_pdf_btn = st.button("Prepare PDF", use_container_width=True)

if uploaded is not None and st.session_state['state']['df_raw'] is None:
    try:
        df_raw = pd.read_csv(uploaded)
        st.session_state['state']['df_raw'] = df_raw
        st.success(f"Dataset loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
    except Exception as e:
        st.error(f"CSV reading error: {e}")

if analyze:
    if st.session_state['state']['df_raw'] is None:
        st.warning("Please upload a CSV file.")
    else:
        with st.spinner("Running preprocessing pipeline..."):
            # LLM config comes from environment variables (.env or system)
            llm_model = os.getenv("LLM_MODEL_NAME", "qwen3-32b")
            pre = AutoPreprocessor(
                target_column=target_col.strip() or None,
                low_card_threshold=int(low_card_threshold),
                high_missing_threshold=float(high_missing_threshold)/100.0,
                apply_outlier_treatment=apply_outlier_treatment,
                scaling_enabled=scaling_enabled,
                llm_model=llm_model,
            )
            result = pre.fit_transform(st.session_state['state']['df_raw'])
            st.session_state['state']['df_processed'] = result.processed_df
            st.session_state['state']['decisions'] = result.decisions
            st.session_state['state']['before_stats'] = result.before_stats
            st.session_state['state']['after_stats'] = result.after_stats
        st.success("Preprocessing completed.")

if st.session_state['state']['df_processed'] is not None:
    st.subheader("Preprocessed Dataset Preview")
    st.dataframe(st.session_state['state']['df_processed'].head(20))

if download_csv_btn:
    if st.session_state['state']['df_processed'] is None:
        st.warning("Please run analysis and preprocessing first.")
    else:
        csv_bytes = st.session_state['state']['df_processed'].to_csv(index=False).encode('utf-8')
        st.session_state['state']['csv_bytes'] = csv_bytes
        st.success("CSV ready for download.")

if download_pdf_btn:
    if st.session_state['state']['df_processed'] is None:
        st.warning("Please run analysis and preprocessing first.")
    else:
        with st.spinner("Generating PDF report..."):
            pdf_buf = generate_pdf_report(
                original_df=st.session_state['state']['df_raw'],
                processed_df=st.session_state['state']['df_processed'],
                decisions=st.session_state['state']['decisions'],
                before_stats=st.session_state['state']['before_stats'],
                after_stats=st.session_state['state']['after_stats'],
            )
            st.session_state['state']['pdf_bytes'] = pdf_buf.getvalue()
        st.success("PDF ready for download.")

# Downloads
if st.session_state['state']['csv_bytes']:
    st.download_button(
        label="Download Preprocessed CSV",
        data=st.session_state['state']['csv_bytes'],
        file_name="dataset_preprocessed.csv",
        mime="text/csv",
        use_container_width=True,
    )

if st.session_state['state']['pdf_bytes']:
    st.download_button(
        label="Download PDF Report",
        data=st.session_state['state']['pdf_bytes'],
        file_name="rapport_preprocessing.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

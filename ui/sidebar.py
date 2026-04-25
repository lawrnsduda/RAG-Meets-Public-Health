"""
Sidebar: pipeline mode, evidence sources, model selection.

Production version that reads its options from config.py, so app.py
stays in sync with the config. app_preview.py uses a hardcoded sidebar
instead.
"""
import streamlit as st
from config import PIPELINE_MODES, EVIDENCE_SOURCES, TOP_K


def render_sidebar() -> tuple[str, list[str], str, int]:
    with st.sidebar:
        st.header("Pipeline Configuration")

        mode = st.radio(
            "Pipeline Mode",
            options=list(PIPELINE_MODES.keys()),
            format_func=lambda m: f"{m} — {PIPELINE_MODES[m]['description']}",
        )

        st.divider()

        st.subheader("Evidence Sources")
        selected_sources = [
            src
            for src, cfg in EVIDENCE_SOURCES.items()
            if st.checkbox(cfg["label"], value=cfg["enabled"], key=f"src_{src}")
        ]

        st.divider()

        st.subheader("Model")
        llm_choice = st.selectbox(
            "LLM",
            options=["qwen3:8b", "mistral:7b"],
            index=0,
        )

        top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=10, value=TOP_K)

        return mode, selected_sources, llm_choice, top_k

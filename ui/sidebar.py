"""
Sidebar: pipeline mode, evidence sources, model selection.

This is the "production" version of the sidebar it reads its options
from config.py. In app_preview.py the sidebar is hardcoded, because we don't need real config for the UI preview.
When the real pipeline is working, app.py uses this module instead so
everything stays in sync with the config.
"""
import streamlit as st
from config import PIPELINE_MODES, EVIDENCE_SOURCES, TOP_K


def render_sidebar() -> tuple[str, list[str], str, int]:
    """
    Render the sidebar and return the user's selections.

    Returns a tuple of:
      - mode:             which pipeline mode to run ("Baseline", "RAG-only", "RAG+NLI")
      - selected_sources: list of source keys the user checked (f.ex. ["WHO", "CDC"])
      - llm_choice:       which Ollama model to use ("qwen3:8b" or "mistral:7b")
      - top_k:            how many evidence passages to retrieve
    """
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

"""
UI preview with hardcoded dummy data, so the UI can be developed without
Ollama, FAISS, or any models running.
"""
import streamlit as st
import time
from ui.styles import inject_css
from ui.components import (
    render_verdict_box,
    render_source_cards,
    render_detail_table,
)

# Must be the very first Streamlit call — raises an error otherwise.
st.set_page_config(page_title="Health Factchecker", page_icon="🩺", layout="wide")

inject_css()

with st.sidebar:
    st.header("Pipeline Configuration")

    st.radio("Pipeline mode", ["Baseline", "RAG-only", "RAG+NLI"], index=2)

    st.divider()

    st.subheader("Evidence Sources")
    st.checkbox("WHO Global Health Observatory", value=True)
    st.checkbox("CDC", value=True)
    st.checkbox("ECDC", value=True)
    st.checkbox("Our World in Data", value=True)
    st.checkbox("PubMed", value=True)

    st.divider()

    st.subheader("LLM")
    st.selectbox("Model", ["qwen3", "mistral"])

    st.slider("Top-k passages", 1, 15, 5)

st.title("🩺 Health Factchecker")
st.caption("Active mode: **RAG+NLI** — Full pipeline — retrieval + NLI + LLM")

with st.form("claim_form"):
    claim = st.text_input(
        "Enter a health claim to verify:",
        value="The MMR vaccine has been scientifically proven to cause autism in children.",
    )
    submitted = st.form_submit_button("Check claim", type="primary")

if submitted:
    with st.spinner("Running pipeline…"):
        time.sleep(1)

    evidence = [
        {"text": "Two doses of MMR vaccine are about 97% effective at preventing measles.", "source": "CDC", "score": 0.1823, "url": "https://www.cdc.gov/measles/vaccines/index.html"},
        {"text": "A 2014 meta-analysis of over 1.2 million children found no association between MMR vaccination and autism.", "source": "PubMed", "score": 0.2451, "url": "https://pubmed.ncbi.nlm.nih.gov/24814559/"},
        {"text": "Measles vaccination has prevented an estimated 56 million deaths between 2000 and 2021.", "source": "WHO", "score": 0.2987, "url": "https://www.who.int/news-room/fact-sheets/detail/measles"},
        {"text": "Measles-containing vaccine first-dose coverage reached 83% globally in 2023.", "source": "OWID", "score": 0.3102, "url": "https://ourworldindata.org/measles-vaccines-save-lives"},
        {"text": "In 2023, measles cases in the WHO European Region increased significantly compared to 2022.", "source": "ECDC", "score": 0.3344, "url": "https://www.ecdc.europa.eu/en/measles"},
    ]

    nli = [
        {"label": "contradiction", "scores": {"contradiction": 0.9412, "entailment": 0.0201, "neutral": 0.0387}},
        {"label": "contradiction", "scores": {"contradiction": 0.9678, "entailment": 0.0112, "neutral": 0.0210}},
        {"label": "contradiction", "scores": {"contradiction": 0.8891, "entailment": 0.0456, "neutral": 0.0653}},
        {"label": "neutral", "scores": {"contradiction": 0.2103, "entailment": 0.1544, "neutral": 0.6353}},
        {"label": "neutral", "scores": {"contradiction": 0.1892, "entailment": 0.2201, "neutral": 0.5907}},
    ]

    st.divider()

    render_verdict_box(
        "REFUTED",
        "NLI analysis confirms strong contradiction between the claim and "
        "retrieved evidence. The CDC source [1] and PubMed meta-analysis [2] "
        "both contradict the claim with high confidence (0.94 and 0.97). "
        "WHO data [3] further contradicts the premise. The claim is "
        "definitively refuted.",
    )

    st.subheader("Retrieved Evidence")
    render_source_cards(evidence, nli)

    with st.expander("Details", expanded=False):
        render_detail_table(evidence, nli)

import streamlit as st
from ui.styles import inject_css
from ui.components import (
    render_verdict_box,
    render_source_cards,
    render_detail_table,
)

st.set_page_config(page_title="Health Factchecker", page_icon="🩺", layout="wide")
inject_css()

with st.sidebar:
    st.header("Pipeline Configuration")

    mode = st.radio("Pipeline mode", ["Baseline", "RAG-only", "RAG+NLI"], index=2)

    st.divider()

    st.subheader("Evidence Sources")
    sources: list[str] = []
    if st.checkbox("WHO Global Health Observatory", value=True):
        sources.append("WHO")
    if st.checkbox("CDC", value=True):
        sources.append("CDC")
    if st.checkbox("ECDC", value=True):
        sources.append("ECDC")
    if st.checkbox("Our World in Data", value=True):
        sources.append("OWID")
    if st.checkbox("PubMed", value=True):
        sources.append("PubMed")

    st.divider()

    st.subheader("LLM")
    llm_choice = st.selectbox("Model", ["qwen3", "mistral"])
    top_k = st.slider("Top-k passages", 1, 15, 5)

MODE_DESCRIPTIONS = {
    "Baseline": "LLM only — no retrieval, no NLI",
    "RAG-only": "FAISS retrieval + LLM",
    "RAG+NLI":  "Full pipeline — retrieval + NLI + LLM",
}

st.title("🩺 Health Factchecker")
st.caption(f"Active mode: **{mode}** — {MODE_DESCRIPTIONS[mode]}")

with st.form("claim_form"):
    claim = st.text_input(
        "Enter a health claim to verify:",
    )
    submitted = st.form_submit_button("Check claim", type="primary")

if submitted and claim.strip():
    evidence: list[dict] = []
    nli: list[dict] = []
    verdict_result: dict = {}

    use_retrieval = mode in ("RAG-only", "RAG+NLI")
    use_nli = mode == "RAG+NLI"

    steps = []
    if use_retrieval:
        steps.append(("Retrieving evidence from index…", 33))
    if use_nli:
        steps.append(("Running NLI classification…", 66))
    steps.append((f"Generating verdict with {llm_choice}…", 100))

    progress = st.progress(0, text="Starting…")

    try:
        if use_retrieval:
            progress.progress(5, text="Retrieving evidence from index… 5%")
            from pipeline.retrieval import retrieve_evidence
            evidence = retrieve_evidence(
                claim,
                top_k=top_k,
                sources=sources if sources else None,
            )
            retrieval_pct = 33 if use_nli else 50
            progress.progress(retrieval_pct, text=f"Evidence retrieved. {retrieval_pct}%")
            if not evidence:
                st.warning("No evidence passages found. Try different sources or a broader claim.")

        if use_nli and evidence:
            progress.progress(40, text="Running NLI classification… 40%")
            from pipeline.nli import classify_nli
            nli = classify_nli(claim, evidence)
            progress.progress(66, text="NLI done. 66%")

        progress.progress(70, text=f"Generating verdict with {llm_choice}… 70%")
        from pipeline.verdict import generate_verdict
        verdict_result = generate_verdict(claim, evidence, nli, mode, llm_choice)
        progress.progress(100, text="Done. 100%")
        progress.empty()

    except FileNotFoundError as e:
        st.error(
            "**FAISS index not found.** "
            "Run the following commands first:\n\n"
            "```\npython -m data.fetch_all\npython data/build_index.py\n```"
        )
        st.stop()
    except Exception as e:
        st.error(f"**Pipeline error:** {e}")
        st.stop()

    st.divider()

    render_verdict_box(
        verdict_result.get("verdict", "UNKNOWN"),
        verdict_result.get("justification", ""),
    )

    if evidence:
        nli_display = nli if nli else [{}] * len(evidence)

        st.subheader("Retrieved Evidence")
        render_source_cards(evidence, nli_display)

        with st.expander("Details", expanded=False):
            render_detail_table(evidence, nli_display)

"""
Reusable UI components for displaying results.

These are the three main blocks of the results section:
  1. render_verdict_box   – the big colored banner at the top
  2. render_source_cards  – one card per evidence passage
  3. render_detail_table  – a table with all the raw numbers

All of them use unsafe_allow_html because Streamlit doesn't support custom-styled divs. The CSS classes are defined in styles.py.
"""
import streamlit as st


def render_verdict_box(verdict: str, justification: str) -> None:
    """
    Show the final verdict as a colored box.
    Green for SUPPORTED, red for REFUTED, yellow for MISLEADING,
    grey for anything else (shouldn't happen but just in case).
    """
    css_class = {
        "SUPPORTED":  "verdict-supported",
        "REFUTED":    "verdict-refuted",
        "MISLEADING": "verdict-misleading",
    }.get(verdict.upper(), "verdict-unknown")

    st.markdown(
        f"""
        <div class="{css_class}">
            <strong>Verdict: {verdict}</strong><br>{justification}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_cards(evidence: list[dict], nli: list[dict]) -> None:
    """
    Render one card per evidence passage.

    Each card shows:
    - Source tag + index number (f.ex. "[1] CDC")
    - NLI label with its confidence score (e.g. "CONTRADICTION 94.12%")
    - The retrieval similarity score
    - The passage text
    - A clickable link to the original source

    The NLI label gets a colored CSS class (green/red/grey) so you
    can immediately see if a passage supports or contradicts the claim.
    """
    for i, ev in enumerate(evidence):
        nli_entry = nli[i] if i < len(nli) else None
        nli_label = nli_entry["label"] if nli_entry else None
        nli_css = f"nli-{nli_label}" if nli_label else ""
        nli_score = (
            f"{nli_entry['scores'][nli_label]:.2%}" if nli_entry else ""
        )

        st.markdown(
            f"""
            <div class="source-card">
                <strong>[{i+1}] {ev['source']}</strong>
                {"&nbsp;·&nbsp;<span class='" + nli_css + "'>" + nli_label.upper() + " " + nli_score + "</span>" if nli_label else ""}
                &nbsp;·&nbsp; score: {ev['score']:.4f}<br>
                {ev['text']}<br>
                <a href="{ev['url']}" target="_blank">{ev['url']}</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detail_table(evidence: list[dict], nli: list[dict]) -> None:
    """
    Show all evidence + NLI scores in a single table.

    This is mainly for debugging and for the thesis evaluation –
    it shows the raw numbers for each passage so you can verify
    that the NLI model and the retriever are working correctly.

    The text column is truncated to 120 chars so the table doesn't
    get ridiculously wide.
    """
    rows = []
    for i, ev in enumerate(evidence):
        nli_entry = nli[i] if i < len(nli) else {}
        scores = nli_entry.get("scores", {})

        rows.append(
            {
                "#": i + 1,
                "Source": ev["source"],
                "Retrieval score": round(ev["score"], 4),
                "NLI label": nli_entry.get("label", "—"),
                "Entailment": round(scores.get("entailment", 0), 4) if scores else "—",
                "Contradiction": round(scores.get("contradiction", 0), 4) if scores else "—",
                "Neutral": round(scores.get("neutral", 0), 4) if scores else "—",
                "Text": ev["text"][:120] + ("…" if len(ev["text"]) > 120 else ""),
            }
        )

    st.dataframe(rows, use_container_width=True)

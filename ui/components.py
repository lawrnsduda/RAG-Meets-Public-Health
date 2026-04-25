"""Reusable UI components for displaying results."""
import streamlit as st


def render_verdict_box(verdict: str, justification: str) -> None:
    css_class = {
        "SUPPORTED":           "verdict-supported",
        "REFUTED":             "verdict-refuted",
        "MISLEADING":          "verdict-misleading",
        "NOT_ENOUGH_EVIDENCE": "verdict-not-enough-evidence",
    }.get(verdict.upper(), "verdict-unknown")

    display_verdict = verdict.replace("_", " ")

    st.markdown(
        f"""
        <div class="{css_class}">
            <strong>Verdict: {display_verdict}</strong><br>{justification}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_cards(evidence: list[dict], nli: list[dict]) -> None:
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
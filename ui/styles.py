"""
Custom CSS for the Health Factchecker UI.

Injected via st.markdown(unsafe_allow_html=True) because Streamlit
doesn't expose easy per-element styling. Call once after set_page_config.
"""
import streamlit as st


def inject_css():
    st.markdown("""
    <style>

    .verdict-supported {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 1rem; border-radius: 6px; margin-bottom: 1rem;
    }
    .verdict-refuted {
        background: #f8d7da; border-left: 5px solid #dc3545;
        padding: 1rem; border-radius: 6px; margin-bottom: 1rem;
    }
    .verdict-misleading {
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 1rem; border-radius: 6px; margin-bottom: 1rem;
    }
    .verdict-unknown {
        background: #e2e3e5; border-left: 5px solid #6c757d;
        padding: 1rem; border-radius: 6px; margin-bottom: 1rem;
    }
    /* Abstain verdict (NOT_ENOUGH_EVIDENCE): neutral grey so users
       don't read it as red = bad or green = good. */
    .verdict-not-enough-evidence {
        background: #e9ecef; border-left: 5px solid #495057;
        padding: 1rem; border-radius: 6px; margin-bottom: 1rem;
    }

    .source-card {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;
    }

    .nli-entailment    { color: #28a745; font-weight: bold; }
    .nli-contradiction { color: #dc3545; font-weight: bold; }
    .nli-neutral       { color: #6c757d; font-weight: bold; }

    </style>
    """, unsafe_allow_html=True)

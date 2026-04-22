"""
Custom CSS for the Health Factchecker UI.

Streamlit doesn't let me style individual elements easily, so I
inject raw CSS via st.markdown(unsafe_allow_html=True). This function
should be called once at the top of the app, right after set_page_config.

The classes defined here are used by the components in ui/components.py.
"""
import streamlit as st


def inject_css():
    """Inject all custom styles into the page. Call this once on app startup."""
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

    /* Source cards */

    .source-card {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;
    }

    .nli-entailment    { color: #28a745; font-weight: bold; }
    .nli-contradiction { color: #dc3545; font-weight: bold; }
    .nli-neutral       { color: #6c757d; font-weight: bold; }

    </style>
    """, unsafe_allow_html=True)

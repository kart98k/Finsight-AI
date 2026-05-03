import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def _get(key: str) -> str:
    """Read from st.secrets first, fall back to os.environ."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, "")

FMP_API_KEY       = _get("FMP_API_KEY")
ANTHROPIC_API_KEY = _get("ANTHROPIC_API_KEY")
NEWS_API_KEY      = _get("NEWS_API_KEY")
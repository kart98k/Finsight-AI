import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str) -> str:
    """Read from st.secrets first, fall back to os.environ."""
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, "")


def get_fmp_key() -> str:
    return _get("FMP_API_KEY")


def get_anthropic_key() -> str:
    return _get("ANTHROPIC_API_KEY")


def get_news_key() -> str:
    return _get("NEWS_API_KEY")


# Keep these for backward compatibility but they now call functions
FMP_API_KEY       = ""
ANTHROPIC_API_KEY = ""
NEWS_API_KEY      = ""
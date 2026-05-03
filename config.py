import os
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY       = os.getenv("FMP_API_KEY",       "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
NEWS_API_KEY      = os.getenv("NEWS_API_KEY",       "")
# 📈 FinSight AI

> An intelligent financial analysis dashboard powered by NLP, LangGraph, and Claude Haiku.

**Built by Srikonda Karthik**
MBA - Business Analytics | Woxsen University
Subject: NLP for Customer & Market Insights

---

## 🌐 Live Demo

[https://finsight-ai.streamlit.app](https://finsight-ai-zdeoxfvmkwg8stefkyfnsw.streamlit.app/)

---

## 🧠 What is FinSight AI?

FinSight AI is a multi-layer NLP-powered financial intelligence dashboard that combines real-time company financials with advanced natural language processing to deliver actionable market insights. It goes beyond numbers — analysing what the market *says* about a company, not just what it *reports*.

---

## 🏗️ System Architecture

```
User Input (Ticker)
        ↓
LangGraph Agent Pipeline
   ├── fetch_data_node       → Financial Modeling Prep API
   ├── analyst_node          → KPI extraction & calculation
   ├── sentiment_runner      → VADER news sentiment analysis
   ├── transcript_runner     → Earnings call NLP summarization
   ├── insights_runner       → Claude Haiku streaming insights
   ├── chart_builder         → Plotly chart data preparation
   └── aggregator            → Final output bundling
        ↓
Streamlit Dashboard
   ├── Company Header (Logo + Live Stock Price)
   ├── KPI Cards (8 metrics)
   ├── AI Insights (streamed)
   ├── News Sentiment Analysis
   │   ├── Sentiment Distribution (Donut + Timeline)
   │   ├── Most Positive / Most Negative Headline
   │   ├── Headlines with Confidence Scores
   │   ├── Keyword Frequency Chart
   │   ├── Word Cloud
   │   ├── NLP Pipeline Step-by-Step Expander
   │   └── VADER Sentiment Model Analysis
   ├── Earnings Call Transcript Analysis
   │   ├── Management Tone
   │   ├── Call Summary
   │   ├── Key Themes
   │   ├── Risks Mentioned
   │   └── Forward Guidance
   └── Financial Charts (5 chart types)
```

---

## 🔬 NLP Pipeline

FinSight AI demonstrates a three-layer NLP architecture:

### Layer 1 - Classical NLP
- **Tokenization** - splitting headlines into individual tokens
- **Stopword Removal** - filtering noise words using a domain-aware stopword list
- **POS Tagging** - part-of-speech classification of financial terms
- **Term Frequency Analysis** - keyword extraction from news headlines
- **Word Cloud** - visual keyword density from recent news

### Layer 2 - Rule-based Sentiment Model
- **VADER** - Valence Aware Dictionary and sEntiment Reasoner, a lexicon-based model designed for social media and financial news
- **Confidence Scoring** - probabilistic output per headline
- **Temporal Analysis** - sentiment distribution over time

> **Architectural Note:** The sentiment module is designed to be model-agnostic. In a production environment with dedicated GPU resources, VADER would be replaced with **FinBERT** (ProsusAI/finbert), a BERT transformer fine-tuned on financial corpora that demonstrates superior performance on domain-specific terminology such as *bearish*, *headwinds*, and *earnings miss*. The separation of the sentiment service ensures this upgrade requires only a single file change.

### Layer 3 - Large Language Models
- **Claude Haiku** - abstractive summarization of earnings call transcripts
- **Structured IE** - extraction of management tone, themes, risks, and forward guidance
- **AI Insights** - streaming financial commentary from KPI data
- **Investment Recommendation** - AI-generated comparative analysis (compare mode)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏢 Company Header | Logo, live stock price, exchange, sector |
| 📊 KPI Dashboard | 8 key financial metrics with YoY comparison |
| 🤖 AI Insights | Streaming Claude Haiku financial commentary |
| 📰 News Sentiment | VADER-powered analysis of recent headlines |
| 🟢🔴 Headline Highlights | Most positive and most negative headline with confidence |
| 🔑 Keyword Frequency | Top 15 keywords extracted via NLP pipeline |
| ☁️ Word Cloud | Visual keyword density from news headlines |
| 🔬 NLP Pipeline | Step-by-step tokenization walkthrough |
| 📊 VADER Analysis | Sentiment scores and analysis table |
| 🎙️ Earnings Call | NLP summarization with tone, themes, risks, guidance |
| 📋 KPI Comparison | Side-by-side dual ticker analysis with winner column |
| 🤖 Investment Rec | AI-generated buy recommendation with reasoning |
| 📈 5 Chart Types | Revenue, Margins, Cash Flow, Debt/Equity, Gauge |
| ⏳ Progress Bar | Step-by-step pipeline progress indicator |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph (StateGraph) |
| LLM | Claude Haiku (claude-haiku-4-5) |
| Sentiment Model | VADER (vaderSentiment) |
| Financial Data | Financial Modeling Prep API (stable) |
| News Data | NewsAPI.org |
| Frontend | Streamlit |
| Charts | Plotly Express |
| Word Cloud | WordCloud + Matplotlib |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub |
| IDE | Cursor |

---

## 📁 Project Structure

```
finsight-ai/
│
├── agent/
│   ├── __init__.py
│   ├── state.py              # LangGraph GraphState TypedDict
│   ├── nodes.py              # 7 pipeline node functions
│   └── graph.py              # StateGraph wiring and compilation
│
├── services/
│   ├── __init__.py
│   ├── fmp_client.py         # Financial Modeling Prep API calls
│   ├── claude_client.py      # Anthropic SDK — insights + recommendation
│   ├── sentiment_service.py  # VADER sentiment analysis
│   ├── nlp_service.py        # Keywords, pipeline, wordcloud
│   └── transcript_service.py # Earnings call NLP summarization
│
├── ui/
│   ├── __init__.py
│   ├── dashboard.py          # Streamlit main app
│   └── charts.py             # Plotly chart builders
│
├── .streamlit/
│   └── config.toml           # Streamlit server configuration
│
├── .env                      # API keys (gitignored)
├── .env.example              # Key template
├── .gitignore
├── config.py                 # Environment variable loader
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Docker Desktop (for local development)
- API keys for Anthropic, FMP, and NewsAPI

### Run locally with Docker

```bash
# 1. Clone the repository
git clone https://github.com/kart98k/finsight-ai.git
cd finsight-ai

# 2. Create your .env file
cp .env.example .env
# Fill in your FMP_API_KEY, ANTHROPIC_API_KEY, NEWS_API_KEY

# 3. Run with Docker Compose
docker compose up --build

# 4. Open in browser
# http://localhost:8501
```

### Run locally without Docker

```bash
pip install -r requirements.txt
streamlit run ui/dashboard.py
```

---

## 🔑 API Keys Required

| Key | Where to get | Cost |
|---|---|---|
| Anthropic API Key | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use (~$0.001/analysis) |
| FMP API Key | [financialmodelingprep.com](https://financialmodelingprep.com) | Free tier |
| NewsAPI Key | [newsapi.org](https://newsapi.org) | Free (100 req/day) |

> **Privacy:** The Anthropic API key is entered by the user in the sidebar and is never stored or logged. FMP and NewsAPI keys are server-side environment variables.

---

## 📊 Supported Tickers (Free Tier)

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, META, NVDA |
| Finance | JPM, BAC, C, WFC, MS |
| Healthcare | JNJ, PFE, MRK, ABBV, UNH |
| Consumer | AMZN, TSLA, WMT, COST, NKE |
| Energy | XOM, CVX |

Custom tickers can also be entered manually in the sidebar.

---

## 🎓 Academic Context

This project was developed for the subject **NLP for Customer & Market Insights** as part of the MBA — Business Analytics programme at Woxsen University.

### NLP Techniques Demonstrated

| # | Technique | Implementation |
|---|---|---|
| 1 | Tokenization | Pure Python tokenizer on news headlines |
| 2 | Stopword Removal | Domain-aware financial stopword list |
| 3 | POS Tagging | Pattern-based part-of-speech classification |
| 4 | Term Frequency | Keyword extraction and ranking |
| 5 | Lexicon Sentiment | VADER rule-based scoring |
| 6 | Confidence Scoring | Probabilistic output per headline |
| 7 | Temporal Analysis | Sentiment distribution over time |
| 8 | Abstractive Summarization | LLM-based earnings call summarization |
| 9 | Information Extraction | Structured IE of tone, themes, risks, guidance |
| 10 | Text Classification | Management tone detection |

### Deployment Note on Model Selection

The production deployment uses VADER for sentiment analysis due to memory constraints on the Streamlit Cloud free tier (1GB RAM). FinBERT (ProsusAI/finbert) is architecturally integrated in the codebase and can be activated locally by installing `torch` and `transformers`. This design decision demonstrates awareness of the tradeoff between model accuracy and deployment constraints - a real-world consideration in production NLP systems.

---

## 📸 Screenshots

> <img width="1919" height="908" alt="Screenshot 2026-05-03 134542" src="https://github.com/user-attachments/assets/f89e7987-33e3-4376-b1ba-d311275bd51f" />


---

## 👤 Author

**Srikonda Karthik**
MBA - Business Analytics
Woxsen University
GitHub: [@kart98k](https://github.com/kart98k)
Email: kartiksrikonda27@gmail.com

---

## 📄 Disclaimer

This application is built for educational purposes only. The AI-generated insights and investment recommendations are not financial advice. Always consult a qualified financial advisor before making investment decisions.

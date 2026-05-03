# 📈 FinSight AI

> An intelligent financial analysis dashboard powered by NLP, LangGraph, and Claude Haiku.

**Built by Srikonda Karthik**
MBA - Business Analytics | Woxsen University
Subject: NLP for Customer & Market Insights

---

## 🧠 What is FinSight AI?

FinSight AI is a multi-layer NLP-powered financial intelligence dashboard that combines real-time company financials with advanced natural language processing to deliver actionable market insights. It goes beyond numbers analysing what the market *says* about a company, not just what it *reports*.

---

## 🏗️ System Architecture

```
User Input (Ticker)
        ↓
LangGraph Agent Pipeline
   ├── fetch_data_node       → Financial Modeling Prep API
   ├── analyst_node          → KPI extraction & calculation
   ├── sentiment_runner      → FinBERT + VADER news sentiment
   ├── transcript_runner     → Earnings call NLP summarization
   ├── insights_runner       → Claude Haiku streaming insights
   ├── chart_builder         → Plotly chart data preparation
   └── aggregator            → Final output bundling
        ↓
Streamlit Dashboard
   ├── KPI Cards
   ├── AI Insights (streamed)
   ├── News Sentiment Analysis
   ├── Keyword Frequency + Word Cloud
   ├── NLP Pipeline Expander
   ├── VADER vs FinBERT Comparison
   ├── Earnings Call Transcript Analysis
   └── Financial Charts (7 chart types)
```

---

## 🔬 NLP Pipeline

FinSight AI demonstrates a three-layer NLP architecture:

### Layer 1 — Classical NLP
- **Tokenization** - splitting headlines into individual tokens
- **Stopword Removal** - filtering noise words using a domain-aware stopword list
- **POS Tagging** - part-of-speech classification of financial terms
- **Term Frequency Analysis** - keyword extraction from news headlines

### Layer 2 - Pre-trained Transformer Models
- **VADER** - rule-based lexicon sentiment scoring
- **FinBERT** - BERT model fine-tuned on financial text (ProsusAI/finbert)
- **Model Comparison** - side-by-side VADER vs FinBERT agreement analysis
- **Confidence Scoring** - probabilistic output per headline

### Layer 3 — Large Language Models
- **Claude Haiku** - abstractive summarization of earnings call transcripts
- **Structured IE** - extraction of tone, themes, risks, and forward guidance
- **AI Insights** - streaming financial commentary from KPI data
- **Investment Recommendation** - AI-generated comparative analysis

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 KPI Dashboard | 8 key financial metrics with YoY comparison |
| 🤖 AI Insights | Streaming Claude Haiku financial commentary |
| 📰 News Sentiment | FinBERT-powered analysis of 30 recent headlines |
| 🔑 Keyword Frequency | Top 15 keywords extracted via NLP pipeline |
| ☁️ Word Cloud | Visual keyword density from news headlines |
| 🔬 NLP Pipeline | Step-by-step tokenization walkthrough |
| 📊 Model Comparison | VADER vs FinBERT agreement/disagreement table |
| 🎙️ Earnings Call | NLP summarization of management transcripts |
| 📋 KPI Comparison | Side-by-side dual ticker analysis with winner column |
| 🤖 Investment Rec | AI-generated buy recommendation with reasoning |
| 📈 5 Chart Types | Revenue, Margins, Cash Flow, Debt/Equity, Gauge |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph (StateGraph) |
| LLM | Claude Haiku (claude-haiku-4-5) |
| Sentiment — Domain | FinBERT (ProsusAI/finbert) |
| Sentiment — Rule-based | VADER (vaderSentiment) |
| Financial Data | Financial Modeling Prep API |
| News Data | NewsAPI.org |
| Frontend | Streamlit |
| Charts | Plotly Express |
| Word Cloud | WordCloud + Matplotlib |
| Container | Docker + Docker Compose |
| Deployment | Render |
| IDE | Cursor |

---

## 📁 Project Structure

```
finsight-ai/
│
├── agent/
│   ├── __init__.py
│   ├── state.py          # LangGraph GraphState TypedDict
│   ├── nodes.py          # 7 pipeline node functions
│   └── graph.py          # StateGraph wiring and compilation
│
├── services/
│   ├── __init__.py
│   ├── fmp_client.py         # Financial Modeling Prep API calls
│   ├── claude_client.py      # Anthropic SDK - insights + recommendation
│   ├── sentiment_service.py  # FinBERT + VADER sentiment analysis
│   ├── nlp_service.py        # Keywords, pipeline, model comparison, wordcloud
│   └── transcript_service.py # Earnings call NLP summarization
│
├── ui/
│   ├── __init__.py
│   ├── dashboard.py      # Streamlit main app
│   └── charts.py         # Plotly chart builders
│
├── .env                  # API keys (gitignored)
├── .env.example          # Key template
├── .gitignore
├── .dockerignore
├── config.py             # Environment variable loader
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

---

## 🚀 Getting Started

### Prerequisites
- Docker Desktop installed
- API keys for Anthropic and NewsAPI

### 1. Clone the repository
```bash
git clone https://github.com/kart98k/finsight-ai.git
cd finsight-ai
```

### 2. Create your `.env` file
```bash
cp .env.example .env
```

Fill in your keys:
```
FMP_API_KEY=your_fmp_key
ANTHROPIC_API_KEY=your_anthropic_key
NEWS_API_KEY=your_newsapi_key
```

### 3. Run with Docker
```bash
docker compose up --build
```

### 4. Open in browser
```
http://localhost:8501
```

### 5. Enter your Anthropic API key in the sidebar and select a ticker

---

## 🔑 API Keys Required

| Key | Where to get | Cost |
|---|---|---|
| Anthropic API Key | [console.anthropic.com](https://console.anthropic.com) | Pay-per-use (~$0.001/analysis) |
| FMP API Key | [financialmodelingprep.com](https://financialmodelingprep.com) | Free tier |
| NewsAPI Key | [newsapi.org](https://newsapi.org) | Free (100 req/day) |

> **Privacy:** The Anthropic API key is entered by the user in the sidebar and is never stored or logged. FMP and NewsAPI keys are server-side only.

---

## 📊 Supported Tickers (Free Tier)

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, META, NVDA |
| Finance | JPM, BAC, C, WFC, MS |
| Healthcare | JNJ, PFE, MRK, ABBV, UNH |
| Consumer | AMZN, TSLA, WMT, COST, NKE |
| Energy | XOM, CVX|

Custom tickers can also be entered manually in the sidebar.

---

## 🎓 Academic Context

This project was developed for the subject **NLP for Customer & Market Insights** as part of the MBA - Business Analytics programme at Woxsen University.

### NLP Techniques Demonstrated

1. **Tokenization** - splitting financial text into tokens
2. **Stopword Removal** - domain-aware noise filtering
3. **POS Tagging** - morphosyntactic classification
4. **Term Frequency** - keyword extraction and ranking
5. **Lexicon-based Sentiment** - VADER rule-based scoring
6. **Transformer Sentiment** - FinBERT domain-specific classification
7. **Comparative Evaluation** - VADER vs FinBERT model comparison
8. **Abstractive Summarization** - LLM-based earnings call summarization
9. **Information Extraction** - structured IE of tone, themes, risks, guidance
10. **Text Classification** - management tone detection

---

## 📸 Screenshots

> Add screenshots of your dashboard here after deployment

---

## 🌐 Live Demo

> Add your Render deployment URL here after deployment

---

## 👤 Author

**Srikonda Karthik**
MBA — Business Analytics
Woxsen University
GitHub: [@kart98k](https://github.com/kart98k)
Email: kartiksrikonda27@gmail.com

---

## 📄 Disclaimer

This application is built for educational purposes only. The AI-generated insights and investment recommendations are not financial advice. Always consult a qualified financial advisor before making investment decisions.
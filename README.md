# AI BI Dashboard Generator

A professional AI-powered analytics platform that automatically generates interactive, Power BI–style dashboards from uploaded datasets. Built with **Streamlit**, **Plotly**, and **Groq/OpenAI**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **📊 Auto Dashboard** | KPI cards, bar/line/scatter/pie charts, correlation heatmap generated automatically |
| **🎛️ Interactive Filters** | Category multi-select, date range, numeric sliders — all charts update dynamically |
| **💬 Ask AI** | Natural language questions → data queries → visualizations (Groq or OpenAI) |
| **💡 Smart Insights** | Automated detection of trends, outliers, correlations, top categories |
| **🔍 Data Explorer** | Paginated table with search, sort, and column statistics |
| **📥 Export** | Download as PDF report, CSV summary, or filtered dataset |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd ai-bi-dashboard
pip install -r requirements.txt
```

### 2. Configure AI (optional)

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Set at least one:
```
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

> **Note:** The dashboard works without an AI key (rule-based analysis), but Ask AI and AI Insights are enhanced with a key.

### 3. Run the app

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 📁 Project Structure

```
ai-bi-dashboard/
├── app.py                     # Main Streamlit application
├── core/
│   ├── data_loader.py         # CSV/Excel loading + validation
│   ├── data_profiler.py       # Automatic data profiling
│   ├── chart_generator.py     # Plotly chart auto-generation
│   ├── nl_query_engine.py     # Natural language → data query
│   └── insight_engine.py      # Statistical insight detection
├── dashboard/
│   ├── dashboard_builder.py   # KPI + chart grid layout
│   └── filters.py             # Sidebar filter system
├── utils/
│   ├── helpers.py             # Formatting, CSS, palettes
│   └── validators.py          # File/data validation
├── data/
│   └── sample_sales.csv       # Sample dataset (500 rows)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📊 Supported File Formats

- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)
- Max file size: **200 MB**
- Max rows: **1,000,000**

---

## 🧠 Tech Stack

- **Python 3.9+**
- **Streamlit** — web UI framework
- **Plotly / Plotly Express** — interactive visualizations
- **Pandas / NumPy / SciPy** — data processing
- **Groq (Llama 3.3)** or **OpenAI (GPT-4o-mini)** — AI layer
- **fpdf2** — PDF report generation

---

## 📝 License

MIT

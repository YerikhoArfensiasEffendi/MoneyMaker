# MoneyMaker

**Master Corporate Financial Analysis & Multi-Agent Stock Market System** (IDX/BEI & Global/US).

## Arsitektur Target
- **Master Financial Analysis Agent (Chief Investment Officer / Lead Analyst)**: Mengorkestrasi para agen spesialis, melakukan sintesis laporan keuangan perusahaan, valuasi intrinsik (DCF, multiples, quality/moat), dan menyusun tesis investasi final.
- **Specialist Agents (LangGraph Multi-Agent Workflow)**:
  - **Fundamental & Corporate Finance Agent**: Analisis laporan laba rugi, neraca, arus kas, rasio profitabilitas/likuiditas/solvabilitas, dan kualitas laba.
  - **Technical & Quantitative Agent**: Indikator teknikal (`ta`), regresi faktor/beta (`statsmodels`), volatilitas & risiko (`scipy`, `scikit-learn`).
  - **Financial Filing Parser**: Ekstraksi tabel & teks laporan keuangan tahunan/kuartalan PDF & Excel (`pdfplumber`, `pypdf`, `openpyxl`).

## Stack Utama
- **Python 3.12+** & **uv**
- **Multi-Agent & LLM**: `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-google-genai`, `google-genai`
- **Market Data & Quant**: `yfinance`, `pandas`, `numpy`, `scipy`, `ta`, `statsmodels`, `scikit-learn`
- **Document Parsing**: `pdfplumber`, `pypdf`, `openpyxl`

## Quick Start

```bash
# 1. Install seluruh dependensi secara deterministik
uv sync --frozen

# 2. Siapkan API key Gemini (jangan commit .env)
cp .env.example .env

# 3. Jalankan smoke test & linter
uv run pytest
uv run ruff check .
```

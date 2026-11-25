"""Environment smoke tests for MoneyMaker financial & multi-agent stack."""

import io
import sqlite3
from typing import TypedDict

import numpy as np
import openpyxl
import pandas as pd
import pdfplumber
import pypdf
import scipy.stats as stats
import statsmodels.api as sm
import ta
import yfinance as yf
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from sklearn.linear_model import Ridge


class _SmokeState(TypedDict):
    ticker: str
    fundamental_score: float
    technical_score: float
    verdict: str


def test_langgraph_multi_agent_orchestration() -> None:
    """Verify LangGraph stateful multi-agent orchestration with SQLite checkpointing."""

    def fundamental_agent(state: _SmokeState) -> dict[str, float]:
        return {"fundamental_score": 82.5}

    def technical_agent(state: _SmokeState) -> dict[str, float]:
        return {"technical_score": 74.0}

    def master_agent(state: _SmokeState) -> dict[str, str]:
        score = 0.65 * state["fundamental_score"] + 0.35 * state["technical_score"]
        return {"verdict": f"BUY ({score:.2f}/100)"}

    builder = StateGraph(_SmokeState)
    builder.add_node("fundamental", fundamental_agent)
    builder.add_node("technical", technical_agent)
    builder.add_node("master", master_agent)
    builder.add_edge(START, "fundamental")
    builder.add_edge("fundamental", "technical")
    builder.add_edge("technical", "master")
    builder.add_edge("master", END)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    try:
        graph = builder.compile(checkpointer=SqliteSaver(conn))
        result = graph.invoke(
            {
                "ticker": "BBCA.JK",
                "fundamental_score": 0.0,
                "technical_score": 0.0,
                "verdict": "",
            },
            config={"configurable": {"thread_id": "env-smoke"}},
        )
    finally:
        conn.close()

    assert result["verdict"] == "BUY (79.53/100)"


def test_financial_and_quantitative_stack() -> None:
    """Verify technical indicators, CAPM regression, and statistical libraries."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    close = pd.Series(9000 + np.cumsum(rng.normal(10, 80, 100)), index=dates)

    rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd = ta.trend.MACD(close=close).macd_diff()
    assert not pd.isna(rsi.iloc[-1])
    assert not pd.isna(macd.iloc[-1])

    mkt_ret = rng.normal(0.0005, 0.01, 100)
    stk_ret = 0.0002 + 1.15 * mkt_ret + rng.normal(0, 0.005, 100)
    ols = sm.OLS(stk_ret, sm.add_constant(mkt_ret)).fit()
    ridge = Ridge().fit(mkt_ret.reshape(-1, 1), stk_ret)

    assert np.isfinite(ols.params[1])
    assert np.isfinite(ridge.coef_[0])
    assert np.isfinite(stats.skew(stk_ret))


def test_financial_document_and_sdk_imports() -> None:
    """Verify PDF/Excel report parsers, yfinance, and Gemini SDK readiness."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "IncomeStatement"
    ws.append(["Metric", "FY2024", "FY2025"])
    ws.append(["Revenue", 100_000, 118_000])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    loaded = openpyxl.load_workbook(buf)
    assert loaded["IncomeStatement"]["C2"].value == 118_000

    assert hasattr(pdfplumber, "open")
    assert hasattr(pypdf, "PdfReader")
    assert hasattr(yf, "Ticker")
    assert hasattr(genai, "Client")
    assert ChatGoogleGenerativeAI is not None

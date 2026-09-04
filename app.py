import os
try:
    import FinanceDataReader as fdr
except ImportError:
    os.system("pip install finance-datareader lxml > /dev/null 2>&1")
    import FinanceDataReader as fdr

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import urllib.parse
import requests
import xml.etree.ElementTree as ET

st.set_page_config(layout="wide", page_title="투자 도우미 프로그램")

# --- [UI 디자인 강제 수정] ---
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.2rem !important; white-space: nowrap !important; }
[data-testid="stMetricLabel"] { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 투자 도우미 프로그램")
st.warning("⚠️ **[투자 유의사항]** 본 프로그램이 제공하는 AI 예측, 차트, 실시간 뉴스 감성 분석 등의 모든 정보는 과거 데이터를 기반으로 한 **참고용 보조 자료**입니다. 미래의 수익을 절대 보장하지 않으며, **모든 투자의 최종 판단과 그에 따른 책임은 전적으로 투자자 본인에게 있습니다.**")

# --- [1. 공통 데이터 엔진] ---
@st.cache_data(ttl=3600)
def load_krx_data():
    return fdr.StockListing('KRX')

@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        krx_df = load_krx_data()
        krx_list = [f"{row['Name']} ({row['Code']})" for _, row in krx_df.iterrows()]
    except:
        krx_list = []
        
    global_list = [
        "애플 (AAPL)", "테슬라 (TSLA)", "엔비디아 (NVDA)", "마이크로소프트 (MSFT)",
        "구글 (GOOGL)", "아마존 (AMZN)", "메타 (META)"
    ]
    return global_list + krx_list

@st.cache_resource
def load_korean_ai(): 
    return pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# ★ 잘림 방지: 수급 데이터 가져오는 부분 코드를 짧게 쪼개서 정리했습니다. ★
@st.cache_data(ttl=3600)
def get_investor_data(stock_code):
    try:
        code_only = stock_code.split('.')[0]
        url = f"https://finance.naver.com/item/frgn.naver?code={code_only}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
        res = requests.get(url, headers=headers, timeout=5)
        dfs = pd.read_html(res.text, encoding='euc-kr')
        
        date_pattern = r"^\d{4}\.\d{2}\.\d{2}$" # 날짜 형식 패턴을 따로 빼서 짧게 만듦
        
        for df in dfs:
            if len(df.columns) >= 7:
                date_col = df.iloc[:, 0].astype(str)
                if date_col.str.match(date_pattern, na=False).any():
                    valid_df = df[date_col.str.match(date_pattern, na=False)]
                    res_df = valid_df.iloc[:, [0, 1, 5, 6]].copy()
                    res_df.columns = ['날짜', '종가', '기관순매수', '외국인순매수']
                    return res_df.head(10)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- [2. 핵심 분석 대시보드 로직] ---
def run_dashboard(ticker_code, company_display_name):
    stock = yf.Ticker(ticker_code)
    df = stock.history(period="2y")
    
    if df.empty or len(df) < 30:
        st.warning("데이터를 불러오지 못했습니다. 종목명이나 코드가 정확한지 확인해주세요.")
        return
        
    info = stock.info
    is_korean = ticker_code.endswith('.KS') or ticker_code.endswith('.KQ')
    currency = "₩" if is_korean else "$"
    current_price = df['Close'].iloc[-1]
    
    # 보조 지표 계산
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
    
    delta = df['Close'].diff()
    rs = (delta.where(delta > 0, 0)).rolling(14).mean() / ((delta.where(delta < 0, 0)).rolling(14).mean().abs() + 1e-9)
    df['RSI'] = 100 - (

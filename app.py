import os
try:
    import FinanceDataReader as fdr
except ImportError:
    os.system("pip install finance-datareader > /dev/null 2>&1")
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

def resolve_ticker(user_input):
    query = user_input.strip()
    if query.endswith('.KS') or query.endswith('.KQ') or (query.isupper() and query.isalpha()):
        return query, query
    clean_query = query.replace(" ", "").upper()
    
    name_map = {
        "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "LG에너지솔루션": "373220.KS",
        "현대차": "005380.KS", "현대자동차": "005380.KS", "기아": "000270.KS", 
        "NAVER": "035420.KS", "네이버": "035420.KS", "카카오": "035720.KS",
        "한화에어로스페이스": "012450.KS", "펩트론": "087010.KQ", "에코프로": "086520.KQ", 
        "애플": "AAPL", "테슬라": "TSLA", "엔비디아": "NVDA", "마이크로소프트": "MSFT"
    }
    if clean_query in name_map:
        return name_map[clean_query], query.upper()

    try:
        krx_df = load_krx_data()
        krx_names_clean = krx_df['Name'].str.replace(" ", "").str.upper()
        match = krx_df[krx_names_clean == clean_query]
        if not match.empty:
            code, market, name = match.iloc[0]['Code'], match.iloc[0]['Market'], match.iloc[0]['Name']
            suffix = '.KQ' if 'KOSDAQ' in str(market).upper() else '.KS'
            return f"{code}{suffix}", name
    except: pass
        
    try:
        q_enc = urllib.parse.quote(query)
        res = requests.get(f"https://ac.finance.naver.com/ac?q={q_enc}&q_enc=utf-8&st=111&r_format=json&r_enc=utf-8", timeout=3).json()
        if res.get('items') and len(res['items'][0]) > 0:
            code, name, market = res['items'][0][0][0], res['items'][0][0][1], res['items'][0][0][2]
            return f"{code}{'.KQ' if '코스닥' in market else '.KS'}", name
    except: pass

    return query, query

# --- [2. 메인 화면 - 개별 종목 분석] ---
user_input = st.text_input("회사명 또는 종목코드를 입력하세요 (예: SK하이닉스, 삼성전자, 테슬라)", "한화에어로스페이스")

if user_input:
    ticker_code, company_display_name = resolve_ticker(user_input)
    stock = yf.Ticker(ticker_code)
    df = stock.history(period="2y")
    
    if not df.empty and len(df) > 30:
        info = stock.info
        is_korean = ticker_code.endswith('.KS') or ticker_code.endswith('.KQ')
        currency = "₩" if is_korean else "$"
        current_price = df['Close'].iloc[-1]
        
        # 보조 지표
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        df['STD20'] = df['Close'].rolling(20).std()
        df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
        
        delta = df['Close'].diff()
        rs = (delta.where(delta > 0, 0)).rolling(14).mean() / ((delta.where(delta < 0, 0)).rolling(14).mean().abs() + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        ema12, ema26 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # ML 학습
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        ml_df = df.dropna().copy()
        if len(ml_df) > 50:
            X, y = ml_df[['MA20', 'MA60', 'RSI', 'MACD', 'Price_Change', 'Volume_Change']], ml_df['Target']
            split_idx = int(len(ml_df) * 0.8)
            test_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X.iloc[:split_idx], y.iloc[:split_idx])
            test_acc = accuracy_score(y.iloc[split_idx:], test_model.predict(X.iloc[split_idx:])) * 100
            
            final_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
            up_prob = final_model.predict_proba(X.iloc[-1:])[0][1] * 100
        else:
            up_prob, test_acc = 50.0, 0.0

        # --- [데이터 강제 추출 (누락 방지)] ---
        price_fmt = f"{currency}{current_price:,.0f}" if is_korean else f"{currency}{current_price:,.2f}"
        
        krx_df = load_krx_data()
        mkt_cap_str = "N/A"
        if is_korean:
            code_only = ticker_code.split('.')[0]
            match = krx_df[krx_df['Code'] == code_only]
            if not match.empty:
                mkt_cap = match.iloc[0]['Marcap']
                mkt_cap_str = f"{mkt_cap / 1_000_000_000_000:.2f}조 원"
        else:
            mkt_cap = info.get('marketCap', 0)
            if mkt_cap: mkt_cap_str = f"${mkt_cap / 1_000_000_000:.2f}B"

        last_252_days = df.tail(252)
        high52_val = last_252_days['High'].max()
        low52_val = last_252_days['Low'].min()
        high52 = f"{currency}{int(high52_val):,}" if is_korean else f"{currency}{high52_val:.2f}"
        low52 = f"{currency}{int(low52_val):,}" if is_korean else f"{currency}{low52_val:.2f}"
        
        latest_rsi = df['RSI'].iloc[-1]

        # ★ 상단 요약 바 (배당수익률 완전 삭제, 5칸으로 구성) ★
        st.success(f"🔍 **{company_display_name}** (`{ticker_code}`) 개별 분석")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("현재 주가", price_fmt)
        c2.metric("시가총액", mkt_cap_str)
        c3.metric("52주 최고", high52)
        c4.metric("52주 최저", low52)
        c5.metric("RSI (과열도)", f"{latest_rsi:.1f}", "과매수 ⚠️" if latest_rsi >= 70 else "과매도 📉" if latest_rsi <= 30 else "중립")
        st.divider()

        # --- [탭 레이아웃] ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 차트 & 뉴스", "🏢 재무제표", "📈 시장 비교", "🧪 백테스트", "🏆 시총 TOP 100 & AI 추천"])

        with tab1:
            col1, col2 = st.columns([1.1, 2.3])
            with col1:
                st.subheader("💡 AI 예측 & 백테스트 검증")
                if test_acc > 0: st.info(f"🧪 **과거 20% 백테스트 적중률**: **{test_acc:.1f}%**")
                if up_prob > 50: st.success(f"📈 **내일 상승 예상 확률**: **{up_prob:.1f}%**")
                else: st.error(f"📉 **내일 하락 예상 확률**: **{100-up_prob:.1f}%**")
                
                st.subheader("📰 실시간 뉴스 분석 (한국어 AI)")
                try:
                    res = requests.get(f"https://news.google.com/rss/search?q={urllib.parse.quote(company_display_name)}&hl=ko&gl=KR&ceid=KR:ko", timeout=5)
                    root = ET.fromstring(res.content)
                    articles = [{'title': item.find('title').text.split(' - ')[0], 'link': item.find('link').text} for item in root.findall('.//item')[:10] if item.find('title') is not None]
                except: articles = []
                
                if articles:
                    @st.cache_resource
                    def load_korean_ai(): return pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
                    ai_model = load_korean_ai()
                    with st.container(height=350, border=True):
                        for art in articles:
                            res = ai_model(art['title'])[0]['label'].upper()
                            icon = "📈 [호재]" if res == "POSITIVE" else "📉 [악재]" if res == "NEGATIVE" else "➖ [중립]"
                            st.markdown(f"{icon} [{

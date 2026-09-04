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

# --- [1. 공통 데이터 엔진 (실시간 자동완성용 데이터 준비)] ---
@st.cache_data(ttl=3600)
def load_krx_data():
    return fdr.StockListing('KRX')

@st.cache_data(ttl=3600)
def get_stock_list():
    try:
        krx_df = load_krx_data()
        # "종목명 (코드)" 형태로 리스트 생성
        krx_list = [f"{row['Name']} ({row['Code']})" for _, row in krx_df.iterrows()]
    except:
        krx_list = []
        
    global_list = [
        "애플 (AAPL)", "테슬라 (TSLA)", "엔비디아 (NVDA)", "마이크로소프트 (MSFT)",
        "구글 (GOOGL)", "아마존 (AMZN)", "메타 (META)"
    ]
    
    # 순수하게 주식 목록만 반환 (안내 문구 제외)
    return global_list + krx_list

@st.cache_resource
def load_korean_ai(): 
    return pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

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
    df['RSI'] = 100 - (100 / (1 + rs))
    
    ema12, ema26 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # ML 모델 학습
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

    # 지표 데이터 추출
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

    # 상단 요약 바 
    st.success(f"🔍 **{company_display_name}** (`{ticker_code}`) 개별 분석 완료")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재 주가", price_fmt)
    c2.metric("시가총액", mkt_cap_str)
    c3.metric("52주 최고", high52)
    c4.metric("52주 최저", low52)
    c5.metric("RSI (과열도)", f"{latest_rsi:.1f}", "과매수 ⚠️" if latest_rsi >= 70 else "과매도 📉" if latest_rsi <= 30 else "중립")
    st.divider()

    chart_config = {'displayModeBar': False, 'scrollZoom': False}

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
                news_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(company_display_name)}&hl=ko&gl=KR&ceid=KR:ko"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                res = requests.get(news_url, headers=headers, timeout=5)
                root = ET.fromstring(res.content)
                
                articles = []
                for item in root.findall('.//item')[:10]:
                    title_tag = item.find('title')
                    link_tag = item.find('link')
                    if title_tag is not None and link_tag is not None:
                        articles.append({'title': title_tag.text.split(' - ')[0], 'link': link_tag.text})
            except:
                articles = []
            
            if articles:
                ai_model = load_korean_ai()
                with st.container(height=350, border=True):
                    for art in articles:
                        res = ai_model(art['title'])[0]['label'].upper()
                        icon = "📈 [호재]" if res == "POSITIVE" else "📉 [악재]" if res == "NEGATIVE" else "➖ [중립]"
                        st.markdown(f"{icon} [{art['title']}]({art['link']})")
            else: st.info("뉴스를 일시적으로 불러오지 못했습니다. (서버 지연)")

        with col2:
            st.subheader("📊 정밀 분석 차트")
            st.caption("📌 **차트 범례**: 🟥/🟩 캔들(주가) | 🟧 **20일선(단기)** | 🔷 **60일선(중기)** | ⚪ **볼린저 밴드** | 🟣 **MACD** | 🔴 **시그널** | 📊 **거래량**")
            
            chart_df = df.tail(120).copy()
            d_str = chart_df.index.strftime('%Y-%m-%d')
            colors = ['#26a69a' if r['Close'] >= r['Open'] else '#ef5350' for _, r in chart_df.iterrows()]
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.2, 0.25])
            fig.add_trace(go.Candlestick(x=d_str, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name='주가(Candle)', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['Upper_Band'], line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name='볼린저 상한'), row=1, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['Lower_Band'], line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name='볼린저 하한'), row=1, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['MA20'], line=dict(color='orange'), name='20일선'), row=1, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['MA60'], line=dict(color='#00bfff'), name='60일선'), row=1, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['MACD'], line=dict(color='#ab47bc'), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['Signal'], line=dict(color='#ff7043', dash='dot'), name='시그널(Signal)'), row=2, col=1)
            fig.add_trace(go.Bar(x=d_str, y=chart_df['Volume'], marker_color=colors, name

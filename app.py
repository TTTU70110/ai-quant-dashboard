import os
import re
import io
try:
    import FinanceDataReader as fdr
except ImportError:
    os.system("pip install finance-datareader lxml html5lib > /dev/null 2>&1")
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
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; white-space: nowrap !important; }
    [data-testid="stMetricLabel"] { font-size: 0.95rem !important; }
    </style>
    """, 
    unsafe_allow_html=True
)

# --- [좌측 사이드바 후원 링크] ---
with st.sidebar:
    st.markdown("### ☕ 개발자에게 마음 전하기")
    st.markdown(
        "<div style='font-size: 0.95em; color: #dddddd; line-height: 1.6; margin-bottom: 20px;'>"
        "본 대시보드가 성공적인 투자에 조금이나마 도움이 되셨다면, 따뜻한 커피 한 잔의 후원을 부탁드립니다.<br><br>"
        "보내주신 귀한 응원은 앞으로 더 유용하고 편리한 기능을 개발하는 데 정말 큰 힘이 됩니다. 늘 성공적인 투자를 기원합니다. 진심으로 감사합니다!"
        "</div>", 
        unsafe_allow_html=True
    )
    
    toon_link = "https://toon.at/donate/tttu70110"
    
    st.markdown(
        f"""
        <a href="{toon_link}" target="_blank" style="text-decoration: none;">
            <div style="background-color: #3b82f6; color: white; padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.05em;">
                💖 투네이션으로 후원하기
            </div>
        </a>
        <div style="text-align: center; font-size: 0.85em; color: gray; margin-top: 8px; margin-bottom: 30px;">
            (카카오페이 · 네이버페이 · 토스 가능)
        </div>
        """, 
        unsafe_allow_html=True
    )

st.title("🤖 투자 도우미 프로그램")
st.warning("⚠️ **[투자 유의사항]** 본 프로그램이 제공하는 정보는 참고용 보조 자료입니다. 모든 투자의 최종 판단과 그에 따른 책임은 전적으로 투자자 본인에게 있습니다.")

# --- [1. 공통 데이터 엔진 (2500개 전체 주식 복구 및 차단 우회)] ---
@st.cache_data(ttl=86400)
def load_krx_data():
    # 1. 봇 차단을 피하기 위한 강력한 사람 위장(User-Agent) 헤더 적용
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    try:
        # 한국거래소(KIND) 직접 접속 및 크롤링
        kospi_res = requests.get('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType=stockMkt', headers=headers, timeout=10)
        kosdaq_res = requests.get('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13&marketType=kosdaqMkt', headers=headers, timeout=10)
        
        kospi_df = pd.read_html(io.StringIO(kospi_res.text), header=0)[0]
        kospi_df['Market'] = 'KOSPI'
        kosdaq_df = pd.read_html(io.StringIO(kosdaq_res.text), header=0)[0]
        kosdaq_df['Market'] = 'KOSDAQ'
        
        df = pd.concat([kospi_df, kosdaq_df], ignore_index=True)
        df = df[['회사명', '종목코드', '업종', 'Market']].rename(columns={'회사명': 'Name', '종목코드': 'Code', '업종': 'Sector'})
        df['Code'] = df['Code'].astype(str).str.zfill(6)
        df['Marcap'] = 0
        df['ChagesRatio'] = 0.0
        return df
    except Exception:
        # 2. KIND 접속 실패 시 파이낸스데이터리더(FDR)로 2차 시도
        try:
            df = fdr.StockListing('KRX')
            return df
        except:
            return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_stock_list():
    global_list = [
        "애플 (AAPL)", "테슬라 (TSLA)", "엔비디아 (NVDA)", "마이크로소프트 (MSFT)", 
        "구글 (GOOGL)", "아마존 (AMZN)", "메타 (META)", "TSMC (TSM)", "브로드컴 (AVGO)", 
        "일라이 릴리 (LLY)", "JP모건 (JPM)", "버크셔 해서웨이 (BRK-B)", "코인베이스 (COIN)"
    ]
    
    krx_list = []
    krx_df = load_krx_data()
    if not krx_df.empty:
        krx_list = [f"{row['Name']} ({row['Code']})" for _, row in krx_df.iterrows()]
        
    final_list = global_list + krx_list
    unique_list = list(dict.fromkeys(final_list))
    return unique_list

@st.cache_resource
def load_korean_ai(): 
    return pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

@st.cache_data(ttl=3600)
def get_fear_and_greed_index():
    try:
        spy = yf.Ticker("SPY").history(period="1mo")
        vix = yf.Ticker("^VIX").history(period="1mo")
        
        spy = spy.dropna(subset=['Close'])
        vix = vix.dropna(subset=['Close'])
        
        delta = spy['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = delta.where(delta < 0, 0).abs().rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        current_vix = vix['Close'].iloc[-1]
        vix_score = 100 - ((current_vix - 10) / 30) * 100
        vix_score = max(0, min(100, vix_score))
        
        fgi_score = (rsi * 0.6) + (vix_score * 0.4)
        return int(fgi_score)
    except:
        return 50

@st.cache_data(ttl=86400)
def get_etf_list():
    fallback_etf = pd.DataFrame([
        {'Symbol': '069500', 'Name': 'KODEX 200', 'Price': 35000},
        {'Symbol': '360750', 'Name': 'TIGER 미국S&P500', 'Price': 15000},
        {'Symbol': '133690', 'Name': 'TIGER 미국나스닥100', 'Price': 80000},
        {'Symbol': '305540', 'Name': 'TIGER 2차전지테마', 'Price': 20000},
        {'Symbol': '091160', 'Name': 'KODEX 반도체', 'Price': 30000},
    ])
    try:
        etf_df = fdr.StockListing('ETF/KR')
        if not etf_df.empty: return etf_df.head(100)
    except:
        pass
    return fallback_etf


# --- [2. 핵심 분석 대시보드 로직] ---
def run_dashboard(ticker_code, company_display_name):
    stock = yf.Ticker(ticker_code)
    df = stock.history(period="2y")
    
    if not df.empty:
        df = df.dropna(subset=['Close', 'High', 'Low'])
    
    if df.empty or len(df) < 30:
        st.warning("데이터를 불러오지 못했습니다. 종목명이나 코드가 정확한지 확인해주세요.")
        return
        
    info = stock.info
    is_korean = ticker_code.endswith('.KS') or ticker_code.endswith('.KQ')
    currency = "₩" if is_korean else "$"
    
    current_price = float(df['Close'].iloc[-1])
    if pd.isna(current_price): current_price = 0.0 
    
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    df['STD20'] = df['Close'].rolling(20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0).abs()
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    df['RSI'] = df['RSI'].fillna(50.0)
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    ml_df = df.dropna().copy()
    
    if len(ml_df) > 50:
        features = ['MA20', 'MA60', 'RSI', 'MACD', 'Price_Change', 'Volume_Change']
        X = ml_df[features]
        y = ml_df['Target']
        split_idx = int(len(ml_df) * 0.8)
        
        test_model = RandomForestClassifier(n_estimators=100, random_state=42)
        test_model.fit(X.iloc[:split_idx], y.iloc[:split_idx])
        test_preds = test_model.predict(X.iloc[split_idx:])
        test_acc = accuracy_score(y.iloc[split_idx:], test_preds) * 100
        
        final_model = RandomForestClassifier(n_estimators=100, random_state=42)
        final_model.fit(X, y)
        up_prob = final_model.predict_proba(X.iloc[-1:])[0][1] * 100
    else:
        up_prob = 50.0
        test_acc = 0.0

    if is_korean:
        price_fmt = f"{currency}{int(current_price):,}"
    else:
        price_fmt = f"{currency}{current_price:,.2f}"
    
    mkt_cap_str = "N/A"
    try:
        krx_df = load_krx_data()
        if not krx_df.empty and is_korean:
            code_only = ticker_code.split('.')[0]
            match = krx_df[krx_df['Code'] == code_only]
            if not match.empty:
                mkt_cap = float(match.iloc[0].get('Marcap', 0))
                if mkt_cap > 0:
                    mkt_cap_str = f"{mkt_cap / 1_000_000_000_000:.2f}조 원"
    except:
        pass
        
    if mkt_cap_str == "N/A" and not is_korean:
        mkt_cap = info.get('marketCap', 0)
        if mkt_cap: 
            mkt_cap_str = f"${mkt_cap / 1_000_000_000:.2f}B"

    last_252_days = df.tail(252)
    high52_val = float(last_252_days['High'].max())
    low52_val = float(last_252_days['Low'].min())
    
    if pd.isna(high52_val): high52_val = current_price
    if pd.isna(low52_val): low52_val = current_price
    
    if is_korean:
        high52 = f"{currency}{int(high52_val):,}"
        low52 = f"{currency}{int(low52_val):,}"
    else:
        high52 = f"{currency}{high52_val:.2f}"
        low52 = f"{currency}{low52_val:.2f}"
        
    latest_rsi = df['RSI'].iloc[-1]
    rsi_status = "과매수 ⚠️" if latest_rsi >= 70 else "과매도 📉" if latest_rsi <= 30 else "중립"
    trend_status = "상승세 📈" if current_price > df['MA20'].iloc[-1] else "하락세 📉"
    macd_status = "매수세 유입(골든크로스)" if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else "매도세 우위(데드크로스)"

    st.success(f"🔍 **{company_display_name}** ({ticker_code}) 종목 분석 완료")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재 주가", price_fmt)
    c2.metric("시가총액", mkt_cap_str)
    c3.metric("52주 최고", high52)
    c4.metric("52주 최저", low52)
    c5.metric("RSI (과열도)", f"{latest_rsi:.1f}", rsi_status)
    
    # ★ 신규 기능 1: 실시간 뉴스 크롤링 및 감성 분석 ★
    articles = []
    pos_arts, neg_arts, neu_arts = [], [], []
    
    try:
        enc_query = urllib.parse.quote(company_display_name)
        news_url = f"https://news.google.com/rss/search?q={enc_query}&hl=ko&gl=KR&ceid=KR:ko"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(news_url, headers=headers, timeout=5)
        root = ET.fromstring(res.content)
        
        for item in root.findall('.//item')[:10]:
            t_tag = item.find('title')
            l_tag = item.find('link')
            if t_tag is not None and l_tag is not None:
                articles.append({'title': t_tag.text.split(' - ')[0], 'link': l_tag.text})
                
        if articles:
            ai_model = load_korean_ai()
            for art in articles:
                res_label = ai_model(art['title'])[0]['label'].upper()
                art['sentiment'] = res_label
                if res_label == "POSITIVE": pos_arts.append(art['title'])
                elif res_label == "NEGATIVE": neg_arts.append(art['title'])
                else: neu_arts.append(art['title'])
    except:
        pass

    # --- [상세 브리핑 문구 생성기] ---
    
    # 1. 하루 요약 텍스트
    if articles:
        pos_ratio = len(pos_arts) / len(articles) * 100
        neg_ratio = len(neg_arts) / len(articles) * 100
        
        if pos_ratio > neg_ratio and pos_arts:
            news_trend = f"오늘 수집된 최신 기사 중 **긍정적 반응이 {pos_ratio:.0f}%**로 시장에서 '호재'가 더 강하게 부각되고 있습니다."
        elif neg_ratio > pos_ratio and neg_arts:
            news_trend = f"오늘 수집된 최신 기사 중 **부정적 반응이 {neg_ratio:.0f}%**로 시장에서 '악재' 우려가 더 큰 상황입니다."
        else:
            news_trend = "현재 뚜렷한 초대형 호재나 악재 없이 **중립적인 뉴스 흐름**을 보이며 관망세가 짙습니다."
            
        news_details_list = []
        for i, art in enumerate(articles[:3]):
            icon = "🟢" if art['sentiment'] == "POSITIVE" else "🔴" if art['sentiment'] == "NEGATIVE" else "⚪"
            news_details_list.append(f"{icon} {art['title']}")
        news_details = "<br>".join(news_details_list)
    else:
        news_trend = "오늘 날짜로 갱신된 주요 뉴스 이슈가 부족하여 뉴스 요약이 제한적입니다."
        news_details = "수집된 최신 기사가 없습니다."

    # 2. 내일의 주가 전망 텍스트
    if up_prob >= 60:
        ml_pred = f"상승 예측 확률이 **{up_prob:.1f}%**로 매우 높게 나타났습니다. 머신러닝 패턴상 내일 **강한 상승 모멘텀**이 기대됩니다."
    elif up_prob >= 50:
        ml_pred = f"상승 예측 확률이 **{up_prob:.1f}%**로 집계되었습니다. 뚜렷한 급등보다는 **소폭 상승 및 보합권 횡보**가 예상됩니다."
    elif up_prob >= 40:
        ml_pred = f"하락 예측 확률이 **{100-up_prob:.1f}%**로 조금 더 높습니다. 내일은 **단기적인 하락이나 지지선 테스트**에 주의해야 합니다."
    else:
        ml_pred = f"하락 예측 확률이 **{100-up_prob:.1f}%**로 높게 분석되었습니다. 내일은 **하락 리스크가 크므로 보수적인 접근**을 권장합니다."
        
    # 3. 상세 기술적 지표 문구
    vol_chg = df['Volume_Change'].iloc[-1]
    if pd.isna(vol_chg): vol_chg = 0
    vol_status = "급격히 증가하며 시장의 관심이 쏠리고" if vol_chg > 0.5 else "다소 감소하며 눈치 보기 장세가 이어지고" if vol_chg < -0.3 else "평이한 수준을 유지하고"
    
    bb_upper = df['Upper_Band'].iloc[-1]
    bb_lower = df['Lower_Band'].iloc[-1]
    if current_price > bb_upper:
        bb_status = "볼린저 밴드 상단을 돌파하며 **단기 과열 상태**를 보이고 있습니다."
    elif current_price < bb_lower:
        bb_status = "볼린저 밴드 하단을 이탈하여 **단기 바닥/과매도 구간**에 진입했습니다."
    else:
        bb_status = "볼린저 밴드 내에서 **안정적인 궤도**를 그리고 있습니다."

    # --- [UI 출력: AI 데일리 브리핑] ---
    st.markdown("### 🤖 AI 데일리 종합 브리핑")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        with st.container(border=True, height=230):
            st.markdown("#### 📰 오늘 뉴스 기반 하루 요약")
            st.markdown(news_trend)
            st.markdown("**[가장 많이 언급된 뉴스 Top 3]**")
            st.markdown(f"<div style='font-size: 0.9em; color: #aaaaaa;'>{news_details}</div>", unsafe_allow_html=True)
            
    with col_b2:
        with st.container(border=True, height=230):
            st.markdown("#### 🔮 내일 주가 및 기술적 전망")
            st.markdown(ml_pred)
            st.markdown(
                f"<div style='font-size: 0.9em; color: #aaaaaa; margin-top: 10px;'>"
                f"- <b>추세/거래량:</b> 현재 주가는 20일선 기준 <b>{trend_status}</b>이며, 전일 대비 거래량은 <b>{vol_status}</b> 있습니다.<br>"
                f"- <b>보조지표:</b> MACD 지표상 <b>{macd_status}</b> 중이며, {bb_status}"
                f"</div>", 
                unsafe_allow_html=True
            )
            
    st.divider()

    chart_config = {'displayModeBar': False, 'scrollZoom': False}

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 차트 & 뉴스 원문", "🏢 재무제표", "📈 시장 비교", "🧪 백테스트", "🚀 AI 스캐너", "🛒 ETF 탐색기"
    ])

    with tab1:
        col1, col2 = st.columns([1.1, 2.3])
        with col1:
            st.subheader("💡 AI 백테스트 신뢰도")
            if test_acc > 0: 
                st.info(f"🧪 **과거 데이터 기반 AI 적중률**: **{test_acc:.1f}%**")
                
            st.subheader("📰 실시간 뉴스 원문 목록")
            if articles:
                with st.container(height=420, border=True):
                    for art in articles:
                        icon = "🟢" if art['sentiment'] == "POSITIVE" else "🔴" if art['sentiment'] == "NEGATIVE" else "⚪"
                        st.markdown(f"{icon} [{art['title']}]({art['link']})")
            else: 
                st.info("뉴스를 일시적으로 불러오지 못했습니다.")

        with col2:
            st.subheader("📊 정밀 분석 차트")
            
            chart_df = df.tail(120).copy()
            d_str = chart_df.index.strftime('%Y-%m-%d')
            colors = ['#26a69a' if r['Close'] >= r['Open'] else '#ef5350' for _, r in chart_df.iterrows()]
            
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.2, 0.25])
            
            fig.add_trace(go.Candlestick(
                x=d_str, open=chart_df['Open'], high=chart_df['High'], 
                low=chart_df['Low'], close=chart_df['Close'], name='주가', 
                increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=d_str, y=chart_df['Upper_Band'], 
                line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name='볼린저 상한'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=d_str, y=chart_df['Lower_Band'], 
                line=dict(color='rgba(255,255,255,0.3)', dash='dash'), name='볼린저 하한'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['MA20'], line=dict(color='orange'), name='20일선'), row=1, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['MA60'], line=dict(color='#00bfff'), name='60일선'), row=1, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['MACD'], line=dict(color='#ab47bc'), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=d_str, y=chart_df['Signal'], line=dict(color='#ff7043', dash='dot'), name='시그널'), row=2, col=1)
            fig.add_trace(go.Bar(x=d_str, y=chart_df['Volume'], marker_color=colors, name='거래량'), row=3, col=1)
            
            fig.update_xaxes(fixedrange=True)
            fig.update_yaxes(fixedrange=True)
            
            fig.update_layout(
                xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False, xaxis3_rangeslider_visible=False, 
                height=600, margin=dict(l=0, r=0, t=30, b=0), template='plotly_dark', 
                showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True, config=chart_config)

    with tab2:
        st.subheader(f"🏢 {company_display_name} 연간 실적 추이")
        try:
            fin = stock.financials.T.sort_index() if not stock.financials.empty else pd.DataFrame()
            if not fin.empty and 'Total Revenue' in fin.columns and 'Net Income' in fin.columns:
                
                fin['Total Revenue'] = fin['Total Revenue'].fillna(0)
                fin['Net Income'] = fin['Net Income'].fillna(0)

                if is_korean:
                    fin['Rev_Disp'] = fin['Total Revenue'] / 100000000
                    fin['Net_Disp'] = fin['Net Income'] / 100000000
                    unit_str = "단위: 억 원"
                else:
                    fin['Rev_Disp'] = fin['Total Revenue'] / 1000000
                    fin['Net_Disp'] = fin['Net Income'] / 1000000
                    unit_str = "단위: 백만 달러 (M)"

                years = fin.index.strftime('%Y년')
                
                fin_fig = go.Figure()
                fin_fig.add_trace(go.Bar(
                    x=years, y=fin['Rev_Disp'], name='매출액', marker_color='#29b6f6', 
                    text=fin['Rev_Disp'].apply(lambda x: f"{x:,.0f}" if x != 0 else ""), textposition='auto'
                ))
                fin_fig.add_trace(go.Bar(
                    x=years, y=fin['Net_Disp'], name='당기순이익', marker_color='#66bb6a', 
                    text=fin['Net_Disp'].apply(lambda x: f"{x:,.0f}" if x != 0 else ""), textposition='auto'
                ))
                
                fin_fig.update_xaxes(fixedrange=True)
                fin_fig.update_yaxes(fixedrange=True)
                fin_fig.update_layout(barmode='group', template='plotly_dark', height=450, yaxis_title=unit_str)
                st.plotly_chart(fin_fig, use_container_width=True, config=chart_config)
            else: 
                st.info("재무 데이터를 제공하지 않습니다.")
        except: 
            st.warning("재무 데이터를 불러오는 중 오류가 발생했습니다.")

    with tab3:
        st.subheader(f"📈 시장 벤치마크 수익률 비교")
        try:
            b_tick, b_name = ("^KS11", "코스피") if is_korean else ("SPY", "S&P 500")
            bench_df = yf.Ticker(b_tick).history(period="2y")
            
            bench_df = bench_df.dropna(subset=['Close'])
            c_dates = df.index.intersection(bench_df.index)
            
            comp_fig = go.Figure()
            y_stock = (df.loc[c_dates, 'Close'] / df.loc[c_dates, 'Close'].iloc[0] - 1) * 100
            y_bench = (bench_df.loc[c_dates, 'Close'] / bench_df.loc[c_dates, 'Close'].iloc[0] - 1) * 100
            
            comp_fig.add_trace(go.Scatter(x=c_dates.strftime('%Y-%m-%d'), y=y_stock, name=company_display_name, line=dict(color='#ffca28')))
            comp_fig.add_trace(go.Scatter(x=c_dates.strftime('%Y-%m-%d'), y=y_bench, name=b_name, line=dict(color='white', dash='dot')))
            
            comp_fig.update_xaxes(fixedrange=True)
            comp_fig.update_yaxes(fixedrange=True)
            comp_fig.update_layout(template='plotly_dark', height=500, yaxis_title="수익률 (%)", hovermode="x unified")
            st.plotly_chart(comp_fig, use_container_width=True, config=chart_config)
        except: 
            st.warning("비교 차트를 불러올 수 없습니다.")

    with tab4:
        st.subheader("🧪 나만의 투자 전략 백테스트")
        sc1, sc2 = st.columns(2)
        sim_short = sc1.slider("단기 이평선", 5, 50, 20)
        sim_long = sc2.slider("장기 이평선", 50, 200, 60)
        
        if sim_short >= sim_long: 
            st.error("⚠️ 단기는 장기보다 작아야 합니다.")
        else:
            sim_df = df.copy().dropna()
            sim_df['S'] = sim_df['Close'].rolling(sim_short).mean()
            sim_df['L'] = sim_df['Close'].rolling(sim_long).mean()
            sim_df['Ret'] = np.where(sim_df['S'] > sim_df['L'], 1, 0)
            sim_df['Ret'] = sim_df['Ret'].shift(1) * sim_df['Price_Change']
            sim_df = sim_df.dropna()
            
            strat_ret = (1 + sim_df['Ret']).cumprod() - 1
            hold_ret = (1 + sim_df['Price_Change']).cumprod() - 1
            
            st.markdown(f"**💡 최종 수익률**: 시뮬레이션 전략 **{strat_ret.iloc[-1]*100:.1f}%** vs 단순 보유 **{hold_ret.iloc[-1]*100:.1f}%**")
            
            sim_fig = go.Figure()
            sim_fig.add_trace(go.Scatter(x=sim_df.index.strftime('%Y-%m-%d'), y=strat_ret*100, name='전략 수익률', line=dict(color='#ff4081')))
            sim_fig.add_trace(go.Scatter(x=sim_df.index.strftime('%Y-%m-%d'), y=hold_ret*100, name='단순 보유', line=dict(color='#90caf9', dash='dot')))
            
            sim_fig.update_xaxes(fixedrange=True)
            sim_fig.update_yaxes(fixedrange=True)
            sim_fig.update_layout(template='plotly_dark', height=450, hovermode="x unified")
            st.plotly_chart(sim_fig, use_container_width=True, config=chart_config)

    with tab5:
        st.subheader("🚀 시가총액 TOP 100 & 내일의 급등주 AI 스캐너")
        
        try:
            krx_df = load_krx_data()
            has_marcap = 'Marcap' in krx_df.columns and pd.to_numeric(krx_df['Marcap'], errors='coerce').sum() > 0
            
            if not krx_df.empty and has_marcap:
                top100 = krx_df.sort_values(by='Marcap', ascending=False).head(100).reset_index(drop=True)
                top100.index = top100.index + 1
                
                st.markdown("#### 🤖 내일의 급등주 AI 스캐너")
                if st.button("🔍 상위 100종목 AI 스캔 시작 (약 15~20초 소요)", type="primary", use_container_width=True):
                    my_bar = st.progress(0, text="AI가 데이터를 분석 중입니다...")
                    ai_results = []
                    
                    for i, row in top100.iterrows():
                        code, name = row['Code'], row['Name']
                        t_code = f"{code}{'.KQ' if 'KOSDAQ' in str(row['Market']).upper() else '.KS'}"
                        
                        try:
                            hist = yf.Ticker(t_code).history(period="3mo")
                            hist = hist.dropna(subset=['Close'])
                            
                            if len(hist) > 20:
                                hist['MA10'] = hist['Close'].rolling(10).mean()
                                hist['MA20'] = hist['Close'].rolling(20).mean()
                                delta2 = hist['Close'].diff()
                                rs2 = (delta2.where(delta2 > 0, 0)).rolling(14).mean() / ((delta2.where(delta2 < 0, 0)).rolling(14).mean().abs() + 1e-9)
                                hist['RSI'] = 100 - (100 / (1 + rs2))
                                hist['Price_Change'] = hist['Close'].pct_change()
                                hist['Volume_Change'] = hist['Volume'].pct_change()
                                hist['Target'] = np.where(hist['Close'].shift(-1) > hist['Close'], 1, 0)
                                
                                ml_df2 = hist.dropna()
                                if len(ml_df2) > 10:
                                    X2 = ml_df2[['MA10', 'MA20', 'RSI', 'Price_Change', 'Volume_Change']]
                                    y2 = ml_df2['Target']
                                    model2 = RandomForestClassifier(n_estimators=50, random_state=42).fit(X2, y2)
                                    prob2 = model2.predict_proba(X2.iloc[-1:])[0][1] * 100
                                    
                                    ai_results.append({
                                        '종목명': name,
                                        '상승 확률(%)': round(prob2, 1),
                                        '현재가': f"₩{int(hist['Close'].iloc[-1]):,}",
                                        'RSI (과열도)': round(hist['RSI'].iloc[-1], 1),
                                    })
                        except: 
                            pass
                        my_bar.progress(i / 100.0, text=f"분석 중... [{i}/100] {name}")
                    
                    my_bar.empty()
                    
                    if ai_results:
                        res_df = pd.DataFrame(ai_results)
                        top10 = res_df.sort_values(by='상승 확률(%)', ascending=False).head(10).reset_index(drop=True)
                        top10.index = top10.index + 1
                        st.success("🎉 **AI 스캔 완료! 내일 상승 확률이 가장 높은 TOP 10 종목입니다.**")
                        st.dataframe(top10, use_container_width=True)
                    else: 
                        st.error("데이터 수집 중 오류가 발생했습니다.")
                
                st.divider()
                st.markdown("#### 🏆 한국 주식 시가총액 순위 (1위 ~ 100위)")
                display_df = top100[['Code', 'Name', 'Close', 'ChagesRatio', 'Marcap']].copy()
                display_df.columns = ['종목코드', '종목명', '현재가', '등락률', '시가총액']
                display_df['현재가'] = display_df['현재가'].apply(lambda x: f"₩{int(x):,}")
                display_df['등락률'] = display_df['등락률'].apply(lambda x: f"{x:.2f}%")
                display_df['시가총액'] = display_df['시가총액'].apply(lambda x: f"{x / 1000000000000:.2f}조 원")
                st.dataframe(display_df, use_container_width=True, height=600)
            else:
                st.warning("⚠️ 현재 데이터 서버의 일시적 통신 문제로 랭킹 정보를 불러올 수 없습니다. 우회 로직을 통해 종목 검색은 정상 작동 중입니다.")
        except Exception:
            st.warning("⚠️ 현재 데이터 서버의 일시적 통신 문제로 랭킹 정보를 불러올 수 없습니다. 우회 로직을 통해 종목 검색은 정상 작동 중입니다.")

    with tab6:
        st.subheader("🛒 한국 상장 인기 ETF 탐색기")
        st.markdown("다양한 테마와 지수를 추종하는 ETF 목록과 개별 시세표를 확인하세요.")
        
        etf_df = get_etf_list()
        
        if not etf_df.empty:
            etf_options = [f"{row['Name']} ({row['Symbol']})" for _, row in etf_df.iterrows()]
            
            selected_etf = st.selectbox(
                "💡 분석할 ETF를 검색하거나 선택하세요:", 
                options=etf_options,
                index=None,
                placeholder="클릭하여 ETF를 선택하세요 (예: TIGER 미국S&P500)",
                key="etf_select_box"
            )
            
            if selected_etf:
                e_name = selected_etf.split(" (")[0]
                e_code = selected_etf.split(" (")[1].replace(")", "")
                
                c_cols1, c_cols2 = st.columns([1, 2])
                
                with c_cols1:
                    st.markdown(f"#### 📊 [{e_name}] 핵심 시세 요약")
                    with st.spinner("야후 파이낸스 데이터 연동 중..."):
                        e_hist = yf.Ticker(f"{e_code}.KS").history(period="1y")
                        if not e_hist.empty:
                            e_hist = e_hist.dropna(subset=['Close'])
                            cur_p = e_hist['Close'].iloc[-1]
                            high52 = e_hist['High'].max()
                            low52 = e_hist['Low'].min()
                            vol = e_hist['Volume'].iloc[-1]
                            
                            st.metric("현재가", f"₩{int(cur_p):,}")
                            st.metric("52주 최고가", f"₩{int(high52):,}")
                            st.metric("52주 최저가", f"₩{int(low52):,}")
                            st.metric("최근 일일 거래량", f"{int(vol):,} 주")
                        else:
                            st.warning("데이터를 불러올 수 없습니다.")
                            
                with c_cols2:
                    st.markdown("#### 📈 최근 1년 수익률 추이")
                    with st.spinner("차트를 그리는 중..."):
                        if not e_hist.empty:
                            fig_e = go.Figure()
                            fig_e.add_trace(go.Scatter(
                                x=e_hist.index.strftime('%Y-%m-%d'), 
                                y=e_hist['Close'], 
                                fill='tozeroy', 
                                line=dict(color='#ab47bc')
                            ))
                            fig_e.update_layout(template='plotly_dark', height=350, margin=dict(l=0, r=0, t=10, b=0))
                            fig_e.update_xaxes(fixedrange=True)
                            fig_e.update_yaxes(fixedrange=True)
                            st.plotly_chart(fig_e, use_container_width=True, config={'displayModeBar': False})
            
            st.divider()
            st.markdown("#### 📋 국내 상장 ETF Top 100 전체 목록")
            
            disp_etf = etf_df.copy()
            cols_to_show = []
            if 'Symbol' in disp_etf.columns: cols_to_show.append('Symbol')
            if 'Name' in disp_etf.columns: cols_to_show.append('Name')
            if 'Price' in disp_etf.columns:
                disp_etf['Price'] = disp_etf['Price'].apply(lambda x: f"₩{int(x):,}" if pd.notnull(x) else "-")
                cols_to_show.append('Price')
                
            disp_etf = disp_etf[cols_to_show]
            disp_etf.columns = ['종목코드', 'ETF명', '현재가'][:len(cols_to_show)]
            st.dataframe(disp_etf, use_container_width=True, hide_index=True)


# --- [3. 메인 화면 레이아웃 (종목 검색 & 홈)] ---
stock_options = get_stock_list()

col_search1, col_search2 = st.columns([1, 0.01])
with col_search1:
    selected_stock = st.selectbox(
        label="🔍 종목 검색",
        options=stock_options,
        index=None, 
        placeholder="🔍 종목명 검색 (엔터 오류 방지를 위해 마우스 클릭을 권장합니다!)",
        label_visibility="collapsed"
    )

if not selected_stock:
    st.divider()
    
    with st.container():
        col_space1, col_gauge, col_space2 = st.columns([1, 2, 1])
        
        with col_gauge:
            fgi_score = get_fear_and_greed_index()
            
            if fgi_score <= 25:
                fgi_color = "#ef5350"
                fgi_text = "극도의 공포 (Extreme Fear) - 저가 매수 찬스일 수 있습니다."
            elif fgi_score <= 45:
                fgi_color = "#ffa726"
                fgi_text = "공포 (Fear) - 시장이 움츠러들어 있습니다."
            elif fgi_score <= 55:
                fgi_color = "#ffca28"
                fgi_text = "중립 (Neutral) - 관망세가 짙은 시장입니다."
            elif fgi_score <= 75:
                fgi_color = "#9ccc65"
                fgi_text = "탐욕 (Greed) - 시장이 달아오르고 있습니다."
            else:
                fgi_color = "#66bb6a"
                fgi_text = "극도의 탐욕 (Extreme Greed) - 차익 실현을 고려할 때입니다."
                
            fig_fgi = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = fgi_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "🔥 오늘의 시장 공포·탐욕 지수 (AI 종합)", 'font': {'size': 20, 'color': 'white'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': fgi_color},
                    'bgcolor': "rgba(255,255,255,0.05)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': "rgba(239, 83, 80, 0.3)"},
                        {'range': [25, 45], 'color': "rgba(255, 167, 38, 0.3)"},
                        {'range': [45, 55], 'color': "rgba(255, 202, 40, 0.3)"},
                        {'range': [55, 75], 'color': "rgba(156, 204, 101, 0.3)"},
                        {'range': [75, 100], 'color': "rgba(102, 187, 106, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': fgi_score
                    }
                }
            ))
            fig_fgi.update_layout(height=350, margin=dict(t=60, b=20, l=20, r=20), template='plotly_dark')
            fig_fgi.update_xaxes(fixedrange=True)
            fig_fgi.update_yaxes(fixedrange=True)
            
            st.plotly_chart(fig_fgi, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f"<h4 style='text-align: center; color: {fgi_color};'>{fgi_text}</h4>", unsafe_allow_html=True)

else:
    company_name = selected_stock.split(" (")[0]
    stock_code = selected_stock.split(" (")[-1].replace(")", "")
    
    if stock_code.isalpha():
        final_ticker = stock_code
    else:
        final_ticker = f"{stock_code}.KS"
        try:
            krx_df = load_krx_data()
            if not krx_df.empty:
                market_info = krx_df[krx_df['Code'] == stock_code]
                if not market_info.empty:
                    market_type = market_info.iloc[0]['Market']
                    suffix = '.KQ' if 'KOSDAQ' in str(market_type).upper() else '.KS'
                    final_ticker = f"{stock_code}{suffix}"
        except:
            pass
            
    run_dashboard(final_ticker, company_name)

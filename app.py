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

st.set_page_config(layout="wide", page_title="AI 퀀트 투자 대시보드 Pro")

# --- [UI 디자인 강제 수정] ---
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.2rem !important; white-space: nowrap !important; }
[data-testid="stMetricLabel"] { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI 투자 보조 프로그램")

# ★ 투자 경고문 추가 ★
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
user_input = st.text_input("회사명 또는 종목코드를 입력하세요 (예: SK하이닉스, 삼성전자, 테슬라)", "SK하이닉스")

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

        # 요약 정보 포맷
        mkt_cap = info.get('marketCap', 0)
        mkt_cap_str = (f"{mkt_cap / 1e12:.2f}조" if is_korean else f"${mkt_cap / 1e9:.2f}B") if mkt_cap else "N/A"
        price_fmt = f"{currency}{current_price:,.0f}" if is_korean else f"{currency}{current_price:,.2f}"
        
        def safe_get(key, default="N/A", fmt=None):
            val = info.get(key); return (fmt(val) if fmt else str(val)) if val else default

        div_yield = safe_get('dividendYield', "0.00%", lambda x: f"{x*100:.2f}%")
        high52 = safe_get('fiftyTwoWeekHigh', "N/A", lambda x: f"{currency}{x:,.0f}" if is_korean else f"${x:,.2f}")
        low52 = safe_get('fiftyTwoWeekLow', "N/A", lambda x: f"{currency}{x:,.0f}" if is_korean else f"${x:,.2f}")
        latest_rsi = df['RSI'].iloc[-1]

        # 상단 요약 바
        st.success(f"🔍 **{company_display_name}** (`{ticker_code}`) 개별 분석")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("현재 주가", price_fmt)
        c2.metric("시가총액", mkt_cap_str)
        c3.metric("52주 최고", high52)
        c4.metric("52주 최저", low52)
        c5.metric("배당수익률", div_yield)
        c6.metric("RSI (과열도)", f"{latest_rsi:.1f}", "과매수 ⚠️" if latest_rsi >= 70 else "과매도 📉" if latest_rsi <= 30 else "중립")
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
                            st.markdown(f"{icon} [{art['title']}]({art['link']})")
                else: st.info("뉴스를 불러오지 못했습니다.")

            with col2:
                st.subheader("📊 정밀 분석 차트")
                
                # ★ 여기에 차트 선 설명(범례) 캡션을 부활시켰습니다! ★
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
                fig.add_trace(go.Bar(x=d_str, y=chart_df['Volume'], marker_color=colors, name='거래량'), row=3, col=1)
                
                # ★ showlegend=True로 변경하여 차트 상단에 클릭 가능한 뱃지 추가 ★
                fig.update_layout(
                    xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False, xaxis3_rangeslider_visible=False, 
                    height=600, margin=dict(l=0,r=0,t=30,b=0), template='plotly_dark', 
                    showlegend=True, 
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"🏢 {company_display_name} 연간 실적 추이")
            try:
                fin = stock.financials.T.sort_index() if not stock.financials.empty else pd.DataFrame()
                if not fin.empty and 'Total Revenue' in fin.columns and 'Net Income' in fin.columns:
                    fin_fig = go.Figure()
                    fin_fig.add_trace(go.Bar(x=fin.index.strftime('%Y-%m-%d'), y=fin['Total Revenue'], name='매출액', marker_color='#29b6f6'))
                    fin_fig.add_trace(go.Bar(x=fin.index.strftime('%Y-%m-%d'), y=fin['Net Income'], name='당기순이익', marker_color='#66bb6a'))
                    fin_fig.update_layout(barmode='group', template='plotly_dark', height=500)
                    st.plotly_chart(fin_fig, use_container_width=True)
                else: st.info("재무 데이터를 제공하지 않습니다.")
            except: st.warning("오류가 발생했습니다.")

        with tab3:
            st.subheader(f"📈 시장 벤치마크 수익률 비교")
            try:
                b_tick, b_name = ("^KS11", "코스피") if is_korean else ("SPY", "S&P 500")
                bench_df = yf.Ticker(b_tick).history(period="2y")
                c_dates = df.index.intersection(bench_df.index)
                comp_fig = go.Figure()
                comp_fig.add_trace(go.Scatter(x=c_dates.strftime('%Y-%m-%d'), y=(df.loc[c_dates,'Close']/df.loc[c_dates,'Close'].iloc[0]-1)*100, name=company_display_name, line=dict(color='#ffca28')))
                comp_fig.add_trace(go.Scatter(x=c_dates.strftime('%Y-%m-%d'), y=(bench_df.loc[c_dates,'Close']/bench_df.loc[c_dates,'Close'].iloc[0]-1)*100, name=b_name, line=dict(color='white', dash='dot')))
                comp_fig.update_layout(template='plotly_dark', height=500, yaxis_title="수익률 (%)", hovermode="x unified")
                st.plotly_chart(comp_fig, use_container_width=True)
            except: st.warning("비교 차트를 불러올 수 없습니다.")

        with tab4:
            st.subheader("🧪 나만의 투자 전략 백테스트")
            sc1, sc2 = st.columns(2)
            sim_short, sim_long = sc1.slider("단기 이평선", 5, 50, 20), sc2.slider("장기 이평선", 50, 200, 60)
            if sim_short >= sim_long: st.error("⚠️ 단기는 장기보다 작아야 합니다.")
            else:
                sim_df = df.copy().dropna()
                sim_df['S'], sim_df['L'] = sim_df['Close'].rolling(sim_short).mean(), sim_df['Close'].rolling(sim_long).mean()
                sim_df['Ret'] = np.where(sim_df['S'] > sim_df['L'], 1, 0)
                sim_df['Ret'] = sim_df['Ret'].shift(1) * sim_df['Price_Change']
                sim_df = sim_df.dropna()
                strat_ret, hold_ret = (1 + sim_df['Ret']).cumprod() - 1, (1 + sim_df['Price_Change']).cumprod() - 1
                st.markdown(f"**💡 최종 수익률**: 시뮬레이션 전략 **{strat_ret.iloc[-1]*100:.1f}%** vs 단순 보유 **{hold_ret.iloc[-1]*100:.1f}%**")
                
                sim_fig = go.Figure()
                sim_fig.add_trace(go.Scatter(x=sim_df.index.strftime('%Y-%m-%d'), y=strat_ret*100, name='전략 수익률', line=dict(color='#ff4081')))
                sim_fig.add_trace(go.Scatter(x=sim_df.index.strftime('%Y-%m-%d'), y=hold_ret*100, name='단순 보유', line=dict(color='#90caf9', dash='dot')))
                sim_fig.update_layout(template='plotly_dark', height=450, hovermode="x unified")
                st.plotly_chart(sim_fig, use_container_width=True)

        with tab5:
            st.subheader("🚀 시가총액 TOP 100 & 내일의 급등주 AI 스캐너")
            st.markdown("한국거래소(KRX) 시가총액 상위 100개 종목의 실시간 데이터를 바탕으로, AI가 내일 상승 확률이 가장 높은 **TOP 10 종목**을 추출합니다.")
            
            krx_df = load_krx_data()
            if 'Marcap' in krx_df.columns:
                top100 = krx_df.sort_values(by='Marcap', ascending=False).head(100).reset_index(drop=True)
                top100.index = top100.index + 1
                
                if st.button("🔍 상위 100종목 AI 스캔 시작 (약 15~20초 소요)", type="primary", use_container_width=True):
                    progress_text = "AI가 100개 종목의 최신 차트 데이터를 분석하여 모델을 학습 중입니다..."
                    my_bar = st.progress(0, text=progress_text)
                    
                    ai_results = []
                    for i, row in top100.iterrows():
                        code, name, market = row['Code'], row['Name'], row['Market']
                        t_code = f"{code}{'.KQ' if 'KOSDAQ' in str(market).upper() else '.KS'}"
                        
                        try:
                            hist = yf.Ticker(t_code).history(period="3mo")
                            if len(hist) > 20:
                                hist['MA10'] = hist['Close'].rolling(10).mean()
                                hist['MA20'] = hist['Close'].rolling(20).mean()
                                delta = hist['Close'].diff()
                                rs = (delta.where(delta > 0, 0)).rolling(14).mean() / ((delta.where(delta < 0, 0)).rolling(14).mean().abs() + 1e-9)
                                hist['RSI'] = 100 - (100 / (1 + rs))
                                hist['Price_Change'] = hist['Close'].pct_change()
                                hist['Volume_Change'] = hist['Volume'].pct_change()
                                hist['Target'] = np.where(hist['Close'].shift(-1) > hist['Close'], 1, 0)
                                
                                ml_df2 = hist.dropna()
                                if len(ml_df2) > 10:
                                    X2, y2 = ml_df2[['MA10', 'MA20', 'RSI', 'Price_Change', 'Volume_Change']], ml_df2['Target']
                                    model2 = RandomForestClassifier(n_estimators=50, random_state=42).fit(X2, y2)
                                    prob2 = model2.predict_proba(X2.iloc[-1:])[0][1] * 100
                                    
                                    ai_results.append({
                                        '종목명': name,
                                        '상승 확률(%)': round(prob2, 1),
                                        '현재가': f"₩{int(hist['Close'].iloc[-1]):,}",
                                        'RSI (과열도)': round(hist['RSI'].iloc[-1], 1),
                                    })
                        except: pass
                        my_bar.progress(i / 100.0, text=f"AI 분석 진행 중... [{i}/100] {name}")
                    my_bar.empty()
                    
                    if ai_results:
                        res_df = pd.DataFrame(ai_results)
                        top10 = res_df.sort_values(by='상승 확률(%)', ascending=False).head(10).reset_index(drop=True)
                        top10.index = top10.index + 1
                        st.success("🎉 **AI 스캔 완료! 내일 상승 확률이 가장 높은 TOP 10 종목입니다.**")
                        st.dataframe(top10, use_container_width=True)
                    else: st.error("데이터 수집 중 오류가 발생했습니다.")
                
                st.divider()
                st.markdown("### 🏆 한국 주식 시가총액 순위 (1위 ~ 100위)")
                display_df = top100[['Code', 'Name', 'Close', 'ChagesRatio', 'Marcap']].copy()
                display_df.columns = ['종목코드', '종목명', '현재가', '등락률', '시가총액']
                display_df['현재가'] = display_df['현재가'].apply(lambda x: f"₩{int(x):,}")
                display_df['등락률'] = display_df['등락률'].apply(lambda x: f"{x:.2f}%")
                display_df['시가총액'] = display_df['시가총액'].apply(lambda x: f"{x / 1000000000000:.2f}조 원")
                st.dataframe(display_df, use_container_width=True, height=600)
            else:
                st.warning("한국거래소(KRX) 데이터를 불러올 수 없습니다.")
    else:
        st.warning("데이터를 불러오지 못했습니다. 종목명이 정확한지 확인해 주세요.")

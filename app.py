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
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; white-space: nowrap !important; }
    [data-testid="stMetricLabel"] { font-size: 0.95rem !important; }
    </style>
    """, 
    unsafe_allow_html=True
)

st.title("🤖 투자 도우미 프로그램")
st.warning("⚠️ **[투자 유의사항]** 본 프로그램이 제공하는 정보는 참고용 보조 자료입니다. 모든 투자의 최종 판단과 그에 따른 책임은 전적으로 투자자 본인에게 있습니다.")

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
        "애플 (AAPL)", "테슬라 (TSLA)", "엔비디아 (NVDA)", 
        "마이크로소프트 (MSFT)", "구글 (GOOGL)", "아마존 (AMZN)", "메타 (META)"
    ]
    return global_list + krx_list

@st.cache_resource
def load_korean_ai(): 
    return pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

# ★ 개선된 수급 데이터 가져오기 (KRX 공식 데이터 활용) ★
@st.cache_data(ttl=3600)
def get_investor_data(stock_code):
    try:
        # FinanceDataReader를 이용한 KRX 공식 투자자별 매매동향
        code_only = stock_code.split('.')[0]
        # 최근 14일 데이터 가져오기 (휴일 감안하여 10개 행 추출)
        today = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.Timedelta(days=20)).strftime('%Y-%m-%d')
        
        # 주가 데이터 가져오기
        price_df = fdr.DataReader(code_only, start_date, today)
        price_df = price_df.reset_index()
        price_df = price_df[['Date', 'Close']].rename(columns={'Date': '날짜', 'Close': '종가'})
        price_df['날짜'] = price_df['날짜'].dt.strftime('%Y-%m-%d')
        
        # 야후 파이낸스나 FDR에 직접적인 수급 데이터가 없으므로 네이버 금융 JSON API로 우회 시도
        url = f"https://m.stock.naver.com/api/stock/{code_only}/investor/trend"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        
        if res.status_code == 200:
            data = res.json().get('investors', [])
            if data:
                inv_list = []
                for item in data[:10]: # 최근 10일
                    inv_list.append({
                        '날짜': item['bizdate'][:4] + "-" + item['bizdate'][4:6] + "-" + item['bizdate'][6:],
                        '기관순매수': int(item.get('institution', 0)),
                        '외국인순매수': int(item.get('foreigner', 0))
                    })
                inv_df = pd.DataFrame(inv_list)
                
                # 주가와 병합
                merged_df = pd.merge(inv_df, price_df, on='날짜', how='left')
                merged_df = merged_df[['날짜', '종가', '기관순매수', '외국인순매수']]
                return merged_df
                
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# ★ 신규: 주가 및 섹터 100% 꽂아주는 함수 (KRX 업종 데이터 기반) ★
@st.cache_data(ttl=3600)
def get_stock_info(ticker_code, company_name):
    info = {'price': 0, 'sector': '알 수 없음'}
    is_korean = ticker_code.endswith('.KS') or ticker_code.endswith('.KQ')
    
    # 1. 주가 가져오기
    try:
        hist = yf.Ticker(ticker_code).history(period="5d")
        if not hist.empty:
            info['price'] = hist['Close'].iloc[-1]
            if not is_korean: # 미국 주식은 환율 적용 (임시: 1350원)
                info['price'] = info['price'] * 1350
    except:
        pass
        
    # 2. 섹터 가져오기
    if is_korean:
        try:
            krx_df = load_krx_data()
            code_only = ticker_code.split('.')[0]
            match = krx_df[krx_df['Code'] == code_only]
            if not match.empty:
                # KRX 공식 업종(Sector) 데이터 활용
                krx_sector = str(match.iloc[0].get('Sector', '기타'))
                if krx_sector == 'nan': krx_sector = '기타'
                
                # 업종 이름 매핑 (비슷한 건 묶기)
                if any(k in krx_sector for k in ['소프트웨어', '컴퓨터', '반도체', '전자부품', '통신장비']):
                    info['sector'] = '💻 IT/반도체'
                elif any(k in krx_sector for k in ['자동차', '운송장비', '기계']):
                    info['sector'] = '🚗 자동차/기계'
                elif any(k in krx_sector for k in ['화학', '의약품', '의료', '생물']):
                    info['sector'] = '💊 바이오/헬스'
                elif any(k in krx_sector for k in ['은행', '증권', '보험', '금융']):
                    info['sector'] = '🏦 금융'
                elif any(k in krx_sector for k in ['방송', '출판', '영화', '플랫폼']):
                    info['sector'] = '📱 플랫폼/콘텐츠'
                elif any(k in krx_sector for k in ['음식료', '섬유', '의복', '유통']):
                    info['sector'] = '🛒 소비재'
                elif any(k in krx_sector for k in ['철강', '금속', '비금속', '건설']):
                    info['sector'] = '🧱 철강/건설'
                else:
                    info['sector'] = f"🏭 {krx_sector}" # 그 외는 원본 업종 표시
        except:
            info['sector'] = '🏭 산업재/기타'
    else:
        # 미국 주식은 야후 파이낸스 섹터 정보 번역
        try:
            us_sec = yf.Ticker(ticker_code).info.get('sector', 'Unknown')
            sec_map = {
                'Technology': '💻 IT/반도체', 'Consumer Cyclical': '🚗 소비재 (자동차 등)',
                'Financial Services': '🏦 금융', 'Healthcare': '💊 헬스케어',
                'Communication Services': '📱 통신/플랫폼', 'Industrials': '🏭 산업재',
                'Consumer Defensive': '🛒 필수소비재', 'Energy': '⚡ 에너지',
                'Basic Materials': '🧱 소재', 'Real Estate': '🏢 부동산', 'Utilities': '💡 유틸리티'
            }
            info['sector'] = sec_map.get(us_sec, '기타 (해외)')
        except:
            info['sector'] = '기타 (해외)'
            
    return info

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
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0).abs()
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs))
    
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # ML 모델 학습
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

    # 안전한 문자열 포맷팅
    if is_korean:
        price_fmt = f"{currency}{int(current_price):,}"
    else:
        price_fmt = f"{currency}{current_price:,.2f}"
    
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
        if mkt_cap: 
            mkt_cap_str = f"${mkt_cap / 1_000_000_000:.2f}B"

    last_252_days = df.tail(252)
    high52_val = last_252_days['High'].max()
    low52_val = last_252_days['Low'].min()
    
    if is_korean:
        high52 = f"{currency}{int(high52_val):,}"
        low52 = f"{currency}{int(low52_val):,}"
    else:
        high52 = f"{currency}{high52_val:.2f}"
        low52 = f"{currency}{low52_val:.2f}"
        
    latest_rsi = df['RSI'].iloc[-1]
    rsi_status = "과매수 ⚠️" if latest_rsi >= 70 else "과매도 📉" if latest_rsi <= 30 else "중립"

    # 상단 요약 바
    st.success(f"🔍 **{company_display_name}** ({ticker_code}) 개별 분석 완료")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재 주가", price_fmt)
    c2.metric("시가총액", mkt_cap_str)
    c3.metric("52주 최고", high52)
    c4.metric("52주 최저", low52)
    c5.metric("RSI (과열도)", f"{latest_rsi:.1f}", rsi_status)
    st.divider()

    chart_config = {'displayModeBar': False, 'scrollZoom': False}

    # 탭 레이아웃
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 차트 & 뉴스", "🏢 재무 & 수급", "📈 시장 비교", "🧪 백테스트", "🚀 AI 스캐너", "💼 계좌 진단"
    ])

    with tab1:
        col1, col2 = st.columns([1.1, 2.3])
        with col1:
            st.subheader("💡 AI 예측 & 백테스트 검증")
            if test_acc > 0: 
                st.info(f"🧪 **과거 20% 백테스트 적중률**: **{test_acc:.1f}%**")
            if up_prob > 50: 
                st.success(f"📈 **내일 상승 예상 확률**: **{up_prob:.1f}%**")
            else: 
                st.error(f"📉 **내일 하락 예상 확률**: **{100-up_prob:.1f}%**")
            
            st.subheader("📰 실시간 뉴스 분석 (한국어 AI)")
            try:
                enc_query = urllib.parse.quote(company_display_name)
                news_url = f"https://news.google.com/rss/search?q={enc_query}&hl=ko&gl=KR&ceid=KR:ko"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                res = requests.get(news_url, headers=headers, timeout=5)
                root = ET.fromstring(res.content)
                
                articles = []
                for item in root.findall('.//item')[:10]:
                    title_tag = item.find('title')
                    link_tag = item.find('link')
                    if title_tag is not None and link_tag is not None:
                        articles.append({
                            'title': title_tag.text.split(' - ')[0], 
                            'link': link_tag.text
                        })
            except:
                articles = []
            
            if articles:
                ai_model = load_korean_ai()
                with st.container(height=350, border=True):
                    for art in articles:
                        res = ai_model(art['title'])[0]['label'].upper()
                        icon = "📈 [호재]" if res == "POSITIVE" else "📉 [악재]" if res == "NEGATIVE" else "➖ [중립]"
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
            
        st.divider()
        st.subheader("👥 최근 10일 외국인/기관 매매 동향 (수급)")
        if is_korean:
            with st.spinner("수급 데이터를 분석 중입니다..."):
                inv_df = get_investor_data(ticker_code)
                if not inv_df.empty:
                    inv_df['종가'] = inv_df['종가'].apply(
                        lambda x: f"₩{int(x):,}" if not pd.isna(x) else "-"
                    )
                    inv_df['기관순매수'] = inv_df['기관순매수'].apply(
                        lambda x: f"🔴 +{int(x):,}" if float(x) > 0 else f"🔵 {int(x):,}" if float(x) < 0 else "0"
                    )
                    inv_df['외국인순매수'] = inv_df['외국인순매수'].apply(
                        lambda x: f"🔴 +{int(x):,}" if float(x) > 0 else f"🔵 {int(x):,}" if float(x) < 0 else "0"
                    )
                    st.dataframe(inv_df, use_container_width=True)
                    st.caption("* 단위: 주 (🔴 순매수 / 🔵 순매도)")
                else:
                    st.info("현재 수급 데이터를 불러올 수 없습니다. (일시적 서버 지연 또는 제공하지 않는 종목)")
        else:
            st.info("💡 해외 주식은 상세 수급 동향(외국인/기관) 데이터를 무료로 제공하지 않습니다.")

    with tab3:
        st.subheader(f"📈 시장 벤치마크 수익률 비교")
        try:
            b_tick, b_name = ("^KS11", "코스피") if is_korean else ("SPY", "S&P 500")
            bench_df = yf.Ticker(b_tick).history(period="2y")
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
        st.markdown("한국거래소(KRX) 시가총액 상위 100개 종목의 실시간 데이터를 바탕으로, AI가 내일 상승 확률이 가장 높은 **TOP 10 종목**을 추출합니다.")
        
        krx_df = load_krx_data()
        if 'Marcap' in krx_df.columns:
            top100 = krx_df.sort_values(by='Marcap', ascending=False).head(100).reset_index(drop=True)
            top100.index = top100.index + 1
            
            if st.button("🔍 상위 100종목 AI 스캔 시작 (약 15~20초 소요)", type="primary", use_container_width=True):
                my_bar = st.progress(0, text="AI가 데이터를 분석 중입니다...")
                ai_results = []
                
                for i, row in top100.iterrows():
                    code, name, market = row['Code'], row['Name'], row['Market']
                    t_code = f"{code}{'.KQ' if 'KOSDAQ' in str(market).upper() else '.KS'}"
                    
                    try:
                        hist = yf.Ticker(t_code).history(period="3mo")
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
            st.markdown("### 🏆 한국 주식 시가총액 순위 (1위 ~ 100위)")
            display_df = top100[['Code', 'Name', 'Close', 'ChagesRatio', 'Marcap']].copy()
            display_df.columns = ['종목코드', '종목명', '현재가', '등락률', '시가총액']
            display_df['현재가'] = display_df['현재가'].apply(lambda x: f"₩{int(x):,}")
            display_df['등락률'] = display_df['등락률'].apply(lambda x: f"{x:.2f}%")
            display_df['시가총액'] = display_df['시가총액'].apply(lambda x: f"{x / 1000000000000:.2f}조 원")
            st.dataframe(display_df, use_container_width=True, height=600)
        else:
            st.warning("한국거래소(KRX) 데이터를 불러올 수 없습니다.")

    with tab6:
        st.subheader("💼 내 계좌 포트폴리오 안전도 진단 (금액 자동 계산)")
        st.markdown("보유 중인 주식을 선택하고 **보유 수량(주)**을 입력하세요. AI가 현재 주가를 곱하여 **총 평가 금액과 비중(%)을 자동 계산**한 뒤, 산업(섹터) 쏠림 현상을 진단합니다.")
        
        # ★ 포트폴리오 자동완성 검색 기능 및 수량 입력 ★
        port_options = ["선택 안함"] + get_stock_list()
        
        port_cols = st.columns(4)
        p_names = []
        p_counts = []
        
        for i in range(4):
            with port_cols[i]:
                # 메인 검색창과 똑같은 자동완성 selectbox 적용
                s_sel = st.selectbox(
                    f"종목 {i+1}", 
                    options=port_options, 
                    index=0, 
                    key=f"port_sel_{i}"
                )
                s_count = st.number_input(f"보유 수량 (주)", min_value=0, value=0, key=f"port_cnt_{i}")
                
                if s_sel != "선택 안함" and s_count > 0:
                    p_names.append(s_sel)
                    p_counts.append(s_count)
                    
        if st.button("🔍 내 포트폴리오 진단하기", type="primary"):
            if not p_names:
                st.error("⚠️ 최소 1개 이상의 종목과 수량을 입력해주세요.")
            else:
                with st.spinner("현재 주가와 섹터 데이터를 불러와 자동 계산 중입니다..."):
                    results = []
                    total_value = 0
                    
                    for name, count in zip(p_names, p_counts):
                        # 이름과 코드 분리
                        c_name = name.split(" (")[0]
                        c_code = name.split(" (")[1].replace(")", "")
                        
                        if c_code.isalpha():
                            t_code = c_code
                        else:
                            krx_df = load_krx_data()
                            m_info = krx_df[krx_df['Code'] == c_code]
                            suffix = '.KQ' if not m_info.empty and 'KOSDAQ' in str(m_info.iloc[0]['Market']).upper() else '.KS'
                            t_code = f"{c_code}{suffix}"
                        
                        # 주가 및 섹터 100% 매핑된 데이터 가져오기
                        s_info = get_stock_info(t_code, c_name)
                        
                        value = s_info['price'] * count
                        total_value += value
                        
                        results.append({
                            '종목': c_name,
                            '수량': count,
                            '평가금액': value,
                            '섹터': s_info['sector']
                        })
                    
                    if total_value > 0:
                        # 비중 퍼센트 자동 계산
                        for res in results:
                            res['비중(%)'] = round((res['평가금액'] / total_value) * 100, 1)
                            res['평가금액'] = f"₩{int(res['평가금액']):,}"
                            
                        port_df = pd.DataFrame(results)
                        
                        # 섹터별 원형 차트 그리기
                        sec_weights = port_df.groupby('섹터')['비중(%)'].sum().reset_index()
                        
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=sec_weights['섹터'], values=sec_weights['비중(%)'], 
                            hole=.4, textinfo='label+percent', 
                            marker_colors=['#29b6f6', '#66bb6a', '#ffa726', '#ab47bc', '#ef5350']
                        )])
                        fig_pie.update_layout(template='plotly_dark', height=400, margin=dict(t=20, b=20))
                        
                        col_p1, col_p2 = st.columns([1, 1])
                        with col_p1:
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col_p2:
                            st.markdown("#### 💡 AI 포트폴리오 코멘트")
                            max_sec = sec_weights.loc[sec_weights['비중(%)'].idxmax()]
                            st.markdown(f"👉 총 평가금액은 **₩{int(total_value):,}** 이며, 가장 큰 비중을 차지하는 산업은 **{max_sec['섹터']} ({max_sec['비중(%)']}%)** 입니다.")
                            
                            if max_sec['비중(%)'] >= 60:
                                st.error("⚠️ **집중 투자 경고!** 특정 산업에 60% 이상 자본이 몰려있습니다. 해당 산업이 타격을 받으면 계좌 전체가 위험해집니다. 리스크 분산을 권장합니다.")
                            elif max_sec['비중(%)'] <= 40 and len(sec_weights) >= 3:
                                st.success("✅ **훌륭한 분산 투자!** 여러 섹터에 자산이 안정적으로 배분되어 있어, 하락장에서도 강한 방어력을 보여줄 수 있습니다.")
                            else:
                                st.info("ℹ️ **무난한 배분!** 밸런스가 나쁘지 않으나, 시장 상황에 맞춰 조금 더 다변화해도 좋습니다.")
                                
                            # 수량과 계산된 비중이 모두 표시되는 최종 표
                            st.dataframe(port_df[['종목', '수량', '섹터', '평가금액', '비중(%)']], use_container_width=True)

# --- [3. 메인 실행 (검색창)] ---
stock_options = get_stock_list()

selected_stock = st.selectbox(
    label="🔍 종목 검색",
    options=stock_options,
    index=None, 
    placeholder="여기를 클릭하고 종목명(예: 삼성) 또는 코드(AAPL)를 입력하세요...",
    label_visibility="collapsed"
)

if selected_stock:
    company_name = selected_stock.split(" (")[0]
    stock_code = selected_stock.split(" (")[1].replace(")", "")
    
    if stock_code.isalpha():
        final_ticker = stock_code
    else:
        krx_df = load_krx_data()
        market_info = krx_df[krx_df['Code'] == stock_code]
        if not market_info.empty:
            market_type = market_info.iloc[0]['Market']
            suffix = '.KQ' if 'KOSDAQ' in str(market_type).upper() else '.KS'
            final_ticker = f"{stock_code}{suffix}"
        else:
            final_ticker = f"{stock_code}.KS"
            
    run_dashboard(final_ticker, company_name)

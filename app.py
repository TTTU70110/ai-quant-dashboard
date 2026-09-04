import os
import re
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

# ★ ETF 전용 엔진 (강력한 보안 차단 우회 헤더 적용) ★
@st.cache_data(ttl=86400)
def get_etf_list():
    try:
        etf_df = fdr.StockListing('ETF/KR')
        if not etf_df.empty:
            return etf_df.head(100)
    except:
        pass
    
    return pd.DataFrame([
        {'Symbol': '069500', 'Name': 'KODEX 200', 'Price': 35000},
        {'Symbol': '360750', 'Name': 'TIGER 미국S&P500', 'Price': 15000},
        {'Symbol': '133690', 'Name': 'TIGER 미국나스닥100', 'Price': 80000},
        {'Symbol': '305540', 'Name': 'TIGER 2차전지테마', 'Price': 20000},
        {'Symbol': '091160', 'Name': 'KODEX 반도체', 'Price': 30000},
    ])

@st.cache_data(ttl=3600)
def get_kr_etf_constituents(code):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
        }
        res = requests.get(url, headers=headers, timeout=5)
        res.encoding = 'euc-kr' 
        
        block = re.search(r'구성종목.*?</table>', res.text, re.DOTALL)
        if block:
            names = re.findall(r'<a href="/item/main.naver\?code=\w+"[^>]*>(.*?)</a>', block.group(0))
            unique_names = []
            for n in names:
                if n not in unique_names:
                    unique_names.append(n)
            return unique_names
        return []
    except:
        return []


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

    st.success(f"🔍 **{company_display_name}** ({ticker_code}) 개별 분석 완료")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재 주가", price_fmt)
    c2.metric("시가총액", mkt_cap_str)
    c3.metric("52주 최고", high52)
    c4.metric("52주 최저", low52)
    c5.metric("RSI (과열도)", f"{latest_rsi:.1f}", rsi_status)
    
    trend_status = "상승세" if current_price > df['MA20'].iloc[-1] else "하락세"
    macd_status = "매수세가 유입" if df['MACD'].iloc[-1] > df['Signal'].iloc[-1] else "매수 심리가 다소 위축"
    
    if latest_rsi >= 70:
        ai_comment = f"현재 주가는 20일선 기준 **{trend_status}**이지만, 지표상 **단기 과열(과매수)** 구간입니다. 신규 매수보다는 관망이나 분할 매도를 고려해볼 수 있는 시점입니다."
    elif latest_rsi <= 30:
        ai_comment = f"현재 낙폭이 과대하여(과매도) **바닥권 반등**을 기대해볼 수 있는 구간입니다. {macd_status}되는지 관찰하며 분할 매수를 검토하기 좋습니다."
    else:
        ai_comment = f"현재 주가는 **{trend_status}**에 있으며, {macd_status}되는 무난한 흐름을 보이고 있습니다. 무리한 단타보다는 시장 추세를 따라가는 것이 좋습니다."
        
    st.info(f"🤖 **AI 한 줄 평:** {ai_comment}")
    st.divider()

    chart_config = {'displayModeBar': False, 'scrollZoom': False}

    # ★ 변경: ETF 탭을 AI 스캐너 바로 옆(6번째) 탭으로 이동 ★
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 차트 & 뉴스", "🏢 재무제표", "📈 시장 비교", "🧪 백테스트", "🚀 AI 스캐너", "🛒 ETF 탐색기"
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
                headers = {'User-Agent': 'Mozilla/5.0'}
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
        
        krx_df = load_krx_data()
        if 'Marcap' in krx_df.columns:
            top100 = krx_df.sort_values(by='Marcap', ascending=False).head(100).reset_index(drop=True)
            top100.index = top100.index + 1
            
            st.markdown("#### 🔥 오늘 시장을 주도하는 핫(Hot) 테마 Top 3")
            st.markdown("한국 시가총액 Top 100 종목 중, **시장에 돈이 가장 많이 몰린 테마 3개와 핵심 정보**를 요약합니다.")
            
            def get_sector_name(sector_str):
                s = str(sector_str)
                if s == 'nan' or not s: return '기타'
                if any(k in s for k in ['소프트웨어', '컴퓨터', '반도체', '전자부품', '통신', 'IT']): return '💻 IT/반도체'
                if any(k in s for k in ['자동차', '운송장비', '기계']): return '🚗 자동차/기계'
                if any(k in s for k in ['화학', '의약품', '의료', '생물', '바이오']): return '💊 바이오/헬스'
                if any(k in s for k in ['은행', '증권', '보험', '금융']): return '🏦 금융'
                if any(k in s for k in ['방송', '출판', '영화', '플랫폼', '엔터']): return '📱 플랫폼/콘텐츠'
                if any(k in s for k in ['음식료', '섬유', '의복', '유통', '소매']): return '🛒 소비재'
                if any(k in s for k in ['철강', '금속', '비금속', '건설']): return '🧱 철강/건설'
                if any(k in s for k in ['전기', '가스', '에너지']): return '⚡ 에너지/유틸리티'
                return f"🏭 {s}"
                
            top100_sec = top100.copy()
            
            if 'Sector' not in top100_sec.columns:
                try:
                    krx_desc = fdr.StockListing('KRX-DESC')
                    if 'Sector' in krx_desc.columns:
                        top100_sec = pd.merge(top100_sec, krx_desc[['Code', 'Sector']], on='Code', how='left')
                    else:
                        top100_sec['Sector'] = '기타'
                except:
                    top100_sec['Sector'] = '기타'
            
            top100_sec['섹터명'] = top100_sec['Sector'].apply(get_sector_name)
            
            sec_counts = top100_sec['섹터명'].value_counts()
            valid_sectors = sec_counts[sec_counts >= 3].index
            valid_top100 = top100_sec[top100_sec['섹터명'].isin(valid_sectors)]
            valid_top100 = valid_top100[valid_top100['섹터명'] != '기타']
            
            sec_stats = []
            for sec_name, group in valid_top100.groupby('섹터명'):
                if len(group) >= 3: 
                    mean_change = group['ChagesRatio'].mean()
                    total_marcap = group['Marcap'].sum() / 1_000_000_000_000 
                    up_cnt = len(group[group['ChagesRatio'] > 0])
                    down_cnt = len(group[group['ChagesRatio'] < 0])
                    flat_cnt = len(group[group['ChagesRatio'] == 0])
                    
                    sec_stats.append({
                        '섹터명': sec_name,
                        '평균등락률': mean_change,
                        '총시총': total_marcap,
                        '상승': up_cnt,
                        '하락': down_cnt,
                        '보합': flat_cnt,
                        '종목데이터': group
                    })
            
            if sec_stats:
                sec_df = pd.DataFrame(sec_stats).sort_values('평균등락률', ascending=False)
                hot_sectors = sec_df[sec_df['평균등락률'] > 0]
                
                if not hot_sectors.empty:
                    top3_sectors = hot_sectors.head(3)
                    cols = st.columns(3)
                    medals = ["🥇 1위", "🥈 2위", "🥉 3위"]
                    
                    for i, (idx, row) in enumerate(top3_sectors.iterrows()):
                        with cols[i]:
                            with st.container(border=True):
                                st.markdown(f"<h4 style='text-align: center; margin-bottom: 0px;'>{medals[i]} {row['섹터명']}</h4>", unsafe_allow_html=True)
                                color = "#ff4b4b" if row['평균등락률'] > 0 else "#00b4d8"
                                st.markdown(f"<h2 style='text-align: center; color: {color}; margin-top: 5px; margin-bottom: 5px;'>{row['평균등락률']:+.2f}%</h2>", unsafe_allow_html=True)
                                st.markdown(f"<div style='text-align: center; font-size: 0.9em; color: #aaaaaa;'>"
                                            f"테마 체급: <b>{row['총시총']:,.0f}조 원</b><br>"
                                            f"🔴 상승 <b>{row['상승']}</b> | ➖ 보합 <b>{row['보합']}</b> | 🔵 하락 <b>{row['하락']}</b>"
                                            f"</div>", unsafe_allow_html=True)
                                st.divider()
                                st.caption("🚀 주도주 Top 5")
                                
                                group_df = row['종목데이터'].sort_values('ChagesRatio', ascending=False)
                                stock_md = ""
                                for _, s_row in group_df.head(5).iterrows():
                                    s_name = s_row['Name']
                                    s_price = s_row['Close']
                                    s_change = s_row['ChagesRatio']
                                    s_icon = "🔺" if s_change > 0 else "🔻" if s_change < 0 else "➖"
                                    s_color = "#ff4b4b" if s_change > 0 else "#00b4d8" if s_change < 0 else "gray"
                                    sign = "+" if s_change > 0 else ""
                                    
                                    stock_md += f"<div style='display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 0.95em;'>" \
                                                f"<span><b>{s_name}</b></span>" \
                                                f"<span>₩{int(s_price):,} <span style='color:{s_color}; font-weight:bold;'>({s_icon} {sign}{s_change:.2f}%)</span></span>" \
                                                f"</div>"
                                st.markdown(stock_md, unsafe_allow_html=True)
                else:
                    st.info("📉 오늘 시장 전체가 하락장이라 뚜렷한 상승 주도 테마가 없습니다.")
            else:
                st.info("📊 데이터를 불러오는 중이거나 분석 가능한 테마가 부족합니다.")
            
            st.divider()
            
            st.markdown("#### 🤖 내일의 급등주 AI 스캐너")
            if st.button("🔍 상위 100종목 AI 스캔 시작 (약 15~20초 소요)", type="primary", use_container_width=True):
                my_bar = st.progress(0, text="AI가 데이터를 분석 중입니다...")
                ai_results = []
                
                for i, row in top100.iterrows():
                    code, name, market = row['Code'], row['Name'], row['Market']
                    t_code = f"{code}{'.KQ' if 'KOSDAQ' in str(market).upper() else '.KS'}"
                    
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
            st.warning("한국거래소(KRX) 데이터를 불러올 수 없습니다.")

    # ★ ETF 탐색기를 AI 스캐너 탭 바로 옆으로 이동 ★
    with tab6:
        st.subheader("🛒 한국 상장 인기 ETF 탐색기")
        st.markdown("다양한 테마와 지수를 추종하는 ETF 목록과 개별 상승/하락 트렌드를 확인하세요.")
        
        etf_df = get_etf_list()
        
        if not etf_df.empty:
            etf_options = [f"{row['Name']} ({row['Symbol']})" for _, row in etf_df.iterrows()]
            
            selected_etf = st.selectbox(
                "💡 분석할 ETF를 검색하거나 선택하세요:", 
                options=etf_options,
                index=None,
                placeholder="클릭하여 ETF를 선택하세요 (예: TIGER 미국S&P500)"
            )
            
            if selected_etf:
                e_name = selected_etf.split(" (")[0]
                e_code = selected_etf.split(" (")[1].replace(")", "")
                
                c_cols1, c_cols2 = st.columns([1, 1.5])
                
                with c_cols1:
                    st.markdown(f"#### 🎁 [{e_name}] 구성 종목 Top 10")
                    with st.spinner("구성 종목을 불러오는 중..."):
                        const_list = get_kr_etf_constituents(e_code)
                        if const_list:
                            clean_list = [c for c in const_list if "현금" not in c and "예금" not in c]
                            for idx, c in enumerate(clean_list[:10]):
                                st.markdown(f"**{idx+1}.** {c}")
                        else:
                            st.info("⚠️ 스트림릿(해외 무료 서버) 환경에서는 국내 포털(네이버/KRX)의 봇 보안 차단으로 인해 구성 종목을 실시간으로 가져올 수 없습니다. 우측 시세 트렌드를 참고해 주세요.")
                            
                with c_cols2:
                    st.markdown("#### 📊 최근 1년 수익률 추이")
                    with st.spinner("차트를 불러오는 중..."):
                        e_hist = yf.Ticker(f"{e_code}.KS").history(period="1y")
                        if not e_hist.empty:
                            e_hist = e_hist.dropna(subset=['Close'])
                            fig_e = go.Figure()
                            fig_e.add_trace(go.Scatter(x=e_hist.index.strftime('%Y-%m-%d'), y=e_hist['Close'], fill='tozeroy', line=dict(color='#ab47bc')))
                            fig_e.update_layout(template='plotly_dark', height=350, margin=dict(l=0, r=0, t=10, b=0))
                            fig_e.update_xaxes(fixedrange=True)
                            fig_e.update_yaxes(fixedrange=True)
                            st.plotly_chart(fig_e, use_container_width=True, config={'displayModeBar': False})
                        else:
                            st.warning("야후 파이낸스에서 차트 데이터를 불러올 수 없습니다.")
            
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

selected_stock = st.selectbox(
    label="🔍 종목 검색",
    options=stock_options,
    index=None, 
    placeholder="🔍 종목명 검색 (엔터 오류 방지를 위해 마우스 클릭을 권장합니다!)",
    label_visibility="collapsed"
)

# ★ 아무 종목도 검색하지 않았을 때의 메인 홈(Home) 화면 ★
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

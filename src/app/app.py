# src/app/app.py
import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine import kau_engine as engine

# ---------------------------------------------------------
# SECTION - Init
# ---------------------------------------------------------
# 페이지 설정
st.set_page_config(
    layout="wide",
    # page_title="동서발전 탄소배출권 시뮬레이터"
)
# st.title("K-ETS AI Simulator")  # 메인 타이틀

# 글로벌 스타일
st.markdown(
    """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"], [class*="st-"], p, div, span, h1, h2, h3, h4, h5, h6, a, li, ul, button, input, select, textarea {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

/* Container */
.stApp .block-container {
    padding: 2rem !important;
}

/* Background */
.stApp {
    background-color: #f1f5f9;
}

/* Custom Cards */
.kpi-card {
    background-color: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 16px;
    border: 1px solid #e2e8f0;
}
.kpi-card-blue { border-top: 4px solid #3b82f6; }
.kpi-card-lightblue { border-top: 4px solid #60a5fa; }
.kpi-card-green { border-top: 4px solid #10b981; }
.kpi-card-purple { border-top: 4px solid #8b5cf6; }
.kpi-card-red { border-top: 4px solid #ef4444; }
.kpi-card-orange { border-top: 4px solid #f59e0b; }

.kpi-title {
    font-size: 14px;
    color: #64748b;
    margin-bottom: 8px;
    font-weight: 600;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}
.kpi-unit {
    font-size: 14px;
    color: #64748b;
    font-weight: normal;
}
.kpi-delta {
    font-size: 13px;
    font-weight: 600;
}
.kpi-delta.positive { color: #10b981; }
.kpi-delta.negative { color: #ef4444; }

/* Main chart container */
.chart-container {
    background-color: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    border: 1px solid #e2e8f0;
}

/* Hide Sidebar Collapse Button */
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Prevent Horizontal Scroll in Sidebar */
section[data-testid="stSidebar"] {
    overflow-x: hidden !important;
    background-color: #172036 !important;
}
section[data-testid="stSidebar"] > div {
    overflow-x: hidden !important;
}
section[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

/* Sidebar Radio Buttons (Menu) */
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 16px 20px !important;
    border-radius: 0px !important;
    margin: 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.2s ease;
    border-left: 4px solid transparent;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: rgba(255, 255, 255, 0.05) !important;
}

/* Hide Radio Circle */
[data-testid="stSidebar"] div[role="radiogroup"] div[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Selected Menu Item */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background-color: #2e3a59 !important;
    border-left: 4px solid #38bdf8 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #38bdf8 !important;
    font-weight: 700 !important;
}

/* Hide Streamlit Header (Deploy button, Main Menu) */
[data-testid="stHeader"] {
    display: none !important;
}

/* Reduce top padding since header is gone */
.stApp .block-container {
    padding-top: 2rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

X_cols = engine.get_x_cols()

# 모델 로드 및 API 조회
with st.spinner("AI 모델 및 API 데이터 로딩 중..."):
    df = engine.load_and_prep_data()
    if df.empty:
        st.error(
            "데이터 로딩 실패 (API 연동 에러 또는 빈 데이터). 캐시를 지우고 다시 시도해주세요."
        )
        st.stop()
    model = engine.get_trained_model(df, X_cols)

# 데이터 기간 표시 (최근 3개월을 검증 데이터셋으로 분리)
today_date = pd.Timestamp("today")
split_date = today_date - pd.DateOffset(months=3)  # 기준 일자
train_start = df["일자"].min().strftime("%Y-%m-%d")  # 학습 데이터셋 시작일
train_end = split_date.strftime("%Y-%m-%d")  # 학습 데이터셋 종료일
test_start = (split_date + pd.Timedelta(days=1)).strftime(  # 검증 데이터셋 시작일
    "%Y-%m-%d"
)
test_end = today_date.strftime("%Y-%m-%d")  # 검증 데이터셋 종료일

st.markdown(
    f"**데이터셋 학습 기간**: `{train_start} ~ {train_end}` &nbsp; | &nbsp; **검증 기간**: `{test_start} ~ {test_end}`"
)

if st.button(
    "최신 데이터 전체 재학습",
    help="외부 API에서 2021년부터 오늘까지의 최신 데이터를 가져와 AI 모델을 처음부터 다시 학습합니다.",
):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
st.markdown("---")

# 최신 데이터 기반 변수들
latest_data_raw = df.iloc[[-1]][X_cols].astype(float)  # 학습데이터의 가장 마지막 행
current_price = df["종가"].iloc[-1]  # 학습데이터의 마지막 행의 종가
prev_price = df["종가"].iloc[-2]  # 학습데이터의 마지막 행의 전일 종가
current_vol = df["거래량"].iloc[-1]  # 학습데이터의 마지막 행의 거래량
current_ma5 = df["rolling_mean_5"].iloc[-1]  # 학습데이터의 마지막 행의 5일 이동평균
mae, rmse, r2, mape, test_count = engine.evaluate_model(model, df, X_cols)
# !SECTION - Init

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom: 30px; margin-top: 10px;">
            <div style="color:white; font-size:18px; font-weight:bold; letter-spacing:-0.5px;">한국동서발전</div>
        </div>
        <hr style="border-color: #334155; margin-bottom: 20px;">
        """,
        unsafe_allow_html=True,
    )
    selected_tab = st.radio(
        "메뉴",
        [
            "통합 대시보드",
            "가격 예측 시뮬레이터",
            # "사업장 · 배출량 관리",
            # "시장 수급 전망",
            "리포트 자동 생성",
        ],
        label_visibility="collapsed",
    )

if "통합 대시보드" in selected_tab:
    main_col = st.container()
    nav_col = st.sidebar
else:
    nav_col, main_col = st.columns([1, 9])

with main_col:

    # ---------------------------------------------------------
    # SECTION - Sidebar
    # ---------------------------------------------------------
    # 세션 관리
    if "fuel_adj" not in st.session_state:
        st.session_state.fuel_adj = 0.0
    if "rate_adj" not in st.session_state:
        st.session_state.rate_adj = 3.5
    if "re_adj" not in st.session_state:
        st.session_state.re_adj = 12.5
    if "exchange_adj" not in st.session_state:
        st.session_state.exchange_adj = 0.0
    if "eua_adj" not in st.session_state:
        st.session_state.eua_adj = 0.0
    if "ng_adj" not in st.session_state:
        st.session_state.ng_adj = 0.0
    if "coal_adj" not in st.session_state:
        st.session_state.coal_adj = 0.0
    if "smp_adj" not in st.session_state:
        st.session_state.smp_adj = 0.0
    if "kospi_adj" not in st.session_state:
        st.session_state.kospi_adj = 0.0
    if "bsi_adj" not in st.session_state:
        st.session_state.bsi_adj = 0.0
    if "temp_adj" not in st.session_state:
        st.session_state.temp_adj = 0.0
    if "op_rate" not in st.session_state:
        st.session_state.op_rate = 85
    if "lng_ratio" not in st.session_state:
        st.session_state.lng_ratio = 20
    if "current_balance" not in st.session_state:
        st.session_state.current_balance = 1800000

    if "simulation_started" not in st.session_state:
        st.session_state.simulation_started = False
    if "llm_tags" not in st.session_state:
        st.session_state.llm_tags = [
            "금리 인상",
            "전기도매요금 인상",
            "EU의 탄소배출권 가격 상승",
        ]
    if "llm_selected_tags" not in st.session_state:
        st.session_state.llm_selected_tags = []
    if "run_llm_simulation" not in st.session_state:
        st.session_state.run_llm_simulation = False

    if "가격 예측 시뮬레이터" in selected_tab:
        nav_col.header("기준 변수")
        # nav_col.warning(f"현재 AI 예측 오차율(MAE): {mae:.4f}")
        # nav_col.caption(f"검증 기간: 최신 데이터 {test_count}일분 기준")

        past_idx = -22 if len(df) > 22 else 0
        past_data = df.iloc[past_idx][X_cols].astype(float)

        def render_sidebar_gauge(label, col_name, unit, format_str="{:,.0f}"):
            if not st.session_state.simulation_started:
                cur_val = 0.0
                pct_change = 0.0
                val_str = "0" if format_str == "{:,.0f}" else "0.00"
                pct_str = "0.0%"
                clamped_pct = 0.0
            else:
                cur_val = float(latest_data_raw[col_name].iloc[0])
                past_val = float(past_data[col_name])

                pct_change = 0.0
                if past_val != 0 and pd.notnull(past_val):
                    pct_change = ((cur_val - past_val) / past_val) * 100.0

                clamped_pct = max(-50.0, min(50.0, pct_change))
                val_str = format_str.format(cur_val)
                pct_str = f"{pct_change:+.1f}%"

            nav_col.markdown(f"**{label}**")
            col1, col2 = nav_col.columns([6, 4])
            with col1:
                st.markdown(
                    f"<h3 style='margin:0; padding:0;'>{val_str} <span style='font-size:0.5em; font-weight:normal;'>{unit}</span></h3>",
                    unsafe_allow_html=True,
                )
            with col2:
                color = (
                    "#ef4444"
                    if pct_change > 0
                    else "#3b82f6" if pct_change < 0 else "gray"
                )
                st.markdown(
                    f"<div style='text-align:right; color:{color}; font-weight:bold; background-color:#f1f5f9; padding:2px 5px; border-radius:5px; margin-top:5px;'>{pct_str}</div>",
                    unsafe_allow_html=True,
                )

            nav_col.slider(
                "_",
                -50.0,
                50.0,
                float(clamped_pct),
                key=f"gauge_{col_name}",
                disabled=True,
                label_visibility="collapsed",
            )
            nav_col.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        render_sidebar_gauge("WTI유가", "WTI_유가", "원/배럴")
        render_sidebar_gauge("원/달러 환율", "환율", "원(KRW)")
        render_sidebar_gauge("EUA", "EUA", "$/tCO2e", "{:,.2f}")
        render_sidebar_gauge("천연가스", "천연가스", "$/MMBtu", "{:,.2f}")
        render_sidebar_gauge("글로벌 석탄", "석탄", "$/mt")
        render_sidebar_gauge("SMP(계통한계가격)", "SMP", "원/kWh")
        render_sidebar_gauge("KOSPI", "KOSPI", "Pt")
        render_sidebar_gauge("기준금리", "기준금리", "%", "{:,.2f}")
        render_sidebar_gauge("업황전망BSI", "업황전망BSI", "Pt")
        render_sidebar_gauge("평균기온", "평균기온", "℃", "{:,.1f}")
        render_sidebar_gauge("제조업 가동률", "제조업가동률", "%", "{:,.1f}")

        nav_col.markdown("##### 보고서 학습")
        uploaded_files = nav_col.file_uploader(
            "보고서 PDF 업로드",
            type=["pdf"],
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        if nav_col.button("보고서 학습", use_container_width=True):
            if uploaded_files:
                with st.spinner("PDF에서 텍스트를 추출하여 학습 중입니다..."):
                    try:
                        import fitz  # PyMuPDF

                        text = ""
                        for file in uploaded_files:
                            doc = fitz.open(stream=file.read(), filetype="pdf")
                            text += f"\n\n--- [보고서: {file.name}] ---\n"
                            for page in doc:
                                text += page.get_text()

                        # 여러 보고서 분량을 고려해 최대 15,000자로 제한
                        st.session_state.learned_report_text = text[:15000]

                        # 요약 기능 호출
                        import importlib
                        from engine import llm_engine as _llm_engine

                        importlib.reload(_llm_engine)
                        from engine.llm_engine import LLMEngine

                        llm = LLMEngine(
                            provider=st.session_state.get("llm_provider", "nemotron")
                        )
                        summary = llm.summarize_report_text(
                            st.session_state.learned_report_text
                        )
                        st.session_state.learned_report_summary = summary

                        nav_col.success(
                            "학습 완료! 시뮬레이션 시 보고서 내용이 융합됩니다."
                        )
                    except Exception as e:
                        nav_col.error(f"오류 발생: {str(e)}")
            else:
                nav_col.warning("먼저 PDF 파일을 업로드해주세요.")

        if st.session_state.get("learned_report_summary"):
            with nav_col.expander("📝 학습된 보고서 요약 보기"):
                st.markdown(st.session_state.learned_report_summary)

        nav_col.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

        def add_llm_tag():
            new_tag = st.session_state.new_llm_tag.strip()
            if new_tag and new_tag not in st.session_state.llm_tags:
                st.session_state.llm_tags.append(new_tag)
                if new_tag not in st.session_state.llm_selected_tags_widget:
                    st.session_state.llm_selected_tags_widget.append(new_tag)
            st.session_state.new_llm_tag = ""

        nav_col.markdown("##### LLM 이벤트 시나리오")

        selected_llm_provider = nav_col.radio(
            "사용할 AI 모델", ["Nemotron", "Gemini"], index=0, horizontal=True
        )
        st.session_state.llm_provider = selected_llm_provider.lower()

        nav_col.text_input(
            "새로운 이벤트 태그 추가 (엔터)",
            key="new_llm_tag",
            on_change=add_llm_tag,
            placeholder="예: 금리 인하",
        )

        if "llm_selected_tags_widget" not in st.session_state:
            st.session_state.llm_selected_tags_widget = []

        nav_col.multiselect(
            "적용할 이벤트 태그",
            options=st.session_state.llm_tags,
            key="llm_selected_tags_widget",
        )

        llm_btn_clicked = nav_col.button(
            "LLM 시뮬레이션 실행", use_container_width=True
        )
        if llm_btn_clicked:
            if not st.session_state.llm_selected_tags_widget:
                nav_col.warning("태그를 하나 이상 선택해주세요.")
            else:
                st.session_state.simulation_started = True

    # 시뮬레이터 변수 조작 기능이 비활성화되었으므로 기본값 0으로 고정
    fuel_price_adj = 0.0
    exchange_rate_adj = 0.0
    eua_price_adj = 0.0
    ng_price_adj = 0.0
    coal_adj = 0.0
    smp_adj = 0.0
    kospi_adj = 0.0
    interest_rate_adj = float(latest_data_raw["기준금리"].iloc[0])
    bsi_adj = 0.0
    temp_adj = 0.0
    re_ratio_adj = float(latest_data_raw["RE_비중"].iloc[0])
    op_rate = float(latest_data_raw["제조업가동률"].iloc[0])
    lng_ratio = 20
    current_balance = 1800000
    # !SECTION - Sidebar

    # ---------------------------------------------------------
    # SECTION - 연산 로직
    # ---------------------------------------------------------
    # 시나리오 데이터 생성 및 외부/내부 변수 주입
    base_scenario = latest_data_raw.copy()
    base_scenario["WTI_유가"] = base_scenario["WTI_유가"] * (1 + (fuel_price_adj / 100))
    base_scenario["환율"] = base_scenario["환율"] * (1 + (exchange_rate_adj / 100))
    base_scenario["EUA"] = base_scenario["EUA"] * (1 + (eua_price_adj / 100))
    base_scenario["천연가스"] = base_scenario["천연가스"] * (1 + (ng_price_adj / 100))
    base_scenario["석탄"] = base_scenario["석탄"] * (1 + (coal_adj / 100))
    base_scenario["SMP"] = base_scenario["SMP"] * (1 + (smp_adj / 100))
    base_scenario["KOSPI"] = base_scenario["KOSPI"] * (1 + (kospi_adj / 100))
    base_scenario["업황전망BSI"] = base_scenario["업황전망BSI"] * (1 + (bsi_adj / 100))
    new_temp = base_scenario["평균기온"] + temp_adj
    base_scenario["평균기온"] = new_temp
    base_scenario["CDD_냉방수요"] = new_temp.apply(lambda x: max(0, x - 24))
    base_scenario["HDD_난방수요"] = new_temp.apply(lambda x: max(0, 18 - x))
    base_scenario["에너지_도입단가"] = base_scenario["환율"] * base_scenario["WTI_유가"]
    base_scenario["기준금리"] = interest_rate_adj
    base_scenario["RE_비중"] = re_ratio_adj
    base_scenario["제조업가동률"] = op_rate
    base_scenario["lng_ratio"] = lng_ratio

    # 가격 예측
    next_day_pred, _ = engine.predict_kau(model, current_price, base_scenario)
    base_pred_price, _ = engine.predict_kau(model, current_price, latest_data_raw)

    linear_adj_factor = 1.0
    linear_adj_factor += (
        fuel_price_adj / 100
    ) * 0.03  # WTI 1% 상승 시 KAU 0.03% 추가 상승
    linear_adj_factor += (
        exchange_rate_adj / 100
    ) * 0.05  # 환율 1% 상승 시 KAU 0.05% 추가 상승
    linear_adj_factor += (
        eua_price_adj / 100
    ) * 0.10  # EUA 1% 상승 시 KAU 0.10% 추가 상승
    linear_adj_factor += (ng_price_adj / 100) * 0.04
    linear_adj_factor += (coal_adj / 100) * 0.03
    linear_adj_factor += (smp_adj / 100) * 0.02
    linear_adj_factor += (kospi_adj / 100) * 0.02
    linear_adj_factor += (bsi_adj / 100) * 0.02
    linear_adj_factor += (temp_adj) * 0.005  # 기온 1도 상승/하락에 따른 변화
    linear_adj_factor += (
        (op_rate - 85) / 100 * 0.05
    )  # 제조업가동률 기준(85) 대비 변화량 적용
    linear_adj_factor += (
        (lng_ratio - 20) / 100 * 0.03
    )  # LNG비중 기준(20) 대비 변화량 적용

    next_day_pred = next_day_pred * linear_adj_factor

    # 예측 생성
    temp_hist = df.tail(11).copy()
    if len(temp_hist) > 0:
        hist_preds = model.predict(temp_hist[X_cols])
        temp_hist["AI 예상 단가_raw"] = temp_hist["lag_1"] * (1 + hist_preds)
        temp_hist["전월(일) 예측값_raw"] = temp_hist["AI 예상 단가_raw"].shift(1)
        prev_pred = temp_hist["전월(일) 예측값_raw"].iloc[-1]
        if pd.isna(prev_pred):
            prev_pred = current_price
    else:
        prev_pred = current_price

    # 전략 추천 산출
    strategy, reason = engine.get_recommendation(
        current_price, next_day_pred, current_ma5
    )

    # 배출량/예산 산출
    base_emissions, _ = engine.calculate_emissions(85, 20)
    base_shortage = max(0, base_emissions - current_balance)
    base_budget = base_shortage * base_pred_price

    total_emissions, _ = engine.calculate_emissions(op_rate, lng_ratio)
    shortage = max(0, total_emissions - current_balance)
    required_budget = shortage * next_day_pred
    financial_gap = required_budget - base_budget
    # !SECTION - 연산 로직

    # ---------------------------------------------------------
    # SECTION - 통합 대시보드
    # ---------------------------------------------------------
    if "통합 대시보드" in selected_tab:
        st.subheader("통합 대시보드")
        last_date = df["일자"].max().strftime("%Y-%m-%d")

        # 1. 상단 6개의 주표 지표(KPI)
        st.markdown("##### 주요지표")

        # 어제 대비 가격 변동 계산
        yesterday_price = df["종가"].iloc[-2] if len(df) > 1 else current_price
        pct_change = (
            ((current_price - yesterday_price) / yesterday_price * 100)
            if yesterday_price
            else 0
        )

        current_volume = df["거래량"].iloc[-1]
        vol_yesterday = df["거래량"].iloc[-2] if len(df) > 1 else current_volume
        vol_change = (
            ((current_volume - vol_yesterday) / vol_yesterday * 100)
            if vol_yesterday
            else 0
        )

        def get_kpi_html(
            title, value, unit, delta_text, color_class, delta_is_positive=True
        ):
            delta_class = "positive" if delta_is_positive else "negative"
            arrow = "▲" if delta_is_positive else "▼"
            if not delta_text:
                delta_html = f"<div class='kpi-delta' style='color:#cbd5e1;'>-</div>"
            else:
                delta_html = (
                    f"<div class='kpi-delta {delta_class}'>{arrow} {delta_text}</div>"
                )

            return f"""
            <div class="kpi-card {color_class}">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value} <span class="kpi-unit">{unit}</span></div>
                {delta_html}
            </div>
            """

        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

        with kpi1:
            st.markdown(
                get_kpi_html(
                    "현재 KAU 가격",
                    f"{current_price:,.0f}",
                    "원",
                    f"{abs(pct_change):.1f}% 전일대비",
                    "kpi-card-blue",
                    pct_change >= 0,
                ),
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                get_kpi_html(
                    "거래량",
                    f"{current_volume:,.0f}",
                    "tCO₂e",
                    f"{abs(vol_change):.1f}% 전일대비",
                    "kpi-card-lightblue",
                    vol_change >= 0,
                ),
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                get_kpi_html(
                    "확보 배출권량",
                    f"{current_balance:,.0f}",
                    "tCO₂e",
                    "",
                    "kpi-card-green",
                    True,
                ),
                unsafe_allow_html=True,
            )
        with kpi4:
            st.markdown(
                get_kpi_html(
                    "예상 배출량",
                    f"{base_emissions:,.0f}",
                    "tCO₂e",
                    "",
                    "kpi-card-purple",
                    True,
                ),
                unsafe_allow_html=True,
            )
        with kpi5:
            st.markdown(
                get_kpi_html(
                    "예상 부족량",
                    f"{base_shortage:,.0f}",
                    "tCO₂e",
                    "",
                    "kpi-card-red",
                    True,
                ),
                unsafe_allow_html=True,
            )
        with kpi6:
            st.markdown(
                get_kpi_html(
                    "예상 구매비용",
                    f"{base_budget/1e8:,.0f}",
                    "억원",
                    "",
                    "kpi-card-orange",
                    True,
                ),
                unsafe_allow_html=True,
            )

        # 2. KPI 아래 그래프 섹션
        with st.container(border=True):
            st.markdown("##### KAU 가격 및 거래량 추이")

            filter_col1, filter_col2, filter_col3 = st.columns([6, 2, 2])
        with filter_col2:
            period_filter = st.radio(
                "기간 필터",
                ["1개월", "3개월", "전체"],
                horizontal=True,
                label_visibility="collapsed",
            )
        with filter_col3:
            freq_filter = st.radio(
                "주기 필터",
                ["일별", "주별", "월별"],
                horizontal=True,
                label_visibility="collapsed",
            )

        # 기간 필터 적용
        chart_df = df.copy()
        if period_filter == "1개월":
            chart_df = chart_df.tail(30)
        elif period_filter == "3개월":
            chart_df = chart_df.tail(90)

        # 주기 필터 적용
        if freq_filter == "주별":
            chart_df = (
                chart_df.resample("W-MON", on="일자")
                .agg({"종가": "last", "거래량": "sum"})
                .reset_index()
            )
        elif freq_filter == "월별":
            chart_df = (
                chart_df.resample("M", on="일자")
                .agg({"종가": "last", "거래량": "sum"})
                .reset_index()
            )

        from plotly.subplots import make_subplots

        fig_main = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
        )
        fig_main.add_trace(
            go.Scatter(
                x=chart_df["일자"],
                y=chart_df["종가"],
                name="KAU 가격",
                line=dict(color="#3b82f6"),
            ),
            row=1,
            col=1,
        )
        fig_main.add_trace(
            go.Bar(
                x=chart_df["일자"],
                y=chart_df["거래량"],
                name="KAU 거래량",
                marker_color="#64748b",
            ),
            row=2,
            col=1,
        )
        fig_main.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # 테이블
        st.markdown("##### 가격 및 예측 데이터")

        # 전역 변수인 temp_hist 활용

        table_src = temp_hist.tail(10).copy()
        table_src["예측 기준일"] = table_src["일자"].dt.strftime("%Y-%m-%d")
        table_src["현재 KAU 가격"] = table_src["종가"].apply(lambda x: f"{x:,.0f}")
        table_src["AI 예상 단가"] = table_src["AI 예상 단가_raw"].apply(
            lambda x: f"{x:,.0f}"
        )
        table_src["전월 예측값"] = table_src["전월(일) 예측값_raw"].apply(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "-"
        )
        table_src["거래량"] = table_src["거래량"].apply(lambda x: f"{x:,.0f}")

        # 예측 증감률 (전일 예측 대비 현재 예측 증감률)
        table_src["예측 증감률_val"] = (
            (table_src["AI 예상 단가_raw"] - table_src["전월(일) 예측값_raw"])
            / table_src["전월(일) 예측값_raw"]
            * 100
        )
        table_src["예측 증감률"] = (
            table_src["예측 증감률_val"]
            .apply(lambda x: f"▲ {x:.1f}%" if x >= 0 else f"▼ {abs(x):.1f}%")
            .replace("nan%", "-")
        )

        # 예측 오차율 (실제가 대비 예측가 오차)
        table_src["예측 오차율_val"] = (
            abs(table_src["AI 예상 단가_raw"] - table_src["종가"])
            / table_src["종가"]
            * 100
        )
        table_src["예측 오차율"] = table_src["예측 오차율_val"].apply(
            lambda x: f"{x:.2f}%"
        )

        disp_cols = [
            "예측 기준일",
            "현재 KAU 가격",
            "AI 예상 단가",
            "전월 예측값",
            "거래량",
            "예측 증감률",
            "예측 오차율",
        ]
        st.dataframe(
            table_src[disp_cols].iloc[::-1], use_container_width=True, hide_index=True
        )

        # 수급 팩터 | 변수 영향 분석 | 배출권 확보 현황
        b_col1, b_col2, b_col3 = st.columns(3)

        # 수급 팩터
        with b_col1:
            with st.container(border=True):
                st.markdown("##### 수급 팩터")
                st.info("추후 개발 예정")

        # 변수 영향 분석
        with b_col2:
            with st.container(border=True):
                st.markdown("##### 변수 영향 분석")
                importance_weights = model.feature_importances_
                feature_names = X_cols

                # 한글 매핑 딕셔너리
                feature_name_map = {
                    "lag_1": "전일 가격",
                    "lag_2": "2일전 가격",
                    "lag_3": "3일전 가격",
                    "rolling_mean_5": "5일 이동평균",
                    "rolling_mean_10": "10일 이동평균",
                    "volatility": "가격 변동성",
                    "WTI_유가": "국제 연료가격(WTI)",
                    "KOSPI": "코스피 지수",
                    "CDD_냉방수요": "냉방수요(CDD)",
                    "HDD_난방수요": "난방수요(HDD)",
                    "SMP": "계통한계가격(SMP)",
                    "EUA": "유럽 배출권(EUA)",
                    "lng_ratio": "LNG 발전 비중",
                    "RE_비중": "신재생에너지 비중",
                    "정산기한_접근도": "정산기한 접근도",
                    "업황전망BSI": "산업 활동 지수(BSI)",
                    "제조업가동률": "제조업 가동률",
                    "환율": "환율",
                    "거래량": "거래량",
                    "천연가스": "천연가스",
                    "석탄": "글로벌 석탄",
                    "기준금리": "기준금리",
                    "에너지_도입단가": "에너지 도입단가",
                }

                importance_df = pd.DataFrame(
                    {"Feature": feature_names, "Importance": importance_weights}
                )
                importance_df["Feature"] = importance_df["Feature"].apply(
                    lambda x: feature_name_map.get(x, x)
                )

                importance_df = (
                    importance_df.groupby("Feature", as_index=False)["Importance"]
                    .sum()
                    .sort_values(by="Importance", ascending=False)
                    .head(8)
                )

                importance_df["Imp_Text"] = importance_df["Importance"].apply(
                    lambda x: f"{x:.4f}"
                )

                fig_imp = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Blues",
                    text="Imp_Text",
                )
                fig_imp.update_traces(textposition="outside")
                fig_imp.update_layout(
                    yaxis={"categoryorder": "total ascending", "title": ""},
                    xaxis={"visible": False},
                    coloraxis_showscale=False,
                    height=300,
                    margin=dict(l=10, r=40, t=10, b=10),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_imp, use_container_width=True)

        # 배출권 확보 현황
        with b_col3:
            with st.container(border=True):
                st.markdown("##### 배출권 확보 현황")
                st.info("추후 개발 예정")

            # with st.container(border=True):
            #     st.markdown("##### 배출권 확보 현황")
            #     secure_ratio = (
            #         (current_balance / total_emissions * 100) if total_emissions > 0 else 0
            #     )
            #     fig_gauge = go.Figure(
            #         go.Indicator(
            #             mode="gauge+number",
            #             value=secure_ratio,
            #             number={"suffix": "%", "font": {"size": 36}},
            #             title={"text": "확보율", "font": {"size": 16}},
            #             gauge={
            #                 "axis": {"range": [0, 100], "tickwidth": 1},
            #                 "bar": {"color": "#3b82f6"},
            #                 "steps": [
            #                     {"range": [0, 50], "color": "#f87171"},
            #                     {"range": [50, 80], "color": "#fbbf24"},
            #                     {"range": [80, 100], "color": "#e2e8f0"},
            #                 ],
            #             },
            #         )
            #     )
            #     fig_gauge.update_layout(
            #         height=300,
            #         margin=dict(l=20, r=20, t=40, b=10),
            #         paper_bgcolor="rgba(0,0,0,0)",
            #         plot_bgcolor="rgba(0,0,0,0)",
            #     )
            #     st.plotly_chart(fig_gauge, use_container_width=True)

        # 보고서 테스트용
        # st.markdown("##### AI 시장 동향 브리핑 (Hybrid XAI)")

        # # SHAP(정량적 XAI) 데이터 추출
        # shap_contributions = engine.get_feature_contributions(
        #     model, latest_data_raw, X_cols
        # )

        # if "dashboard_llm_briefing" not in st.session_state:
        #     st.session_state.dashboard_llm_briefing = None

        # if st.button("브리핑 생성 / 갱신", key="btn_llm_briefing"):
        #     with st.spinner(
        #         "LLM이 정량적 수치(SHAP)를 바탕으로 시장 동향을 분석 중입니다..."
        #     ):
        #         try:
        #             from engine.llm_engine import LLMEngine

        #             llm = LLMEngine()

        #             # 모든 변수 동적 할당
        #             external_factors = {}
        #             for col in latest_data_raw.columns:
        #                 val = latest_data_raw[col].iloc[0]
        #                 external_factors[col] = (
        #                     float(val)
        #                     if isinstance(val, (int, float, np.integer, np.floating))
        #                     else str(val)
        #                 )

        #             context_snapshot = {
        #                 "Data_Date": (
        #                     str(latest_data_raw["일자"].iloc[0].date())
        #                     if "일자" in latest_data_raw.columns
        #                     else "2026-07-28"
        #                 ),
        #                 "KAU_Price_Analysis": {
        #                     "current_price": float(current_price),
        #                     "predicted_price": float(next_day_pred),
        #                     "expected_trend": (
        #                         "상승" if next_day_pred > current_price else "하락"
        #                     ),
        #                 },
        #                 "External_Factors": external_factors,
        #                 "Feature_Contributions_SHAP": shap_contributions,
        #                 "Recommendation_Engine": {"strategy": strategy, "reason": reason},
        #             }

        #             briefing_text = llm.analyze_market(context_snapshot)
        #             st.session_state.dashboard_llm_briefing = briefing_text
        #         except Exception as e:
        #             st.error(f"LLM 호출 실패: {str(e)}")

        # if st.session_state.dashboard_llm_briefing:
        #     st.info(st.session_state.dashboard_llm_briefing)
    # !SECTION - 통합 대시보드

    # ---------------------------------------------------------
    # SECTION - 가격 예측 시뮬레이터
    # ---------------------------------------------------------
    elif "가격 예측 시뮬레이터" in selected_tab:
        if not st.session_state.simulation_started:
            st.markdown("### 가격 예측 시뮬레이터")
            st.info("아래 버튼을 눌러 가격 예측 시뮬레이터를 실행해보세요.")

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                if st.button(
                    "시뮬레이션 실행하기", type="primary", use_container_width=True
                ):
                    import time

                    loading_overlay = """
                    <style>
                    .fullscreen-overlay {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100vw;
                        height: 100vh;
                        background-color: rgba(255, 255, 255, 0.4);
                        backdrop-filter: blur(8px);
                        -webkit-backdrop-filter: blur(8px);
                        z-index: 999999;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                    }
                    .loader {
                        border: 6px solid #e2e8f0;
                        border-top: 6px solid #3b82f6;
                        border-radius: 50%;
                        width: 60px;
                        height: 60px;
                        animation: spin 1s linear infinite;
                        margin-bottom: 24px;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    .overlay-text {
                        font-size: 1.25rem;
                        font-weight: 700;
                        color: #0f172a;
                        text-shadow: 0 2px 4px rgba(255,255,255,0.5);
                    }
                    </style>
                    <div class="fullscreen-overlay">
                        <div class="loader"></div>
                    </div>
                    """
                    placeholder = st.empty()
                    placeholder.markdown(loading_overlay, unsafe_allow_html=True)
                    time.sleep(10)
                    placeholder.empty()
                    st.session_state.simulation_started = True
                    st.rerun()
            st.stop()
        else:
            # -- LLM 적용 스위치 --
            llm_toggle = False
            llm_adj_rate = 0.0
            original_next_day_pred = next_day_pred

            if (
                "llm_parsed_result" in st.session_state
                and "error" not in st.session_state["llm_parsed_result"]
            ):
                llm_toggle = st.toggle("LLM 이벤트 시나리오 분석결과 반영", value=False)
                st.markdown("</div>", unsafe_allow_html=True)

                if llm_toggle:
                    llm_adj_rate = float(
                        st.session_state["llm_parsed_result"].get(
                            "adjustment_rate", 0.0
                        )
                    )
                    next_day_pred = next_day_pred * (1 + llm_adj_rate)
                    required_budget = shortage * next_day_pred
                    st.session_state["latest_return"] = llm_adj_rate

            # 1. 상단 6개의 KPI
            st.markdown("##### 주요지표")
            s_kpi1, s_kpi2, s_kpi3, s_kpi4, s_kpi5, s_kpi6 = st.columns(6)

        # 어제 대비 가격 변동 계산
        yesterday_price = df["종가"].iloc[-2] if len(df) > 1 else current_price
        pct_change = (
            ((current_price - yesterday_price) / yesterday_price * 100)
            if yesterday_price
            else 0
        )

        ai_pct_change = (
            ((next_day_pred - current_price) / current_price * 100)
            if current_price
            else 0
        )

        prev_pct_change = (
            ((prev_pred - current_price) / current_price * 100) if current_price else 0
        )

        def get_kpi_html(
            title, value, unit, delta_text, color_class, delta_is_positive=True
        ):
            delta_class = "positive" if delta_is_positive else "negative"
            arrow = "▲" if delta_is_positive else "▼"
            if not delta_text:
                delta_html = f"<div class='kpi-delta' style='color:#cbd5e1;'>-</div>"
            else:
                delta_html = (
                    f"<div class='kpi-delta {delta_class}'>{arrow} {delta_text}</div>"
                )

            return f"""
            <div class="kpi-card {color_class}">
                <div class="kpi-title">{title}</div>
                <div class="kpi-value">{value} <span class="kpi-unit">{unit}</span></div>
                {delta_html}
            </div>
            """

        with s_kpi1:
            st.markdown(
                get_kpi_html(
                    "현재 KAU 가격",
                    f"{current_price:,.0f}",
                    "원",
                    f"{abs(pct_change):.1f}% 전일대비",
                    "kpi-card-blue",
                    pct_change >= 0,
                ),
                unsafe_allow_html=True,
            )
        with s_kpi2:
            st.markdown(
                get_kpi_html(
                    "AI 예상 가격",
                    f"{next_day_pred:,.0f}",
                    "원",
                    f"{abs(ai_pct_change):.1f}% 전일대비",
                    "kpi-card-lightblue",
                    ai_pct_change >= 0,
                ),
                unsafe_allow_html=True,
            )
        with s_kpi3:
            st.markdown(
                get_kpi_html(
                    "전월 예측값",
                    f"{prev_pred:,.0f}",
                    "원",
                    f"{abs(prev_pct_change):.1f}% 전일대비",
                    "kpi-card-green",
                    prev_pct_change >= 0,
                ),
                unsafe_allow_html=True,
            )
        with s_kpi4:
            st.markdown(
                get_kpi_html(
                    "확보 배출권량",
                    f"{current_balance:,.0f}",
                    "tCO₂e",
                    "",
                    "kpi-card-purple",
                    True,
                ),
                unsafe_allow_html=True,
            )
        with s_kpi5:
            st.markdown(
                get_kpi_html(
                    "부족량", f"{shortage:,.0f}", "tCO₂e", "", "kpi-card-red", True
                ),
                unsafe_allow_html=True,
            )
        with s_kpi6:
            st.markdown(
                get_kpi_html(
                    "예상 구매비용",
                    f"{required_budget/1e8:,.0f}",
                    "억원",
                    "",
                    "kpi-card-orange",
                    True,
                ),
                unsafe_allow_html=True,
            )

        # 2. 시나리오별 가격 전망
        with st.container(border=True):
            st.markdown("##### 시나리오별 가격 전망 (원/tCO2e)")

            # --- 필터 영역 ---
            f_col1, f_col2, f_col3 = st.columns([6, 2, 2])
            with f_col2:
                sim_period_filter = st.radio(
                    "시뮬레이터 기간 필터",
                    ["1개월", "3개월", "전체"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="sim_period_filter",
                )
            with f_col3:
                sim_freq_filter = st.radio(
                    "시뮬레이터 주기 필터",
                    ["일별", "주별", "월별"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="sim_freq_filter",
                )

            sc_col1, sc_col2 = st.columns([7, 3])

            future_dates = pd.date_range(
                start=df["일자"].max() + pd.Timedelta(days=1), periods=30
            )

            def generate_random_walk(start_val, end_val, steps, vol, seed=None):
                if seed is not None:
                    np.random.seed(seed)
                t = np.linspace(0, 1, steps)
                W = np.random.standard_normal(size=steps)
                W = np.cumsum(W) * np.sqrt(1.0 / steps)
                W = W - W[0]
                bridge = W - t * W[-1]
                trend = np.linspace(start_val, end_val, steps)
                return trend + bridge * vol

            vol = current_price * 0.05
            base_pred_series = generate_random_walk(
                current_price, original_next_day_pred, len(future_dates), vol, seed=42
            )
            up_pred_series = generate_random_walk(
                current_price,
                next_day_pred * 1.15,
                len(future_dates),
                vol * 1.2,
                seed=43,
            )
            down_pred_series = generate_random_walk(
                current_price,
                next_day_pred * 0.85,
                len(future_dates),
                vol * 1.2,
                seed=44,
            )

            # 데이터프레임 구성 및 필터링
            past_df = df[["일자", "종가"]].copy()

            future_df = pd.DataFrame(
                {
                    "일자": future_dates,
                    "base_pred": base_pred_series,
                    "up_pred": up_pred_series,
                    "down_pred": down_pred_series,
                }
            )

            llm_pred_series_upper = None
            if llm_toggle:
                llm_pred_series_upper = st.session_state.get("llm_pred_series")
                if llm_pred_series_upper is not None:
                    future_df["llm_pred"] = llm_pred_series_upper

            # 기간 필터
            if sim_period_filter == "1개월":
                past_df = past_df.tail(30)
            elif sim_period_filter == "3개월":
                past_df = past_df.tail(90)

            # 주기 필터
            if sim_freq_filter == "주별":
                past_df = (
                    past_df.resample("W-MON", on="일자")
                    .agg({"종가": "last"})
                    .reset_index()
                )
                future_df = future_df.resample("W-MON", on="일자").last().reset_index()
            elif sim_freq_filter == "월별":
                past_df = (
                    past_df.resample("M", on="일자").agg({"종가": "last"}).reset_index()
                )
                future_df = future_df.resample("M", on="일자").last().reset_index()

            fig_scenario = go.Figure()

            fig_scenario.add_trace(
                go.Scatter(
                    x=past_df["일자"],
                    y=past_df["종가"],
                    name="과거 가격",
                    line=dict(color="#cbd5e1"),
                )
            )
            fig_scenario.add_trace(
                go.Scatter(
                    x=future_df["일자"],
                    y=future_df["base_pred"],
                    name="기준 시나리오 (XGBoost)",
                    line=dict(color="#3b82f6"),
                )
            )
            if "llm_pred" in future_df.columns:
                fig_scenario.add_trace(
                    go.Scatter(
                        x=future_df["일자"],
                        y=future_df["llm_pred"],
                        name="LLM 이벤트 반영 시나리오 (After)",
                        line=dict(color="#8b5cf6", width=3, dash="dash"),
                    )
                )
            fig_scenario.add_trace(
                go.Scatter(
                    x=future_df["일자"],
                    y=future_df["up_pred"],
                    name="상승 시나리오",
                    line=dict(color="#ef4444"),
                )
            )
            fig_scenario.add_trace(
                go.Scatter(
                    x=future_df["일자"],
                    y=future_df["down_pred"],
                    name="하락 시나리오",
                    line=dict(color="#f59e0b"),
                )
            )

            fig_scenario.update_layout(
                height=400,
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            with sc_col1:
                st.plotly_chart(fig_scenario, use_container_width=True)
            with sc_col2:
                st.markdown("###### 예상 구매비용")
                st.error(
                    f"**상승 시나리오**\n\n{next_day_pred * 1.15:,.0f} 원\n\n**{shortage * next_day_pred * 1.15 / 1e8:,.0f} 억원**"
                )
                scenario_title = (
                    "**LLM 적용 시나리오**" if llm_toggle else "**기준 시나리오**"
                )
                st.info(
                    f"{scenario_title}\n\n{next_day_pred:,.0f} 원\n\n**{required_budget / 1e8:,.0f} 억원**"
                )
                st.warning(
                    f"**하락 시나리오**\n\n{next_day_pred * 0.85:,.0f} 원\n\n**{shortage * next_day_pred * 0.85 / 1e8:,.0f} 억원**"
                )

            # 시나리오 테이블
            scenario_table = pd.DataFrame(
                {
                    "시나리오": ["기준 시나리오", "상승 시나리오", "하락 시나리오"],
                    "평균가격(원/tCO2e)": [
                        next_day_pred,
                        next_day_pred * 1.15,
                        next_day_pred * 0.85,
                    ],
                    "최고가": [
                        next_day_pred * 1.05,
                        next_day_pred * 1.15 * 1.05,
                        next_day_pred * 0.85 * 1.05,
                    ],
                    "최저가": [
                        next_day_pred * 0.95,
                        next_day_pred * 1.15 * 0.95,
                        next_day_pred * 0.85 * 0.95,
                    ],
                    "누적구매량(tCO2e)": [shortage, shortage, shortage],
                    "구매비용(억원)": [
                        required_budget / 1e8,
                        shortage * next_day_pred * 1.15 / 1e8,
                        shortage * next_day_pred * 0.85 / 1e8,
                    ],
                    "전일대비": [
                        ai_pct_change,
                        ((next_day_pred * 1.15 - current_price) / current_price * 100),
                        ((next_day_pred * 0.85 - current_price) / current_price * 100),
                    ],
                    "매입(%)": [100.0, 100.0, 100.0],
                    "신뢰도": ["보통", "높음", "낮음"],
                }
            )

            st.dataframe(
                scenario_table.style.format(
                    {
                        "평균가격(원/tCO2e)": "{:,.0f}",
                        "최고가": "{:,.0f}",
                        "최저가": "{:,.0f}",
                        "누적구매량(tCO2e)": "{:,.0f}",
                        "구매비용(억원)": "{:,.0f}",
                        "전일대비": "{:.2f}%",
                        "매입(%)": "{:.1f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # LLM 이벤트 기반 시뮬레이터 추가
            if llm_btn_clicked and st.session_state.get("llm_selected_tags_widget"):
                event_text = ", ".join(st.session_state.llm_selected_tags_widget)
                st.session_state.llm_last_tags = event_text
                with st.spinner(
                    "LLM이 이벤트를 분석하고 JSON 수치 파라미터로 변환 중입니다..."
                ):
                    try:
                        import importlib
                        from engine import llm_engine as _llm_engine

                        importlib.reload(_llm_engine)
                        from engine.llm_engine import LLMEngine

                        llm = LLMEngine(
                            provider=st.session_state.get("llm_provider", "nemotron")
                        )

                        external_factors = {}
                        for col in latest_data_raw.columns:
                            val = latest_data_raw[col].iloc[0]
                            external_factors[col] = (
                                float(val)
                                if isinstance(
                                    val, (int, float, np.integer, np.floating)
                                )
                                else str(val)
                            )

                        context_snapshot = {
                            "Data_Date": (
                                str(latest_data_raw["일자"].iloc[0].date())
                                if "일자" in latest_data_raw.columns
                                else "2026-07-28"
                            ),
                            "External_Factors": external_factors,
                        }

                        if (
                            "learned_report_text" in st.session_state
                            and st.session_state.learned_report_text
                        ):
                            context_snapshot["학습된_비정형_보고서_내용"] = (
                                st.session_state.learned_report_text
                            )

                        parsed_json = llm.parse_scenario(
                            event_text, int(next_day_pred), context_snapshot
                        )

                        st.session_state.llm_parsed_result = parsed_json
                        st.session_state.llm_provider_name = getattr(
                            llm, "provider_name", "Unknown Model"
                        )

                        if "error" not in parsed_json:
                            adj_rate = float(parsed_json.get("adjustment_rate", 0.0))
                            new_predicted_price = original_next_day_pred * (
                                1 + adj_rate
                            )
                            trajectory = parsed_json.get("predicted_trajectory")
                            if (
                                trajectory
                                and isinstance(trajectory, list)
                                and len(trajectory) > 1
                            ):
                                if len(trajectory) == len(future_dates):
                                    st.session_state.llm_pred_series = trajectory
                                else:
                                    import numpy as np

                                    old_indices = np.linspace(0, 1, len(trajectory))
                                    new_indices = np.linspace(0, 1, len(future_dates))
                                    st.session_state.llm_pred_series = np.interp(
                                        new_indices, old_indices, trajectory
                                    ).tolist()
                            else:
                                st.session_state.llm_pred_series = generate_random_walk(
                                    current_price,
                                    new_predicted_price,
                                    len(future_dates),
                                    vol * 1.5,
                                )
                        st.rerun()
                    except Exception as e:
                        st.error(f"실행 중 오류 발생: {str(e)}")
                        st.session_state.llm_parsed_result = {"error": str(e)}

            if st.session_state.get("llm_parsed_result"):
                with st.container(border=True):
                    st.markdown("##### LLM 시나리오 분석 결과")
                    provider_info = st.session_state.get(
                        "llm_provider_name", "Unknown Model"
                    )
                    st.info(
                        f"**적용된 이벤트 태그:** {st.session_state.get('llm_last_tags', '')}  \n"
                        f"**사용된 AI 모델:** {provider_info}"
                    )

                    parsed_json = st.session_state.llm_parsed_result
                    if "error" in parsed_json:
                        st.error(f"파싱 실패: {parsed_json['error']}")
                    else:
                        adj_rate = float(parsed_json.get("adjustment_rate", 0.0))
                        impact = parsed_json.get("impact_level", "Unknown")
                        reason = parsed_json.get(
                            "reasoning", "분석 내용을 찾을 수 없습니다."
                        )
                        sectors = parsed_json.get("impacted_sectors", [])

                        ev_c1, ev_c2 = st.columns([1, 2])
                        with ev_c1:
                            st.metric("예상 가격 변동률", f"{adj_rate * 100:+.1f}%")
                            color = "gray"
                            if impact.lower() == "high":
                                color = "red"
                            elif impact.lower() == "medium":
                                color = "orange"
                            elif impact.lower() == "low":
                                color = "green"
                            st.markdown(
                                f"**이벤트 영향도:** <span style='color:{color}; font-weight:bold;'>{impact}</span>",
                                unsafe_allow_html=True,
                            )
                        with ev_c2:
                            st.markdown(
                                f"**분석 이유:**<br>{reason}", unsafe_allow_html=True
                            )
                            if sectors:
                                st.markdown(
                                    f"**주요 영향 산업군:** {', '.join(sectors)}"
                                )

                        new_predicted_price = original_next_day_pred * (1 + adj_rate)
                        st.success(
                            f"**이벤트 적용 시 새로운 AI 예측 단가:** {new_predicted_price:,.0f} 원"
                        )

                        llm_pred_series = st.session_state.get("llm_pred_series")
                        if llm_pred_series is not None:
                            fig_llm = go.Figure()
                            fig_llm.add_trace(
                                go.Scatter(
                                    x=past_dates,
                                    y=past_prices,
                                    name="과거 가격",
                                    line=dict(color="#cbd5e1"),
                                )
                            )
                            fig_llm.add_trace(
                                go.Scatter(
                                    x=future_dates,
                                    y=llm_pred_series,
                                    name="이벤트 시나리오",
                                    line=dict(color="#8b5cf6", dash="dot"),
                                )
                            )
                            fig_llm.update_layout(
                                height=350,
                                margin=dict(t=20, b=20, l=10, r=10),
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="left",
                                    x=0,
                                ),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                            )
                            st.plotly_chart(fig_llm, use_container_width=True)

        st.markdown("---")

        # 3. 수요/공급 기반 전망
        st.markdown("##### 수요·공급 기반 전망")
        month_names = [f"26.{m:02d}" for m in range(7, 13)]
        np.random.seed(42)  # 시드 고정으로 위젯 상호작용 시 그래프 변동 방지
        supply = [total_emissions * np.random.uniform(0.8, 1.2) for _ in range(6)]
        demand = [total_emissions * np.random.uniform(0.9, 1.1) for _ in range(6)]
        excess = [s - d for s, d in zip(supply, demand)]
        market_excess = [e * 1.5 for e in excess]

        fig_sd = go.Figure()
        fig_sd.add_trace(
            go.Bar(x=month_names, y=supply, name="공급량", marker_color="#3b82f6")
        )
        fig_sd.add_trace(
            go.Bar(x=month_names, y=demand, name="수요량", marker_color="#10b981")
        )
        fig_sd.add_trace(
            go.Bar(x=month_names, y=excess, name="과부족량", marker_color="#ef4444")
        )
        fig_sd.add_trace(
            go.Bar(
                x=month_names,
                y=market_excess,
                name="시장 잉여량(누락)",
                marker_color="#f59e0b",
            )
        )
        fig_sd.update_layout(
            barmode="group",
            height=300,
            margin=dict(t=20, b=20, l=10, r=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_sd, use_container_width=True)

        sd_table = pd.DataFrame(
            {
                "시나리오": ["공급량", "수요량", "과부족량", "시장 잉여량(누락)"],
                "26.07": [supply[0], demand[0], excess[0], market_excess[0]],
                "26.08": [supply[1], demand[1], excess[1], market_excess[1]],
                "26.09": [supply[2], demand[2], excess[2], market_excess[2]],
                "26.10": [supply[3], demand[3], excess[3], market_excess[3]],
                "26.11": [supply[4], demand[4], excess[4], market_excess[4]],
                "26.12": [supply[5], demand[5], excess[5], market_excess[5]],
            }
        )
        st.dataframe(
            sd_table.style.format(
                {c: "{:,.0f}" for c in sd_table.columns if c != "시나리오"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # 민감도 데이터 계산
        sim_base = latest_data_raw.copy()

        sim_data_wti = []
        base_val_wti = latest_data_raw["WTI_유가"].iloc[0]
        for pct in range(-30, 35, 5):
            temp_df = sim_base.copy()
            temp_df["WTI_유가"] = base_val_wti * (1 + (pct / 100.0))
            temp_df["에너지_도입단가"] = temp_df["환율"] * temp_df["WTI_유가"]
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_wti.append({"변동률(%)": pct, "예측가(원)": pred})
        fig_wti = px.line(
            pd.DataFrame(sim_data_wti),
            x="변동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#ef4444"],
        )
        fig_wti.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_rate = []
        for rate in np.arange(1.0, 5.5, 0.5):
            temp_df = sim_base.copy()
            temp_df["기준금리"] = rate
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_rate.append({"기준금리(%)": rate, "예측가(원)": pred})
        fig_rate = px.line(
            pd.DataFrame(sim_data_rate),
            x="기준금리(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#f59e0b"],
        )
        fig_rate.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_re = []
        for rate in range(0, 35, 5):
            temp_df = sim_base.copy()
            temp_df["RE_비중"] = rate
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_re.append({"RE_비중(%)": rate, "예측가(원)": pred})
        fig_re = px.line(
            pd.DataFrame(sim_data_re),
            x="RE_비중(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#8b5cf6"],
        )
        fig_re.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_op = []
        for rate in range(50, 105, 5):
            temp_df = sim_base.copy()
            temp_df["제조업가동률"] = rate
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_op.append({"가동률(%)": rate, "예측가(원)": pred})
        fig_op = px.line(
            pd.DataFrame(sim_data_op),
            x="가동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#10b981"],
        )
        fig_op.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_lng = []
        for rate in range(0, 105, 10):
            temp_df = sim_base.copy()
            temp_df["lng_ratio"] = rate
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_lng.append({"LNG비율(%)": rate, "예측가(원)": pred})
        fig_lng = px.line(
            pd.DataFrame(sim_data_lng),
            x="LNG비율(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#ec4899"],
        )
        fig_lng.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_exch = []
        base_val_exch = latest_data_raw["환율"].iloc[0]
        for pct in range(-20, 25, 5):
            temp_df = sim_base.copy()
            temp_df["환율"] = base_val_exch * (1 + (pct / 100.0))
            temp_df["에너지_도입단가"] = temp_df["환율"] * temp_df["WTI_유가"]
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_exch.append({"변동률(%)": pct, "예측가(원)": pred})
        fig_exch = px.line(
            pd.DataFrame(sim_data_exch),
            x="변동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#64748b"],
        )
        fig_exch.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_eua = []
        base_val_eua = latest_data_raw["EUA"].iloc[0]
        for pct in range(-30, 35, 5):
            temp_df = sim_base.copy()
            temp_df["EUA"] = base_val_eua * (1 + (pct / 100.0))
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_eua.append({"변동률(%)": pct, "예측가(원)": pred})
        fig_eua = px.line(
            pd.DataFrame(sim_data_eua),
            x="변동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#0ea5e9"],
        )
        fig_eua.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_ng = []
        base_val_ng = latest_data_raw["천연가스"].iloc[0]
        for pct in range(-30, 35, 5):
            temp_df = sim_base.copy()
            temp_df["천연가스"] = base_val_ng * (1 + (pct / 100.0))
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_ng.append({"변동률(%)": pct, "예측가(원)": pred})
        fig_ng = px.line(
            pd.DataFrame(sim_data_ng),
            x="변동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#84cc16"],
        )
        fig_ng.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_coal = []
        base_val_coal = latest_data_raw["석탄"].iloc[0]
        for pct in range(-30, 35, 5):
            temp_df = sim_base.copy()
            temp_df["석탄"] = base_val_coal * (1 + (pct / 100.0))
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_coal.append({"변동률(%)": pct, "예측가(원)": pred})
        fig_coal = px.line(
            pd.DataFrame(sim_data_coal),
            x="변동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#57534e"],
        )
        fig_coal.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_smp = []
        base_val_smp = latest_data_raw["SMP"].iloc[0]
        for pct in range(-30, 35, 5):
            temp_df = sim_base.copy()
            temp_df["SMP"] = base_val_smp * (1 + (pct / 100.0))
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_smp.append({"변동률(%)": pct, "예측가(원)": pred})
        fig_smp = px.line(
            pd.DataFrame(sim_data_smp),
            x="변동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#fb923c"],
        )
        fig_smp.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_kospi = []
        base_val_kospi = latest_data_raw["KOSPI"].iloc[0]
        for pct in range(-20, 25, 5):
            temp_df = sim_base.copy()
            temp_df["KOSPI"] = base_val_kospi * (1 + (pct / 100.0))
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_kospi.append({"변동률(%)": pct, "예측가(원)": pred})
        fig_kospi = px.line(
            pd.DataFrame(sim_data_kospi),
            x="변동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#3b82f6"],
        )
        fig_kospi.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_bsi = []
        base_val_bsi = latest_data_raw["업황전망BSI"].iloc[0]
        for pct in range(-20, 25, 5):
            temp_df = sim_base.copy()
            temp_df["업황전망BSI"] = base_val_bsi * (1 + (pct / 100.0))
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_bsi.append({"변동률(%)": pct, "예측가(원)": pred})
        fig_bsi = px.line(
            pd.DataFrame(sim_data_bsi),
            x="변동률(%)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#a855f7"],
        )
        fig_bsi.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        sim_data_temp = []
        base_val_temp = latest_data_raw["평균기온"].iloc[0]
        for adj in np.arange(-10.0, 11.0, 2.0):
            temp_df = sim_base.copy()
            new_temp = base_val_temp + adj
            temp_df["평균기온"] = new_temp
            temp_df["CDD_냉방수요"] = max(0, new_temp - 24)
            temp_df["HDD_난방수요"] = max(0, 18 - new_temp)
            pred, _ = engine.predict_kau(model, current_price, temp_df)
            sim_data_temp.append({"증감(℃)": adj, "예측가(원)": pred})
        fig_temp = px.line(
            pd.DataFrame(sim_data_temp),
            x="증감(℃)",
            y="예측가(원)",
            markers=True,
            color_discrete_sequence=["#ef4444"],
        )
        fig_temp.update_layout(height=300, margin=dict(t=20, b=20, l=10, r=10))

        # 하단 2분할 (변수 영향 분석 / 민감도 분석)
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("##### 변수 영향 분석")
            # importance_df 재정의
            importance_weights = model.feature_importances_
            feature_names = X_cols

            feature_name_map = {
                "lag_1": "전일 가격",
                "lag_2": "2일전 가격",
                "lag_3": "3일전 가격",
                "rolling_mean_5": "5일 이동평균",
                "rolling_mean_10": "10일 이동평균",
                "volatility": "가격 변동성",
                "WTI_유가": "국제 연료가격(WTI)",
                "KOSPI": "코스피 지수",
                "CDD_냉방수요": "냉방수요(CDD)",
                "HDD_난방수요": "난방수요(HDD)",
                "SMP": "계통한계가격(SMP)",
                "EUA": "유럽 배출권(EUA)",
                "lng_ratio": "LNG 발전 비중",
                "RE_비중": "신재생에너지 비중",
                "정산기한_접근도": "정산기한 접근도",
                "업황전망BSI": "산업 활동 지수(BSI)",
                "제조업가동률": "제조업 가동률",
                "환율": "환율",
                "거래량": "거래량",
                "천연가스": "천연가스",
                "석탄": "글로벌 석탄",
                "기준금리": "기준금리",
                "에너지_도입단가": "에너지 도입단가",
            }

            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance_weights}
            )
            importance_df["Feature"] = importance_df["Feature"].apply(
                lambda x: feature_name_map.get(x, x)
            )

            importance_df = (
                importance_df.groupby("Feature", as_index=False)["Importance"]
                .sum()
                .sort_values(by="Importance", ascending=False)
                .head(8)
            )

            importance_df["Imp_Text"] = importance_df["Importance"].apply(
                lambda x: f"{x:.4f}"
            )
            fig_imp2 = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                text="Imp_Text",
            )
            fig_imp2.update_traces(textposition="outside")
            fig_imp2.update_layout(
                yaxis={"categoryorder": "total ascending", "title": ""},
                xaxis={"visible": False},
                coloraxis_showscale=False,
                height=300,
                margin=dict(l=10, r=40, t=10, b=10),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_imp2, use_container_width=True, key="fig_imp2_tab2")

        with s_col2:
            st.markdown("##### 민감도 분석")
            sens_var = st.radio(
                "분석 변수",
                [
                    "WTI유가",
                    "원/달러 환율",
                    "EUA",
                    "천연가스",
                    "석탄",
                    "SMP",
                    "KOSPI",
                    "업황전망BSI",
                    "평균기온",
                    "기준금리",
                    "재생에너지 비중",
                    "가동률",
                    "LNG 전환 비율",
                ],
                horizontal=True,
                label_visibility="collapsed",
            )

            sens_map = {
                "WTI유가": fig_wti,
                "원/달러 환율": fig_exch,
                "EUA": fig_eua,
                "천연가스": fig_ng,
                "석탄": fig_coal,
                "SMP": fig_smp,
                "KOSPI": fig_kospi,
                "업황전망BSI": fig_bsi,
                "평균기온": fig_temp,
                "기준금리": fig_rate,
                "재생에너지 비중": fig_re,
                "가동률": fig_op,
                "LNG 전환 비율": fig_lng,
            }
            selected_fig = sens_map[sens_var]
            selected_fig.data[0].name = "기준 시나리오"
            selected_fig.data[0].showlegend = True

            if llm_toggle:
                x_data = selected_fig.data[0].x
                y_data = selected_fig.data[0].y
                llm_y_data = [y * (1 + llm_adj_rate) for y in y_data]

                selected_fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=llm_y_data,
                        mode="lines+markers",
                        name="LLM 반영 시나리오",
                        line=dict(dash="dash", color="#8b5cf6"),
                    )
                )

            selected_fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0
                ),
            )

            st.plotly_chart(selected_fig, use_container_width=True)
    # !SECTION - 가격 예측 시뮬레이터

    # ---------------------------------------------------------
    # SECTION - 실시간 예측 비교
    # ---------------------------------------------------------
    # elif "실시간 예측 비교" in selected_tab:
    #     st.subheader("실시간 일일 예측 비교 (2026-07-02 이후)")

    #     test = df[df["일자"] > "2026-07-01"].copy()
    #     if len(test) > 0:
    #         pred_returns = model.predict(test[X_cols])
    #         test["예측가"] = test["lag_1"] * (1 + pred_returns)
    #         test["상단"] = test["예측가"] * 1.03
    #         test["하단"] = test["예측가"] * 0.97

    #         st.markdown("##### 일별 가격 예측 및 신뢰 구간")
    #         st.line_chart(test.set_index("일자")[["종가", "예측가", "상단", "하단"]])

    #         st.markdown("##### 일별 상세 비교 데이터")
    #         compare_df = test[["일자", "종가", "예측가"]].copy()
    #         compare_df.rename(
    #             columns={"종가": "실제 가격(원)", "예측가": "예측 가격(원)"}, inplace=True
    #         )
    #         compare_df["오차율(%)"] = (
    #             (compare_df["예측 가격(원)"] - compare_df["실제 가격(원)"])
    #             / compare_df["실제 가격(원)"]
    #             * 100
    #         ).round(2)
    #         compare_df["일자"] = compare_df["일자"].dt.strftime("%Y-%m-%d")
    #         compare_df = compare_df.set_index("일자")
    #         st.dataframe(compare_df.style.format("{:.2f}"))
    #     else:
    #         st.info("2026-07-01 이후의 실제 가격 데이터가 아직 수집되지 않았습니다.")
    # !SECTION - 실시간 예측 비교

    # ---------------------------------------------------------
    # SECTION - AI 모델 성능 리포트
    # ---------------------------------------------------------
    # elif "AI 모델 성능 리포트" in selected_tab:
    #     st.subheader("AI 모델 성능 리포트 (XGBoost)")
    #     st.markdown("##### 1. 예측 성능 핵심 지표")

    #     col1, col2, col3, col4 = st.columns(4)
    #     col1.metric("R² (설명력)", f"{r2:.4f}")
    #     col2.metric("RMSE (평균 제곱근 오차)", f"{rmse:,.0f} 원")
    #     col3.metric("MAPE (오차율)", f"{mape:.2f} %")
    #     col4.metric("MAE (평균 절대 오차)", f"{mae:,.0f} 원")

    #     st.markdown("---")
    #     st.markdown("##### 2. 실제값 vs 예측값 산점도 (Scatter Plot)")

    #     if len(test) > 0:
    #         fig_scatter = px.scatter(
    #             compare_df.reset_index(),
    #             x="실제 가격(원)",
    #             y="예측 가격(원)",
    #             title="실제 가격 대비 AI 예측 가격",
    #         )
    #         min_val = min(
    #             compare_df["실제 가격(원)"].min(), compare_df["예측 가격(원)"].min()
    #         )
    #         max_val = max(
    #             compare_df["실제 가격(원)"].max(), compare_df["예측 가격(원)"].max()
    #         )
    #         fig_scatter.add_shape(
    #             type="line",
    #             x0=min_val,
    #             y0=min_val,
    #             x1=max_val,
    #             y1=max_val,
    #             line=dict(color="#ef4444", width=2, dash="dash"),
    #         )
    #         fig_scatter.update_layout(height=400, margin=dict(t=40, b=20, l=10, r=10))
    #         st.plotly_chart(fig_scatter, use_container_width=True)
    #     else:
    #         st.info("검증할 데이터가 부족하여 산점도를 표시할 수 없습니다.")
    # !SECTION - AI 모델 성능 리포트

    # ---------------------------------------------------------
    # SECTION - API 연동 테스트
    # ---------------------------------------------------------
    # elif "API 연동 테스트" in selected_tab:
    #     st.subheader("API 연동 테스트 (1일/1개월 단위 조회)")

    #     if st.button("API 연동 테스트 실행"):
    #         try:
    #             from engine.utils.external_api_test import ExternalAPITestClient

    #             api_client = ExternalAPITestClient()

    #             with st.spinner("API 데이터 호출 중..."):
    #                 c1, c2 = st.columns(2)

    #                 with c1:
    #                     st.markdown("**1. 금융위 일반상품시세 (KAU)**")
    #                     st.json(api_client.fetch_general_product_info_kau())

    #                     st.markdown("**2. 에너지공단 온실가스 배출량 통계**")
    #                     st.json(api_client.fetch_ghg_emission_stat())

    #                     st.markdown("**3. 전력거래소 발전원별 발전량**")
    #                     st.json(api_client.fetch_power_gen_by_source())

    #                     st.markdown("**4. 전력거래소 실시간 발전량 현황**")
    #                     st.json(api_client.fetch_power_gen_status_5m())

    #                     st.markdown("**5. 전력거래소 SMP 및 수요예측**")
    #                     st.json(api_client.fetch_smp_and_demand_forecast())

    #                     st.markdown("**6. 동서발전 연료원별 발전량**")
    #                     st.json(api_client.fetch_ewp_daily_power_gen())

    #                     st.markdown("**7. KEPCO 산업분류별 전력사용량**")
    #                     st.json(api_client.fetch_kepco_power_usage())

    #                 with c2:
    #                     st.markdown("**8. ECOS 기준금리**")
    #                     st.json(api_client.fetch_ecos_base_rate())

    #                     st.markdown("**9. ECOS 환율**")
    #                     st.json(api_client.fetch_ecos_exchange_rate())

    #                     st.markdown("**10. ECOS BSI (기업경기조사)**")
    #                     st.json(api_client.fetch_ecos_business_survey_bsi())

    #                     st.markdown("**11. KOSIS 광공업생산지수**")
    #                     st.json(api_client.fetch_kosis_mining_industry_index())

    #                     st.markdown("**12. KOSIS 제조업 생산능력지수**")
    #                     st.json(api_client.fetch_kosis_manufacturing_index())

    #                     st.markdown("**13. Yahoo Finance 데이터**")
    #                     st.write("KOSPI 200:", api_client.fetch_yahoo_kospi())
    #                     st.write(
    #                         "Global Energy (WTI, TTF, Coal):",
    #                         api_client.fetch_yahoo_global_energy_prices(),
    #                     )
    #                     st.write("EUA Carbon Price:", api_client.fetch_yahoo_eua_price())

    #                     st.markdown("**14. 기상청 API 허브 (ASOS)**")
    #                     st.json(api_client.fetch_kma_asos_data())

    #             st.success("API 호출 완료")
    #         except Exception as e:
    #             st.error(f"테스트 중 오류 발생: {e}")
    # !SECTION - API 연동 테스트

    # ---------------------------------------------------------
    # SECTION - LLM 브리핑 테스트
    # ---------------------------------------------------------
    # elif "LLM 브리핑 테스트" in selected_tab:
    #     st.subheader("LLM (Gemini) 연동 및 브리핑 테스트")
    #     st.markdown("Phase 1: `LLMEngine` 모듈이 정상적으로 동작하는지 테스트합니다.")

    #     if st.button("시장 분석 브리핑 생성"):
    #         with st.spinner("LLM이 시장 데이터를 분석 중입니다..."):
    #             try:
    #                 from engine.llm_engine import LLMEngine

    #                 llm = LLMEngine()

    #                 # 모든 변수 동적 할당
    #                 external_factors = {}
    #                 for col in latest_data_raw.columns:
    #                     val = latest_data_raw[col].iloc[0]
    #                     external_factors[col] = (
    #                         float(val)
    #                         if isinstance(val, (int, float, np.integer, np.floating))
    #                         else str(val)
    #                     )

    #                 # 테스트용 컨텍스트(스냅샷) 데이터 구성
    #                 context_snapshot = {
    #                     "Data_Date": (
    #                         str(latest_data_raw["일자"].iloc[0].date())
    #                         if "일자" in latest_data_raw.columns
    #                         else "2026-07-28"
    #                     ),
    #                     "KAU_Price": {
    #                         "current_price": float(current_price),
    #                         "next_day_predicted": float(next_day_pred),
    #                         "price_change_pct": (
    #                             round(
    #                                 (
    #                                     (
    #                                         current_price
    #                                         - (
    #                                             df["종가"].iloc[-2]
    #                                             if len(df) > 1
    #                                             else current_price
    #                                         )
    #                                     )
    #                                     / (
    #                                         df["종가"].iloc[-2]
    #                                         if len(df) > 1
    #                                         else current_price
    #                                     )
    #                                     * 100
    #                                 ),
    #                                 2,
    #                             )
    #                             if len(df) > 1
    #                             else 0
    #                         ),
    #                     },
    #                     "External_Factors": external_factors,
    #                     "Recommendation_Engine": {"strategy": strategy, "reason": reason},
    #                 }

    #                 st.markdown("#### 주입된 데이터 스냅샷")
    #                 st.json(context_snapshot)

    #                 st.markdown("#### LLM 분석 결과")
    #                 # 스트리밍 방식 출력
    #                 stream_generator = llm.analyze_market_stream(context_snapshot)
    #                 st.write_stream(stream_generator)

    #             except Exception as e:
    #                 st.error(f"LLM 호출 실패: {str(e)}")
    # !SECTION - LLM 브리핑 테스트

    # ---------------------------------------------------------
    # SECTION - 리포트 자동 생성
    # ---------------------------------------------------------
    elif "리포트 자동 생성" in selected_tab:
        st.subheader("월간 배출권 매매계획 리포트")
        st.markdown(
            "현재 모델 예측치와 시장 데이터를 바탕으로 보고서를 자동 생성합니다."
        )

        import datetime

        current_date_str = datetime.datetime.now().strftime("%Y년 %m월")

        # Prepare context data
        context_data = {
            "current_date_str": current_date_str,
            "current_date": test_end,
            "current_price": float(current_price),
            "predicted_price": float(
                current_price * (1 + getattr(st.session_state, "latest_return", 0))
            ),
            "predicted_return_pct": float(
                getattr(st.session_state, "latest_return", 0) * 100
            ),
            "trading_volume": float(current_vol),
            "macro_wti": float(latest_data_raw["WTI_유가"].iloc[0]),
            "macro_exchange": float(latest_data_raw["환율"].iloc[0]),
            "macro_eua": float(latest_data_raw["EUA"].iloc[0]),
        }

        if st.button("보고서 생성", type="primary", use_container_width=True):
            with st.spinner(
                "AI가 데이터를 분석하여 보고서를 작성하고 있습니다... (약 10~15초 소요)"
            ):
                import importlib
                import engine.llm_engine

                importlib.reload(engine.llm_engine)
                from engine.llm_engine import LLMEngine
                from engine.utils.pdf_generator import generate_pdf_report

                llm = LLMEngine(
                    provider=st.session_state.get("llm_provider", "nemotron")
                )
                report_data = llm.generate_report_text(context_data)

                # 생성된 데이터와 PDF를 session_state에 저장
                st.session_state["report_data"] = report_data
                st.session_state["pdf_bytes"] = generate_pdf_report(
                    context_data, report_data
                )

        # session_state에 보고서 데이터가 있으면 렌더링
        if "report_data" in st.session_state:
            report_data = st.session_state["report_data"]

            st.markdown("---")
            st.markdown(f"### {current_date_str} 온실가스 배출권 매매계획(안)")
            st.markdown("---")

            st.markdown("#### Ⅰ. 배출권 시장현황")
            st.markdown(
                f"**현재가:** {current_price:,.0f}원 | **최근 거래량:** {current_vol:,.0f}톤"
            )
            st.info(
                report_data.get(
                    "market_trend", "시장 동향 텍스트 생성 중 오류가 발생했습니다."
                )
            )

            st.markdown("#### Ⅱ. 향후 전망 (AI 예측 기반)")
            st.markdown(
                f"**AI 예측가 (단기):** {context_data['predicted_price']:,.0f}원 ({context_data['predicted_return_pct']:+.2f}%)"
            )
            st.success(
                report_data.get(
                    "future_outlook", "전망 텍스트 생성 중 오류가 발생했습니다."
                )
            )

            st.markdown("#### Ⅲ. 배출권 확보계획(안)")
            st.warning(
                report_data.get(
                    "purchasing_strategy", "구매 전략 생성 중 오류가 발생했습니다."
                )
            )

            if "pdf_bytes" in st.session_state and st.session_state["pdf_bytes"]:
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📄 월간 보고서 PDF 다운로드",
                    data=st.session_state["pdf_bytes"],
                    file_name="월간_배출권_매매계획_보고서.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

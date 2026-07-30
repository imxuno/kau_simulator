# src/engine/kau_engine.py
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import os


# ---------------------------------------------------------
# SECTION - 데이터 로드 및 전처리
# ---------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    from engine.utils.external_api import ExternalAPIClient  # 외부 API 연동

    api_client = ExternalAPIClient()

    # 1. KAU 가격
    today = datetime.now()
    begin_date_str = "20210101"  # 제 3차 계획기간 시작일
    end_date_str = today.strftime("%Y%m%d")  # 오늘

    df = pd.DataFrame()
    try:
        items = []
        for p in [1, 2, 3]:
            kau_res = api_client.fetch_general_product_info_kau(
                begin_date=begin_date_str,
                end_date=end_date_str,
                page_no=p,
                num_of_rows=10000,
            )
            page_items = (
                kau_res.get("response", {})
                .get("body", {})
                .get("items", {})
                .get("item", [])
            )
            if not page_items:
                break
            items.extend(page_items)

        if items:
            df = pd.DataFrame(items)
            if "itmsNm" in df.columns:
                df = df[df["itmsNm"].str.startswith("KAU", na=False)]
                df = df.rename(
                    columns={
                        "basDt": "일자",
                        "clpr": "종가",
                        "mkp": "시가",
                        "hipr": "고가",
                        "lopr": "저가",
                        "trqu": "거래량",
                    }
                )
                # 거래량 기준 정렬을 위해 임시로 float 변환
                df["거래량_num"] = (
                    df["거래량"].astype(str).str.replace(",", "").astype(float)
                )
                df = df.sort_values("거래량_num", ascending=False).drop_duplicates(
                    subset=["일자"], keep="first"
                )
                df = df.drop(columns=["거래량_num"])

                if "일자" in df.columns:
                    df["일자"] = pd.to_datetime(df["일자"], format="%Y%m%d")
                    df["종목명"] = "KAU_Benchmark"
                    df = df.sort_values("일자")
    except Exception as e:
        st.warning(f"⚠️ KAU 실시간 데이터 연동 실패: {e}")
        return pd.DataFrame()

    cols_to_fix = ["종가", "시가", "고가", "저가", "거래량"]
    for col in cols_to_fix:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.replace(",", "").astype(float)
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan).ffill()

    start_date_str = df["일자"].min().strftime("%Y-%m-%d")
    end_date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    # 2. Yahoo Finance 데이터
    try:

        def fetch_and_prep_yf(ticker, start, end, col_name):
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data.empty:
                return pd.DataFrame(columns=[col_name])
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if "Close" in data.columns:
                res = data[["Close"]].rename(columns={"Close": col_name})
                if res.index.tz is not None:
                    res.index = res.index.tz_localize(None)
                res = res[~res.index.duplicated(keep='first')]
                return res
            return pd.DataFrame(columns=[col_name])

        # 글로벌 에너지(WTI, 천연가스, 석탄) 및 EUA, KOSPI
        wti = fetch_and_prep_yf("CL=F", start_date_str, end_date_str, "WTI_유가")
        ng = fetch_and_prep_yf("NG=F", start_date_str, end_date_str, "천연가스")
        kospi = fetch_and_prep_yf("^KS11", start_date_str, end_date_str, "KOSPI")
        coal = fetch_and_prep_yf("MTF=F", start_date_str, end_date_str, "석탄")
        eua = fetch_and_prep_yf("KE=F", start_date_str, end_date_str, "EUA")

        ext_data = (
            wti.join(ng, how="outer")
            .join(kospi, how="outer")
            .join(coal, how="outer")
            .join(eua, how="outer")
        )
        ext_data.index = pd.to_datetime(ext_data.index)

        # 한국은행 환율 (ECOS)
        try:
            krw_res = api_client.fetch_ecos_exchange_rate(
                start_date=start_date_str.replace("-", ""),
                end_date=end_date_str.replace("-", ""),
            )
            krw_items = krw_res.get("StatisticSearch", {}).get("row", [])
            if krw_items:
                krw = pd.DataFrame(krw_items)
                krw["일자"] = pd.to_datetime(krw["TIME"], format="%Y%m%d")
                krw = (
                    krw.set_index("일자")[["DATA_VALUE"]]
                    .rename(columns={"DATA_VALUE": "환율"})
                    .astype(float)
                )
                ext_data = ext_data.join(krw, how="outer")
            else:
                krw = fetch_and_prep_yf("KRW=X", start_date_str, end_date_str, "환율")
                ext_data = ext_data.join(krw, how="outer")
        except Exception:
            krw = fetch_and_prep_yf("KRW=X", start_date_str, end_date_str, "환율")
            ext_data = ext_data.join(krw, how="outer")

        # 종관기상관측
        try:
            weather_res = api_client.fetch_kma_asos_data(
                start_date=start_date_str.replace("-", ""),
                end_date=end_date_str.replace("-", ""),
            )
            txt = weather_res.get("data", "")
            lines = [l for l in txt.split("\n") if not l.startswith("#") and l.strip()]
            if lines:
                import io

                weather_raw = pd.read_csv(
                    io.StringIO("\n".join(lines)),
                    delim_whitespace=True,
                    header=None,
                    usecols=[0, 11],
                )
                weather_raw.columns = ["TIME", "TA"]
                weather_raw["일자"] = pd.to_datetime(
                    weather_raw["TIME"].astype(str).str[:8],
                    format="%Y%m%d",
                    errors="coerce",
                )
                weather_raw["TA"] = pd.to_numeric(weather_raw["TA"], errors="coerce")
                weather = (
                    weather_raw.groupby("일자")["TA"]
                    .mean()
                    .reset_index()
                    .rename(columns={"TA": "평균기온"})
                )
                weather = weather.set_index("일자")
                weather["CDD_냉방수요"] = weather["평균기온"].apply(
                    lambda x: max(0, x - 24) if pd.notnull(x) else 0
                )
                weather["HDD_난방수요"] = weather["평균기온"].apply(
                    lambda x: max(0, 18 - x) if pd.notnull(x) else 0
                )
                ext_data = ext_data.join(weather, how="outer")
            else:
                ext_data["평균기온"] = 15.0
                ext_data["CDD_냉방수요"] = 0.0
                ext_data["HDD_난방수요"] = 3.0
        except Exception as e:
            ext_data["평균기온"] = 15.0
            ext_data["CDD_냉방수요"] = 0.0
            ext_data["HDD_난방수요"] = 3.0

        # 발전원별 발전량 (RE_비중, lng_ratio 산출)
        try:
            pwr_res = api_client.fetch_power_gen_by_source(1, 10000)
            items = (
                pwr_res.get("response", {})
                .get("body", {})
                .get("items", {})
                .get("item", [])
            )
            if items:
                pwr = pd.DataFrame(items)
                pwr["일자"] = pd.to_datetime(pwr["tradeYmd"], format="%Y%m%d")
                pwr["amgo"] = pd.to_numeric(pwr["amgo"], errors="coerce")
                pwr_pivot = pwr.pivot_table(
                    index="일자", columns="fuelTpCd", values="amgo", aggfunc="sum"
                ).fillna(0)
                pwr_pivot["총발전량"] = pwr_pivot.sum(axis=1)
                pwr_pivot["lng_ratio"] = (
                    pwr_pivot.get("LNG", 0) / pwr_pivot["총발전량"]
                ) * 100
                pwr_pivot["RE_비중"] = (
                    (
                        pwr_pivot.get("태양광", 0)
                        + pwr_pivot.get("풍력", 0)
                        + pwr_pivot.get("수력", 0)
                    )
                    / pwr_pivot["총발전량"]
                    * 100
                )
                ext_data = ext_data.join(
                    pwr_pivot[["lng_ratio", "RE_비중"]], how="outer"
                )
            else:
                ext_data["lng_ratio"] = 20.0
                ext_data["RE_비중"] = 10.0
        except Exception as e:
            ext_data["lng_ratio"] = 20.0
            ext_data["RE_비중"] = 10.0

    except Exception as e:
        st.warning(f"⚠️ 외부 데이터 연동 일부 실패: {e}")
        ext_data = pd.DataFrame(index=pd.date_range(start_date_str, end_date_str))
        for c in [
            "환율",
            "WTI_유가",
            "천연가스",
            "KOSPI",
            "석탄",
            "EUA",
            "평균기온",
            "CDD_냉방수요",
            "HDD_난방수요",
            "lng_ratio",
            "RE_비중",
        ]:
            ext_data[c] = 0.0

    # 3. 공공데이터포털 SMP 실시간 캐싱 (하루치씩만 조회되므로 캐싱본과 조합)
    os.makedirs("data", exist_ok=True)
    smp_cache_path = "data/smp_api_cache.csv"
    try:
        if os.path.exists(smp_cache_path):
            smp = pd.read_csv(smp_cache_path, index_col=0, parse_dates=True)
        else:
            # 최초 캐시 파일 생성 (과거 로컬 데이터를 시드(Seed)로 1회 활용)
            legacy_path = "data/HOME_전력거래_계통한계가격_시간별SMP.csv"
            if os.path.exists(legacy_path):
                smp_legacy = pd.read_csv(legacy_path, encoding="cp949")
                smp_legacy["기간"] = pd.to_datetime(smp_legacy["기간"])
                smp = smp_legacy.set_index("기간")[["가중평균"]].rename(
                    columns={"가중평균": "SMP"}
                )
            else:
                # 레거시 파일도 없을 경우 빈 데이터프레임으로 시작
                smp = pd.DataFrame(columns=["SMP"])
                smp.index.name = "기간"
            smp.to_csv(smp_cache_path)

        if not smp.empty:
            last_smp_date = smp.index.max()
        else:
            last_smp_date = pd.to_datetime("2026-07-28") - timedelta(
                days=7
            )  # 데이터가 아예 없을 경우 최근 7일치만 가져오도록

        today = datetime.now()
        # 마지막 저장일 이후로 누락된 SMP만 하루씩 API로 긁어오기 (최소화)
        if last_smp_date < today - timedelta(days=1):
            import time

            missing_dates = pd.date_range(
                last_smp_date + timedelta(days=1), today - timedelta(days=1)
            )
            new_smp_rows = []
            for d in missing_dates:
                try:
                    res = api_client.fetch_smp_and_demand_forecast(
                        target_date=d.strftime("%Y%m%d"), num_of_rows=24
                    )
                    if not res:
                        st.warning(
                            "⚠️ SMP API 수집 중 빈 데이터가 반환되었습니다. (Rate Limit 또는 트래픽 제한 추정)"
                        )
                        break

                    items = (
                        res.get("response", {})
                        .get("body", {})
                        .get("items", {})
                        .get("item", [])
                    )
                    if items:
                        avg_smp = np.mean(
                            [
                                float(item.get("smp", 0))
                                for item in items
                                if "smp" in item
                            ]
                        )
                        if avg_smp > 0:
                            new_smp_rows.append({"기간": d, "SMP": avg_smp})
                    time.sleep(
                        1.0
                    )  # 429 Rate Limit 방지용 딜레이 (0.3초 -> 1.0초로 증가)
                except Exception as loop_e:
                    st.warning(f"⚠️ SMP API 연동 중단 (예외 발생): {loop_e}")
                    break  # 에러 발생 시 현재까지 모은 것만 저장하고 루프 탈출
            if new_smp_rows:
                new_smp_df = pd.DataFrame(new_smp_rows).set_index("기간")
                smp = pd.concat([smp, new_smp_df]).sort_index()
                smp = smp[~smp.index.duplicated(keep="last")]
                smp.to_csv(smp_cache_path)  # 캐시 업데이트
    except Exception as e:
        st.warning(f"⚠️ SMP API 연동 실패 (기존 캐시 사용): {e}")
        smp = pd.DataFrame(columns=["SMP"])

    ext_data = ext_data.join(smp, how="outer")
    ext_data.index.name = "일자"
    ext_data = ext_data[~ext_data.index.duplicated(keep="first")]
    df = pd.merge(df, ext_data, on="일자", how="left")

    # 5. 월간 거시 지표 데이터 (BSI, 제조업 가동률)
    df["연월"] = df["일자"].dt.strftime("%Y-%m")

    start_month = start_date_str.replace("-", "")[:6]
    end_month = end_date_str.replace("-", "")[:6]

    # 5-1. BSI (ECOS)
    try:
        bsi_res = api_client.fetch_ecos_business_survey_bsi(
            start_month=start_month, end_month=end_month
        )
        bsi_items = bsi_res.get("StatisticSearch", {}).get("row", [])
        if bsi_items:
            bsi = pd.DataFrame(bsi_items)
            bsi["연월"] = bsi["TIME"].str[:4] + "-" + bsi["TIME"].str[4:6]
            bsi = (
                bsi.set_index("연월")[["DATA_VALUE"]]
                .rename(columns={"DATA_VALUE": "업황전망BSI"})
                .astype(float)
            )
            df = pd.merge(df, bsi, on="연월", how="left")
        else:
            df["업황전망BSI"] = 75.0
    except Exception:
        df["업황전망BSI"] = 75.0

    # 5-2. 제조업 가동률 (KOSIS)
    try:
        mfg_res = api_client.fetch_kosis_manufacturing_index(start_month=start_month)
        if isinstance(mfg_res, list) and len(mfg_res) > 0:
            mfg = pd.DataFrame(mfg_res)
            if "PRD_DE" in mfg.columns:
                mfg["연월"] = (
                    mfg["PRD_DE"].astype(str).str[:4]
                    + "-"
                    + mfg["PRD_DE"].astype(str).str[4:6]
                )
                mfg = mfg[(mfg["ITM_ID"] == "T30") & (mfg["C1"] == "C")]
                mfg = (
                    mfg.set_index("연월")[["DT"]]
                    .rename(columns={"DT": "제조업가동률"})
                    .astype(float)
                )
                df = pd.merge(df, mfg, on="연월", how="left")
            else:
                df["제조업가동률"] = 100.0
        else:
            df["제조업가동률"] = 100.0
    except Exception:
        df["제조업가동률"] = 100.0

    df["에너지_도입단가"] = df["환율"] * df["WTI_유가"]

    # 결측치 보정 (Daily & Monthly FFill)
    cols_to_fill = [
        "환율",
        "WTI_유가",
        "천연가스",
        "KOSPI",
        "평균기온",
        "CDD_냉방수요",
        "HDD_난방수요",
        "SMP",
        "석탄",
        "EUA",
        "lng_ratio",
        "RE_비중",
        "업황전망BSI",
        "제조업가동률",
    ]
    for col in cols_to_fill:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0)

    def get_days_to_compliance(date):
        compliance_date = datetime(
            date.year if date.month <= 6 else date.year + 1, 6, 30
        )
        return (compliance_date - date).days

    df["정산기한_접근도"] = df["일자"].apply(get_days_to_compliance)

    # 4. ECOS API - 기준금리 실시간 연동 (기존 3.50 하드코딩 대체)
    try:
        ecos_res = api_client.fetch_ecos_base_rate(
            start_date=start_date_str.replace("-", "")[:6],
            end_date=end_date_str.replace("-", "")[:6],
        )
        ecos_items = ecos_res.get("StatisticSearch", {}).get("row", [])
        if ecos_items:
            rates = pd.DataFrame(ecos_items)
            rates["일자"] = pd.to_datetime(rates["TIME"], format="%Y%m")
            rates = rates.set_index("일자")[["DATA_VALUE"]].rename(
                columns={"DATA_VALUE": "기준금리"}
            )
            rates["기준금리"] = rates["기준금리"].astype(float)
            rates = rates.resample("D").ffill()  # 월간 데이터를 일별로 확장
            df = pd.merge(df, rates, on="일자", how="left")
            df["기준금리"] = df["기준금리"].ffill().bfill().fillna(0)
        else:
            df["기준금리"] = 3.50
    except Exception:
        df["기준금리"] = 3.50

    # 특징 생성
    df["target"] = df["종가"].pct_change()
    df["lag_1"] = df["종가"].shift(1)
    df["lag_2"] = df["종가"].shift(2)
    df["lag_3"] = df["종가"].shift(3)
    df["rolling_mean_5"] = df["종가"].rolling(window=5).mean().shift(1)
    df["rolling_mean_10"] = df["종가"].rolling(window=10).mean().shift(1)
    df["volatility"] = (df["고가"] - df["저가"]).shift(1)

    ext_cols = [
        "거래량",
        "환율",
        "WTI_유가",
        "천연가스",
        "KOSPI",
        "평균기온",
        "CDD_냉방수요",
        "HDD_난방수요",
        "SMP",
        "석탄",
        "EUA",
        "에너지_도입단가",
        "정산기한_접근도",
        "기준금리",
        "업황전망BSI",
        "제조업가동률",
        "lng_ratio",
        "RE_비중",
    ]
    # 이전에는 여기서 ext_cols를 shift(1) 했으나, target이 이미 shift(-1) 되어 있으므로
    # 당일 데이터를 사용해 익일 가격을 예측하는 것이 맞습니다. shift(1)을 제거하여 거래량 및 외부 변수가 하루씩 밀리는 현상 수정.

    return df.dropna()


# !SECTION - 데이터 로드 및 전처리


# ---------------------------------------------------------
# SECTION - 피쳐 컬럼 가져오기
# ---------------------------------------------------------
def get_x_cols():
    return [
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_5",
        "rolling_mean_10",
        "volatility",
        "거래량",
        "환율",
        "WTI_유가",
        "천연가스",
        "KOSPI",
        "평균기온",
        "CDD_냉방수요",
        "HDD_난방수요",
        "SMP",
        "석탄",
        "EUA",
        "에너지_도입단가",
        "정산기한_접근도",
        "기준금리",
        "업황전망BSI",
        "제조업가동률",
        "lng_ratio",
        "RE_비중",
    ]


# !SECTION - 피쳐 컬럼 가져오기


# ---------------------------------------------------------
# SECTION - 학습 모델 가져오기
# ---------------------------------------------------------
@st.cache_resource
def get_trained_model(df, X_cols, force_retrain=False):
    """최초 접속 또는 캐시 만료 시 모델 로드 또는 강제 재학습 시 새 모델 학습"""
    os.makedirs("models", exist_ok=True)
    model_path = "models/kau_xgboost_model.json"

    # 최근 3개월 기준으로 학습/검증 분할 (Rolling Window)
    split_date = df["일자"].max() - pd.DateOffset(months=3)
    train = df[df["일자"] <= split_date]
    test = df[df["일자"] > split_date]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        early_stopping_rounds=50,
        random_state=42,
    )

    if os.path.exists(model_path) and not force_retrain:
        # 이미 모델이 존재하면 읽기 전용으로 로드 (앱 새로고침 시 중복 학습 방지)
        model.load_model(model_path)
    else:
        # 최초 실행 혹은 강제 전체 재학습
        model.fit(
            train[X_cols],
            train["target"],
            eval_set=(
                [(test[X_cols], test["target"])]
                if len(test) > 0
                else [(train[X_cols], train["target"])]
            ),
            verbose=False,
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)

    return model


# !SECTION - 학습 모델 가져오기


# ---------------------------------------------------------
# SECTION - 점진적 학습
# ---------------------------------------------------------
def incremental_train(model, new_df, X_cols):
    """명시적으로 업데이트 버튼을 눌렀을 때만 동작"""
    model_path = "models/kau_xgboost_model.json"

    # 기존 트리에 이어서 추가 부스팅
    model.fit(
        new_df[X_cols],
        new_df["target"],
        eval_set=[(new_df[X_cols], new_df["target"])],  # early_stopping 에러 방지
        xgb_model=model_path,
        verbose=False,
    )
    # 덮어쓰기 저장
    model.save_model(model_path)
    return model


# !SECTION - 점진적 학습


# ---------------------------------------------------------
# SECTION - KAU 예측
# ---------------------------------------------------------
def predict_kau(model, current_price, input_data):
    # X_cols로 지정된 특성만 필터링 (오류 방지)
    # X_cols는 전역 변수나 상수로 접근할 수 없으므로, 모델의 feature_names_in_를 사용
    features = (
        model.feature_names_in_
        if hasattr(model, "feature_names_in_")
        else input_data.columns
    )
    prediction_return = model.predict(input_data[features])[0]
    base_ai_price = current_price * (1 + prediction_return)

    # 하이브리드 보정
    AUCTION_PREMIUM = 400  # 5월 120만 톤 경매 (응찰률 1.8배) 선반영
    POLICY_PREMIUM = 200  # K-MSR 도입 대비 이월 목적 매수세 선반영

    # AI 예측가에 프리미엄을 얹어 최종 타겟 가격 산출
    final_target_price = base_ai_price + AUCTION_PREMIUM + POLICY_PREMIUM

    # 최종 가격과 변동률(보정된 최종 가격 기준) 반환
    final_return = (final_target_price - current_price) / current_price

    return final_target_price, final_return


# !SECTION - KAU 예측


# ---------------------------------------------------------
# SECTION - 시나리오 추천
# ---------------------------------------------------------
def get_recommendation(today_price, predicted_price, ma_5):
    pct_change = (predicted_price - today_price) / today_price * 100
    if predicted_price > today_price and today_price < ma_5:
        return "🟢 매수 추천", f"예측 상승률 {pct_change:.2f}% & 단기 저평가 구간"
    elif predicted_price < today_price:
        return "🔴 매수 대기", f"예측 하락률 {pct_change:.2f}% (가격 하락 예상)"
    else:
        return "⚪ 관망", "가격 변동 추세 미미"


# !SECTION - 시나리오 추천


# ---------------------------------------------------------
# SECTION - 예상 배출량 계산
# ---------------------------------------------------------
def calculate_emissions(op_rate, lng_ratio):
    """
    발전소 가동률(%)과 LNG 사용 비율(%)을 받아 예상 탄소 배출량과 부족분을 계산
    """
    MAX_GEN_MWH = 3000000  # 월 최대 발전 가능량 (예시)
    MONTHLY_ALLOWANCE = 1800000  # 월간 무상 할당량 (예시)

    # 단순화된 배출 계수 (tCO2 / MWh)
    COAL_EF = 0.82  # 석탄 배출계수
    LNG_EF = 0.39  # LNG 배출계수

    # 실제 발전량 산출
    actual_gen = MAX_GEN_MWH * (op_rate / 100.0)

    # 연료별 사용 비중
    coal_ratio_pct = 1.0 - (lng_ratio / 100.0)
    lng_ratio_pct = lng_ratio / 100.0

    # 총 예상 배출량 계산 = 발전량 * ((석탄비중 * 석탄계수) + (LNG비중 * LNG계수))
    total_emissions = actual_gen * (
        (coal_ratio_pct * COAL_EF) + (lng_ratio_pct * LNG_EF)
    )

    # 부족량 계산 (배출량이 할당량을 넘으면 부족)
    shortage = max(0, total_emissions - MONTHLY_ALLOWANCE)

    return total_emissions, shortage


# !SECTION - 예상 배출량 계산


# ---------------------------------------------------------
# SECTION - 모델 평가
# ---------------------------------------------------------
def evaluate_model(model, df, X_cols):
    split_date = pd.to_datetime("2026-07-01")
    test_data = df[df["일자"] > split_date]
    if len(test_data) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    predictions = model.predict(test_data[X_cols])

    actual_returns = test_data["target"]
    mae = mean_absolute_error(actual_returns, predictions)
    rmse = np.sqrt(mean_squared_error(actual_returns, predictions))
    r2 = r2_score(actual_returns, predictions)

    pred_price = test_data["lag_1"] * (1 + predictions)
    actual_price = test_data["종가"]
    mape = np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100

    return mae, rmse, r2, mape, len(test_data)


# !SECTION - 모델 평가


# ---------------------------------------------------------
# SECTION - 특성 기여도
# ---------------------------------------------------------
def get_feature_contributions(model, input_df, X_cols):
    """
    최신 데이터 샘플에 대한 SHAP 값을 계산하여 각 피처의 기여도 반환
    LLM 프롬프트에 주입할 정량적 XAI 기반 데이터로 활용
    """
    try:
        import shap

        # TreeExplainer 생성
        explainer = shap.TreeExplainer(model)
        # 2D 배열 혹은 단일 행 DataFrame에 대한 SHAP 값 계산
        shap_values = explainer.shap_values(input_df[X_cols])

        # shap_values가 2D 배열인 경우 (다수 샘플), 마지막 샘플(최신) 값만 가져옴
        if len(shap_values.shape) > 1:
            latest_shap = shap_values[-1]
        else:
            latest_shap = shap_values

        contributions = {}
        for i, col in enumerate(X_cols):
            contributions[col] = float(latest_shap[i])

        # 기여도 크기(절대값) 기준으로 내림차순 정렬
        sorted_contributions = dict(
            sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        )
        return sorted_contributions
    except ImportError:
        # shap 패키지가 없을 경우 Feature Importances를 대용으로 반환
        importances = model.feature_importances_
        contributions = {col: float(imp) for col, imp in zip(X_cols, importances)}
        return dict(
            sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        )
    except Exception as e:
        import logging

        logging.getLogger(__name__).error(f"SHAP 계산 중 오류: {e}")
        return {}


# !SECTION - 특성 기여도

import os
import requests
from typing import Dict, Any, Optional
import logging
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ExternalAPIClient:
    """
    각종 공공 데이터 API를 가져오는 클래스
    """

    def __init__(self):
        # API 키 로드
        self.data_go_kr_key = os.environ.get("DATA_GO_KR_KEY", "")  # 공공 데이터 포털
        self.ecos_key = os.environ.get("ECOS_KEY", "")  # ECOS
        self.kma_data_hub_key = os.environ.get(
            "KMA_DATA_HUB_KEY", ""
        )  # 기상청 API 허브
        self.kosis_api_key = os.environ.get("KOSIS_API_KEY", "")  # KOSIS API
        self.kepco_api_key = os.environ.get("KEPCO_DATA_API_KEY", "")  # 한전 API

        # 기본 타임아웃 등 세션 설정
        self.session = requests.Session()
        self.timeout = 30

    # ---------------------------------------------------------
    # 공공데이터포털 (DATA.GO.KR)
    # ---------------------------------------------------------

    """
    NOTE: 공공 데이터 포털 - 금융위원회_일반상품시세정보 (배출권)
    """

    def fetch_general_product_info_kau(
        self,
        begin_date: str = "20210101",
        end_date: str = None,
        page_no: int = 1,
        num_of_rows: int = 100,
    ) -> Dict[str, Any]:
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        try:
            logger.info("fetch_general_product_info_kau 호출")
            url = "https://apis.data.go.kr/1160100/service/GetGeneralProductInfoService/getCertifiedEmissionReductionPriceInfo"
            params = {
                "serviceKey": self.data_go_kr_key,
                "pageNo": page_no,
                "numOfRows": num_of_rows,
                "resultType": "json",
                "beginBasDt": begin_date,
                "endBasDt": end_date,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"배출권 데이터 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: 공공 데이터 포털 - 한국에너지공단_에너지사용 및 온실가스배출량 통계-산업부문
    """

    def fetch_ghg_emission_stat(
        self,
        year: str = "2020",
        ind_code: str = "C211",
        gas_type: str = "CO2",
        energy_type_name: str = "석유류",
        energy_name: str = "프로판",
        page_no: int = 1,
        num_of_rows: int = 10,
    ) -> Dict[str, Any]:
        try:
            logger.info("fetch_ghg_emission_stat 호출")
            url = "https://apis.data.go.kr/B553530/GHG_LIST_01/GHG_LIST_01_08_VIEW"
            params = {
                "ServiceKey": self.data_go_kr_key,
                "pageNo": page_no,
                "numOfRows": num_of_rows,
                "apiType": "JSON",
                "q1": year,
                "q2": ind_code,
                "q3": gas_type,
                "q4": energy_type_name,
                "q5": energy_name,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"온실가스배출량 통계 데이터 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: 공공 데이터 포털 - 한국전력거래소_발전원별 발전량
    """

    def fetch_power_gen_by_source(
        self, page_no: int = 1, num_of_rows: int = 100
    ) -> Dict[str, Any]:
        try:
            logger.info("fetch_power_gen_by_source 호출")
            url = "https://apis.data.go.kr/B552115/PvAmountByPwrGen/getPvAmountByPwrGen"
            params = {
                "serviceKey": self.data_go_kr_key,
                "pageNo": page_no,
                "numOfRows": num_of_rows,
                "dataType": "JSON",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"발전원별 발전량 데이터 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: 공공 데이터 포털 - 한국전력거래소_발전원별 발전량 현황조회 (실시간/단기) (출력값 = XML)
    """

    def fetch_power_gen_status_5m(self) -> Dict[str, Any]:
        try:
            logger.info("fetch_power_gen_status_5m 호출")
            url = "https://openapi.kpx.or.kr/openapi/sumperfuel5m/getSumperfuel5m"
            params = {
                "serviceKey": self.data_go_kr_key,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            # XML 고정 반환이므로 JSON 변환 시도 대신 텍스트 자체를 반환
            return {"status": "success", "dataType": "XML", "data": response.text}
        except Exception as e:
            logger.error(f"발전원별 발전량 현황 데이터 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: 공공 데이터 포털 - 한국전력거래소_계통한계가격 및 수요예측(하루전 발전계획용)
    """

    def fetch_smp_and_demand_forecast(
        self, target_date: str = "20210101", page_no: int = 1, num_of_rows: int = 100
    ) -> Dict[str, Any]:
        try:
            logger.info("fetch_smp_and_demand_forecast 호출")
            url = "https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand"
            params = {
                "serviceKey": self.data_go_kr_key,
                "pageNo": page_no,
                "numOfRows": num_of_rows,
                "dataType": "JSON",
                "date": target_date,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"계통한계가격 및 수요예측 데이터 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: 공공 데이터 포털 - 한국전력거래소_전력수급예보조회
    """

    def fetch_power_supply_forecast_by_source(self) -> Dict[str, Any]:
        try:
            logger.info("fetch_power_supply_forecast_by_source 호출")
            url = (
                "https://openapi.kpx.or.kr/openapi/forecast1dMaxBaseDate/getForecast1dMaxBaseDate"
                f"?serviceKey={self.data_go_kr_key}"
            )
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            # XML 고정 반환이므로 JSON 변환 시도 대신 텍스트 자체를 반환
            return {"status": "success", "dataType": "XML", "data": response.text}
        except Exception as e:
            logger.error(f"전력수급예보조회 데이터 수집 중 오류 발생: {e}")
            return {}

    # """
    # NOTE: 공공 데이터 포털 - 한국전력거래소_전력수급예보조회
    # """

    # def fetch_power_supply_forecast_by_source(
    #     self, page_no: int = 1, num_of_rows: int = 100
    # ) -> Dict[str, Any]:
    #     try:
    #         logger.info("fetch_power_supply_forecast_by_source 호출")
    #         url = "https://apis.data.go.kr/B552115/forecast1dMaxBaseDate/getForecast1DMaxBaseDate"
    #         params = {
    #             "serviceKey": self.data_go_kr_key,
    #             "pageNo": page_no,
    #             "numOfRows": num_of_rows,
    #             "dataType": "json",
    #         }
    #     except Exception as e:
    #         logger.error(f"계통한계가격 및 수요예측 데이터 수집 중 오류 발생: {e}")
    #         return {}

    """
    NOTE: 공공 데이터 포털 - 한국동서발전(주)_연료원별 일별 발전량 현황 정보
    """

    def fetch_ewp_daily_power_gen(
        self, target_date: str = "20210101", page_no: int = 1, num_of_rows: int = 100
    ) -> Dict[str, Any]:
        try:
            logger.info("fetch_ewp_daily_power_gen 호출")
            url = "https://apis.data.go.kr/B552070/dilyDvlpService/oamsFile13"
            params = {
                "serviceKey": self.data_go_kr_key,
                "pageNo": page_no,
                "numOfRows": num_of_rows,
                "apiType": "JSON",
                "date": target_date,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"동서발전 연료원별 발전량 데이터 수집 중 오류 발생: {e}")
            return {}

    # ---------------------------------------------------------
    # 한국은행 경제통계시스템 (ECOS)
    # ---------------------------------------------------------

    """
    NOTE: ECOS - 1.3.1. 한국은행 기준금리 및 여수신금리 -> 한국은행 기준금리
    """

    def fetch_ecos_base_rate(
        self, cycle: str = "M", start_date: str = "202101", end_date: str = None
    ) -> Dict[str, Any]:
        if not end_date:
            end_date = datetime.now().strftime("%Y%m")
        try:
            logger.info("fetch_ecos_base_rate 호출")
            if not self.ecos_key:
                raise ValueError("ECOS_KEY is not set.")
            url = f"https://ecos.bok.or.kr/api/StatisticSearch/{self.ecos_key}/json/kr/1/10000/722Y001/{cycle}/{start_date}/{end_date}/0101000/"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"ECOS 기준금리 데이터 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: ECOS - 3.1.1.1. 주요국 통화의 대원화환율 -> 원/미국달러(매매기준율)
    """

    def fetch_ecos_exchange_rate(
        self, start_date: str = "20210101", end_date: str = None
    ) -> Dict[str, Any]:
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        try:
            logger.info("fetch_ecos_exchange_rate 호출")
            if not self.ecos_key:
                raise ValueError("ECOS_KEY is not set.")
            url = f"https://ecos.bok.or.kr/api/StatisticSearch/{self.ecos_key}/json/kr/1/10000/731Y001/D/{start_date}/{end_date}/0000001/"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"ECOS 환율 데이터 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: ECOS - 6.1.1.4. 기업경기조사(매출액가중 전망) BSI
    업종코드: 99988(전산업), C0000(제조업), Y9900(비제조업)
    BSI코드: BA(업황전망), BB(매출전망), BM(수출전망) 등
    """

    def fetch_ecos_business_survey_bsi(
        self,
        start_month: str = "202001",
        end_month: str = None,
        industry_code: str = "C0000",
        bsi_code: str = "BA",
    ) -> Dict[str, Any]:
        if not end_month:
            end_month = datetime.now().strftime("%Y%m")
        try:
            logger.info("fetch_ecos_business_survey_bsi 호출")
            if not self.ecos_key:
                raise ValueError("ECOS_KEY is not set.")
            url = f"https://ecos.bok.or.kr/api/StatisticSearch/{self.ecos_key}/json/kr/1/10000/512Y016/M/{start_month}/{end_month}/{industry_code}/{bsi_code}/"
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"ECOS BSI 데이터 수집 중 오류 발생: {e}")
            return {}

    # ---------------------------------------------------------
    # YAHOO FINANCE
    # ---------------------------------------------------------

    """
    NOTE: Yahoo Finance - KOSPI 200
    """

    def fetch_yahoo_kospi(self, start_date: str = "2021-01-01") -> Any:
        try:
            logger.info("fetch_yahoo_kospi 호출")
            ticker = "^KS200"
            data = yf.download(ticker, start=start_date)
            return data
        except Exception as e:
            logger.error(f"KOSPI 200 데이터 수집 중 오류 발생: {e}")
            return None

    """
    NOTE: Yahoo Finance - 글로벌 핵심 에너지원 가격
    """

    def fetch_yahoo_global_energy_prices(
        self, start_date: str = "2021-01-01"
    ) -> Dict[str, Any]:
        try:
            logger.info("fetch_yahoo_global_energy_prices 호출")
            # 1. 국제 석탄(Newcastle Coal): 'MTF=F'
            # 2. 국제 천연가스(Dutch TTF LNG): 'TTF=F'
            # 3. 국제 유가(Brent): 'BZ=F'
            tickers = ["MTF=F", "TTF=F", "BZ=F"]
            data = yf.download(tickers, start=start_date)
            return data
        except Exception as e:
            logger.error(f"글로벌 에너지 가격 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: Yahoo Finance - 유럽 탄소배출권(EUA) 선물 가격
    """

    def fetch_yahoo_eua_price(self, start_date: str = "2021-01-01") -> Any:
        try:
            logger.info("fetch_yahoo_eua_price 호출")
            ticker = "KE=F"
            data = yf.download(ticker, start=start_date)
            return data
        except Exception as e:
            logger.error(f"EUA 배출권 데이터 수집 중 오류 발생: {e}")
            return None

    # ---------------------------------------------------------
    # KOSIS (국가통계포털)
    # ---------------------------------------------------------

    """
    NOTE: KOSIS - 시도/산업별 광공업생산지수 (2020=100)
    """

    def fetch_kosis_mining_industry_index(
        self, start_month: str = "202001"
    ) -> Dict[str, Any]:
        try:
            logger.info("fetch_kosis_mining_industry_index 호출")
            url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
            params = {
                "method": "getList",
                "apiKey": self.kosis_api_key,
                "itmId": "T10+T11+T12+T20+T21+T22+",
                "objL1": "ALL",
                "objL2": "0+A+B+C+D+",
                "format": "json",
                "jsonVD": "Y",
                "prdSe": "M",
                "startPrdDe": start_month,
                "endPrdDe": datetime.now().strftime("%Y%m"),
                "orgId": "101",
                "tblId": "DT_1F02001",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"광공업생산지수 데이터 수집 중 오류 발생: {e}")
            return {}

    """
    NOTE: KOSIS - 제조업 생산능력 및 가동률 지수 (2020=100)
    """

    def fetch_kosis_manufacturing_index(
        self, start_month: str = "202001"
    ) -> Dict[str, Any]:
        try:
            logger.info("fetch_kosis_manufacturing_index 호출")
            url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
            params = {
                "method": "getList",
                "apiKey": self.kosis_api_key,
                "itmId": "T10+T20+T30+",
                "objL1": "ALL",
                "format": "json",
                "jsonVD": "Y",
                "prdSe": "M",
                "startPrdDe": start_month,
                "endPrdDe": datetime.now().strftime("%Y%m"),
                "orgId": "101",
                "tblId": "DT_1F32001",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(
                f"제조업 생산능력 및 가동률 지수 데이터 수집 중 오류 발생: {e}"
            )
            return {}

    # ---------------------------------------------------------
    # 전력데이터 개방 포털시스템 (KEPCO)
    # ---------------------------------------------------------

    """
    NOTE: 전력데이터 개방 포털시스템 - 산업분류별 전력사용량
    """

    def fetch_kepco_power_usage(
        self,
        year: str = "2020",
        month: str = "01",
        metro_cd: str = "11",
        city_cd: str = "110",
        biz_cd: str = "C",
    ) -> Dict[str, Any]:
        try:
            logger.info("fetch_kepco_power_usage 호출")
            url = "https://bigdata.kepco.co.kr/openapi/v1/powerUsage/industryType.do"
            params = {
                "year": year,
                "month": month,
                "metroCd": metro_cd,
                "cityCd": city_cd,
                "bizCd": biz_cd,
                "apiKey": self.kepco_api_key,
                "returnType": "json",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"산업분류별 전력사용량 데이터 수집 중 오류 발생: {e}")
            return {}

    # ---------------------------------------------------------
    # 기상청 API 허브 (KMA)
    # ---------------------------------------------------------

    """
    NOTE: 기상청 API 허브 - 종관기상관측(ASOS)
    """

    def fetch_kma_asos_data(
        self, start_date: str = "20210101", end_date: str = None
    ) -> Dict[str, Any]:
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        try:
            logger.info("fetch_kma_asos_data 호출")
            url = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
            params = {
                "authKey": self.kma_data_hub_key,
                "tm1": f"{start_date}0000",
                "tm2": f"{end_date}2359",
                "help": "0",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return {"status": "success", "data": response.text}
        except Exception as e:
            logger.error(f"종관기상관측(ASOS) 데이터 수집 중 오류 발생: {e}")
            return {}

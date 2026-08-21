import os
import requests
from typing import Dict, Any, Optional
import logging
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ExternalAPITestClient:
    """
    각종 공공 데이터 API를 하루치(혹은 1개월치)만 테스트로 가져오는 클래스
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

        # 테스트용 기준 날짜 세팅 (데이터 존재가 보장되는 최근 과거 특정일 사용)
        self.test_date = "20240502"  # 2024년 5월 2일 (평일)
        self.test_date_dash = "2024-05-02"
        self.test_month = "202405"
        self.test_year = "2023"

    # ---------------------------------------------------------
    # 공공데이터포털 (DATA.GO.KR)
    # ---------------------------------------------------------
    def fetch_general_product_info_kau(self) -> Dict[str, Any]:
        try:
            url = "https://apis.data.go.kr/1160100/service/GetGeneralProductInfoService/getCertifiedEmissionReductionPriceInfo"
            params = {
                "serviceKey": self.data_go_kr_key,
                "pageNo": 1,
                "numOfRows": 10,
                "resultType": "json",
                "beginBasDt": self.test_date,
                "endBasDt": self.test_date,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {
                    "error": "JSON 파싱 실패. 원본 데이터",
                    "raw_text": response.text,
                }
        except Exception as e:
            return {"error": str(e)}

    def fetch_ghg_emission_stat(self) -> Dict[str, Any]:
        try:
            url = "https://apis.data.go.kr/B553530/GHG_LIST_01/GHG_LIST_01_08_VIEW"
            params = {
                "ServiceKey": self.data_go_kr_key,
                "pageNo": 1,
                "numOfRows": 10,
                "apiType": "JSON",
                "q1": self.test_year,
                "q2": "C211",
                "q3": "CO2",
                "q4": "석유류",
                "q5": "프로판",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    def fetch_power_gen_by_source(self) -> Dict[str, Any]:
        try:
            url = "https://apis.data.go.kr/B552115/PvAmountByPwrGen/getPvAmountByPwrGen"
            params = {
                "serviceKey": self.data_go_kr_key,
                "pageNo": 1,
                "numOfRows": 10,
                "dataType": "JSON",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    # 이 API는 출력값이 XML로 고정되어 있음
    def fetch_power_gen_status_5m(self) -> Dict[str, Any]:
        try:
            url = "https://openapi.kpx.or.kr/openapi/sumperfuel5m/getSumperfuel5m"
            params = {
                "serviceKey": self.data_go_kr_key,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            # XML 고정 반환이므로 JSON 변환 시도 대신 텍스트 자체를 반환
            return {
                "status": "success",
                "dataType": "XML",
                "data": response.text[:1000],
            }  # 너무 길 수 있으므로 1000자 제한
        except Exception as e:
            return {"error": str(e)}

    def fetch_smp_and_demand_forecast(self) -> Dict[str, Any]:
        try:
            url = "https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand"
            params = {
                "serviceKey": self.data_go_kr_key,
                "pageNo": 1,
                "numOfRows": 10,
                "dataType": "JSON",
                "date": self.test_date,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    def fetch_ewp_daily_power_gen(self) -> Dict[str, Any]:
        try:
            url = "https://apis.data.go.kr/B552070/dilyDvlpService/oamsFile13"
            params = {
                "serviceKey": self.data_go_kr_key,
                "pageNo": 1,
                "numOfRows": 10,
                "apiType": "JSON",
                "date": self.test_date,
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    # ---------------------------------------------------------
    # 한국은행 경제통계시스템 (ECOS)
    # ---------------------------------------------------------

    def fetch_ecos_base_rate(self) -> Dict[str, Any]:
        try:
            url = f"https://ecos.bok.or.kr/api/StatisticSearch/{self.ecos_key}/json/kr/1/10/722Y001/M/{self.test_month}/{self.test_month}/0101000/"
            response = self.session.get(url, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    def fetch_ecos_exchange_rate(self) -> Dict[str, Any]:
        try:
            url = f"https://ecos.bok.or.kr/api/StatisticSearch/{self.ecos_key}/json/kr/1/10/731Y001/D/{self.test_date}/{self.test_date}/0000001/"
            response = self.session.get(url, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    def fetch_ecos_business_survey_bsi(self) -> Dict[str, Any]:
        try:
            url = f"https://ecos.bok.or.kr/api/StatisticSearch/{self.ecos_key}/json/kr/1/10/512Y016/M/{self.test_month}/{self.test_month}/C0000/BA/"
            response = self.session.get(url, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    # ---------------------------------------------------------
    # YAHOO FINANCE
    # ---------------------------------------------------------

    def fetch_yahoo_kospi(self) -> str:
        try:
            data = yf.download(
                "^KS200",
                start=self.test_date_dash,
                end=(
                    datetime.strptime(self.test_date_dash, "%Y-%m-%d")
                    + timedelta(days=1)
                ).strftime("%Y-%m-%d"),
            )
            return data.to_json()
        except Exception as e:
            return str(e)

    def fetch_yahoo_global_energy_prices(self) -> str:
        try:
            tickers = ["MTF=F", "TTF=F", "BZ=F"]
            data = yf.download(
                tickers,
                start=self.test_date_dash,
                end=(
                    datetime.strptime(self.test_date_dash, "%Y-%m-%d")
                    + timedelta(days=1)
                ).strftime("%Y-%m-%d"),
            )
            return data.to_json()
        except Exception as e:
            return str(e)

    def fetch_yahoo_eua_price(self) -> str:
        try:
            data = yf.download(
                "KE=F",
                start=self.test_date_dash,
                end=(
                    datetime.strptime(self.test_date_dash, "%Y-%m-%d")
                    + timedelta(days=1)
                ).strftime("%Y-%m-%d"),
            )
            return data.to_json()
        except Exception as e:
            return str(e)

    # ---------------------------------------------------------
    # KOSIS (국가통계포털)
    # ---------------------------------------------------------

    def fetch_kosis_mining_industry_index(self) -> Dict[str, Any]:
        try:
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
                "startPrdDe": self.test_month,
                "endPrdDe": self.test_month,
                "orgId": "101",
                "tblId": "DT_1F02001",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    def fetch_kosis_manufacturing_index(self) -> Dict[str, Any]:
        try:
            url = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
            params = {
                "method": "getList",
                "apiKey": self.kosis_api_key,
                "itmId": "T10+T20+T30+",
                "objL1": "ALL",
                "format": "json",
                "jsonVD": "Y",
                "prdSe": "M",
                "startPrdDe": self.test_month,
                "endPrdDe": self.test_month,
                "orgId": "101",
                "tblId": "DT_1F32001",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    # ---------------------------------------------------------
    # 전력데이터 개방 포털시스템 (KEPCO)
    # ---------------------------------------------------------

    def fetch_kepco_power_usage(self) -> Dict[str, Any]:
        try:
            url = "https://bigdata.kepco.co.kr/openapi/v1/powerUsage/industryType.do"
            params = {
                "year": self.test_month[:4],
                "month": self.test_month[4:],
                "metroCd": "11",
                "cityCd": "110",
                "bizCd": "C",
                "apiKey": self.kepco_api_key,
                "returnType": "json",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            try:
                return response.json()
            except Exception:
                return {"error": "JSON 파싱 실패", "raw_text": response.text}
        except Exception as e:
            return {"error": str(e)}

    # ---------------------------------------------------------
    # 기상청 API 허브 (KMA)
    # ---------------------------------------------------------

    def fetch_kma_asos_data(self) -> Dict[str, Any]:
        try:
            url = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
            params = {
                "authKey": self.kma_data_hub_key,
                "tm1": f"{self.test_date}0000",
                "tm2": f"{self.test_date}2359",
                "help": "0",
            }
            response = self.session.get(url, params=params, timeout=self.timeout)
            return {
                "status": "success",
                "data": response.text[:500],
            }  # 너무 길면 잘라서 리턴
        except Exception as e:
            return {"error": str(e)}

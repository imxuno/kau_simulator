import os
import json
import re
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class GeminiLLMEngine:
    def __init__(self):
        self.provider_name = "Google Gemini 3.6 Flash"
        import google.generativeai as genai

        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY가 환경변수에 설정되어 있지 않습니다.")
        else:
            genai.configure(api_key=self.api_key)

        self.system_prompt_report = """
        당신은 최고 수준의 한국 배출권 시장(K-ETS) 전문 분석가이자 퀀트 트레이더입니다.
        제공되는 정량적 데이터(시세, 에너지 가격 변동, 거시경제 지표)와 정성적 뉴스/이벤트를 분석하여,
        명확하고 논리적인 시장 브리핑을 한국어로 작성해야 합니다.

        규칙:
        1. 단순한 사실 나열을 피하고, '왜' 오르거나 내리는지에 대한 인사이트를 제공하십시오.
        2. 어조는 전문적이고 객관적인 '리포트 형태(경어체)'로 작성하십시오.
        3. 마크다운(Markdown) 포맷을 사용하여 가독성을 높이십시오.
        4. 데이터가 주어지면, 데이터 기반으로 수치를 인용하십시오.
        """

        self.system_prompt_json = """
        당신은 대한민국 온실가스 배출권(K-ETS) 수석 애널리스트입니다.
        이벤트가 주어지면 KAU 가격 변동률을 추정하고 반드시 JSON 형태로만 반환하세요.
        입력되는 데이터에 '학습된_비정형_보고서_내용'이 있다면, 해당 보고서의 정책 방향과 수급 전망을 최우선적으로 반영하여 가격 궤적(trajectory)과 변동률(adjustment_rate)을 산출해야 합니다.
        마크다운 코드블록(```json)이나 다른 텍스트는 절대 포함하지 마십시오.

        응답 형식:
        {
            "adjustment_rate": (float) 예측 가격 변동률 (예: 5% 상승은 0.05, 3% 하락은 -0.03),
            "impact_level": (string) "High", "Medium", "Low",
            "reasoning": (string) 이벤트가 KAU 수급 및 가격에 미치는 영향을 3문장 이내로 요약,
            "impacted_sectors": (list of strings) 주요 영향 산업군,
            "predicted_trajectory": (list of float) 해당 이벤트 발생 직후 30일간의 가격 변화 궤적 (시작가는 '현재 KAU 예측가', 마지막 날은 예상 가격 변동률이 반영된 최종가에 수렴하도록 총 30개의 가격 수치를 배열로 작성)
        }
        """

        try:
            self.model = genai.GenerativeModel(
                model_name="models/gemini-3.6-flash",
                system_instruction=self.system_prompt_report,
                generation_config=genai.GenerationConfig(temperature=0.2),
            )
            self.json_model = genai.GenerativeModel(
                model_name="models/gemini-3.6-flash",
                system_instruction=self.system_prompt_json,
                generation_config=genai.GenerationConfig(
                    temperature=0.1, response_mime_type="application/json"
                ),
            )
        except Exception as e:
            logger.error(f"GenerativeModel 초기화 중 오류: {e}")
            self.model = None
            self.json_model = None

    def analyze_market(self, context_data: Dict[str, Any]) -> str:
        if not self.api_key or not self.model:
            return "⚠️ GEMINI_API_KEY가 등록되지 않았거나 모델 초기화에 실패하여 분석을 수행할 수 없습니다."
        prompt = f"아래는 현재 배출권 시장 및 외부 변수의 데이터 스냅샷입니다.\n이 데이터를 바탕으로 현재 시장 상황을 2~3단락으로 분석해 주세요.\n\n데이터 스냅샷:\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"LLM API 호출 중 오류 발생: {e}")
            return f"분석 중 오류가 발생했습니다: {str(e)}"

    def analyze_market_stream(self, context_data: Dict[str, Any]):
        if not self.api_key or not self.model:
            yield "⚠️ GEMINI_API_KEY가 없거나 모델 초기화 실패"
            return
        prompt = f"아래는 현재 배출권 시장 및 외부 변수의 데이터 스냅샷입니다.\n이 데이터를 바탕으로 현재 시장 상황을 2~3단락으로 분석해 주세요.\n\n데이터 스냅샷:\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"
        try:
            response = self.model.generate_content(prompt, stream=True)
            for chunk in response:
                yield chunk.text
        except Exception as e:
            logger.error(f"LLM API 스트리밍 오류 발생: {e}")
            yield f"\n[오류 발생: {str(e)}]"

    def parse_scenario(
        self, event_text: str, base_price: int, context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not self.api_key or not self.json_model:
            return {"error": "API Key missing or json_model init failed"}
        user_prompt = f"현재 KAU 예측가: {base_price}원. 발생 이벤트: {event_text}"
        if context_data:
            user_prompt += f"\n\n현재 시장 데이터 스냅샷:\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"
        try:
            response = self.json_model.generate_content(user_prompt)
            return self._extract_json(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return {"error": "JSON 디코딩 실패"}
        except Exception as e:
            logger.error(f"시나리오 분석 중 오류 발생: {e}")
            return {"error": f"분석 오류: {str(e)}"}

    def generate_report_text(self, context_data: Dict[str, Any]) -> Dict[str, str]:
        if not self.api_key or not self.json_model:
            return {"error": "API Key missing"}
        prompt = f"""당신은 한국동서발전의 탄소배출권 전문 애널리스트입니다.
아래의 시장 데이터(XGBoost 예측값 및 현재 변수들)를 바탕으로 '월간 배출권 매매계획' 보고서의 텍스트 요약을 작성해주세요.

[현재 시장 데이터 스냅샷]
{json.dumps(context_data, ensure_ascii=False, indent=2)}

아래 JSON 형식에 맞춰서 내용을 반환해주세요. 문체는 반드시 '~함', '~예상됨', '~전망'과 같은 공문서 개조식(개요) 형태여야 합니다.

{{
    "market_trend": "최근 배출권 시장의 가격 동향 및 거래량 요약 (2~3문장)",
    "future_outlook": "AI 예측 모델(XGBoost)을 바탕으로 한 향후 가격 전망 (2~3문장)",
    "purchasing_strategy": "현재 상황에 따른 구체적인 배출권 확보(매수/매도/관망) 전략 제안 (2~3문장)"
}}"""
        try:
            response = self.json_model.generate_content(prompt)
            return self._extract_json(response.text)
        except Exception as e:
            logger.error(f"보고서 생성 실패: {e}")
            return {
                "market_trend": "시장 동향 텍스트 생성 실패",
                "future_outlook": "향후 전망 텍스트 생성 실패",
                "purchasing_strategy": "구매 전략 텍스트 생성 실패",
            }

    def summarize_report_text(self, report_text: str) -> str:
        prompt = f"다음은 K-ETS 관련 보고서의 텍스트입니다. 이 보고서가 향후 배출권 가격이나 수급에 미칠 핵심적인 시사점을 3~4문장으로 요약해 주세요.\n\n[보고서 내용]\n{report_text}"
        if not self.api_key or not self.model:
            return "API Key missing or model init failed"
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"보고서 요약 중 오류 발생: {e}")
            return f"보고서 요약 실패: {str(e)}"

    def _extract_json(self, text: str) -> Dict[str, Any]:
        # <think> 블록 제거 (Nemotron 등의 추론 과정 포함 시 대비)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(1).strip()
        else:
            clean_text = text.strip()
            start_idx = clean_text.find("{")
            end_idx = clean_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean_text = clean_text[start_idx : end_idx + 1]
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            return {"error": f"JSON 디코딩 실패: {e}"}


class NemotronLLMEngine:
    def __init__(self):
        self.provider_name = "NVIDIA Nemotron-3 Ultra (550B)"
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "NVIDIA Nemotron 모델을 사용하려면 'openai' 패키지가 필요합니다. (pip install openai)"
            )

        self.api_key = os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            logger.warning("NVIDIA_API_KEY가 환경변수에 설정되어 있지 않습니다.")

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1", api_key=self.api_key
        )
        self.model_name = "nvidia/nemotron-3-ultra-550b-a55b"

        self.system_prompt_report = """
        당신은 최고 수준의 한국 배출권 시장(K-ETS) 전문 분석가이자 퀀트 트레이더입니다.
        제공되는 정량적 데이터(시세, 에너지 가격 변동, 거시경제 지표)와 정성적 뉴스/이벤트를 분석하여,
        명확하고 논리적인 시장 브리핑을 한국어로 작성해야 합니다.

        규칙:
        1. 단순한 사실 나열을 피하고, '왜' 오르거나 내리는지에 대한 인사이트를 제공하십시오.
        2. 어조는 전문적이고 객관적인 '리포트 형태(경어체)'로 작성하십시오.
        3. 마크다운(Markdown) 포맷을 사용하여 가독성을 높이십시오.
        4. 데이터가 주어지면, 데이터 기반으로 수치를 인용하십시오.
        """

        self.system_prompt_json = """
        당신은 대한민국 온실가스 배출권(K-ETS) 수석 애널리스트입니다.
        이벤트가 주어지면 KAU 가격 변동률을 추정하고 반드시 JSON 형태로만 반환하세요.
        입력되는 데이터에 '학습된_비정형_보고서_내용'이 있다면, 해당 보고서의 정책 방향과 수급 전망을 최우선적으로 반영하여 가격 궤적(trajectory)과 변동률(adjustment_rate)을 산출해야 합니다.
        마크다운 코드블록(```json)이나 다른 텍스트는 절대 포함하지 마십시오.

        응답 형식:
        {
            "adjustment_rate": (float) 예측 가격 변동률 (예: 5% 상승은 0.05, 3% 하락은 -0.03),
            "impact_level": (string) "High", "Medium", "Low",
            "reasoning": (string) 이벤트가 KAU 수급 및 가격에 미치는 영향을 3문장 이내로 요약,
            "impacted_sectors": (list of strings) 주요 영향 산업군,
            "predicted_trajectory": (list of float) 해당 이벤트 발생 직후 30일간의 가격 변화 궤적 (시작가는 '현재 KAU 예측가', 마지막 날은 예상 가격 변동률이 반영된 최종가에 수렴하도록 총 30개의 가격 수치를 배열로 작성)
        }
        """

    def _call_api(self, system_prompt, user_prompt, temperature=0.2, stream=False):
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY is missing")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            stream=stream,
        )

    def analyze_market(self, context_data: Dict[str, Any]) -> str:
        prompt = f"아래는 현재 배출권 시장 및 외부 변수의 데이터 스냅샷입니다.\n이 데이터를 바탕으로 현재 시장 상황을 2~3단락으로 분석해 주세요.\n\n데이터 스냅샷:\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"
        try:
            completion = self._call_api(
                self.system_prompt_report, prompt, temperature=0.2
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM API 호출 중 오류 발생: {e}")
            return f"분석 중 오류가 발생했습니다: {str(e)}"

    def analyze_market_stream(self, context_data: Dict[str, Any]):
        prompt = f"아래는 현재 배출권 시장 및 외부 변수의 데이터 스냅샷입니다.\n이 데이터를 바탕으로 현재 시장 상황을 2~3단락으로 분석해 주세요.\n\n데이터 스냅샷:\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"
        try:
            completion = self._call_api(
                self.system_prompt_report, prompt, temperature=0.2, stream=True
            )
            for chunk in completion:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            logger.error(f"LLM API 스트리밍 오류 발생: {e}")
            yield f"\n[오류 발생: {str(e)}]"

    def parse_scenario(
        self, event_text: str, base_price: int, context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        user_prompt = f"현재 KAU 예측가: {base_price}원. 발생 이벤트: {event_text}"
        if context_data:
            user_prompt += f"\n\n현재 시장 데이터 스냅샷:\n{json.dumps(context_data, ensure_ascii=False, indent=2)}"
        try:
            completion = self._call_api(
                self.system_prompt_json, user_prompt, temperature=0.1
            )
            response_text = completion.choices[0].message.content
            return self._extract_json(response_text)
        except Exception as e:
            logger.error(f"시나리오 분석 중 오류 발생: {e}")
            return {"error": f"분석 오류: {str(e)}"}

    def generate_report_text(self, context_data: Dict[str, Any]) -> Dict[str, str]:
        system_prompt = "당신은 탄소배출권 시장 전문가입니다. 반드시 사용자가 요청한 JSON 형식으로만 응답해야 하며, 다른 텍스트나 마크다운은 포함하지 마세요."
        prompt = f"""당신은 한국동서발전의 탄소배출권 전문 애널리스트입니다.
아래의 시장 데이터(XGBoost 예측값 및 현재 변수들)를 바탕으로 '월간 배출권 매매계획' 보고서의 텍스트 요약을 작성해주세요.

[현재 시장 데이터 스냅샷]
{json.dumps(context_data, ensure_ascii=False, indent=2)}

아래 JSON 형식에 맞춰서 내용을 반환해주세요. 문체는 반드시 '~함', '~예상됨', '~전망'과 같은 공문서 개조식(개요) 형태여야 합니다.

{{
    "market_trend": "최근 배출권 시장의 가격 동향 및 거래량 요약 (2~3문장)",
    "future_outlook": "AI 예측 모델(XGBoost)을 바탕으로 한 향후 가격 전망 (2~3문장)",
    "purchasing_strategy": "현재 상황에 따른 구체적인 배출권 확보(매수/매도/관망) 전략 제안 (2~3문장)"
}}"""
        try:
            completion = self._call_api(system_prompt, prompt, temperature=0.1)
            response_text = completion.choices[0].message.content
            return self._extract_json(response_text)
        except Exception as e:
            logger.error(f"보고서 생성 실패: {e}")
            return {
                "market_trend": "시장 동향 텍스트 생성 실패",
                "future_outlook": "향후 전망 텍스트 생성 실패",
                "purchasing_strategy": "구매 전략 텍스트 생성 실패",
            }

    def summarize_report_text(self, report_text: str) -> str:
        prompt = f"다음은 K-ETS 관련 보고서의 텍스트입니다. 이 보고서가 향후 배출권 가격이나 수급에 미칠 핵심적인 시사점을 3~4문장으로 요약해 주세요.\n\n[보고서 내용]\n{report_text}"
        try:
            completion = self._call_api(
                self.system_prompt_report, prompt, temperature=0.2
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"보고서 요약 실패: {e}")
            return f"요약 중 오류 발생: {str(e)}"

    def _extract_json(self, text: str) -> Dict[str, Any]:
        # <think> 블록 제거 (Nemotron 등의 추론 과정 포함 시 대비)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(1).strip()
        else:
            clean_text = text.strip()
            start_idx = clean_text.find("{")
            end_idx = clean_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean_text = clean_text[start_idx : end_idx + 1]
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError as e:
            return {"error": f"JSON 디코딩 실패: {e}"}


def LLMEngine(provider=None):
    """팩토리 함수: 인자 또는 환경변수 LLM_PROVIDER 값에 따라 적절한 엔진 인스턴스를 반환합니다."""
    if not provider:
        provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    else:
        provider = provider.lower()

    if provider == "nvidia" or provider == "nemotron":
        logger.info("NVIDIA Nemotron 모델을 백엔드로 사용합니다.")
        return NemotronLLMEngine()
    else:
        logger.info("Google Gemini 모델을 백엔드로 사용합니다.")
        return GeminiLLMEngine()

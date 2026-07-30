# 탄소배출권 시뮬레이터

- 사전점검 테스트용 시뮬레이터
- 프레임워크: streamlit(python)

---

## 사용 API (필요한 API 키)

- Google AI Studio: aistudio.google.com
- KOSIS: kosis.kr
- 공공데이터포털: data.go.kr
- 한국은행 Open API 서비스(ECOS): ecos.bok.or.kr
- 전력데이터 개방 포털시스템: bigdata.kepco.co.kr
- 기상청 API허브: apihub.kma.go.kr

- Yahoo Finance: finance.yahoo.com (Key X)

---

### 프로젝트 클론

```bash
git clone https://github.com/imxuno/kau_simulataor.git
```

### 가상환경(venv) 생성

- python 버전: 3.10.19

```bash
python -m venv venv
```

### 가상환경 활성화

```bash
# 활성화
source venv/bin/activate

# 비활성화
source venv/bin/deactivate
```

### 패키지 설치

```bash
pip install -r requirements.txt
```

### Streamlit 앱 실행

```bash
streamlit run src/app/app.py
```

---

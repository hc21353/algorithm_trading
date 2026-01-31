# Finite MACD Strategy Optimizer
알고리듬 트레이딩 macd 최적화 과제 (3조)

전통적인 MACD의 한계를 보완하기 위해 Finite 시계열 방식의 MACD를 구현하고, KOSPI 지수 및 국내 개별종목에 최적화된 트레이딩 파라미터를 찾아내는 정밀 분석 도구입니다.

# 설치 및 실행 방법

1. 필수 라이브러리 설치
터미널(또는 CMD)을 열고 아래 명령어를 입력하여 필요한 패키지를 한 번에 설치합니다.
```
pip install pandas numpy yfinance matplotlib seaborn koreanize_matplotlib tqdm openpyxl numba scipy
```

핵심 라이브러리 설명:
yfinance: 야후 파이낸스에서 실시간 주가 데이터를 가져옵니다.
numba: Finite MACD의 복잡한 연산을 초고속으로 처리합니다.
koreanize_matplotlib: 차트 내 한글 깨짐을 자동으로 방지합니다.
tqdm: 최적화 진행률을 시각적으로 보여줍니다.

2. 실행 방법
터미널에서 파일이 있는 폴더로 이동합니다.
아래 명령어를 실행합니다.
```
python "{파일명}.py"
```

4. 주요 설정 변경
분석하고 싶은 종목이나 기간을 바꾸려면 코드 하단의 ``` if __name__ == "__main__": ```부분을 수정하세요.

KOSPI 지수: ``` ticker="^KS11" ```

KB금융: ``` ticker="105560.ks" ```

기간 설정: ``` start_date="2010-01-01", end_date="2026-01-25" ```

# Troubleshooting
1. Numba 관련 오류 
본 프로젝트는 Finite MACD 연산 속도를 극대화하기 위해 Numba 라이브러리를 사용합니다. 관련 오류가 발생할 경우 아래 명령어로 최신 버전을 설치해 주세요.
```
pip install --upgrade numba
```
3. OS별 한글 폰트 설정 (Mac/Windows/Linux)
차트 내 한글 깨짐 현상을 방지하기 위해 각 OS에 최적화된 폰트 설정을 자동으로 수행합니다.

- Apple Silicon (M1/M2/M3) 사용자: koreanize_matplotlib 패키지 외에도 시스템 내 AppleGothic 폰트를 자동으로 찾아 차트에 적용하도록 설계되었습니다.

- 기타 환경: Windows(Malgun Gothic), Linux(NanumBarunGothic) 모두 platform.system() 분기 처리를 통해 별도의 코드 수정 없이 실행 가능합니다.

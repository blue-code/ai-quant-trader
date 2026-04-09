@echo off
chcp 65001 >nul
echo ========================================
echo  AI Quant Trader - 환경 설치
echo ========================================
echo.

cd /d "%~dp0"

:: Python 버전 확인
python --version 2>nul
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)

:: venv 생성
if not exist ".venv" (
    echo [1/3] 가상환경 생성 중...
    python -m venv .venv
    echo       완료.
) else (
    echo [1/3] 가상환경 이미 존재함.
)

:: venv 활성화 및 패키지 설치
echo [2/3] 패키지 설치 중...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

echo [3/3] 설치 확인 중...
python -c "import pandas, numpy, openai, click, omegaconf; print('       핵심 패키지 정상')"
if errorlevel 1 (
    echo [오류] 일부 패키지 설치 실패. 로그를 확인하세요.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  설치 완료!
echo  실행: ideate.bat / backtest.bat / optimize.bat / trade.bat
echo ========================================
pause

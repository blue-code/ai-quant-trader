@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo ========================================
echo  전체 파이프라인 실행
echo  (전략 생성 → 백테스트 → 최적화 → 리포트)
echo ========================================

set UNIVERSE=%1
if "%UNIVERSE%"=="" set UNIVERSE=kospi_bluechip

set NUM=%2
if "%NUM%"=="" set NUM=10

echo  유니버스: %UNIVERSE%
echo  전략 수 : %NUM%
echo.

python -m src.main run --universe %UNIVERSE% --num-strategies %NUM%

echo.
pause

@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo ========================================
echo  백테스트 실행
echo ========================================

set STRATEGIES=%1
if "%STRATEGIES%"=="" set STRATEGIES=./results/ideas.json

set UNIVERSE=%2
if "%UNIVERSE%"=="" set UNIVERSE=kospi_bluechip

echo  전략 파일: %STRATEGIES%
echo  유니버스 : %UNIVERSE%
echo.

python -m src.main backtest --strategies %STRATEGIES% --universe %UNIVERSE% --start 2020-01-01 --end 2024-12-31

echo.
pause

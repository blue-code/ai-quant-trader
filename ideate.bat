@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo ========================================
echo  전략 생성 (Ideation)
echo ========================================

set UNIVERSE=%1
if "%UNIVERSE%"=="" set UNIVERSE=kospi_bluechip

set NUM=%2
if "%NUM%"=="" set NUM=10

echo  유니버스: %UNIVERSE%
echo  생성 수 : %NUM%
echo.

python -m src.main ideate --universe %UNIVERSE% --num-strategies %NUM% --output ./results/ideas.json

echo.
pause

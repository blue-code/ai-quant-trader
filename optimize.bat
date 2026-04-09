@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo ========================================
echo  트리 탐색 최적화 (전체 파이프라인)
echo ========================================

set UNIVERSE=%1
if "%UNIVERSE%"=="" set UNIVERSE=kospi_bluechip

set STRATEGIES=%2

echo  유니버스 : %UNIVERSE%
if not "%STRATEGIES%"=="" (
    echo  전략 파일: %STRATEGIES%
    python -m src.main optimize --universe %UNIVERSE% --strategies %STRATEGIES%
) else (
    echo  전략: 자동 생성
    python -m src.main optimize --universe %UNIVERSE%
)

echo.
pause

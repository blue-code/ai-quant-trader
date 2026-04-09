@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo ========================================
echo  트레이딩 실행
echo ========================================

set STRATEGY=%1
if "%STRATEGY%"=="" set STRATEGY=./results/best_strategy.json

set MODE=%2
if "%MODE%"=="" set MODE=paper

echo  전략 파일: %STRATEGY%
echo  모드     : %MODE%
echo.

if "%MODE%"=="live" (
    echo [경고] 실거래 모드입니다. 실제 주문이 실행됩니다.
    set /p CONFIRM="계속하시겠습니까? (y/n): "
    if /i not "%CONFIRM%"=="y" (
        echo 취소됨.
        pause
        exit /b 0
    )
)

python -m src.main trade --strategy %STRATEGY% --mode %MODE%

echo.
pause

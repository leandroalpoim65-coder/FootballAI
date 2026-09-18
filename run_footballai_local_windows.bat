@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo ============================================================
echo FootballAI - LOCAL
echo Sem GitHub Actions
echo Sem Turso para a app local
echo ============================================================
echo.

set "PY=.venv_goal_backtest\Scripts\python.exe"

if not exist "%PY%" (
    echo [1/3] A criar ambiente Python local...
    py -3.11 -m venv .venv_goal_backtest
    if errorlevel 1 (
        echo ERRO: Python 3.11 nao encontrado.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Ambiente Python ja existe.
)

echo [2/3] A validar/instalar dependencias...
"%PY%" -c "import numpy,pandas,joblib,sklearn,streamlit" >nul 2>nul
if errorlevel 1 (
    "%PY%" -m pip install --disable-pip-version-check -r requirements.txt
    if errorlevel 1 (
        echo ERRO ao instalar dependencias.
        pause
        exit /b 1
    )
)

echo [3/3] A abrir a dashboard...
start "" http://localhost:8501
"%PY%" -m streamlit run app.py --server.port 8501

pause

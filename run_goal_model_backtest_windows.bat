@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo FootballAI - Goal Model Multi-League Backtest (LOCAL)
echo Nao consome minutos GitHub Actions e nao escreve na Turso.
echo ============================================================
echo.

where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py -3"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERRO: Python nao foi encontrado no PATH.
        echo Instala Python 3.11+ e volta a executar este ficheiro.
        pause
        exit /b 1
    )
    set "PY=python"
)

if not exist ".venv_goal_backtest\Scripts\python.exe" (
    echo [1/4] A criar ambiente virtual isolado...
    %PY% -m venv .venv_goal_backtest
    if errorlevel 1 goto :fail
)

set "VPY=.venv_goal_backtest\Scripts\python.exe"

echo [2/4] A preparar dependencias minimas...
"%VPY%" -m pip install --disable-pip-version-check --quiet --upgrade pip
if errorlevel 1 goto :fail
"%VPY%" -m pip install --disable-pip-version-check --quiet numpy pandas
if errorlevel 1 goto :fail

echo [3/4] A validar o Goal Model e o backtest...
"%VPY%" -m py_compile ml\goal_model.py goal_model_multileague_backtest.py
if errorlevel 1 goto :fail

echo [4/4] A executar 3 epocas x 7 ligas em walk-forward...
echo Isto usa 2021/22 como development, 2022/23 como validation e 2023/24 como holdout final.
echo.
"%VPY%" goal_model_multileague_backtest.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo BACKTEST TERMINADO COM SUCESSO

echo Relatorios criados em: %CD%\reports

echo Principais ficheiros:
echo   goal_model_multileague_summary.json
echo   goal_model_multileague_probability_metrics.csv
echo   goal_model_multileague_holdout_strategy.csv
echo   goal_model_multileague_holdout_by_league.csv
echo ============================================================
start "" "%CD%\reports"
pause
exit /b 0

:fail
echo.
echo ============================================================
echo O BACKTEST PAROU COM ERRO.
echo Copia ou envia a mensagem que aparece acima para diagnosticarmos.
echo ============================================================
pause
exit /b 1

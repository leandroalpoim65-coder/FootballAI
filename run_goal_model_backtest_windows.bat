@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist "reports" mkdir "reports"

echo ============================================================
echo FootballAI - Goal Model Multi-League Backtest LOCAL v2
echo Nao consome GitHub Actions e nao escreve na Turso.
echo ============================================================
echo.

if not exist "goal_model_multileague_backtest.py" (
    echo ERRO: goal_model_multileague_backtest.py nao existe nesta pasta.
    echo Confirma que estas na raiz do repositorio FootballAI.
    goto :fail
)

if not exist "goal_model_multileague_backtest_local.py" (
    echo ERRO: falta goal_model_multileague_backtest_local.py.
    echo O repositorio local esta desatualizado. Faz git pull e volta a tentar.
    goto :fail
)

if not exist "ml\goal_model.py" (
    echo ERRO: ml\goal_model.py nao existe nesta pasta.
    goto :fail
)

set "PY_CMD="
py -3.11 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3.11"

if not defined PY_CMD (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo ERRO: Python 3.10 ou superior nao foi encontrado.
    echo Instala Python 3.11 e ativa a opcao Add Python to PATH.
    goto :fail
)

echo Python encontrado:
%PY_CMD% --version
if errorlevel 1 goto :fail

echo.
set "VENV=.venv_goal_backtest"
set "VPY=%VENV%\Scripts\python.exe"

if exist "%VPY%" (
    "%VPY%" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo Ambiente virtual antigo/corrompido. A recriar...
        rmdir /s /q "%VENV%"
    )
)

if not exist "%VPY%" (
    echo [1/4] A criar ambiente virtual isolado...
    %PY_CMD% -m venv "%VENV%"
    if errorlevel 1 goto :fail
)

echo [2/4] A instalar dependencias...
"%VPY%" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto :fail
"%VPY%" -m pip install --disable-pip-version-check numpy pandas requests certifi
if errorlevel 1 goto :fail

echo.
echo [3/4] A validar codigo...
"%VPY%" -m py_compile ml\goal_model.py goal_model_multileague_backtest.py goal_model_multileague_backtest_local.py
if errorlevel 1 goto :fail

echo.
echo [4/4] A executar 3 epocas x 7 ligas em walk-forward...
echo 2021/22 = development ^| 2022/23 = validation ^| 2023/24 = holdout final
echo Os CSV ficam em backtest_data_cache para nao repetir downloads.
echo.
"%VPY%" goal_model_multileague_backtest_local.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo BACKTEST TERMINADO COM SUCESSO
echo Relatorios: %CD%\reports
echo ============================================================
start "" "%CD%\reports"
pause
exit /b 0

:fail
echo.
echo ============================================================
echo O BACKTEST PAROU COM ERRO.
if exist "reports\goal_model_backtest_error.log" (
    echo.
    echo Ultimo diagnostico:
    echo ------------------------------------------------------------
    type "reports\goal_model_backtest_error.log"
    echo ------------------------------------------------------------
    echo Tambem ficou guardado em reports\goal_model_backtest_error.log
)
echo.
echo Envia-me uma foto desta janela ou o ficheiro de erro se continuar.
echo ============================================================
pause
exit /b 1

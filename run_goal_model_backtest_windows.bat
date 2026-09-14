@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "LAUNCHER_DIR=%~dp0"
set "REPO_DIR="

call :check_repo "%LAUNCHER_DIR%"
if defined REPO_DIR goto :repo_found

call :check_repo "%USERPROFILE%\FootballAI"
if defined REPO_DIR goto :repo_found
call :check_repo "%USERPROFILE%\Desktop\FootballAI"
if defined REPO_DIR goto :repo_found
call :check_repo "%USERPROFILE%\Documents\FootballAI"
if defined REPO_DIR goto :repo_found
call :check_repo "%USERPROFILE%\Downloads\FootballAI"
if defined REPO_DIR goto :repo_found
call :check_repo "%USERPROFILE%\OneDrive\Desktop\FootballAI"
if defined REPO_DIR goto :repo_found
call :check_repo "%USERPROFILE%\OneDrive\Documents\FootballAI"
if defined REPO_DIR goto :repo_found

where git >nul 2>nul
if not errorlevel 1 (
    for /f "usebackq delims=" %%R in (`git -C "%LAUNCHER_DIR%" rev-parse --show-toplevel 2^>nul`) do (
        call :check_repo "%%R"
        if defined REPO_DIR goto :repo_found
    )
)

echo A procurar automaticamente a pasta FootballAI no teu perfil...
for /f "usebackq delims=" %%R in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$f = Get-ChildItem -Path $env:USERPROFILE -Filter 'goal_model_multileague_backtest.py' -File -Recurse -ErrorAction SilentlyContinue ^| Select-Object -First 1; if ($f) { $f.DirectoryName }"`) do (
    call :check_repo "%%R"
    if defined REPO_DIR goto :repo_found
)

echo.
echo ============================================================
echo ERRO: nao encontrei uma copia completa do repositorio FootballAI.
echo O ficheiro que abriste esta sozinho ou fora da pasta do projeto.
echo.
echo Para executar o teste, a mesma pasta tem de conter:
echo   goal_model_multileague_backtest.py
echo   goal_model_multileague_backtest_local.py
echo   ml\goal_model.py
echo.
echo Solucao: atualiza/clona o repositorio FootballAI completo e executa este BAT la dentro.
echo ============================================================
pause
exit /b 1

:repo_found
cd /d "%REPO_DIR%"
if not exist "reports" mkdir "reports"

echo ============================================================
echo FootballAI - Goal Model Multi-League Backtest LOCAL v4
echo Nao consome GitHub Actions e nao escreve na Turso.
echo ============================================================
echo Repositorio encontrado: %CD%
echo.

if not exist "goal_model_multileague_backtest.py" (
    echo ERRO: goal_model_multileague_backtest.py nao existe nesta pasta.
    goto :fail
)
if not exist "goal_model_multileague_backtest_local.py" (
    echo ERRO: falta goal_model_multileague_backtest_local.py.
    echo Atualiza a copia local do FootballAI e volta a tentar.
    goto :fail
)
if not exist "ml\goal_model.py" (
    echo ERRO: ml\goal_model.py nao existe nesta pasta.
    goto :fail
)

set "PY_CMD="
python -c "import sys; print('FOOTBALLAI_OK' if sys.version_info[:2] in [(3,10),(3,11),(3,12),(3,13),(3,14),(3,15)] else 'NO')" 2>nul | findstr /x /c:"FOOTBALLAI_OK" >nul
if not errorlevel 1 set "PY_CMD=python"

if not defined PY_CMD (
    py -3.11 -c "print('FOOTBALLAI_OK')" 2>nul | findstr /x /c:"FOOTBALLAI_OK" >nul
    if not errorlevel 1 set "PY_CMD=py -3.11"
)

if not defined PY_CMD (
    py -3 -c "import sys; print('FOOTBALLAI_OK' if sys.version_info[:2] in [(3,10),(3,11),(3,12),(3,13),(3,14),(3,15)] else 'NO')" 2>nul | findstr /x /c:"FOOTBALLAI_OK" >nul
    if not errorlevel 1 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    echo ERRO: nao existe um runtime Python 3.10+ instalado.
    echo O launcher 'py' pode existir sem ter o Python instalado.
    echo.
    echo Executa este comando numa janela CMD:
    echo     py install 3.11
    echo.
    echo Depois confirma com:
    echo     py -3.11 --version
    echo.
    echo E volta a executar este BAT.
    goto :fail
)

echo Python encontrado e validado:
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

:check_repo
if exist "%~1\goal_model_multileague_backtest.py" (
    if exist "%~1\ml\goal_model.py" set "REPO_DIR=%~1"
)
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
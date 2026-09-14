@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "LAUNCHER_DIR=%~dp0"
set "REPO_DIR="

call :check_repo "%LAUNCHER_DIR%"
if defined REPO_DIR goto :repo_found

for /f "usebackq delims=" %%R in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$roots=@($env:USERPROFILE,'B:\'); foreach($root in $roots){ if(Test-Path $root){ $f=Get-ChildItem -Path $root -Filter 'goal_model_v6_home_o25_train.py' -File -Recurse -ErrorAction SilentlyContinue ^| Select-Object -First 1; if($f){$f.DirectoryName; break} } }"`) do (
    call :check_repo "%%R"
    if defined REPO_DIR goto :repo_found
)

echo ERRO: nao encontrei o repositorio FootballAI atualizado.
echo Faz download/atualizacao do repositorio e tenta novamente.
pause
exit /b 1

:repo_found
cd /d "%REPO_DIR%"
if not exist "reports" mkdir "reports"

echo ============================================================
echo FootballAI - V6 TRAINED HOME TEAM OVER 2.5
echo Expanding-window CV: 2021/22 a 2025/26
echo Fit final: todos os jogos completos ate 2025/26
echo A partir daqui a avaliacao valida e apenas prospetiva.
echo Nao usa GitHub Actions e nao escreve na Turso.
echo ============================================================
echo Repositorio: %CD%
echo.

set "VPY=.venv_goal_backtest\Scripts\python.exe"
if not exist "%VPY%" (
    echo Ambiente do backtest nao encontrado. A criar...
    py -3.11 -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo ERRO: Python 3.11 nao encontrado.
        goto :fail
    )
    py -3.11 -m venv .venv_goal_backtest
    if errorlevel 1 goto :fail
)

"%VPY%" -c "import numpy,pandas,requests,joblib,sklearn" >nul 2>nul
if errorlevel 1 (
    echo A instalar dependencias minimas...
    "%VPY%" -m pip install --disable-pip-version-check numpy pandas requests certifi joblib scikit-learn==1.9.0
    if errorlevel 1 goto :fail
)

echo [1/2] A validar os ficheiros...
"%VPY%" -m py_compile goal_model_v6_home_o25_train.py goal_model_v6_home_o25_research.py goal_model_v4_structural_research.py goal_model_v2_multileague_research.py goal_model_multileague_backtest_local.py ml\goal_model.py
if errorlevel 1 goto :fail

echo [2/2] A treinar o V6...
echo.
"%VPY%" goal_model_v6_home_o25_train.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo V6 TRAINED TERMINADO COM SUCESSO
echo Modelo criado:
echo   ml\goal_model_v6_home_o25_trained.pkl
echo Envia-me:
echo   reports\goal_model_v6_trained_summary.json
echo   reports\goal_model_v6_trained_oof.csv
echo ============================================================
start "" "%CD%\reports"
pause
exit /b 0

:check_repo
if exist "%~1\goal_model_v6_home_o25_train.py" (
    if exist "%~1\goal_model_v6_home_o25_research.py" (
        if exist "%~1\goal_model_v4_structural_research.py" (
            if exist "%~1\goal_model_v2_multileague_research.py" set "REPO_DIR=%~1"
        )
    )
)
exit /b 0

:fail
echo.
echo ============================================================
echo O TREINO V6 PAROU COM ERRO.
echo Envia-me uma foto desta janela.
echo ============================================================
pause
exit /b 1

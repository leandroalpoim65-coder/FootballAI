@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "LAUNCHER_DIR=%~dp0"
set "REPO_DIR="

call :check_repo "%LAUNCHER_DIR%"
if defined REPO_DIR goto :repo_found

for /f "usebackq delims=" %%R in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$roots=@($env:USERPROFILE,'B:\'); foreach($root in $roots){ if(Test-Path $root){ $f=Get-ChildItem -Path $root -Filter 'goal_model_home_o25_challenger_shadow_cloud.py' -File -Recurse -ErrorAction SilentlyContinue ^| Select-Object -First 1; if($f){$f.DirectoryName; break} } }"`) do (
    call :check_repo "%%R"
    if defined REPO_DIR goto :repo_found
)

echo ERRO: nao encontrei o repositorio FootballAI atualizado.
pause
exit /b 1

:repo_found
cd /d "%REPO_DIR%"

echo ============================================================
echo FootballAI - HOME TEAM O2.5 PROSPECTIVE SHADOW
echo Modelos: V2 / V5 / V6 Frozen / V6 Trained
echo Modos: STRICT + WARM
echo Horizonte default: 7 dias
echo As probabilidades ficam congeladas na primeira execucao.
echo ============================================================
echo Repositorio: %CD%
echo.

if not exist "ml\goal_model_v6_home_o25_trained.pkl" (
    echo ERRO: falta ml\goal_model_v6_home_o25_trained.pkl
    echo Executa primeiro run_goal_model_v6_train_windows.bat
    goto :fail
)

if "%TURSO_DATABASE_URL%"=="" (
    echo ERRO: TURSO_DATABASE_URL nao esta definida nesta janela do Windows.
    echo Nao coloques credenciais no GitHub. Define-as apenas localmente e volta a executar.
    goto :fail
)
if "%TURSO_AUTH_TOKEN%"=="" (
    echo ERRO: TURSO_AUTH_TOKEN nao esta definida nesta janela do Windows.
    echo Nao coloques credenciais no GitHub. Define-as apenas localmente e volta a executar.
    goto :fail
)

set "VPY=.venv_goal_backtest\Scripts\python.exe"
if not exist "%VPY%" (
    echo Ambiente local nao encontrado. A criar...
    py -3.11 -c "import sys" >nul 2>nul
    if errorlevel 1 goto :fail
    py -3.11 -m venv .venv_goal_backtest
    if errorlevel 1 goto :fail
)

"%VPY%" -c "import numpy,pandas,requests,joblib,sklearn,libsql" >nul 2>nul
if errorlevel 1 (
    echo A instalar dependencias...
    "%VPY%" -m pip install --disable-pip-version-check numpy pandas requests certifi joblib scikit-learn==1.9.0 libsql
    if errorlevel 1 goto :fail
)

if "%HOME_O25_SHADOW_HORIZON_DAYS%"=="" set "HOME_O25_SHADOW_HORIZON_DAYS=7"

echo [1/2] A validar o shadow...
"%VPY%" -m py_compile goal_model_home_o25_challenger_shadow_cloud.py goal_model_v6_home_o25_train.py goal_model_v6_home_o25_research.py goal_model_v4_structural_research.py goal_model_v2_multileague_research.py
if errorlevel 1 goto :fail

echo [2/2] A congelar/atualizar o shadow na Turso...
echo.
"%VPY%" goal_model_home_o25_challenger_shadow_cloud.py
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo SHADOW HOME O2.5 ATUALIZADO COM SUCESSO
echo Volta a executar antes dos jogos para atualizar as odds finais.
echo Depois dos resultados, volta a executar para liquidar os registos.
echo ============================================================
pause
exit /b 0

:check_repo
if exist "%~1\goal_model_home_o25_challenger_shadow_cloud.py" (
    if exist "%~1\goal_model_v6_home_o25_train.py" (
        if exist "%~1\goal_model_v6_home_o25_research.py" set "REPO_DIR=%~1"
    )
)
exit /b 0

:fail
echo.
echo ============================================================
echo O SHADOW PAROU COM ERRO.
echo Envia-me uma foto desta janela e eu corrijo o ponto exato.
echo ============================================================
pause
exit /b 1

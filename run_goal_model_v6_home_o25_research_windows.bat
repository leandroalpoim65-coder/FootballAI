@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "LAUNCHER_DIR=%~dp0"
set "REPO_DIR="

call :check_repo "%LAUNCHER_DIR%"
if defined REPO_DIR goto :repo_found

for /f "usebackq delims=" %%R in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$roots=@($env:USERPROFILE,'B:\'); foreach($root in $roots){ if(Test-Path $root){ $f=Get-ChildItem -Path $root -Filter 'goal_model_v6_home_o25_research.py' -File -Recurse -ErrorAction SilentlyContinue ^| Select-Object -First 1; if($f){$f.DirectoryName; break} } }"`) do (
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
echo FootballAI - Goal Model V6 HOME O2.5 SPECIALIST
echo Escolha de parametros: 2021/22 + 2022/23 + 2023/24 APENAS
echo Avaliacao: 2024/25 + 2025/26 + stress 2020/21
echo Inclui bootstrap de logloss V6 vs V2 e V6 vs V5.
echo Nao usa GitHub Actions e nao escreve na Turso.
echo ============================================================
echo Repositorio: %CD%
echo.

set "VPY=.venv_goal_backtest\Scripts\python.exe"
if not exist "%VPY%" (
    echo Ambiente do backtest nao encontrado. A criar...
    py -3.11 -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo ERRO: Python 3.11 nao encontrado. Executa: py install 3.11
        goto :fail
    )
    py -3.11 -m venv .venv_goal_backtest
    if errorlevel 1 goto :fail
)

"%VPY%" -c "import numpy,pandas,requests" >nul 2>nul
if errorlevel 1 (
    echo A instalar dependencias minimas...
    "%VPY%" -m pip install --disable-pip-version-check numpy pandas requests certifi
    if errorlevel 1 goto :fail
)

echo [1/3] A validar os ficheiros V6...
"%VPY%" -m py_compile goal_model_v6_home_o25_research.py goal_model_v4_structural_research.py goal_model_v2_multileague_research.py goal_model_multileague_backtest_local.py ml\goal_model.py
if errorlevel 1 goto :fail

echo [2/3] A testar calibracao especialista Home O2.5...
echo Sao 125 configuracoes pequenas + bootstrap final.
echo Pode demorar alguns minutos. Nao feches esta janela.
echo.
"%VPY%" goal_model_v6_home_o25_research.py
if errorlevel 1 goto :fail

echo [3/3] Teste terminado.
echo.
echo ============================================================
echo V6 TERMINADO COM SUCESSO
echo Envia-me da pasta reports:
echo   goal_model_v6_home_o25_summary.json
echo   goal_model_v6_home_o25_metrics.csv
echo ============================================================
start "" "%CD%\reports"
pause
exit /b 0

:check_repo
if exist "%~1\goal_model_v6_home_o25_research.py" (
    if exist "%~1\goal_model_v4_structural_research.py" (
        if exist "%~1\goal_model_v2_multileague_research.py" (
            if exist "%~1\ml\goal_model.py" set "REPO_DIR=%~1"
        )
    )
)
exit /b 0

:fail
echo.
echo ============================================================
echo O TESTE V6 PAROU COM ERRO.
echo Envia-me uma foto desta janela para corrigirmos o ponto exato.
echo ============================================================
pause
exit /b 1

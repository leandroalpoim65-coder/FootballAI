@'
# ============================================================
# FOOTBALLAI - EXECUCAO AUTOMATICA
# ============================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogsDirectory = Join-Path $ProjectRoot "logs"
$LockFile = Join-Path $LogsDirectory "footballai_automation.lock"

if (-not (Test-Path -LiteralPath $LogsDirectory)) {
    New-Item -ItemType Directory -Path $LogsDirectory -Force | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogsDirectory "footballai_$Timestamp.log"


function Write-Log {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    $Line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"

    Write-Host $Line

    Add-Content `
        -LiteralPath $LogFile `
        -Value $Line `
        -Encoding UTF8
}


function Find-Python {
    $PythonCommand = Get-Command `
        python.exe `
        -ErrorAction SilentlyContinue

    if ($null -ne $PythonCommand) {
        return @{
            Executable = $PythonCommand.Source
            Arguments = @()
        }
    }

    $PyCommand = Get-Command `
        py.exe `
        -ErrorAction SilentlyContinue

    if ($null -ne $PyCommand) {
        return @{
            Executable = $PyCommand.Source
            Arguments = @("-3")
        }
    }

    throw "Python nao foi encontrado no sistema."
}


function Invoke-PythonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $ScriptPath = Join-Path $ProjectRoot $RelativePath

    if (-not (Test-Path -LiteralPath $ScriptPath)) {
        throw "Ficheiro nao encontrado: $ScriptPath"
    }

    Write-Log "A executar: $RelativePath"

    $CommandArguments = @()

    foreach ($Argument in $PythonArguments) {
        $CommandArguments += $Argument
    }

    $CommandArguments += $ScriptPath

    & $PythonExecutable @CommandArguments 2>&1 |
        Tee-Object `
            -LiteralPath $LogFile `
            -Append

    $PythonExitCode = $LASTEXITCODE

    if ($PythonExitCode -ne 0) {
        throw (
            "O ficheiro $RelativePath terminou " +
            "com o codigo $PythonExitCode."
        )
    }

    Write-Log "Concluido: $RelativePath"
}


$ExitCode = 0

try {
    Write-Log "========================================"
    Write-Log "FOOTBALLAI - INICIO"
    Write-Log "========================================"

    Set-Location -LiteralPath $ProjectRoot

    if (Test-Path -LiteralPath $LockFile) {
        $LockItem = Get-Item -LiteralPath $LockFile

        $LockAge = (
            (Get-Date) - $LockItem.LastWriteTime
        )

        if ($LockAge.TotalHours -gt 2) {
            Write-Log "Bloqueio antigo encontrado."

            Remove-Item `
                -LiteralPath $LockFile `
                -Force
        }
        else {
            Write-Log "Ja existe uma execucao ativa."
            exit 0
        }
    }

    Set-Content `
        -LiteralPath $LockFile `
        -Value $PID `
        -Encoding ASCII

    $PythonInformation = Find-Python

    $PythonExecutable = $PythonInformation.Executable
    $PythonArguments = @($PythonInformation.Arguments)

    Write-Log "Projeto: $ProjectRoot"
    Write-Log "Python: $PythonExecutable"

    Invoke-PythonFile `
        -RelativePath "download_upcoming.py"

    Invoke-PythonFile `
        -RelativePath "ml\paper_trade.py"

    Write-Log "========================================"
    Write-Log "FOOTBALLAI - EXECUCAO CONCLUIDA"
    Write-Log "========================================"
}
catch {
    $ExitCode = 1

    Write-Log "ERRO: $($_.Exception.Message)"
    Write-Log "Log: $LogFile"
}
finally {
    if (Test-Path -LiteralPath $LockFile) {
        Remove-Item `
            -LiteralPath $LockFile `
            -Force `
            -ErrorAction SilentlyContinue
    }
}


'@ | Set-Content -LiteralPath ".\run_footballai.ps1" -Encoding ASCII

exit $ExitCode
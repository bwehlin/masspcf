@echo off
:: Must be run with: call remove_cuda.bat

set "NEW_PATH="
setlocal EnableDelayedExpansion
for %%i in ("%PATH:;=" "%") do (
    echo %%~i | findstr /i /c:"cuda" >nul 2>&1
    if errorlevel 1 (
        if defined NEW_PATH (
            set "NEW_PATH=!NEW_PATH!;%%~i"
        ) else (
            set "NEW_PATH=%%~i"
        )
    )
)
endlocal & set "PATH=%NEW_PATH%"

set "CUDA_PATH="
set "CUDA_HOME="
set "CUDA_ROOT="
set "CUDA_BIN_PATH="
set "CUDADIR="
set "NVTOOLSEXT_PATH="

:: Remove versioned CUDA_PATH variables (e.g. CUDA_PATH_V12_3)
for /f "tokens=1 delims==" %%v in ('set CUDA_PATH_ 2^>nul') do set "%%v="

echo Done. CUDA removed from environment for this session.

@echo off
setlocal enabledelayedexpansion
:: Usage: collect_sources.bat [dir1] [dir2] ... [/o output_file]
:: Example: collect_sources.bat src include .github /o collected.txt
:: Defaults: dir=. output=collected_sources.txt

set OUTPUT=collected_sources.txt
set DIRS=

:parse_args
if "%~1"=="" goto done_parsing
if /i "%~1"=="/o" (
    set OUTPUT=%~2
    shift
    shift
    goto parse_args
)
set DIRS=!DIRS! "%~1"
shift
goto parse_args

:done_parsing
if "!DIRS!"=="" set DIRS=.

type nul > "%OUTPUT%"

for %%D in (!DIRS!) do (
    echo Scanning %%D...
    for /f "delims=" %%F in ('dir /b /s /a-d %%D 2^>nul ^| findstr /i "\.cpp$ \.h$ \.cu$ \.tpp$ \.cmake$ \.yml$ \.yaml$" ^| findstr /v /i "\\build\\ \.git\\ \\3rd\\ \\benchmarking\\ \\examples\\ \\docs\\ \\cmake-build\\"') do (
        echo ======================================== >> "%OUTPUT%"
        echo FILE: %%F >> "%OUTPUT%"
        echo ======================================== >> "%OUTPUT%"
        type "%%F" >> "%OUTPUT%"
        echo. >> "%OUTPUT%"
        echo FILE: %%F
    )
)

echo.
echo Collected into %OUTPUT%
find /c /v "" "%OUTPUT%"
endlocal
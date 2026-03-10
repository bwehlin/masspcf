@echo off
call make.bat html
if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b %ERRORLEVEL%
)
start "" "_build\html\index.html"

@echo off
cd /d "%~dp0"
if exist "%~dp0..\..\work\venv\Scripts\python.exe" (
  "%~dp0..\..\work\venv\Scripts\python.exe" Go_Project.py demo
) else (
  python Go_Project.py demo
)
if errorlevel 1 pause

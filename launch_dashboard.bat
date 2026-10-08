@echo off
title AI Dynamic Pricing Server
cd /d "%~dp0"
echo ========================================================
echo Starting AI Dynamic Pricing Dashboard on port 8000...
echo ========================================================
start "" http://localhost:8000
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
pause

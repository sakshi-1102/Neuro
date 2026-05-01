@echo off
echo Starting NeuroVoice Backend...
cd C:\Users\Admin\PROJECTS\Python\neuro
call venv\Scripts\activate.bat
venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
pause
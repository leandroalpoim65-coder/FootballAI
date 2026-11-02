@echo off
cd /d "%~dp0"

call .venv\Scripts\activate.bat

start "" http://localhost:8503

python -m streamlit run app_cloud_v3.py --server.port 8503

pause
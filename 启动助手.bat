@echo off
echo 正在启动你的专属 AI 助手...
cd /d %~dp0
python -m streamlit run app.py
pause
@echo off
cd /d %~dp0

call Scripts\activate

echo Starting Smart Attendance System...
streamlit run app.py

pause

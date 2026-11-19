@echo off
REM run.bat — ARES one-command launcher (Windows). Phase 3, Task 5.
setlocal

cd /d "%~dp0"

set PY=python
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
    echo Using .venv\Scripts\python.exe
) else (
    echo WARNING: .venv\ not found. Using python from PATH.
)

for %%F in (
    models\raw_xgb.joblib
    results\model_manifest.json
    results\model_selection.json
    results\pca_report.json
    results\calculus_experiment.json
) do (
    if not exist "%%F" (
        echo ERROR: required artifact missing: %%F 1>&2
        exit /b 1
    )
)

echo Verifying model manifest...
%PY% scripts\model_manifest.py --check
if errorlevel 1 (
    echo ERROR: model manifest does not match on-disk artifacts. 1>&2
    exit /b 1
)

%PY% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERROR: streamlit not installed. Run: %PY% -m pip install streamlit 1>&2
    exit /b 1
)

echo Launching ARES (Streamlit)...
%PY% -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
endlocal
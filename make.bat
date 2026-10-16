@echo off
REM Cross-platform-friendly wrapper so `make <target>` semantics work on Windows
REM without GNU Make installed. Usage: make.bat setup|test|demo|eval|booklets|exam|exam-kb|clean
setlocal
set VENV=.venv
set PYV=%VENV%\Scripts\python.exe

if "%1"=="setup" (
    py -m venv %VENV%
    %PYV% -m pip install --upgrade pip
    %PYV% -m pip install -e ".[dev,ui,nlp]"
    goto :eof
)
if "%1"=="test" (
    %PYV% -m pytest
    goto :eof
)
if "%1"=="demo" (
    %PYV% -m streamlit run app\streamlit_app.py
    goto :eof
)
if "%1"=="eval" (
    %PYV% -m eval.run_eval
    goto :eof
)
if "%1"=="booklets" (
    %PYV% tools\make_sample_booklets.py
    goto :eof
)
if "%1"=="exam-kb" (
    %PYV% tools\build_exam_kb.py com226
    goto :eof
)
if "%1"=="exam" (
    %PYV% -m ees.exam data\exam\com226\sample_student_script.txt
    goto :eof
)
if "%1"=="clean" (
    rmdir /s /q .pytest_cache 2>nul
    for /d /r %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
    goto :eof
)
echo Unknown target: %1
echo Usage: make.bat setup^|test^|demo^|eval^|booklets^|exam^|exam-kb^|clean

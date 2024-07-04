@echo off
for /F "tokens=1* delims==" %%a in (.env) do (
    setx %%a "%%b"
)
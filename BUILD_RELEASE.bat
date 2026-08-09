@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ================================================
echo  cHiDeScaler-Neo v343g Release Build
echo ================================================
echo.

where cargo >nul 2>nul
if errorlevel 1 (
  echo [ERROR] cargo was not found.
  echo Install Rust with rustup, then reopen this window.
  pause
  exit /b 1
)

if not exist Cargo.toml (
  echo [ERROR] Cargo.toml was not found in this folder.
  pause
  exit /b 1
)

echo [1/3] Formatting source...
cargo fmt --package chidescaler-neo
if errorlevel 1 (
  echo [ERROR] cargo fmt failed.
  pause
  exit /b 1
)

echo [2/3] Building release executable...
cargo build --release --bin chidescaler-neo
if errorlevel 1 (
  echo.
  echo [ERROR] Build failed. Copy the first error through the final error.
  pause
  exit /b 1
)

if not exist "target\release\chidescaler-neo.exe" (
  echo [ERROR] target\release\chidescaler-neo.exe was not generated.
  pause
  exit /b 1
)

echo [3/3] Copying executable to the application folder...
if exist "cHiDeScaler-Neo.exe" copy /y "cHiDeScaler-Neo.exe" "cHiDeScaler-Neo_previous.exe" >nul
copy /y "target\release\chidescaler-neo.exe" "cHiDeScaler-Neo.exe" >nul
if errorlevel 1 (
  echo [ERROR] Could not replace cHiDeScaler-Neo.exe.
  echo Close the running application and try again.
  pause
  exit /b 1
)

echo.
echo [SUCCESS] %~dp0cHiDeScaler-Neo.exe
pause

@echo off
echo Starting build...
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" > nul 2>&1
if errorlevel 1 (
    echo ERROR: Could not initialize VS environment
    exit /b 1
)
echo VS environment loaded
cd /d "C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja\src\csharp\SKIENinjaML"
echo Building project...
msbuild SKIENinjaML.csproj /p:Configuration=Release /v:normal
echo Build complete with exit code %errorlevel%

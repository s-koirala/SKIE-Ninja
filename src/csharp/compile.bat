@echo off
setlocal

set CSC=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe
set SRCDIR=C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja\src\csharp\SKIENinjaML
set PKGDIR=C:\Users\skoir\Documents\SKIE Enterprises\SKIE-Ninja\SKIE-Ninja-Project\SKIE_Ninja\src\csharp\packages
set OUTDIR=%SRCDIR%\bin\Release
set NETSTD=C:\Program Files\dotnet\packs\NETStandard.Library.Ref\2.1.0\ref\netstandard2.1\netstandard.dll

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Compiling SKIENinjaML.dll...
"%CSC%" /target:library /out:"%OUTDIR%\SKIENinjaML.dll" ^
    /reference:"%PKGDIR%\Microsoft.ML.OnnxRuntime.Managed.1.16.3\lib\netstandard2.0\Microsoft.ML.OnnxRuntime.dll" ^
    /reference:"%PKGDIR%\Newtonsoft.Json.13.0.3\lib\net45\Newtonsoft.Json.dll" ^
    "%SRCDIR%\SKIENinjaPredictor.cs"

if %errorlevel% equ 0 (
    echo Build succeeded!
    echo Output: %OUTDIR%\SKIENinjaML.dll
) else (
    echo Build failed with error %errorlevel%
)

endlocal

@echo off
rem Apply the Ornith-1.5 (qwen35moe) patches to a Strata v0.2.9 source tree, on plain Windows.
rem
rem   apply.cmd "C:\path\to\Strata"
rem
rem One consolidated diff, cut against the released v0.2.9 source (engine 0.1.55, commit c62d5e3a).
rem patches\legacy-0.1.54 contains the older five-diff set, which targets the pre-release tree and does NOT apply
rem to v0.2.9; it is kept for history only.
setlocal
set "HERE=%~dp0"
set "TREE=%~1"
if "%TREE%"=="" set /p TREE=Strata folder (the one with src\ and include\): 
if "%TREE%"=="" goto :usage
if not exist "%TREE%\src\qwen35" goto :notree

where patch >nul 2>&1
if errorlevel 1 goto :nopython

pushd "%TREE%"
patch -p1 --forward --batch --no-backup-if-mismatch < "%HERE%patches\01-ornith-qwen35-0.2.9.diff"
rem --kv-grow (v0.2.9-r2): the KV holds VRAM only as far as the context reaches.
patch -p1 --forward --batch --no-backup-if-mismatch < "%HERE%patches\02-kv-grow.diff"

if errorlevel 1 (
    popd
    echo.
    echo FAILED: this tree is not v0.2.9 (engine 0.1.55).  Fix the hunk against the new source, never the patch.
    exit /b 1
)
popd
echo.
echo patched: %TREE%
goto :next

:nopython
echo patch.exe (Git for Windows) is not on PATH.  Install Git for Windows, or apply the diff with any tool you have.
exit /b 1

:next
echo.
echo Build (what the released engine was built with):
echo   cmake -S "%TREE%" -B "%TREE%\build-qwen35" -G Ninja -DCMAKE_BUILD_TYPE=Release
echo     -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
echo     -DSTRATA_Q6K_EXPERTS=ON
echo     -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"
echo     -DSTRATA_GGML_DIR="C:\path\to\pinned\llama.cpp"
echo   ninja -C "%TREE%\build-qwen35" strata strata_kq
exit /b 0

:usage
echo usage: apply.cmd "C:\path\to\Strata"
exit /b 2

:notree
echo %TREE%: no src\qwen35 in it - point at the folder that holds src\ and include\.
exit /b 2

@echo off
rem Apply the Ornith engine patches to a Strata source tree, on plain Windows.
rem
rem   apply.cmd "C:\path\to\Strata"
rem
rem Uses Git's patch.exe when it is on PATH, otherwise the bundled Python applier. Either way the three
rem diffs go in order: 01 adds the dequant include that the others extend.
setlocal enabledelayedexpansion
set "HERE=%~dp0"
set "TREE=%~1"
if "%TREE%"=="" set /p TREE=Strata folder (the one with src\ and include\): 
if "%TREE%"=="" goto :usage
if not exist "%TREE%\src\qwen35" goto :notree

where patch >nul 2>&1
if errorlevel 1 goto :python

for %%F in ("%HERE%patches\*.diff") do (
    echo == %%~nxF
    pushd "%TREE%"
    patch -p1 --forward --batch --no-backup-if-mismatch < "%%~fF"
    if errorlevel 1 (
        popd
        echo.
        echo FAILED: %%~nxF -- this tree is probably a newer Strata than the patches were cut for.
        echo Fix that hunk against the new source, do not patch the patch.
        exit /b 1
    )
    popd
)
echo.
echo patched: %TREE%
goto :next

:python
where python >nul 2>&1
if errorlevel 1 (
    echo Neither patch.exe (Git for Windows) nor python is on PATH.
    echo Install either one, or use a prebuilt engine instead of patching the source.
    exit /b 1
)
python "%HERE%tools\apply_patch.py" "%TREE%" "%HERE%patches\01-pr3925-kquant-ssm.diff" "%HERE%patches\02-pr3903-kv-modes.diff" "%HERE%patches\03-qwen35moe-kernels.diff" "%HERE%patches\04-vram-headroom.diff" "%HERE%patches\05-shared-card.diff"
if errorlevel 1 exit /b 1
goto :next

:next
echo.
echo Build (MSVC 2022 + CUDA 13, adjust the two paths):
echo   cmake -S "%TREE%" -B "%TREE%\build-qwen35" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
echo     -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF ^
echo     -DSTRATA_Q6K_EXPERTS=ON -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" ^
echo     -DSTRATA_GGML_DIR="C:\path\to\pinned\llama.cpp"
echo   ninja -C "%TREE%\build-qwen35" strata strata_kq
exit /b 0

:usage
echo usage: apply.cmd "C:\path\to\Strata"
exit /b 2

:notree
echo %TREE%: no src\qwen35 in it -- point at the folder that holds src\ and include\.
exit /b 2

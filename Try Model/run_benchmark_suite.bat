@echo off
echo ==========================================================
echo Starting Sequential Benchmark Suite Execution
echo ==========================================================

cd /d "d:\New folder (2)\Data\OneDrive_2025-12-13\Try Model"

echo.
echo [1/8] Running mr_tae_ultimate.py...
call myenv\Scripts\python.exe ..\mr_tae_ultimate.py

echo.
echo [2/8] Running compare_methods.py...
call myenv\Scripts\python.exe ..\compare_methods.py

echo.
echo [3/8] Running train_extended.py (root)...
call myenv\Scripts\python.exe ..\train_extended.py

echo.
echo [4/8] Running examples/train_and_visualize.py...
call myenv\Scripts\python.exe examples\train_and_visualize.py

echo.
echo [5/8] Running examples/train_demo.py...
call myenv\Scripts\python.exe examples\train_demo.py

echo.
echo [6/8] Running examples/train_extended.py...
call myenv\Scripts\python.exe examples\train_extended.py

echo.
echo [7/8] Running examples/train_extended_500.py...
call myenv\Scripts\python.exe examples\train_extended_500.py

echo.
echo [8/8] Running examples/train_improved.py...
call myenv\Scripts\python.exe examples\train_improved.py

echo.
echo ==========================================================
echo ALL TRAINING AND BENCHMARKING COMPLETE
echo Check outputs/benchmark/results_summary.csv for full metrics.
echo ==========================================================
pause

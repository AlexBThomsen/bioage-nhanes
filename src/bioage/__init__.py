import os

# xgboost and torch each ship their own OpenMP runtime; with >1 thread they deadlock or segfault on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

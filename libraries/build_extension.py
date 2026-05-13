"""
Compiles xgboost_scratch.cpp as a Python extension module (xgboost_cpp).
Run from the project root with the venv activated and g++ on PATH:

    cd libraries
    python build_extension.py

The output .pyd file is placed in the notebooks/ folder so the notebook
can import it directly without any sys.path changes.
"""

import subprocess
import sys
import sysconfig
import os
import shutil

# ---- Paths ----------------------------------------------------------------

import pybind11

pybind_inc  = pybind11.get_include()
python_inc  = sysconfig.get_path("include")
ext_suffix  = sysconfig.get_config_var("EXT_SUFFIX")   # e.g. .cp311-win_amd64.pyd
output_name = f"xgboost_cpp{ext_suffix}"

# Python .lib lives in the base installation, not the venv
base_prefix = getattr(sys, "real_prefix", sys.base_prefix)
python_libs = os.path.join(base_prefix, "libs")
python_lib  = f"python{sys.version_info.major}{sys.version_info.minor}"

src   = os.path.join(os.path.dirname(__file__), "xgboost_scratch.cpp")
out   = os.path.join(os.path.dirname(__file__), output_name)

# ---- Compile ---------------------------------------------------------------

cmd = [
    r"C:\msys64\ucrt64\bin\g++",
    "-O3", "-march=native", "-fopenmp", "-std=c++17",
    "-shared",                      # build shared library (.pyd / .so)
    "-DXGB_EXTENSION",              # enables pybind11 block, disables main()
    f"-I{pybind_inc}",
    f"-I{python_inc}",
    src,
    f"-o{out}",
    f"-L{python_libs}",
    f"-l{python_lib}",
    "-Wl,--enable-auto-import",     # Windows: allow implicit DLL symbol imports
]

print("Building:", output_name)
print("Command :", " ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("--- stderr ---")
    print(result.stderr)
    sys.exit(1)

# Copy to notebooks/ so it can be imported directly from the notebook
notebooks_dir = os.path.join(os.path.dirname(__file__), "..", "notebooks")
dest = os.path.join(notebooks_dir, output_name)
shutil.copy2(out, dest)

print(f"Success: {out}")
print(f"Copied : {dest}")
print()
print("In your notebook:")
print("    from xgboost_cpp import XGBoostCpp")
print("    model = XGBoostCpp(n_estimators=100, learning_rate=0.1, max_depth=3)")
print("    model.fit(X_train, y_train)")
print("    y_pred = model.predict(X_test)")

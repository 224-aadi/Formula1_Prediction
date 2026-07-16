import os
import sys
import ctypes
import glob
import site

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

for site_packages in site.getsitepackages():
    for libgomp_path in glob.glob(os.path.join(site_packages, "*.libs", "libgomp*.so*")):
        ctypes.CDLL(libgomp_path, mode=ctypes.RTLD_GLOBAL)
        break

# Add the src directory to the python path so `f1outcome` can be imported by Vercel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from f1outcome.api.app import app

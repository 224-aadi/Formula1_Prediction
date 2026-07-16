import os
import sys
import ctypes
import glob
import site

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

for site_packages in [*site.getsitepackages(), *sys.path]:
    if not site_packages:
        continue
    for libgomp_path in glob.glob(os.path.join(site_packages, "*.libs", "libgomp*.so*")):
        for link_dir in [
            os.path.dirname(libgomp_path),
            os.path.join(site_packages, "lightgbm", "lib"),
        ]:
            try:
                os.makedirs(link_dir, exist_ok=True)
                link_path = os.path.join(link_dir, "libgomp.so.1")
                if not os.path.exists(link_path):
                    os.symlink(libgomp_path, link_path)
            except OSError:
                pass

        ctypes.CDLL(libgomp_path, mode=ctypes.RTLD_GLOBAL)
        break

# Add the src directory to the python path so `f1outcome` can be imported by Vercel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from f1outcome.api.app import app

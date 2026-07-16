import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

# Add the src directory to the python path so `f1outcome` can be imported by Vercel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from f1outcome.runtime import prepare_lightgbm_runtime

prepare_lightgbm_runtime()

from f1outcome.api.app import app

import faulthandler; faulthandler.enable()
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")   # kept from the original WSL fix; JIT stays at its default (enabled)

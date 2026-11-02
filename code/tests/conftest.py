"""Make `pytest` work from the repository root as well as from code/ (modules live in code/)."""
import os
import sys

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

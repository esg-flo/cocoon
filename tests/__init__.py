import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.joinpath("src")
sys.path.insert(0, str(REPO_ROOT))

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent
print(ROOT_PATH)
sys.path.insert(0, ROOT_PATH)

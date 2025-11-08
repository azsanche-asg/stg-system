import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / 'stg-stsg-model' / 'src'))

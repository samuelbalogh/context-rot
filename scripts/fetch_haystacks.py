import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import load_env
from src.haystack.pg_essays import fetch_all as fetch_pg
from src.haystack.arxiv import fetch_all as fetch_arxiv
from src.haystack.oss_code import fetch_all as fetch_oss_code

load_env()
fetch_pg()
fetch_arxiv()
fetch_oss_code()

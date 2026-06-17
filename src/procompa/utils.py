import os
from pathlib import Path
    
from dotenv import load_dotenv

def get_project_root() -> Path:
    """Returns the root directory of the project."""
    return Path(__file__).parent.parent.parent

def get_data_dir() -> str:
    load_dotenv()
    return Path(os.getenv("DATA_DIR_POOLED_PPI_YEASTMAP"))
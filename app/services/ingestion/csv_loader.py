import pandas as pd
from pathlib import Path
from app.core.logger import get_logger
from app.core.constants import ARXIV_CSV_PATH

logger = get_logger(__name__)


class CSVLoader:
    def __init__(self, path: Path = ARXIV_CSV_PATH):
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV not found at {self.path}")
        logger.info(f"Loading CSV from {self.path}")
        df = pd.read_csv(self.path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        required_columns = ["paper_id", "title", "abstract", "authors", "primary_category"]
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        logger.info("CSV validation passed")
        return True
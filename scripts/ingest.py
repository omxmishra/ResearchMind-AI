import pandas as pd
import pickle
from pathlib import Path
from app.core.logger import get_logger
from app.core.constants import ARXIV_CSV_PATH, PROCESSED_DATA_DIR

logger = get_logger(__name__)


def load_raw_data() -> pd.DataFrame:
    logger.info(f"Loading raw data from {ARXIV_CSV_PATH}")
    df = pd.read_csv(ARXIV_CSV_PATH)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def inspect_data(df: pd.DataFrame):
    logger.info("Running data inspection")
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nSample row:\n{df.iloc[0].to_dict()}")


def save_raw_snapshot(df: pd.DataFrame):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "raw_snapshot.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(df, f)
    logger.info(f"Saved raw snapshot to {output_path}")


if __name__ == "__main__":
    df = load_raw_data()
    inspect_data(df)
    save_raw_snapshot(df)
    logger.info("Ingestion complete")
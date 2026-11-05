"""
Robust Data Loader and Missing Value Cleaning Utilities
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_data(file_path: str | Path) -> Optional[pd.DataFrame]:
    """Load a dataset from a CSV or Parquet file."""
    path = Path(file_path)

    if not path.is_file():
        logger.error("File not found: %s", path)
        return None

    try:
        if path.suffix.lower() == ".parquet":
            logger.info("Loading Parquet file: %s", path)
            df = pd.read_parquet(path, engine="pyarrow")
        elif path.suffix.lower() in {".csv", ".gz"}:
            logger.info("Loading CSV file: %s", path)
            df = pd.read_csv(path, compression="infer")
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        logger.info("Loaded %d rows × %d columns", df.shape[0], df.shape[1])
        return df

    except Exception as exc:
        logger.exception("Failed to load %s: %s", path, exc)
        raise


def clean_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_value: Optional[dict] = None,
) -> pd.DataFrame:
    """Simple missing-value handling."""
    if strategy == "drop":
        logger.info("Dropping rows with any missing values")
        return df.dropna()

    if strategy == "mean":
        logger.info("Filling numeric columns with column mean")
        return df.fillna(df.mean())

    if strategy == "median":
        logger.info("Filling numeric columns with column median")
        return df.fillna(df.median())

    if strategy == "mode":
        logger.info("Filling columns with mode")
        return df.fillna(df.mode().iloc[0])

    if strategy == "custom":
        if not fill_value:
            raise ValueError("`fill_value` dict required for custom strategy")
        logger.info("Filling columns with custom values")
        return df.fillna(fill_value)

    raise ValueError(f"Unknown missing-value strategy: {strategy}")
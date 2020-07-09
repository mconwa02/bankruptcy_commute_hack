import os
import pandas as pd

from config import SHORTEN_NAMES, DATA_DIR
from utils import get_configured_logger

logger = get_configured_logger(__name__)


def read_data(file_name):
    df = pd.read_csv(os.path.join(DATA_DIR, file_name))
    df = df.set_index("company_id")
    df = df.fillna(0)
    df.columns = SHORTEN_NAMES.values()
    return df


if __name__ == "__main__":
    df = read_data("train_dataset.csv")
    logger.info(df.columns)
    logger.info(df.info)
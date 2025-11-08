"""Project configuration: paths, model parameters, train/test split ratios, etc."""
from pathlib import Path

BASE_DIR = Path("/tmp")  # Change as needed
DATA_URL = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
DATA_LOCAL_PATH = BASE_DIR / "iris.csv"
OUTPUT_DIR = BASE_DIR / "output"

TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
SEED = 42

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABEL_COLUMN = "species"

LR_PARAMS = {"maxIter": 20, "regParam": 0.1, "elasticNetParam": 0.0}
GBT_PARAMS = {"maxIter": 50, "maxDepth": 5, "stepSize": 0.2}

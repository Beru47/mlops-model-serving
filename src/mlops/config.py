"""Settings read from the environment at call time (keeps tests and containers simple)."""
import os

FEATURES = ["med_inc", "house_age", "avg_rooms", "population", "latitude", "longitude"]
TARGET = "price"
HOLDOUT_SEED = 1234
HOLDOUT_SIZE = 2000
STAGING = "staging"
PRODUCTION = "production"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def model_name() -> str:
    return os.environ.get("MODEL_NAME", "housing-regressor")

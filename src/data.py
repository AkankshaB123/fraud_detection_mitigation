import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def load_and_clean_data(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["data"]["raw_path"])

    if config["preprocessing"]["drop_duplicates"]:
        df = df.drop_duplicates()

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

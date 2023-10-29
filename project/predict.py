import pandas as pd
from surprise import AlgoBase

import os

DATA_DIR = "data"
PREDICTIONS_DIR = "predictions"


def predict(svd_model: AlgoBase, filename: str) -> None:
    X = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    predictions = pd.DataFrame({}, columns=["Id", "rating"])

    for Id, (user_id, item_id) in enumerate(X.values):
        predictions.loc[-1] = [Id, svd_model.predict(user_id, item_id).est]
        predictions.index = predictions.index + 1
    predictions = predictions.set_index("Id").sort_index()
    predictions.index = predictions.index.astype(int)
    predictions.to_csv(os.path.join(PREDICTIONS_DIR, filename))

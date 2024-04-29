"""Create predictions with a fitted model"""

import argparse
import os
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

PREDITIONS_PATH = pathlib.Path("./predictions")


def _make_predictions(test_data: pd.DataFrame, model: Pipeline) -> np.ndarray:
    return model.predict(test_data)


def main():
    """Run and create the CLI"""

    # Create CLI
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "-d", "--data", required=True, help="Data used to make predictions."
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Model that will be use to make predictions",
    )
    parser.add_argument(
        "-s", "--save", required=True, help="Filename to save the predictions"
    )

    # Get values from the CLI
    args = parser.parse_args()

    # Get the model and load test dataset
    df_test = pd.read_csv(args.data)
    model = joblib.load(args.model)

    # Make predictions
    print("Making predictions...")
    preds = _make_predictions(df_test, model)

    # Save the predictions
    df_preds = pd.DataFrame({"Id": df_test.Id, "Osteoporosis": preds})

    if not PREDITIONS_PATH.exists():
        os.mkdir(PREDITIONS_PATH)

    print(f"Saving predictions to {args.save}.csv")
    df_preds.to_csv(f"predictions/{args.save}.csv")


if __name__ == "__main__":
    main()

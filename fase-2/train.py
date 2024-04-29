"""Fit a model with training data given by user"""

import argparse
import os
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODELS_PATH = pathlib.Path("./models")


def _preprocess_train(df: pd.DataFrame) -> Pipeline:
    """Preprocess and fits the model with the training data.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataset.

    Returns
    -------
    Pipeline
        Pipeline of preprocessing steps and fitted model
    """

    # Split data into features and target
    X = df.drop("Osteoporosis", axis=1)
    y = df["Osteoporosis"]

    # Create the preprocessing process
    num_cols = [col for col in X.columns if X[col].dtype in ["float", "int"]]
    cat_cols = [col for col in X.columns if X[col].dtype == "object"]

    num_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    # Define the model
    model = RandomForestClassifier()

    # Create the pipeline
    pipe = Pipeline(steps=[("prep", preprocessor), ("rdf", model)])

    # Fit the pipeline
    print("Fitting model...")
    pipe.fit(X, y)

    # Compute model's accuracy
    results = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    print(f"Model's Accuracy: {np.mean(results)}")

    return pipe


def main():
    """Runs the program and the CLI"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data", required=True, help="Dataset to train the model with."
    )

    parser.add_argument(
        "-f", "--model-file", help="Filename to save the model.", required=True
    )

    # Get the arguments from the CLI
    args = parser.parse_args()

    # Load data and get model
    df = pd.read_csv(args.data)
    model = _preprocess_train(df)

    if not MODELS_PATH.exists():
        os.mkdir(MODELS_PATH)

    # Save the model
    print(f"Saving model to models/{args.model_file}.joblib")
    joblib.dump(model, f"models/{args.model_file}.joblib")


if __name__ == "__main__":
    main()

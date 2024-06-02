import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def preprocess_train(df: pd.DataFrame):
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
    X = df.drop(["Osteoporosis", "Id"], axis=1)
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
    pipe.fit(X, y)

    # Compute model's accuracy
    results = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    avg_acc = np.mean(results)

    return pipe, avg_acc
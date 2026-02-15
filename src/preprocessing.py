from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


def split_features_target(df):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def train_test_split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor(X):
    # Columns where zero is invalid
    zero_invalid_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    # Replace zero with NaN for these columns
    X[zero_invalid_columns] = X[zero_invalid_columns].replace(0, np.nan)

    numeric_features = X.columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ]
    )

    return preprocessor

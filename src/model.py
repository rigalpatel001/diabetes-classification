from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_regression():
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

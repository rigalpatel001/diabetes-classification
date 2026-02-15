from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def train_model(preprocessor, model, X_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate_model(preprocessor, model, X, y):
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="roc_auc"
    )

    return scores.mean(), scores.std()


def tune_logistic_regression(preprocessor, X_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 50, 100],
        # "model__penalty": ["l2"],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="recall",   # 🔥 Focus on recall
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_


def train_with_smote(preprocessor, model, X_train, y_train):
    smote = SMOTE(random_state=42)

    pipeline = ImbPipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("smote", smote),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline

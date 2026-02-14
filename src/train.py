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

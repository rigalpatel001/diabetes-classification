from src.data_loader import load_dataset
from src.preprocessing import (
    split_features_target,
    train_test_split_data,
    build_preprocessor,
)
from src.model import get_logistic_regression, get_random_forest
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    df = load_dataset("data/raw/diabetes.csv")

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    preprocessor = build_preprocessor(X_train)

    # Logistic Regression
    lr = get_logistic_regression()
    lr_model = train_model(preprocessor, lr, X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)

    print("\nLogistic Regression Results:")
    for k, v in lr_metrics.items():
        print(f"{k}: {v}")

    # Random Forest
    rf = get_random_forest()
    rf_model = train_model(preprocessor, rf, X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    print("\nRandom Forest Results:")
    for k, v in rf_metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

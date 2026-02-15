from src.evaluate import plot_roc_curve
from src.data_loader import load_dataset
from src.preprocessing import (
    split_features_target,
    train_test_split_data,
    build_preprocessor,
)
from src.model import get_logistic_regression, get_random_forest
from src.train import train_model
from src.evaluate import evaluate_model
from src.threshold import evaluate_with_threshold
from src.train import cross_validate_model
from src.train import tune_logistic_regression
from src.train import train_with_smote
from src.explain import explain_model
from src.save_load import save_model


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

    print("\n--- Threshold Optimization (Logistic Regression) ---")

    for t in [0.5, 0.45, 0.4, 0.35, 0.3]:
        metrics = evaluate_with_threshold(
            lr_model, X_test, y_test, threshold=t)

        print(f"\nThreshold: {t}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1: {metrics['f1']:.3f}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

    mean_auc, std_auc = cross_validate_model(preprocessor, lr, X, y)

    print("\nCross-Validation ROC-AUC (Logistic Regression):")
    print(f"Mean AUC: {mean_auc:.3f}")
    print(f"Std: {std_auc:.3f}")

    print("\n--- Recall-Focused Hyperparameter Tuning ---")

    best_model, best_params = tune_logistic_regression(
        preprocessor, X_train, y_train)

    print("Best Parameters:", best_params)

    tuned_metrics = evaluate_model(best_model, X_test, y_test)

    print("\nTuned Logistic Regression Results:")
    for k, v in tuned_metrics.items():
        print(f"{k}: {v}")

    print("\n--- Logistic Regression with SMOTE ---")

    from sklearn.linear_model import LogisticRegression

    smote_model = train_with_smote(
        preprocessor,
        LogisticRegression(class_weight=None, max_iter=1000),
        X_train,
        y_train,
    )

    smote_metrics = evaluate_model(smote_model, X_test, y_test)

    for k, v in smote_metrics.items():
        print(f"{k}: {v}")

    # Random Forest
    rf = get_random_forest()
    rf_model = train_model(preprocessor, rf, X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)

    print("\nRandom Forest Results:")
    for k, v in rf_metrics.items():
        print(f"{k}: {v}")

    plot_roc_curve(lr_model, X_test, y_test)

    print("\n--- Generating SHAP Explanation ---")

    # Use small sample for explanation
    explain_model(lr_model, X_test)

    save_model(lr_model, "model.joblib")
    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()

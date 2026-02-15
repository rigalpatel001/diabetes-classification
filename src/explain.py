import shap
import matplotlib.pyplot as plt


def explain_model(model, X_sample):
    """
    Generate SHAP explanation for a trained pipeline model.
    """

    # Extract trained logistic regression model
    logistic_model = model.named_steps["model"]

    # Transform input data using preprocessing
    X_transformed = model.named_steps["preprocessing"].transform(X_sample)

    explainer = shap.LinearExplainer(logistic_model, X_transformed)
    shap_values = explainer(X_transformed)

    shap.summary_plot(shap_values, X_transformed, show=True)

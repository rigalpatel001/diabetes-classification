# import numpy as np

# def apply_custom_threshold(model, X, threshold=0.5):
#     probabilities = model.predict_proba(X)[:, 1]
#     return np.where(probabilities >= threshold, 1, 0)

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    probabilities = model.predict_proba(X_test)[:, 1]
    y_pred_custom = np.where(probabilities >= threshold, 1, 0)

    precision = precision_score(y_test, y_pred_custom)
    recall = recall_score(y_test, y_pred_custom)
    f1 = f1_score(y_test, y_pred_custom)
    cm = confusion_matrix(y_test, y_pred_custom)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }

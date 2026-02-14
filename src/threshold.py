import numpy as np

def apply_custom_threshold(model, X, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]
    return np.where(probabilities >= threshold, 1, 0)

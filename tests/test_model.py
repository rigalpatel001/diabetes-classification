import pandas as pd
from src.save_load import load_model

def test_prediction_output():
    model = load_model("model.joblib")

    sample = pd.DataFrame(
        [[6, 148, 72, 35, 0, 33.6, 0.627, 50]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    )

    pred = model.predict(sample)

    assert pred[0] in [0, 1]

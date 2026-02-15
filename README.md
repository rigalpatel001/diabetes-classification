# 🩺 Diabetes Risk Prediction System  
### End-to-End Machine Learning Pipeline with Interpretability & Deployment

---

## 📌 Project Overview

This project implements a **production-ready machine learning system** to predict diabetes risk using clinical health indicators.

The system is designed with a healthcare mindset:

- Minimize False Negatives (missed diabetic patients)
- Handle class imbalance properly
- Ensure model interpretability
- Provide a REST API for real-world deployment

---

## 🎯 Business Objective

In healthcare screening, **missing a diabetic patient (False Negative)** is more dangerous than a false alarm.

Therefore, this system prioritizes:

- High Recall
- Controlled Precision
- Stable cross-validation performance

---

## 📊 Dataset

**Pima Indians Diabetes Dataset**

Features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:
- Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## ⚙️ Machine Learning Pipeline

### 1️⃣ Data Cleaning
- Replaced medically impossible zeros with NaN
- Median imputation using `SimpleImputer`
- Standardization using `StandardScaler`

### 2️⃣ Class Imbalance Handling
- Used `class_weight="balanced"` in Logistic Regression
- Evaluated SMOTE (but class_weight performed better)

### 3️⃣ Model Training
- Logistic Regression
- Random Forest (baseline comparison)

### 4️⃣ Threshold Optimization
Instead of default 0.5 threshold, optimized to reduce False Negatives.

Best operational threshold ≈ **0.35–0.40**

### 5️⃣ Cross-Validation
- 5-Fold Stratified Cross-Validation
- ROC-AUC Mean ≈ 0.83
- Stable performance (low std)

### 6️⃣ Hyperparameter Tuning
- GridSearchCV optimizing for Recall
- Selected optimal regularization strength

### 7️⃣ Model Explainability
- SHAP used for feature attribution
- Provides patient-level interpretability
- Healthcare-friendly transparency

### 8️⃣ REST API Deployment
- Built with FastAPI
- Input validation via Pydantic
- Logging and error handling
- Risk categorization

---

## 📈 Final Model Performance

At optimized threshold:

- **Recall:** ~0.88  
- **Precision:** ~0.56  
- **False Negatives reduced from 26 → 6**
- ROC-AUC ≈ 0.83

This configuration prioritizes patient safety in screening scenarios.

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model & Evaluate
```python
python -m scripts.quick_test
```

### 4️⃣ Run Inference on New Data
```python
python -m scripts.predict
```

### 5️⃣ Run API
```python
uvicorn api:app --reload
Open:  http://127.0.0.1:8000/docs
```


👨‍💻 Author

Rigal Patel
Applied Machine Learning Portfolio Project


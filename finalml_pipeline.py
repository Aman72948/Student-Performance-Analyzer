# =========================================
# 📦 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, r2_score

import joblib

# =========================================
# 📂 2. LOAD DATA
# =========================================
data = pd.read_csv(r"C:\Users\prince kumar\Downloads\studentperformance.csv")

# =========================================
# 🧹 3. DATA CLEANING
# =========================================

# Fill missing values
data = data.fillna(data.mode().iloc[0])

# =========================================
# 🔄 4. ENCODING (Categorical → Numeric)
# =========================================

# One-hot encoding (best practice)
data = pd.get_dummies(data, drop_first=True)

# =========================================
# 🎯 5. FEATURE & TARGET SPLIT
# =========================================

target = "Exam_Score"

X = data.drop(target, axis=1)
y = data[target]

# =========================================
# 🔥 6. FEATURE SELECTION (SelectKBest)
# =========================================

selector = SelectKBest(score_func=f_regression, k=10)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]

print("\n✅ Selected Features:")
print(selected_features)

# Convert back to DataFrame
X_selected = pd.DataFrame(X_selected, columns=selected_features)

# =========================================
# ✂️ 7. TRAIN-TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# =========================================
# ⚖️ 8. FEATURE SCALING
# =========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================
# 🤖 9. MODEL TRAINING (BEST: Random Forest)
# =========================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================
# 🔮 10. PREDICTION
# =========================================

y_pred = model.predict(X_test)

# =========================================
# 📊 11. EVALUATION
# =========================================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print("MAE:", mae)
print("R2 Score:", r2)

# =========================================
# 💾 12. SAVE MODEL + SCALER + FEATURES
# =========================================

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features.tolist(), "features.pkl")

print("\n✅ Model Saved Successfully!")
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib


# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("data/climate_features.csv")


# ----------------------------
# Prepare ML data
# ----------------------------
X = df.drop(columns=["PRECTOTCORR", "DATE"])
y = np.log1p(df["PRECTOTCORR"])   # log transform rainfall


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----------------------------
# XGBoost Model
# ----------------------------
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)


# ----------------------------
# Predictions
# ----------------------------
pred_log = model.predict(X_test)

pred = np.expm1(pred_log)       # reverse log transform
y_test_actual = np.expm1(y_test)


# ----------------------------
# Evaluation
# ----------------------------
rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
r2 = r2_score(y_test_actual, pred)

print("Model Training Completed")
print("RMSE:", rmse)
print("R² Score:", r2)


# ----------------------------
# Feature Importance
# ----------------------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()


# ----------------------------
# Actual vs Predicted
# ----------------------------
plt.figure(figsize=(6,6))
plt.scatter(y_test_actual, pred, alpha=0.5)
plt.xlabel("Actual Rainfall")
plt.ylabel("Predicted Rainfall")
plt.title("Actual vs Predicted Rainfall")
plt.tight_layout()
plt.show()


# ----------------------------
# Save trained model
# ----------------------------
joblib.dump(model, "models/xgboost_rainfall_model.pkl")

print("Model saved successfully!")
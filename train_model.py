"""
train_model.py
---------------
Run this SECOND (after generate_data.py).

Trains a model to predict monthly demand (units_sold) using ONLY:
  - which product it is
  - its price
  - the month

We intentionally leave out things like "last month's sales" as an input,
because the goal here is to answer: "if I change the price, or if the
month changes, how does demand change?" - not just to copy old numbers.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from database import get_connection

# ---------- 1. LOAD DATA FROM THE DATABASE ----------
conn = get_connection()
df = pd.read_sql_query("""
    SELECT demand_history.product_id, products.name as product_name,
           products.category, demand_history.year, demand_history.month,
           demand_history.price, demand_history.units_sold
    FROM demand_history
    JOIN products ON demand_history.product_id = products.product_id
""", conn)
conn.close()

print("Loaded", len(df), "rows from the database")

# ---------- 2. ENCODE PRODUCT AS A NUMBER ----------
le = LabelEncoder()
df["product_encoded"] = le.fit_transform(df["product_id"])
joblib.dump(le, "product_encoder.pkl")

# ---------- 3. FEATURES AND TARGET ----------
features = ["product_encoded", "price", "month"]
X = df[features]
y = df["units_sold"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

# ---------- 4. TRAIN AND COMPARE MODELS ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=250, random_state=42)
}

best_model, best_name, best_r2 = None, None, -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{name}\n  MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.3f}")
    if r2 > best_r2:
        best_r2, best_name, best_model = r2, name, model

print(f"\n>>> Best model: {best_name} (R2 = {best_r2:.3f})")

# ---------- 5. SAVE MODEL ----------
joblib.dump(best_model, "demand_model.pkl")
joblib.dump(features, "model_features.pkl")
print("Saved demand_model.pkl")

# ---------- 6. GRAPHS FOR PRESENTATION ----------
preds_best = best_model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds_best, alpha=0.3, color="teal")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Units Sold")
plt.ylabel("Predicted Units Sold")
plt.title(f"Actual vs Predicted ({best_name})")
plt.tight_layout()
plt.savefig("static/actual_vs_predicted.png", dpi=120)
plt.close()

if best_name == "Random Forest":
    importances = best_model.feature_importances_
    plt.figure(figsize=(6, 3.5))
    plt.barh(features, importances, color="steelblue")
    plt.title("Which factor affects demand more: product, price, or month?")
    plt.tight_layout()
    plt.savefig("static/feature_importance.png", dpi=120)
    plt.close()

print("Saved graphs in static/ folder")
print("\nDONE. Model is trained and ready. Now run: python app.py")

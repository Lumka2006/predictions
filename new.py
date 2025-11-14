import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import joblib
import os

new_data_path = "data.xlsx"                 # Path to your new Excel file
model_path = "optimized_rf_model.pkl"       # Saved Random Forest model
target_col = "Target"                       # Target column name
output_path = "new_data_predictions.xlsx"   # Save predictions

# ---------------------------------------------------------------------
# 1. LOAD MODEL
# ---------------------------------------------------------------------
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found!")

rf_model = joblib.load(model_path)

# ✔️ Use the EXACT feature names the model was trained with
top_features = list(rf_model.feature_names_in_)

# ---------------------------------------------------------------------
# 2. LOAD NEW DATA
# ---------------------------------------------------------------------
if not os.path.exists(new_data_path):
    raise FileNotFoundError(f"{new_data_path} not found!")

df_new = pd.read_excel(new_data_path)

# ---------------------------------------------------------------------
# 3. CLEAN COLUMN NAMES (minimal cleaning ONLY)
# ---------------------------------------------------------------------
df_new.columns = (
    df_new.columns
    .str.replace(" ", "_")
    .str.replace("(", "_")
    .str.replace(")", "_")
    .str.strip()
)

# ---------------------------------------------------------------------
# 4. FORCE COLUMNS TO MATCH TRAINING COLUMNS EXACTLY
# ---------------------------------------------------------------------
df_new = df_new.reindex(columns=top_features)

# ---------------------------------------------------------------------
# 5. HANDLE MISSING VALUES
# ---------------------------------------------------------------------
num_cols_new = df_new.select_dtypes(include=[np.number]).columns
cat_cols_new = df_new.select_dtypes(exclude=[np.number]).columns

df_new[num_cols_new] = SimpleImputer(strategy="median").fit_transform(df_new[num_cols_new])

if len(cat_cols_new):
    df_new[cat_cols_new] = SimpleImputer(strategy="most_frequent").fit_transform(df_new[cat_cols_new])

# ---------------------------------------------------------------------
# 6. PREDICT
# ---------------------------------------------------------------------
y_new_pred = rf_model.predict(df_new)
df_new['Predicted_Target'] = y_new_pred

# ---------------------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------------------
df_new.to_excel(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")

# ---------------------------------------------------------------------
# 8. OPTIONAL: EVALUATE IF TRUE TARGET EXISTS
# ---------------------------------------------------------------------
original = pd.read_excel(new_data_path)
if target_col in original.columns:
    y_new_true = original[target_col]

    # Map numeric predictions to original class names
    label_map = {
        0: "Dropout",
        1: "Enrolled",
        2: "Graduate"
    }
    y_new_pred_decoded = pd.Series(y_new_pred).map(label_map)

    from sklearn.metrics import accuracy_score, confusion_matrix

    acc_new = accuracy_score(y_new_true, y_new_pred_decoded)
    cm_new = confusion_matrix(y_new_true, y_new_pred_decoded)

    print(f"Accuracy on new data: {acc_new:.4f}")
    print("Confusion Matrix:")
    print(cm_new)


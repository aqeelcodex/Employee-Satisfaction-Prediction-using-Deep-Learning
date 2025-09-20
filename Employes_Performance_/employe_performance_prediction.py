import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
emp = pd.read_csv("Extended_Employee_Performance_and_Productivity_Data.csv")
print(emp.head())
print(emp.isnull().sum())
print(emp.shape)
print(emp.describe())

# Drop unnecessary columns
emp.drop(columns= ["Employee_ID", "Hire_Date"], inplace=True)

# Explore categorical columns
cat_cols = emp.select_dtypes(include= "object").columns
for col in cat_cols:
    print(f"[{col}] ..... {emp[col].nunique()} unique values.")

# Convert target column to integer
emp["Resigned"] = emp["Resigned"].astype(int)

# Encode ordinal column
ordinal_col = "Education_Level"
oe = OrdinalEncoder()
emp[ordinal_col] = oe.fit_transform(emp[[ordinal_col]])

# One-hot encode selected categorical columns
ohe_cols = ["Department", "Gender", "Job_Title"]
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_features = ohe.fit_transform(emp[ohe_cols])
ohe_features_name = ohe.get_feature_names_out(ohe_cols)

# Merge OHE features into dataframe
df_ohe_features = pd.DataFrame(ohe_features, columns=ohe_features_name, index=emp.index)
employe = pd.concat([emp.drop(columns=ohe_cols), df_ohe_features], axis=1)

print(employe.dtypes)

# Define features and target
X = employe.drop(columns="Employee_Satisfaction_Score")
y = employe["Employee_Satisfaction_Score"]

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_train_mae, cv_test_mae = [], []

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale numerical features
    scaling = StandardScaler()
    X_train_scaled = scaling.fit_transform(X_train)
    X_test_scaled = scaling.transform(X_test)
    
    # Define neural network
    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="linear")   # Regression output
    ])

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    # Early stopping for overfitting prevention
    early_stopping = EarlyStopping(monitor="val_mae", patience=5, restore_best_weights=True)

    # Train model
    model.fit(X_train_scaled, y_train,
              validation_data=(X_test_scaled, y_test),
              epochs=50, batch_size=32,
              callbacks=[early_stopping],
              verbose=1)

    # Evaluate model
    train_loss, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)

    print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    cv_train_mae.append(train_mae)
    cv_test_mae.append(test_mae)

# Final results
print("\n=== Cross-Validation Results ===")
print(f"Average Train MAE: {np.mean(cv_train_mae):.4f}")
print(f"Average Test MAE: {np.mean(cv_test_mae):.4f}")

# Save model and preprocessing objects for deployment
import pickle
model_filename = "models.pkl"
scale_filename = "scale.pkl"
ohe_filename = "ohe.pkl"
oe_filename = "oe.pkl"

with open(model_filename, "wb") as file:
    pickle.dump(model, file)   # Trained model
with open(scale_filename, "wb") as file:
    pickle.dump(scaling, file) # Scaler
with open(ohe_filename, "wb") as file:
    pickle.dump(ohe, file)     # One-hot encoder
with open(oe_filename, "wb") as file:
    pickle.dump(oe, file)      # Ordinal encoder

print("All files saved successfully !")

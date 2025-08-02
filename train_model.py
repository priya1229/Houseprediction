import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv("AmesHousing.csv")

# Select features
selected_features = [
    "GrLivArea", "BedroomAbvGr", "Neighborhood", "OverallQual",
    "YearBuilt", "FullBath", "GarageCars", "TotRmsAbvGrd"
]

# Ensure required columns are present
df_model = df[selected_features + ["SalePrice"]].dropna()

# Encode categorical feature: Neighborhood
le = LabelEncoder()
df_model["Neighborhood"] = le.fit_transform(df_model["Neighborhood"])

# Save label encoder
joblib.dump(le, "le_neighborhood.pkl")

# Prepare X and y
X = df_model.drop("SalePrice", axis=1)
y = df_model["SalePrice"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model and label encoder saved successfully.")

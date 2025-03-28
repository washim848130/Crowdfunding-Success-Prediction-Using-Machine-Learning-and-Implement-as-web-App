import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
file_path = "data.csv"  # Update the path to your dataset
data = pd.read_csv(file_path)

# Data preprocessing
data_cleaned = data.drop(columns=["Unnamed: 0"])  # Drop unnecessary columns
data_cleaned["state"] = data_cleaned["state"].apply(lambda x: 1 if x == "successful" else 0)  # Convert target to binary

# One-hot encode categorical features
data_encoded = pd.get_dummies(data_cleaned, columns=["parent_category", "sub_category", "country"], drop_first=True)

# Define features and target
X = data_encoded.drop(columns=["state"])
y = data_encoded["state"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model and metadata
model_metadata = {
    "model": model,
    "feature_names": X.columns.tolist(),
}
joblib.dump(model_metadata, "1compatible_model_sklearn_1.6.1.pkl")

print("Model training complete. Saved as '1compatible_model_sklearn_1.6.1.pkl'.")

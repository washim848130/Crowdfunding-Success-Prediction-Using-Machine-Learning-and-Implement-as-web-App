import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the cleaned dataset
file_path = "crowdfunding_dataset.csv"  # Ensure this file is in the same directory

data = pd.read_csv(file_path)

# Define feature columns and target
feature_columns = [
    "parent_category", "sub_category", "days", "backers_count", "pledged_amt",
    "converted_pledged_amt", "goal", "country"
]
target_column = "success"

# One-hot encode categorical features
data = pd.get_dummies(data, columns=["parent_category", "sub_category", "country"], drop_first=True)

# Split features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and metadata
model_metadata = {
    "model": model,
    "feature_names": X.columns.tolist()
}
joblib.dump(model_metadata, "compatible_model_sklearn_1.6.1.pkl")

print("Model retrained and saved as compatible_model_sklearn_1.6.1.pkl")

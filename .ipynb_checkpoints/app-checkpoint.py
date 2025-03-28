import streamlit as st
import pandas as pd
import joblib

# Load the compatible trained model with metadata
model_metadata = joblib.load("compatible_model_sklearn_1.6.1.pkl")
model = model_metadata["model"]
training_columns = model_metadata["feature_names"]

# Define input features
input_features = [
    "parent_category", "sub_category", "days", "backers_count", "pledged_amt", 
    "converted_pledged_amt", "goal", "country"
]

# Define categorical mapping for one-hot encoding
categories = {
    "parent_category": ["Art", "Design", "Games", "Music", "Technology"],
    "sub_category": ["Comics", "Film & Video", "Gadgets", "Hardware", "Software"],
    "country": ["AU", "CA", "DE", "GB", "US"]
}

def main():
    st.title("Crowdfunding Success Prediction")
    st.write("Enter the details of your campaign to predict its success.")

    # User inputs
    user_input = {}
    for feature in input_features:
        if feature in categories:
            user_input[feature] = st.selectbox(f"{feature.replace('_', ' ').capitalize()}", categories[feature])
        else:
            user_input[feature] = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0)

    if st.button("Predict Success"):
        # Convert user input to DataFrame for prediction
        input_df = pd.DataFrame([user_input])

        # One-hot encode categorical features
        input_df = pd.get_dummies(input_df, columns=["parent_category", "sub_category", "country"], drop_first=True)

        # Align input_df with training columns
        for col in training_columns:
            if col not in input_df:
                input_df[col] = 0

        # Ensure column order matches
        input_df = input_df[training_columns]

        # Predict
        prediction = model.predict(input_df)[0]
        st.write("Prediction: Success" if prediction == 1 else "Prediction: Not Successful")

if __name__ == "__main__":
    main()

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


import joblib
import pandas as pd
import numpy as np

# Load the trained model, scaler and columns
model = joblib.load("random_forest_compressed.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

def preprocess_input(data):
    # Convert input data to DataFrame
    df_input = pd.DataFrame([data])

    # One-Hot Encode categorical features (handle unseen categories)
    for col in df_input.select_dtypes(include=['object']).columns:
        if df_input[col].nunique() > 2:
            df_input = pd.get_dummies(df_input, columns=[col], drop_first=True)
        else: # Handle binary columns using LabelEncoder if needed, though get_dummies is robust
             from sklearn.preprocessing import LabelEncoder
             le = LabelEncoder()
             # Fit to all possible values (including those not in the small input df)
             # This is a simplification; in a real app, you'd fit the encoder on the training data
             # For this example, we'll just apply it assuming "Yes"/"No" or "Male"/"Female"
             if col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
                 df_input[col] = le.fit_transform(df_input[col])
             else: # For other binary features, get_dummies will work
                  df_input = pd.get_dummies(df_input, columns=[col], drop_first=True)


    # Ensure the input data has the same columns as the training data
    # Add missing columns with a value of 0
    for col in X_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # Ensure the order of columns is the same
    df_input = df_input[X_columns]

    # Scale numerical features
    # Identify numerical columns by checking their dtype and if they are not OHE columns
    numerical_cols = df_input.select_dtypes(include=np.number).columns.tolist()
    # Exclude potential OHE columns that might be numbers
    numerical_cols = [col for col in numerical_cols if col not in X_columns or df_input[col].nunique() > 2] # Simple heuristic

    if numerical_cols:
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])

    return df_input

# Replace the prediction part to use the loaded model and preprocessed input

model = joblib.load("random_forest.pkl") # Assuming you want to use the Random Forest model

# Load the scaler used during training
# Note: You would need to save the scaler object during training as well
# For demonstration purposes, we'll create a dummy scaler based on the training data structure
# In a real application, save and load the actual scaler
# scaler = joblib.load("scaler.pkl")

# Assume X_train from the notebook is available or recreate it
# For this example, we'll use the first few rows of the original dataframe to simulate the structure
# of the data before scaling and encoding to create a dummy scaler
# In a real scenario, you would save and load the actual scaler used on X_train

# Load the dataset again to simulate the original structure for scaler fitting
# In a real application, you would have saved the fitted scaler.
file_path = "/content/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df_original = pd.read_csv(file_path)
df_original = df_original.drop("customerID", axis=1)
y_original = df_original['Churn'].map({'Yes': 1, 'No': 0})
X_original = df_original.drop("Churn", axis=1)

# Apply the same encoding steps as in the notebook to get the structure for scaler fitting
for col in X_original.select_dtypes(include=['object']).columns:
    if X_original[col].nunique() == 2:
        le = LabelEncoder()
        X_original[col] = le.fit_transform(X_original[col])
    else:
        # Handle potential errors with 'TotalCharges' which is not purely numerical initially
        if col == 'TotalCharges':
            X_original[col] = pd.to_numeric(X_original[col], errors='coerce')
            X_original.dropna(subset=[col], inplace=True) # Drop rows with NaN
        X_original = pd.get_dummies(X_original, columns=[col], drop_first=True)

# Re-split the original data after cleaning and encoding to fit the scaler
X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
    X_original, y_original, test_size=0.2, random_state=42, stratify=y_original
)

scaler = StandardScaler()
scaler.fit(X_train_original) # Fit the scaler on the processed training data

# Streamlit App Title
st.title("Telco Customer Churn Prediction")

# Input fields for user data
st.header("Customer Information")

gender = st.selectbox("Gender", ['Female', 'Male'])
senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
partner = st.selectbox("Partner", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['Yes', 'No'])
tenure = st.slider("Tenure (months)", 0, 72, 1)
phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)


# Create a dictionary from the input data
input_data = {
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Apply the same preprocessing steps as used for training
# Handle 'TotalCharges' before encoding
input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
# Note: In a real app, you might need to handle the case where TotalCharges is NaN after conversion

# Apply encoding (binary and one-hot)
for col in input_df.select_dtypes(include=['object']).columns:
    if input_df[col].nunique() == 2 and col in X_original.columns: # Check if column exists in original data
        le = LabelEncoder()
        # Fit on all unique values from original training data to handle unseen values in input
        le.fit(df_original[col].unique())
        input_df[col] = le.transform(input_df[col])
    else:
        # Apply one-hot encoding, ensuring all original columns are present
        # This requires knowing the columns from the original training data after one-hot encoding
        # A safer approach is to use the columns from the fitted scaler or a saved list of columns
        # For demonstration, we'll apply get_dummies and then reindex
        input_df = pd.get_dummies(input_df, columns=[col], drop_first=True)


# Ensure the input DataFrame has the same columns and order as the training data (X_train_original)
# This is crucial for the model to work correctly. Missing columns should be added with 0, extra columns dropped.
# The columns of X_train_original after preprocessing are in scaler.feature_names_in_
input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)


# Scale the numerical features
input_scaled = scaler.transform(input_df)


# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[:, 1] # Probability of churn

    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error(f"This customer is likely to churn with a probability of {prediction_proba[0]:.2f}")
    else:
        st.success(f"This customer is unlikely to churn with a probability of {prediction_proba[0]:.2f}")

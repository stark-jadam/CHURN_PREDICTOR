import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load Saved Model, Scaler, Columns, and Median Years ---
try:
    model = joblib.load('rf_churn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    feature_columns = joblib.load('feature_columns.joblib')
    median_years_for_loyalty = joblib.load('median_years_for_loyalty.joblib')
except FileNotFoundError:
    st.error(
        "Model/Scaler/Columns/Median Years files not found. Please ensure they are in the same directory as app.py or provide the correct path.")
    st.stop()  # Stop execution if files are missing
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()


# --- Helper function to preprocess input data ---
def preprocess_input(data_dict, original_feature_columns, median_years):
    # Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([data_dict])

    # --- 1. Handle Categorical Features (One-Hot Encoding) ---
    # Gender: Base category is 'Female' (Gender_Male=0, Gender_Other=0)
    input_df['Gender_Male'] = 1 if input_df['Gender'].iloc[0] == 'Male' else 0
    input_df['Gender_Other'] = 1 if input_df['Gender'].iloc[0] == 'Other' else 0

    # Promotion_Response: Base category is 'Ignored'
    input_df['Promotion_Response_Responded'] = 1 if input_df['Promotion_Response'].iloc[0] == 'Responded' else 0
    input_df['Promotion_Response_Unsubscribed'] = 1 if input_df['Promotion_Response'].iloc[0] == 'Unsubscribed' else 0

    # Drop original categorical columns used for input
    input_df = input_df.drop(['Gender', 'Promotion_Response'], axis=1)

    # --- 2. Convert Boolean to Int (already done by Streamlit for checkbox if used like that) ---
    # Assuming Email_Opt_In comes as True/False from st.checkbox, convert to 1/0
    if 'Email_Opt_In' in input_df.columns:  # Make sure it exists
        input_df['Email_Opt_In'] = input_df['Email_Opt_In'].astype(int)

    # --- 3. Feature Engineering (must match training phase) ---
    input_df['Spend_per_Purchase'] = np.where(
        input_df['Num_of_Purchases'] > 0,
        input_df['Total_Spend'] / input_df['Num_of_Purchases'], 0
    )
    input_df['Return_Rate'] = np.where(
        input_df['Num_of_Purchases'] > 0,
        input_df['Num_of_Returns'] / input_df['Num_of_Purchases'], 0
    )
    if 'Years_as_Customer' in input_df.columns:
        input_df['Purchases_per_Year'] = np.where(
            input_df['Years_as_Customer'] > 0,
            input_df['Num_of_Purchases'] / input_df['Years_as_Customer'],
            input_df['Num_of_Purchases']
        )
    if 'Years_as_Customer' in input_df.columns and 'Satisfaction_Score' in input_df.columns:
        input_df['Loyal_High_Sat'] = (
                (input_df['Years_as_Customer'] > median_years) &
                (input_df['Satisfaction_Score'] >= 4)
        ).astype(int)

    # --- 4. Ensure all feature columns are present and in correct order ---
    # Create a DataFrame with all expected columns, initialized to 0 or appropriate default
    processed_df = pd.DataFrame(columns=original_feature_columns)
    processed_df = pd.concat([processed_df, input_df], ignore_index=False)  # Add input_df's data

    # Fill any missing columns (that were not in input_df but are in original_feature_columns) with 0
    # This handles cases where some one-hot encoded columns might not be explicitly created if input has only base category
    for col in original_feature_columns:
        if col not in processed_df.columns:
            processed_df[col] = 0  # Or an appropriate default

    # Ensure order and select only the feature columns
    processed_df = processed_df[original_feature_columns].fillna(0)  # Fill any NaNs that might have occurred

    return processed_df


# --- Streamlit User Interface ---
st.set_page_config(page_title="The Cypher on Elm - Churn Prediction", layout="wide")
st.title("The Cypher on Elm - Customer Churn Prediction")
st.markdown("""
Enter customer details below to predict the likelihood of churn.
This app uses a Random Forest model trained on historical customer data.
""")

# --- Input Fields ---
# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Account")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", options=['Female', 'Male', 'Other'])
    annual_income = st.number_input("Annual Income (e.g., 50000)", min_value=0.0, value=50000.0, step=1000.0,
                                    format="%.2f")
    years_as_customer = st.number_input("Years as Customer", min_value=0, max_value=50, value=3, step=1)
    email_opt_in = st.checkbox("Email Opt-In", value=True)

with col2:
    st.subheader("Purchase & Interaction History")
    total_spend = st.number_input("Total Spend (e.g., 2500.00)", min_value=0.0, value=2500.0, step=100.0, format="%.2f")
    num_of_purchases = st.number_input("Number of Purchases", min_value=0, value=10, step=1)
    avg_transaction_amount = st.number_input("Average Transaction Amount", min_value=0.0, value=250.0, step=10.0,
                                             format="%.2f")
    num_of_returns = st.number_input("Number of Returns", min_value=0, value=1, step=1)
    last_purchase_days_ago = st.number_input("Last Purchase Days Ago", min_value=0, value=90, step=1)

with col3:
    st.subheader("Engagement & Feedback")
    num_of_support_contacts = st.number_input("Number of Support Contacts", min_value=0, value=2, step=1)
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    promotion_response = st.selectbox("Promotion Response", options=['Ignored', 'Responded', 'Unsubscribed'])

# --- Prediction Button ---
if st.button("Predict Churn Likelihood", type="primary"):
    # 1. Collect input data into a dictionary
    input_data = {
        'Age': age,
        'Annual_Income': annual_income,
        'Total_Spend': total_spend,
        'Years_as_Customer': years_as_customer,
        'Num_of_Purchases': num_of_purchases,
        'Average_Transaction_Amount': avg_transaction_amount,
        'Num_of_Returns': num_of_returns,
        'Num_of_Support_Contacts': num_of_support_contacts,
        'Satisfaction_Score': satisfaction_score,
        'Last_Purchase_Days_Ago': last_purchase_days_ago,
        'Email_Opt_In': email_opt_in,  # This will be True/False
        # These are the original categorical inputs, preprocess_input will handle them
        'Gender': gender,
        'Promotion_Response': promotion_response
    }

    # 2. Preprocess the input data
    try:
        processed_input_df = preprocess_input(input_data, feature_columns, median_years_for_loyalty)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    # 3. Scale the processed input features
    try:
        scaled_input = scaler.transform(processed_input_df)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()

    # 4. Make prediction
    try:
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        churn_probability = prediction_proba[0][1]  # Probability of class 1 (Churn)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"This customer is LIKELY TO CHURN (Probability: {churn_probability:.2%})")
            st.markdown(f"""
            **Why this might be the case (based on common churn drivers):**
            Consider their recent activity, satisfaction, and overall engagement pattern.
            Targeted outreach or special offers might be beneficial.
            """)
        else:
            st.success(f"This customer is UNLIKELY TO CHURN (Churn Probability: {churn_probability:.2%})")
            st.markdown(f"""
            **Keep up the good work!**
            Continue to engage this customer and ensure they have a positive experience.
            """)



    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Enter customer details and click 'Predict Churn Likelihood'.")

st.sidebar.header("About")
st.sidebar.info("""
This is a demo web application for 'The Cypher on Elm' customer churn prediction project.
It uses a machine learning model to predict whether a customer is likely to churn based on their entered characteristics.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed as part of a data science project.")
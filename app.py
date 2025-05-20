import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Load original data for a contextual plot ---
@st.cache_data  # Cache the data loading to improve performance
def load_original_data():
    try:
        # This assumes 'online_retail_customer_churn.csv' is in the same directory as app.py
        df_viz = pd.read_csv('online_retail_customer_churn.csv')
        # Ensure Target_Churn is 0 or 1 for plotting
        if 'Target_Churn' in df_viz.columns and df_viz['Target_Churn'].dtype == 'bool':
            df_viz['Target_Churn_Int'] = df_viz['Target_Churn'].astype(int)
        elif 'Target_Churn' in df_viz.columns and df_viz[
            'Target_Churn'].dtype != 'int':  # if it's object like 'True'/'False' strings
            # Handle potential string 'True'/'False' to int, robustly
            df_viz['Target_Churn_Int'] = df_viz['Target_Churn'].apply(
                lambda x: 1 if str(x).lower() == 'true' else 0).astype(int)
        else:  # Assuming it's already int or needs no change
            df_viz['Target_Churn_Int'] = df_viz['Target_Churn']

        return df_viz # Return the whole df for more flexible plotting
    except FileNotFoundError:
        st.sidebar.error("Original dataset ('online_retail_customer_churn.csv') for visualization not found.")
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading data for viz: {e}")
        return None


df_for_plot = load_original_data()


# --- Helper function to preprocess input data ---
def preprocess_input(data_dict, original_feature_columns, median_years):
    # Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([data_dict])

    # --- 1. Handle Categorical Features (One-Hot Encoding) ---
    input_df['Gender_Male'] = 1 if input_df['Gender'].iloc[0] == 'Male' else 0
    input_df['Gender_Other'] = 1 if input_df['Gender'].iloc[0] == 'Other' else 0

    input_df['Promotion_Response_Responded'] = 1 if input_df['Promotion_Response'].iloc[0] == 'Responded' else 0
    input_df['Promotion_Response_Unsubscribed'] = 1 if input_df['Promotion_Response'].iloc[0] == 'Unsubscribed' else 0

    input_df = input_df.drop(['Gender', 'Promotion_Response'], axis=1)

    # --- 2. Convert Boolean to Int ---
    if 'Email_Opt_In' in input_df.columns:
        input_df['Email_Opt_In'] = input_df['Email_Opt_In'].astype(int)

    # --- 3. Feature Engineering ---
    input_df['Spend_per_Purchase'] = np.where(
        input_df['Num_of_Purchases'] > 0,
        input_df['Total_Spend'] / input_df['Num_of_Purchases'], 0
    )
    input_df['Return_Rate'] = np.where(
        input_df['Num_of_Purchases'] > 0,
        input_df['Num_of_Returns'] / input_df['Num_of_Purchases'], 0
    )
    if 'Years_as_Customer' in input_df.columns:  # Check if key exists before division
        input_df['Purchases_per_Year'] = np.where(
            input_df['Years_as_Customer'].iloc[0] > 0,  # Use .iloc[0] for Series access
            input_df['Num_of_Purchases'].iloc[0] / input_df['Years_as_Customer'].iloc[0],
            input_df['Num_of_Purchases'].iloc[0]
        )
    else:
        input_df['Purchases_per_Year'] = 0  # Default if Years_as_Customer is not in input

    if 'Years_as_Customer' in input_df.columns and 'Satisfaction_Score' in input_df.columns:
        input_df['Loyal_High_Sat'] = (
                (input_df['Years_as_Customer'].iloc[0] > median_years) &
                (input_df['Satisfaction_Score'].iloc[0] >= 4)
        ).astype(int)
    else:
        input_df['Loyal_High_Sat'] = 0  # Default if keys are missing

    # --- 4. Ensure all feature columns are present and in correct order ---
    output_df_row = pd.DataFrame(0, index=[0], columns=original_feature_columns)

    for col in input_df.columns:
        if col in output_df_row.columns:
            output_df_row[col] = input_df[col].iloc[0]

    return output_df_row


# --- Streamlit User Interface ---
st.set_page_config(page_title="The Cypher on Elm - Churn Prediction", layout="wide")
st.title("The Cypher on Elm - Customer Churn Prediction")
st.markdown("""
Enter customer details below to predict the likelihood of churn.
This app uses a Random Forest model trained on historical customer data.
""")

# --- Dashboard Section ---
st.subheader("Understanding Churn Drivers (Based on Training Data)")

if df_for_plot is not None and not df_for_plot.empty:
    if 'Target_Churn_Int' in df_for_plot.columns:
        # Create two columns for the dashboard plots
        dash_col1, dash_col2 = st.columns(2)

        with dash_col1:
            # --- Donut Chart for Overall Churn Rate ---
            churn_counts = df_for_plot['Target_Churn_Int'].value_counts()
            if not churn_counts.empty:
                labels = ['Did Not Churn (0)', 'Churned (1)']
                sizes = [churn_counts.get(0, 0), churn_counts.get(1, 0)] # Handle cases where one category might be missing
                colors = ['#5cb85c', '#d9534f'] # Green for no churn, Red for churn
                explode = (0.05, 0)  # explode the 1st slice (Not Churned)

                fig_donut, ax_donut = plt.subplots(figsize=(6, 5)) # Adjusted figure size
                ax_donut.pie(sizes, explode=explode, labels=labels, colors=colors,
                             autopct='%1.1f%%', shadow=False, startangle=90,
                             pctdistance=0.85) # pctdistance for donut hole

                # Draw a circle at the center to make it a donut
                centre_circle = plt.Circle((0, 0), 0.70, fc='white')
                fig_donut.gca().add_artist(centre_circle)

                ax_donut.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                ax_donut.set_title('Overall Churn Distribution in Training Data', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig_donut)
                st.caption("This donut chart shows the proportion of customers who churned versus those who did not in the training dataset.")
            else:
                st.write("Not enough data to display churn distribution donut chart.")

        with dash_col2:
            # --- Box Plot for Last Purchase Days Ago ---
            if 'Last_Purchase_Days_Ago' in df_for_plot.columns:
                fig_boxplot, ax_boxplot = plt.subplots(figsize=(6, 5))  # Adjusted figure size
                sns.boxplot(x='Target_Churn_Int', y='Last_Purchase_Days_Ago', data=df_for_plot, ax=ax_boxplot, palette="pastel")
                ax_boxplot.set_title('Context: Last Purchase Days Ago vs. Churn', fontsize=14)
                ax_boxplot.set_xticklabels(['Did Not Churn (0)', 'Churned (1)'], fontsize=10)
                ax_boxplot.set_xlabel("Customer Churn Status", fontsize=12)
                ax_boxplot.set_ylabel("Last Purchase Days Ago", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_boxplot)
                st.caption(
                    "This plot shows how the days since a customer's last purchase related to churn in the training data. Higher days often correlate with higher churn likelihood.")
            else:
                st.write("Required 'Last_Purchase_Days_Ago' column for boxplot is missing.")
    else:
        st.write("Required 'Target_Churn_Int' column for dashboard plots is missing from the loaded data.")
else:
    st.write("Contextual visualization data could not be loaded or is empty.")

st.markdown("---")  # Separator

st.subheader("Enter Customer Details for Churn Prediction:")
# --- Input Fields ---
col1, col2, col3 = st.columns(3)


with col1:
    st.markdown("##### Demographics & Account")
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    gender = st.selectbox("Gender", options=['Female', 'Male', 'Other'], index=0)  # Default to Female
    annual_income = st.number_input("Annual Income (e.g., 50000)", min_value=0.0, value=50000.0, step=1000.0,
                                    format="%.2f")
    years_as_customer = st.number_input("Years as Customer", min_value=0, max_value=50, value=3, step=1)
    email_opt_in = st.checkbox("Email Opt-In", value=True)

with col2:
    st.markdown("##### Purchase & Interaction History")
    total_spend = st.number_input("Total Spend (e.g., 2500.00)", min_value=0.0, value=2500.0, step=100.0, format="%.2f")
    num_of_purchases = st.number_input("Number of Purchases", min_value=0, value=10, step=1)
    avg_transaction_amount = st.number_input("Average Transaction Amount", min_value=0.0, value=250.0, step=10.0,
                                             format="%.2f")
    num_of_returns = st.number_input("Number of Returns", min_value=0, value=1, step=1)
    last_purchase_days_ago = st.number_input("Last Purchase Days Ago", min_value=0, value=90, step=1)

with col3:
    st.markdown("##### Engagement & Feedback")
    num_of_support_contacts = st.number_input("Number of Support Contacts", min_value=0, value=2, step=1)
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
    promotion_response = st.selectbox("Promotion Response", options=['Ignored', 'Responded', 'Unsubscribed'],
                                      index=0)  # Default to Ignored

# --- Prediction Button ---
if st.button("Predict Churn Likelihood", type="primary", use_container_width=True):
    input_data = {
        'Age': age, 'Annual_Income': annual_income, 'Total_Spend': total_spend,
        'Years_as_Customer': years_as_customer, 'Num_of_Purchases': num_of_purchases,
        'Average_Transaction_Amount': avg_transaction_amount, 'Num_of_Returns': num_of_returns,
        'Num_of_Support_Contacts': num_of_support_contacts, 'Satisfaction_Score': satisfaction_score,
        'Last_Purchase_Days_Ago': last_purchase_days_ago, 'Email_Opt_In': email_opt_in,
        'Gender': gender, 'Promotion_Response': promotion_response
    }

    try:
        processed_input_df = preprocess_input(input_data, feature_columns, median_years_for_loyalty)
        scaled_input = scaler.transform(processed_input_df)

        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)
        churn_probability = prediction_proba[0][1]

        st.markdown("---")
        st.subheader("Prediction Result:")

        # Display Churn Probability Bar Chart first
        st.markdown("##### Churn Risk Assessment")
        prob_data = pd.DataFrame({
            'Outcome': ['Will Not Churn', 'Will Churn'],
            'Probability': [1 - churn_probability, churn_probability]
        })

        fig_prob, ax_prob = plt.subplots(figsize=(6, 1.5))  # Adjusted for horizontal bar
        sns.barplot(x='Probability', y='Outcome', data=prob_data, ax=ax_prob, palette=["#5cb85c", "#d9534f"],
                    orient='h')  # Green for no churn, Red for churn
        ax_prob.set_xlim(0, 1)
        ax_prob.set_xlabel("Probability", fontsize=10)
        ax_prob.set_ylabel("")
        ax_prob.tick_params(axis='y', labelsize=10)
        ax_prob.tick_params(axis='x', labelsize=8)

        # Add probability text on bars
        for i, v in enumerate(prob_data['Probability']):
            ax_prob.text(v + 0.02, i, f"{v:.1%}", color='black', va='center', fontweight='bold', fontsize=10)
        st.pyplot(fig_prob)

        # Display textual prediction result
        if prediction[0] == 1:
            st.error(f"This customer is LIKELY TO CHURN (Model Score: {churn_probability:.0%})")
            st.markdown("""
            **Potential Action:** Consider targeted outreach or special offers to retain this customer.
            Review their interaction history for specific pain points.
            """)
        else:
            st.success(f"This customer is UNLIKELY TO CHURN (Model Score: {churn_probability:.0%})")
            st.markdown("""
            **Keep up the good work!** Continue to engage this customer and ensure they have a positive experience.
            """)

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Enter customer details above and click 'Predict Churn Likelihood'.")

st.sidebar.header("About")
st.sidebar.info("""
This is a demo web application for 'The Cypher on Elm' (an online retailer) customer churn prediction project.
It uses a machine learning model to predict whether a customer is likely to churn based on their entered characteristics.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed as part of Adam Stark's WGU Capstone Project, 2025.")
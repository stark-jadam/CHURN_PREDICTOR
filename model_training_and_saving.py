import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

if __name__ == '__main__':
    # --- 1. Load the Dataset ---
    file_path = 'online_retail_customer_churn.csv'
    print(f"Loading dataset from: {file_path}")
    try:
        df_original = pd.read_csv(file_path)
        print("Successfully loaded the dataset.")
        print(f"Original dataset shape: {df_original.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the file path.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the CSV: {e}")
        exit()

    # --- 2. Separate Customer_IDs for Later Use ---
    if 'Customer_ID' in df_original.columns:
        customer_ids_full_dataset = df_original['Customer_ID'].copy()
        print("\nStored Customer_IDs separately.")
    else:
        print("\nWarning: Customer_ID column not found in the original dataset.")
        customer_ids_full_dataset = None

    # --- 3. Data Preprocessing ---
    print("\nStarting Data Preprocessing...")
    df_processed = df_original.copy()

    if 'Customer_ID' in df_processed.columns:
        df_processed = df_processed.drop('Customer_ID', axis=1)
        print("Dropped Customer_ID column from the processed DataFrame.")

    categorical_cols_to_encode = df_processed.select_dtypes(include='object').columns
    if not categorical_cols_to_encode.empty:
        print(f"Categorical columns to be one-hot encoded: {list(categorical_cols_to_encode)}")
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols_to_encode, drop_first=True)
    else:
        print("No object-type categorical columns found for one-hot encoding.")

    bool_cols_final_conversion = []
    # Check original boolean columns first if they exist in the processed dataframe
    for col_name in ['Email_Opt_In', 'Target_Churn']:
        if col_name in df_processed.columns and df_original[col_name].dtype == 'bool':
            bool_cols_final_conversion.append(col_name)

    # Check all columns in df_processed, add if bool and not already added
    for col in df_processed.columns:
        if df_processed[col].dtype == 'bool' and col not in bool_cols_final_conversion:
            bool_cols_final_conversion.append(col)

    if bool_cols_final_conversion:
        for col in bool_cols_final_conversion:
            df_processed[col] = df_processed[col].astype(int)
        print(f"Converted boolean columns to int: {list(bool_cols_final_conversion)}")
    else:
        print(
            "No boolean columns needed explicit conversion to int (get_dummies might have created int/uint8 or they were already int).")

    print("\nProcessed DataFrame info after initial preprocessing:")
    df_processed.info()

    # --- 4. Feature Engineering ---
    print("\nStarting Feature Engineering...")
    df_processed['Spend_per_Purchase'] = np.where(
        df_processed['Num_of_Purchases'] > 0,
        df_processed['Total_Spend'] / df_processed['Num_of_Purchases'], 0
    )
    print("Created 'Spend_per_Purchase'.")

    df_processed['Return_Rate'] = np.where(
        df_processed['Num_of_Purchases'] > 0,
        df_processed['Num_of_Returns'] / df_processed['Num_of_Purchases'], 0
    )
    print("Created 'Return_Rate'.")

    median_years_for_loyalty = 0  # Initialize
    if 'Years_as_Customer' in df_processed.columns:
        df_processed['Purchases_per_Year'] = np.where(
            df_processed['Years_as_Customer'] > 0,
            df_processed['Num_of_Purchases'] / df_processed['Years_as_Customer'],
            df_processed['Num_of_Purchases']  # Or 0, if tenure is 0, purchases this year is all purchases
        )
        print("Created 'Purchases_per_Year'.")

        if not df_processed['Years_as_Customer'].empty and 'Satisfaction_Score' in df_processed.columns:
            median_years_for_loyalty = df_processed['Years_as_Customer'].median()
            df_processed['Loyal_High_Sat'] = (
                    (df_processed['Years_as_Customer'] > median_years_for_loyalty) &
                    (df_processed['Satisfaction_Score'] >= 4)
            ).astype(int)
            print(f"Created 'Loyal_High_Sat' using median_years: {median_years_for_loyalty}.")
        else:
            print("Skipped 'Loyal_High_Sat' creation as 'Years_as_Customer' or 'Satisfaction_Score' is missing/empty.")
    else:
        print("Skipped 'Purchases_per_Year' and 'Loyal_High_Sat' creation as 'Years_as_Customer' column is missing.")

    print("\nDataFrame info after Feature Engineering:")
    df_processed.info()

    # --- 5. Define Target Variable (y) and Features (X) ---
    print("\nDefining Target Variable and Features...")
    try:
        y = df_processed['Target_Churn']
        X = df_processed.drop('Target_Churn', axis=1)
        print("Target variable 'y' and Features 'X' defined.")
        print(f"Shape of Features (X): {X.shape}")
        print(f"Shape of Target Variable (y): {y.shape}")
    except KeyError:
        print("Error: 'Target_Churn' column not found in df_processed. Cannot define X and y.")
        exit()

    # Store feature column names for saving later
    feature_columns_for_saving = list(X.columns)

    # --- 6. Split Data into Training and Testing Sets ---
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")
    print("Data splitting complete.")

    X_test_indices = X_test.index

    # --- 7. Feature Scaling ---
    print("\nStarting Feature Scaling...")
    scaler = StandardScaler()  # Initialize scaler here to save it later
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling complete.")

    # --- 8. Model Training and Evaluation: Logistic Regression (Baseline) ---
    print("\n\n--- Starting Model Training (Logistic Regression) ---")
    log_reg_model = LogisticRegression(random_state=42, max_iter=1000)
    log_reg_model.fit(X_train_scaled, y_train)
    print("Logistic Regression training complete.")

    y_pred_log_reg = log_reg_model.predict(X_test_scaled)
    y_pred_proba_log_reg = log_reg_model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Logistic Regression Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_log_reg):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_log_reg))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_log_reg, target_names=['Non-Churn (0)', 'Churn (1)']))

    # --- 9. Model Training and Evaluation: Random Forest Classifier ---
    print("\n\n--- Starting Model Training (Random Forest Classifier) ---")
    # This will be the model to save
    rf_model_to_save = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    rf_model_to_save.fit(X_train_scaled, y_train)
    print("Random Forest model training complete.")

    y_pred_rf = rf_model_to_save.predict(X_test_scaled)
    y_pred_proba_rf = rf_model_to_save.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Random Forest Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
    print("Confusion Matrix (Random Forest):")
    print(confusion_matrix(y_test, y_pred_rf))
    print("Classification Report (Random Forest):")
    print(classification_report(y_test, y_pred_rf, target_names=['Non-Churn (0)', 'Churn (1)']))

    print("\n--- Feature Importances (Random Forest) ---")
    rf_importances = rf_model_to_save.feature_importances_
    feature_importance_rf_df = pd.DataFrame({'Feature': feature_columns_for_saving, 'Importance': rf_importances})
    feature_importance_rf_df = feature_importance_rf_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_rf_df.head(10))

    # --- 10. Model Training and Evaluation: Gradient Boosting Classifier ---
    print("\n\n--- Starting Model Training (Gradient Boosting Classifier) ---")
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train_scaled, y_train)
    print("Gradient Boosting model training complete.")

    y_pred_gb = gb_model.predict(X_test_scaled)
    y_pred_proba_gb = gb_model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Gradient Boosting Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_gb):.4f}")
    print("Confusion Matrix (Gradient Boosting):")
    print(confusion_matrix(y_test, y_pred_gb))
    print("Classification Report (Gradient Boosting):")
    print(classification_report(y_test, y_pred_gb, target_names=['Non-Churn (0)', 'Churn (1)']))

    print("\n--- Feature Importances (Gradient Boosting) ---")
    gb_importances = gb_model.feature_importances_
    feature_importance_gb_df = pd.DataFrame({'Feature': feature_columns_for_saving, 'Importance': gb_importances})
    feature_importance_gb_df = feature_importance_gb_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_gb_df.head(10))

    # --- 11. Identify Specific Customers Predicted to Churn (Using Random Forest) ---
    print("\n\n--- Identifying Specific Customers from Test Set Predicted to Churn (Random Forest Model) ---")
    if customer_ids_full_dataset is not None and X_test_indices is not None:
        test_customer_ids_actual = customer_ids_full_dataset.loc[X_test_indices]
        churn_predictions_df = pd.DataFrame({
            'Customer_ID': test_customer_ids_actual.values,
            'Actual_Churn': y_test.values,
            'Predicted_Churn_RF': y_pred_rf,
            'Churn_Probability_RF': y_pred_proba_rf
        })
        predicted_to_churn_list_rf = churn_predictions_df[churn_predictions_df['Predicted_Churn_RF'] == 1]
        print(f"\nTotal customers in test set predicted to churn by Random Forest: {len(predicted_to_churn_list_rf)}")
        print("Top 5 customers from test set predicted by Random Forest to churn (with their churn probability):")
        print(predicted_to_churn_list_rf[['Customer_ID', 'Churn_Probability_RF']].sort_values(by='Churn_Probability_RF',
                                                                                              ascending=False).head())
        true_positive_churners = predicted_to_churn_list_rf[predicted_to_churn_list_rf['Actual_Churn'] == 1]
        print(f"\nOf those, {len(true_positive_churners)} were correctly predicted as churners (True Positives).")
        print("Top 5 True Positive Churners (predicted to churn and actually churned):")
        print(true_positive_churners[['Customer_ID', 'Churn_Probability_RF']].sort_values(by='Churn_Probability_RF',
                                                                                          ascending=False).head())
    else:
        print("Customer IDs were not available or X_test indices were not stored; cannot list specific customers.")

    # --- 12. Save the Model, Scaler, Feature Columns, and Median Years ---
    #    (Focusing on Random Forest as it was slightly than logical regression)
    print("\n\n--- Saving Model Artifacts ---")

    # Save the Random Forest Model
    model_filename = 'rf_churn_model.joblib'
    joblib.dump(rf_model_to_save, model_filename)  # Use rf_model_to_save
    print(f"Random Forest model saved as {model_filename}")

    # Save the Scaler
    scaler_filename = 'scaler.joblib'
    joblib.dump(scaler, scaler_filename)  # scaler was defined and fit before
    print(f"Scaler saved as {scaler_filename}")

    # Save the list of feature columns (in the correct order)
    columns_filename = 'feature_columns.joblib'
    joblib.dump(feature_columns_for_saving, columns_filename)
    print(f"Feature columns saved as {columns_filename}")

    # Save median_years_for_loyalty (calculated during feature engineering)
    if 'median_years_for_loyalty' in locals() and median_years_for_loyalty != 0:  # Check if it was calculated
        median_years_filename = 'median_years_for_loyalty.joblib'
        joblib.dump(median_years_for_loyalty, median_years_filename)
        print(f"Median years for loyalty feature saved: {median_years_for_loyalty} as {median_years_filename}")
    else:
        print(
            "Warning: 'median_years_for_loyalty' was not calculated or available for saving. Streamlit app might need a default.")

        if 'Years_as_Customer' in df_processed.columns:
            fallback_median_years = df_processed['Years_as_Customer'].median()
            print(
                f"Calculated fallback median years from df_processed: {fallback_median_years}. Consider saving this if needed.")

    print("\n\n--- End of Script: model_training_and_saving.py ---")

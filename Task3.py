import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def prepare_data(file_path='Task 3 and 4_Loan_Data.csv'):
    """Load the dataset and prepare the features and target variable."""
    df = pd.read_csv(file_path)
    X = df[['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
    y = df['default']  # Assuming 'default' is the target variable
    return X, y

def train_and_evaluate(X, y):
    """Train the logistic regression model and evaluate its performance."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10000)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

   
    
    return model, scaler

def calculate_expected_loss(model, scaler, features):
    """Calculate the expected loss of a loan given its features."""
    features_df = pd.DataFrame([features])
    features_scaled = scaler.transform(features_df)
    probability_of_default = model.predict_proba(features_scaled)[0][1]  # Probability of default
    
    ead = features['loan_amt_outstanding']  # Exposure at default
    lgd = 0.9  # Assuming a recovery rate of 10%
    expected_loss = round(probability_of_default * ead * lgd,3)
    
    return expected_loss

# Main execution
def main():
    # The function now uses the default file path if no argument is provided
    X, y = prepare_data()
    model, scaler = train_and_evaluate(X, y)

    # Example of how to use the calculate_expected_loss function:
    example_features = {
        'credit_lines_outstanding': 5,
        'loan_amt_outstanding': 10000,
        'total_debt_outstanding': 15000,
        'income': 50000,
        'years_employed': 10,
        'fico_score': 680
    }
    
    print("Expected Loss:", calculate_expected_loss(model, scaler, example_features))

main()
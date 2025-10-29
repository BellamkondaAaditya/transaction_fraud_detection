"""
Make fraud predictions on new transaction data.

This script loads the trained model and makes predictions on new transactions.
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path


def load_model_artifacts(model_dir='../models'):
    """Load the trained model and preprocessing artifacts."""
    model_dir = Path(model_dir)

    model = joblib.load(model_dir / 'fraud_detection_model.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')

    with open(model_dir / 'feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]

    return model, scaler, feature_names


def preprocess_transaction(transaction_df, feature_names):
    """Preprocess a single transaction for prediction."""

    # Feature engineering (same as training)
    transaction_df['amount_vs_avg_ratio'] = transaction_df['amount'] / (transaction_df['avg_transaction_amount'] + 1)
    transaction_df['high_amount'] = (transaction_df['amount'] > 100).astype(int)
    transaction_df['frequent_transactions'] = (transaction_df['num_transactions_24h'] > 5).astype(int)
    transaction_df['unusual_location'] = (transaction_df['location_distance'] > 50).astype(int)

    # One-hot encode categorical variables
    categorical_cols = ['merchant_category', 'transaction_type', 'device_type']
    transaction_encoded = pd.get_dummies(transaction_df, columns=categorical_cols, drop_first=True)

    # Ensure all features are present
    for feature in feature_names:
        if feature not in transaction_encoded.columns:
            transaction_encoded[feature] = 0

    # Select only the features used in training
    transaction_encoded = transaction_encoded[feature_names]

    return transaction_encoded


def predict_fraud(transaction_data, model, scaler, feature_names):
    """Predict fraud probability for a transaction."""

    # Preprocess
    transaction_processed = preprocess_transaction(transaction_data, feature_names)

    # Scale
    transaction_scaled = scaler.transform(transaction_processed)

    # Predict
    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0, 1]

    return prediction, probability


def main():
    """Example usage of the prediction script."""

    print("Loading trained model...")
    model, scaler, feature_names = load_model_artifacts()
    print("Model loaded successfully!\n")

    # Example transaction
    example_transaction = pd.DataFrame([{
        'amount': 250.00,
        'merchant_category': 'online_retail',
        'transaction_type': 'online',
        'location_distance': 85.5,
        'num_transactions_24h': 8,
        'avg_transaction_amount': 45.00,
        'time_since_last_transaction': 12.5,
        'device_type': 'desktop',
        'hour_of_day': 2,
        'day_of_week': 3,
        'is_weekend': 0,
        'is_night': 1
    }])

    print("Example Transaction:")
    print("-" * 50)
    for col, val in example_transaction.iloc[0].items():
        print(f"{col:30s}: {val}")

    # Predict
    prediction, probability = predict_fraud(example_transaction, model, scaler, feature_names)

    print("\n" + "="*50)
    print("FRAUD DETECTION RESULT")
    print("="*50)
    print(f"Fraud Probability: {probability:.2%}")
    print(f"Prediction: {'FRAUDULENT' if prediction == 1 else 'LEGITIMATE'}")
    print(f"Risk Level: {'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'}")
    print("="*50)


if __name__ == '__main__':
    main()

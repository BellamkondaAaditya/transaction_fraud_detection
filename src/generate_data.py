"""
Download and prepare transaction data for fraud detection analysis.

This script downloads the Credit Card Fraud Detection dataset from Kaggle
or generates synthetic data if the download fails.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def download_kaggle_dataset():
    """
    Download the Credit Card Fraud Detection dataset from Kaggle.

    This dataset contains transactions made by credit cards in September 2013
    by European cardholders. It presents transactions that occurred in two days,
    where we have 492 frauds out of 284,807 transactions.

    Dataset URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """
    try:
        print("Attempting to download Credit Card Fraud Detection dataset from Kaggle...")
        print("Note: This requires Kaggle API credentials configured.")
        print("If download fails, synthetic data will be generated instead.\n")

        # Try to import kaggle
        import kaggle

        # Download dataset
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path='../data',
            unzip=True
        )

        # Load the dataset
        df = pd.read_csv('../data/creditcard.csv')

        # Rename columns for clarity
        df = df.rename(columns={'Class': 'is_fraud'})

        print(f"Successfully downloaded dataset from Kaggle!")
        print(f"Total transactions: {len(df)}")
        print(f"Fraudulent transactions: {df['is_fraud'].sum()}")
        print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%\n")

        return df

    except Exception as e:
        print(f"Could not download from Kaggle: {e}")
        print("Generating synthetic dataset instead...\n")
        return None


def generate_transactions(n_samples=10000, fraud_ratio=0.02):
    """
    Generate synthetic transaction dataset.

    Parameters:
    -----------
    n_samples : int
        Total number of transactions to generate
    fraud_ratio : float
        Proportion of fraudulent transactions (default: 2%)

    Returns:
    --------
    pd.DataFrame
        DataFrame containing transaction data
    """

    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud

    # Generate timestamps (last 30 days)
    start_date = datetime.now() - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=random.randint(0, 30*24*60*60)
    ) for _ in range(n_samples)]

    # Initialize lists for features
    data = {
        'transaction_id': [f'TXN{str(i).zfill(6)}' for i in range(n_samples)],
        'timestamp': timestamps,
        'amount': [],
        'merchant_category': [],
        'transaction_type': [],
        'location_distance': [],
        'num_transactions_24h': [],
        'avg_transaction_amount': [],
        'time_since_last_transaction': [],
        'device_type': [],
        'is_fraud': []
    }

    merchant_categories = ['grocery', 'restaurant', 'gas_station', 'online_retail',
                          'electronics', 'travel', 'entertainment', 'healthcare']
    transaction_types = ['chip', 'swipe', 'online', 'contactless']
    device_types = ['mobile', 'desktop', 'tablet', 'pos_terminal']

    # Generate legitimate transactions
    for i in range(n_legitimate):
        data['amount'].append(np.random.lognormal(3.5, 1.2))  # Mean ~$50
        data['merchant_category'].append(np.random.choice(merchant_categories, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05, 0.05]))
        data['transaction_type'].append(np.random.choice(transaction_types, p=[0.4, 0.3, 0.2, 0.1]))
        data['location_distance'].append(np.random.exponential(10))  # km from usual location
        data['num_transactions_24h'].append(np.random.poisson(3))
        data['avg_transaction_amount'].append(np.random.lognormal(3.5, 0.8))
        data['time_since_last_transaction'].append(np.random.exponential(120))  # minutes
        data['device_type'].append(np.random.choice(device_types, p=[0.35, 0.25, 0.15, 0.25]))
        data['is_fraud'].append(0)

    # Generate fraudulent transactions (with distinct patterns)
    for i in range(n_fraud):
        # Fraudulent transactions tend to have:
        # - Higher amounts
        # - More online transactions
        # - Greater distance from usual location
        # - Multiple transactions in short time
        # - Higher transaction amounts than user average

        data['amount'].append(np.random.lognormal(4.5, 1.5))  # Higher amounts
        data['merchant_category'].append(np.random.choice(merchant_categories, p=[0.05, 0.05, 0.05, 0.35, 0.30, 0.10, 0.05, 0.05]))
        data['transaction_type'].append(np.random.choice(transaction_types, p=[0.1, 0.1, 0.7, 0.1]))  # More online
        data['location_distance'].append(np.random.exponential(100) + 50)  # Far from usual location
        data['num_transactions_24h'].append(np.random.poisson(8) + 5)  # Burst of transactions
        data['avg_transaction_amount'].append(np.random.lognormal(3.0, 0.6))  # Lower historical average
        data['time_since_last_transaction'].append(np.random.exponential(15))  # Short time since last
        data['device_type'].append(np.random.choice(device_types, p=[0.20, 0.50, 0.20, 0.10]))  # More desktop
        data['is_fraud'].append(1)

    # Create DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)

    # Add derived features
    df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['hour_of_day'].isin(range(0, 6)).astype(int)

    # Round numerical columns
    df['amount'] = df['amount'].round(2)
    df['location_distance'] = df['location_distance'].round(2)
    df['avg_transaction_amount'] = df['avg_transaction_amount'].round(2)
    df['time_since_last_transaction'] = df['time_since_last_transaction'].round(2)

    return df


if __name__ == '__main__':
    # Try to download real dataset first
    df = download_kaggle_dataset()

    # If download failed, generate synthetic data
    if df is None:
        print("Generating synthetic transaction data...")
        df = generate_transactions(n_samples=10000, fraud_ratio=0.02)
        output_path = '../data/transactions.csv'
    else:
        # Use the Kaggle dataset
        output_path = '../data/creditcard.csv'
        # Also create a processed version
        df_processed = df.copy()
        df_processed.to_csv('../data/transactions_processed.csv', index=False)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"\nDataset saved successfully!")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Legitimate transactions: {(1-df['is_fraud']).sum()} ({(1-df['is_fraud'].mean())*100:.2f}%)")
    print(f"Saved to: {output_path}")

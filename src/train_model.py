"""
Train fraud detection models and save the best performer.

This script handles the complete model training pipeline including:
- Data loading and preprocessing
- Feature engineering
- Handling class imbalance with SMOTE
- Training multiple classifiers
- Model evaluation and comparison
- Saving the best model
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(data_path):
    """Load and preprocess transaction data."""
    print("Loading data...")
    df = pd.read_csv(data_path)

    print(f"Dataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

    # Feature engineering
    print("\nEngineering features...")
    df['amount_vs_avg_ratio'] = df['amount'] / (df['avg_transaction_amount'] + 1)
    df['high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
    df['frequent_transactions'] = (df['num_transactions_24h'] > 5).astype(int)
    df['unusual_location'] = (df['location_distance'] > df['location_distance'].quantile(0.90)).astype(int)

    # One-hot encode categorical variables
    categorical_cols = ['merchant_category', 'transaction_type', 'device_type']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Select features
    feature_cols = [col for col in df_encoded.columns if col not in
                    ['transaction_id', 'timestamp', 'is_fraud']]

    X = df_encoded[feature_cols]
    y = df_encoded['is_fraud']

    return X, y, feature_cols


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results."""

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    print("Applying SMOTE for balanced training...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"Training samples after SMOTE: {X_train_balanced.shape[0]}")

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    trained_models = {}

    print("\n" + "="*60)
    print("Training and evaluating models...")
    print("="*60)

    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 40)

        # Train
        model.fit(X_train_balanced, y_train_balanced)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }

        trained_models[name] = model

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0, 0]:>6} | FP: {cm[0, 1]:>6}")
        print(f"FN: {cm[1, 0]:>6} | TP: {cm[1, 1]:>6}")

    return trained_models, scaler, results


def save_best_model(trained_models, scaler, results, feature_cols, output_dir):
    """Save the best performing model and artifacts."""

    # Find best model based on ROC-AUC
    best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
    best_model = trained_models[best_model_name]
    best_results = results[best_model_name]

    print("\n" + "="*60)
    print(f"Best Model: {best_model_name}")
    print(f"ROC-AUC Score: {best_results['roc_auc']:.4f}")
    print("="*60)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / 'fraud_detection_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save scaler
    scaler_path = output_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Save feature names
    features_path = output_dir / 'feature_names.txt'
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Feature names saved to: {features_path}")

    # Save results
    results_path = output_dir / 'training_results.json'
    # Convert numpy types to Python types for JSON serialization
    results_json = {}
    for model_name, metrics in results.items():
        results_json[model_name] = {
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc']),
            'confusion_matrix': metrics['confusion_matrix']
        }

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Training results saved to: {results_path}")

    return best_model_name


def main():
    """Main training pipeline."""
    print("="*60)
    print("Transaction Fraud Detection - Model Training")
    print("="*60)

    # Paths
    data_path = '../data/transactions.csv'
    model_dir = '../models'

    # Load and preprocess
    X, y, feature_cols = load_and_preprocess_data(data_path)

    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples ({y_train.sum()} frauds)")
    print(f"Test set: {X_test.shape[0]} samples ({y_test.sum()} frauds)")

    # Train models
    trained_models, scaler, results = train_models(X_train, X_test, y_train, y_test)

    # Save best model
    best_model = save_best_model(trained_models, scaler, results, feature_cols, model_dir)

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()

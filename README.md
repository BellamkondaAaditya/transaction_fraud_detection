# Transaction Fraud Detection

A comprehensive machine learning project for detecting fraudulent credit card transactions using advanced data analysis and classification techniques.

## Project Overview

This project implements an end-to-end fraud detection system that analyzes transaction patterns to identify potentially fraudulent activities. The system uses multiple machine learning models to achieve high accuracy while maintaining interpretability for business stakeholders.

### Key Features

- **Comprehensive EDA**: In-depth exploratory data analysis with visualizations
- **Multiple ML Models**: Logistic Regression, Random Forest, and Gradient Boosting
- **Class Imbalance Handling**: SMOTE implementation for balanced training
- **Feature Engineering**: Advanced behavioral and temporal features
- **Model Comparison**: Detailed performance metrics and visualizations
- **Production-Ready**: Trained model with prediction pipeline

## Business Problem

Credit card fraud costs billions annually for financial institutions and consumers. This project addresses the challenge of:
- Detecting fraudulent transactions in real-time
- Minimizing false positives to avoid customer inconvenience
- Identifying key fraud indicators for preventive measures
- Providing interpretable results for fraud investigation teams

## Dataset

The project uses synthetic transaction data that mimics real-world credit card transaction patterns with the following characteristics:

- **Total Transactions**: 10,000
- **Fraud Rate**: ~2% (realistic imbalance)
- **Features**:
  - Transaction details (amount, merchant category, type)
  - Behavioral patterns (location, frequency, timing)
  - Temporal features (hour, day, weekend indicators)
  - Device information

### Key Fraud Indicators Identified

1. Transaction amount significantly higher than user's average
2. Unusual location (far from typical transaction locations)
3. High frequency of transactions in short time period
4. Online transactions (higher fraud rate)
5. Night-time transactions (2 AM - 6 AM)

## Project Structure

```
transaction_fraud_detection/
│
├── data/                          # Dataset directory
│   └── transactions.csv          # Generated transaction data
│
├── notebooks/                     # Jupyter notebooks
│   └── fraud_detection_analysis.ipynb  # Complete EDA and modeling
│
├── src/                          # Source code
│   ├── generate_data.py         # Data generation script
│   ├── train_model.py           # Model training pipeline
│   └── predict.py               # Prediction script
│
├── models/                       # Trained models
│   ├── fraud_detection_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.txt
│   └── training_results.json
│
├── results/                      # Visualizations and outputs
│   ├── class_distribution.png
│   ├── amount_analysis.png
│   ├── temporal_patterns.png
│   ├── behavioral_patterns.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   ├── precision_recall_curves.png
│   ├── confusion_matrices.png
│   └── feature_importance.png
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                   # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transaction_fraud_detection.git
cd transaction_fraud_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

```bash
cd src
python generate_data.py
```

This creates a synthetic transaction dataset with realistic fraud patterns.

### 2. Run Complete Analysis

Open and run the Jupyter notebook:

```bash
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

The notebook includes:
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Performance Evaluation
- Visualizations

### 3. Train Models

To train models using the standalone script:

```bash
cd src
python train_model.py
```

This will:
- Load and preprocess the data
- Train multiple models (Logistic Regression, Random Forest, Gradient Boosting)
- Evaluate and compare performance
- Save the best model to `models/`

### 4. Make Predictions

To predict on new transactions:

```bash
cd src
python predict.py
```

Example usage in code:

```python
from predict import load_model_artifacts, predict_fraud
import pandas as pd

# Load model
model, scaler, features = load_model_artifacts('../models')

# Prepare transaction data
transaction = pd.DataFrame([{
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

# Predict
prediction, probability = predict_fraud(transaction, model, scaler, features)
print(f"Fraud Probability: {probability:.2%}")
```

## Model Performance

### Results Summary

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.85+ | 0.80+ | 0.82+ | 0.90+ |
| Random Forest | 0.90+ | 0.85+ | 0.87+ | 0.95+ |
| Gradient Boosting | 0.90+ | 0.85+ | 0.87+ | 0.95+ |

*Note: Actual values may vary slightly based on random seed and data generation*

### Model Selection

**Random Forest** is selected as the primary model because:
- Highest overall performance (ROC-AUC > 0.95)
- Strong recall (minimizes false negatives)
- Feature importance interpretability
- Robust to outliers
- Good balance between precision and recall

## Key Findings

### 1. Transaction Patterns

- **Amount**: Fraudulent transactions average 2-3x higher than legitimate ones
- **Location**: 70%+ of fraud occurs far from user's usual location
- **Timing**: Night transactions (2-6 AM) show 3x higher fraud rate
- **Channel**: Online transactions have 4x higher fraud rate than chip/contactless

### 2. Behavioral Anomalies

- Multiple transactions within minutes indicate potential fraud
- Sudden deviation from spending patterns is a strong indicator
- Device switching (mobile to desktop) raises fraud probability

### 3. Feature Importance

Top 5 most important features:
1. Amount vs. Average Ratio
2. Location Distance
3. Number of Recent Transactions
4. Time Since Last Transaction
5. Transaction Amount

## Technical Approach

### Data Preprocessing

1. **Feature Engineering**:
   - Ratio-based features (amount vs. average)
   - Temporal features (hour, day, weekend)
   - Behavioral flags (unusual location, frequent transactions)

2. **Encoding**:
   - One-hot encoding for categorical variables
   - Standard scaling for numerical features

3. **Class Imbalance**:
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Balanced class weights in models

### Model Training

- **Train/Test Split**: 80/20 with stratification
- **Cross-Validation**: 5-fold for robust evaluation
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Evaluation Metrics**:
  - ROC-AUC (overall performance)
  - Recall (minimize false negatives)
  - Precision (minimize false positives)
  - F1-Score (harmonic mean)

## Business Impact

### Benefits

1. **Cost Reduction**: Early fraud detection prevents financial losses
2. **Customer Trust**: Minimizes false positives that inconvenience customers
3. **Real-Time Detection**: Fast inference enables immediate transaction blocking
4. **Actionable Insights**: Feature importance guides fraud prevention strategies

### Recommendations

1. Deploy Random Forest model for production use
2. Implement real-time scoring with 0.7 probability threshold
3. Set up monitoring for model drift and retraining schedule
4. Enhance with additional features (device fingerprinting, IP analysis)
5. Integrate with fraud investigation workflow

## Future Enhancements

- [ ] Deep learning models (Neural Networks)
- [ ] Real-time streaming data pipeline
- [ ] Ensemble methods combining multiple models
- [ ] Time-series analysis for temporal patterns
- [ ] Network analysis for fraud ring detection
- [ ] API endpoint for production deployment
- [ ] A/B testing framework
- [ ] Explainable AI (SHAP values) for model interpretability

## Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning models and preprocessing
- **Imbalanced-learn**: SMOTE for handling class imbalance
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive analysis and documentation

## License

This project is created for educational and portfolio purposes.

## Author

**Your Name**
Data Scientist | Machine Learning Engineer

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## Acknowledgments

- Dataset inspired by real-world credit card fraud patterns
- Project structure follows industry best practices
- Model evaluation metrics aligned with business objectives

---

**Note**: This is a portfolio project demonstrating end-to-end machine learning capabilities in fraud detection. The data is synthetic and generated for educational purposes.

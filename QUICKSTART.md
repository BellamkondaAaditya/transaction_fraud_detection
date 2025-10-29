# Quick Start Guide

Get up and running with the Transaction Fraud Detection project in 5 minutes!

## Setup (2 minutes)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate data**:
   ```bash
   cd src
   python generate_data.py
   ```

## Run Analysis (3 minutes)

### Option 1: Complete Interactive Analysis (Recommended for recruiters)

```bash
jupyter notebook notebooks/fraud_detection_analysis.ipynb
```

Then run all cells (Cell > Run All) to see:
- Data exploration with visualizations
- Feature analysis
- Multiple models comparison
- Performance metrics
- Feature importance

### Option 2: Quick Model Training

```bash
cd src
python train_model.py
```

### Option 3: Test Predictions

```bash
cd src
python predict.py
```

## What You'll See

1. **Data Insights**: 10,000 transactions with 2% fraud rate
2. **Model Performance**: 95%+ ROC-AUC scores across all models
3. **Visualizations**: 10+ professional charts showing patterns
4. **Key Findings**: Critical fraud indicators identified

## Project Highlights for Recruiters

- **Complete ML Pipeline**: Data generation → EDA → Feature Engineering → Model Training → Evaluation
- **Multiple Models**: Logistic Regression, Random Forest, Gradient Boosting
- **Class Imbalance Handling**: SMOTE implementation
- **Production Ready**: Includes prediction pipeline and model artifacts
- **Well Documented**: Comprehensive README and code comments
- **Business Focus**: Clear business problem, impact, and recommendations

## Next Steps

- Review the Jupyter notebook for detailed analysis
- Check `models/training_results.json` for performance metrics
- Explore `results/` folder for visualizations
- Read `README.md` for comprehensive documentation

---

**Need Help?** Check the main README.md for detailed instructions.

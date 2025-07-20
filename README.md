# Credit Card Fraud Detection System

A machine learning system for detecting fraudulent credit card transactions achieving 97.6% precision and 81.6% recall using Random Forest classification.

## Project Overview

This project develops a fraud detection model using a dataset of 284,807 credit card transactions from European cardholders. The main challenge was building an accurate classifier for highly imbalanced data with only 0.173% fraud rate while maintaining low false positive rates for production use.

The final Random Forest model achieves 97.6% precision with 81.6% recall, resulting in only 2 false positives per 57,000 transactions. This translates to minimal customer friction while catching approximately 4 out of 5 fraudulent transactions.

## Technical Approach

### Data Analysis

Initial analysis revealed key fraud patterns. Night transactions between 00:00-06:00 are 3x more likely to be fraudulent. Additionally, 50.6% of frauds involve small amounts under 10 EUR compared to 34.1% for normal transactions. The V-features, which are PCA-transformed variables, showed strong discriminative power with V17, V14, and V12 having the highest effect sizes.

### Feature Engineering

The original 31 features were reduced to 9 optimized features that maintained 90% of model performance. The core V-features V17, V14, V12, V10, and V16 were kept as the strongest fraud indicators. 

I engineered four additional features: is_night_hour as a binary indicator for transactions during 00:00-06:00, is_small_amount for amounts under 10 EUR, hour as a continuous time feature, and amount_log as the log-transformed transaction amount.

### Model Selection

I tested four algorithms optimized for class imbalance. Random Forest achieved the best results with 97.6% precision, 81.6% recall, 88.9% F1-score and 87.9% PR-AUC. XGBoost came second with 82.5% precision and 86.7% recall. LightGBM had high recall at 87.8% but poor precision at 57.3%. Logistic Regression performed poorly with only 6.2% precision despite 90.8% recall.

Random Forest was selected for its superior precision-recall balance, natural handling of class imbalance with balanced weights, robust ensemble performance without overfitting, and minimal false positives critical for production deployment.

The model uses balanced class weights to address the 1:578 class imbalance ratio. Feature importance analysis shows V14 contributes 39.6%, V10 contributes 18.5%, V12 contributes 16.0%, V17 contributes 13.4%, and V16 contributes 5.2%.

## Usage

```python
from fraud_predictor import FraudPredictor

predictor = FraudPredictor()
predictor.load_model()

test_case = {
    'V17': -5.0, 'V14': -3.0, 'V12': -2.0,
    'V10': -1.5, 'V16': -1.0, 'is_night_hour': 1,
    'is_small_amount': 1, 'hour': 2.5, 'amount_log': 2.3
}

result = predictor.predict_from_dict(test_case)
predictor.print_prediction(result)
```

Output:
```
FRAUD PREDICTION ANALYSIS
Risk Level: High
Fraud Probability: 78.5%
Prediction: FRAUD

Analysis:
V14 is low (-3.00) - increases fraud risk; 
V17 is low (-5.00) - increases fraud risk; 
Night transaction (00:00-06:00) - increases fraud risk
```

The confusion matrix shows 56,862 true negatives, 2 false positives, 18 false negatives, and 80 true positives. This results in a false positive rate of 0.004% and estimated fraud prevention of 9,760 EUR per test batch.

## Project Structure

The project contains the processed dataset in the data folder, trained models in the models folder including the Random Forest classifier and feature metadata, and the main prediction system script.

## Methodology and Findings

The exploratory data analysis identified temporal and amount-based fraud patterns and calculated effect sizes for V-features. Feature engineering created time-based risk indicators and reduced the feature space while maintaining performance.

Model development used stratified train/test split to preserve fraud ratio, PR-AUC and F1-score as primary metrics, and class balancing techniques. Validation included confusion matrix analysis and production readiness assessment.

Key insights include clear temporal fraud patterns, significantly different amount distributions between fraud and normal transactions, and V14, V17, V12 providing the strongest fraud signals. The class imbalance requires specialized handling, feature selection is critical for performance, ensemble methods outperform linear models, and precision optimization is essential for production systems.

The Random Forest model achieves optimal precision/recall tradeoff with 97.6% precision enabling production deployment and 81.6% recall providing substantial fraud prevention capability.

## Requirements

Python 3.8 or higher with scikit-learn 1.6+, pandas, numpy, and joblib.

## Dataset

European credit card transactions from September 2013 containing 284,807 total transactions with 492 fraudulent cases. The dataset has 31 features with PCA transformation applied to V1-V28 for privacy protection and spans 2 days of transaction data.

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class FraudPredictor:
    """
    Yksittäisten tapahtumien fraud-ennustaja
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def train_model(self, data_file='../data/creditcard_minimal_features.csv'):
        """Train model if not already saved"""
        print("Training Random Forest model...")

        # Load data
        df = pd.read_csv(data_file)
        X = df.drop('Class', axis=1)
        y = df['Class']

        self.feature_names = X.columns.tolist()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))

        # Save model to models folder (relative to project root)
        import os
        os.makedirs('../models', exist_ok=True)

        joblib.dump(self.model, '../models/fraud_model.pkl')
        joblib.dump(self.feature_names, '../models/feature_names.pkl')
        joblib.dump(self.feature_importance, '../models/feature_importance.pkl')

        print("Model trained and saved!")
        return self.model

    def load_model(self):
        """Load saved model"""
        try:
            self.model = joblib.load('../models/fraud_model.pkl')
            self.feature_names = joblib.load('../models/feature_names.pkl')
            self.feature_importance = joblib.load('../models/feature_importance.pkl')
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("Saved model not found - train first!")
            return False

    def predict_fraud(self, V17, V14, V12, V10, V16, is_night_hour, is_small_amount, hour, amount_log):
        """
        Ennusta yksittäisen tapahtuman fraud-riski

        Args:
            V17, V14, V12, V10, V16: PCA-muunnettuja piirteitä (float)
            is_night_hour: Onko yöaika (0 tai 1)
            is_small_amount: Onko pieni summa (<10€) (0 tai 1)
            hour: Tunti (0-23)
            amount_log: log(1 + amount)

        Returns:
            dict: Ennustustulos
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("Malli ei ole ladattu eikä tallennettua mallia löydy!")

        # Luo input array
        input_data = np.array([[V17, V14, V12, V10, V16, is_night_hour, is_small_amount, hour, amount_log]])

        # Ennuste
        probability = self.model.predict_proba(input_data)[0, 1]  # Fraud probability
        prediction = self.model.predict(input_data)[0]

        # Fraud probability (no need for score/10)
        probability = self.model.predict_proba(input_data)[0, 1]  # Fraud probability
        prediction = self.model.predict(input_data)[0]

        # Risk level based on probability
        if probability < 0.05:
            risk_level = "Very Low"
            risk_color = "🟢"
        elif probability < 0.25:
            risk_level = "Low"
            risk_color = "🟡"
        elif probability < 0.6:
            risk_level = "Medium"
            risk_color = "🟠"
        elif probability < 0.85:
            risk_level = "High"
            risk_color = "🔴"
        else:
            risk_level = "Critical"
            risk_color = "🚨"

        # Smart feature contribution analysis
        feature_values = np.array([V17, V14, V12, V10, V16, is_night_hour, is_small_amount, hour, amount_log])

        # Calculate proper contributions using feature importance and deviation from baseline
        contributions = []
        baselines = {
            'V17': 0.0, 'V14': 0.0, 'V12': 0.0, 'V10': 0.0, 'V16': 0.0,
            'is_night_hour': 0.0, 'is_small_amount': 0.0, 'hour': 12.0, 'amount_log': 3.5
        }

        for i, (name, value) in enumerate(zip(self.feature_names, feature_values)):
            baseline = baselines[name]
            deviation = value - baseline
            importance = self.feature_importance[name]

            # For V features (negative correlation with fraud)
            if name.startswith('V'):
                # Lower values = higher fraud risk
                contribution = -deviation * importance * 10
            elif name in ['is_night_hour', 'is_small_amount']:
                # Binary features: 1 = higher risk
                contribution = value * importance * 20
            elif name == 'hour':
                # Night hours (0-6) are riskier
                night_risk = 1.0 if 0 <= value <= 6 else 0.0
                contribution = night_risk * importance * 15
            else:
                # Other features
                contribution = deviation * importance * 5

            contributions.append({
                'feature': name,
                'value': value,
                'contribution': contribution,
                'importance': importance,
                'impact': 'increases' if contribution > 0 else 'decreases'
            })

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

        # Create explanation focusing on top risk factors
        explanation_parts = []
        for contrib in contributions[:3]:
            if abs(contrib['contribution']) > 0.01:  # Only significant contributions
                if contrib['feature'].startswith('V'):
                    if contrib['contribution'] > 0:
                        explanation_parts.append(
                            f"{contrib['feature']} is low ({contrib['value']:.2f}) - increases fraud risk")
                    else:
                        explanation_parts.append(
                            f"{contrib['feature']} is normal ({contrib['value']:.2f}) - decreases fraud risk")
                elif contrib['feature'] == 'is_night_hour':
                    if contrib['value'] == 1:
                        explanation_parts.append("Night transaction (00:00-06:00) - increases fraud risk")
                    else:
                        explanation_parts.append("Daytime transaction - normal risk")
                elif contrib['feature'] == 'is_small_amount':
                    if contrib['value'] == 1:
                        explanation_parts.append("Small amount (<10€) - increases fraud risk")
                    else:
                        explanation_parts.append("Normal amount - lower risk")
                else:
                    explanation_parts.append(f"{contrib['feature']} {contrib['impact']} risk")

        explanation = "; ".join(explanation_parts) if explanation_parts else "All features indicate normal transaction"

        return {
            'probability': probability,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'prediction': 'FRAUD' if prediction == 1 else 'NORMAL',
            'explanation': explanation,
            'top_contributions': contributions[:5],
            'input_values': {
                'V17': V17, 'V14': V14, 'V12': V12, 'V10': V10, 'V16': V16,
                'is_night_hour': is_night_hour, 'is_small_amount': is_small_amount,
                'hour': hour, 'amount_log': amount_log
            }
        }

    def predict_from_dict(self, transaction_data):
        """Ennusta dictionary-datasta"""
        return self.predict_fraud(
            V17=transaction_data['V17'],
            V14=transaction_data['V14'],
            V12=transaction_data['V12'],
            V10=transaction_data['V10'],
            V16=transaction_data['V16'],
            is_night_hour=transaction_data['is_night_hour'],
            is_small_amount=transaction_data['is_small_amount'],
            hour=transaction_data['hour'],
            amount_log=transaction_data['amount_log']
        )

    def batch_predict(self, transactions_df):
        """Ennusta useita tapahtumia kerralla"""
        results = []
        for _, transaction in transactions_df.iterrows():
            result = self.predict_from_dict(transaction.to_dict())
            results.append(result)
        return results

    def print_prediction(self, result):
        """Print prediction results"""
        print(f"\nFRAUD PREDICTION ANALYSIS")
        print("=" * 25)
        print(f"Risk Level: {result['risk_level']}")
        print(f"Fraud Probability: {result['probability']:.1%}")
        print(f"Prediction: {result['prediction']}")
        print(f"\nAnalysis:")
        print(f"   {result['explanation']}")

        print(f"\nFeature Impact Details:")
        for i, contrib in enumerate(result['top_contributions']):
            direction = "increases risk" if contrib['contribution'] > 0 else "decreases risk"
            print(
                f"   {i + 1}. {contrib['feature']:15} {direction:15} (value: {contrib['value']:.3f}, impact: {contrib['contribution']:+.3f})")

    def create_examples(self):
        """Luo esimerkkitapauksia testausta varten"""
        examples = {
            'high_risk_night': {
                'name': 'High Risk Night Transaction',
                'data': {
                    'V17': -5.0,  # Matala V17 = riski
                    'V14': -3.0,  # Matala V14 = riski
                    'V12': -2.0,  # Matala V12 = riski
                    'V10': -1.5,  # Matala V10 = riski
                    'V16': -1.0,  # Matala V16 = riski
                    'is_night_hour': 1,  # Yöaika = riski
                    'is_small_amount': 1,  # Pieni summa = riski
                    'hour': 2.5,  # Keskiyön aikaan
                    'amount_log': 2.3  # log(10) ≈ pienet summat
                }
            },
            'normal_day': {
                'name': 'Normal Day Transaction',
                'data': {
                    'V17': 1.0,  # Normaali V17
                    'V14': 0.5,  # Normaali V14
                    'V12': 0.0,  # Normaali V12
                    'V10': 0.2,  # Normaali V10
                    'V16': 0.1,  # Normaali V16
                    'is_night_hour': 0,  # Päiväaika
                    'is_small_amount': 0,  # Iso summa
                    'hour': 14.0,  # Iltapäivä
                    'amount_log': 4.6  # log(100) ≈ 100€
                }
            },
            'borderline': {
                'name': 'Borderline Case',
                'data': {
                    'V17': -1.0,  # Hieman matala
                    'V14': -0.5,  # Hieman matala
                    'V12': 0.0,  # Neutraali
                    'V10': 0.0,  # Neutraali
                    'V16': 0.0,  # Neutraali
                    'is_night_hour': 0,  # Päiväaika
                    'is_small_amount': 1,  # Pieni summa
                    'hour': 18.0,  # Ilta
                    'amount_log': 3.0  # log(20) ≈ 20€
                }
            }
        }
        return examples


def main():
    """Demo of the fraud prediction function"""
    print("FRAUD PREDICTION FUNCTION DEMO")
    print("==============================")

    # Create predictor
    predictor = FraudPredictor()

    # Try to load model, if not found then train
    if not predictor.load_model():
        print("Training new model...")
        try:
            predictor.train_model()
        except FileNotFoundError:
            print("ERROR: ../data/creditcard_minimal_features.csv not found!")
            print("Run feature engineering --minimal first")
            return

    # Create examples
    examples = predictor.create_examples()

    print("\nTesting predictions:")
    print("-------------------")

    # Test examples
    for example_name, example_data in examples.items():
        print(f"\n{example_data['name'].upper()}")
        print("-" * len(example_data['name']))

        result = predictor.predict_from_dict(example_data['data'])
        predictor.print_prediction(result)

    # Interactive section
    print("\nInteractive usage:")
    print("-----------------")
    print("Use predictor.predict_fraud() or predictor.predict_from_dict() for custom predictions")

    return predictor


if __name__ == "__main__":
    predictor = main()
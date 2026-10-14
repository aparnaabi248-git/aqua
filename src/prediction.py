"""Prediction module for water quality classification."""
import numpy as np
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


def load_prediction_models():
    """Load trained models for prediction."""
    quality_model_path = os.path.join(MODELS_DIR, 'best_model_quality.pkl')
    risk_model_path = os.path.join(MODELS_DIR, 'best_model_risk.pkl')

    if not os.path.exists(quality_model_path):
        return None, None

    quality_model = joblib.load(quality_model_path)
    risk_model = joblib.load(risk_model_path) if os.path.exists(risk_model_path) else None

    return quality_model, risk_model


def predict_water_quality(model_dict, input_features):
    """Predict water quality class and probabilities."""
    model = model_dict['model']
    scaler = model_dict['scaler']
    le = model_dict['label_encoder']

    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    class_name = le.inverse_transform([prediction])[0]

    return class_name, probabilities, le.classes_


def predict_risk(model_dict, input_features):
    """Predict contamination risk class and probabilities."""
    model = model_dict['model']
    scaler = model_dict['scaler']
    le = model_dict['label_encoder']

    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    class_name = le.inverse_transform([prediction])[0]

    return class_name, probabilities, le.classes_

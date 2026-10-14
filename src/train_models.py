"""Model training module."""
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42


def get_models():
    """Get dictionary of models to train."""
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, multi_class='multinomial'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE, min_samples_split=5
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE,
            min_samples_split=5, n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='mlogloss'
        )
    return models


def train_models(X, y, feature_names):
    """Train all models and return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
        model.fit(X_train_scaled, y_train)
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'X_test': X_test_scaled,
            'y_test': y_test,
        }

    return results, scaler

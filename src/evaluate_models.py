"""Model evaluation module."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, classification_report, confusion_matrix,
                            roc_auc_score)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'figures')


def evaluate_model(model, X_test, y_test, le, model_name, target_name):
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)

    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        else:
            roc_auc = 0.0
    except Exception:
        roc_auc = 0.0

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision Macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall Macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1 Macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'F1 Weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc,
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix - {model_name} ({target_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'confusion_matrix_{target_name.lower()}.png'), dpi=150)
    plt.close()

    return metrics, y_pred


def compare_models(results):
    """Create model comparison table."""
    comparison = []
    for name, metrics in results.items():
        comparison.append({
            'Model': name,
            'Accuracy': metrics['Accuracy'],
            'F1 Macro': metrics['F1 Macro'],
            'F1 Weighted': metrics['F1 Weighted'],
            'ROC-AUC': metrics['ROC-AUC'],
        })
    return pd.DataFrame(comparison)

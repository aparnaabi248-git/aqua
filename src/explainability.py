"""Model explainability module using SHAP."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'figures')


def perform_shap_analysis(model, X_test, feature_names, target_name='Quality'):
    """Perform SHAP analysis and generate summary plot."""
    if not SHAP_AVAILABLE:
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                         show=False, plot_size=(12, 8))
        plt.title(f'SHAP Summary Plot - {target_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'shap_summary_{target_name.lower()}.png'), dpi=150)
        plt.close()

        return shap_values
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None


def get_feature_importance(model, feature_names):
    """Get feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return feat_imp
    return None

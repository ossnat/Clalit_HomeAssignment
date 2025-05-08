import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import shap

def plot_rf_importances(model, feature_names):
    importances = model.best_estimator_.named_steps["classifier"].feature_importances_
    feature_names = model.best_estimator_.named_steps["preprocessor"].get_feature_names_out()
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[indices][::-1], importances[indices][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, label_encoder):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_shap_summary(model, X_sample):
    # Extract fitted classifier
    classifier = model.best_estimator_.named_steps["classifier"]
    preprocessor = model.best_estimator_.named_steps["preprocessor"]

    # Preprocess a sample
    X_transformed = preprocessor.transform(X_sample)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    shap.summary_plot(shap_values, X_transformed, feature_names=X_train.columns)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, accuracy_score
)

def evaluate_model(model, X_test, y_test, class_names):
    """
    Generates and prints evaluation metrics for the test set.
    """
    print("\n--- Evaluating Best Model on Hold-Out Test Set ---")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # --- Report Metrics ---
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # --- Plot and Save ROC Curve ---
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('ROC Curve - SVM Classifier (Test Set)')
    save_path = 'figures/roc_curve.png'
    plt.savefig(save_path)
    print(f"\nROC curve saved to {save_path}")
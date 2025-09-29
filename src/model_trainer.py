from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from .feature_extractor import HOGTransformer # Relative import

def train_and_tune_model(X_train, y_train):
    """
    Creates a pipeline and uses GridSearchCV to find the best SVM model.
    """
    print("\n--- Starting Model Training and Hyperparameter Tuning ---")
    
    # Define the full pipeline
    # 1. Extract HOG features using our custom transformer
    # 2. Standardize the resulting feature vectors
    # 3. Train an SVM classifier
    pipeline = Pipeline([
        ('hog', HOGTransformer(pixels_per_cell=(16, 16), cells_per_block=(2, 2))),
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True, random_state=42))
    ])
    
    # Define the parameter grid to search over
    # This grid will test linear, RBF, and polynomial kernels with different params.
    param_grid = [
        {'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 10, 100]},
        {'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 0.01, 0.001]},
        {'svc__kernel': ['poly'], 'svc__C': [0.1, 1, 10], 'svc__degree': [2, 3]}
    ]
    
    # Set up GridSearchCV
    # It will use 5-fold stratified cross-validation and search for the params
    # that maximize the ROC-AUC score.
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    
    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)
    
    # Print the best parameters found
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
    
    # The best_estimator_ attribute is the fully trained pipeline with the best params
    return grid_search.best_estimator_
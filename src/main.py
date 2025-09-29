import os
from sklearn.model_selection import train_test_split

# Import our custom modules using relative imports
from .data_loader import load_images
from .model_trainer import train_and_tune_model
from .evaluation import evaluate_model

def main_pipeline():
    """
    Runs the full SVM image classification pipeline.
    """
    print("Starting the SVM Image Classification Pipeline... 🩺")
    
    # --- Setup ---
    DATA_DIR = 'data/raw'
    CLASS_NAMES = sorted(os.listdir(DATA_DIR))
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    # 1. Load Data
    images, labels = load_images(DATA_DIR)
    
    # 2. Split Data (80% train, 20% test)
    # Stratify by labels to ensure train/test sets have proportional class representation.
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"\nData split into {len(X_train)} training and {len(X_test)} testing images.")
    
    # 3. Train and Tune Model using GridSearchCV
    best_model = train_and_tune_model(X_train, y_train)
    
    # 4. Evaluate the Best Model
    evaluate_model(best_model, X_test, y_test, CLASS_NAMES)
    
    print("\nSVM Image Classification Pipeline finished successfully! ✨")

if __name__ == "__main__":
    main_pipeline()
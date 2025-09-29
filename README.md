# CST-600-Assignment4

This project develops a prototype for a medical diagnostic tool using Support Vector Machines (SVMs) to classify chest X-ray images. The model is trained to distinguish between "Normal" and "Pneumonia" cases using features extracted via the Histogram of Oriented Gradients (HOG) method.

## Business Scenario
As a data scientist in a healthcare technology company, I was tasked with prototyping a machine learning classifier for image-based diagnostics. This project serves as a proof-of-concept to evaluate the feasibility of using traditional computer vision features (HOG) with a powerful classifier (SVM) for this task.

---
## Dataset
* **Source**: [Chest X-Ray Images (Pneumonia) - Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* **Preparation**: To create a manageable and balanced dataset for this prototype, a **curated subset** was created.
  * **250 images** from the `NORMAL` class.
  * **250 images** from the `PNEUMONIA` class.
* The raw images are stored locally in `data/raw/<class>/` and should be excluded from version control via `.gitignore`.

---
## Environment Setup
Follow these steps to set up the local environment on a Windows machine.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd svm-image-classifier
    ```

2.  **Create and Activate the Virtual Environment**
    ```bash
    # Create the virtual environment folder named 'venv'
    python -m venv venv

    # Activate it
    venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
---
## How to Run
After setting up the environment and placing the dataset in the `data/raw/` folder:

1.  Navigate to the project's **root directory** in your terminal (the folder containing `src/`, `README.md`, etc.).
2.  Run the main script **as a module** with the following command:
    ```bash
    python -m src.main
    ```
    > **Note:** Running the script as a module (`-m`) is necessary for Python to correctly handle the relative imports between files within the `src` package.

---
## Summary of Decisions & Results
* **Feature Extraction**: **HOG (Histogram of Oriented Gradients)** was chosen because it effectively captures shape and texture information by analyzing local gradient directions, which is ideal for identifying structural changes in X-rays.
* **Preprocessing Pipeline**: A `scikit-learn Pipeline` was implemented with a custom `HOGTransformer` and a `StandardScaler`. This ensures that feature extraction and scaling are applied consistently and prevents data leakage from the test set.
* **Model Tuning**: `GridSearchCV` was used to systematically test and compare **linear, RBF, and polynomial SVM kernels** with various hyperparameters (`C`, `gamma`, `degree`).
* **Results**: The best performance was achieved with the **RBF kernel** (`C=10`, `gamma='scale'`), resulting in **98% accuracy** and a **0.998 ROC-AUC** score on the hold-out test set.
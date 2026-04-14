import numpy as np
from sklearn.metrics import classification_report
from src.config import SAME_DIST_PATH, DIFF_1_DIST_PATH, TEST_DIST_PATH
from src.utils.data_loader import load_feather_data
from src.models.cnn_model import CNNModel
from src.algorithms.tr_adaboost import MultiClassTrAdaBoostCNN

def main():
    print("Loading data...")
    # Target domain (Same distribution)
    target_X, target_y = load_feather_data(SAME_DIST_PATH)
    
    # Source domain (Diff distribution)
    source_X, source_y = load_feather_data(DIFF_1_DIST_PATH)
    
    # Test domain
    test_X, test_y = load_feather_data(TEST_DIST_PATH)
    
    if target_X is None or source_X is None or test_X is None:
        print("Error loading datasets. Please check paths in src/config.py")
        return

    print(f"Target shape: {target_X.shape}, Source shape: {source_X.shape}, Test shape: {test_X.shape}")

    # Initialize TrAdaBoost
    print("Initializing MultiClassTrAdaBoostCNN...")
    model = MultiClassTrAdaBoostCNN(CNNModel, n_estimators=10)
    
    # Train
    model.fit(target_X, target_y, source_X, source_y)
    
    # Predict
    print("Predicting on test set...")
    predictions = model.predict(test_X)
    
    # Evaluate
    print("\nClassification Report:\n")
    print(classification_report(test_y, predictions))

if __name__ == "__main__":
    main()

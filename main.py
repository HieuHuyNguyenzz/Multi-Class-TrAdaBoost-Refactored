import numpy as np
import os
import argparse
import src.config as config
from sklearn.metrics import classification_report
from src.config import SOURCE_PATH, TARGET_PATH, TARGET_TEST_RATIO, DEVICE, set_seed, SEED, NUM_ESTIMATORS, GATING_K
from src.utils.data_loader import load_source_data, load_target_data
from src.models.cnn_model import CNNModel
from src.algorithms.original_tr_adaboost import MultiClassTrAdaBoostCNN
from src.algorithms.gated_tr_adaboost import GatedMultiClassTrAdaBoostCNN

# Set seeds for reproducibility before any training
set_seed(SEED)

def main():
    parser = argparse.ArgumentParser(description="Multi-Class TrAdaBoost-CNN Training and Evaluation")
    parser.add_argument('--mode', type=str, default='test', 
                        choices=['train_full', 'train_gate', 'tradaboost_only', 'test_no_gating', 'test_with_gating', 'test'], 
                        help="Execution mode: 'train_full' to train everything, 'train_gate' to only train Gating Network, "
                             "'tradaboost_only' to train/test only original TrAdaBoost, "
                             "'test_no_gating' for full ensemble evaluation, 'test_with_gating' for sparse evaluation, "
                             "'test' for both.")
    parser.add_argument('--gate_data', type=str, default='both',
                        choices=['both', 'target_only'],
                        help="Data to use for training the gating network: 'both' (source + target) or 'target_only'.")
    parser.add_argument('--use_semi', action='store_true',
                        help="Whether to use semi-supervised pre-training with unlabeled target data.")
    parser.add_argument('--use_soft_labels', action='store_true',
                        help="Whether to use weighted soft labels instead of binary oracle labels for gating training.")
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Execution Mode: {args.mode}")

    # Print config compactly
    print("\n[Config]")
    gen = [f"{k}={v}" for k, v in vars(config).items() if k.isupper() and not k.startswith(('GATING_', 'LAMBDA_', 'SOURCE_', 'TARGET_', 'DATA_'))]
    gat = [f"{k}={v}" for k, v in vars(config).items() if k.isupper() and (k.startswith(('GATING_', 'LAMBDA_')))]
    pth = [f"{k}={v}" for k, v in vars(config).items() if k.isupper() and (k.startswith(('SOURCE_', 'TARGET_', 'DATA_')))]
    print(f" General: {', '.join(gen)}")
    print(f" Gating: {', '.join(gat)}")
    print(f" Paths: {', '.join(pth)}")
    
    print("\nLoading data...")
    # Source domain: Full dataset for training
    source_X, source_y = load_source_data(SOURCE_PATH)
    
    # Target domain: Split into labeled, unlabeled, and test
    target_labeled_X, target_labeled_y, target_unlabeled_X, target_unlabeled_y, test_X, test_y = load_target_data(TARGET_PATH, TARGET_TEST_RATIO)
    
    if source_X is None or target_labeled_X is None or test_X is None:
        print("Error loading datasets. Please check paths in src/config.py")
        return

    input_shape = target_labeled_X[0].shape
    print(f"\nSource domain (Domain 1): {source_X.shape} flows")
    print(f"Target domain (Domain 2): Labeled {target_labeled_X.shape}, Unlabeled {target_unlabeled_X.shape}, Test {test_X.shape}")
    
    # Paths for saving models
    ORIG_MODEL_PATH = "model_orig.pth"
    GATED_MODEL_PATH = "model_gated.pth"
    
    # Always train from scratch (delete existing models for training modes)
    if args.mode in ['train_full', 'tradaboost_only']:
        if os.path.exists(ORIG_MODEL_PATH):
            print("Removing existing model files...")
            os.remove(ORIG_MODEL_PATH)
        if os.path.exists(GATED_MODEL_PATH):
            os.remove(GATED_MODEL_PATH)
    elif args.mode == 'train_gate':
        if os.path.exists(GATED_MODEL_PATH):
            print("Removing existing gated model file...")
            os.remove(GATED_MODEL_PATH)
    
    # --- Setup Models ---
    model_orig = MultiClassTrAdaBoostCNN(CNNModel, n_estimators=NUM_ESTIMATORS)
    
    # Only create gated model if needed
    if args.mode not in ['tradaboost_only']:
        model_gated = GatedMultiClassTrAdaBoostCNN(CNNModel, n_estimators=NUM_ESTIMATORS)
    
    # 1. Handle Original Model
    if args.mode == 'train_full' or args.mode == 'tradaboost_only':
        print("\nTraining Original TrAdaBoost Model...")
        model_orig.fit(target_labeled_X, target_labeled_y, source_X, source_y)
        model_orig.save(ORIG_MODEL_PATH, input_shape)
    elif os.path.exists(ORIG_MODEL_PATH):
        print("Loading pre-trained Original Model...")
        model_orig.load(ORIG_MODEL_PATH)
    else:
        print("Error: No pre-trained Original Model found. Please run with --mode train_full or tradaboost_only first.")
        return
    
    # --- Evaluation: Original TrAdaBoost Only ---
    print("\n" + "="*60)
    print(" EVALUATION: Original Multi-class TrAdaBoost-CNN ")
    print("="*60)
    orig_predictions, orig_time = model_orig.predict(test_X, return_time=True)
    print(classification_report(test_y, orig_predictions))
    print(f"\n>>> Inference Time: {orig_time:.4f} ms/sample ({model_orig.n_estimators} learners)")
    
    # If only TrAdaBoost requested, stop here
    if args.mode == 'tradaboost_only':
        return
    
    # 2. Handle Gated Model (for modes beyond tradaboost_only)
    if args.mode == 'train_full':
        # Reuse learners from model_orig instead of training from scratch
        print("\nReusing learners from Original model for Gated model...")
        model_gated.learners = model_orig.learners
        model_gated.alphas = model_orig.alphas
        model_gated.n_estimators = model_orig.n_estimators
        
        print("\nTraining Gating Network for Sparse Inference...")
        # Use source data only if --gate_data is 'both'
        s_X, s_y = (source_X, source_y) if args.gate_data == 'both' else (None, None)
        
        # Only pass unlabeled data if --use_semi flag is set
        u_X = target_unlabeled_X if args.use_semi else None
        model_gated.train_gate(target_labeled_X, target_labeled_y, s_X, s_y, X_unlabeled=u_X, use_soft_labels=args.use_soft_labels)
        model_gated.save(GATED_MODEL_PATH, input_shape)
        
    elif args.mode == 'train_gate':
        if os.path.exists(GATED_MODEL_PATH):
            print("Loading existing Gated Model to re-train gate...")
            model_gated.load(GATED_MODEL_PATH)
        elif os.path.exists(ORIG_MODEL_PATH):
            print("Gated model not found. Initializing Gated model from Original ensemble...")
            model_gated.learners = model_orig.learners
            model_gated.alphas = model_orig.alphas
            model_gated.n_estimators = model_orig.n_estimators
        else:
            print("Error: No pre-trained models found. Please run with --mode train_full first.")
            return
            
        print("\nRe-training Gating Network...")
        # Use source data only if --gate_data is 'both'
        s_X, s_y = (source_X, source_y) if args.gate_data == 'both' else (None, None)
        
        # Only pass unlabeled data if --use_semi flag is set
        u_X = target_unlabeled_X if args.use_semi else None
        model_gated.train_gate(target_labeled_X, target_labeled_y, s_X, s_y, X_unlabeled=u_X, use_soft_labels=args.use_soft_labels)
        model_gated.save(GATED_MODEL_PATH, input_shape)
    else:
        if os.path.exists(GATED_MODEL_PATH):
            print("Loading pre-trained Gated Model...")
            model_gated.load(GATED_MODEL_PATH)
        else:
            print("Error: No pre-trained Gated Model found. Please run with --mode train_full or train_gate first.")
            return

    # --- Evaluation Phase ---
    
    # Scenario A: WITHOUT GATING (Full Ensemble)
    if args.mode in ['test_no_gating', 'test', 'train_full', 'train_gate']:
        print("\n" + "="*60)
        print(" EVALUATION WITHOUT GATING (Full Ensemble) ")
        print("="*60)
        
        # Gated Model (Full mode)
        print("\n[1] Gated Model (Running in Full Mode):")
        gated_full_predictions, gated_full_time = model_gated.predict(test_X, return_time=True)
        print(classification_report(test_y, gated_full_predictions))
        print(f"\n>>> Inference Time: {gated_full_time:.4f} ms/sample ({model_gated.n_estimators} learners)")

    # Scenario B: WITH GATING (Sparse Inference)
    if args.mode in ['test_with_gating', 'test', 'train_full', 'train_gate']:
        print("\n" + "="*60)
        print(" EVALUATION WITH GATING (Sparse Inference) ")
        print("="*60)
        
        k_values = [GATING_K]
        k_values = [min(k, model_gated.n_estimators) for k in k_values]
        
        for k in k_values:
            gated_sparse_predictions, sparse_time = model_gated.predict_sparse(test_X, k=k, return_time=True)
            print(f"\n--- Gated Sparse AdaBoost (k={k}) ---")
            print(classification_report(test_y, gated_sparse_predictions))
            print(f">>> Inference Time: {sparse_time:.4f} ms/sample ({k} learners)")

if __name__ == "__main__":
    main()
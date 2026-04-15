import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from src.config import (
    DEVICE, BATCH_SIZE, NUM_CLASSES, 
    GATING_K, GATING_TAU, GATING_LR, GATING_EPOCHS,
    GATING_VAL_RATIO, GATING_PATIENCE, GATING_MIN_DELTA,
    NUM_WORKERS
)
from src.models.gating_net import GatingNetwork
from src.utils.dataset import ETCDataset
from src.algorithms.original_tr_adaboost import MultiClassTrAdaBoostCNN, PIN_MEMORY

class GatedMultiClassTrAdaBoostCNN(MultiClassTrAdaBoostCNN):
    """
    Improved Multi-class TrAdaBoost with Gating Network for Sparse Inference.
    """
    def __init__(self, model_class, n_estimators=10):
        super().__init__(model_class, n_estimators)
        self.gate = None
    
    def train_gate(self, X_train, y_train, X_source=None, y_source=None):
        """
        Train the gating network using multi-label classification.
        The gate learns to predict which learners should be in top-k.
        
        Args:
            X_train: Target domain training data
            y_train: Target domain labels
            X_source: Source domain data (optional)
            y_source: Source domain labels (optional)
        """
        print("Generating multi-label ground truth for Gating Network...")
        
        # Use both target and source data if provided
        if X_source is not None and y_source is not None:
            X_combined = np.concatenate([X_train, X_source], axis=0)
            y_combined = np.concatenate([y_train, y_source], axis=0)
            print(f"  Using target ({len(X_train)}) + source ({len(X_source)}) = {len(X_combined)} samples")
        else:
            X_combined = X_train
            y_combined = y_train
            print(f"  Using {len(X_combined)} samples")
        
        preds = self._get_all_predictions(X_combined) 
        alphas = np.array(self.alphas) 
        
        contributions = (preds == y_combined[:, np.newaxis]) * alphas
        
        # Multi-label: create binary labels (learner in top-k = 1, else = 0)
        k = 3
        top_k_idx = np.argsort(contributions, axis=1)[:, -k:]
        binary_labels = np.zeros((len(X_combined), self.n_estimators), dtype=np.float32)
        for i, row in enumerate(top_k_idx):
            binary_labels[i, row] = 1.0
        
        binary_labels_tensor = torch.from_numpy(binary_labels).float().to(DEVICE)
        
        X_dataset = ETCDataset(X_combined, y_combined)
        dataloader = DataLoader(X_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        self.gate = GatingNetwork(input_shape=X_train[0].shape, num_learners=self.n_estimators).to(DEVICE)
        optimizer = optim.Adam(self.gate.parameters(), lr=GATING_LR)
        criterion = nn.BCEWithLogitsLoss()
        
        print("Training Gating Network (multi-label)...")
        self.gate.train()
        for epoch in range(GATING_EPOCHS):
            total_loss = 0
            correct = 0
            total = 0
            
            for i, (data, _) in enumerate(dataloader):
                data = data.to(DEVICE)
                batch_idx = i * BATCH_SIZE
                if batch_idx >= len(binary_labels_tensor):
                    break
                batch_labels = binary_labels_tensor[batch_idx : batch_idx + BATCH_SIZE]
                if batch_labels.size(0) == 0:
                    continue
                
                optimizer.zero_grad()
                p_logits = self.gate(data)
                
                loss = criterion(p_logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate top-k accuracy (how often correct learners are in predicted top-k)
                p_probs = torch.sigmoid(p_logits)
                _, topk_pred = torch.topk(p_probs, k, dim=1)
                batch_range = torch.arange(batch_labels.size(0)).to(DEVICE)
                # Check if target learners are in predicted top-k
                for b in range(batch_labels.size(0)):
                    target_set = set(torch.where(batch_labels[b] > 0.5)[0].tolist())
                    pred_set = set(topk_pred[b].tolist())
                    if target_set & pred_set:  # intersection
                        correct += 1
                    total += 1
            
            accuracy = 100 * correct / total if total > 0 else 0
            print(f"Gate Epoch {epoch+1}/{GATING_EPOCHS}, Loss: {total_loss/max(1, len(dataloader)):.4f}, Top-{k} Acc: {accuracy:.2f}%")
    
    def predict_sparse(self, X_test, k=None, return_time=False):
        """
        Gated Sparse Inference prediction.
        
        Args:
            X_test: Test data.
            k (int, optional): Number of top learners to select. Defaults to GATING_K from config.
            return_time: If True, returns (predictions, avg_time_per_sample_ms)
        """
        if self.gate is None:
            raise ValueError("Gating network not trained. Call train_gate first.")
            
        # Use provided k or fallback to config
        selected_k = k if k is not None else GATING_K
            
        X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
        if X_test_tensor.dim() == 3:
            X_test_tensor = X_test_tensor.unsqueeze(1)
        
        # Warm up
        if DEVICE.type == 'mps':
            with torch.no_grad():
                warmup_input = X_test_tensor[:1]
                if warmup_input.dim() == 3:
                    warmup_input = warmup_input.unsqueeze(1)
                _ = self.gate(warmup_input)
                if len(self.learners) > 0:
                    _ = self.learners[0](warmup_input)
                torch.mps.synchronize()
        
        # Start timing
        start_time = time.time()
        
        self.gate.eval()
        with torch.no_grad():
            g_scores = []
            for i in range(0, X_test_tensor.size(0), BATCH_SIZE):
                batch = X_test_tensor[i : i + BATCH_SIZE]
                out = self.gate(batch)
                g_scores.append(out.cpu().numpy())
            g_scores = np.concatenate(g_scores) 
            
            top_k_idx = np.argsort(g_scores, axis=1)[:, -selected_k:]
            
            vote_matrix = np.zeros((X_test.shape[0], NUM_CLASSES))
            
            for t in range(self.n_estimators):
                mask = np.any(top_k_idx == t, axis=1)
                if not np.any(mask):
                    continue
                
                X_masked = X_test[mask]
                X_masked_tensor = torch.from_numpy(X_masked).float().to(DEVICE)
                if X_masked_tensor.dim() == 3:
                    X_masked_tensor = X_masked_tensor.unsqueeze(1)
                
                self.learners[t].eval()
                preds_masked = []
                with torch.no_grad():
                    for i in range(0, X_masked_tensor.size(0), BATCH_SIZE):
                        batch = X_masked_tensor[i : i + BATCH_SIZE]
                        out = self.learners[t](batch)
                        preds_masked.append(torch.argmax(out, dim=1).cpu().numpy())
                
                preds_masked = np.concatenate(preds_masked)
                indices = np.where(mask)[0]
                alpha_t = self.alphas[t]
                for idx, pred in zip(indices, preds_masked):
                    vote_matrix[idx, pred] += alpha_t
        
        # Synchronize
        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        
        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_per_sample_ms = total_time_ms / X_test.shape[0]
        
        predictions = np.argmax(vote_matrix, axis=1)
        
        if return_time:
            return predictions, avg_time_per_sample_ms
        return predictions

    def save(self, path, input_shape):
        """Saves the gated ensemble model to a file."""
        # Save base ensemble info
        checkpoint = {
            'n_estimators': self.n_estimators,
            'alphas': self.alphas,
            'learners_state_dicts': [learner.state_dict() for learner in self.learners],
            'input_shape': input_shape,
            'gate_state_dict': self.gate.state_dict() if self.gate is not None else None
        }
        torch.save(checkpoint, path)
        print(f"Gated model saved to {path}")

    def load(self, path):
        """Loads the gated ensemble model from a file."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.n_estimators = checkpoint['n_estimators']
        self.alphas = checkpoint['alphas']
        input_shape = checkpoint['input_shape']
        
        # Load base learners
        self.learners = []
        state_dicts = checkpoint['learners_state_dicts']
        for sd in state_dicts:
            learner = self.model_class(input_shape=input_shape, num_classes=NUM_CLASSES).to(DEVICE)
            learner.load_state_dict(sd)
            self.learners.append(learner)
        
        # Load Gating Network
        gate_sd = checkpoint.get('gate_state_dict')
        if gate_sd is not None:
            self.gate = GatingNetwork(input_shape=input_shape, num_learners=self.n_estimators).to(DEVICE)
            self.gate.load_state_dict(gate_sd)
            
        print(f"Gated model loaded from {path}")

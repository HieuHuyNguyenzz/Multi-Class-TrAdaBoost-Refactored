import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from src.config import (
    DEVICE, BATCH_SIZE, NUM_CLASSES, 
    GATING_K, GATING_TAU, GATING_LR, GATING_EPOCHS,
    LAMBDA_KL, LAMBDA_SPARSE, NUM_WORKERS
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
    
    def train_gate(self, X_train, y_train):
        """
        Train the gating network based on learner contributions.
        """
        print("Generating contribution labels for Gating Network...")
        preds = self._get_all_predictions(X_train) 
        alphas = np.array(self.alphas) 
        
        contributions = (preds == y_train[:, np.newaxis]) * alphas
        
        exp_c = np.exp(contributions / GATING_TAU)
        q = exp_c / np.sum(exp_c, axis=1, keepdims=True)
        q_tensor = torch.from_numpy(q).float().to(DEVICE)
        
        X_dataset = ETCDataset(X_train, y_train)
        dataloader = DataLoader(X_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        self.gate = GatingNetwork(input_shape=X_train[0].shape, num_learners=self.n_estimators).to(DEVICE)
        optimizer = optim.Adam(self.gate.parameters(), lr=GATING_LR)
        
        print("Training Gating Network...")
        self.gate.train()
        for epoch in range(GATING_EPOCHS):
            total_loss = 0
            for i, (data, _) in enumerate(dataloader):
                data = data.to(DEVICE)
                batch_q = q_tensor[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                if batch_q.size(0) == 0: continue
                
                optimizer.zero_grad()
                p_logits = self.gate(data)
                p = torch.softmax(p_logits, dim=1)
                
                kl_loss = torch.sum(batch_q * (torch.log(batch_q + 1e-10) - torch.log(p + 1e-10)), dim=1).mean()
                sparse_loss = torch.sum(p * torch.log(p + 1e-10), dim=1).mean()
                
                loss = LAMBDA_KL * kl_loss + LAMBDA_SPARSE * sparse_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            print(f"Gate Epoch {epoch+1}/{GATING_EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    
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
                _ = self.gate(X_test_tensor[:1])
                if len(self.learners) > 0:
                    _ = self.learners[0](X_test_tensor[:1])
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
        checkpoint = torch.load(path, map_location=DEVICE)
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

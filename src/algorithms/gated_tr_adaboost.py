import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.cluster import KMeans
from src.config import (
    DEVICE, BATCH_SIZE, NUM_CLASSES, NUM_ESTIMATORS,
    GATING_K, GATING_LR, GATING_EPOCHS,
    GATING_GRAD_CLIP, GATING_LAMBDA_LB, GATING_WEIGHT_DECAY,
    NUM_WORKERS, GRPO_EPOCHS
)
from src.models.gating_net import GatingNetwork, GatingCNN
from src.utils.dataset import ETCDataset
from src.algorithms.original_tr_adaboost import MultiClassTrAdaBoostCNN, PIN_MEMORY
from src.algorithms.grpo_trainer import GRPOTrainer


def load_balance_loss(gate_logits, num_experts):
    """
    Load balancing loss to prevent expert collapse.
    Encourages gating to distribute tokens evenly across experts.
    """
    probs = F.softmax(gate_logits, dim=-1)
    mean_prob = probs.mean(dim=0)
    target = torch.ones(num_experts) / num_experts
    target = target.to(gate_logits.device)
    return F.mse_loss(mean_prob, target)


def compute_gating_metrics(gate_logits, oracle_labels, k):
    """
    Compute monitoring metrics for gating network.
    
    Returns:
        dict with topk_hit_rate, expert_utilization, entropy, coverage
    """
    B, n = gate_logits.shape
    probs = F.softmax(gate_logits, dim=-1)
    topk_indices = probs.topk(k, dim=-1).indices
    
    oracle_bool = oracle_labels.bool()
    
    topk_oh = torch.zeros_like(oracle_labels)
    topk_oh.scatter_(1, topk_indices, 1)
    
    hit = (topk_oh * oracle_bool).any(dim=-1).float()
    topk_hit_rate = hit.mean().item()
    
    utilization = topk_oh.mean(dim=0)
    utilization_std = utilization.std().item()
    
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
    
    coverage = (topk_oh * oracle_bool).sum(dim=-1) / (oracle_bool.sum(dim=-1) + 1e-8)
    coverage_mean = coverage.mean().item()
    
    return {
        'topk_hit_rate': topk_hit_rate,
        'utilization_std': utilization_std,
        'entropy': entropy,
        'coverage': coverage_mean
    }


class GatedMultiClassTrAdaBoostCNN(MultiClassTrAdaBoostCNN):
    """
    Improved Multi-class TrAdaBoost with Gating Network for Sparse Inference.
    """
    def __init__(self, model_class, n_estimators=NUM_ESTIMATORS):
        super().__init__(model_class, n_estimators)
        self.gate = None
    
    def pretrain_gate(self, X_unlabeled):
        """
        Pre-trains the gating network using K-Means clustering on unlabeled target data.
        The cluster IDs act as pseudo-labels for experts.
        """
        if self.gate is None:
            self.gate = GatingCNN(input_shape=X_unlabeled[0].shape, num_learners=self.n_estimators).to(DEVICE)
        
        print(f"  Performing K-Means clustering on {len(X_unlabeled)} samples...")
        # Flatten data for KMeans: (N, 20, 256) -> (N, 20*256)
        X_flat = X_unlabeled.reshape(len(X_unlabeled), -1)
        
        kmeans = KMeans(n_clusters=self.n_estimators, random_state=42, n_init='auto')
        cluster_ids = kmeans.fit_predict(X_flat)
        
        # Convert cluster IDs to one-hot labels (N, num_estimators)
        pseudo_labels = np.zeros((len(X_unlabeled), self.n_estimators), dtype=np.float32)
        pseudo_labels[np.arange(len(X_unlabeled)), cluster_ids] = 1.0
        pseudo_labels_tensor = torch.from_numpy(pseudo_labels).float().to(DEVICE)
        
        # Data Loader for unlabeled data
        train_dataset = ETCDataset(X_unlabeled, np.zeros(len(X_unlabeled))) # y is not used
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        optimizer = optim.Adam(self.gate.parameters(), lr=GATING_LR)
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"  Training Gating Network on pseudo-labels (K-Means) for {GATING_EPOCHS // 2} epochs...")
        self.gate.train()
        for epoch in range(GATING_EPOCHS // 2):
            total_loss = 0
            for i, (data, _) in enumerate(train_loader):
                data = data.to(DEVICE)
                batch_idx = i * BATCH_SIZE
                if batch_idx >= len(pseudo_labels_tensor):
                    break
                batch_labels = pseudo_labels_tensor[batch_idx : batch_idx + BATCH_SIZE]
                
                optimizer.zero_grad()
                logits = self.gate(data)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Pre-train Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    def train_gate(self, X_train, y_train, X_source=None, y_source=None, X_unlabeled=None, use_soft_labels=True):
        """
        Train the gating network following the MoE design document.
        Optional: pre-trains with X_unlabeled using clustering.
        
        Args:
            X_train: Target domain training data (labeled)
            y_train: Target domain labels
            X_source: Source domain data (optional)
            y_source: Source domain labels (optional)
            X_unlabeled: Target domain unlabeled data for pre-training (optional)
            use_soft_labels: Whether to use weighted soft labels or binary oracle labels.
        """
        if X_unlabeled is not None:
            print("\n[Semi-supervised] Starting pre-training on unlabeled data...")
            self.pretrain_gate(X_unlabeled)

        print("Generating oracle labels for Gating Network...")
        
        # Use both target and source data if provided
        if X_source is not None and y_source is not None:
            X_combined = np.concatenate([X_train, X_source], axis=0)
            y_combined = np.concatenate([y_train, y_source], axis=0)
            print(f"  Using target ({len(X_train)}) + source ({len(X_source)}) = {len(X_combined)} samples")
        else:
            X_combined = X_train
            y_combined = y_train
            print(f"  Using {len(X_combined)} samples")
        
        # Precompute oracle labels (which experts predict correctly)
        preds = self._get_all_predictions(X_combined)
        alphas = np.array(self.alphas)
        
        contributions = (preds == y_combined[:, np.newaxis]) * alphas
        
        if use_soft_labels:
            # Normalize contributions with a power transformation to increase contrast (Soft Labels)
            # Squaring the contributions amplifies the gap between the best and mediocre experts
            contributions_sq = contributions ** 2
            row_sums = contributions_sq.sum(axis=1, keepdims=True)
            binary_labels = contributions_sq / (row_sums + 1e-8)
            binary_labels = binary_labels.astype(np.float32)
        else:
            # Multi-label: create binary labels (learner in top-k = 1, else = 0)
            # Use a fixed k=3 for binary labels as per original design
            k = 3
            top_k_idx = np.argsort(contributions, axis=1)[:, -k:]
            binary_labels = np.zeros((len(X_combined), self.n_estimators), dtype=np.float32)
            for i, row in enumerate(top_k_idx):
                binary_labels[i, row] = 1.0
        
        binary_labels_tensor = torch.from_numpy(binary_labels).float().to(DEVICE)
        
        train_dataset = ETCDataset(X_combined, y_combined)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        self.gate = GatingCNN(input_shape=X_train[0].shape, num_learners=self.n_estimators).to(DEVICE)
        
        # AdamW optimizer with weight decay
        optimizer = optim.AdamW(self.gate.parameters(), lr=GATING_LR, weight_decay=GATING_WEIGHT_DECAY)
        
        # OneCycleLR scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=GATING_LR,
            steps_per_epoch=len(train_loader),
            epochs=GATING_EPOCHS,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"\nTraining Gating Network (Supervised Oracle + Load Balance)...")
        print(f"  LR: {GATING_LR}, Weight Decay: {GATING_WEIGHT_DECAY}")
        print(f"  Lambda LB: {GATING_LAMBDA_LB}, Grad Clip: {GATING_GRAD_CLIP}")
        
        for epoch in range(GATING_EPOCHS):
            self.gate.train()
            train_loss = 0
            train_loss_task = 0
            train_loss_lb = 0
            
            for i, (data, _) in enumerate(train_loader):
                data = data.to(DEVICE)
                batch_idx = i * BATCH_SIZE
                if batch_idx >= len(binary_labels_tensor):
                    break
                batch_labels = binary_labels_tensor[batch_idx : batch_idx + BATCH_SIZE]
                if batch_labels.size(0) == 0:
                    continue
                
                optimizer.zero_grad()
                gate_logits = self.gate(data)
                
                # Task loss (BCE)
                loss_task = criterion(gate_logits, batch_labels)
                
                # Load balancing loss
                loss_lb = load_balance_loss(gate_logits, self.n_estimators)
                
                # Total loss
                loss = loss_task + GATING_LAMBDA_LB * loss_lb
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.gate.parameters(), GATING_GRAD_CLIP)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_loss_task += loss_task.item()
                train_loss_lb += loss_lb.item()
                
                # Monitoring metrics every batch
                if i % 10 == 0:
                    with torch.no_grad():
                        metrics = compute_gating_metrics(gate_logits, batch_labels, GATING_K)
                        # Just log briefly or store. For now, let's print periodically.
            
            train_loss = train_loss / max(1, len(train_loader))
            train_loss_task = train_loss_task / max(1, len(train_loader))
            train_loss_lb = train_loss_lb / max(1, len(train_loader))
            
            # Print metrics at end of epoch
            with torch.no_grad():
                # Use a small sample to compute epoch metrics
                sample_data = next(iter(train_loader))[0].to(DEVICE)
                sample_labels = binary_labels_tensor[:sample_data.size(0)]
                metrics = compute_gating_metrics(self.gate(sample_data), sample_labels, GATING_K)
            
            print(f"Gate Epoch {epoch+1}/{GATING_EPOCHS}, Loss: {train_loss:.4f} (Task: {train_loss_task:.4f}, LB: {train_loss_lb:.4f}) | HitRate: {metrics['topk_hit_rate']:.4f}, UtilStd: {metrics['utilization_std']:.4f}")
    
    def train_gate_grpo(self, X_train, y_train, epochs=GRPO_EPOCHS, lr=GATING_LR):
        """
        Train the gating network using Group Relative Policy Optimization (GRPO).
        This allows the model to dynamically choose the number of experts.
        """
        print(f"\nTraining Gating Network with GRPO for {epochs} epochs...")
        
        # Initialize Gating Network if not already done
        if self.gate is None:
            self.gate = GatingCNN(input_shape=X_train[0].shape, num_learners=self.n_estimators).to(DEVICE)
            
        # Prepare data
        train_dataset = ETCDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        
        trainer = GRPOTrainer(
            gating_net=self.gate,
            learners=self.learners,
            alphas=self.alphas,
            lr=lr
        )
        
        for epoch in range(epochs):
            total_loss = 0
            avg_reward = 0
            
            for data, labels in train_loader:
                data = data.to(DEVICE)
                labels = labels.to(DEVICE)
                
                loss, reward = trainer.train_step(data, labels)
                total_loss += loss
                avg_reward += reward
                
            print(f"GRPO Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Avg Reward: {avg_reward/len(train_loader):.4f}")

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
                self.gate = GatingCNN(input_shape=input_shape, num_learners=self.n_estimators).to(DEVICE)
                self.gate.load_state_dict(gate_sd)
            
        print(f"Gated model loaded from {path}")

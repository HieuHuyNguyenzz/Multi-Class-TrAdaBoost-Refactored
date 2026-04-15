import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.config import DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES, NUM_WORKERS
from src.utils.dataset import ETCDataset

# Optimize pin_memory for CUDA only (MPS doesn't benefit from it)
PIN_MEMORY = DEVICE.type == 'cuda'

class MultiClassTrAdaBoostCNN:
    """
    Original Multi-class TrAdaBoost with CNN as weak learner.
    """
    def __init__(self, model_class, n_estimators=10):
        self.model_class = model_class
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = [] # alpha_t values
        
    def fit(self, target_X, target_y, source_X, source_y):
        n_target = len(target_y)  # n in paper (target domain size)
        n_source = len(source_y)  # m in paper (source domain size)
        total_samples = n_target + n_source
        
        # Combine target and source data: target comes first, then source
        X_combined = np.concatenate([target_X, source_X], axis=0)
        y_combined = np.concatenate([target_y, source_y], axis=0)
        combined_dataset = ETCDataset(X_combined, y_combined)
        
        # Step 1: Initialize weights (uniform distribution)
        beta = np.ones(total_samples) / total_samples
        
        # Calculate alpha (source domain weight adjustment parameter)
        # alpha = ln(1 / (1 + sqrt(2*ln(n) / N)))
        alpha_s = np.log(1 / (1 + np.sqrt(2 * np.log(n_target) / self.n_estimators)))
        
        for t in range(self.n_estimators):
            # Step 2.1: Calculate probability distribution p_t
            p_t = beta / beta.sum()
            
            # Create sampler with current distribution
            sampler = WeightedRandomSampler(p_t, num_samples=total_samples, replacement=True)
            dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
            
            # Step 2.2: Train CNNModel (weak learner) on combined data
            learner = self.model_class(input_shape=X_combined[0].shape, num_classes=NUM_CLASSES).to(DEVICE)
            optimizer = optim.Adam(learner.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            learner.train()
            for epoch in range(NUM_EPOCHS):
                for data, target in dataloader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    output = learner(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Step 2.3: Calculate error rate epsilon_t on TARGET domain
            learner.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                eval_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
                for data, target in eval_loader:
                    data = data.to(DEVICE)
                    output = learner(data)
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Extract predictions and labels for TARGET domain (first n_target samples)
            preds_target = all_preds[:n_target]
            labels_target = all_labels[:n_target]
            beta_target = beta[:n_target]
            
            # indicator: 1 if prediction != label, 0 otherwise
            indicator_target = (preds_target != labels_target).astype(float)
            
            # epsilon_t = sum(beta_i * I(h_t(x_i) != c(x_i))) / sum(beta_i) for target domain
            eps_t = np.sum(beta_target * indicator_target) / np.sum(beta_target)
            eps_t = np.clip(eps_t, 1e-10, (NUM_CLASSES - 1) / NUM_CLASSES - 1e-10)
            
            # Step 2.4: Calculate parameters alpha_t and C^t
            # alpha_t = ln((1 - eps_t) / eps_t) + ln(K - 1)
            alpha_t = np.log((1 - eps_t) / eps_t) + np.log(NUM_CLASSES - 1)
            
            # C^t = K * (1 - eps_t)
            C_t = NUM_CLASSES * (1 - eps_t)
            
            # Step 2.5: Update weights
            # indicator for ALL samples
            indicator_all = (all_preds != all_labels).astype(float)
            
            # Target domain (1 to n): beta^{t+1}_i = beta^t_i * exp(alpha_t * I(h_t(x_i) != c(x_i)))
            beta[:n_target] *= np.exp(alpha_t * indicator_all[:n_target])
            
            # Source domain (n+1 to n+m): beta^{t+1}_i = C^t * beta^t_i * exp(alpha_s * I(h_t(x_i) != c(x_i)))
            beta[n_target:] *= C_t * np.exp(alpha_s * indicator_all[n_target:])
            
            # Normalize weights
            beta = beta / beta.sum()
            
            print(f"Iteration {t+1}/{self.n_estimators}, Target Error: {eps_t:.4f}, Alpha_t: {alpha_t:.4f}")
            
            # Save learner and alpha_t
            self.learners.append(learner)
            self.alphas.append(alpha_t)
    
    def _get_all_predictions(self, X):
        """Helper to get predictions from all learners for a given X."""
        X_tensor = torch.from_numpy(X).float().to(DEVICE)
        if X_tensor.dim() == 3:
            X_tensor = X_tensor.unsqueeze(1)
            
        all_preds = []
        with torch.no_grad():
            for learner in self.learners:
                learner.eval()
                preds = []
                for i in range(0, X_tensor.size(0), BATCH_SIZE):
                    batch = X_tensor[i : i + BATCH_SIZE]
                    out = learner(batch)
                    preds.append(torch.argmax(out, dim=1).cpu().numpy())
                all_preds.append(np.concatenate(preds))
        
        return np.array(all_preds).T # (samples, T)
    
    def predict(self, X_test, return_time=False):
        """
        Final hypothesis: h_f(x) = argmax_k sum(alpha_t * I(h_t(x) = k))
        
        Args:
            X_test: Test data array
            return_time: If True, returns (predictions, avg_time_per_sample_ms)
        """
        X_test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
        if X_test_tensor.dim() == 3:
            X_test_tensor = X_test_tensor.unsqueeze(1)
        
        # Warm up (run a few passes to stabilize timing)
        n_warmup = min(10, X_test_tensor.size(0))
        with torch.no_grad():
            for i in range(n_warmup):
                warmup_input = X_test_tensor[i:i+1]
                if warmup_input.dim() == 3:
                    warmup_input = warmup_input.unsqueeze(1)
                _ = self.learners[0](warmup_input)
        
        # Synchronize before timing
        if DEVICE.type == 'mps':
            torch.mps.synchronize()
        
        # Start timing
        start_time = time.time()
        
        vote_matrix = np.zeros((X_test.shape[0], NUM_CLASSES))
        
        with torch.no_grad():
            for alpha_t, learner in zip(self.alphas, self.learners):
                learner.eval()
                outputs = []
                for i in range(0, X_test_tensor.size(0), BATCH_SIZE):
                    batch = X_test_tensor[i : i + BATCH_SIZE]
                    out = learner(batch)
                    outputs.append(torch.argmax(out, dim=1).cpu().numpy())
                preds = np.concatenate(outputs)
                for cls in range(NUM_CLASSES):
                    vote_matrix[preds == cls, cls] += alpha_t
        
        # Synchronize after timing
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
        """Saves the ensemble model to a file."""
        checkpoint = {
            'n_estimators': self.n_estimators,
            'alphas': self.alphas,
            'learners_state_dicts': [learner.state_dict() for learner in self.learners],
            'input_shape': input_shape
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Loads the ensemble model from a file."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.n_estimators = checkpoint['n_estimators']
        self.alphas = checkpoint['alphas']
        input_shape = checkpoint['input_shape']
        
        self.learners = []
        state_dicts = checkpoint['learners_state_dicts']
        for sd in state_dicts:
            learner = self.model_class(input_shape=input_shape, num_classes=NUM_CLASSES).to(DEVICE)
            learner.load_state_dict(sd)
            self.learners.append(learner)
        print(f"Model loaded from {path}")
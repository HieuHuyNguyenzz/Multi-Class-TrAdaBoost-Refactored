import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
from src.config import DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_CLASSES

class ETCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X.float()
        self.y = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y.long()
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Returns (1, H, W) to match CNN input
        return self.X[idx].unsqueeze(0), self.y[idx]

class MultiClassTrAdaBoostCNN:
    def __init__(self, model_class, n_estimators=10):
        self.model_class = model_class
        self.n_estimators = n_estimators
        self.learners = []
        self.alphas = []
        
    def fit(self, target_X, target_y, source_X, source_y):
        """
        Train Multi-class TrAdaBoost.
        
        Args:
            target_X, target_y: Target domain data
            source_X, source_y: Source domain data
        """
        n_target = len(target_y)
        n_source = len(source_y)
        total_samples = n_target + n_source
        
        # Combine datasets
        X_combined = np.concatenate([target_X, source_X], axis=0)
        y_combined = np.concatenate([target_y, source_y], axis=0)
        combined_dataset = ETCDataset(X_combined, y_combined)
        
        # Hyperparameters
        alpha_s = np.log(1 / (1 + np.sqrt(2 * np.log(n_target) / self.n_estimators)))
        beta = np.ones(total_samples) / total_samples
        
        for t in range(self.n_estimators):
            # 1. Weighted Sampling
            p_t = beta / beta.sum()
            sampler = WeightedRandomSampler(p_t, num_samples=total_samples, replacement=True)
            dataloader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, sampler=sampler)
            
            # 2. Base Learner Training
            learner = self.model_class(input_shape=X_combined[0].shape, num_classes=NUM_CLASSES).to(DEVICE)
            optimizer = optim.Adam(learner.parameters(), lr=1e-4)
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
            
            # 3. Evaluation & Weight Update
            learner.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                eval_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=False)
                for data, target in eval_loader:
                    data = data.to(DEVICE)
                    output = learner(data)
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(target.numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Source domain error
            preds_source = all_preds[n_target:]
            labels_source = all_labels[n_target:]
            beta_source = beta[n_target:]
            
            # Vectorized error calculation
            indicator_source = (preds_source != labels_source).astype(float)
            eps_t = np.sum(beta_source * indicator_source) / np.sum(beta_source)
            eps_t = np.clip(eps_t, 1e-10, (NUM_CLASSES - 1) / NUM_CLASSES - 1e-10)
            
            # 4. Update Alphas and Beta
            alpha_t = np.log((1 - eps_t) / eps_t) + np.log(NUM_CLASSES - 1)
            C_t = NUM_CLASSES * (1 - eps_t)
            
            # Vectorized Beta Update
            indicator_all = (all_preds != all_labels).astype(float)
            
            # Target weights update
            beta[:n_target] *= np.exp(alpha_t * indicator_all[:n_target])
            
            # Source weights update
            beta[n_target:] *= C_t * np.exp(alpha_s * indicator_all[n_target:])
            
            print(f"Iteration {t+1}/{self.n_estimators}, Source Error: {eps_t:.4f}")
            
            self.learners.append(learner)
            self.alphas.append(alpha_t)
            
    def predict(self, X_test):
        """
        Weighted voting prediction.
        """
        X_test = torch.from_numpy(X_test).float().to(DEVICE)
        # Add channel dim
        if X_test.dim() == 3:
            X_test = X_test.unsqueeze(1)
            
        vote_matrix = np.zeros((X_test.size(0), NUM_CLASSES))
        
        with torch.no_grad():
            for alpha_t, learner in zip(self.alphas, self.learners):
                learner.eval()
                # Process in batches to avoid OOM
                outputs = []
                for i in range(0, X_test.size(0), BATCH_SIZE):
                    batch = X_test[i : i + BATCH_SIZE]
                    out = learner(batch)
                    outputs.append(torch.argmax(out, dim=1).cpu().numpy())
                
                preds = np.concatenate(outputs)
                
                # Vectorized vote accumulation
                for cls in range(NUM_CLASSES):
                    vote_matrix[preds == cls, cls] += alpha_t
                    
        return np.argmax(vote_matrix, axis=1)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cpu")

class LogitDGP:
    def __init__(self, d=5, seed=42):
        self.d = d
        self.rng = np.random.default_rng(seed)
        
    def alpha_func(self, x):
        return 0.5 + 0.5*x[:,0]
    
    def beta_func(self, x):
        return -1.0 + 0.5*x[:,1] + 0.2*x[:,0]*x[:,1]
        
    def generate(self, n):
        X = self.rng.normal(0, 1, (n, self.d))
        R = 0.5 * X[:,0] + 0.5 * self.rng.normal(0, 1, n)
        alpha = self.alpha_func(X)
        beta = self.beta_func(X)
        logits = alpha + beta * R
        probs = 1 / (1 + np.exp(-logits))
        Y = self.rng.binomial(1, probs)
        return X, R, Y, alpha, beta, probs

class StructuralNet(nn.Module):
    def __init__(self, d, out_dim=2):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.out = nn.Linear(16, out_dim)
        
    def forward(self, x):
        feat = self.hidden(x)
        return self.out(feat)

def train_logit_model(X, R, Y, epochs=200, lr=0.001):
    model = StructuralNet(X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr) # No weight decay for debug
    
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(R), torch.FloatTensor(Y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    losses = []
    for ep in range(epochs):
        ep_loss = 0
        for bx, br, by in loader:
            opt.zero_grad()
            out = model(bx)
            alpha = out[:, 0]
            beta = out[:, 1]
            logits = alpha + beta * br
            loss = nn.BCEWithLogitsLoss()(logits, by)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(loader))
        if ep % 20 == 0:
            print(f"Epoch {ep}: Loss {losses[-1]:.4f}")
            
    return model

if __name__ == "__main__":
    dgp = LogitDGP()
    X, R, Y, alpha_true, beta_true, _ = dgp.generate(2000)
    
    print("Training StructuralNet...")
    model = train_logit_model(X, R, Y, epochs=200, lr=0.001)
    
    model.eval()
    with torch.no_grad():
        out = model(torch.FloatTensor(X))
        alpha_hat = out[:, 0].numpy()
        beta_hat = out[:, 1].numpy()
        
    mse_alpha = np.mean((alpha_hat - alpha_true)**2)
    mse_beta = np.mean((beta_hat - beta_true)**2)
    
    print(f"MSE Alpha: {mse_alpha:.4f}")
    print(f"MSE Beta: {mse_beta:.4f}")
    
    print(f"True Mean Beta: {np.mean(beta_true):.4f}")
    print(f"Est Mean Beta: {np.mean(beta_hat):.4f}")
    
    # Check correlation
    corr_beta = np.corrcoef(beta_true, beta_hat)[0,1]
    print(f"Corr Beta: {corr_beta:.4f}")

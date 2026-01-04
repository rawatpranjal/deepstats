
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
import torch.autograd.functional as F_grad

# Set device
device = torch.device("cpu")

# ==========================================
# 1. Data Generation Processes (DGPs)
# ==========================================

class LinearDGP:
    def __init__(self, d=5, seed=42):
        self.d = d
        self.rng = np.random.default_rng(seed)
        
    def alpha_func(self, x):
        return 1.0 + 0.5*x[:,0] - 0.2*x[:,1]
    
    def beta_func(self, x):
        return -1.0 + 0.3*x[:,2] + 0.4*x[:,0]*x[:,1]
        
    def generate(self, n):
        X = self.rng.normal(0, 1, (n, self.d))
        # Binary Treatment T
        logits = 0.5 * X[:,0]
        p = 1 / (1 + np.exp(-logits))
        T = self.rng.binomial(1, p)
        
        alpha = self.alpha_func(X)
        beta = self.beta_func(X)
        
        # Y = alpha + beta*T + noise
        noise = 0.5 * self.rng.normal(0, 1, n)
        Y = alpha + beta * T + noise
        
        return X, T, Y, alpha, beta, p

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
        # Continuous Treatment R
        R = 0.5 * X[:,0] + 0.5 * self.rng.normal(0, 1, n)
        
        alpha = self.alpha_func(X)
        beta = self.beta_func(X)
        
        # Logit Model
        logits = alpha + beta * R
        probs = 1 / (1 + np.exp(-logits))
        Y = self.rng.binomial(1, probs)
        
        return X, R, Y, alpha, beta, probs


from joblib import Parallel, delayed
import multiprocessing

# ... (Previous imports remain, ensure joblib is imported)

# ==========================================
# 2. Models (Reverted to Smaller Capacity)
# ==========================================

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

class TreatmentNet(nn.Module):
    def __init__(self, d, is_binary=True):
        super().__init__()
        self.is_binary = is_binary
        self.hidden = nn.Sequential(
            nn.Linear(d, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        out_dim = 1 if is_binary else 2
        self.out = nn.Linear(16, out_dim)
        
    def forward(self, x):
        feat = self.hidden(x)
        return self.out(feat)

# ==========================================
# 3. Loss Functions & FLM Logic
# ==========================================

def linear_loss(y, t, theta):
    # theta = [alpha, beta]
    # y_pred = alpha + beta * t
    # loss = (y - y_pred)**2
    alpha = theta[0]
    beta = theta[1]
    y_pred = alpha + beta * t
    return (y - y_pred)**2

def logit_loss(y, t, theta):
    # theta = [alpha, beta]
    # logits = alpha + beta * t
    # loss = BCEWithLogits(logits, y)
    alpha = theta[0]
    beta = theta[1]
    logits = alpha + beta * t
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

def target_functional(theta):
    # Target: beta (the second parameter)
    return theta[1]

# ==========================================
# 4. Estimation & Inference (Updated Defaults)
# ==========================================

def train_model(model, loader, loss_wrapper, lr=0.001, epochs=200): # Tuned based on debug
    opt = optim.Adam(model.parameters(), lr=lr) # Removed weight_decay for now to match debug
    
    for ep in range(epochs):
        for bx, bt, by in loader:
            opt.zero_grad()
            theta = model(bx)
            if loss_wrapper == linear_loss_batch:
                loss = torch.mean(loss_wrapper(by, bt, theta))
            elif loss_wrapper == logit_loss_batch:
                loss = torch.mean(loss_wrapper(by, bt, theta))
            else:
                loss = loss_wrapper(theta, by)
            loss.backward()
            opt.step()
    return model

def linear_loss_batch(y, t, theta):
    return (y - (theta[:,0] + theta[:,1]*t))**2

def logit_loss_batch(y, t, theta):
    logits = theta[:,0] + theta[:,1]*t
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')

def train_treatment(X, T, is_binary=True, epochs=200): # Increased epochs
    model = TreatmentNet(X.shape[1], is_binary).to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(T))
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    for ep in range(epochs):
        for bx, bt in loader:
            opt.zero_grad()
            out = model(bx)
            if is_binary:
                loss = nn.BCEWithLogitsLoss()(out.squeeze(), bt)
            else:
                mu = out[:,0]
                logvar = out[:,1]
                var = torch.exp(logvar)
                loss = torch.mean(0.5*logvar + 0.5*(bt - mu)**2/var)
            loss.backward()
            opt.step()
    return model

def compute_flm_influence(X, T, Y, struct_model, treat_model, loss_single_fn, is_binary_treatment):
    """
    Computes the influence function values using the FLM formula:
    psi = H(theta) - grad(H)^T * E[Hess(L)]^{-1} * grad(L)
    """
    n = len(X)
    X_torch = torch.FloatTensor(X).to(device)
    T_torch = torch.FloatTensor(T).to(device)
    Y_torch = torch.FloatTensor(Y).to(device)
    
    struct_model.eval()
    treat_model.eval()
    
    scores = []
    
    # We process one by one or in small batches for Autograd Hessian
    # For prototype, one by one is safer to avoid batch-Hessian complexity
    
    with torch.no_grad():
        theta_pred = struct_model(X_torch) # (N, 2)
        if is_binary_treatment:
            treat_logits = treat_model(X_torch).squeeze()
            treat_probs = torch.sigmoid(treat_logits)
        else:
            treat_out = treat_model(X_torch)
            treat_mu = treat_out[:,0]
            treat_std = torch.exp(0.5*treat_out[:,1])

    for i in range(n):
        theta_i = theta_pred[i].clone().detach().requires_grad_(True)
        y_i = Y_torch[i]
        t_i = T_torch[i]
        
        # 1. Compute Gradient of Loss w.r.t theta
        # L(y, t, theta)
        l_val = loss_single_fn(y_i, t_i, theta_i)
        grad_l = torch.autograd.grad(l_val, theta_i, create_graph=True)[0]
        
        # 2. Compute Expected Hessian E[Hess(L) | X]
        # Monte Carlo Integration for Expected Hessian
        M = 100
        hess_sum = torch.zeros(2, 2).to(device)
        
        if is_binary_treatment:
            # Binary T: Sum over T=0, 1
            ts = [0.0, 1.0]
            ps = [1 - treat_probs[i], treat_probs[i]]
            
            for t_val, p_val in zip(ts, ps):
                t_tens = torch.tensor(t_val).to(device)
                
                for _ in range(10): # Small inner sample
                    if loss_single_fn == linear_loss:
                        # Y ~ N(alpha + beta*t, 1)
                        mu_y = theta_i[0] + theta_i[1]*t_val
                        y_sim = mu_y + torch.randn(1).to(device)[0]
                    else:
                        # Y ~ Bern(sigmoid(alpha + beta*t))
                        logits_y = theta_i[0] + theta_i[1]*t_val
                        prob_y = torch.sigmoid(logits_y)
                        y_sim = torch.bernoulli(prob_y)
                    
                    def loss_func_h(th):
                        return loss_single_fn(y_sim, t_tens, th)
                    
                    hess = torch.autograd.functional.hessian(loss_func_h, theta_i)
                    hess_sum += hess * p_val / 10.0
                    
        else:
            # Continuous T: Sample T ~ N(mu, std)
            for _ in range(M):
                t_sim = treat_mu[i] + treat_std[i] * torch.randn(1).to(device)[0]
                
                # Sample Y | T_sim
                if loss_single_fn == linear_loss:
                    mu_y = theta_i[0] + theta_i[1]*t_sim
                    y_sim = mu_y + torch.randn(1).to(device)[0]
                else:
                    logits_y = theta_i[0] + theta_i[1]*t_sim
                    prob_y = torch.sigmoid(logits_y)
                    y_sim = torch.bernoulli(prob_y)
                
                def loss_func_h(th):
                    return loss_single_fn(y_sim, t_sim, th)
                
                hess = torch.autograd.functional.hessian(loss_func_h, theta_i)
                hess_sum += hess / M
                
        J = hess_sum
        
        # 3. Compute Gradient of Target Functional H(theta)
        # H(theta) = theta[1] (beta)
        # grad H = [0, 1]
        grad_H = torch.tensor([0.0, 1.0]).to(device)
        
        # 4. Compute Correction
        # psi = H(theta) - grad_H^T * J^{-1} * grad_L
        # Regularize J (Stronger regularization)
        J_reg = J + 1e-2 * torch.eye(2).to(device)
        try:
            J_inv = torch.inverse(J_reg)
        except:
            J_inv = torch.eye(2).to(device) # Fallback
            
        correction = grad_H @ J_inv @ grad_l
        
        # Clip Correction Term
        correction = torch.clamp(correction, -10.0, 10.0)
        
        # Plug-in
        h_val = theta_i[1]
        
        # The formula in prompt: psi = H - mu - correction
        # We return (H - correction). The mean subtraction happens later.
        psi = h_val - correction
        scores.append(psi.item())
        
    return np.array(scores)

# ==========================================
# 5. Simulation Loop (Parallelized)
# ==========================================

def run_single_sim(i, dgp_type, n_obs, mu_true, n_boot):
    # Ensure single thread per process to avoid contention
    torch.set_num_threads(1)
    
    if dgp_type == 'Linear':
        dgp = LinearDGP()
        loss_batch = linear_loss_batch
        loss_single = linear_loss
        is_binary = True
    else:
        dgp = LogitDGP()
        loss_batch = logit_loss_batch
        loss_single = logit_loss
        is_binary = False
        
    X, T, Y, _, _, _ = dgp.generate(n_obs)
    
    # Split Data
    indices = np.random.permutation(n_obs)
    half = n_obs // 2
    idx1, idx2 = indices[:half], indices[half:]
    
    psi_all = []
    
    for train_idx, eval_idx in [(idx1, idx2), (idx2, idx1)]:
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_ev, T_ev, Y_ev = X[eval_idx], T[eval_idx], Y[eval_idx]
        
        # Train Structural
        struct_model = StructuralNet(X.shape[1]).to(device)
        ds_str = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(T_tr), torch.FloatTensor(Y_tr))
        ld_str = DataLoader(ds_str, batch_size=64, shuffle=True)
        train_model(struct_model, ld_str, loss_batch, epochs=100)
        
        # Train Treatment
        treat_model = train_treatment(X_tr, T_tr, is_binary=is_binary, epochs=100)
        
        # Compute Scores
        scores = compute_flm_influence(X_ev, T_ev, Y_ev, struct_model, treat_model, loss_single, is_binary)
        psi_all.extend(scores)
        
    mu_inf = np.mean(psi_all)
    se_inf = np.std(psi_all, ddof=1) / np.sqrt(n_obs)
    cov_inf = (mu_inf - 1.96*se_inf <= mu_true <= mu_inf + 1.96*se_inf)
    
    # Naive (Full fit)
    struct_full = StructuralNet(X.shape[1]).to(device)
    ds_full = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(T), torch.FloatTensor(Y))
    ld_full = DataLoader(ds_full, batch_size=64, shuffle=True)
    train_model(struct_full, ld_full, loss_batch, epochs=100)
    struct_full.eval()
    with torch.no_grad():
        thetas = struct_full(torch.FloatTensor(X).to(device))
        betas = thetas[:,1].numpy()
    mu_naive = np.mean(betas)
    se_naive = np.std(betas, ddof=1) / np.sqrt(n_obs)
    cov_naive = (mu_naive - 1.96*se_naive <= mu_true <= mu_naive + 1.96*se_naive)
    
    res_list = []
    # MSE = Bias^2 + Var = (Est - True)^2
    mse_naive = (mu_naive - mu_true)**2
    mse_inf = (mu_inf - mu_true)**2
    
    res_list.append({'Model': dgp_type, 'Method': 'Naive', 'Estimate': mu_naive, 'SE': se_naive, 'Bias': mu_naive - mu_true, 'MSE': mse_naive, 'Coverage': cov_naive})
    res_list.append({'Model': dgp_type, 'Method': 'Influence', 'Estimate': mu_inf, 'SE': se_inf, 'Bias': mu_inf - mu_true, 'MSE': mse_inf, 'Coverage': cov_inf})
    
    # Bootstrap Naive (Only for first few iterations if running many, but here parallel makes it easier)
    # Let's run bootstrap only if i < 10 to save time even in parallel
    if i < 10:
        boot_ests = []
        for b in range(n_boot):
            idx_b = np.random.randint(0, n_obs, n_obs)
            X_b, T_b, Y_b = X[idx_b], T[idx_b], Y[idx_b]
            
            model_b = StructuralNet(X.shape[1]).to(device)
            ds_b = TensorDataset(torch.FloatTensor(X_b), torch.FloatTensor(T_b), torch.FloatTensor(Y_b))
            ld_b = DataLoader(ds_b, batch_size=64, shuffle=True)
            train_model(model_b, ld_b, loss_batch, epochs=50) # Fewer epochs for boot
            
            with torch.no_grad():
                thetas_b = model_b(torch.FloatTensor(X_b).to(device))
                mu_b = np.mean(thetas_b[:,1].numpy())
            boot_ests.append(mu_b)
        
        se_boot = np.std(boot_ests, ddof=1)
        cov_boot = (mu_naive - 1.96*se_boot <= mu_true <= mu_naive + 1.96*se_boot)
        mse_boot = (mu_naive - mu_true)**2 # Bootstrap estimate is same as Naive, just SE differs
        res_list.append({'Model': dgp_type, 'Method': 'Bootstrap', 'Estimate': mu_naive, 'SE': se_boot, 'Bias': mu_naive - mu_true, 'MSE': mse_boot, 'Coverage': cov_boot})
        
    return res_list

def run_flm_simulation(n_sims=50, n_obs=5000, n_boot=30): # Increased N to 5000
    results = []
    num_cores = multiprocessing.cpu_count()
    print(f"Running simulation on {num_cores} cores with N={n_obs}...")
    
    # 1. Linear
    print("Running Linear Simulation...")
    dgp_lin = LinearDGP()
    X_large = np.random.normal(0, 1, (100000, 5))
    mu_true_lin = np.mean(dgp_lin.beta_func(X_large))
    print(f"True Mu (Linear): {mu_true_lin:.4f}")
    
    res_lin = Parallel(n_jobs=num_cores)(
        delayed(run_single_sim)(i, 'Linear', n_obs, mu_true_lin, n_boot) for i in tqdm(range(n_sims))
    )
    for r in res_lin:
        results.extend(r)

    # 2. Logit
    print("Running Logit Simulation...")
    dgp_log = LogitDGP()
    mu_true_log = np.mean(dgp_log.beta_func(X_large))
    print(f"True Mu (Logit): {mu_true_log:.4f}")
    
    res_log = Parallel(n_jobs=num_cores)(
        delayed(run_single_sim)(i, 'Logit', n_obs, mu_true_log, n_boot) for i in tqdm(range(n_sims))
    )
    for r in res_log:
        results.extend(r)

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_flm_simulation(n_sims=50, n_obs=5000) # Increased N to 5000
    print(df.groupby(['Model', 'Method']).mean())
    df.to_csv("prototypes/misra_validation_results.csv", index=False)

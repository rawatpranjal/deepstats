import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# 1) DATA GENERATION
###############################################################################
def generate_synthetic_data(
    seed=999,
    N=5000,         # Number of consumers
    J=200,          # Number of products
    K=100,          # Dimensionality of each product's feature vector M_j
    alpha_true=-1.0
):
    """
    Creates a synthetic dataset of dimension-K product attributes M_j,
    a 'true' function g_true_j = j/(J-1) mapped in [0,1],
    and random prices p_j ~ Uniform(1,2).
    Then simulates discrete choices from N identical consumers
    (outside good included, utility=0) via logit with error ~ Gumbel(0,1).
    
    Returns:
      M: [J, K] float tensor of product attributes
      p: [J] float tensor of prices
      g_true: [J] float tensor of true 'intrinsic utility offsets'
      c_counts: [J+1] int tensor of choice counts (index 0 is outside good).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1) Create product feature matrix M_j in R^K
    #    e.g. M_j each dimension ~ Uniform(0,1).
    M = torch.rand(J, K)  # shape [J, K]

    # 2) g_true_j = j/(J-1), mapped to [0, 1]
    if J > 1:
        g_true = torch.arange(J, dtype=torch.float32) / (J-1)
    else:
        g_true = torch.zeros(J, dtype=torch.float32)

    # 3) Price p_j ~ Uniform(1,2)
    p = 1.0 + torch.rand(J)

    # 4) Simulate consumer choices via logit
    #    Utility_j = alpha_true * p_j + g_true_j + e_j
    #    Probability(choose j) = exp(util_j) / sum over all (including outside good=0)
    with torch.no_grad():
        util_j = alpha_true * p + g_true  # shape [J]
        stacked_utils = torch.cat([torch.zeros(1), util_j], dim=0)  # [J+1] w/ outside good
        log_denom = torch.logsumexp(stacked_utils, dim=0)
        prob0 = 1.0 / torch.exp(log_denom)               # Probability outside good
        probs_j = torch.exp(util_j - log_denom)          # [J], sum_j + prob0 = 1
        cat_probs = torch.cat([prob0.view(1), probs_j])  # [J+1]

        # Sample each consumer's choice
        c_counts = torch.zeros(J+1, dtype=torch.int64)
        choices = np.random.choice(J+1, size=N, p=cat_probs.numpy())
        for c in choices:
            c_counts[c] += 1

    return M, p, g_true, c_counts


###############################################################################
# 2) MODEL DEFINITION
###############################################################################
class MLPUtility(nn.Module):
    """
    A simple feedforward neural net mapping M_j (size K) -> scalar utility offset g(M_j),
    with final layer 'fc_out' having no bias to reduce intercept overshadowing price.
    """
    def __init__(self, K=100, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(K, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)  # No bias

    def forward(self, x):
        """
        x shape: [batch_size, K]
        returns shape: [batch_size] (scalar offset)
        """
        x = x.view(x.size(0), -1)  # Ensure the shape is [batch_size, K]
        x = F.relu(self.fc1(x))    # [batch_size, hidden_dim]
        out = self.fc2(x).squeeze(-1)  # [batch_size]
        return out


###############################################################################
# 3) NEGATIVE LOG-LIKELIHOOD WITH REGULARIZATION
###############################################################################
def neg_log_likelihood(alpha_param, model, M, p, c_counts, regularization_factor=1e-4):
    """
    c_counts[0] = # of times outside good was chosen
    c_counts[j] = # chosen count for product j>0
    Utility_j = alpha_param * p_j + model(M_j).
    """
    # model(M) -> [J], utility offsets
    g_j = model(M)
    util_j = alpha_param * p + g_j  # [J]
    all_utils = torch.cat([torch.zeros(1, device=util_j.device), util_j], dim=0)  # [J+1]
    log_denom = torch.logsumexp(all_utils, dim=0)
    
    total_counts = c_counts.sum().float()
    weighted_utils = torch.sum(c_counts * all_utils)
    ll = weighted_utils - total_counts * log_denom
    
    # Apply regularization on the network's output (L2 regularization)
    regularization_loss = regularization_factor * torch.sum(g_j**2)
    
    # Add regularization loss to the NLL
    nll = -ll + regularization_loss
    return nll


###############################################################################
# 4) TRAINING LOOP
###############################################################################
def train_model(
    model,
    alpha_param,
    M,
    p,
    g_true,
    c_counts,
    epochs=300,
    lr=1e-3,
    regularization_factor=1e-4,
    verbose=False
):
    """
    Jointly train alpha_param and model(M_j) by maximizing likelihood
    (minimizing negative log-likelihood).
    We'll track:
      - NLL
      - alpha
      - R^2(g_true,g_est)
    and return them for plotting.
    """
    optimizer = optim.Adam(list(model.parameters()) + [alpha_param], lr=lr, weight_decay=regularization_factor)

    # For R^2 calculation
    mean_g = torch.mean(g_true)
    sst = torch.sum((g_true - mean_g)**2).item() if len(g_true) > 1 else 1e-8

    nll_history = []
    alpha_history = []
    r2_history = []

    for ep in range(epochs):
        optimizer.zero_grad()
        nll = neg_log_likelihood(alpha_param, model, M, p, c_counts, regularization_factor)
        nll.backward()
        optimizer.step()

        with torch.no_grad():
            # Evaluate alpha, NLL, R^2
            g_est = model(M)
            sse = torch.sum((g_est - g_true)**2).item()
            r2_val = 1.0 - sse / sst if sst > 0 else 0.0

            nll_val = nll.item()
            a_val = alpha_param.item()
            nll_history.append(nll_val)
            alpha_history.append(a_val)
            r2_history.append(r2_val)

        # Print progress
        if verbose and ((ep+1) % 50 == 0 or ep == 0):
            print(f"Epoch {ep+1}/{epochs}, NLL={nll_val:.3f}, alpha={a_val:.3f}, R^2={r2_val:.3f}")

    return nll_history, alpha_history, r2_history


###############################################################################
# 5) PLOTTING AND EVALUATION
###############################################################################
def final_evaluation_plot(nll_hist, alpha_hist, r2_hist, alpha_true, g_true, g_est):
    """
    Shows:
      - NLL vs epoch
      - alpha & R^2 vs epoch
      - scatter of g_true vs. g_est
    Prints final alpha and final R^2.
    """
    alpha_est = alpha_hist[-1]

    sse = torch.sum((g_est - g_true)**2).item()
    mean_g = torch.mean(g_true).item()
    sst = torch.sum((g_true - mean_g)**2).item() if len(g_true) > 1 else 1e-8
    r2_final = 1.0 - sse / sst if sst > 0 else 0.0

    print("\n--- Final Results ---")
    print(f"True alpha = {alpha_true:.3f}, Estimated alpha = {alpha_est:.3f}")
    print(f"Final R^2(g_true,g_est) = {r2_final:.3f}\n")

    fig, axes = plt.subplots(1, 3, figsize=(15,4))

    # (1) NLL
    axes[0].plot(nll_hist, color='gray')
    axes[0].set_title("NLL")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Neg Log-Likelihood")

    # (2) alpha & R^2
    ax1 = axes[1]
    ax2 = ax1.twinx()
    l1 = ax1.plot(alpha_hist, color='blue', label='alpha')
    l2 = ax2.plot(r2_hist, color='red', label='R^2')
    ax1.set_title("Alpha & R^2")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("alpha")
    ax2.set_ylabel("R^2")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    # (3) scatter
    axes[2].scatter(g_true.cpu().numpy(), g_est.cpu().numpy(), alpha=0.4, edgecolors='k')
    axes[2].set_xlabel("True g(M_j)")
    axes[2].set_ylabel("Estimated g(M_j)")
    axes[2].set_title("g(M_j): True vs. Predicted")

    plt.tight_layout()
    plt.show()


###############################################################################
# 6) MAIN (minimal)
###############################################################################
def main():
    M, p, g_true, c_counts = generate_synthetic_data(
        seed=999, N=100_000, J=20_000, K=5, alpha_true=-1.0
    )
    model = MLPUtility(K=100, hidden_dim=64)
    alpha_param = nn.Parameter(torch.tensor(0.0))

    nll_hist, alpha_hist, r2_hist = train_model(
        model, alpha_param, M, p, g_true, c_counts,
        epochs=10_000, lr=3e-4, regularization_factor=1e-2, verbose=True
    )

    with torch.no_grad():
        g_est = model(M)

    final_evaluation_plot(nll_hist, alpha_hist, r2_hist, alpha_true=-1.0, g_true=g_true, g_est=g_est)


if __name__ == "__main__":
    main()

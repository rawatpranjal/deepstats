import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import norm
import math

###############################################################################
# 1. DATA GENERATION
###############################################################################

def generate_data_example3_fractional(n=20000, d=5, seed=456):
    """
    Example 3 (Fractional):
    Y in [0,1], with E[Y|X,D] = logistic( alpha(X) + beta(X)*D ).
    For demonstration, we let Y be Bernoulli with that probability (common approach).
    We have a "treatment" D ~ Bernoulli(prob=some function of X).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0,1,(n,d))

    def alpha_func(x):
        return 0.2 + 0.3*x[:,0] - 0.2*x[:,1]

    def beta_func(x):
        return -0.6 + 0.4*x[:,2] + 0.3*x[:,3]*x[:,1]

    # Generate D ~ Bernoulli(prob= logistic(0.3*X[:,1]))
    pD = 1/(1+np.exp(-0.3*X[:,1]))
    D = rng.binomial(1, pD)

    alphaX = alpha_func(X)
    betaX  = beta_func(X)

    # logistic index => alpha(X)+ beta(X)*D
    index = alphaX + betaX*D
    probY = 1/(1 + np.exp(-index))

    # now Y is Bernoulli(probY)
    Y = rng.binomial(1, probY).astype(float)

    return X, D, Y, alpha_func, beta_func

def generate_data_example4_tobit(n=20000, d=4, seed=999):
    """
    Example 4 (Tobit):
    Y = max(0, Y*).
    Y* ~ Normal( mu(X) + gamma(X)*D, sigma(X)^2 ).
    We let:
      mu(X)= 1 + 0.5*X0
      gamma(X)= -0.8 + 0.2*X1*X2
      sigma(X)= exp(0.1*X3)
    D ~ Bernoulli(0.5).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0,1,(n,d))

    def mu_func(x):
        return 1.0 + 0.5*x[:,0]

    def gamma_func(x):
        return -0.8 + 0.2*x[:,1]*x[:,2]

    def sigma_func(x):
        return np.exp(0.1*x[:,3])

    D = rng.binomial(1, 0.5, size=n)
    muX     = mu_func(X)
    gammaX  = gamma_func(X)
    sigmaX  = sigma_func(X)

    Y_star  = muX + gammaX*D + sigmaX*rng.normal(0,1,n)
    Yobs    = np.maximum(0, Y_star)

    return X, D, Yobs, mu_func, gamma_func, sigma_func

###############################################################################
# 2. MODEL ARCHITECTURES
###############################################################################

class DeepFractionalNet(nn.Module):
    """
    Outputs (alpha(x), beta(x)).
    We'll interpret E[Y|X,D]= logistic(alpha+beta*D). We train by BCE.
    """
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d,64)
        self.fc2 = nn.Linear(64,32)
        self.out = nn.Linear(32,2)

    def forward(self, x):
        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        ab = self.out(z) # shape (bs,2)
        return ab

class DeepTobitNet(nn.Module):
    """
    Outputs (mu(x), gamma(x), log_sigma(x)).
    We'll define the tobit log-likelihood to handle Y= max(0,Y*).
    """
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d,64)
        self.fc2 = nn.Linear(64,32)
        self.out = nn.Linear(32,3)

    def forward(self, x):
        z = torch.relu(self.fc1(x))
        z = torch.relu(self.fc2(z))
        out_ = self.out(z) # shape (bs,3)
        return out_

###############################################################################
# 3. PRINT DATA OVERVIEW
###############################################################################

def print_data_overview(X, D, Y, label="Dataset"):
    n, d = X.shape
    print(f"\n--- {label} Overview ---")
    print(f"N={n}, d={d}")
    print(f"Shapes => X:{X.shape}, D:{D.shape}, Y:{Y.shape}")

    df_head = pd.DataFrame(np.hstack([X[:5], D[:5,None], Y[:5,None]]),
                           columns=[f"X{j}" for j in range(d)] + ["D","Y"])
    print("\nFirst 5 observations:\n", df_head.to_string(index=False))

    df_summary = pd.DataFrame(np.hstack([X, D[:,None], Y[:,None]]),
                              columns=[f"X{j}" for j in range(d)] + ["D","Y"])
    print("\nSummary stats:\n", df_summary.describe().to_string())

###############################################################################
# 4. TRAINING (Fractional, Tobit) + Prediction
###############################################################################

def train_fractional_net(Xtrain, Dtrain, Ytrain, epochs=10, lr=1e-3, batch_size=128):
    """
    Train a net that produces alpha, beta for E[Y]= logistic(alpha+ beta*D),
    using BCE on the Bernoulli outcome.
    """
    ds = TensorDataset(torch.from_numpy(Xtrain).float(),
                       torch.from_numpy(Dtrain).float().unsqueeze(-1),
                       torch.from_numpy(Ytrain).float().unsqueeze(-1))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = DeepFractionalNet(Xtrain.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    bceloss = nn.BCELoss()

    for e in range(epochs):
        run_loss = 0.0
        for xx, dd, yy in dl:
            opt.zero_grad()
            ab = model(xx)  # shape (bs,2)
            alpha_ = ab[:,0:1]
            beta_  = ab[:,1:2]
            index_ = alpha_ + beta_*dd
            p_     = torch.sigmoid(index_)
            loss   = bceloss(p_, yy)
            loss.backward()
            opt.step()
            run_loss+=loss.item()
    return model

def predict_fractional_net(model, X, D):
    """
    model => alpha(x), beta(x).
    Then p = sigmoid(alpha+ beta*D).
    Return alpha_est, beta_est, p_est arrays.
    """
    with torch.no_grad():
        xt = torch.from_numpy(X).float()
        dt = torch.from_numpy(D).float().unsqueeze(-1)
        ab = model(xt)
        alpha_ = ab[:,0:1]
        beta_  = ab[:,1:2]
        index_ = alpha_ + beta_*dt
        p_     = torch.sigmoid(index_)
    return alpha_.numpy().flatten(), beta_.numpy().flatten(), p_.numpy().flatten()

def tobit_negloglik(y, d, net_out):
    """
    net_out => (mu, gamma, log_sigma).
    Y= max(0, Y*). If Y>0 => pdf normal, if Y=0 => cdf normal.
    negative log-likelihood.
    """
    mu_     = net_out[:,0:1]
    gamma_  = net_out[:,1:2]
    logsig_ = net_out[:,2:3]
    sigma_  = torch.exp(logsig_)
    mean_   = mu_ + gamma_*d
    pos_mask= (y>1e-12).float()

    # logpdf if y>0
    zscore = (y-mean_)/sigma_
    logpdf = -0.5*zscore**2 - torch.log(sigma_) -0.5*math.log(2*math.pi)
    # logcdf if y=0
    cdf_   = 0.5*(1+ torch.erf((0.0-mean_)/(sigma_*math.sqrt(2.0))) )
    logcdf = torch.log(torch.clamp(cdf_,min=1e-15))

    ll = pos_mask*logpdf + (1-pos_mask)*logcdf
    return -torch.mean(ll)

def train_tobit_net(Xtrain, Dtrain, Ytrain, epochs=10, lr=1e-3, batch_size=128):
    ds = TensorDataset(torch.from_numpy(Xtrain).float(),
                       torch.from_numpy(Dtrain).float().unsqueeze(-1),
                       torch.from_numpy(Ytrain).float().unsqueeze(-1))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = DeepTobitNet(Xtrain.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        run_loss=0.0
        for xx, dd, yy in dl:
            opt.zero_grad()
            out_ = model(xx)
            loss = tobit_negloglik(yy, dd, out_)
            loss.backward()
            opt.step()
            run_loss+=loss.item()
    return model

def predict_tobit_net(model, X, D):
    """
    model => (mu(x), gamma(x), log_sigma(x)).
    Return mu_hat, gamma_hat, sigma_hat.
    Possibly also compute E[Y], etc.  We'll keep it simple.
    """
    with torch.no_grad():
        xt = torch.from_numpy(X).float()
        dt = torch.from_numpy(D).float().unsqueeze(-1)
        out_ = model(xt)
        mu_     = out_[:,0:1]
        gamma_  = out_[:,1:2]
        logsig_ = out_[:,2:3]
        sigma_  = torch.exp(logsig_)
    return mu_.numpy().flatten(), gamma_.numpy().flatten(), sigma_.numpy().flatten()

###############################################################################
# 5. SINGLE-RUN EVALUATIONS
###############################################################################

def run_example_fractional(X, D, Y, alpha_func, beta_func,
                           example_name="Example3-Fractional",
                           epochs=10, lr=1e-3, batch_size=128, verb_mode=2):
    print_data_overview(X, D, Y, label=example_name)
    # train
    net = train_fractional_net(X, D, Y, epochs=epochs, lr=lr, batch_size=batch_size)
    alpha_est, beta_est, p_est = predict_fractional_net(net, X, D)

    # get average alpha,beta, compare to true
    a_true = alpha_func(X).mean()
    b_true = beta_func(X).mean()
    a_hat  = alpha_est.mean()
    b_hat  = beta_est.mean()

    # quick table
    def confint(e, se):
        z=1.96
        return (e-z*se, e+z*se)
    # naive se => sample stdev / sqrt(n)
    n = X.shape[0]
    a_se = np.std(alpha_est,ddof=1)/np.sqrt(n)
    b_se = np.std(beta_est,ddof=1)/np.sqrt(n)
    cA   = confint(a_hat,a_se)
    cB   = confint(b_hat,b_se)

    data = [
      ["Alpha", f"{a_true:.3f}", f"{a_hat:.3f}", f"{a_se:.3f}", f"{cA[0]:.3f}", f"{cA[1]:.3f}"],
      ["Beta" , f"{b_true:.3f}", f"{b_hat:.3f}", f"{b_se:.3f}", f"{cB[0]:.3f}", f"{cB[1]:.3f}"]
    ]
    headers=["Param","True","Est","StdErr","CI_Low","CI_High"]
    table_str = tabulate(data, headers, tablefmt="github")

    if verb_mode>=1:
        print(f"\nResults for {example_name}")
        print(table_str)

def run_example_tobit(X, D, Y, mu_func, gamma_func, sigma_func,
                      example_name="Example4-Tobit",
                      epochs=10, lr=1e-3, batch_size=128, verb_mode=2):
    print_data_overview(X, D, Y, label=example_name)
    # train
    net = train_tobit_net(X, D, Y, epochs=epochs, lr=lr, batch_size=batch_size)
    mu_hat, gamma_hat, sigma_hat = predict_tobit_net(net, X, D)

    # average mu, gamma
    m_true = mu_func(X).mean()
    g_true = gamma_func(X).mean()
    # ignoring sigma
    m_hat  = mu_hat.mean()
    g_hat  = gamma_hat.mean()

    n = X.shape[0]
    m_se = np.std(mu_hat,ddof=1)/np.sqrt(n)
    g_se = np.std(gamma_hat,ddof=1)/np.sqrt(n)

    def confint(e,se):
        z=1.96
        return (e-z*se, e+z*se)

    cm = confint(m_hat, m_se)
    cg = confint(g_hat, g_se)

    data = [
      ["Mu",     f"{m_true:.3f}", f"{m_hat:.3f}", f"{m_se:.3f}", f"{cm[0]:.3f}", f"{cm[1]:.3f}"],
      ["Gamma",  f"{g_true:.3f}", f"{g_hat:.3f}", f"{g_se:.3f}", f"{cg[0]:.3f}", f"{cg[1]:.3f}"]
    ]
    heads = ["Param","True","Est","StdErr","CI_Low","CI_High"]
    table_str = tabulate(data, heads, tablefmt="github")

    if verb_mode>=1:
        print(f"\nResults for {example_name} (Tobit partial means):")
        print(table_str)

###############################################################################
# 6. MONTE CARLO ILLUSTRATION (OPTIONAL)
###############################################################################

def mc_example3_fractional(nrep=20, n=5000, d=5, verb_mode=2):
    """
    Repeated draws from generate_data_example3_fractional
    for a quick demonstration.
    We'll store the mean of alpha(x), beta(x).
    """
    rng = np.random.default_rng(9999)
    alpha_ests=[]
    beta_ests=[]
    for i in range(nrep):
        X, D, Y, aF, bF = generate_data_example3_fractional(n=n,d=d,seed=rng.integers(1e9))
        net = train_fractional_net(X,D,Y,epochs=5,lr=1e-3,batch_size=256)
        a_, b_, _ = predict_fractional_net(net,X,D)
        alpha_ests.append(a_.mean())
        beta_ests.append(b_.mean())

    alpha_ests = np.array(alpha_ests)
    beta_ests  = np.array(beta_ests)
    # approximate true from a large sample
    Xb, Db, _, aFb, bFb = generate_data_example3_fractional(n=50000,d=d,seed=12345)
    a_true = aFb(Xb).mean()
    b_true = bFb(Xb).mean()

    if verb_mode>=1:
        print("\n=== MC Example3 (Fractional) ===")
        print(f"Approx true alpha={a_true:.3f}, beta={b_true:.3f}")
    # summary
    ma = alpha_ests.mean()
    mb = beta_ests.mean()
    sa = alpha_ests.std(ddof=1)
    sb = beta_ests.std(ddof=1)
    data = [
      ["Alpha", f"{a_true:.3f}", f"{ma:.3f}", f"{sa:.3f}"],
      ["Beta" , f"{b_true:.3f}", f"{mb:.3f}", f"{sb:.3f}"]
    ]
    heads = ["Param","True","Mean(Est)","Std(Est)"]
    tab_str = tabulate(data, heads, tablefmt="github")
    if verb_mode>=1:
        print(tab_str)

def mc_example4_tobit(nrep=20, n=5000, d=4, verb_mode=2):
    """
    Repeated draws from generate_data_example4_tobit.
    We'll store the average mu(x), gamma(x).
    """
    rng = np.random.default_rng(8888)
    mu_ests=[]
    gamma_ests=[]
    for i in range(nrep):
        X, D, Y, mf, gf, sf = generate_data_example4_tobit(n=n,d=d,seed=rng.integers(1e9))
        net = train_tobit_net(X,D,Y,epochs=5,lr=1e-3,batch_size=256)
        m_, g_, s_ = predict_tobit_net(net,X,D)
        mu_ests.append(m_.mean())
        gamma_ests.append(g_.mean())

    mu_ests   = np.array(mu_ests)
    gamma_ests= np.array(gamma_ests)
    # approximate true from large sample
    Xb, Db, Yb, mfB, gfB, sfB = generate_data_example4_tobit(n=30000,d=d,seed=77777)
    m_true = mfB(Xb).mean()
    g_true = gfB(Xb).mean()

    if verb_mode>=1:
        print("\n=== MC Example4 (Tobit) ===")
        print(f"Approx true mu={m_true:.3f}, gamma={g_true:.3f}")
    mm = mu_ests.mean()
    gm = gamma_ests.mean()
    sm = mu_ests.std(ddof=1)
    sg = gamma_ests.std(ddof=1)
    data = [
      ["Mu",    f"{m_true:.3f}", f"{mm:.3f}", f"{sm:.3f}"],
      ["Gamma", f"{g_true:.3f}", f"{gm:.3f}", f"{sg:.3f}"]
    ]
    heads = ["Param","True","Mean(Est)","Std(Est)"]
    tab_str = tabulate(data, heads, tablefmt="github")
    if verb_mode>=1:
        print(tab_str)

###############################################################################
# 7. MAIN
###############################################################################

def main(verb_mode=2, folder_name="results"):
    out_dir = os.path.join("data", folder_name)
    os.makedirs(out_dir, exist_ok=True)

    orig_stdout = sys.stdout
    log_path = os.path.join(out_dir, "stdout.log")
    log_file = open(log_path, "w")
    sys.stdout = log_file

    # Example3: Single run
    X3, D3, Y3, aF3, bF3 = generate_data_example3_fractional(n=5000, d=5, seed=456)
    run_example_fractional(X3, D3, Y3, aF3, bF3,
                           example_name="Example3-Fractional",
                           epochs=10, lr=1e-3, batch_size=256, verb_mode=verb_mode)

    # Example4: Single run
    X4, D4, Y4, muF4, gammaF4, sigmaF4 = generate_data_example4_tobit(n=5000, d=4, seed=999)
    run_example_tobit(X4, D4, Y4, muF4, gammaF4, sigmaF4,
                      example_name="Example4-Tobit",
                      epochs=10, lr=1e-3, batch_size=256, verb_mode=verb_mode)

    # Monte Carlo for Example3
    mc_example3_fractional(nrep=10, n=2000, d=5, verb_mode=verb_mode)

    # Monte Carlo for Example4
    mc_example4_tobit(nrep=10, n=2000, d=4, verb_mode=verb_mode)

    sys.stdout = orig_stdout
    log_file.close()
    if verb_mode>=1:
        print(f"\nLogs and any printouts are in '{out_dir}/stdout.log'.")

if __name__=="__main__":
    main(verb_mode=2, folder_name="frac_tobit_results")

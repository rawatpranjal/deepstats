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

###############################################################################
# 1. DATA GENERATION
###############################################################################

def generate_data_example1(n=20000, d=5, seed=123):
    """
    Example1 data: Y = alpha(X) + beta(X)*D + eps
    alpha(X) = 1 + 0.5*X0 - 0.2*X1
    beta(X)  = -1 + 0.3*X2 + 0.4*X0*X1
    D ~ Bernoulli( p=logistic(0.5*X0) ), eps ~ N(0,0.5^2)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0,1,(n,d))
    def alpha_func(x):
        return 1.0 + 0.5*x[:,0] - 0.2*x[:,1]
    def beta_func(x):
        return -1.0 + 0.3*x[:,2] + 0.4*x[:,0]*x[:,1]
    p = 1/(1+np.exp(-0.5*X[:,0]))
    D = rng.binomial(1, p)
    alphaX = alpha_func(X)
    betaX  = beta_func(X)
    eps = 0.5*rng.normal(0,1,n)
    Y = alphaX + betaX*D + eps
    return X, D, Y, alpha_func, beta_func

def generate_data_example2(n=20000, d=5, seed=999):
    """
    Example2 data: Y = alpha(X) + beta(X)*D + eps
    alpha(X) = 2.0 + 0.8*(X1^2)
    beta(X)  = 0.3*X0 - 0.3*X2 + 0.1*(X3^3)
    D ~ Bernoulli( p=logistic(-0.3*X1) ), eps ~ N(0,0.7^2)
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0,1,(n,d))
    def alpha_func(x):
        return 2.0 + 0.8*(x[:,1]**2)
    def beta_func(x):
        return 0.3*x[:,0] - 0.3*x[:,2] + 0.1*(x[:,3]**3)
    p = 1/(1+np.exp(0.3*X[:,1]))
    D = rng.binomial(1, p)
    alphaX = alpha_func(X)
    betaX  = beta_func(X)
    eps = 0.7*rng.normal(0,1,n)
    Y = alphaX + betaX*D + eps
    return X, D, Y, alpha_func, beta_func

###############################################################################
# 2. MODEL ARCHITECTURES
###############################################################################

class DeepLinearNet(nn.Module):
    """
    Simple feed-forward net: final output is [alpha(x), beta(x)].
    """
    def __init__(self, d):
        super().__init__()
        self.hidden1 = nn.Linear(d, 100)
        self.hidden2 = nn.Linear(100, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        z = torch.relu(self.hidden1(x))
        z = torch.relu(self.hidden2(z))
        ab = self.out(z)  # ab[:,0]=alpha, ab[:,1]=beta
        return ab

###############################################################################
# 3. PRINT DATA OVERVIEW
###############################################################################

def print_data_overview(X, D, Y, label="Dataset"):
    import pandas as pd
    n, d = X.shape
    print(f"\n--- {label} Overview ---")
    print(f"N={n}, d={d}")
    print(f"Shapes => X:{X.shape}, D:{D.shape}, Y:{Y.shape}")

    df_head = pd.DataFrame(np.hstack([X[:5], D[:5,None], Y[:5,None]]),
                           columns=[f"X{j}" for j in range(d)] + ["D","Y"])
    print("\nFirst 5 observations:\n", df_head.to_string(index=False))

    df_summary = pd.DataFrame(np.hstack([X, D[:,None], Y[:,None]]),
                              columns=[f"X{j}" for j in range(d)]+["D","Y"])
    print("\nSummary stats:\n", df_summary.describe().to_string())

###############################################################################
# 4. TRAINING alpha,beta net + logistic net for p(x)
###############################################################################

def get_r2(true_vals, pred_vals):
    ss_res = np.sum((true_vals - pred_vals)**2)
    ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
    if ss_tot < 1e-12:
        return 1.0
    return 1 - ss_res/ss_tot

def estimate_alpha_beta(Xtrain, Dtrain, Ytrain, alpha_func, beta_func,
                        epochs=10, lr=1e-3, batch_size=128,
                        verb_mode=2, seed=42):
    torch.manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(Xtrain).float(),
                       torch.from_numpy(Dtrain).float().unsqueeze(1),
                       torch.from_numpy(Ytrain).float().unsqueeze(1))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = DeepLinearNet(Xtrain.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)

    metrics = {
        "loss": [],
        "r2_alpha": [],
        "r2_beta": [],
        "mean_alpha": [],
        "mean_beta": [],
        "eps_var": []
    }

    alpha_true_full = alpha_func(Xtrain) if alpha_func else None
    beta_true_full  = beta_func(Xtrain)  if beta_func else None
    Xall_ten = torch.from_numpy(Xtrain).float()
    Dtrain_2d = Dtrain.reshape(-1,1)

    for e in range(epochs):
        running_loss=0.0
        for xx, dd, yy in dl:
            opt.zero_grad()
            ab = model(xx)
            alpha_pred = ab[:,0].unsqueeze(1)
            beta_pred  = ab[:,1].unsqueeze(1)
            yhat = alpha_pred + beta_pred*dd
            loss = torch.mean((yy - yhat)**2)
            loss.backward()
            opt.step()
            running_loss+=loss.item()

        train_loss = running_loss/len(dl)
        metrics["loss"].append(train_loss)

        with torch.no_grad():
            ab_all = model(Xall_ten)
            alpha_est = ab_all[:,0].numpy()
            beta_est  = ab_all[:,1].numpy()
            yhat_all = alpha_est + beta_est*Dtrain
            residuals = Ytrain - yhat_all
            eps_var = np.mean(residuals**2)

            if alpha_true_full is not None and len(alpha_true_full)==len(alpha_est):
                r2a = get_r2(alpha_true_full, alpha_est)
            else:
                r2a = np.nan
            if beta_true_full is not None and len(beta_true_full)==len(beta_est):
                r2b = get_r2(beta_true_full, beta_est)
            else:
                r2b = np.nan
            ma = alpha_est.mean()
            mb = beta_est.mean()

        metrics["r2_alpha"].append(r2a)
        metrics["r2_beta"].append(r2b)
        metrics["mean_alpha"].append(ma)
        metrics["mean_beta"].append(mb)
        metrics["eps_var"].append(eps_var)

        if verb_mode==2:
            print(f"epoch {e} loss={train_loss:.4f}")

    return model, metrics

def estimate_propensity(Xtrain, Dtrain, epochs=10, lr=1e-3, batch_size=128,
                        verb_mode=2, seed=999):
    torch.manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(Xtrain).float(),
                       torch.from_numpy(Dtrain).float().unsqueeze(1))
    dl = DataLoader(ds,batch_size=batch_size,shuffle=True)

    net = nn.Sequential(
        nn.Linear(Xtrain.shape[1],32),
        nn.ReLU(),
        nn.Linear(32,16),
        nn.ReLU(),
        nn.Linear(16,1)
    )
    opt = optim.Adam(net.parameters(), lr=lr)
    bceloss = nn.BCEWithLogitsLoss()

    for e in range(epochs):
        runloss=0.0
        for xx, dd in dl:
            opt.zero_grad()
            logits = net(xx)
            loss = bceloss(logits, dd)
            loss.backward()
            opt.step()
            runloss+=loss.item()
        if verb_mode==2:
            print(f"ep {e} cls loss={runloss/len(dl):.4f}")

    return net

###############################################################################
# 5. DOUBLY-ROBUST + CROSS-FIT
###############################################################################

def get_alpha_beta(model, X):
    with torch.no_grad():
        xt = torch.from_numpy(X).float()
        ab = model(xt)
        alpha_est, beta_est = ab[:,0].numpy(), ab[:,1].numpy()
    return alpha_est, beta_est

def get_phat(net, X):
    with torch.no_grad():
        xt = torch.from_numpy(X).float()
        prob = torch.sigmoid(net(xt).flatten())
    return prob.numpy()

def dr_alpha_inference(Y, D, alpha_hatX, beta_hatX, p_hatX):
    score = alpha_hatX + (1-D)/(1-p_hatX)*(Y - alpha_hatX - beta_hatX*D)
    est = np.mean(score)
    se  = np.std(score, ddof=1)/np.sqrt(len(score))
    return est, se, score

def dr_beta_inference(Y, D, alpha_hatX, beta_hatX, p_hatX):
    sc = beta_hatX + (D - p_hatX)/(p_hatX*(1-p_hatX))*(Y - alpha_hatX - beta_hatX*D)
    e  = np.mean(sc)
    s  = np.std(sc, ddof=1)/np.sqrt(len(sc))
    return e, s, sc

def cross_fit_inference(X, D, Y, alpha_func, beta_func,
                        epochs_ab=10, epochs_prop=10,
                        lr=1e-3, batch_size=128,
                        verb_mode=2, seed_offset=0):
    np.random.seed(123+seed_offset)
    n = X.shape[0]
    idx = np.random.permutation(n)
    half = n//2
    idx1, idx2 = idx[:half], idx[half:]
    X1, D1, Y1 = X[idx1], D[idx1], Y[idx1]
    X2, D2, Y2 = X[idx2], D[idx2], Y[idx2]

    # ab on fold1
    model_ab_1, metrics1 = estimate_alpha_beta(
        X1, D1, Y1, alpha_func, beta_func,
        epochs=epochs_ab, lr=lr, batch_size=batch_size,
        verb_mode=verb_mode, seed=11+seed_offset
    )
    model_p_1 = estimate_propensity(
        X1, D1, epochs=epochs_prop, lr=lr, batch_size=batch_size,
        verb_mode=verb_mode, seed=101+seed_offset
    )

    # ab on fold2
    model_ab_2, metrics2 = estimate_alpha_beta(
        X2, D2, Y2, alpha_func, beta_func,
        epochs=epochs_ab, lr=lr, batch_size=batch_size,
        verb_mode=verb_mode, seed=22+seed_offset
    )
    model_p_2 = estimate_propensity(
        X2, D2, epochs=epochs_prop, lr=lr, batch_size=batch_size,
        verb_mode=verb_mode, seed=202+seed_offset
    )

    # Evaluate on other fold
    ahat2, bhat2 = get_alpha_beta(model_ab_1, X2)
    p2           = get_phat(model_p_1, X2)
    ahat1, bhat1 = get_alpha_beta(model_ab_2, X1)
    p1           = get_phat(model_p_2, X1)

    est1_alpha, se1_alpha, sc1_a = dr_alpha_inference(Y2,D2,ahat2,bhat2,p2)
    est1_beta,  se1_beta,  sc1_b = dr_beta_inference(Y2,D2,ahat2,bhat2,p2)
    est2_alpha, se2_alpha, sc2_a = dr_alpha_inference(Y1,D1,ahat1,bhat1,p1)
    est2_beta,  se2_beta,  sc2_b = dr_beta_inference(Y1,D1,ahat1,bhat1,p1)

    alpha_hat = 0.5*(est1_alpha + est2_alpha)
    beta_hat  = 0.5*(est1_beta  + est2_beta)

    all_scores_a = np.concatenate([sc1_a,sc2_a])
    alpha_se = np.std(all_scores_a, ddof=1)/np.sqrt(n)

    all_scores_b = np.concatenate([sc1_b,sc2_b])
    beta_se = np.std(all_scores_b, ddof=1)/np.sqrt(n)

    return alpha_hat, alpha_se, beta_hat, beta_se, metrics1, metrics2

###############################################################################
# 6. PLOTTING SINGLE RUN
###############################################################################

def plot_training_metrics(metrics1, metrics2, example_name, out_dir):
    """
    2x3 plot: loss, r2(alpha), r2(beta), mean(alpha), mean(beta), eps var.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2,3,figsize=(12,8))

    def _plot(ax, data1, data2, labely, ttl):
        ep1 = np.arange(len(data1))
        ep2 = np.arange(len(data2))
        ax.plot(ep1, data1, label='fold1')
        ax.plot(ep2, data2, label='fold2')
        ax.set_title(ttl)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(labely)
        ax.legend()

    _plot(axes[0,0], metrics1["loss"],     metrics2["loss"],     "Loss",        "Training Loss")
    _plot(axes[0,1], metrics1["r2_alpha"], metrics2["r2_alpha"], "R^2 alpha",   "R^2( alpha )")
    _plot(axes[0,2], metrics1["r2_beta"],  metrics2["r2_beta"],  "R^2 beta",    "R^2( beta )")
    _plot(axes[1,0], metrics1["mean_alpha"], metrics2["mean_alpha"], "Mean alpha", "Mean[ alpha(x) ]")
    _plot(axes[1,1], metrics1["mean_beta"],  metrics2["mean_beta"],  "Mean beta",  "Mean[ beta(x) ]")
    _plot(axes[1,2], metrics1["eps_var"],     metrics2["eps_var"],    "Var",        "Fitted eps var")

    fig.suptitle(f"Training Metrics: {example_name}", fontsize=14)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"training_metrics_{example_name}.png")
    plt.savefig(fname, dpi=100)
    plt.close()

def create_single_run_table(alpha_est, alpha_se, beta_est, beta_se,
                            alpha_true, beta_true):
    """
    Table: Param, True, Est, StdErr, t=Est/SE, p, CI
    Null: param=0
    """
    def confint(e,s):
        z=1.96
        return (e - z*s, e + z*s)

    cialpha = confint(alpha_est, alpha_se)
    cibet   = confint(beta_est,  beta_se)

    t_a = alpha_est/alpha_se if abs(alpha_se)>1e-12 else 0.0
    t_b = beta_est/beta_se   if abs(beta_se)>1e-12 else 0.0
    p_a = 2*(1 - norm.cdf(abs(t_a)))
    p_b = 2*(1 - norm.cdf(abs(t_b)))

    data = [
      ["Alpha", f"{alpha_true:.3f}", f"{alpha_est:.3f}", f"{alpha_se:.3f}",
       f"{t_a:.2f}", f"{p_a:.3f}", f"{cialpha[0]:.3f}", f"{cialpha[1]:.3f}"],
      ["Beta" , f"{beta_true:.3f}",  f"{beta_est:.3f}",  f"{beta_se:.3f}",
       f"{t_b:.2f}", f"{p_b:.3f}", f"{cibet[0]:.3f}", f"{cibet[1]:.3f}"]
    ]
    headers = ["Param","True","Est","StdErr","t(= est/se)","p","CI_Low","CI_High"]
    return tabulate(data, headers, tablefmt="github")

###############################################################################
# 7. RUN SINGLE EXAMPLE
###############################################################################

def run_example_single(X, D, Y, alpha_func, beta_func,
                       example_name="Example",
                       epochs_ab=10, epochs_prop=10, lr=1e-3, batch_size=128,
                       verb_mode=2, out_dir="data/results"):
    print_data_overview(X, D, Y, label=example_name)
    alpha_true_val = alpha_func(X).mean()
    beta_true_val  = beta_func(X).mean()

    alpha_est, alpha_se, beta_est, beta_se, m1, m2 = cross_fit_inference(
        X, D, Y, alpha_func, beta_func,
        epochs_ab=epochs_ab, epochs_prop=epochs_prop,
        lr=lr, batch_size=batch_size,
        verb_mode=verb_mode
    )
    plot_training_metrics(m1, m2, example_name, out_dir)

    table_str = create_single_run_table(alpha_est, alpha_se, beta_est, beta_se,
                                        alpha_true_val, beta_true_val)
    if verb_mode>=1:
        print(f"\nResults for {example_name}, testing param=0, but also showing true param:")
        print(table_str)

###############################################################################
# 8. MONTE CARLO
###############################################################################

def approximate_true_mean_ab(alpha_func, beta_func, d=5, seed=9999, nLarge=200000):
    """
    We'll assume X ~ N(0,1,d).  If you want the exact same distribution as the example,
    that's typically the case for these examples.  This is a quick approximate method.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0,1,(nLarge,d))
    A = alpha_func(X)
    B = beta_func(X)
    return A.mean(), B.mean()

def monte_carlo_simulation(gen_data_fn, alpha_func, beta_func,
                           label="MC_Study", out_dir="data/results",
                           nrep=30, verb_mode=0, d=5,
                           mc_epochs_ab=5, mc_epochs_prop=5):
    alphaT, betaT = approximate_true_mean_ab(alpha_func, beta_func, d=d, seed=9999)
    if verb_mode>=1:
        print(f"\nMonte Carlo {label}, True alpha={alphaT:.3f}, beta={betaT:.3f}")

    alpha_ests = []
    beta_ests  = []
    for i in range(nrep):
        X, D, Y, _, _ = gen_data_fn()
        a_est, a_se, b_est, b_se, _, _ = cross_fit_inference(
            X, D, Y, alpha_func, beta_func,
            epochs_ab=mc_epochs_ab, epochs_prop=mc_epochs_prop,
            lr=1e-3, batch_size=256,
            verb_mode=0, seed_offset=i
        )
        alpha_ests.append(a_est)
        beta_ests.append(b_est)

    alpha_ests = np.array(alpha_ests)
    beta_ests  = np.array(beta_ests)

    # KDE
    sns.set_style("whitegrid")
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    sns.kdeplot(alpha_ests, shade=True, color='gray', ax=axs[0])
    axs[0].axvline(alphaT, color='red', linestyle='--', label=f"True={alphaT:.3f}")
    axs[0].set_title(f"{label}: alpha")
    axs[0].legend()

    sns.kdeplot(beta_ests, shade=True, color='gray', ax=axs[1])
    axs[1].axvline(betaT, color='red', linestyle='--', label=f"True={betaT:.3f}")
    axs[1].set_title(f"{label}: beta")
    axs[1].legend()

    plt.tight_layout()
    fname = os.path.join(out_dir, f"monte_carlo_kde_{label}.png")
    plt.savefig(fname,dpi=100)
    plt.close()

    # Summaries
    meanA = alpha_ests.mean()
    stdA  = alpha_ests.std(ddof=1)
    rmseA = np.sqrt(np.mean((alpha_ests - alphaT)**2))

    meanB = beta_ests.mean()
    stdB  = beta_ests.std(ddof=1)
    rmseB = np.sqrt(np.mean((beta_ests - betaT)**2))

    data = [
      ["Alpha", f"{alphaT:.3f}", f"{meanA:.3f}", f"{stdA:.3f}", f"{rmseA:.3f}"],
      ["Beta" , f"{betaT:.3f}",  f"{meanB:.3f}", f"{stdB:.3f}", f"{rmseB:.3f}"]
    ]
    headers = ["Param","True","Mean(Est)","Std(Est)","RMSE"]
    table_str = tabulate(data, headers, tablefmt="github")
    if verb_mode>=1:
        print(f"\nMC {label} summary table:")
        print(table_str)

###############################################################################
# 9. MAIN
###############################################################################

def main(verb_mode=2, folder_name="results"):
    """
    1) Single-run for Example1, Example2 with user-set hyperparams
    2) Monte Carlo for each example with user-set hyperparams
    All outputs -> data/<folder_name>/...

    Adjust hyperparameters here for single-run & MC.
    """
    # SINGLE-RUN HYPERPARAMS
    single_n = 10000       # sample size for single example
    single_epochs_ab = 100  # epochs for alpha,beta net
    single_epochs_prop = 100  # epochs for p net
    single_lr = 1e-3
    single_batch_size = 256

    # MONTE CARLO HYPERPARAMS
    mc_n = 10_000          # sample size per replicate
    mc_reps = 20         # number of Monte Carlo replications
    mc_epochs_ab = 5     # epochs for alpha,beta net
    mc_epochs_prop = 5   # epochs for p net
    mc_lr = 1e-3
    mc_batch_size = 256

    out_dir = os.path.join("data", folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # capture stdout
    orig_stdout = sys.stdout
    log_path = os.path.join(out_dir, "stdout.log")
    log_file = open(log_path, "w")
    sys.stdout = log_file

    # Example1 single run
    X1, D1, Y1, aF1, bF1 = generate_data_example1(n=single_n,d=5,seed=123)
    run_example_single(X1, D1, Y1, aF1, bF1,
                       example_name="Example1",
                       epochs_ab=single_epochs_ab,
                       epochs_prop=single_epochs_prop,
                       lr=single_lr,
                       batch_size=single_batch_size,
                       verb_mode=verb_mode,
                       out_dir=out_dir)

    # Example2 single run
    X2, D2, Y2, aF2, bF2 = generate_data_example2(n=single_n,d=5,seed=999)
    run_example_single(X2, D2, Y2, aF2, bF2,
                       example_name="Example2",
                       epochs_ab=single_epochs_ab,
                       epochs_prop=single_epochs_prop,
                       lr=single_lr,
                       batch_size=single_batch_size,
                       verb_mode=verb_mode,
                       out_dir=out_dir)

    # Monte Carlo for Example1
    if verb_mode>=1:
        print("\n=== Monte Carlo for Example1 ===")
    def gen_ex1():
        return generate_data_example1(n=mc_n, d=5, seed=np.random.randint(999999))
    monte_carlo_simulation(
        gen_ex1, aF1, bF1,
        label="Example1_MC",
        out_dir=out_dir,
        nrep=mc_reps,
        verb_mode=verb_mode,
        d=5,
        mc_epochs_ab=mc_epochs_ab,
        mc_epochs_prop=mc_epochs_prop
    )

    # Monte Carlo for Example2
    if verb_mode>=1:
        print("\n=== Monte Carlo for Example2 ===")
    def gen_ex2():
        return generate_data_example2(n=mc_n, d=5, seed=np.random.randint(999999))
    monte_carlo_simulation(
        gen_ex2, aF2, bF2,
        label="Example2_MC",
        out_dir=out_dir,
        nrep=mc_reps,
        verb_mode=verb_mode,
        d=5,
        mc_epochs_ab=mc_epochs_ab,
        mc_epochs_prop=mc_epochs_prop
    )

    sys.stdout = orig_stdout
    log_file.close()
    if verb_mode>=1:
        print(f"\nAll logs & plots saved in '{out_dir}'. See stdout.log for details.")


if __name__=="__main__":
    # example usage
    main(verb_mode=2, folder_name="linear_logit")

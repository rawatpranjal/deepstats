import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

###############################################################################
# 1. DATA GENERATION
###############################################################################

def generate_data_example1(n=20000, d=5, seed=123):
    """
    Example1: Y = alpha(X) + beta(X)*D + eps
    alpha(X) = 1 + 0.5*X0 -0.2*X1
    beta(X)  = -1 + 0.3*X2 + 0.4*X0*X1
    D ~ Bernoulli( p=logistic(0.5*X0) )
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
    betaX = beta_func(X)
    eps = 0.5*rng.normal(0,1,n)
    Y = alphaX + betaX*D + eps
    return X, D, Y, alpha_func, beta_func

def generate_data_example2(n=20000, d=5, seed=999):
    """
    Example2: Y = alpha(X) + beta(X)*D + eps
    alpha(X) = 2.0 + 0.8*X1^2
    beta(X)  = 0.3*X0 - 0.3*X2 + 0.1*X3^3
    D ~ Bernoulli( p=logistic(-0.3*X1) )
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
    betaX = beta_func(X)
    eps = 0.7*rng.normal(0,1,n)
    Y = alphaX + betaX*D + eps
    return X, D, Y, alpha_func, beta_func

###############################################################################
# 2. NEURAL NETS FOR alpha(X), beta(X) & PROPENSITY
###############################################################################

class DeepLinearNet(nn.Module):
    """
    Simple feed-forward net to learn [alpha(x), beta(x)] in final output.
    """
    def __init__(self, d):
        super().__init__()
        self.hidden1 = nn.Linear(d, 100)
        self.hidden2 = nn.Linear(100, 64)
        self.out = nn.Linear(64, 2)
    def forward(self, x):
        z = torch.relu(self.hidden1(x))
        z = torch.relu(self.hidden2(z))
        ab = self.out(z)
        # ab[:,0] = alpha(x), ab[:,1] = beta(x)
        return ab

def get_r2(true_vals, pred_vals):
    """
    Compute R^2 = 1 - RSS/TSS for vector of predictions vs. ground truth.
    """
    ss_res = np.sum((true_vals - pred_vals)**2)
    ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
    if ss_tot<1e-12:
        return 1.0
    return 1 - ss_res/ss_tot

def estimate_alpha_beta(Xtrain, Dtrain, Ytrain, alpha_true, beta_true,
                        epochs=10, lr=1e-3, batch_size=128, verb_mode=2, seed=42):
    """
    Train a net that outputs [alpha(x), beta(x)].
    Track: loss, r2(alpha), r2(beta), E[alpha], E[beta], fitted eps variance.
    Return model + dict of metrics (lists over epochs).
    """
    torch.manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(Xtrain).float(),
                       torch.from_numpy(Dtrain).float().unsqueeze(1),
                       torch.from_numpy(Ytrain).float().unsqueeze(1))
    dl = DataLoader(ds,batch_size=batch_size,shuffle=True)

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

    Xall_ten = torch.from_numpy(Xtrain).float()
    Dall_ten = torch.from_numpy(Dtrain).float().unsqueeze(1)
    Yall_ten = torch.from_numpy(Ytrain).float().unsqueeze(1)

    N = Xtrain.shape[0]
    alpha_true_full = alpha_true(Xtrain) if alpha_true else None
    beta_true_full = beta_true(Xtrain) if beta_true else None

    for e in range(epochs):
        running_loss=0.0
        for xx, dd, yy in dl:
            opt.zero_grad()
            out = model(xx)
            alpha_pred = out[:,0].unsqueeze(1)
            beta_pred  = out[:,1].unsqueeze(1)
            yhat = alpha_pred + beta_pred*dd
            loss = torch.mean((yy - yhat)**2)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        train_loss = running_loss/len(dl)

        # evaluate on entire training set
        with torch.no_grad():
            ab_all = model(Xall_ten)
            alpha_est = ab_all[:,0].numpy()
            beta_est  = ab_all[:,1].numpy()
            yhat_all = alpha_est + beta_est*Dtrain
            residuals = Ytrain - yhat_all
            eps_var = np.mean(residuals**2)

            # r2
            if alpha_true_full is not None:
                r2a = get_r2(alpha_true_full, alpha_est)
            else:
                r2a = np.nan
            if beta_true_full is not None:
                r2b = get_r2(beta_true_full, beta_est)
            else:
                r2b = np.nan

            ma = np.mean(alpha_est)
            mb = np.mean(beta_est)

        metrics["loss"].append(train_loss)
        metrics["r2_alpha"].append(r2a)
        metrics["r2_beta"].append(r2b)
        metrics["mean_alpha"].append(ma)
        metrics["mean_beta"].append(mb)
        metrics["eps_var"].append(eps_var)

        if verb_mode==2:
            print(f"epoch {e} loss {round(train_loss,4)}")

    return model, metrics

def estimate_propensity(Xtrain, Dtrain, epochs=10, lr=1e-3, batch_size=128, verb_mode=2, seed=999):
    """
    Fit p(x)=P(D=1|X=x) via logistic net. Return model.
    """
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
            print(f"ep {e} cls loss {round(runloss/len(dl),4)}")

    return net

###############################################################################
# 3. INFERENCE
###############################################################################

def get_alpha_beta(model, X):
    with torch.no_grad():
        xt = torch.from_numpy(X).float()
        ab = model(xt)
        a, b = ab[:,0].numpy(), ab[:,1].numpy()
    return a, b

def get_phat(net, X):
    with torch.no_grad():
        xt = torch.from_numpy(X).float()
        logit = net(xt).flatten()
        prob = torch.sigmoid(logit)
    return prob.numpy()

def dr_alpha_inference(Y, D, alpha_hatX, beta_hatX, p_hatX):
    """
    Y = alpha + beta*D + eps
    DR score for E[alpha(X)]:
      alpha_hat + (1-D)/(1-p) * (Y - alpha_hat - beta_hat*D).
    """
    score = alpha_hatX + (1-D)/(1-p_hatX)*(Y - alpha_hatX - beta_hatX*D)
    est = np.mean(score)
    se = np.std(score, ddof=1)/np.sqrt(len(score))
    return est, se, score

def dr_beta_inference(Y, D, alpha_hatX, beta_hatX, p_hatX):
    """
    DR score for E[beta(X)]:
      beta_hat + [ (D - p)/(p*(1-p)) ] * (Y - alpha_hat - beta_hat*D ).
    """
    sc = beta_hatX + (D - p_hatX)/(p_hatX*(1-p_hatX))*(Y - alpha_hatX - beta_hatX*D)
    e = np.mean(sc)
    s = np.std(sc, ddof=1)/np.sqrt(len(sc))
    return e, s, sc

###############################################################################
# 4. CROSS-FIT + METRICS + PLOTTING + TABLE
###############################################################################

def cross_fit_inference(X, D, Y, alpha_true, beta_true,
                        epochs1=10, epochs2=10, lr=1e-3, batch_size=128,
                        verb_mode=2):
    """
    Cross-fitting for nuisance. Then DR inference on E[alpha(X)], E[beta(X)].
    Return final estimates + stderrs + (model metrics).
    If alpha_true/beta_true not None, we track R^2, etc.
    """
    n = X.shape[0]
    idx = np.random.permutation(n)
    half = n//2
    idx1, idx2 = idx[:half], idx[half:]
    X1, X2 = X[idx1], X[idx2]
    D1, D2 = D[idx1], D[idx2]
    Y1, Y2 = Y[idx1], Y[idx2]

    aTrue1 = alpha_true(X1) if alpha_true else None
    bTrue1 = beta_true(X1) if beta_true else None
    aTrue2 = alpha_true(X2) if alpha_true else None
    bTrue2 = beta_true(X2) if beta_true else None

    # model ab on fold1
    model_ab_1, metrics1 = estimate_alpha_beta(
        X1, D1, Y1, aTrue1, bTrue1,
        epochs=epochs1, lr=lr, batch_size=batch_size, verb_mode=verb_mode, seed=11
    )
    # model ab on fold2
    model_ab_2, metrics2 = estimate_alpha_beta(
        X2, D2, Y2, aTrue2, bTrue2,
        epochs=epochs1, lr=lr, batch_size=batch_size, verb_mode=verb_mode, seed=22
    )
    # model p on fold1
    model_p_1 = estimate_propensity(X1, D1, epochs=epochs2, lr=lr, batch_size=batch_size, verb_mode=verb_mode, seed=101)
    # model p on fold2
    model_p_2 = estimate_propensity(X2, D2, epochs=epochs2, lr=lr, batch_size=batch_size, verb_mode=verb_mode, seed=202)

    # predictions on other fold
    ahat2, bhat2 = get_alpha_beta(model_ab_1, X2)
    p2 = get_phat(model_p_1, X2)
    ahat1, bhat1 = get_alpha_beta(model_ab_2, X1)
    p1 = get_phat(model_p_2, X1)

    # DR inference
    est1_alpha, se1_alpha, scores1_alpha = dr_alpha_inference(Y2, D2, ahat2, bhat2, p2)
    est1_beta,  se1_beta,  scores1_beta  = dr_beta_inference(Y2, D2, ahat2, bhat2, p2)
    est2_alpha, se2_alpha, scores2_alpha = dr_alpha_inference(Y1, D1, ahat1, bhat1, p1)
    est2_beta,  se2_beta,  scores2_beta  = dr_beta_inference(Y1, D1, ahat1, bhat1, p1)

    alpha_hat = 0.5*(est1_alpha + est2_alpha)
    beta_hat  = 0.5*(est1_beta + est2_beta)
    # stderrs
    all_scores_alpha = np.concatenate([scores1_alpha,scores2_alpha])
    all_scores_beta  = np.concatenate([scores1_beta,scores2_beta])
    alpha_se = np.std(all_scores_alpha,ddof=1)/np.sqrt(n)
    beta_se  = np.std(all_scores_beta,ddof=1)/np.sqrt(n)

    # combine metrics from both folds
    # for brevity, we just return them in a dict
    return alpha_hat, alpha_se, beta_hat, beta_se, metrics1, metrics2

def plot_training_metrics(metrics1, metrics2, example_name="Example"):
    """
    Plot in a 2x3 grid: loss, r2(alpha), r2(beta), mean alpha, mean beta, eps var.
    Each from fold1 and fold2 in same subplot.
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2,3,figsize=(12,8))
    # row0: loss, r2a, r2b; row1: mean_alpha, mean_beta, eps_var
    # combine
    def _pl(ax, data1, data2, ylabel, ttl):
        ep1 = np.arange(len(data1))
        ep2 = np.arange(len(data2))
        ax.plot(ep1, data1, label='fold1')
        ax.plot(ep2, data2, label='fold2')
        ax.set_title(ttl)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    _pl(axes[0,0], metrics1["loss"], metrics2["loss"], "Loss", "Training Loss")
    _pl(axes[0,1], metrics1["r2_alpha"], metrics2["r2_alpha"], "R2", "R2 of alpha")
    _pl(axes[0,2], metrics1["r2_beta"], metrics2["r2_beta"], "R2", "R2 of beta")
    _pl(axes[1,0], metrics1["mean_alpha"], metrics2["mean_alpha"], "Mean alpha", "Running E[alpha]")
    _pl(axes[1,1], metrics1["mean_beta"], metrics2["mean_beta"], "Mean beta",  "Running E[beta]")
    _pl(axes[1,2], metrics1["eps_var"],    metrics2["eps_var"],    "Var",        "Fitted epsilon var")

    fig.suptitle(f"Training Metrics: {example_name}", fontsize=14)
    plt.tight_layout()
    fname = f"training_metrics_{example_name}.png"
    plt.savefig(fname,dpi=100)
    plt.close()

def create_results_table(true_alpha, est_alpha, se_alpha,
                         true_beta,  est_beta,  se_beta):
    """
    Use tabulate to create a table with columns:
      Param, TrueVal, EstVal, StdErr, t, p, 95%CI low, 95%CI high
    """
    def confint(e, s):
        z = 1.96
        return (e - z*s, e + z*s)
    cialpha = confint(est_alpha, se_alpha)
    cibet   = confint(est_beta, se_beta)

    # t, p for alpha
    talpha = (est_alpha - true_alpha)/se_alpha if se_alpha>1e-12 else 0
    tbeta  = (est_beta - true_beta)/se_beta   if se_beta>1e-12 else 0

    from scipy.stats import norm
    palpha = 2*(1 - norm.cdf(abs(talpha)))
    pbeta  = 2*(1 - norm.cdf(abs(tbeta)))

    data = [
      ["Alpha", f"{true_alpha:.2f}", f"{est_alpha:.2f}", f"{se_alpha:.2f}",
       f"{talpha:.2f}", f"{palpha:.3f}",
       f"{cialpha[0]:.2f}", f"{cialpha[1]:.2f}"],
      ["Beta" , f"{true_beta:.2f}" , f"{est_beta:.2f}" , f"{se_beta:.2f}" ,
       f"{tbeta:.2f}" , f"{pbeta:.3f}" ,
       f"{cibet[0]:.2f}", f"{cibet[1]:.2f}"]
    ]
    headers = ["Param","True","Est","StdErr","t","p","CI_Low","CI_High"]
    return tabulate(data, headers, tablefmt="github")

###############################################################################
# 5. MAIN
###############################################################################

def run_example(X, D, Y, alpha_func, beta_func,
                example_name="Example", 
                epochs_ab=10, epochs_prop=10, lr=1e-3, batch_size=128,
                verb_mode=2):
    """
    1) Cross-fit
    2) Plot training metrics
    3) Table of results
    """
    # True E[alpha(X)] and E[beta(X)]:
    trueAlpha = np.mean(alpha_func(X))
    trueBeta  = np.mean(beta_func(X))

    # cross fit
    alpha_est, alpha_se, beta_est, beta_se, m1, m2 = cross_fit_inference(
        X, D, Y, alpha_func, beta_func,
        epochs1=epochs_ab, epochs2=epochs_prop,
        lr=lr, batch_size=batch_size, verb_mode=verb_mode
    )

    # final table
    table_str = create_results_table(trueAlpha, alpha_est, alpha_se,
                                     trueBeta,  beta_est,  beta_se)
    # optional prints
    if verb_mode>=1:
        print(f"\n\nResults for {example_name}")
        print(table_str)

    # plot training metrics
    plot_training_metrics(m1, m2, example_name=example_name)

def main(verb_mode=2):
    """
    Run two examples with expanded tracking of metrics, final table, etc.
    verb_mode=2 -> full prints
    verb_mode=1 -> only table + final results
    verb_mode=0 -> silent
    """
    # Example1
    X1, D1, Y1, aF1, bF1 = generate_data_example1(n=20000,d=5,seed=123)
    run_example(X1, D1, Y1, aF1, bF1,
                example_name="Example1",
                epochs_ab=10, epochs_prop=10, lr=1e-3, batch_size=256,
                verb_mode=verb_mode)

    # Example2
    X2, D2, Y2, aF2, bF2 = generate_data_example2(n=20000,d=5,seed=999)
    run_example(X2, D2, Y2, aF2, bF2,
                example_name="Example2",
                epochs_ab=10, epochs_prop=10, lr=1e-3, batch_size=256,
                verb_mode=verb_mode)

if __name__=="__main__":
    # Adjust verb_mode as desired
    main(verb_mode=2)

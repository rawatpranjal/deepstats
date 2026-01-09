#!/usr/bin/env Rscript
#
# Run R implementation of FLM2 inference on shared test data.
#
# This implements the core FLM algorithm:
# 1. K-fold cross-fitting with neural networks
# 2. Influence function correction
# 3. Within-fold variance SE estimation
#

library(torch)
library(jsonlite)

# Configuration
K_FOLDS <- 50
EPOCHS <- 100
LR <- 0.01
HIDDEN_DIMS <- c(64, 32)
BATCH_SIZE <- 64

# Set seed for reproducibility
set.seed(42)
torch_manual_seed(42)

#' Create a simple MLP for structural estimation
create_structural_net <- function(input_dim, hidden_dims = c(64, 32)) {
  nn_module(
    initialize = function() {
      self$fc1 <- nn_linear(input_dim, hidden_dims[1])
      self$fc2 <- nn_linear(hidden_dims[1], hidden_dims[2])
      self$out <- nn_linear(hidden_dims[2], 2)  # [alpha, beta]
    },
    forward = function(x) {
      x %>%
        self$fc1() %>%
        nnf_relu() %>%
        self$fc2() %>%
        nnf_relu() %>%
        self$out()
    }
  )
}

#' Train structural model on data
train_structural <- function(X, T, Y, epochs = 100, lr = 0.01) {
  n <- nrow(X)
  input_dim <- ncol(X)

  # Convert to tensors
  X_t <- torch_tensor(X, dtype = torch_float())
  T_t <- torch_tensor(T, dtype = torch_float())
  Y_t <- torch_tensor(Y, dtype = torch_float())

  # Create model
  model <- create_structural_net(input_dim, HIDDEN_DIMS)()
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = 1e-4)

  # Training loop
  for (epoch in 1:epochs) {
    model$train()
    optimizer$zero_grad()

    # Forward pass
    theta <- model(X_t)
    alpha <- theta[, 1]
    beta <- theta[, 2]

    # Linear family loss: (Y - alpha - beta*T)^2
    mu <- alpha + beta * T_t
    loss <- torch_mean((Y_t - mu)^2)

    # Backward pass
    loss$backward()
    optimizer$step()
  }

  model$eval()
  return(model)
}

#' Compute Hessian Lambda for linear family
compute_hessian <- function(T_vec, n) {
  # For linear family: Lambda = E[[1,T] @ [1,T]'] = [[1, E[T]], [E[T], E[T^2]]]
  T_mean <- mean(T_vec)
  T2_mean <- mean(T_vec^2)

  Lambda <- matrix(c(1, T_mean, T_mean, T2_mean), nrow = 2)

  # Add ridge for stability
  Lambda <- Lambda + 1e-4 * diag(2)

  return(Lambda)
}

#' Compute influence scores for linear family
compute_influence_scores <- function(Y, T, theta, Lambda_inv) {
  n <- length(Y)
  alpha <- as.numeric(theta[, 1])
  beta <- as.numeric(theta[, 2])

  # Residuals
  mu <- alpha + beta * T
  r <- Y - mu

  # Design matrix [1, T]
  T_design <- cbind(rep(1, n), T)

  # Score: l_theta = -r * [1, T]
  l_theta <- -r * T_design

  # Gradient of H(theta) = beta: H_grad = [0, 1]
  H_grad <- c(0, 1)

  # Influence score: psi = beta - l_theta @ Lambda_inv @ H_grad
  correction <- l_theta %*% Lambda_inv %*% H_grad
  psi <- beta - as.numeric(correction)

  return(list(psi = psi, correction = correction, beta = beta))
}

#' Run K-fold cross-fitting inference
run_influence_inference <- function(X, T, Y, K = 50) {
  n <- nrow(X)
  psi_all <- numeric(n)
  beta_all <- numeric(n)
  fold_indices <- sample(rep(1:K, length.out = n))

  for (k in 1:K) {
    train_idx <- which(fold_indices != k)
    eval_idx <- which(fold_indices == k)

    X_train <- X[train_idx, , drop = FALSE]
    T_train <- T[train_idx]
    Y_train <- Y[train_idx]

    X_eval <- X[eval_idx, , drop = FALSE]
    T_eval <- T[eval_idx]
    Y_eval <- Y[eval_idx]

    # Train model on training fold
    model <- train_structural(X_train, T_train, Y_train, epochs = EPOCHS, lr = LR)

    # Compute Hessian from training data
    Lambda <- compute_hessian(T_train, length(T_train))
    Lambda_inv <- solve(Lambda)

    # Get predictions on eval fold
    X_eval_t <- torch_tensor(X_eval, dtype = torch_float())
    with_no_grad({
      theta_eval <- as.matrix(model(X_eval_t))
    })

    # Compute influence scores
    result <- compute_influence_scores(Y_eval, T_eval, theta_eval, Lambda_inv)

    psi_all[eval_idx] <- result$psi
    beta_all[eval_idx] <- result$beta
  }

  # Final inference
  mu_hat <- mean(psi_all)

  # Within-fold variance SE
  variance_sum <- 0
  for (k in 1:K) {
    psi_k <- psi_all[fold_indices == k]
    if (length(psi_k) > 0) {
      mu_k <- mean(psi_k)
      variance_k <- mean((psi_k - mu_k)^2)
      variance_sum <- variance_sum + variance_k
    }
  }
  Psi_hat <- variance_sum / K
  se <- sqrt(Psi_hat / n)

  return(list(
    mu_hat = mu_hat,
    se = se,
    psi = psi_all,
    beta = beta_all
  ))
}

#' Run naive inference (no IF correction)
run_naive_inference <- function(X, T, Y) {
  n <- nrow(X)

  # Train on full data
  model <- train_structural(X, T, Y, epochs = EPOCHS, lr = LR)

  # Get predictions
  X_t <- torch_tensor(X, dtype = torch_float())
  with_no_grad({
    theta <- as.matrix(model(X_t))
  })

  beta <- theta[, 2]
  mu_hat <- mean(beta)
  se <- sd(beta) / sqrt(n)

  return(list(mu_hat = mu_hat, se = se, beta = beta))
}

#' Run single simulation
run_single_sim <- function(sim_id, data_dir) {
  # Load data
  X <- as.matrix(read.csv(file.path(data_dir, sprintf("X_%d.csv", sim_id)), header = FALSE))
  T <- as.numeric(read.csv(file.path(data_dir, sprintf("T_%d.csv", sim_id)), header = FALSE)[, 1])
  Y <- as.numeric(read.csv(file.path(data_dir, sprintf("Y_%d.csv", sim_id)), header = FALSE)[, 1])
  beta_star <- as.numeric(read.csv(file.path(data_dir, sprintf("beta_star_%d.csv", sim_id)), header = FALSE)[, 1])
  mu_true <- mean(beta_star)

  # Set seed for this simulation
  set.seed(42 + sim_id)
  torch_manual_seed(42 + sim_id)

  # Run influence function inference
  result_if <- run_influence_inference(X, T, Y, K = K_FOLDS)

  # Run naive inference
  result_naive <- run_naive_inference(X, T, Y)

  # Check coverage
  ci_lower <- result_if$mu_hat - 1.96 * result_if$se
  ci_upper <- result_if$mu_hat + 1.96 * result_if$se
  if_covered <- (ci_lower <= mu_true) && (mu_true <= ci_upper)

  naive_ci_lower <- result_naive$mu_hat - 1.96 * result_naive$se
  naive_ci_upper <- result_naive$mu_hat + 1.96 * result_naive$se
  naive_covered <- (naive_ci_lower <= mu_true) && (mu_true <= naive_ci_upper)

  return(list(
    sim_id = sim_id,
    mu_true = mu_true,
    if_mu = result_if$mu_hat,
    if_se = result_if$se,
    if_covered = if_covered,
    naive_mu = result_naive$mu_hat,
    naive_se = result_naive$se,
    naive_covered = naive_covered
  ))
}

# Main
main <- function() {
  script_dir <- dirname(sys.frame(1)$ofile)
  if (is.null(script_dir) || script_dir == "") {
    script_dir <- "."
  }

  data_dir <- file.path(script_dir, "data")
  output_dir <- file.path(script_dir, "results")
  dir.create(output_dir, showWarnings = FALSE)

  # Load metadata
  metadata <- fromJSON(file.path(data_dir, "metadata.json"))
  M <- metadata$M

  cat("============================================================\n")
  cat("Running R (FLM2-style) Pipeline\n")
  cat("============================================================\n")
  cat(sprintf("M = %d simulations\n", M))
  cat(sprintf("N = %d, d = %d\n", metadata$N, metadata$d))
  cat("\n")

  results <- list()
  for (sim_id in 0:(M-1)) {
    cat(sprintf("R: Simulation %d/%d\r", sim_id + 1, M))
    result <- run_single_sim(sim_id, data_dir)
    results[[sim_id + 1]] <- result
  }
  cat("\n")

  # Aggregate results
  if_covered <- sapply(results, function(r) r$if_covered)
  naive_covered <- sapply(results, function(r) r$naive_covered)
  if_mu <- sapply(results, function(r) r$if_mu)
  naive_mu <- sapply(results, function(r) r$naive_mu)
  if_se <- sapply(results, function(r) r$if_se)
  naive_se <- sapply(results, function(r) r$naive_se)
  mu_true <- sapply(results, function(r) r$mu_true)

  if_coverage <- mean(if_covered)
  naive_coverage <- mean(naive_covered)

  if_se_emp <- sd(if_mu)
  if_se_est <- mean(if_se)
  if_ratio <- if_se_est / if_se_emp

  naive_se_emp <- sd(naive_mu)
  naive_se_est <- mean(naive_se)
  naive_ratio <- naive_se_est / naive_se_emp

  if_bias <- mean(if_mu - mu_true)
  naive_bias <- mean(naive_mu - mu_true)

  summary <- list(
    implementation = "r_flm2",
    M = M,
    N = metadata$N,
    d = metadata$d,
    if_coverage = if_coverage,
    if_se_empirical = if_se_emp,
    if_se_estimated = if_se_est,
    if_se_ratio = if_ratio,
    if_bias = if_bias,
    naive_coverage = naive_coverage,
    naive_se_empirical = naive_se_emp,
    naive_se_estimated = naive_se_est,
    naive_se_ratio = naive_ratio,
    naive_bias = naive_bias,
    raw_results = results
  )

  # Save results
  write_json(summary, file.path(output_dir, "r_results.json"), pretty = TRUE, auto_unbox = TRUE)

  cat("\n")
  cat("============================================================\n")
  cat("R Results\n")
  cat("============================================================\n")
  cat(sprintf("%20s %12s %12s\n", "", "IF", "Naive"))
  cat(strrep("-", 44), "\n")
  cat(sprintf("%20s %11.1f%% %11.1f%%\n", "Coverage", if_coverage * 100, naive_coverage * 100))
  cat(sprintf("%20s %12.4f %12.4f\n", "SE (empirical)", if_se_emp, naive_se_emp))
  cat(sprintf("%20s %12.4f %12.4f\n", "SE (estimated)", if_se_est, naive_se_est))
  cat(sprintf("%20s %12.2f %12.2f\n", "SE Ratio", if_ratio, naive_ratio))
  cat(sprintf("%20s %12.4f %12.4f\n", "Bias", if_bias, naive_bias))
  cat("\n")
  cat(sprintf("Results saved to: %s\n", file.path(output_dir, "r_results.json")))
}

# Run main
main()

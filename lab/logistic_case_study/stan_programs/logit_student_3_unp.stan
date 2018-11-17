// Stan program thesis --> QR decomposition

data {

  int<lower = 1> N;                        // Number of observations
  int<lower = 1> K;                        // Number of confounders
  matrix[N, K] X;                          // Confounders
  int<lower = 0, upper = 1> Y[N];          // Outcome

}

parameters {

  real alpha;           // Intercept
  vector[K] beta;       // Coefficients confounders

}

model {

  // Linear predictor
  vector[N] eta = alpha + X * beta;

  // Priors: stan uses uniform priors by default
  target += cauchy_lpdf(alpha |0, 10) +
            student_t_lpdf(beta[1:K-1] |3, 0, 1) +
            student_t_lpdf(beta[K] |3, 0, 10);

  // Likelihood
  target += bernoulli_logit_lpmf(Y | eta);

}

generated quantities {

  // OR, predicted probabilities, log_likelihood and replicated outcome
  // to evaluate the fit of the model
  vector[N] pred_probs;
  vector[N] log_lik;
  int y_rep[N];

  // Simulate replicated data from the fitted model
  for (n in 1:N) {

    // Linear predictor
    real eta_rep = alpha + X[n] * beta;

    // Predicted probabilities
    pred_probs[n] = inv_logit(eta_rep);

    // Log likelihood for model comparison with LOO
    log_lik[n] = bernoulli_logit_lpmf(Y[n] | eta_rep);

    // Replicated outcome
    y_rep[n] = bernoulli_logit_rng(eta_rep);

  }

}


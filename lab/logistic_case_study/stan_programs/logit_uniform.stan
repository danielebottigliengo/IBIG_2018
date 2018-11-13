// Stan program thesis --> QR decomposition

data {

  int<lower = 1> N;                     // Number of observations
  int<lower = 1> K;                     // Number of covariates
  matrix[N, K] X;                       // Matrix design
  int<lower = 0, upper = 1> Y[N];       // Outcome

}

parameters {

  real alpha;       // Intercept
  vector[K] beta;   // Coefficients

}

model {

  // Linear predictor
  vector[N] eta = alpha + X * beta;

  // Priors: stan uses uniform priors by default
  // target += uniform_lpdf(alpha | -10, 10) +
  //           uniform_lpdf(beta | -10, 10);

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
    real eta_rep = X[n] * beta + alpha;

    // Predicted probabilities
    pred_probs[n] = inv_logit(eta_rep);

    // Log likelihood for model comparison with LOO
    log_lik[n] = bernoulli_logit_lpmf(Y[n] | eta_rep);

    // Replicated outcome
    y_rep[n] = bernoulli_logit_rng(eta_rep);

  }

}


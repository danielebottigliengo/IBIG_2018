// Birth weight: regularizing priors

data {

  int<lower = 1> N;                     // Number of observations
  int<lower = 1> K;                     // Number of covariates
  matrix[N, K] X;                       // Matrix design
  vector<lower = 0>[N] Y;               // Outcome

}

parameters {

  real alpha;                  // Intercept
  vector[K] beta;              // Coefficients
  real<lower = 0> sigma;       // Sd

}

model {

  // Linear predictor
  vector[N] eta = alpha + X * beta;

  // Priors: stan uses uniform priors by default
  target += normal_lpdf(alpha |0, 10) +
            normal_lpdf(beta |0, 1) +
            cauchy_lpdf(sigma |0, 2.5);

  // Likelihood
  target += normal_lpdf(Y | eta, sigma);

}

generated quantities {

  vector[N] log_lik;
  vector[N] y_rep;

  // Simulate replicated data from the fitted model
  for (n in 1:N) {

    // Linear predictor
    real eta_rep = X[n] * beta + alpha;

    // Log likelihood for model comparison with LOO
    log_lik[n] = normal_lpdf(Y[n] | eta_rep, sigma);

    // Replicated outcome
    y_rep[n] = normal_rng(eta_rep, sigma);

  }

}


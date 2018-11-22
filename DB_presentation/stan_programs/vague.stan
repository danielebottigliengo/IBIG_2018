// Birth weight: vague priors (with truncation at 0)

// Random number generator from a lower-bound normal distribution
functions {
  real normal_lb_rng(real mu, real sigma, real lb) {

    real p = normal_cdf(lb, mu, sigma);  // cdf for bounds

    real u = uniform_rng(p, 1);

    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
}
}

data {

  int<lower = 1> N;                     // Number of observations
  int<lower = 1> K;                     // Number of covariates
  matrix[N, K] X;                       // Matrix design
  real lower_b;                         // Lower bound
  vector<lower = lower_b>[N] Y;         // Outcome

}

parameters {

  real alpha;                  // Intercept
  vector[K] beta;              // Coefficients
  real<lower = 0> sigma;       // Sd

}

model {

  // Likelihood
  for (n in 1:N) {

    Y[n] ~ normal(alpha + X[n] * beta, sigma) T[lower_b, ];

  }

  // Priors: stan uses uniform priors by default
  alpha~ normal(0, 100);
  beta ~ normal(0, 100);
  sigma ~ normal(0, 100);

}

generated quantities {

  vector[N] log_lik;
  vector<lower = lower_b>[N] y_rep;

  // Simulate replicated data from the fitted model
  for (n in 1:N) {

    // Linear predictor
    real eta_rep = X[n] * beta + alpha;

    // Log likelihood for model comparison with LOO
    log_lik[n] = normal_lpdf(Y[n] | eta_rep, sigma);

    // Replicated outcome
    y_rep[n] = normal_lb_rng(eta_rep, sigma, lower_b);

  }

}


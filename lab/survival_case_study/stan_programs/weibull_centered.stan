// Weibull survival model

data {

  int<lower = 0> n_obs;             // Number of deaths
  int<lower = 0> n_cens;            // Number of censored
  vector[n_obs] y_obs;              // Death vector
  vector[n_cens] y_cens;            // Censored vector
  int<lower = 0> k;                 // Number of covariates
  matrix[n_obs, k] x_obs;           // Design matrix for deaths
  matrix[n_cens, k] x_cens;         // Design matrix for censoring

}

transformed data {

  real<lower = 0> tau_beta_0;       // Sd of intercept
  real<lower = 0> tau_alpha;        // Sd alpha

  tau_beta_0 = 10;
  tau_alpha = 10;

}

parameters {

  real<lower = 0> alpha;           // Alpha parameter on the log scale
  real beta_0;                     // Intercept
  vector[k] beta;                  // Coefficients of covariates

}

model {

  // Linear predictors
  vector[n_obs] eta_obs = beta_0 + x_obs * beta;
  vector[n_cens] eta_cens = beta_0 + x_cens * beta;

  // Define the priors
  target += normal_lpdf(alpha | 0, tau_alpha) +
            normal_lpdf(beta_0 | 0, tau_beta_0) +
            normal_lpdf(beta | 0, 1);

  // Define the likelihood
  target += weibull_lpdf(y_obs | alpha, exp(-eta_obs/alpha)) +
            weibull_lccdf(y_cens | alpha, exp(-eta_cens/alpha));

}

generated quantities {

  // Replicated data and log predictive density
  vector[n_obs + n_cens] y_rep;
  vector[n_obs + n_cens] log_lik;

  // Simulations for deaths
  for(i in 1:n_obs) {

    // Linear predictor deaths
    real eta_obs_rep = beta_0 + x_obs[i] * beta;

    y_rep[i] = weibull_rng(alpha, exp(-eta_obs_rep/alpha));

    log_lik[i] = weibull_lpdf(
      y_obs[i] | alpha, exp(-eta_obs_rep/alpha)
    );

  }

  // Simulations for censored
  for(i in 1:n_cens) {

    // Linear predictor deaths
    real eta_cens_rep = beta_0 + x_cens[i] * beta;

    y_rep[n_obs + i] = weibull_rng(
      alpha, exp(-eta_cens_rep/alpha)
    );

    log_lik[n_obs + i] = weibull_lccdf(
      y_cens[i] | alpha, exp(-eta_cens_rep/alpha)
    );

  }

}



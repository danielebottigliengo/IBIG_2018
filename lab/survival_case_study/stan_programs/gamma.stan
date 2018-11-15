// Gamma survival model

data {

  int<lower = 0> n_obs;             // Number of deaths
  int<lower = 0> n_cens;            // Number of censored
  vector[n_obs] y_obs;              // Death vector
  vector[n_cens] y_cens;            // Censored vector
  int<lower = 0> k;                 // Number of covariates
  matrix[n_obs, k] x_obs;           // Design matrix for deaths
  matrix[n_cens, k] x_cens;         // Design matrix for censoring

}

parameters {

  real<lower = 0> shape;           // Shape parameter on log scale
  real beta_0;                     // Intercept
  vector[k] beta;                  // Coefficients of covariates

}

model {

  // Linear predictors
  vector[n_obs] eta_obs = exp(-(beta_0 + x_obs * beta));
  vector[n_cens] eta_cens = exp(-(beta_0 + x_cens * beta));

  // Define the priors
  target += student_t_lpdf(shape |3, 0, 1) +
            student_t_lpdf(beta_0 | 3, 0, 10) +
            student_t_lpdf(beta |3, 0, 1);

  // Define the likelihood
  target += gamma_lpdf(y_obs | shape, eta_obs) +
            gamma_lccdf(y_cens | shape, eta_cens);

}

generated quantities {

  // Replicated data and log predictive density
  vector[n_obs + n_cens] y_rep;
  vector[n_obs + n_cens] log_lik;

  // Simulations for deaths
  for(i in 1:n_obs) {

    // Linear predictor deaths
    real eta_obs_rep = exp(-(beta_0 + x_obs[i] * beta));

    y_rep[i] = gamma_rng(shape, eta_obs_rep);

    log_lik[i] = gamma_lpdf(y_obs[i] | shape, eta_obs_rep);

  }

  // Simulations for censored
  for(i in 1:n_cens) {

    // Linear predictor deaths
    real eta_cens_rep = exp(-(beta_0 + x_cens[i] * beta));

    y_rep[n_obs + i] = gamma_rng(shape, eta_cens_rep);

    log_lik[n_obs + i] = gamma_lccdf(y_cens[i] | shape, eta_cens_rep);

  }

}



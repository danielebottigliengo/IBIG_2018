// Lognormal survival model

data {

  int<lower = 0> n_obs;             // Number of deaths
  int<lower = 0> n_cens;            // Number of censored
  vector[n_obs] y_obs;              // Death vector
  vector[n_cens] y_cens;            // Censored vector
  int<lower = 0> k;                 // Number of covariates
  matrix[n_obs, k] x_obs;           // Design matrix for deaths
  matrix[n_cens, k] x_cens;         // Design matrix for censoring

  // New observations
  int<lower = 0> n_new;             // Number new observations
  matrix[n_new, k] x_new;           // New design matrix
  vector[n_new] times;              // New times

}

parameters {

  real<lower = 0> sigma;           // Sigma parameter on log scales
  real beta_0;                     // Intercept
  vector[k] beta;                  // Coefficients of covariates

}

model {

  // Linear predictors
  vector[n_obs] eta_obs = beta_0 + x_obs * beta;
  vector[n_cens] eta_cens = beta_0 + x_cens * beta;

  // Define the priors
  target += student_t_lpdf(sigma |3, 0, 1) +
            student_t_lpdf(beta_0 | 3, 0, 10) +
            student_t_lpdf(beta |3, 0, 1);

  // Define the likelihood
  target += lognormal_lpdf(y_obs | eta_obs, sigma) +
            lognormal_lccdf(y_cens | eta_cens, sigma);

}

generated quantities {

  // Replicated data, log predictive density and predicted data
  vector[n_obs + n_cens] y_rep;
  vector[n_obs + n_cens] log_lik;
  vector[n_new] surv_pred;

  // Simulations for deaths
  for(i in 1:n_obs) {

    // Linear predictor deaths
    real eta_obs_rep = beta_0 + x_obs[i] * beta;

    y_rep[i] = lognormal_rng(eta_obs_rep, sigma);

    log_lik[i] = lognormal_lpdf(y_obs[i] | eta_obs_rep, sigma);

  }

  // Simulations for censored
  for(i in 1:n_cens) {

    // Linear predictor deaths
    real eta_cens_rep = beta_0 + x_cens[i] * beta;

    y_rep[n_obs + i] = lognormal_rng(eta_cens_rep, sigma);

    log_lik[n_obs + i] = lognormal_lccdf(
      y_cens[i] | eta_cens_rep, sigma
    );

  }

  // Simulated follow-up time for each new observation
  for(i in 1:n_new) {

    // Linear predictor deaths
    real eta_pred = beta_0 + x_new[i] * beta;

    surv_pred[i] = 1 - normal_cdf(log(times[i]), eta_pred, sigma);

  }

}



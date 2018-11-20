// Weibull data generating process

data {

  int<lower = 0> n_obs;     // number of deaths
  int<lower = 0> n_cens;     // number of censored

}

parameters {

}

model {

}

generated quantities {

  // Declare simulated variables
  vector[n_obs + n_cens] age;
  int resid_ds[n_obs + n_cens];
  int ecog_ps[n_obs + n_cens];
  int rx[n_obs + n_cens];
  vector[n_obs + n_cens] futime;

  // Generate parameter values from the prior predictive distribution
  real alpha = uniform_rng(1, 100);
  real beta_0 = uniform_rng(-100, 100);
  real beta_age = uniform_rng(-100, 100);
  real beta_resid_ds = uniform_rng(-100, 100);
  real beta_ecog_ps = uniform_rng(-100, 100);
  real beta_rx = uniform_rng(-100, 100);

  // Simulate fake data
  real eta_obs;
  real eta_cens;

  for(i in 1:n_obs) {

    // Covariates for deaths
    age[i] = normal_rng(0, 1);
    resid_ds[i] = bernoulli_rng(0.58);
    ecog_ps[i] = bernoulli_rng(0.46);
    rx[i] = bernoulli_rng(0.50);

    // Linear predictor for deaths
    eta_obs = beta_0 +
              beta_age * age[i] +
              beta_resid_ds * resid_ds[i] +
              beta_ecog_ps * ecog_ps[i] +
              beta_rx * rx[i];

    futime[i] = weibull_rng(alpha, exp(-eta_obs/alpha));

  }

  for(i in 1:n_cens) {

    // Covariates for deaths
    age[n_obs + i] = normal_rng(0, 1);
    resid_ds[n_obs + i] = bernoulli_rng(0.58);
    ecog_ps[n_obs + i] = bernoulli_rng(0.46);
    rx[n_obs + i] = bernoulli_rng(0.50);

    // Linear predictor for deaths
    eta_cens = beta_0 +
              beta_age * age[i] +
              beta_resid_ds * resid_ds[i] +
              beta_ecog_ps * ecog_ps[i] +
              beta_rx * rx[i];

    futime[n_obs + i] = weibull_rng(
      alpha, exp(-eta_cens/alpha)
    );

  }

}

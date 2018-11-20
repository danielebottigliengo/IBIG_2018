// Logit data generating process uniform

data {

  int<lower = 1> N;     // number of observations

}

parameters {

}

model {

}

generated quantities {

  // Declare simulated variables
  vector[N] age;
  int smoke[N];
  int ui[N];
  vector[N] lwt;
  vector[N] bwt;

  // Generate parameter values from the prior predictive distribution
  real<lower = 0> sigma = lognormal_rng(0, 100);
  real intercept = normal_rng(0, 100);
  real b_age = normal_rng(0, 100);
  real b_smoke = normal_rng(0, 100);
  real b_ui = normal_rng(0, 100);
  real b_lwt = normal_rng(0, 100);

  // Simulate fake data
  real eta;

  for(n in 1:N) {

    // Generate covariates
    age[n] = normal_rng(0, 1);
    smoke[n] = bernoulli_rng(0.35);
    ui[n] = bernoulli_rng(0.10);
    lwt[n] = normal_rng(0, 1);

    // Linear predictor
    eta = intercept +
          b_age * age[n] +
          b_smoke * smoke[n] +
          b_ui * ui[n] +
          b_lwt * lwt[n];

    // Outcome
    bwt[n] = normal_rng(eta, sigma);


  }

}

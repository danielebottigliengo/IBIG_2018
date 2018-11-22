// Normal dgp with weakly priors

// Random number generator from a lower-bound normal distribution
functions {
  real normal_lb_rng(real mu, real sigma, real lb) {

    real p = normal_cdf(lb, mu, sigma);  // cdf for bounds

    real u = uniform_rng(p, 1);

    return (sigma * inv_Phi(u)) + mu;  // inverse cdf for value
}
}

data {

  int<lower = 1> N;     // number of observations
  real lower_b;           // lower bound of the normal

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
  vector<lower = lower_b>[N] bwt;

  // Generate parameter values from the prior predictive distribution
  real<lower = lower_b> sigma = normal_lb_rng(0, 100, lower_b);
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
    bwt[n] = normal_lb_rng(eta, sigma, lower_b);


  }

}

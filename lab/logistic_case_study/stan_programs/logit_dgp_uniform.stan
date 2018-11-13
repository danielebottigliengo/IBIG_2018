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
  int race[N];
  int home_oxygen_use[N];
  int gender[N];
  int current_smoking[N];
  int diabetes_mellitus[N];
  vector[N] age;
  int vaccine[N];
  int influenza[N];

  // Generate parameter values from the prior predictive distribution
  real intercept = uniform_rng(-5, 5);
  real b_race = uniform_rng(-5, 5);
  real b_home_oxygen_use = uniform_rng(-5, 5);
  real b_gender = uniform_rng(-5, 5);
  real b_current_smoking = uniform_rng(-5, 5);
  real b_diabetes_mellitus = uniform_rng(-5, 5);
  real b_age = uniform_rng(-5, 5);
  real b_vaccine = uniform_rng(-5, 5);

  // Simulate fake data
  real eta;

  for(n in 1:N) {

    // Generate covariates
    race[n] = bernoulli_rng(0.36);
    home_oxygen_use[n] = bernoulli_rng(0.29);
    gender[n] = bernoulli_rng(0.64);
    current_smoking[n] = bernoulli_rng(0.26);
    diabetes_mellitus[n] = bernoulli_rng(0.29);
    age[n] = normal_rng(73, 22);
    vaccine[n] = bernoulli_rng(0.60);

    // Linear predictor
    eta = intercept +
          b_race * race[n] +
          b_home_oxygen_use * home_oxygen_use[n] +
          b_gender * gender[n] +
          b_current_smoking * current_smoking[n] +
          b_diabetes_mellitus * diabetes_mellitus[n] +
          b_age * age[n] +
          b_vaccine * vaccine[n];

    // Outcome
    influenza[n] = bernoulli_logit_rng(eta);


  }

}

# IBIG case study analysis: logistic regression ------------------------
library(tidyverse)
library(rstan)
  options(mc.cores = parallel::detectCores() - 1)
  rstan_options(auto_write = TRUE)
  Sys.setenv(LOCAL_CPPFLAGS = c('-march=native'))
library(bayesplot)
library(loo)

load(here::here("data/logistic_case_study.rda"))
source(here::here("lab/logistic_case_study/logit_misc_functions.R"))

# The goal of the case study is tho show how priors can impact the
# inference, especially with noisy and small size samples. The
# importance of the prios is shown by fitting different models with
# different priors. Priors on the coefficients will gradually be more
# defined. We will start with non-informative prior and we will end
# with weakly informative priors, showing the benefits that they could
# provide in terms of regularization (shrinkage) of the inference
# process.

# The analysis will be presented focusing of the following aspects
# of the Bayesian data analysis:
#   1) Exploratory data analysis
#   2) Fitting models with different priors
#   3) The importance of weakly informative priors on reguralization
#   4) Model comparison

mcmc_seed <- 140509

# 1) Exploratory data analysis plot ------------------------------------
# Plot jittered point of the influenza status given vaccine
p1 <- ggplot(
  data = db_logit, mapping = aes(x =  vaccine, y = influenza)
) +
  geom_jitter(width = 0.3, height = 0.3) +
  xlab("Vaccination status") +
  ylab("Influenza status") +
  theme_bw()

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/flu_vac.png"
  ),
  plot = p1, width = 10, height = 8
)

# Plot jittered point of influenza given vaccine and the timing of
# admission relative to the onset of influenza season
p2 <- ggplot(
  data = db_logit, mapping = aes(
    x = timing_onset_influenza_days, y = as.numeric(influenza) - 1,
    colour = vaccine, fill = vaccine
  )
) +
  geom_jitter(width = 0.3, height = 0.1) +
  geom_smooth(
    method = "glm", method.args = list(family = "binomial"), se = FALSE,
    alpha = 0.25
  ) +
  xlab("Timing of admission") +
  ylab("Influenza status") +
  theme_bw()

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/flu_vac_time.png"
  ),
  plot = p2, width = 10, height = 8
)

# Plot jittered point of influenza given vaccine and the timing of
# admission relative to the onset of influenza season and diabetes
# mellitus
p3 <- ggplot(
  data = db_logit, mapping = aes(
    x = timing_onset_influenza_days, y = as.numeric(influenza) - 1,
    colour = vaccine, fill = vaccine
  )
) +
  geom_jitter(width = 0.3, height = 0.1) +
  geom_smooth(
    method = "glm", method.args = list(family = "binomial"), se = FALSE,
    alpha = 0.25
  ) +
  xlab("Timing of admission") +
  ylab("Influenza status") +
  facet_grid(. ~ diabetes_mellitus) +
  theme_bw()

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/flu_vac_time_diab.png"
  ),
  plot = p3, width = 10, height = 8
)

# Plot jittered point of influenza given vaccine and the timing of
# admission relative to the onset of influenza season, diabetes
# mellitus and gender
p4 <- ggplot(
  data = db_logit, mapping = aes(
    x = timing_onset_influenza_days, y = as.numeric(influenza) - 1,
    colour = vaccine, fill = vaccine
  )
) +
  geom_jitter(width = 0.3, height = 0.1) +
  geom_smooth(
    method = "glm", method.args = list(family = "binomial"), se = FALSE,
    alpha = 0.25
  ) +
  xlab("Timing of admission") +
  ylab("Influenza status") +
  facet_grid(gender ~ diabetes_mellitus) +
  theme_bw()

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/flu_vac_time_diab_gender.png"
  ),
  plot = p4, width = 12, height = 8
)

# Plot jittered point of influenza given vaccine and the age, diabetes
# mellitus and gender
p5 <- ggplot(
  data = db_logit, mapping = aes(
    x = age_year, y = as.numeric(influenza) - 1,
    colour = vaccine, fill = vaccine
  )
) +
  geom_jitter(width = 0.3, height = 0.1) +
  geom_smooth(
    method = "glm", method.args = list(family = "binomial"), se = FALSE,
    alpha = 0.25
  ) +
  xlab("Year") +
  ylab("Influenza status") +
  facet_grid(gender ~ diabetes_mellitus) +
  theme_bw()

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/flu_vac_age_diab_gender.png"
  ),
  plot = p5, width = 12, height = 8
)

# Scaling covariates by a constant is always good for computation ------
db_logit_analysis <- db_logit %>%
  mutate_at(
    vars(
      race, home_oxygen_use, current_smoking:other_diseases, vaccine
    ),
    funs(as.numeric(if_else(. == "Yes", 1, 0)))
  ) %>%
  mutate(
    gender_female = as.numeric(if_else(gender == "female", 1, 0)),
    influenza = as.integer(if_else(influenza == "Yes", 1, 0))
  )  %>%
  mutate_at(
    vars(race:home_oxygen_use, current_smoking:vaccine, gender_female),
    funs(as.numeric(scale(x = ., scale = FALSE)))
  ) %>%
  mutate(
    age_100 = age_year/100,
    timing_onset_influenza_months = timing_onset_influenza_days/30
  ) %>%
  dplyr::select(
    race:home_oxygen_use, gender_female, current_smoking:other_diseases,
    age_100, timing_onset_influenza_months, vaccine, influenza
  )

design_matrix_logit <- db_logit_analysis %>%
  dplyr::select(race:vaccine)

db_logit_list <- list(
  N = nrow(db_logit_analysis),
  K = ncol(design_matrix_logit),
  X = as.matrix(design_matrix_logit),
  Y = db_logit_analysis[["influenza"]]
)

# 2) Non-informative priors: uniform -----------------------------------
# 2A) Prior Predictive checks ------------------------------------------
uniform_dgp <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_dgp_uniform.stan"
)

fitted_uniform_dgp <- sampling(
  object = uniform_dgp, data = list(N = 1000L),
  chains = 1L, cores = 1L, iter = 1L, algorithm = "Fixed_param",
  seed = mcmc_seed
)

# Get fake data and store them into a list ready to be passed to
# the stan program of the model. Covariates need to be putted roughly
# on unit scale.

fake_data <- rstan::extract(fitted_uniform_dgp)

fake_data_covs <- data_frame(
  race = fake_data$race[1, ],
  home_oxygen_use = fake_data$home_oxygen_use[1, ],
  gender = fake_data$gender[1, ],
  current_smoking = fake_data$current_smoking[1, ],
  diabetes_mellitus = fake_data$diabetes_mellitus[1, ],
  age = fake_data$age[1, ],
  vaccine = fake_data$vaccine[1, ]
) %>%
  mutate(
    age = age/100
  ) %>%
  mutate_all(funs(scale(x = ., scale = FALSE)))

fake_data_list <- list(
  N = nrow(fake_data_covs),
  K = ncol(fake_data_covs),
  X = as.matrix(fake_data_covs),
  Y = fake_data$influenza[1, ]
)

# Now I have to fit the model to fake data and check if true parameter
# values are recovered by the model
comp_uniform <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_uniform.stan"
)

fitted_fake <- sampling(
  object = comp_uniform, data = fake_data_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

# Does the model recover the true parameter values?
post_coefs <- as.matrix(fitted_fake, pars = c("alpha", "beta"))

true_coefs <- c(
  fake_data$intercept, fake_data$b_race, fake_data$b_home_oxygen_use,
  fake_data$b_gender, fake_data$b_current_smoking,
  fake_data$b_diabetes_mellitus, fake_data$b_age, fake_data$b_vaccine
)

uniform_fake_recover <- mcmc_recover_hist(
  x = post_coefs, true = true_coefs
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/uniform_fake_recover.png"
  ),
  plot = uniform_fake_recover,
  width = 10, height = 8
)

# Not so good (divergent transitions problems...)

# 2B) Fit the model to real data ---------------------------------------
fitted_uniform <- sampling(
  object = comp_uniform, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

# Many problematic transitions: inference is biased
post_uniform <- rstan::extract(fitted_uniform)
post_coefs_uniform <- as.data.frame(
  fitted_uniform, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 2C) Diagnose divergent transitions -----------------------------------
print(fitted_uniform, pars = c("alpha", "beta"))

uniform_recover <- mcmc_recover_hist(
  x = post_coefs_uniform, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/uniform_recover.png"
  ),
  plot = uniform_recover,
  width = 10, height = 8
)

# 2D) Posterior predictive checks --------------------------------------
y_rep <- as.matrix(fitted_uniform, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

uniform_zero <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/uniform_zero.png"
  ),
  plot = uniform_zero,
  width = 10, height = 8
)

uniform_one <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/uniform_one.png"
  ),
  plot = uniform_one,
  width = 10, height = 8
)

# 3) Vague priors ------------------------------------------------------
# 3B) Fit the model to real data ---------------------------------------
comp_vague <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_vague.stan"
)

fitted_vague <- sampling(
  object = comp_vague, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

# Many problematic transitions: inference is biased
post_vague <- rstan::extract(fitted_vague)
post_coefs_vague <- as.matrix(
  fitted_vague, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 3C) Diagnose divergent transitions -----------------------------------
print(fitted_vague, pars = c("alpha", "beta"))

vague_recover <- mcmc_recover_hist(
  x = post_coefs_vague, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/vague_recover.png"
  ),
  plot = vague_recover,
  width = 10, height = 8
)

# 3D) Posterior predictive checks --------------------------------------
y_rep <- as.matrix(fitted_vague, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

vague_zero <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/vague_zero.png"
  ),
  plot = vague_zero,
  width = 10, height = 8
)

vague_one <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/vague_one.png"
  ),
  plot = vague_one,
  width = 10, height = 8
)

# 4) Weakly informative priors 2: cauchy -------------------------------
# 4B) Fit the model to real data ---------------------------------------
comp_cauchy <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_cauchy.stan"
)

fitted_cauchy <- sampling(
  object = comp_cauchy, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

# Many problematic transitions: inference is biased
post_cauchy <- rstan::extract(fitted_cauchy)
post_coefs_cauchy <- as.matrix(
  fitted_cauchy, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 4C) Diagnose divergent transitions -----------------------------------
print(fitted_cauchy, pars = c("alpha", "beta"))

cauchy_recover <- mcmc_recover_hist(
  x = post_coefs_cauchy, true = true_betas
)

ggsave(
  file = here::here(
    "lab/logistic_case_study/figures/cauchy_recover.png"
  ),
  plot = cauchy_recover,
  width = 10, height = 8
)

# 4D) Posterior predictive checks --------------------------------------
y_rep <- as.matrix(fitted_cauchy, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

cauchy_zero <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/cauchy_zero.png"
  ),
  plot = cauchy_zero,
  width = 10, height = 8
)

cauchy_one <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/cauchy_one.png"
  ),
  plot = cauchy_one,
  width = 10, height = 8
)

# 5) Weakly informative priors 2: t-student 3 --------------------------
# 5B) Fit the model to real data ---------------------------------------
comp_student_3 <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_student_3.stan"
)

fitted_student_3 <- sampling(
  object = comp_student_3, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

# Many problematic transitions: inference is biased
post_student_3 <- rstan::extract(fitted_student_3)
post_coefs_student_3 <- as.matrix(
  fitted_student_3, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 5C) Diagnose divergent transitions -----------------------------------
print(fitted_student_3, pars = c("alpha", "beta"))

student_recover_3 <- mcmc_recover_hist(
  x = post_coefs_student_3, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_recover_3.png"
  ),
  plot = student_recover_3,
  width = 10, height = 8
)

# 5D) Posterior predictive checks --------------------------------------
y_rep <- as.matrix(fitted_student_3, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

student_zero_3 <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_zero_3.png"
  ),
  plot = student_zero_3,
  width = 10, height = 8
)

student_one_3 <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_one_3.png"
  ),
  plot = student_one_3,
  width = 10, height = 8
)

# 6) Weakly informative priors 2: t-student 7 --------------------------
# 6B) Fit the model to real data ---------------------------------------
comp_student_7 <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_student_7.stan"
)

fitted_student_7 <- sampling(
  object = comp_student_7, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

# Many problematic transitions: inference is biased
post_student_7 <- rstan::extract(fitted_student_7)
post_coefs_student_7 <- as.matrix(
  fitted_student_7, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 6C) Diagnose divergent transitions -----------------------------------
print(fitted_student_7, pars = c("alpha", "beta"))

student_recover_7 <- mcmc_recover_hist(
  x = post_coefs_student_7, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_recover_7.png"
  ),
  plot = student_recover_7,
  width = 10, height = 8
)

# 6D) Posterior predictive checks --------------------------------------
y_rep <- as.matrix(fitted_student_7, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

student_zero_7 <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_zero_7.png"
  ),
  plot = student_zero_7,
  width = 10, height = 8
)

student_one_7 <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_one_7.png"
  ),
  plot = student_one_7,
  width = 10, height = 8
)

# 7) Weakly informative priors 2: t-student 3 unpenalized --------------
# 7B) Fit the model to real data ---------------------------------------
comp_student_3_unp <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_student_3_unp.stan"
)

fitted_student_3_unp <- sampling(
  object = comp_student_3_unp, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

# Many problematic transitions: inference is biased
post_student_3_unp <- rstan::extract(fitted_student_3_unp)
post_coefs_student_3_unp <- as.matrix(
  fitted_student_3_unp, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 7C) Diagnose divergent transitions -----------------------------------
print(fitted_student_3_unp, pars = c("alpha", "beta"))

student_recover_3_unp <- mcmc_recover_hist(
  x = post_coefs_student_3_unp, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_recover_3_unp.png"
  ),
  plot = student_recover_3_unp,
  width = 10, height = 8
)

# 7D) Posterior predictive checks --------------------------------------
y_rep <- as.matrix(fitted_student_3_unp, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

student_zero_3_unp <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_zero_3_unp.png"
  ),
  plot = student_zero_3_unp,
  width = 10, height = 8
)

student_one_3_unp <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_one_3_unp.png"
  ),
  plot = student_one_3_unp,
  width = 10, height = 8
)

# 8) Weakly informative priors 2: t-student 7 unpenalized --------------
# 8B) Fit the model to real data ---------------------------------------
comp_student_7_unp <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_student_7_unp.stan"
)

fitted_student_7_unp <- sampling(
  object = comp_student_7_unp, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

post_student_7_unp <- rstan::extract(fitted_student_7_unp)
post_coefs_student_7_unp <- as.matrix(
  fitted_student_7_unp, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 8C) Diagnose divergent transitions -----------------------------------
print(fitted_student_7_unp, pars = c("alpha", "beta"))

student_recover_7_unp <- mcmc_recover_hist(
  x = post_coefs_student_7_unp, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_recover_7_unp.png"
  ),
  plot = student_recover_7_unp,
  width = 10, height = 8
)

# 8D) Posterior predictive checks --------------------------------------
y_rep <- as.matrix(fitted_student_7_unp, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

student_zero_7_unp <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_zero_7_unp.png"
  ),
  plot = student_zero_7_unp,
  width = 10, height = 8
)

student_one_7_unp <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_one_7_unp.png"
  ),
  plot = student_one_7_unp,
  width = 10, height = 8
)

# 9) Weakly informative priors 2: t-student 2 --------------------------
# 9B) Fit the model to real data ---------------------------------------
comp_student_2 <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_student_2.stan"
)

fitted_student_2 <- sampling(
  object = comp_student_2, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

post_student_2 <- rstan::extract(fitted_student_2)
post_coefs_student_2 <- as.matrix(
  fitted_student_2, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 9C) Diagnose divergent transitions -----------------------------------
print(fitted_student_2, pars = c("alpha", "beta"))

student_recover_2 <- mcmc_recover_hist(
  x = post_coefs_student_2, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_recover_2.png"
  ),
  plot = student_recover_2,
  width = 10, height = 8
)

# 9D) Posterior predictive checks --------------------------------------
y_rep <- as.matrix(fitted_student_2, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

student_zero_2 <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_zero_2.png"
  ),
  plot = student_zero_2,
  width = 10, height = 8
)

student_one_2 <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_one_2.png"
  ),
  plot = student_one_2,
  width = 10, height = 8
)

# 10) Weakly informative priors 2: t-student 2 unpenalized -------------
# 10B) Fit the model to real data --------------------------------------
comp_student_2_unp <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_student_2_unp.stan"
)

fitted_student_2_unp <- sampling(
  object = comp_student_2_unp, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

post_student_2_unp <- rstan::extract(fitted_student_2_unp)
post_coefs_student_2_unp <- as.matrix(
  fitted_student_2_unp, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 10C) Diagnose divergent transitions ----------------------------------
print(fitted_student_2_unp, pars = c("alpha", "beta"))

student_recover_2_unp <- mcmc_recover_hist(
  x = post_coefs_student_2_unp, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_recover_2_unp.png"
  ),
  plot = student_recover_2_unp,
  width = 10, height = 8
)

# 10D) Posterior predictive checks -------------------------------------
y_rep <- as.matrix(fitted_student_2_unp, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

student_zero_2_unp <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_zero_2_unp.png"
  ),
  plot = student_zero_2_unp,
  width = 10, height = 8
)

student_one_2_unp <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_one_2_unp.png"
  ),
  plot = student_one_2_unp,
  width = 10, height = 8
)



# 11) Model comparison -------------------------------------------------
model_list <- list(
  fitted_uniform, fitted_vague, fitted_cauchy, fitted_student_3,
  fitted_student_7, fitted_student_3_unp, fitted_student_7_unp,
  fitted_student_2, fitted_student_2_unp
)

comp_model_list <- map(
  .x = model_list, ~ model_comparison_loo(stan_fit = .x)
) %>%
  setNames(
    object = ., nm = c(
      "uniform", "vague", "cauchy", "student_t_3", "student_t_7",
      "student_t_3_unp", "student_t_7_unp", "student_t_2",
      "student_t_2_unp"
    )
  )

loo_list <- map(.x = comp_model_list, ~ .x[["loo"]])
stacking_weights <- loo_model_weights(loo_list[4:length(loo_list)])
pseudo_bma <- loo_model_weights(
  loo_list[4:length(loo_list)], method = "pseudobma", BB = FALSE
)

pseudo_bma_bb <- loo_model_weights(
  loo_list[4:length(loo_list)], method = "pseudobma", BB = TRUE
)

# What if we know the disease is rare? ---------------------------------
# 12) T-student 2 rare -------------------------------------------------
# 12B) Fit the model to real data --------------------------------------
comp_student_2_rare <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_student_2_rare.stan"
)

fitted_student_2_rare <- sampling(
  object = comp_student_2_rare, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

post_student_2_rare <- rstan::extract(fitted_student_2_rare)
post_coefs_student_2_rare <- as.matrix(
  fitted_student_2_rare, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 12C) Diagnose divergent transitions ----------------------------------
print(fitted_student_2_rare, pars = c("alpha", "beta"))

student_recover_2_rare <- mcmc_recover_hist(
  x = post_coefs_student_2_rare, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_recover_2_rare.png"
  ),
  plot = student_recover_2_rare,
  width = 10, height = 8
)

# 12D) Posterior predictive checks -------------------------------------
y_rep <- as.matrix(fitted_student_2_rare, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

student_zero_2_rare <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_zero_2_rare.png"
  ),
  plot = student_zero_2_rare,
  width = 10, height = 8
)

student_one_2_rare <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_one_2_rare.png"
  ),
  plot = student_one_2_rare,
  width = 10, height = 8
)

# 13)T-student 2 rare unpenalized --------------------------------------
# 13B) Fit the model to real data --------------------------------------
comp_student_2_unp_rare <- stan_model(
  file = "lab/logistic_case_study/stan_programs/logit_student_2_unp_rare.stan"
)

fitted_student_2_unp_rare <- sampling(
  object = comp_student_2_unp_rare, data = db_logit_list,
  chains = 4L, warmup = 1000L, iter = 2000L, seed = mcmc_seed
)

post_student_2_unp_rare <- rstan::extract(fitted_student_2_unp_rare)
post_coefs_student_2_unp_rare <- as.matrix(
  fitted_student_2_unp_rare, pars = c("alpha", "beta")
) %>%
  setNames(c("intercept", names(design_matrix_logit)))

# 13C) Diagnose divergent transitions ----------------------------------
print(fitted_student_2_unp_rare, pars = c("alpha", "beta"))

student_recover_2_unp_rare <- mcmc_recover_hist(
  x = post_coefs_student_2_unp_rare, true = true_betas
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_recover_2_unp_rare.png"
  ),
  plot = student_recover_2_unp_rare,
  width = 10, height = 8
)

# 13D) Posterior predictive checks -------------------------------------
y_rep <- as.matrix(fitted_student_2_unp_rare, pars = "y_rep")

prop_zero <- function(x) mean(x == 0)
prop_one <- function(x) mean(x == 1)

student_zero_2_unp_rare <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_zero
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_zero_2_unp_rare.png"
  ),
  plot = student_zero_2_unp_rare,
  width = 10, height = 8
)

student_one_2_unp_rare <- ppc_stat(
  y = db_logit_analysis[["influenza"]], yrep = y_rep[1:200, ],
  stat = prop_one
)

ggsave(
  filename = here::here(
    "lab/logistic_case_study/figures/student_one_2_unp_rare.png"
  ),
  plot = student_one_2_unp_rare,
  width = 10, height = 8
)

# 14) Model comparison -------------------------------------------------
model_list_rare <- list(
  fitted_uniform, fitted_vague, fitted_cauchy, fitted_student_3,
  fitted_student_7, fitted_student_3_unp, fitted_student_7_unp,
  fitted_student_2, fitted_student_2_unp, fitted_student_2_rare,
  fitted_student_2_unp_rare
)

comp_model_list_rare <- map(
  .x = model_list_rare, ~ model_comparison_loo(stan_fit = .x)
) %>%
  setNames(
    object = ., nm = c(
      "uniform", "vague", "cauchy", "student_t_3", "student_t_7",
      "student_t_3_unp", "student_t_7_unp", "student_t_2",
      "student_t_2_unp", "student_2_rare", "student_2_unp_rare"
    )
  )

loo_list_rare <- map(.x = comp_model_list_rare, ~ .x[["loo"]])
stacking_weights_rare <- loo_model_weights(
  loo_list_rare[4:length(loo_list_rare)]
)
pseudo_bma_rare <- loo_model_weights(
  loo_list_rare[4:length(loo_list_rare)],
  method = "pseudobma", BB = FALSE
)

pseudo_bma_bb_rare <- loo_model_weights(
  loo_list_rare[4:length(loo_list_rare)],
  method = "pseudobma", BB = TRUE
)

# Save everything needed for the slides --------------------------------
save(
  post_uniform, post_vague, post_cauchy, post_student_3,
  post_student_7, post_student_3_unp, post_student_7_unp,
  post_student_2, post_student_2_unp, post_student_2_rare,
  post_student_2_unp_rare,
  post_coefs_uniform, post_coefs_vague, post_coefs_cauchy,
  post_coefs_student_3, post_coefs_student_7, post_coefs_student_3_unp,
  post_coefs_student_7_unp, post_coefs_student_2,
  post_coefs_student_2_unp, post_coefs_student_2_rare,
  post_coefs_student_2_unp_rare,
  comp_model_list, loo_list, stacking_weights, pseudo_bma,
  pseudo_bma_bb, comp_model_list_rare, loo_list_rare,
  stacking_weights_rare, pseudo_bma_rare, pseudo_bma_bb_rare,
  file = here::here("lab/logistic_case_study/logit_analysis.rda")
)






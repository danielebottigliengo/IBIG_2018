# Dataset creation based on Chen et al. (2016) -------------------------
library(tidyverse)
library(janitor)
library(MASS)
library(glue)

source(here::here("lab/logistic_case_study/logit_misc_functions.R"))

# Dataset is created from the study of Chen et al. (2016) on the
# the effect of vaccination status on occurrence of influenza. The
# study was chosen because of the rare outcome and the high number
# of potential confounders. The idea is to show how priors distributions
# can be effectively used to regularize inference.

# Both outcome and exposure were measured as binary. 13 potential
# confounders were present, 11 of which were binary and 2 were
# continuous. Simulation procedure will follows the instructions
# provided by the authors of the paper.

# 1) Creation of confounders -------------------------------------------
# 1A) Binary variables from z2 to z7 -----------------------------------
# 6 binary variables will be generated using latent Gaussian
# distribution to allow for correlation between them.
# First 2 multivariate normal will be generated. After that, cutoff
# values will be used for each variable to choose their marginal
# probabilities.

n <- 200

sigma_r_1 <- matrix(
  data = c(4, 2, 2, 2, 3, 1, 2, 1, 1), nrow = 3L, ncol = 3L
)

sigma_r_2 <- matrix(
  data = c(4, 2, 2, 2, 4, 2, 2, 2, 4), nrow = 3L, ncol = 3L
)

set.seed(140509)    # seed for reproducibility
r_1 <- mvrnorm(n = n, mu = c(0, 0, 0), Sigma = sigma_r_1) %>%
  as_data_frame() %>%
  rename(z2 = V1, z3 = V2, z4 = V3)

set.seed(140509)    # seed for reproducibility
r_2 <- mvrnorm(n = n, mu = c(0, 0, 0), Sigma = sigma_r_2) %>%
  as_data_frame() %>%
  rename(z5 = V1, z6 = V2, z7 = V3)

# List of marginal probabilities should be in the same order of the
# columns of the dataframe as define in Chen et al. (2016)
marg_prob_list <- c(0.29, 0.64, 0.26, 0.29, 0.66, 0.50)
cut_off_list <- seq(from = -10, to = 10, by = 0.001)
delta <- 0.01

latent_conf_df <- bind_cols(r_1, r_2)

binary_corr_df <- return_binary_conf_df(
  latent_df = latent_conf_df, cut_off_list = cut_off_list,
  marg_prob_list = marg_prob_list, delta = delta
)

# 1B) Adding other 5 binary and 2 continuous confounders ---------------
set.seed(140509)    # seed for reproducibility
conf_df <- binary_corr_df %>%
  mutate(
    z1 = rbinom(n = n, size = 1, prob = 0.36),
    z8 = rbinom(n = n, size = 1, prob = 0.32),
    z9 = rbinom(n = n, size = 1, prob = 0.26),
    z10 = rbinom(n = n, size = 1, prob = 0.12),
    z11 = rbinom(n = n, size = 1, prob = 0.10),
    z12 = round(rnorm(n = n, mean = 73, sd = 22)),
    z13 = round(rnorm(n = n, mean = 59, sd = 15))
  ) %>%
  dplyr::select(
    z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13
  ) %>%
  rename(
    race = z1, home_oxygen_use = z2, gender = z3, current_smoking = z4,
    diabetes_mellitus = z5, asthma_copd = z6, chd = z7,
    immunosuppression = z8, liver_kidney = z9, asplenia = z10,
    other_diseases = z11, age_year = z12,
    timing_onset_influenza_days = z13
  )

message("Confounders are over!")

# 2) Exposure assignment -----------------------------------------------
# Coefficients for the exposure assignment are taken from Chen et
# al. (2016). The intercept will be chosen with a grid approach such
# the exposure rate will be near 0.60 (moderate exposure), the rate
# observed in the study of Talbot et al. (2013)
betas <- c(
  -0.001, 0.001, -0.459, -0.001, 0.608, -0.441, -0.358, 0.504, -0.292,
  -0.201, -0.1, 0.022, -0.005
)

exposure_rate <- 0.6

# 2A) Define the intercept of the exposure model -----------------------
intercept_grid <- seq(from = -8, to = 8, by = 0.01)

exposure_intercept <- choose_intercept_logit(
  x = as.matrix(conf_df), betas = betas,
  intercept_grid = intercept_grid, given_rate = exposure_rate,
  delta = 0.01
)

# 2B) Generate the exposure --------------------------------------------
set.seed(140509)    # seed for reproducibility

coefs <- c(exposure_intercept, betas)
x_mat <- cbind(rep(1, nrow(conf_df)), as.matrix(conf_df))
exposure_eta <- x_mat %*% coefs

exposure_prob <- exp(exposure_eta)/(1 + exp(exposure_eta))

set.seed(140509)    # seed for reproducibility
exposure <- rbinom(n = n, size = 1, prob = exposure_prob)

design_matrix <- conf_df %>%
  mutate(vaccine = exposure)

message("Vaccine is over!")

# 3) Influenza status --------------------------------------------------
# Coefficients for the influenza status are taken from Chen et
# al. (2016). The intercept will be chosen with a grid approach such
# the influenza will be near 0.05 (rare outcome), similar to the one
# observed in the study of Talbot et al. (2013). The coefficient for
# the Vaccine Effect (VE) was set equal to -1.59, to one estimated
# by best method in Chen et al. (2016), i.e. ridge regression with
# no penalty on the vaccination status
alphas <- c(
  -0.286, 1.992, 0.918, -0.432, 0.127, -0.833, 0.019, -0.292, -0.113,
  -0.388, -0.001, -0.045, 0.10, -1.59
)

influenza_rate <- 0.1

# 3A) Define the intercept of the exposure model -----------------------
intercept_grid <- seq(from = -8, to = 8, by = 0.01)

influenza_intercept <- choose_intercept_logit(
  x = as.matrix(design_matrix), betas = alphas,
  intercept_grid = intercept_grid, given_rate = influenza_rate,
  delta = 0.01
)

true_betas <- c(influenza_intercept, alphas)

# 3B) Generate the exposure --------------------------------------------
set.seed(140509)    # seed for reproducibility

coefs <- c(influenza_intercept, alphas)
x_mat <- cbind(rep(1, nrow(design_matrix)), as.matrix(design_matrix))
influenza_eta <- x_mat %*% coefs

influenza_prob <- exp(influenza_eta)/(1 + exp(influenza_eta))

set.seed(140509)    # seed for reproducibility
influenza <- rbinom(n = n, size = 1, prob = influenza_prob)

db <- design_matrix %>%
  mutate(influenza = influenza) %>%
  mutate_at(
    vars(
      home_oxygen_use, current_smoking:other_diseases,
      vaccine, influenza
    ), funs(factor(if_else(. == 1, "Yes", "No")))
  ) %>%
  mutate(
    race = factor(if_else(race == 1, "black", "not_black")),
    gender = factor(if_else(gender == 1, "female", "male"))
  )

message("Influenza is over!")

# Test: fix the sample size up to 10000, fit a logistic regression and
# check that coefficients estimates are similar to the fixed one
# logit_vaccine <- glm(
#   vaccine ~ ., family = binomial("logit"), data = design_matrix
# )
#
# logit_influenza <- glm(
#   influenza ~ ., family = binomial("logit"), data = db
# )

# Cofficients are all ok but the race one (being black reduces the
# risk of influenza...)

# Let's save the dataset with a sample size of 200 (roughly the sample
# size of the study of Talbot et al. (2013))
db_logit <- db

save(
  db_logit, true_betas,
  file = here::here("data/logistic_case_study.rda")
)



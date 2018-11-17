# IBIG case study analysis: logistic regression ------------------------
library(tidyverse)
library(rstan)
  options(mc.cores = parallel::detectCores() - 1)
  rstan_options(auto_write = TRUE)
  Sys.setenv(LOCAL_CPPFLAGS = c('-march=native'))
library(brms)
library(bayesplot)
library(loo)
library(survival)
library(survminer)

source(here::here("lab/logistic_case_study/logit_misc_functions.R"))

# The goal of this case study is to give a simple example of a
# Bayesian survival analysis with a Weibull model. The main idea is
# to focus on this two aspects of the Bayesian Workflow:
#   1) Prior Predictive checks
#   2) Algorithm diagnostics
#   3) Reparameterization of the model

mcmc_seed <- 140509

# Rescale covariates for computational ease
db_ovarian <- ovarian

db_ovarian_scaled <- ovarian %>%
  janitor::clean_names() %>%
  dplyr::select(age, resid_ds, ecog_ps, rx, futime, fustat) %>%
  mutate(age = age/100) %>%
  mutate_at(
    vars(age:rx), funs(as.numeric(scale(x = ., scale = FALSE)))
  ) %>%
  mutate(futime = futime/365.25)

ovarian_scaled_list <- list(
  n_obs = db_ovarian_scaled %>% filter(fustat == 1) %>% nrow(),
  n_cens = db_ovarian_scaled %>% filter(fustat == 0) %>% nrow(),
  y_obs = db_ovarian_scaled %>% filter(fustat == 1) %>% .[["futime"]],
  y_cens = db_ovarian_scaled %>% filter(fustat == 0) %>% .[["futime"]],
  k = ncol(db_ovarian_scaled) - 2L,
  x_obs = db_ovarian_scaled %>%
    filter(fustat == 1) %>%
    dplyr::select(age:rx) %>%
    as.matrix(),
  x_cens = db_ovarian_scaled %>%
    filter(fustat == 0) %>%
    dplyr::select(age:rx) %>%
    as.matrix()
)

# 1) Exploratory data analysis -----------------------------------------
# Draw two simple Kaplan-Meier for residual disease and treatment
km_trt <- survfit(Surv(futime, fustat) ~ rx, data = db_ovarian)
km_resid_ds <- survfit(
  Surv(futime, fustat) ~ resid.ds, data = db_ovarian
)

km_ecog_ps <- survfit(
  Surv(futime, fustat) ~ ecog.ps, data = db_ovarian
)

p_trt <- ggsurvplot(
  km_trt, data = db_ovarian, risk.table = TRUE, pval = FALSE,
  conf.int = TRUE, xlab = "Follow-up time in days",
  surv.median.line = "hv", risk.table.col = "strata",
  legend.labs = c("Cyclo", "Cyclo + Adria"),
  risk.table.height = 0.25, ggtheme = theme_bw(),
  main = "Treatment groups"
)

p_resid_ds <- ggsurvplot(
  km_resid_ds, data = db_ovarian, risk.table = TRUE, pval = FALSE,
  conf.int = TRUE, xlab = "Follow-up time in days",
  surv.median.line = "hv", risk.table.col = "strata",
  palette = c("dodgerblue2", "firebrick"), legend.labs = c("No", "Yes"),
  risk.table.height = 0.25, ggtheme = theme_bw(),
  main = "Residual Disease"
)

p_ecog_ps <- ggsurvplot(
  km_ecog_ps, data = db_ovarian, risk.table = TRUE, pval = FALSE,
  conf.int = TRUE, xlab = "Follow-up time in days",
  surv.median.line = "hv", risk.table.col = "strata",
  legend.labs = c("ECOG = 1", "ECOG = 2"),
  palette = c("#E7B800", "#2E9FDF"),
  risk.table.height = 0.25, ggtheme = theme_bw(),
  main = "ECOG performance"
)

# 2) Prior Predictive Checking -----------------------------------------
# Simulate fake data in R and check if the weibull model can recover
# parameter values
n_obs <- 46L
n_cens <- 54L

# 2A) Generate fake data -----------------------------------------------
weibull_fake_comp <- stan_model(
  file = "lab/survival_case_study/stan_programs/weibull_dgp.stan"
)

fitted_weibull_fake <- sampling(
  object = weibull_fake_comp, data = list(n_obs = 46, n_cens = 54),
  chains = 1L, cores = 1L, iter = 1L, algorithm = "Fixed_param",
  seed = mcmc_seed
)

fake_data_weibull <- rstan::extract(fitted_weibull_fake)

# Prepare fake data for stan program
fake_data_db <- data_frame(
  age = fake_data_weibull$age[1, ],
  resid_ds = fake_data_weibull$resid_ds[1, ],
  ecog_ps = fake_data_weibull$ecog_ps[1, ],
  rx = fake_data_weibull$rx[1, ],
  fustat = c(rep(1, n_obs), rep(0, n_cens)),
  futime = fake_data_weibull$futime[1, ]
)

weibull_fake_list <- list(
  n_obs = n_obs, n_cens = n_cens,
  y_obs = fake_data_db %>% filter(fustat == 1) %>% .[["futime"]],
  y_cens = fake_data_db %>% filter(fustat == 0) %>% .[["futime"]],
  k = ncol(fake_data_db) - 2L,
  x_obs = fake_data_db %>%
    filter(fustat == 1) %>%
    dplyr::select(age:rx) %>%
    as.matrix(),
  x_cens = fake_data_db %>%
    filter(fustat == 0) %>%
    dplyr::select(age:rx) %>%
    as.matrix()
)

# 2B) Fit the model to the fake data -----------------------------------
weibull_centered_comp <- stan_model(
  file = "lab/survival_case_study/stan_programs/weibull_centered.stan"
)

fitted_weibull_centered_fake <- sampling(
  object = weibull_centered_comp,
  data = weibull_fake_list,
  warmup = 1000L,
  iter = 2000L,
  chains = 4L,
  seed = mcmc_seed
)

# Did the model recover the parameters?
post_fake_weibull_centered <- as.matrix(
  fitted_weibull_centered_fake,
  pars = c("alpha", "beta_0", "beta")
)

true_fake_weibull_centered <- c(
  fake_data_weibull$alpha, fake_data_weibull$beta_0,
  fake_data_weibull$beta_age, fake_data_weibull$beta_resid_ds,
  fake_data_weibull$beta_ecog_ps, fake_data_weibull$beta_rx
)

recover_fake_weibull_centered <- mcmc_recover_hist(
  x = post_fake_weibull_centered, true = true_fake_weibull_centered
)

# Yessssss!!!!!!

# 3) Fit the model to the real data ------------------------------------
fitted_weibull_centered <- sampling(
  object = weibull_centered_comp,
  data = ovarian_scaled_list,
  warmup = 1000L,
  iter = 2000L,
  chains = 4L,
  seed = mcmc_seed
)

post_weibull_centered <- as.array(fitted_weibull_centered)

# 4) Posterior Predictive checks ---------------------------------------
coef_weibull_centered <- as.data.frame(
  fitted_weibull_centered, pars = c("alpha", "beta_0", "beta")
) %>%
  setNames(
    object = .,
    nm = c("alpha", "intercept", colnames(ovarian_scaled_list$x_obs))
  )

# Simulated quantiles follow-up times seems to be too greater than those
# observed. We can revise the model by changing family distribution:
# we can try a log-normal or a Gamma family

# 5) Revise the model --------------------------------------------------
# 5A) Lognormal --------------------------------------------------------
lognormal_comp <- stan_model(
  file = "lab/survival_case_study/stan_programs/lognormal.stan"
)

fitted_lognormal <- sampling(
  object = lognormal_comp,
  data = ovarian_scaled_list,
  warmup = 1000L,
  iter = 2000L,
  chains = 4L,
  seed = mcmc_seed
)

coef_lognormal <- as.data.frame(
  fitted_lognormal, pars = c("sigma", "beta_0", "beta")
) %>%
  setNames(
    object = .,
    nm = c("sigma", "intercept", colnames(ovarian_scaled_list$x_obs))
  )

# 5B) Gamma ------------------------------------------------------------
gamma_comp <- stan_model(
  file = "lab/survival_case_study/stan_programs/gamma.stan"
)

fitted_gamma <- sampling(
  object = gamma_comp,
  data = ovarian_scaled_list,
  warmup = 1000L,
  iter = 2000L,
  chains = 4L,
  seed = mcmc_seed
)

coef_gamma <- as.data.frame(
  fitted_gamma, pars = c("shape", "beta_0", "beta")
) %>%
  setNames(
    object = .,
    nm = c("shape", "intercept", colnames(ovarian_scaled_list$x_obs))
  )

# 6) Compare models ----------------------------------------------------
model_list_surv <- list(
  fitted_weibull_centered, fitted_lognormal, fitted_gamma
) %>%
  setNames(object = ., nm = c("weibull", "lognormal", "gamma"))

comp_model_list_surv <- map(
  .x = model_list_surv, ~ model_comparison_loo(stan_fit = .x)
) %>%
  setNames(
    object = ., nm = c("weibull", "lognormal", "gamma")
  )

loo_list_surv <- map(.x = comp_model_list_surv, ~ .x[["loo"]])

stacking_weights_surv <- loo_model_weights(loo_list_surv)

pseudo_bma_surv <- loo_model_weights(
  loo_list_surv, method = "pseudobma", BB = FALSE
)

pseudo_bma_bb_surv <- loo_model_weights(
  loo_list_surv, method = "pseudobma", BB = TRUE
)

# Lognormal seems to be better in this case...

# 7) Predictions with the model ----------------------------------------
# We will use lognormal model to describe the effect of the treatment.
# To do this, I need to define some time of the follow-up of a future
# study and compare this time with time to event. If time-to-event
# is higher than time of follow-up, then the patient is dead, otherwise
# it is right censored
pred_df <- expand.grid(
  age = median(db_ovarian_scaled$age),
  redis_ds = max(db_ovarian_scaled$resid_ds),
  ecog_ps = max(db_ovarian_scaled$ecog_ps),
  rx = c(-0.5, 0.5),
  study_time = seq(from = 0, to = 3, by = 0.1)
) %>%
  mutate(sub_id = seq_along(1:nrow(.)))

# Pass everything into stan as a list
ovarian_scaled_pred_list <- list(
  n_obs = db_ovarian_scaled %>% filter(fustat == 1) %>% nrow(),
  n_cens = db_ovarian_scaled %>% filter(fustat == 0) %>% nrow(),
  y_obs = db_ovarian_scaled %>% filter(fustat == 1) %>% .[["futime"]],
  y_cens = db_ovarian_scaled %>% filter(fustat == 0) %>% .[["futime"]],
  k = ncol(db_ovarian_scaled) - 2L,
  x_obs = db_ovarian_scaled %>%
    filter(fustat == 1) %>%
    dplyr::select(age:rx) %>%
    as.matrix(),
  x_cens = db_ovarian_scaled %>%
    filter(fustat == 0) %>%
    dplyr::select(age:rx) %>%
    as.matrix(),
  n_new = nrow(pred_df),
  x_new = as.matrix(pred_df %>% dplyr::select(age:rx)),
  times = pred_df[["study_time"]]
)

lognormal_pred_comp <- stan_model(
  file = "lab/survival_case_study/stan_programs/lognormal_pred.stan"
)

fitted_lognormal_pred <- sampling(
  object = lognormal_pred_comp,
  data = ovarian_scaled_pred_list,
  warmup = 1000L,
  iter = 2000L,
  chains = 4L,
  seed = mcmc_seed
)

# Get the predicted values and compute 80% posterior intervals
surv_pred <- as.data.frame(
  x = fitted_lognormal_pred, pars = c("surv_pred")
) %>%
  map_df(
    .x = .,
    ~ data_frame(
      post_median = median(.x),
      lower_80 = quantile(.x, probs = 0.10),
      upper_80 = quantile(.x, probs = 0.90)
    )
  )

post_surv_pred <- bind_cols(pred_df, surv_pred) %>%
  rename(treatment = rx) %>%
  mutate(
    treatment = factor(
      case_when(
        treatment == -0.5 ~ "cyclo",
        treatment == 0.5 ~ "cyclo + adria"
      )
    )
  )

# Plot the survival curves with relative 80% posterior intervals
lognormal_survplot <- ggplot(
  data = post_surv_pred, mapping = aes(
    x = study_time, y = post_median, colour = treatment,
    fill = treatment
  )
) +
  geom_line() +
  geom_ribbon(
    mapping = aes(ymin = lower_80, ymax = upper_80), alpha = 0.2
  ) +
  scale_colour_discrete(
    name = "Group"
  ) +
  scale_fill_discrete(
    name = "Group"
  ) +
  xlab("Follow-up time (years)") +
  ylab("Survival rate") +
  theme_bw()

# Save everything for the slides ---------------------------------------
save(
  p_trt, p_resid_ds, p_ecog_ps,
  recover_fake_weibull_centered, model_list_surv, loo_list_surv,
  stacking_weights_surv, pseudo_bma_surv, pseudo_bma_bb_surv,
  fitted_lognormal_pred, lognormal_survplot,
  file = here::here("lab/survival_case_study/survival_analysis.rda")
)

# IBIG congress: birthweight example -----------------------------------
library(tidyverse)
library(rstan)
  options(mc.cores = parallel::detectCores() - 1)
  rstan_options(auto_write = TRUE)
  Sys.setenv(LOCAL_CPPFLAGS = c('-march=native'))
library(bayesplot)
library(loo)
library(MASS)
library(knitr)

mcmc_seed <- 140509

source(here::here("lab/logistic_case_study/logit_misc_functions.R"))

# Birth weight dataset -------------------------------------------------
db <- birthwt %>%
  mutate_at(vars(age, lwt, bwt), funs(as.double)) %>%
  mutate(
    bwt = bwt/1000, lwt = (lwt * 0.453592)/100, age = age/10
  ) %>%
  dplyr::select(-low) %>%
  mutate_at(
    vars(age:ftv), funs(as.numeric(scale(x = ., scale = FALSE)))
  )

db_list <- list(
  N = nrow(db), K = 4L,
  X = db %>% dplyr::select(age, smoke, ui, lwt) %>% as.matrix(),
  Y = db[["bwt"]]
)

# Prior Predictive Checks ----------------------------------------------
dgp_comp_vague <- stan_model(
  file = "DB_presentation/stan_programs/vague_dgp.stan"
)

dgp_comp_reg <- stan_model(
  file = "DB_presentation/stan_programs/regularized_dgp.stan"
)

fake_vague <- sampling(
  object = dgp_comp_vague, data = list(N = 189L),
  chains = 1L, cores = 1L, iter = 1L, algorithm = "Fixed_param",
  seed = mcmc_seed
)

fake_reg <- sampling(
  object = dgp_comp_reg, data = list(N = 189L),
  chains = 1L, cores = 1L, iter = 1L, algorithm = "Fixed_param",
  seed = mcmc_seed
)

fake_vague_post <- rstan::extract(fake_vague)
fake_reg_post <- rstan::extract(fake_reg)

bwt_rep_fake_vague <- fake_vague_post$bwt[1, ]
bwt_rep_fake_reg <- fake_reg_post$bwt[1, ]

obs_fake_plot_vague <- ggplot(
  mapping = aes(x = db$bwt, y = bwt_rep_fake_vague)
) +
  geom_point(color = "forestgreen") +
  xlab("Observed birth weights") +
  ylab("Simulated birth weights") +
  theme_bw()

obs_fake_plot_reg <- ggplot(
  mapping = aes(x = db$bwt, y = bwt_rep_fake_reg)
) +
  geom_point(color = "forestgreen") +
  xlab("Observed birth weights") +
  ylab("Simulated birth weights") +
  theme_bw()

# Fitting models to real data -----------------------------------------
vague_comp <- stan_model(
  file = "DB_presentation/stan_programs/vague.stan"
)

reg_comp <- stan_model(
  file = "DB_presentation/stan_programs/regularized.stan"
)

fitted_vague <- sampling(
  object = vague_comp, data = db_list,
  chains = 4L, warmup = 1000L, iter = 2000L,
  seed = mcmc_seed
)

fitted_reg <- sampling(
  object = reg_comp, data = db_list,
  chains = 4L, warmup = 1000L, iter = 2000L,
  seed = mcmc_seed
)

design_mat_int <- db %>%
  dplyr::select(age, smoke, ui, lwt) %>%
  model.matrix(object = ~ age + smoke + ui * lwt, data = .) %>%
  .[, -1]

db_int_list <- list(
  N = nrow(db), K = 5L,
  X = design_mat_int,
  Y = db[["bwt"]]
)

fitted_reg_int <- sampling(
  object = reg_comp, data = db_int_list,
  chains = 4L, warmup = 1000L, iter = 2000L,
  seed = mcmc_seed
)

# Model comparison -----------------------------------------------------
model_list_bwt <- list(fitted_vague, fitted_reg, fitted_reg_int) %>%
  setNames(object = ., nm = c("vague", "reg", "reg_int"))

comp_model_list <- map(
  .x = model_list_bwt, ~ model_comparison_loo(stan_fit = .x)
)

loo_list_bwt <- map(.x = comp_model_list, ~ .x[["loo"]])

stacking_weights_bwt <- loo_model_weights(loo_list_bwt)

pseudo_bma_bwt <- loo_model_weights(
  loo_list_bwt, method = "pseudobma", BB = FALSE
)

pseudo_bma_bb_bwt <- loo_model_weights(
  loo_list_bwt, method = "pseudobma", BB = TRUE
)

# Save everything for the slides ---------------------------------------
save(
  model_list_bwt, loo_list_bwt, stacking_weights_bwt, fake_vague_post,
  fake_reg_post, pseudo_bma_bwt, pseudo_bma_bb_bwt, obs_fake_plot_vague,
  obs_fake_plot_reg,
  file = here::here("DB_presentation/bwt_example.rda")
)


# Logistic case study miscellaneous functions --------------------------
# Define marginal probabilities for binary confounders -----------------
marginal_prob_conf <- function(latent_x, cut_off){

  transf_latent <- dplyr::if_else(latent_x < cut_off, 1, 0)

  dplyr::data_frame(
    cut_off = cut_off, marg_prob = mean(transf_latent)
  )

}

# Gives the cutoff that returns the defined marginal probabilitiy ------
define_cut_off_conf <- function(
  latent_x, cut_off_list, marg_prob, delta
){

  list_marg_prob <- purrr::map_df(
    .x = cut_off_list, ~ marginal_prob_conf(
      latent_x = latent_x, cut_off = .x
    )
  )

  selected_row <- which(list_marg_prob$marg_prob >= marg_prob - delta &
    list_marg_prob$marg_prob <= marg_prob + delta)[1]

  list_marg_prob[[selected_row, 1]]

}

# Return the binary confounders given latent gaussian ------------------
return_binary_conf <- function(
  latent_x, cut_off_list, marg_prob, delta
){

  cut_off <- define_cut_off_conf(
    latent_x = latent_x, cut_off_list = cut_off_list,
    marg_prob = marg_prob, delta = delta
  )

  dplyr::if_else(latent_x < cut_off, 1, 0)

}

# Return binary confounders given a latent gaussian df -----------------
return_binary_conf_df <- function(
  latent_df, cut_off_list, marg_prob_list, delta
){

  purrr::map2_df(
    .x = latent_df, .y = marg_prob_list, ~ return_binary_conf(
      latent_x = .x, cut_off_list = cut_off_list,
      marg_prob = .y, delta = delta
    )
  )

}

# Return the the outocome given an intercept ---------------------------
rate_logit <- function(x, betas, intercept) {

  coefs <- c(intercept, betas)
  x_mat <- cbind(rep(1, nrow(x)), as.matrix(x))
  eta <-  x_mat %*% coefs

  prob <- exp(eta)/(1 + exp(eta))

  set.seed(140509)    # seed for reproducibility
  outcome <- stats::rbinom(n = length(prob), size = 1, prob = prob)

  dplyr::data_frame(intercept = intercept, rate = mean(outcome))
}

# Choose intercept for a logit model given the rate --------------------
choose_intercept_logit <- function(
  x, betas, intercept_grid, given_rate, delta
) {

  list_int_rate <- purrr::map_df(
    .x = intercept_grid,
    ~ rate_logit(x = x, betas = betas, intercept = .x)
  ) %>%
    dplyr::filter(
      rate >= given_rate - delta & rate <= given_rate + delta
    ) %>%
    dplyr::sample_n(size = 1L)

  list_int_rate[["intercept"]]
}

# Model comparison with LOO --------------------------------------------
model_comparison_loo <- function(stan_fit){

  # Extract log likelihood from stan object
  log_lik <- extract_log_lik(stanfit = stan_fit, merge_chains = FALSE)

  # Get relative effective sample sizes
  r_eff <- relative_eff(exp(log_lik))

  # Compute loo
  loo <- loo(log_lik, r_eff = r_eff)

  list("log_lik" = log_lik, "r_eff" = r_eff, "loo" = loo)

}

# Compute relative survival lognormal ----------------------------------
lognorm_surv <- function(time, sigma) {

  assertive::is_non_negative(time)
  assertive::is_positive(sigma)

  1 - stats::pnorm(q = log(time)/sigma)

}

# Predictions survival -------------------------------------------------
post_pred_surv <- function(pred_df, time_fu_draw) {

  sigma <- time_fu_draw[["sigma"]]

  time_event <- time_fu_draw %>%
    dplyr::select(contains("y_pred")) %>%
    t() %>%
    as_data_frame() %>%
    .[[1]]

  pred_surv_df <- cbind(pred_df, time_fu_draw) %>%
    dplyr::mutate(
      time_to_event = time_event,
      fustat = if_else(time_to_event < time_stop, 1, 0),
      futime = if_else(fustat == 1, time_to_event, time_stop)
    )

  fu_time <- pred_surv_df[["futime"]]

  map_dbl(
    .x = fu_time, ~ lognorm_surv(time = .x, sigma = sigma)
  )

}



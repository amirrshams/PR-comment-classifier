# R script file to run the comp dataset model

#importing libraries
library(readr)
library(dplyr)
library(forcats)
library(lme4)
library(glmnet)
library(optimx)

library(car)
library(stargazer)
library(texreg)
library(plyr)
library(xtable)
library(splitstackshape)
library(scales)

library(Hmisc)
library(lmerTest)
library(e1071)
library(ggplot2)

#loading the data
df_sampled <- read_csv("/home/a2shamso/projects/def-m2nagapp/a2shamso/pr_classification/dataset/sampled_data.csv")
df_comp <- read_csv("/home/a2shamso/projects/def-m2nagapp/a2shamso/pr_classification/dataset/pr_final_April_2023.csv")
df_TSE <- read_csv("/home/a2shamso/projects/def-m2nagapp/a2shamso/pr_classification/dataset/pull_requests.csv")

#getting other important features
df_comp <- left_join(df_comp, select(df_TSE, pr_id, prs_pri_same_nationality, prs_experience, prs_succ_rate, prs_popularity, prs_watched_repo, prs_followed_pri, prs_tenure_mnth, prs_main_team_member, repo_pr_tenure_mnth, repo_pr_popularity,repo_pr_team_size, perc_external_contribs, pr_opened_at, pr_files_changed, pr_lines_changed, intra_branch) , by= "pr_id")


# Building the model for all merged and nonmerged pull reqests

#preprocessing the data

preprocess_data_norm <- function(data) {
  # Convert status to factor
  data$status <- factor(data$status, levels = c("merged", "not-merged"))

  # List of numeric variables to apply log transformation and robust scaling
  numeric_vars <- c("comments_counts", "commit_counts", "prs_experience", "prs_succ_rate", "prs_popularity", "prs_tenure_mnth", "repo_pr_tenure_mnth", "repo_pr_popularity", "repo_pr_team_size", "perc_external_contribs", "pr_lines_changed")

  # Apply log transformation and robust scaling
  for (var in numeric_vars) {
    # Log transformation (adding 1 to avoid log(0))
    data[[var]]<-as.integer(data[[var]])
    #data[[var]] <- scale(log(data[[var]] + 1))
 
    #square root
    data[[var]] <- sqrt(data[[var]])

  }

  # Convert binary variables to factors
  binary_vars <- c("prs_main_team_member", "prs_watched_repo", "prs_followed_pri", "intra_branch", "pr_files_changed")
  for (var in binary_vars) {
    data[[var]] <- factor(data[[var]], levels = c(0, 1))
  }

  return(data)
}
# unique_repo_ids_sampled <- unique(df_sampled$repo_id)



df_comp_factored <- preprocess_data_norm(df_comp)


# df_comp_factored_filtered<- df_comp_factored[df_comp_factored$repo_id %in% unique_repo_ids_sampled, ]


formula <- status ~  comments_counts + commit_counts + prs_experience + prs_succ_rate + prs_popularity + prs_watched_repo + prs_tenure_mnth + prs_main_team_member + repo_pr_tenure_mnth + prs_followed_pri + repo_pr_popularity + repo_pr_team_size + perc_external_contribs + intra_branch + pr_files_changed + pr_lines_changed + (1 | repo_id) + (1 | author_id)

model_all <- glmer(formula, data = df_comp_factored, family = binomial,
                        control = glmerControl(optimizer = "nloptwrap", calc.derivs = FALSE, optCtrl = list(maxeval = 50)))

model_all_summary <- summary(model_all)
fixed_effects <- fixef(model_all)
odds_ratios <- exp(fixed_effects)
#conf_intervals <- confint(bootstrapped_model_alls[[i]], which = "fixed")
print('The model_all summary and correlation:')
print(model_all_summary, correlation=TRUE)
print('Odds Ratio:')
print(odds_ratios)
print('VIF:')
car::vif(model_all)

create_model_summary_df <- function(model_all) {
  # Extract model_all coefficients
  coefs <- summary(model_all)$coefficients
  
  # Create a dataframe from the coefficients
  df <- as.data.frame(coefs)
  
  # Calculate odds ratios for fixed effects
  df$odds_ratio <- exp(df[, "Estimate"])
  
  # Add row names as a new column for the terms
  df$term <- rownames(df)
  
  # Determine significance levels
  df$Significance <- ifelse(df[, "Pr(>|z|)"] < 0.001, '***',
                            ifelse(df[, "Pr(>|z|)"] < 0.01, '**',
                            ifelse(df[, "Pr(>|z|)"] < 0.05, '*',
                            ifelse(df[, "Pr(>|z|)"] < 0.1, '.', ' '))))
  
  # Return the dataframe
  return(df)
}

# Usage
model_all_summary_df <- create_model_summary_df(model_all)
write.csv(model_all_summary_df, "/home/a2shamso/projects/def-m2nagapp/a2shamso/pr_classification/model_all_square_df.csv", row.names = FALSE)


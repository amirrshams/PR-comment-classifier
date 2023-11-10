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
df_comp <- read_csv("/home/a2shamso/projects/def-m2nagapp/a2shamso/pr_classification/dataset/pr_final_April_2023.csv")
df_TSE <- read_csv("/home/a2shamso/projects/def-m2nagapp/a2shamso/pr_classification/dataset/pull_requests.csv")

#getting other important features
df_comp <- left_join(df_comp, select(df_TSE, pr_id, prs_pri_same_nationality, prs_experience, prs_succ_rate, prs_popularity, prs_watched_repo, prs_followed_pri, prs_tenure_mnth, prs_main_team_member, repo_pr_tenure_mnth, repo_pr_popularity,repo_pr_team_size, perc_external_contribs, pr_opened_at, pr_files_changed, pr_lines_changed, intra_branch) , by= "pr_id")


# Building the model for all merged and nonmerged pull reqests

#preprocessing the data

preprocess_data_norm <- function(data){
  
  data$status <- factor(data$status, levels = c("merged", "not-merged"))

  data$closer_country <- factor(data$closer_country)                                    
  data$author_country <- factor(data$author_country)                                    
  data$author_continent <- factor(data$author_continent,levels=c("Asia","Africa","South America","Antarctica", "Unknown", "North America", "Europe", "Oceania"))
  
  data$comments_counts<-as.integer(data$comments_counts)
  data$comments_counts<-scale(log(data$comments_counts +1))
  data$commit_counts<-as.integer(data$commit_counts)
  data$commit_counts<-scale(log(data$commit_counts +1))
  data$code_changes_counts<-as.integer(data$code_changes_counts)
  data$code_changes_counts<-scale(log(data$code_changes_counts +1))
  data$prs_experience<-as.integer(data$prs_experience)
  data$prs_experience<-scale(log(data$prs_experience +1))
  data$prs_succ_rate<-as.integer(data$prs_succ_rate)
  data$prs_succ_rate<-scale(log(data$prs_succ_rate +1))
  data$prs_popularity<-as.integer(data$prs_popularity)
  data$prs_popularity<-scale(log(data$prs_popularity +1))
  data$prs_tenure_mnth<-as.integer(data$prs_tenure_mnth)
  data$prs_tenure_mnth<-scale(log(data$prs_tenure_mnth +1))
  data$repo_pr_tenure_mnth<-as.integer(data$repo_pr_tenure_mnth)
  data$repo_pr_tenure_mnth<-scale(log(data$repo_pr_tenure_mnth +1))
  data$repo_pr_popularity<-as.integer(data$repo_pr_popularity)
  data$repo_pr_popularity<-scale(log(data$repo_pr_popularity +1))
  data$repo_pr_team_size<-as.integer(data$repo_pr_team_size)
  data$repo_pr_team_size<-scale(log(data$repo_pr_team_size +1))
  data$perc_external_contribs<-as.integer(data$perc_external_contribs)
  data$perc_external_contribs<-scale(log(data$perc_external_contribs +1))
  data$pr_lines_changed<-as.integer(data$pr_lines_changed)
  data$pr_lines_changed<-scale(log(data$pr_lines_changed +1))
                                      

  data$prs_main_team_member<-factor(data$prs_main_team_member,levels=c(0,1))
  data$prs_watched_repo<-factor(data$prs_watched_repo,levels=c(0,1))
  data$prs_followed_pri<-factor(data$prs_followed_pri,levels=c(0,1))
  data$same_eth<-factor(data$same_eth,levels=c(0,1))
  data$same_country<-factor(data$same_country,levels=c(0,1))
  data$intra_branch<-factor(data$intra_branch,levels=c(0,1))
  data$pr_files_changed<-factor(data$pr_files_changed,levels=c(0,1))
  data$intra_branch<-factor(data$intra_branch,levels=c(0,1))

  
  
    # List of columns to normalize
  # columns_to_normalize <- c("comments_counts", "commit_counts", "code_changes_counts", "prs_experience", 
  #                           "prs_succ_rate", "prs_popularity", "prs_tenure_mnth", "repo_pr_tenure_mnth", 
  #                           "repo_pr_popularity", "repo_pr_team_size", "perc_external_contribs", 
  #                           "pr_files_changed", "pr_lines_changed")
  
  # for (col in columns_to_normalize) {
  #   data[[col]] <- rescale(data[[col]])
  # }

  return(data)

}

df_comp_factor <- preprocess_data_norm(df_comp)


formula <- status ~  comments_counts + commit_counts + code_changes_counts + prs_experience + prs_succ_rate + prs_popularity + prs_watched_repo + prs_tenure_mnth + prs_main_team_member +
  repo_pr_tenure_mnth + prs_followed_pri + repo_pr_popularity + repo_pr_team_size + perc_external_contribs + intra_branch + pr_files_changed + pr_lines_changed + (1 | repo_id) + (1 | author_id)


model_all <- glmer(formula, data = df_comp_factor, family = binomial,
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
write.csv(model_all_summary_df, "/home/a2shamso/projects/def-m2nagapp/a2shamso/pr_classification/model_all_df.csv", row.names = FALSE)


# R script file to run the big dataset model

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

library(Hmisc)
library(lmerTest)
library(e1071)
library(ggplot2)

#loading the data
df_comp <- read_csv("/Users/amirrshams/Library/CloudStorage/OneDrive-UniversityofWaterloo/Thesis/Dataset/Dataset/pr_final_April_2023.csv")
df_TSE <- read_csv("/Users/amirrshams/Library/CloudStorage/OneDrive-UniversityofWaterloo/Thesis/Dataset/Reza's Dataset/TSE paper/2020-TSE-Developers-Perceptible-Ethnicity-and-PR-evaluation-main/Dataset/pull_requests.csv")

#getting other important features
df_comp <- left_join(df_comp, select(df_TSE, pr_id, prs_pri_same_nationality, prs_experience, prs_succ_rate, prs_popularity, prs_watched_repo, prs_followed_pri, prs_tenure_mnth, prs_main_team_member, repo_pr_tenure_mnth, repo_pr_popularity,repo_pr_team_size, perc_external_contribs, pr_opened_at, pr_files_changed, pr_lines_changed, intra_branch) , by= "pr_id", all.y = TRUE)


# Building the model for all merged and nonmerged pull reqests

#preprocessing the data
df_comp_factor <- df_comp


df_comp_factor$status<-factor(df_comp_factor$status,levels=c("not-merged","merged"))

df_comp_factor$repo_pr_tenure_mnth<-as.integer(df_comp_factor$repo_pr_tenure_mnth)
df_comp_factor$repo_pr_tenure_mnth<-scale(log(df_comp_factor$repo_pr_tenure_mnth +1))
df_comp_factor$repo_pr_popularity<-as.integer(df_comp_factor$repo_pr_popularity)
df_comp_factor$repo_pr_popularity<-scale(log(df_comp_factor$repo_pr_popularity +1))
df_comp_factor$repo_pr_team_size<-as.integer(df_comp_factor$repo_pr_team_size)
df_comp_factor$repo_pr_team_size<-scale(log(df_comp_factor$repo_pr_team_size +1))
df_comp_factor$perc_external_contribs<-as.integer(df_comp_factor$perc_external_contribs)
df_comp_factor$perc_external_contribs<-scale(log(df_comp_factor$perc_external_contribs +1))

df_comp_factor$closer_country <- factor(df_comp_factor$closer_country)                                    
df_comp_factor$author_country <- factor(df_comp_factor$author_country)                                    
df_comp_factor$author_continent <- factor(df_comp_factor$author_continent,levels=c("Asia","Africa","South America","Antarctica", "Unknown", "North America", "Europe", "Oceania"))



df_comp_factor$prs_experience<-as.integer(df_comp_factor$prs_experience)
df_comp_factor$prs_experience<-scale(log(df_comp_factor$prs_experience +1))
df_comp_factor$prs_succ_rate<-as.integer(df_comp_factor$prs_succ_rate)
df_comp_factor$prs_succ_rate<-scale(log(df_comp_factor$prs_succ_rate +1))
df_comp_factor$prs_main_team_member<-factor(df_comp_factor$prs_main_team_member,levels=c(0,1))
df_comp_factor$prs_popularity<-as.integer(df_comp_factor$prs_popularity)
df_comp_factor$prs_popularity<-scale(log(df_comp_factor$prs_popularity +1))
df_comp_factor$prs_tenure_mnth<-as.integer(df_comp_factor$prs_tenure_mnth)
df_comp_factor$prs_tenure_mnth<-scale(log(df_comp_factor$prs_tenure_mnth +1))
df_comp_factor$comments_counts<-as.integer(df_comp_factor$comments_counts)
df_comp_factor$comments_counts<-scale(log(df_comp_factor$comments_counts +1))
df_comp_factor$commit_counts<-as.integer(df_comp_factor$commit_counts)
df_comp_factor$commit_counts<-scale(log(df_comp_factor$commit_counts +1))
df_comp_factor$prs_watched_repo<-factor(df_comp_factor$prs_watched_repo,levels=c(0,1))
df_comp_factor$prs_followed_pri<-factor(df_comp_factor$prs_followed_pri,levels=c(0,1))
df_comp_factor$same_eth<-factor(df_comp_factor$same_eth,levels=c(0,1))
df_comp_factor$same_country<-factor(df_comp_factor$same_country,levels=c(0,1))
df_comp_factor$intra_branch<-factor(df_comp_factor$intra_branch,levels=c(0,1))
df_comp_factor$pr_files_changed<-factor(df_comp_factor$pr_files_changed,levels=c(0,1))
df_comp_factor$intra_branch<-factor(df_comp_factor$intra_branch,levels=c(0,1))
df_comp_factor$pr_files_changed<-as.integer(df_comp_factor$pr_files_changed)
df_comp_factor$pr_files_changed<-scale(log(df_comp_factor$pr_files_changed +1))
df_comp_factor$pr_lines_changed<-as.integer(df_comp_factor$pr_lines_changed)
df_comp_factor$pr_lines_changed<-scale(log(df_comp_factor$pr_lines_changed +1))


formula <- status ~  comments_counts + commit_counts + code_changes_counts + prs_pri_same_nationality + prs_experience + prs_succ_rate + prs_popularity + prs_watched_repo + prs_tenure_mnth + prs_main_team_member +
  repo_pr_tenure_mnth + repo_pr_popularity + repo_pr_team_size + perc_external_contribs + intra_branch + pr_files_changed + pr_lines_changed + (1 | repo_id) + (1 | author_id)


model <- glmer(formula, data = df_comp_factor, family = binomial,
                        control = glmerControl(optimizer = "nloptwrap", calc.derivs = FALSE, optCtrl = list(maxeval = 50)))

print(model, correlation=TRUE)


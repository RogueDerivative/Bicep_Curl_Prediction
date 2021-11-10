# 1. Specific Question
## What type of movement are they doing

if(!file.exists("./data")){dir.create("./data")}
url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#download.file(url1, destfile = "./data/pml-training.csv")
url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#download.file(url2, destfile = "./data/pml-testing.csv")
## Data Source
website <- "http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har."

## libraries
library(tidyverse) #*
library(caret)#*
library(tidymodels) #* for the dials fuction
library(parsnip)#*
library(recipes)#*
library(workflows)#*
library(yardstick)#* for the roc_auc
library(vip)
# 2. Input Data
validation <- read_csv("./data/pml-testing.csv")
df <- read_csv("./data/pml-training.csv")
# 3. Features: what to use as predictors
# Looking at the original data we see alot of NA values.
# The user might make a difference so turn them into factors.
# Am not going to use time slices so we can drop the time columns
# making NA explicit
df <- df %>% complete(.)
# Remove the columns with more than 90% NA values
df <- 
        df[,!sapply(df,
                           function(x) mean(is.na(x)))>0.9]
# convert user_name into factors
df$user_name <-
        parse_factor(df$user_name, 
                     levels = c("carlitos", "pedro","adelmo",
                                "charles","eurico","jeremy"),
                     ordered = FALSE)
# get rid of the index, time, and window cols
df <- 
        df %>% 
        select(-(...1 | raw_timestamp_part_1:num_window))
# Split the Data
inTrain = createDataPartition(df$classe, p = 0.8, list = FALSE)
training <- df[inTrain,]
testing <- df[-inTrain,]
# Convert classe into factors
training$classe <- 
        parse_factor(training$classe, 
                     levels = c("A","B","C","D","E"), 
                     ordered = FALSE)
testing$classe <- 
        parse_factor(testing$classe, 
                     levels = c("A","B","C","D","E"), 
                     ordered = FALSE)
# Using the Parsnip package I created two different recipes to pre-process the data. 
# The first recipe was to normalize the data: to account for any variables that
# might of had a large variance by centering them around zero and a standard deviation
# of one. The normalized pre-process also looks for and eliminates any predictors
# that have a variance reall close to zero. The second recipe uses the Principal
# Component Analysis (pca in the rest of the code). Both recipes convert the 
# user names into dummy variables. 
## Recipes 
norm_recipe <-
        recipe(classe ~ ., data = training) %>%
        step_dummy(all_nominal_predictors()) %>%
        step_nzv(all_predictors()) %>%
        step_normalize(all_numeric_predictors())
# Principal Component Analysis Recipe
pca_recipe <-
        recipe(classe ~., data = training) %>%
        step_dummy(all_nominal_predictors()) %>%
        step_pca(all_numeric_predictors(), num_comp = 4)
# Algorithms  
#We analyzed the training data using three different methods: multinomial regression, a tree model, and a random forest model. Each model was run twice: once with the normalized and centered recipe, and then again with the PCA recipe.  

## Multinomial Regression  
mr_model <- 
        multinom_reg(penalty = 0.1) %>%
        set_engine("glmnet") %>%
        set_mode("classification")
# Normalized and Centered
mr_workflow_n <-
        workflow() %>%
        add_model(mr_model) %>%
        add_recipe(norm_recipe)
# Fit the mr n model
set.seed(234)
mr_fit_n <- 
        mr_workflow_n %>%
        fit(data = training)
# Multinomial pca
mr_workflow_pca <- 
        workflow() %>%
        add_model(mr_model) %>%
        add_recipe(pca_recipe)
## Fit the Mutlinomial pca model
set.seed(234)
mr_fit_pca <- 
        mr_workflow_pca %>%
        fit(data = training)
## Tree Model
tune_spec <- 
        decision_tree(cost_complexity = 0.02,
                      tree_depth = 10,
                      min_n = 1000) %>%
        set_engine("rpart") %>%
        set_mode("classification")
# Normalized Tree
tree_wf_n <- 
        workflow() %>%
        add_model(tune_spec) %>%
        add_recipe(norm_recipe)
# Fit the Normalized Tree
set.seed(234)
tree_fit_n <- 
        tree_wf_n %>%
        fit(data=training)
# Tree pca
tree_wf_pca <-
        workflow() %>%
        add_model(tune_spec) %>%
        add_recipe(pca_recipe)
# Fit Tree pca
set.seed(567)
tree_fit_pca <- 
        tree_wf_pca %>%
        fit(data=training)
## Random Forest Model
randf_model <- 
        rand_forest(mtry = 5, trees = 10) %>%
        set_engine("ranger", importance = "impurity") %>%
        set_mode("classification")
# Random Forest pca
randf_wf_pca <-
        workflow() %>%
        add_model(randf_model) %>%
        add_recipe(pca_recipe)
## Fit the Random Forest pca
set.seed(567)
randf_fit_pca <- 
        randf_wf_pca %>%
        fit(data=training)
# Normalized Random Forest
randf_wf_n <-
        workflow() %>%
        add_model(randf_model) %>%
        add_recipe(norm_recipe)
## Fit Normalized Random Forest
set.seed(234)
randf_fit_n <- 
        randf_wf_n %>%
        fit(data=training)
# 5. Parameters - which predictors were used?
mr_n_vip <- mr_fit_n %>%
        extract_fit_parsnip() %>%
        vip()
mr_n_vip$data
mr_pca_vip <- mr_fit_pca %>%
        extract_fit_parsnip() %>%
        vip()
mr_pca_vip$data
tree_n_vip <- tree_fit_n %>%
        extract_fit_parsnip() %>%
        vip()
tree_pca_vip <- tree_fit_pca %>%
        extract_fit_parsnip() %>%
        vip()
randf_n_vip <- randf_fit_n %>%
        extract_fit_parsnip() %>%
        vip()
randf_pca_vip <- randf_fit_pca %>%
        extract_fit_parsnip() %>%
        vip()

# 6. How good were the models?
# Predict using the norm center model
mr_pred_n <- 
        predict(mr_fit_n, testing) %>%
        bind_cols(predict(mr_fit_n, testing, type = "prob")) %>%
        bind_cols(testing %>% select(classe))
# Check the quality of fit
# mr_n_cm <- confusionMatrix(mr_pred_n$classe, testing$classe)
# mr_n_cm$table
# mr_n_cm$overall
mr_n_roc_auc <- 
        mr_pred_n %>%
        roc_auc(truth = testing$classe,
                .pred_A, .pred_B, .pred_C, .pred_D, .pred_E)
mr_n_roc_auc_est <- round(mr_n_roc_auc$.estimate,3)
mr_n_accuracy <-
        mr_pred_n %>%
        accuracy(truth = testing$classe, .pred_class)
mr_n_acc_est <- round(mr_n_accuracy$.estimate, 3)
## predict with the mr pca model
mr_pred_pca <- 
        predict(mr_fit_pca, testing) %>%
        bind_cols(predict(mr_fit_pca, testing, type = "prob")) %>%
        bind_cols(testing %>% select(classe))
## Check the quality of fit
mr_pca_roc_auc <-
        mr_pred_pca %>%
        roc_auc(truth = testing$classe,
                .pred_A, .pred_B, .pred_C, .pred_D, .pred_E)
mr_pca_accuracy <- 
        mr_pred_pca %>%
        accuracy(truth = testing$classe, .pred_class)
mr_pca_roc_auc_est <- round(mr_pca_roc_auc$.estimate,3)
mr_pca_acc_est <- round(mr_pca_accuracy$.estimate ,3)
# Predict tree using the norm center model
tree_pred_n <- 
        predict(tree_fit_n, testing) %>%
        bind_cols(predict(tree_fit_n, testing, type = "prob")) %>%
        bind_cols(testing %>% select(classe))
## Check quality of fit
tree_n_roc_auc <-
        tree_pred_n %>%
        roc_auc(truth = testing$classe, 
                .pred_A, .pred_B, .pred_C, .pred_D, .pred_E)
tree_n_roc_auc_est <- round(tree_n_roc_auc$.estimate, 3)
tree_n_accuracy <-
        tree_pred_n %>%
        accuracy(truth = testing$classe, .pred_class)
tree_n_acc_est <- round(tree_n_accuracy$.estimate, 3)
# Predict tree using the pca recipe
tree_pred_pca <- 
        predict(tree_fit_pca, testing) %>%
        bind_cols(predict(tree_fit_pca, testing, type = "prob")) %>%
        bind_cols(testing %>% select(classe))
tree_pca_roc_auc <-
        tree_pred_pca %>%
        roc_auc(truth = testing$classe,
                .pred_A, .pred_B, .pred_C, .pred_D, .pred_E)
tree_pca_roc_auc_est <- round(tree_pca_roc_auc$.estimate, 3)
tree_pca_accuracy <-
        tree_pred_pca %>%
        accuracy(truth = testing$classe, .pred_class)
tree_pca_acc_est <- round(tree_pca_accuracy$.estimate, 3)
# Predict randf using the pca recipe
randf_predict_pca <- 
        predict(randf_fit_pca, testing) %>%
        bind_cols(predict(randf_fit_pca, testing, type = "prob")) %>%
        bind_cols(testing %>% select(classe))
## Check quality of fit
randf_pca_cm <- 
        confusionMatrix(randf_predict_pca$classe , testing$classe)
randf_pca_roc_auc <-
        randf_predict_pca %>%
        roc_auc(truth = testing$classe,
                .pred_A, .pred_B, .pred_C, .pred_D, .pred_E)
randf_pca_roc_auc_est <- 
        round(randf_pca_roc_auc$.estimate, 3)
randf_pca_accuracy <-
        randf_predict_pca %>%
        accuracy(truth = testing$classe, .pred_class)
randf_pca_acc_est <- 
        round(randf_pca_accuracy$.estimate, 3)
## Predict using the Normailzed Random Forest----------------------------------
randf_pred_n <- 
        predict(randf_fit_n, testing) %>%
        bind_cols(predict(randf_fit_n, testing, type = "prob")) %>%
        bind_cols(testing %>% select(classe))
## Check quality of fit
randf_n_roc_auc <-
        randf_pred_n %>%
        roc_auc(truth = testing$classe,
                .pred_A, .pred_B, .pred_C, .pred_D, .pred_E)
randf_n_accuracy <-
        randf_pred_n %>%
        accuracy(truth = testing$classe, .pred_class)
randf_n_est <-
        round(randf_n_roc_auc$.estimate, 3)
randf_n_acc_est <-
        round(randf_n_accuracy$.estimate, 3)
# Tables of estimate values
my_roc_auc_table <- data.frame(ROC_AUC_estimate = c("Normalized", "PCA"),
                Multinomial = c(mr_n_roc_auc_est, mr_pca_roc_auc_est),
                Tree = c(tree_n_roc_auc_est, tree_pca_roc_auc_est),
                Random_Forest = c(randf_n_est,randf_pca_roc_auc_est))
my_acc_table <- data.frame(accuracy_estimate = c("Normalized", "PCA"),
                           Multinomial = c(mr_n_acc_est, mr_pca_acc_est),
                           Tree = c(tree_n_acc_est, tree_pca_acc_est),
                           Random_Forest = c(randf_n_acc_est,randf_pca_acc_est))
# Test the predictors:  
## Test the randf norm center model
final_randf_n_predict <-
        predict(randf_fit_n, new_data = validation) %>%
        bind_cols(
                predict(randf_fit_n, validation),
                predict(randf_fit_n, validation, type = "prob"))

randf_n_test_table <-
        final_randf_n_predict
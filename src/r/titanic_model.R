# Titanic survival prediction with logistic regression (R version).
#
# Loads train.csv, explores and cleans it, engineers a few features, fits a
# logistic regression with glm(), reports training/validation accuracy, then
# predicts on test.csv. Every step prints what it does.
#
# Run from the repo root:  Rscript src/r/titanic_model.R

suppressPackageStartupMessages(library(dplyr))

data_dir <- file.path("src", "data")
seed <- 42

section <- function(title) {
  line <- strrep("=", 70)
  cat("\n", line, "\n", title, "\n", line, "\n", sep = "")
}

print_rate <- function(df, col) {
  df %>%
    group_by(.data[[col]]) %>%
    summarise(n = n(), survival_rate = round(mean(Survived), 3), .groups = "drop") %>%
    as.data.frame() %>%
    print(row.names = FALSE)
}

# Blank strings (e.g. Embarked, Cabin) are treated as missing.
read_titanic <- function(file) {
  read.csv(file.path(data_dir, file), stringsAsFactors = FALSE, na.strings = c("", "NA"))
}

# ---------------------------------------------------------------------------
# 1. Load the training data
# ---------------------------------------------------------------------------
section("1. Loading train.csv")
train <- read_titanic("train.csv")
cat("Loaded", file.path(data_dir, "train.csv"), "\n")
cat("Shape:", nrow(train), "rows x", ncol(train), "columns\n")
cat("\nStructure:\n")
str(train)

# ---------------------------------------------------------------------------
# 2. Explore the data
# ---------------------------------------------------------------------------
section("2. Exploring the data")
cat("Missing values per column:\n")
print(colSums(is.na(train)))
cat("\nDuplicate rows:", sum(duplicated(train)), "\n")
cat(sprintf("\nOverall survival rate: %.3f\n", mean(train$Survived)))
cat("\nSurvival rate by Sex:\n");      print_rate(train, "Sex")
cat("\nSurvival rate by Pclass:\n");   print_rate(train, "Pclass")
cat("\nSurvival rate by Embarked:\n"); print_rate(filter(train, !is.na(Embarked)), "Embarked")

# ---------------------------------------------------------------------------
# 3. Feature engineering + cleaning
# ---------------------------------------------------------------------------
add_features <- function(df) {
  df %>%
    mutate(
      Title = trimws(sub("^[^,]*,\\s*([^.]+)\\..*$", "\\1", Name)),
      Title = recode(Title, Mlle = "Miss", Ms = "Miss", Mme = "Mrs"),
      Title = ifelse(Title %in% c("Mr", "Mrs", "Miss", "Master"), Title, "Rare"),
      FamilySize = SibSp + Parch + 1,
      IsAlone = as.integer(FamilySize == 1)
    )
}

# Imputation values are learned from train.csv only, then reused on test.csv.
fit_fill_values <- function(df) {
  list(
    age_by_title = df %>% group_by(Title) %>%
      summarise(age_fill = median(Age, na.rm = TRUE), .groups = "drop"),
    age_overall = median(df$Age, na.rm = TRUE),
    fare = median(df$Fare, na.rm = TRUE),
    embarked = names(which.max(table(df$Embarked)))
  )
}

fill_missing <- function(df, fill) {
  df %>%
    left_join(fill$age_by_title, by = "Title") %>%
    mutate(
      Age = coalesce(Age, age_fill, fill$age_overall),
      Fare = coalesce(Fare, fill$fare),
      Embarked = coalesce(Embarked, fill$embarked)
    ) %>%
    select(-age_fill)
}

categorical <- c("Pclass", "Embarked", "Title")
numeric <- c("Age", "Fare", "FamilySize", "IsAlone")

# Keep only model columns and convert categoricals to factors with fixed levels
# (taken from train.csv) so train and test are encoded identically.
prepare <- function(df, levels_list) {
  out <- df[, c(numeric, categorical)]
  for (col in categorical) {
    out[[col]] <- factor(as.character(df[[col]]), levels = levels_list[[col]])
  }
  out
}

section("3. Feature engineering and cleaning")
train <- add_features(train)
cat("Added 'Title' (extracted from Name; Mlle/Ms -> Miss, Mme -> Mrs, uncommon titles -> Rare):\n")
print_rate(train, "Title")
cat("\nAdded 'FamilySize' = SibSp + Parch + 1 and 'IsAlone' = (FamilySize == 1)\n")
print_rate(train, "IsAlone")

fill <- fit_fill_values(train)
cat("\nImputation values learned from train.csv:\n")
cat("  Age  -> median Age for the passenger's Title:\n        ",
    paste(sprintf("%s=%.1f", fill$age_by_title$Title, fill$age_by_title$age_fill), collapse = ", "),
    "\n", sep = "")
cat(sprintf("  Fare -> overall median (%.4f)\n", fill$fare))
cat(sprintf("  Embarked -> most common port ('%s')\n", fill$embarked))
n_age <- sum(is.na(train$Age)); n_emb <- sum(is.na(train$Embarked))
train <- fill_missing(train, fill)
cat("Filled", n_age, "missing Age and", n_emb, "missing Embarked values\n")

cat("\nDropping columns not used as features:\n")
cat(sprintf("  Cabin  -> %.0f%% missing\n", 100 * mean(is.na(train$Cabin))))
cat("  PassengerId, Name, Ticket -> identifiers / free text (Title already taken from Name)\n")
cat("  Sex    -> redundant with Title (Mr/Master = male, Miss/Mrs = female)\n")
cat("  SibSp, Parch -> combined into FamilySize\n")
levels_list <- lapply(setNames(categorical, categorical),
                      function(col) sort(unique(as.character(train[[col]]))))
cat("Numeric features:", paste(numeric, collapse = ", "), "\n")
cat("Categorical features (factors; glm uses the first level as baseline):",
    paste(categorical, collapse = ", "), "\n")
X_all <- prepare(train, levels_list)
y_all <- train$Survived
cat("Model data:", nrow(X_all), "rows x", ncol(X_all), "columns\n")
cat("Missing values remaining:", sum(is.na(X_all)), "\n")

# ---------------------------------------------------------------------------
# 4. Fit logistic regression + training accuracy
# ---------------------------------------------------------------------------
section("4. Logistic regression")

fit_glm <- function(X, y) {
  glm(Survived ~ ., data = cbind(Survived = y, X), family = binomial())
}
predict_class <- function(model, X) {
  as.integer(predict(model, newdata = X, type = "response") >= 0.5)
}
accuracy <- function(y, pred) mean(y == pred)

# Stratified 80/20 split: sample 80% of survivors and 80% of non-survivors.
set.seed(seed)
train_idx <- unlist(lapply(split(seq_along(y_all), y_all),
                           function(idx) sample(idx, round(0.8 * length(idx)))))
val_idx <- setdiff(seq_along(y_all), train_idx)
cat(sprintf("Split train.csv 80/20 (stratified, seed %d): %d training rows, %d validation rows\n",
            seed, length(train_idx), length(val_idx)))

model <- fit_glm(X_all[train_idx, ], y_all[train_idx])
cat("Fitted glm(Survived ~ ., family = binomial) on the training rows\n")
cat(sprintf("Training accuracy (80%% split):   %.4f\n",
            accuracy(y_all[train_idx], predict_class(model, X_all[train_idx, ]))))
cat(sprintf("Validation accuracy (20%% split): %.4f\n",
            accuracy(y_all[val_idx], predict_class(model, X_all[val_idx, ]))))

final_model <- fit_glm(X_all, y_all)
cat(sprintf("\nRefit on all %d rows of train.csv for the final model\n", nrow(X_all)))
cat(sprintf("Training accuracy (full train.csv): %.4f\n",
            accuracy(y_all, predict_class(final_model, X_all))))

cat("\nCoefficients (log-odds; positive = higher survival odds):\n")
coefs <- summary(final_model)$coefficients
print(round(coefs[order(coefs[, "Estimate"]), c("Estimate", "Pr(>|z|)")], 3))
dropped <- names(coef(final_model))[is.na(coef(final_model))]
if (length(dropped) > 0) {
  cat("Dropped as perfectly collinear:", paste(dropped, collapse = ", "), "\n")
}

# ---------------------------------------------------------------------------
# 5. Predict on the test set
# ---------------------------------------------------------------------------
section("5. Predicting on test.csv")
test <- read_titanic("test.csv")
cat("Loaded test.csv:", nrow(test), "rows x", ncol(test), "columns\n")
cat("Missing values per column:\n")
print(colSums(is.na(test)))

test <- add_features(test)
n_missing <- colSums(is.na(test[, c("Age", "Fare", "Embarked")]))
test <- fill_missing(test, fill)
cat(sprintf("\nApplied the same feature engineering; filled %d Age, %d Fare, %d Embarked using train.csv values\n",
            n_missing[["Age"]], n_missing[["Fare"]], n_missing[["Embarked"]]))
X_test <- prepare(test, levels_list)
cat("Test model data:", nrow(X_test), "rows x", ncol(X_test), "columns\n")

test_pred <- predict_class(final_model, X_test)
cat(sprintf("\nPredicted survivors: %d of %d (%.1f%%)\n",
            sum(test_pred), length(test_pred), 100 * mean(test_pred)))
preview <- data.frame(PassengerId = test$PassengerId, Survived = test_pred)
cat("First 10 predictions:\n")
print(head(preview, 10), row.names = FALSE)

# ---------------------------------------------------------------------------
# 6. Test accuracy
# ---------------------------------------------------------------------------
section("6. Test set accuracy")
cat("test.csv has no 'Survived' column (Kaggle keeps those labels private),\n")
cat("so true test accuracy can't be computed here. As the closest available\n")
cat("reference, predictions are compared to gender_submission.csv, Kaggle's\n")
cat("sample answer file (all women survive, all men die).\n")
reference <- read_titanic("gender_submission.csv")
merged <- inner_join(preview, reference, by = "PassengerId", suffix = c("_pred", "_ref"))
cat(sprintf("\nTest accuracy vs gender_submission.csv: %.4f\n",
            accuracy(merged$Survived_ref, merged$Survived_pred)))
cat("This measures agreement with the gender baseline, not correctness. The\n")
cat("validation accuracy in step 4 is the honest estimate on unseen passengers.\n")

section("Done")

# Titanic survival prediction with logistic regression (R version).
#
# Loads train.csv, cleans it, fits a logistic regression with glm(), reports
# training accuracy, then prints survival predictions for test.csv.
#
# Run from the repo root:  Rscript src/r/titanic_model.R

suppressPackageStartupMessages(library(dplyr))

data_dir <- file.path("src", "data")
features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")

section <- function(title) {
  line <- strrep("=", 60)
  cat("\n", line, "\n", title, "\n", line, "\n", sep = "")
}

# Blank strings (e.g. Embarked, Cabin) are read as missing.
read_titanic <- function(file) {
  read.csv(file.path(data_dir, file), na.strings = c("", "NA"))
}

# Fill missing values with train.csv statistics and encode text columns.
clean <- function(df, fill) {
  df |>
    mutate(
      Age = coalesce(.data$Age, fill$Age),
      Fare = coalesce(.data$Fare, fill$Fare),
      Embarked = factor(
        coalesce(.data$Embarked, fill$Embarked),
        levels = c("C", "Q", "S")
      ),
      Sex = as.integer(.data$Sex == "female")
    ) |>
    select(all_of(features))
}

# 1. Load the training data
section("1. Loading train.csv")
train <- read_titanic("train.csv")
cat("Shape:", nrow(train), "rows x", ncol(train), "columns\n")
print(head(train))

# 2. Explore
section("2. Exploring the data")
cat("Missing values per column:\n")
print(colSums(is.na(train)))
cat(sprintf("\nOverall survival rate: %.3f\n", mean(train$Survived)))
for (col in c("Sex", "Pclass", "Embarked")) {
  cat("\nSurvival rate by", col, ":\n")
  print(round(tapply(train$Survived, train[[col]], mean), 3))
}

# 3. Clean
section("3. Cleaning the data")
fill <- list(
  Age = median(train$Age, na.rm = TRUE),
  Fare = median(train$Fare, na.rm = TRUE),
  Embarked = names(which.max(table(train$Embarked)))
)
cat(sprintf("Missing Age      -> train median (%.1f)\n", fill$Age))
cat(sprintf("Missing Fare     -> train median (%.2f)\n", fill$Fare))
cat(sprintf("Missing Embarked -> most common port ('%s')\n", fill$Embarked))
cat("Sex              -> 1 if female, 0 if male\n")
cat("Embarked         -> factor; glm creates EmbarkedQ, EmbarkedS ",
    "(C is the baseline)\n", sep = "")
cat("Dropped PassengerId, Name, Ticket (identifiers) and Cabin (77% missing)\n")
x_train <- clean(train, fill)
y_train <- train$Survived
cat("\nFeatures:", paste(features, collapse = ", "), "\n")
cat("Training data:", nrow(x_train), "rows x", ncol(x_train), "columns,",
    sum(is.na(x_train)), "missing values\n")

# 4. Fit logistic regression
section("4. Logistic regression")
model <- glm(
  Survived ~ .,
  data = cbind(x_train, Survived = y_train),
  family = binomial()
)
cat("Coefficients (positive = higher survival odds):\n")
print(round(coef(model), 3))
train_pred <- as.integer(predict(model, type = "response") >= 0.5)
cat(sprintf("\nTraining accuracy: %.4f\n", mean(train_pred == y_train)))

# 5. Predict on the test set
section("5. Predicting on test.csv")
test <- read_titanic("test.csv")
cat("Shape:", nrow(test), "rows x", ncol(test), "columns\n")
cat("Missing values filled with the train.csv values above:\n")
print(colSums(is.na(test[, c("Age", "Fare", "Embarked")])))
test_prob <- predict(model, newdata = clean(test, fill), type = "response")
test_pred <- as.integer(test_prob >= 0.5)
cat("\nPredictions (1 = survived, 0 = did not survive):\n")
predictions <- data.frame(PassengerId = test$PassengerId, Survived = test_pred)
print(predictions, row.names = FALSE)
cat(sprintf("\nPredicted survivors: %d of %d (%.1f%%)\n",
            sum(test_pred), length(test_pred), 100 * mean(test_pred)))

"""Titanic survival prediction with logistic regression (Python version).

Loads train.csv, cleans it, fits a logistic regression, reports training
accuracy, then prints survival predictions for test.csv.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def clean(df, fill):
    """Fill missing values with train.csv statistics and encode text columns."""
    df = df.copy()
    df["Age"] = df["Age"].fillna(fill["Age"])
    df["Fare"] = df["Fare"].fillna(fill["Fare"])
    df["Embarked"] = df["Embarked"].fillna(fill["Embarked"])
    df["Sex"] = (df["Sex"] == "female").astype(int)
    df["Embarked_Q"] = (df["Embarked"] == "Q").astype(int)
    df["Embarked_S"] = (df["Embarked"] == "S").astype(int)
    return df[FEATURES]


# 1. Load the training data
section("1. Loading train.csv")
train = pd.read_csv(DATA_DIR / "train.csv")
print(f"Shape: {train.shape[0]} rows x {train.shape[1]} columns")
print(train.head().to_string())

# 2. Explore
section("2. Exploring the data")
print("Missing values per column:")
print(train.isna().sum().to_string())
print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
for col in ["Sex", "Pclass", "Embarked"]:
    print(f"\nSurvival rate by {col}:")
    print(train.groupby(col)["Survived"].mean().round(3).to_string())

# 3. Clean
section("3. Cleaning the data")
fill = {
    "Age": train["Age"].median(),
    "Fare": train["Fare"].median(),
    "Embarked": train["Embarked"].mode()[0],
}
print(f"Missing Age      -> train median ({fill['Age']:.1f})")
print(f"Missing Fare     -> train median ({fill['Fare']:.2f})")
print(f"Missing Embarked -> most common port ('{fill['Embarked']}')")
print("Sex              -> 1 if female, 0 if male")
print("Embarked         -> dummy columns Embarked_Q, Embarked_S (C is the baseline)")
print("Dropped PassengerId, Name, Ticket (identifiers) and Cabin (77% missing)")
X_train = clean(train, fill)
y_train = train["Survived"]
print(f"\nFeatures: {FEATURES}")
print(f"Training matrix: {X_train.shape[0]} rows x {X_train.shape[1]} columns, "
      f"{int(X_train.isna().sum().sum())} missing values")

# 4. Fit logistic regression
section("4. Logistic regression")
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("Coefficients (positive = higher survival odds):")
print(pd.Series(model.coef_[0], index=FEATURES).round(3).to_string())
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"\nTraining accuracy: {train_acc:.4f}")

# 5. Predict on the test set
section("5. Predicting on test.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"Shape: {test.shape[0]} rows x {test.shape[1]} columns")
print("Missing values filled with the train.csv values above:")
print(test[["Age", "Fare", "Embarked"]].isna().sum().to_string())
test_pred = model.predict(clean(test, fill))
predictions = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
print("\nPredictions (1 = survived, 0 = did not survive):")
print(predictions.to_string(index=False))
print(f"\nPredicted survivors: {test_pred.sum()} of {len(test_pred)} ({test_pred.mean():.1%})")

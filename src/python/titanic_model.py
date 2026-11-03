"""Titanic survival prediction with logistic regression (Python version).

Loads train.csv, explores and cleans it, engineers a few features, fits a
logistic regression, reports training/validation accuracy, then predicts on
test.csv. Every step prints what it does so the run can be followed from the
terminal.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RANDOM_STATE = 42


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Load the training data
# ---------------------------------------------------------------------------
section("1. Loading train.csv")
train = pd.read_csv(DATA_DIR / "train.csv")
print(f"Loaded {DATA_DIR / 'train.csv'}")
print(f"Shape: {train.shape[0]} rows x {train.shape[1]} columns")
print("\nFirst 5 rows:")
print(train.head().to_string())
print("\nColumn types:")
print(train.dtypes.to_string())

# ---------------------------------------------------------------------------
# 2. Explore the data
# ---------------------------------------------------------------------------
section("2. Exploring the data")
print("Missing values per column:")
print(train.isna().sum().to_string())
print(f"\nDuplicate rows: {train.duplicated().sum()}")
print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
print("\nSurvival rate by Sex:")
print(train.groupby("Sex")["Survived"].mean().round(3).to_string())
print("\nSurvival rate by Pclass:")
print(train.groupby("Pclass")["Survived"].mean().round(3).to_string())
print("\nSurvival rate by Embarked:")
print(train.groupby("Embarked")["Survived"].mean().round(3).to_string())


# ---------------------------------------------------------------------------
# 3. Feature engineering + cleaning
# ---------------------------------------------------------------------------
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
COMMON_TITLES = {"Mr", "Mrs", "Miss", "Master"}


def add_features(df):
    """Create new columns from existing ones (same rules for train and test)."""
    df = df.copy()
    title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace(TITLE_MAP)
    df["Title"] = title.where(title.isin(COMMON_TITLES), "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def fit_fill_values(df):
    """Learn imputation values from the training data only."""
    return {
        "age_by_title": df.groupby("Title")["Age"].median(),
        "age_overall": df["Age"].median(),
        "fare": df["Fare"].median(),
        "embarked": df["Embarked"].mode()[0],
    }


def fill_missing(df, fill):
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Title"].map(fill["age_by_title"]))
    df["Age"] = df["Age"].fillna(fill["age_overall"])
    df["Fare"] = df["Fare"].fillna(fill["fare"])
    df["Embarked"] = df["Embarked"].fillna(fill["embarked"])
    return df


NUMERIC = ["Age", "Fare", "FamilySize", "IsAlone"]
CATEGORICAL = ["Pclass", "Embarked", "Title"]


def to_model_matrix(df, columns=None):
    """One-hot encode categoricals; align to the training columns if given."""
    X = pd.get_dummies(df[NUMERIC + CATEGORICAL], columns=CATEGORICAL,
                       drop_first=True, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


section("3. Feature engineering and cleaning")
train = add_features(train)
print("Added 'Title' (extracted from Name; Mlle/Ms -> Miss, Mme -> Mrs, "
      "uncommon titles -> Rare):")
print(train["Title"].value_counts().to_string())
print("\nSurvival rate by Title:")
print(train.groupby("Title")["Survived"].mean().round(3).to_string())
print("\nAdded 'FamilySize' = SibSp + Parch + 1 and 'IsAlone' = (FamilySize == 1)")
print("Survival rate by IsAlone:")
print(train.groupby("IsAlone")["Survived"].mean().round(3).to_string())

fill = fit_fill_values(train)
print("\nImputation values learned from train.csv:")
print("  Age  -> median Age for the passenger's Title:")
print("        " + ", ".join(f"{t}={a:.1f}" for t, a in fill["age_by_title"].items()))
print(f"  Fare -> overall median ({fill['fare']:.4f})")
print(f"  Embarked -> most common port ('{fill['embarked']}')")
before = train[["Age", "Embarked"]].isna().sum()
train = fill_missing(train, fill)
print(f"Filled {before['Age']} missing Age and {before['Embarked']} missing Embarked values")

print("\nDropping columns not used as features:")
print(f"  Cabin  -> {train['Cabin'].isna().mean():.0%} missing")
print("  PassengerId, Name, Ticket -> identifiers / free text (Title already taken from Name)")
print("  Sex    -> redundant with Title (Mr/Master = male, Miss/Mrs = female)")
print("  SibSp, Parch -> combined into FamilySize")
print(f"Numeric features: {NUMERIC}")
print(f"Categorical features (one-hot encoded, first level dropped): {CATEGORICAL}")
X_all = to_model_matrix(train)
y_all = train["Survived"]
print(f"Model matrix: {X_all.shape[0]} rows x {X_all.shape[1]} columns")
print(f"Columns: {list(X_all.columns)}")
print(f"Missing values remaining: {int(X_all.isna().sum().sum())}")

# ---------------------------------------------------------------------------
# 4. Fit logistic regression + training accuracy
# ---------------------------------------------------------------------------
section("4. Logistic regression")


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


X_tr, X_val, y_tr, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE)
print(f"Split train.csv 80/20 (stratified, seed {RANDOM_STATE}): "
      f"{len(X_tr)} training rows, {len(X_val)} validation rows")

model = make_model().fit(X_tr, y_tr)
print("Fitted StandardScaler + LogisticRegression on the training rows")
print(f"Training accuracy (80% split):   {accuracy_score(y_tr, model.predict(X_tr)):.4f}")
print(f"Validation accuracy (20% split): {accuracy_score(y_val, model.predict(X_val)):.4f}")

final_model = make_model().fit(X_all, y_all)
train_acc = accuracy_score(y_all, final_model.predict(X_all))
print(f"\nRefit on all {len(X_all)} rows of train.csv for the final model")
print(f"Training accuracy (full train.csv): {train_acc:.4f}")

coefs = pd.Series(final_model[-1].coef_[0], index=X_all.columns).sort_values()
print("\nStandardized coefficients (positive = higher survival odds):")
print(coefs.round(3).to_string())

# ---------------------------------------------------------------------------
# 5. Predict on the test set
# ---------------------------------------------------------------------------
section("5. Predicting on test.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"Loaded test.csv: {test.shape[0]} rows x {test.shape[1]} columns")
print("Missing values per column:")
print(test.isna().sum().to_string())

test = add_features(test)
before = test[["Age", "Fare", "Embarked"]].isna().sum()
test = fill_missing(test, fill)
print(f"\nApplied the same feature engineering; filled {before['Age']} Age, "
      f"{before['Fare']} Fare, {before['Embarked']} Embarked using train.csv values")
X_test = to_model_matrix(test, columns=X_all.columns)
print(f"Test model matrix: {X_test.shape[0]} rows x {X_test.shape[1]} columns")

test_pred = final_model.predict(X_test)
print(f"\nPredicted survivors: {test_pred.sum()} of {len(test_pred)} "
      f"({test_pred.mean():.1%})")
preview = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred})
print("First 10 predictions:")
print(preview.head(10).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Test accuracy
# ---------------------------------------------------------------------------
section("6. Test set accuracy")
print("test.csv has no 'Survived' column (Kaggle keeps those labels private),")
print("so true test accuracy can't be computed here. As the closest available")
print("reference, predictions are compared to gender_submission.csv, Kaggle's")
print("sample answer file (all women survive, all men die).")
reference = pd.read_csv(DATA_DIR / "gender_submission.csv")
merged = preview.merge(reference, on="PassengerId", suffixes=("_pred", "_ref"))
agree = accuracy_score(merged["Survived_ref"], merged["Survived_pred"])
print(f"\nTest accuracy vs gender_submission.csv: {agree:.4f}")
print("This measures agreement with the gender baseline, not correctness. The")
print("validation accuracy in step 4 is the honest estimate on unseen passengers.")

section("Done")

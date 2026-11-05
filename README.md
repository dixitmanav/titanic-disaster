# titanic-disaster

This repo predicts which passengers survived the Titanic using logistic regression. It contains two versions of the same model, one in **Python** and one in **R**, and each runs in its own Docker container.

Each script:

1. Loads `train.csv` and prints its shape and first rows.
2. Explores the data: missing values, plus survival rates by sex, class and port.
3. Cleans the data: fills missing Age, Fare and Embarked values, encodes Sex and Embarked, and drops unused columns.
4. Fits a logistic regression and prints the coefficients and the **training accuracy**.
5. Loads `test.csv`, applies the same cleaning, predicts survival for every test passenger, and prints **how many are predicted to survive**.

Every step prints what it is doing, so you can follow everything from the terminal. The dataset is **not** included in this repo. Step 2 below explains how to download it.

---

## Prerequisites

| Tool | Why | Get it |
|---|---|---|
| **Docker** (Docker Desktop on Mac or Windows) | Builds and runs both containers | https://docs.docker.com/get-docker/ |
| **Git** | Clones this repo | https://git-scm.com/downloads |
| **Kaggle account** | Downloads the dataset | https://www.kaggle.com/account/login |

You do **not** need Python or R on your computer, because Docker installs everything inside the containers. Make sure Docker Desktop is **running** before you go on. To check, run `docker --version` and `docker info`; the second command should finish without errors.

---

## Step 1: Clone the repo

```bash
git clone https://github.com/dixitmanav/titanic-disaster.git
cd titanic-disaster
```

Run **every command below from this `titanic-disaster` folder**, the repo root.

---

## Step 2: Download the data

You need two files, `train.csv` and `test.csv`. Put them in `src/data/`.

### Option A: Kaggle website (easiest)

1. Sign in to Kaggle and open the competition page: https://www.kaggle.com/competitions/titanic/data
2. If asked, click **Join Competition** and accept the rules. Kaggle won't let you download the files until you do.
3. Click **Download All** to get `titanic.zip`.
4. Unzip it and move `train.csv` and `test.csv` into `src/data/` inside this repo. The zip also contains `gender_submission.csv`, which these scripts don't use; you can leave it there or delete it.

### Option B: Kaggle command-line tool

This assumes you have the Kaggle CLI installed and set up with an API token (see https://github.com/Kaggle/kaggle-api), and that you've accepted the competition rules on the website.

```bash
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data
```

### Check the layout

Your folder should look like this:

```
titanic-disaster/
└── src/
    └── data/
        ├── train.csv
        └── test.csv
```

On Mac or Linux, `ls src/data` should list `test.csv` and `train.csv`. `src/data/` is in `.gitignore`, so the data is never committed.

---

## Step 3: Run the Python model

```bash
docker build -f src/python/Dockerfile -t titanic-python .
docker run --rm titanic-python
```

- `docker build` creates an image with Python 3.12 and the pinned packages from `src/python/requirements.txt`, then copies in the script and the data. The `.` at the end is required: it tells Docker to build from the repo root so it can find `src/data/`.
- `docker run --rm` runs the script and removes the container once it finishes.

---

## Step 4: Run the R model

```bash
docker build -f src/r/Dockerfile -t titanic-r .
docker run --rm titanic-r
```

- `docker build` starts from `rocker/r-ver:4.5.1` and runs `src/r/install_packages.R` to install `dplyr`. The first build takes a few minutes; later builds reuse the cached layer.
- `docker run --rm` runs `src/r/titanic_model.R`.

---

## What you should see

Both containers print the same five numbered sections. Here's an excerpt from the Python run:

```
============================================================
4. Logistic regression
============================================================
Coefficients (positive = higher survival odds):
Pclass       -1.066
Sex           2.617
Age          -0.038
...
Training accuracy: 0.8002

============================================================
5. Predicting on test.csv
============================================================
...
Predicted survivors: 155 of 418 (37.1%)
```

| | Python | R |
|---|---|---|
| Training accuracy | 0.8002 | 0.7980 |
| Predicted survivors (out of 418) | 155 | 156 |

The two models make the same prediction for 417 of the 418 test passengers. The small difference is because scikit-learn's `LogisticRegression` adds a mild L2 penalty by default, while R's `glm()` doesn't.

**Why there's no test accuracy:** Kaggle keeps the true `Survived` labels for `test.csv` private, so the scripts can't score their own test predictions. They report how many test passengers are predicted to survive instead.

---

## Model details

| Step | What's done |
|---|---|
| Features used | `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked` |
| Missing `Age` | Filled with the training median (28.0) |
| Missing `Fare` (1 test row) | Filled with the training median (14.45) |
| Missing `Embarked` (2 training rows) | Filled with the most common port, `S` |
| `Sex` | Coded as 1 for female, 0 for male |
| `Embarked` | Split into dummy columns for `Q` and `S`, with `C` as the baseline |
| Dropped | `PassengerId`, `Name`, `Ticket` (identifiers) and `Cabin` (77% missing) |
| Model | Python: `sklearn.linear_model.LogisticRegression`. R: `glm(family = binomial)` |

Fill values come **only from `train.csv`** and are reused for `test.csv`, so no test data feeds into the model.

---

## Repo structure

```
titanic-disaster/
├── .github/CODEOWNERS
├── .gitignore              # ignores src/data/, .DS_Store, virtual environments, etc.
├── LICENSE
├── README.md
└── src/
    ├── data/               # put train.csv and test.csv here (not committed)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── titanic_model.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic_model.R
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says it's running. |
| Build fails with `"/src/data": not found` | The data isn't in `src/data/` (Step 2), or you didn't run the command from the repo root. |
| Script fails with `FileNotFoundError` (Python) or `cannot open file` (R) | `train.csv` or `test.csv` is missing or misnamed. File names are case-sensitive. |
| You added or changed the data after building | Run the `docker build` command again; the data is copied in at build time. |
| `unable to prepare context: path "." not found` | You left off the `.` at the end of `docker build`. |

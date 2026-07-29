import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

cell1 = """import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder"""

cell2 = """titanic = sns.load_dataset("titanic")

leaks = ["alive", "who", "adult_male", "class", "embark_town", "alone", "deck"]
df = titanic.drop(columns=leaks)
print(df.shape)
df.head()"""

cell3 = """df["is_alone"] = ((df["sibsp"] + df["parch"]) == 0).astype(int)

# did we rebuild seaborn's "alone" column correctly?
matches = (df["is_alone"] == titanic["alone"].astype(int)).all()
print("is_alone matches the original alone column:", matches)

numeric = ["age", "fare", "sibsp", "parch", "is_alone"]
categorical = ["sex", "embarked"]
ordinal = ["pclass"]"""

cell4 = """X = df.drop(columns="survived")
y = df["survived"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape, X_val.shape)"""

cell5 = """numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])"""

cell6 = """ordinal_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=[[1, 2, 3]])),
])"""

cell7 = """preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", categorical_pipe, categorical),
    ("ord", ordinal_pipe, ordinal),
])

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
print(X.shape[1], "->", X_train_t.shape[1])"""

cell8 = """names = preprocessor.get_feature_names_out()
clean = pd.DataFrame(X_train_t, columns=names)
clean.head()"""

cell9 = """plt.hist(X_train["fare"], bins=40)
plt.title("Fare - raw")
plt.show()"""

cell10 = """plt.hist(clean["num__fare"], bins=40)
plt.title("Fare - after imputation and scaling")
plt.show()"""

cell11 = """def build_titanic_pipeline():
    \"\"\"Return the unfitted preprocessor built in Cells 5, 6 and 7.\"\"\"
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
        ("ord", ordinal_pipe, ordinal),
    ])

def validate_pipeline(pipeline, X_train, X_val):
    train = pipeline.transform(X_train)
    val = pipeline.transform(X_val)

    assert not np.isnan(train).any(), "missing values left in training data"
    assert train.shape[1] == val.shape[1], "train and val columns differ"
    assert train.shape[1] > 6, "too few features - encoders did not run"

    print("Pipeline validation PASSED")
    print("Features:", train.shape[1], " Train rows:", train.shape[0])

pipeline = build_titanic_pipeline()
pipeline.fit(X_train)
validate_pipeline(pipeline, X_train, X_val)

joblib.dump(pipeline, "titanic_preprocessor.joblib")"""

cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10, cell11]

nb.cells = [nbf.v4.new_code_cell(c) for c in cells]

with open('skill_lab2.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Notebook regenerated safely')

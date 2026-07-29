import json
import io
import sys
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

cells_code = [
    # Cell 1
    ("""import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder""", False),

    # Cell 2
    ("""titanic = sns.load_dataset("titanic")

leaks = ["alive", "who", "adult_male", "class", "embark_town", "alone", "deck"]
df = titanic.drop(columns=leaks)
print("df.shape:", df.shape)""", True), # has df.head() display

    # Cell 3
    ("""df["is_alone"] = ((df["sibsp"] + df["parch"]) == 0).astype(int)

# did we rebuild seaborn's "alone" column correctly?
matches = (df["is_alone"] == titanic["alone"].astype(int)).all()
print("is_alone matches the original alone column:", matches)

numeric = ["age", "fare", "sibsp", "parch", "is_alone"]
categorical = ["sex", "embarked"]
ordinal = ["pclass"]""", False),

    # Cell 4
    ("""X = df.drop(columns="survived")
y = df["survived"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("X_train.shape, X_val.shape:", X_train.shape, X_val.shape)""", False),

    # Cell 5
    ("""numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])""", False),

    # Cell 6
    ("""ordinal_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=[[1, 2, 3]])),
])""", False),

    # Cell 7
    ("""preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", categorical_pipe, categorical),
    ("ord", ordinal_pipe, ordinal),
])

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
print("Transformation:", X.shape[1], "->", X_train_t.shape[1])""", False),

    # Cell 8
    ("""names = preprocessor.get_feature_names_out()
clean = pd.DataFrame(X_train_t, columns=names)""", True), # has clean.head() display

    # Cell 9
    ("""plt.hist(X_train["fare"], bins=40)
plt.title("Fare - raw")
plt.show()""", False),

    # Cell 10
    ("""plt.hist(clean["num__fare"], bins=40)
plt.title("Fare - after imputation and scaling")
plt.show()""", False),

    # Cell 11
    ("""def build_titanic_pipeline():
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

joblib.dump(pipeline, "titanic_preprocessor.joblib")
print("Successfully saved fitted pipeline to titanic_preprocessor.joblib")""", False)
]

global_env = {}
cells = []

for idx, (code_str, eval_head) in enumerate(cells_code, start=1):
    plt.close('all')
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf
    
    outputs = []
    
    try:
        exec(code_str, global_env)
        disp_res = None
        if eval_head:
            if idx == 2:
                disp_res = global_env['df'].head()
            elif idx == 8:
                disp_res = global_env['clean'].head()
    except Exception as e:
        print(f"Error in cell {idx}: {e}", file=sys.stderr)
        raise e
    finally:
        sys.stdout = old_stdout

    printed = stdout_buf.getvalue()
    if printed:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": [line + "\n" for line in printed.splitlines()]
        })

    # Check for plot figures
    fig_nums = plt.get_fignums()
    if fig_nums:
        for fignum in fig_nums:
            fig = plt.figure(fignum)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            b64_data = base64.b64encode(img_buf.read()).decode('utf-8')
            outputs.append({
                "data": {
                    "image/png": b64_data,
                    "text/plain": ["<Figure size 640x480 with 1 Axes>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
        plt.close('all')

    if disp_res is not None:
        outputs.append({
            "data": {
                "text/html": [disp_res._repr_html_()],
                "text/plain": [repr(disp_res)]
            },
            "execution_count": idx,
            "metadata": {},
            "output_type": "execute_result"
        })

    # Add source code for cell
    full_source = code_str
    if eval_head:
        if idx == 2:
            full_source += "\ndf.head()"
        elif idx == 8:
            full_source += "\nclean.head()"

    source_lines = [line + "\n" for line in full_source.splitlines()]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip("\n")

    cell_obj = {
        "cell_type": "code",
        "execution_count": idx,
        "metadata": {},
        "outputs": outputs,
        "source": source_lines
    }
    cells.append(cell_obj)

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

filepath = "executed_out.json"
with open(filepath, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

import os
print(f"SUCCESS: Wrote executed_out.json ({os.path.getsize(filepath)} bytes)!")



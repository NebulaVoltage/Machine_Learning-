import json
import io
import sys
import base64
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def run_cell_code(code_str, global_env):
    # Capture stdout
    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf

    # Clear current matplotlib figure before running
    plt.close('all')

    outputs = []
    
    # Parse code AST to separate trailing expression if present
    tree = ast.parse(code_str)
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body.pop()
    
    try:
        # Execute leading statements
        if tree.body:
            exec(compile(tree, filename="<cell>", mode="exec"), global_env)
        
        # Evaluate last expression if present
        eval_res = None
        if last_expr:
            eval_res = eval(compile(ast.Expression(body=last_expr.value), filename="<cell>", mode="eval"), global_env)
    finally:
        sys.stdout = old_stdout

    stdout_text = stdout_buf.getvalue()
    if stdout_text:
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": stdout_text.splitlines(keepends=True)
        })

    # Check if a plot was generated
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
                    "text/plain": ["<Figure size ...>"]
                },
                "metadata": {},
                "output_type": "display_data"
            })
        plt.close('all')

    # If there is an evaluated result (like df.head()), format display
    if eval_res is not None:
        data_dict = {}
        if isinstance(eval_res, (pd.DataFrame, pd.Series)):
            data_dict["text/html"] = [eval_res._repr_html_()]
            data_dict["text/plain"] = [repr(eval_res)]
        else:
            data_dict["text/plain"] = [repr(eval_res)]
        
        outputs.append({
            "data": data_dict,
            "execution_count": len(global_env),
            "metadata": {},
            "output_type": "execute_result"
        })

    return outputs

cells_source = [
    # Cell 1
    """import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder""",

    # Cell 2
    """titanic = sns.load_dataset("titanic")

leaks = ["alive", "who", "adult_male", "class", "embark_town", "alone", "deck"]
df = titanic.drop(columns=leaks)
print("df.shape:", df.shape)
df.head()""",

    # Cell 3
    """df["is_alone"] = ((df["sibsp"] + df["parch"]) == 0).astype(int)

# did we rebuild seaborn's "alone" column correctly?
matches = (df["is_alone"] == titanic["alone"].astype(int)).all()
print("is_alone matches the original alone column:", matches)

numeric = ["age", "fare", "sibsp", "parch", "is_alone"]
categorical = ["sex", "embarked"]
ordinal = ["pclass"]""",

    # Cell 4
    """X = df.drop(columns="survived")
y = df["survived"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("X_train.shape, X_val.shape:", X_train.shape, X_val.shape)""",

    # Cell 5
    """numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])""",

    # Cell 6
    """ordinal_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=[[1, 2, 3]])),
])""",

    # Cell 7
    """preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric),
    ("cat", categorical_pipe, categorical),
    ("ord", ordinal_pipe, ordinal),
])

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)
print("Transformation completed:", X.shape[1], "->", X_train_t.shape[1])""",

    # Cell 8
    """names = preprocessor.get_feature_names_out()
clean = pd.DataFrame(X_train_t, columns=names)
clean.head()""",

    # Cell 9
    """plt.hist(X_train["fare"], bins=40)
plt.title("Fare - raw")
plt.show()""",

    # Cell 10
    """plt.hist(clean["num__fare"], bins=40)
plt.title("Fare - after imputation and scaling")
plt.show()""",

    # Cell 11
    """def build_titanic_pipeline():
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
print("Successfully saved fitted pipeline to titanic_preprocessor.joblib")"""
]

global_env = {}
notebook_cells = []

for idx, code in enumerate(cells_source, start=1):
    outputs = run_cell_code(code, global_env)
    
    source_lines = code.splitlines(keepends=True)

    cell = {
        "cell_type": "code",
        "execution_count": idx,
        "metadata": {},
        "outputs": outputs,
        "source": source_lines
    }
    notebook_cells.append(cell)

notebook_json = {
    "cells": notebook_cells,
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

with open("skill_lab2.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

print("Successfully executed notebook cells and updated skill_lab2.ipynb!")

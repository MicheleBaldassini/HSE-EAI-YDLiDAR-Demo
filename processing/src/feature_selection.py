# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from joblib import dump, load
import os


exec(open(os.path.abspath(os.path.join(__file__, '..', '..', '..', 'remove_hidden_items.py'))).read())


# --------------------------
# Load data
# --------------------------
base_path = os.path.abspath(os.path.join(__file__, '..', '..'))

results_path = os.path.join(base_path, 'results')
os.makedirs(results_path, exist_ok=True)
fig_path = os.path.join(base_path, 'fig')
os.makedirs(fig_path, exist_ok=True)

dataset = pd.read_csv(os.path.join(base_path, 'dataset', 'dataset.csv'))
X = dataset.iloc[:, :-1]
T = dataset.iloc[:, -1]

variableNames = X.columns

# --------------------------
# Holdout partition for test
# --------------------------
X_train, X_test, T_train, T_test = train_test_split(
    X, T, test_size=0.2, stratify=T, random_state=42
)

trainset = pd.concat([X_train, T_train], axis=1)
testset = pd.concat([X_test, T_test], axis=1)

trainset.to_csv(os.path.join(base_path, 'dataset', "trainset.csv"), index=False)
testset.to_csv(os.path.join(base_path, 'dataset', "testset.csv"), index=False)

# --------------------------
# Feature selection function
# --------------------------
def feature_selection(X, T, model, niter=30):
    """Run sequential forward selection multiple times."""
    res = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for _ in range(niter):
        sfs = SequentialFeatureSelector(
            model,
            n_features_to_select="auto",
            direction="forward",
            scoring="accuracy",
            cv=cv,
            n_jobs=-1
        )
        sfs.fit(X, T)
        fs = sfs.get_support()
        res.append({"fs": fs})
    return res

# --------------------------
# Define models
# --------------------------
models = {
    "svm": SVC(kernel="rbf", gamma="scale"),
    "knn": KNeighborsClassifier(n_neighbors=5),
    # "mlp": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42),
    "dt": DecisionTreeClassifier(random_state=42)
}

niter = 1

for name, model in models.items():
    print(f"Running feature selection with {name.upper()}...")
    sequential_fs = feature_selection(X_train.values, T_train, model, niter=niter)

    # Save results
    dump(sequential_fs, os.path.join(results_path, f"sequential_fs_{model.__class__.__name__}.joblib"))

    # --------------------------
    # Plot bar
    # --------------------------
    D = np.vstack([r["fs"] for r in sequential_fs])
    s = np.sum(D, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(range(1, len(s) + 1), s)
    ax.set_xticks(range(1, len(s) + 1))
    ax.set_yticks(range(1, int(np.max(s)) + 1))
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Selection count")
    ax.set_title(f"Sequential Feature Selection ({name.upper()})")
    fig.tight_layout()
    fig.savefig(os.path.join(base_path, 'fig', f"sequential_fs_{name}.png"), dpi=300)
    plt.close(fig)
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from joblib import dump, load
import os

import conf_mat


exec(open(os.path.abspath(os.path.join(__file__, '..', '..', '..', 'remove_hidden_items.py'))).read())


# --------------------------
# Carica dataset
# --------------------------
base_path = os.path.abspath(os.path.join(__file__, '..', '..'))
trainset = pd.read_csv(os.path.join(base_path, 'dataset', 'trainset.csv'))
testset = pd.read_csv(os.path.join(base_path, 'dataset', 'testset.csv'))

# --------------------------
# Define models + param grids
# --------------------------
param_grids = {
    'SVC': (
        SVC(),
        {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        }
    ),
    'KNeighborsClassifier': (
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    ),
    # 'MLPClassifier': (
    #     MLPClassifier(max_iter=1000, random_state=42),
    #     {
    #         'hidden_layer_sizes': [(50,), (100,), (50,50)],
    #         'activation': ['relu', 'tanh'],
    #         'alpha': [0.0001, 0.001]
    #     }
    # ),
    'DecisionTreeClassifier': (
        DecisionTreeClassifier(random_state=42),
        {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20]
        }
    )
}

results = []

# --------------------------
# Cross-validation con GridSearch
# --------------------------
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, (model, grid) in param_grids.items():
    print(f'\nRunning {name} with GridSearchCV...')

    # --------------------------
    # Carica risultati feature selection
    # --------------------------
    sequential_fs = load(os.path.join(base_path, 'results', f'sequential_fs_{model.__class__.__name__}.joblib'))

    D = np.vstack([r['fs'] for r in sequential_fs])
    s = np.sum(D, axis=0)

    # Ordina feature per importanza (decrescente)
    sorted_idx = np.argsort(s)[::-1]

    # Prendi sempre le prime 5
    selected_features = sorted_idx[:5].tolist()

    # Continua a prendere finché il numero di selezioni è almeno metà della precedente
    for i in range(5, len(sorted_idx)):
        if s[sorted_idx[i]] >= s[sorted_idx[i-1]] / 2:
            selected_features.append(sorted_idx[i])
        else:
            break

    print('Selected features:', selected_features)

    X_train = trainset.iloc[:, selected_features].values
    T_train = trainset.iloc[:, -1].values - 1

    X_test = testset.iloc[:, selected_features].values
    T_test = testset.iloc[:, -1].values - 1

    gridsearch = GridSearchCV(
        model,
        param_grid=grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    gridsearch.fit(X_train, T_train)

    best_model = gridsearch.best_estimator_
    best_params = gridsearch.best_params_
    best_acc = gridsearch.best_score_

    print(f'Best {name} params: {best_params}')
    print(f'Best {name} CV accuracy: {best_acc:.4f}')

    results.append({
        'model': name,
        'best_params': best_params,
        'mean_accuracy': best_acc,
        'estimator': best_model
    })

# --------------------------
# Save CV results
# --------------------------
cv_df = pd.DataFrame([{
    'model': r['model'],
    'best_params': r['best_params'],
    'mean_accuracy': r['mean_accuracy']
} for r in results])

cv_df.to_csv(os.path.join(base_path, 'results', 'cv_results.csv'), index=False)
print('GridSearch results saved to cv_results.csv')

# --------------------------
# Select best model
# --------------------------
best = max(results, key=lambda r: r['mean_accuracy'])
best_model_name = best['model']
best_model = best['estimator']

print(f"\nBest model overall: {best_model_name} with accuracy {best['mean_accuracy']:.4f}")

# --------------------------
# Train best model on full training set
# --------------------------
best_model.fit(X_train, T_train)
dump(best_model, os.path.join(base_path, 'results', 'best_model.pth'))

best_model = load(os.path.join(base_path, 'results', 'best_model.pth'))
# --------------------------
# Test the best model
# --------------------------
y_score = best_model.predict_proba(X_test)
y_pred = np.argmax(y_score, axis=-1)
test_acc = accuracy_score(T_test, y_pred)
print(f'Test accuracy = {test_acc:.4f}')


fig, ax = plt.subplots(1, 1, figsize=(5, 5))

cm = confusion_matrix(T_test, y_pred)
conf_mat.plot_confusion_matrix(ax, cm, ['a', 'b', 'c', 'd', 'e', 'f'], fontsize=10)
ax.set_title(f'Confusion Matrix ({best_model_name})')
fig.tight_layout()
plt.draw()
fig.savefig(os.path.join(base_dir, 'fig', f'cm_{best_model_name}.png'))
plt.cla()
plt.close(fig)

one_hot_test = pd.get_dummies(T_test, dtype=float).values
num_classes = one_hot_test.shape[1]

fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(one_hot_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

classes = ['a', 'b', 'c', 'd', 'e', 'f']
for i in range(num_classes):
    ax.plot(fpr[i], tpr[i],
            lw=2,
            label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

ax.plot([0, 1], [0, 1], 'k--', lw=2)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.legend(loc='lower right')

fig.tight_layout()
fig.savefig(os.path.join(base_dir, 'fig', f'roc_{best_model_name}.png'))
plt.close(fig)

# --------------------------
# Reload model example
# --------------------------
reloaded_model = load(os.path.join(base_path, 'results', 'best_model.pth'))
print('Reloaded model type:', type(reloaded_model))
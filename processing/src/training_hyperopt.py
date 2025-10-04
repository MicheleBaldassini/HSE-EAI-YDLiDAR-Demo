# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
from joblib import dump, load
import os

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import conf_mat


exec(open(os.path.abspath(os.path.join(__file__, '..', '..', '..', 'remove_hidden_items.py'))).read())


# --------------------------
# Carica dataset
# --------------------------
base_path = os.path.abspath(os.path.join(__file__, '..', '..'))
trainset = pd.read_csv(os.path.join(base_path, 'dataset', 'trainset.csv'))
testset = pd.read_csv(os.path.join(base_path, 'dataset', 'testset.csv'))

# --------------------------
# Definizione search space
# --------------------------
search_spaces = {
    'SVC': hp.choice('svm', [
        {
            'model': 'svm',
            'C': hp.loguniform('svm_C', np.log(1e-3), np.log(1e3)),
            'kernel': hp.choice('svm_kernel', ['linear', 'rbf', 'poly']),
            'gamma': hp.choice('svm_gamma', ['scale', 'auto']),
            'degree': hp.quniform('svm_degree', 2, 5, 1)
        }
    ]),
    'KNeighborsClassifier': hp.choice('knn', [
        {
            'model': 'knn',
            'n_neighbors': hp.quniform('knn_n_neighbors', 3, 30, 1),
            'weights': hp.choice('knn_weights', ['uniform', 'distance']),
            'p': hp.choice('knn_p', [1, 2])  # manhattan o euclidea
        }
    ]),
    # 'MLPClassifier': hp.choice('mlp', [
    #     {
    #         'model': 'mlp',
    #         'hidden_layer_sizes': hp.choice('mlp_hidden', [(50,), (100,), (50,50), (100,50)]),
    #         'activation': hp.choice('mlp_activation', ['relu', 'tanh', 'logistic']),
    #         'alpha': hp.loguniform('mlp_alpha', np.log(1e-5), np.log(1e-1)),
    #         'learning_rate_init': hp.loguniform('mlp_lr', np.log(1e-4), np.log(1e-1))
    #     }
    # ]),
    'DecisionTreeClassifier': hp.choice('dt', [
        {
            'model': 'dt',
            'criterion': hp.choice('dt_criterion', ['gini', 'entropy']),
            'max_depth': hp.choice('dt_max_depth', [None, 5, 10, 20, 50]),
            'min_samples_split': hp.quniform('dt_min_split', 2, 20, 1),
            'min_samples_leaf': hp.quniform('dt_min_leaf', 1, 10, 1)
        }
    ])
}

# --------------------------
# Funzione obiettivo
# --------------------------
def objective(params):
    model_type = params['model']

    if model_type == 'svm':
        model = SVC(
            C=params['C'],
            kernel=params['kernel'],
            gamma=params['gamma'],
            degree=int(params['degree']),
            probability=False
        )
    elif model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=int(params['n_neighbors']),
            weights=params['weights'],
            p=params['p']
        )
    elif model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            activation=params['activation'],
            alpha=params['alpha'],
            learning_rate_init=params['learning_rate_init'],
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'dt':
        model = DecisionTreeClassifier(
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            random_state=42
        )
    else:
        raise ValueError('Unknown model type')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, T_train, cv=cv, scoring='accuracy')
    acc = scores.mean()

    return {'loss': -acc, 'status': STATUS_OK, 'params': params}

# --------------------------
# Esegui Hyperopt su tutti i modelli
# --------------------------
results = []
best_models = {}

for model_name, space in search_spaces.items():
    print(f'\n=== Optimizing {model_name} with Hyperopt ===')
    # --------------------------
    # Carica risultati feature selection
    # --------------------------
    sequential_fs = load(os.path.join(base_path, 'results', f'sequential_fs_{model_name}.joblib'))

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

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,   # aumenta se vuoi una ricerca più lunga
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    best_trial = min(trials.results, key=lambda x: x['loss'])
    acc = -best_trial['loss']
    params = best_trial['params']

    print(f'Best {model_name} accuracy = {acc:.4f}')
    print(f'Params = {params}')

    results.append({'model': model_name, 'mean_accuracy': acc, 'best_params': params})
    best_models[model_name] = params


cv_df = pd.DataFrame([{
    'model': r['model'],
    'best_params': r['best_params'],
    'mean_accuracy': r['mean_accuracy']
} for r in results])

cv_df.to_csv(os.path.join(base_path, 'results', 'hp_results.csv'), index=False)

# --------------------------
# Seleziona best model globale
# --------------------------
best = max(results, key=lambda r: r['mean_accuracy'])
best_model_name = best['model']
best_params = best['best_params']
print(f"\nBest model overall: {best_model_name} with accuracy {best['mean_accuracy']:.4f}")

# Ricostruisci best_model con parametri trovati
_ = objective(best_params)  # per rigenerare model identico
if best_params['model'] == 'svm':
    best_model = SVC(
        C=best_params['C'],
        kernel=best_params['kernel'],
        gamma=best_params['gamma'],
        degree=int(best_params['degree'])
    )
elif best_params['model'] == 'knn':
    best_model = KNeighborsClassifier(
        n_neighbors=int(best_params['n_neighbors']),
        weights=best_params['weights'],
        p=best_params['p']
    )
# elif best_params['model'] == 'mlp':
#     best_model = MLPClassifier(
#         hidden_layer_sizes=best_params['hidden_layer_sizes'],
#         activation=best_params['activation'],
#         alpha=best_params['alpha'],
#         learning_rate_init=best_params['learning_rate_init'],
#         max_iter=1000,
#         random_state=42
#     )
elif best_params['model'] == 'dt':
    best_model = DecisionTreeClassifier(
        criterion=best_params['criterion'],
        max_depth=best_params['max_depth'],
        min_samples_split=int(best_params['min_samples_split']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        random_state=42
    )

# --------------------------
# Train e salva modello
# --------------------------
best_model.fit(X_train, T_train)
dump(best_model, os.path.join(base_path, 'results', 'best_model.pth'))

best_model = load(os.path.join(base_path, 'results', 'best_model.pth'))
# --------------------------
# Test su set di test
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
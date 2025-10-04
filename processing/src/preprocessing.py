# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy.stats import gmean, hmean, kurtosis, skew, trim_mean, mode
from scipy.signal import find_peaks


exec(open(os.path.abspath(os.path.join(__file__, '..', '..', '..', 'remove_hidden_items.py'))).read())


# --------------------------
# Paths
# --------------------------
base_path = os.path.abspath(os.path.join(__file__, '..', '..'))
data_path = os.path.join(base_path, 'data')

dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

# --------------------------
# Compute minDistance
# --------------------------
minDistance = np.inf
minSamples = np.inf

for d in dirs:
    subdir_path = os.path.join(data_path, d)
    subdirs = [s for s in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, s))]

    for s in subdirs:
        files = [f for f in os.listdir(os.path.join(subdir_path, s)) if f.endswith('.csv')]

        for file in files:
            filepath = os.path.join(subdir_path, s, file)
            data = pd.read_csv(filepath)

            data = data[(data['Angle'] >= 135) & (data['Angle'] <= 225)]
            data = data[(data['Range'] > 0) & (data['Range'] < 1.5)]

            if len(data) < minSamples:
                minSamples = len(data)

            minDist = data['Range'].min()
            if minDist < minDistance:
                minDistance = minDist

print(f'Min samples: {minSamples}')

# --------------------------
# Compute features
# --------------------------
features = []
targets = []

for i, d in enumerate(dirs, start=1):
    subdir_path = os.path.join(data_path, d)
    subdirs = [s for s in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, s))]

    for j, s in enumerate(subdirs, start=1):
        files = [f for f in os.listdir(os.path.join(subdir_path, s)) if f.endswith('.csv')]

        for file in files:
            filepath = os.path.join(subdir_path, s, file)
            data = pd.read_csv(filepath)

            # Preprocessing
            data = data[(data['Angle'] >= 135) & (data['Angle'] <= 225)]
            data = data[(data['Range'] > 0) & (data['Range'] < 1.5)]

            # Translate all distance values at 0.5 metres
            offset = data['Range'].min() - minDistance
            range_vals = data['Range'].values - offset

            # Features
            peaks, _ = find_peaks(range_vals)
            peak_vals = range_vals[peaks] if len(peaks) > 0 else np.array([np.nan])

            feat = [
                np.mean(range_vals),
                gmean(range_vals),
                hmean(range_vals),
                trim_mean(range_vals, 0.2),  # 40% trimmed mean
                np.max(range_vals),
                np.min(range_vals),
                np.std(range_vals),
                np.var(range_vals),
                mode(range_vals, keepdims=False).mode,
                np.median(range_vals),
                kurtosis(range_vals),
                skew(range_vals),
                np.ptp(range_vals),  # peak-to-peak
                np.ptp(range_vals) / np.sqrt(np.mean(range_vals**2)),  # peak2rms
                np.sqrt(np.mean(range_vals**2)),  # rms
                np.sqrt(np.sum(range_vals**2)),   # rssq
                len(peaks),
                np.nanmean(peak_vals)
            ]
            features.append(feat)

            # Targets
            targets.append(j)

variableNames = [
    'mean', 'geomean', 'harmmean', 'trimmean',
    'max', 'min', 'std',
    'var', 'mode', 'median',
    'kurtosis', 'skewness', 'peak2peak',
    'peak2rms', 'rms', 'rssq',
    'num_maxima', 'mean_maxima',
    'targets'
]

dataset = pd.DataFrame(np.column_stack([features, targets]), columns=variableNames)

os.makedirs(os.path.join(base_path, 'dataset'), exist_ok=True)
dataset.to_csv(os.path.join(base_path, 'dataset', 'dataset_raw.csv'), index=False)

# --------------------------
# Clean dataset
# --------------------------
dataset = pd.read_csv(os.path.join(base_path, 'dataset', 'dataset_raw.csv'))
features = dataset.iloc[:, :-1]
targets = dataset.iloc[:, -1]

# Remove rows with NaN
rowsWithMissing = features.isna().sum(axis=1)
print(f'Dataset rows with NaN = {np.sum(rowsWithMissing > 0)}')
print(np.where(rowsWithMissing > 0))
features = features[rowsWithMissing == 0]
targets = targets[rowsWithMissing == 0]

# Remove columns with NaN
colsWithMissing = features.isna().sum(axis=0)
print(f'Dataset cols with NaN = {np.sum(colsWithMissing > 0)}')
features = features.loc[:, colsWithMissing == 0]

# Remove rows with all zeros
mask = (features != 0).any(axis=1)
features = features[mask]
targets = targets[mask]

# Remove columns with all zeros
features = features.loc[:, (features != 0).any(axis=0)]

# Remove rows with all ones
mask = ~(features == 1).all(axis=1)
features = features[mask]
targets = targets[mask]

# Remove columns with all ones
features = features.loc[:, ~(features == 1).all(axis=0)]

# (Optional) Remove correlated features
corr = features.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.99)]
features = features.drop(columns=to_drop)

dataset = pd.concat([features, targets.reset_index(drop=True)], axis=1)
dataset.to_csv(os.path.join(base_path, 'dataset', 'dataset.csv'), index=False)
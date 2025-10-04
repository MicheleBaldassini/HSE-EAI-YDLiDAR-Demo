import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Cartella dei CSV ---
data_path = os.path.abspath(os.path.join(__file__, '..', '..', 'demo'))

# --- Raccolta dati ---
records = []
for file in glob.glob(os.path.join(data_path, '*.csv')):
    df = pd.read_csv(file)
    first_row = df.iloc[0]
    pred = first_row['Pred']
    filename = os.path.basename(file).replace('.csv', '')
    
    ts_raw = int(filename)
    ts_sec = ts_raw // 1_000_000_000
    ts_dt = datetime.fromtimestamp(ts_sec)

    records.append((ts_dt, pred))

# --- Creazione DataFrame ---
timeline_df = pd.DataFrame(records, columns=['timestamp', 'Pred'])
timeline_df['Pred'] = timeline_df['Pred'].apply(lambda x: int(x))
timeline_df = timeline_df.sort_values('timestamp').reset_index(drop=True)

# --- Mappa classi e colori ---
labels_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
color_map = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red', 4: 'tab:purple', 5: 'tab:brown'}

# Classi presenti nel dataset
present_classes = sorted(timeline_df['Pred'].unique())
labels_present = [labels_map[i] for i in present_classes]
colors_present = [color_map[i] for i in present_classes]

# --- Rimappo classi in indici consecutivi per eliminare spazi vuoti ---
class_to_index = {cls: i for i, cls in enumerate(present_classes)}
timeline_df['Pred_compact'] = timeline_df['Pred'].map(class_to_index)

# Conteggio classi
class_counts = timeline_df['Pred'].value_counts().sort_index()

# --- Creazione figura ---
fig, ax = plt.subplots(1, 2, figsize=(16,6), gridspec_kw={'width_ratios':[2,1]})

# --- Scatter plot ---
ax[0].scatter(
    range(len(timeline_df)),
    timeline_df['Pred_compact'],
    c=[color_map[p] for p in timeline_df['Pred']],
    alpha=0.7
)

# Y-ticks solo per classi presenti (compatti)
ax[0].set_yticks(range(len(present_classes)))
ax[0].set_yticklabels(labels_present, fontsize=16)

# Colori ai tick Y
for tick, cls in zip(ax[0].get_yticklabels(), present_classes):
    tick.set_color(color_map[cls])

ax[0].set_ylabel('Posizione (Pred)', fontsize=16)
ax[0].set_title('Timeline delle posizioni', fontsize=18)

# --- Etichette X ad intervalli di tempo ---
interval = timedelta(seconds=60)
start_time = timeline_df['timestamp'].iloc[0]
xticks = []
xticklabels = []

for i, t in enumerate(timeline_df['timestamp']):
    if not xticks or t - start_time >= interval:
        xticks.append(i)
        xticklabels.append(t.strftime('%d-%m-%Y %H:%M:%S'))
        start_time = t

ax[0].set_xticks(xticks)
ax[0].set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=14)
ax[0].tick_params(axis='y', labelsize=14)

# --- Pie chart ---
wedges, texts, autotexts = ax[1].pie(
    class_counts[class_counts.index.isin(present_classes)],
    labels=labels_present,
    colors=colors_present,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 16}
)

# Etichette colorate come gli spicchi
for text, color in zip(texts, colors_present):
    text.set_color(color)

# Percentuali in bianco
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)

ax[1].set_title('Distribuzione delle posizioni', fontsize=18)

plt.tight_layout()
plt.show()

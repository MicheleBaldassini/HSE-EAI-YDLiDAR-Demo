import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# cartella dei csv
data_path = os.path.abspath(os.path.join(__file__, '..', 'demo'))

# lista per accumulare i risultati
records = []

# ciclo su tutti i file csv
for file in glob.glob(os.path.join(data_path, "*.csv")):
    df = pd.read_csv(file)

    # prendo la prima riga
    first_row = df.iloc[0]

    # assumo che ci sia una colonna 'timestamp' e 'Pred'
    timestamp = pd.to_datetime(first_row["timestamp"])
    pred = first_row["Pred"]

    records.append((timestamp, pred))

# creo DataFrame con i risultati
timeline_df = pd.DataFrame(records, columns=["timestamp", "pred"])

# ordino per timestamp
timeline_df = timeline_df.sort_values("timestamp")

labels_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
labels = [labels_map.get(i, str(i)) for i in class_counts.index]

fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
ax[0].scatter(timeline_df["timestamp"], timeline_df["pred"], c="blue", alpha=0.7)
# ax[0].set_yticks(range(int(timeline_df["pred"].min())), int(timeline_df["pred"].max))
ax[0].set_yticks(range(len(labels)))
ax[0].set_yticklabels(labels)
ax[0].set_xlabel("Timestamp")
ax[0].set_ylabel("Posizione (Pred)")
ax[0].set_title("Timeline delle posisizoni")
ax[0].set_xticks(rotation=45)

# --- Pie chart delle occorrenze ---
class_counts = timeline_df["pred"].value_counts()
ax[1].pie(class_counts, labels=labels, autopct="%1.1f%%", startangle=90)
ax[1].set_title("Distribuzione percentuale delle posisizoni")

plt.tight_layout()
plt.show()
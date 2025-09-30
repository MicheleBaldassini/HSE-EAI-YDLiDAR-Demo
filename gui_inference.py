# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from threading import Thread
import tkinter as tk

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from YDLidar import YDLidar
from constants import *

from joblib import load
from scipy.stats import gmean, hmean, kurtosis, skew, trim_mean, mode
from scipy.signal import find_peaks


exec(open(os.path.abspath(os.path.join(__file__, '..', '..', 'remove_hidden_items.py'))).read())


# --------------------------
# Init model and feature
# --------------------------
def init_model():
    global model, selected_features
    model = load(os.path.abspath(os.path.join(__file__, '..', 'processing', 'results', 'best_model.pth')))
    sequential_fs = load(os.path.join(__file__, '..', 'processing', 'results', f"sequential_fs_{model.__class__.__name__}.joblib"))

    D = np.vstack([r["fs"] for r in sequential_fs])
    s = np.sum(D, axis=0)

    sorted_idx = np.argsort(s)[::-1]
    selected_features = sorted_idx[:5].tolist()
    for i in range(5, len(sorted_idx)):
        if s[sorted_idx[i]] >= s[sorted_idx[i-1]] / 2:
            selected_features.append(sorted_idx[i])
        else:
            break


# --------------------------
# Preprocess LiDAR signal
# --------------------------
def preprocess(angles, ranges):
    angles %= (2 * np.pi)
    mask = (angles >= MIN_ANGLE_RAD) & (angles <= MAX_ANGLE_RAD)
    ranges = ranges[mask]
    mask = (ranges > 0) & (ranges < 1.5)
    ranges = ranges[mask]

    ranges = ranges - ranges.min()
    return np.array(ranges)


# --------------------------
# Compute features from preprocessed LiDAR signal
# --------------------------
def compute_features(ranges, selected_features):
    global selected_features
    peaks, _ = find_peaks(range_vals)
    peak_vals = range_vals[peaks] if len(peaks) > 0 else np.array([np.nan])
    feat = [
        np.mean(range_vals),
        gmean(range_vals),
        hmean(range_vals),
        trim_mean(range_vals, 0.2),
        np.max(range_vals),
        np.min(range_vals),
        np.std(range_vals),
        np.var(range_vals),
        mode(range_vals, keepdims=False).mode,
        np.median(range_vals),
        kurtosis(range_vals),
        skew(range_vals),
        np.ptp(range_vals),
        np.ptp(range_vals) / np.sqrt(np.mean(range_vals**2)),
        np.sqrt(np.mean(range_vals**2)),
        np.sqrt(np.sum(range_vals**2)),
        len(peaks),
        np.nanmean(peak_vals)
    ]
    feat = feat[selected_features]
    return np.array(feat).reshape(1, -1)


# --------------------------
# Inference thread
# --------------------------
def predict(angles, ranges):
    global model, inference_label
    ranges = preprocess(angles, ranges)
    features = compute_features(ranges)
    pred = model.predict(features)[0]
    # inference_label.after(0, lambda: inference_label.config(text=f"Position: {pred}"))
    inference_label['text'] = f"Position: {pred}"


# --------------------------
# Init polar plot
# --------------------------
def init_plot():
    lidar_plot.clear()
    lidar_plot.grid(False)
    lidar_plot.autoscale_view(True, False, False)
    lidar_plot.set_rmin(MIN_RANGE)
    lidar_plot.set_rmax(MAX_RANGE)
    lidar_plot.set_theta_direction(-1)
    lidar_plot.set_theta_offset(-np.pi/2)
    lidar_plot.set_thetamin(MIN_ANGLE_DEG)
    lidar_plot.set_thetamax(MAX_ANGLE_DEG)
    lidar_plot.set_xticks([])
    lidar_plot.set_yticks([])


# --------------------------
# Write CSV
# --------------------------
def write_csv(filename, angles, ranges, intensity):
    # df = pd.DataFrame({'Angle': angles, 'Range': ranges, 'Intensity': intensity})
    # df['Angle'] %= (2 * np.pi)
    # df = df[(df['Angle'] >= MIN_ANGLE_RAD) & (df['Angle'] <= MAX_ANGLE_RAD)]
    # df['Angle'] = df['Angle'] * 180.0 / np.pi
    # df = df.sort_values(by=['Angle'])

    mask = (angles >= MIN_ANGLE_RAD) & (angles <= MAX_ANGLE_RAD)
    angles = angles[mask]
    ranges = ranges[mask]
    intensity = intensity[mask]

    angles = angles * 180.0 / np.pi
    sorted_idx = np.argsort(angles)

    angles = angles[sorted_idx]
    ranges = ranges[sorted_idx]
    intensity = intensity[sorted_idx]

    df = pd.DataFrame({'Angle': angles, 'Range': ranges, 'Intensity': intensity})
    df.to_csv(os.path.join(data_path, f"{filename}.csv"), index=False)


# --------------------------
# Animate
# --------------------------
def animate(num):
    global lidar, anim, thread_record, thread_inference, num_record, first_inference

    if start_button['text'] == 'Stop':
        lidar.scan_task()
        init_plot()
        lidar_plot.scatter(lidar.angle, lidar.range, c='k', cmap='hsv', alpha=0.95)

        # Record
        if record_button['text'] == 'No record':
            if num_record > 0:
                thread_record = Thread(target=write_csv, args=(lidar.scan.stamp, lidar.angle, lidar.range, lidar.intensity))
                thread_record.start()
            num_record += 1
            # record_label['text'] = num_record
            record_label.after(0, lambda: record_label.config(text=f'{num_record}'))

        # Inferenza
        if inference_button['text'] == 'No inference':
            if first_inference == False:
                thread_inference = Thread(target=predict, args=(lidar.angle, lidar.range))
                thread_inference.start()
            first_inference = False
    else:
        if thread_record is not None:
            thread_record.join()
            thread_record = None
        if thread_inference is not None:
            thread_inference.join()
            thread_inference = None
        anim.event_source.stop()
        anim = None
        lidar.turnOff()
        init_plot()
        num_record = 0
        first_inference = True


# --------------------------
# Start/Stop streaming
# --------------------------
def start():
    global lidar, anim
    if start_button['text'] == 'Start':
        start_button['text'] = 'Stop'
        lidar = YDLidar()
        if lidar.turnOn():
            interval = (1.0 / FREQUENCY) * 1000.0
            anim = animation.FuncAnimation(fig, animate, interval=interval)
            canvas.draw()
        else:
            start_button['text'] = 'Start'
    else:
        # start_button['text'] = 'Start'
        # record_button['text'] = 'Record'
        # record_label['text'] = ''
        # record_button['text'] = 'Inference'
        # inference_label['text'] = ''
        start_button.after(0, lambda: start_button.config(text="Start"))
        record_button.after(0, lambda: record_button.config(text="Record"))
        record_label.after(0, lambda: record_label.config(text=""))
        record_button.after(0, lambda: record_button.config(text="Inference"))
        inference_label.after(0, lambda: inference_label.config(text=""))


# --------------------------
# Start/Stop recording
# --------------------------
def record():
    global thread_record, num_record
    if start_button['text'] == 'Stop':
        if record_button['text'] == 'Record':
            os.makedirs(data_path, exist_ok=True)
            # record_button['text'] = 'No record'
            record_button.after(0, lambda: record_button.config(text="No record"))
        else:
            if thread_record is not None:
                thread_record.join()
                thread_record = None
            num_record = 0
            # record_button['text'] = 'Record'
            # record_label['text'] = ''
            record_button.after(0, lambda: record_button.config(text="Record"))
            record_label.after(0, lambda: record_label.config(text=""))


# --------------------------
# Start/Stop inference
# --------------------------
def inference():
    global thread_inference, first_inference
    if start_button['text'] == 'Stop':
        if inference_button['text'] == 'Inference':
            # record_button['text'] = 'No inference'
            record_button.after(0, lambda: record_button.config(text="No inference"))
        else:
            if thread_inference is not None:
                thread_inference.join()
                thread_inference = None
            first_inference = True
            # inference_button['text'] = 'Inference'
            # inference_label['text'] = ''
            inference_button.after(0, lambda: record_button.config(text="Inference"))
            inference_label.after(0, lambda: record_label.config(text=""))


# --------------------------
# Close GUI
# --------------------------
def quit():
    window.quit()
    window.destroy()


# --------------------------
# Center window
# --------------------------
def center(window):
    window.update_idletasks()
    width = window.winfo_width()
    frm_width = window.winfo_rootx() - window.winfo_x()
    win_width = width + 2 * frm_width
    height = window.winfo_height()
    titlebar_height = window.winfo_rooty() - window.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = window.winfo_screenwidth() // 2 - win_width // 2
    y = window.winfo_screenheight() // 2 - win_height // 2
    window.geometry(f'{width}x{height}+{x}+{y}')
    window.deiconify()


# --------------------------
# Init GUI
# --------------------------
def initGUI():
    global window, fig, lidar_plot, canvas
    global start_button, record_button, record_label
    global pred_button, prediction_box

    init_model()

    window_width, window_height = 600, 600
    window = tk.Tk()
    window.minsize(window_width, window_height)
    window.title('LiDAR Monitor')

    # Matplotlib
    fig = Figure(figsize=(16, 9), dpi=100)
    lidar_plot = fig.add_subplot(projection='polar')
    init_plot()

    # A tk.DrawingArea
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().place(relx=-0.015, rely=0.025, relwidth=1.0, relheight=1.0)

    # Start/Stop
    start_button = tk.Button(master=window, text='Start', command=start, bg='white')
    start_button.place(relx=0.3, rely=0.05, relwidth=0.125, relheight=0.05)

    # Record
    record_button = tk.Button(master=window, text='Record', command=record, bg='white')
    record_button.place(relx=0.575, rely=0.05, relwidth=0.125, relheight=0.05)

    # Prediction toggle
    inference_button = tk.Button(master=window, text='Inference', command=inference, bg='white')
    inference_button.place(relx=0.05, rely=0.05, relwidth=0.2, relheight=0.05)

    # Num recording
    record_label = tk.Label(master=window, text='', anchor='w', bg='white')
    record_label.place(relx=0.7, rely=0.05, relwidth=0.05, relheight=0.05)

    # Prediction label
    inference_label = tk.Label(master=window, text='', anchor='w', bg='white', font=('Arial', 12, 'bold'))
    inference_label.place(relx=0.05, rely=0.12, relwidth=0.2, relheight=0.05)

    window.configure(background='white')
    center(window)
    window.mainloop()


# --------------------------
# Main
# --------------------------
if __name__ == '__main__':
    lidar = None
    anim = None
    thread_record = None
    thread_inference = None
    num_record = 0
    first_inference = True

    initGUI()
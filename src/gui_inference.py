# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from datetime import datetime
from joblib import load

from scipy.signal import find_peaks
from scipy.stats import gmean, hmean, trim_mean, mode, kurtosis, skew

from YDLidar import YDLidar
from constants import *


exec(open(os.path.abspath(os.path.join(__file__, '..', '..', 'remove_hidden_items.py'))).read())


# --- GLOBAL DATA ---
demo_path = os.path.abspath(os.path.join(__file__, '..', '..', 'demo'))


# Init polar plot
def init_plot():
	lidar_plot.clear()
	lidar_plot.grid(False)
	lidar_plot.autoscale_view(True, False, False)
	lidar_plot.set_rmin(MIN_RANGE)
	lidar_plot.set_rmax(MAX_RANGE)
	lidar_plot.set_theta_direction(-1)
	# np.pi/2 ('N') to set the zero up
	# -np.pi/2 ('S') to set the zero down
	lidar_plot.set_theta_offset(-np.pi/2)
	# lidar_plot.set_theta_zero_location('N')

	lidar_plot.set_thetamin(MIN_ANGLE_DEG)
	lidar_plot.set_thetamax(MAX_ANGLE_DEG)

	lidar_plot.set_xticks([])
	lidar_plot.set_yticks([])


def init_timeline():
	timeline_plot.clear()

	timeline_plot.set_xticks([])
	timeline_plot.set_xticklabels([])

	timeline_plot.set_ylim(-0.5, 5.5)
	timeline_plot.set_yticks(range(6))
	timeline_plot.set_yticklabels([labels_map[i] for i in range(6)], fontsize=18, fontweight='bold')
	for tick, i in zip(timeline_plot.get_yticklabels(), range(6)):
		tick.set_color(color_map[i])

	timeline_plot.set_title('Timeline', fontsize=18, fontweight='bold')
	timeline_plot.set_xlabel('Time', fontsize=18)
	timeline_plot.set_ylabel('Position', fontsize=18)
	timeline_plot.figure.autofmt_xdate()
	canvas_timeline.draw()


# Init model and feature
def init_model():
	global model, selected_features
	model = load(os.path.abspath(os.path.join(__file__, '..', '..',  'processing', 'results', 'best_model.pth')))
	sequential_fs = load(os.path.abspath(os.path.join(__file__, '..', '..', 'processing', 'results', f'sequential_fs_{model.__class__.__name__}.joblib')))

	D = np.vstack([r['fs'] for r in sequential_fs])
	s = np.sum(D, axis=0)

	sorted_idx = np.argsort(s)[::-1]
	selected_features = sorted_idx[:5].tolist()
	for i in range(5, len(sorted_idx)):
		if s[sorted_idx[i]] >= s[sorted_idx[i-1]] / 2:
			selected_features.append(sorted_idx[i])
		else:
			break


# Preprocess LiDAR signal
def preprocess(angles, ranges):
	angles = angles * 180.0 / np.pi
	mask = (
		(angles >= MIN_ANGLE_DEG) & (angles <= MAX_ANGLE_DEG) &
		(ranges > 0) & (ranges < 1.5)
	)

	return mask


# Compute features from preprocessed LiDAR signal
def compute_features(range_vals):
	peaks, _ = find_peaks(range_vals)
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
	]
	feat = np.array(feat)
	feat = feat[selected_features]
	return np.array(feat).reshape(1, -1)


# Process data and predict position
def process_data(timestamp, angles, ranges, intensity):
	global inference_label
	angles = np.array(angles)
	ranges = np.array(ranges)
	intensity = np.array(intensity)

	mask = preprocess(angles, ranges)
	angles = angles[mask]
	ranges = ranges[mask]
	intensity = intensity[mask]

	if len(ranges) < 30:
		inference_label['text'] = 'Position: -'
		return

	features = compute_features(ranges)
	pred = np.argmax(model.predict_proba(features), axis=1)[0]

	df = pd.DataFrame({'Angle': angles, 'Range': ranges, 'Intensity': intensity, 'Pred': pred})
	df.to_csv(os.path.join(demo_path, f'{timestamp}.csv'), header=True, index=False)

	window.after(0, lambda: update_gui(timestamp, pred))


# Update GUI in the main thread
def update_gui(timestamp, pred):
	inference_label.config(text=f'Position: {labels_map[pred]}')

	ts_raw = int(timestamp)
	ts_sec = ts_raw // 1_000_000_000
	ts_dt = datetime.fromtimestamp(ts_sec)
	timeline_data.append((ts_dt, pred))
	
	if len(timeline_data) > 10:
		timeline_data.pop(0)

	update_timeline()


# Timeline update
# def update_timeline():
# 	init_timeline()

# 	times = [t for t, _ in timeline_data]
# 	preds = [p for _, p in timeline_data]
# 	colors = [color_map[p] for p in preds]

# 	timeline_plot.scatter(times, preds, c=colors, alpha=0.7)
# 	timeline_plot.set_xticks(times)
# 	timeline_plot.set_xticklabels([t.strftime('%H:%M:%S') for t in times], fontsize=18, fontweight='bold', rotation=45, ha='right')

# 	if len(times) > 1:
# 		timeline_plot.set_xlim(times[0], times[-1])

# 	canvas_timeline.draw()
def update_timeline():
    init_timeline()

    max_points = 10
    n_points = len(timeline_data)

    # x_positions = list(range(max_points - n_points, max_points))
    x_positions = list(range(n_points))

    preds = [p for _, p in timeline_data]
    colors = [color_map[p] for p in preds]
    times = [t.strftime('%H:%M:%S') for t, _ in timeline_data]

    timeline_plot.scatter(x_positions, preds, c=colors, alpha=0.8, s=120)

    timeline_plot.set_xlim(-0.5, 9.5)

    # ticks_to_show = list(range(max_points - n_points, max_points))
    ticks_to_show = list(range(n_points))
    timeline_plot.set_xticks(ticks_to_show)
    timeline_plot.set_xticklabels(times, fontsize=14, fontweight='bold', rotation=45, ha='right')

    canvas_timeline.draw()


# Animate loop
def animate(num):
	global anim, thread

	if start_button['text'] == 'Stop':
		lidar.scan_task()

		init_plot()
		lidar_plot.scatter(lidar.angle, lidar.range, c=lidar.intensity, cmap='hsv', alpha=0.95)

		thread = Thread(target=process_data,
						args=(lidar.scan.stamp, lidar.angle, lidar.range, lidar.intensity))
		thread.start()
	else:
		if thread is not None:
			thread.join()
			thread = None
		anim.event_source.stop()
		anim = None

		lidar.turnOff()
		init_plot()
		timeline_data.clear()
		init_timeline()


# Start real-time plot
def start_animation():
	global anim
	if not os.path.exists(demo_path):
		os.makedirs(demo_path)
	anim = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
	canvas.draw()
	canvas_timeline.draw()


# Streaming data
def start():
	global lidar, inference_label
	if start_button['text'] == 'Start':
		start_button['text'] = 'Stop'
		if lidar is None:
			lidar = YDLidar()
		if lidar.turnOn():
			window.after(500, start_animation)
		else:
			start_button['text'] = 'Start'
	else:
		start_button['text'] = 'Start'
		inference_label['text'] = 'Position: -'


# Close tkinter gui
def quit():
	window.quit()
	window.destroy()


# Centers a tkinter window
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
	window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
	window.deiconify()


# GUI
def initGUI():
	global window, fig, lidar_plot, canvas, start_button, inference_label
	global timeline_plot, canvas_timeline

	init_model()

	window_width = 1600
	window_height = 900

	window = tk.Tk()
	window.minsize(window_width, window_height)
	window.title('LiDAR Monitor')

	# --- LiDAR polar plot ---
	fig = Figure(figsize=(10, 10), dpi=100)
	lidar_plot = fig.add_subplot(projection='polar')
	init_plot()
	canvas = FigureCanvasTkAgg(fig, master=window)
	canvas.draw()
	canvas.get_tk_widget().place(relx=0.0, rely=0.15, relwidth=0.5, relheight=0.8)

	# --- Timeline plot ---
	fig_timeline = Figure(figsize=(10, 10), dpi=100)
	timeline_plot = fig_timeline.add_subplot(111)
	
	canvas_timeline = FigureCanvasTkAgg(fig_timeline, master=window)
	init_timeline()
	canvas_timeline.get_tk_widget().place(relx=0.5, rely=0.15, relwidth=0.5, relheight=0.8)

	# Start/Stop
	start_button = tk.Button(master=window, text='Start', command=start, bg='white', font=('Arial', 18, 'bold'))
	start_button.place(relx=0.25-0.0625, rely=0.08, relwidth=0.125, relheight=0.06)
	
	# Label prediction
	inference_label = tk.Label(master=window, text='Position: -', anchor='center', bg='white', font=('Arial', 18, 'bold'))
	inference_label.place(relx=0.75-0.125, rely=0.08, relwidth=0.25, relheight=0.06)
	
	window.configure(background='white')
	center(window)
	window.mainloop()



if __name__ == '__main__':

	initGUI()

	if os.path.exists(os.path.abspath(os.path.join(__file__, '..', '__pycache__'))):
		import shutil
		shutil.rmtree(os.path.abspath(os.path.join(__file__, '..', '__pycache__')))
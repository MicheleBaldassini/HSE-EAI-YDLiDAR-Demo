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
from matplotlib.backend_bases import key_press_handler

from threading import Thread

from YDLidar import YDLidar
from constants import *


exec(open(os.path.abspath(os.path.join(__file__, '..', '..', '..', 'remove_hidden_items.py'))).read())


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


# Write data to csv file
def write_csv(filename, angle, ran, intensity):
	df = pd.DataFrame(columns=['Angle', 'Range', 'Intensity'])
	df['Angle'] = angle
	df['Range'] = ran
	df['Intensity'] = intensity

	df['Angle'] %= (2 * np.pi)
	df = df[(df['Angle'] >= MIN_ANGLE_RAD) & (df['Angle'] <= MAX_ANGLE_RAD)]
	df['Angle'] = df['Angle'] * 180.0 / np.pi
	df = df.sort_values(by=['Angle'])
	df.to_csv(os.path.join(data_path, str(filename) + '.csv'), header=True, index=False)


# Plot data in real-time
def animate(num):
	global lidar, anim, thread, num_record

	# start_time = time.time()

	if start_button['text'] == 'Stop':
		
		lidar.scan_task()
			
		init_plot()
		lidar_plot.scatter(lidar.angle, lidar.range, c='k', cmap='hsv', alpha=0.95)

		#if (1.0 / lidar.scan.config.scan_time) >= 4.0 and lidar.scan.points.size() < 1000:
		if record_button['text'] == 'No record':
			if num_record > 0:

				thread = Thread(target=write_csv,
								args=(lidar.scan.stamp, lidar.angle, lidar.range, lidar.intensity))
				thread.start()

			num_record = num_record + 1
			record_label['text'] = num_record				
	else:
		if thread is not None:
			thread.join()
			thread = None

		anim.event_source.stop()
		anim = None

		lidar.turnOff()
		init_plot()

		num_record = 0

	# end_time = time.time()
	# print(f'Elapsed time: {end_time - start_time} seconds')


# Streaming data
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
		start_button['text'] = 'Start'
		record_button['text'] = 'Record'
		record_label['text'] = ''


# Record data
def record():
	global thread, filename, num_record

	if start_button['text'] == 'Stop':
		if record_button['text'] == 'Record':

			if not os.path.exists(data_path):
				os.makedirs(data_path)

			record_button['text'] = 'No record'

		else:
			if thread is not None:
				thread.join()
				thread = None

			# filename = None
			num_record = 0
			record_button['text'] = 'Record'
			record_label['text'] = ''


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


# Create a tkinter GUI
def initGUI():

	global window, fig, lidar_plot, canvas, start_button, record_button, record_label
																																														  
	window_width = 600
	window_height = 600

	window = tk.Tk()
	window.minsize(window_width, window_height)
	window.title('LiDAR Monitor')

	# Matplotlib figure
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

	# Start/Stop recording
	record_button = tk.Button(master=window, text='Record', command=record, bg='white')
	record_button.place(relx=0.575, rely=0.05, relwidth=0.125, relheight=0.05)

	# Num recording
	record_label = tk.Label(master=window, text='', anchor='w')
	record_label.place(relx=0.7, rely=0.05, relwidth=0.05, relheight=0.05)
	record_label['bg'] = 'white'


	window.configure(background='white')
	center(window)
	window.mainloop()



if __name__ == '__main__':

	data_path = os.path.abspath(__file__ + '/../../data')

	MIN_ANGLE_RAD = MIN_ANGLE_DEG * np.pi / 180.0
	MAX_ANGLE_RAD = MAX_ANGLE_DEG * np.pi / 180.0

	lidar = None
	anim = None
	thread = None
	num_record = 0

	initGUI()

	# Clear the __pycache__ folder
	if os.path.exists(os.path.abspath(__file__ + '/../__pycache__')):
		import shutil
		shutil.rmtree(os.path.abspath(__file__ + '/../__pycache__'))
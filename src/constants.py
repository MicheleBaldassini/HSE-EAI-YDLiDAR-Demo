# -*- coding: utf-8 -*-
port = '/dev/ttyUSB0'

model = 'G2'

if model == 'X4':
	BAUDRATE = 128000
elif model == 'G2':
	BAUDRATE = 128000

SAMPLE_RATE = 5
FREQUENCY = 5

# Min and Max range (distance)
MIN_RANGE = 0.15
MAX_RANGE = 1.5

# Min and Max angles (degrees)
MIN_ANGLE_DEG = 135.0
MAX_ANGLE_DEG = 225.0

interval = int((1.0 / FREQUENCY) * 1000.0)
lidar = None
anim = None
thread = None
num_record = 0


# Inference
labels_map = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f'}
color_map = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red', 4: 'tab:purple', 5: 'tab:brown'}
timeline_data = []